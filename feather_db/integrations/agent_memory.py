"""AgentMemory — the five kinds of thing an agent stores, over a FeatherStore.

`FeatherStore` is a correct LangGraph `BaseStore`: namespaces, keys, JSON values,
semantic search. That is the storage layer, and it is deliberately generic — it
will store anything you hand it under any namespace you invent.

Which is the problem. An agent memory is not generic. Five kinds of thing get
stored, they have different lifetimes, and the difference is not decorative:

    PREFERENCES   how a person wants to be treated      revised in place
    EPISODES      what happened and how it turned out   append-only, never edited
    FACTS         world knowledge, rules, positioning   corrected, with history
    PROCEDURES    how to do something                   versioned
    ENTITIES      things that recur                     enriched, merged

Write an episode with upsert semantics and you have destroyed the history that
made it an episode. Write a preference append-only and the agent now holds six
contradictory answers to "how do they want to be contacted" with no way to tell
which is current. The kind determines the write semantics, so the kind has to be
in the API rather than in the caller's head.

The namespace ordering matters just as much, and is just as easy to get wrong.
Search takes a namespace *prefix*, so `(org, user, kind)` makes "everything I
know about this user" a prefix read, while `(org, kind, user)` makes it a filter
over every user you have. Same data, and the second one does not scale. This
module owns that ordering so a caller cannot invert it.

    mem = AgentMemory(store, org="hawky", user="user_842", brand="nike")

    mem.preferences.set("comms", "prefers written async over meetings")
    mem.episodes.record("q3 report shipped late", outcome="missed_deadline")
    mem.facts.state("positioning", "premium, never discount-led")
    mem.entities.enrich("nike", {"industry": "apparel"})

    mem.about()                      # every kind, one prefix read
    mem.recall("how should I contact them", kinds=["preferences"])

Everything underneath is still a `.feather` file and a plain `BaseStore`, so a
LangGraph agent can use this and `store.get(...)` interchangeably.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Any, Iterable, Optional

PREFERENCES = "preferences"
EPISODES    = "episodes"
FACTS       = "facts"
PROCEDURES  = "procedures"
ENTITIES    = "entities"
OBSERVATIONS = "observations"

KINDS = (PREFERENCES, EPISODES, FACTS, PROCEDURES, ENTITIES, OBSERVATIONS)

_SLUG = re.compile(r"[^a-z0-9]+")
_APPEND_LOCK = threading.Lock()   # check-then-write for episode keys


def _slug(text: str, limit: int = 40) -> str:
    return _SLUG.sub("_", text.lower()).strip("_")[:limit] or "item"


def _event_time(meta) -> float:
    """Event time from an episode key, `<unix>_<slug>`. Falls back to the engine
    timestamp for anything not written by _Episodes.record()."""
    head = (meta.entity_id or "").split("_", 1)[0]
    try:
        return float(head)
    except ValueError:
        return float(meta.timestamp)


class _Kind:
    """One kind of memory, bound to its namespace. Not constructed directly."""

    def __init__(self, memory: "AgentMemory", kind: str, namespace: tuple[str, ...]):
        self._mem = memory
        self.kind = kind
        self.namespace = namespace

    # ── shared plumbing ──────────────────────────────────────────────────
    def _write(self, key: str, value: dict, *, ttl: Optional[float] = None,
               index: Any = None) -> str:
        payload = {"kind": self.kind, "recorded_at": time.time(), **value}
        self._mem.store.put(self.namespace, key, payload, ttl=ttl,
                            **({"index": index} if index is not None else {}))
        return key

    def get(self, key: str) -> Optional[dict]:
        item = self._mem.store.get(self.namespace, key)
        return item.value if item else None

    def all(self, limit: int = 100) -> list[dict]:
        """Every item of this kind, newest first.

        Uses the store's `newest`, not `search`. `search` has no ordering, so
        sorting its arbitrary slice returned the newest of a random subset and
        silently dropped real records.

        Ordering is by the engine timestamp, which has one-second resolution, so
        items written within the same second are in arbitrary order relative to
        each other. That is fine for the kinds this matters for — preferences,
        facts and procedures are upserted by key and few — and `_Episodes`
        overrides it. Sub-second ordering needs the second timestamp field
        format v10 is scoped to carry.
        """
        return [i.value for i in self._mem.store.newest(self.namespace, limit)]

    def delete(self, key: str) -> None:
        self._mem.store.delete(self.namespace, key)

    def __len__(self) -> int:
        # A count, not a fetch. The old version pulled 1000 items and parsed
        # every one's JSON just to call len() on the list — 17 ms at 2000
        # memories, and wrong past 1000.
        return self._mem.store.count(self.namespace)

    def __repr__(self) -> str:
        return f"<{self.kind} {'.'.join(self.namespace)} n={len(self)}>"


class _Preferences(_Kind):
    """Revised in place. The current answer is the only answer.

    `at` is when the preference was stated, if not now. A statement older
    than the one held is not newer information and is dropped, so
    ingesting sessions out of order cannot roll a preference back."""

    def set(self, key: str, text: str, *, confidence: float = 1.0,
            at: Optional[float] = None, **extra) -> str:
        if at is not None:
            held = (self.get(key) or {}).get("since")
            if held is not None and at < held:
                return key
            extra["since"] = extra["valid_at"] = at
        return self._write(key, {"text": text, "confidence": confidence, **extra})


class _Episodes(_Kind):
    """Append-only. An episode that can be overwritten is not a record of
    anything. The key is `<unix second>_<slug>`; the same event recorded twice
    in one second (a retry, a loop) gets `#2`, `#3`… rather than replacing the
    first, which is what the bare key used to do."""

    def record(self, what: str, *, outcome: Optional[str] = None,
               at: Optional[float] = None, ttl: Optional[float] = None,
               **extra) -> str:
        at = at if at is not None else time.time()
        base = key = f"{int(at)}_{_slug(what)}"
        with _APPEND_LOCK:
            n = 1
            while self._mem.store.get(self.namespace, key) is not None:
                n += 1
                key = f"{base}#{n}"
            return self._write(key, {"text": what, "outcome": outcome,
                                     "happened_at": at, "valid_at": at,
                                     **extra}, ttl=ttl)

    def all(self, limit: int = 100) -> list[dict]:
        """Episodes order by event time, not by write time — same as recent().

        The generic `all()` sorts on the engine timestamp, which is whole
        seconds, so 300 episodes written in one second tie and come back in an
        arbitrary order: asking for 3 returned events 298, 297, 296 rather than
        299, 298, 297.
        """
        return self.recent(limit)

    def recent(self, limit: int = 10) -> list[dict]:
        """The `limit` most recent episodes, by when they HAPPENED.

        An episode may be backdated with `at=`, so the engine's own timestamp
        (when it was written) is the wrong field. Each episode carries its
        event time as the engine's valid time, so this is an index walk.
        Episodes written before valid time existed have none; while any
        remain, this falls back to the event time in their keys, which reads
        and sorts the namespace.
        """
        store = self._mem.store
        hits = store.newest(self.namespace, limit, by_valid=True)
        if any("valid_at" not in h.value for h in hits):
            hits = store.newest(self.namespace, limit, key=_event_time)
        return [h.value for h in hits]


class _Facts(_Kind):
    """Corrected, not replaced silently — a correction keeps what it replaced,
    so an agent can answer 'since when' and spot a fact that keeps flipping."""

    def state(self, key: str, text: str, *, source: Optional[str] = None,
              at: Optional[float] = None, **extra) -> str:
        """`at`: when the fact became true (valid time), if not now. The
        superseded value's `until` is then that time, not the time of the
        write, and a fact older than the current one goes into the history
        in date order instead of replacing what is current."""
        previous = self.get(key)
        history = (previous or {}).get("supersedes", [])
        since = at if at is not None else time.time()
        held = (previous or {}).get("since")
        if previous and held is not None and since < held:
            if previous.get("text") == text or any(
                    h.get("text") == text and h.get("since") == since
                    for h in history):
                return key
            history = sorted([*history, {"text": text, "since": since,
                                         "until": None, "source": source}],
                             key=lambda h: h.get("since") or 0)
            for h, nxt in zip(history, [*history[1:], {"since": held}]):
                h["until"] = nxt.get("since")
            return self._write(key, {**previous, "supersedes": history[-10:]})
        if previous and previous.get("text") != text:
            history = [*history, {"text": previous["text"],
                                  "since": held, "until": since,
                                  "source": previous.get("source")}][-10:]
        return self._write(key, {"text": text, "source": source,
                                 "since": since, "valid_at": since,
                                 "supersedes": history, **extra})

    def history(self, key: str) -> list[dict]:
        return (self.get(key) or {}).get("supersedes", [])


class _Procedures(_Kind):
    """Versioned. `name` is always the current version; `name@N` is kept so a
    procedure that regressed can be compared against the one that worked."""

    def define(self, name: str, steps: Iterable[str], *,
               checks: Optional[Iterable[str]] = None, **extra) -> str:
        current = self.get(name)
        version = int((current or {}).get("version", 0)) + 1
        body = {"name": name, "steps": list(steps),
                "checks": list(checks or []), "version": version, **extra}
        if current:
            self._write(f"{name}@{current['version']}", current)
        return self._write(name, body)

    def version(self, name: str, n: int) -> Optional[dict]:
        return self.get(name) if (self.get(name) or {}).get("version") == n \
            else self.get(f"{name}@{n}")


class _Entities(_Kind):
    """Enriched. Each write merges into what is already known rather than
    replacing it, because the next thing you learn about an entity is an
    addition, not a correction."""

    def enrich(self, name: str, attributes: dict, *, text: Optional[str] = None,
               **extra) -> str:
        current = self.get(name) or {}
        merged = {**current.get("attributes", {}), **attributes}
        seen = int(current.get("seen_count", 0)) + 1
        return self._write(name, {"name": name, "attributes": merged,
                                  "text": text or current.get("text") or name,
                                  "seen_count": seen, **extra})


class _Observations(_Kind):
    """The reflection layer. An observation cites the records that support it,
    so a conclusion the agent drew can be traced back or retracted."""

    def note(self, key: str, text: str, *, evidence: Optional[Iterable[str]] = None,
             confidence: float = 0.5, supersedes: Optional[Iterable[str]] = None) -> str:
        return self._write(key, {"text": text,
                                 "evidence": list(evidence or []),
                                 "confidence": confidence,
                                 "supersedes": list(supersedes or [])})


class AgentMemory:
    """Typed agent memory over a `FeatherStore`.

    `org` is always the root. `user`, `agent` and `brand` are the subjects a kind
    hangs off; a kind whose subject was not supplied raises rather than quietly
    writing somewhere surprising.
    """

    def __init__(self, store, org: str, *, user: Optional[str] = None,
                 agent: Optional[str] = None, brand: Optional[str] = None):
        if not org:
            raise ValueError("org is required — it is the root of the namespace tree")
        self.store = store
        self.org, self.user, self.agent, self.brand = org, user, agent, brand

        self.preferences  = _Preferences(self, PREFERENCES, self._ns(user, PREFERENCES, "user"))
        self.episodes     = _Episodes(self, EPISODES, self._ns(user, EPISODES, "user"))
        self.facts        = _Facts(self, FACTS, self._ns(brand or user, FACTS, "brand or user"))
        self.procedures   = _Procedures(self, PROCEDURES, self._ns(agent, PROCEDURES, "agent"))
        self.entities     = _Entities(self, ENTITIES, (org, ENTITIES))
        self.observations = _Observations(self, OBSERVATIONS, self._ns(user, OBSERVATIONS, "user"))

    def _ns(self, subject: Optional[str], kind: str, needs: str) -> tuple[str, ...]:
        if subject is None:
            return _Missing(kind, needs)           # type: ignore[return-value]
        return (self.org, subject, kind)

    # ── reads across kinds ───────────────────────────────────────────────

    def about(self, subject: Optional[str] = None, *, limit: int = 100) -> dict[str, list[dict]]:
        """Everything stored about a subject, grouped by kind.

        One prefix read, which is the whole reason the namespace is ordered
        tenant → subject → kind.
        """
        subject = subject or self.user or self.brand or self.agent
        if subject is None:
            raise ValueError("no subject — pass one, or construct with user/brand/agent")
        out: dict[str, list[dict]] = {}
        for item in self.store.search((self.org, subject), query=None, limit=limit):
            out.setdefault(item.value.get("kind", "unknown"), []).append(item.value)
        return out

    def recall(self, query: str, *, kinds: Optional[Iterable[str]] = None,
               subject: Optional[str] = None, limit: int = 5) -> list[dict]:
        """Semantic search, optionally narrowed to certain kinds."""
        subject = subject or self.user or self.brand or self.agent
        if not kinds:
            prefix = (self.org, subject) if subject else (self.org,)
            return [h.value for h in self.store.search(prefix, query=query, limit=limit)]
        # Each kind is its own namespace: search those, inside the engine, and
        # merge. Searching the subject and keeping the wanted kinds afterwards
        # (it fetched limit x 4) came back empty for a subject whose other
        # kinds outranked them.
        hits = []
        for kind in dict.fromkeys(kinds):
            ns = (self.org, ENTITIES) if kind == ENTITIES else (self.org, subject, kind)
            hits += self.store.search(ns, query=query, limit=limit)
        hits.sort(key=lambda h: -(h.score if h.score is not None else 0.0))
        return [h.value for h in hits[:limit]]

    def ingest_facts(self, facts: Iterable[Any], *,
                     source: Optional[str] = None) -> dict[str, int]:
        """Route extracted facts (feather_db.extractors.Fact, e.g. from
        SessionFactExtractor) to the kind whose lifetime fits each:

            event       -> episodes, at the date it happened; the same
                           event (entity + day) told twice is one episode
            preference  -> preferences, keyed by subject/attribute
            state/other -> facts, keyed by subject/attribute; a later
                           value supersedes with its valid-from date

        Every write carries the fact's valid time (`valid_at`), not the
        time of ingest, so sessions can be ingested in any order.
        Returns how many facts went to each kind (and how many were
        repeats of an episode already held).
        """
        from feather_db.extractors import fact_key   # no import cycle at load
        counts = {"episodes": 0, "preferences": 0, "facts": 0, "repeats": 0}
        for f in facts:
            m = f.metadata or {}
            at = float(f.valid_at) if f.valid_at is not None else None
            text = m.get("statement") or f"{f.subject} {f.predicate}: {f.object}"
            kind = m.get("kind", "other")
            src = source or m.get("session_id")
            if kind == "event":
                day = int(at if at is not None else time.time())
                key = f"{day}_{_slug(m.get('entity') or text)}"
                if self.store.get(self.episodes.namespace, key) is not None:
                    counts["repeats"] += 1
                    continue
                self.episodes._write(key, {
                    "text": text, "outcome": None, "happened_at": day,
                    "valid_at": day,
                    "entity": m.get("entity", ""), "amount": m.get("amount", ""),
                    "own": m.get("own", True), "source": src})
                counts["episodes"] += 1
            elif kind == "preference":
                self.preferences.set(_slug(fact_key(f), 80), text, at=at,
                                     value=f.object, source=src)
                counts["preferences"] += 1
            else:
                self.facts.state(_slug(fact_key(f), 80), text, at=at,
                                 source=src, value=f.object)
                counts["facts"] += 1
        return counts

    def summary(self) -> dict[str, int]:
        """How much of each kind is held — cheap enough to log every turn."""
        out = {}
        for name in ("preferences", "episodes", "facts", "procedures",
                     "entities", "observations"):
            kind = getattr(self, name)
            out[name] = 0 if isinstance(kind.namespace, _Missing) else len(kind)
        return out   # len() is now a count, so this is six cheap scans

    def __repr__(self) -> str:
        who = ", ".join(f"{k}={v}" for k, v in
                        (("user", self.user), ("agent", self.agent), ("brand", self.brand))
                        if v)
        return f"<AgentMemory {self.org}{' ' + who if who else ''}>"


class _Missing(tuple):
    """Stands in for a namespace whose subject was never supplied.

    Returning this instead of raising in `__init__` means constructing
    `AgentMemory(store, org, user=...)` without an `agent` is fine — you only hit
    the error if you actually touch `.procedures`, and the error says which
    argument was missing rather than surfacing as an opaque tuple error later.
    """

    def __new__(cls, kind: str, needs: str):
        self = super().__new__(cls)
        self.kind, self.needs = kind, needs
        return self

    def _die(self, *_a, **_k):
        raise ValueError(
            f"{self.kind} needs `{self.needs}` — construct AgentMemory with it, "
            f"e.g. AgentMemory(store, org, {self.needs.split(' or ')[0]}='...')"
        )

    __iter__ = __len__ = __getitem__ = __hash__ = _die
