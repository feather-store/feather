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
    INTENTIONS    what to do, and when                  held until due, then
                                                        carried until done
    NOTES         what was said, in its own words       kept as said; found
                                                        through what was
                                                        extracted from it

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
    mem.intentions.add("q3_followup", "send Dana the Q3 numbers", due_at=friday)

    mem.intentions.due()             # what is due now, oldest first
    mem.about()                      # every kind, one prefix read
    mem.recall("how should I contact them", kinds=["preferences"])

Every item carries an `origin` (feather_db.origin): `user` by default, since
the code calling these vouches for what it stores; `agent` for observations,
which are inferences by definition, and for `ingest_facts`, which a model
extracted. Pass `origin="tool"` or `"external"` for what was read rather than
told. Anything but `user` is quoted as data when it reaches a prompt.

Everything underneath is still a `.feather` file and a plain `BaseStore`, so a
LangGraph agent can use this and `store.get(...)` interchangeably.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Any, Iterable, Optional

from feather_db import clock as _clock
from feather_db import origin as _origin
from feather_db.pocket import due_at as _due_at

PREFERENCES = "preferences"
EPISODES    = "episodes"
FACTS       = "facts"
PROCEDURES  = "procedures"
ENTITIES    = "entities"
OBSERVATIONS = "observations"
INTENTIONS  = "intentions"
NOTES       = "notes"

KINDS = (PREFERENCES, EPISODES, FACTS, PROCEDURES, ENTITIES, OBSERVATIONS, INTENTIONS,
         NOTES)

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

    default_origin = _origin.USER

    def __init__(self, memory: "AgentMemory", kind: str, namespace: tuple[str, ...]):
        self._mem = memory
        self.kind = kind
        self.namespace = namespace

    # ── shared plumbing ──────────────────────────────────────────────────
    def _write(self, key: str, value: dict, *, ttl: Optional[float] = None,
               index: Any = None) -> str:
        payload = {"kind": self.kind, "recorded_at": time.time(), **value}
        payload["origin"] = _origin.check(payload.get("origin") or self.default_origin)
        self._mem.store.put(self.namespace, key, payload, ttl=ttl,
                            **({"index": index} if index is not None else {}))
        return key

    def _write_many(self, items: list) -> list:
        """[(key, value)] in one store batch: one save, not one per item."""
        from langgraph.store.base import PutOp
        ops = []
        for key, value in items:
            payload = {"kind": self.kind, "recorded_at": time.time(), **value}
            payload["origin"] = _origin.check(payload.get("origin") or self.default_origin)
            ops.append(PutOp(self.namespace, key, payload))
        if ops:
            self._mem.store.batch(ops)
        return [k for k, _ in items]

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
            # The record now holds this statement too, in its history.
            origin = _origin.least(_origin.check(extra.get("origin") or self.default_origin),
                                   previous.get("origin") or _origin.USER)
            return self._write(key, {**previous, "supersedes": history[-10:],
                                     "origin": origin})
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
        # The merged entity holds what every write added, so it is only as
        # trusted as the least trusted of them: otherwise a `user` enrichment
        # would launder the attributes a tool or web page put there.
        origin = _origin.check(extra.pop("origin", None) or self.default_origin)
        if current:
            origin = _origin.least(origin, current.get("origin") or _origin.USER)
        return self._write(name, {"name": name, "attributes": merged,
                                  "text": text or current.get("text") or name,
                                  "seen_count": seen, "origin": origin, **extra})


class _Observations(_Kind):
    """The reflection layer. An observation cites the records that support it,
    so a conclusion the agent drew can be traced back or retracted. Its origin
    is `agent`: it is an inference, whoever's evidence it cites."""

    default_origin = _origin.AGENT

    def note(self, key: str, text: str, *, evidence: Optional[Iterable[str]] = None,
             confidence: float = 0.5, supersedes: Optional[Iterable[str]] = None,
             origin: str = _origin.AGENT) -> str:
        evidence = list(evidence or [])
        self._write(key, {"text": text,
                          "evidence": evidence,
                          "confidence": confidence,
                          "supersedes": list(supersedes or []),
                          "origin": origin})
        # The citation as an edge as well as a key in the value: the engine
        # reads edges, so erasing what an observation cites erases it too.
        store = self._mem.store
        obs = store.id_of(self.namespace, key)
        for cited in self._mem._resolve_keys(evidence):
            store.db.link(obs, cited, "cites", 1.0)
        return key


class _Intentions(_Kind):
    """Prospective memory: something to do, and when. An intention is held
    back until `due_at`, then surfaces until it is marked done; a reminder
    that decays like a note is forgotten exactly when it matters.

    The value's `due_at` is mapped onto the record (FeatherStore), so a Pocket
    reading this namespace keeps an open intention out of its working set
    until it is due and carries it as if pinned after. `due()` is the same
    read for code that assembles its own prompt. `done()` ends it: an
    intention kept after it is done reads like one still to do."""

    def add(self, key: str, text: str, *, due_at: float, **extra) -> str:
        """Hold `text` until `due_at` (unix seconds). Same key replaces, so a
        rescheduled reminder is one reminder."""
        return self._write(key, {"text": text, "due_at": float(due_at),
                                 "status": "open", **extra})

    def due(self, now: Optional[float] = None, limit: int = 100) -> list[dict]:
        """Open intentions due at `now` (default: the handle's clock, so an
        as_of replay sees what was due then), the longest overdue first.
        Reads the due time off each record and parses only the due ones."""
        store = self._mem.store
        now = _clock.now(store.db) if now is None else now
        rows = [(d, meta) for _rid, meta in store.scan(self.namespace)
                if (d := _due_at(meta)) is not None and d <= now]
        rows.sort(key=lambda r: r[0])
        return [store._value_of(meta) for _d, meta in rows[:limit]]

    def done(self, key: str, *, outcome: Optional[str] = None) -> Optional[str]:
        """Remove the intention; with a user, record its completion as an
        episode, so "did I send it?" still has an answer. Returns the
        episode's key, or None (unknown key, or no episodes to record in)."""
        current = self.get(key)
        if current is None:
            return None
        episode = None
        if not isinstance(self._mem.episodes.namespace, _Missing):
            episode = self._mem.episodes.record(
                f"done: {current.get('text', key)}", outcome=outcome,
                intention=key, was_due_at=current.get("due_at"))
        self.delete(key)
        return episode

class _Notes(_Kind):
    """Raw memory: what was said, kept in its own words, and found through
    the facts extracted from it (`FeatherStore.index`): facts as index,
    notes as payload. A fact with a one-value key also chains the notes
    that state a value of it, so `recall` returns the newest with the older
    ones in its `earlier`.

    Why not store the facts themselves: on LoCoMo a reader given extracted
    facts scored J 64, given the raw turns 78.8, and given the raw turns
    found through their facts too, 82.4 (bench.locomo). Extraction keeps a
    few facts per session and drops the detail a question asks about."""

    def add(self, text: str, *, key: Optional[str] = None, at: Optional[float] = None,
            index: Optional[Iterable[Any]] = None, **extra) -> str:
        """Keep `text` as said, at `at` (when it was said or became true,
        default now), indexed by `index` (IndexEntry, str or extractor Fact).
        Without `key` it is appended (`<unix>_<slug>`, `#n` on a clash); a
        key rewrites the note at it and replaces its index."""
        at = at if at is not None else time.time()
        if key is None:
            base = key = f"{int(at)}_{_slug(text)}"
            with _APPEND_LOCK:
                n = 1
                while self._mem.store.get(self.namespace, key) is not None:
                    n += 1
                    key = f"{base}#{n}"
                self._write(key, {"text": text, "valid_at": at, **extra})
        else:
            self._write(key, {"text": text, "valid_at": at, **extra})
        if index is not None:
            self._mem.store.index(self.namespace, key, index)
        return key

    def known_keys(self) -> list[str]:
        """The one-value keys these notes are indexed by: pass them to the
        extractor (`known_keys`) so it names an attribute the way it did
        before."""
        return self._mem.store.known_keys(self.namespace)


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
        self.intentions   = _Intentions(self, INTENTIONS,
                                        self._ns(user or agent, INTENTIONS, "user or agent"))
        self.notes        = _Notes(self, NOTES, self._ns(user, NOTES, "user"))

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

    def ingest_session(self, turns: list[dict], session_date: Any, extractor: Any = None, *,
                       session_id: Optional[str] = None,
                       user_name: Optional[str] = None, queue: Any = None) -> dict[str, int]:
        """Keep a conversation session as notes, one per turn in its own words
        ("user: ...", "assistant: ..."), indexed by the facts `extractor` (a
        SessionFactExtractor) draws from it: each fact indexes the turn it
        cites (the first turn when it cites none). The extractor is told
        `user_name` and the one-value keys the notes already hold, so an
        attribute keeps one name across sessions.

        The measured alternative to `ingest_facts`, which stores the facts as
        the memory: LoCoMo J 82.4 against 64 (bench.locomo --memory index /
        facts). Idempotent per session: notes are keyed `<session>#<turn>`,
        so ingesting a session again rewrites it. User turns are stored with
        origin `user`, the rest `agent`. Returns counts.

        With `queue` (feather_db.ingest.IngestQueue) the notes are written now,
        in one batch, and the extraction is a job: the call returns before
        any model is asked, and a failed call is retried instead of leaving
        the session unindexed. `extractor` is then the queue's."""
        import datetime as _dt
        from feather_db.extractors.session_facts import _parse_day
        day = (session_date if isinstance(session_date, _dt.datetime)
               else _parse_day(session_date))
        at = day.timestamp() if day else time.time()
        sid = session_id or str(int(at))
        if queue is not None:
            notes = []
            for i, t in enumerate(turns):
                text = str(t.get("content") or "").strip()
                if not text:
                    continue
                role = t.get("role") or "user"
                notes.append((f"{sid}#{i}", {
                    "text": f"{role}: {text}", "valid_at": at, "role": role,
                    "session": sid, "turn": i,
                    "origin": _origin.USER if role == "user" else _origin.AGENT}))
            written = self.notes._write_many(notes)
            job = queue.submit_session(".".join(self.notes.namespace), written,
                                       session_date, session_id=sid, user_name=user_name)
            return {"turns": len(written), "queued": job}
        if extractor is None:
            raise ValueError("ingest_session needs an extractor, or a queue that has one")
        facts = extractor.extract_session(
            turns, session_date,
            context={"user_name": user_name, "known_keys": self.notes.known_keys(),
                     "session_id": sid})
        cited: dict[int, list] = {}
        for f in facts:
            t = f.metadata.get("turn")
            cited.setdefault(t if isinstance(t, int) and 0 <= t < len(turns) else 0,
                             []).append(f)
        written = 0
        for i, t in enumerate(turns):
            text = str(t.get("content") or "").strip()
            if not text:
                continue
            role = t.get("role") or "user"
            self.notes.add(f"{role}: {text}", key=f"{sid}#{i}", at=at,
                           index=cited.get(i, []), role=role, session=sid, turn=i,
                           origin=_origin.USER if role == "user" else _origin.AGENT)
            written += 1
        return {"turns": written, "facts": len(facts),
                "one_value": sum(1 for f in facts if f.metadata.get("replaces"))}

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
        time of ingest, so sessions can be ingested in any order, and
        origin `agent`: a model extracted it, from a transcript that may
        hold tool output or pasted text as well as what the user said.
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
                    "own": m.get("own", True), "source": src,
                    "origin": _origin.AGENT})
                counts["episodes"] += 1
            elif kind == "preference":
                self.preferences.set(_slug(fact_key(f), 80), text, at=at,
                                     value=f.object, source=src,
                                     origin=_origin.AGENT)
                counts["preferences"] += 1
            else:
                self.facts.state(_slug(fact_key(f), 80), text, at=at,
                                 source=src, value=f.object,
                                 origin=_origin.AGENT)
                counts["facts"] += 1
        return counts

    # ── erasure ──────────────────────────────────────────────────────────

    def erase_subject(self, subject: Optional[str] = None, *,
                      dry_run: bool = False) -> dict:
        """Erase everything held about `subject` (default: this user), so that
        nothing of it remains: every kind under (org, subject), the org entity
        named after it, and every record derived from or citing any of those.
        The right to erasure, for one person. Returns the engine's receipt."""
        subject = subject or self.user
        if not subject:
            raise ValueError("no subject: pass one, or construct with user=...")
        ids = [i for i in [self.store.id_of((self.org, ENTITIES), subject)] if i is not None]
        return self.store.db.erase(ids=ids, namespace=f"{self.org}.{subject}",
                                   prefix=True, dry_run=dry_run)

    def erase(self, kind: str, key: str, *, dry_run: bool = False) -> dict:
        """Erase one item, and every record derived from or citing it, so that
        nothing of it remains. Observations cite by edge since 0.22; one
        written before cites only by key in its value, and is found that way
        too (over-matching a key shared by two kinds erases an observation more,
        never one less)."""
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}")
        ns = getattr(self, kind).namespace
        ids = []
        if not isinstance(self.observations.namespace, _Missing):
            import json as _json
            for oid, meta in self.store.scan(self.observations.namespace):
                try:
                    value = _json.loads(meta.get_attribute("_lg_value") or "{}")
                except ValueError:
                    continue
                if key in (value.get("evidence") or []):
                    ids.append(oid)
        # The item by (namespace, entity = its key), a deleted one's shell
        # included; the observations citing it by key, by id.
        return self.store.db.erase(ids=ids, namespace=".".join(ns), entity=key,
                                   dry_run=dry_run)

    def _resolve_keys(self, keys_: Iterable[str]) -> list[int]:
        """Record ids for bare keys, looked up in every kind of this subject and
        in the org's entities. A key held by two kinds resolves to both."""
        spaces = [getattr(self, k).namespace for k in
                  ("preferences", "episodes", "facts", "procedures", "observations",
                   "intentions", "notes", "entities")]
        out = []
        for k in keys_:
            for ns in spaces:
                if isinstance(ns, _Missing):
                    continue
                rid = self.store.id_of(ns, k)
                if rid is not None:
                    out.append(rid)
        return out

    def summary(self) -> dict[str, int]:
        """How much of each kind is held — cheap enough to log every turn."""
        out = {}
        for name in ("preferences", "episodes", "facts", "procedures",
                     "entities", "observations", "intentions", "notes"):
            kind = getattr(self, name)
            out[name] = 0 if isinstance(kind.namespace, _Missing) else len(kind)
        return out   # len() is now a count, so this is one cheap scan per kind

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
