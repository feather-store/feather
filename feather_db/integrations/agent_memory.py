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


def _slug(text: str, limit: int = 40) -> str:
    return _SLUG.sub("_", text.lower()).strip("_")[:limit] or "item"


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
        """Every item of this kind, newest first."""
        items = self._mem.store.search(self.namespace, query=None, limit=limit)
        return sorted((i.value for i in items),
                      key=lambda v: v.get("recorded_at", 0), reverse=True)

    def delete(self, key: str) -> None:
        self._mem.store.delete(self.namespace, key)

    def __len__(self) -> int:
        return len(self._mem.store.search(self.namespace, query=None, limit=1000))

    def __repr__(self) -> str:
        return f"<{self.kind} {'.'.join(self.namespace)} n={len(self)}>"


class _Preferences(_Kind):
    """Revised in place. The current answer is the only answer."""

    def set(self, key: str, text: str, *, confidence: float = 1.0, **extra) -> str:
        return self._write(key, {"text": text, "confidence": confidence, **extra})


class _Episodes(_Kind):
    """Append-only. An episode that can be overwritten is not a record of
    anything — the key embeds the time so two writes never collide."""

    def record(self, what: str, *, outcome: Optional[str] = None,
               at: Optional[float] = None, ttl: Optional[float] = None,
               **extra) -> str:
        at = at if at is not None else time.time()
        key = f"{int(at)}_{_slug(what)}"
        return self._write(key, {"text": what, "outcome": outcome,
                                 "happened_at": at, **extra}, ttl=ttl)

    def recent(self, limit: int = 10) -> list[dict]:
        items = self._mem.store.search(self.namespace, query=None, limit=limit * 4)
        return sorted((i.value for i in items),
                      key=lambda v: v.get("happened_at", 0), reverse=True)[:limit]


class _Facts(_Kind):
    """Corrected, not replaced silently — a correction keeps what it replaced,
    so an agent can answer 'since when' and spot a fact that keeps flipping."""

    def state(self, key: str, text: str, *, source: Optional[str] = None,
              **extra) -> str:
        previous = self.get(key)
        history = (previous or {}).get("supersedes", [])
        if previous and previous.get("text") != text:
            history = [*history, {"text": previous["text"],
                                  "until": time.time(),
                                  "source": previous.get("source")}][-10:]
        return self._write(key, {"text": text, "source": source,
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
        prefix = (self.org, subject) if subject else (self.org,)
        wanted = set(kinds) if kinds else None
        hits = self.store.search(prefix, query=query, limit=limit * 4 if wanted else limit)
        values = [h.value for h in hits
                  if wanted is None or h.value.get("kind") in wanted]
        return values[:limit]

    def summary(self) -> dict[str, int]:
        """How much of each kind is held — cheap enough to log every turn."""
        out = {}
        for name in ("preferences", "episodes", "facts", "procedures",
                     "entities", "observations"):
            kind = getattr(self, name)
            out[name] = 0 if isinstance(kind.namespace, _Missing) else len(kind)
        return out

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
