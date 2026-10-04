"""Agent pocket memory — a token-budgeted working set over a Feather DB.

An agent does not want "the top 10 results for a query". It wants *the things it
should be carrying right now*, sized to fit the context window it has left. That
is a different question, and answering it by hand is what every agent codebase
ends up doing badly: slice the tool output to 20 items, keep four fields, hash
the query and cache the result, hope the budget holds.

Feather already computes the signal this needs and nobody reads it as a tier.
Every search hit increments `recall_count`; `stickiness = 1 + ln(1+recall_count)`
already slows decay for things the agent keeps coming back to. That IS hotness —
measured continuously, persisted, and currently used only to re-rank search.

Three tiers, one file:

  HOT    the working set. Fits a token budget, ordered by heat. Free to read —
         no query, no embedding, no model call.
  WARM   everything else in the namespace. Reachable by `recall()`, and a hit
         promotes it, so the working set follows what the agent actually uses.
  CACHE  keyed results (a ClickHouse query, a tool call) with a TTL, evicted by
         the same rules rather than by a separate expiry thread.

Nothing here is a new storage engine. It is an interpretation of counters the
database already keeps.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import numpy as np

import feather_db
from feather_db.filter import FilterBuilder

# ~4 characters per token is the usual rough figure for English prose. It is an
# estimate on purpose: an exact tokenizer would tie Feather to one model family,
# and the budget only has to be approximately right to be useful.
CHARS_PER_TOKEN = 4

# How many recalls it takes for usage to contribute half its weight. Saturating
# rather than linear so a record read 200 times does not permanently crowd out
# everything else — past a point, more reads are not more evidence.
USAGE_HALF_LIFE = 3.0
# What a never-recalled record keeps. Non-zero because "not yet used" is not the
# same as "not useful" — a fact stored a minute ago has had no chance to be read.
UNUSED_FLOOR = 0.4
# Below this, a memory is not worth context space even when the budget is free.
# "Fits" and "worth carrying" are different questions: a 500-day-old note at
# heat 0.0001 costs few tokens but spends the model's attention, and an agent
# reasoning from stale context is worse off than one reasoning from none.
MIN_HEAT = 0.01
# How much heat may influence recall() ranking. Small on purpose: recall answers
# an explicit question, so a memory being frequently used is weak evidence next
# to it actually matching the query. At 0.15 a pinned record (heat 1.0 by
# definition) outranked a row scoring 0.04 higher on relevance — heat should
# separate near-ties, not overturn the search.
RECALL_HEAT_WEIGHT = 0.05

# `hot()` is O(namespace): it reads every record in scope, scores it, and sorts.
# Measured, that is 171 ms at 10,000 memories in one agent's scope — far too slow
# to call once per turn, which is exactly how an agent uses it.
#
# The access pattern is heavily read-biased: an agent assembles context many
# times between writes. So the result is memoised against a write generation,
# and any write through any pocket on the same DB invalidates it. Keyed per DB
# rather than per pocket because a parent's write must invalidate a child's
# cached view — inheritance means the child can see it.
_WRITE_GEN: dict[int, int] = {}


def _bump(db) -> None:
    _WRITE_GEN[id(db)] = _WRITE_GEN.get(id(db), 0) + 1


def _gen(db) -> int:
    return _WRITE_GEN.get(id(db), 0)

_CACHE_KIND   = "_pocket_cache"
_CACHE_EXPIRY = "_pocket_expires_at"
_PINNED       = "_pocket_pinned"

# Scopes are hierarchical, joined with "." — the same convention FeatherStore
# uses, so a namespace written by one is readable by the other.
SCOPE_SEP = "."

# An agent's own memory outranks what it inherits. Without this a brand-wide
# fact written months ago and read by every agent would outrank the note this
# agent made ten minutes ago, purely because it has more recalls across the
# fleet. Depth is per level of inheritance distance.
INHERIT_DECAY = 0.7


def _scope(scope) -> tuple[str, ...]:
    """Accept "a.b.c", ("a","b","c") or "a" and normalise to a tuple."""
    if isinstance(scope, str):
        return tuple(x for x in scope.split(SCOPE_SEP) if x)
    return tuple(scope)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class PocketItem:
    id: int
    key: str
    content: str
    heat: float
    tokens: int
    recalls: int
    age_days: float
    pinned: bool
    scope: tuple[str, ...] = ()      # which layer it came from
    inherited: bool = False          # True when it came from an ancestor scope
    relevance: float = 0.0           # normalised search score; 0 outside recall()

    def __repr__(self) -> str:          # readable in a debugger / log line
        p = "*" if self.pinned else " "
        i = "^" if self.inherited else " "
        return f"<{p}{i}{self.key} heat={self.heat:.3f} tok={self.tokens} recalls={self.recalls}>"


class Pocket:
    """A working set for one agent, with a token budget.

    Scopes are hierarchical, and an agent reads its own scope **plus every
    ancestor**:

        ("hawky",)                              org — brand-agnostic rules
        ("hawky", "brand_a")                    everything about one brand
        ("hawky", "brand_a", "creative_agent")  this agent's own memory

    A pocket opened at the deepest scope carries all three, because that is what
    an agent actually needs: its own notes AND the brand rules AND the org
    policy, in one budget, ranked together. Writes always land in the pocket's
    own scope, so one agent cannot silently mutate a sibling's memory — sharing
    is something you do by writing higher up, deliberately.
    """

    def __init__(
        self,
        db,
        scope,
        *,
        budget_tokens: int = 4000,
        half_life_days: float = 7.0,
        min_heat: float = MIN_HEAT,
        inherit: bool = True,
        embed: Optional[Callable[[str], Any]] = None,
    ):
        self.scope = _scope(scope)
        self.namespace = SCOPE_SEP.join(self.scope)   # where writes land
        self.db = db
        self.budget = budget_tokens
        self.half_life = half_life_days
        self.min_heat = min_heat
        self.inherit = inherit
        self._embed = embed
        self._hot_cache: dict = {}

    # ── scope resolution ─────────────────────────────────────────────────

    def scopes(self) -> list[tuple[str, ...]]:
        """This scope and its ancestors, deepest first."""
        if not self.inherit:
            return [self.scope]
        return [self.scope[:i] for i in range(len(self.scope), 0, -1)]

    def _scoped_ids(self):
        """(id, scope, depth) for everything this pocket can see.

        depth 0 is the agent's own scope; 1 is its parent, and so on. Feather's
        namespace index is exact-match, so each level is looked up separately
        rather than by prefix — O(levels) lookups, each still the O(matches)
        index hit.
        """
        for depth, sc in enumerate(self.scopes()):
            flat = SCOPE_SEP.join(sc)
            for rid in self.db.ids_in_namespace(flat):
                yield rid, sc, depth

    # ── heat ─────────────────────────────────────────────────────────────

    def heat(self, meta, now: Optional[float] = None, depth: int = 0) -> float:
        """How much this record deserves context space right now, in [0, ~1].

        Deliberately the same shape as the engine's own decay scorer, so a
        record's position in the pocket and its rank in a scored search cannot
        disagree:

            stickiness = 1 + ln(1 + recalls)     things used often age slower
            recency    = 0.5 ^ (age / stickiness / half_life)
            heat       = recency * importance

        A pinned record is always maximally hot. That is the escape hatch for
        the handful of facts an agent must never lose — a system prompt, a
        brand rule — without inventing a second storage path for them.

        Usage appears TWICE and deliberately. Through stickiness it slows decay,
        which is the engine's existing behaviour. It also enters directly,
        because recency alone cannot rank a working set: every record written in
        the same session has age about zero and therefore recency exactly 1.0,
        so without a usage term the pocket is ordered arbitrarily among
        everything stored today — which is most of what an agent holds.
        """
        if meta.get_attribute(_PINNED) == "true":
            return 1.0
        now = now if now is not None else time.time()
        age_days = max(0.0, (now - meta.timestamp) / 86400.0)
        stickiness = 1.0 + math.log(1.0 + meta.recall_count)
        recency = 0.5 ** (age_days / stickiness / self.half_life)
        usage = 1.0 - 0.5 ** (meta.recall_count / USAGE_HALF_LIFE)
        use_weight = UNUSED_FLOOR + (1.0 - UNUSED_FLOOR) * usage
        # Inherited memory is DISCOUNTED per level, not subordinated. A
        # brand-wide fact read across the whole fleet accumulates recalls that
        # say little about whether THIS agent needs it now, so each level of
        # distance costs it 30%. It can still outrank the agent's own note — and
        # should: a rule read twenty times is more likely to matter than a note
        # written once and never looked at again. The discount breaks ties in
        # favour of the local, it does not override evidence.
        inherit_weight = INHERIT_DECAY ** depth
        return (recency * use_weight * inherit_weight
                * max(0.0, min(1.0, meta.importance)))

    # ── HOT ──────────────────────────────────────────────────────────────

    def hot(self, budget_tokens: Optional[int] = None) -> list[PocketItem]:
        """The working set: the hottest records that fit the budget.

        No query, no embedding, no model call — this is a scan of one namespace
        ordered by a number already in memory. That is the point: an agent
        assembling its context should not have to pay for a search to find out
        what it already knows.
        """
        budget = budget_tokens if budget_tokens is not None else self.budget
        now = time.time()

        # Heat decays on a half-life measured in days, so a result up to a
        # second stale is indistinguishable from a fresh one. Bucketing by whole
        # seconds keeps the cache from being invalidated by the clock alone.
        ck = (budget, _gen(self.db), int(now))
        hit = self._hot_cache.get(ck)
        if hit is not None:
            return hit

        scored: list[PocketItem] = []
        seen: set[str] = set()
        for rid, sc, depth in self._scoped_ids():
            meta = self.db.get_metadata(rid)
            if meta is None or meta.source == "_forgotten":
                continue
            # Cache entries live in the same namespace but are not memories —
            # they must never spend the agent's context budget. importance=0.0
            # already zeroes their heat, but a generous budget would still admit
            # them at the tail, so exclude them by kind rather than by score.
            if meta.source == _CACHE_KIND:
                continue
            key = meta.entity_id or str(rid)
            # Shadowing: the agent's own version of a key wins over an inherited
            # one. Scopes are walked deepest first, so the first sighting is the
            # most specific — the same rule as a variable shadowing an outer one.
            if key in seen:
                continue
            seen.add(key)
            scored.append(PocketItem(
                id=rid,
                key=key,
                content=meta.content,
                heat=self.heat(meta, now, depth),
                tokens=estimate_tokens(meta.content),
                recalls=meta.recall_count,
                age_days=(now - meta.timestamp) / 86400.0,
                pinned=meta.get_attribute(_PINNED) == "true",
                scope=sc,
                inherited=depth > 0,
            ))

        scored.sort(key=lambda x: -x.heat)
        out, spent = [], 0
        for item in scored:
            if item.heat < self.min_heat:
                break             # sorted by heat, so nothing below is worth it
            if spent + item.tokens > budget:
                continue          # skip, don't stop — a long cold item should
                                  # not block several short hot ones behind it
            out.append(item)
            spent += item.tokens
        # One entry: the budget and generation both change rarely, and an
        # unbounded dict here would be a slow leak in a long-running agent.
        self._hot_cache = {ck: out}
        return out

    def hot_text(self, budget_tokens: Optional[int] = None, sep: str = "\n") -> str:
        """The working set as a single block, ready to drop into a prompt."""
        return sep.join(i.content for i in self.hot(budget_tokens))

    # ── writing ──────────────────────────────────────────────────────────

    def remember(
        self, key: str, content: str, *,
        importance: float = 1.0, pinned: bool = False,
        attributes: Optional[dict] = None,
    ) -> int:
        """Store a memory. Same key overwrites, so re-learning is not duplication."""
        rid = self._id(key)
        prev = self.db.get_metadata(rid)

        meta = feather_db.Metadata()
        meta.timestamp    = int(time.time())
        meta.namespace_id = self.namespace
        meta.entity_id    = key
        meta.content      = content
        meta.importance   = importance
        meta.source       = "pocket"
        if pinned:
            meta.set_attribute(_PINNED, "true")
        for k, v in (attributes or {}).items():
            meta.set_attribute(str(k), str(v))
        # Carry the recall history forward — an update is new information about
        # something the agent already cared about, not a reset of that interest.
        if prev is not None and prev.recall_count:
            meta.recall_count = prev.recall_count

        self.db.add(rid, self._vec(content), meta)
        _bump(self.db)
        return rid

    def pin(self, key: str, pinned: bool = True) -> None:
        rid = self._id(key)
        meta = self.db.get_metadata(rid)
        if meta is None:
            return
        meta.set_attribute(_PINNED, "true" if pinned else "false")
        self.db.update_metadata(rid, meta)
        _bump(self.db)

    def forget(self, key: str) -> None:
        self.db.forget(self._id(key))
        _bump(self.db)

    # ── WARM ─────────────────────────────────────────────────────────────

    def recall(self, query: str, k: int = 5) -> list[PocketItem]:
        """Search the whole namespace, including what is too cold for the pocket.

        A hit promotes: `search()` increments `recall_count`, so the next `hot()`
        reflects what the agent just needed. The working set follows real usage
        rather than a policy someone guessed at.
        """
        now = time.time()
        visible = {SCOPE_SEP.join(sc): d for d, sc in enumerate(self.scopes())}

        hits = []
        for flat, depth in visible.items():
            if self._embed is not None:
                f = FilterBuilder().namespace(flat).build()
                hits += [(h, depth) for h in self.db.search(
                    np.asarray(self._embed(query), dtype=np.float32), k=k, filter=f)]
            else:
                hits += [(h, depth) for h in self.db.keyword_search(query, k=k * 3)
                         if h.metadata.namespace_id == flat]
        # Each scope is searched separately, so raw scores are only comparable
        # within one. Normalise before ranking across them.
        top = max((h.score for h, _ in hits), default=1.0) or 1.0

        out, seen = [], set()
        for h, depth in hits:
            m = h.metadata
            if m.source == "_forgotten" or m.source == _CACHE_KIND:
                continue
            key = m.entity_id or str(h.id)
            if key in seen:
                continue
            seen.add(key)
            out.append(PocketItem(
                id=h.id, key=key, content=m.content,
                heat=self.heat(m, now, depth), tokens=estimate_tokens(m.content),
                recalls=m.recall_count,
                age_days=(now - m.timestamp) / 86400.0,
                pinned=m.get_attribute(_PINNED) == "true",
                scope=tuple(m.namespace_id.split(SCOPE_SEP)) if m.namespace_id else (),
                inherited=depth > 0,
                relevance=float(h.score) / top,
            ))
        # Rank by RELEVANCE, with heat as a modest tiebreaker.
        #
        # Sorting by heat alone made recall() ignore the query: a pinned record
        # has heat 1.0 by definition, so an org policy was returned first for
        # "which tiktok creative performed best" as readily as for "what should
        # I not claim". Heat answers "what should I be carrying"; relevance
        # answers "what did you ask for". Only hot() may confuse the two.
        out.sort(key=lambda x: -(x.relevance + RECALL_HEAT_WEIGHT * x.heat))
        # recall() looks like a read but IS a write: search() increments
        # recall_count on every hit, which changes heat and therefore the
        # working set. Without this the memoised hot() would keep serving a view
        # that predates the promotion — the exact thing promotion exists to do.
        if out:
            _bump(self.db)
        return out[:k]

    # ── CACHE ────────────────────────────────────────────────────────────

    def cache(self, key: str, compute: Callable[[], Any], *, ttl_seconds: float = 3600) -> Any:
        """Memoize an expensive call in the same file as the memory.

        Agent codebases grow a second, separate cache for tool results — a dict
        keyed on a hash, with no eviction and no notion of what is worth keeping.
        Putting it here means one store, one expiry mechanism, and cached results
        that can be inspected alongside everything else the agent knows.
        """
        rid = self._id(f"__cache__/{key}")
        meta = self.db.get_metadata(rid)
        if meta is not None and meta.source != "_forgotten":
            expires = float(meta.get_attribute(_CACHE_EXPIRY) or 0)
            if time.time() < expires:
                try:
                    return json.loads(meta.content)
                except Exception:
                    pass          # unparseable cache entry: recompute

        value = compute()
        m = feather_db.Metadata()
        m.timestamp    = int(time.time())
        m.namespace_id = self.namespace
        m.entity_id    = f"__cache__/{key}"
        m.content      = json.dumps(value, default=str)
        m.source       = _CACHE_KIND
        m.importance   = 0.0      # never competes with real memory for budget
        m.set_attribute(_CACHE_EXPIRY, str(time.time() + ttl_seconds))
        m.ttl = int(ttl_seconds)  # so forget_expired() reaps it
        self.db.add(rid, self._vec(key), m)
        _bump(self.db)
        return value

    def evict_expired(self) -> int:
        """Drop cache entries past their TTL. Real memories are untouched."""
        n, now = 0, time.time()
        for rid in self.db.ids_in_namespace(self.namespace):
            meta = self.db.get_metadata(rid)
            if meta is None or meta.source != _CACHE_KIND:
                continue
            if now >= float(meta.get_attribute(_CACHE_EXPIRY) or 0):
                self.db.forget(rid); n += 1
        if n:
            _bump(self.db)
        return n

    # ── introspection ────────────────────────────────────────────────────

    # ── lifecycle ────────────────────────────────────────────────────────

    def session(self, session_id: Optional[str] = None, **kw) -> "PocketSession":
        """Open a run. Use as a context manager so scratch is always cleaned up."""
        return PocketSession(self, session_id, **kw)

    def maintain(self) -> dict:
        """Periodic upkeep: reap expired cache and abandoned session scratch.

        A crashed agent leaves its scratch scope behind. Nothing reads it —
        session scopes are children, so only that session's pocket saw them —
        but they occupy the file and the index until something removes them.
        """
        evicted = self.evict_expired()
        orphans = 0
        mine = SCOPE_SEP.join(self.scope) + SCOPE_SEP + "_session" + SCOPE_SEP
        for flat in self.db.list_namespaces():
            if not flat.startswith(mine):
                continue
            for rid in self.db.ids_in_namespace(flat):
                self.db.forget(rid); orphans += 1
        if orphans:
            _bump(self.db)
        return {"cache_evicted": evicted, "orphan_scratch_dropped": orphans}

    def stats(self) -> dict:
        now = time.time()
        hot = self.hot()
        hot_ids = {i.id for i in hot}
        total = cache = warm_tokens = 0
        per_scope: dict[str, int] = {}
        for rid, sc, _depth in self._scoped_ids():
            meta = self.db.get_metadata(rid)
            if meta is None or meta.source == "_forgotten":
                continue
            total += 1
            flat = SCOPE_SEP.join(sc)
            per_scope[flat] = per_scope.get(flat, 0) + 1
            if meta.source == _CACHE_KIND:
                cache += 1
            elif rid not in hot_ids:
                warm_tokens += estimate_tokens(meta.content)
        return {
            "scope":        self.namespace,
            "inherits":     [SCOPE_SEP.join(s) for s in self.scopes()[1:]],
            "budget":       self.budget,
            "hot_items":    len(hot),
            "hot_tokens":   sum(i.tokens for i in hot),
            "hot_own":      sum(1 for i in hot if not i.inherited),
            "hot_inherited":sum(1 for i in hot if i.inherited),
            "warm_items":   total - len(hot) - cache,
            "warm_tokens":  warm_tokens,
            "cache_items":  cache,
            "total_items":  total,
            "per_scope":    per_scope,
        }

    # ── internals ────────────────────────────────────────────────────────

    def _id(self, key: str) -> int:
        """Deterministic, so `remember()` on the same key updates in place."""
        h = hashlib.sha1(f"{self.namespace}\x00{key}".encode()).hexdigest()
        return int(h[:14], 16) & ((1 << 53) - 1)

    def _vec(self, text: str) -> np.ndarray:
        dim = self.db.dim("text")
        if self._embed is not None:
            return np.asarray(self._embed(text), dtype=np.float32)
        v = np.zeros(dim, dtype=np.float32); v[0] = 1.0   # valid, never zero-norm
        return v


def pocket(db, namespace: str, **kw) -> Pocket:
    """`from feather_db.pocket import pocket` → `pocket(db, "agent_x")`."""
    return Pocket(db, namespace, **kw)


# ─────────────────────────────────────────────────────────────────────────
# Lifecycle
# ─────────────────────────────────────────────────────────────────────────

class PocketSession:
    """One agent run, with a scratch tier that does not pollute long-term memory.

    An agent run produces two very different kinds of writing. There is what it
    *learned* — worth keeping, worth carrying into the next run. And there is
    what it was *thinking* — intermediate results, tool output, a plan it
    revised twice. Both get written to memory by agents that have only one
    place to put things, and the second kind is what turns a useful memory into
    a landfill within a week.

    A session gives the scratch its own scope, so it is real memory during the
    run — searchable, rankable, budgeted alongside everything else — and is then
    dropped at the end unless the agent explicitly promoted it.

        with p.session() as s:
            s.note("tried carousel, CPA 4x")      # scratch, this run only
            ctx = s.context()                     # hot text incl. scratch
            s.keep("finding", "carousel fails for Brand A")   # promote
        # scratch is gone; the promoted finding is in the agent's own scope

    Nothing here is automatic magic: `keep()` is an explicit decision, because
    an agent that guesses what is worth remembering guesses wrong, and the cost
    of guessing wrong accumulates.
    """

    def __init__(self, parent: "Pocket", session_id: Optional[str] = None,
                 scratch_ratio: float = 0.4):
        self.parent = parent
        self.id = session_id or hashlib.sha1(
            f"{parent.namespace}{time.time_ns()}".encode()).hexdigest()[:10]
        # Scratch is a CHILD scope, so it inherits everything the agent can see
        # and is ranked against it, while remaining trivially droppable.
        self.scratch = Pocket(
            parent.db, tuple(parent.scope) + ("_session", self.id),
            budget_tokens=parent.budget,
            half_life_days=parent.half_life,
            min_heat=parent.min_heat,
            embed=parent._embed,
        )
        # Scratch is fresh and therefore hot; without a cap one chatty run would
        # push every durable memory out of the context window.
        self.scratch_ratio = scratch_ratio
        self._kept: list[str] = []
        self._closed = False

    # ── during the run ───────────────────────────────────────────────────

    def note(self, text: str, *, key: Optional[str] = None, importance: float = 0.6) -> str:
        """Write working memory for this run only."""
        key = key or f"n{len(self._kept) + int(time.time_ns() % 100000)}"
        self.scratch.remember(key, text, importance=importance)
        return key

    def keep(self, key: str, text: str, *, importance: float = 1.0,
             pinned: bool = False) -> None:
        """Promote something to the agent's durable memory. Survives close()."""
        self.parent.remember(key, text, importance=importance, pinned=pinned)
        self._kept.append(key)

    def recall(self, query: str, k: int = 5) -> list[PocketItem]:
        """Search durable memory and this run's scratch together."""
        return self.scratch.recall(query, k=k)

    def context(self, budget_tokens: Optional[int] = None) -> str:
        """The prompt block: durable memory plus this run's scratch, budgeted.

        Scratch is capped at `scratch_ratio` of the budget. Everything written
        this run has age ~0 and therefore maximal recency, so without the cap a
        long run would evict the brand rules it is supposed to be following.
        """
        budget = budget_tokens if budget_tokens is not None else self.parent.budget
        scratch_budget = int(budget * self.scratch_ratio)

        own_scope = SCOPE_SEP.join(self.scratch.scope)
        scratch_items, durable_items = [], []
        for item in self.scratch.hot(budget):
            (scratch_items if SCOPE_SEP.join(item.scope) == own_scope
             else durable_items).append(item)

        out, spent = [], 0
        for item in scratch_items:
            if spent + item.tokens > scratch_budget:
                continue
            out.append(item); spent += item.tokens
        for item in durable_items:
            if spent + item.tokens > budget:
                continue
            out.append(item); spent += item.tokens

        out.sort(key=lambda x: -x.heat)
        return "\n".join(i.content for i in out)

    # ── ending the run ───────────────────────────────────────────────────

    def close(self, *, discard_scratch: bool = True) -> dict:
        """End the run: drop the scratch, reap expired cache, report.

        Called automatically by the context manager. Idempotent.
        """
        if self._closed:
            return {"session": self.id, "already_closed": True}
        dropped = 0
        if discard_scratch:
            for rid in self.parent.db.ids_in_namespace(SCOPE_SEP.join(self.scratch.scope)):
                self.parent.db.forget(rid); dropped += 1
        if dropped:
            _bump(self.parent.db)
        evicted = self.parent.evict_expired()
        self.parent.db.save()
        self._closed = True
        return {
            "session": self.id,
            "scratch_dropped": dropped,
            "kept": list(self._kept),
            "cache_evicted": evicted,
        }

    def __enter__(self) -> "PocketSession":
        return self

    def __exit__(self, *exc) -> None:
        # Close even on an exception: a crashed run must not leave its scratch
        # behind to be inherited by the next one as if it were durable memory.
        self.close()
