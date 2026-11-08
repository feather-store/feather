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

import json
import math
import re
import time
import uuid
import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import numpy as np

import feather_db
from feather_db import clock as _clock
from feather_db import keys
from feather_db import origin as _origin

try:                                       # the engine's, so the two cannot drift
    from feather_db.core import RECALL_HALF_LIFE_DAYS
except ImportError:                        # pragma: no cover - a core built before it
    RECALL_HALF_LIFE_DAYS = 30.0
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


# Running ingest queues (feather_db.ingest), per DB: a read that returns a
# memory indexed by another extractor tells its queue, which re-makes that
# index first. Reads do no I/O for it (stale_seen only queues).
_INDEXERS: dict = {}


def register_indexer(db, queue) -> None:
    _INDEXERS[id(db)] = queue


def unregister_indexer(db, queue) -> None:
    if _INDEXERS.get(id(db)) is queue:
        del _INDEXERS[id(db)]


def note_stale(db, metas) -> None:
    """Called by reads with the memories they return: nothing unless a queue
    runs on `db`, then one attribute comparison per memory."""
    if not _INDEXERS:
        return
    q = _INDEXERS.get(id(db))
    if q is None:
        return
    for m in metas:
        if m is not None and m.get_attribute(INDEX_V):
            q.stale_seen(m)


# DB.erase() removes records a pocket may hold in its memoised hot() view, and
# that view must not hand erased text back, even for the second it lives. The
# engine knows nothing of pockets, so the binding is wrapped here. RemoteDB's
# erase() bumps the same way.
_core_erase = feather_db.DB.erase


def _erase(self, *args, **kwargs):
    dry = kwargs.get("dry_run", False)
    try:
        return _core_erase(self, *args, **kwargs)
    finally:                      # an erase that raised may have applied already
        if not dry:
            _after_erase(self)


def _after_erase(db) -> None:
    """What an erasure must also clear in this process: every pocket's
    memoised view of `db`, and every embedding cache (a vector per text,
    keyed by its hash). Embedders are only imported if something made one."""
    _bump(db)
    import sys
    emb = sys.modules.get("feather_db.integrations.embedders")
    if emb is not None:
        emb.clear_caches()


_erase.__doc__ = _core_erase.__doc__
feather_db.DB.erase = _erase

_CACHE_KIND   = "_pocket_cache"
_CACHE_EXPIRY = "_pocket_expires_at"
_PINNED       = "_pocket_pinned"
# Prospective memory: an intention (a reminder, an obligation) is held back
# from the working set until this unix time, then carried as if pinned until
# it is rewritten without one or forgotten. FeatherStore maps a numeric
# `due_at` in an item's value onto it, so AgentMemory's intentions use it too.
DUE_ATTR      = "_pocket_due_at"
# Set with a due time. The engine indexes attributes by exact value, so this
# finds every intention without a scan (a due time differs per record).
INTENTION_ATTR = "_pocket_intention"

# Index records (facts as index, memories as payload). What an extractor
# derives from a memory finds it without replacing it: `remember(index=...)`
# writes each statement as a record of its own, in the memory's scope, whose
# search hit counts for the memory. A reader never sees one. On LoCoMo,
# memory made of extracted facts scored J 64 against raw turns' 78.8; raw
# turns found through their facts as well scored 82.4 (bench.locomo
# --memory index), same reader, same context size.
_INDEX_KIND = "_pocket_index"
INDEX_FOR   = "_pocket_index_for"   # on an index record: its memory's key
INDEX_N     = "_pocket_index_n"     # on a memory: how many index records it has
# Supersession on raw memory. An index entry with a one-value key ("alice|home
# city") puts its memory in that key's chain; recall() shows a chain once,
# its newest statement with the older ones under it as history, and brings
# the newest in when only an older one matched. SUPKEY (on the index record)
# is "<namespace>|<key>", so the engine's exact attribute index finds a chain
# without a scan; SUPKEYS (on the memory) lists its keys.
SUPKEY      = "_pocket_supkey"
SUPVAL      = "_pocket_supval"
SUPKEYS     = "_pocket_supkeys"
_KEY_SEP    = "\x1f"
MAX_EARLIER = 5                      # history lines carried under one memory
# The index is derived data: on the memory, which extractor derived it
# (`name@version+fingerprint`) and which ingest job derives it again
# (feather_db.ingest), so a new extractor re-indexes what the old one made.
INDEX_V        = "_pocket_index_v"
INDEX_JOB      = "_pocket_index_job"
INDEX_JOB_ARGS = "_pocket_index_job_args"
INDEX_ATTRS    = (INDEX_N, SUPKEYS, INDEX_V, INDEX_JOB, INDEX_JOB_ARGS)
# The key registry: one record per one-value key a scope's index holds, found
# by the engine's exact attribute index (KEYREG_NS = the scope), so
# known_keys() reads O(keys), not the scope. Every index record with a key
# links to its entry (REGISTERED_IN), so an erasure of the data takes the
# key's name with it: a key can name a person ("alice|employer").
_KEYREG_KIND  = "_pocket_keyreg"
KEYREG_NS     = "_pocket_keyreg_ns"
KEYREG_KEY    = "_pocket_keyreg_key"
REGISTERED_IN = "registered_in"
# A pending ingest job (feather_db.ingest): durable, in the scope it indexes.
_JOB_KIND     = "_pocket_job"
# Which extractors made the indexes in a scope: one registry entry per
# (scope, extractor id), all found by IXVER_ALL, so a sweep finds the stale
# ones by extractor without reading any scope.
IXVER         = "_pocket_ixver"
IXVER_ALL     = "_pocket_ixver_all"


def due_at(meta) -> Optional[float]:
    """The record's due time (unix seconds), or None if it is not an intention."""
    raw = meta.get_attribute(DUE_ATTR) if meta is not None else ""
    try:
        return float(raw) if raw else None
    except ValueError:
        return None

# Scopes are hierarchical, joined with "." — the same convention FeatherStore
# uses, so a namespace written by one is readable by the other.
SCOPE_SEP = "."

# An agent's own memory outranks what it inherits. Without this a brand-wide
# fact written months ago and read by every agent would outrank the note this
# agent made ten minutes ago, purely because it has more recalls across the
# fleet. Depth is per level of inheritance distance.
INHERIT_DECAY = 0.7


def scoped_id(namespace: str, key: str) -> int:
    """The id a pre-0.22 pocket derived for `key` in `namespace` (a 53-bit SHA-1).

    Kept for code that still calls it. It no longer places records: two keys
    that collided shared one, and the second write overwrote the first. Records
    are now addressed by native key — see feather_db.keys.resolve().
    """
    return keys.legacy_id(namespace, key)


# (id(db), namespace) pairs already migrated in this process: migrate() is
# idempotent, but it reads every record of the scope.
_MIGRATED: set = set()


def _scope(scope) -> tuple[str, ...]:
    """Accept "a.b.c", ("a","b","c") or "a" and normalise to a tuple."""
    if isinstance(scope, str):
        return tuple(x for x in scope.split(SCOPE_SEP) if x)
    return tuple(scope)


def effective_recalls(meta, now: float) -> float:
    """`recall_count` as of now: n * 0.5^(days since last recall / H), with
    H = RECALL_HALF_LIFE_DAYS, the engine's (`Scorer::effective_recalls`).

    `recall_count` counts recent recalls: each recall fades the count so far
    at H before adding one (`Metadata::note_recall`); this fades the rest. One
    H for every reader, so the pocket and scored search credit a record's use
    alike; only their recency half-lives differ. A count with no recall time
    fades from the record's timestamp."""
    n = meta.recall_count
    if not n:
        return 0.0
    at = meta.last_recalled_at or max(0, meta.timestamp)
    since_days = max(0.0, now - at) / 86400.0
    return n * 0.5 ** (since_days / RECALL_HALF_LIFE_DAYS)


def _overrides(origin: str, held: str, trusted) -> bool:
    """Does a record of `origin` from a further scope replace the nearer one of
    the same key, of origin `held`? Only when it is trusted and that one is not.
    Nearest wins otherwise; but if an untrusted record could shadow a trusted
    one, writing a key in the agent's own scope would hide the org's record of
    it from every context."""
    return origin in trusted and held not in trusted


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


# FeatherStore keeps an item's JSON value here (langgraph_store._VALUE_ATTR).
_LG_VALUE = "_lg_value"


def display_text(meta) -> str:
    """What a prompt shows for a record: its content, or for a FeatherStore
    item whose value has a `text` string, that text.

    A FeatherStore record's content is its indexed text, by default every
    field of the value ("kind: facts\\nrecorded_at: ...\\nsupersedes: [...]"),
    which is right to search and wrong to read: a keyed fact rendered its
    whole history, ~10x the tokens and every superseded value with it
    (bench.contradiction). Its date goes on the line instead (`date_tag`)."""
    raw = meta.get_attribute(_LG_VALUE)
    if raw:
        try:
            text = json.loads(raw).get("text")
        except (ValueError, AttributeError):
            text = None
        if isinstance(text, str) and text:
            return text
    return meta.content


def _day(t: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(t))


# What date_tag() puts before an item, for a reader that parses rendered text.
DATE_TAG_RE = re.compile(r"\[(?:due )?\d{4}-\d{2}-\d{2}\] ")


def date_tag(item) -> str:
    """'[YYYY-MM-DD] ' (UTC): when the memory happened or became true, its
    valid time; '[due YYYY-MM-DD] ' for an intention; '' when unknown.

    Without dates a reader handed "Alice lives in Denver" and "Alice moved to
    Lisbon" cannot tell which is current, and answered with the stale value
    about half the time (bench.contradiction)."""
    if getattr(item, "due_at", None):
        return f"[due {_day(item.due_at)}] "
    t = getattr(item, "valid_time", 0)
    return f"[{_day(t)}] " if t and t > 0 else ""


# Records that live in a scope but are not memory: never carried, never
# returned, never resolved as a required key.
_NOT_MEMORY = ("_forgotten", _CACHE_KIND, _INDEX_KIND, _KEYREG_KIND, _JOB_KIND)
# Derived or bookkeeping records: never an item, never carried.
DERIVED_KINDS = (_INDEX_KIND, _KEYREG_KIND, _JOB_KIND)


@dataclass(frozen=True)
class IndexEntry:
    """One way to find a memory (`Pocket.remember(index=...)`).

    `text` is searched in place of the memory and never shown. `key`
    ('subject|attribute') says the memory states a value of something that
    holds one value at a time, `value` which one, `valid_at` since when
    (default: the memory's valid time): the newest value of a key is
    current, and the memories holding older ones become its history. A
    SessionFactExtractor `Fact` converts to one (`IndexEntry.of`): its
    statement, and its key when it `replaces()`."""
    text: str = ""
    key: Optional[str] = None
    value: str = ""
    valid_at: Optional[float] = None

    @staticmethod
    def of(x) -> "IndexEntry":
        if isinstance(x, IndexEntry):
            return x
        if isinstance(x, str):
            return IndexEntry(text=x)
        if hasattr(x, "predicate") and hasattr(x, "object"):      # an extractor Fact
            from feather_db.extractors.session_facts import fact_key, replaces
            md = x.metadata or {}
            return IndexEntry(
                text=md.get("statement") or f"{x.subject} {x.predicate}: {x.object}",
                key=fact_key(x) if replaces(x) else None,
                value=str(x.object or ""),
                valid_at=float(x.valid_at) if x.valid_at is not None else None)
        raise TypeError(f"an index entry is an IndexEntry, a str or a Fact, not {type(x)!r}")


def _norm_key(key: str) -> str:
    """'User | Home City' -> 'user|home city': the identity two entries share."""
    parts = [" ".join(re.sub(r"[^\w\s']", " ", p.lower()).split())
             for p in str(key).split("|")]
    return "|".join(p for p in parts if p)


def _norm_value(v: str) -> str:
    return " ".join(str(v).lower().split())


# ── index records and supersession chains, for any (db, namespace) ─────
# Pocket and FeatherStore both use these: a memory is identified by
# (namespace, key), its index records by `<key>\x1fix<i>` in its namespace.

def _index_key(key: str, i: int) -> str:
    return f"{key}{_KEY_SEP}ix{i}"


def index_supkeys(namespace: str, entries) -> list[str]:
    """The SUPKEYS a memory with `entries` carries: '<namespace>|<key>' each."""
    return list(dict.fromkeys(f"{namespace}|{_norm_key(e.key)}"
                              for e in entries if e.key and _norm_key(e.key)))


def _index_meta(namespace: str, key: str, e, memory, i: int):
    """The metadata of the memory's index record i for entry `e`. Its entity
    is the memory's key, so an erase by entity takes it too; its origin is
    the memory's."""
    m = feather_db.Metadata()
    m.timestamp = int(time.time())
    m.namespace_id = namespace
    m.key = _index_key(key, i)
    m.entity_id = key
    m.source = _INDEX_KIND
    m.importance = memory.importance
    k = _norm_key(e.key) if e.key else ""
    m.content = e.text or (f"{k.replace('|', ' ')}: {e.value}" if k else "")
    at = e.valid_at if e.valid_at is not None else memory.valid_time
    if at:
        m.valid_at = int(at)
    m.set_attribute(INDEX_FOR, key)
    m.set_attribute(_origin.ATTR, _origin.of(memory))
    if k:
        m.set_attribute(SUPKEY, f"{namespace}|{k}")
        m.set_attribute(SUPVAL, _norm_value(e.value))
    return m


def _reg_key(k: str) -> str:
    return f"{_KEY_SEP}keyreg{_KEY_SEP}{k}"


def _registry_meta(namespace: str, k: str):
    m = feather_db.Metadata()
    m.timestamp = int(time.time())
    m.namespace_id = namespace
    m.key = _reg_key(k)
    m.source = _KEYREG_KIND
    m.importance = 0.0
    m.set_attribute(KEYREG_NS, namespace)
    m.set_attribute(KEYREG_KEY, k)
    return m


def _version_meta(namespace: str, extractor_id: str):
    m = feather_db.Metadata()
    m.timestamp = int(time.time())
    m.namespace_id = namespace
    m.key = f"{_KEY_SEP}ixver{_KEY_SEP}{extractor_id}"
    m.source = _KEYREG_KIND
    m.importance = 0.0
    m.set_attribute(IXVER, extractor_id)
    m.set_attribute(IXVER_ALL, "1")
    return m


def _index_pos(meta) -> int:
    """i of an index record keyed `<memory key>\x1fix<i>`; -1 if not one."""
    tail = (meta.key or "").rsplit(_KEY_SEP + "ix", 1)
    return int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else -1


def _stale_index_ids(db, namespace: str, key: str, keep: int) -> list[int]:
    """Ids of the memory's live index records at i >= keep: one attribute
    lookup and one metadata read, wherever the records are."""
    ids = list(db.ids_with_attribute(INDEX_FOR, key))
    metas = db.get_metadata_many(ids) if ids else []
    return [rid for rid, m in zip(ids, metas)
            if m is not None and m.source == _INDEX_KIND and m.namespace_id == namespace
            and _index_pos(m) >= keep]


@dataclass
class IndexWrite:
    """One memory's index, for apply_index(). `memory` is the memory's
    metadata as it is to be written (apply_index sets its index attributes).
    With `vec` the memory itself is written too (upsert by key); without, its
    record is updated in place, `expected_version` making that conditional:
    a memory rewritten since it was read is not given an index derived from
    its old text."""
    namespace: str
    key: str
    entries: list
    memory: Any
    vec: Any = None
    expected_version: Optional[int] = None
    old_n: int = 0


def apply_index(db, writes: Iterable[IndexWrite], vec: Callable[[str], Any], *,
                extractor_id: Optional[str] = None, job_ref: Optional[str] = None,
                job_args: Optional[str] = None):
    """Write memories' indexes, ALL in one atomic WriteBatch: one round trip
    over the Cloud API, where index records written one by one cost two or
    three each. Per memory: its index records (upsert by key), the registry
    entry of each one-value key and a REGISTERED_IN link to it, its index
    attributes (INDEX_N, SUPKEYS, and INDEX_V / INDEX_JOB when given), and
    the forgetting of a previous index's records beyond the new one's (read
    first, only when the index shrinks: a batch refuses to forget a key that
    is not there). Returns the BatchResult."""
    b = feather_db.WriteBatch()
    versioned = set()
    for w in writes:
        if extractor_id is not None and w.namespace not in versioned:
            b.upsert(_version_meta(w.namespace, extractor_id))
            versioned.add(w.namespace)
        entries = [IndexEntry.of(x) for x in w.entries]
        m = w.memory
        m.namespace_id, m.key = w.namespace, w.key
        m.set_attribute(INDEX_N, str(len(entries)))
        m.set_attribute(SUPKEYS, _KEY_SEP.join(index_supkeys(w.namespace, entries)))
        if extractor_id is not None:
            m.set_attribute(INDEX_V, extractor_id)
        if job_ref is not None:
            m.set_attribute(INDEX_JOB, job_ref)
            m.set_attribute(INDEX_JOB_ARGS, job_args or "")
        if w.vec is not None:
            b.upsert(m, w.vec, expected_version=w.expected_version)
        else:
            b.update((w.namespace, w.key), m, expected_version=w.expected_version)
        registered = set()
        for i, e in enumerate(entries):
            im = _index_meta(w.namespace, w.key, e, m, i)
            b.upsert(im, vec(im.content or m.content))
            k = _norm_key(e.key) if e.key else ""
            if k:
                if k not in registered:
                    b.upsert(_registry_meta(w.namespace, k))
                    registered.add(k)
                b.link((w.namespace, im.key), (w.namespace, _reg_key(k)), REGISTERED_IN)
        if w.old_n > len(entries):
            for rid in _stale_index_ids(db, w.namespace, w.key, len(entries)):
                b.forget(rid)
    r = db.write(b) if len(b) else None
    _bump(db)
    return r


def write_index(db, namespace: str, key: str, entries, memory, old_n: int,
                vec: Callable[[str], Any]) -> None:
    """The memory at (namespace, key), whose metadata is `memory`, indexed by
    `entries`, replacing an index of `old_n` records: apply_index() for one
    memory, updated in place."""
    apply_index(db, [IndexWrite(namespace, key, list(entries), memory, old_n=old_n)], vec)


def forget_index(db, namespace: str, key: str, n: int, start: int = 0) -> None:
    """Forget the memory's index records start..n-1 (one lookup, one batch)."""
    if n <= start:
        return
    ids = _stale_index_ids(db, namespace, key, start)
    if ids:
        b = feather_db.WriteBatch()
        for rid in ids:
            b.forget(rid)
        db.write(b)


def known_keys(db, namespace: str) -> list[str]:
    """The one-value keys ('subject|attribute') indexed in `namespace`,
    sorted: what SessionFactExtractor takes as `known_keys`. Reads the key
    registry: one attribute lookup and one metadata read, O(keys). A key
    stays known after its last memory is forgotten (the name is still the
    right one for that attribute); an erasure of its data removes it."""
    ids = list(db.ids_with_attribute(KEYREG_NS, namespace))
    metas = db.get_metadata_many(ids) if ids else []
    return sorted({m.get_attribute(KEYREG_KEY) for m in metas
                   if m is not None and m.source == _KEYREG_KIND
                   and m.get_attribute(KEYREG_NS) == namespace and m.get_attribute(KEYREG_KEY)})


def backfill_registry(db, namespace: str, ids: list, metas: list) -> int:
    """Registry entries and REGISTERED_IN links for index records written
    before the registry existed, from a scope's records already read (the
    keys.migrate pass on open: no read of its own). Returns how many entries
    it wrote."""
    have = {m.get_attribute(KEYREG_KEY) for m in metas
            if m is not None and m.source == _KEYREG_KIND}
    b, new = feather_db.WriteBatch(), set()
    for rid, m in zip(ids, metas):
        if m is None or m.source != _INDEX_KIND or not m.get_attribute(SUPKEY):
            continue
        sk = m.get_attribute(SUPKEY)
        if not sk.startswith(namespace + "|"):
            continue
        k = sk[len(namespace) + 1:]
        if k not in have and k not in new:
            b.upsert(_registry_meta(namespace, k))
            new.add(k)
        if not any(e.rel_type == REGISTERED_IN for e in m.edges):
            b.link(rid, (namespace, _reg_key(k)), REGISTERED_IN)
    if len(b):
        db.write(b)
    return len(new)


def memories_of(db, pairs) -> dict:
    """{(namespace, key): (id, metadata)} for the live records of `pairs`,
    one metadata read."""
    pairs = [p for p in pairs if p[1]]
    found = list(zip(pairs, keys.resolve_many(db, pairs)))
    found = [(p, rid) for p, rid in found if rid is not None]
    metas = db.get_metadata_many([rid for _, rid in found]) if found else []
    return {p: (rid, m) for (p, rid), m in zip(found, metas)
            if m is not None and m.source not in _NOT_MEMORY}


def index_targets(db, metas) -> dict:
    """For the index records among `metas`, {(namespace, key of the memory
    it indexes): (id, metadata of that memory)}."""
    return memories_of(db, {(m.namespace_id, m.get_attribute(INDEX_FOR))
                            for m in metas if m is not None and m.source == _INDEX_KIND})


def supkeys_of(meta) -> tuple[str, ...]:
    raw = meta.get_attribute(SUPKEYS) if meta is not None else ""
    return tuple(raw.split(_KEY_SEP)) if raw else ()


def chain_of(db, supkey: str) -> list:
    """[(order, value, namespace, key, id, metadata)] of the live memories
    holding a value of `supkey`, oldest first: by the index entry's valid
    time, then the memory's, then its version. One entry per memory."""
    ids = list(db.ids_with_attribute(SUPKEY, supkey))
    latest: dict = {}
    for m in db.get_metadata_many(ids) if ids else []:
        if m is None or m.source != _INDEX_KIND:
            continue
        k = (m.namespace_id, m.get_attribute(INDEX_FOR))
        if k not in latest or m.valid_time >= latest[k][0]:
            latest[k] = (m.valid_time, m.get_attribute(SUPVAL))
    found = memories_of(db, latest)
    chain = [((valid, m.valid_time, m.version), value, ns, key, rid, m)
             for (ns, key), (valid, value) in latest.items()
             for rid, m in [found.get((ns, key), (None, None))] if m is not None]
    chain.sort(key=lambda c: c[0])
    return chain


def group_records(db, records, *, trusted, make, cap: int = MAX_EARLIER) -> list:
    """Supersession for any reader. `records`: [((namespace, key), supkeys,
    origin, obj)] in rank order. Returns [(obj, [earlier objs, newest
    first], source obj)]: each one-value key's chain once, as the memory
    holding its newest value, at the place of the first record of the chain
    (`source obj`); an older memory that matched brings the newest in.
    `make(namespace, key, id, metadata)` builds an obj for a memory not in
    `records` (None to leave it out).

    The current value is the newest TRUSTED one when the chain holds any:
    an untrusted memory never makes a trusted one history. A memory that
    repeats the current value is not history."""
    records = list(records)
    if not any(r[1] for r in records):
        return [(r[3], [], r[3]) for r in records]
    have = {r[0]: r for r in records}
    chains: dict = {}

    def resolve(supkey):
        """((ident, supkeys, obj) current, [(ident, obj)] older, oldest first)."""
        if supkey not in chains:
            chain = []
            for _order, value, ns, key, rid, m in chain_of(db, supkey):
                ident = (ns, key)
                obj = have[ident][3] if ident in have else make(ns, key, rid, m)
                if obj is not None:
                    chain.append((value, ident, _origin.of(m), supkeys_of(m), obj))
            if not chain:
                chains[supkey] = None
            else:
                pool = [c for c in chain if c[2] in trusted] or chain
                cur = pool[-1]
                older = [(c[1], c[4]) for c in chain[:chain.index(cur)]
                         if c[0] != cur[0] and c[1] != cur[1]]
                chains[supkey] = ((cur[1], cur[3], cur[4]), older)
        return chains[supkey]

    out, placed, shown = [], set(), set()
    for ident, supkeys, _origin_, obj in records:
        if ident in shown:
            continue
        target = (ident, supkeys, obj)
        for k in supkeys:
            got = resolve(k)
            if got and got[0][0] != ident and any(o[0] == ident for o in got[1]):
                target = got[0]
                break
        if target[0] in placed:
            continue                          # its chain is already shown
        history: dict = {}
        for k in target[1]:
            got = resolve(k)
            if got and got[0][0] == target[0]:
                for o_ident, o_obj in reversed(got[1]):
                    if o_ident not in shown and o_ident not in history:
                        history[o_ident] = o_obj
        kept = list(history.items())[:cap]
        placed.add(target[0])
        shown.add(target[0])
        shown.update(i for i, _ in kept)
        out.append((target[2], [o for _, o in kept], obj))
    return out


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
    version: int = 0                 # the record's version: moves on writes, never on reads
    origin: str = _origin.USER       # who it came from: user, agent, tool or external
    due_at: Optional[float] = None   # an intention's due time (unix seconds); None otherwise
    valid_time: float = 0.0          # when it happened or became true (unix seconds)
    supkeys: tuple[str, ...] = ()    # its one-value keys ("<namespace>|subject|attribute")
    # Memories this one supersedes, newest first: history, shown under it.
    earlier: tuple["PocketItem", ...] = ()
    # The index facts a recall() matched it by: what it was found for
    # (PacketBuilder(select="coverage") counts information by them).
    facts: tuple[str, ...] = ()

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
        trusted_origins: Iterable[str] = _origin.TRUSTED,
    ):
        """`trusted_origins` (default: `user` only) are the origins this pocket
        renders as plain text, lets a pin lift, and lets shadow a record of the
        same key; `PacketBuilder` takes the same set unless told otherwise.
        Widen it only for a source an attacker cannot write into; see
        feather_db.origin."""
        self.scope = _scope(scope)
        self.trusted = frozenset(_origin.check(o) for o in trusted_origins)
        self.namespace = SCOPE_SEP.join(self.scope)   # where writes land
        self.db = db
        self.budget = budget_tokens
        self.half_life = half_life_days
        self.min_heat = min_heat
        self.inherit = inherit
        self._embed = embed
        self._hot_cache: dict = {}
        # Records written before native keys get theirs, so they are found by
        # key — keyword search included — like everything written since.
        if (id(db), self.namespace) not in _MIGRATED and not db.is_read_only():
            # The same read backfills the key registry of an index written
            # before it existed.
            if keys.migrate(db, [self.namespace], visit=lambda ns, ids, metas:
                            backfill_registry(db, ns, ids, metas)):
                _bump(db)
            _MIGRATED.add((id(db), self.namespace))

    # ── scope resolution ─────────────────────────────────────────────────

    def scopes(self) -> list[tuple[str, ...]]:
        """This scope and its ancestors, deepest first."""
        if not self.inherit:
            return [self.scope]
        return [self.scope[:i] for i in range(len(self.scope), 0, -1)]

    def _scoped_records(self):
        """(id, metadata, scope, depth) for everything this pocket can see;
        metadata is None for a record that is gone.

        depth 0 is the agent's own scope; 1 is its parent, and so on. Feather's
        namespace index is exact-match, so each level is looked up separately
        rather than by prefix — O(levels) lookups, each still the O(matches)
        index hit. Each level's records are read in one call: per record, a
        RemoteDB paid one HTTP round trip each (282 ms for 500 on localhost).
        """
        for depth, sc in enumerate(self.scopes()):
            ids = list(self.db.ids_in_namespace(SCOPE_SEP.join(sc)))
            if ids:
                for rid, meta in zip(ids, self.db.get_metadata_many(ids)):
                    yield rid, meta, sc, depth

    # ── heat ─────────────────────────────────────────────────────────────

    def heat(self, meta, now: Optional[float] = None, depth: int = 0) -> float:
        """How much this record deserves context space right now, in [0, ~1].

        The same shape as the engine's decay scorer (scoring.h), with the same
        inputs: age from `valid_time`, and recalls faded at the one salience
        half-life H (30 days) every reader shares. Only the recency half-life
        differs: a Pocket's (7 days) is shorter than `ScoringConfig`'s (30),
        because a working set should turn over faster than a search ranking:

            recalls    = n * 0.5 ^ (days since last recall / H)
            stickiness = 1 + ln(1 + recalls)     things used often age slower
            recency    = 0.5 ^ (age / stickiness / half_life)
            heat       = recency * importance

        Recalls fade from the last one, as in the engine (`effective_recalls`
        in scoring.h): a record used heavily months ago and not since is not
        hot now. Counted raw, a count only ever grew, so it never was.

        A pinned record is always maximally hot. That is the escape hatch for
        the handful of facts an agent must never lose — a system prompt, a
        brand rule — without inventing a second storage path for them. Only a
        trusted record (feather_db.origin) is lifted by its pin: a pin puts a
        record in every context from then on, and that must not be something
        an injected page can talk the agent into. An untrusted pinned record
        ranks by heat like any other.

        An intention (a record with a due time, `remember(due_at=...)`) is a
        pin set in advance: heat 0 until it is due, so it stays out of the
        working set, then 1.0 until it is rewritten without one or forgotten.
        Only for a trusted record, for the same reason as a pin.

        Nor does use lift an untrusted record: its recalls are not counted.
        Planted text stuffed with every topic's words matches nearly every
        query, so counting its use let it buy its way into the working set
        (bench.poisoning: in 81% of hot sets against 51% uncounted, while
        benign records went 58% -> 62%). Its age and importance still rank it.

        Usage appears TWICE and deliberately. Through stickiness it slows decay,
        which is the engine's existing behaviour. It also enters directly,
        because recency alone cannot rank a working set: every record written in
        the same session has age about zero and therefore recency exactly 1.0,
        so without a usage term the pocket is ordered arbitrarily among
        everything stored today — which is most of what an agent holds.
        """
        trusted = _origin.of(meta) in self.trusted
        if meta.get_attribute(_PINNED) == "true" and trusted:
            return 1.0
        now = now if now is not None else _clock.now(self.db)
        # An intention is not context until it is due, and then it is context
        # until it is done: a reminder that decays like a note is forgotten
        # exactly when it matters. Trusted only, as a pin is: a due time is a
        # pin set in advance.
        due = due_at(meta) if trusted else None
        if due is not None:
            return 1.0 if now >= due else 0.0
        age_days = max(0.0, (now - meta.valid_time) / 86400.0)   # as the engine
        recalls = effective_recalls(meta, now) if trusted else 0.0
        stickiness = 1.0 + math.log(1.0 + recalls)
        recency = 0.5 ** (age_days / stickiness / self.half_life)
        usage = 1.0 - 0.5 ** (recalls / USAGE_HALF_LIFE)
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
        now = _clock.now(self.db)

        # Heat decays on a half-life measured in days, so a result up to a
        # second stale is indistinguishable from a fresh one. Bucketing by whole
        # seconds keeps the cache from being invalidated by the clock alone.
        ck = (budget, _gen(self.db), int(now))
        hit = self._hot_cache.get(ck)
        if hit is not None:
            return hit
        out, metas = self._hot_select(budget, now)
        if out is None:
            out, metas = self._hot_scan(budget, now)
        # One entry: the budget and generation both change rarely, and an
        # unbounded dict here would be a slow leak in a long-running agent.
        self._hot_cache = {ck: out}
        note_stale(self.db, (metas[i.id] for i in out))
        return out

    def _hot_select(self, budget, now):
        """hot() by the engine (DB.heat_select, which mirrors _hot_scan):
        only the picks become items. (None, None) when the DB cannot."""
        select = getattr(self.db, "heat_select", None)
        if select is None:
            return None, None
        scopes = self.scopes()
        picks = select([SCOPE_SEP.join(sc) for sc in scopes], float(now), budget,
                       self.half_life, self.min_heat, sorted(self.trusted), list(_NOT_MEMORY),
                       usage_half_life=USAGE_HALF_LIFE, unused_floor=UNUSED_FLOOR,
                       inherit_decay=INHERIT_DECAY, chars_per_token=CHARS_PER_TOKEN,
                       pinned_attr=_PINNED, due_attr=DUE_ATTR, origin_attr=_origin.ATTR,
                       text_attr=_LG_VALUE, origins=list(_origin.ORIGINS),
                       default_origin=_origin.USER, unknown_origin=_origin.EXTERNAL)
        ids = [p[0] for p in picks]
        out, metas = [], {}
        for (rid, depth, _, _), meta in zip(picks, self.db.get_metadata_many(ids) if ids else []):
            if meta is None:
                continue                      # gone since: never mind it
            metas[rid] = meta
            out.append(self._item(rid, meta, scopes[depth], depth, now,
                                  meta.key or meta.entity_id or str(rid), _origin.of(meta)))
        return out, metas

    def _hot_scan(self, budget, now):
        """hot() in Python: every record in scope becomes an item. The
        specification DB.heat_select follows, and the path for a DB without
        it."""
        scored: list[PocketItem] = []
        at: dict[str, int] = {}               # key -> its index in `scored`
        metas: dict = {}
        for rid, meta, sc, depth in self._scoped_records():
            if meta is None or meta.source == "_forgotten":
                continue
            # Cache entries live in the same namespace but are not memories —
            # they must never spend the agent's context budget. importance=0.0
            # already zeroes their heat, but a generous budget would still admit
            # them at the tail, so exclude them by kind rather than by score.
            # Index records find a memory; they are never carried themselves.
            if meta.source in _NOT_MEMORY:
                continue
            key = meta.key or meta.entity_id or str(rid)
            # Shadowing: the agent's own version of a key wins over an inherited
            # one. Scopes are walked deepest first, so the first sighting is the
            # most specific — the same rule as a variable shadowing an outer one.
            origin = _origin.of(meta)
            if key in at and not _overrides(origin, scored[at[key]].origin, self.trusted):
                continue
            item = self._item(rid, meta, sc, depth, now, key, origin)
            metas[rid] = meta
            if key in at:
                scored[at[key]] = item
            else:
                at[key] = len(scored)
                scored.append(item)

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
        return out, metas

    def _item(self, rid, meta, sc, depth, now, key, origin,
              relevance: float = 0.0) -> PocketItem:
        content = display_text(meta)
        sup = meta.get_attribute(SUPKEYS)
        return PocketItem(
            id=rid,
            key=key,
            content=content,
            heat=self.heat(meta, now, depth),
            tokens=estimate_tokens(content),
            recalls=meta.recall_count,
            age_days=(now - meta.valid_time) / 86400.0,
            pinned=meta.get_attribute(_PINNED) == "true",
            scope=tuple(sc),
            inherited=depth > 0,
            version=meta.version,
            origin=origin,
            due_at=due_at(meta),
            valid_time=float(meta.valid_time),
            relevance=relevance,
            supkeys=tuple(sup.split(_KEY_SEP)) if sup else (),
        )

    def standing(self) -> list[PocketItem]:
        """What every context carries: trusted pins and due intentions in this
        scope and its ancestors, in key order.

        Also the part of a context that is the same from one request to the
        next, which is why `PacketBuilder` puts it first: a prompt cache reuses
        a prefix only up to the first byte that changed. Found through the
        engine's attribute index, not by reading the scope: `hot()` reads every
        record (1.7 ms at 10k in the engine), and a packet would pay that cold each time,
        since a packet's promotion invalidates hot()'s memo. Shadowing is
        hot()'s: a nearer trusted record of the same key hides an ancestor's.
        """
        db = self.db
        now = _clock.now(db)
        scopes = self.scopes()
        depth_of = {SCOPE_SEP.join(sc): d for d, sc in enumerate(scopes)}
        ids = list(dict.fromkeys([*db.ids_with_attribute(_PINNED, "true"),
                                  *db.ids_with_attribute(INTENTION_ATTR, "true")]))
        best: dict[str, PocketItem] = {}
        for rid, meta in zip(ids, db.get_metadata_many(ids) if ids else []):
            if meta is None or meta.source in _NOT_MEMORY:
                continue
            depth = depth_of.get(meta.namespace_id)
            if depth is None or self.heat(meta, now, depth) < 1.0:
                continue            # out of scope, untrusted, or not yet due
            key = meta.key or meta.entity_id or str(rid)
            if key in best and len(best[key].scope) >= len(scopes[depth]):
                continue            # a nearer one of the key is already held
            if self._shadowed(key, depth):
                continue
            best[key] = self._item(rid, meta, scopes[depth], depth, now, key,
                                   _origin.of(meta))
        return [best[k] for k in sorted(best)]

    def _shadowed(self, key: str, depth: int) -> bool:
        """Does a nearer scope than `depth` hold a trusted live record of `key`?"""
        for sc in self.scopes()[:depth]:
            _rid, meta = keys.get(self.db, SCOPE_SEP.join(sc), key)
            if (meta is not None and meta.source not in _NOT_MEMORY
                    and _origin.of(meta) in self.trusted):
                return True
        return False

    def hot_text(self, budget_tokens: Optional[int] = None, sep: str = "\n", *,
                 dates: bool = True, history: str = "section") -> str:
        """The working set as a single block, ready to drop into a prompt,
        each item after its date (`date_tag`) unless `dates=False`.

        Records not stated by a principal (agent, tool, external) come after
        the rest, quoted, under a header that says they are data. See
        feather_db.origin.

        A memory superseded on a one-value key is shown as history
        (`group_superseded`; `history` as in origin.render, a section of its
        own by default); those lines are outside the budget."""
        return _origin.render(self.group_superseded(self.hot(budget_tokens)),
                              trusted=self.trusted, sep=sep,
                              tag=date_tag if dates else None, history=history)

    # ── writing ──────────────────────────────────────────────────────────

    def remember(
        self, key: str, content: str, *,
        importance: float = 1.0, pinned: bool = False,
        attributes: Optional[dict] = None, origin: str = _origin.USER,
        due_at: Optional[float] = None,
        valid_at: Optional[float] = None,
        index: Optional[Iterable] = None,
    ) -> int:
        """Store a memory. Same key overwrites, so re-learning is not duplication.

        `valid_at` (unix seconds): when what it says happened or became true,
        if not now. It dates the memory in a prompt and orders supersession.

        `index`: other ways to find it — `IndexEntry`s, strings, or the Facts
        a SessionFactExtractor drew from `content`. Each is written as an
        index record that a search can hit in place of the memory; the memory
        is what comes back. An entry with a one-value key puts the memory in
        that key's chain (see `recall`). A rewrite replaces the index; leaving
        `index` out keeps the one it has.

        `due_at` (unix seconds) makes it an intention: out of `hot()` until
        then, carried as if pinned from then on. Done is a rewrite without
        one, or `forget(key)`.

        `origin` is where the content came from: `user` (stated by a principal,
        the default for a direct call), `agent` (the agent's own inference),
        `tool` (a tool's output) or `external` (third-party text). Anything but
        `user` is quoted as data in a prompt, cannot satisfy a required rule,
        and is not lifted by `pinned`. See feather_db.origin."""
        _origin.check(origin)
        # One request remotely (keys.get is two). The scope's legacy records
        # were given native keys when this pocket opened.
        prev = self.db.get_by_key(key, self.namespace)

        meta = feather_db.Metadata()
        meta.timestamp    = int(time.time())
        meta.namespace_id = self.namespace
        meta.entity_id    = key
        meta.content      = content
        meta.importance   = importance
        meta.source       = "pocket"
        if pinned:
            meta.set_attribute(_PINNED, "true")
        if due_at is not None:
            meta.set_attribute(DUE_ATTR, repr(float(due_at)))
            meta.set_attribute(INTENTION_ATTR, "true")
        for k, v in (attributes or {}).items():
            meta.set_attribute(str(k), str(v))
        meta.set_attribute(_origin.ATTR, origin)
        if valid_at is not None:
            meta.valid_at = int(valid_at)
        entries = None if index is None else [IndexEntry.of(x) for x in index]
        if entries is None and prev is not None:
            # Keep the index it has: its count, keys and provenance are on
            # the record.
            for attr in INDEX_ATTRS:
                if prev.get_attribute(attr):
                    meta.set_attribute(attr, prev.get_attribute(attr))
        # Carry the recall history forward — an update is new information about
        # something the agent already cared about, not a reset of that interest.
        # With its time, or the history would fade as if it were the record's age.
        if prev is not None and prev.recall_count:
            meta.recall_count = prev.recall_count
            meta.last_recalled_at = prev.last_recalled_at

        if entries is None:
            rid = keys.write(self.db, self.namespace, key, meta, self._vec(content))
        else:
            # The memory and its whole index in one batch (one round trip).
            r = apply_index(self.db, [IndexWrite(
                self.namespace, key, entries, meta, vec=self._vec(content),
                old_n=int((prev.get_attribute(INDEX_N) if prev is not None else "") or 0))],
                self._vec)
            rid = r.ids[0]
        _bump(self.db)
        return rid

    def known_keys(self) -> list[str]:
        """The one-value keys ('subject|attribute') this scope's index holds,
        sorted. What a SessionFactExtractor is given as `known_keys` so a new
        memory names an attribute the way earlier ones did: told nothing,
        separate calls named one city "address", "home city" and "location"
        (bench.contradiction: ~70% of an attribute's statements on one key
        without them, 91% with). Reads the scope: O(records), like hot() cold."""
        return known_keys(self.db, self.namespace)

    def pin(self, key: str, pinned: bool = True) -> None:
        rid, meta = keys.get(self.db, self.namespace, key)
        if meta is None:
            return
        meta.set_attribute(_PINNED, "true" if pinned else "false")
        self.db.update_metadata(rid, meta)
        _bump(self.db)

    def forget(self, key: str) -> None:
        """Hide a memory: it leaves reads, but its bytes stay in the file until
        a vacuum. For personal data use erase()."""
        rid, meta = keys.get(self.db, self.namespace, key)
        if rid is not None:
            forget_index(self.db, self.namespace, key, int(meta.get_attribute(INDEX_N) or 0))
            self.db.forget(rid)
        _bump(self.db)

    def erase(self, key: str) -> dict:
        """Erase the memory at `key` in this scope, and anything derived from
        or citing it, so that nothing of it remains on disk (DB.erase). A
        memory's entity is its key, so a forgotten one is found too. Returns
        the receipt."""
        return self.db.erase(namespace=self.namespace, entity=key)

    def erase_all(self) -> dict:
        """Erase everything this scope holds, its session scratch included.
        Inherited scopes are not this agent's to erase, and are untouched."""
        return self.db.erase(namespace=self.namespace, prefix=True)

    # ── WARM ─────────────────────────────────────────────────────────────

    def recall(self, query: str, k: int = 5, *, promote: bool = True,
               supersede: bool = True) -> list[PocketItem]:
        """Search the whole namespace, including what is too cold for the pocket.

        What it returns is promoted: its `recall_count` goes up, so the next
        `hot()` reflects what the agent just needed. Only those k — the
        candidates fetched to rank them are searched without counting, or every
        recall would credit up to 3k per scope it never handed back.

        `promote=False` returns without counting, for a caller that will use
        only some of the results and counts those itself (`PacketBuilder`
        counts what it puts in a packet).

        A hit on an index record (`remember(index=...)`) counts for the memory
        it indexes, at the better of the two scores. With `supersede` (the
        default; nothing changes for memories written without keyed index
        entries), memories in one key's chain come back once: the newest,
        with the older ones in its `earlier`, brought in when only an older
        one matched (`group_superseded`).
        """
        now = _clock.now(self.db)
        visible = {SCOPE_SEP.join(sc): d for d, sc in enumerate(self.scopes())}

        hits = []
        # 3k per scope: a memory and its index records can each take a slot.
        for flat, depth in visible.items():
            # Filter in the engine, never after it. Taking the global top hits
            # and then keeping this scope's lost a scope's own note whenever
            # another scope held more records that scored higher on the query.
            f = FilterBuilder().namespace(flat).build()
            if self._embed is not None:
                hits += [(h, depth) for h in self.db.search(
                    np.asarray(self._embed(query), dtype=np.float32), k=k * 3, filter=f,
                    record_salience=False)]
            else:
                hits += [(h, depth) for h in self.db.keyword_search(
                    query, k=k * 3, filter=f, record_salience=False)]
        # Each scope is searched separately, so raw scores are only comparable
        # within one. Normalise before ranking across them.
        top = max((h.score for h, _ in hits), default=1.0) or 1.0
        memory_of = index_targets(self.db, [h.metadata for h, _ in hits])

        out, at, metas, facts = [], {}, {}, {}
        for h, depth in hits:
            rid, m = h.id, h.metadata
            if m.source == _INDEX_KIND:
                fact = m.content
                rid, m = memory_of.get((m.namespace_id, m.get_attribute(INDEX_FOR)),
                                       (None, None))
                if m is None:
                    continue
                facts.setdefault(rid, []).append(fact)
            if m.source in _NOT_MEMORY:
                continue
            key = m.key or m.entity_id or str(rid)
            origin = _origin.of(m)
            relevance = float(h.score) / top
            if key in at:
                held = out[at[key]]
                if held.id == rid:          # the same memory, through its index
                    held.relevance = max(held.relevance, relevance)
                    continue
                if not _overrides(origin, held.origin, self.trusted):
                    continue
            sc = tuple(m.namespace_id.split(SCOPE_SEP)) if m.namespace_id else ()
            item = self._item(rid, m, sc, depth, now, key, origin, relevance)
            metas[rid] = m
            if key in at:
                out[at[key]] = item
            else:
                at[key] = len(out)
                out.append(item)
        # Rank by RELEVANCE, with heat as a modest tiebreaker.
        #
        # Sorting by heat alone made recall() ignore the query: a pinned record
        # has heat 1.0 by definition, so an org policy was returned first for
        # "which tiktok creative performed best" as readily as for "what should
        # I not claim". Heat answers "what should I be carrying"; relevance
        # answers "what did you ask for". Only hot() may confuse the two.
        out.sort(key=lambda x: -(x.relevance + RECALL_HEAT_WEIGHT * x.heat))
        out = out[:k]
        for item in out:
            if item.id in facts:
                item.facts = tuple(dict.fromkeys(facts[item.id]))
        note_stale(self.db, (metas[i.id] for i in out))
        if supersede:
            out = self.group_superseded(out)
        if promote:
            self.promote(out)
        return out

    # ── supersession ─────────────────────────────────────────────────────

    def group_superseded(self, items: Iterable[PocketItem]) -> list[PocketItem]:
        """`items` with every one-value key's chain shown once: the memory
        holding its newest value, carrying the older ones in `earlier`
        (newest first, at most MAX_EARLIER), at the place of the first item
        of the chain. An older memory that matched brings the newest in.

        Raw memory stays the payload and history stays readable; only which
        statement is current is decided. Measured (bench.contradiction,
        dense search, gpt-4o-mini reader): notes alone 84-86% current, the
        same notes with their chains grouped 97%, keyed facts 99-100%.

        Newest is by the index entry's valid time, then the memory's. An
        untrusted memory never makes a trusted one history (as with
        shadowing): the current value is the newest trusted one when the
        chain holds any. A memory repeating the current value is left alone.
        """
        items = list(items)
        if not any(i.supkeys for i in items):
            return items
        now = _clock.now(self.db)
        depth_of = {SCOPE_SEP.join(sc): d for d, sc in enumerate(self.scopes())}

        def make(ns, key, rid, meta):
            if ns not in depth_of:
                return None
            return self._item(rid, meta, tuple(ns.split(SCOPE_SEP)), depth_of[ns], now, key,
                              _origin.of(meta))

        grouped = group_records(
            self.db, [((SCOPE_SEP.join(i.scope), i.key), i.supkeys, i.origin, i) for i in items],
            trusted=self.trusted, make=make)
        # Copies: hot() memoises its items, and they must not change.
        return [dataclasses.replace(obj, relevance=src.relevance, earlier=tuple(hist))
                for obj, hist, src in grouped]

    def promote(self, items: Iterable[PocketItem]) -> None:
        """Count `items` as used: each one's `recall_count` goes up by one.
        Untrusted items are skipped: heat() does not count their use."""
        touched = False
        for item in items:
            if item.origin not in self.trusted:
                continue
            self.db.touch(item.id)
            touched = True
        # A promotion is a write: it changes heat and therefore the working
        # set. Without this the memoised hot() would keep serving a view that
        # predates it — the exact thing promotion exists to change.
        if touched:
            _bump(self.db)

    # ── CACHE ────────────────────────────────────────────────────────────

    def cache(self, key: str, compute: Callable[[], Any], *, ttl_seconds: float = 3600) -> Any:
        """Memoize an expensive call in the same file as the memory.

        Agent codebases grow a second, separate cache for tool results — a dict
        keyed on a hash, with no eviction and no notion of what is worth keeping.
        Putting it here means one store, one expiry mechanism, and cached results
        that can be inspected alongside everything else the agent knows.
        """
        _, meta = keys.get(self.db, self.namespace, f"__cache__/{key}")
        if meta is not None and meta.source != "_forgotten":
            expires = float(meta.get_attribute(_CACHE_EXPIRY) or 0)
            if _clock.now(self.db) < expires:
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
        keys.write(self.db, self.namespace, f"__cache__/{key}", m, self._vec(key))
        _bump(self.db)
        return value

    def evict_expired(self) -> int:
        """Drop cache entries past their TTL. Real memories are untouched."""
        n, now = 0, _clock.now(self.db)
        ids = list(self.db.ids_in_namespace(self.namespace))
        for rid, meta in zip(ids, self.db.get_metadata_many(ids) if ids else []):
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
        now = _clock.now(self.db)
        hot = self.hot()
        hot_ids = {i.id for i in hot}
        total = cache = warm_tokens = index = 0
        per_scope: dict[str, int] = {}
        for rid, meta, sc, _depth in self._scoped_records():
            if meta is None or meta.source == "_forgotten":
                continue
            if meta.source == _INDEX_KIND:
                index += 1
                continue
            if meta.source in DERIVED_KINDS:
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
            "index_items":  index,
            "total_items":  total,
            "per_scope":    per_scope,
        }

    # ── internals ────────────────────────────────────────────────────────

    def _id(self, key: str) -> Optional[int]:
        """The record holding `key` in this scope, or None."""
        return keys.resolve(self.db, self.namespace, key)

    def _vec(self, text: str) -> np.ndarray:
        if self._embed is not None:
            return np.asarray(self._embed(text), dtype=np.float32)
        # The dimension once per pocket: over the Cloud API each dim() is a
        # request, and an index of 20 entries asked 20 times.
        dim = getattr(self, "_dim", None)
        if dim is None:
            dim = self._dim = self.db.dim("text")
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
        # Random, not clock-derived: time_ns() ticks in microseconds on macOS,
        # so two sessions opened back to back could hash to one id and share
        # (and drop) each other's scratch.
        self.id = session_id or uuid.uuid4().hex[:10]
        # Scratch is a CHILD scope, so it inherits everything the agent can see
        # and is ranked against it, while remaining trivially droppable.
        self.scratch = Pocket(
            parent.db, tuple(parent.scope) + ("_session", self.id),
            budget_tokens=parent.budget,
            half_life_days=parent.half_life,
            min_heat=parent.min_heat,
            embed=parent._embed,
            trusted_origins=parent.trusted,
        )
        # Scratch is fresh and therefore hot; without a cap one chatty run would
        # push every durable memory out of the context window.
        self.scratch_ratio = scratch_ratio
        self._kept: list[str] = []
        self._closed = False

    # ── during the run ───────────────────────────────────────────────────

    def note(self, text: str, *, key: Optional[str] = None, importance: float = 0.6,
             origin: str = _origin.AGENT) -> str:
        """Write working memory for this run only. `origin` defaults to `agent`:
        scratch is what the agent was thinking; pass `tool` or `external` for
        what it read."""
        # Random, not `time_ns() % 100000`: with microsecond ticks that left
        # 100 possible keys, so ~1 in 100 note pairs collided and the second
        # silently overwrote the first.
        key = key or f"n{uuid.uuid4().hex[:12]}"
        self.scratch.remember(key, text, importance=importance, origin=origin)
        return key

    def keep(self, key: str, text: str, *, importance: float = 1.0,
             pinned: bool = False, origin: str = _origin.AGENT) -> None:
        """Promote something to the agent's durable memory. Survives close().
        `origin` defaults to `agent`: a finding is the agent's conclusion. A
        pin holds only for a trusted origin, so pinning a finding takes
        `origin="user"`: the caller vouching for it."""
        self.parent.remember(key, text, importance=importance, pinned=pinned,
                             origin=origin)
        self._kept.append(key)

    def recall(self, query: str, k: int = 5) -> list[PocketItem]:
        """Search durable memory and this run's scratch together."""
        return self.scratch.recall(query, k=k)

    def context(self, budget_tokens: Optional[int] = None, *, dates: bool = True) -> str:
        """The prompt block: durable memory plus this run's scratch, budgeted,
        each item after its date unless `dates=False`.

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
        return _origin.render(out, trusted=self.parent.trusted,
                              tag=date_tag if dates else None)

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
