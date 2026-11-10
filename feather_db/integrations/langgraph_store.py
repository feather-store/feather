"""FeatherStore — a LangGraph `BaseStore` backed by Feather.

LangGraph has two memory sockets. `BaseCheckpointSaver` holds per-run graph
state; `BaseStore` holds **long-term, cross-thread** memory — the things an agent
should still know next week. Feather is the second one.

`BaseStore` has exactly two abstract methods, `batch()` and `abatch()`. Every
convenience method (`get`/`put`/`search`/`delete`/`list_namespaces`) is concrete
and routes through them, so implementing those two puts Feather inside every
LangGraph agent.

Why this matters more than the existing adapters: `FeatherVectorStore` makes
Feather *a search index an agent queries*. A `BaseStore` makes Feather *the thing
an agent remembers into*. Same engine, different position in the stack.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Iterable, Optional

import numpy as np

# LangGraph is an optional dependency; the module imports without it so the
# rest of feather_db.integrations keeps working.
try:
    from langgraph.store.base import (
        BaseStore, GetOp, PutOp, SearchOp, ListNamespacesOp, Item, SearchItem,
    )
    _LG = True
except ImportError:                                           # pragma: no cover
    _LG = False
    BaseStore = object                                        # type: ignore
    GetOp = PutOp = SearchOp = ListNamespacesOp = Item = SearchItem = None  # type: ignore

import feather_db
from feather_db import DB, keys
from feather_db import clock as _clock
from feather_db import origin as _origin
from feather_db.filter import FilterBuilder
from feather_db.pocket import DUE_ATTR as _DUE_ATTR
from feather_db.pocket import unit as _unit
from feather_db.pocket import INTENTION_ATTR as _INTENTION_ATTR
from feather_db.pocket import _bump as _pocket_bump
from feather_db.pocket import (_INDEX_KIND, _KEY_SEP, DERIVED_KINDS, INDEX_ATTRS,
                               INDEX_FOR, INDEX_N, SUPKEYS, IndexEntry, IndexWrite,
                               _stale_index_ids, apply_index, backfill_registry,
                               group_records, index_supkeys, index_targets, note_stale,
                               supkeys_of)
from feather_db.pocket import known_keys as _known_keys

# LangGraph forbids "." inside a namespace element, which makes it the one
# separator guaranteed not to collide when flattening the tuple.
NS_SEP = "."
_VALUE_ATTR   = "_lg_value"       # the JSON payload
_CREATED_ATTR = "_lg_created_at"  # Feather has ONE timestamp; created_at lives here
                                  # until format v10 adds recorded_at/valid_from.
_UNINDEXED    = "_lg_unindexed"   # index=False: stored, but not semantically findable


def _expired(meta, now: Optional[float] = None) -> bool:
    """Feather's own rule: ttl seconds from timestamp, 0 meaning never.

    Checked on every read because nothing sweeps automatically — `forget_expired()`
    exists but has to be called. Without this an expired item stayed readable
    until someone happened to run the sweep, which for an agent memory means a
    fact the caller asked to be temporary outliving the reason it was temporary.
    """
    ttl = getattr(meta, "ttl", 0) or 0
    if ttl <= 0:
        return False
    return (now if now is not None else time.time()) > meta.timestamp + ttl


def _is_dead(meta, now: Optional[float] = None) -> bool:
    """Mirror the Cloud API's `_is_dead_meta` exactly.

    `db.forget()` marks a record by setting source="_forgotten"; an earlier
    version of this file invented its own sentinel and therefore never saw a
    deleted item, so `get()` kept returning records after `delete()`.
    """
    if meta is None:
        return True
    if meta.source == "_forgotten":
        return True
    if meta.get_attribute("_deleted") == "true":
        return True
    # An index record (FeatherStore.index) finds an item; it is not one. Nor
    # is a key-registry entry or an ingest job.
    if meta.source in DERIVED_KINDS:
        return True
    return _expired(meta, now)


def _require_langgraph() -> None:
    if not _LG:
        raise ImportError(
            "FeatherStore needs langgraph. Install with: pip install langgraph"
        )


def _indexed_text(value: dict, fields: Optional[list[str]]) -> str:
    """The text that gets embedded and BM25-indexed.

    `fields` mirrors LangGraph's index config: None means the whole object,
    otherwise only the named top-level keys. Keeping this separate from the
    stored payload means an agent can store structure and still search prose.
    """
    if fields:
        parts = [str(value.get(f, "")) for f in fields]
    else:
        parts = [f"{k}: {v}" for k, v in value.items()]
    return "\n".join(p for p in parts if p)


def _matches(value: dict, flt: Optional[dict]) -> bool:
    """LangGraph filter operators, applied over the candidate set.

    Feather's own attribute filter is exact-match only, so `$gt`/`$lt` and
    friends are evaluated here. That is correct but O(candidates) — when typed
    attributes land in format v10 this moves into the engine.
    """
    if not flt:
        return True
    for field, cond in flt.items():
        actual = value.get(field)
        if not isinstance(cond, dict):
            if actual != cond:
                return False
            continue
        for op, want in cond.items():
            try:
                if   op == "$eq"  and not actual == want: return False
                elif op == "$ne"  and not actual != want: return False
                elif op == "$gt"  and not actual >  want: return False
                elif op == "$gte" and not actual >= want: return False
                elif op == "$lt"  and not actual <  want: return False
                elif op == "$lte" and not actual <= want: return False
            except TypeError:
                return False          # incomparable types never match
    return True


class FeatherStore(BaseStore):
    """A LangGraph long-term memory store backed by one `.feather` file.

    No server, no network hop, no container. The whole store is a file next to
    the agent.
    """

    # BaseStore.put() refuses a ttl unless the subclass declares support, and
    # raises before reaching batch() — so without this, Feather's own `ttl` and
    # forget_expired() were unreachable through the LangGraph API even though
    # both already worked.
    supports_ttl: bool = True

    # Puts per WriteBatch: one request each over the Cloud API, and a bound
    # on its body (a 768-d vector is ~8 KB of JSON).
    PUT_CHUNK = 256

    def __init__(
        self,
        path: str = "agent_memory.feather",
        *,
        dim: Optional[int] = None,
        embed: Optional[Callable[[str], Any]] = None,
        index_fields: Optional[list[str]] = None,
        auto_save: Optional[bool] = None,
        db: Any = None,
    ):
        """A store over the file at `path`, or over `db`: an open handle,
        local or a `feather_db.RemoteDB` (the Cloud API), which the store
        uses without owning — close() leaves it open. `auto_save` (save
        after every write) defaults to on for a file the store opened and
        off for a handle it was given: a remote server checkpoints on its
        own, and a save per write would be one more request each."""
        _require_langgraph()
        self._owns_db = db is None
        if db is None:
            dim = 768 if dim is None else dim
            db = DB.open(path, dim=dim)
        elif dim is None:
            dim = db.dim()
        self.db = db
        self.dim = dim
        if embed is None:
            # Default to whatever the environment configures
            # (FEATHER_EMBED_PROVIDER / FEATHER_EMBED_MODEL), so a deployment
            # gets real semantic recall without every call site threading an
            # embedder through. None still means keyword-only; a provider that
            # is set but broken raises here rather than degrading silently.
            from feather_db.integrations.embedders import default_embedder
            embed = default_embedder(dim)
        self._embed = embed
        self._index_fields = index_fields
        self._auto_save = self._owns_db if auto_save is None else auto_save
        # Items are addressed by native key. Before 0.22 an item's id was a
        # 53-bit SHA-1 of (namespace, key), and two keys that collided shared
        # one record. Items written then get their key now, which also makes
        # them findable by keyword search on it.
        # Once per file (keys.upkeep): later starts cost one key lookup, not
        # a read of every record of every namespace.
        keys.upkeep(self.db, visit=lambda ns, ids, metas:
                    backfill_registry(self.db, ns, ids, metas))

    def id_of(self, namespace: tuple[str, ...], key: str) -> Optional[int]:
        """The record id of the item at (namespace, key), or None."""
        return keys.resolve(self.db, NS_SEP.join(namespace), key)

    # ── the two abstract methods ─────────────────────────────────────────

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        ops = list(ops)
        out: list[Any] = [None] * len(ops)
        dirty = False
        i = 0
        while i < len(ops):
            op = ops[i]
            if isinstance(op, PutOp):
                # A run of puts is written together: one key lookup, one
                # metadata read and one WriteBatch for all of them, where
                # each put cost two or three round trips over the Cloud API.
                # The run ends at the next read, which must see it.
                j = i
                while j < len(ops) and isinstance(ops[j], PutOp):
                    j += 1
                self._put_many(ops[i:j])
                dirty, i = True, j
                continue
            if isinstance(op, GetOp):
                out[i] = self._get(op)
            elif isinstance(op, SearchOp):
                out[i] = self._search(op)
            elif isinstance(op, ListNamespacesOp):
                out[i] = self._list_namespaces(op)
            i += 1
        if dirty and self._auto_save:
            self.db.save()
        return out

    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        # Feather releases the GIL inside search/add, so the sync path is
        # already concurrent across threads. A thread executor here would add a
        # hop without adding parallelism.
        return self.batch(ops)

    # ── operations ───────────────────────────────────────────────────────

    def _get(self, op: Any) -> Optional[Any]:
        _, meta = keys.get(self.db, NS_SEP.join(op.namespace), op.key)
        if _is_dead(meta, _clock.now(self.db)):
            return None
        return self._to_item(op.namespace, op.key, meta)

    def _put(self, op: Any) -> None:
        self._put_many([op])

    def _put_many(self, ops: list) -> None:
        # The last put of a (namespace, key) is the one that counts.
        last: dict = {}
        for op in ops:
            last[(NS_SEP.join(op.namespace), op.key)] = op
        refs = list(last)
        for c in range(0, len(refs), self.PUT_CHUNK):
            self._write_puts([(r, last[r]) for r in refs[c:c + self.PUT_CHUNK]])
        # A Pocket over this DB memoises hot() per write generation: bump it
        # after the write, so a done intention or a new note is seen at once.
        _pocket_bump(self.db)

    def _write_puts(self, puts: list) -> None:
        ids = keys.resolve_many(self.db, [r for r, _ in puts])
        found = [rid for rid in ids if rid is not None]
        prev_of = dict(zip(found, self.db.get_metadata_many(found))) if found else {}
        b = feather_db.WriteBatch()
        for ((flat, key), op), rid in zip(puts, ids):
            prev = prev_of.get(rid) if rid is not None else None
            if op.value is None:                   # delete
                if rid is not None:
                    n = int(prev.get_attribute(INDEX_N) or 0) if prev is not None else 0
                    if n:
                        for iid in _stale_index_ids(self.db, flat, key, 0):
                            b.forget(iid)
                    b.forget(rid)
                continue
            meta, vec = self._record(op, flat, key, prev)
            if rid is not None and prev is not None and not prev.key:
                # Written before native keys, at its hashed id: it keeps the
                # id and gains its key (keys.write).
                b.put(rid, vec, meta)
            else:
                b.upsert(meta, vec)
        if len(b):
            self.db.write(b)

    def _record(self, op: Any, flat: str, key: str, prev):
        """The metadata and vector a put writes."""
        created = (prev.get_attribute(_CREATED_ATTR) if prev else "") or str(int(time.time()))

        meta = feather_db.Metadata()
        meta.timestamp    = int(time.time())       # doubles as updated_at
        meta.namespace_id = flat
        meta.key          = key
        meta.entity_id    = key
        meta.source       = "langgraph"
        meta.content      = _indexed_text(dict(op.value), self._index_fields)
        meta.set_attribute(_VALUE_ATTR, json.dumps(dict(op.value)))
        meta.set_attribute(_CREATED_ATTR, created)
        # A rewrite keeps the item's index (FeatherStore.index), as Pocket does.
        if prev is not None and prev.source != "_forgotten":
            for attr in INDEX_ATTRS:
                if prev.get_attribute(attr):
                    meta.set_attribute(attr, prev.get_attribute(attr))
        # A numeric `valid_at` in the value is when the item happened or
        # became true (unix seconds), as opposed to when it was put: the
        # engine indexes it, so newest(by_valid=True) and a valid-time
        # filter use it without parsing JSON.
        valid = dict(op.value).get("valid_at")
        if isinstance(valid, (int, float)) and not isinstance(valid, bool):
            meta.valid_at = int(valid)
        # A numeric `due_at` makes the item an intention: a Pocket reading
        # this namespace holds it back until then and carries it as if pinned
        # after (feather_db.pocket.DUE_ATTR).
        due = dict(op.value).get("due_at")
        if isinstance(due, (int, float)) and not isinstance(due, bool):
            meta.set_attribute(_DUE_ATTR, repr(float(due)))
            meta.set_attribute(_INTENTION_ATTR, "true")
        # An `origin` naming one of feather_db.origin.ORIGINS labels the record,
        # so a Pocket or PacketBuilder reading it quotes untrusted content. Any
        # other value is the caller's own field and is left alone.
        origin = dict(op.value).get("origin")
        if isinstance(origin, str) and origin in _origin.ORIGINS:
            meta.set_attribute(_origin.ATTR, origin)
        if op.ttl is not None:
            # LangGraph ttl is in MINUTES; Feather stores whole SECONDS, where 0
            # means "never expires". Truncating would turn any ttl under one
            # second into permanent — the worst possible direction for a field
            # whose entire purpose is to make something temporary. Round, and
            # floor at one second so a positive ttl can never mean forever.
            meta.ttl = max(1, int(round(op.ttl * 60)))

        # Only embed when indexing is on for this item AND an embedder exists.
        # op.index is False to opt out, a list to pick fields, None for default.
        # index=False means "store it, do not make it semantically findable".
        # The placeholder vector below is IDENTICAL for every unindexed record,
        # so without a marker they all match each other perfectly and an
        # index=False item came back as a top semantic hit — the opposite of
        # what was asked for.
        if op.index is False:
            meta.set_attribute(_UNINDEXED, "true")

        vec = None
        if op.index is not False and self._embed is not None:
            text = _indexed_text(dict(op.value),
                                 op.index if isinstance(op.index, list) else self._index_fields)
            if text.strip():
                vec = _unit(self._embed(text))
        if vec is None:
            vec = np.zeros(self.dim, dtype=np.float32)
            vec[0] = 1.0        # a valid unit vector; never a zero-norm query target
        return meta, vec

    def _ids_under(self, prefix: tuple[str, ...]) -> list[int]:
        """Every id whose namespace starts with `prefix`.

        LangGraph namespaces are hierarchical — searching ("hawky","user_842")
        must reach ("hawky","user_842","preferences"). Feather's namespace index
        is an exact-match dict, so expand the prefix over the known namespaces
        first. O(namespaces), which is small; the per-namespace lookup underneath
        is still the O(matches) index hit.
        """
        flat = NS_SEP.join(prefix)
        if flat:          # one call for the subtree, where each namespace was one
            return list(self.db.ids_in_namespace(flat, prefix=True))
        ids: list[int] = []
        for known in self.db.list_namespaces():
            if known:
                ids.extend(self.db.ids_in_namespace(known))
        return ids

    def _search(self, op: Any) -> list[Any]:
        want = op.limit + op.offset
        semantic = bool(op.query and self._embed is not None)
        if not semantic:
            ids = (self._ids_under(op.namespace_prefix) if op.namespace_prefix
                   else self.db.get_all_ids("text"))
            # One read for every record (get_metadata_many): per id it was one
            # request each over the Cloud API, N + 2 for about().
            ids = list(ids)
            cand = [(i, None, m) for i, m in zip(ids, self.db.get_metadata_many(ids))
                    if m is not None]
            return self._select(cand, op, semantic)[op.offset: op.offset + op.limit]

        # Ranked WITHIN the prefix by the engine (an indexed namespace_prefix
        # filter). It used to rank everything and keep what was under the
        # prefix afterwards: with enough other tenants, all k hits belonged to
        # them and a user's own search came back empty. Only what the engine
        # cannot filter — expired and unindexed items, value filters — is
        # dropped here, and k grows until enough survive or the prefix runs out.
        q = _unit(self._embed(op.query))
        flt = (FilterBuilder().namespace_prefix(NS_SEP.join(op.namespace_prefix)).build()
               if op.namespace_prefix else None)
        k = max(want, 10) * 3            # an item and its index records share hits
        while True:
            hits = self.db.search(q, k=k, filter=flt, record_salience=False)
            resolved = self._resolve_index(hits)
            results = self._select(resolved, op, semantic)
            if len(results) >= want or len(hits) < k:
                note_stale(self.db, (m for _, _, m in resolved[:want]))
                return self._group(results[op.offset: op.offset + op.limit], op)
            k *= 4

    def _resolve_index(self, hits) -> list:
        """(id, score, metadata) per hit, an index record's hit standing for
        the item it indexes at the better of the two scores, in rank order."""
        target = index_targets(self.db, [h.metadata for h in hits])
        out, at = [], {}
        for h in hits:
            rid, score, m = h.id, h.score, h.metadata
            if m.source == _INDEX_KIND:
                rid, m = target.get((m.namespace_id, m.get_attribute(INDEX_FOR)), (None, None))
                if m is None:
                    continue
            if rid in at:
                i = at[rid]
                out[i] = (rid, max(out[i][1], score), out[i][2])
                continue
            at[rid] = len(out)
            out.append((rid, score, m))
        return out

    def _group(self, items: list, op: Any) -> list:
        """Items in one key's supersession chain come back once: the newest,
        with the older values (newest first) in its value's `earlier`; an
        older item that matched brings the newest in, at its score. Items
        without keyed index entries are returned as they are."""
        # Every item's key and record in two reads (keys.resolve_many,
        # get_metadata_many), not two or three round trips per result.
        refs = [(NS_SEP.join(it.namespace), it.key) for it in items]
        rids = keys.resolve_many(self.db, refs)
        found = [r for r in rids if r is not None]
        by_rid = dict(zip(found, self.db.get_metadata_many(found)))
        metas = {ref: (by_rid.get(rid) if rid is not None else None)
                 for ref, rid in zip(refs, rids)}
        if not any(supkeys_of(m) for m in metas.values()):
            return items
        now = _clock.now(self.db)

        def make(ns, key, rid, m):
            tns = tuple(ns.split(NS_SEP))
            if _is_dead(m, now) or (op.namespace_prefix and
                                    tns[:len(op.namespace_prefix)] != tuple(op.namespace_prefix)):
                return None
            return self._to_item(tns, key, m, score=0.0)

        records = [((NS_SEP.join(it.namespace), it.key),
                    supkeys_of(metas[(NS_SEP.join(it.namespace), it.key)]),
                    _origin.of(metas[(NS_SEP.join(it.namespace), it.key)])
                    if metas[(NS_SEP.join(it.namespace), it.key)] is not None else _origin.USER,
                    it) for it in items]
        out = []
        for obj, hist, src in group_records(self.db, records, trusted=_origin.TRUSTED, make=make):
            value = dict(obj.value)
            if hist:
                value["earlier"] = [dict(h.value) for h in hist]
            out.append(SearchItem(value=value, key=obj.key, namespace=obj.namespace,
                                  created_at=obj.created_at, updated_at=obj.updated_at,
                                  score=src.score))
        return out

    def _select(self, cand, op: Any, semantic: bool) -> list[Any]:
        results, now = [], _clock.now(self.db)
        for rid, score, meta in cand:
            if _is_dead(meta, now):
                continue
            # Unindexed records are still returned by a filter-only search —
            # they exist, they are just not semantically retrievable.
            if semantic and meta.get_attribute(_UNINDEXED) == "true":
                continue
            ns = tuple(meta.namespace_id.split(NS_SEP)) if meta.namespace_id else ()
            if op.namespace_prefix and ns[:len(op.namespace_prefix)] != op.namespace_prefix:
                continue
            value = self._value_of(meta)
            if not _matches(value, op.filter):
                continue
            results.append(self._to_item(ns, meta.key or meta.entity_id, meta, score=score))
        return results

    @staticmethod
    def _matches_condition(ns: tuple[str, ...], cond: Any) -> bool:
        """LangGraph prefix/suffix matching, with `*` as a single-element wildcard."""
        path = tuple(cond.path)
        seg = ns[: len(path)] if cond.match_type == "prefix" else ns[-len(path):]
        if len(seg) != len(path):
            return False
        return all(p == "*" or p == s for p, s in zip(path, seg))

    def _list_namespaces(self, op: Any) -> list[tuple[str, ...]]:
        out: list[tuple[str, ...]] = []
        for flat in self.db.list_namespaces():
            if not flat:
                continue
            ns = tuple(flat.split(NS_SEP))
            # Filter on the FULL namespace, then truncate. Truncating first would
            # make ("a","b","c") match a suffix condition on ("b",) at max_depth=2,
            # which is a match against a path the caller never stored.
            if op.match_conditions:
                if not all(self._matches_condition(ns, c) for c in op.match_conditions):
                    continue
            if op.max_depth is not None:
                ns = ns[: op.max_depth]
            if ns not in out:
                out.append(ns)
        out.sort()
        return out[op.offset: op.offset + op.limit]

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _value_of(meta) -> dict:
        raw = meta.get_attribute(_VALUE_ATTR) or "{}"
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _to_item(self, namespace, key, meta, score=None):
        created = meta.get_attribute(_CREATED_ATTR)
        from datetime import datetime, timezone
        def _dt(ts):
            try: return datetime.fromtimestamp(int(ts), tz=timezone.utc)
            except Exception: return datetime.now(tz=timezone.utc)
        common = dict(
            value=self._value_of(meta),
            key=key or "",
            namespace=tuple(namespace),
            created_at=_dt(created or meta.timestamp),
            updated_at=_dt(meta.timestamp),
        )
        return SearchItem(score=score, **common) if score is not None else Item(**common)

    # ── extensions beyond BaseStore ──────────────────────────────────────
    # BaseStore's `search` has no ordering and no cheap count, so anything that
    # wants "the newest N" or "how many" had to fetch and materialise the whole
    # namespace. These three give callers a bounded path. They are NOT part of
    # the LangGraph interface — a plain BaseStore consumer never sees them.

    def scan(self, namespace: tuple[str, ...], *, prefix: bool = False):
        """Live (id, metadata) rows in a namespace. Does not parse the value.

        The value is a JSON blob in an attribute, and parsing it is most of the
        cost of building an Item. Callers that only need to count, or to sort on
        a metadata field, should never pay it.
        """
        ids = (self._ids_under(namespace) if prefix
               else self.db.ids_in_namespace(NS_SEP.join(namespace)))
        now = _clock.now(self.db)
        ids = list(ids)
        return [(rid, meta) for rid, meta in zip(ids, self.db.get_metadata_many(ids))
                if meta is not None and not _is_dead(meta, now)]

    def erase(self, namespace: tuple[str, ...], key: Optional[str] = None, *,
              dry_run: bool = False) -> dict:
        """Erase the item at (namespace, key), or with no key everything under
        the namespace prefix, so that nothing of it remains: not in the file,
        not in the WAL, and not in a record derived from or citing it.
        `delete()` only forgets: the bytes stay until a vacuum, so erasing a
        key already deleted is how to finish the job. Returns the engine's
        erase receipt."""
        if key is not None:
            # An item's entity_id is its key, and a deleted item's shell keeps
            # both until the next save: this finds the live item or the shell.
            # Matching nothing still compacts and rewrites the file.
            return self.db.erase(namespace=NS_SEP.join(namespace), entity=key,
                                 dry_run=dry_run)
        if not namespace:
            raise ValueError("erase() needs a namespace; it will not erase the whole store")
        return self.db.erase(namespace=NS_SEP.join(namespace), prefix=True, dry_run=dry_run)

    def index(self, namespace: tuple[str, ...], key: str, entries: Iterable) -> None:
        """Index the item at (namespace, key) by `entries` (IndexEntry, str,
        or a SessionFactExtractor Fact), replacing any index it had: each is
        a record a semantic search can hit in place of the item, which is
        what comes back. An entry with a one-value `key` puts the item in
        that key's supersession chain: search returns a chain once, the
        newest item with the older values in its value's `earlier`. See
        feather_db.pocket (facts as index; LoCoMo J 78.8 -> 82.4). Needs an
        embedder to be searchable."""
        flat = NS_SEP.join(namespace)
        rid, meta = keys.get(self.db, flat, key)
        if _is_dead(meta, _clock.now(self.db)):
            raise KeyError(f"no item at {namespace!r}/{key!r}")
        # The item's attributes, every index record and every key's registry
        # entry in ONE batch: over the Cloud API, one round trip instead of
        # two or three per entry.
        apply_index(self.db, [IndexWrite(flat, key, [IndexEntry.of(x) for x in entries],
                                         meta, old_n=int(meta.get_attribute(INDEX_N) or 0))],
                    self.index_vector)
        if self._auto_save:
            self.db.save()

    def index_vector(self, text: str):
        """The vector an index record is searched by: the store's embedder,
        or a fixed placeholder without one."""
        if self._embed is None:
            v = np.zeros(self.dim, dtype=np.float32)
            v[0] = 1.0
            return v
        return _unit(self._embed(text))

    def known_keys(self, namespace: tuple[str, ...]) -> list[str]:
        """The one-value keys indexed under `namespace`, for an extractor's
        `known_keys`."""
        return _known_keys(self.db, NS_SEP.join(namespace))

    def count(self, namespace: tuple[str, ...], *, prefix: bool = False) -> int:
        """How many live items. Cannot use the engine's namespace_size(), which
        counts tombstones — forgotten records stay in the namespace index until
        compaction, so it would report deleted memories as present."""
        return len(self.scan(namespace, prefix=prefix))

    def newest(self, namespace: tuple[str, ...], limit: int = 10, *,
               key=None, prefix: bool = False, by_valid: bool = False) -> list[Any]:
        """The `limit` most recent items, newest first.

        Exists because `search(limit=n)` cannot answer this. It has no ordering,
        so it returns an arbitrary n and a caller sorting those n is sorting the
        wrong set — at 300 episodes, asking for the 5 newest returned events
        299, 298, 296, 295, 293, silently skipping two.

        By the engine timestamp (updated_at), it walks the engine's
        (timestamp, id) index from the end: O(log n + limit), where it used to
        read and sort the whole namespace. Items that expired are skipped and
        the walk pages on. `by_valid` walks the (valid time, id) index
        instead: newest by when it happened (the value's `valid_at`, else
        updated_at). A custom `key(meta)` cannot use either index, so it
        still scans the namespace and sorts.
        """
        if key is not None:
            rows = self.scan(namespace, prefix=prefix)
            rows.sort(key=lambda r: key(r[1]), reverse=True)
            return [self._item_of(m) for _, m in rows[:limit]]
        flat, out, before = NS_SEP.join(namespace), [], None
        now = _clock.now(self.db)
        while len(out) < limit:
            want = limit - len(out)
            ids = self.db.newest(flat, want, prefix=prefix, before=before,
                                 by_valid=by_valid)
            cursor = before
            for rid, meta in zip(ids, self.db.get_metadata_many(list(ids))):
                if meta is None:              # removed since: nothing to show
                    continue
                cursor = (meta.valid_time if by_valid else meta.timestamp, rid)
                if not _is_dead(meta, now):
                    out.append(self._item_of(meta))
            # A short page is the end; a page the cursor cannot move past (all
            # of it removed meanwhile) would repeat forever.
            if len(ids) < want or cursor == before:
                break
            before = cursor
        return out

    def _item_of(self, m):
        return self._to_item(tuple(m.namespace_id.split(NS_SEP)) if m.namespace_id else (),
                             m.key or m.entity_id, m)

    def close(self) -> None:
        """Close the file the store opened. A handle passed as `db=` is the
        caller's: it is saved if auto_save is on, and left open."""
        if not self._owns_db:
            if self._auto_save:
                self.db.save()
            return
        self.db.save()
        # Release the single-owner file lock so the path can be reopened in
        # this process; save() alone keeps the handle (and the lock) alive.
        close = getattr(self.db, "close", None)
        if close is not None:
            close()
