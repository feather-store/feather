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
from feather_db.filter import FilterBuilder

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

    def __init__(
        self,
        path: str = "agent_memory.feather",
        *,
        dim: int = 768,
        embed: Optional[Callable[[str], Any]] = None,
        index_fields: Optional[list[str]] = None,
        auto_save: bool = True,
    ):
        _require_langgraph()
        self.db = DB.open(path, dim=dim)
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
        self._auto_save = auto_save
        # Items are addressed by native key. Before 0.22 an item's id was a
        # 53-bit SHA-1 of (namespace, key), and two keys that collided shared
        # one record. Items written then get their key now, which also makes
        # them findable by keyword search on it.
        keys.migrate(self.db)

    def id_of(self, namespace: tuple[str, ...], key: str) -> Optional[int]:
        """The record id of the item at (namespace, key), or None."""
        return keys.resolve(self.db, NS_SEP.join(namespace), key)

    # ── the two abstract methods ─────────────────────────────────────────

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        out: list[Any] = []
        dirty = False
        for op in ops:
            if isinstance(op, GetOp):
                out.append(self._get(op))
            elif isinstance(op, PutOp):
                self._put(op); dirty = True
                out.append(None)
            elif isinstance(op, SearchOp):
                out.append(self._search(op))
            elif isinstance(op, ListNamespacesOp):
                out.append(self._list_namespaces(op))
            else:
                out.append(None)
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
        if _is_dead(meta):
            return None
        return self._to_item(op.namespace, op.key, meta)

    def _put(self, op: Any) -> None:
        flat = NS_SEP.join(op.namespace)
        rid, prev = keys.get(self.db, flat, op.key)

        if op.value is None:                       # delete
            if rid is not None:
                self.db.forget(rid)
            return

        created = (prev.get_attribute(_CREATED_ATTR) if prev else "") or str(int(time.time()))

        meta = feather_db.Metadata()
        meta.timestamp    = int(time.time())       # doubles as updated_at
        meta.namespace_id = NS_SEP.join(op.namespace)
        meta.entity_id    = op.key
        meta.source       = "langgraph"
        meta.content      = _indexed_text(dict(op.value), self._index_fields)
        meta.set_attribute(_VALUE_ATTR, json.dumps(dict(op.value)))
        meta.set_attribute(_CREATED_ATTR, created)
        # A numeric `valid_at` in the value is when the item happened or
        # became true (unix seconds), as opposed to when it was put: the
        # engine indexes it, so newest(by_valid=True) and a valid-time
        # filter use it without parsing JSON.
        valid = dict(op.value).get("valid_at")
        if isinstance(valid, (int, float)) and not isinstance(valid, bool):
            meta.valid_at = int(valid)
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
                vec = np.asarray(self._embed(text), dtype=np.float32)
        if vec is None:
            vec = np.zeros(self.dim, dtype=np.float32)
            vec[0] = 1.0        # a valid unit vector; never a zero-norm query target

        keys.write(self.db, flat, op.key, meta, vec)

    def _ids_under(self, prefix: tuple[str, ...]) -> list[int]:
        """Every id whose namespace starts with `prefix`.

        LangGraph namespaces are hierarchical — searching ("hawky","user_842")
        must reach ("hawky","user_842","preferences"). Feather's namespace index
        is an exact-match dict, so expand the prefix over the known namespaces
        first. O(namespaces), which is small; the per-namespace lookup underneath
        is still the O(matches) index hit.
        """
        flat = NS_SEP.join(prefix)
        ids: list[int] = []
        for known in self.db.list_namespaces():
            if not known:
                continue
            if flat and not (known == flat or known.startswith(flat + NS_SEP)):
                continue
            ids.extend(self.db.ids_in_namespace(known))
        return ids

    def _search(self, op: Any) -> list[Any]:
        want = op.limit + op.offset
        semantic = bool(op.query and self._embed is not None)
        if not semantic:
            ids = (self._ids_under(op.namespace_prefix) if op.namespace_prefix
                   else self.db.get_all_ids("text"))
            cand = []
            for i in ids:
                m = self.db.get_metadata(i)
                if m is not None:
                    cand.append((i, None, m))
            return self._select(cand, op, semantic)[op.offset: op.offset + op.limit]

        # Ranked WITHIN the prefix by the engine (an indexed namespace_prefix
        # filter). It used to rank everything and keep what was under the
        # prefix afterwards: with enough other tenants, all k hits belonged to
        # them and a user's own search came back empty. Only what the engine
        # cannot filter — expired and unindexed items, value filters — is
        # dropped here, and k grows until enough survive or the prefix runs out.
        q = np.asarray(self._embed(op.query), dtype=np.float32)
        flt = (FilterBuilder().namespace_prefix(NS_SEP.join(op.namespace_prefix)).build()
               if op.namespace_prefix else None)
        k = max(want, 10)
        while True:
            hits = self.db.search(q, k=k, filter=flt, record_salience=False)
            results = self._select([(h.id, h.score, h.metadata) for h in hits], op, semantic)
            if len(results) >= want or len(hits) < k:
                return results[op.offset: op.offset + op.limit]
            k *= 4

    def _select(self, cand, op: Any, semantic: bool) -> list[Any]:
        results = []
        for rid, score, meta in cand:
            if _is_dead(meta):
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
        now = time.time()
        out = []
        for rid in ids:
            meta = self.db.get_metadata(rid)
            if meta is None or _is_dead(meta, now):
                continue
            out.append((rid, meta))
        return out

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
        now = time.time()
        while len(out) < limit:
            want = limit - len(out)
            ids = self.db.newest(flat, want, prefix=prefix, before=before,
                                 by_valid=by_valid)
            cursor = before
            for rid in ids:
                meta = self.db.get_metadata(rid)
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
        self.db.save()
        # Release the single-owner file lock so the path can be reopened in
        # this process; save() alone keeps the handle (and the lock) alive.
        close = getattr(self.db, "close", None)
        if close is not None:
            close()
