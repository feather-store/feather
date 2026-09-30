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

import hashlib
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
from feather_db import DB
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


def _ns_key(namespace: tuple[str, ...], key: str) -> int:
    """Deterministic uint64 id for (namespace, key).

    Deterministic on purpose: `put()` on the same (namespace, key) must overwrite
    in place, and Feather's `id` is a true upsert key. A time-seeded id — which
    is what the MCP helper uses — would create a new record per write and the
    store would grow without ever updating anything.
    """
    raw = NS_SEP.join(namespace) + "\x00" + key
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return int(h[:14], 16) & ((1 << 53) - 1)      # 53-bit: survives JSON/JS


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
        self._embed = embed
        self._index_fields = index_fields
        self._auto_save = auto_save

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
        meta = self.db.get_metadata(_ns_key(op.namespace, op.key))
        if _is_dead(meta):
            return None
        return self._to_item(op.namespace, op.key, meta)

    def _put(self, op: Any) -> None:
        rid = _ns_key(op.namespace, op.key)

        if op.value is None:                       # delete
            self.db.forget(rid)
            return

        prev = self.db.get_metadata(rid)
        created = (prev.get_attribute(_CREATED_ATTR) if prev else "") or str(int(time.time()))

        meta = feather_db.Metadata()
        meta.timestamp    = int(time.time())       # doubles as updated_at
        meta.namespace_id = NS_SEP.join(op.namespace)
        meta.entity_id    = op.key
        meta.source       = "langgraph"
        meta.content      = _indexed_text(dict(op.value), self._index_fields)
        meta.set_attribute(_VALUE_ATTR, json.dumps(dict(op.value)))
        meta.set_attribute(_CREATED_ATTR, created)
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

        self.db.add(rid, vec, meta)

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
        if op.query and self._embed is not None:
            # Rank semantically over everything, then keep what is under the
            # prefix. Over-fetch because the prefix filter is applied after —
            # a namespace-prefix filter does not exist in the engine yet.
            q = np.asarray(self._embed(op.query), dtype=np.float32)
            want = (op.limit + op.offset)
            hits = self.db.search(q, k=max(want * 8, 50), record_salience=False)
            cand = [(h.id, h.score, h.metadata) for h in hits]
        else:
            ids = (self._ids_under(op.namespace_prefix) if op.namespace_prefix
                   else self.db.get_all_ids("text"))
            cand = []
            for i in ids:
                m = self.db.get_metadata(i)
                if m is not None:
                    cand.append((i, None, m))

        results = []
        semantic = bool(op.query and self._embed is not None)
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
            results.append(self._to_item(ns, meta.entity_id, meta, score=score))

        return results[op.offset: op.offset + op.limit]

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

    def close(self) -> None:
        self.db.save()
