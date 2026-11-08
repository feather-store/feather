"""Records addressed by (namespace, key) — the one place that knows how.

Pocket, the packet builder and FeatherStore all store records under a string
key. Until 0.22 each derived the record id from a 53-bit SHA-1 of the
namespace and key, so two keys that collided shared one record and the second
write silently overwrote the first. The engine now has native keys
(`Metadata.key`, unique per namespace_id, ids allocated rather than hashed),
and BM25 indexes the key, so a search for the name a memory was stored under
finds it.

Files written before that still hold records at the hashed id, with the key
only in `entity_id`. `resolve()` finds them; `write()` adopts one into a native
key the first time it is written (same id, so edges into it stay valid); and
`migrate()` adopts them all at once, which is what makes their keys searchable.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Optional

import feather_db

# The writers whose records used the hashed id. migrate() touches only those.
LEGACY_SOURCES = ("pocket", "_pocket_cache", "langgraph")


def legacy_id(namespace: str, key: str) -> int:
    """The id a pre-0.22 writer derived for (namespace, key). Only for finding
    records written then; never use it to place a new one."""
    h = hashlib.sha1(f"{namespace}\x00{key}".encode("utf-8")).hexdigest()
    return int(h[:14], 16) & ((1 << 53) - 1)


def _legacy(db, namespace: str, key: str) -> Optional[int]:
    lid = legacy_id(namespace, key)
    m = db.get_metadata(lid)
    # The id alone proves nothing — under a collision it holds ANOTHER key's
    # record. It is this key's only if the record says so.
    if (m is not None and not m.key and m.source != "_forgotten"
            and m.namespace_id == namespace and m.entity_id == key):
        return lid
    return None


def resolve(db, namespace: str, key: str) -> Optional[int]:
    """The id of the live record keyed (namespace, key), or None."""
    rid = db.id_for(key, namespace)
    return rid if rid is not None else _legacy(db, namespace, key)


def resolve_many(db, refs) -> list:
    """resolve() for many (namespace, key) pairs: ids (None where there is no
    live record), in order. One key lookup for all of them, and one metadata
    read for the misses' legacy ids, instead of a round trip per key."""
    refs = list(refs)
    if not refs:
        return []
    if hasattr(db, "ids_for_many"):
        ids = list(db.ids_for_many(refs))
    else:                                        # a DB built before ids_for_many
        ids = [db.id_for(k, ns) for ns, k in refs]
    miss = [i for i, x in enumerate(ids) if x is None]
    if miss:
        lids = [legacy_id(*refs[i]) for i in miss]
        for i, lid, m in zip(miss, lids, db.get_metadata_many(lids)):
            ns, k = refs[i]
            if (m is not None and not m.key and m.source != "_forgotten"
                    and m.namespace_id == ns and m.entity_id == k):
                ids[i] = lid
    return ids


def get(db, namespace: str, key: str):
    """(id, metadata) of the record keyed (namespace, key), or (None, None)."""
    rid = resolve(db, namespace, key)
    return (rid, db.get_metadata(rid)) if rid is not None else (None, None)


def write(db, namespace: str, key: str, meta, vec=None, modality: str = "text") -> int:
    """Write the record keyed (namespace, key) and return its id: the existing
    record (a legacy one keeps its id and gains a native key), else a new one
    with an engine-allocated id."""
    meta.namespace_id = namespace
    meta.key = key
    if db.id_for(key, namespace) is None:
        lid = _legacy(db, namespace, key)
        if lid is not None:
            if vec is not None:
                db.add(lid, vec, meta, modality=modality)
            else:
                db.update_metadata(lid, meta)
            return lid
    return db.upsert(meta, vec, modality=modality)[0]


def migrate(db, namespaces: Optional[Iterable[str]] = None, batch: int = 512,
            visit=None) -> int:
    """Give every legacy record in `namespaces` (default: all) its native key.
    Idempotent, and cheap once done: a record that already has a key is
    skipped. Returns how many records were adopted. `visit(namespace, ids,
    metas)`, if given, sees each namespace's records as read: other one-off
    upkeep shares this read instead of making its own."""
    names = list(namespaces) if namespaces is not None else db.list_namespaces()
    adopted, pending = 0, []
    for ns in names:
        ids = list(db.ids_in_namespace(ns))
        # One read per namespace, not per record: an agent runs this on every
        # start, and over the Cloud API each record was a round trip.
        metas = db.get_metadata_many(ids) if ids else []
        if visit is not None:
            visit(ns, ids, metas)
        for rid, m in zip(ids, metas):
            if (m is None or m.key or not m.entity_id or m.source not in LEGACY_SOURCES
                    or rid != legacy_id(ns, m.entity_id)):
                continue
            m.key = m.entity_id
            pending.append((rid, m))
            if len(pending) >= batch:
                adopted += _adopt(db, pending)
                pending = []
    return adopted + _adopt(db, pending)


def _adopt(db, pending) -> int:
    if not pending:
        return 0
    b = feather_db.WriteBatch()
    for rid, m in pending:
        b.update(rid, m)
    try:
        db.write(b)
        return len(pending)
    except feather_db.DuplicateKeyError:
        # A native record already holds one of these keys (written by a
        # build that did not see the legacy one). Adopt the rest one by one,
        # so the conflict leaves one record unkeyed instead of the batch.
        n = 0
        for rid, m in pending:
            try:
                db.update_metadata(rid, m)
                n += 1
            except feather_db.DuplicateKeyError:
                pass
        return n
