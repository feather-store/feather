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
import time
from typing import Iterable, Optional

import feather_db

# The writers whose records used the hashed id. migrate() touches only those.
LEGACY_SOURCES = ("pocket", "_pocket_cache", "langgraph")


def legacy_id(namespace: str, key: str) -> int:
    """The id a pre-0.22 writer derived for (namespace, key). Only for finding
    records written then; never use it to place a new one."""
    h = hashlib.sha1(f"{namespace}\x00{key}".encode("utf-8")).hexdigest()
    return int(h[:14], 16) & ((1 << 53) - 1)


def _legacy_record(db, namespace: str, key: str):
    """(legacy id, its metadata) if a pre-0.22 record holds (namespace, key),
    else (None, None)."""
    lid = legacy_id(namespace, key)
    m = db.get_metadata(lid)
    # The id alone proves nothing — under a collision it holds ANOTHER key's
    # record. It is this key's only if the record says so.
    if (m is not None and not m.key and m.source != "_forgotten"
            and m.namespace_id == namespace and m.entity_id == key):
        return lid, m
    return None, None


def _legacy(db, namespace: str, key: str) -> Optional[int]:
    return _legacy_record(db, namespace, key)[0]


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
    """(id, metadata) of the record keyed (namespace, key), or (None, None).
    One lookup returns both (DB.lookup_key: one request over the Cloud API,
    where resolving then reading was two); a legacy record costs one more."""
    if hasattr(db, "lookup_key"):
        hit = db.lookup_key(key, namespace)
        if hit is not None:
            return hit[0], hit[1]
    else:                                        # a DB built before lookup_key
        rid = db.id_for(key, namespace)
        if rid is not None:
            return rid, db.get_metadata(rid)
    return _legacy_record(db, namespace, key)


_UNKNOWN = object()


def write(db, namespace: str, key: str, meta, vec=None, modality: str = "text",
          found=_UNKNOWN) -> int:
    """Write the record keyed (namespace, key) and return its id: the existing
    record (a legacy one keeps its id and gains a native key), else a new one
    with an engine-allocated id. `found`: what get() returned for this key,
    when the caller has just read it, so the key is not looked up again."""
    meta.namespace_id = namespace
    meta.key = key
    rid, prev = get(db, namespace, key) if found is _UNKNOWN else found
    if rid is not None and prev is not None and not prev.key:      # legacy: adopt it
        if vec is not None:
            db.add(rid, vec, meta, modality=modality)
        else:
            db.update_metadata(rid, meta)
        return rid
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


# A file whose records all have their native keys (and whose index records
# their registry entries, backfill_registry) carries this record, so a start
# skips that pass. It read every record of the file on every start: over the
# Cloud API the whole dataset downloaded per start, long after there was
# nothing left to migrate. In the namespace "" (FeatherStore and Pocket never
# list or carry it), keyed, naming no subject. Bump UPKEEP_VERSION to run
# the pass once more on every file.
UPKEEP_KIND = "_feather_upkeep"
UPKEEP_KEY = "\x1fupkeep"
UPKEEP_ATTR = "_feather_upkeep_v"
UPKEEP_VERSION = "1"


def upkeep_done(db) -> bool:
    """Has this file had the one-off upkeep pass (one key lookup)?"""
    m = db.get_by_key(UPKEEP_KEY, "")
    return m is not None and m.get_attribute(UPKEEP_ATTR) == UPKEEP_VERSION


def upkeep(db, visit=None) -> int:
    """migrate() every namespace of the file, once: after the pass the file
    is marked and later calls cost one lookup. A read-only handle does
    nothing (it could write neither the keys nor the mark; resolve() still
    finds a legacy record by its id). Returns how many records were adopted.

    A record written at a hashed id after the mark (by a pre-0.22 client of
    the same file) is not adopted at start, but resolve() still finds it."""
    if db.is_read_only() or upkeep_done(db):
        return 0
    n = migrate(db, visit=visit)
    m = feather_db.Metadata()
    m.namespace_id, m.key, m.source = "", UPKEEP_KEY, UPKEEP_KIND
    m.timestamp = int(time.time())
    m.set_attribute(UPKEEP_ATTR, UPKEEP_VERSION)
    db.upsert(m)
    return n


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
