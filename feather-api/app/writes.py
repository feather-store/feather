"""Record writes for the Cloud API: atomic batches, compare-and-set, retries.

Every route that writes records builds a list of `Op`s and hands them to
`apply()`, which sends them to the engine as ONE `WriteBatch`: all or nothing,
one WAL write, one lock. Two HTTP features ride on that:

* `Idempotency-Key` header. The engine stores the key, a fingerprint of the
  request and the response body in the same WAL write as the batch, so a retry
  — even one after a crash — returns the first response (with the header
  `Idempotent-Replayed: true`) instead of applying the write again. The same key
  with a different request is a 422. A route checks `prior()` BEFORE its own
  checks: a DELETE retried after it succeeded must not answer 404.
* `If-Match: <version>` (records carry `version` and an `ETag`). A write based
  on a stale read is refused with 412 instead of silently overwriting the write
  it never saw.

On a feather-db wheel without WriteBatch (< 0.22) the ops are applied one by
one, as before, and both headers are refused with 501 rather than ignored.
"""
import hashlib
import json
import re
from typing import Callable, List, NamedTuple, Optional

from fastapi import HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool

import feather_db

from .errors import APIError

HAS_BATCH = hasattr(feather_db, "WriteBatch")
VersionConflictError = getattr(feather_db, "VersionConflictError", None)
IdempotencyKeyReusedError = getattr(feather_db, "IdempotencyKeyReusedError", None)
DuplicateKeyError = getattr(feather_db, "DuplicateKeyError", None)

IDEMPOTENCY_HEADER = "Idempotency-Key"
REPLAYED_HEADER = "Idempotent-Replayed"
MAX_KEY_LEN = 255


class Idem(NamedTuple):
    key: str = ""
    fingerprint: str = ""


class Op(NamedTuple):
    kind: str                       # "put" | "update" | "link" | "forget" | "upsert"
    id: int = 0
    meta: object = None             # feather_db.Metadata
    vec: object = None              # np.ndarray for put (and optionally upsert)
    modality: str = "text"
    to_id: int = 0
    rel_type: str = "related_to"
    weight: float = 1.0
    expected: Optional[int] = None
    ref: Optional[tuple] = None     # (namespace, key) instead of `id`
    to_ref: Optional[tuple] = None  # link: (namespace, key) instead of `to_id`
    require_target: bool = True     # link: false allows a forward reference


async def idempotency(request: Request) -> Idem:
    """FastAPI dependency: the request's key, and a fingerprint of everything
    the request says (method, path, query, body). The fingerprint is what
    tells a retry apart from a different request reusing the key."""
    key = _checked_key(request)
    if not key:
        return Idem()
    h = hashlib.sha256()
    h.update(f"{request.method} {request.url.path}?{request.url.query}\n".encode())
    body = await request.body()
    if len(body) > _HASH_INLINE_MAX:
        await run_in_threadpool(h.update, body)     # tens of ms for a large batch
    else:
        h.update(body)
    return Idem(key, h.hexdigest())


_HASH_INLINE_MAX = 1 << 20


def idempotency_of(request: Request, fingerprint: str) -> Idem:
    """As `idempotency`, for a route that fingerprints its own request: a
    multipart upload, whose body the form parser has already consumed and
    which is too large to hash a second time."""
    key = _checked_key(request)
    return Idem(key, fingerprint) if key else Idem()


def _checked_key(request: Request) -> str:
    key = request.headers.get(IDEMPOTENCY_HEADER, "")
    if key and len(key) > MAX_KEY_LEN:
        raise HTTPException(400, f"{IDEMPOTENCY_HEADER} is longer than {MAX_KEY_LEN} characters")
    if key and not HAS_BATCH:
        raise HTTPException(501, f"{IDEMPOTENCY_HEADER} needs feather-db >= 0.22 on the server")
    return key


def if_match_version(value: Optional[str]) -> Optional[int]:
    """`If-Match: 42`, `"42"` or `W/"42"` -> 42. `*` means any existing
    record, which every update requires anyway -> no condition."""
    if value is None or value.strip() in ("", "*"):
        return None
    v = value.strip()
    if v.startswith("W/"):
        v = v[2:]
    v = v.strip('"')
    if not v.isdigit():
        raise HTTPException(400, f"If-Match must be a record version, got {value!r}")
    if not HAS_BATCH:
        raise HTTPException(501, "If-Match needs feather-db >= 0.22 on the server")
    return int(v)


def etag(version: int) -> str:
    return f'"{version}"'


def prior(db, idem: Idem):
    """The stored result of an earlier attempt with this key, or None."""
    if not idem.key:
        return None
    try:
        return db.idempotent_result(idem.key, idem.fingerprint)
    except Exception as e:
        _raise_mapped(e)


def apply(db, ops: List[Op], idem: Idem = Idem(), note: Optional[dict] = None):
    """Write `ops` atomically. Returns the engine's BatchResult (None when
    there was nothing to write, or on an engine without batches). `note` is
    the response body to hand back to a retry."""
    if not ops:
        return None
    if not HAS_BATCH:
        if (idem.key or any(op.expected is not None or op.ref or op.to_ref for op in ops)
                or any(op.kind == "upsert" for op in ops)):
            raise HTTPException(501, "conditional, keyed or idempotent writes need feather-db >= 0.22")
        for op in ops:
            if op.kind == "put":
                db.add(id=op.id, vec=op.vec, meta=op.meta, modality=op.modality)
            elif op.kind == "update":
                db.update_metadata(op.id, op.meta)
            elif op.kind == "link":
                db.link(from_id=op.id, to_id=op.to_id, rel_type=op.rel_type, weight=op.weight)
            elif op.kind == "forget":
                db.forget(op.id)
        return None
    b = feather_db.WriteBatch()
    for op in ops:
        target = op.ref if op.ref else op.id
        if op.kind == "put":
            b.put(op.id, op.vec, op.meta, modality=op.modality, expected_version=op.expected)
        elif op.kind == "upsert":
            b.upsert(op.meta, op.vec, modality=op.modality, expected_version=op.expected)
        elif op.kind == "update":
            b.update(target, op.meta, expected_version=op.expected)
        elif op.kind == "link":
            b.link(target, op.to_ref if op.to_ref else op.to_id, rel_type=op.rel_type,
                   weight=op.weight, expected_version=op.expected, require_target=op.require_target)
        elif op.kind == "forget":
            b.forget(target, expected_version=op.expected)
        else:
            raise HTTPException(400, f"unknown op {op.kind!r}")
    try:
        return db.write(b, idempotency_key=idem.key or None, fingerprint=idem.fingerprint,
                        note=json.dumps(note) if idem.key and note is not None else "")
    except Exception as e:
        _raise_mapped(e)


def respond(response: Optional[Response], body: dict, result,
            extra: Optional[Callable[[object], dict]] = None) -> dict:
    """The route's response: `body` (or, for a replay, the body the first
    attempt stored) plus whatever `extra` derives from the engine result."""
    if result is not None and getattr(result, "replayed", False):
        if result.note:
            body = json.loads(result.note)
        if response is not None:
            response.headers[REPLAYED_HEADER] = "true"
    if result is not None and extra is not None:
        body = {**body, **extra(result)}
    return body


def _raise_mapped(e: Exception):
    if VersionConflictError is not None and isinstance(e, VersionConflictError):
        raise APIError(409, str(e), "version_conflict", id=e.id,
                       expected_version=e.expected, actual_version=e.actual)
    if DuplicateKeyError is not None and isinstance(e, DuplicateKeyError):
        raise APIError(409, str(e), "duplicate_key")
    if IdempotencyKeyReusedError is not None and isinstance(e, IdempotencyKeyReusedError):
        raise APIError(422, f"{IDEMPOTENCY_HEADER} was already used for a different request",
                       "idempotency_key_reused")
    if isinstance(e, ValueError):            # e.g. a link to a record that does not exist
        raise HTTPException(400, str(e))
    if isinstance(e, RuntimeError) and "Dimension mismatch" in str(e):
        raise HTTPException(400, str(e))
    raise e


def precondition_failed(e: HTTPException) -> HTTPException:
    """A 409 version conflict, re-raised as the 412 that If-Match calls for."""
    extra = getattr(e, "extra", {})
    if e.status_code == 409 and "actual_version" in extra:
        return APIError(412, e.detail, "precondition_failed",
                        headers={"ETag": etag(extra["actual_version"])}, **extra)
    return e


# Routes that write and cannot honour an Idempotency-Key: a key there is
# refused, not silently ignored, because the caller is relying on it. Routes
# that are idempotent by nature (purge, compact, save, flush, settings) accept
# and ignore it; reads ignore it.
_KEY_UNSUPPORTED = [
    ("POST", re.compile(r"^/v1/[^/]+/(import|seed)$")),
    ("POST", re.compile(r"^/v1/admin/upload$")),
    ("POST", re.compile(r"^/v1/namespaces$")),
    ("DELETE", re.compile(r"^/v1/namespaces/[^/]+$")),
]


def key_unsupported(method: str, path: str) -> bool:
    return any(method == m and rx.match(path) for m, rx in _KEY_UNSUPPORTED)
