"""Backpressure: bound the requests in flight, refuse the rest with 429.

Without a bound, overload shows up as latency: the write pool's queue is
unbounded, so a burst queues behind 8 threads, every request in it eventually
times out client-side, and the clients' retries arrive on top of the backlog
— the retry storm that turns a slow minute into an outage. With one, a request
the server cannot start soon is refused at once with `429` + `Retry-After`,
which costs nothing to send and tells the client exactly what to do.

Requests are counted per class, because they wait on different resources:

    read    GETs and the read-only POSTs (search, ...)   FEATHER_MAX_INFLIGHT_READS   (256)
    write   every other POST / PUT / PATCH / DELETE        FEATHER_MAX_INFLIGHT_WRITES  (64)
    upload  POST /v1/admin/upload (a disk copy + a load)   FEATHER_MAX_INFLIGHT_UPLOADS (2)
    file    POST /v1/{ns}/ingest_file (a model reads it)   FEATHER_MAX_INFLIGHT_FILES   (8)

Health, the SPAs and the docs are never refused. A limit of 0 disables it.
The count lives on the event loop (incremented and decremented there only),
so it needs no lock.
"""
import os
import re

from fastapi.responses import JSONResponse

from . import errors


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default


LIMITS = {
    "read": _env_int("FEATHER_MAX_INFLIGHT_READS", 256),
    "write": _env_int("FEATHER_MAX_INFLIGHT_WRITES", 64),
    "upload": _env_int("FEATHER_MAX_INFLIGHT_UPLOADS", 2),
    "file": _env_int("FEATHER_MAX_INFLIGHT_FILES", 8),
}
RETRY_AFTER_S = _env_int("FEATHER_RETRY_AFTER_S", 1) or 1

_FILE_POST = re.compile(r"^/v1/[^/]+/ingest_file$")
_READ_POSTS = re.compile(r"^/v1/[^/]+/(search|keyword_search|hybrid_search|context_chain)$")
_EXEMPT = ("/health", "/admin", "/app", "/static", "/docs", "/redoc", "/openapi.json", "/dashboard")


def classify(method: str, path: str):
    if path == "/" or path.startswith(_EXEMPT) or not path.startswith("/v1/"):
        return None
    if path == "/v1/admin/upload":
        return "upload"
    if method == "POST" and _FILE_POST.match(path):
        return "file"
    if method in ("GET", "HEAD", "OPTIONS") or (method == "POST" and _READ_POSTS.match(path)):
        return "read"
    return "write"


class Admission:
    """Pure ASGI middleware (see _MetricsMiddleware for why not BaseHTTPMiddleware)."""

    inflight = {k: 0 for k in LIMITS}
    rejected = {k: 0 for k in LIMITS}

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        cls = classify(scope.get("method", "GET"), scope.get("path", ""))
        limit = LIMITS.get(cls, 0) if cls else 0
        if not limit:
            return await self.app(scope, receive, send)
        if Admission.inflight[cls] >= limit:
            Admission.rejected[cls] += 1
            resp = JSONResponse(
                status_code=429, headers={"Retry-After": str(RETRY_AFTER_S)},
                content=errors.body(429, f"server busy: {limit} {cls} requests already in "
                                         f"flight; retry after {RETRY_AFTER_S}s",
                                    limit=limit, kind=cls))
            return await resp(scope, receive, send)
        Admission.inflight[cls] += 1
        try:
            await self.app(scope, receive, send)
        finally:
            Admission.inflight[cls] -= 1

    @classmethod
    def stats(cls) -> dict:
        return {k: {"inflight": cls.inflight[k], "limit": LIMITS[k], "rejected": cls.rejected[k]}
                for k in LIMITS}
