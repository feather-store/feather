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

Fair between tenants. A class's slots are shared by the namespaces using
them (the namespace in the path; /v1/admin/* is one tenant): with n holding
slots, a tenant may hold at most limit / n, and a tenant alone at most
limit minus a reserve (FEATHER_ADMISSION_RESERVE, 1/8 of the limit, for
limits of 4 or more). So one tenant's bulk import cannot take every write
slot: another tenant's request finds the reserve free and is admitted at
once, and from then on the two share. Nothing is preempted; a tenant over
its share is refused (429, `fair_share`) until its own requests finish.
O(1) per request. There is no queue to schedule (a request is admitted or
refused at once, see above), so this is deficit round robin's fairness
without its queue. FEATHER_FAIR_ADMISSION=0 turns it off.
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
FAIR = os.getenv("FEATHER_FAIR_ADMISSION", "1") != "0"
try:
    RESERVE = min(0.5, max(0.0, float(os.getenv("FEATHER_ADMISSION_RESERVE", "0.125"))))
except ValueError:
    RESERVE = 0.125

_TENANT = re.compile(r"^/v1/([^/]+)")


def tenant_of(path: str) -> str:
    m = _TENANT.match(path)
    return m.group(1) if m else ""


def fair_cap(limit: int, others: int) -> int:
    """The most slots of `limit` one tenant may hold while `others` other
    tenants hold some."""
    if others == 0:
        reserve = max(1, int(limit * RESERVE)) if limit >= 4 and RESERVE > 0 else 0
        return max(1, limit - reserve)
    return max(1, limit // (others + 1))

_FILE_POST = re.compile(r"^/v1/[^/]+/ingest_file$")
_READ_POSTS = re.compile(
    r"^/v1/[^/]+/(search|keyword_search|hybrid_search|context_chain|records/get_many|by_keys|heat_select)$")
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
    rejected_fair = {k: 0 for k in LIMITS}
    held = {k: {} for k in LIMITS}          # class -> tenant -> requests in flight

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        cls = classify(scope.get("method", "GET"), path)
        limit = LIMITS.get(cls, 0) if cls else 0
        if not limit:
            return await self.app(scope, receive, send)
        tenant = tenant_of(path)
        held = Admission.held[cls]
        mine = held.get(tenant, 0)
        if Admission.inflight[cls] >= limit:
            Admission.rejected[cls] += 1
            return await self._refuse(scope, receive, send,
                                      f"server busy: {limit} {cls} requests already in flight",
                                      limit=limit, kind=cls)
        if FAIR:
            cap = fair_cap(limit, len(held) - (1 if mine else 0))
            if mine >= cap:
                Admission.rejected[cls] += 1
                Admission.rejected_fair[cls] += 1
                return await self._refuse(
                    scope, receive, send,
                    f"fair share: namespace {tenant!r} already has {mine} of {limit} {cls} "
                    f"requests in flight", limit=limit, kind=cls, share=cap,
                    error="fair_share")
        Admission.inflight[cls] += 1
        held[tenant] = mine + 1
        try:
            await self.app(scope, receive, send)
        finally:
            Admission.inflight[cls] -= 1
            n = held[tenant] - 1
            if n:
                held[tenant] = n
            else:
                del held[tenant]

    @staticmethod
    async def _refuse(scope, receive, send, why: str, error: str = None, **extra):
        body = errors.body(429, f"{why}; retry after {RETRY_AFTER_S}s", error, **extra)
        resp = JSONResponse(status_code=429, headers={"Retry-After": str(RETRY_AFTER_S)},
                            content=body)
        return await resp(scope, receive, send)

    @classmethod
    def stats(cls) -> dict:
        return {k: {"inflight": cls.inflight[k], "limit": LIMITS[k], "rejected": cls.rejected[k],
                    "rejected_fair": cls.rejected_fair[k], "tenants": len(cls.held[k])}
                for k in LIMITS}
