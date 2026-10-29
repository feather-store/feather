"""
feather-gateway — scale the Feather Cloud API out across processes/machines.

Each feather-api process is single-owner for its namespaces (the engine's file
lock enforces it) and bound to one core by the GIL. Run N of them, each with
FEATHER_SHARD_IDS=a,b,c and its own FEATHER_SHARD_ID, and put this gateway in
front (FEATHER_SHARD_COUNT=N / FEATHER_SHARD_INDEX=i still work: ids "0".."N-1"):

    FEATHER_SHARDS=a=http://api0:8000,b=http://api1:8000,... \
    uvicorn gateway:app --host 0.0.0.0 --port 8080

Routing
  /v1/{namespace}/...                 -> shard owner(namespace)
  /v1/namespaces/{namespace}[/...]    -> shard owner(namespace)
  POST /v1/namespaces {"name": ns}    -> shard owner(ns)
  GET  /v1/namespaces                 -> fan out, merged list
  GET  /v1/admin/overview             -> fan out, merged totals + top namespaces
  PUT  /v1/admin/embedding_config     -> broadcast to every shard
  POST /v1/admin/upload?namespace=ns  -> shard owner(ns)  (namespace must be in the
                                         query string or X-Feather-Namespace header:
                                         the gateway does not parse multipart bodies)
  GET  /v1/admin/placement            -> this gateway's placement table
  POST /v1/admin/placement/move       -> move one namespace, no restart
  other /v1/admin/*                   -> ?shard=<id> (default the first); per-shard data
  /health                             -> gateway + every shard
  everything else (/admin SPA, docs)  -> the first shard

owner(ns) = the placement table's override for ns, else rendezvous hashing
over the shard ids — the rule in feather-api/app/placement.py
(tests/test_gateway.py keeps them in sync). Adding a shard moves ~1/(N+1) of
the namespaces; see docs/deploy-sharding.md.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import zlib
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

def _parse_shards(spec: str):
    """FEATHER_SHARDS: `id=url,...`, or bare urls (ids "0".."N-1", in order)."""
    ids, urls = [], {}
    for n, entry in enumerate(e.strip() for e in spec.split(",") if e.strip()):
        sid, url = entry.split("=", 1) if "=" in entry.split("://", 1)[0] else (str(n), entry)
        sid = sid.strip()
        if sid in urls:
            raise RuntimeError(f"FEATHER_SHARDS names shard {sid!r} twice")
        ids.append(sid)
        urls[sid] = url.strip().rstrip("/")
    return ids, urls


SHARD_IDS, SHARD_URLS = _parse_shards(os.getenv("FEATHER_SHARDS", ""))
SHARDS: List[str] = [SHARD_URLS[i] for i in SHARD_IDS]
SHARD_HASH = os.getenv("FEATHER_SHARD_HASH", "rendezvous").strip().lower()
TIMEOUT_S = float(os.getenv("FEATHER_GATEWAY_TIMEOUT_S", "120"))
# The placement table (feather-api/app/placement.py): namespace -> shard id
# overrides, with an epoch. Loaded from the shards at startup; changed by
# POST /v1/admin/placement/move.
_placement = {"epoch": 0, "overrides": {}}

# Hop-by-hop headers must not be forwarded (RFC 7230 §6.1); length is recomputed.
_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
        "trailers", "transfer-encoding", "upgrade", "host", "content-length"}

_client: Optional[httpx.AsyncClient] = None


# ── ownership: a copy of feather-api/app/placement.py's (tests/test_gateway.py
# checks they agree) ─────────────────────────────────────────────────────────
def rendezvous_owner(namespace: str, shard_ids: List[str]) -> str:
    def weight(sid: str) -> int:
        h = hashlib.blake2b(f"{sid}\x00{namespace}".encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "big")
    return max(shard_ids, key=lambda sid: (weight(sid), sid))


def modulo_owner(namespace: str, shard_ids: List[str]) -> str:
    return shard_ids[zlib.crc32(namespace.encode("utf-8")) % len(shard_ids)]


def hash_owner(namespace: str, shard_ids: List[str] = None) -> str:
    ids = shard_ids if shard_ids is not None else SHARD_IDS
    return (modulo_owner if SHARD_HASH == "modulo" else rendezvous_owner)(namespace, ids)


def owner_of(namespace: str) -> str:
    o = _placement["overrides"].get(namespace)
    return o if o in SHARD_URLS else hash_owner(namespace)


def shard_owner(namespace: str, shard_count: int) -> int:
    """The owner's index among shards "0".."N-1", by the hash alone."""
    return int(hash_owner(namespace, [str(k) for k in range(max(1, shard_count))]))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    if not SHARDS:
        raise RuntimeError("FEATHER_SHARDS is empty: set it to the shard base URLs, "
                           "in shard-index order (http://api0:8000,http://api1:8000,...)")
    _client = httpx.AsyncClient(timeout=TIMEOUT_S,
                                limits=httpx.Limits(max_connections=512, max_keepalive_connections=128))
    await _load_placement()
    yield
    await _client.aclose()


app = FastAPI(title="Feather DB Gateway", lifespan=lifespan, docs_url=None, redoc_url=None)


def _fwd_headers(request: Request) -> dict:
    return {k: v for k, v in request.headers.items() if k.lower() not in _HOP}


def _error(status: int, detail: str, error: str, retryable: bool, **extra) -> JSONResponse:
    """The API's typed error shape (feather-api/app/errors.py)."""
    headers = {"Retry-After": "1"} if retryable else None
    return JSONResponse(status_code=status, headers=headers, content={
        "detail": detail, "error": error, "retryable": retryable, **extra})


async def _forward(shard: str, request: Request, body: Optional[bytes] = None,
                   redirected: bool = False) -> Response:
    url = SHARD_URLS[shard] + request.url.path
    if request.url.query:
        url += "?" + request.url.query
    data = body if body is not None else await request.body()
    try:
        r = await _client.request(request.method, url, headers=_fwd_headers(request), content=data)
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        # Never reached the shard, so nothing was applied: safe to send again.
        return _error(503, f"shard {shard} unreachable: {e}", "shard_unreachable", True, shard=shard)
    except httpx.TimeoutException as e:
        # Sent, but no answer: it may have been applied. Not retryable as
        # such; a client may resend only what is idempotent.
        return _error(504, f"shard {shard} did not answer in {TIMEOUT_S:.0f}s: {e}",
                      "timeout", False, shard=shard)
    except httpx.HTTPError as e:
        return _error(502, f"shard {shard} failed: {e}", "upstream_error", False, shard=shard)
    if r.status_code == 421 and not redirected:
        # The shard says another one owns the namespace: this gateway's view
        # is behind (a namespace just moved, or a reshard in progress). A 421
        # is refused before any work, so resending is safe for every method.
        # Once only: a second 421 goes to the client.
        try:
            owner = r.json().get("owner_shard")
        except ValueError:
            owner = None
        owner = None if owner is None else str(owner)
        if owner in SHARD_URLS and owner != shard:
            return await _forward(owner, request, data, redirected=True)
    headers = {k: v for k, v in r.headers.items() if k.lower() not in _HOP | {"content-encoding"}}
    headers["X-Feather-Shard"] = str(shard)
    return Response(content=r.content, status_code=r.status_code, headers=headers,
                    media_type=r.headers.get("content-type"))


async def _fan_out(request: Request, path: str, method: str = "GET", body: bytes = b""):
    async def one(i):
        try:
            r = await _client.request(method, SHARD_URLS[i] + path, headers=_fwd_headers(request),
                                      content=body, params=dict(request.query_params))
            return i, r
        except httpx.HTTPError as e:
            return i, e
    return await asyncio.gather(*[one(i) for i in SHARD_IDS])


# The API's own /v1/admin/<route> endpoints. Anything else under /v1/admin/ is
# a namespace that happens to be called "admin" (/v1/{namespace}/search ...).
_ADMIN_ROUTES = {"overview", "upload", "metrics", "activity", "ops_timeseries",
                 "connection_info", "embedding_config", "embedding_models"}


def _namespace_of(path: str, method: str = "GET") -> Optional[str]:
    parts = [p for p in path.split("/") if p]
    if len(parts) < 2 or parts[0] != "v1":
        return None
    if parts[1] == "namespaces":
        if len(parts) == 2:
            return None                                   # list / create
        if len(parts) == 3 and method == "DELETE":
            return parts[2]                               # DELETE /v1/namespaces/{ns}
        if len(parts) == 4 and parts[3] == "stats":
            return parts[2]                               # /v1/namespaces/{ns}/stats
        return "namespaces"                               # /v1/namespaces/search etc.
    if parts[1] == "admin" and (len(parts) == 2 or parts[2] in _ADMIN_ROUTES):
        return None
    return parts[1]


# ── fan-out / broadcast routes ──────────────────────────────────────────────
@app.get("/health")
async def health(request: Request):
    res = await _fan_out(request, "/health")
    shards = []
    for i, r in res:
        ok = isinstance(r, httpx.Response) and r.status_code == 200
        shards.append({"shard": i, "url": SHARD_URLS[i], "ok": ok,
                       **(r.json() if ok else {"error": str(r) if not isinstance(r, httpx.Response)
                                                 else r.status_code})})
    all_ok = all(s["ok"] for s in shards)
    return JSONResponse(status_code=200 if all_ok else 503,
                        content={"status": "ok" if all_ok else "degraded", "shards": shards,
                                 "namespaces_loaded": sum(s.get("namespaces_loaded", 0) for s in shards)})


@app.get("/v1/namespaces")
async def list_namespaces(request: Request):
    names, errors = [], []
    for i, r in await _fan_out(request, "/v1/namespaces"):
        if isinstance(r, httpx.Response) and r.status_code == 200:
            names.extend(r.json().get("namespaces", []))
        elif isinstance(r, httpx.Response) and r.status_code in (401, 403):
            return Response(r.content, status_code=r.status_code, media_type="application/json")
        else:
            errors.append(i)
    body = {"namespaces": sorted(set(names))}
    if errors:
        body["unreachable_shards"] = errors
    return JSONResponse(body, status_code=200 if not errors else 206)


@app.get("/v1/admin/overview")
async def overview(request: Request):
    parts = []
    for i, r in await _fan_out(request, "/v1/admin/overview"):
        if isinstance(r, httpx.Response) and r.status_code in (401, 403):
            return Response(r.content, status_code=r.status_code, media_type="application/json")
        if isinstance(r, httpx.Response) and r.status_code == 200:
            parts.append(r.json())
    if not parts:
        return _error(503, "no shard answered", "shard_unreachable", True)
    top = sorted((ns for p in parts for ns in p.get("topNamespaces", [])),
                 key=lambda x: x.get("records", 0), reverse=True)[:10]
    return {"version": parts[0].get("version"), "uptime": min(p.get("uptime", 0) for p in parts),
            "dim": parts[0].get("dim"), "nsCount": sum(p.get("nsCount", 0) for p in parts),
            "totalRecords": sum(p.get("totalRecords", 0) for p in parts), "topNamespaces": top,
            "shards": len(SHARDS), "shardsAnswered": len(parts)}


@app.put("/v1/admin/embedding_config")
async def embedding_config_put(request: Request):
    body = await request.body()
    res = await _fan_out(request, "/v1/admin/embedding_config", method="PUT", body=body)
    failed = [i for i, r in res if not (isinstance(r, httpx.Response) and r.status_code < 300)]
    first = next((r for _, r in res if isinstance(r, httpx.Response)), None)
    # Every shard refused the request itself (bad key, invalid body): that is
    # the client's error, so pass it through instead of calling it a 502.
    codes = {r.status_code for _, r in res if isinstance(r, httpx.Response)}
    if first is not None and len(codes) == 1 and 400 <= first.status_code < 500 \
            and all(isinstance(r, httpx.Response) for _, r in res):
        return Response(first.content, status_code=first.status_code, media_type="application/json")
    if failed or first is None:
        # A PUT: re-sending converges, so it is retryable (still a 502).
        return _error(502, f"embedding config not applied on shards {failed}; re-send to converge",
                      "upstream_error", True, failed_shards=failed)
    return Response(first.content, status_code=first.status_code, media_type="application/json")


@app.post("/v1/namespaces")
async def create_namespace(request: Request):
    body = await request.body()
    try:
        name = (await request.json()).get("name", "")
    except Exception:  # noqa: BLE001
        name = ""
    if not name or not isinstance(name, str):
        return await _forward(SHARD_IDS[0], request, body)   # let the shard produce the 422
    return await _forward(owner_of(name), request, body)


@app.post("/v1/admin/upload")
async def upload(request: Request):
    ns = request.query_params.get("namespace") or request.headers.get("x-feather-namespace")
    if not ns:
        return JSONResponse(status_code=400, content={
            "error": "invalid_request", "retryable": False,
            "detail": "behind the gateway, pass the target namespace as ?namespace=<ns> or an "
                      "X-Feather-Namespace header (in addition to the form field) so the upload "
                      "can be routed without parsing the multipart body"})
    return await _forward(owner_of(ns), request)


# ── placement: moving a namespace with no restart ───────────────────────────
async def _load_placement(headers: Optional[dict] = None):
    """The newest table any shard has (they may share one file). At startup
    with FEATHER_API_KEY, if the gateway has one; a shard it cannot read from
    still corrects a stale route with a 421, which _forward follows."""
    if headers is None:
        key = os.getenv("FEATHER_API_KEY", "")
        headers = {"X-API-Key": key} if key else {}
    for sid in SHARD_IDS:
        try:
            r = await _client.get(SHARD_URLS[sid] + "/v1/admin/placement", headers=headers)
            st = r.json() if r.status_code == 200 else {}
        except (httpx.HTTPError, ValueError):
            continue
        if int(st.get("epoch", 0)) > _placement["epoch"]:
            _placement.update(epoch=int(st["epoch"]), overrides=dict(st.get("overrides", {})))


@app.get("/v1/admin/placement")
async def get_placement():
    return {"shards": SHARD_IDS, "hash": SHARD_HASH, **_placement}


@app.post("/v1/admin/placement/move")
async def move_namespace(request: Request):
    """Move one namespace to another shard, with no restart: {"namespace", "shard"}.

    1. The target takes the new table: it owns the namespace from here on (it
       cannot open the file yet: the source still holds its lock).
    2. The source takes it and hands the namespace over: its in-flight writes
       finish, it checkpoints and closes the file, releasing the lock. From here
       it answers 421, which this gateway follows to the target.
    3. This gateway takes it and routes straight to the target; then the other
       shards, so their 421s point the right way.
    Requests in between are either served by the source or redirected to the
    target, which opens the file on the first one. Needs a data volume both
    shards see: a target that cannot see the namespace's file is refused.
    Moving a namespace back to its hash owner removes its override."""
    try:
        body = await request.json()
        ns, target = str(body["namespace"]), str(body["shard"])
    except Exception:  # noqa: BLE001
        return _error(400, 'body must be {"namespace": ..., "shard": ...}', "invalid_request", False)
    if target not in SHARD_URLS:
        return _error(400, f"shard must be one of {SHARD_IDS}", "invalid_request", False)
    hdrs = _fwd_headers(request)
    await _load_placement(hdrs)
    source = owner_of(ns)
    if source == target:
        return {"namespace": ns, "from": source, "to": target, "moved": False, **_placement}
    checks = {}
    for sid in (source, target):
        try:
            r = await _client.get(SHARD_URLS[sid] + "/v1/admin/placement", params={"namespace": ns},
                                  headers=hdrs)
            checks[sid] = r.json()["namespace"]
        except Exception as e:  # noqa: BLE001
            return _error(503, f"shard {sid} unreachable: {e}", "shard_unreachable", True, shard=sid)
    if checks[source]["has_file"] and not checks[target]["has_file"]:
        return _error(409, f"shard {target} cannot see {ns}.feather: a live move needs a data "
                           "volume both shards share", "conflict", False)
    overrides = dict(_placement["overrides"])
    if hash_owner(ns) == target:
        overrides.pop(ns, None)
    else:
        overrides[ns] = target
    table = {"epoch": _placement["epoch"] + 1, "overrides": overrides}

    async def put(sid):
        r = await _client.put(SHARD_URLS[sid] + "/v1/admin/placement", json=table, headers=hdrs)
        if r.status_code >= 300:
            raise RuntimeError(f"shard {sid}: HTTP {r.status_code} {r.text[:200]}")
        return r.json()
    try:
        await put(target)
        released = (await put(source)).get("released", [])
    except Exception as e:  # noqa: BLE001
        return _error(502, f"move of {ns} incomplete: {e}; re-send to finish it",
                      "upstream_error", True, namespace=ns)
    _placement.update(table)
    unreached = []
    for sid in SHARD_IDS:
        if sid not in (source, target):
            try:
                await put(sid)
            except Exception:  # noqa: BLE001
                unreached.append(sid)
    return {"namespace": ns, "from": source, "to": target, "moved": True,
            "released": ns in released, "unreached_shards": unreached, **_placement}


# ── everything else ─────────────────────────────────────────────────────────
@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy(full_path: str, request: Request):
    path = "/" + full_path
    ns = _namespace_of(path, request.method)
    if ns is not None:
        return await _forward(owner_of(ns), request)
    shard = SHARD_IDS[0]
    if path.startswith("/v1/admin/"):
        shard = request.query_params.get("shard", SHARD_IDS[0])
        if shard not in SHARD_URLS:
            return _error(400, f"shard must be one of {SHARD_IDS}", "invalid_request", False)
    return await _forward(shard, request)
