"""
feather-gateway — scale the Feather Cloud API out across processes/machines.

Each feather-api process is single-owner for its namespaces (the engine's file
lock enforces it) and bound to one core by the GIL. Run N of them with
FEATHER_SHARD_COUNT=N / FEATHER_SHARD_INDEX=i and put this gateway in front:

    FEATHER_SHARDS=http://api0:8000,http://api1:8000,... \
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
  other /v1/admin/*                   -> ?shard=i (default 0); per-shard data
  /health                             -> gateway + every shard
  everything else (/admin SPA, docs)  -> shard 0

owner(ns) = crc32(utf-8 ns) % N — byte-for-byte the rule in
feather-api/app/db_manager.py (tests/test_gateway.py keeps them in sync).
Changing N moves namespaces between shards: stop all shards (close() releases
the file locks), change N everywhere, start again. See docs/deploy-sharding.md.
"""
from __future__ import annotations

import asyncio
import os
import zlib
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

SHARDS: List[str] = [s.strip().rstrip("/") for s in os.getenv("FEATHER_SHARDS", "").split(",") if s.strip()]
TIMEOUT_S = float(os.getenv("FEATHER_GATEWAY_TIMEOUT_S", "120"))

# Hop-by-hop headers must not be forwarded (RFC 7230 §6.1); length is recomputed.
_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
        "trailers", "transfer-encoding", "upgrade", "host", "content-length"}

_client: Optional[httpx.AsyncClient] = None


def shard_owner(namespace: str, shard_count: int) -> int:
    return zlib.crc32(namespace.encode("utf-8")) % max(1, shard_count)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    if not SHARDS:
        raise RuntimeError("FEATHER_SHARDS is empty: set it to the shard base URLs, "
                           "in shard-index order (http://api0:8000,http://api1:8000,...)")
    _client = httpx.AsyncClient(timeout=TIMEOUT_S,
                                limits=httpx.Limits(max_connections=512, max_keepalive_connections=128))
    yield
    await _client.aclose()


app = FastAPI(title="Feather DB Gateway", lifespan=lifespan, docs_url=None, redoc_url=None)


def _fwd_headers(request: Request) -> dict:
    return {k: v for k, v in request.headers.items() if k.lower() not in _HOP}


async def _forward(shard: int, request: Request, body: Optional[bytes] = None) -> Response:
    url = SHARDS[shard] + request.url.path
    if request.url.query:
        url += "?" + request.url.query
    data = body if body is not None else await request.body()
    try:
        r = await _client.request(request.method, url, headers=_fwd_headers(request), content=data)
    except httpx.HTTPError as e:
        return JSONResponse(status_code=502, content={"detail": f"shard {shard} unreachable: {e}",
                                                      "shard": shard})
    headers = {k: v for k, v in r.headers.items() if k.lower() not in _HOP | {"content-encoding"}}
    headers["X-Feather-Shard"] = str(shard)
    return Response(content=r.content, status_code=r.status_code, headers=headers,
                    media_type=r.headers.get("content-type"))


async def _fan_out(request: Request, path: str, method: str = "GET", body: bytes = b""):
    async def one(i):
        try:
            r = await _client.request(method, SHARDS[i] + path, headers=_fwd_headers(request),
                                      content=body, params=dict(request.query_params))
            return i, r
        except httpx.HTTPError as e:
            return i, e
    return await asyncio.gather(*[one(i) for i in range(len(SHARDS))])


def _namespace_of(path: str) -> Optional[str]:
    parts = [p for p in path.split("/") if p]
    if len(parts) < 2 or parts[0] != "v1":
        return None
    if parts[1] == "namespaces":
        return parts[2] if len(parts) >= 3 else None
    if parts[1] == "admin":
        return None
    return parts[1]


# ── fan-out / broadcast routes ──────────────────────────────────────────────
@app.get("/health")
async def health(request: Request):
    res = await _fan_out(request, "/health")
    shards = []
    for i, r in res:
        ok = isinstance(r, httpx.Response) and r.status_code == 200
        shards.append({"shard": i, "url": SHARDS[i], "ok": ok,
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
        return JSONResponse(status_code=502, content={"detail": "no shard answered"})
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
    if failed or first is None:
        return JSONResponse(status_code=502, content={
            "detail": f"embedding config not applied on shards {failed}; re-send to converge",
            "failed_shards": failed})
    return Response(first.content, status_code=first.status_code, media_type="application/json")


@app.post("/v1/namespaces")
async def create_namespace(request: Request):
    body = await request.body()
    try:
        name = (await request.json()).get("name", "")
    except Exception:  # noqa: BLE001
        name = ""
    if not name:
        return await _forward(0, request, body)      # let the shard produce the 422
    return await _forward(shard_owner(name, len(SHARDS)), request, body)


@app.post("/v1/admin/upload")
async def upload(request: Request):
    ns = request.query_params.get("namespace") or request.headers.get("x-feather-namespace")
    if not ns:
        return JSONResponse(status_code=400, content={
            "detail": "behind the gateway, pass the target namespace as ?namespace=<ns> or an "
                      "X-Feather-Namespace header (in addition to the form field) so the upload "
                      "can be routed without parsing the multipart body"})
    return await _forward(shard_owner(ns, len(SHARDS)), request)


# ── everything else ─────────────────────────────────────────────────────────
@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy(full_path: str, request: Request):
    path = "/" + full_path
    ns = _namespace_of(path)
    if ns is not None:
        return await _forward(shard_owner(ns, len(SHARDS)), request)
    shard = 0
    if path.startswith("/v1/admin/"):
        try:
            shard = int(request.query_params.get("shard", "0"))
        except ValueError:
            shard = -1
        if not 0 <= shard < len(SHARDS):
            return JSONResponse(status_code=400, content={"detail": f"shard must be 0..{len(SHARDS) - 1}"})
    return await _forward(shard, request)
