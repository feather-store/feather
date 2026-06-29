"""
Feather DB Cloud API — FastAPI application.

Architecture: multi-tenant via DBManager.
Each namespace (brand, org, tenant) gets an isolated .feather file.

Endpoints:
  GET  /health
  GET  /v1/namespaces
  GET  /v1/namespaces/{namespace}/stats

  POST /v1/{namespace}/vectors          — add a vector
  POST /v1/{namespace}/search           — search
  GET  /v1/{namespace}/records/{id}     — get metadata
  PUT  /v1/{namespace}/records/{id}     — update full metadata
  PUT  /v1/{namespace}/records/{id}/importance  — update importance
  POST /v1/{namespace}/records/{id}/link        — link two records
  POST /v1/{namespace}/save             — flush to disk

Authentication: X-API-Key header (set FEATHER_API_KEY env var).
If FEATHER_API_KEY is unset, auth is disabled (dev mode).
"""

import os
import time
import logging
import pathlib
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional, List, Tuple

from fastapi import FastAPI, HTTPException, Depends, Header, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

import feather_db
from feather_db import Metadata, ContextType, ScoringConfig
from feather_db.core import SearchFilter

from .db_manager import DBManager
from .metrics import METRICS, classify, namespace_from_path
from .embedding import EMBEDDING, SUPPORTED_MODELS
from .models import (
    AddVectorRequest, SearchRequest, SearchResponse, SearchResultItem,
    KeywordSearchRequest, HybridSearchRequest,
    LinkRequest, UpdateImportanceRequest, UpdateMetadataRequest, PurgeRequest,
    MetadataOut, NamespaceStats, HealthResponse, AdminOverview,
    SeedRequest, ContextChainRequest, EdgesResponse, EdgeOut, IncomingEdgeOut,
    ContextChainNode, ContextChainEdge, ContextChainResponse,
    CreateNamespaceRequest, NamespaceSchema, SchemaAttribute,
    TopRecalledItem, OpsTimeseriesResponse, OpsTimeseriesPoint,
    ConnectionInfo, EmbeddingConfig, EmbeddingConfigUpdate,
    ImportRequest, ImportResponse, IngestTextRequest,
    HierarchyNode, HierarchyResponse,
    AutoCompactRequest, QuantizeRequest, IndexStatsResponse,
)
import numpy as np
import json
import random


_PROCESS_START = time.time()

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("feather-api")

# ─────────────────────────────────────────────
# App lifecycle
# ─────────────────────────────────────────────
manager: Optional[DBManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    logger.info("Starting Feather DB Cloud API...")
    manager = DBManager()
    logger.info(f"Loaded namespaces: {manager.list_namespaces()}")
    yield
    logger.info("Shutting down — saving all DBs...")
    manager.save_all()

app = FastAPI(
    title="Feather DB Cloud API",
    description="REST API for Feather DB — embedded vector database with living context.",
    version=feather_db.__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    dt = (time.perf_counter() - t0) * 1000
    path = request.url.path
    if path.startswith("/admin") or path.startswith("/static"):
        return response          # don't count static asset hits
    METRICS.record(
        op=classify(request.method, path),
        latency_ms=dt,
        namespace=namespace_from_path(path),
        status=response.status_code,
    )
    return response

# ─────────────────────────────────────────────
# Mount Atlas-style admin SPA at /admin (static).
# Custom route for /admin/ serves index.html with no-cache headers so a fresh
# deploy is always picked up by the browser (was: aggressive caching of old JS
# left users stuck on stale state after a deploy).
# /dashboard redirects to /admin so old links keep working.
# ─────────────────────────────────────────────
from fastapi.responses import FileResponse

_STATIC_DIR = pathlib.Path(__file__).parent.parent / "static" / "admin"
_INDEX_HTML = _STATIC_DIR / "index.html"

@app.get("/admin", include_in_schema=False)
@app.get("/admin/", include_in_schema=False)
def _admin_index():
    if not _INDEX_HTML.is_file():
        raise HTTPException(404, "admin SPA not deployed")
    return FileResponse(
        str(_INDEX_HTML),
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )

if _STATIC_DIR.is_dir():
    # any other static subpath (we don't have any today, but safe to keep)
    app.mount("/admin/static", StaticFiles(directory=str(_STATIC_DIR)), name="admin_static")
else:
    logger.warning("admin static dir not found at %s", _STATIC_DIR)

@app.get("/dashboard", include_in_schema=False)
@app.get("/dashboard/", include_in_schema=False)
def _dashboard_redirect():
    return RedirectResponse(url="/admin/", status_code=308)

@app.get("/", include_in_schema=False)
def _root():
    return RedirectResponse(url="/admin/", status_code=308)

# ─────────────────────────────────────────────
# Auth middleware
# ─────────────────────────────────────────────
API_KEY = os.getenv("FEATHER_API_KEY", "")

def verify_api_key(x_api_key: str = Header(default="")):
    if not API_KEY:
        return   # dev mode — no key required
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _meta_from_model(m_in) -> Metadata:
    meta = Metadata()
    meta.timestamp       = m_in.timestamp if m_in.timestamp else int(time.time())
    meta.importance      = m_in.importance
    meta.type            = ContextType(int(m_in.type))
    meta.source          = m_in.source
    meta.content         = m_in.content
    meta.tags_json       = m_in.tags_json
    meta.namespace_id    = m_in.namespace_id
    meta.entity_id       = m_in.entity_id
    for k, v in m_in.attributes.items():
        meta.set_attribute(k, v)
    return meta


def _meta_to_model(meta: Metadata) -> MetadataOut:
    return MetadataOut(
        timestamp       = meta.timestamp,
        importance      = meta.importance,
        type            = int(meta.type),
        source          = meta.source,
        content         = meta.content,
        tags_json       = meta.tags_json,
        namespace_id    = meta.namespace_id,
        entity_id       = meta.entity_id,
        attributes      = dict(meta.attributes),
        recall_count    = meta.recall_count,
        last_recalled_at= meta.last_recalled_at,
        links           = list(meta.links),
    )


def _build_filter(req: SearchRequest) -> Optional[SearchFilter]:
    has_filter = any([
        req.namespace_id, req.entity_id, req.attributes_match,
        req.source, req.source_prefix, req.importance_gte,
        req.tags_contains, req.timestamp_after, req.timestamp_before,
    ])
    if not has_filter:
        return None

    f = SearchFilter()
    if req.namespace_id:      f.namespace_id    = req.namespace_id
    if req.entity_id:         f.entity_id       = req.entity_id
    if req.attributes_match:  f.attributes_match = req.attributes_match
    if req.source:            f.source          = req.source
    if req.source_prefix:     f.source_prefix   = req.source_prefix
    if req.importance_gte is not None: f.importance_gte = req.importance_gte
    if req.tags_contains:     f.tags_contains   = req.tags_contains
    if req.timestamp_after:   f.timestamp_after = req.timestamp_after
    if req.timestamp_before:  f.timestamp_before = req.timestamp_before
    return f


def _build_scoring(req: SearchRequest) -> Optional[ScoringConfig]:
    if req.scoring_half_life is None and req.scoring_weight is None:
        return None
    return ScoringConfig(
        half_life = req.scoring_half_life or 30.0,
        weight    = req.scoring_weight    or 0.3,
        min       = req.scoring_min       or 0.0,
    )


def _established_dim(db, modality: str = "text"):
    """The dim of `modality`'s index IF it already exists, else None.

    A namespace has no fixed dimension until its first vector lands — the engine
    adopts that vector's length (vec.size()). An empty namespace's db.dim() only
    reports the server default (768); treating that as the real dim is exactly
    what silently forced every non-768 embedding to be padded or rejected. So we
    enforce a dim ONLY once the index actually exists; otherwise the first
    inserted vector is free to define any dimension."""
    try:
        if modality in set(db.modality_names()):
            return db.dim(modality)
    except Exception:
        pass
    return None


def _check_query_dim(db, vector, modality: str):
    """Reject a query whose length doesn't match an *established* index dim,
    with a clean 400 instead of letting the C++ HNSW read out of bounds."""
    expected = _established_dim(db, modality)
    if expected and len(vector) != expected:
        raise HTTPException(
            400,
            f"Query vector dim {len(vector)} != index dim {expected} "
            f"for modality '{modality}'",
        )


# Bounded thread pool so a large auto-embed import fires provider calls
# concurrently instead of one-at-a-time (each EMBEDDING.embed is a blocking HTTP
# call; the config lock is only held briefly, so concurrent calls are safe).
_EMBED_WORKERS = max(1, int(os.getenv("FEATHER_EMBED_CONCURRENCY", "8")))


def _embed_many(texts: List[str]) -> List[Tuple[Optional[list], Optional[str]]]:
    """Embed many strings concurrently, preserving order. Returns one
    (vector, error) tuple per input — never raises for a single bad item."""
    if not texts:
        return []
    out: List[Tuple[Optional[list], Optional[str]]] = [(None, None)] * len(texts)

    def work(i: int):
        try:
            return i, EMBEDDING.embed(texts[i]), None
        except Exception as e:  # noqa: BLE001
            return i, None, str(e)

    workers = min(_EMBED_WORKERS, len(texts))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, vec, err in ex.map(work, range(len(texts))):
            out[i] = (vec, err)
    return out

# ─────────────────────────────────────────────
# Routes — health & meta
# ─────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(
        status="ok",
        version=feather_db.__version__,
        namespaces_loaded=len(manager.list_namespaces()),
    )

@app.get("/v1/namespaces", tags=["meta"], dependencies=[Depends(verify_api_key)])
def list_namespaces():
    return {"namespaces": manager.list_namespaces()}


@app.post("/v1/namespaces", status_code=201, tags=["meta"],
          dependencies=[Depends(verify_api_key)])
def create_namespace(req: CreateNamespaceRequest):
    """Create an empty namespace (a new .feather file on disk).

    A namespace has no fixed dimension until its first vector — pass an optional
    `dim` to pin it up front (so the dashboard shows the right dimension and
    clients get an early 400 on mismatch). With no data yet this is only a
    reported default; the first inserted vector is what truly fixes the dim."""
    db = manager.get(req.name, dim=req.dim)
    db.save()
    return {"name": req.name, "dim": db.dim(), "created": True}


@app.delete("/v1/namespaces/{namespace}", tags=["meta"],
            dependencies=[Depends(verify_api_key)])
def delete_namespace(namespace: str):
    """Hard-delete a namespace. Drops in-memory state + removes .feather + .wal from disk."""
    if namespace not in manager.list_namespaces():
        raise HTTPException(404, f"Namespace '{namespace}' not found")
    removed = manager.delete(namespace)
    return {"namespace": namespace, "deleted": bool(removed)}


def _is_dead_record(db, rec_id: int) -> bool:
    """True if a record is missing, forgotten, or soft-deleted."""
    meta = db.get_metadata(rec_id)
    if meta is None:
        return True
    if meta.source == "_forgotten":
        return True
    if meta.get_attribute("_deleted") == "true":
        return True
    return False


def _prune_edges_to(db, dead_id: int) -> int:
    """Cascade: remove every edge in every record that points to `dead_id`.
    Called whenever a record is deleted so graph state stays consistent.
    Returns the number of edges removed.
    """
    removed = 0
    for rec_id in db.get_all_ids(modality="text"):
        if rec_id == dead_id:
            continue
        meta = db.get_metadata(rec_id)
        if meta is None:
            continue
        current = list(meta.edges)
        kept    = [e for e in current if e.target_id != dead_id]
        if len(kept) != len(current):
            meta.edges = kept
            db.update_metadata(rec_id, meta)
            removed += len(current) - len(kept)
    return removed


def _prune_dead_edges(db) -> int:
    """Sweep every record's edges and drop ones pointing at deleted / forgotten /
    missing records. Returns the number of dead edges removed.
    """
    all_ids = set(db.get_all_ids(modality="text"))
    # cache liveness so we don't re-fetch metadata per edge
    live: dict[int, bool] = {}
    def is_live(rid: int) -> bool:
        if rid in live:
            return live[rid]
        live[rid] = (rid in all_ids) and not _is_dead_record(db, rid)
        return live[rid]
    removed = 0
    for rec_id in all_ids:
        if not is_live(rec_id):
            continue
        meta = db.get_metadata(rec_id)
        if meta is None:
            continue
        current = list(meta.edges)
        kept    = [e for e in current if is_live(e.target_id)]
        if len(kept) != len(current):
            meta.edges = kept
            db.update_metadata(rec_id, meta)
            removed += len(current) - len(kept)
    return removed


def _all_record_ids(db) -> list:
    """Every id that has metadata, regardless of which modality holds its
    vector. Falls back to the "text" index on older cores without all_ids()."""
    try:
        return db.all_ids()
    except AttributeError:
        return db.get_all_ids(modality="text")


def _modalities(db) -> list:
    """Actual modality names present (e.g. a pipeline may name them
    "embeddings"); falls back to ["text"] on older cores or empty DBs."""
    try:
        names = list(db.modality_names())
    except AttributeError:
        names = []
    return names or ["text"]


def _primary_modality(db) -> str:
    mods = _modalities(db)
    return "text" if "text" in mods else mods[0]


def _primary_dim(db) -> int:
    try:
        return db.dim(_primary_modality(db))
    except Exception:
        try:
            return db.dim()
        except Exception:
            return 0


def _live_record_count(db) -> int:
    """Walk metadata, skipping soft-deleted ones."""
    n = 0
    for record_id in _all_record_ids(db):
        meta = db.get_metadata(record_id)
        if meta is None:
            continue
        if meta.source == "_forgotten":
            continue
        if meta.get_attribute("_deleted") == "true":
            continue
        n += 1
    return n


@app.get("/v1/namespaces/{namespace}/stats", response_model=NamespaceStats,
         tags=["meta"], dependencies=[Depends(verify_api_key)])
def namespace_stats(namespace: str):
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")
    return NamespaceStats(
        namespace  = namespace,
        db_path    = f"{namespace}.feather",
        dim        = _primary_dim(db),
        modalities = _modalities(db),
        records    = _live_record_count(db),
    )


@app.get("/v1/admin/overview", response_model=AdminOverview,
         tags=["meta"], dependencies=[Depends(verify_api_key)])
def admin_overview():
    """Aggregate stats for the dashboard overview screen."""
    items = []
    total = 0
    for name in manager.list_namespaces():
        try:
            db = manager.get(name, create=False)
            n = _live_record_count(db)
            total += n
            items.append({"name": name, "dim": _primary_dim(db),
                          "modalities": _modalities(db), "records": n})
        except Exception:
            continue
    items.sort(key=lambda x: x["records"], reverse=True)

    return AdminOverview(
        version       = feather_db.__version__,
        uptime        = int(time.time() - _PROCESS_START),
        dim           = int(os.getenv("FEATHER_DB_DIM", "768")),
        nsCount       = len(items),
        totalRecords  = total,
        topNamespaces = items[:10],
    )


# Max size for an uploaded .feather. Streamed to disk in 1 MiB chunks so we never
# hold the whole file in RAM — but the file IS loaded into the in-RAM HNSW index
# on adopt, so the cap also bounds peak memory on the (small) prod VM. Tune with
# FEATHER_MAX_UPLOAD_MB.
MAX_UPLOAD_BYTES = int(os.getenv("FEATHER_MAX_UPLOAD_MB", "256")) * 1024 * 1024


@app.post("/v1/admin/upload", tags=["meta"], dependencies=[Depends(verify_api_key)])
async def upload_feather(
    file: UploadFile = File(...),
    namespace: str = Form(...),
    overwrite: bool = Form(False),
):
    """Adopt a locally-built `.feather` file as a cloud namespace.

    Stream the upload straight into the data dir, validate its magic+version,
    atomically move it into place, and serve it — so it shows up in the
    dashboard immediately. This is the "push my local DB to the cloud" path:
    the graph, attributes and persisted HNSW index come over intact (no
    re-embedding, no per-record API calls).
    """
    namespace = (namespace or "").strip()
    if not namespace:
        raise HTTPException(400, "namespace is required")

    # Stream to a temp file inside the data dir so the final move is an atomic
    # same-filesystem rename (and so a huge upload never lands in RAM).
    fd, staged = tempfile.mkstemp(suffix=".upload", dir=manager.data_dir())
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413,
                        f"file exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit",
                    )
                out.write(chunk)
    except HTTPException:
        try: os.remove(staged)
        except OSError: pass
        raise

    try:
        db = manager.adopt(namespace, staged, overwrite=overwrite)
    except FileExistsError:
        raise HTTPException(
            409,
            f"Namespace '{namespace}' already exists. "
            f"Re-send with overwrite=true to replace it.",
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    return {
        "namespace": namespace,
        "records": _live_record_count(db),
        "dim": db.dim(),
        "bytes": size,
        "imported": True,
    }


# ─────────────────────────────────────────────
# Routes — vector operations
# ─────────────────────────────────────────────
@app.post("/v1/{namespace}/vectors", status_code=201, tags=["vectors"],
          dependencies=[Depends(verify_api_key)])
def add_vector(namespace: str, req: AddVectorRequest):
    db = manager.get(namespace)
    meta = _meta_from_model(req.metadata) if req.metadata else Metadata()

    # Reject a mismatch only against an *established* dim. On an empty namespace
    # the first vector defines the dim (any dimension allowed), so we never
    # coerce it to the server default.
    ns_dim = _established_dim(db, req.modality)
    if ns_dim and len(req.vector) != ns_dim:
        raise HTTPException(
            400,
            f"vector dim {len(req.vector)} != index dim {ns_dim} "
            f"for modality '{req.modality}'",
        )

    # Always stamp namespace_id from the URL namespace if not set in body
    if not meta.namespace_id:
        meta.namespace_id = namespace

    with manager.lock(namespace):
        db.add(id=req.id, vec=req.vector, meta=meta, modality=req.modality)

    return {"id": req.id, "namespace": namespace, "modality": req.modality}


@app.post("/v1/{namespace}/search", response_model=SearchResponse, tags=["search"],
          dependencies=[Depends(verify_api_key)])
def search_vectors(namespace: str, req: SearchRequest):
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")

    sf = _build_filter(req)
    sc = _build_scoring(req)
    _check_query_dim(db, req.vector, req.modality)

    raw = db.search(req.vector, k=req.k, filter=sf, scoring=sc, modality=req.modality)

    items = [
        SearchResultItem(id=r.id, score=r.score, metadata=_meta_to_model(r.metadata))
        for r in raw
    ]
    return SearchResponse(results=items, count=len(items))


@app.get("/v1/{namespace}/records/{record_id}", response_model=MetadataOut,
         tags=["records"], dependencies=[Depends(verify_api_key)])
def get_record(namespace: str, record_id: int):
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")

    meta = db.get_metadata(record_id)
    if meta is None:
        raise HTTPException(404, f"Record {record_id} not found in namespace '{namespace}'")
    if meta.source == "_forgotten" or meta.get_attribute("_deleted") == "true":
        raise HTTPException(404, f"Record {record_id} not found in namespace '{namespace}'")
    return _meta_to_model(meta)


@app.put("/v1/{namespace}/records/{record_id}", tags=["records"],
         dependencies=[Depends(verify_api_key)])
def update_record_metadata(namespace: str, record_id: int, req: UpdateMetadataRequest):
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")

    meta = _meta_from_model(req.metadata)
    with manager.lock(namespace):
        db.update_metadata(record_id, meta)
    return {"id": record_id, "updated": True}


@app.put("/v1/{namespace}/records/{record_id}/importance", tags=["records"],
         dependencies=[Depends(verify_api_key)])
def update_importance(namespace: str, record_id: int, req: UpdateImportanceRequest):
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")

    with manager.lock(namespace):
        db.update_importance(record_id, req.importance)
    return {"id": record_id, "importance": req.importance}


@app.post("/v1/{namespace}/records/{record_id}/link", tags=["records"],
          dependencies=[Depends(verify_api_key)])
def link_records(namespace: str, record_id: int, req: LinkRequest):
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")

    with manager.lock(namespace):
        db.link(from_id=record_id, to_id=req.to_id)
    return {"from_id": record_id, "to_id": req.to_id, "linked": True}


@app.delete("/v1/{namespace}/records/{record_id}", tags=["records"],
            dependencies=[Depends(verify_api_key)])
def delete_record(namespace: str, record_id: int):
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")

    meta = db.get_metadata(record_id)
    if meta is None:
        raise HTTPException(404, f"Record {record_id} not found")
    if meta.source == "_forgotten" or meta.get_attribute("_deleted") == "true":
        raise HTTPException(404, f"Record {record_id} not found")

    with manager.lock(namespace):
        db.forget(record_id)
        # Cascade: drop any edges pointing at this id so the graph isn't left
        # with dangling pointers to a deleted record. Set ?cascade=false to opt
        # out (rare; mostly for bulk-delete sequences that compact afterwards).
        edges_pruned = _prune_edges_to(db, record_id)
        db.save()
    return {"id": record_id, "deleted": True, "edges_pruned": edges_pruned}


@app.delete("/v1/{namespace}/records/{from_id}/link/{to_id}", tags=["records"],
            dependencies=[Depends(verify_api_key)])
def unlink_records(namespace: str, from_id: int, to_id: int):
    """Remove a single edge from `from_id` to `to_id`. Returns the number of
    edges removed (0 if the edge didn't exist, 1+ if multiple rel_types matched).
    """
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")
    meta = db.get_metadata(from_id)
    if meta is None:
        raise HTTPException(404, f"Record {from_id} not found")
    current = list(meta.edges)
    kept    = [e for e in current if e.target_id != to_id]
    removed = len(current) - len(kept)
    if removed > 0:
        with manager.lock(namespace):
            meta.edges = kept
            db.update_metadata(from_id, meta)
            db.save()
    return {"from_id": from_id, "to_id": to_id, "removed": removed}


@app.post("/v1/{namespace}/purge", tags=["records"],
          dependencies=[Depends(verify_api_key)])
def purge_namespace(namespace: str, req: PurgeRequest):
    """Hard-delete all records whose metadata.namespace_id matches req.namespace_id.
    Removes from HNSW indices, metadata store, and reverse edge index. Returns count removed.
    """
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")

    with manager.lock(namespace):
        removed = db.purge(req.namespace_id)
        db.save()
    return {"namespace": namespace, "namespace_id": req.namespace_id, "removed": removed}


@app.post("/v1/{namespace}/compact", tags=["records"],
          dependencies=[Depends(verify_api_key)])
def compact_namespace(namespace: str, prune_dead_edges: bool = True):
    """Rebuild HNSW indices, physically dropping any soft-deleted records.
    By default also sweeps every record's outgoing edges and drops those that
    point at a deleted / forgotten / missing record (set `?prune_dead_edges=false`
    to skip). Returns counts for both reclamations.
    """
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")

    edges_pruned = 0
    with manager.lock(namespace):
        reclaimed = db.compact()
        if prune_dead_edges:
            edges_pruned = _prune_dead_edges(db)
        db.save()
    return {"namespace": namespace, "reclaimed": reclaimed, "edges_pruned": edges_pruned}


# ── Index maintenance / stats (Phase 7–8 capabilities) ──────────────
@app.get("/v1/{namespace}/admin/index_stats", response_model=IndexStatsResponse,
         tags=["admin"], dependencies=[Depends(verify_api_key)])
def index_stats(namespace: str):
    """Operational index health: record count, dim, on-disk quantization state,
    auto-compaction threshold, and the per-namespace secondary-index sizes."""
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")
    mns = [{"id": n, "size": db.namespace_size(n)} for n in db.list_namespaces()]
    mns.sort(key=lambda x: -x["size"])
    pm = _primary_modality(db)
    try:
        pquant = db.is_quantized(pm)
    except Exception:
        pquant = False
    return IndexStatsResponse(
        namespace=namespace,
        record_count=db.size(),
        dim=_primary_dim(db),
        text_quantized=pquant,
        auto_compact_ratio=db.get_auto_compact(),
        metadata_namespaces=mns[:50],
    )


@app.put("/v1/{namespace}/admin/auto_compact", tags=["admin"],
         dependencies=[Depends(verify_api_key)])
def set_auto_compact(namespace: str, req: AutoCompactRequest):
    """Enable/adjust incremental auto-compaction (rebuild a modality once its
    deleted/total ratio crosses `ratio`). 0 disables."""
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")
    with manager.lock(namespace):
        db.set_auto_compact(req.ratio)
    return {"namespace": namespace, "auto_compact_ratio": db.get_auto_compact()}


@app.put("/v1/{namespace}/admin/quantize", tags=["admin"],
         dependencies=[Depends(verify_api_key)])
def set_quantize(namespace: str, req: QuantizeRequest):
    """Toggle on-disk int8 quantization for a modality (~3x smaller .feather).
    Takes effect on save, which we do immediately so it's persisted now."""
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")
    with manager.lock(namespace):
        db.set_quantized(req.modality, req.on)
        db.save()
    return {"namespace": namespace, "modality": req.modality,
            "quantized": db.is_quantized(req.modality)}


@app.get("/v1/{namespace}/records", tags=["records"],
         dependencies=[Depends(verify_api_key)])
def list_records(namespace: str, limit: int = 50, after: int = -1, modality: str = "text"):
    """Cursor-based record listing. Returns up to `limit` records with id > `after`.
    Default after=-1 so a record with id 0 appears on the first page."""
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")

    # List by metadata id (not a single modality index) so records whose
    # vectors live under a non-"text" modality still show up.
    all_ids = sorted(_all_record_ids(db))
    page_ids = [i for i in all_ids if i > after][:limit]

    results = []
    for record_id in page_ids:
        meta = db.get_metadata(record_id)
        if meta is None:
            continue
        if meta.get_attribute("_deleted") == "true":
            continue
        results.append(SearchResultItem(
            id=record_id, score=float(meta.importance),
            metadata=_meta_to_model(meta)
        ))

    next_cursor = page_ids[-1] if page_ids else after
    return {
        "results": results,
        "count": len(results),
        "next_cursor": next_cursor,
        "has_more": len(page_ids) == limit,
    }


@app.post("/v1/{namespace}/keyword_search", response_model=SearchResponse, tags=["search"],
          dependencies=[Depends(verify_api_key)])
def keyword_search(namespace: str, req: KeywordSearchRequest):
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")

    sf = _build_filter(req)
    raw = db.keyword_search(req.query, k=req.k, filter=sf)
    items = [
        SearchResultItem(id=r.id, score=r.score, metadata=_meta_to_model(r.metadata))
        for r in raw
    ]
    return SearchResponse(results=items, count=len(items))


@app.post("/v1/{namespace}/hybrid_search", response_model=SearchResponse, tags=["search"],
          dependencies=[Depends(verify_api_key)])
def hybrid_search(namespace: str, req: HybridSearchRequest):
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")

    sf = _build_filter(req)
    sc = _build_scoring(req)
    _check_query_dim(db, req.vector, req.modality)
    raw = db.hybrid_search(req.vector, req.query, k=req.k,
                            rrf_k=req.rrf_k, filter=sf, scoring=sc,
                            modality=req.modality)
    items = [
        SearchResultItem(id=r.id, score=r.score, metadata=_meta_to_model(r.metadata))
        for r in raw
    ]
    return SearchResponse(results=items, count=len(items))


@app.post("/v1/{namespace}/save", tags=["admin"], dependencies=[Depends(verify_api_key)])
def save_namespace(namespace: str):
    try:
        manager.save(namespace)
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"namespace": namespace, "saved": True}


# ─────────────────────────────────────────────
# Bulk seeder — generates N records with random vectors
# ─────────────────────────────────────────────
@app.post("/v1/{namespace}/seed", tags=["admin"], dependencies=[Depends(verify_api_key)])
def seed_namespace(namespace: str, req: SeedRequest):
    db = manager.get(namespace)
    rng = np.random.default_rng(req.seed)
    dim = db.dim()
    ns_tag = req.namespace_id or namespace

    inserted = []
    with manager.lock(namespace):
        for i in range(req.count):
            # Cap random IDs at 2^53 - 1 so the dashboard (JavaScript) can address
            # them precisely. Larger IDs lose precision on round-trip and can't be
            # individually viewed/deleted.
            rec_id = (req.base_id + i + 1) if req.base_id else int(rng.integers(1, 2**53))
            meta = Metadata()
            meta.timestamp    = int(time.time())
            meta.importance   = 1.0
            meta.type         = ContextType.FACT
            meta.namespace_id = ns_tag
            meta.entity_id    = f"seed_{rec_id}"
            meta.content      = req.content_template.replace("{i}", str(i + 1))
            vec = rng.random(dim).astype(np.float32)
            db.add(id=rec_id, vec=vec, meta=meta, modality="text")
            inserted.append(rec_id)
        db.save()
    return {"namespace": namespace, "inserted": len(inserted),
            "first_id": inserted[0] if inserted else None,
            "last_id":  inserted[-1] if inserted else None}


# ─────────────────────────────────────────────
# Edges — read outgoing + incoming for a record
# ─────────────────────────────────────────────
@app.get("/v1/{namespace}/records/{record_id}/edges",
         response_model=EdgesResponse, tags=["graph"],
         dependencies=[Depends(verify_api_key)])
def get_record_edges(namespace: str, record_id: int):
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")
    out = [EdgeOut(target_id=e.target_id, rel_type=e.rel_type, weight=e.weight)
           for e in db.get_edges(record_id)]
    inc = [IncomingEdgeOut(source_id=e.source_id, rel_type=e.rel_type, weight=e.weight)
           for e in db.get_incoming(record_id)]
    return EdgesResponse(id=record_id, outgoing=out, incoming=inc)


# ─────────────────────────────────────────────
# Context chain — vector search + BFS expansion
# ─────────────────────────────────────────────
@app.post("/v1/{namespace}/context_chain", response_model=ContextChainResponse,
          tags=["graph"], dependencies=[Depends(verify_api_key)])
def context_chain(namespace: str, req: ContextChainRequest):
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")

    if req.vector is not None:
        _check_query_dim(db, req.vector, req.modality)
        vec = np.asarray(req.vector, dtype=np.float32)
    else:
        rng = np.random.default_rng(req.seed)
        vec = rng.random(db.dim()).astype(np.float32)

    result = db.context_chain(q=vec, k=req.k, hops=req.hops, modality=req.modality)
    nodes = [ContextChainNode(
                id=n.id, score=float(n.score), hop_distance=int(n.hop),
                metadata=_meta_to_model(n.metadata) if n.metadata else None)
             for n in result.nodes]
    edges = [ContextChainEdge(
                source_id=e.source, target_id=e.target,
                rel_type=e.rel_type, weight=float(e.weight))
             for e in result.edges]
    return ContextChainResponse(nodes=nodes, edges=edges)


# ─────────────────────────────────────────────
# Graph export — D3-friendly nodes + links JSON
# ─────────────────────────────────────────────
GRAPH_NODE_LIMIT = int(os.getenv("FEATHER_GRAPH_NODE_LIMIT", "4000"))


@app.get("/v1/{namespace}/graph", tags=["graph"],
         dependencies=[Depends(verify_api_key)])
def export_graph(namespace: str, ns_filter: str = "", entity_filter: str = ""):
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")

    # A browser force-graph can't render hundreds of thousands of nodes, and
    # serializing them all is pointless. If the namespace is large and the
    # caller didn't narrow it down, return a friendly guard instead of trying
    # (and either freezing the tab or crashing on dirty bytes — see below).
    if not ns_filter and not entity_filter:
        try:
            total = db.size()
        except Exception:
            total = 0
        if total > GRAPH_NODE_LIMIT:
            return {
                "nodes": [], "links": [], "truncated": True, "total": total,
                "note": (f"{total:,} records — too large to render as a graph. "
                         f"Filter by namespace_id or entity to view a subgraph "
                         f"(limit {GRAPH_NODE_LIMIT:,})."),
            }

    # export_graph_json builds one big JSON string in C++; pybind11 decodes it
    # as strict UTF-8. A single record with non-UTF-8 bytes (latin-1, truncated
    # multibyte) anywhere in the namespace makes that decode throw. Never let
    # that 500 the endpoint — degrade to an empty graph with a clear note.
    try:
        raw = db.export_graph_json(ns_filter, entity_filter)
    except (UnicodeDecodeError, ValueError) as e:
        return {
            "nodes": [], "links": [], "dirty": True,
            "note": ("graph contains non-UTF-8 text in at least one record and "
                     "can't be rendered; filter to a clean subset. "
                     f"({type(e).__name__})"),
        }
    try:
        return json.loads(raw)
    except Exception:
        raise HTTPException(500, "graph export returned invalid JSON")


# ─────────────────────────────────────────────
# Admin metrics + activity feed
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# Schema discovery — distinct attribute keys + value frequencies
# ─────────────────────────────────────────────
@app.get("/v1/{namespace}/schema", response_model=NamespaceSchema, tags=["meta"],
         dependencies=[Depends(verify_api_key)])
def namespace_schema(namespace: str, sample_limit: int = 8):
    """Walk live records, tally attributes + value samples + type inference."""
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")

    from collections import Counter, defaultdict
    attr_count: Counter = Counter()
    attr_values: dict = defaultdict(set)
    namespace_ids: Counter = Counter()
    entity_ids: list = []
    sources: Counter = Counter()
    n = 0
    for record_id in db.get_all_ids(modality="text"):
        meta = db.get_metadata(record_id)
        if meta is None:
            continue
        if meta.source == "_forgotten":
            continue
        if meta.get_attribute("_deleted") == "true":
            continue
        n += 1
        if meta.namespace_id:
            namespace_ids[meta.namespace_id] += 1
        if meta.entity_id and len(entity_ids) < 20:
            entity_ids.append(meta.entity_id)
        if meta.source:
            sources[meta.source] += 1
        for k, v in dict(meta.attributes).items():
            if k.startswith("_"):
                continue
            attr_count[k] += 1
            if len(attr_values[k]) < 64:
                attr_values[k].add(v)

    def infer(values: set) -> str:
        types = set()
        for v in list(values)[:32]:
            if isinstance(v, bool): types.add("bool"); continue
            try: int(v); types.add("int"); continue
            except (TypeError, ValueError): pass
            try: float(v); types.add("float"); continue
            except (TypeError, ValueError): pass
            if v in ("true", "false", "True", "False"):
                types.add("bool"); continue
            types.add("str")
        if len(types) == 1:
            return next(iter(types))
        return "mixed" if types else "str"

    attrs = []
    for k, c in attr_count.most_common():
        vals = list(attr_values[k])
        attrs.append(SchemaAttribute(
            key=k, count=c, distinct=len(vals),
            sample_values=sorted(map(str, vals))[:sample_limit],
            type=infer(vals),
        ))

    return NamespaceSchema(
        namespace=namespace,
        record_count=n,
        attributes=attrs,
        namespace_ids=[k for k, _ in namespace_ids.most_common(20)],
        entity_ids_sample=entity_ids,
        sources=[k for k, _ in sources.most_common(10)],
    )


@app.get("/v1/admin/metrics", tags=["meta"], dependencies=[Depends(verify_api_key)])
def admin_metrics(window: int = 3600):
    return METRICS.snapshot(since_seconds=float(window))


@app.get("/v1/admin/activity", tags=["meta"], dependencies=[Depends(verify_api_key)])
def admin_activity(limit: int = 50):
    return {"events": METRICS.activity(limit=limit)}


# ─────────────────────────────────────────────
# v0.10.0 — Hawky edition: monitoring, connection info, hierarchy, embedding, import
# ─────────────────────────────────────────────
@app.get("/v1/admin/ops_timeseries", response_model=OpsTimeseriesResponse,
         tags=["meta"], dependencies=[Depends(verify_api_key)])
def ops_timeseries(window: int = 3600, bucket_seconds: int = 60):
    pts = METRICS.bucketed(bucket_seconds=bucket_seconds, since_seconds=float(window))
    return OpsTimeseriesResponse(
        bucketSeconds=bucket_seconds,
        points=[OpsTimeseriesPoint(**p) for p in pts],
    )


@app.get("/v1/{namespace}/top_recalled", response_model=List[TopRecalledItem],
         tags=["meta"], dependencies=[Depends(verify_api_key)])
def top_recalled(namespace: str, limit: int = 10):
    """Records sorted by recall_count desc — what's actually being used."""
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")
    rows = []
    for record_id in db.get_all_ids(modality="text"):
        meta = db.get_metadata(record_id)
        if meta is None or meta.source == "_forgotten":
            continue
        if meta.get_attribute("_deleted") == "true":
            continue
        rows.append(TopRecalledItem(
            id=record_id,
            recall_count=int(meta.recall_count),
            last_recalled_at=int(meta.last_recalled_at),
            importance=float(meta.importance),
            content=(meta.content or "")[:200],
            namespace_id=meta.namespace_id or "",
        ))
    rows.sort(key=lambda r: (-r.recall_count, -r.last_recalled_at))
    return rows[:limit]


@app.get("/v1/admin/connection_info", response_model=ConnectionInfo,
         tags=["meta"], dependencies=[Depends(verify_api_key)])
def connection_info(request: Request):
    """Returns the canonical API base URL + ready-to-paste code samples."""
    base = str(request.base_url).rstrip("/")
    key_placeholder = "<YOUR_API_KEY>"
    sample_curl = (
        f'curl -X POST "{base}/v1/my_brand/search" \\\n'
        f'  -H "X-API-Key: {key_placeholder}" -H "Content-Type: application/json" \\\n'
        f'  -d \'{{"vector":[/* {EMBEDDING.snapshot()["dim"]} floats */],"k":10}}\''
    )
    sample_python = (
        "import requests\n"
        f'r = requests.post("{base}/v1/my_brand/search",\n'
        f'                  headers={{"X-API-Key": "{key_placeholder}"}},\n'
        '                  json={"vector": [...], "k": 10})\n'
        "print(r.json())"
    )
    sample_js = (
        f"await fetch('{base}/v1/my_brand/search', {{\n"
        "  method: 'POST',\n"
        f"  headers: {{ 'X-API-Key': '{key_placeholder}', 'Content-Type': 'application/json' }},\n"
        "  body: JSON.stringify({ vector: [/* floats */], k: 10 }),\n"
        "}}).then(r => r.json())"
    )
    return ConnectionInfo(
        base_url=base,
        api_key_header="X-API-Key",
        sample_curl=sample_curl,
        sample_python=sample_python,
        sample_javascript=sample_js,
        docs_url=f"{base}/docs",
    )


@app.get("/v1/admin/embedding_config", response_model=EmbeddingConfig,
         tags=["meta"], dependencies=[Depends(verify_api_key)])
def embedding_config_get():
    return EmbeddingConfig(**EMBEDDING.snapshot())


@app.put("/v1/admin/embedding_config", response_model=EmbeddingConfig,
         tags=["meta"], dependencies=[Depends(verify_api_key)])
def embedding_config_put(req: EmbeddingConfigUpdate):
    snap = EMBEDDING.update(
        provider=req.provider, model=req.model, base_url=req.base_url,
        deployment=req.deployment, api_version=req.api_version,
        api_key=req.api_key, dim=req.dim,
    )
    return EmbeddingConfig(**snap)


@app.get("/v1/admin/embedding_models", tags=["meta"],
         dependencies=[Depends(verify_api_key)])
def embedding_models():
    """Per-provider curated model lists for the dashboard dropdown."""
    return SUPPORTED_MODELS


@app.post("/v1/{namespace}/ingest_text", tags=["records"],
          dependencies=[Depends(verify_api_key)])
def ingest_text(namespace: str, req: IngestTextRequest):
    """Embed `text` via the configured provider, then ingest as a new record."""
    try:
        vec = EMBEDDING.embed(req.text)
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    db = manager.get(namespace)
    # The embedding model's output dim is fixed. If this namespace already has an
    # established dim and the model doesn't match it, reject honestly rather than
    # padding/truncating (which silently corrupts the vector). On an empty
    # namespace the embedding defines the dim.
    ns_dim = _established_dim(db, req.modality)
    if ns_dim and len(vec) != ns_dim:
        raise HTTPException(
            400,
            f"embedding dim {len(vec)} != namespace dim {ns_dim} for modality "
            f"'{req.modality}'; the configured model doesn't match this namespace",
        )
    rec_id = int(np.random.default_rng().integers(1, 2**53))   # JS-safe id
    meta = _meta_from_model(req.metadata) if req.metadata else Metadata()
    if not meta.namespace_id:
        meta.namespace_id = namespace
    meta.content = meta.content or req.text
    if not meta.timestamp:
        meta.timestamp = int(time.time())

    with manager.lock(namespace):
        db.add(id=rec_id, vec=np.asarray(vec, dtype=np.float32),
               meta=meta, modality=req.modality)
        db.save()
    return {"id": rec_id, "namespace": namespace, "embedded": True, "dim": len(vec)}


@app.post("/v1/{namespace}/import", response_model=ImportResponse, tags=["records"],
          dependencies=[Depends(verify_api_key)])
def bulk_import(namespace: str, req: ImportRequest):
    """Bulk insert N records. Each item has an id and EITHER a precomputed
    `vector` (must match the namespace dim) OR `metadata.content`, which is
    embedded server-side via the configured provider. Without this, records
    pasted as plain text (no vector) were silently skipped and the namespace
    stayed empty."""
    db = manager.get(namespace)
    # No padding: an established dim is authoritative; otherwise the first
    # supplied (or embedded) vector defines it — any dimension is allowed.
    locked_dim = _established_dim(db, req.modality)
    skipped = 0
    embedded = 0
    errors: List[str] = []

    def _err(i, msg):
        nonlocal skipped
        skipped += 1
        if len(errors) < 20:
            errors.append(f"item {i}: {msg}")

    # ── Pass 1: parse every item; collect the ones that need server-side
    # embedding so we can embed them all concurrently (not one HTTP call at a
    # time, which timed out / crawled on large text imports).
    parsed: List[dict] = []          # validated entries, in order
    embed_texts: List[str] = []      # texts to embed
    embed_back: List[int] = []       # embed_texts[k] → parsed index
    for i, item in enumerate(req.items):
        try:
            rec_id = int(item["id"])
            meta_data = item.get("metadata") or {}
            content = str(meta_data.get("content", ""))
            vec = item.get("vector")
            entry = {"i": i, "id": rec_id, "md": meta_data, "content": content,
                     "vec": None, "auto": False}
            if vec is None or (isinstance(vec, (list, tuple)) and len(vec) == 0):
                if not content.strip():
                    raise ValueError("no 'vector' and no 'metadata.content' to embed")
                entry["auto"] = True
                embed_back.append(len(parsed))
                embed_texts.append(content)
            else:
                vlen = len(vec)
                if locked_dim is None:
                    locked_dim = vlen            # first vector defines the dim
                elif vlen != locked_dim:
                    raise ValueError(f"dim mismatch: got {vlen}, expected {locked_dim}")
                entry["vec"] = list(vec)
            parsed.append(entry)
        except Exception as e:
            _err(i, e)

    # ── Pass 2: embed the collected texts concurrently.
    for k, (vec, err) in enumerate(_embed_many(embed_texts)):
        entry = parsed[embed_back[k]]
        if err is not None:
            entry["error"] = (f"no 'vector' supplied and auto-embed failed: {err} "
                              f"— configure an embedding provider in Settings, or include vectors")
            continue
        vlen = len(vec)
        if locked_dim is None:
            locked_dim = vlen                # all-text import: model defines dim
        if vlen != locked_dim:
            entry["error"] = (f"auto-embedded dim {vlen} != namespace dim {locked_dim}; "
                              f"the configured model doesn't match this namespace")
        else:
            entry["vec"] = list(vec)         # no padding

    # ── Pass 3: assemble the batch.
    ids: List[int] = []
    vecs: List[np.ndarray] = []
    metas: List[Metadata] = []
    for entry in parsed:
        if entry.get("error"):
            _err(entry["i"], entry["error"])
            continue
        if entry["vec"] is None:
            continue
        md = entry["md"]
        meta = Metadata()
        meta.timestamp    = int(md.get("timestamp") or time.time())
        meta.importance   = float(md.get("importance", 1.0))
        meta.type         = ContextType(int(md.get("type", 0)))
        meta.source       = str(md.get("source", ""))
        meta.content      = entry["content"]
        meta.tags_json    = str(md.get("tags_json", ""))
        meta.namespace_id = str(md.get("namespace_id", namespace))
        meta.entity_id    = str(md.get("entity_id", ""))
        for k, v in (md.get("attributes") or {}).items():
            meta.set_attribute(str(k), str(v))
        ids.append(entry["id"])
        vecs.append(np.asarray(entry["vec"], dtype=np.float32))
        metas.append(meta)
        if entry["auto"]:
            embedded += 1

    with manager.lock(namespace):
        if ids:
            db.add_batch(ids, np.asarray(vecs, dtype=np.float32), metas, modality=req.modality)
        db.save()
    return ImportResponse(namespace=namespace, inserted=len(ids),
                          skipped=skipped, embedded=embedded, errors=errors)


@app.get("/v1/{namespace}/hierarchy", response_model=HierarchyResponse,
         tags=["meta"], dependencies=[Depends(verify_api_key)])
def namespace_hierarchy(namespace: str):
    """Build a Brand → Channel → Campaign → AdSet → Ad → Creative tree
    from record metadata.attributes. Walks live records once, groups by levels.
    """
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")

    LEVELS = ["brand", "channel", "campaign", "adset", "ad", "creative"]
    counts: dict = {}    # tuple of values -> record_count
    for record_id in db.get_all_ids(modality="text"):
        meta = db.get_metadata(record_id)
        if meta is None or meta.source == "_forgotten":
            continue
        if meta.get_attribute("_deleted") == "true":
            continue
        path = []
        for lvl in LEVELS:
            v = meta.get_attribute(lvl, "")
            if not v:
                break
            path.append((lvl, v))
        if not path:
            # fallback to namespace_id as brand
            if meta.namespace_id:
                path = [("brand", meta.namespace_id)]
            else:
                continue
        for depth in range(1, len(path) + 1):
            key = tuple(path[:depth])
            counts[key] = counts.get(key, 0) + 1

    def build(prefix: tuple, depth: int) -> List[HierarchyNode]:
        children: dict = {}   # name -> count
        for key, c in counts.items():
            if len(key) == depth + 1 and key[:depth] == prefix:
                children[key[depth][1]] = c
        nodes = []
        for name, c in sorted(children.items(), key=lambda kv: -kv[1]):
            new_prefix = prefix + ((LEVELS[depth], name),)
            level_name = LEVELS[depth] if depth < len(LEVELS) else "leaf"
            nodes.append(HierarchyNode(
                level=level_name,
                name=name,
                record_count=c,
                children=build(new_prefix, depth + 1),
            ))
        return nodes

    return HierarchyResponse(
        namespace=namespace,
        levels=LEVELS,
        root=build((), 0),
    )
