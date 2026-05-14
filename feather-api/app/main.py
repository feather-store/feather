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
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends, Header, Request
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
    """Create an empty namespace (a new .feather file on disk)."""
    db = manager.get(req.name)
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


def _live_record_count(db) -> int:
    """Walk metadata, skipping soft-deleted ones."""
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
        dim        = db.dim(),
        modalities = ["text"],
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
            items.append({"name": name, "dim": db.dim(),
                          "modalities": ["text"], "records": n})
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


# ─────────────────────────────────────────────
# Routes — vector operations
# ─────────────────────────────────────────────
@app.post("/v1/{namespace}/vectors", status_code=201, tags=["vectors"],
          dependencies=[Depends(verify_api_key)])
def add_vector(namespace: str, req: AddVectorRequest):
    db = manager.get(namespace)
    meta = _meta_from_model(req.metadata) if req.metadata else Metadata()

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


@app.get("/v1/{namespace}/records", tags=["records"],
         dependencies=[Depends(verify_api_key)])
def list_records(namespace: str, limit: int = 50, after: int = 0, modality: str = "text"):
    """Cursor-based record listing. Returns up to `limit` records with id > `after`."""
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")

    all_ids = sorted(db.get_all_ids(modality=modality))
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
                source_id=e.source_id, target_id=e.target_id,
                rel_type=e.rel_type, weight=float(e.weight))
             for e in result.edges]
    return ContextChainResponse(nodes=nodes, edges=edges)


# ─────────────────────────────────────────────
# Graph export — D3-friendly nodes + links JSON
# ─────────────────────────────────────────────
@app.get("/v1/{namespace}/graph", tags=["graph"],
         dependencies=[Depends(verify_api_key)])
def export_graph(namespace: str, ns_filter: str = "", entity_filter: str = ""):
    try:
        db = manager.get(namespace, create=False)
    except KeyError:
        raise HTTPException(404, f"Namespace '{namespace}' not found")
    raw = db.export_graph_json(ns_filter, entity_filter)
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
    """Bulk insert N records {id, vector, metadata}. Vectors must match namespace dim."""
    db = manager.get(namespace)
    expected_dim = db.dim()
    inserted = 0
    skipped = 0
    errors: List[str] = []
    with manager.lock(namespace):
        for i, item in enumerate(req.items):
            try:
                rec_id = int(item["id"])
                vec = item["vector"]
                if len(vec) != expected_dim:
                    raise ValueError(f"dim mismatch: got {len(vec)}, expected {expected_dim}")
                meta_data = item.get("metadata") or {}
                meta = Metadata()
                meta.timestamp    = int(meta_data.get("timestamp") or time.time())
                meta.importance   = float(meta_data.get("importance", 1.0))
                meta.type         = ContextType(int(meta_data.get("type", 0)))
                meta.source       = str(meta_data.get("source", ""))
                meta.content      = str(meta_data.get("content", ""))
                meta.tags_json    = str(meta_data.get("tags_json", ""))
                meta.namespace_id = str(meta_data.get("namespace_id", namespace))
                meta.entity_id    = str(meta_data.get("entity_id", ""))
                for k, v in (meta_data.get("attributes") or {}).items():
                    meta.set_attribute(str(k), str(v))
                db.add(id=rec_id, vec=np.asarray(vec, dtype=np.float32),
                       meta=meta, modality=req.modality)
                inserted += 1
            except Exception as e:
                skipped += 1
                if len(errors) < 20:
                    errors.append(f"item {i}: {e}")
        db.save()
    return ImportResponse(namespace=namespace, inserted=inserted,
                          skipped=skipped, errors=errors)


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
