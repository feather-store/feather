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
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import JSONResponse

import feather_db
from feather_db import Metadata, ContextType, ScoringConfig
from feather_db.core import SearchFilter

from .db_manager import DBManager
from .models import (
    AddVectorRequest, SearchRequest, SearchResponse, SearchResultItem,
    LinkRequest, UpdateImportanceRequest, UpdateMetadataRequest,
    MetadataOut, NamespaceStats, HealthResponse,
)

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
        modalities = ["text"],   # future: expose per-modality info
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


@app.post("/v1/{namespace}/save", tags=["admin"], dependencies=[Depends(verify_api_key)])
def save_namespace(namespace: str):
    try:
        manager.save(namespace)
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"namespace": namespace, "saved": True}
