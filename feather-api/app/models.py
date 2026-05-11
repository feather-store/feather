"""
Pydantic request / response models for Feather DB Cloud API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import IntEnum


class ContextTypeEnum(IntEnum):
    FACT = 0
    PREFERENCE = 1
    EVENT = 2
    CONVERSATION = 3


class MetadataIn(BaseModel):
    timestamp: Optional[int] = None          # defaults to now
    importance: float = 1.0
    type: ContextTypeEnum = ContextTypeEnum.FACT
    source: str = ""
    content: str = ""
    tags_json: str = ""
    namespace_id: str = ""
    entity_id: str = ""
    attributes: Dict[str, str] = Field(default_factory=dict)


class MetadataOut(MetadataIn):
    recall_count: int = 0
    last_recalled_at: int = 0
    links: List[int] = Field(default_factory=list)


class AddVectorRequest(BaseModel):
    id: int
    vector: List[float]
    metadata: Optional[MetadataIn] = None
    modality: str = "text"


class SearchRequest(BaseModel):
    vector: List[float]
    k: int = 10
    modality: str = "text"
    # Filters
    namespace_id: Optional[str] = None
    entity_id: Optional[str] = None
    attributes_match: Optional[Dict[str, str]] = None
    source: Optional[str] = None
    source_prefix: Optional[str] = None
    importance_gte: Optional[float] = None
    tags_contains: Optional[List[str]] = None
    timestamp_after: Optional[int] = None
    timestamp_before: Optional[int] = None
    # Scoring
    scoring_half_life: Optional[float] = None    # days
    scoring_weight: Optional[float] = None       # 0.0–1.0
    scoring_min: Optional[float] = None


class SearchResultItem(BaseModel):
    id: int
    score: float
    metadata: MetadataOut


class SearchResponse(BaseModel):
    results: List[SearchResultItem]
    count: int


class KeywordSearchRequest(BaseModel):
    query: str
    k: int = 10
    # Filters (same as SearchRequest)
    namespace_id: Optional[str] = None
    entity_id: Optional[str] = None
    attributes_match: Optional[Dict[str, str]] = None
    source: Optional[str] = None
    source_prefix: Optional[str] = None
    importance_gte: Optional[float] = None
    tags_contains: Optional[List[str]] = None
    timestamp_after: Optional[int] = None
    timestamp_before: Optional[int] = None


class HybridSearchRequest(BaseModel):
    vector: List[float]
    query: str
    k: int = 10
    rrf_k: int = 60
    modality: str = "text"
    # Filters
    namespace_id: Optional[str] = None
    entity_id: Optional[str] = None
    attributes_match: Optional[Dict[str, str]] = None
    source: Optional[str] = None
    source_prefix: Optional[str] = None
    importance_gte: Optional[float] = None
    tags_contains: Optional[List[str]] = None
    timestamp_after: Optional[int] = None
    timestamp_before: Optional[int] = None
    # Scoring
    scoring_half_life: Optional[float] = None
    scoring_weight: Optional[float] = None
    scoring_min: Optional[float] = None


class LinkRequest(BaseModel):
    to_id: int


class UpdateImportanceRequest(BaseModel):
    importance: float


class UpdateMetadataRequest(BaseModel):
    metadata: MetadataIn


class PurgeRequest(BaseModel):
    namespace_id: str = Field(..., description="metadata.namespace_id to hard-delete (e.g. brand_id)")


class CreateNamespaceRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_\-]+$")


class SchemaAttribute(BaseModel):
    key: str
    count: int                 # how many records have this attribute
    distinct: int              # number of distinct values seen (capped)
    sample_values: List[str]   # up to 8 example values
    type: str                  # "int" | "float" | "bool" | "str" | "mixed"


class NamespaceSchema(BaseModel):
    namespace: str
    record_count: int
    attributes: List[SchemaAttribute]
    namespace_ids: List[str]   # distinct namespace_id values seen
    entity_ids_sample: List[str]
    sources: List[str]


# v0.10.0 — Hawky edition models
class TopRecalledItem(BaseModel):
    id: int
    recall_count: int
    last_recalled_at: int
    importance: float
    content: str
    namespace_id: str = ""


class OpsTimeseriesPoint(BaseModel):
    ts: int
    ops: int
    errors: int = 0


class OpsTimeseriesResponse(BaseModel):
    bucketSeconds: int
    points: List[OpsTimeseriesPoint]


class ConnectionInfo(BaseModel):
    base_url: str
    api_key_header: str = "X-API-Key"
    sample_curl: str
    sample_python: str
    sample_javascript: str
    docs_url: str


class EmbeddingConfig(BaseModel):
    provider: str = "none"             # "openai" | "voyage" | "cohere" | "ollama" | "none"
    model: str = ""
    base_url: str = ""                 # for self-hosted (e.g. ollama)
    api_key_set: bool = False          # never echo the key — just whether configured
    dim: int = 768


class EmbeddingConfigUpdate(BaseModel):
    provider: str
    model: str = ""
    base_url: str = ""
    api_key: Optional[str] = None      # set to update; omit to leave unchanged
    dim: int = 768


class ImportRequest(BaseModel):
    """Inline-bulk import.  Each item must have id + vector (dim must match)."""
    items: List[Dict[str, Any]]        # [{ id, vector, metadata: {...} }, ...]
    modality: str = "text"


class ImportResponse(BaseModel):
    namespace: str
    inserted: int
    skipped: int
    errors: List[str] = []


class HierarchyNode(BaseModel):
    level: str            # 'brand' | 'channel' | 'campaign' | 'adset' | 'ad' | 'creative'
    name: str
    record_count: int
    children: List["HierarchyNode"] = []


HierarchyNode.model_rebuild()


class HierarchyResponse(BaseModel):
    namespace: str
    levels: List[str]
    root: List[HierarchyNode]


class IngestTextRequest(BaseModel):
    """Add a record by text — server embeds via configured provider, then stores."""
    text: str
    metadata: Optional[MetadataIn] = None
    modality: str = "text"


class NamespaceStats(BaseModel):
    namespace: str
    db_path: str
    dim: int
    modalities: List[str]
    records: int = 0


class AdminOverview(BaseModel):
    version: str
    uptime: int                       # seconds since process start
    dim: int                          # default dim
    nsCount: int
    totalRecords: int
    topNamespaces: List[Dict[str, Any]]


class SeedRequest(BaseModel):
    count: int = Field(..., ge=1, le=10000, description="Number of records to seed")
    namespace_id: str = Field("", description="metadata.namespace_id stamped on each record")
    content_template: str = Field(
        "auto-seed record {i}",
        description="String template; {i} is replaced with the record index",
    )
    base_id: int = Field(0, description="Starting id; new ids = base_id + 1..count, or random if 0")
    seed: int = Field(42, description="RNG seed for reproducibility")


class ContextChainRequest(BaseModel):
    vector: Optional[List[float]] = None     # if omitted, server generates random with `seed`
    seed: int = 42
    k: int = 5
    hops: int = 2
    modality: str = "text"


class EdgeOut(BaseModel):
    target_id: int
    rel_type: str
    weight: float


class IncomingEdgeOut(BaseModel):
    source_id: int
    rel_type: str
    weight: float


class EdgesResponse(BaseModel):
    id: int
    outgoing: List[EdgeOut]
    incoming: List[IncomingEdgeOut]


class ContextChainNode(BaseModel):
    id: int
    score: float
    hop_distance: int
    metadata: Optional[MetadataOut] = None


class ContextChainEdge(BaseModel):
    source_id: int
    target_id: int
    rel_type: str
    weight: float


class ContextChainResponse(BaseModel):
    nodes: List[ContextChainNode]
    edges: List[ContextChainEdge]


class HealthResponse(BaseModel):
    status: str
    version: str
    namespaces_loaded: int
