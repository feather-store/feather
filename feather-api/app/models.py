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


class LinkRequest(BaseModel):
    to_id: int


class UpdateImportanceRequest(BaseModel):
    importance: float


class UpdateMetadataRequest(BaseModel):
    metadata: MetadataIn


class NamespaceStats(BaseModel):
    namespace: str
    db_path: str
    dim: int
    modalities: List[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    namespaces_loaded: int
