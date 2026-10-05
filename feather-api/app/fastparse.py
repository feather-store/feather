"""Fast request-body decoding for the hot routes (msgspec).

Validating a search body through pydantic dominated the per-request cost once
the engine work became cheap: a 768-3072-float JSON array is parsed into Python
floats and then validated element by element. msgspec decodes straight into
typed Structs several times faster. The pydantic models in models.py stay the
documented request schemas (OpenAPI is generated from them); these Structs
mirror their fields, defaults and bounds, and the vector rules (exactly one of
`vector` / `vector_b64`, valid base64 float32) are re-checked here.

Validation failures are returned as HTTP 422 with a FastAPI-style `detail` list.
"""
from __future__ import annotations

import base64
import binascii
from typing import Annotated, Any, Dict, List, Optional

import msgspec
import numpy as np
from fastapi import HTTPException, Request

K = Annotated[int, msgspec.Meta(ge=1, le=1000)]
RRF_K = Annotated[int, msgspec.Meta(ge=1, le=10000)]
HOPS = Annotated[int, msgspec.Meta(ge=0, le=10)]


class _VectorMixin:
    """Methods shared by bodies that carry a vector (Struct fields supply the data)."""
    __slots__ = ()
    VECTOR_REQUIRED = True

    def has_vector(self) -> bool:
        return self.vector is not None or self.vector_b64 is not None

    def vector_array(self) -> np.ndarray:
        if self.vector_b64 is not None:
            return np.frombuffer(base64.b64decode(self.vector_b64), dtype="<f4")
        return np.asarray(self.vector, dtype=np.float32)

    def check_vector(self) -> None:
        both = self.vector is not None and self.vector_b64 is not None
        if both or (not self.has_vector() and self.VECTOR_REQUIRED):
            _fail("provide exactly one of `vector` or `vector_b64`")
        if self.vector_b64 is not None:
            try:
                raw = base64.b64decode(self.vector_b64, validate=True)
            except (binascii.Error, ValueError):
                _fail("`vector_b64` is not valid base64")
            if len(raw) == 0 or len(raw) % 4:
                _fail("`vector_b64` must decode to a whole number of float32 values")


class SearchBody(_VectorMixin, msgspec.Struct, kw_only=True):
    vector: Optional[List[float]] = None
    vector_b64: Optional[str] = None
    k: K = 10
    modality: str = "text"
    track: bool = True
    raw_score: bool = False
    include_metadata: bool = True
    namespace_id: Optional[str] = None
    entity_id: Optional[str] = None
    attributes_match: Optional[Dict[str, str]] = None
    source: Optional[str] = None
    source_prefix: Optional[str] = None
    importance_gte: Optional[float] = None
    tags_contains: Optional[List[str]] = None
    timestamp_after: Optional[int] = None
    timestamp_before: Optional[int] = None
    scoring_half_life: Optional[float] = None
    scoring_weight: Optional[float] = None
    scoring_min: Optional[float] = None


class HybridBody(_VectorMixin, msgspec.Struct, kw_only=True):
    query: str
    vector: Optional[List[float]] = None
    vector_b64: Optional[str] = None
    k: K = 10
    rrf_k: RRF_K = 60
    modality: str = "text"
    include_metadata: bool = True
    namespace_id: Optional[str] = None
    entity_id: Optional[str] = None
    attributes_match: Optional[Dict[str, str]] = None
    source: Optional[str] = None
    source_prefix: Optional[str] = None
    importance_gte: Optional[float] = None
    tags_contains: Optional[List[str]] = None
    timestamp_after: Optional[int] = None
    timestamp_before: Optional[int] = None
    scoring_half_life: Optional[float] = None
    scoring_weight: Optional[float] = None
    scoring_min: Optional[float] = None


class KeywordBody(msgspec.Struct, kw_only=True):
    query: str
    k: K = 10
    include_metadata: bool = True
    namespace_id: Optional[str] = None
    entity_id: Optional[str] = None
    attributes_match: Optional[Dict[str, str]] = None
    source: Optional[str] = None
    source_prefix: Optional[str] = None
    importance_gte: Optional[float] = None
    tags_contains: Optional[List[str]] = None
    timestamp_after: Optional[int] = None
    timestamp_before: Optional[int] = None


class ContextChainBody(_VectorMixin, msgspec.Struct, kw_only=True):
    VECTOR_REQUIRED = False            # omitted -> server generates one from `seed`
    vector: Optional[List[float]] = None
    vector_b64: Optional[str] = None
    seed: int = 42
    k: K = 5
    hops: HOPS = 2
    modality: str = "text"


class AddVectorBody(_VectorMixin, msgspec.Struct, kw_only=True):
    id: int
    vector: Optional[List[float]] = None
    vector_b64: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None   # validated by pydantic MetadataIn (strict keys)
    modality: str = "text"


class ImportBody(msgspec.Struct, kw_only=True):
    items: List[Dict[str, Any]]
    modality: str = "text"
    flush: bool = False


_DECODERS: Dict[type, msgspec.json.Decoder] = {}


def _fail(msg: str, loc: Optional[str] = None):
    raise HTTPException(status_code=422, detail=[{
        "type": "value_error", "loc": ["body"] + ([loc] if loc else []), "msg": msg}])


async def parse_body(request: Request, struct: type):
    """Decode + validate a JSON body into `struct`; 422 on any problem."""
    dec = _DECODERS.get(struct)
    if dec is None:
        dec = _DECODERS[struct] = msgspec.json.Decoder(struct)
    body = await request.body()
    try:
        obj = dec.decode(body)
    except msgspec.ValidationError as e:
        _fail(str(e))
    except msgspec.DecodeError as e:
        _fail(f"invalid JSON: {e}")
    if isinstance(obj, _VectorMixin):
        obj.check_vector()
    return obj
