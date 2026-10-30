"""Fast request-body decoding for the hot routes (msgspec).

Validating a search body through pydantic dominated the per-request cost once
the engine work became cheap: a 768-3072-float JSON array is parsed into Python
floats and then validated element by element. msgspec decodes straight into
typed Structs several times faster. The pydantic models in models.py stay the
documented request schemas (OpenAPI is generated from them); these Structs
mirror their fields, defaults and bounds, and the vector rules (exactly one of
`vector` / `vector_b64`, valid base64 float32) are re-checked here.

Validation failures are returned as HTTP 422 with the exact `detail` list
FastAPI + pydantic produced before this fast path existed: on the error path
only, the body is re-validated with the pydantic model (see _legacy_error).
"""
from __future__ import annotations

import base64
import binascii
import json
from typing import Annotated, Any, Dict, List, Optional

import msgspec
import numpy as np
from fastapi import HTTPException, Request
from starlette.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from . import models

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
    namespace_prefix: Optional[str] = None
    types: Optional[List[int]] = None
    entity_id: Optional[str] = None
    attributes_match: Optional[Dict[str, str]] = None
    source: Optional[str] = None
    source_prefix: Optional[str] = None
    importance_gte: Optional[float] = None
    confidence_gte: Optional[float] = None   # only records at least this certain
    tags_contains: Optional[List[str]] = None
    timestamp_after: Optional[int] = None
    timestamp_before: Optional[int] = None
    scoring_half_life: Optional[float] = None
    scoring_weight: Optional[float] = None
    scoring_min: Optional[float] = None


class HybridBody(_VectorMixin, msgspec.Struct, kw_only=True):
    query: str
    vector: Optional[List[float]] = None
    VECTOR_REQUIRED = False         # the route embeds `query` instead
    vector_b64: Optional[str] = None
    k: K = 10
    rrf_k: RRF_K = 60
    modality: str = "text"
    include_metadata: bool = True
    namespace_id: Optional[str] = None
    namespace_prefix: Optional[str] = None
    types: Optional[List[int]] = None
    entity_id: Optional[str] = None
    attributes_match: Optional[Dict[str, str]] = None
    source: Optional[str] = None
    source_prefix: Optional[str] = None
    importance_gte: Optional[float] = None
    confidence_gte: Optional[float] = None   # only records at least this certain
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
    namespace_prefix: Optional[str] = None
    types: Optional[List[int]] = None
    entity_id: Optional[str] = None
    attributes_match: Optional[Dict[str, str]] = None
    source: Optional[str] = None
    source_prefix: Optional[str] = None
    importance_gte: Optional[float] = None
    confidence_gte: Optional[float] = None   # only records at least this certain
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

# The pydantic model each Struct mirrors, used to reproduce its 422 detail.
_MODELS: Dict[type, type] = {
    SearchBody: models.SearchRequest, HybridBody: models.HybridSearchRequest,
    KeywordBody: models.KeywordSearchRequest, ContextChainBody: models.ContextChainRequest,
    AddVectorBody: models.AddVectorRequest, ImportBody: models.ImportRequest,
}


def _fail(msg: str, loc: Optional[str] = None):
    raise HTTPException(status_code=422, detail=[{
        "type": "value_error", "loc": ["body"] + ([loc] if loc else []), "msg": msg}])


def _legacy_error(struct: type, body: bytes) -> Optional[list]:
    """The 422 detail pydantic gave this body before the msgspec path, or None.

    Clients parse `detail` (loc/msg/type), so the error shape is part of the
    API. Only ever called after a failure, so its cost is off the hot path.
    """
    try:
        data = json.loads(body)
    except ValueError as e:
        if not isinstance(e, json.JSONDecodeError):
            return None
        return [{"type": "json_invalid", "loc": ["body", e.pos], "msg": "JSON decode error",
                 "input": {}, "ctx": {"error": e.msg}}]
    model = _MODELS.get(struct)
    if model is None:
        return None
    # `vector` was a required field before `vector_b64` existed; a body with
    # neither got pydantic's "missing", not the either/or rule's message.
    if (getattr(struct, "VECTOR_REQUIRED", False) and isinstance(data, dict)
            and "vector" not in data and "vector_b64" not in data):
        return [{"type": "missing", "loc": ["body", "vector"], "msg": "Field required", "input": data}]
    try:
        model.model_validate(data, from_attributes=True)   # as FastAPI validates bodies
    except ValidationError as e:
        return [{**err, "loc": ["body", *err["loc"]]}
                for err in jsonable_encoder(e.errors(include_url=False))]
    return None


# A body larger than this is decoded on a worker thread: a 50 MB /import
# decodes in ~100 ms, and on the event loop that is 100 ms in which no other
# request on the worker makes progress. Small bodies stay inline, where a
# thread hop would cost more than the decode.
_INLINE_DECODE_MAX = 256 * 1024


async def parse_body(request: Request, struct: type):
    """Decode + validate a JSON body into `struct`; 422 on any problem."""
    body = await request.body()
    if len(body) > _INLINE_DECODE_MAX:
        return await run_in_threadpool(_parse, struct, body)
    return _parse(struct, body)


def _parse(struct: type, body: bytes):
    dec = _DECODERS.get(struct)
    if dec is None:
        # strict=False matches pydantic's lax coercion, which these routes used
        # before: "5" -> 5, 5.0 -> 5, "true" -> True, ["0.1"] -> [0.1]. Strict
        # mode 422s those, breaking existing clients that send them.
        dec = _DECODERS[struct] = msgspec.json.Decoder(struct, strict=False)
    try:
        obj = dec.decode(body)
        if isinstance(obj, _VectorMixin):
            obj.check_vector()
        return obj
    except (msgspec.ValidationError, msgspec.DecodeError, HTTPException) as e:
        legacy = _legacy_error(struct, body)
        if legacy:
            raise HTTPException(status_code=422, detail=legacy) from None
        if isinstance(e, HTTPException):
            raise
        # pydantic accepted what msgspec refused (a NaN/Infinity literal, which
        # Python's json parser allows): take the body the way it always was
        # taken rather than 422 a client that used to work.
        try:
            obj = msgspec.convert(json.loads(body), struct, strict=False)
        except (msgspec.ValidationError, ValueError, TypeError):
            _fail(f"invalid JSON: {e}" if isinstance(e, msgspec.DecodeError) else str(e))
        if isinstance(obj, _VectorMixin):
            obj.check_vector()
        return obj
