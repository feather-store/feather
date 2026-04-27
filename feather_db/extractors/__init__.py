"""Phase 9.1 — agentic ingestion primitives.

Pluggable LLM-backed (and rule-based) extractors that turn raw text into
structured facts, canonical entities, ISO timestamps, and ontology edges.

All extractors implement the protocols in `base.py`. They accept any
`feather_db.providers.LLMProvider` (Claude / OpenAI / Gemini / Ollama),
so users on any model stack work without code change.

Status (2026-04-27): scaffolds + TemporalParser shipping; LLM-backed
extractors land week-by-week per docs/architecture/phase9-plan.md.
"""
from .base import (
    Extractor,
    Resolver,
    Linker,
    Fact,
    Entity,
    ExtractedTimestamp,
    OntologyEdge,
)
from .temporal import TemporalParser
from .facts import FactExtractor          # stub
from .entities import EntityResolver      # stub
from .ontology import OntologyLinker      # stub

__all__ = [
    "Extractor", "Resolver", "Linker",
    "Fact", "Entity", "ExtractedTimestamp", "OntologyEdge",
    "TemporalParser",
    "FactExtractor",
    "EntityResolver",
    "OntologyLinker",
]
