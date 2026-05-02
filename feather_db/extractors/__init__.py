"""Phase 9.1 — agentic ingestion primitives.

Pluggable LLM-backed (and rule-based) extractors that turn raw text into
structured facts, canonical entities, ISO timestamps, ontology edges, and
contradiction findings.

All extractors implement the protocols in `base.py`. They accept any
`feather_db.providers.LLMProvider` (Claude / OpenAI / Gemini / Ollama),
so users on any model stack work without code change.

Status (Phase 9.1 Week 3): full extractor suite shipping. Detect-only
contradiction handling — no auto-resolve, audit-trail-friendly.
"""
from .base import (
    Extractor,
    Resolver,
    Linker,
    Fact,
    Entity,
    ExtractedTimestamp,
    OntologyEdge,
    ContradictionFinding,
)
from .temporal import TemporalParser
from .facts import FactExtractor
from .entities import EntityResolver
from .ontology import OntologyLinker
from .contradictions import ContradictionResolver

__all__ = [
    "Extractor", "Resolver", "Linker",
    "Fact", "Entity", "ExtractedTimestamp", "OntologyEdge",
    "ContradictionFinding",
    "TemporalParser",
    "FactExtractor",
    "EntityResolver",
    "OntologyLinker",
    "ContradictionResolver",
]
