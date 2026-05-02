"""Protocols + dataclasses for the Phase 9 extractor pipeline.

Every extractor / resolver / linker accepts an LLMProvider (where
applicable) and returns serializable dataclasses. The dataclasses are
the wire format between pipeline stages.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


# ── Wire-format dataclasses ────────────────────────────────────────────────

@dataclass
class Fact:
    """A single atomic fact extracted from raw text.

    `subject`, `predicate`, `object` form the triple; `confidence` is the
    extractor's self-rated likelihood the triple is supported by source
    text; `provenance` is the source identifier (memory_id of the raw
    turn, file path, URL, etc.); `extracted_at` is when extraction ran.
    Optional `valid_at` records a fact's *content* time (e.g. "in 2024"),
    distinct from when we extracted it.
    """
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    provenance: Optional[str] = None
    extracted_at: int = 0
    valid_at: Optional[int] = None
    raw_text: Optional[str] = None
    extractor_name: Optional[str] = None
    extractor_version: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    """A canonical entity reference. surface_form is what appeared in the
    source text; canonical_id is the resolved canonical identifier (e.g.
    'campaign_42'); kind is the entity type (Brand, Campaign, Person,
    etc.) per the active ontology.
    """
    surface_form: str
    canonical_id: str
    kind: str
    confidence: float = 1.0
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedTimestamp:
    """A natural-language temporal expression resolved to ISO time.

    `surface_form` is the original text ('last March', 'three weeks
    ago'); `iso` is the resolved ISO-8601 string; `kind` is one of:
    'absolute', 'relative', 'range_start', 'range_end', 'duration'.
    """
    surface_form: str
    iso: str
    kind: str
    confidence: float = 1.0
    range_end_iso: Optional[str] = None  # populated when kind=range_start
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OntologyEdge:
    """A typed weighted edge inferred between two facts/entities.

    rel_type is from the linker's allowed relations (e.g. 'caused_by',
    'supersedes', 'contradicts', 'part_of', 'refers_to').
    """
    source_id: str
    target_id: str
    rel_type: str
    weight: float = 1.0
    confidence: float = 1.0
    rationale: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContradictionFinding:
    """A surfaced conflict between a new fact and an existing one.

    Detect-only: this is what the resolver returns. Whether to act on it
    (auto-supersede, route to human review, mark both as low-confidence)
    is a downstream policy decision — Cloud applies tenant rules.

    severity:
        - 'definite'  — same subject+predicate, different non-numeric
                        object, overlapping or absent valid_at.
        - 'probable'  — same subject+predicate, different numeric object
                        outside tolerance.
        - 'possible'  — same subject+predicate, different object, but
                        the new fact's valid_at is later (likely
                        supersedure rather than contradiction).

    suggested_resolution:
        - 'supersedes' — new fact replaces old (clear temporal ordering).
        - 'review'     — surface to a human (default for definite/probable).
        - 'merge'      — same numeric value within tolerance.
    """
    new_fact: Fact
    conflicting_with: Fact
    severity: str = "definite"
    rationale: Optional[str] = None
    suggested_resolution: str = "review"
    detected_at: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Protocols ──────────────────────────────────────────────────────────────

class Extractor(ABC):
    """Text -> list of structured outputs (Facts, Entities, etc.)."""

    name: str = "base"
    version: str = "0.1.0"

    @abstractmethod
    def extract(self, text: str, *, context: Optional[dict] = None) -> list[Any]:
        """Extract structured items from `text`.

        Args:
            text:    raw text to process.
            context: optional metadata (source_id, namespace, prior turns,
                     vertical-specific hints from a Cloud agent).
        Returns:
            A list of dataclass instances (subclass-specific type).
        """


class Resolver(ABC):
    """Surface forms -> canonical references."""

    name: str = "base"

    @abstractmethod
    def resolve(self, surface_forms: list[str], *,
                context: Optional[dict] = None) -> list[Entity]:
        """Resolve each surface form to a canonical Entity.

        If a surface cannot be resolved with sufficient confidence, the
        resolver MAY return a low-confidence Entity with a synthesized
        canonical_id (e.g. ``unknown::<sha256(surface)[:12]>``) so the
        pipeline does not lose the mention.
        """


class Linker(ABC):
    """Pairs of facts/entities -> typed edges."""

    name: str = "base"
    allowed_relations: list[str] = []

    @abstractmethod
    def link(self, items: list[Any], *,
             context: Optional[dict] = None) -> list[OntologyEdge]:
        """Infer ontology edges between items.

        Implementations should self-restrict to `allowed_relations`. The
        items list is heterogeneous (Facts and/or Entities); subclasses
        can declare which combinations they support.
        """
