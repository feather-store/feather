"""LLM-backed OntologyLinker — fact pairs -> typed weighted edges.

Status: stub (Phase 9.1 Week 3).
"""
from __future__ import annotations
from typing import Optional, Any

from .base import Linker, OntologyEdge

DEFAULT_RELATIONS = [
    "caused_by",
    "supports",
    "contradicts",
    "supersedes",
    "part_of",          # hierarchical (e.g. ad → ad_set → campaign)
    "refers_to",
    "extracted_from",   # provenance back-edge to source
    "uses_creative",    # ad → creative
    "targets_segment",  # campaign → audience segment
    "same_segment_as",  # cross-source: own data ↔ competitor data
    "similar_creative_to",  # cross-source: visual similarity above threshold
]

ONTOLOGY_LINK_SYSTEM_PROMPT = """\
You infer typed relationships between pairs of facts/entities.

Given a list of items (facts and/or entities) and an allowed-relations
list, return zero or more typed edges (source_id → target_id) drawn
ONLY from the allowed list.

Rules:
- Do not invent new relation types.
- Edge weight (0.0–1.0) reflects strength of the relationship as
  supported by the items' raw text.
- If you propose `contradicts` or `supersedes`, include a brief
  rationale explaining the conflict.
- Skip any pair where evidence is too weak (no edge) — better to miss
  an edge than fabricate one.

Output: JSON array of {source_id, target_id, rel_type, weight,
confidence, rationale}.
"""


class OntologyLinker(Linker):
    """LLM-backed typed-edge inference.

    Args:
        provider:           LLMProvider instance.
        allowed_relations:  whitelist of edge types this linker may
                            emit. Cloud verticals override with
                            domain-specific lists (Marketing adds
                            "uses_creative", "targets_segment", etc.).
        max_pairs_per_call: cap pairs evaluated per LLM call (default 50).

    Example:
        >>> linker = OntologyLinker(
        ...     provider=ClaudeProvider(),
        ...     allowed_relations=["caused_by", "supports", "contradicts"],
        ... )
        >>> edges = linker.link([fact1, fact2, fact3])
        >>> # edges -> [
        >>> #   OntologyEdge(source_id='fact_1', target_id='fact_2',
        >>> #                rel_type='supports', weight=0.85, ...),
        >>> # ]
    """

    name = "ontology_linker"
    version = "0.1.0-stub"

    def __init__(self, provider, *,
                 allowed_relations: Optional[list[str]] = None,
                 max_pairs_per_call: int = 50):
        self._provider = provider
        self.allowed_relations = (allowed_relations
                                  if allowed_relations is not None
                                  else list(DEFAULT_RELATIONS))
        self._max_pairs = max_pairs_per_call

    def link(self, items: list[Any], *,
             context: Optional[dict] = None) -> list[OntologyEdge]:
        """Infer typed edges between items.

        IMPLEMENTATION STATUS: stub. See Phase 9.1 Week 3 in
        docs/architecture/phase9-plan.md.
        """
        raise NotImplementedError(
            "OntologyLinker.link — Phase 9.1 Week 3."
        )
