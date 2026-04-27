"""LLM-backed EntityResolver — surface forms -> canonical Entity references.

Status: stub (Phase 9.1 Week 2).
"""
from __future__ import annotations
from typing import Optional

from .base import Resolver, Entity

ENTITY_RESOLUTION_SYSTEM_PROMPT = """\
You canonicalize surface forms to entities the agent already knows.

Given a list of surface forms and (optionally) a list of known entities,
return the canonical_id and entity kind for each surface form.

Rules:
- If a surface form clearly refers to a known entity, return its
  canonical_id and kind verbatim.
- If a surface form refers to a new entity, propose a canonical_id of
  the form `<kind>::<slug>` (e.g. "campaign::summer_sale_2024") and
  rate confidence ≤ 0.7 to flag it for downstream merge.
- If a surface form is too ambiguous to resolve, return canonical_id =
  "unknown::<surface_slug>" with confidence ≤ 0.3.
- Aliases: include any other phrasings of the same entity present in
  the surface_forms list.

Output: JSON array of {surface_form, canonical_id, kind, confidence,
aliases: []}.
"""


class EntityResolver(Resolver):
    """LLM-backed entity canonicalization.

    Used by Phase 9 IngestPipeline after FactExtractor to deduplicate
    surface mentions ("Acme", "Acme Corp", "ACME Inc.") to a single
    canonical_id.

    Args:
        provider:       LLMProvider instance.
        known_entities: optional pre-existing list of Entities the
                        resolver should preserve canonical_ids for.
                        Cloud verticals seed this from the brand-context
                        bootstrap.

    Example:
        >>> resolver = EntityResolver(provider=ClaudeProvider())
        >>> entities = resolver.resolve(
        ...     ["Acme Corp", "ACME", "their summer campaign"],
        ...     context={"namespace": "acme_corp"},
        ... )
        >>> # entities -> [
        >>> #   Entity(surface_form='Acme Corp',
        >>> #          canonical_id='brand::acme', kind='Brand',
        >>> #          aliases=['ACME']),
        >>> #   Entity(surface_form='their summer campaign',
        >>> #          canonical_id='campaign::summer_2024',
        >>> #          kind='Campaign'),
        >>> # ]
    """

    name = "entity_resolver"
    version = "0.1.0-stub"

    def __init__(self, provider, *,
                 known_entities: Optional[list[Entity]] = None):
        self._provider = provider
        self._known = known_entities or []

    def resolve(self, surface_forms: list[str], *,
                context: Optional[dict] = None) -> list[Entity]:
        """Resolve each surface form to a canonical Entity.

        IMPLEMENTATION STATUS: stub. See Phase 9.1 Week 2 in
        docs/architecture/phase9-plan.md.
        """
        raise NotImplementedError(
            "EntityResolver.resolve — Phase 9.1 Week 2."
        )
