"""LLM-backed EntityResolver — surface forms -> canonical Entity references."""
from __future__ import annotations
import hashlib
import re
import time
from typing import Optional

from .base import Resolver, Entity
from ._jsonparse import extract_json

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
    version = "0.1.0"

    def __init__(self, provider, *,
                 known_entities: Optional[list[Entity]] = None,
                 max_tokens: int = 1500,
                 max_retries: int = 2):
        self._provider = provider
        self._known = known_entities or []
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    def resolve(self, surface_forms: list[str], *,
                context: Optional[dict] = None) -> list[Entity]:
        """Resolve each surface form to a canonical Entity.

        Always returns one Entity per input surface form (even if
        unresolved — those get a synthesized canonical_id with low
        confidence so the pipeline doesn't lose the mention).
        """
        if not surface_forms:
            return []
        ctx = context or {}
        on_error = ctx.get("on_error")

        messages = [
            {"role": "system", "content": ENTITY_RESOLUTION_SYSTEM_PROMPT},
            {"role": "user", "content": self._user_message(surface_forms, ctx)},
        ]

        raw = None
        last_err: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                raw = self._provider.complete(
                    messages,
                    max_tokens=self._max_tokens,
                    temperature=0.0,
                )
                break
            except Exception as e:
                last_err = e
                if attempt < self._max_retries:
                    time.sleep(1.5 ** (attempt + 1))
        if raw is None:
            if on_error:
                on_error(f"EntityResolver LLM call failed: {last_err}")
            return [self._fallback(s) for s in surface_forms]

        parsed, ok, _ = extract_json(raw)
        if not ok or not isinstance(parsed, list):
            if on_error:
                on_error(f"EntityResolver JSON parse failed; raw[:200]={raw[:200]!r}")
            return [self._fallback(s) for s in surface_forms]

        # Index resolver output by surface_form (case-insensitive) so we
        # can guarantee one Entity per input even if the LLM dropped some.
        by_surface: dict[str, dict] = {}
        for item in parsed:
            if isinstance(item, dict) and item.get("surface_form"):
                by_surface[item["surface_form"].strip().lower()] = item

        out: list[Entity] = []
        for s in surface_forms:
            item = by_surface.get(s.strip().lower())
            if not item:
                out.append(self._fallback(s))
                continue
            cid = (item.get("canonical_id") or "").strip()
            kind = (item.get("kind") or "Unknown").strip()
            if not cid:
                out.append(self._fallback(s))
                continue
            try:
                conf = float(item.get("confidence", 0.7))
            except (TypeError, ValueError):
                conf = 0.5
            conf = max(0.0, min(1.0, conf))
            aliases = [a for a in (item.get("aliases") or [])
                       if isinstance(a, str)]
            out.append(Entity(
                surface_form=s,
                canonical_id=cid,
                kind=kind,
                confidence=conf,
                aliases=aliases,
            ))
        return out

    def _user_message(self, surface_forms: list[str], ctx: dict) -> str:
        lines = []
        if self._known:
            lines.append("KNOWN ENTITIES (preserve these canonical_ids):")
            for e in self._known[:50]:
                aliases = (" / aliases: " + ", ".join(e.aliases[:5])
                           if e.aliases else "")
                lines.append(f"  - {e.canonical_id} ({e.kind}){aliases}")
            lines.append("")
        if ctx.get("namespace"):
            lines.append(f"Namespace: {ctx['namespace']}")
            lines.append("")
        lines.append("SURFACE FORMS TO RESOLVE:")
        for s in surface_forms:
            lines.append(f"  - {s}")
        return "\n".join(lines)

    @staticmethod
    def _fallback(surface: str) -> Entity:
        slug = re.sub(r"[^a-z0-9]+", "_",
                      surface.lower().strip()).strip("_") or "entity"
        # short, deterministic, unique per surface
        h = hashlib.sha1(surface.encode("utf-8")).hexdigest()[:8]
        return Entity(
            surface_form=surface,
            canonical_id=f"unknown::{slug[:24]}_{h}",
            kind="Unknown",
            confidence=0.2,
        )
