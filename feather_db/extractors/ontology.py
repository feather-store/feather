"""LLM-backed OntologyLinker — fact/entity items -> typed weighted edges.

Phase 9.1 Week 3. Detect-only: emits OntologyEdge instances without
auto-resolving conflicts. Marketing teams need provenance + audit trail,
so contradictions surface as `contradicts` edges with rationale rather
than silently merging facts.
"""
from __future__ import annotations
import time
from typing import Any, Optional

from .base import Linker, OntologyEdge, Fact, Entity
from ._jsonparse import extract_json

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

# Relations whose existence is a high-stakes claim — require LLM rationale.
REQUIRES_RATIONALE = frozenset({"contradicts", "supersedes"})

ONTOLOGY_LINK_SYSTEM_PROMPT = """\
You infer typed relationships between pairs of facts/entities.

Given a list of items (facts and/or entities) and an allowed-relations
list, return zero or more typed edges (source_id → target_id) drawn
ONLY from the allowed list.

Rules:
- Use ONLY the source_id / target_id values shown in the ITEMS list.
  Do not invent ids.
- Do not invent new relation types.
- Edge weight (0.0–1.0) reflects strength of the relationship as
  supported by the items' raw text.
- If you propose `contradicts` or `supersedes`, include a brief
  rationale explaining the conflict.
- Skip any pair where evidence is too weak (no edge) — better to miss
  an edge than fabricate one.
- A source_id may not equal its target_id (no self-loops).

Output: JSON array of {source_id, target_id, rel_type, weight,
confidence, rationale}.
If no edges are warranted, return: []
"""


class OntologyLinker(Linker):
    """LLM-backed typed-edge inference between facts and/or entities.

    Detect-only — emits edges including `contradicts` but does NOT
    auto-resolve. Downstream pipelines (Cloud) decide whether to surface
    contradictions to a human reviewer or apply policy rules.

    Args:
        provider:           LLMProvider instance (anything with .complete).
        allowed_relations:  whitelist of edge types this linker may
                            emit. Cloud verticals override with
                            domain-specific lists (Marketing adds
                            "uses_creative", "targets_segment", etc.).
        max_pairs_per_call: cap items considered per LLM call. Each item
                            can pair with every other, so N items yield
                            up to N*(N-1)/2 candidate edges. Default 50
                            (≈1225 candidate pairs per call).
        max_tokens:         LLM completion budget.
        max_retries:        retries on provider exception (exponential
                            backoff).

    Example:
        >>> from feather_db.providers import ClaudeProvider
        >>> from feather_db.extractors import OntologyLinker, Fact
        >>> linker = OntologyLinker(
        ...     provider=ClaudeProvider(),
        ...     allowed_relations=["caused_by", "supports", "contradicts"],
        ... )
        >>> facts = [
        ...     Fact("Acme", "had_CTR", "4.5%", confidence=0.9),
        ...     Fact("Acme", "had_CTR", "4.2%", confidence=0.95,
        ...          valid_at=1712000000),
        ... ]
        >>> edges = linker.link(facts)
        >>> # [OntologyEdge(source_id='f_1', target_id='f_0',
        >>> #              rel_type='supersedes', rationale='...')]
    """

    name = "ontology_linker"
    version = "0.1.0"

    def __init__(self, provider, *,
                 allowed_relations: Optional[list[str]] = None,
                 max_pairs_per_call: int = 50,
                 max_tokens: int = 1500,
                 max_retries: int = 2):
        self._provider = provider
        self.allowed_relations = (allowed_relations
                                  if allowed_relations is not None
                                  else list(DEFAULT_RELATIONS))
        self._allowed_set = set(self.allowed_relations)
        self._max_items = max_pairs_per_call
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    def link(self, items: list[Any], *,
             context: Optional[dict] = None) -> list[OntologyEdge]:
        """Infer typed edges between items.

        Returns an empty list if there are fewer than 2 items, the
        provider call fails, or the response can't be parsed. Edges
        whose source_id/target_id don't match a rendered item id are
        dropped (defends against id hallucination).
        """
        if not items or len(items) < 2:
            return []
        ctx = context or {}
        on_error = ctx.get("on_error")

        items = list(items)[: self._max_items]
        rendered, valid_ids = self._render_items(items)
        if len(valid_ids) < 2:
            return []

        messages = [
            {"role": "system", "content": ONTOLOGY_LINK_SYSTEM_PROMPT},
            {"role": "user", "content": self._user_message(rendered, ctx)},
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
                on_error(f"OntologyLinker LLM call failed: {last_err}")
            return []

        parsed, ok, _ = extract_json(raw)
        if not ok or not isinstance(parsed, list):
            if on_error:
                on_error(
                    f"OntologyLinker JSON parse failed; "
                    f"raw[:200]={raw[:200]!r}"
                )
            return []

        valid_set = set(valid_ids)
        edges: list[OntologyEdge] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            sid = (item.get("source_id") or "").strip()
            tid = (item.get("target_id") or "").strip()
            rel = (item.get("rel_type") or "").strip()
            if not (sid and tid and rel):
                continue
            if sid == tid:
                continue
            if sid not in valid_set or tid not in valid_set:
                continue
            if rel not in self._allowed_set:
                continue

            try:
                weight = float(item.get("weight", 1.0))
            except (TypeError, ValueError):
                weight = 0.5
            weight = max(0.0, min(1.0, weight))

            conf_raw = item.get("confidence", weight)
            try:
                conf = float(conf_raw)
            except (TypeError, ValueError):
                conf = weight
            conf = max(0.0, min(1.0, conf))

            rationale = item.get("rationale")
            if isinstance(rationale, str):
                rationale = rationale.strip() or None
            else:
                rationale = None
            if rel in REQUIRES_RATIONALE and not rationale:
                continue

            edges.append(OntologyEdge(
                source_id=sid,
                target_id=tid,
                rel_type=rel,
                weight=weight,
                confidence=conf,
                rationale=rationale,
            ))
        return edges

    # ── Item rendering ─────────────────────────────────────────────────

    def _render_items(self, items: list[Any]) -> tuple[list[str], list[str]]:
        """Render items as `id: summary` lines + return the id list."""
        rendered: list[str] = []
        ids: list[str] = []
        f_idx = 0
        e_idx = 0
        for it in items:
            if isinstance(it, Fact):
                iid = f"f_{f_idx}"
                f_idx += 1
                summary = self._summarize_fact(it)
            elif isinstance(it, Entity):
                iid = f"e_{e_idx}"
                e_idx += 1
                summary = self._summarize_entity(it)
            else:
                continue
            ids.append(iid)
            rendered.append(f"{iid}: {summary}")
        return rendered, ids

    @staticmethod
    def _summarize_fact(f: Fact) -> str:
        bits = [f"FACT [{f.subject} | {f.predicate} | {f.object}]"]
        if f.valid_at:
            bits.append(f"valid_at={f.valid_at}")
        if f.confidence and f.confidence < 1.0:
            bits.append(f"confidence={f.confidence:.2f}")
        return " ".join(bits)

    @staticmethod
    def _summarize_entity(e: Entity) -> str:
        bits = [f"ENTITY [{e.kind}: {e.surface_form} → {e.canonical_id}]"]
        if e.aliases:
            bits.append(f"aliases={', '.join(e.aliases[:5])}")
        return " ".join(bits)

    def _user_message(self, rendered: list[str], ctx: dict) -> str:
        lines: list[str] = []
        if ctx.get("namespace"):
            lines.append(f"Namespace: {ctx['namespace']}")
            lines.append("")
        if ctx.get("vertical_hint"):
            lines.append(f"Domain: {ctx['vertical_hint']}")
            lines.append("")
        lines.append("ALLOWED RELATIONS:")
        lines.append(", ".join(self.allowed_relations))
        lines.append("")
        lines.append("ITEMS:")
        lines.extend(rendered)
        return "\n".join(lines)
