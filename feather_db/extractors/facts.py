"""LLM-backed FactExtractor — converts raw text to atomic (subj, pred, obj)
triples with provenance and confidence.

Status: stub (Phase 9.1 Week 1 — implementation queued for Week 2).
The class signature, docstring, and prompt template are locked here so
the pipeline + tests can be wired against the API today.
"""
from __future__ import annotations
from typing import Optional

from .base import Extractor, Fact

# Prompt is locked here so changes are versioned via git.
FACT_EXTRACTION_SYSTEM_PROMPT = """\
You extract atomic, verifiable facts from text and return them as JSON.

Each fact must be a triple: (subject, predicate, object).

Rules:
- Only extract facts substantively supported by the text. Do not infer.
- Subject and object should be specific noun phrases (canonical names
  preferred). Predicate is a short verb-phrase or relation.
- If a fact has a temporal anchor in the text, include it under valid_at
  in ISO 8601 format.
- Rate your confidence (0.0–1.0) for each fact based on how directly
  the text supports it.
- Return at most 20 facts per input.

Output format (JSON array):
[
  {
    "subject": "...",
    "predicate": "...",
    "object": "...",
    "confidence": 0.0,
    "valid_at": "2024-03-15T00:00:00Z" | null
  },
  ...
]

If the text contains no extractable facts, return: []
"""


class FactExtractor(Extractor):
    """LLM-backed atomic-fact extractor.

    Pluggable on any feather_db.providers.LLMProvider. By default uses
    the system LLM (Claude Haiku) — Cloud locks this; OSS users can
    swap.

    Args:
        provider:        LLMProvider instance.
        max_facts_per_call: cap to keep token cost bounded (default 20).
        min_confidence:  drop facts below this threshold (default 0.5).

    Example:
        >>> from feather_db.providers import ClaudeProvider
        >>> from feather_db.extractors import FactExtractor
        >>> extractor = FactExtractor(provider=ClaudeProvider())
        >>> facts = extractor.extract(
        ...     "Acme launched the Summer Sale campaign on March 15, "
        ...     "2024. CTR averaged 4.5% in the first week.",
        ...     context={"source_id": "memo-2024-03-20"},
        ... )
        >>> # facts -> [
        >>> #   Fact(subject='Acme', predicate='launched',
        >>> #        object='Summer Sale campaign',
        >>> #        valid_at=1710460800, confidence=0.95, ...),
        >>> #   Fact(subject='Summer Sale campaign first week',
        >>> #        predicate='had_average_CTR', object='4.5%', ...),
        >>> # ]
    """

    name = "fact_extractor"
    version = "0.1.0-stub"

    def __init__(self, provider, *,
                 max_facts_per_call: int = 20,
                 min_confidence: float = 0.5):
        self._provider = provider
        self._max_facts = max_facts_per_call
        self._min_confidence = min_confidence

    def extract(self, text: str, *,
                context: Optional[dict] = None) -> list[Fact]:
        """Extract atomic facts from `text`.

        Returns a list of Fact instances with `provenance` populated from
        context["source_id"] when present, and `extracted_at` stamped to
        the call time.

        IMPLEMENTATION STATUS: stub. See Phase 9.1 Week 2 in
        docs/architecture/phase9-plan.md. The wiring is in place for
        downstream code to import and instantiate; calling .extract()
        raises NotImplementedError until the LLM prompt + JSON parse +
        validation loop ship.
        """
        raise NotImplementedError(
            "FactExtractor.extract — Phase 9.1 Week 2. "
            "The class signature is locked; implementation arrives next."
        )
