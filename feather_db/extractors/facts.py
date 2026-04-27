"""LLM-backed FactExtractor — converts raw text to atomic (subj, pred, obj)
triples with provenance and confidence.
"""
from __future__ import annotations
import datetime as _dt
import time
from typing import Optional

from .base import Extractor, Fact
from ._jsonparse import extract_json

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
    version = "0.1.0"

    def __init__(self, provider, *,
                 max_facts_per_call: int = 20,
                 min_confidence: float = 0.5,
                 max_tokens: int = 1500,
                 max_retries: int = 2):
        self._provider = provider
        self._max_facts = max_facts_per_call
        self._min_confidence = min_confidence
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    def extract(self, text: str, *,
                context: Optional[dict] = None) -> list[Fact]:
        """Extract atomic facts from `text`.

        Returns a list of Fact instances with `provenance` populated from
        context["source_id"] when present, and `extracted_at` stamped to
        the call time. Empty list on any unrecoverable parse failure
        (logged via context["on_error"] callback if provided).
        """
        if not text or not text.strip():
            return []
        ctx = context or {}
        source_id = ctx.get("source_id")
        on_error = ctx.get("on_error")

        messages = [
            {"role": "system", "content": FACT_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": self._user_message(text, ctx)},
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
                on_error(f"FactExtractor LLM call failed: {last_err}")
            return []

        parsed, ok, _ = extract_json(raw)
        if not ok or not isinstance(parsed, list):
            if on_error:
                on_error(f"FactExtractor JSON parse failed; raw[:200]={raw[:200]!r}")
            return []

        now_ts = int(time.time())
        facts: list[Fact] = []
        for item in parsed[: self._max_facts]:
            if not isinstance(item, dict):
                continue
            subj = (item.get("subject") or "").strip()
            pred = (item.get("predicate") or "").strip()
            obj_ = (item.get("object") or "").strip()
            if not (subj and pred and obj_):
                continue
            try:
                confidence = float(item.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))
            if confidence < self._min_confidence:
                continue

            valid_at: Optional[int] = None
            v = item.get("valid_at")
            if v:
                try:
                    valid_at = int(_dt.datetime.fromisoformat(
                        str(v).replace("Z", "+00:00")
                    ).timestamp())
                except (TypeError, ValueError):
                    valid_at = None

            facts.append(Fact(
                subject=subj,
                predicate=pred,
                object=obj_,
                confidence=confidence,
                provenance=source_id,
                extracted_at=now_ts,
                valid_at=valid_at,
                raw_text=text if len(text) <= 2000 else None,
                extractor_name=self.name,
                extractor_version=self.version,
            ))
        return facts

    def _user_message(self, text: str, ctx: dict) -> str:
        hints = []
        if ctx.get("namespace"):
            hints.append(f"Namespace: {ctx['namespace']}")
        if ctx.get("vertical_hint"):
            hints.append(f"Domain: {ctx['vertical_hint']}")
        prefix = ("\n".join(hints) + "\n\n") if hints else ""
        return f"{prefix}TEXT TO PROCESS:\n{text}"
