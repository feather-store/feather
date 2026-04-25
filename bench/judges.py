"""Pluggable judges for memory-QA scoring.

Phase 1: substring match (free, deterministic).
Phase 2: LLM judge using LongMemEval's rubric prompt with Claude Haiku.

Both return a score in [0.0, 1.0] and a short rationale string.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class JudgeResult:
    score: float            # 0.0 .. 1.0
    rationale: str = ""     # short explanation


class Judge(ABC):
    name: str

    @abstractmethod
    def score(self, predicted: str, gold: str, question: str = "") -> JudgeResult:
        ...


class SubstringJudge(Judge):
    """Returns 1.0 iff gold (lowercased, stripped) is a substring of predicted.

    Phase 1: cheap and crude. We use this to validate that the pipeline
    surfaces the right memory, not that the LLM phrases an answer well.
    On real LongMemEval text, expect this to under-score the system —
    use LLMJudge for the published numbers.
    """
    name = "substring"

    def score(self, predicted, gold, question="") -> JudgeResult:
        p = str(predicted or "").lower()
        g = str(gold if gold is not None else "").lower().strip()
        if not g:
            return JudgeResult(0.0, "empty gold")
        hit = g in p
        return JudgeResult(1.0 if hit else 0.0,
                           "substring hit" if hit else "substring miss")


def get_judge(name: str, **kwargs) -> Judge:
    if name == "substring":
        return SubstringJudge()
    if name == "llm":
        from .judges_llm import LLMJudge  # lazy import
        return LLMJudge(**kwargs)
    raise ValueError(f"unknown judge: {name}")
