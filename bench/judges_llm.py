"""LLM-based judge + answerer for memory QA.

A single class wraps two LLMProvider calls:
    1. answer(question, context) -> str       generation given retrieved context
    2. score(predicted, gold, question) -> 0/1 binary correctness vs reference

Uses feather_db.providers.* under the hood, so the same code works with
Claude / OpenAI / Gemini / Ollama just by swapping the provider object.
"""
from __future__ import annotations
import os
import re
import time
from typing import Optional, Callable

from .judges import Judge, JudgeResult


def _retry(fn: Callable, *, attempts: int = 3, base_delay: float = 1.5,
           label: str = "llm"):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            if i + 1 < attempts:
                time.sleep(base_delay ** (i + 1))
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last}")


# ---------------- Prompts ----------------

ANSWER_SYSTEM = (
    "You are a helpful assistant answering questions about a user's past "
    "conversations. Use the memory context to answer the question. The "
    "answer may require combining facts across multiple snippets, doing "
    "arithmetic, or making reasonable inferences from what was said. "
    "Think step by step before answering when the question requires "
    "synthesis. Only reply 'I don't know' if the relevant facts are "
    "genuinely absent from the context. Keep the final answer concise "
    "(1–2 sentences) and put it on the last line."
)

ANSWER_USER_TMPL = (
    "Memory context (snippets retrieved from past conversations, separated "
    "by '---'):\n\n{context}\n\n"
    "Question: {question}\n\n"
    "Reason briefly, then give the final answer on the last line:"
)

# Modeled after the LongMemEval rubric (Xu et al., 2024). Binary judgment.
JUDGE_SYSTEM = (
    "You are an expert evaluator for chat-assistant memory benchmarks. "
    "Determine whether the assistant's answer correctly addresses the "
    "user's question, given a reference answer. Be lenient about phrasing "
    "but strict about facts. Output a JSON object: "
    '{"correct": true|false, "reason": "short rationale"}.'
)

JUDGE_USER_TMPL = (
    "Question: {question}\n"
    "Reference answer: {gold}\n"
    "Assistant answer: {predicted}\n\n"
    "Is the assistant answer correct?"
)


def _provider_from_name(name: str, model: Optional[str] = None):
    """Construct an LLMProvider by short name. Lazy import so optional deps don't break import."""
    name = (name or "").lower()
    if name == "gemini":
        from feather_db.providers import GeminiProvider
        return GeminiProvider(model=model or "gemini-2.0-flash")
    if name == "claude":
        from feather_db.providers import ClaudeProvider
        return ClaudeProvider(model=model or "claude-haiku-4-5-20251001")
    if name == "openai":
        from feather_db.providers import OpenAIProvider
        return OpenAIProvider(model=model or "gpt-4o-mini")
    if name == "ollama":
        from feather_db.providers import OllamaProvider
        return OllamaProvider(model=model or "llama3.1:8b")
    raise ValueError(f"unknown LLM provider: {name}")


class LLMJudge(Judge):
    """LLM-based answerer + judge. Holds two LLMProvider objects.

    By default, both roles use the same provider (cheap, simple). Pass
    `answerer_provider=` to override the answer model independently —
    useful for "ceiling" runs where the answerer is a frontier model
    and the judge is a cheap one.
    """

    name = "llm"

    def __init__(self,
                 provider: str = "gemini",
                 answerer_provider: Optional[str] = None,
                 model: Optional[str] = None,
                 answerer_model: Optional[str] = None,
                 max_context_chars: int = 12_000,
                 answer_max_tokens: int = 1024,
                 judge_max_tokens: int = 256):
        self._judge_llm = _provider_from_name(provider, model)
        self._answer_llm = (_provider_from_name(answerer_provider, answerer_model)
                            if answerer_provider else self._judge_llm)
        self._max_context_chars = max_context_chars
        # 2.5-tier reasoning models (thinking mode) burn many tokens before
        # producing the final answer; 200 is not enough. 1024 covers
        # chain-of-thought + a concise final answer comfortably.
        self._answer_max_tokens = answer_max_tokens
        self._judge_max_tokens = judge_max_tokens
        self.name = (f"llm_judge={provider}/{model or 'default'}"
                     f"_ans={answerer_provider or provider}"
                     f"/{answerer_model or model or 'default'}")

    # ---------------- Answer step ----------------

    def answer(self, question: str, context: str) -> str:
        # Trim context to keep token bills bounded.
        ctx = (context or "")[: self._max_context_chars]
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user",   "content": ANSWER_USER_TMPL.format(
                context=ctx, question=question)},
        ]
        try:
            return _retry(
                lambda: self._answer_llm.complete(
                    messages, max_tokens=self._answer_max_tokens, temperature=0.0
                ).strip(),
                label="answerer",
            )
        except Exception as e:
            return f"[answerer error: {e}]"

    # ---------------- Judge step ----------------

    def score(self, predicted, gold, question: str = "") -> JudgeResult:
        p = str(predicted or "").strip()
        g = str(gold if gold is not None else "").strip()
        if not g:
            return JudgeResult(0.0, "empty gold")

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": JUDGE_USER_TMPL.format(
                question=question, gold=g, predicted=p)},
        ]
        try:
            raw = _retry(
                lambda: self._judge_llm.complete(
                    messages, max_tokens=self._judge_max_tokens, temperature=0.0
                ),
                label="judge",
            )
        except Exception as e:
            return JudgeResult(0.0, f"judge call failed: {e}")

        # Robust JSON extraction — tolerate markdown fences, prose, etc.
        score_val, reason = _parse_judge_output(raw)
        return JudgeResult(score_val, reason)


_JSON_BOOL_RE = re.compile(r'"correct"\s*:\s*(true|false)', re.IGNORECASE)


def _parse_judge_output(raw: str) -> tuple[float, str]:
    if not raw:
        return 0.0, "empty judge response"
    m = _JSON_BOOL_RE.search(raw)
    if m:
        is_correct = m.group(1).lower() == "true"
        # try to grab reason
        reason = ""
        m2 = re.search(r'"reason"\s*:\s*"([^"]*)"', raw)
        if m2:
            reason = m2.group(1)
        return (1.0 if is_correct else 0.0), reason or raw[:120]
    # Fallback: scan for keywords.
    low = raw.lower()
    if "correct: true" in low or "is correct" in low or "correct." in low.replace("incorrect", ""):
        return 1.0, raw[:120]
    if "incorrect" in low or "wrong" in low or "false" in low:
        return 0.0, raw[:120]
    return 0.0, f"unparseable: {raw[:120]}"
