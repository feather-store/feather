"""LLM-based judge + answerer for memory QA.

A single class wraps two LLMProvider calls:
    1. answer(question, context) -> str       generation given retrieved context
    2. score(predicted, gold, question) -> 0/1 binary correctness vs reference

Uses feather_db.providers.* under the hood, so the same code works with
Claude / OpenAI / Gemini / Ollama just by swapping the provider object.
"""
from __future__ import annotations
import os
import random
import time
from typing import Optional, Callable

from .judges import Judge, JudgeResult
from . import lme_answer, lme_eval


def _retry(fn: Callable, *, attempts: int = 6, base_delay: float = 2.0,
           max_delay: float = 60.0, label: str = "llm"):
    """Exponential backoff with jitter, ~2 minutes in all. Three tries over
    ~4 s did not outlast a rate-limit window at 8 workers, and every call
    that gave up became a question scored wrong."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            if i + 1 < attempts:
                delay = min(max_delay, base_delay ** (i + 1))
                time.sleep(delay * random.uniform(0.75, 1.25))
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last}")


# ---------------- Prompts ----------------

# The answer prompt (procedures per question category + the DAYS_BETWEEN
# tool) is in lme_answer; both judge prompts, the official LongMemEval one
# and the lenient one used here, are in lme_eval. All shared with
# benchmarks/longmemeval_qa.py.


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
    if name == "gemini-text":
        from .providers_extra import GeminiTextProvider
        return GeminiTextProvider(model=model)
    if name == "openrouter":
        from .providers_extra import OpenRouterProvider
        return OpenRouterProvider(model=model)
    if name in ("azure", "azure-openai"):
        from .providers_azure import AzureChatProvider
        return AzureChatProvider(model=model)
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
                 max_context_chars: int = 160_000,
                 answer_max_tokens: int = 2048,
                 judge_max_tokens: int = 256,
                 count_votes: int = 1,
                 map_provider: Optional[str] = None,
                 map_model: Optional[str] = None):
        self._judge_llm = _provider_from_name(provider, model)
        self._answer_llm = (_provider_from_name(answerer_provider, answerer_model)
                            if answerer_provider else self._judge_llm)
        # The count map step: ~22 short calls per count question. A strong
        # answerer need not make them; default is the answerer.
        self._map_llm = (_provider_from_name(
            map_provider or answerer_provider or provider, map_model)
            if (map_provider or map_model) else self._answer_llm)
        self._max_context_chars = max_context_chars
        # 2.5-tier reasoning models (thinking mode) burn many tokens before
        # producing the final answer; 200 is not enough. 2048 covers the
        # counting procedure's one-line-per-instance list + a final answer.
        self._answer_max_tokens = answer_max_tokens
        self._judge_max_tokens = judge_max_tokens
        # >1: count questions are answered this many times, majority wins.
        self._count_votes = count_votes
        self.name = (f"llm_judge={provider}/{model or 'default'}"
                     f"_ans={answerer_provider or provider}"
                     f"/{answerer_model or model or 'default'}"
                     + (f"_map={map_provider or answerer_provider or provider}"
                        f"/{map_model or 'default'}"
                        if (map_provider or map_model) else ""))

    def _chat(self, llm, max_tokens: int, label: str,
              temperature: float = 0.0) -> lme_eval.Chat:
        def chat(system: str, user: str) -> str:
            messages = ([{"role": "system", "content": system}] if system
                        else []) + [{"role": "user", "content": user}]
            return _retry(lambda: llm.complete(
                messages, max_tokens=max_tokens,
                temperature=temperature).strip(),
                label=label)
        return chat

    # ---------------- Answer step ----------------

    def answer_full(self, question: str, context: str,
                    question_date: str = "") -> tuple[str, str]:
        """(final answer, the model's whole last reply)."""
        # The scenarios budget their own context; this is only a backstop.
        ctx = (context or "")[: self._max_context_chars]
        if self._count_votes > 1 and lme_answer.is_count_question(question):
            # One greedy reading plus sampled ones; majority wins.
            chats = [self._chat(self._answer_llm, self._answer_max_tokens,
                                "answerer", 0.0 if i == 0 else 0.7)
                     for i in range(self._count_votes)]
            final, transcript = lme_answer.answer_vote(chats, question, ctx,
                                                       question_date)
            if not transcript:
                return "[answerer error: every reading failed]", ""
            return final, transcript
        try:
            reply = lme_answer.answer(
                self._chat(self._answer_llm, self._answer_max_tokens,
                           "answerer"), question, ctx, question_date)
        except Exception as e:
            return f"[answerer error: {e}]", ""
        return lme_answer.final_answer(reply), reply

    def answer(self, question: str, context: str,
               question_date: str = "") -> str:
        """The model's FINAL ANSWER line (reasoning stripped, so the judge
        grades the answer, not the working)."""
        return self.answer_full(question, context, question_date)[0]

    def answer_chat(self) -> lme_eval.Chat:
        """The answer model as configured for answering (count reduce)."""
        return self._chat(self._answer_llm, self._answer_max_tokens, "reducer")

    def map_chat(self) -> lme_eval.Chat:
        """The answer model, short and without thinking: one per session
        in a count's map step."""
        return self._chat(self._map_llm, 512, "mapper")

    def planner_chat(self) -> lme_eval.Chat:
        """The answer model, for lme_retrieve's follow-up queries."""
        return self._chat(self._answer_llm, 512, "planner")

    # ---------------- Judge step (lenient) ----------------

    def verdict(self, question: str, gold: str, predicted: str) -> dict:
        try:
            return lme_eval.lenient_verdict(
                self._chat(self._judge_llm, self._judge_max_tokens, "judge"),
                question, str(gold or "").strip(), str(predicted or "").strip())
        except Exception as e:
            return {"correct": False, "raw": "", "error": str(e)}

    def score(self, predicted, gold, question: str = "") -> JudgeResult:
        if not str(gold if gold is not None else "").strip():
            return JudgeResult(0.0, "empty gold")
        v = self.verdict(question, gold, predicted)
        if v.get("error"):
            return JudgeResult(0.0, f"judge call failed: {v['error']}")
        return JudgeResult(1.0 if v["correct"] else 0.0, v["raw"])


class OfficialJudge:
    """LongMemEval's official per-type judge (lme_eval). Run it on
    gpt-4o-2024-08-06 for numbers comparable with published ones."""

    def __init__(self, provider: str = "openai",
                 model: str = lme_eval.OFFICIAL_JUDGE_MODEL):
        self._llm = _provider_from_name(provider, model)
        self.name = f"official={provider}/{model}"

    def verdict(self, qtype: str, question: str, gold: str, predicted: str,
                abstention: bool) -> dict:
        def chat(system: str, user: str) -> str:
            # The official call: one user message, temperature 0, 10 tokens.
            return _retry(lambda: self._llm.complete(
                [{"role": "user", "content": user}], max_tokens=10,
                temperature=0.0), label="official judge")
        try:
            return lme_eval.official_verdict(chat, qtype, question, gold,
                                             predicted, abstention)
        except Exception as e:
            return {"correct": False, "raw": "", "error": str(e)}
