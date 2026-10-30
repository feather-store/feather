"""Providers the benchmark needs beyond feather_db.providers.

  gemini-text   Gemini in plain-text mode. feather_db's GeminiProvider forces
                JSON output (right for extractors), which breaks the
                answerer's "FINAL ANSWER:" line and DAYS_BETWEEN protocol.
  openrouter    any OpenRouter model through the OpenAI SDK, text mode.
                Serves openai/gpt-4o-2024-08-06 for the official judge.
  GeminiEmbedder  gemini-embedding-001, batched.

All retry rate limits with exponential backoff: a full run is thousands
of calls, and a 429 must cost a wait, not a question.
"""
from __future__ import annotations
import os
import random
import time
from typing import Optional

import numpy as np

from .embedders import Embedder

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _backoff(fn, *, attempts: int = 8, base: float = 2.0, cap: float = 90.0,
             label: str = "call"):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:        # 429 / 5xx / timeouts; auth errors too,
            last = e                  # but those fail fast on every attempt
            msg = str(e)
            if any(s in msg for s in ("401", "403", "API key not valid",
                                      "INVALID_ARGUMENT")):
                break
            time.sleep(min(cap, base ** i) * (0.5 + random.random()))
    raise RuntimeError(f"{label} failed: {last}")


def _no_thinking(model: str) -> tuple[int, int]:
    """(thinking_budget, extra output tokens) for a call that should not
    think. 2.5 Pro cannot switch thinking off (its minimum budget is 128)
    and rejects 0: every planner call failed, and plan_followups turns a
    failure into "no follow-ups", so multi-round retrieval stopped
    silently."""
    return (128, 128) if "pro" in (model or "").lower() else (0, 0)


class GeminiTextProvider:
    """complete(messages, max_tokens, temperature) -> str, text mode.

    Gemini 2.5 thinks by default and thinking tokens count against
    max_output_tokens: a 256-token judge call could come back empty. So a
    short call (max_tokens <= 512: judges, planner) runs without thinking,
    and a long one gets `thinking_headroom` extra output tokens. think=
    False forces it off (fact extraction: a transcription task, where
    thinking costs ~3x the output tokens for no gain)."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 thinking_headroom: int = 8192, think: Optional[bool] = None):
        from google import genai
        from google.genai import types
        self._model = model or "gemini-2.5-flash"
        self._client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY")
                                    or os.environ.get("GEMINI_API_KEY"))
        self._types = types
        self._headroom = thinking_headroom
        self._think = think          # None: by max_tokens, as described above

    def complete(self, messages: list[dict], max_tokens: int = 512,
                 temperature: float = 0.0) -> str:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [self._types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[self._types.Part(text=m["content"])])
            for m in messages if m["role"] != "system"]
        think = max_tokens > 512 if self._think is None else self._think
        budget, extra = (-1, self._headroom) if think else _no_thinking(self._model)
        cfg = self._types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens + extra,
            system_instruction=system or None,
            thinking_config=self._types.ThinkingConfig(thinking_budget=budget),
        )
        r = _backoff(lambda: self._client.models.generate_content(
            model=self._model, contents=contents, config=cfg), label=self._model)
        return r.text or ""

    def __repr__(self) -> str:
        return f"GeminiTextProvider({self._model})"


class OpenRouterProvider:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        from openai import OpenAI
        self._model = model or "openai/gpt-4o-2024-08-06"
        self._client = OpenAI(base_url=OPENROUTER_BASE, max_retries=0,
                              api_key=api_key or os.environ.get("OPENROUTER_API_KEY"))

    def complete(self, messages: list[dict], max_tokens: int = 512,
                 temperature: float = 0.0) -> str:
        r = _backoff(lambda: self._client.chat.completions.create(
            model=self._model, messages=messages, max_tokens=max_tokens,
            temperature=temperature), label=self._model)
        return (r.choices[0].message.content or "") if r.choices else ""

    def __repr__(self) -> str:
        return f"OpenRouterProvider({self._model})"


class GeminiEmbedder(Embedder):
    """gemini-embedding-001 (3072-d native; `dim` truncates, then
    renormalises). Texts are clipped: the model reads ~2k tokens."""

    BATCH = 100
    MAX_CHARS = 8000

    def __init__(self, model: str = "gemini-embedding-001",
                 dim: Optional[int] = None, api_key: Optional[str] = None):
        from google import genai
        from google.genai import types
        self._client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY")
                                    or os.environ.get("GEMINI_API_KEY"))
        self._types = types
        self._model = model
        self.dim = dim or 3072
        self.name = f"gemini_{model}_d{self.dim}"

    def _call(self, texts: list[str]) -> np.ndarray:
        cfg = (self._types.EmbedContentConfig(output_dimensionality=self.dim)
               if self.dim != 3072 else None)
        r = _backoff(lambda: self._client.models.embed_content(
            model=self._model, contents=[(t or " ")[: self.MAX_CHARS] for t in texts],
            config=cfg), label=self._model)
        v = np.asarray([e.values for e in r.embeddings], dtype=np.float32)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(n > 0, n, 1.0)

    def embed(self, text: str) -> np.ndarray:
        return self._call([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.concatenate([self._call(texts[i:i + self.BATCH])
                               for i in range(0, len(texts), self.BATCH)])
