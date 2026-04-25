"""OpenAI embedder.

Uses the official OpenAI SDK and OPENAI_API_KEY. Default model is
text-embedding-3-small (1536-dim) — what Mem0 and Zep used in their
LongMemEval baselines, so it's the apples-to-apples choice.

Costs (text-embedding-3-small): ~$0.02 per 1M input tokens.
For LongMemEval oracle on 20 questions: ~$0.005 total.
"""
from __future__ import annotations
import os
import time
from typing import Optional
import numpy as np

from .embedders import Embedder


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = "text-embedding-3-small",
                 api_key: Optional[str] = None,
                 dim: Optional[int] = None,
                 max_retries: int = 3,
                 retry_backoff: float = 1.5):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "pip install openai  # required for OpenAIEmbedder"
            ) from e

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "Provide api_key= or set OPENAI_API_KEY env var."
            )
        self._client = OpenAI(api_key=key)
        self._model = model
        # text-embedding-3-small native dim is 1536. text-embedding-3-large is 3072.
        # Both support `dimensions=` parameter to truncate; default to native.
        self._native_dim = 3072 if "large" in model else 1536
        self.dim = dim or self._native_dim
        self.name = f"openai_{model}_d{self.dim}"
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff

    def _call(self, texts: list[str]) -> list[list[float]]:
        kwargs = {"model": self._model, "input": texts}
        if self.dim != self._native_dim:
            kwargs["dimensions"] = self.dim
        last_err = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.embeddings.create(**kwargs)
                return [d.embedding for d in resp.data]
            except Exception as e:
                last_err = e
                if attempt + 1 < self._max_retries:
                    time.sleep(self._retry_backoff ** attempt)
        raise RuntimeError(f"OpenAI embeddings failed after retries: {last_err}")

    def embed(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.dim, dtype=np.float32)
        out = self._call([text])[0]
        return np.asarray(out, dtype=np.float32)

    def embed_batch(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            # OpenAI rejects empty strings — substitute a single space.
            sanitized = [t if t else " " for t in chunk]
            out.extend(self._call(sanitized))
        return np.asarray(out, dtype=np.float32)
