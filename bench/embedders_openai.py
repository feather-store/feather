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
    """OpenAI or Azure OpenAI embedder. Env-driven Azure mode.

    Azure mode is auto-enabled when AZURE_OPENAI_ENDPOINT is set. In that
    case the constructor's `model` arg is interpreted as the Azure
    deployment name (because Azure routes by deployment, not model id).
    """

    def __init__(self, model: str = "text-embedding-3-small",
                 api_key: Optional[str] = None,
                 dim: Optional[int] = None,
                 max_retries: int = 3,
                 retry_backoff: float = 1.5,
                 azure_endpoint: Optional[str] = None,
                 azure_api_version: Optional[str] = None,
                 azure_deployment: Optional[str] = None):
        # Azure mode if endpoint provided (or AZURE_OPENAI_ENDPOINT in env).
        endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self._is_azure = bool(endpoint)

        if self._is_azure:
            try:
                from openai import AzureOpenAI
            except ImportError as e:
                raise ImportError("pip install openai>=1.0  # required") from e
            key = (api_key or os.environ.get("AZURE_OPENAI_API_KEY")
                   or os.environ.get("OPENAI_API_KEY"))
            if not key:
                raise ValueError(
                    "Provide api_key= or set AZURE_OPENAI_API_KEY env var."
                )
            api_version = (azure_api_version
                           or os.environ.get("AZURE_OPENAI_API_VERSION")
                           or "2024-02-01")
            self._client = AzureOpenAI(
                api_key=key,
                api_version=api_version,
                azure_endpoint=endpoint,
            )
            # In Azure, `model` parameter at request time = deployment name.
            self._model = (azure_deployment
                           or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
                           or model)
            self._is_legacy_api = api_version < "2024-01-01"
            self.name = f"azure_{self._model}"
        else:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("pip install openai  # required") from e
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("Provide api_key= or set OPENAI_API_KEY env var.")
            self._client = OpenAI(api_key=key)
            self._model = model
            self._is_legacy_api = False
            self.name = f"openai_{model}"

        # text-embedding-3-small native dim 1536, large 3072.
        self._native_dim = 3072 if "large" in self._model else 1536
        # On legacy Azure api-versions, `dimensions=` is rejected — embeddings
        # always come back at native dim, so we must use that.
        if self._is_legacy_api:
            if dim and dim != self._native_dim:
                print(f"[embedder] api-version pre-2024 ignores `dimensions` "
                      f"— forcing dim={self._native_dim}")
            self.dim = self._native_dim
        else:
            self.dim = dim or self._native_dim
        self.name = f"{self.name}_d{self.dim}"
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff

    def _call(self, texts: list[str]) -> list[list[float]]:
        kwargs = {"model": self._model, "input": texts}
        # `dimensions` parameter is a 2024+ feature; old Azure api-versions
        # (e.g. 2023-05-15) reject it. Skip if we know we're on legacy.
        if self.dim != self._native_dim and not self._is_legacy_api:
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
        raise RuntimeError(f"embeddings failed after retries: {last_err}")

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
