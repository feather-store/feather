"""Pluggable embedding service.

Stores config in-memory (single process). On every restart the operator
must re-set credentials via PUT /v1/admin/embedding-config.
The API key itself is never echoed back to clients.

Supported providers:
  openai   — OpenAI embeddings (default model: text-embedding-3-small)
  voyage   — Voyage AI embeddings
  cohere   — Cohere embed
  ollama   — local Ollama HTTP API (no key required)
  none     — no embedding service configured (caller must pass a vector)
"""
from __future__ import annotations
import os
import json
from typing import List, Optional
from threading import Lock

import urllib.request
import urllib.error


_DEFAULT_MODELS = {
    "openai":  "text-embedding-3-small",
    "voyage":  "voyage-2",
    "cohere":  "embed-english-v3.0",
    "ollama":  "nomic-embed-text",
}


class EmbeddingProvider:
    def __init__(self):
        self._lock = Lock()
        # bootstrap from env if present
        self.provider: str = os.getenv("FEATHER_EMBED_PROVIDER", "none")
        self.model:    str = os.getenv("FEATHER_EMBED_MODEL", _DEFAULT_MODELS.get(self.provider, ""))
        self.base_url: str = os.getenv("FEATHER_EMBED_BASE_URL", "")
        self._api_key: str = os.getenv("FEATHER_EMBED_API_KEY", "")
        self.dim:      int = int(os.getenv("FEATHER_EMBED_DIM", "768"))

    # ── public state introspection ──────────────────────────────
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "provider":     self.provider,
                "model":        self.model,
                "base_url":     self.base_url,
                "api_key_set":  bool(self._api_key),
                "dim":          self.dim,
            }

    def update(self, *, provider: str, model: str = "", base_url: str = "",
               api_key: Optional[str] = None, dim: int = 768) -> dict:
        with self._lock:
            self.provider = (provider or "none").lower()
            self.model    = model or _DEFAULT_MODELS.get(self.provider, "")
            self.base_url = base_url or ""
            self.dim      = int(dim or 768)
            if api_key is not None and api_key.strip():
                self._api_key = api_key.strip()
            elif provider == "none":
                self._api_key = ""
        return self.snapshot()

    # ── embed dispatch ──────────────────────────────────────────
    def embed(self, text: str) -> List[float]:
        """Embed a single string. Raises RuntimeError on misconfig or upstream error."""
        with self._lock:
            provider, model, base_url, key, dim = (
                self.provider, self.model, self.base_url, self._api_key, self.dim
            )
        if provider == "none":
            raise RuntimeError("No embedding provider configured.")
        if provider == "openai":
            vec = _openai_embed(text, model or _DEFAULT_MODELS["openai"], key)
        elif provider == "voyage":
            vec = _voyage_embed(text, model or _DEFAULT_MODELS["voyage"], key)
        elif provider == "cohere":
            vec = _cohere_embed(text, model or _DEFAULT_MODELS["cohere"], key)
        elif provider == "ollama":
            vec = _ollama_embed(text, model or _DEFAULT_MODELS["ollama"],
                                base_url or "http://localhost:11434")
        else:
            raise RuntimeError(f"Unknown provider: {provider}")
        # safety: pad/truncate to configured dim if shape mismatch (lets users
        # plug in a different model without crashing the whole pipeline)
        if len(vec) != dim:
            if len(vec) < dim:
                vec = vec + [0.0] * (dim - len(vec))
            else:
                vec = vec[:dim]
        return vec


# ── provider drivers (urllib only — no extra deps) ──────────────────────────

def _http_post_json(url: str, headers: dict, body: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={**headers, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url} → HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"{url} → URL error: {e.reason}")


def _openai_embed(text: str, model: str, key: str) -> List[float]:
    if not key:
        raise RuntimeError("OpenAI provider needs an API key.")
    j = _http_post_json(
        "https://api.openai.com/v1/embeddings",
        {"Authorization": f"Bearer {key}"},
        {"input": text, "model": model},
    )
    return j["data"][0]["embedding"]


def _voyage_embed(text: str, model: str, key: str) -> List[float]:
    if not key:
        raise RuntimeError("Voyage provider needs an API key.")
    j = _http_post_json(
        "https://api.voyageai.com/v1/embeddings",
        {"Authorization": f"Bearer {key}"},
        {"input": [text], "model": model},
    )
    return j["data"][0]["embedding"]


def _cohere_embed(text: str, model: str, key: str) -> List[float]:
    if not key:
        raise RuntimeError("Cohere provider needs an API key.")
    j = _http_post_json(
        "https://api.cohere.ai/v1/embed",
        {"Authorization": f"Bearer {key}"},
        {"texts": [text], "model": model, "input_type": "search_document"},
    )
    return j["embeddings"][0]


def _ollama_embed(text: str, model: str, base_url: str) -> List[float]:
    j = _http_post_json(
        f"{base_url.rstrip('/')}/api/embeddings",
        {},
        {"model": model, "prompt": text},
    )
    return j["embedding"]


EMBEDDING = EmbeddingProvider()
