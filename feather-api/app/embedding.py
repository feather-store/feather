"""Pluggable embedding service.

Stores config in-memory (single process). On every restart the operator
must re-set credentials via PUT /v1/admin/embedding-config.
The API key itself is never echoed back to clients.

Supported providers:
  openai        — OpenAI embeddings
  azure_openai  — Azure OpenAI (custom endpoint + deployment + api_version)
  gemini        — Google AI / Gemini (text-embedding-004 etc.)
  voyage        — Voyage AI embeddings
  cohere        — Cohere embed
  ollama        — local Ollama HTTP API (no key required)
  none          — no embedding service configured (caller must pass a vector)
"""
from __future__ import annotations
import os
import json
from typing import List, Optional
from threading import Lock

import urllib.request
import urllib.error


# Default model per provider — chosen for compatibility with dim=768 where
# possible.  All models can be padded/truncated to namespace dim by the caller.
_DEFAULT_MODELS = {
    "openai":       "text-embedding-3-small",   # native 1536 dim
    "azure_openai": "text-embedding-3-small",   # deployment must exist
    "gemini":       "gemini-embedding-001",     # native 768 dim (text-embedding-004 deprecated)
    "voyage":       "voyage-3",                 # native 1024 dim
    "cohere":       "embed-english-v3.0",       # native 1024 dim
    "ollama":       "nomic-embed-text",         # native 768 dim
}

# Curated lists used by the dashboard's model dropdown. Keep small + current.
SUPPORTED_MODELS: dict = {
    "openai": [
        {"name": "text-embedding-3-small", "dim": 1536, "label": "OpenAI 3-small (1536d, default)"},
        {"name": "text-embedding-3-large", "dim": 3072, "label": "OpenAI 3-large (3072d, highest quality)"},
        {"name": "text-embedding-ada-002", "dim": 1536, "label": "OpenAI ada-002 (legacy, 1536d)"},
    ],
    "azure_openai": [
        {"name": "text-embedding-3-small", "dim": 1536, "label": "Azure OpenAI 3-small (1536d)"},
        {"name": "text-embedding-3-large", "dim": 3072, "label": "Azure OpenAI 3-large (3072d)"},
        {"name": "text-embedding-ada-002", "dim": 1536, "label": "Azure OpenAI ada-002 (1536d)"},
    ],
    "gemini": [
        {"name": "gemini-embedding-001",       "dim": 768, "label": "Gemini embedding-001 (768d, current)"},
        {"name": "gemini-embedding-exp-03-07", "dim": 768, "label": "Gemini embedding exp (768d, experimental)"},
        {"name": "text-embedding-004",         "dim": 768, "label": "Gemini text-embedding-004 (768d, deprecated)"},
        {"name": "embedding-001",              "dim": 768, "label": "Gemini embedding-001 (768d, legacy)"},
    ],
    "voyage": [
        {"name": "voyage-3",         "dim": 1024, "label": "Voyage 3 (1024d, balanced)"},
        {"name": "voyage-3-lite",    "dim": 512,  "label": "Voyage 3 Lite (512d, cheap)"},
        {"name": "voyage-large-2",   "dim": 1536, "label": "Voyage Large 2 (1536d, highest)"},
        {"name": "voyage-code-2",    "dim": 1536, "label": "Voyage Code 2 (1536d, code-tuned)"},
    ],
    "cohere": [
        {"name": "embed-english-v3.0",       "dim": 1024, "label": "Cohere embed-english-v3 (1024d)"},
        {"name": "embed-multilingual-v3.0",  "dim": 1024, "label": "Cohere embed-multilingual-v3 (1024d)"},
        {"name": "embed-english-light-v3.0", "dim": 384,  "label": "Cohere embed-english-light-v3 (384d)"},
    ],
    "ollama": [
        {"name": "nomic-embed-text",  "dim": 768,  "label": "Ollama nomic-embed-text (768d, default)"},
        {"name": "mxbai-embed-large", "dim": 1024, "label": "Ollama mxbai-embed-large (1024d)"},
        {"name": "all-minilm",        "dim": 384,  "label": "Ollama all-minilm (384d, smallest)"},
    ],
    "none": [],
}


class EmbeddingProvider:
    def __init__(self):
        self._lock = Lock()
        # bootstrap from env if present
        self.provider:    str = os.getenv("FEATHER_EMBED_PROVIDER", "none")
        self.model:       str = os.getenv("FEATHER_EMBED_MODEL", _DEFAULT_MODELS.get(self.provider, ""))
        self.base_url:    str = os.getenv("FEATHER_EMBED_BASE_URL", "")
        self.deployment:  str = os.getenv("FEATHER_EMBED_DEPLOYMENT", "")
        self.api_version: str = os.getenv("FEATHER_EMBED_API_VERSION", "2024-02-01")
        self._api_key:    str = os.getenv("FEATHER_EMBED_API_KEY", "")
        self.dim:         int = int(os.getenv("FEATHER_EMBED_DIM", "768"))

    # ── public state introspection ──────────────────────────────
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "provider":     self.provider,
                "model":        self.model,
                "base_url":     self.base_url,
                "deployment":   self.deployment,
                "api_version":  self.api_version,
                "api_key_set":  bool(self._api_key),
                "dim":          self.dim,
            }

    def update(self, *, provider: str, model: str = "", base_url: str = "",
               deployment: str = "", api_version: str = "",
               api_key: Optional[str] = None, dim: int = 768) -> dict:
        with self._lock:
            self.provider    = (provider or "none").lower()
            self.model       = model or _DEFAULT_MODELS.get(self.provider, "")
            self.base_url    = base_url or ""
            self.deployment  = deployment or ""
            self.api_version = api_version or self.api_version or "2024-02-01"
            self.dim         = int(dim or 768)
            if api_key is not None and api_key.strip():
                self._api_key = api_key.strip()
            elif provider == "none":
                self._api_key = ""
        return self.snapshot()

    # ── embed dispatch ──────────────────────────────────────────
    def embed(self, text: str) -> List[float]:
        """Embed a single string. Raises RuntimeError on misconfig or upstream error."""
        with self._lock:
            provider     = self.provider
            model        = self.model
            base_url     = self.base_url
            deployment   = self.deployment
            api_version  = self.api_version
            key          = self._api_key
            dim          = self.dim

        if provider == "none":
            raise RuntimeError("No embedding provider configured.")
        if provider == "openai":
            vec = _openai_embed(text, model or _DEFAULT_MODELS["openai"], key)
        elif provider == "azure_openai":
            vec = _azure_openai_embed(text, deployment or model or _DEFAULT_MODELS["azure_openai"],
                                       base_url, api_version or "2024-02-01", key)
        elif provider == "gemini":
            vec = _gemini_embed(text, model or _DEFAULT_MODELS["gemini"], key, base_url)
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


def _extract_vec(j: dict, path, provider: str) -> List[float]:
    """Walk `path` (keys/indices) into the response, raising RuntimeError —
    the documented embed() contract — instead of KeyError/IndexError when the
    upstream returns an error-shaped or otherwise unexpected payload."""
    cur = j
    try:
        for p in path:
            cur = cur[p]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"{provider}: unexpected embedding response shape: "
                           f"{json.dumps(j)[:300]}")
    if not isinstance(cur, list) or not cur:
        raise RuntimeError(f"{provider}: embedding response missing vector: "
                           f"{json.dumps(j)[:300]}")
    return cur


def _openai_embed(text: str, model: str, key: str) -> List[float]:
    if not key:
        raise RuntimeError("OpenAI provider needs an API key.")
    j = _http_post_json(
        "https://api.openai.com/v1/embeddings",
        {"Authorization": f"Bearer {key}"},
        {"input": text, "model": model},
    )
    return _extract_vec(j, ["data", 0, "embedding"], "openai")


def _azure_openai_embed(text: str, deployment: str, endpoint: str,
                         api_version: str, key: str) -> List[float]:
    """Azure OpenAI:  POST {endpoint}/openai/deployments/{deployment}/embeddings?api-version=...
    Header `api-key` (not Bearer).  Endpoint shape: https://<resource>.openai.azure.com
    """
    if not key:
        raise RuntimeError("Azure OpenAI needs an API key (api-key header).")
    if not endpoint:
        raise RuntimeError("Azure OpenAI needs Base URL (https://<resource>.openai.azure.com).")
    if not deployment:
        raise RuntimeError("Azure OpenAI needs the deployment name.")
    base = endpoint.rstrip("/")
    url  = f"{base}/openai/deployments/{deployment}/embeddings?api-version={api_version}"
    j = _http_post_json(url, {"api-key": key}, {"input": text})
    return _extract_vec(j, ["data", 0, "embedding"], "azure_openai")


def _gemini_embed(text: str, model: str, key: str, base_url: str = "") -> List[float]:
    """Gemini (Google AI):  POST {base}/v1beta/models/{model}:embedContent
    Header: x-goog-api-key.  Default base: https://generativelanguage.googleapis.com
    """
    if not key:
        raise RuntimeError("Gemini provider needs a Google AI API key.")
    base = (base_url or "https://generativelanguage.googleapis.com").rstrip("/")
    url  = f"{base}/v1beta/models/{model}:embedContent"
    body = {"content": {"parts": [{"text": text}]}}
    j = _http_post_json(url, {"x-goog-api-key": key}, body)
    # Response shape: { "embedding": { "values": [...] } }
    return _extract_vec(j, ["embedding", "values"], "gemini")


def _voyage_embed(text: str, model: str, key: str) -> List[float]:
    if not key:
        raise RuntimeError("Voyage provider needs an API key.")
    j = _http_post_json(
        "https://api.voyageai.com/v1/embeddings",
        {"Authorization": f"Bearer {key}"},
        {"input": [text], "model": model},
    )
    return _extract_vec(j, ["data", 0, "embedding"], "voyage")


def _cohere_embed(text: str, model: str, key: str) -> List[float]:
    if not key:
        raise RuntimeError("Cohere provider needs an API key.")
    j = _http_post_json(
        "https://api.cohere.ai/v1/embed",
        {"Authorization": f"Bearer {key}"},
        {"texts": [text], "model": model, "input_type": "search_document"},
    )
    return _extract_vec(j, ["embeddings", 0], "cohere")


def _ollama_embed(text: str, model: str, base_url: str) -> List[float]:
    j = _http_post_json(
        f"{base_url.rstrip('/')}/api/embeddings",
        {},
        {"model": model, "prompt": text},
    )
    return _extract_vec(j, ["embedding"], "ollama")


EMBEDDING = EmbeddingProvider()
