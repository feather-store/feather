"""
Feather DB — client-side embedders for the MCP server
=====================================================
A tiny, dependency-free (urllib only) embedder factory so `feather-serve` can
turn text into vectors locally — needed by the *remote* MCP backend (which calls
the Cloud API's plain vector endpoints) and handy for the local backend too.

    from feather_db.integrations.embedders import make_embedder
    embed = make_embedder("gemini", api_key=KEY, dim=768)
    vec = embed("some text")          # -> list[float], length == dim

Providers: gemini, openai, voyage, cohere, ollama, hash (deterministic, no key).
The returned vector is padded/truncated to `dim` so it always matches the
namespace dimension. For an exact native match use Gemini `text-embedding-004`
(768) or OpenAI `text-embedding-3-small` with `dim=768` (it supports a
`dimensions` request).
"""
from __future__ import annotations

import os
import json
import hashlib
import urllib.request
import urllib.error
from typing import Callable, List, Optional

_DEFAULT_MODELS = {
    "openai":  "text-embedding-3-small",
    "gemini":  "text-embedding-004",
    "voyage":  "voyage-3-lite",
    "cohere":  "embed-english-v3.0",
    "ollama":  "nomic-embed-text",
}


def _http_post_json(url: str, headers: dict, body: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={**headers, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url} -> HTTP {e.code}: {e.read().decode('utf-8','replace')[:300]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"{url} -> {e.reason}")


def _dig(j: dict, path, provider: str) -> List[float]:
    cur = j
    try:
        for p in path:
            cur = cur[p]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"{provider}: unexpected embedding response: {json.dumps(j)[:300]}")
    if not isinstance(cur, list) or not cur:
        raise RuntimeError(f"{provider}: no vector in response: {json.dumps(j)[:300]}")
    return cur


def _fit(vec: List[float], dim: int) -> List[float]:
    if len(vec) == dim:
        return [float(x) for x in vec]
    if len(vec) > dim:
        return [float(x) for x in vec[:dim]]
    return [float(x) for x in vec] + [0.0] * (dim - len(vec))


# ── provider drivers ────────────────────────────────────────────────────────
def _openai(text, model, key, dim, base_url):
    base = (base_url or "https://api.openai.com").rstrip("/")
    body = {"input": text, "model": model, "dimensions": dim}   # native 768 support
    j = _http_post_json(f"{base}/v1/embeddings", {"Authorization": f"Bearer {key}"}, body)
    return _dig(j, ["data", 0, "embedding"], "openai")


def _gemini(text, model, key, dim, base_url):
    base = (base_url or "https://generativelanguage.googleapis.com").rstrip("/")
    url = f"{base}/v1beta/models/{model}:embedContent"
    body = {"content": {"parts": [{"text": text}]}}
    j = _http_post_json(url, {"x-goog-api-key": key}, body)
    return _dig(j, ["embedding", "values"], "gemini")


def _voyage(text, model, key, dim, base_url):
    j = _http_post_json("https://api.voyageai.com/v1/embeddings",
                        {"Authorization": f"Bearer {key}"},
                        {"input": [text], "model": model})
    return _dig(j, ["data", 0, "embedding"], "voyage")


def _cohere(text, model, key, dim, base_url):
    j = _http_post_json("https://api.cohere.ai/v1/embed",
                        {"Authorization": f"Bearer {key}"},
                        {"texts": [text], "model": model, "input_type": "search_document"})
    return _dig(j, ["embeddings", 0], "cohere")


def _ollama(text, model, key, dim, base_url):
    base = (base_url or "http://localhost:11434").rstrip("/")
    j = _http_post_json(f"{base}/api/embeddings", {}, {"model": model, "prompt": text})
    return _dig(j, ["embedding"], "ollama")


_DRIVERS = {"openai": _openai, "gemini": _gemini, "voyage": _voyage,
            "cohere": _cohere, "ollama": _ollama}


def _hash_embedder(dim: int) -> Callable[[str], List[float]]:
    def embed(text: str) -> List[float]:
        v = [0.0] * dim
        for tok in (text or "").lower().split():
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v] if n > 0 else v
    return embed


def make_embedder(provider: str, model: Optional[str] = None,
                  api_key: Optional[str] = None, dim: int = 768,
                  base_url: Optional[str] = None) -> Callable[[str], List[float]]:
    """Return an `embed(text) -> list[float]` of length `dim` for the provider.
    api_key falls back to FEATHER_EMBED_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY etc."""
    provider = (provider or "hash").lower()
    if provider in ("hash", "none", ""):
        return _hash_embedder(dim)
    if provider not in _DRIVERS:
        raise ValueError(f"unknown embed provider: {provider} "
                         f"(choose from {', '.join(_DRIVERS)} or hash)")
    model = model or _DEFAULT_MODELS[provider]
    key = (api_key or os.getenv("FEATHER_EMBED_API_KEY") or {
        "openai": os.getenv("OPENAI_API_KEY"),
        "gemini": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        "voyage": os.getenv("VOYAGE_API_KEY"),
        "cohere": os.getenv("COHERE_API_KEY"),
        "ollama": "",
    }.get(provider, "") or "")
    if provider != "ollama" and not key:
        raise RuntimeError(f"{provider} embedder needs an API key "
                           f"(pass api_key= or set the provider env var).")
    driver = _DRIVERS[provider]

    def embed(text: str) -> List[float]:
        return _fit(driver(text, model, key, dim, base_url), dim)
    return embed
