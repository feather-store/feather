"""Pluggable embedding service.

Stores config in-memory (single process). On every restart the operator
must re-set credentials via PUT /v1/admin/embedding-config.
The API key itself is never echoed back to clients.

Supported providers:
  openai        — OpenAI embeddings
  azure_openai  — Azure OpenAI (custom endpoint + deployment + api_version)
  gemini        — Google AI / Gemini (gemini-embedding-001 etc.)
  voyage        — Voyage AI embeddings
  cohere        — Cohere embed
  ollama        — local Ollama HTTP API (no key required)
  mock          — deterministic hash vectors, FEATHER_DEV_MODE only (tests/benchmarks)
  none          — no embedding service configured (caller must pass a vector)

Transport (0.19): pooled httpx clients (one TLS handshake per connection, not
per call), a sync path for the bulk importer and an async path for
/ingest_text, provider-side batching, a small LRU cache, and an asyncio
micro-batcher (EmbedBatcher) that coalesces concurrent ingest requests into one
provider call.

Dimensions: vectors are NO LONGER silently padded or truncated to the
configured dim. OpenAI/Azure text-embedding-3-* and Gemini are asked for the
configured dim natively (Matryoshka truncation done by the provider, which is
the correct way to shorten them). Any other mismatch raises RuntimeError
naming both numbers — zero-padding or chopping a vector corrupts it silently.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections import OrderedDict
from threading import Lock
from typing import Callable, Dict, List, Optional, Tuple

import httpx


# Default model per provider.
_DEFAULT_MODELS = {
    "openai":       "text-embedding-3-small",   # native 1536 dim, `dimensions` supported
    "azure_openai": "text-embedding-3-small",   # deployment must exist
    "gemini":       "gemini-embedding-001",     # native 3072, outputDimensionality supported
    "voyage":       "voyage-3",                 # native 1024 dim
    "cohere":       "embed-english-v3.0",       # native 1024 dim
    "ollama":       "nomic-embed-text",         # native 768 dim
    "mock":         "mock-hash",
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
        {"name": "gemini-embedding-001",       "dim": 768, "label": "Gemini embedding-001 (768d requested, current)"},
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

_TIMEOUT_S = float(os.getenv("FEATHER_EMBED_TIMEOUT_S", "30"))
_MAX_PER_REQUEST = 96          # Cohere's cap; safe for every provider
_DEV_MODE = os.getenv("FEATHER_DEV_MODE", "").strip().lower() in ("1", "true", "yes")


def _limits() -> httpx.Limits:
    return httpx.Limits(max_connections=64, max_keepalive_connections=32)


# ─────────────────────────────────────────────────────────────────────────────
# Request building (shared by the sync and async paths)
# ─────────────────────────────────────────────────────────────────────────────
class _Cfg:
    __slots__ = ("provider", "model", "base_url", "deployment", "api_version", "key", "dim")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def cache_key(self) -> tuple:
        return (self.provider, self.model, self.base_url, self.deployment, self.dim)


def _extract(j, path, provider: str):
    """Walk `path` into a response, raising RuntimeError (the documented embed()
    contract) instead of KeyError/IndexError on an error-shaped payload."""
    cur = j
    try:
        for p in path:
            cur = cur[p]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"{provider}: unexpected embedding response shape: "
                           f"{json.dumps(j)[:300]}")
    return cur


def _vec(v, provider: str, j) -> List[float]:
    if not isinstance(v, list) or not v:
        raise RuntimeError(f"{provider}: embedding response missing vector: {json.dumps(j)[:300]}")
    return v


def _build(cfg: _Cfg, texts: List[str]) -> Tuple[str, dict, dict, Callable[[dict], List[List[float]]]]:
    """(url, headers, json body, parse(response_json) -> vectors) for one batch."""
    p, n = cfg.provider, len(texts)
    model = cfg.model or _DEFAULT_MODELS.get(p, "")

    if p in ("openai", "azure_openai"):
        if not cfg.key:
            raise RuntimeError(f"{p} provider needs an API key.")
        body = {"input": texts}
        if p == "openai":
            url, headers = "https://api.openai.com/v1/embeddings", {"Authorization": f"Bearer {cfg.key}"}
            body["model"] = model
        else:
            if not cfg.base_url:
                raise RuntimeError("Azure OpenAI needs Base URL (https://<resource>.openai.azure.com).")
            dep = cfg.deployment or model
            if not dep:
                raise RuntimeError("Azure OpenAI needs the deployment name.")
            url = (f"{cfg.base_url.rstrip('/')}/openai/deployments/{dep}/embeddings"
                   f"?api-version={cfg.api_version or '2024-02-01'}")
            headers = {"api-key": cfg.key}
        if "text-embedding-3" in (cfg.deployment or model):
            body["dimensions"] = cfg.dim            # native Matryoshka shortening

        def parse(j):
            data = sorted(_extract(j, ["data"], p), key=lambda d: d.get("index", 0))
            return [_vec(d.get("embedding"), p, j) for d in data]
        return url, headers, body, parse

    if p == "gemini":
        if not cfg.key:
            raise RuntimeError("Gemini provider needs a Google AI API key.")
        base = (cfg.base_url or "https://generativelanguage.googleapis.com").rstrip("/")
        url = f"{base}/v1beta/models/{model}:batchEmbedContents"
        reqs = []
        for t in texts:
            r = {"model": f"models/{model}", "content": {"parts": [{"text": t}]}}
            if model.startswith("gemini-embedding"):
                r["outputDimensionality"] = cfg.dim
            reqs.append(r)

        def parse(j):
            return [_vec(_extract(e, ["values"], p), p, j) for e in _extract(j, ["embeddings"], p)]
        return url, {"x-goog-api-key": cfg.key}, {"requests": reqs}, parse

    if p == "voyage":
        if not cfg.key:
            raise RuntimeError("Voyage provider needs an API key.")

        def parse(j):
            return [_vec(d.get("embedding"), p, j) for d in _extract(j, ["data"], p)]
        return ("https://api.voyageai.com/v1/embeddings", {"Authorization": f"Bearer {cfg.key}"},
                {"input": texts, "model": model}, parse)

    if p == "cohere":
        if not cfg.key:
            raise RuntimeError("Cohere provider needs an API key.")

        def parse(j):
            return [_vec(v, p, j) for v in _extract(j, ["embeddings"], p)]
        return ("https://api.cohere.ai/v1/embed", {"Authorization": f"Bearer {cfg.key}"},
                {"texts": texts, "model": model, "input_type": "search_document"}, parse)

    if p == "ollama":
        base = (cfg.base_url or "http://localhost:11434").rstrip("/")

        def parse(j):
            return [_vec(v, p, j) for v in _extract(j, ["embeddings"], p)]
        return f"{base}/api/embed", {}, {"model": model, "input": texts}, parse

    raise RuntimeError(f"Unknown provider: {p}")


def _ollama_legacy(cfg: _Cfg, text: str):
    """Pre-0.3 Ollama has no batch /api/embed; fall back to /api/embeddings."""
    base = (cfg.base_url or "http://localhost:11434").rstrip("/")
    model = cfg.model or _DEFAULT_MODELS["ollama"]
    return (f"{base}/api/embeddings", {}, {"model": model, "prompt": text},
            lambda j: [_vec(_extract(j, ["embedding"], "ollama"), "ollama", j)])


def _mock_vectors(cfg: _Cfg, texts: List[str]) -> List[List[float]]:
    out = []
    for t in texts:
        seed = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:8], "little")
        # cheap deterministic pseudo-random vector (xorshift), non-zero
        v, x = [], seed or 1
        for _ in range(cfg.dim):
            x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF; x ^= x >> 7; x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
            v.append(((x & 0xFFFF) / 32768.0) - 1.0)
        out.append(v)
    return out


def _check_dims(cfg: _Cfg, vecs: List[List[float]]) -> List[List[float]]:
    for v in vecs:
        if len(v) != cfg.dim:
            raise RuntimeError(
                f"{cfg.provider}/{cfg.model or _DEFAULT_MODELS.get(cfg.provider, '')} returned "
                f"{len(v)}-dim vectors but the configured dim is {cfg.dim}. Set the embedding "
                f"config's dim to {len(v)} (or choose a model that produces {cfg.dim}). "
                f"Vectors are never padded or truncated: that silently corrupts them.")
    return vecs


def _http_error(url: str, e: Exception) -> RuntimeError:
    if isinstance(e, httpx.HTTPStatusError):
        return RuntimeError(f"{url} -> HTTP {e.response.status_code}: {e.response.text[:300]}")
    return RuntimeError(f"{url} -> {type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Provider (config + sync/async embedding + cache)
# ─────────────────────────────────────────────────────────────────────────────
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
        self._mock_delay_s = float(os.getenv("FEATHER_EMBED_MOCK_DELAY_MS", "0")) / 1000.0

        self._cache_cap = max(0, int(os.getenv("FEATHER_EMBED_CACHE", "2048")))
        self._cache: "OrderedDict[tuple, List[float]]" = OrderedDict()
        self._cache_lock = Lock()
        self.stats = {"calls": 0, "texts": 0, "cache_hits": 0}

        self._client: Optional[httpx.Client] = None
        self._aclient: Optional[httpx.AsyncClient] = None
        self._aclient_loop = None

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

    def _cfg(self) -> _Cfg:
        with self._lock:
            cfg = _Cfg(provider=self.provider, model=self.model, base_url=self.base_url,
                       deployment=self.deployment, api_version=self.api_version,
                       key=self._api_key, dim=self.dim)
        if cfg.provider == "none":
            raise RuntimeError("No embedding provider configured.")
        if cfg.provider == "mock" and not _DEV_MODE:
            raise RuntimeError("The mock embedding provider is only available with FEATHER_DEV_MODE=1.")
        return cfg

    # ── cache ───────────────────────────────────────────────────
    def _cache_get(self, key: tuple) -> Optional[List[float]]:
        if not self._cache_cap:
            return None
        with self._cache_lock:
            v = self._cache.get(key)
            if v is not None:
                self._cache.move_to_end(key)
                self.stats["cache_hits"] += 1
            return v

    def _cache_put(self, key: tuple, v: List[float]) -> None:
        if not self._cache_cap:
            return
        with self._cache_lock:
            self._cache[key] = v
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_cap:
                self._cache.popitem(last=False)

    @staticmethod
    def _key(cfg: _Cfg, text: str) -> tuple:
        return cfg.cache_key() + (hashlib.sha256(text.encode("utf-8")).digest(),)

    def _split_cached(self, cfg: _Cfg, texts: List[str]):
        out: List[Optional[List[float]]] = [None] * len(texts)
        missing: List[int] = []
        for i, t in enumerate(texts):
            hit = self._cache_get(self._key(cfg, t))
            if hit is None:
                missing.append(i)
            else:
                out[i] = hit
        return out, missing

    # ── sync path (bulk import thread pool) ─────────────────────
    def _sync_client(self) -> httpx.Client:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = httpx.Client(timeout=_TIMEOUT_S, limits=_limits())
        return self._client

    def _post_sync(self, url, headers, body, parse):
        try:
            r = self._sync_client().post(url, headers=headers, json=body)
            r.raise_for_status()
            return parse(r.json())
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001
            raise _http_error(url, e)

    def _provider_batch_sync(self, cfg: _Cfg, texts: List[str]) -> List[List[float]]:
        self.stats["calls"] += 1
        self.stats["texts"] += len(texts)
        if cfg.provider == "mock":
            if self._mock_delay_s:
                time.sleep(self._mock_delay_s)
            return _mock_vectors(cfg, texts)
        url, headers, body, parse = _build(cfg, texts)
        try:
            return self._post_sync(url, headers, body, parse)
        except RuntimeError as e:
            if cfg.provider == "ollama" and "HTTP 404" in str(e):
                return [self._post_sync(*_ollama_legacy(cfg, t))[0] for t in texts]
            raise

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """Embed several strings (sync), batching provider calls. Raises RuntimeError."""
        cfg = self._cfg()
        out, missing = self._split_cached(cfg, texts)
        for s in range(0, len(missing), _MAX_PER_REQUEST):
            idx = missing[s:s + _MAX_PER_REQUEST]
            vecs = _check_dims(cfg, self._provider_batch_sync(cfg, [texts[i] for i in idx]))
            if len(vecs) != len(idx):
                raise RuntimeError(f"{cfg.provider}: asked for {len(idx)} embeddings, got {len(vecs)}")
            for i, v in zip(idx, vecs):
                out[i] = v
                self._cache_put(self._key(cfg, texts[i]), v)
        return out  # type: ignore[return-value]

    def embed(self, text: str) -> List[float]:
        """Embed a single string. Raises RuntimeError on misconfig or upstream error."""
        return self.embed_many([text])[0]

    # ── async path (/ingest_text) ───────────────────────────────
    def _async_client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._aclient is None or self._aclient_loop is not loop:
            self._aclient = httpx.AsyncClient(timeout=_TIMEOUT_S, limits=_limits())
            self._aclient_loop = loop
        return self._aclient

    async def _post_async(self, url, headers, body, parse):
        try:
            r = await self._async_client().post(url, headers=headers, json=body)
            r.raise_for_status()
            return parse(r.json())
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001
            raise _http_error(url, e)

    async def _provider_batch_async(self, cfg: _Cfg, texts: List[str]) -> List[List[float]]:
        self.stats["calls"] += 1
        self.stats["texts"] += len(texts)
        if cfg.provider == "mock":
            if self._mock_delay_s:
                await asyncio.sleep(self._mock_delay_s)
            return _mock_vectors(cfg, texts)
        url, headers, body, parse = _build(cfg, texts)
        try:
            return await self._post_async(url, headers, body, parse)
        except RuntimeError as e:
            if cfg.provider == "ollama" and "HTTP 404" in str(e):
                res = await asyncio.gather(*[self._post_async(*_ollama_legacy(cfg, t)) for t in texts])
                return [r[0] for r in res]
            raise

    async def aembed_many(self, texts: List[str]) -> List[List[float]]:
        cfg = self._cfg()
        out, missing = self._split_cached(cfg, texts)
        chunks = [missing[s:s + _MAX_PER_REQUEST] for s in range(0, len(missing), _MAX_PER_REQUEST)]
        results = await asyncio.gather(
            *[self._provider_batch_async(cfg, [texts[i] for i in idx]) for idx in chunks])
        for idx, vecs in zip(chunks, results):
            _check_dims(cfg, vecs)
            if len(vecs) != len(idx):
                raise RuntimeError(f"{cfg.provider}: asked for {len(idx)} embeddings, got {len(vecs)}")
            for i, v in zip(idx, vecs):
                out[i] = v
                self._cache_put(self._key(cfg, texts[i]), v)
        return out  # type: ignore[return-value]

    async def aclose(self) -> None:
        if self._aclient is not None:
            try:
                await self._aclient.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._aclient = None
        if self._client is not None:
            self._client.close()
            self._client = None


# ─────────────────────────────────────────────────────────────────────────────
# Micro-batcher: many concurrent /ingest_text calls -> one provider request
# ─────────────────────────────────────────────────────────────────────────────
class EmbedBatcher:
    """Collects texts for up to FEATHER_EMBED_BATCH_WINDOW_MS (default 5 ms) or
    FEATHER_EMBED_BATCH_MAX texts (default 64), embeds them in one provider call
    and resolves each caller's future. A failed batch fails every caller in it.
    Awaiting callers hold no worker thread, so slow providers can't starve the
    thread pool that searches run on."""

    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider
        self.window_s = max(0.0, float(os.getenv("FEATHER_EMBED_BATCH_WINDOW_MS", "5")) / 1000.0)
        self.max_batch = max(1, int(os.getenv("FEATHER_EMBED_BATCH_MAX", "64")))
        self._queue: Optional[asyncio.Queue] = None
        self._task: Optional[asyncio.Task] = None
        self._inflight: set = set()

    def start(self) -> None:
        if self._task is None:
            self._queue = asyncio.Queue()
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None
        if self._inflight:
            await asyncio.gather(*self._inflight, return_exceptions=True)

    async def embed(self, text: str) -> List[float]:
        if self._task is None:                    # not started (e.g. tests): direct call
            return (await self.provider.aembed_many([text]))[0]
        fut = asyncio.get_running_loop().create_future()
        await self._queue.put((text, fut))
        return await fut

    async def _run(self) -> None:
        q = self._queue
        while True:
            first = await q.get()
            batch = [first]
            deadline = asyncio.get_running_loop().time() + self.window_s
            while len(batch) < self.max_batch:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(q.get(), timeout))
                except asyncio.TimeoutError:
                    break
            # Dispatch without awaiting, so the next batch can start collecting
            # while this one is in flight.
            t = asyncio.get_running_loop().create_task(self._dispatch(batch))
            self._inflight.add(t)
            t.add_done_callback(self._inflight.discard)

    async def _dispatch(self, batch) -> None:
        texts = [t for t, _ in batch]
        try:
            vecs = await self.provider.aembed_many(texts)
            for (_, fut), v in zip(batch, vecs):
                if not fut.done():
                    fut.set_result(v)
        except Exception as e:  # noqa: BLE001
            for _, fut in batch:
                if not fut.done():
                    fut.set_exception(e if isinstance(e, RuntimeError) else RuntimeError(str(e)))


EMBEDDING = EmbeddingProvider()
EMBED_BATCHER = EmbedBatcher(EMBEDDING)
