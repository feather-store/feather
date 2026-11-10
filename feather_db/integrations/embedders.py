"""
Feather DB — client-side embedders for the MCP server
=====================================================
A tiny, dependency-free (urllib only) embedder factory so `feather-serve` can
turn text into vectors locally — needed by the *remote* MCP backend (which calls
the Cloud API's plain vector endpoints) and handy for the local backend too.

    from feather_db.integrations.embedders import make_embedder
    embed = make_embedder("gemini", api_key=KEY, dim=768)
    vec = embed("some text")          # -> list[float], length == dim

Providers: gemini, openai, voyage, cohere, ollama, local, hash (deterministic,
no key). `local` runs a small ONNX model on the CPU (`pip install fastembed`;
bge-small, 384-d, ~4-10 ms a text against 100-300 ms for an API call, no key,
works offline). Vectors from different models are not comparable: pick one
per database.
Provider embedders keep an LRU of recent texts (`FEATHER_EMBED_CACHE` entries,
default 2048, 0 disables), as the Cloud API does: an agent re-embeds the same
query and the same memory text turn after turn, and each miss is a 100-300 ms
round trip — more than the search it feeds, by two orders of magnitude.
The returned vector is padded/truncated to `dim` so it always matches the
namespace dimension. Gemini `gemini-embedding-001` (the default; Google shut
`text-embedding-004` down) and OpenAI `text-embedding-3-*` are asked for `dim`
natively. Requests reuse their connection (one per thread and host).
Hedging (a duplicate request at the p95) was measured and not kept: 450
calls, 13 hedges, none won; the multi-second stalls hit the duplicate too.
"""
from __future__ import annotations

import os
import json
import hashlib
import http.client
import threading
import weakref
import urllib.parse
import urllib.request
import urllib.error
from collections import OrderedDict
from threading import Lock
from typing import Callable, List, Optional

_DEFAULT_MODELS = {
    "openai":  "text-embedding-3-small",
    "gemini":  "gemini-embedding-001",   # text-embedding-004 answers 404
    "voyage":  "voyage-3-lite",
    "cohere":  "embed-english-v3.0",
    "ollama":  "nomic-embed-text",
    "local":   "BAAI/bge-small-en-v1.5",
}


def _http_post_json(url: str, headers: dict, body: dict, timeout: float = 30.0) -> dict:
    """POST JSON, over a kept-alive connection (one per thread and host).

    A new HTTPS connection per call cost every embed its TCP + TLS
    handshake: gemini-embedding-001 from one client, 60 calls each, p50
    602 -> 488 ms and p95 776 -> 535 ms with the connection reused, and the
    only multi-second calls (3.0 s, 6.1 s) were on fresh connections. A
    configured proxy keeps the urllib path, which honours it."""
    data = json.dumps(body).encode("utf-8")
    headers = {**headers, "Content-Type": "application/json"}
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or \
            urllib.request.getproxies().get(parts.scheme):
        return _urllib_post(url, headers, data, timeout)
    path = parts.path + (f"?{parts.query}" if parts.query else "")
    for attempt in (0, 1):
        conn, reused = _connection(parts.scheme, parts.netloc, timeout)
        try:
            conn.request("POST", path, data, headers)
            resp = conn.getresponse()
            raw = resp.read()
        except (http.client.HTTPException, OSError) as e:
            _drop_connection(parts.scheme, parts.netloc)
            # A kept-alive connection the server closed while idle fails
            # on first use: retry once on a fresh one. A fresh one failing
            # is a real error.
            if reused and attempt == 0 and not isinstance(e, TimeoutError):
                continue
            raise RuntimeError(f"{url} -> {e}")
        if resp.will_close:
            _drop_connection(parts.scheme, parts.netloc)
        if resp.status >= 400:
            raise RuntimeError(
                f"{url} -> HTTP {resp.status}: {raw.decode('utf-8', 'replace')[:300]}")
        return json.loads(raw.decode("utf-8"))
    raise AssertionError("unreachable")


_conns = threading.local()


def _connection(scheme: str, netloc: str, timeout: float):
    """(this thread's connection to scheme://netloc, whether it was reused)."""
    pool = getattr(_conns, "pool", None)
    if pool is None:
        pool = _conns.pool = {}
    conn = pool.get((scheme, netloc))
    if conn is not None:
        return conn, True
    cls = (http.client.HTTPSConnection if scheme == "https"
           else http.client.HTTPConnection)
    conn = pool[(scheme, netloc)] = cls(netloc, timeout=timeout)
    return conn, False


def _drop_connection(scheme: str, netloc: str) -> None:
    conn = getattr(_conns, "pool", {}).pop((scheme, netloc), None)
    if conn is not None:
        conn.close()


def _urllib_post(url: str, headers: dict, data: bytes, timeout: float) -> dict:
    req = urllib.request.Request(url, data=data, headers=headers)
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


def _unit(vec: List[float]) -> List[float]:
    n = sum(x * x for x in vec) ** 0.5
    return [x / n for x in vec] if n > 0 else vec


def _fit(vec: List[float], dim: int) -> List[float]:
    """`vec` at `dim`: padded with zeros, or cut and renormalised. A cut
    vector is not unit length (Gemini's 768-d prefix is ~0.59), and search
    ranks by L2 distance, which is cosine order only for one length."""
    if len(vec) == dim:
        return [float(x) for x in vec]
    if len(vec) > dim:
        return _unit([float(x) for x in vec[:dim]])
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
    # Matryoshka: the API's d-dim output is exactly the first d values of
    # the 3072-d vector (checked live: cosine 1.000000, same norm ~0.59, not
    # renormalised), so it is renormalised here, as _fit renormalises a cut.
    # Asking for d sends 4x less JSON at 768.
    cut = model.startswith("gemini-embedding") and 128 <= dim < 3072
    if cut:
        body["outputDimensionality"] = dim
    j = _http_post_json(url, {"x-goog-api-key": key}, body)
    v = _dig(j, ["embedding", "values"], "gemini")
    return _unit([float(x) for x in v]) if cut else v      # the prefix is ~0.59 long


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


def _local_embedder(model: str, dim: int) -> Callable[[str], List[float]]:
    try:
        from fastembed import TextEmbedding
    except ImportError:
        raise RuntimeError("local embedder needs fastembed: pip install fastembed")
    m = TextEmbedding(model)            # loads now, so a bad name fails at startup

    def embed(text: str) -> List[float]:
        return _fit(next(iter(m.embed([text]))).tolist(), dim)
    return embed


def default_embedder(dim: int = 768, *, provider: Optional[str] = None,
                     model: Optional[str] = None):
    """Build an embedder from the environment, or return None if none is set.

    Reads `FEATHER_EMBED_PROVIDER` and `FEATHER_EMBED_MODEL`. Returns None when
    no provider is configured, so a caller can fall back to keyword search.

    If a provider IS configured but cannot be built — missing key, unknown name
    — this raises instead of returning None. A silent fall back to keyword
    search would be the worst outcome: BM25 does no stemming, so a paraphrased
    query returns "nothing found", and the caller cannot tell a
    misconfiguration from a memory the agent genuinely never had. Better to
    fail at startup than to look like amnesia at runtime.
    """
    provider = provider or os.getenv("FEATHER_EMBED_PROVIDER")
    if not provider:
        return None
    return make_embedder(provider, model=model or os.getenv("FEATHER_EMBED_MODEL"),
                         dim=dim)


def make_embedder(provider: str, model: Optional[str] = None,
                  api_key: Optional[str] = None, dim: int = 768,
                  base_url: Optional[str] = None) -> Callable[[str], List[float]]:
    """Return an `embed(text) -> list[float]` of length `dim` for the provider.
    api_key falls back to FEATHER_EMBED_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY etc."""
    provider = (provider or "hash").lower()
    if provider in ("hash", "none", ""):
        return _hash_embedder(dim)
    if provider == "local":
        return _cached(_local_embedder(model or _DEFAULT_MODELS["local"], dim),
                       int(os.getenv("FEATHER_EMBED_CACHE", "2048")))
    if provider not in _DRIVERS:
        raise ValueError(f"unknown embed provider: {provider} "
                         f"(choose from {', '.join(_DRIVERS)}, local or hash)")
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
    return _cached(embed, int(os.getenv("FEATHER_EMBED_CACHE", "2048")))


def _cached(embed: Callable[[str], List[float]], cap: int) -> Callable[[str], List[float]]:
    """Wrap `embed` in a thread-safe LRU keyed by the text's SHA-256.

    One wrapper serves one (provider, model, dim, base_url), so the text is the
    whole key. Failures are not cached. Each hit returns a copy, so a caller
    that edits its vector in place cannot change what the next caller gets.
    `embed.stats` counts calls and hits."""
    if cap <= 0:
        return embed
    cache: "OrderedDict[bytes, List[float]]" = OrderedDict()
    lock = Lock()
    stats = {"calls": 0, "cache_hits": 0}

    def cached(text: str) -> List[float]:
        k = hashlib.sha256((text or "").encode("utf-8")).digest()
        with lock:
            stats["calls"] += 1
            v = cache.get(k)
            if v is not None:
                cache.move_to_end(k)
                stats["cache_hits"] += 1
                return list(v)
        v = embed(text)                       # outside the lock: never serialise I/O
        with lock:
            cache[k] = v
            cache.move_to_end(k)
            while len(cache) > cap:
                cache.popitem(last=False)
        return list(v)

    def clear() -> None:
        with lock:
            cache.clear()

    cached.stats = stats
    cached.clear = clear
    _CACHES.add(cached)
    return cached


# Every cache this process holds, so an erasure can drop them all (weak: a
# cache goes with its embedder).
_CACHES: "weakref.WeakSet" = weakref.WeakSet()


def clear_caches() -> int:
    """Empty every embedding cache in this process; returns how many.

    A cache holds a vector per text, keyed by the text's SHA-256: an embedding
    of what an erasure removed. DB.erase() and RemoteDB.erase() call this."""
    caches = list(_CACHES)
    for c in caches:
        c.clear()
    return len(caches)
