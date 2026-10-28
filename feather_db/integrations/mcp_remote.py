"""
Feather DB — Remote MCP backend
===============================
A drop-in backend for the Feather MCP server that talks to a *deployed* Feather
Cloud API (the `feather-api/` FastAPI service) over HTTP instead of opening a
local `.feather` file. This lets Claude (Desktop or Code) use a shared, hosted
Feather instance as a persona context engine.

Design: embedding happens **client-side** here (in the MCP server process), so we
only call the API's plain vector endpoints — no server-side embedding config is
required, and any dim works as long as the embedder matches the namespace dim.

Tools (persona-oriented subset that maps cleanly to the REST API):
  feather_ingest          Embed + store a piece of context
  feather_recall          Semantic recall (embed query → vector search)
  feather_keyword_recall  BM25 keyword recall (no embedding)
  feather_context_chain   Vector search + n-hop graph expansion
  feather_get_record      Fetch one record by id
  feather_link            Create a typed edge between two records
  feather_stats           Index/health stats for a namespace
  feather_list_namespaces List namespaces on the server

Use a real embedder for quality (pass `embedder=`); the built-in default is a
deterministic hash embedder — fine for wiring/tests, weak for real semantics.
"""
from __future__ import annotations

import json
import time
import random
import uuid
import hashlib
import urllib.request
import urllib.error
from typing import Callable, Optional, List


# ── Typed API errors ─────────────────────────────────────────────────────────
# The API's error body is {"detail", "error", "retryable", ...}. The class says
# what to do: retry (Overloaded, Unavailable), go elsewhere (WrongShard), fix
# the request (RequestError), or report a server bug (ServerError).
class FeatherAPIError(RuntimeError):
    def __init__(self, method: str, path: str, status: int, error: str, detail,
                 retryable: bool, retry_after: Optional[float] = None, body: Optional[dict] = None):
        super().__init__(f"{method} {path} -> HTTP {status} {error}: {str(detail)[:300]}")
        self.status, self.error, self.detail = status, error, detail
        self.retryable, self.retry_after, self.body = retryable, retry_after, body or {}


class OverloadedError(FeatherAPIError):
    """429: the server refused to queue the request. Retry after a pause."""


class UnavailableError(FeatherAPIError):
    """503: temporarily unable (a write failed and is recovering, the
    namespace is loading, the embedding provider is rate-limited)."""


class WrongShardError(FeatherAPIError):
    """421: another shard owns the namespace (`owner_shard`). Retrying here
    cannot succeed; send it to the gateway, or to that shard."""

    @property
    def owner_shard(self) -> Optional[int]:
        return self.body.get("owner_shard")


class RequestError(FeatherAPIError):
    """Any other 4xx: the request itself is wrong. Never retried."""


class ServerError(FeatherAPIError):
    """5xx other than 503: a server-side failure. Never retried — repeating a
    request that crashed the server is how one bad request becomes an outage."""


class ConnectionFailed(RuntimeError):
    """No HTTP answer. `sent`: the request may have reached the server."""

    def __init__(self, msg: str, sent: bool):
        super().__init__(msg)
        self.sent = sent


_ERROR_CLASSES = {429: OverloadedError, 503: UnavailableError, 421: WrongShardError}
_DEFAULT_TYPES = {429: "overloaded", 503: "unavailable", 421: "wrong_shard"}
# Repeating these has the same effect as sending them once (RFC 9110 §9.2.2).
_IDEMPOTENT_METHODS = {"GET", "HEAD", "OPTIONS", "PUT", "DELETE"}


# ── Default deterministic embedder (no API key; for wiring/tests) ───────────
def default_hash_embedder(dim: int) -> Callable[[str], List[float]]:
    def embed(text: str) -> List[float]:
        v = [0.0] * dim
        toks = (text or "").lower().split()
        for tok in toks:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % dim] += 1.0
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v] if n > 0 else v
    return embed


def _gen_id(text: str) -> int:
    # 53-bit id so it round-trips safely through JS/JSON on the dashboard too
    h = hashlib.sha1(f"{text}|{time.time_ns()}".encode()).hexdigest()
    return int(h[:14], 16) & ((1 << 53) - 1)


# ── Tool specs (MCP-style: name/description/parameters/required) ─────────────
REMOTE_TOOL_SPECS = [
    {
        "name": "feather_ingest",
        "description": "Store a piece of context (note, fact, decision) in the persona's "
                       "Feather namespace. The text is embedded and indexed for later recall.",
        "parameters": {
            "text":       {"type": "string", "description": "The content to remember."},
            "namespace":  {"type": "string", "description": "Target namespace (defaults to the server's configured one)."},
            "source":     {"type": "string", "description": "Where this came from (e.g. 'meeting', 'email', 'user')."},
            "entity_id":  {"type": "string", "description": "Optional subject/entity this is about."},
            "importance": {"type": "number", "description": "Relevance weight 0..1 (default 1.0)."},
        },
        "required": ["text"],
    },
    {
        "name": "feather_recall",
        "description": "Semantically recall the most relevant stored context for a query "
                       "(embeds the query and runs vector search, with adaptive-decay scoring).",
        "parameters": {
            "query":     {"type": "string", "description": "What to recall context about."},
            "k":         {"type": "integer", "description": "How many results (default 5)."},
            "namespace": {"type": "string", "description": "Namespace to search."},
        },
        "required": ["query"],
    },
    {
        "name": "feather_keyword_recall",
        "description": "Recall context by keyword (BM25 over content) — exact-term matching, no embedding.",
        "parameters": {
            "query":     {"type": "string", "description": "Keywords to match."},
            "k":         {"type": "integer", "description": "How many results (default 5)."},
            "namespace": {"type": "string", "description": "Namespace to search."},
        },
        "required": ["query"],
    },
    {
        "name": "feather_context_chain",
        "description": "Retrieve relevant context plus its graph-linked neighbours "
                       "(vector search seeds + n-hop BFS expansion).",
        "parameters": {
            "query":     {"type": "string", "description": "Seed query."},
            "k":         {"type": "integer", "description": "Seed results (default 5)."},
            "hops":      {"type": "integer", "description": "Graph expansion hops (default 2)."},
            "namespace": {"type": "string", "description": "Namespace."},
        },
        "required": ["query"],
    },
    {
        "name": "feather_get_record",
        "description": "Fetch one stored record (its content + metadata) by id.",
        "parameters": {
            "id":        {"type": "integer", "description": "Record id."},
            "namespace": {"type": "string", "description": "Namespace."},
        },
        "required": ["id"],
    },
    {
        "name": "feather_link",
        "description": "Create a typed, weighted edge between two records (build the context graph).",
        "parameters": {
            "from_id":   {"type": "integer", "description": "Source record id."},
            "to_id":     {"type": "integer", "description": "Target record id."},
            "rel_type":  {"type": "string", "description": "Relationship type (e.g. 'caused_by', 'supports')."},
            "namespace": {"type": "string", "description": "Namespace."},
        },
        "required": ["from_id", "to_id"],
    },
    {
        "name": "feather_stats",
        "description": "Index/health stats for a namespace: record count, dim, quantization, "
                       "auto-compaction, and per-namespace secondary-index sizes.",
        "parameters": {
            "namespace": {"type": "string", "description": "Namespace."},
        },
        "required": [],
    },
    {
        "name": "feather_list_namespaces",
        "description": "List the namespaces available on the server.",
        "parameters": {},
        "required": [],
    },
]


class RemoteFeatherTools:
    """HTTP-backed tool dispatcher mirroring FeatherTools.handle(name, args) -> str."""

    def __init__(self, api_url: str, api_key: str = "", namespace: str = "default",
                 dim: int = 768, embedder: Optional[Callable[[str], List[float]]] = None,
                 timeout: float = 20.0, max_retries: int = 3, backoff_s: float = 0.25,
                 max_backoff_s: float = 10.0):
        self.base = api_url.rstrip("/")
        self.api_key = api_key
        self.default_ns = namespace
        self.dim = dim
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.max_backoff_s = max_backoff_s
        self._embed = embedder or default_hash_embedder(dim)

    # ── HTTP helper ─────────────────────────────────────────────────────────
    def _req(self, method: str, path: str, body: Optional[dict] = None, *,
             read_only: bool = False, idempotency_key: Optional[str] = None) -> dict:
        """Send a request, retrying only what is safe to repeat.

        A retry happens when the server says it is worth it (429, 503: the
        error body's `retryable`) AND the request is idempotent: an idempotent
        method, a read-only POST (`read_only`), or a write carrying an
        Idempotency-Key, which the server applies once however often it
        arrives. A request that never reached the server (connection refused)
        is always safe to resend. Everything else raises at once: a 4xx cannot
        succeed by repetition, and repeating a 5xx or a request whose answer was
        lost can apply a write twice or pile load on a server already failing.
        """
        idempotent = method in _IDEMPOTENT_METHODS or read_only or bool(idempotency_key)
        attempt = 0
        while True:
            try:
                return self._send(method, path, body, idempotency_key)
            except FeatherAPIError as e:
                if e.status == 501 and idempotency_key and "Idempotency-Key" in str(e.detail):
                    # A server older than keys: send it once without one. It
                    # is then not idempotent, so it is no longer retried.
                    idempotency_key = None
                    idempotent = method in _IDEMPOTENT_METHODS or read_only
                    continue
                if not (e.retryable and idempotent) or attempt >= self.max_retries:
                    raise
                delay = e.retry_after
            except ConnectionFailed as e:
                if (e.sent and not idempotent) or attempt >= self.max_retries:
                    raise
                delay = None
            if delay is None:                       # exponential backoff, full jitter
                delay = random.uniform(0, self.backoff_s * (2 ** attempt))
            time.sleep(min(delay, self.max_backoff_s))
            attempt += 1

    def _send(self, method: str, path: str, body: Optional[dict], idempotency_key: Optional[str]) -> dict:
        url = f"{self.base}{path}"
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            raise _api_error(method, path, e) from None
        except urllib.error.URLError as e:
            # urlopen wraps failures to connect or to send: no answer was
            # produced from this request.
            raise ConnectionFailed(f"{method} {path} -> {e.reason}", sent=False) from None
        except OSError as e:
            # Sent, then the connection dropped or timed out waiting for the
            # answer: the server may have applied it.
            raise ConnectionFailed(f"{method} {path} -> {type(e).__name__}: {e}", sent=True) from None

    def _ns(self, args: dict) -> str:
        return args.get("namespace") or self.default_ns

    # ── Dispatch ────────────────────────────────────────────────────────────
    def handle(self, name: str, args: dict) -> str:
        try:
            fn = getattr(self, f"_t_{name[len('feather_'):]}", None)
            if fn is None:
                return json.dumps({"error": f"unknown tool: {name}"})
            return json.dumps(fn(args or {}), default=str)
        except FeatherAPIError as exc:
            # The type tells the model what to do: wait (overloaded), go
            # elsewhere (wrong_shard), or change the request.
            return json.dumps({"error": str(exc), "tool": name, "error_type": exc.error,
                               "status": exc.status, "retryable": exc.retryable})
        except Exception as exc:
            return json.dumps({"error": str(exc), "tool": name})

    # ── Tool implementations ────────────────────────────────────────────────
    def _t_ingest(self, a: dict) -> dict:
        ns = self._ns(a)
        text = a["text"]
        rid = _gen_id(text)
        vec = self._embed(text)
        meta = {"content": text, "namespace_id": ns,
                "source": a.get("source", "claude"),
                "importance": float(a.get("importance", 1.0))}
        if a.get("entity_id"):
            meta["entity_id"] = a["entity_id"]
        self._req("POST", f"/v1/{ns}/vectors", {"id": rid, "vector": vec, "metadata": meta},
                  idempotency_key=str(uuid.uuid4()))
        return {"ok": True, "id": rid, "namespace": ns}

    def _search(self, a: dict, path: str, extra: dict) -> dict:
        ns = self._ns(a)
        body = {"vector": self._embed(a["query"]), "k": int(a.get("k", 5)), **extra}
        d = self._req("POST", f"/v1/{ns}/{path}", body, read_only=True)
        return {"namespace": ns, "results": [
            {"id": r["id"], "score": r.get("score"),
             "content": (r.get("metadata") or {}).get("content", ""),
             "source": (r.get("metadata") or {}).get("source", "")}
            for r in d.get("results", [])]}

    def _t_recall(self, a: dict) -> dict:
        return self._search(a, "search", {})

    def _t_keyword_recall(self, a: dict) -> dict:
        ns = self._ns(a)
        d = self._req("POST", f"/v1/{ns}/keyword_search",
                      {"query": a["query"], "k": int(a.get("k", 5))}, read_only=True)
        return {"namespace": ns, "results": [
            {"id": r["id"], "score": r.get("score"),
             "content": (r.get("metadata") or {}).get("content", "")}
            for r in d.get("results", [])]}

    def _t_context_chain(self, a: dict) -> dict:
        ns = self._ns(a)
        body = {"vector": self._embed(a["query"]), "k": int(a.get("k", 5)),
                "hops": int(a.get("hops", 2)), "modality": "text"}
        d = self._req("POST", f"/v1/{ns}/context_chain", body, read_only=True)
        return {"namespace": ns,
                "nodes": [{"id": n["id"], "score": n.get("score"),
                           "hop": n.get("hop_distance"),
                           "content": (n.get("metadata") or {}).get("content", "")}
                          for n in d.get("nodes", [])],
                "edges": d.get("edges", [])}

    def _t_get_record(self, a: dict) -> dict:
        ns = self._ns(a)
        return self._req("GET", f"/v1/{ns}/records/{int(a['id'])}")

    def _t_link(self, a: dict) -> dict:
        ns = self._ns(a)
        body = {"to_id": int(a["to_id"])}
        if a.get("rel_type"):
            body["rel_type"] = a["rel_type"]
        return self._req("POST", f"/v1/{ns}/records/{int(a['from_id'])}/link", body,
                         idempotency_key=str(uuid.uuid4()))

    def _t_stats(self, a: dict) -> dict:
        ns = self._ns(a)
        return self._req("GET", f"/v1/{ns}/admin/index_stats")

    def _t_list_namespaces(self, a: dict) -> dict:
        return self._req("GET", "/v1/namespaces")


def _api_error(method: str, path: str, e: "urllib.error.HTTPError") -> FeatherAPIError:
    raw = e.read().decode("utf-8", "replace")
    try:
        body = json.loads(raw) if raw else {}
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    status = e.code
    retry_after = None
    try:
        retry_after = float(e.headers.get("Retry-After")) if e.headers.get("Retry-After") else None
    except (TypeError, ValueError):
        pass
    # A server before typed errors sends {"detail"} alone: derive the rest.
    error = body.get("error") or _DEFAULT_TYPES.get(status, "client_error" if status < 500 else "server_error")
    retryable = bool(body.get("retryable", status in (429, 503)))
    cls = _ERROR_CLASSES.get(status, RequestError if status < 500 else ServerError)
    return cls(method, path, status, error, body.get("detail", raw[:300]), retryable, retry_after, body)
