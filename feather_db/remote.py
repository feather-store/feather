"""RemoteDB — embedded Feather's `DB`, served by the Cloud API.

    db = RemoteDB("https://feather.example.com", "acme", api_key="...")
    db.add(1, vec, meta)
    db.search(q, k=10, filter=FilterBuilder().namespace_prefix("acme.u1").build())

Same methods, same arguments, same return types and the same exceptions as
`feather_db.DB`, so code moves between a file and a server without changing a
line. That is the contract (CONTRACTS.md, "One feature set in every mode"),
and it is checked two ways:

  * `PARITY` maps every public `DB` method to the HTTP route that serves it,
    or names why it has none. tests/test_parity.py fails when the engine
    gains a method that is not in it, when a mapped route does not exist, or
    when RemoteDB lacks the method.
  * tests/test_contracts.py runs the same contract tests against embedded
    and remote.

A Cloud API namespace is a file: `RemoteDB(url, "acme")` is
`DB.open("acme.feather")`, and a record's `namespace_id` partitions it.
"""
from __future__ import annotations

import base64
import http.client
import json
import select
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

import feather_db

# method -> "METHOD /path" (path relative to /v1/{namespace}), or a reason it
# has no route of its own ("-: ...").
PARITY = {
    # writes
    "add":               "POST /vectors",
    "add_batch":         "POST /batch",
    "upsert":            "POST /batch",
    "update_metadata":   "PUT /records/{record_id}",
    "update_importance": "PUT /records/{record_id}/importance",
    "put_if":            "POST /batch",
    "write":             "POST /batch",
    "forget":            "DELETE /records/{record_id}",
    "forget_expired":    "POST /forget_expired",
    "purge":             "POST /purge",
    "erase":             "POST /erase",
    "link":              "POST /records/{record_id}/link",
    "auto_link":         "POST /auto_link",
    "touch":             "POST /records/{record_id}/touch",
    # reads
    "get_metadata":      "GET /records/{record_id}",
    "get_metadata_many": "POST /records/get_many",
    "get_vector":        "GET /records/{record_id}",
    "version":           "GET /records/{record_id}",
    "get_by_key":        "GET /by_key",
    "id_for":            "GET /by_key",
    "ids_for_many":      "POST /by_keys",
    "heat_select":       "POST /heat_select",
    "get_edges":         "GET /records/{record_id}/edges",
    "get_incoming":      "GET /records/{record_id}/edges",
    "search":            "POST /search",
    "keyword_search":    "POST /keyword_search",
    "hybrid_search":     "POST /hybrid_search",
    "context_chain":     "POST /context_chain",
    "newest":            "GET /newest",
    "ids_in_namespace":  "GET /ids",
    "ids_for_entity":    "GET /ids",
    "ids_with_attribute": "GET /ids",
    "get_all_ids":       "GET /ids",
    "all_ids":           "GET /ids",
    "list_namespaces":   "GET /metadata_namespaces",
    "namespace_size":    "GET /metadata_namespaces",
    "export_graph_json": "GET /graph",
    # maintenance and settings
    "save":              "POST /save",
    "vacuum":            "POST /vacuum",
    "compact":           "POST /compact",
    "size":              "GET /admin/engine",
    "dim":               "GET /admin/engine",
    "modality_names":    "GET /admin/engine",
    "storage_stats":     "GET /admin/engine",
    "wal_size":          "GET /admin/engine",
    "wal_fsync_count":   "GET /admin/engine",
    "is_compacting":     "GET /admin/engine",
    "wait_for_compaction": "GET /admin/engine",
    "get_ef":            "GET /admin/engine",
    "set_ef":            "PUT /admin/settings",
    "get_auto_compact":  "GET /admin/engine",
    "set_auto_compact":  "PUT /admin/settings",
    "is_quantized":      "GET /admin/engine",
    "set_quantized":     "PUT /admin/settings",
    "is_int8_ram":       "GET /admin/engine",
    "int8_info":         "GET /admin/engine",
    "background_cosine": "GET /admin/engine",
    "set_int8_ram":      "PUT /admin/settings",
    "namespace_ann_info": "GET /admin/engine",
    "set_namespace_ann_min": "PUT /admin/settings",
    "wait_for_namespace_graphs": "GET /admin/engine",
    # history (time travel): RemoteDB(url, ns, as_of=...) sends X-Feather-As-Of
    "history":           "GET /history",
    "tags":              "GET /history",
    "history_retention": "GET /history",
    "tag":               "POST /tags",
    "untag":             "DELETE /tags/{name}",
    "set_history_retention": "PUT /admin/settings",
    "as_of":             "GET /as_of",
    "now":               "GET /as_of",
    "record_versions":   "GET /record_versions",
    # the handle itself, not a feature of the data
    "open":              "-: a server opens its files; RemoteDB(url, namespace) is the open, "
                         "and RemoteDB(..., as_of=, generation=) the X-Feather-As-Of header",
    "close":             "-: RemoteDB.close() ends the client session; the server keeps the file",
    "closed":            "-: client-side (RemoteDB.closed)",
    "is_closed":         "-: client-side (RemoteDB.is_closed)",
    "is_read_only":      "-: client-side (True only for RemoteDB(..., as_of=))",
    "write_error":       "-: the server recovers a failed handle itself (503 write_failed)",
    "idempotent_result": "-: over HTTP, the Idempotency-Key header",
}


def _as_of_header(as_of, generation):
    """DB.open's as_of= / generation= as the X-Feather-As-Of header value."""
    if generation is not None:
        return f"generation:{int(generation)}"
    if as_of is None:
        return None
    if isinstance(as_of, str):
        return "tag:" + urllib.parse.quote(as_of, safe="")
    if hasattr(as_of, "timestamp"):
        as_of = as_of.timestamp()
    return f"time:{float(as_of)!r}"


@dataclass
class RemoteResult:
    """A search hit: SearchResult's attributes."""
    id: int
    score: float
    metadata: Any
    cosine: Optional[float] = None


@dataclass
class RemoteIncomingEdge:
    source_id: int
    rel_type: str
    weight: float


@dataclass
class RemoteContextNode:
    id: int
    score: float
    similarity: float
    hop: int
    metadata: Any


@dataclass
class RemoteContextEdge:
    source: int
    target: int
    rel_type: str
    weight: float


@dataclass
class RemoteContextChainResult:
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)


@dataclass
class RemoteBatchResult:
    ids: list
    versions: list
    replayed: bool = False
    note: str = ""


# ── transport ─────────────────────────────────────────────────────────────

class _Response:
    def __init__(self, status: int, body: bytes, headers):
        self.status_code, self._body, self.headers = status, body, headers

    def json(self):
        return json.loads(self._body) if self._body else None


class _StdlibSession:
    """The client's default transport: the standard library, no dependency.

    Keeps one connection per thread open between calls. A fresh connection
    per request (what urlopen does) pays a TCP handshake, and a TLS one over
    https, on every call: several round trips for a search that takes 0.1 ms.
    An idle connection the server has closed is noticed before reuse and
    replaced. The server can still close it between that check and the send;
    then an idempotent request is sent once more on a fresh connection, as
    urllib3 does, and a POST raises, since it may have been applied. Behind a
    proxy from the environment it falls back to urlopen, which honours it."""

    _IDEMPOTENT = frozenset({"GET", "HEAD", "PUT", "DELETE", "OPTIONS"})

    def __init__(self, timeout: float):
        self.timeout = timeout
        self._local = threading.local()
        self._all: list = []
        self._lock = threading.Lock()

    def _conn(self, scheme: str, netloc: str):
        conns = self._local.__dict__.setdefault("conns", {})
        c = conns.get((scheme, netloc))
        if c is not None and c.sock is not None and select.select([c.sock], [], [], 0)[0]:
            c.close()       # readable while idle: the server closed it (or sent junk)
        if c is None:
            cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
            c = conns[(scheme, netloc)] = cls(netloc, timeout=self.timeout)
            with self._lock:
                self._all.append(c)
        return c

    def request(self, method, url, json=None, params=None, headers=None):
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        data = None
        hdrs = dict(headers or {})
        if json is not None:
            data = _json_dumps(json).encode()
            hdrs["Content-Type"] = "application/json"
        u = urllib.parse.urlsplit(url)
        if u.scheme in urllib.request.getproxies() and not urllib.request.proxy_bypass(u.hostname or ""):
            return self._urlopen(method, url, data, hdrs)
        target = urllib.parse.urlunsplit(("", "", u.path or "/", u.query, ""))
        for attempt in (0, 1):
            c = self._conn(u.scheme, u.netloc)
            reused = c.sock is not None
            try:
                c.request(method, target, body=data, headers=hdrs)
                r = c.getresponse()
                return _Response(r.status, r.read(), r.headers)
            except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError):
                c.close()
                if not (reused and attempt == 0 and method.upper() in self._IDEMPOTENT):
                    raise
            except BaseException:
                c.close()   # half-used: the next call starts a fresh connection
                raise

    def _urlopen(self, method, url, data, hdrs):
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return _Response(r.status, r.read(), r.headers)
        except urllib.error.HTTPError as e:
            return _Response(e.code, e.read(), e.headers)

    def close(self):
        with self._lock:
            for c in self._all:
                c.close()
            self._all.clear()


def _json_dumps(obj) -> str:
    return json.dumps(obj, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))


# ── Metadata <-> JSON ─────────────────────────────────────────────────────

def _meta_json(m) -> dict:
    """Every field, explicitly: the API stores what is given as given."""
    m = m if m is not None else feather_db.Metadata()
    return {
        "timestamp": int(m.timestamp), "importance": float(m.importance), "type": int(m.type),
        "source": m.source, "content": m.content, "tags_json": m.tags_json,
        "namespace_id": m.namespace_id, "entity_id": m.entity_id, "key": m.key,
        "attributes": dict(m.attributes), "confidence": float(m.confidence), "ttl": int(m.ttl),
        "valid_at": int(m.valid_at),
        "recall_count": int(m.recall_count), "last_recalled_at": int(m.last_recalled_at),
        "edges": [{"target_id": e.target_id, "rel_type": e.rel_type, "weight": float(e.weight)}
                  for e in m.edges],
    }


def _meta_from(d: Optional[dict]):
    if d is None:
        return None
    m = feather_db.Metadata()
    m.timestamp = int(d.get("timestamp", 0))
    m.importance = float(d.get("importance", 1.0))
    m.type = feather_db.ContextType(int(d.get("type", 0)))
    m.source = d.get("source", "")
    m.content = d.get("content", "")
    m.tags_json = d.get("tags_json", "")
    m.namespace_id = d.get("namespace_id") or ""
    m.entity_id = d.get("entity_id", "")
    m.key = d.get("key") or ""
    m.confidence = float(d.get("confidence", 1.0))
    m.ttl = int(d.get("ttl", 0))
    m.valid_at = int(d.get("valid_at", 0) or 0)
    m.version = int(d.get("version", 0))
    m.recall_count = int(d.get("recall_count", 0))
    m.last_recalled_at = int(d.get("last_recalled_at", 0))
    for k, v in (d.get("attributes") or {}).items():
        m.set_attribute(k, v)
    m.edges = [feather_db.Edge(int(e["target_id"]), e.get("rel_type", "related_to"),
                               float(e.get("weight", 1.0))) for e in (d.get("edges") or [])]
    return m


def _filter_json(f) -> dict:
    if f is None:
        return {}
    out = {}
    for name in ("namespace_id", "namespace_prefix", "entity_id", "source", "source_prefix",
                 "importance_gte", "confidence_gte", "timestamp_after", "timestamp_before",
                 "valid_after", "valid_before"):
        v = getattr(f, name, None)
        if v is not None:
            out[name] = v
    if f.attributes_match is not None:
        out["attributes_match"] = dict(f.attributes_match)
    if f.tags_contains is not None:
        out["tags_contains"] = list(f.tags_contains)
    if f.types is not None:
        out["types"] = [int(t) for t in f.types]
    return out


def _scoring_json(s) -> dict:
    if s is None:
        return {}
    return {"scoring_half_life": s.decay_half_life_days, "scoring_weight": s.time_weight,
            "scoring_min": s.min_weight, "scoring_relevance_floor": s.relevance_floor}


def _vec(v) -> str:
    return base64.b64encode(np.ascontiguousarray(v, dtype=np.float32).tobytes()).decode()


# ── the client ────────────────────────────────────────────────────────────

class RemoteDB:
    """See the module docstring. `session` may be any object with
    `request(method, url, json=, params=, headers=)` returning a response with
    `status_code`, `json()` and `headers` — httpx.Client, requests.Session,
    FastAPI's TestClient; by default the standard library."""

    BATCH_MAX = 10_000          # ops per POST /batch

    def __init__(self, base_url: str, namespace: str, api_key: str = "",
                 session=None, timeout: float = 30.0, as_of=None, generation=None):
        self.base = base_url.rstrip("/")
        self.namespace = namespace
        self._headers = {"X-API-Key": api_key} if api_key else {}
        # As DB.open(..., as_of=, generation=): every request reads that commit.
        if as_of is not None and generation is not None:
            raise ValueError("RemoteDB: pass as_of or generation, not both")
        if isinstance(as_of, bool):
            raise ValueError("as_of is a time (unix seconds or datetime) or a tag name")
        self._as_of_header = _as_of_header(as_of, generation)
        if self._as_of_header:
            self._headers["X-Feather-As-Of"] = self._as_of_header
        self._as_of_info = None
        self._own_session = session is None      # a caller's session is theirs to close
        self._session = session if session is not None else _StdlibSession(timeout)
        self._closed = False

    # ── plumbing ──────────────────────────────────────────────────────
    def _call(self, method: str, path: str, *, body=None, params=None, headers=None,
              ok_404: bool = False, raw: bool = False):
        if self._closed:
            raise RuntimeError("RemoteDB is closed")
        url = f"{self.base}/v1/{urllib.parse.quote(self.namespace, safe='')}{path}"
        r = self._session.request(method, url, json=body, params=params,
                                  headers={**self._headers, **(headers or {})})
        if r.status_code == 404 and ok_404:
            return None
        if r.status_code >= 400:
            _raise(r)
        return r if raw else r.json()

    def _engine(self) -> dict:
        return self._call("GET", "/admin/engine")

    def _settings(self, **kw):
        return self._call("PUT", "/admin/settings", body=kw)

    # ── handle ────────────────────────────────────────────────────────
    def ensure_namespace(self, dim=None):
        """Create this namespace if it does not exist; a no-op if it does.

        `DB.open` creates a missing file. The server creates a namespace on its
        first write but answers a READ of a missing one with 404, so a client
        whose first call is a read (an agent loading its memory) must call this
        first. `dim` only applies when the namespace is created."""
        if self._closed:
            raise RuntimeError("RemoteDB is closed")
        body = {"name": self.namespace, **({"dim": int(dim)} if dim else {})}
        r = self._session.request("POST", f"{self.base}/v1/namespaces", json=body,
                                  params=None, headers=self._headers)
        if r.status_code >= 400:
            _raise(r)

    def close(self, save: bool = True):
        if not self._closed and save and not self._as_of_header:
            self.save()
        self._closed = True
        if self._own_session:
            self._session.close()

    @property
    def closed(self) -> bool:
        return self._closed

    def is_closed(self) -> bool:
        return self._closed

    def is_read_only(self) -> bool:
        return self._as_of_header is not None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── writes ────────────────────────────────────────────────────────
    def add(self, id, vec, meta=None, modality="text"):
        # No meta: the vector only, and a live record keeps its metadata.
        body = {"id": int(id), "vector_b64": _vec(vec), "modality": modality}
        if meta is not None:
            body["metadata"] = _meta_json(meta)
        self._call("POST", "/vectors", body=body)

    def add_batch(self, ids, vecs, metas=None, modality="text"):
        vecs = np.asarray(vecs, dtype=np.float32)
        ops = [{"op": "put", "id": int(i), "vector": vecs[n].tolist(), "modality": modality,
                **({"metadata": _meta_json(metas[n])} if metas else {})}
               for n, i in enumerate(ids)]
        for s in range(0, len(ops), self.BATCH_MAX):
            self._call("POST", "/batch", body={"ops": ops[s:s + self.BATCH_MAX]})

    def upsert(self, meta, vec=None, modality="text", expected_version=None):
        op = {"op": "upsert", "metadata": _meta_json(meta), "modality": modality,
              "expected_version": expected_version}
        if vec is not None:
            op["vector"] = np.asarray(vec, dtype=np.float32).tolist()
        r = self._call("POST", "/batch", body={"ops": [op]})["results"][0]
        return r["id"], r["version"]

    def update_metadata(self, id, meta):
        self._call("PUT", f"/records/{int(id)}", body={"metadata": _meta_json(meta)})

    def update_importance(self, id, importance):
        self._call("PUT", f"/records/{int(id)}/importance", body={"importance": float(importance)})

    def put_if(self, id, meta, expected_version, vec=None, modality="text"):
        ops = []
        if vec is not None:
            ops.append({"op": "put", "id": int(id), "vector": np.asarray(vec, np.float32).tolist(),
                        "modality": modality, "metadata": _meta_json(meta),
                        "expected_version": int(expected_version)})
        up = {"op": "update", "id": int(id), "metadata": _meta_json(meta)}
        if vec is None:
            up["expected_version"] = int(expected_version)
        ops.append(up)
        return self._call("POST", "/batch", body={"ops": ops})["results"][-1]["version"]

    def write(self, batch, idempotency_key=None, fingerprint="", note=""):
        ops = []
        for o in batch.ops():
            op = {"op": o["kind"], "modality": o["modality"], "expected_version": o["expected"]}
            if o["ref"] is not None:
                op["key_namespace"], op["key"] = o["ref"]
            elif o["kind"] != "upsert":
                op["id"] = o["id"]
            if o["kind"] in ("put", "update", "upsert"):
                op["metadata"] = _meta_json(o["meta"])
            if o["vec"] is not None:
                op["vector"] = o["vec"].tolist()
            if o["kind"] == "link":
                if o["to_ref"] is not None:
                    op["to_key_namespace"], op["to_key"] = o["to_ref"]
                else:
                    op["to_id"] = o["to_id"]
                op["rel_type"], op["weight"] = o["rel_type"], o["weight"]
                op["require_target"] = o["require_target"]
            ops.append(op)
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        r = self._call("POST", "/batch", body={"ops": ops}, headers=headers, raw=True)
        res = r.json()["results"]
        return RemoteBatchResult([x["id"] for x in res], [x["version"] for x in res],
                                 replayed=r.headers.get("Idempotent-Replayed") == "true")

    def forget(self, id):
        # cascade=false: embedded forget() leaves edges into the record alone.
        self._call("DELETE", f"/records/{int(id)}", params={"cascade": "false"}, ok_404=True)

    def forget_expired(self):
        return self._call("POST", "/forget_expired")["forgotten"]

    def purge(self, namespace_id):
        return self._call("POST", "/purge", body={"namespace_id": namespace_id})["removed"]

    def erase(self, ids=None, namespace="", entity="", prefix=False, cites=None,
              feeds=None, dry_run=False):
        body = {"ids": [int(i) for i in (ids or [])], "namespace_id": namespace,
                "prefix": bool(prefix), "entity_id": entity, "dry_run": bool(dry_run)}
        if cites is not None:
            body["cites"] = list(cites)
        if feeds is not None:
            body["feeds"] = list(feeds)
        try:
            return self._call("POST", "/erase", body=body)
        finally:                  # as DB.erase: pockets and embedding caches
            if not dry_run:
                from feather_db.pocket import _after_erase
                _after_erase(self)

    def link(self, from_id, to_id, rel_type="related_to", weight=1.0):
        r = self._call("POST", f"/records/{int(from_id)}/link",
                       body={"to_id": int(to_id), "rel_type": rel_type, "weight": float(weight)},
                       ok_404=True)
        return r is not None

    def auto_link(self, modality="text", threshold=0.80, rel_type="related_to", candidates=15):
        return self._call("POST", "/auto_link", body={
            "modality": modality, "threshold": threshold, "rel_type": rel_type,
            "candidates": candidates})["links_created"]

    def touch(self, id):
        self._call("POST", f"/records/{int(id)}/touch")

    # ── reads ─────────────────────────────────────────────────────────
    def get_metadata(self, id, include_dead=False):
        r = self._call("GET", f"/records/{int(id)}",
                       params={"include_deleted": "true" if include_dead else "false"}, ok_404=True)
        return _meta_from(r["metadata"]) if r else None

    def get_metadata_many(self, ids, include_dead=False):
        ids = [int(i) for i in ids]
        out = []
        for i in range(0, len(ids), self.BATCH_MAX):
            r = self._call("POST", "/records/get_many",
                           body={"ids": ids[i:i + self.BATCH_MAX],
                                 "include_deleted": bool(include_dead)})
            out += [_meta_from(x["metadata"]) if x else None for x in r["records"]]
        return out

    def get_vector(self, id, modality="text"):
        r = self._call("GET", f"/records/{int(id)}", params={"include_vector": "true",
                                                              "modality": modality}, ok_404=True)
        v = (r or {}).get("vector")
        return np.asarray(v if v else [], dtype=np.float32)

    def version(self, id):
        r = self._call("GET", f"/records/{int(id)}", ok_404=True)
        return int(r["version"]) if r else 0

    def get_by_key(self, key, namespace=""):
        r = self._call("GET", "/by_key", params={"key": key, "key_namespace": namespace}, ok_404=True)
        return _meta_from(r["metadata"]) if r else None

    def id_for(self, key, namespace=""):
        r = self._call("GET", "/by_key", params={"key": key, "key_namespace": namespace}, ok_404=True)
        return int(r["id"]) if r else None

    def heat_select(self, namespaces, now, budget, half_life_days, min_heat, trusted,
                    exclude_sources, usage_half_life=3.0, unused_floor=0.4, inherit_decay=0.7,
                    chars_per_token=4, pinned_attr="_pocket_pinned", due_attr="_pocket_due_at",
                    origin_attr="_origin", text_attr="_lg_value",
                    origins=("user", "agent", "tool", "external"), default_origin="user",
                    unknown_origin="external"):
        """DB.heat_select: an agent's working set in one request (Pocket.hot()
        otherwise reads every record in scope over the wire)."""
        body = dict(namespaces=list(namespaces), now=float(now), budget=float(budget),
                    half_life_days=float(half_life_days), min_heat=float(min_heat),
                    trusted=list(trusted), exclude_sources=list(exclude_sources),
                    usage_half_life=usage_half_life, unused_floor=unused_floor,
                    inherit_decay=inherit_decay, chars_per_token=chars_per_token,
                    pinned_attr=pinned_attr, due_attr=due_attr, origin_attr=origin_attr,
                    text_attr=text_attr, origins=list(origins), default_origin=default_origin,
                    unknown_origin=unknown_origin)
        r = self._call("POST", "/heat_select", body=body)
        return [(int(i), int(d), float(h), int(t)) for i, d, h, t in r["picks"]]

    def ids_for_many(self, refs):
        refs = [[str(ns), str(k)] for ns, k in refs]
        out = []
        for s in range(0, len(refs), 10_000):
            r = self._call("POST", "/by_keys", body={"refs": refs[s:s + 10_000]})
            out += [int(x) if x is not None else None for x in r["ids"]]
        return out

    def get_edges(self, id):
        r = self._call("GET", f"/records/{int(id)}/edges")
        return [feather_db.Edge(e["target_id"], e["rel_type"], e["weight"]) for e in r["outgoing"]]

    def get_incoming(self, id):
        r = self._call("GET", f"/records/{int(id)}/edges")
        return [RemoteIncomingEdge(e["source_id"], e["rel_type"], e["weight"]) for e in r["incoming"]]

    def _hits(self, r):
        return [RemoteResult(h["id"], h["score"], _meta_from(h.get("metadata")), h.get("cosine"))
                for h in r["results"]]

    def search(self, q, k=5, filter=None, scoring=None, modality="text",
               record_salience=True, with_cosine=False):
        body = {"vector_b64": _vec(q), "k": int(k), "modality": modality,
                "track": bool(record_salience), "raw_score": bool(with_cosine),
                **_filter_json(filter), **_scoring_json(scoring)}
        return self._hits(self._call("POST", "/search", body=body))

    def keyword_search(self, query, k=10, filter=None, record_salience=True):
        return self._hits(self._call("POST", "/keyword_search",
                                     body={"query": query, "k": int(k),
                                           "track": bool(record_salience),
                                           **_filter_json(filter)}))

    def hybrid_search(self, vec, query, k=10, rrf_k=60, filter=None, scoring=None, modality="text",
                      fusion="score"):
        body = {"vector_b64": _vec(vec), "query": query, "k": int(k), "rrf_k": int(rrf_k),
                "fusion": fusion,
                "modality": modality, **_filter_json(filter), **_scoring_json(scoring)}
        return self._hits(self._call("POST", "/hybrid_search", body=body))

    def context_chain(self, q, k=5, hops=2, modality="text"):
        r = self._call("POST", "/context_chain", body={"vector_b64": _vec(q), "k": int(k),
                                                         "hops": int(hops), "modality": modality})
        return RemoteContextChainResult(
            [RemoteContextNode(n["id"], n["score"], n.get("similarity", 0.0), n["hop_distance"],
                               _meta_from(n.get("metadata"))) for n in r["nodes"]],
            [RemoteContextEdge(e["source_id"], e["target_id"], e["rel_type"], e["weight"])
             for e in r["edges"]])

    def newest(self, namespace="", limit=10, prefix=False, before=None, by_valid=False):
        params = {"metadata_namespace": namespace, "limit": int(limit),
                  "prefix": "true" if prefix else "false"}
        if by_valid:
            params["by_valid"] = "true"
        if isinstance(before, tuple):
            params["before_ts"], params["before_id"] = int(before[0]), int(before[1])
        elif before is not None:
            params["before_ts"] = int(before)
        return self._call("GET", "/newest", params=params)["ids"]

    def ids_in_namespace(self, namespace_id):
        return self._call("GET", "/ids", params={"metadata_namespace": namespace_id})["ids"]

    def ids_for_entity(self, entity_id):
        return self._call("GET", "/ids", params={"entity_id": entity_id})["ids"]

    def ids_with_attribute(self, key, value):
        return self._call("GET", "/ids", params={"attr_key": key, "attr_value": value})["ids"]

    def get_all_ids(self, modality="text"):
        return self._call("GET", "/ids", params={"modality": modality})["ids"]

    def all_ids(self):
        return self._call("GET", "/ids")["ids"]

    def list_namespaces(self):
        return [n["id"] for n in self._call("GET", "/metadata_namespaces")["namespaces"]]

    def namespace_size(self, namespace_id):
        for n in self._call("GET", "/metadata_namespaces")["namespaces"]:
            if n["id"] == namespace_id:
                return n["size"]
        return 0

    def export_graph_json(self, namespace_filter="", entity_filter=""):
        return json.dumps(self._call("GET", "/graph", params={
            "ns_filter": namespace_filter, "entity_filter": entity_filter, "full": "true"}))

    # ── maintenance and settings ──────────────────────────────────────
    def save(self):
        self._call("POST", "/save")

    def vacuum(self):
        self._call("POST", "/vacuum")

    def compact(self):
        return self._call("POST", "/compact", params={"prune_dead_edges": "false"})["reclaimed"]

    def size(self):
        return self._engine()["size"]

    def dim(self, modality="text"):
        e = self._engine()                 # one request, not two
        m = e["modalities"].get(modality)
        return m["dim"] if m else e["dim"]

    def modality_names(self):
        return list(self._engine()["modalities"])

    def storage_stats(self):
        return self._engine()["storage"]

    # ── history (time travel) ─────────────────────────────────────────
    def history(self):
        return self._call("GET", "/history")["commits"]

    def tags(self):
        return {k: int(v) for k, v in self._call("GET", "/history")["tags"].items()}

    def history_retention(self):
        r = self._call("GET", "/history")["retention_s"]
        return float("inf") if r is None else float(r)

    def tag(self, name):
        self._call("POST", "/tags", body={"name": name})

    def untag(self, name):
        self._call("DELETE", f"/tags/{urllib.parse.quote(name, safe='')}")

    def set_history_retention(self, seconds):
        seconds = float(seconds)
        if seconds != seconds or seconds < 0:
            raise ValueError("history retention is seconds >= 0 (infinity: keep everything)")
        self._settings(history_retention_s="inf" if seconds == float("inf") else seconds)

    def as_of(self):
        if self._as_of_header is None:
            return None
        if self._as_of_info is None:                 # one commit for the handle's life
            self._as_of_info = self._call("GET", "/as_of")["as_of"]
        return self._as_of_info

    def now(self):
        a = self.as_of()
        return a["clock"] if a else time.time()

    def record_versions(self, namespace=""):
        rows = self._call("GET", "/record_versions", params={"namespace_prefix": namespace})["versions"]
        return {int(i): int(v) for i, v in rows}

    def wal_size(self):
        return self._engine()["wal_size"]

    def wal_fsync_count(self):
        return self._engine()["wal_fsync_count"]

    def is_compacting(self):
        return self._engine()["compacting"]

    def wait_for_compaction(self, timeout=0.0):
        return self._poll(lambda e: not e["compacting"], timeout)

    def get_ef(self, modality="text"):
        return self._engine()["modalities"][modality]["ef"]

    def set_ef(self, ef, modality=""):
        self._settings(ef=int(ef), ef_modality=modality)

    def get_auto_compact(self):
        return self._engine()["auto_compact"]

    def set_auto_compact(self, ratio):
        self._settings(auto_compact=float(ratio))

    def is_quantized(self, modality):
        m = self._engine()["modalities"].get(modality)
        return bool(m and m["quantized"])

    def set_quantized(self, modality, on=True):
        self._settings(quantized={modality: bool(on)})

    def is_int8_ram(self, modality):
        m = self._engine()["modalities"].get(modality)
        return bool(m and m["int8_ram"])

    def int8_info(self, modality):
        return self._engine()["modalities"][modality]["int8"]

    def background_cosine(self, modality="text"):
        m = self._engine()["modalities"].get(modality)
        return float(m["background_cosine"]) if m else 0.0

    def set_int8_ram(self, modality, max_abs=None, refine=2):
        self._settings(int8_ram={modality: {"max_abs": max_abs, "refine": refine}})

    def namespace_ann_info(self):
        return self._engine()["namespace_ann"]

    def set_namespace_ann_min(self, n):
        self._settings(namespace_ann_min=int(n))

    def wait_for_namespace_graphs(self, timeout=0.0):
        return self._poll(lambda e: e["namespace_ann"]["pending"] == 0, timeout)

    def _poll(self, done, timeout):
        end = time.monotonic() + timeout if timeout > 0 else None
        while True:
            if done(self._engine()):
                return True
            if end is not None and time.monotonic() >= end:
                return False
            time.sleep(0.02)


def _raise(r):
    """The engine's exception for the API's typed error."""
    try:
        body = r.json() or {}
    except ValueError:
        body = {}
    detail = body.get("detail", "")
    kind = body.get("error", "")
    msg = detail if isinstance(detail, str) else json.dumps(detail)
    if kind in ("version_conflict", "precondition_failed"):
        e = feather_db.VersionConflictError(msg)
        e.id, e.expected, e.actual = body.get("id"), body.get("expected_version"), body.get("actual_version")
        raise e
    if kind == "duplicate_key":
        raise feather_db.DuplicateKeyError(msg)
    if kind == "idempotency_key_reused":
        raise feather_db.IdempotencyKeyReusedError(msg)
    if kind == "write_failed":
        raise feather_db.WriteFailedError(msg)
    if r.status_code in (400, 422):
        if "Dimension mismatch" in msg or "dim" in msg:
            raise RuntimeError(msg)
        raise ValueError(msg)
    raise RuntimeError(f"HTTP {r.status_code} {kind}: {msg}")
