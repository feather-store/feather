"""Send the same requests to feather-api (Python) and feather-server (Rust)
and compare the answers: status, error type, headers that carry meaning, and
the JSON body (floats to 1e-6, wall-clock fields to a few seconds).

    .venv/bin/python feather-server/tests/parity.py [--keep]

Builds a small dataset, copies it into two data directories, starts both
servers on free ports (dev mode), and exits non-zero on any difference.
Needs feather-server built: cargo build --release (in feather-server/).
"""
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import feather_db  # noqa: E402

RUST_BIN = os.path.join(ROOT, "feather-server", "target", "release", "feather-server")
DIM = 64
N = 2000
NS = "parity"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def build(path):
    rng = np.random.default_rng(7)
    db = feather_db.DB.open(path, dim=DIM)
    now = int(time.time())
    for i in range(1, N + 1):
        m = feather_db.Metadata()
        m.timestamp = now - (i % 400) * 86400
        m.importance = [1.0, 0.9, 0.5, 0.25][i % 4]
        m.confidence = [1.0, 0.7][i % 2]
        m.type = feather_db.ContextType(i % 4)
        m.source = ["user", "agent", "tool"][i % 3]
        m.content = f"record {i}: the quick brown fox é中 \"quoted\" \\ line\nbreak {i % 17}"
        m.tags_json = json.dumps([f"t{i % 5}", f"g{i % 3}"])
        m.namespace_id = ["org.a", "org.b", "org.a.sub"][i % 3]
        m.entity_id = f"user_{i % 50}"
        m.set_attribute("channel", ["ig", "fb"][i % 2])
        m.set_attribute("bucket", str(i % 7))
        if i % 10 == 0:
            m.key = f"k{i}"
        if i % 6 == 0:
            m.valid_at = now - i * 3600
        if i % 9 == 0:
            m.ttl = 86400 * 365
        v = rng.standard_normal(DIM).astype(np.float32)
        db.add(id=i, vec=v / np.linalg.norm(v), meta=m)
    for i in range(1, 200, 7):
        db.link(from_id=i, to_id=i + 1, rel_type="caused_by", weight=0.75)
        db.link(from_id=i, to_id=i + 2)
    for i in range(5, 100, 25):
        db.forget(i)
    db.save()
    db.close()


def start(cmd, env, port, cwd):
    p = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for _ in range(200):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5)
            return p
        except Exception:
            if p.poll() is not None:
                raise SystemExit(f"server exited: {p.stderr.read().decode()[-2000:]}")
            time.sleep(0.1)
    p.kill()
    raise SystemExit("server did not start")


def call(port, method, path, body=None, headers=None):
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                 headers={"content-type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


WALL_CLOCK = {"last_recalled_at", "timestamp", "completed_at"}
# An erase receipt's fields that differ by run, not by server: a random id,
# and the sizes and generation of files each server checkpoints on its own.
RECEIPT_RUN = {"receipt_id", "file_bytes_before", "file_bytes_after", "wal_bytes_after", "generation"}


def same(a, b, path="$", ignore=frozenset()):
    """Differences between two JSON values, as strings."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if path == "$" and k in ignore:
                continue
            if k not in a or k not in b:
                out.append(f"{path}.{k}: only in {'python' if k in a else 'rust'}")
            else:
                out += same(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in same(x, y, f"{path}[{i}]")]
    if isinstance(a, bool) or isinstance(b, bool):
        return [] if a == b else [f"{path}: {a!r} vs {b!r}"]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if path.rsplit(".", 1)[-1] in WALL_CLOCK and abs(a - b) <= 3:
            return []
        if a == b or abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b)):
            return []
        return [f"{path}: {a!r} vs {b!r}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def main():
    keep = "--keep" in sys.argv
    if not os.path.exists(RUST_BIN):
        raise SystemExit(f"build it first: (cd feather-server && cargo build --release); {RUST_BIN} missing")
    work = tempfile.mkdtemp(prefix="feather-parity-")
    base = os.path.join(work, "base.feather")
    build(base)
    dirs = {}
    for side in ("py", "rs"):
        d = os.path.join(work, side)
        os.makedirs(d)
        shutil.copy(base, os.path.join(d, f"{NS}.feather"))
        dirs[side] = d
    py_port, rs_port = free_port(), free_port()
    common = {k: v for k, v in os.environ.items() if not k.startswith("FEATHER_")}
    common.update(FEATHER_DEV_MODE="1", FEATHER_CHECKPOINT_POLL_S="3600")
    py = start([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(py_port), "--log-level", "warning"],
               {**common, "FEATHER_DATA_DIR": dirs["py"], "PYTHONPATH": ROOT}, py_port,
               os.path.join(ROOT, "feather-api"))
    rs = start([RUST_BIN, "--host", "127.0.0.1", "--port", str(rs_port)],
               {**common, "FEATHER_DATA_DIR": dirs["rs"]}, rs_port, ROOT)

    rng = np.random.default_rng(11)

    def vec():
        v = rng.standard_normal(DIM).astype(np.float32)
        return (v / np.linalg.norm(v)).tolist()

    def b64(v):
        return base64.b64encode(np.asarray(v, dtype="<f4").tobytes()).decode()

    q = vec()
    cases = [
        ("POST", f"/v1/{NS}/search", {"vector": q, "k": 10}),
        ("POST", f"/v1/{NS}/search", {"vector_b64": b64(q), "k": 5}),
        ("POST", f"/v1/{NS}/search", {"vector": vec(), "k": 7, "raw_score": True}),
        ("POST", f"/v1/{NS}/search", {"vector": vec(), "k": 10, "include_metadata": False}),
        ("POST", f"/v1/{NS}/search", {"vector": vec(), "k": 10, "track": False}),
        ("POST", f"/v1/{NS}/search", {"vector": vec(), "k": 10, "namespace_id": "org.a"}),
        ("POST", f"/v1/{NS}/search", {"vector": vec(), "k": 10, "namespace_prefix": "org.a"}),
        ("POST", f"/v1/{NS}/search", {"vector": vec(), "k": 10, "entity_id": "user_3", "types": [1, 3]}),
        ("POST", f"/v1/{NS}/search", {"vector": vec(), "k": 10, "attributes_match": {"channel": "ig", "bucket": "2"}}),
        ("POST", f"/v1/{NS}/search", {"vector": vec(), "k": 10, "source": "tool", "importance_gte": 0.5}),
        ("POST", f"/v1/{NS}/search", {"vector": vec(), "k": 10, "source_prefix": "ag", "confidence_gte": 0.8}),
        ("POST", f"/v1/{NS}/search", {"vector": vec(), "k": 10, "tags_contains": ["t1"]}),
        ("POST", f"/v1/{NS}/search", {"vector": vec(), "k": 10, "timestamp_after": int(time.time()) - 30 * 86400}),
        ("POST", f"/v1/{NS}/search", {"vector": vec(), "k": 10, "timestamp_after": 0, "namespace_id": ""}),
        ("POST", f"/v1/{NS}/search", {"vector": vec(), "k": 10, "valid_after": 0}),
        ("POST", f"/v1/{NS}/search", {"vector": vec(), "k": 10, "scoring_half_life": 10.0}),
        ("POST", f"/v1/{NS}/search", {"vector": vec(), "k": 10, "scoring_weight": 0.0, "scoring_relevance_floor": 0.0}),
        ("POST", f"/v1/{NS}/search", {"vector": vec(), "k": 10, "scoring_min": 0.5}),
        ("POST", f"/v1/{NS}/search", {"vector": vec(), "unknown_field": 1}),
        ("POST", f"/v1/{NS}/keyword_search", {"query": "quick fox 17", "k": 10}),
        ("POST", f"/v1/{NS}/keyword_search", {"query": "record", "k": 10, "namespace_id": "org.b", "track": False}),
        ("POST", f"/v1/{NS}/keyword_search", {"query": "nothingmatches", "k": 3}),
        ("POST", f"/v1/{NS}/hybrid_search", {"query": "brown fox", "vector": vec(), "k": 10}),
        ("POST", f"/v1/{NS}/hybrid_search", {"query": "brown fox", "vector_b64": b64(vec()), "k": 10, "fusion": "rrf", "rrf_k": 30}),
        ("POST", f"/v1/{NS}/hybrid_search", {"query": "record", "vector": vec(), "k": 10, "scoring_half_life": 5.0, "entity_id": "user_4"}),
        ("GET", f"/v1/{NS}/records/10", None),
        ("GET", f"/v1/{NS}/records/8", None),
        ("GET", f"/v1/{NS}/records/12?include_vector=true", None),
        ("GET", f"/v1/{NS}/records/12?include_vector=1&modality=nope", None),
        ("GET", f"/v1/{NS}/records/5", None),
        ("GET", f"/v1/{NS}/records/5?include_deleted=true", None),
        ("GET", f"/v1/{NS}/records/999999", None),
        ("GET", f"/v1/{NS}/records/abc", None),
        ("GET", f"/v1/{NS}/records/1?include_vector=maybe", None),
        ("POST", f"/v1/{NS}/records/get_many", {"ids": [1, 5, 999999, 8, 10]}),
        ("POST", f"/v1/{NS}/records/get_many", {"ids": [5, 30], "include_deleted": True}),
        ("POST", f"/v1/{NS}/records/get_many", {"ids": []}),
        # errors
        ("POST", f"/v1/{NS}/search", {"vector": q, "k": 0}),
        ("POST", f"/v1/{NS}/search", {"vector": q, "k": 1001}),
        ("POST", f"/v1/{NS}/search", {"vector": [0.0] * DIM}),
        ("POST", f"/v1/{NS}/search", {"vector": [0.1] * (DIM + 1)}),
        ("POST", f"/v1/{NS}/search", {"vector": q, "vector_b64": b64(q)}),
        ("POST", f"/v1/{NS}/search", {"k": 3}),
        ("POST", f"/v1/{NS}/search", {"vector_b64": "not base64!!"}),
        ("POST", f"/v1/{NS}/search", {"vector_b64": base64.b64encode(b"abc").decode()}),
        ("POST", f"/v1/{NS}/search", b"{not json"),
        ("POST", "/v1/nosuchns/search", {"vector": q}),
        ("POST", "/v1/bad.name/search", {"vector": q}),
        ("POST", f"/v1/{NS}/keyword_search", {"k": 3}),
        ("POST", f"/v1/{NS}/hybrid_search", {"query": "  "}),
        ("POST", f"/v1/{NS}/hybrid_search", {"query": "x", "vector": q, "fusion": "max"}),
        # writes
        ("POST", f"/v1/{NS}/vectors", {"id": 5001, "vector": vec(), "metadata": {
            "content": "new one", "importance": 0.9, "type": 2, "attributes": {"a": "b"},
            "edges": [{"target_id": 1, "rel_type": "about", "weight": 0.5}], "key": "nk1",
            "valid_at": 1700000000, "confidence": 0.8, "ttl": 60, "timestamp": 1690000000}}),
        ("GET", f"/v1/{NS}/records/5001", None),
        ("POST", f"/v1/{NS}/vectors", {"id": 10, "vector_b64": b64(vec())}),
        ("GET", f"/v1/{NS}/records/10", None),
        ("POST", f"/v1/{NS}/vectors", {"id": 5002, "vector": vec(), "metadata": {"content": "x", "bogus": 1}}),
        ("POST", f"/v1/{NS}/vectors", {"id": 5002, "vector": vec(), "metadata": {"type": 7}}),
        ("POST", f"/v1/{NS}/vectors", {"id": 5002, "vector": [0.1] * 3}),
        ("POST", f"/v1/{NS}/vectors", {"id": 5003, "vector": vec(), "metadata": {"key": "nk1"}}),
        ("POST", f"/v1/{NS}/vectors", {"id": 5004, "vector": vec(), "metadata": {"content": "idem"}},
         {"Idempotency-Key": "abc"}),
        ("POST", f"/v1/{NS}/vectors", None, {"Idempotency-Key": "abc"}),       # replay: same body as above
        ("POST", f"/v1/{NS}/vectors", {"id": 5005, "vector": vec()}, {"Idempotency-Key": "abc"}),
        ("POST", "/v1/newns/vectors", {"id": 1, "vector": [1.0, 2.0, 3.0]}),
        ("POST", "/v1/newns/search", {"vector": [1.0, 2.0, 3.5], "k": 3}),
        ("GET", "/v1/namespaces", None),
    ]
    # The agent-memory routes: everything AgentMemory / Pocket /
    # PacketBuilder / FeatherStore send over RemoteDB besides search and get.
    week_ago = int(time.time()) - 7 * 86400
    heat = {"namespaces": ["org.a.sub", "org.a"], "now": time.time(), "budget": 400.0,
            "half_life_days": 7.0, "min_heat": 0.01, "trusted": ["user"], "exclude_sources": ["tool"]}
    cases += [
        ("GET", f"/v1/{NS}/by_key?key=k10&key_namespace=org.b", None),
        ("GET", f"/v1/{NS}/by_key?key=k10&key_namespace=org.a", None),
        ("GET", f"/v1/{NS}/by_key?key_namespace=org.a", None),
        ("GET", "/v1/nosuchns/by_key?key=x", None),
        ("POST", f"/v1/{NS}/by_keys", {"refs": [["org.b", "k10"], ["org.a", "k30"], ["org.a", "nope"],
                                                ["org.a.sub", "k20"], ["org.a.sub", "k50"]]}),
        ("POST", f"/v1/{NS}/by_keys", {"refs": [["org.b"]]}),
        ("POST", f"/v1/{NS}/by_keys", {"refs": [], "extra": 1}),
        ("POST", f"/v1/{NS}/by_keys", {"refs": [["a", "b"]] * 10_001}),
        ("GET", f"/v1/{NS}/newest?metadata_namespace=org.a&limit=5", None),
        ("GET", f"/v1/{NS}/newest?metadata_namespace=org&prefix=true&limit=7", None),
        ("GET", f"/v1/{NS}/newest?metadata_namespace=org.a&limit=4&before_ts={week_ago}", None),
        ("GET", f"/v1/{NS}/newest?metadata_namespace=org.a&limit=4&before_ts={week_ago}&before_id=500", None),
        ("GET", f"/v1/{NS}/newest?metadata_namespace=org.a&by_valid=1&limit=4", None),
        ("GET", f"/v1/{NS}/newest?metadata_namespace=org.a&limit=-3", None),
        ("GET", f"/v1/{NS}/newest?limit=abc", None),
        ("GET", f"/v1/{NS}/count_live?metadata_namespace=org.a", None),
        ("GET", f"/v1/{NS}/count_live?metadata_namespace=org&prefix=true&exclude_source=tool&exclude_source=agent", None),
        ("GET", f"/v1/{NS}/ids?metadata_namespace=org.b", None),
        ("GET", f"/v1/{NS}/ids?metadata_namespace=org.a&prefix=true", None),
        ("GET", f"/v1/{NS}/ids?entity_id=user_3", None),
        ("GET", f"/v1/{NS}/ids?attr_key=bucket&attr_value=2", None),
        ("GET", f"/v1/{NS}/ids?modality=text", None),
        ("GET", f"/v1/{NS}/ids", None),
        ("GET", f"/v1/{NS}/metadata_namespaces", None),
        ("POST", f"/v1/{NS}/heat_select", heat),
        ("POST", f"/v1/{NS}/heat_select", {**heat, "budget": 50.0, "inherit_decay": 0.5, "trusted": []}),
        ("POST", f"/v1/{NS}/heat_select", {**heat, "budget": -1}),
        ("POST", f"/v1/{NS}/heat_select", {k: v for k, v in heat.items() if k != "trusted"}),
        ("POST", f"/v1/{NS}/heat_select", {**heat, "bogus": 1}),
        ("GET", f"/v1/{NS}/admin/engine", None),
        ("POST", f"/v1/{NS}/records/touch_many", {"ids": [1, 2, 3, 999999]}),
        ("GET", f"/v1/{NS}/records/2", None),
        ("POST", f"/v1/{NS}/records/7/touch", b""),
        ("POST", f"/v1/{NS}/records/abc/touch", b""),
        # batch: every op kind, by id and by key
        ("POST", f"/v1/{NS}/batch", {"ops": [
            {"op": "put", "id": 6001, "vector": vec(), "metadata": {
                "content": "batch put", "key": "bk1", "namespace_id": "org.a", "timestamp": 1700000000}},
            {"op": "upsert", "vector": vec(), "metadata": {
                "key": "bk2", "namespace_id": "org.a", "content": "upserted", "timestamp": 1700000001}},
            {"op": "update", "key": "k50", "key_namespace": "org.a.sub", "metadata": {
                "content": "updated by key", "namespace_id": "org.a.sub", "timestamp": 1700000002}},
            {"op": "link", "id": 6001, "to_key": "k10", "to_key_namespace": "org.b", "rel_type": "about", "weight": 0.5},
            {"op": "forget", "id": 1000},
            {"op": "put", "id": 12, "vector": vec()},
        ]}),
        ("GET", f"/v1/{NS}/records/50", None),
        ("GET", f"/v1/{NS}/records/6001", None),
        ("GET", f"/v1/{NS}/records/12", None),
        ("GET", f"/v1/{NS}/records/1000", None),
        ("GET", f"/v1/{NS}/by_key?key=bk2&key_namespace=org.a", None),
        ("POST", f"/v1/{NS}/batch", {"ops": [{"op": "upsert", "metadata": {
            "key": "bk2", "namespace_id": "org.a", "content": "upserted again", "timestamp": 1700000003}}]}),
        ("POST", f"/v1/{NS}/batch", {"ops": [{"op": "update", "id": 6001, "metadata": {
            "content": "idem", "namespace_id": "org.a", "timestamp": 1700000004}}]}, {"Idempotency-Key": "b-1"}),
        ("POST", f"/v1/{NS}/batch", None, {"Idempotency-Key": "b-1"}),          # replay
        ("POST", f"/v1/{NS}/batch", {"ops": [{"op": "forget", "id": 6001}]}, {"Idempotency-Key": "b-1"}),
        ("POST", f"/v1/{NS}/batch", {"ops": [{"op": "update", "id": 6001, "expected_version": 1,
                                              "metadata": {"content": "stale"}}]}),
        ("POST", f"/v1/{NS}/batch", {"ops": [{"op": "put", "id": 6002, "vector": vec(), "expected_version": 0,
                                              "metadata": {"key": "bk1", "namespace_id": "org.a"}}]}),
        ("POST", f"/v1/{NS}/batch", {"ops": [{"op": "put", "id": 6002}]}),
        ("POST", f"/v1/{NS}/batch", {"ops": [{"op": "update", "id": 6001}]}),
        ("POST", f"/v1/{NS}/batch", {"ops": [{"op": "link", "id": 6001}]}),
        ("POST", f"/v1/{NS}/batch", {"ops": [{"op": "forget"}]}),
        ("POST", f"/v1/{NS}/batch", {"ops": [{"op": "upsert", "metadata": {"content": "no key"}}]}),
        ("POST", f"/v1/{NS}/batch", {"ops": [{"op": "link", "id": 6001, "to_id": 99999999}]}),
        ("POST", f"/v1/{NS}/batch", {"ops": [{"op": "update", "key": "nope", "metadata": {"content": "x"}}]}),
        ("POST", f"/v1/{NS}/batch", {"ops": [{"op": "put", "id": 6003, "vector": [0.1, 0.2]}]}),
        ("POST", f"/v1/{NS}/batch", {"ops": [{"op": "nope", "id": 1}]}),
        ("POST", f"/v1/{NS}/batch", {"ops": []}),
        ("POST", f"/v1/{NS}/batch", {"ops": [{"op": "forget", "id": 1, "expected_version": -1}]}),
        ("POST", f"/v1/{NS}/batch", {"ops": [{"op": "forget", "id": 1, "bogus": True}]}),
        ("POST", "/v1/batchns/batch", {"ops": [{"op": "put", "id": 1, "vector": [1.0, 2.0, 3.0]}]}),
        # one edge (observations.note links to its evidence)
        ("POST", f"/v1/{NS}/records/6001/link", {"to_id": 11, "rel_type": "cites", "weight": 0.25}),
        ("POST", f"/v1/{NS}/records/6001/link", {"to_id": 888888}),                 # forward reference
        ("POST", f"/v1/{NS}/records/13/link", {"to_id": 14}, {"Idempotency-Key": "l-1"}),
        ("POST", f"/v1/{NS}/records/13/link", None, {"Idempotency-Key": "l-1"}),    # replay
        ("GET", f"/v1/{NS}/records/13", None),
        ("POST", f"/v1/{NS}/records/5/link", {"to_id": 1}),                         # forgotten source
        ("POST", f"/v1/{NS}/records/777777/link", {"to_id": 1}),
        ("POST", f"/v1/{NS}/records/13/link", {"rel_type": "x"}),
        ("POST", "/v1/nosuchns/records/1/link", {"to_id": 1}),
        # corrections: PUT replaces metadata, keeping edges / key / valid_at
        # the request leaves out (Pocket.pin, keys.write)
        ("PUT", f"/v1/{NS}/records/20", {"metadata": {"content": "edited", "namespace_id": "org.a.sub",
                                                      "timestamp": 1700000010}}),
        ("GET", f"/v1/{NS}/records/20", None),
        ("PUT", f"/v1/{NS}/records/15", {"metadata": {"content": "unlinked", "edges": [], "key": "",
                                                      "valid_at": 0, "timestamp": 1700000011}}),
        ("GET", f"/v1/{NS}/records/15", None),
        ("PUT", f"/v1/{NS}/records/22", {"metadata": {"content": "stale"}}, {"If-Match": "1"}),
        ("PUT", f"/v1/{NS}/records/22", {"metadata": {"content": "weak stale"}}, {"If-Match": 'W/"1"'}),
        ("PUT", f"/v1/{NS}/records/22", {"metadata": {"content": "x"}}, {"If-Match": "abc"}),
        ("PUT", f"/v1/{NS}/records/22", {"metadata": {"content": "any", "timestamp": 1700000012}},
         {"If-Match": "*"}),
        ("GET", f"/v1/{NS}/records/22", None),
        ("PUT", f"/v1/{NS}/records/22", {"metadata": {"content": "current", "timestamp": 1700000013}},
         {"If-Match": "@etag"}),                                                 # the ETag just read
        ("PUT", f"/v1/{NS}/records/26", {"metadata": {"content": "idem put", "timestamp": 1700000014}},
         {"Idempotency-Key": "p-1"}),
        ("PUT", f"/v1/{NS}/records/26", None, {"Idempotency-Key": "p-1"}),     # replay
        ("PUT", f"/v1/{NS}/records/27", {"metadata": {"content": "other"}}, {"Idempotency-Key": "p-1"}),
        ("PUT", f"/v1/{NS}/records/40", {"metadata": {"key": "k10", "namespace_id": "org.b"}}),
        ("PUT", f"/v1/{NS}/records/5", {"metadata": {"content": "forgotten one", "timestamp": 1700000015}}),
        ("PUT", f"/v1/{NS}/records/999999", {"metadata": {"content": "nobody"}}),
        ("PUT", f"/v1/{NS}/records/abc", {"metadata": {"content": "x"}}),
        ("PUT", f"/v1/{NS}/records/26", {"metadata": {"bogus": 1}}),
        ("PUT", f"/v1/{NS}/records/26", {}),
        ("PUT", "/v1/nosuchns/records/1", {"metadata": {"content": "x"}}),
        # forgets: DELETE prunes edges into the record unless cascade=false
        ("DELETE", f"/v1/{NS}/records/9", None),
        ("GET", f"/v1/{NS}/records/8", None),
        ("DELETE", f"/v1/{NS}/records/24?cascade=false", None),
        ("GET", f"/v1/{NS}/records/22", None),
        ("DELETE", f"/v1/{NS}/records/9", None),
        ("DELETE", f"/v1/{NS}/records/36", None, {"Idempotency-Key": "d-1"}),
        ("DELETE", f"/v1/{NS}/records/36", None, {"Idempotency-Key": "d-1"}),  # replay, record gone
        ("DELETE", f"/v1/{NS}/records/37", None, {"Idempotency-Key": "d-1"}),
        ("DELETE", f"/v1/{NS}/records/999999", None),
        ("DELETE", f"/v1/{NS}/records/abc", None),
        ("DELETE", f"/v1/{NS}/records/38?cascade=maybe", None),
        ("DELETE", "/v1/nosuchns/records/1", None),
        # erasure: the selection, everything derived from or citing it
        ("POST", f"/v1/{NS}/erase", {"namespace_id": "org.b", "entity_id": "user_7", "dry_run": True}),
        ("POST", f"/v1/{NS}/erase", {"ids": [11, 999999]}),     # 6001 cites 11: erased with it
        ("GET", f"/v1/{NS}/records/11", None),
        ("GET", f"/v1/{NS}/records/6001", None),
        ("POST", f"/v1/{NS}/erase", {"namespace_id": "org.b", "entity_id": "user_4"}),
        ("POST", f"/v1/{NS}/erase", {"namespace_id": "org", "prefix": True, "entity_id": "user_9"}),
        ("POST", f"/v1/{NS}/erase", {"ids": [13], "cites": [], "feeds": []}),
        ("GET", f"/v1/{NS}/records/14", None),
        ("POST", f"/v1/{NS}/erase", {}),
        ("POST", f"/v1/{NS}/erase", {"ids": [1], "bogus": 1}),
        ("POST", f"/v1/{NS}/erase", {"ids": "x"}),
        ("POST", "/v1/nosuchns/erase", {"ids": [1]}),
        ("POST", f"/v1/{NS}/search", {"vector": q, "k": 10}),
        # setup and close
        ("POST", f"/v1/{NS}/normalize_vectors", b""),
        ("POST", f"/v1/{NS}/normalize_vectors?modality=nope", b""),
        ("POST", "/v1/nosuchns/normalize_vectors", b""),
        ("POST", "/v1/namespaces", {"name": "created1", "dim": 32}),
        ("POST", "/v1/namespaces", {"name": "created2"}),
        ("POST", "/v1/namespaces", {"name": "bad.name"}),
        ("POST", "/v1/namespaces", {"name": "x", "dim": 0}),
        ("POST", "/v1/namespaces", {"name": "y"}, {"Idempotency-Key": "k"}),
        ("GET", "/v1/created1/admin/engine", None),
        # an idempotent write in a non-ASCII namespace: the stored note (and
        # so the WAL) must be byte-identical
        ("POST", "/v1/na%C3%AFve/vectors", {"id": 1, "vector": [1.0, 0.0]}, {"Idempotency-Key": "u1"}),
        ("GET", "/v1/na%C3%AFve/admin/engine", None),
        ("GET", f"/v1/{NS}/admin/engine", None),
        ("POST", f"/v1/{NS}/save", b""),
        ("POST", "/v1/nosuchns/save", b""),
        ("GET", "/v1/namespaces", None),
    ]
    failures = 0
    last_body = None
    last_etag = None
    for c in cases:
        method, path, body = c[0], c[1], c[2]
        headers = dict(c[3]) if len(c) > 3 else None
        if body is None and method in ("POST", "PUT"):
            body = last_body                   # the idempotent replay resends the same bytes
        if headers and headers.get("If-Match") == "@etag":
            headers["If-Match"] = last_etag    # what the previous response said (the same on both)
        raw = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
        last_body = raw
        a = call(py_port, method, path, raw, headers)
        b = call(rs_port, method, path, raw, headers)
        last_etag = {k.lower(): v for k, v in a[1].items()}.get("etag")
        diffs = []
        if a[0] != b[0]:
            diffs.append(f"status {a[0]} vs {b[0]}")
        for h in ("etag", "idempotent-replayed", "retry-after"):
            ha = {k.lower(): v for k, v in a[1].items()}.get(h)
            hb = {k.lower(): v for k, v in b[1].items()}.get(h)
            if ha != hb:
                diffs.append(f"header {h}: {ha!r} vs {hb!r}")
        try:
            ja, jb = json.loads(a[2]), json.loads(b[2])
        except ValueError:
            ja, jb = a[2], b[2]
        if a[0] >= 400 and isinstance(ja, dict) and isinstance(jb, dict):
            # Error bodies: same type and retryability. A 422's detail is
            # pydantic's wording on one side and serde's on the other.
            for k in ("error", "retryable"):
                if ja.get(k) != jb.get(k):
                    diffs.append(f"{k}: {ja.get(k)!r} vs {jb.get(k)!r}")
            if a[0] != 422 and ja.get("detail") != jb.get("detail"):
                diffs.append(f"detail: {ja.get('detail')!r} vs {jb.get('detail')!r}")
        elif method == "GET" and path == "/v1/namespaces":
            pass                               # the data dirs differ only by name
        else:
            diffs += same(ja, jb, ignore=RECEIPT_RUN if path.endswith("/erase") else frozenset())
        shown = body.decode(errors="replace") if isinstance(body, bytes) else json.dumps(body)
        tag = f"{method} {path} {shown[:60]}"
        if diffs:
            failures += 1
            print(f"DIFF  {tag}")
            for d in diffs[:12]:
                print(f"      {d}")
        else:
            print(f"same  {a[0]} {tag}")
    for p in (py, rs):
        p.terminate()
        p.wait(timeout=30)
    # Both servers checkpoint on shutdown: the files they leave must load and
    # hold the same records.
    for side in ("py", "rs"):
        db = feather_db.DB.open(os.path.join(dirs[side], f"{NS}.feather"), dim=DIM, read_only=True)
        dirs[side + "_ids"] = sorted(db.all_ids())
        db.close()
    if dirs["py_ids"] != dirs["rs_ids"]:
        failures += 1
        print("DIFF  files after shutdown hold different ids")
    print(f"\n{len(cases) - failures}/{len(cases)} cases identical" + ("" if failures else " — parity holds"))
    if keep:
        print("kept", work)
    else:
        shutil.rmtree(work, ignore_errors=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
