"""Cloud API: typed errors, 429 backpressure, nothing blocking the event loop.

  * every error says what to do: `error` (type) and `retryable`;
  * past the in-flight limit a request is refused at once with 429 +
    Retry-After, instead of queueing until the client times out and retries;
  * an upload's copy and load run off the event loop, and outside the
    DBManager lock: other requests, other namespaces and new namespaces keep
    being served while a large file loads, and the namespace being loaded
    answers 503 namespace_loading.
"""
import contextlib
import os
import sys
import threading
import time

import numpy as np
import pytest

import feather_db

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(REPO, "feather-api")

pytest.importorskip("fastapi", reason="Cloud API tests need fastapi")
pytest.importorskip("httpx", reason="fastapi TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

NS = "bp"
DIM = 8


@contextlib.contextmanager
def _client(tmp_path, monkeypatch, **env):
    monkeypatch.setenv("FEATHER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("FEATHER_DB_DIM", str(DIM))
    monkeypatch.delenv("FEATHER_API_KEY", raising=False)
    monkeypatch.setenv("FEATHER_DEV_MODE", "1")
    monkeypatch.setenv("FEATHER_EMBED_PROVIDER", "mock")
    monkeypatch.setenv("FEATHER_EMBED_DIM", str(DIM))
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    if API_DIR not in sys.path:
        sys.path.insert(0, API_DIR)
    for mod in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]
    from app import main  # noqa: WPS433
    with TestClient(main.app) as c:
        yield c, main


@pytest.fixture
def api(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as c:
        yield c


def _add(c, rec_id, ns=NS):
    return c.post(f"/v1/{ns}/vectors", json={"id": rec_id, "vector": [0.1] * DIM})


def _in_parallel(*fns):
    out = [None] * len(fns)

    def run(i, f):
        out[i] = f()
    ts = [threading.Thread(target=run, args=(i, f)) for i, f in enumerate(fns)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


# ── typed errors ──────────────────────────────────────────────────────────

def test_errors_carry_a_type_and_whether_to_retry(api):
    c, _ = api
    _add(c, 1)
    nf = c.get(f"/v1/{NS}/records/99")
    assert nf.status_code == 404
    assert nf.json()["error"] == "not_found" and nf.json()["retryable"] is False
    assert "detail" in nf.json()                       # the field existing clients read
    bad = c.post(f"/v1/{NS}/vectors", json={"id": "x"})
    assert bad.status_code == 422 and bad.json()["error"] == "invalid_request"
    ns = c.get("/v1/bad.name/records/1")
    assert ns.status_code == 400 and ns.json()["error"] == "invalid_namespace"


def test_wrong_shard_is_421_and_names_the_owner(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, FEATHER_SHARD_COUNT=2, FEATHER_SHARD_INDEX=0) as (c, main):
        from app.db_manager import shard_owner  # noqa: WPS433
        ns = next(f"t{i}" for i in range(100) if shard_owner(f"t{i}", 2) == 1)
        r = _add(c, 1, ns=ns)
        assert r.status_code == 421
        body = r.json()
        assert body["error"] == "wrong_shard" and body["retryable"] is False
        assert body["owner_shard"] == "1" and body["this_shard"] == "0"


def test_a_rate_limited_provider_is_503_not_the_clients_fault(api, monkeypatch):
    c, main = api
    from app.embedding import UpstreamError  # noqa: WPS433

    async def limited(text):
        raise UpstreamError("https://provider -> HTTP 429: slow down", 429)

    async def refused(text):
        raise UpstreamError("https://provider -> HTTP 401: bad key", 401)

    monkeypatch.setattr(main.EMBED_BATCHER, "embed", limited)
    r = c.post(f"/v1/{NS}/ingest_text", json={"text": "x"})
    assert r.status_code == 503 and r.json()["error"] == "upstream_unavailable" and r.json()["retryable"]
    assert r.headers["Retry-After"]
    monkeypatch.setattr(main.EMBED_BATCHER, "embed", refused)
    r = c.post(f"/v1/{NS}/ingest_text", json={"text": "x"})
    assert r.status_code == 502 and r.json()["error"] == "upstream_error" and not r.json()["retryable"]


# ── backpressure ──────────────────────────────────────────────────────────

def test_past_the_write_limit_requests_get_429_and_reads_still_flow(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, FEATHER_MAX_INFLIGHT_WRITES=2,
                 FEATHER_EMBED_MOCK_DELAY_MS=600) as (c, main):
        _add(c, 1)
        ingest = lambda: c.post(f"/v1/{NS}/ingest_text", json={"text": "slow"})   # noqa: E731
        reads = []

        def read_while_busy():
            time.sleep(0.2)
            reads.append(c.post(f"/v1/{NS}/search", json={"vector": [0.1] * DIM, "k": 1}))
            return None

        rs = _in_parallel(ingest, ingest, ingest, ingest, read_while_busy)[:4]
        codes = sorted(r.status_code for r in rs)
        assert codes == [200, 200, 429, 429], codes
        refused = next(r for r in rs if r.status_code == 429)
        assert refused.headers["Retry-After"] == "1"
        assert refused.json()["error"] == "overloaded" and refused.json()["retryable"] is True
        assert reads[0].status_code == 200, "reads were refused because writes were busy"
        adm = c.get("/health").json()["admission"]
        assert adm["write"]["rejected"] == 2 and adm["write"]["inflight"] == 0


# ── the event loop and the DBManager lock ─────────────────────────────────

def _feather_file(tmp_path, n=200):
    path = str(tmp_path / "local.feather")
    db = feather_db.DB.open(path, dim=DIM)
    db.add_batch(list(range(n)), np.random.default_rng(0).random((n, DIM), dtype=np.float32))
    db.close()
    return open(path, "rb").read()


def _upload(c, ns, data):
    return c.post("/v1/admin/upload", data={"namespace": ns, "overwrite": "true"},
                  files={"file": ("x.feather", data, "application/octet-stream")})


def test_a_slow_upload_does_not_stall_other_requests(tmp_path, monkeypatch):
    data = _feather_file(tmp_path)
    with _client(tmp_path, monkeypatch) as (c, main):
        _add(c, 1, ns="other")
        import app.db_manager as dbm  # noqa: WPS433
        real = dbm.DB

        class SlowDB:                       # a large file: the load takes a while
            @staticmethod
            def open(path, **kw):
                if "uploaded" in path:
                    time.sleep(1.5)
                return real.open(path, **kw)
        monkeypatch.setattr(dbm, "DB", SlowDB)

        timings = {}

        def timed(name, fn):
            def run():
                time.sleep(0.3)             # land while the load is running
                t = time.perf_counter()
                r = fn()
                timings[name] = (time.perf_counter() - t, r)
            return run

        up, *_ = _in_parallel(
            lambda: _upload(c, "uploaded", data),
            timed("health", lambda: c.get("/health")),
            timed("other", lambda: c.get("/v1/other/records/1")),
            timed("new", lambda: _add(c, 1, ns="brandnew")),
            timed("loading", lambda: c.get("/v1/uploaded/records/1")),
        )
        assert up.status_code == 200 and up.json()["records"] == 200, up.text
        for name in ("health", "other", "new"):
            took, r = timings[name]
            assert r.status_code < 300, (name, r.text)
            assert took < 0.5, f"{name} waited {took:.2f}s behind the upload's load"
        took, r = timings["loading"]
        assert r.status_code == 503 and r.json()["error"] == "namespace_loading", r.text
        assert r.headers["Retry-After"] and took < 0.5
        assert c.get("/v1/uploaded/records/1").status_code == 200


def test_a_failed_upload_load_rolls_back_to_the_old_file(tmp_path, monkeypatch):
    data = _feather_file(tmp_path)
    with _client(tmp_path, monkeypatch) as (c, main):
        assert _upload(c, "keep", data).status_code == 200
        _add(c, 9999, ns="keep")
        # A valid header over a corrupt section: it passes the upload's header
        # check and fails the load, which is what reaches the rollback.
        bad = bytearray(data)
        for i in range(9000, 9064):
            bad[i] ^= 0xFF
        bad = bytes(bad)
        r = _upload(c, "keep", bad)
        assert r.status_code == 400, r.text
        assert c.get("/v1/keep/records/5").status_code == 200, "the old namespace was lost"
        assert c.get("/v1/keep/records/9999").status_code == 200, "a write only in the WAL was lost"
        assert not any(p.endswith(".upload") for p in os.listdir(tmp_path / "data"))
