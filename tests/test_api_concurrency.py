"""Cloud API changes from the concurrency work (0.19).

* vector_b64 input, include_metadata, raw_score computed in the engine
* deletes/purges no longer rewrite the namespace file (WAL + background checkpoint)
* edge pruning on delete covers records in any modality
* ingest_text: async, micro-batched, never pads/truncates vectors
* sharding: a process answers 421 for namespaces it doesn't own
"""
import base64
import os
import sys
import threading
import time

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi", reason="Cloud API tests need fastapi")
pytest.importorskip("httpx", reason="fastapi TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(REPO, "feather-api")
NS = "cc"
DIM = 16


def _client(tmp_path, monkeypatch, **env):
    monkeypatch.setenv("FEATHER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FEATHER_DB_DIM", str(DIM))
    monkeypatch.delenv("FEATHER_API_KEY", raising=False)
    monkeypatch.setenv("FEATHER_DEV_MODE", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    if API_DIR not in sys.path:
        sys.path.insert(0, API_DIR)
    for mod in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]
    from app.main import app  # noqa: WPS433
    return TestClient(app)


@pytest.fixture
def client(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, FEATHER_EMBED_PROVIDER="mock",
                 FEATHER_EMBED_DIM=DIM) as c:
        yield c


def _b64(v):
    return base64.b64encode(np.asarray(v, dtype="<f4").tobytes()).decode()


def _seed(client, n=20):
    rng = np.random.default_rng(0)
    V = rng.random((n, DIM)).astype(np.float32)
    items = [{"id": i + 1, "vector": V[i].tolist(), "metadata": {"content": f"doc {i} alpha"}}
             for i in range(n)]
    assert client.post(f"/v1/{NS}/import", json={"items": items}).status_code == 200
    return V


def test_vector_b64_matches_json_vector(client):
    V = _seed(client)
    a = client.post(f"/v1/{NS}/search", json={"vector": V[3].tolist(), "k": 5, "track": False}).json()
    b = client.post(f"/v1/{NS}/search", json={"vector_b64": _b64(V[3]), "k": 5, "track": False}).json()
    assert [r["id"] for r in a["results"]] == [r["id"] for r in b["results"]]
    assert a["results"][0]["id"] == 4


def test_vector_and_b64_are_mutually_exclusive(client):
    V = _seed(client, 3)
    r = client.post(f"/v1/{NS}/search", json={"vector": V[0].tolist(), "vector_b64": _b64(V[0])})
    assert r.status_code == 422
    assert client.post(f"/v1/{NS}/search", json={"k": 3}).status_code == 422
    assert client.post(f"/v1/{NS}/search", json={"vector_b64": "!!notb64"}).status_code == 422


def test_msgspec_bodies_validate_like_pydantic(client):
    """Hot routes decode with msgspec; bounds, types and bad JSON still 422."""
    V = _seed(client, 5)
    q = V[0].tolist()
    assert client.post(f"/v1/{NS}/search", json={"vector": q, "k": 0}).status_code == 422
    assert client.post(f"/v1/{NS}/search", json={"vector": q, "k": 1001}).status_code == 422
    assert client.post(f"/v1/{NS}/search", json={"vector": q, "k": "ten"}).status_code == 422
    assert client.post(f"/v1/{NS}/search", json={"vector": ["x"] * 16}).status_code == 422
    r = client.post(f"/v1/{NS}/search", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422 and "JSON" in r.text
    # unknown top-level keys are ignored, exactly like pydantic's default
    ok = client.post(f"/v1/{NS}/search", json={"vector": q, "k": 2, "unexpected": 1})
    assert ok.status_code == 200 and ok.json()["count"] == 2
    # pydantic's lax coercions still work: existing clients send these
    lax = {"vector": [str(x) for x in q], "k": "2", "track": "true",
           "importance_gte": "0", "timestamp_after": 0.0}
    ok = client.post(f"/v1/{NS}/search", json=lax)
    assert ok.status_code == 200 and ok.json()["count"] == 2, ok.text
    assert client.post(f"/v1/{NS}/hybrid_search", json={"vector": q, "k": 2}).status_code == 422  # no query
    assert client.post(f"/v1/{NS}/keyword_search", json={"query": "doc", "k": 2}).status_code == 200
    assert client.post(f"/v1/{NS}/context_chain", json={"k": 2, "hops": 11}).status_code == 422
    assert client.post(f"/v1/{NS}/import", json={"items": "nope"}).status_code == 422


def test_openapi_still_documents_request_bodies(client):
    spec = client.get("/openapi.json").json()
    body = spec["paths"]["/v1/{namespace}/search"]["post"]["requestBody"]
    ref = body["content"]["application/json"]["schema"]["$ref"]
    assert ref.endswith("/SearchRequest")
    schemas = spec["components"]["schemas"]
    assert "vector_b64" in schemas["SearchRequest"]["properties"]
    assert "MetadataIn" in schemas              # nested model of AddVectorRequest


def test_include_metadata_false_returns_ids_and_scores_only(client):
    V = _seed(client)
    r = client.post(f"/v1/{NS}/search",
                    json={"vector": V[0].tolist(), "k": 3, "include_metadata": False}).json()
    assert r["count"] == 3
    assert all(set(x) == {"id", "score", "cosine"} for x in r["results"])
    full = client.post(f"/v1/{NS}/search", json={"vector": V[0].tolist(), "k": 3}).json()
    md = full["results"][0]["metadata"]
    assert md["content"].startswith("doc ") and "attributes" in md and "links" in md


def test_raw_score_is_true_cosine(client):
    V = _seed(client)
    q = V[5] + 0.01
    r = client.post(f"/v1/{NS}/search", json={"vector": q.tolist(), "k": 5, "raw_score": True}).json()
    for hit in r["results"]:
        v = V[hit["id"] - 1]
        expect = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v)))
        assert abs(hit["cosine"] - expect) < 1e-4
    r2 = client.post(f"/v1/{NS}/search", json={"vector": q.tolist(), "k": 2}).json()
    assert all(h["cosine"] is None for h in r2["results"])


def test_delete_does_not_rewrite_the_namespace_file(client, tmp_path):
    _seed(client)
    assert client.post(f"/v1/{NS}/flush").status_code == 200
    path = tmp_path / f"{NS}.feather"
    before = path.stat().st_mtime_ns
    time.sleep(0.02)
    assert client.delete(f"/v1/{NS}/records/3").status_code == 200
    assert path.stat().st_mtime_ns == before, "DELETE rewrote the whole namespace file"
    assert client.get(f"/v1/{NS}/records/3").status_code == 404


def test_delete_prunes_edges_from_any_modality(client):
    _seed(client, 5)
    # record 100 lives only in a non-text modality and links to record 2
    body = {"id": 100, "vector": [0.5] * 8, "modality": "visual", "metadata": {"content": "img"}}
    assert client.post(f"/v1/{NS}/vectors", json=body).status_code == 201
    assert client.post(f"/v1/{NS}/records/100/link", json={"to_id": 2}).status_code == 200
    r = client.delete(f"/v1/{NS}/records/2").json()
    assert r["edges_pruned"] == 1
    assert client.get(f"/v1/{NS}/records/100").json()["links"] == []


def test_ingest_text_embeds_via_batcher(client):
    rs = [client.post(f"/v1/{NS}/ingest_text", json={"text": f"note {i}"}) for i in range(5)]
    assert all(r.status_code == 200 for r in rs), [r.text for r in rs]
    assert all(r.json()["dim"] == DIM for r in rs)
    q = client.post(f"/v1/{NS}/keyword_search", json={"query": "note", "k": 10}).json()
    assert q["count"] == 5


def test_embedding_dim_mismatch_is_an_error_not_padding(tmp_path, monkeypatch):
    # mock provider emits FEATHER_EMBED_DIM floats; the namespace is fixed at 16
    with _client(tmp_path, monkeypatch, FEATHER_EMBED_PROVIDER="mock", FEATHER_EMBED_DIM=24) as c:
        body = {"id": 1, "vector": [0.1] * DIM, "metadata": {"content": "x"}}
        assert c.post(f"/v1/{NS}/vectors", json=body).status_code == 201
        r = c.post(f"/v1/{NS}/ingest_text", json={"text": "hello"})
        assert r.status_code == 400 and "24" in r.text


def test_provider_dim_mismatch_fits_by_default_and_errors_when_strict(monkeypatch):
    """A provider vector of the wrong length is padded/truncated (the pre-0.19
    behaviour existing namespaces were filled with) unless
    FEATHER_EMBED_STRICT_DIM=1, which raises and asks providers for the dim."""
    if API_DIR not in sys.path:
        sys.path.insert(0, API_DIR)
    from app.embedding import _Cfg, _build, _check_dims  # noqa: WPS433
    cfg = _Cfg(provider="openai", model="text-embedding-3-small", key="k", base_url="",
               deployment="", api_version="", dim=4)
    monkeypatch.delenv("FEATHER_EMBED_STRICT_DIM", raising=False)
    assert _check_dims(cfg, [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]) == \
        [[1.0, 2.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]]
    assert "dimensions" not in _build(cfg, ["x"])[2]
    monkeypatch.setenv("FEATHER_EMBED_STRICT_DIM", "1")
    with pytest.raises(RuntimeError, match="never padded"):
        _check_dims(cfg, [[1.0, 2.0]])
    assert _build(cfg, ["x"])[2]["dimensions"] == 4


def test_mock_provider_requires_dev_mode(monkeypatch):
    if API_DIR not in sys.path:
        sys.path.insert(0, API_DIR)
    monkeypatch.setenv("FEATHER_DEV_MODE", "")
    for mod in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]
    from app.embedding import EmbeddingProvider  # noqa: WPS433
    p = EmbeddingProvider()
    p.update(provider="mock", dim=8)
    with pytest.raises(RuntimeError, match="DEV_MODE"):
        p.embed("x")


def test_shard_returns_421_for_foreign_namespace(tmp_path, monkeypatch):
    if API_DIR not in sys.path:
        sys.path.insert(0, API_DIR)
    for mod in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]
    from app.db_manager import shard_owner  # noqa: WPS433
    mine = next(n for n in (f"ns{i}" for i in range(100)) if shard_owner(n, 2) == 0)
    other = next(n for n in (f"ns{i}" for i in range(100)) if shard_owner(n, 2) == 1)
    with _client(tmp_path, monkeypatch, FEATHER_SHARD_COUNT=2, FEATHER_SHARD_INDEX=0) as c:
        ok = c.post(f"/v1/{mine}/vectors", json={"id": 1, "vector": [0.1] * DIM})
        assert ok.status_code == 201
        r = c.post(f"/v1/{other}/vectors", json={"id": 1, "vector": [0.1] * DIM})
        assert r.status_code == 421 and r.json()["owner_shard"] == "1"     # shard ids are strings
        assert not (tmp_path / f"{other}.feather").exists()


def test_shard_owner_is_stable_across_processes():
    import subprocess
    code = (f"import sys; sys.path.insert(0, {API_DIR!r}); "
            "from app.db_manager import shard_owner; "
            "print([shard_owner(f'n{i}', 7) for i in range(20)])")
    runs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "FEATHER_DATA_DIR": "/tmp"}).stdout for _ in range(2)}
    assert len(runs) == 1 and runs.pop().strip()


def test_empty_metadata_object_still_stamps_now(client):
    """`"metadata": {}` was a MetadataIn() before the msgspec path, which sets
    timestamp=now; a bare engine Metadata() would date the record 1970."""
    before = int(time.time())
    r = client.post(f"/v1/{NS}/vectors", json={"id": 1, "vector": [0.1] * DIM, "metadata": {}})
    assert r.status_code == 201
    assert client.get(f"/v1/{NS}/records/1").json()["timestamp"] >= before


def test_metadata_type_error_detail_matches_pydantic(client):
    r = client.post(f"/v1/{NS}/vectors", json={"id": 1, "vector": [0.1] * DIM, "metadata": [1]})
    assert r.status_code == 422
    d = r.json()["detail"][0]
    assert d["type"] == "model_attributes_type" and d["loc"] == ["body", "metadata"]


def test_one_bad_text_does_not_fail_the_rest_of_its_batch():
    """Providers reject a whole request for one invalid input; each caller must
    only ever see its own text's outcome."""
    import asyncio
    if API_DIR not in sys.path:
        sys.path.insert(0, API_DIR)
    from app.embedding import EmbedBatcher  # noqa: WPS433

    class Provider:
        calls = 0

        async def aembed_many(self, texts):
            Provider.calls += 1
            if "BAD" in texts:
                raise RuntimeError("HTTP 400: input invalid")
            return [[float(len(t))] for t in texts]

    async def run():
        b = EmbedBatcher(Provider())
        b.window_s = 0.05
        b.start()
        res = await asyncio.gather(*[b.embed(t) for t in ("aa", "BAD", "cccc", "d")],
                                   return_exceptions=True)
        await b.stop()
        return res

    res = asyncio.run(run())
    assert res[0] == [2.0] and res[2] == [4.0] and res[3] == [1.0]
    assert isinstance(res[1], RuntimeError) and "input invalid" in str(res[1])


def test_embedding_imports_do_not_hold_the_write_pool(tmp_path, monkeypatch):
    """Server-side embedding runs at provider latency; on the 8-thread write
    pool a few such imports stalled every write on the server."""
    with _client(tmp_path, monkeypatch, FEATHER_EMBED_PROVIDER="mock", FEATHER_EMBED_DIM=DIM,
                 FEATHER_WRITE_THREADS=1) as c:
        import app.main as m  # noqa: WPS433
        m.EMBEDDING._mock_delay_s = 0
        orig = m.EMBEDDING.embed_many
        gate = threading.Event()

        def slow(texts):
            gate.wait(10)
            return orig(texts)
        m.EMBEDDING.embed_many = slow
        t = threading.Thread(target=lambda: c.post(
            f"/v1/{NS}/import", json={"items": [{"id": 1, "metadata": {"content": "x"}}]}))
        t.start()
        time.sleep(0.3)                                   # import is now blocked embedding
        t0 = time.time()
        r = c.post(f"/v1/{NS}/vectors", json={"id": 2, "vector": [0.1] * DIM})
        waited = time.time() - t0
        gate.set(); t.join(10)
        m.EMBEDDING.embed_many = orig
        assert r.status_code == 201 and waited < 2, f"add waited {waited:.1f}s behind an embedding import"


def test_confidence_gte_filters_search_keyword_and_hybrid(client):
    """Additive: absent, every route behaves as before."""
    for i, conf in enumerate((0.2, 0.5, 0.9), start=1):
        client.post(f"/v1/{NS}/vectors", json={"id": i, "vector": [0.1 * i] * DIM,
                                               "metadata": {"content": "shared word", "confidence": conf}})
    q = [0.1] * DIM
    ids = lambda r: sorted(x["id"] for x in r.json()["results"])
    assert ids(client.post(f"/v1/{NS}/search", json={"vector": q, "k": 10})) == [1, 2, 3]
    assert ids(client.post(f"/v1/{NS}/search", json={"vector": q, "k": 10, "confidence_gte": 0.5})) == [2, 3]
    assert ids(client.post(f"/v1/{NS}/keyword_search", json={"query": "shared", "k": 10, "confidence_gte": 0.9})) == [3]
    assert ids(client.post(f"/v1/{NS}/hybrid_search", json={"vector": q, "query": "shared", "k": 10,
                                                             "confidence_gte": 0.5})) == [2, 3]
