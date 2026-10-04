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
        assert r.status_code == 421 and r.json()["owner_shard"] == 1
        assert not (tmp_path / f"{other}.feather").exists()


def test_shard_owner_is_stable_across_processes():
    import subprocess
    code = (f"import sys; sys.path.insert(0, {API_DIR!r}); "
            "from app.db_manager import shard_owner; "
            "print([shard_owner(f'n{i}', 7) for i in range(20)])")
    runs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "FEATHER_DATA_DIR": "/tmp"}).stdout for _ in range(2)}
    assert len(runs) == 1 and runs.pop().strip()
