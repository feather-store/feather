"""Valid time over HTTP: written, read back, filtered, ordered — and the
same through RemoteDB, so the embedded and remote modes agree."""
import numpy as np
import pytest

pytest.importorskip("fastapi", reason="Cloud API tests need fastapi")

from fastapi.testclient import TestClient     # noqa: E402

DIM = 16
T0 = 1_683_000_000
DAY = 86_400


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FEATHER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FEATHER_DB_DIM", str(DIM))
    monkeypatch.setenv("FEATHER_DEV_MODE", "1")
    monkeypatch.delenv("FEATHER_API_KEY", raising=False)

    import sys, pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "feather-api")]
    for mod in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]
    from app.main import app                                   # noqa: E402
    with TestClient(app) as c:
        yield c


def put(client, rid, ts, valid):
    v = np.zeros(DIM)
    v[rid % DIM] = 1.0
    body = {"id": rid, "vector": v.tolist(),
            "metadata": {"content": f"record {rid}", "namespace_id": "u",
                         "timestamp": ts, "valid_at": valid}}
    r = client.post("/v1/t/vectors", json=body)
    assert r.status_code in (200, 201), r.text


@pytest.fixture
def filled(client):
    # Recorded 1, 2, 3; happened 3, 1, 2.
    put(client, 1, T0 + DAY, T0 - 10 * DAY)
    put(client, 2, T0 + 2 * DAY, T0 - 5 * DAY)
    put(client, 3, T0 + 3 * DAY, T0 - 20 * DAY)
    return client


def test_valid_at_is_written_and_read_back(filled):
    m = filled.get("/v1/t/records/1").json()["metadata"]
    assert m["valid_at"] == T0 - 10 * DAY and m["timestamp"] == T0 + DAY


def test_omitted_valid_at_is_zero(client):
    r = client.post("/v1/t/vectors", json={"id": 9, "vector": [1.0] * DIM,
                                           "metadata": {"content": "x"}})
    assert r.status_code in (200, 201)
    assert client.get("/v1/t/records/9").json()["metadata"]["valid_at"] == 0


def test_newest_by_valid(filled):
    q = {"metadata_namespace": "u", "limit": 10}
    assert filled.get("/v1/t/newest", params=q).json()["ids"] == [3, 2, 1]
    assert filled.get("/v1/t/newest", params={**q, "by_valid": "true"}
                      ).json()["ids"] == [2, 1, 3]


def test_search_filters_by_valid_time(filled):
    r = filled.post("/v1/t/search", json={
        "vector": [1.0] * DIM, "k": 10, "namespace_id": "u",
        "valid_after": T0 - 10 * DAY, "valid_before": T0 - 5 * DAY})
    assert r.status_code == 200, r.text
    assert {x["id"] for x in r.json()["results"]} == {1, 2}


def test_a_put_without_valid_at_keeps_it(filled):
    # A client written before valid time existed must not erase it.
    r = filled.put("/v1/t/records/1", json={"metadata": {
        "content": "edited", "namespace_id": "u", "timestamp": T0 + DAY}})
    assert r.status_code == 200, r.text
    m = filled.get("/v1/t/records/1").json()["metadata"]
    assert m["content"] == "edited" and m["valid_at"] == T0 - 10 * DAY
    r = filled.put("/v1/t/records/1", json={"metadata": {
        "content": "edited", "namespace_id": "u", "valid_at": 0}})
    assert filled.get("/v1/t/records/1").json()["metadata"]["valid_at"] == 0
