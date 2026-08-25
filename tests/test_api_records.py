"""Cloud API record-lifecycle regressions.

These exist because of a production incident: `GET /v1/{ns}/records` kept
listing records that `GET`/`DELETE /v1/{ns}/records/{id}` returned 404 for. The
listing checked only the "_deleted" attribute, while `db.forget()` — what DELETE
actually calls — marks a record by setting source="_forgotten". So deletes
succeeded, the listing never stopped showing the tombstones, the caller's
reconciler retried, got a correct 404, and concluded deletion was broken. The
record count never dropped because the listing was counting deleted records.

The invariant these tests protect: **every endpoint must agree on which records
exist.** No C++-level test could catch this — the disagreement was entirely in
the API layer.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(REPO, "feather-api")

fastapi = pytest.importorskip("fastapi", reason="Cloud API tests need fastapi")
pytest.importorskip("httpx", reason="fastapi TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

NS = "testns"
DIM = 16


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient over a throwaway data dir, with auth off (as in dev)."""
    monkeypatch.setenv("FEATHER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FEATHER_DB_DIM", str(DIM))
    monkeypatch.delenv("FEATHER_API_KEY", raising=False)
    monkeypatch.setenv("FEATHER_DEV_MODE", "1")   # no-auth now has to be explicit
    if API_DIR not in sys.path:
        sys.path.insert(0, API_DIR)
    # Drop cached modules so the patched env is picked up per test.
    for mod in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]
    from app.main import app  # noqa: WPS433
    with TestClient(app) as c:
        yield c


def _add(client, rec_id, content="reference asset", **attrs):
    body = {
        "id": rec_id,
        "vector": np.random.rand(DIM).astype(np.float32).tolist(),
        "metadata": {"content": content, "attributes": attrs},
    }
    r = client.post(f"/v1/{NS}/vectors", json=body)
    assert r.status_code in (200, 201), r.text
    return r


def _list_ids(client, limit=100):
    r = client.get(f"/v1/{NS}/records", params={"limit": limit})
    assert r.status_code == 200, r.text
    return [item["id"] for item in r.json()["results"]]


# ── The incident ──────────────────────────────────────────────────────────

def test_deleted_record_disappears_from_the_listing(client):
    """The exact production symptom: list showed it, get/delete 404'd on it."""
    for rid in (1, 2, 3):
        _add(client, rid)
    assert sorted(_list_ids(client)) == [1, 2, 3]

    assert client.delete(f"/v1/{NS}/records/2").status_code == 200
    assert 2 not in _list_ids(client), "deleted record still listed (the incident)"
    assert sorted(_list_ids(client)) == [1, 3]


def test_every_endpoint_agrees_on_which_records_exist(client):
    """Listing, stats and per-id fetch must not disagree. Drift between them is
    what produced 226 listed records where only a fraction were real."""
    for rid in range(1, 11):
        _add(client, rid)
    for rid in (2, 4, 6):
        assert client.delete(f"/v1/{NS}/records/{rid}").status_code == 200

    listed = set(_list_ids(client))
    assert listed == {1, 3, 5, 7, 8, 9, 10}

    stats = client.get(f"/v1/namespaces/{NS}/stats")
    assert stats.status_code == 200, stats.text
    assert stats.json()["records"] == len(listed), (
        "stats and listing disagree about the live record count")

    # Anything listed must be fetchable; anything deleted must 404.
    for rid in listed:
        assert client.get(f"/v1/{NS}/records/{rid}").status_code == 200, (
            f"listed id {rid} is not fetchable")
    for rid in (2, 4, 6):
        assert client.get(f"/v1/{NS}/records/{rid}").status_code == 404


def test_record_count_actually_drops_after_delete(client):
    """'Record count stays 226 after the delete' was the reported tell."""
    for rid in range(1, 6):
        _add(client, rid)
    before = client.get(f"/v1/namespaces/{NS}/stats").json()["records"]
    client.delete(f"/v1/{NS}/records/3")
    after = client.get(f"/v1/namespaces/{NS}/stats").json()["records"]
    assert after == before - 1, f"count did not drop: {before} -> {after}"


def test_deleting_twice_is_a_404_not_a_silent_success(client):
    _add(client, 1)
    assert client.delete(f"/v1/{NS}/records/1").status_code == 200
    assert client.delete(f"/v1/{NS}/records/1").status_code == 404


def test_batch_delete_removes_all_ids_from_the_listing(client):
    for rid in range(1, 8):
        _add(client, rid)
    r = client.post(f"/v1/{NS}/records/batch_delete", json={"ids": [2, 3, 5]})
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] == 3
    assert sorted(_list_ids(client)) == [1, 4, 6, 7]


def test_deleted_records_stay_gone_across_a_reload(client):
    """A tombstone must not come back when the namespace is re-opened."""
    for rid in (1, 2, 3):
        _add(client, rid)
    client.delete(f"/v1/{NS}/records/2")
    assert client.post(f"/v1/{NS}/save").status_code == 200
    assert 2 not in _list_ids(client)


# ── Behaviours the incident report asked about ────────────────────────────

def test_posting_an_existing_id_upserts_rather_than_duplicating(client):
    """Q4 from the report. Re-POSTing an id must update in place."""
    _add(client, 1, content="original", s3_key="old/path.png")
    _add(client, 1, content="updated", s3_key="new/path.png")

    assert _list_ids(client) == [1], "re-POSTing an id created a duplicate"
    meta = client.get(f"/v1/{NS}/records/1").json()
    assert meta["content"] == "updated"
    assert meta["attributes"]["s3_key"] == "new/path.png"


def test_put_updates_attributes_without_touching_the_vector(client):
    """Q3: repoint a file path without re-embedding."""
    _add(client, 1, content="asset", s3_key="old/path.png")
    before = client.post(f"/v1/{NS}/search",
                         json={"vector": np.random.rand(DIM).tolist(), "k": 1}).json()

    r = client.put(f"/v1/{NS}/records/1",
                   json={"metadata": {"content": "asset",
                                      "attributes": {"s3_key": "new/path.png"}}})
    assert r.status_code == 200, r.text
    assert client.get(f"/v1/{NS}/records/1").json()["attributes"]["s3_key"] == "new/path.png"

    after = client.post(f"/v1/{NS}/search",
                        json={"vector": np.random.rand(DIM).tolist(), "k": 1}).json()
    assert before["results"][0]["id"] == after["results"][0]["id"] == 1


def test_deleted_records_do_not_come_back_in_any_search_path(client):
    """The downstream harm: a deleted asset resurfacing in a retrieval pool and
    handing callers a dead URL. Vector, keyword and hybrid must all exclude it.
    """
    _add(client, 1, content="alpha reference render")
    _add(client, 2, content="beta reference render")
    client.delete(f"/v1/{NS}/records/2")

    q = np.random.rand(DIM).tolist()
    for path, body in (
        (f"/v1/{NS}/search", {"vector": q, "k": 10}),
        (f"/v1/{NS}/keyword_search", {"query": "reference render", "k": 10}),
        (f"/v1/{NS}/hybrid_search", {"vector": q, "query": "reference render", "k": 10}),
    ):
        r = client.post(path, json=body)
        assert r.status_code == 200, f"{path}: {r.text}"
        ids = [item["id"] for item in r.json()["results"]]
        assert 2 not in ids, f"deleted record resurfaced via {path}"
