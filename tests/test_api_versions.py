"""Cloud API: If-Match, Idempotency-Key, and atomic POST /batch.

  * a write based on a stale read is refused (412 / 409), not applied over
    the write it never saw;
  * a retried request with the same Idempotency-Key gets the first response
    and is not applied again — including a DELETE retried after it succeeded,
    an ingest_text that would otherwise mint a second random id, and a retry
    after the server restarted;
  * POST /batch writes every op or none.
"""
import contextlib
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(REPO, "feather-api")

pytest.importorskip("fastapi", reason="Cloud API tests need fastapi")
pytest.importorskip("httpx", reason="fastapi TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

NS = "vns"
DIM = 8


@contextlib.contextmanager
def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("FEATHER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FEATHER_DB_DIM", str(DIM))
    monkeypatch.delenv("FEATHER_API_KEY", raising=False)
    monkeypatch.setenv("FEATHER_DEV_MODE", "1")
    monkeypatch.setenv("FEATHER_EMBED_PROVIDER", "mock")
    monkeypatch.setenv("FEATHER_EMBED_DIM", str(DIM))
    if API_DIR not in sys.path:
        sys.path.insert(0, API_DIR)
    for mod in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]
    from app.main import app  # noqa: WPS433
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as c:
        yield c


def _vec(seed):
    return np.random.default_rng(seed).random(DIM).tolist()


def _add(client, rec_id, content="x", **headers):
    return client.post(f"/v1/{NS}/vectors", headers=headers,
                       json={"id": rec_id, "vector": _vec(rec_id), "metadata": {"content": content}})


def _get(client, rec_id):
    return client.get(f"/v1/{NS}/records/{rec_id}")


# ── If-Match ──────────────────────────────────────────────────────────────

def test_a_stale_if_match_is_refused(client):
    v1 = _add(client, 1, "first").json()["version"]
    r = _get(client, 1)
    assert r.json()["version"] == v1 and r.headers["ETag"] == f'"{v1}"'

    # Agent A and agent B both read v1. A writes first; B's write must fail.
    a = client.put(f"/v1/{NS}/records/1", headers={"If-Match": f'"{v1}"'},
                   json={"metadata": {"content": "from A"}})
    assert a.status_code == 200, a.text
    v2 = a.json()["version"]
    assert v2 > v1 and a.headers["ETag"] == f'"{v2}"'
    b = client.put(f"/v1/{NS}/records/1", headers={"If-Match": f'"{v1}"'},
                   json={"metadata": {"content": "from B"}})
    assert b.status_code == 412, b.text
    assert b.headers["ETag"] == f'"{v2}"'
    assert _get(client, 1).json()["content"] == "from A", "B overwrote A"

    imp = client.put(f"/v1/{NS}/records/1/importance", headers={"If-Match": str(v1)},
                     json={"importance": 0.1})
    assert imp.status_code == 412
    assert client.put(f"/v1/{NS}/records/1", headers={"If-Match": "nope"},
                      json={"metadata": {"content": "x"}}).status_code == 400


def test_writes_without_if_match_still_work(client):
    _add(client, 1)
    assert client.put(f"/v1/{NS}/records/1", json={"metadata": {"content": "y"}}).status_code == 200
    assert client.put(f"/v1/{NS}/records/1/importance", json={"importance": 0.3}).status_code == 200
    assert client.put(f"/v1/{NS}/records/9/importance", json={"importance": 0.3}).status_code == 404


# ── Idempotency-Key ───────────────────────────────────────────────────────

def test_a_retried_put_is_not_applied_over_a_later_write(client):
    _add(client, 1, "original")
    k = {"Idempotency-Key": "put-1"}
    first = client.put(f"/v1/{NS}/records/1", headers=k, json={"metadata": {"content": "attempt"}})
    client.put(f"/v1/{NS}/records/1", json={"metadata": {"content": "later, by someone else"}})
    retry = client.put(f"/v1/{NS}/records/1", headers=k, json={"metadata": {"content": "attempt"}})
    assert retry.status_code == 200 and retry.json() == first.json()
    assert retry.headers.get("Idempotent-Replayed") == "true"
    assert "Idempotent-Replayed" not in first.headers
    assert _get(client, 1).json()["content"] == "later, by someone else"


def test_the_same_key_with_another_body_is_422(client):
    assert _add(client, 1, "a", **{"Idempotency-Key": "k"}).status_code == 201
    r = _add(client, 2, "b", **{"Idempotency-Key": "k"})
    assert r.status_code == 422, r.text
    assert _get(client, 2).status_code == 404


def test_a_retried_delete_gets_the_first_answer_not_404(client):
    _add(client, 1)
    _add(client, 2)
    client.post(f"/v1/{NS}/records/2/link", json={"to_id": 1})
    k = {"Idempotency-Key": "del-1"}
    first = client.delete(f"/v1/{NS}/records/1", headers=k)
    assert first.status_code == 200 and first.json()["edges_pruned"] == 1
    retry = client.delete(f"/v1/{NS}/records/1", headers=k)
    assert retry.status_code == 200 and retry.json() == first.json()
    assert client.delete(f"/v1/{NS}/records/1").status_code == 404   # no key: a real 404


def test_a_retried_ingest_does_not_create_a_second_record(client):
    k = {"Idempotency-Key": "ingest-1"}
    first = client.post(f"/v1/{NS}/ingest_text", headers=k, json={"text": "a note"})
    retry = client.post(f"/v1/{NS}/ingest_text", headers=k, json={"text": "a note"})
    assert first.status_code == retry.status_code == 200, retry.text
    assert retry.json()["id"] == first.json()["id"]
    hits = client.post(f"/v1/{NS}/keyword_search", json={"query": "note", "k": 10}).json()
    assert hits["count"] == 1


def test_a_retried_batch_delete_and_unlink(client):
    for i in range(1, 5):
        _add(client, i)
    client.post(f"/v1/{NS}/records/4/link", json={"to_id": 1})
    k = {"Idempotency-Key": "bd"}
    body = {"ids": [1, 2], "cascade": True}
    first = client.post(f"/v1/{NS}/records/batch_delete", headers=k, json=body)
    assert first.json()["deleted"] == 2 and first.json()["edges_pruned"] == 1
    retry = client.post(f"/v1/{NS}/records/batch_delete", headers=k, json=body)
    assert retry.json() == first.json()
    client.post(f"/v1/{NS}/records/4/link", json={"to_id": 3})
    k = {"Idempotency-Key": "ul"}
    assert client.delete(f"/v1/{NS}/records/4/link/3", headers=k).json()["removed"] == 1
    assert client.delete(f"/v1/{NS}/records/4/link/3", headers=k).json()["removed"] == 1


def test_keys_outlive_a_restart(tmp_path, monkeypatch):
    k = {"Idempotency-Key": "across-restart"}
    with _client(tmp_path, monkeypatch) as c:
        first = _add(c, 1, "a", **k)
    with _client(tmp_path, monkeypatch) as c:
        retry = _add(c, 1, "a", **k)
        assert retry.json() == first.json() and retry.headers.get("Idempotent-Replayed") == "true"


def test_a_key_on_a_route_that_cannot_honour_it_is_refused(client):
    _add(client, 1)
    r = client.post(f"/v1/{NS}/import", headers={"Idempotency-Key": "imp"},
                    json={"items": [{"id": 1, "vector": _vec(1)}]})
    assert r.status_code == 400 and "Idempotency-Key" in r.text
    # Reads ignore it.
    s = client.post(f"/v1/{NS}/search", headers={"Idempotency-Key": "s"}, json={"vector": _vec(1), "k": 1})
    assert s.status_code == 200


def test_link_follows_embedded_rules(client):
    """As embedded link(): the source must exist (404 — it used to say
    "linked" and link nothing); the target may be a forward reference. A /batch
    link needs both, unless it asks otherwise."""
    _add(client, 1)
    assert client.post(f"/v1/{NS}/records/99/link", json={"to_id": 1}).status_code == 404
    assert client.post(f"/v1/{NS}/records/1/link", json={"to_id": 99}).status_code == 200
    assert _get(client, 1).json()["links"] == [99]
    strict = client.post(f"/v1/{NS}/batch", json={"ops": [{"op": "link", "id": 1, "to_id": 98}]})
    assert strict.status_code == 400
    lenient = client.post(f"/v1/{NS}/batch", json={"ops": [
        {"op": "link", "id": 1, "to_id": 98, "require_target": False}]})
    assert lenient.status_code == 200


# ── POST /batch ───────────────────────────────────────────────────────────

def _fact_ops(fact, entity):
    return [
        {"op": "put", "id": entity, "vector": _vec(entity), "metadata": {"content": f"entity {entity}"}},
        {"op": "put", "id": fact, "vector": _vec(fact),
         "metadata": {"content": "acme raised a series B", "attributes": {"source_doc": "memo-7"}}},
        {"op": "link", "id": fact, "to_id": entity, "rel_type": "about"},
    ]


def test_batch_writes_a_fact_and_its_links(client):
    r = client.post(f"/v1/{NS}/batch", json={"ops": _fact_ops(10, 20)})
    assert r.status_code == 200, r.text
    res = r.json()["results"]
    assert [x["id"] for x in res] == [20, 10, 10]
    assert _get(client, 10).json()["links"] == [20]
    assert _get(client, 10).json()["version"] == res[-1]["version"]


@pytest.mark.parametrize("bad, status", [
    ({"op": "link", "id": 10, "to_id": 404}, 400),                         # dangling edge
    ({"op": "put", "id": 30, "vector": [0.1] * (DIM + 1)}, 400),          # wrong dim
    ({"op": "update", "id": 1, "metadata": {"content": "z"}, "expected_version": 1}, 409),
])
def test_a_bad_op_writes_nothing(client, bad, status):
    _add(client, 1, "keep")
    # Move record 1 past version 1, so the stale expected_version=1 conflicts.
    client.put(f"/v1/{NS}/records/1", json={"metadata": {"content": "keep"}})
    r = client.post(f"/v1/{NS}/batch", json={"ops": _fact_ops(10, 20) + [bad]})
    assert r.status_code == status, r.text
    for rid in (10, 20, 30):
        assert _get(client, rid).status_code == 404, f"record {rid} was written by a refused batch"
    assert _get(client, 1).json()["content"] == "keep"
    if status == 409:
        body = r.json()
        assert body["error"] == "version_conflict" and not body["retryable"]
        assert body["id"] == 1 and body["actual_version"] == _get(client, 1).json()["version"]


def test_batch_create_only_and_retry(client):
    ops = [{"op": "put", "id": 5, "vector": _vec(5), "expected_version": 0}]
    k = {"Idempotency-Key": "b1"}
    first = client.post(f"/v1/{NS}/batch", headers=k, json={"ops": ops})
    assert first.status_code == 200
    retry = client.post(f"/v1/{NS}/batch", headers=k, json={"ops": ops})
    assert retry.status_code == 200 and retry.json() == first.json()
    again = client.post(f"/v1/{NS}/batch", json={"ops": ops})      # no key: create-only refuses
    assert again.status_code == 409
