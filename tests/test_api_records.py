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


# ── Pagination must not be spent on tombstones ────────────────────────────
# Filed by the dev team against 0.16.0: tombstones were interleaved with live
# records and counted against `limit`, so a paging client received mostly
# ghosts. The first fix excluded them from the listing — but excluded them
# AFTER slicing to `limit`, which turned "your page is full of ghosts" into
# "your page is empty". Measured on 47 live records behind 127 tombstones:
# limit=10/50/100 each returned 0 rows. Filtering has to happen before the cut.

def _seed_churned(client, live=8, deleted=12):
    """Deleted ids sort BEFORE live ones, so a naive slice sees only tombstones."""
    for rid in range(deleted + live):
        _add(client, rid, content=f"record {rid}")
    for rid in range(deleted):
        assert client.delete(f"/v1/{NS}/records/{rid}").status_code == 200
    return deleted, live


def test_a_small_page_returns_live_records_not_an_empty_page(client):
    """The regression: every early id is a tombstone, so slicing first yields 0."""
    _seed_churned(client, live=8, deleted=12)
    rows = _list_ids(client, limit=5)
    assert len(rows) == 5, f"page of 5 returned {len(rows)} rows — tombstones ate the page"
    assert all(r >= 12 for r in rows), f"a tombstone was returned: {rows}"


def test_every_page_size_sees_all_live_records(client):
    """Correctness must not depend on over-fetching. Before the fix only a
    limit >= total-ever-written returned the full live set."""
    deleted, live = _seed_churned(client, live=8, deleted=12)
    for limit in (1, 3, 5, 8, 20, 100):
        rows = _list_ids(client, limit=limit)
        assert len(rows) == min(limit, live), (
            f"limit={limit} returned {len(rows)} rows, expected {min(limit, live)}")


def test_paging_through_a_churned_namespace_yields_each_live_record_once(client):
    deleted, live = _seed_churned(client, live=8, deleted=12)
    seen, cursor, guard = [], -1, 0
    while guard < 50:
        guard += 1
        r = client.get(f"/v1/{NS}/records", params={"limit": 3, "after": cursor})
        body = r.json()
        seen += [x["id"] for x in body["results"]]
        if not body["has_more"]:
            break
        cursor = body["next_cursor"]
    assert sorted(seen) == list(range(12, 20)), f"paging lost or duplicated records: {seen}"
    assert len(seen) == len(set(seen)), "a record was returned on two pages"


def test_include_deleted_is_opt_in_and_flags_rows_explicitly(client):
    """The team asked not to have to recognise an undocumented sentinel."""
    _seed_churned(client, live=3, deleted=2)
    r = client.get(f"/v1/{NS}/records", params={"limit": 100, "include_deleted": "true"})
    rows = r.json()["results"]
    assert len(rows) == 5, "include_deleted=true should surface tombstones too"
    flags = {row["id"]: row.get("deleted") for row in rows}
    assert flags[0] is True and flags[1] is True, "tombstones not flagged deleted=true"
    assert flags[2] is False, "live record wrongly flagged deleted"


def test_has_more_is_false_once_the_live_records_run_out(client):
    """A trailing run of tombstones must not advertise another page."""
    for rid in range(10):
        _add(client, rid)
    for rid in range(5, 10):
        client.delete(f"/v1/{NS}/records/{rid}")
    body = client.get(f"/v1/{NS}/records", params={"limit": 50}).json()
    assert [x["id"] for x in body["results"]] == [0, 1, 2, 3, 4]
    assert body["has_more"] is False, "claimed another page when only tombstones remain"


# ── Integration feedback: the API must not fail silently ──────────────────
# All five reproduced against v0.17.0 by a team integrating Feather as the
# vector store for creative analysis. The common thread is silence: the server
# returned 200 and the caller found out much later, or never.

def test_import_that_stores_nothing_is_not_a_200(client):
    """A backfill that wrote zero vectors answered a flat 200, so a caller
    checking the status recorded a successful import of nothing."""
    _add(client, 1)                                   # establishes dim
    r = client.post(f"/v1/{NS}/import",
                    json={"items": [{"id": 99, "vector": [0.1] * 4}]})
    assert r.status_code == 400, f"stored nothing but returned {r.status_code}"
    detail = r.json()["detail"]
    assert detail["inserted"] == 0 and detail["skipped"] == 1
    assert "dim mismatch" in " ".join(detail["errors"]).lower()


def test_partial_import_reports_207_not_200(client):
    _add(client, 1)
    good = np.random.rand(DIM).astype(np.float32).tolist()
    r = client.post(f"/v1/{NS}/import", json={"items": [
        {"id": 10, "vector": good},
        {"id": 11, "vector": [0.1] * 4},          # wrong dim
    ]})
    assert r.status_code == 207, f"partial import returned {r.status_code}"
    body = r.json()
    assert body["inserted"] == 1 and body["skipped"] == 1
    assert body["partial"] is True


def test_unknown_metadata_keys_are_rejected_by_name(client):
    """They were accepted, reported inserted, and absent on read back. The team
    only noticed after building a UI and seeing every media URL come back empty."""
    _add(client, 1)
    vec = np.random.rand(DIM).astype(np.float32).tolist()
    r = client.post(f"/v1/{NS}/import", json={"items": [
        {"id": 2, "vector": vec,
         "metadata": {"content": "x", "creative_hash": "abc", "creative_url": "https://e/x.jpg"}},
    ]})
    assert r.status_code == 400
    errs = " ".join(r.json()["detail"]["errors"])
    assert "creative_hash" in errs and "creative_url" in errs, "offending keys not named"
    assert "attributes" in errs, "error should point at the supported alternative"

    # The same guard on the single-add route.
    r2 = client.post(f"/v1/{NS}/vectors", json={
        "id": 3, "vector": vec, "metadata": {"content": "x", "creative_hash": "abc"}})
    assert r2.status_code == 422, "unknown key accepted on /vectors"


def test_a_zero_vector_query_is_rejected(client):
    """An empty string embeds to exactly this. The engine ranks by L2 and
    returned confident-looking rows for a query with no direction."""
    for rid in (1, 2, 3):
        _add(client, rid)
    r = client.post(f"/v1/{NS}/search", json={"vector": [0.0] * DIM, "k": 3})
    assert r.status_code == 400, f"zero-norm query returned {r.status_code}"
    assert "zero" in r.json()["detail"].lower()


def test_stored_vectors_can_be_read_back(client):
    """Without this, 'find records like this one' had to re-embed text that had
    already been embedded — an API call and a cost per lookup."""
    _add(client, 1)
    plain = client.get(f"/v1/{NS}/records/1").json()
    assert "vector" not in plain, "vector returned without being asked for"

    withvec = client.get(f"/v1/{NS}/records/1", params={"include_vector": "true"}).json()
    assert len(withvec["vector"]) == DIM


def test_single_get_can_be_read_like_every_other_route(client):
    """Client code reading record["metadata"]["content"] worked for search hits
    and silently returned nothing for a single get."""
    _add(client, 1, content="hello")
    got = client.get(f"/v1/{NS}/records/1").json()
    assert got["id"] == 1
    assert got["metadata"]["content"] == "hello"     # nested, like search
    assert got["content"] == "hello"                 # flat, as before — non-breaking


def test_repeated_identical_import_errors_are_collapsed(client):
    """A misconfigured provider produced one copy of the same message per item;
    a 1,000-item import returned 1,000 identical strings."""
    _add(client, 1)
    items = [{"id": 900 + i, "metadata": {"content": f"n{i}"}} for i in range(6)]
    r = client.post(f"/v1/{NS}/import", json={"items": items})
    errs = r.json().get("detail", r.json()).get("errors", [])
    assert len(errs) == 1, f"expected the repeats collapsed, got {len(errs)}"
    assert "5 more" in errs[0], f"collapsed message should say how many: {errs[0]}"
