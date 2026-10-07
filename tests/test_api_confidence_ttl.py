"""`confidence` and `ttl` over HTTP.

Both have been engine fields for a long time and both are persisted, but
neither was reachable through the Cloud API: `MetadataIn` sets
`extra="forbid"`, so a POST carrying either was rejected with 422, and
`MetadataOut` omitted them, so a client could not read them back either.

That blocked two things outright. A caller could not record how certain a fact
was — made worse in 0.21.0, which added `confidence_gte` and so shipped a
filter for a field the API refused to accept. And there was no way to write an
expiring record at all, which is exactly what a short-lived agent run scope
needs.

The `extra="forbid"` behaviour itself is right and is left alone: silently
dropping unknown keys behind a 200 is how an integration once stored nothing
and only noticed after building a UI on the empty results.
"""
import numpy as np
import pytest

pytest.importorskip("fastapi", reason="Cloud API tests need fastapi")

from fastapi.testclient import TestClient     # noqa: E402

DIM = 16


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
    # `with TestClient(...)` so startup runs — without it `manager` is still
    # None and every route fails on NoneType.
    with TestClient(app) as c:
        yield c


def vec(seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(DIM).tolist()


def put(client, rid, **meta):
    body = {"id": rid, "vector": vec(rid),
            "metadata": {"content": f"record {rid}", **meta}}
    return client.post("/v1/t/vectors", json=body)


# ── writable ──────────────────────────────────────────────────────────────

def test_confidence_can_be_written_and_read_back(client):
    assert put(client, 1, confidence=0.42).status_code in (200, 201)
    got = client.get("/v1/t/records/1").json()["metadata"]
    assert got["confidence"] == pytest.approx(0.42, abs=1e-6)


def test_ttl_can_be_written_and_read_back(client):
    assert put(client, 2, ttl=1209600).status_code in (200, 201)
    assert client.get("/v1/t/records/2").json()["metadata"]["ttl"] == 1209600


def test_both_default_so_existing_callers_are_unaffected(client):
    """A client that never sends them must keep working, and must not suddenly
    find its records expiring."""
    assert put(client, 3).status_code in (200, 201)
    m = client.get("/v1/t/records/3").json()["metadata"]
    assert m["confidence"] == 1.0        # certain unless told otherwise
    assert m["ttl"] == 0                 # 0 means never expires


# ── validated ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_confidence_outside_zero_to_one_is_rejected(client, bad):
    """It is a probability-shaped field; 1.5 confident is not a thing, and
    letting it through would silently skew every `confidence_gte` filter."""
    r = put(client, 4, confidence=bad)
    assert r.status_code == 422
    assert "confidence" in r.text


def test_a_negative_ttl_is_rejected(client):
    """A negative ttl would make `timestamp + ttl` land in the past, so the
    record would be born expired."""
    r = put(client, 5, ttl=-1)
    assert r.status_code == 422
    assert "ttl" in r.text


def test_unknown_keys_are_still_rejected_by_name(client):
    """The guard these two fields were caught by stays on for everything else."""
    r = put(client, 6, creative_url="https://example.com/a.png")
    assert r.status_code == 422
    assert "creative_url" in r.text


# ── visible everywhere a record is ────────────────────────────────────────

def test_search_results_carry_them_too(client):
    put(client, 7, confidence=0.33, ttl=600)
    r = client.post("/v1/t/search", json={"vector": vec(7), "k": 3})
    assert r.status_code == 200
    hit = next(x for x in r.json()["results"] if x["id"] == 7)
    assert hit["metadata"]["confidence"] == pytest.approx(0.33, abs=1e-6)
    assert hit["metadata"]["ttl"] == 600


def test_listing_carries_them_too(client):
    """Feather Desk reads the listing, not search, to lay out a scope."""
    put(client, 8, confidence=0.77, ttl=99)
    row = next(x for x in client.get("/v1/t/records?limit=50").json()["results"]
               if x["id"] == 8)
    assert row["metadata"]["confidence"] == pytest.approx(0.77, abs=1e-6)
    assert row["metadata"]["ttl"] == 99


def test_they_survive_a_save_and_reload(client):
    put(client, 9, confidence=0.5, ttl=1234)
    client.post("/v1/t/save")
    m = client.get("/v1/t/records/9").json()["metadata"]
    assert m["confidence"] == pytest.approx(0.5, abs=1e-6)
    assert m["ttl"] == 1234
