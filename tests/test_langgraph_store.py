"""FeatherStore — Feather as a LangGraph `BaseStore`.

LangGraph has two memory sockets. `BaseCheckpointSaver` holds per-run graph
state; `BaseStore` holds long-term, cross-thread memory — what an agent should
still know next week. Feather is the second one, and `BaseStore` needs exactly
two methods, so this adapter is what moves Feather from "a search index an agent
queries" to "the thing an agent remembers into".

Three bugs surfaced the first time the prototype ran, and each has a test here:
delete used a sentinel that `db.forget()` does not set, and namespace-PREFIX
search returned nothing on both the semantic and the filter-only path because
Feather's namespace index is exact-match.
"""
import asyncio
import hashlib
import time

import numpy as np
import pytest

pytest.importorskip("langgraph", reason="FeatherStore needs langgraph")

from feather_db.integrations.langgraph_store import FeatherStore  # noqa: E402

DIM = 128


def _embed(text: str):
    """Deterministic stand-in so the suite needs no API key."""
    seed = int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).normal(0, 1, DIM).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def store(tmp_path):
    s = FeatherStore(str(tmp_path / "agent.feather"), dim=DIM, embed=_embed)
    yield s
    s.close()


NS = ("acme", "user_1", "preferences")


# ── the basics ────────────────────────────────────────────────────────────

def test_put_then_get_round_trips_the_value(store):
    store.put(NS, "comms", {"kind": "preference", "text": "async only", "confidence": 0.9})
    it = store.get(NS, "comms")
    assert it is not None
    assert it.value["text"] == "async only"
    assert it.value["confidence"] == 0.9        # numbers survive the JSON round trip
    assert it.namespace == NS and it.key == "comms"


def test_get_of_a_missing_key_is_none(store):
    assert store.get(NS, "nope") is None


def test_put_is_an_upsert_not_an_append(store):
    """The property that makes re-running an indexer safe. It depends on the id
    being sha1(namespace+key) — a time-seeded id would grow the store forever."""
    store.put(NS, "comms", {"text": "v1", "confidence": 0.5})
    store.put(NS, "comms", {"text": "v2", "confidence": 0.9})
    assert len(store.search(NS, limit=50)) == 1
    assert store.get(NS, "comms").value["text"] == "v2"


def test_created_at_is_preserved_across_an_update(store):
    """An agent needs to tell when it first learned something from when it last
    confirmed it. Feather has one timestamp, so created_at lives in an attribute."""
    store.put(NS, "comms", {"text": "v1"})
    first = store.get(NS, "comms").created_at
    store.put(NS, "comms", {"text": "v2"})
    again = store.get(NS, "comms")
    assert again.created_at == first
    assert again.updated_at >= first


# ── the three bugs the prototype found ────────────────────────────────────

def test_delete_actually_deletes(store):
    """Bug 1: the adapter invented its own sentinel while db.forget() sets
    source="_forgotten", so get() kept returning deleted items."""
    store.put(NS, "tone", {"text": "blunt"})
    assert store.get(NS, "tone") is not None
    store.delete(NS, "tone")
    assert store.get(NS, "tone") is None
    assert all(i.key != "tone" for i in store.search(NS, limit=50))


def test_semantic_search_reaches_down_the_namespace_tree(store):
    """Bug 2: Feather's namespace index is exact-match, so searching a PARENT
    namespace silently returned nothing."""
    store.put(("acme", "user_1", "preferences"), "comms", {"text": "prefers async written updates"})
    store.put(("acme", "user_1", "episodes"), "ep1", {"text": "cancelled the campaign early"})

    hits = store.search(("acme", "user_1"), query="how do they like to communicate", limit=5)
    assert hits, "prefix search over a parent namespace returned nothing"
    assert {h.namespace[-1] for h in hits} >= {"preferences", "episodes"}


def test_filter_only_search_reaches_down_the_tree(store):
    """Bug 3: same root cause on the non-semantic path, different symptom."""
    store.put(("acme", "user_1", "episodes"), "ep1", {"kind": "episode", "text": "a"})
    store.put(("acme", "user_2", "episodes"), "ep2", {"kind": "episode", "text": "b"})
    store.put(("acme", "user_1", "preferences"), "p1", {"kind": "preference", "text": "c"})

    eps = store.search(("acme",), filter={"kind": "episode"}, limit=20)
    assert {e.key for e in eps} == {"ep1", "ep2"}


# ── search semantics ──────────────────────────────────────────────────────

def test_operator_filters(store):
    store.put(NS, "a", {"confidence": 0.95, "text": "high"})
    store.put(NS, "b", {"confidence": 0.40, "text": "low"})

    assert {i.key for i in store.search(NS, filter={"confidence": {"$gte": 0.9}}, limit=9)} == {"a"}
    assert {i.key for i in store.search(NS, filter={"confidence": {"$lt": 0.5}}, limit=9)} == {"b"}
    assert {i.key for i in store.search(NS, filter={"text": "high"}, limit=9)} == {"a"}


def test_search_results_carry_a_score(store):
    store.put(NS, "a", {"text": "prefers written communication"})
    hits = store.search(NS, query="communication style", limit=3)
    assert hits and hits[0].score is not None


def test_search_does_not_leak_across_sibling_namespaces(store):
    store.put(("acme", "user_1", "preferences"), "a", {"text": "alpha"})
    store.put(("acme", "user_2", "preferences"), "b", {"text": "beta"})
    hits = store.search(("acme", "user_1"), limit=20)
    assert {h.key for h in hits} == {"a"}


def test_limit_and_offset(store):
    for i in range(6):
        store.put(NS, f"k{i}", {"text": f"item {i}", "n": i})
    page1 = store.search(NS, limit=2)
    page2 = store.search(NS, limit=2, offset=2)
    assert len(page1) == 2 and len(page2) == 2
    assert {i.key for i in page1}.isdisjoint({i.key for i in page2})


def test_list_namespaces(store):
    store.put(("acme", "user_1", "preferences"), "a", {"text": "x"})
    store.put(("acme", "brand_a", "facts"), "b", {"text": "y"})
    ns = store.list_namespaces()
    assert ("acme", "user_1", "preferences") in ns
    assert ("acme", "brand_a", "facts") in ns


# ── it is still just a file ───────────────────────────────────────────────

def test_memory_survives_a_restart(tmp_path):
    path = str(tmp_path / "persist.feather")
    s1 = FeatherStore(path, dim=DIM, embed=_embed)
    s1.put(("acme", "brand_a", "facts"), "positioning", {"text": "premium minimalist"})
    s1.close()

    s2 = FeatherStore(path, dim=DIM, embed=_embed)
    assert s2.get(("acme", "brand_a", "facts"), "positioning").value["text"] == "premium minimalist"
    s2.close()


def test_works_without_an_embedder(tmp_path):
    """No embed function means no semantic search, but get/put/filter must still
    work — a store that needs an API key to hold a preference is not useful."""
    s = FeatherStore(str(tmp_path / "noembed.feather"), dim=DIM)
    s.put(NS, "a", {"kind": "preference", "text": "x"})
    assert s.get(NS, "a").value["text"] == "x"
    assert len(s.search(NS, filter={"kind": "preference"}, limit=9)) == 1
    s.close()


# ── the LangGraph contract, beyond the happy path ─────────────────────────
# Four behaviours the interface specifies that the first implementation either
# silently ignored or actively inverted. Each was found by exercising the
# contract rather than the code.

def test_ttl_is_declared_and_enforced(store):
    """BaseStore.put() refuses a ttl unless the subclass sets supports_ttl, and
    raises before reaching batch() — so Feather's own ttl and forget_expired()
    were unreachable through the LangGraph API even though both already worked.
    """
    assert store.supports_ttl is True

    store.put(NS, "short", {"text": "expires"}, ttl=0.001)   # sub-second
    store.put(NS, "long", {"text": "stays"}, ttl=60)
    store.put(NS, "never", {"text": "no ttl"})
    assert store.get(NS, "short") is not None

    time.sleep(1.4)
    assert store.get(NS, "short") is None, "expired item still readable"
    assert store.get(NS, "long") is not None
    assert store.get(NS, "never") is not None


def test_a_sub_second_ttl_does_not_become_permanent(store):
    """LangGraph ttl is in MINUTES, Feather stores whole SECONDS, and 0 means
    'never expires'. Truncating turned any ttl under a second into permanent —
    the worst possible direction for a field whose purpose is impermanence."""
    store.put(NS, "tiny", {"text": "x"}, ttl=0.0001)         # 6 ms
    rid = store.db.get_metadata(
        __import__("feather_db.integrations.langgraph_store", fromlist=["_ns_key"])
        ._ns_key(NS, "tiny"))
    assert rid.ttl >= 1, "a positive ttl was floored to 0, which means forever"


def test_expired_items_are_excluded_from_search(store):
    store.put(NS, "gone", {"text": "alpha expiring"}, ttl=0.001)
    store.put(NS, "kept", {"text": "alpha staying"})
    time.sleep(1.4)
    keys = {i.key for i in store.search(NS, limit=10)}
    assert keys == {"kept"}


def test_index_false_stores_but_does_not_make_it_findable(store):
    """The placeholder vector is IDENTICAL for every unindexed record, so
    without a marker they match each other perfectly and an index=False item
    came back as a TOP semantic hit — the opposite of what was asked."""
    store.put(NS, "hidden", {"text": "secret internal note"}, index=False)
    store.put(NS, "visible", {"text": "a normal memory"})

    assert store.get(NS, "hidden") is not None, "index=False must still store it"
    semantic = {i.key for i in store.search(NS, query="secret internal note", limit=5)}
    assert "hidden" not in semantic

    # but it is still there for a filter-only listing
    assert "hidden" in {i.key for i in store.search(NS, limit=10)}


def test_list_namespaces_honours_prefix(store):
    store.put(("acme", "u1", "prefs"), "a", {"x": 1})
    store.put(("acme", "u2", "prefs"), "b", {"x": 1})
    store.put(("other", "u3", "prefs"), "c", {"x": 1})

    got = store.list_namespaces(prefix=("acme",))
    assert got and all(ns[0] == "acme" for ns in got)
    assert ("other", "u3", "prefs") not in got


def test_list_namespaces_honours_suffix(store):
    store.put(("acme", "u1", "prefs"), "a", {"x": 1})
    store.put(("acme", "u1", "episodes"), "b", {"x": 1})

    got = store.list_namespaces(suffix=("prefs",))
    assert ("acme", "u1", "prefs") in got
    assert ("acme", "u1", "episodes") not in got


def test_list_namespaces_wildcard(store):
    store.put(("acme", "u1", "prefs"), "a", {"x": 1})
    store.put(("acme", "u2", "prefs"), "b", {"x": 1})
    got = store.list_namespaces(prefix=("acme", "*", "prefs"))
    assert len(got) == 2


def test_list_namespaces_filters_before_truncating(store):
    """Truncating first would let ("a","b","c") match a suffix on ("b",) at
    max_depth=2 — a match against a path the caller never stored."""
    store.put(("a", "b", "c"), "k", {"x": 1})
    assert store.list_namespaces(suffix=("b",), max_depth=2) == []


def test_the_async_path_works(store):
    async def go():
        await store.aput(NS, "async_key", {"text": "written via aput"})
        got = await store.aget(NS, "async_key")
        hits = await store.asearch(NS, query="written via", limit=3)
        return got, hits

    got, hits = asyncio.run(go())
    assert got is not None and got.value["text"] == "written via aput"
    assert any(h.key == "async_key" for h in hits)


def test_batch_applies_every_op_in_order(store):
    from langgraph.store.base import GetOp, PutOp
    ops = [PutOp(NS, f"b{i}", {"n": i}, None, None) for i in range(5)]
    ops.append(GetOp(NS, "b2"))
    results = store.batch(ops)
    assert results[-1] is not None and results[-1].value["n"] == 2
    assert len(store.search(NS, limit=20)) == 5
