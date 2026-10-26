"""The five kinds of agent memory, and the lifetimes that distinguish them.

A generic store will happily let you overwrite an episode or append a preference.
Both are silent data defects: the first destroys history, the second leaves the
agent holding several contradictory answers with no way to tell which is current.
These tests pin the write semantics per kind, because that is the entire reason
this layer exists on top of a correct `BaseStore`.
"""
import time

import numpy as np
import pytest

pytest.importorskip("langgraph", reason="needs langgraph")

from feather_db.integrations.langgraph_store import FeatherStore   # noqa: E402
from feather_db.integrations.agent_memory import AgentMemory       # noqa: E402


def embedder(dim=64):
    """Bag-of-words, so related text is actually near — a hash-random embedder
    has no semantics and would make every recall assertion meaningless."""
    def embed(text: str):
        v = np.zeros(dim, dtype=np.float32)
        for tok in str(text).lower().split():
            v[hash(tok) % dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v
    return embed


@pytest.fixture
def mem(tmp_path):
    store = FeatherStore(str(tmp_path / "m.feather"), dim=64, embed=embedder())
    return AgentMemory(store, org="hawky", user="user_842",
                       agent="creative", brand="nike")


# ── preferences: revised in place ─────────────────────────────────────────

def test_a_preference_is_revised_not_accumulated(mem):
    mem.preferences.set("comms", "prefers phone calls")
    mem.preferences.set("comms", "prefers written async updates")

    assert mem.preferences.get("comms")["text"] == "prefers written async updates"
    assert len(mem.preferences) == 1, "the revision was stored alongside the old answer"


# ── episodes: append-only ─────────────────────────────────────────────────

def test_episodes_accumulate_and_never_overwrite(mem):
    mem.episodes.record("q3 report shipped late", outcome="missed_deadline",
                        at=time.time() - 86400)
    mem.episodes.record("q4 report shipped on time", outcome="ok")

    assert len(mem.episodes) == 2
    assert mem.episodes.recent(1)[0]["text"] == "q4 report shipped on time"


def test_two_identical_episodes_are_still_two_events(mem):
    """Same text, different moments. Collapsing them loses that it recurred."""
    now = time.time()
    mem.episodes.record("deploy failed", at=now - 600)
    mem.episodes.record("deploy failed", at=now)
    assert len(mem.episodes) == 2


def test_the_same_episode_twice_in_one_second_is_two_events(mem):
    """The key is `<second>_<slug>`, so a retry in the same second used to
    produce the same key and replace the first record."""
    now = time.time()
    k1 = mem.episodes.record("deploy failed", at=now, outcome="first")
    k2 = mem.episodes.record("deploy failed", at=now, outcome="second")
    k3 = mem.episodes.record("deploy failed", at=now, outcome="third")
    assert k1 == f"{int(now)}_deploy_failed", "an uncontested key changed format"
    assert len({k1, k2, k3}) == 3 and len(mem.episodes) == 3
    assert {e["outcome"] for e in mem.episodes.all()} == {"first", "second", "third"}


# ── facts: corrected, with history ────────────────────────────────────────

def test_correcting_a_fact_keeps_what_it_replaced(mem):
    mem.facts.state("positioning", "value brand", source="2024 deck")
    mem.facts.state("positioning", "premium, never discount-led", source="2026 deck")

    assert mem.facts.get("positioning")["text"] == "premium, never discount-led"
    history = mem.facts.history("positioning")
    assert len(history) == 1
    assert history[0]["text"] == "value brand"
    assert history[0]["source"] == "2024 deck"


def test_restating_a_fact_unchanged_does_not_fabricate_history(mem):
    mem.facts.state("positioning", "premium")
    mem.facts.state("positioning", "premium")
    assert mem.facts.history("positioning") == []


# ── procedures: versioned ─────────────────────────────────────────────────

def test_a_redefined_procedure_keeps_the_version_that_worked(mem):
    mem.procedures.define("weekly_report", ["pull numbers", "send"])
    mem.procedures.define("weekly_report", ["pull numbers", "sanity check", "send"],
                          checks=["roas is not null"])

    current = mem.procedures.get("weekly_report")
    assert current["version"] == 2
    assert "sanity check" in current["steps"]

    first = mem.procedures.version("weekly_report", 1)
    assert first["steps"] == ["pull numbers", "send"]


# ── entities: enriched ────────────────────────────────────────────────────

def test_enriching_an_entity_merges_rather_than_replaces(mem):
    mem.entities.enrich("nike", {"industry": "apparel"})
    mem.entities.enrich("nike", {"tier": "enterprise"})

    nike = mem.entities.get("nike")
    assert nike["attributes"] == {"industry": "apparel", "tier": "enterprise"}
    assert nike["seen_count"] == 2


def test_entities_are_org_wide_not_per_user(mem):
    """An entity learned while serving one user is known for the next."""
    assert mem.entities.namespace == ("hawky", "entities")


# ── observations: the reflection layer ────────────────────────────────────

def test_an_observation_cites_its_evidence(mem):
    mem.observations.note("comms_style",
                          "consistently prefers written async over synchronous",
                          evidence=["comms", "ep_2026_09_14"], confidence=0.92)
    o = mem.observations.get("comms_style")
    assert o["confidence"] == 0.92
    assert "comms" in o["evidence"]


# ── the namespace tree ────────────────────────────────────────────────────

def test_subject_comes_before_kind_so_about_is_one_prefix_read(mem):
    """If kind came before subject, 'everything about this user' would be a
    filter over every user in the org instead of a prefix."""
    assert mem.preferences.namespace == ("hawky", "user_842", "preferences")
    assert mem.episodes.namespace == ("hawky", "user_842", "episodes")
    assert mem.procedures.namespace == ("hawky", "creative", "procedures")
    assert mem.facts.namespace == ("hawky", "nike", "facts")


def test_about_returns_every_kind_for_the_subject(mem):
    mem.preferences.set("comms", "written async")
    mem.episodes.record("kickoff call went well", outcome="ok")
    mem.observations.note("style", "decisive", confidence=0.7)

    about = mem.about()
    assert set(about) == {"preferences", "episodes", "observations"}
    assert about["preferences"][0]["text"] == "written async"


def test_about_does_not_leak_another_user(tmp_path):
    store = FeatherStore(str(tmp_path / "m.feather"), dim=64, embed=embedder())
    a = AgentMemory(store, org="hawky", user="user_a")
    b = AgentMemory(store, org="hawky", user="user_b")
    a.preferences.set("comms", "user a likes email")

    assert b.about() == {}
    assert "likes email" not in str(b.about())
    assert "likes email" in str(a.about())           # it is genuinely stored
    assert b.recall("email") == []                   # and not reachable by search


# ── recall ────────────────────────────────────────────────────────────────

def test_recall_can_be_narrowed_to_one_kind(mem):
    mem.preferences.set("comms", "prefers written async updates")
    mem.episodes.record("sent written async update, it worked well")

    only_prefs = mem.recall("written async", kinds=["preferences"])
    assert len(only_prefs) == 1
    assert only_prefs[0]["kind"] == "preferences"

    both = mem.recall("written async")
    assert len(both) >= 2


# ── the guardrail ─────────────────────────────────────────────────────────

def test_a_kind_whose_subject_was_never_given_says_which_one(tmp_path):
    """Writing a procedure with no agent set must name `agent`, not fail with
    some opaque tuple error three frames down."""
    store = FeatherStore(str(tmp_path / "m.feather"), dim=64, embed=embedder())
    mem = AgentMemory(store, org="hawky", user="u1")       # no agent=

    with pytest.raises(ValueError, match="agent"):
        mem.procedures.define("anything", ["a step"])

    mem.preferences.set("comms", "still works")            # unaffected


def test_summary_counts_only_the_kinds_that_are_usable(mem):
    mem.preferences.set("comms", "written")
    mem.episodes.record("something happened")
    s = mem.summary()
    assert s["preferences"] == 1 and s["episodes"] == 1
    assert s["facts"] == 0


def test_org_is_required():
    with pytest.raises(ValueError, match="org"):
        AgentMemory(None, org="")


# ── it is still a plain BaseStore underneath ──────────────────────────────

def test_the_underlying_store_sees_the_same_records(mem):
    mem.preferences.set("comms", "written async")
    item = mem.store.get(("hawky", "user_842", "preferences"), "comms")
    assert item.value["text"] == "written async"
    assert item.value["kind"] == "preferences"


def test_memory_survives_reopening_the_file(tmp_path):
    path = str(tmp_path / "m.feather")
    store = FeatherStore(path, dim=64, embed=embedder())
    AgentMemory(store, org="hawky", user="u1").preferences.set("comms", "written async")
    store.close()

    reopened = AgentMemory(FeatherStore(path, dim=64, embed=embedder()),
                           org="hawky", user="u1")
    assert reopened.preferences.get("comms")["text"] == "written async"


# ── the example must keep working ─────────────────────────────────────────

def test_the_langgraph_example_still_runs(tmp_path, monkeypatch):
    """`examples/langgraph_agent_memory.py` is the gate the spec set: memory
    added to an existing agent by changing one line. An example that no longer
    runs is worse than no example, so it is executed here rather than read."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples"))
    import langgraph_agent_memory as ex

    store = FeatherStore(str(tmp_path / "ex.feather"), dim=64, embed=ex.embedder())
    graph = ex.build_graph(store)

    graph.invoke({"message": "I prefer written async updates, not calls"},
                 config={"configurable": {"thread_id": "t1"}})
    # a different thread: a checkpointer would remember nothing here
    out = graph.invoke({"message": "what do you know about me?"},
                       config={"configurable": {"thread_id": "t2"}})

    assert "written async" in out["reply"]
    assert "2 time(s)" in out["reply"]


# ── ordering and counting at scale ────────────────────────────────────────
# These exist because the first implementation was wrong and the tests above
# did not catch it: with two episodes, "the newest 1 of an arbitrary 8" and
# "the newest 1" are the same answer. They stop being the same at 300.

def test_recent_returns_the_actually_newest_episodes(mem):
    """`recent()` used to ask search() for limit*4 items and sort those.
    search() has no ordering, so it sorted an arbitrary subset: at 300 episodes
    the newest 5 came back as 299, 298, 296, 295, 293 — silently skipping two.
    """
    base = time.time()
    for i in range(300):
        mem.episodes.record(f"event {i}", at=base + i)

    assert [v["text"] for v in mem.episodes.recent(5)] == \
        [f"event {i}" for i in (299, 298, 297, 296, 295)]


def test_episodes_all_orders_by_event_time_not_write_time(mem):
    """The generic all() sorts on the engine timestamp, which is whole seconds,
    so 300 records written inside one second tie and order arbitrarily — that
    returned 298, 297, 296 for the newest three. Episodes override it."""
    base = time.time()
    for i in range(300):
        mem.episodes.record(f"event {i}", at=base + i)

    assert [v["text"] for v in mem.episodes.all(3)] == \
        [f"event {i}" for i in (299, 298, 297)]


def test_a_backdated_episode_sorts_by_when_it_happened(mem):
    """`at=` exists so a past event can be recorded now. Sorting on write time
    would put it first, which is the opposite of the truth."""
    now = time.time()
    mem.episodes.record("happened last year", at=now - 365 * 86400)
    mem.episodes.record("happened an hour ago", at=now - 3600)

    assert mem.episodes.recent(1)[0]["text"] == "happened an hour ago"


def test_len_is_exact_beyond_the_old_thousand_item_fetch(mem):
    """len() used to fetch 1000 items and measure the list, so it was both slow
    and silently capped. 1200 records reported 1000."""
    base = time.time()
    for i in range(1200):
        mem.episodes.record(f"event {i}", at=base + i)

    assert len(mem.episodes) == 1200
    assert mem.summary()["episodes"] == 1200


def test_counting_excludes_deleted_records(mem):
    """The engine's namespace_size() counts tombstones — forgotten records stay
    in the namespace index until compaction — so it would report deleted
    memories as present. count() checks liveness."""
    mem.preferences.set("a", "first")
    mem.preferences.set("b", "second")
    assert len(mem.preferences) == 2

    mem.preferences.delete("a")
    assert len(mem.preferences) == 1
    assert mem.preferences.get("a") is None


def test_the_store_primitives_are_bounded(mem):
    """`newest` must parse only what it returns, and `count` must parse nothing
    — that is the whole reason they exist next to search()."""
    base = time.time()
    for i in range(500):
        mem.episodes.record(f"event {i}", at=base + i)

    ns = mem.episodes.namespace
    assert mem.store.count(ns) == 500
    assert len(mem.store.newest(ns, 7)) == 7
    assert len(mem.store.scan(ns)) == 500
    # scan returns (id, metadata) and never touches the value JSON
    rid, meta = mem.store.scan(ns)[0]
    assert isinstance(rid, int) and meta.namespace_id == ".".join(ns)
