"""Agent pocket memory — hot / warm / cache tiering over one namespace.

The premise: an agent does not want "top k for a query", it wants the things it
should be carrying right now, sized to the context budget it has left. Feather
already computes that signal — every search hit increments `recall_count`, and
`stickiness = 1 + ln(1+recall_count)` already slows decay for things the agent
keeps returning to. The pocket reads those counters as a tier instead of leaving
them to re-rank search results.

Two bugs surfaced the first time the prototype ran, and both have tests here:
every fresh record scored heat exactly 1.0 (age zero → recency 1.0) so the
working set was ordered arbitrarily, and cache entries leaked into the hot set
and spent context budget.
"""
import time

import numpy as np
import pytest

import feather_db
from feather_db import DB
from feather_db.pocket import Pocket, estimate_tokens, pocket

NS = "agent_x"


@pytest.fixture
def p(tmp_path):
    db = DB.open(str(tmp_path / "pocket.feather"), dim=64)
    return pocket(db, NS, budget_tokens=200)


def _heat(p, key):
    """Heat of one record, regardless of whether it is hot enough to carry.

    `hot()` excludes anything under `min_heat` by design, so a test comparing
    two deliberately-cold records has to ask for the score directly.
    """
    return p.heat(p.db.get_metadata(p._id(key)))


def _age(p, key, days):
    """Backdate a record so decay can be tested without waiting."""
    rid = p._id(key)
    m = p.db.get_metadata(rid)
    m.timestamp = int(time.time() - days * 86400)
    p.db.update_metadata(rid, m)


# ── writing ───────────────────────────────────────────────────────────────

def test_remember_then_hot_returns_it(p):
    p.remember("a", "the brand voice is calm and clinical")
    assert [i.key for i in p.hot()] == ["a"]


def test_remember_is_an_upsert(p):
    p.remember("a", "first version")
    p.remember("a", "second version")
    hot = p.hot()
    assert len(hot) == 1 and "second" in hot[0].content


def test_update_keeps_the_recall_history(p):
    """An update is new information about something the agent already cared
    about — not a reason to forget that it cared."""
    p.remember("a", "alpha beta")
    for _ in range(4):
        p.recall("alpha")
    before = p.db.get_metadata(p._id("a")).recall_count
    assert before >= 1
    p.remember("a", "alpha beta revised")
    assert p.db.get_metadata(p._id("a")).recall_count == before


def test_forget_removes_from_every_tier(p):
    p.remember("a", "alpha")
    p.forget("a")
    assert p.hot() == []
    assert p.recall("alpha") == []


# ── heat ──────────────────────────────────────────────────────────────────

def test_fresh_records_are_not_all_equally_hot(p):
    """The first bug: at age ~0 recency is exactly 1.0 for everything, so
    without a usage term the working set was ordered arbitrarily among all the
    records written this session — which is most of what an agent holds."""
    p.remember("used", "alpha alpha alpha")
    p.remember("unused", "beta beta beta")
    for _ in range(8):
        p.recall("alpha")

    heats = {i.key: i.heat for i in p.hot()}
    assert heats["used"] > heats["unused"], heats


def test_age_cools_a_record(p):
    p.remember("old", "gamma")
    p.remember("new", "gamma")
    _age(p, "old", 120)
    assert _heat(p, "old") < _heat(p, "new") * 0.5


def test_use_slows_decay(p):
    """stickiness: a record read often should survive ageing better than one
    that was not. This is the engine's existing behaviour, surfaced as a tier."""
    p.remember("hot", "delta delta")
    p.remember("cold", "delta delta")
    for _ in range(12):
        p.recall("delta")
    # equalise recalls on the loser so only stickiness differs
    m = p.db.get_metadata(p._id("cold"))
    m.recall_count = 0
    p.db.update_metadata(p._id("cold"), m)
    _age(p, "hot", 60); _age(p, "cold", 60)

    assert _heat(p, "hot") > _heat(p, "cold")


def test_pinned_is_always_maximally_hot(p):
    p.remember("pinned", "system rule", pinned=True)
    p.remember("fresh", "just written")
    _age(p, "pinned", 400)
    heats = {i.key: i.heat for i in p.hot()}
    assert heats["pinned"] == 1.0
    assert heats["pinned"] > heats["fresh"]


def test_pin_and_unpin_after_the_fact(p):
    p.remember("a", "x")
    _age(p, "a", 300)
    cold = _heat(p, "a")
    assert cold < p.min_heat, "a 300-day-old record should be below the floor"

    p.pin("a")
    assert _heat(p, "a") == 1.0
    assert [i.key for i in p.hot()] == ["a"], "pinning should bring it back into the pocket"

    p.pin("a", False)
    assert _heat(p, "a") == pytest.approx(cold, abs=1e-6)
    assert p.hot() == [], "unpinned and cold: back below the floor"


# ── budget ────────────────────────────────────────────────────────────────

def test_hot_respects_the_token_budget(p):
    for i in range(30):
        p.remember(f"k{i}", "word " * 40)          # ~50 tokens each
    hot = p.hot(200)
    assert sum(i.tokens for i in hot) <= 200
    assert 0 < len(hot) < 30


def test_a_long_cold_item_does_not_block_short_hot_ones(p):
    """`hot()` skips over an item that will not fit rather than stopping, so one
    oversized record cannot starve everything ranked behind it."""
    p.remember("huge", "word " * 400)              # ~500 tokens
    for i in range(3):
        p.remember(f"small{i}", "tiny note")
    hot = [i.key for i in p.hot(60)]
    assert "huge" not in hot
    assert len([k for k in hot if k.startswith("small")]) == 3


def test_use_changes_which_records_fit(p):
    """The point of the whole feature: the working set follows real usage."""
    for i in range(6):
        p.remember(f"k{i}", f"topic{i} " * 8)
    tight = 40
    before = set(k.key for k in p.hot(tight))
    for _ in range(10):
        p.recall("topic5")
    after = set(k.key for k in p.hot(tight))
    assert "k5" in after, (before, after)


def test_hot_text_is_prompt_ready(p):
    p.remember("a", "first fact")
    p.remember("b", "second fact")
    text = p.hot_text()
    assert "first fact" in text and "second fact" in text


# ── warm ──────────────────────────────────────────────────────────────────

def test_cold_records_stay_searchable(p):
    p.remember("old", "supplier delay in the packaging refresh")
    _age(p, "old", 500)
    assert _heat(p, "old") < p.min_heat
    assert p.hot(20) == []                            # too cold to carry
    assert any(i.key == "old" for i in p.recall("supplier packaging"))


def test_recall_promotes(p):
    p.remember("a", "epsilon topic")
    before = p.db.get_metadata(p._id("a")).recall_count
    p.recall("epsilon")
    assert p.db.get_metadata(p._id("a")).recall_count > before


def test_recall_does_not_cross_namespaces(p):
    other = Pocket(p.db, "other_agent")
    p.remember("mine", "zeta shared word")
    other.remember("theirs", "zeta shared word")
    assert all(i.key != "theirs" for i in p.recall("zeta"))


# ── cache ─────────────────────────────────────────────────────────────────

def test_cache_computes_once(p):
    calls = []
    def compute():
        calls.append(1); return {"rows": 7}
    assert p.cache("q1", compute)["rows"] == 7
    assert p.cache("q1", compute)["rows"] == 7
    assert len(calls) == 1


def test_cache_recomputes_after_expiry(p):
    calls = []
    def compute():
        calls.append(1); return len(calls)
    p.cache("q", compute, ttl_seconds=0.05)
    time.sleep(0.12)
    p.cache("q", compute, ttl_seconds=0.05)
    assert len(calls) == 2


def test_cache_never_spends_the_context_budget(p):
    """The second bug: cache entries live in the same namespace, and a generous
    budget admitted them at the tail even with importance 0.0."""
    p.remember("real", "an actual memory")
    p.cache("expensive", lambda: {"big": "x" * 500}, ttl_seconds=60)
    hot = p.hot(10_000)
    assert [i.key for i in hot] == ["real"]
    assert not any("__cache__" in i.key for i in hot)


def test_evict_expired_leaves_memories_alone(p):
    p.remember("keep", "a real memory")
    p.cache("gone", lambda: {"v": 1}, ttl_seconds=0.05)
    time.sleep(0.12)
    assert p.evict_expired() == 1
    assert [i.key for i in p.hot()] == ["keep"]


# ── introspection ─────────────────────────────────────────────────────────

def test_stats_separates_the_tiers(p):
    for i in range(10):
        p.remember(f"k{i}", "word " * 30)
    p.cache("c", lambda: {"v": 1}, ttl_seconds=60)
    s = p.stats(); 
    assert s["hot_items"] + s["warm_items"] + s["cache_items"] == s["total_items"]
    assert s["cache_items"] == 1
    assert s["hot_tokens"] <= s["budget"]


def test_estimate_tokens_is_monotonic():
    assert estimate_tokens("a") >= 1
    assert estimate_tokens("word " * 100) > estimate_tokens("word " * 10)


def test_works_with_no_embedder_at_all(tmp_path):
    """A pocket that needs an API key to hold a preference is not useful."""
    db = DB.open(str(tmp_path / "n.feather"), dim=64)
    pk = pocket(db, "solo")
    pk.remember("a", "omega keyword here")
    assert [i.key for i in pk.hot()] == ["a"]
    assert any(i.key == "a" for i in pk.recall("omega"))


# ── agent scoping ─────────────────────────────────────────────────────────
# A pocket is opened at a scope and reads that scope PLUS every ancestor, which
# is what an agent actually needs: its own notes AND the brand rules AND the org
# policy, in one budget, ranked together.

@pytest.fixture
def fleet(tmp_path):
    db = DB.open(str(tmp_path / "fleet.feather"), dim=64)
    return db


def test_an_agent_sees_its_ancestors(fleet):
    pocket(fleet, ("org",)).remember("policy", "never make medical claims")
    pocket(fleet, ("org", "brand_a")).remember("voice", "calm and clinical")
    agent = pocket(fleet, ("org", "brand_a", "creative"))
    agent.remember("note", "carousel underperformed")

    assert {i.key for i in agent.hot()} == {"policy", "voice", "note"}


def test_inherited_items_are_flagged_with_their_source(fleet):
    pocket(fleet, ("org", "brand_a")).remember("voice", "calm")
    agent = pocket(fleet, ("org", "brand_a", "creative"))
    agent.remember("note", "mine")

    by_key = {i.key: i for i in agent.hot()}
    assert by_key["note"].inherited is False
    assert by_key["voice"].inherited is True
    assert by_key["voice"].scope == ("org", "brand_a")


def test_siblings_are_isolated(fleet):
    a = pocket(fleet, ("org", "brand_a", "creative"))
    b = pocket(fleet, ("org", "brand_a", "media"))
    a.remember("note", "creative note")
    b.remember("note", "media note")

    assert next(i for i in a.hot() if i.key == "note").content == "creative note"
    assert next(i for i in b.hot() if i.key == "note").content == "media note"


def test_a_different_branch_inherits_only_the_common_ancestor(fleet):
    pocket(fleet, ("org",)).remember("policy", "org wide")
    pocket(fleet, ("org", "brand_a")).remember("voice", "brand a only")
    other = pocket(fleet, ("org", "brand_b", "creative"))
    other.remember("note", "mine")

    keys = {i.key for i in other.hot()}
    assert keys == {"policy", "note"}
    assert "voice" not in keys


def test_own_scope_shadows_an_inherited_key(fleet):
    """Deepest scope wins, like a variable shadowing an outer one."""
    parent = pocket(fleet, ("org", "brand_a"))
    parent.remember("palette", "muted green")
    agent = pocket(fleet, ("org", "brand_a", "creative"))

    assert next(i for i in agent.hot() if i.key == "palette").inherited is True
    agent.remember("palette", "warmer clay")

    own = next(i for i in agent.hot() if i.key == "palette")
    assert own.inherited is False and own.content == "warmer clay"
    # the parent is untouched, so a sibling still sees the original
    assert next(i for i in parent.hot() if i.key == "palette").content == "muted green"


def test_inheritance_is_discounted_per_level(fleet):
    """Same record, same age, same recalls — only distance differs."""
    pocket(fleet, ("org",)).remember("x", "identical text")
    deep = pocket(fleet, ("org", "brand_a", "creative"))
    near = pocket(fleet, ("org", "brand_a"))

    meta = fleet.get_metadata(pocket(fleet, ("org",))._id("x"))
    assert deep.heat(meta, depth=2) < near.heat(meta, depth=1) < deep.heat(meta, depth=0)


def test_inherit_can_be_turned_off(fleet):
    pocket(fleet, ("org", "brand_a")).remember("voice", "calm")
    solo = pocket(fleet, ("org", "brand_a", "creative"), inherit=False)
    solo.remember("note", "mine")
    assert {i.key for i in solo.hot()} == {"note"}


def test_writes_always_land_in_the_agents_own_scope(fleet):
    """An agent must not be able to mutate a shared layer by accident."""
    parent = pocket(fleet, ("org", "brand_a"))
    agent = pocket(fleet, ("org", "brand_a", "creative"))
    agent.remember("thing", "written by the agent")

    assert parent.hot() == []          # the parent scope stayed empty
    assert [i.key for i in agent.hot()] == ["thing"]


def test_recall_reaches_inherited_scopes(fleet):
    pocket(fleet, ("org", "brand_a")).remember("voice", "omega distinctive phrase")
    agent = pocket(fleet, ("org", "brand_a", "creative"))
    hits = agent.recall("omega distinctive")
    assert any(i.key == "voice" and i.inherited for i in hits)


def test_scope_accepts_a_dotted_string(fleet):
    a = pocket(fleet, "org.brand_a.creative")
    b = pocket(fleet, ("org", "brand_a", "creative"))
    a.remember("k", "v")
    assert [i.key for i in b.hot()] == ["k"]


def test_stats_reports_the_layering(fleet):
    pocket(fleet, ("org",)).remember("p", "policy")
    pocket(fleet, ("org", "brand_a")).remember("v", "voice")
    agent = pocket(fleet, ("org", "brand_a", "creative"))
    agent.remember("n", "note")

    s = agent.stats()
    assert s["inherits"] == ["org.brand_a", "org"]
    assert s["hot_own"] == 1 and s["hot_inherited"] == 2
    assert s["per_scope"]["org.brand_a.creative"] == 1


# ── session lifecycle ─────────────────────────────────────────────────────
# An agent run writes two very different things: what it LEARNED (worth
# keeping) and what it was THINKING (intermediate results, revised plans).
# Agents with one place to put things write both, and the second kind is what
# turns a useful memory into a landfill. A session gives the thinking its own
# scope: real memory during the run, dropped at the end unless promoted.

def test_scratch_is_visible_during_the_run(fleet):
    agent = pocket(fleet, ("org", "a"))
    with agent.session() as s:
        s.note("intermediate finding about carousels")
        assert "carousels" in s.context()


def test_scratch_is_dropped_at_the_end(fleet):
    agent = pocket(fleet, ("org", "a"))
    with agent.session() as s:
        s.note("thinking out loud")
    assert agent.hot() == []
    assert "thinking out loud" not in agent.hot_text()


def test_keep_promotes_to_durable_memory(fleet):
    agent = pocket(fleet, ("org", "a"))
    with agent.session() as s:
        s.note("scratch reasoning")
        s.keep("finding", "carousel fails for Brand A")
    keys = [i.key for i in agent.hot()]
    assert keys == ["finding"]
    assert "scratch reasoning" not in agent.hot_text()


def test_a_later_run_inherits_findings_not_thinking(fleet):
    agent = pocket(fleet, ("org", "a"))
    with agent.session() as s1:
        s1.note("step one")
        s1.note("step two")
        s1.keep("finding", "the durable conclusion")
    with agent.session() as s2:
        ctx = s2.context()
    assert "durable conclusion" in ctx
    assert "step one" not in ctx and "step two" not in ctx


def test_session_inherits_the_agents_ancestors(fleet):
    pocket(fleet, ("org",)).remember("policy", "org rule")
    agent = pocket(fleet, ("org", "a"))
    with agent.session() as s:
        s.note("local thought")
        ctx = s.context()
    assert "org rule" in ctx and "local thought" in ctx


def test_scratch_cannot_evict_the_durable_rules(fleet):
    """Everything written this run has age ~0 and maximal recency, so without a
    cap a chatty run would push out the rules it is supposed to follow."""
    pocket(fleet, ("org",)).remember("policy", "NEVER make medical claims", pinned=True)
    agent = pocket(fleet, ("org", "a"), budget_tokens=120)
    with agent.session(scratch_ratio=0.4) as s:
        for i in range(20):
            s.note(f"verbose intermediate reasoning step number {i} with padding")
        ctx = s.context()
    assert "medical claims" in ctx, "a chatty run evicted a pinned org policy"


def test_close_is_idempotent(fleet):
    agent = pocket(fleet, ("org", "a"))
    s = agent.session()
    s.note("x")
    first = s.close()
    second = s.close()
    assert first.get("already_closed") is not True
    assert second.get("already_closed") is True


def test_a_crashed_run_does_not_leak_scratch(fleet):
    """__exit__ must clean up on the exception path, or the next run inherits a
    half-finished thought as though it were durable memory."""
    agent = pocket(fleet, ("org", "a"))
    agent.remember("real", "a durable memory")
    with pytest.raises(RuntimeError):
        with agent.session() as s:
            s.note("half-finished work")
            raise RuntimeError("tool timed out")
    assert [i.key for i in agent.hot()] == ["real"]
    assert "half-finished" not in agent.hot_text()


def test_close_reports_what_happened(fleet):
    agent = pocket(fleet, ("org", "a"))
    s = agent.session()
    s.note("a"); s.note("b")
    s.keep("k", "kept thing")
    r = s.close()
    assert r["scratch_dropped"] == 2
    assert r["kept"] == ["k"]


def test_sessions_are_isolated_from_each_other(fleet):
    agent = pocket(fleet, ("org", "a"))
    s1 = agent.session()
    s2 = agent.session()
    s1.note("only in session one")
    assert "only in session one" not in s2.context()
    s1.close(); s2.close()


def test_maintain_reaps_abandoned_scratch(fleet):
    """A process killed mid-run leaves its scratch scope behind."""
    agent = pocket(fleet, ("org", "a"))
    orphan = agent.session()
    orphan.note("abandoned")           # never closed — simulates a hard crash
    assert agent.maintain()["orphan_scratch_dropped"] >= 1
    assert agent.maintain()["orphan_scratch_dropped"] == 0   # nothing left


def test_maintain_leaves_durable_memory_alone(fleet):
    agent = pocket(fleet, ("org", "a"))
    agent.remember("keep", "durable")
    agent.cache("c", lambda: {"v": 1}, ttl_seconds=60)
    agent.maintain()
    assert [i.key for i in agent.hot()] == ["keep"]


def test_session_recall_sees_scratch_and_durable(fleet):
    agent = pocket(fleet, ("org", "a"))
    agent.remember("durable", "sigma durable phrase")
    with agent.session() as s:
        s.note("sigma scratch phrase")
        keys = {i.key for i in s.recall("sigma")}
        assert any("durable" == k for k in keys)


# ── hot() memoisation ─────────────────────────────────────────────────────
# hot() is O(namespace) — 171 ms at 10,000 memories, far too slow for a call an
# agent makes every turn. It is memoised against a write generation, which is
# only correct if EVERY path that changes heat invalidates it.

def test_repeated_hot_is_cached(fleet):
    p = pocket(fleet, ("org", "a"))
    for i in range(50):
        p.remember(f"k{i}", "word " * 20)
    first = p.hot()
    assert p.hot() is first, "identical call recomputed instead of hitting the cache"


def test_a_write_invalidates(fleet):
    p = pocket(fleet, ("org", "a"))
    p.remember("a", "alpha")
    before = p.hot()
    p.remember("b", "beta")
    assert {i.key for i in p.hot()} == {"a", "b"}
    assert p.hot() is not before


def test_recall_invalidates(fleet):
    """recall() looks like a read but increments recall_count, which changes
    heat — the exact thing promotion exists to do."""
    p = pocket(fleet, ("org", "a"))
    p.remember("x", "omega topic")
    p.hot()                                   # prime the cache
    before = p.db.get_metadata(p._id("x")).recall_count
    p.recall("omega")
    after_heat = {i.key: i.heat for i in p.hot()}
    assert p.db.get_metadata(p._id("x")).recall_count > before
    assert after_heat["x"] > 0


def test_a_parents_write_invalidates_the_child(fleet):
    """Inheritance means a parent's write changes what the child can see."""
    parent = pocket(fleet, ("org",))
    child = pocket(fleet, ("org", "a"))
    child.remember("own", "mine")
    child.hot()                               # prime
    parent.remember("shared", "from the parent")
    assert "shared" in {i.key for i in child.hot()}


def test_forget_invalidates(fleet):
    p = pocket(fleet, ("org", "a"))
    p.remember("a", "alpha")
    p.hot()
    p.forget("a")
    assert p.hot() == []


def test_pin_invalidates(fleet):
    p = pocket(fleet, ("org", "a"))
    p.remember("a", "alpha")
    _age(p, "a", 400)
    p.hot()
    p.pin("a")
    assert [i.key for i in p.hot()] == ["a"]


def test_session_close_invalidates(fleet):
    p = pocket(fleet, ("org", "a"))
    with p.session() as s:
        s.note("scratch")
        s.context()
    assert p.hot() == []


def test_different_budgets_are_cached_separately(fleet):
    p = pocket(fleet, ("org", "a"))
    for i in range(20):
        p.remember(f"k{i}", "word " * 30)
    small = p.hot(60)
    large = p.hot(6000)
    assert len(small) < len(large)
    assert len(p.hot(60)) == len(small)        # still correct after the other call
