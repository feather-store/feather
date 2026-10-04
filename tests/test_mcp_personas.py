"""Three ways Feather gets used, driven end to end over MCP.

The unit tests check one behaviour per test against a clean pocket. Real memory
problems are not like that: they appear after a dozen writes, a correction, a
restart, and enough unrelated memories to put the working set under budget
pressure. The ad-hoc version of these scenarios found two defects that every
unit test had passed over — `recall` ranking by heat instead of relevance, and
pinned memories crowding out every query result — so they live here now.

  1. PERSONAL     one user, one assistant, memory across restarts, corrections
  2. PERFORMANCE  marketing agents per brand, inherited policy, ranked recall
  3. FLEET        many agents on one database, isolation and cost at scale
"""
import asyncio
import json
import time

import pytest

from feather_db import DB
from feather_db.pocket import pocket
from feather_db.integrations.mcp_agent import build
from tests.mcp_harness import connect


def session(path, scope, **kw):
    """A fresh server for `scope`, as a client connection. Models a restart."""
    kw.setdefault("dim", 64)
    return connect(build(path, scope, **kw))


# ══ 1. personal ═══════════════════════════════════════════════════════════
# One user, one assistant, remembering across restarts.

def test_personal_assistant_across_four_sessions(tmp_path):
    path = str(tmp_path / "me.feather")
    scope = "me.assistant"

    async def go():
        # ── onboarding ────────────────────────────────────────────────────
        async with session(path, scope) as mcp:
            await mcp.call("remember", key="allergy",
                           content="severely allergic to shellfish",
                           pinned=True, importance=1.0)
            await mcp.call("remember", key="timezone", content="based in Chennai, IST")
            await mcp.call("remember", key="tone",
                           content="prefers short replies, no bullet lists")

        # ── weeks later: a new process, nothing in the conversation ───────
        async with session(path, scope) as mcp:
            context = await mcp.call("context")
            assert "shellfish" in context
            assert "Chennai" in context

            answer = await mcp.call("recall", query="shellfish")
            assert "shellfish" in answer

        # ── a correction: same key overwrites, it does not accumulate ─────
        async with session(path, scope) as mcp:
            await mcp.call("remember", key="timezone", content="moved to Berlin, CET")
            context = await mcp.call("context")
            assert "Berlin" in context
            assert "Chennai" not in context, "the correction left the stale fact behind"

        # ── 'forget that' ─────────────────────────────────────────────────
        async with session(path, scope) as mcp:
            await mcp.call("forget", key="tone")
            context = await mcp.call("context")
            assert "bullet lists" not in context
            assert "shellfish" in context, "forget removed more than it was asked to"

    asyncio.run(go())


def test_a_pinned_safety_fact_survives_budget_pressure(tmp_path):
    """The allergy must not fall out of context because forty dull memories
    arrived after it. This is the failure that matters most in this persona:
    it is silent, and the assistant just stops knowing."""
    path = str(tmp_path / "me.feather")

    async def go():
        async with session(path, "me.assistant", budget_tokens=300) as mcp:
            await mcp.call("remember", key="allergy",
                           content="severely allergic to shellfish", pinned=True)
            for i in range(40):
                await mcp.call("remember", key=f"chat_{i}",
                               content=f"discussed weekend plans, option {i}, "
                                       f"nothing important was decided")

            context = await mcp.call("context")
            assert "shellfish" in context

            stats = json.loads(await mcp.read("feather://stats"))
            assert stats["total_items"] == 41
            assert stats["warm_items"] > 0, "nothing was demoted; the budget did not bind"
            assert stats["hot_tokens"] <= stats["budget"]

    asyncio.run(go())


def test_recall_reaches_memories_too_cold_to_be_carried(tmp_path):
    """The point of warm storage: it is out of context but not out of reach."""
    path = str(tmp_path / "me.feather")

    async def go():
        async with session(path, "me.assistant", budget_tokens=200) as mcp:
            await mcp.call("remember", key="passport",
                           content="passport number expires in March 2031")
            for i in range(40):
                await mcp.call("remember", key=f"noise_{i}",
                               content=f"unrelated conversational filler number {i}")

            assert "passport" not in await mcp.call("context")
            assert "2031" in await mcp.call("recall", query="passport expiry")

    asyncio.run(go())


def test_without_an_embedder_recall_is_exact_keyword_matching(tmp_path):
    """Characterisation, not an endorsement. `feather-agent` with no
    --embed-provider falls back to BM25, which does no stemming: a memory whose
    content says "allergic" is not found by searching "allergy", and the key is
    not searchable at all. The model cannot tell that from "never told me", so
    any deployment relying on paraphrased recall needs a real embedder.

    If this starts failing because more queries match, retrieval improved —
    update the expectations rather than restoring the old behaviour.
    """
    path = str(tmp_path / "kw.feather")

    async def go():
        async with session(path, "me.assistant") as mcp:
            await mcp.call("remember", key="allergy",
                           content="severely allergic to shellfish")

            for query in ("shellfish", "allergic", "severely allergic"):
                assert "shellfish" in await mcp.call("recall", query=query), query

            for query in ("allergy", "allergies", "what food should be avoided"):
                answer = await mcp.call("recall", query=query)
                assert answer == "nothing found", f"{query!r} now matches: {answer}"

    asyncio.run(go())


# ══ 2. performance / marketing ════════════════════════════════════════════
# Agents per brand under a shared org, with a policy they all inherit.

@pytest.fixture
def agency(tmp_path):
    """One org policy, two brands. The policy is written once, centrally."""
    path = str(tmp_path / "agency.feather")
    db = DB.open(path, dim=64)
    pocket(db, ("acme",)).remember(
        "compliance", "never make medical or health claims in any creative",
        pinned=True)
    pocket(db, ("acme",)).remember(
        "brand_safety", "no competitor names in paid copy")
    db.save()
    del db
    return path


def test_brand_agent_inherits_the_org_policy(agency):
    async def go():
        async with session(agency, "acme.nike.creative") as mcp:
            context = await mcp.call("context")
            assert "medical" in context
            assert "competitor names" in context

            stats = json.loads(await mcp.read("feather://stats"))
            assert stats["inherits"] == ["acme.nike", "acme"]
            assert stats["hot_inherited"] == 2
            assert stats["hot_own"] == 0

    asyncio.run(go())


def test_performance_facts_are_recalled_by_relevance_not_by_heat(agency):
    """Regression. `recall` used to rank by heat, so the pinned compliance rule
    was the top hit for every query — the agent asked about ROAS and was told
    about medical claims."""
    async def go():
        async with session(agency, "acme.nike.creative") as mcp:
            await mcp.call("remember", key="roas_q3",
                           content="the ugc video creative delivered 4.2 ROAS in Q3, "
                                   "the highest of any format")
            await mcp.call("remember", key="ctr_q3",
                           content="static carousel CTR was 0.8 percent, below target")
            await mcp.call("remember", key="audience",
                           content="lookalike audiences outperformed interest targeting")

            top = (await mcp.call("recall", query="which creative had the best ROAS",
                                  limit=3)).splitlines()[0]
            assert "4.2" in top, f"pinned policy outranked the answer: {top}"

    asyncio.run(go())


def test_sibling_brands_do_not_leak_into_each_other(agency):
    """Two brands in one agency, one file. A leak here is a client incident."""
    async def go():
        db = DB.open(agency, dim=64)
        async with connect(build("", "acme.nike.creative", dim=64, db=db)) as nike:
            await nike.call("remember", key="budget",
                            content="nike q4 budget is 2.4 million dollars")
        async with connect(build("", "acme.adidas.creative", dim=64, db=db)) as adidas:
            assert "nike" not in (await adidas.call("context")).lower()
            assert "2.4 million" not in await adidas.call("recall", query="q4 budget")
            # but the shared policy still reaches them both
            assert "medical" in await adidas.call("context")
        db.save()

    asyncio.run(go())


def test_an_agent_updates_a_metric_instead_of_appending_a_second_truth(agency):
    """Performance numbers get restated constantly. Keys are how a restatement
    replaces the old number rather than sitting next to it."""
    async def go():
        async with session(agency, "acme.nike.paid") as mcp:
            await mcp.call("remember", key="roas", content="blended ROAS is 2.1")
            await mcp.call("remember", key="roas", content="blended ROAS is 3.4")

            context = await mcp.call("context")
            assert "3.4" in context and "2.1" not in context

            stats = json.loads(await mcp.read("feather://stats"))
            assert stats["hot_own"] == 1

    asyncio.run(go())


# ══ 3. fleet ══════════════════════════════════════════════════════════════
# Many agents, one database. Isolation must hold and cost must not grow.

FLEET = 50


@pytest.fixture
def fleet(tmp_path):
    """One DB, one shared root policy, `FLEET` agent scopes under it."""
    db = DB.open(str(tmp_path / "fleet.feather"), dim=64)
    pocket(db, ("fleet",)).remember("charter", "all agents log their sources",
                                    pinned=True)
    for i in range(FLEET):
        pkt = pocket(db, ("fleet", f"agent_{i:03d}"))
        pkt.remember("identity", f"I am agent {i:03d} and I own queue {i:03d}")
        pkt.remember("finding", f"agent {i:03d} observed anomaly code {1000 + i}")
    db.save()
    return db


def test_every_agent_in_the_fleet_sees_only_itself_and_the_charter(fleet):
    async def go():
        for i in (0, 7, FLEET - 1):
            scope = f"fleet.agent_{i:03d}"
            async with connect(build("", scope, dim=64, db=fleet)) as mcp:
                context = await mcp.call("context")
                assert f"agent {i:03d}" in context
                assert "log their sources" in context, "the charter did not reach it"

                others = [j for j in (0, 7, FLEET - 1) if j != i]
                for j in others:
                    assert f"queue {j:03d}" not in context

                stats = json.loads(await mcp.read("feather://stats"))
                assert stats["hot_own"] == 2
                assert stats["hot_inherited"] == 1

    asyncio.run(go())


def test_one_agents_context_does_not_get_slower_as_the_fleet_grows(fleet):
    """Context assembly must cost what one agent holds, not what the fleet
    holds — otherwise the 500th agent pays for the other 499. Best-of-three, so
    a scheduling hiccup on a loaded CI box does not fail the build."""
    async def go():
        async with connect(build("", "fleet.agent_000", dim=64, db=fleet)) as mcp:
            await mcp.call("context")                      # warm the caches
            timings = []
            for _ in range(3):
                start = time.perf_counter()
                await mcp.call("context")
                timings.append(time.perf_counter() - start)
            return min(timings)

    elapsed = asyncio.run(go())
    # 101 memories live in the database; this agent holds 3. A generous bound —
    # it catches an O(fleet) scan per call, not a few milliseconds of drift.
    assert elapsed < 0.25, f"context took {elapsed * 1000:.0f}ms for a 3-item pocket"


def test_a_fleet_wide_policy_change_reaches_every_agent(fleet):
    """Write once at the root; every agent picks it up without being told."""
    async def go():
        async with connect(build("", "fleet", dim=64, db=fleet)) as root:
            await root.call("remember", key="incident",
                            content="pause all outbound writes until 18:00 UTC",
                            pinned=True)

        for i in (0, 23, FLEET - 1):
            async with connect(build("", f"fleet.agent_{i:03d}", dim=64, db=fleet)) as mcp:
                assert "18:00 UTC" in await mcp.call("context")

    asyncio.run(go())


def test_two_servers_on_one_path_are_a_data_loss_bug(tmp_path):
    """Documents why `build(db=...)` exists. Two `DB.open` calls on one path are
    two independent instances; the second save overwrites the first's records.
    A host that gives each agent its own server on a shared file loses memory.
    If this ever stops failing, file locking landed and `build` should say so."""
    path = str(tmp_path / "collide.feather")
    a, b = DB.open(path, dim=64), DB.open(path, dim=64)
    assert a is not b
    pocket(a, ("x",)).remember("from_a", "written by a")
    a.save()
    pocket(b, ("y",)).remember("from_b", "written by b")
    b.save()
    del a, b

    reopened = DB.open(path, dim=64)
    assert len(reopened.get_all_ids()) == 1, (
        "both records survived — concurrent instances are now safe, so "
        "`build(db=...)` is no longer the only correct fleet pattern"
    )
