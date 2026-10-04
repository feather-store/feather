"""The MCP surface as a client sees it, over the real protocol.

`test_mcp_agent.py` covers behaviour by calling the tool functions. This file
covers the wire: handshake, the schemas the model is shown, argument validation,
error signalling, and session lifetime. A server can pass every test in that
file and still be unusable in Claude Desktop — the replaced `mcp_server` module
was exactly that.
"""
import asyncio
import json

import pytest

from feather_db import DB
from feather_db.pocket import pocket
from feather_db.integrations.mcp_agent import build
from tests.mcp_harness import connect, over_mcp

SCOPE = "org.brand_a.creative"


@pytest.fixture
def server(tmp_path):
    path = str(tmp_path / "agent.feather")
    db = DB.open(path, dim=64)
    pocket(db, ("org",)).remember("policy", "never make medical claims", pinned=True)
    db.save()
    del db
    return build(path, SCOPE, dim=64, budget_tokens=400)


# ── handshake ─────────────────────────────────────────────────────────────

@over_mcp
async def test_the_handshake_completes(mcp):
    assert mcp.init.server_info.name == "feather"
    assert mcp.init.protocol_version


@over_mcp
async def test_the_handshake_carries_the_memory_policy(mcp):
    """`instructions` is how the model learns when to write. It is delivered
    once, at initialize — if it does not survive the handshake the model gets a
    memory server with no policy and writes nothing."""
    text = (mcp.init.instructions or "").lower()
    assert "recall" in text and "remember" in text
    assert "feather://context" in text


@over_mcp
async def test_all_three_capabilities_are_advertised(mcp):
    assert {"remember", "recall", "context", "forget", "memory_stats"} <= set(await mcp.tools())
    assert {"remember", "what-do-you-know", "catch-up"} <= set(await mcp.prompts())
    assert {"feather://context", "feather://stats"} <= set(await mcp.resources())


# ── the schemas the model is shown ────────────────────────────────────────

@over_mcp
async def test_remember_advertises_the_right_argument_contract(mcp):
    """The schema is derived from the type hints, so a signature change silently
    changes what the model is allowed to send."""
    schema = (await mcp.tools())["remember"].input_schema
    assert set(schema["required"]) == {"key", "content"}
    props = schema["properties"]
    assert props["key"]["type"] == "string"
    assert props["importance"]["type"] == "number"
    assert props["pinned"]["type"] == "boolean"
    # defaults must be advertised, or the model is forced to supply them
    assert props["importance"]["default"] == 1.0
    assert props["pinned"]["default"] is False


@over_mcp
async def test_optional_arguments_are_genuinely_optional(mcp):
    """`context` takes a budget override. If Optional[int] ever renders as
    required, every bare `context` call from the model starts failing."""
    schema = (await mcp.tools())["context"].input_schema
    assert not schema.get("required")
    assert await mcp.call("context")


@over_mcp
async def test_every_tool_is_described_well_enough_to_choose_between(mcp):
    for name, tool in (await mcp.tools()).items():
        assert tool.description and len(tool.description) > 40, name
    remember = (await mcp.tools())["remember"].description.lower()
    assert "duplicate" in remember        # tells the model to reuse keys
    assert "not" in remember              # tells the model what NOT to store


@over_mcp
async def test_resources_declare_their_mime_types(mcp):
    resources = await mcp.resources()
    assert resources["feather://context"].mime_type == "text/plain"
    assert resources["feather://stats"].mime_type == "application/json"


@over_mcp
async def test_prompts_declare_their_arguments(mcp):
    prompts = await mcp.prompts()
    assert [a.name for a in prompts["remember"].arguments] == ["what"]
    assert [a.name for a in prompts["what-do-you-know"].arguments] == ["topic"]
    assert not (prompts["catch-up"].arguments or [])


# ── round trips ───────────────────────────────────────────────────────────

@over_mcp
async def test_a_write_in_one_call_is_visible_to_the_next(mcp):
    await mcp.call("remember", key="tone", content="captions stay lowercase")
    assert "lowercase" in await mcp.call("recall", query="captions")
    assert "lowercase" in await mcp.call("context")


@over_mcp
async def test_the_resource_and_the_tool_agree(mcp):
    await mcp.call("remember", key="tone", content="captions stay lowercase")
    assert await mcp.read("feather://context") == await mcp.call("context")


@over_mcp
async def test_stats_resource_is_parseable_json(mcp):
    """Published as application/json, so the key names are a contract — a
    rename breaks every dashboard reading the resource, silently."""
    stats = json.loads(await mcp.read("feather://stats"))
    assert set(stats) == {
        "scope", "inherits", "budget",
        "hot_items", "hot_tokens", "hot_own", "hot_inherited",
        "warm_items", "warm_tokens", "cache_items", "total_items", "per_scope",
    }
    assert stats["scope"] == SCOPE
    assert stats["inherits"] == ["org.brand_a", "org"]
    assert stats["hot_tokens"] <= stats["budget"]


@over_mcp
async def test_stats_track_writes_made_over_the_wire(mcp):
    before = json.loads(await mcp.read("feather://stats"))["total_items"]
    await mcp.call("remember", key="new", content="a freshly stored fact")
    after = json.loads(await mcp.read("feather://stats"))
    assert after["total_items"] == before + 1
    assert after["hot_own"] >= 1


@over_mcp
async def test_inherited_scope_arrives_over_the_wire(mcp):
    """The ancestor memory was written by a different process entirely."""
    assert "medical claims" in await mcp.call("context")


@over_mcp
async def test_forget_takes_effect_across_calls(mcp):
    await mcp.call("remember", key="wrong", content="launch is in June")
    assert "June" in await mcp.call("context")
    await mcp.call("forget", key="wrong")
    assert "June" not in await mcp.call("context")


@over_mcp
async def test_prompts_render_with_the_argument_substituted(mcp):
    rendered = await mcp.prompt("remember", what="the client hates stock photos")
    assert "stock photos" in rendered
    assert "remember" in rendered.lower()
    assert "catch-up" not in rendered

    topic = await mcp.prompt("what-do-you-know", topic="pricing")
    assert "pricing" in topic

    assert "feather://context" in await mcp.prompt("catch-up")


# ── failure signalling ────────────────────────────────────────────────────

@over_mcp
async def test_an_unknown_tool_is_an_error_not_a_crash(mcp):
    assert "does_not_exist" in await mcp.fails("does_not_exist")


@over_mcp
async def test_a_missing_required_argument_is_reported_to_the_model(mcp):
    """The model must be told which field it omitted, or it cannot retry."""
    assert "key" in await mcp.fails("remember", content="no key given")


@over_mcp
async def test_a_wrong_argument_type_is_rejected(mcp):
    assert "importance" in await mcp.fails("remember", key="k", content="c",
                                           importance="very high")


@over_mcp
async def test_the_session_survives_a_failed_call(mcp):
    """A validation error must not poison the connection — the model retries on
    the same session, and a dead session looks to the user like memory loss."""
    await mcp.fails("remember", content="no key")
    await mcp.call("remember", key="retry", content="second attempt worked")
    assert "second attempt" in await mcp.call("context")


@over_mcp
async def test_recall_with_no_match_says_so_rather_than_inventing(mcp):
    assert "nothing" in (await mcp.call("recall", query="quarterly gross margin")).lower()


# ── session lifetime ──────────────────────────────────────────────────────

def test_memory_written_in_one_session_survives_the_next(tmp_path):
    """Claude Desktop restarts the server on every launch."""
    path = str(tmp_path / "a.feather")

    async def first():
        async with connect(build(path, SCOPE, dim=64)) as mcp:
            await mcp.call("remember", key="brand", content="voice is dry and factual")

    async def second():
        async with connect(build(path, SCOPE, dim=64)) as mcp:
            return await mcp.call("context")

    asyncio.run(first())
    assert "dry and factual" in asyncio.run(second())


def test_two_agents_on_one_file_do_not_read_each_other(tmp_path):
    """Sibling scopes are the isolation boundary; a shared file is not a leak."""
    path = str(tmp_path / "shared.feather")

    async def go():
        async with connect(build(path, "org.team.alice", dim=64)) as alice:
            await alice.call("remember", key="secret", content="alice's private note")
            async with connect(build(path, "org.team.bob", dim=64)) as bob:
                return await bob.call("context"), await bob.call("recall", query="private note")

    context, recalled = asyncio.run(go())
    assert "alice" not in context.lower()
    assert "alice" not in recalled.lower()


def test_concurrent_sessions_against_one_server_stay_independent(tmp_path):
    """Two clients, one server object, overlapping in time."""
    server = build(str(tmp_path / "c.feather"), SCOPE, dim=64)

    async def go():
        async with connect(server) as a, connect(server) as b:
            await a.call("remember", key="from_a", content="written by client a")
            # b shares the pocket, so it sees the write — but its own session,
            # request ids and handshake are separate
            assert b.init.server_info.name == "feather"
            return await b.call("context")

    assert "client a" in asyncio.run(go())
