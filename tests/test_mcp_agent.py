"""Feather over MCP — agent memory as tools, resources and prompts.

The previous server exposed 16 low-level database verbs and is BROKEN against
the current SDK: `Server.list_tools` was removed in 2.0, so `create_server()`
raises. This module replaces it with a surface shaped like what an agent does
rather than like the database underneath.

The three capabilities are used for what each is actually good at: tools for
recall (models recall eagerly), prompts for capture (models write almost never,
so capture must be user-invoked), and resources for context (attachable without
spending a tool call).
"""
import asyncio
import json
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="needs the MCP SDK")

from feather_db import DB                                  # noqa: E402
from feather_db.pocket import pocket                       # noqa: E402
from feather_db.integrations.mcp_agent import build        # noqa: E402

SCOPE = "org.brand_a.creative"


@pytest.fixture
def server(tmp_path):
    path = str(tmp_path / "agent.feather")
    # a shared ancestor scope, so inheritance is exercised through MCP too
    db = DB.open(path, dim=64)
    pocket(db, ("org",)).remember("policy", "never make medical claims", pinned=True)
    db.save()
    del db
    return build(path, SCOPE, dim=64, budget_tokens=400)


def _text(result):
    """Pull plain text out of whatever shape the SDK returns."""
    c = getattr(result, "content", result)
    if isinstance(c, list) and c:
        return getattr(c[0], "text", str(c[0]))
    return str(c)


def call(server, name, **kw):
    return _text(asyncio.run(server.call_tool(name, kw)))


# ── the surface ───────────────────────────────────────────────────────────

def test_the_server_exposes_all_three_capabilities(server):
    tools = {t.name for t in asyncio.run(server.list_tools())}
    prompts = {p.name for p in asyncio.run(server.list_prompts())}
    resources = {str(r.uri) for r in asyncio.run(server.list_resources())}

    assert {"remember", "recall", "context", "forget", "memory_stats"} <= tools
    assert {"remember", "what-do-you-know", "catch-up"} <= prompts
    assert {"feather://context", "feather://stats"} <= resources


def test_tools_carry_descriptions_the_model_can_act_on(server):
    """A tool the model cannot tell apart from another is a tool it will not
    call. `remember` in particular has to say what NOT to store."""
    by_name = {t.name: t for t in asyncio.run(server.list_tools())}
    assert "duplicate" in by_name["remember"].description.lower()
    assert "not" in by_name["remember"].description.lower()
    assert by_name["recall"].description


# ── tools ─────────────────────────────────────────────────────────────────

def test_remember_then_recall(server):
    call(server, "remember", key="voice", content="brand voice is calm and clinical")
    out = call(server, "recall", query="brand voice", limit=3)
    assert "calm and clinical" in out


def test_remember_is_an_upsert(server):
    call(server, "remember", key="k", content="first")
    call(server, "remember", key="k", content="second")
    ctx = call(server, "context")
    assert "second" in ctx and "first" not in ctx


def test_context_includes_inherited_scopes(server):
    call(server, "remember", key="own", content="my own note")
    ctx = call(server, "context")
    assert "my own note" in ctx
    assert "medical claims" in ctx, "inherited org policy missing from context"


def test_context_respects_the_budget(server):
    for i in range(40):
        call(server, "remember", key=f"k{i}", content="word " * 40)
    stats = json.loads(call(server, "memory_stats"))
    assert stats["hot_tokens"] <= stats["budget"]


def test_forget_removes_it(server):
    call(server, "remember", key="tmp", content="temporary thing")
    assert "temporary thing" in call(server, "context")
    call(server, "forget", key="tmp")
    assert "temporary thing" not in call(server, "context")


def test_memory_stats_reports_the_layering(server):
    call(server, "remember", key="own", content="mine")
    s = json.loads(call(server, "memory_stats"))
    assert s["scope"] == SCOPE
    assert s["inherits"] == ["org.brand_a", "org"]
    assert s["hot_own"] >= 1 and s["hot_inherited"] >= 1


def test_recall_on_empty_memory_says_so(server):
    """Silence is worse than "nothing found" — the model cannot tell an empty
    result from a failed call."""
    assert "nothing" in call(server, "recall", query="zzz nonexistent").lower()


# ── resources ─────────────────────────────────────────────────────────────

def test_context_resource_matches_the_tool(server):
    call(server, "remember", key="a", content="a durable fact")
    res = asyncio.run(server.read_resource("feather://context"))
    body = getattr(res[0], "content", None) if isinstance(res, list) else None
    text = body if isinstance(body, str) else _text(res)
    assert "a durable fact" in text


def test_stats_resource_is_json(server):
    res = asyncio.run(server.read_resource("feather://stats"))
    body = getattr(res[0], "content", None) if isinstance(res, list) else None
    text = body if isinstance(body, str) else _text(res)
    assert json.loads(text)["scope"] == SCOPE


# ── prompts ───────────────────────────────────────────────────────────────

def test_remember_prompt_tells_the_model_to_reuse_keys(server):
    """The capture path. It must steer toward a stable key, or repeated
    learning of the same fact accumulates near-duplicates."""
    g = asyncio.run(server.get_prompt("remember", {"what": "prefers async updates"}))
    text = g.messages[0].content.text
    assert "prefers async updates" in text
    assert "key" in text.lower()


def test_recall_prompt_forbids_guessing(server):
    g = asyncio.run(server.get_prompt("what-do-you-know", {"topic": "pricing"}))
    text = g.messages[0].content.text.lower()
    assert "pricing" in text
    assert "guess" in text or "say so" in text


def test_catchup_prompt_points_at_the_resource(server):
    g = asyncio.run(server.get_prompt("catch-up", {}))
    assert "feather://context" in g.messages[0].content.text


# ── persistence ───────────────────────────────────────────────────────────

def test_memory_survives_a_server_restart(tmp_path):
    path = str(tmp_path / "persist.feather")
    s1 = build(path, SCOPE, dim=64)
    call(s1, "remember", key="durable", content="this must survive")

    s2 = build(path, SCOPE, dim=64)
    assert "this must survive" in call(s2, "context")


def test_sibling_agents_do_not_see_each_other(tmp_path):
    path = str(tmp_path / "fleet.feather")
    a = build(path, "org.brand_a.creative", dim=64)
    b = build(path, "org.brand_a.media", dim=64)
    call(a, "remember", key="note", content="creative private note")
    call(b, "remember", key="note", content="media private note")

    assert "creative private note" not in call(b, "context")
    assert "media private note" not in call(a, "context")


# ── the old server ────────────────────────────────────────────────────────

def test_the_legacy_server_is_known_broken_on_sdk_2():
    """Documents why this module exists. If this ever passes, the legacy server
    was fixed or the SDK regained the old API — either way, revisit."""
    from feather_db.integrations.mcp_server import create_server
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(Exception):
            create_server(db_path=str(Path(d) / "x.feather"), dim=64)
