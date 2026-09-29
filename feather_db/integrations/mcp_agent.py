"""Feather over MCP — agent memory as tools, resources and prompts.

The existing `mcp_server` exposes 16 low-level database operations and is broken
against the current SDK: `Server.list_tools` was removed in 2.0, so
`create_server()` raises on import-and-build. This module is the replacement, and
it is deliberately a different surface.

Sixteen database verbs make the model do memory management. A model handed
`feather_add_intel`, `feather_mmr_search` and `feather_consolidate` has to decide
which one means "remember this", and it usually decides not to. So this exposes
what an agent actually does — remember, recall, carry context — and keeps the
database underneath.

Three MCP capabilities, used for what each is good at:

  TOOLS      the model calls these when it decides to. Good for recall, weak for
             capture: models recall eagerly and write almost never.
  PROMPTS    the USER invokes these. `/remember` is the reliable capture path,
             because it does not depend on the model choosing to act.
  RESOURCES  the user attaches these. `feather://context` is the working set,
             readable without spending a tool call or a round trip.

Run:
    feather-agent --db agent.feather --scope hawky.brand_a.creative
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

import numpy as np

try:
    from mcp.server import MCPServer
    _MCP = True
except ImportError:                                    # pragma: no cover
    _MCP = False
    MCPServer = object                                 # type: ignore

import feather_db
from feather_db import DB
from feather_db.pocket import Pocket, SCOPE_SEP, _scope


def _require_mcp() -> None:
    if not _MCP:
        raise ImportError(
            "needs the MCP SDK 2.0+:  pip install 'feather-db[mcp]'\n"
            "(the SDK requires Python 3.10 or newer)"
        )


def build(
    db_path: str,
    scope: str,
    *,
    dim: int = 768,
    budget_tokens: int = 4000,
    embed=None,
) -> "MCPServer":
    """Build the server. Kept separate from main() so tests can drive it."""
    _require_mcp()

    db = DB.open(db_path, dim=dim)
    pkt = Pocket(db, _scope(scope), budget_tokens=budget_tokens, embed=embed)

    server = MCPServer(
        name="feather",
        title="Feather agent memory",
        instructions=(
            "Persistent memory for this agent, stored in a single local file.\n\n"
            "Call `recall` before answering anything that depends on earlier "
            "sessions. Call `remember` when the user states a durable preference, "
            "a decision, or a fact that should outlive this conversation — not for "
            "intermediate reasoning.\n\n"
            "The `feather://context` resource is the current working set. Read it "
            "instead of calling `recall` with a vague query."
        ),
    )

    # ── tools ────────────────────────────────────────────────────────────

    @server.tool(
        name="remember",
        title="Remember something durable",
        description=(
            "Store a fact, preference or decision that should outlive this "
            "conversation. Use a stable `key` — writing the same key again "
            "updates in place rather than creating a duplicate. Do NOT use this "
            "for intermediate reasoning or tool output."
        ),
    )
    def remember(key: str, content: str, importance: float = 1.0,
                 pinned: bool = False) -> str:
        pkt.remember(key, content, importance=importance, pinned=pinned)
        db.save()
        return f"remembered '{key}'" + (" (pinned)" if pinned else "")

    @server.tool(
        name="recall",
        title="Search memory",
        description=(
            "Search everything this agent knows, including memories too cold to "
            "be carried in the working set. A hit promotes the memory, so what "
            "gets recalled tends to stay available."
        ),
    )
    def recall(query: str, limit: int = 5) -> str:
        hits = pkt.recall(query, k=limit)
        if not hits:
            return "nothing found"
        return "\n".join(
            f"[{i.key}]{' (inherited)' if i.inherited else ''} {i.content}"
            for i in hits
        )

    @server.tool(
        name="context",
        title="Current working set",
        description=(
            "The memories worth carrying right now, ranked by how recently and "
            "how often they were used, trimmed to a token budget. Cheaper and "
            "more reliable than guessing a recall query."
        ),
    )
    def context(budget_tokens: Optional[int] = None) -> str:
        return pkt.hot_text(budget_tokens) or "(memory is empty)"

    @server.tool(
        name="forget",
        title="Forget a memory",
        description="Remove a memory by key. Use when the user says something is wrong or no longer true.",
    )
    def forget(key: str) -> str:
        pkt.forget(key)
        db.save()
        return f"forgot '{key}'"

    @server.tool(
        name="memory_stats",
        title="Inspect memory",
        description="What this agent is holding: hot/warm counts, token usage, and which scopes it inherits.",
    )
    def memory_stats() -> str:
        return json.dumps(pkt.stats(), indent=2)

    # ── resources ────────────────────────────────────────────────────────

    @server.resource(
        "feather://context",
        name="Working set",
        description="The memories this agent is currently carrying.",
        mime_type="text/plain",
    )
    def context_resource() -> str:
        return pkt.hot_text() or "(memory is empty)"

    @server.resource(
        "feather://stats",
        name="Memory stats",
        description="Tier counts, token usage and inherited scopes.",
        mime_type="application/json",
    )
    def stats_resource() -> str:
        return json.dumps(pkt.stats(), indent=2)

    # ── prompts ──────────────────────────────────────────────────────────
    # The capture path that does not depend on the model choosing to act.

    @server.prompt(
        name="remember",
        title="Remember this",
        description="Store something from the conversation as durable memory.",
    )
    def remember_prompt(what: str) -> str:
        return (
            f"Store this in Feather memory using the `remember` tool: {what}\n\n"
            "Choose a short stable key — reuse an existing one if this updates "
            "something already known, so it overwrites rather than duplicating. "
            "Write the content so it is still understandable months from now, "
            "without the surrounding conversation."
        )

    @server.prompt(
        name="what-do-you-know",
        title="What do you know about this?",
        description="Recall everything stored about a topic before answering.",
    )
    def recall_prompt(topic: str) -> str:
        return (
            f"Use the `recall` tool to find everything stored about: {topic}\n\n"
            "Then answer using what you found. If nothing is stored, say so "
            "plainly rather than guessing — an unremembered fact and a fact that "
            "does not exist are different things."
        )

    @server.prompt(
        name="catch-up",
        title="Catch up on this agent's memory",
        description="Load the working set before starting a task.",
    )
    def catchup_prompt() -> str:
        return (
            "Read the `feather://context` resource and summarise what you "
            "already know, grouping it into: standing rules that constrain what "
            "you may do, and findings from earlier work. Then ask what to do next."
        )

    return server


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Feather agent memory over MCP",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--db", default=os.getenv("FEATHER_DB", "agent_memory.feather"),
                    help="path to the .feather file (created if absent)")
    ap.add_argument("--scope", default=os.getenv("FEATHER_SCOPE", "agent"),
                    help="dotted memory scope, e.g. hawky.brand_a.creative — the "
                         "agent reads this scope and every ancestor")
    ap.add_argument("--dim", type=int, default=int(os.getenv("FEATHER_DIM", "768")))
    ap.add_argument("--budget", type=int, default=4000,
                    help="token budget for the working set")
    ap.add_argument("--embed-provider", default=os.getenv("FEATHER_EMBED_PROVIDER"),
                    choices=["gemini", "openai", "voyage", "cohere", "ollama"],
                    help="real embeddings for semantic recall; without one, "
                         "recall falls back to BM25 keyword search")
    args = ap.parse_args()

    _require_mcp()

    embed = None
    if args.embed_provider:
        from feather_db.integrations.embedders import make_embedder
        embed = make_embedder(args.embed_provider, dim=args.dim)

    server = build(args.db, args.scope, dim=args.dim,
                   budget_tokens=args.budget, embed=embed)

    print(f"feather agent memory → {args.db}", file=sys.stderr)
    print(f"  scope  : {args.scope}", file=sys.stderr)
    print(f"  budget : {args.budget} tokens", file=sys.stderr)
    print(f"  recall : {args.embed_provider or 'BM25 (no embedder)'}", file=sys.stderr)

    import asyncio
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
