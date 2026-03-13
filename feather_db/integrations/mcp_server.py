"""
Feather DB — MCP Server
========================
Exposes the full Feather DB tool set as an MCP (Model Context Protocol) server.

Claude Desktop, Cursor, and any MCP-compatible agent can plug into this server
and get all 14 Feather tools as first-class MCP tools — zero code required.

Install:
    pip install mcp feather-db

Run:
    python -m feather_db.integrations.mcp_server --db my.feather --dim 3072
    # or via entry point:
    feather-serve --db my.feather --dim 3072

Claude Desktop config (~/.claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "feather": {
          "command": "feather-serve",
          "args": ["--db", "/path/to/my.feather", "--dim", "3072"]
        }
      }
    }

Cursor config (.cursor/mcp.json):
    {
      "mcpServers": {
        "feather": {
          "command": "feather-serve",
          "args": ["--db", "/path/to/my.feather", "--dim", "3072"]
        }
      }
    }

Tools exposed (14 total):
  feather_search          Semantic search over the knowledge graph
  feather_context_chain   Vector search + BFS graph expansion (n hops)
  feather_get_node        Full metadata for a node by ID
  feather_get_related     Graph neighbours (edges in / out / both)
  feather_add_intel       Ingest a new intelligence node
  feather_link_nodes      Create a typed weighted edge between nodes
  feather_timeline        Chronological node list by product / entity
  feather_forget          Soft-delete a node (removes from search)
  feather_health          Knowledge graph health report
  feather_why             Score breakdown explaining a retrieval decision
  feather_mmr_search      MMR-diverse semantic search
  feather_consolidate     Cluster + merge similar nodes into summaries
  feather_episode_get     Retrieve ordered nodes in a named episode
  feather_expire          Scan and soft-delete TTL-expired nodes
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import os

# ── Try to import the MCP SDK ─────────────────────────────────────────────────
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types as mcp_types
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


def _require_mcp():
    if not _MCP_AVAILABLE:
        print(
            "ERROR: MCP SDK not installed.\n"
            "Install with:  pip install mcp\n"
            "Then re-run:   feather-serve --db my.feather --dim 3072",
            file=sys.stderr,
        )
        sys.exit(1)


# ── JSON schema conversion ────────────────────────────────────────────────────

def _spec_to_mcp_schema(spec: dict) -> dict:
    """Convert a TOOL_SPEC parameters dict to an MCP-compatible JSON Schema."""
    properties: dict = {}
    for pname, pdef in spec["parameters"].items():
        prop: dict = {"type": pdef["type"]}
        if "description" in pdef:
            prop["description"] = pdef["description"]
        if "enum" in pdef:
            prop["enum"] = pdef["enum"]
        properties[pname] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": spec.get("required", []),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Server factory
# ──────────────────────────────────────────────────────────────────────────────

def create_server(
    db_path: str,
    dim: int = 3072,
    embed_fn=None,
    server_name: str = "feather-db",
) -> "Server":
    """
    Build and return an MCP Server instance with all Feather tools registered.
    """
    _require_mcp()

    from feather_db.integrations.base import FeatherTools, TOOL_SPECS

    tools = FeatherTools(db_path=db_path, dim=dim, embedder=embed_fn)
    server = Server(server_name)

    # ── Tool list handler ─────────────────────────────────────────────────────

    @server.list_tools()
    async def list_tools() -> list[mcp_types.Tool]:
        result: list[mcp_types.Tool] = []
        for spec in TOOL_SPECS:
            result.append(mcp_types.Tool(
                name        = spec["name"],
                description = spec["description"],
                inputSchema = _spec_to_mcp_schema(spec),
            ))
        return result

    # ── Tool call handler ─────────────────────────────────────────────────────

    @server.call_tool()
    async def call_tool(
        name: str,
        arguments: dict,
    ) -> list[mcp_types.TextContent]:
        """Dispatch tool calls to FeatherTools.handle() in a thread executor."""
        loop = asyncio.get_event_loop()
        try:
            result_str = await loop.run_in_executor(
                None,
                tools.handle,
                name,
                arguments,
            )
        except Exception as exc:
            result_str = json.dumps({"error": str(exc), "tool": name})

        # Try to pretty-print JSON; fall through on non-JSON results
        try:
            parsed = json.loads(result_str)
            formatted = json.dumps(parsed, indent=2)
        except Exception:
            formatted = result_str

        return [mcp_types.TextContent(type="text", text=formatted)]

    # ── Resource: DB info ────────────────────────────────────────────────────

    @server.list_resources()
    async def list_resources() -> list[mcp_types.Resource]:
        return [
            mcp_types.Resource(
                uri         = f"feather://db/info",
                name        = "Feather DB Info",
                description = "Current DB path, size, and tool count",
                mimeType    = "application/json",
            )
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        if "info" in str(uri):
            return json.dumps({
                "db_path":    db_path,
                "dim":        dim,
                "tool_count": len(TOOL_SPECS),
                "tools":      [s["name"] for s in TOOL_SPECS],
            }, indent=2)
        return json.dumps({"error": "Unknown resource"})

    return server


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Feather DB MCP Server — expose Feather DB tools to any MCP-compatible agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db",    required=True,
        help="Path to the .feather file (created if it does not exist)",
    )
    parser.add_argument(
        "--dim",   type=int, default=3072,
        help="Vector dimension (default: 3072 for Gemini Embedding 2)",
    )
    parser.add_argument(
        "--name",  default="feather-db",
        help="MCP server name shown in agent UIs (default: feather-db)",
    )
    parser.add_argument(
        "--embedder", default=None,
        help="Optional: path to a Python module exporting an `embed(text) -> list[float]` function",
    )
    args = parser.parse_args()

    _require_mcp()

    # Optional custom embedder
    embed_fn = None
    if args.embedder:
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location("_custom_embedder",
                                                       pathlib.Path(args.embedder))
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        embed_fn = getattr(mod, "embed", None)
        if embed_fn is None:
            print(f"ERROR: {args.embedder} must export a function named `embed`",
                  file=sys.stderr)
            sys.exit(1)

    server = create_server(
        db_path     = args.db,
        dim         = args.dim,
        embed_fn    = embed_fn,
        server_name = args.name,
    )

    print(f"Feather DB MCP server starting…", file=sys.stderr)
    print(f"  DB:   {args.db}", file=sys.stderr)
    print(f"  Dim:  {args.dim}", file=sys.stderr)
    print(f"  Name: {args.name}", file=sys.stderr)

    from feather_db.integrations.base import TOOL_SPECS
    print(f"  Tools ({len(TOOL_SPECS)}):", file=sys.stderr)
    for spec in TOOL_SPECS:
        print(f"    • {spec['name']}", file=sys.stderr)
    print("", file=sys.stderr)

    asyncio.run(stdio_server(server))


if __name__ == "__main__":
    main()
