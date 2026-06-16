"""
Feather DB — MCP Server
========================
Exposes the full Feather DB tool set as an MCP (Model Context Protocol) server.

Claude Desktop, Cursor, and any MCP-compatible agent can plug into this server
and get all 14 Feather tools as first-class MCP tools — zero code required.

Install:
    pip install "feather-db[mcp]"        # feather_db + the mcp SDK

Two backends:
    LOCAL   — open a .feather file directly (full 16-tool set):
        feather-serve --db my.feather --dim 3072
    REMOTE  — talk to a deployed Feather Cloud API (persona context engine);
              embedding runs client-side, so the server needs no embedder:
        feather-serve --api-url http://HOST:8000 --api-key KEY --namespace persona --dim 768
        # (or set FEATHER_API_URL / FEATHER_API_KEY in the env)

── Claude Desktop  (~/Library/Application Support/Claude/claude_desktop_config.json)
    LOCAL:
      { "mcpServers": { "feather": {
          "command": "feather-serve",
          "args": ["--db", "/path/to/my.feather", "--dim", "3072"] } } }
    REMOTE:
      { "mcpServers": { "feather": {
          "command": "feather-serve",
          "args": ["--api-url", "http://HOST:8000", "--namespace", "persona", "--dim", "768"],
          "env": { "FEATHER_API_KEY": "your-key" } } } }

── Claude Code   (project .mcp.json, or `claude mcp add`)
    LOCAL:
      { "mcpServers": { "feather": {
          "command": "feather-serve",
          "args": ["--db", "./persona.feather", "--dim", "3072"] } } }
    REMOTE (with a real embedder so recall is genuinely semantic):
      { "mcpServers": { "feather": {
          "command": "feather-serve",
          "args": ["--api-url", "http://HOST:8000", "--namespace", "persona",
                   "--dim", "768", "--embed-provider", "gemini"],
          "env": { "FEATHER_API_KEY": "your-key", "GOOGLE_API_KEY": "your-gemini-key" } } } }
    # CLI shortcut:
    #   GOOGLE_API_KEY=… claude mcp add feather -- feather-serve \
    #     --api-url http://HOST:8000 --namespace persona --dim 768 --embed-provider gemini

Embedder: pass --embed-provider gemini|openai|voyage|cohere|ollama for REAL semantic
embeddings (Gemini text-embedding-004 is native 768 and matches the hosted namespaces;
OpenAI text-embedding-3-small honours --dim 768 directly). Key via --embed-key or
FEATHER_EMBED_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY. Or --embedder <module.py>.
With neither, a deterministic HASH embedder is used — fine for wiring, weak for semantics.
The --dim must match the embedder AND (remote) the namespace dim.

Remote tools (8, persona context engine): feather_ingest, feather_recall,
feather_keyword_recall, feather_context_chain, feather_get_record, feather_link,
feather_stats, feather_list_namespaces.

Local tools (16 total):
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
  feather_ingest          Phase 9 structured ingestion (FactExtractor + EntityResolver)
  feather_recall          Hybrid retrieval with adaptive decay scoring
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
    db_path: str = None,
    dim: int = 3072,
    embed_fn=None,
    server_name: str = "feather-db",
    system_provider=None,
    namespace: str = "default",
    api_url: str = None,
    api_key: str = "",
) -> "Server":
    """
    Build and return an MCP Server instance with all Feather tools registered.

    Two backends:
      • local  — opens a `.feather` file directly (pass db_path).
      • remote — talks to a deployed Feather Cloud API over HTTP (pass api_url);
                 embedding happens client-side here so no server embedding config
                 is needed.
    """
    _require_mcp()

    if api_url:
        from feather_db.integrations.mcp_remote import RemoteFeatherTools, REMOTE_TOOL_SPECS
        tools = RemoteFeatherTools(
            api_url=api_url, api_key=api_key, namespace=namespace,
            dim=dim, embedder=embed_fn,
        )
        specs = REMOTE_TOOL_SPECS
    else:
        from feather_db.integrations.base import FeatherTools, TOOL_SPECS
        tools = FeatherTools(
            db_path=db_path, dim=dim, embedder=embed_fn,
            system_provider=system_provider, namespace=namespace,
        )
        specs = TOOL_SPECS

    server = Server(server_name)

    # ── Tool list handler ─────────────────────────────────────────────────────

    @server.list_tools()
    async def list_tools() -> list[mcp_types.Tool]:
        result: list[mcp_types.Tool] = []
        for spec in specs:
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
                "backend":    "remote" if api_url else "local",
                "target":     api_url or db_path,
                "namespace":  namespace,
                "dim":        dim,
                "tool_count": len(specs),
                "tools":      [s["name"] for s in specs],
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
        "--db",    default=None,
        help="LOCAL backend: path to the .feather file (created if absent). "
             "Omit when using --api-url.",
    )
    parser.add_argument(
        "--api-url", default=os.getenv("FEATHER_API_URL"),
        help="REMOTE backend: base URL of a deployed Feather Cloud API "
             "(e.g. http://host:8000). Env: FEATHER_API_URL. Embedding runs "
             "client-side, so the server needs no embedding config.",
    )
    parser.add_argument(
        "--api-key", default=os.getenv("FEATHER_API_KEY", ""),
        help="REMOTE backend: X-API-Key for the Cloud API. Env: FEATHER_API_KEY.",
    )
    parser.add_argument(
        "--dim",   type=int, default=3072,
        help="Vector dimension. Must match the embedder AND (remote) the "
             "namespace dim. Default 3072; use 768 for the hosted dim-768 namespaces.",
    )
    parser.add_argument(
        "--name",  default="feather-db",
        help="MCP server name shown in agent UIs (default: feather-db)",
    )
    parser.add_argument(
        "--embedder", default=None,
        help="Optional: path to a Python module exporting an `embed(text) -> list[float]` function",
    )
    parser.add_argument(
        "--embed-provider", default=os.getenv("FEATHER_EMBED_PROVIDER"),
        choices=["gemini", "openai", "voyage", "cohere", "ollama", "hash"],
        help="Real embedder for the persona engine: gemini (text-embedding-004, "
             "native 768) / openai / voyage / cohere / ollama / hash. Key from "
             "--embed-key or FEATHER_EMBED_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY. "
             "Env: FEATHER_EMBED_PROVIDER. Takes precedence over --embedder.",
    )
    parser.add_argument("--embed-model", default=os.getenv("FEATHER_EMBED_MODEL"),
                        help="Override the embedding model for --embed-provider.")
    parser.add_argument("--embed-key", default=None,
                        help="API key for --embed-provider (else taken from env).")
    parser.add_argument("--embed-base-url", default=os.getenv("FEATHER_EMBED_BASE_URL"),
                        help="Override base URL (e.g. Ollama host, Azure endpoint).")
    parser.add_argument(
        "--system-provider", default=None,
        choices=["claude", "openai", "gemini", "ollama"],
        help="Phase 9: LLM provider for FactExtractor + EntityResolver. "
             "Requires the matching API key env var (ANTHROPIC_API_KEY, "
             "OPENAI_API_KEY, GOOGLE_API_KEY). When omitted, feather_ingest "
             "falls back to raw text storage.",
    )
    parser.add_argument(
        "--system-model", default=None,
        help="Phase 9: override model for --system-provider.",
    )
    parser.add_argument(
        "--namespace", default="default",
        help="Default namespace for Phase 9 ingestion (default: 'default').",
    )
    args = parser.parse_args()

    _require_mcp()

    if not args.api_url and not args.db:
        print("ERROR: provide either --db <file.feather> (local) or "
              "--api-url <url> (remote).", file=sys.stderr)
        sys.exit(1)

    # Embedder: --embed-provider (real, built-in providers) takes precedence;
    # otherwise a custom --embedder module; otherwise the backend's default.
    embed_fn = None
    if args.embed_provider:
        from feather_db.integrations.embedders import make_embedder
        try:
            embed_fn = make_embedder(args.embed_provider, model=args.embed_model,
                                     api_key=args.embed_key, dim=args.dim,
                                     base_url=args.embed_base_url)
            print(f"  Embedder: {args.embed_provider} "
                  f"({args.embed_model or 'default'}, dim={args.dim})", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: embedder init failed: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.embedder:
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

    # Optional Phase 9 system provider
    system_provider = None
    if args.system_provider:
        try:
            from feather_db.providers import (
                ClaudeProvider, OpenAIProvider, GeminiProvider, OllamaProvider,
            )
            _pmap = {
                "claude": lambda m: ClaudeProvider(model=m or "claude-haiku-4-5-20251001"),
                "openai": lambda m: OpenAIProvider(model=m or "gpt-4o-mini"),
                "gemini": lambda m: GeminiProvider(model=m or "gemini-2.0-flash"),
                "ollama": lambda m: OllamaProvider(model=m or "llama3.1:8b"),
            }
            system_provider = _pmap[args.system_provider](args.system_model)
            print(f"  Phase 9 system provider: {args.system_provider} "
                  f"({args.system_model or 'default'})", file=sys.stderr)
        except Exception as e:
            print(f"WARNING: could not init system provider: {e}", file=sys.stderr)

    server = create_server(
        db_path         = args.db,
        dim             = args.dim,
        embed_fn        = embed_fn,
        server_name     = args.name,
        system_provider = system_provider,
        namespace       = args.namespace,
        api_url         = args.api_url,
        api_key         = args.api_key,
    )

    print(f"Feather DB MCP server starting…", file=sys.stderr)
    print(f"  Backend: {'remote ' + args.api_url if args.api_url else 'local ' + str(args.db)}", file=sys.stderr)
    print(f"  Namespace: {args.namespace}", file=sys.stderr)
    print(f"  Dim:  {args.dim}", file=sys.stderr)
    print(f"  Name: {args.name}", file=sys.stderr)

    if args.api_url:
        from feather_db.integrations.mcp_remote import REMOTE_TOOL_SPECS as _specs
    else:
        from feather_db.integrations.base import TOOL_SPECS as _specs
    print(f"  Tools ({len(_specs)}):", file=sys.stderr)
    for spec in _specs:
        print(f"    • {spec['name']}", file=sys.stderr)
    print("", file=sys.stderr)

    asyncio.run(stdio_server(server))


if __name__ == "__main__":
    main()
