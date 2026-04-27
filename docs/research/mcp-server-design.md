---
id: research-mcp-server-design
title: "MCP Server Best Practices for Memory/Vector Databases"
status: research-complete
date: 2026-04-27
author: Hawky.ai (research agent)
---

# MCP Server Design — Best Practices for Memory DBs

> Source: research workstream for Phase 9 outbound surfaces. Decisions below feed `feather_db/integrations/mcp_server.py` redesign and the Cloud-hosted MCP endpoint.

## TL;DR

| Decision | Recommendation |
|---|---|
| **Tool count** | **6 tools max** (currently 14 — over the cliff) |
| **Tool surface** | `feather.search`, `feather.context_chain`, `feather.ask`, `feather.add`, `feather.link`, `feather.list_namespaces` |
| **Drop / move to resources** | `get_node`, `get_related`, `timeline`, `health`, `why`, `episode_get` → MCP resources at `feather://node/{id}` |
| **Drop entirely from LLM-facing surface** | `expire`, `consolidate`, `forget`, `mmr_search` → admin HTTP only |
| **OSS transport** | stdio only (Claude Desktop / Cursor / Cline / Windsurf compat) |
| **Cloud transport** | Streamable HTTP + OAuth 2.1 + PKCE (mandatory per spec for hosted) |
| **Cloud tenancy** | `tenant_id` from JWT claim — **never** trust namespace argument from LLM |
| **Distribution priorities** | (1) Smithery.ai · (2) modelcontextprotocol/servers README · (3) Glama.ai |

## The tool-count cliff (2026 evidence)

| Tools | Behavior |
|---|---|
| 10 tools | Perfect routing |
| 20 tools | 19/20 correct |
| 107 tools | Models fail |

GitHub Copilot cut 40→13. Block cut Linear MCP 30→2. A 106-tool MySQL server consumed **207KB / ~54.6K tokens per request** before any actual call. Feather's current 14-tool surface is at the danger edge — Phase 9 cuts to 6.

## Recommended tool surface (full schemas)

```jsonc
// 1. feather.search — hybrid retrieval, replaces feather_search + feather_mmr_search
{
  "name": "feather.search",
  "description": "Hybrid semantic + BM25 search over Feather memory.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query":  {"type": "string"},
      "top_k":  {"type": "integer", "default": 8, "maximum": 50},
      "mode":   {"type": "string", "enum": ["hybrid","semantic","bm25","mmr"], "default": "hybrid"},
      "filter": {
        "type": "object",
        "properties": {
          "namespace":  {"type": "string"},
          "entity":     {"type": "string"},
          "type":       {"type": "string", "enum": ["FACT","PREFERENCE","EVENT","CONVERSATION"]},
          "since":      {"type": "string", "format": "date-time"},
          "attributes": {"type": "object", "additionalProperties": {"type": "string"}}
        }
      }
    },
    "required": ["query"]
  }
}

// 2. feather.context_chain — vector + n-hop graph BFS
// 3. feather.ask — Phase 9 reasoner (returns answer + plan + provenance)
// 4. feather.add — write
// 5. feather.link — typed edges (graph is core to Feather's value prop)
// 6. feather.list_namespaces — discoverability
```

## Transport landscape (April 2026)

| Transport | Status | Where it works |
|---|---|---|
| **stdio** | Stable, dominant for local | Claude Desktop (native); Cursor, Cline, Windsurf, Claude Code |
| **Streamable HTTP** | Replaced HTTP+SSE in March 2025 spec | Cursor, Claude Code, Cloudflare/Koyeb gateways. Claude Desktop only via `mcp-remote` proxy |
| **HTTP+SSE** | Deprecated (back-compat only) | Older clients |

Streamable HTTP uses one endpoint supporting POST + GET, optional `Mcp-Session-Id` headers, designed to work behind load balancers. The 2026 roadmap is **not** adding more transports — it's evolving session/state semantics for horizontal scaling without sticky sessions.

## Auth + multi-tenancy (Cloud)

The spec mandates **OAuth 2.1 with PKCE** for public remote servers. MCP server acts as OAuth 2.1 Resource Server.

```
JWT claims:
  tenant_id    (mandatory, derived from auth)
  scopes:      feather:read · feather:write · feather:admin
  exp / iat    (standard)

At every tool invocation:
  1. Validate JWT
  2. Extract tenant_id from claims
  3. Use that tenant — IGNORE any namespace arg the LLM passed
  4. Apply scope check (read vs write tools)
```

Authorization Server can be co-hosted (Cloud control plane) or delegated (Auth0, Clerk, WorkOS). API-key fallback only for first-party CLI/CI on a separate non-MCP HTTP endpoint.

## Memory MCP servers in the wild

| Server | Pattern | Tools |
|---|---|---|
| `@modelcontextprotocol/server-memory` (official ref) | Knowledge-graph CRUD: Entities → Observations → Relations | 9 tools (create_entities, create_relations, add_observations, ...) |
| Mem0 / OpenMemory MCP | Multi-signal retrieval (semantic + BM25 + entity match, RRF) | 9 tools (add_memories, search_memory, list_memories, ...) |
| Letta (MemGPT) | Three-tier (Core / Recall / Archival), agents call tools to read/edit memory blocks | Schemas only — execution external |
| memento-mcp | KG + temporal awareness | 8 tools |
| mcp-memory-service | LangGraph/CrewAI/Claude integration with consolidation | 6 tools |
| cognee | Graph-aware RAG over MCP | 5 tools |

**Common pattern across all:** small surface (4–9 tools), JSON outputs with `{id, content, score, source, timestamp}`, namespace/user-scoped.

## Distribution channels (in priority)

1. **Smithery.ai** — `npx -y @smithery/cli install <server> --client claude` writes the config automatically. Largest installable MCP registry.
2. **Official MCP Registry** at `modelcontextprotocol.io` — listing in `modelcontextprotocol/servers` README brings the highest credibility signal; auto-feeds MCPfinder/Glama.
3. **Glama.ai** — second-largest registry, strong SEO, programmatic discovery API.
4. **Cursor `.cursor/mcp.json`** + Cline/Windsurf equivalents — same JSON shape as Claude Desktop.
5. **PyPI / npm** — table stakes for actual install, but not how users discover.

## Pitfalls (must avoid)

| Pitfall | Mitigation |
|---|---|
| Tool count cliff (>20 tools degrade) | Hold to 6 |
| Schema bloat / token tax | Outcome-oriented tools; don't 1:1 wrap REST endpoints |
| API-wrapper anti-pattern (one tool per endpoint) | Polymorphic tools (e.g. `search` with `mode` enum) |
| No provenance | Every result has `id`, `source`, `score`, `timestamp` |
| Over-eager retrieval | Cap `top_k` at 50; default 8 |
| Stateful sessions in HTTP | Stateless where possible; rely on auth for session identity |
| Trusting tool args for tenant isolation | **Always** derive `tenant_id` from token, not arguments |

## Implementation plan

**OSS (refactor existing `feather_db/integrations/mcp_server.py`):**
- [ ] Trim 14 tools → 6
- [ ] Keep stdio transport
- [ ] Move `get_node`, `health`, `why`, `timeline` to MCP resources at `feather://*` URIs
- [ ] Drop `expire`, `consolidate`, `forget`, `mmr_search` from LLM surface (keep as Python/CLI)
- [ ] Update `feather_db/integrations/base.py::TOOL_SPECS`

**Cloud (new):**
- [ ] Streamable HTTP server (FastAPI base)
- [ ] OAuth 2.1 + PKCE; well-known endpoint at `/.well-known/oauth-authorization-server`
- [ ] JWT tenant scoping middleware
- [ ] Mount the same 6 tools, different dispatcher
- [ ] Distribute via Smithery + register in MCP servers repo

## Sources

- [MCP spec — Transports (2025-03-26)](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports)
- [MCP spec — Authorization](https://modelcontextprotocol.io/specification/draft/basic/authorization)
- [The 2026 MCP Roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/)
- [Cloudflare Agents MCP transport](https://developers.cloudflare.com/agents/model-context-protocol/transport/)
- [Mem0 — Introducing OpenMemory MCP](https://mem0.ai/blog/introducing-openmemory-mcp)
- [Letta memory guide](https://docs.letta.com/guides/agents/memory/)
- [Six-Tool Pattern](https://www.mcpbundles.com/blog/mcp-tool-design-pattern)
- [STRAP pattern (96→10 tools)](https://almatuck.com/articles/reduced-mcp-tools-96-to-10-strap-pattern)
- [Junia.ai — Tool count cliff](https://www.junia.ai/blog/mcp-context-window-problem)
- [Layered.dev — Schema bloat](https://layered.dev/mcp-tool-schema-bloat-the-hidden-token-tax-and-how-to-fix-it/)
- [Smithery.ai](https://smithery.ai)
- [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)
