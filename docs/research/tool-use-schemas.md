---
id: research-tool-use-schemas
title: "Official LLM Tool Integration Spec — OpenAI + Anthropic"
status: research-complete
date: 2026-04-27
author: Hawky.ai (research agent)
---

# Official LLM Tool Integration Spec

> Source: research workstream for Phase 9 outbound surfaces. Decisions below feed `feather_db/integrations/openai_tools.py`, `anthropic_tools.py`, and `feather-cloud-sdk`.

## TL;DR

Ship **four tools** with both OpenAI (strict-mode) and Anthropic (rich-description) variants:

| Tool | Purpose | When to use |
|---|---|---|
| `feather_search` | Hybrid BM25+vector retrieval | Direct lookup of facts/docs/conversations by topic |
| `feather_context_chain` | Vector seed + n-hop graph BFS | Multi-hop reasoning across linked memories |
| `feather_ask` | **Phase 9 reasoner**: plans, retrieves, synthesizes, returns answer + provenance | Synthesis questions; user wants a final answer |
| `feather_add` | Persist a new memory | User states a stable preference / fact / decision |

Generate both vendor schemas from a **single source-of-truth** at `feather_db/integrations/_spec.json` so OpenAI and Anthropic variants stay in lockstep.

## Key behavioral differences (April 2026)

| Concern | OpenAI | Anthropic |
|---|---|---|
| Description length | Terse, action-first (1–3 sentences) | **Rich, instructive (3–4+ sentences) with "use when / don't use when"** |
| Strict mode | `strict: true` is default. Optional fields modeled as `["T", "null"]` and forced into `required`. `additionalProperties: false`. | Validated, not strict. Optional fields can be omitted. |
| Routing at scale | Degrades past 8–12 tools | Holds well to 20+ when descriptions disambiguate |
| Caching | n/a | `cache_control: ephemeral` on tool list — important for stable agent loops |
| Parameter style | snake_case + enums preferred | snake_case |

## Description-writing best practices

Anthropic's tool descriptions are **routing prompts, not docstrings**. The right description for `feather_ask` distinguishes it from `feather_search` along three axes:
1. **Capability surface** — what `feather_ask` does that search can't (planning, contradiction detection, synthesis).
2. **Use-when triggers** — "summarize", "have we decided", "is X consistent with Y".
3. **Don't-use-when triggers** — explicit pointers to `feather_search` (for raw hits) or `feather_context_chain` (for graph-only).

Per Anthropic March 2026 eval: this pattern reduced wrong-tool selection by **~40%** on ambiguous queries vs a single-paragraph description.

For OpenAI, compress the same logic into one sentence; OpenAI's router relies more on parameter shape than prose.

## Pitfalls + mitigations

| Pitfall | Mitigation |
|---|---|
| Overlapping `feather_search` ↔ `feather_ask` descriptions cause wrong-tool calls | Explicit "use X instead when..." cross-references in each |
| Ambiguous parameter names (`k` vs `top_k` across tools) | `top_k` for raw retrieval, `k` only for "seeds before expansion"; document in description |
| LLM doesn't know what to do on empty results | Return `{"results": [], "reason": "no_match_above_threshold"}` not `[]` |
| Same tool fails on OpenAI strict but works on Anthropic (or vice versa) | Generate from one source spec |
| Over-broad `metadata`/`filter` objects (`additionalProperties: true`) → hallucinated keys | Enumerate supported keys in description, or use closed schema |
| `feather_add` re-writes same memory in agent loops | Server hashes content for dedupe; surface `dedupe_window_seconds` |
| Token bloat from rich Anthropic descriptions | `cache_control: ephemeral` on last tool |
| Namespace omitted by LLM → cross-tenant data leak | **Make namespace server-side from auth context, never trust LLM tool call** |

## Implementation files (to create)

```
feather_db/integrations/
├── _spec.json                 source of truth — both vendors generated from here
├── openai_tools.py            OPENAI_TOOLS list + dispatch helper
├── anthropic_tools.py         ANTHROPIC_TOOLS list + dispatch helper
├── langchain_tools.py         @tool wrappers
└── llamaindex_tools.py        FunctionTool factories

feather-api/app/routes/tools.py   REST endpoints these dispatch into
                                  (/v1/search, /v1/context_chain, /v1/ask, /v1/add)
```

## Full schemas + SDK wiring examples

See the [appendix below this line](#full-schemas) for:
- Full OpenAI + Anthropic JSON schemas for all 4 tools
- 10–20 line SDK wiring examples for OpenAI Agents, Anthropic SDK, LangChain, LlamaIndex

---

## <a name="full-schemas"></a>Appendix — Full schemas

### `feather_search`

**OpenAI:**
```json
{
  "type": "function",
  "name": "feather_search",
  "description": "Hybrid semantic + BM25 search over the agent's memory. Returns top-k matching memories with scores. Use for direct lookup of facts, prior conversations, or documents by topic.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Natural-language search query."},
      "top_k": {"type": "integer", "description": "Number of results.", "default": 10, "minimum": 1, "maximum": 100},
      "namespace": {"type": ["string", "null"], "description": "Memory partition (e.g. user_id or tenant). Null = default."},
      "filter": {"type": ["object", "null"], "description": "Optional metadata filter, e.g. {\"type\":\"FACT\",\"importance_gte\":0.7}.", "additionalProperties": true}
    },
    "required": ["query", "top_k", "namespace", "filter"],
    "additionalProperties": false
  },
  "strict": true
}
```

**Anthropic:**
```json
{
  "name": "feather_search",
  "description": "Search the agent's persistent memory using hybrid semantic + BM25 retrieval. Returns the top-k most relevant memories ranked by combined vector similarity and keyword score, with adaptive decay applied (frequently-recalled items are boosted).\n\nUse this tool when the user asks about a specific fact, document, prior message, or topic that may exist in memory. Prefer this over `feather_ask` for simple lookups where you only need raw evidence, not a synthesized answer. Prefer `feather_context_chain` when the question requires following relationships across memories (e.g. \"why did X happen\").\n\nResults are returned as `[{id, content, score, metadata}]` ordered by descending score.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "The natural-language query. Be specific - keywords matter for BM25 component."},
      "top_k": {"type": "integer", "description": "Maximum number of results to return.", "default": 10, "minimum": 1, "maximum": 100},
      "namespace": {"type": "string", "description": "Memory partition key (e.g. user ID, tenant ID, project slug). Omit to search the default namespace."},
      "filter": {"type": "object", "description": "Metadata constraints. Supported keys: type, source, source_prefix, importance_gte, timestamp_after, timestamp_before, tags_contains, attributes_match.", "additionalProperties": true}
    },
    "required": ["query"]
  }
}
```

### `feather_context_chain`

**OpenAI:**
```json
{
  "type": "function",
  "name": "feather_context_chain",
  "description": "Multi-hop reasoning retrieval: runs a semantic search, then walks the typed memory graph N hops outward to surface causally or relationally connected memories. Use when the question requires connecting facts across sessions or following 'why/because' chains.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Seed query for the initial vector search."},
      "k": {"type": "integer", "description": "Seeds before graph expansion.", "default": 5, "minimum": 1, "maximum": 50},
      "hops": {"type": "integer", "description": "Graph traversal depth.", "default": 2, "minimum": 1, "maximum": 4},
      "namespace": {"type": ["string", "null"], "description": "Memory partition. Null = default."}
    },
    "required": ["query", "k", "hops", "namespace"],
    "additionalProperties": false
  },
  "strict": true
}
```

**Anthropic:**
```json
{
  "name": "feather_context_chain",
  "description": "Retrieve memories via vector search and then expand the result set by walking the typed memory graph (caused_by, supports, contradicts, related_to, etc.) up to N hops. Each returned node includes its hop distance from the seed and the edge types traversed.\n\nUse this when the user's question involves reasoning across linked memories - causation, justification, history of a decision, or 'why did the agent do X'. Do NOT use this for simple factual lookup (use `feather_search`). Do NOT use this when you need a synthesized natural-language answer (use `feather_ask`).\n\nReturns `[{id, content, hop_distance, edge_types}]` where hop_distance=0 means a direct vector hit and >0 means reached via graph traversal.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Seed query. The vector search runs first, then graph expansion fans out."},
      "k": {"type": "integer", "description": "Number of seed nodes from the initial vector search before expansion.", "default": 5},
      "hops": {"type": "integer", "description": "How many graph hops to traverse outward from each seed. 1-2 is typical; 3+ is expensive and noisy.", "default": 2},
      "namespace": {"type": "string", "description": "Memory partition. Omit for default."}
    },
    "required": ["query", "k"]
  }
}
```

### `feather_ask` (the Phase 9 reasoner)

**OpenAI:**
```json
{
  "type": "function",
  "name": "feather_ask",
  "description": "Ask Feather's reasoner a natural-language question. It plans retrieval, gathers evidence, detects contradictions, and returns a synthesized answer with citations. Use when you need a final answer grounded in memory, not raw search hits.",
  "parameters": {
    "type": "object",
    "properties": {
      "question": {"type": "string", "description": "The natural-language question to answer from memory."},
      "namespace": {"type": "string", "description": "Memory partition to reason over."}
    },
    "required": ["question", "namespace"],
    "additionalProperties": false
  },
  "strict": true
}
```

**Anthropic:**
```json
{
  "name": "feather_ask",
  "description": "Ask Feather's built-in reasoner a natural-language question against the agent's memory. The reasoner internally: (1) plans a retrieval strategy, (2) executes hybrid search and graph expansion, (3) ranks evidence, (4) detects contradictions between memories, and (5) returns a synthesized answer with full provenance.\n\nUse this tool when:\n - The user asks a question requiring synthesis across multiple memories ('summarize what we decided about X', 'has the user mentioned Y before').\n - You need a grounded answer with citations rather than raw search results.\n - You want contradictions surfaced (e.g. user said one thing in March, opposite in April).\n\nDo NOT use this tool when:\n - You only need raw matching memories - use `feather_search` instead (cheaper, faster).\n - You only need to traverse the graph - use `feather_context_chain`.\n - You are writing a memory - use `feather_add`.\n\nReturns `{answer, plan, evidence: [{id, content, why_chosen}], contradictions: [{id_a, id_b, reason}]}`. Always pass the `evidence` IDs through to the user when citing.",
  "input_schema": {
    "type": "object",
    "properties": {
      "question": {"type": "string", "description": "The user's question, as a complete natural-language sentence. The reasoner handles its own query planning, so do not pre-decompose."},
      "namespace": {"type": "string", "description": "Memory partition to reason over (typically the user_id or tenant)."}
    },
    "required": ["question", "namespace"]
  }
}
```

### `feather_add`

**OpenAI:**
```json
{
  "type": "function",
  "name": "feather_add",
  "description": "Persist a new memory. Use to record facts learned about the user, decisions made, or observations the agent should remember across sessions. Do NOT use for transient chat turns the user did not ask to be remembered.",
  "parameters": {
    "type": "object",
    "properties": {
      "content": {"type": "string", "description": "The memory text. Write in third-person declarative form, e.g. 'User prefers dark mode'."},
      "namespace": {"type": "string", "description": "Memory partition (typically user_id or tenant)."},
      "metadata": {"type": ["object", "null"], "description": "Optional metadata: {type, importance, tags, attributes}.", "additionalProperties": true}
    },
    "required": ["content", "namespace", "metadata"],
    "additionalProperties": false
  },
  "strict": true
}
```

**Anthropic:**
```json
{
  "name": "feather_add",
  "description": "Write a new memory to the agent's persistent store. The memory is embedded, indexed for both vector and BM25 search, and timestamped. Future calls to feather_search / feather_ask / feather_context_chain will be able to retrieve it.\n\nUse this tool when:\n - The user states a stable preference or fact ('I'm vegetarian', 'my project is called Atlas').\n - A decision is made that future sessions should remember.\n - The user explicitly asks the agent to remember something.\n\nDo NOT use this tool for:\n - Routine conversation turns (those are handled by the chat transcript layer).\n - Information the user said was temporary or hypothetical.\n - Duplicates - search first if unsure whether the fact is already stored.\n\nReturns `{id, indexed_at}`.",
  "input_schema": {
    "type": "object",
    "properties": {
      "content": {"type": "string", "description": "The memory content as declarative text. Be self-contained - the memory will be retrieved without surrounding context."},
      "namespace": {"type": "string", "description": "Memory partition. Almost always the user_id or tenant_id."},
      "metadata": {"type": "object", "description": "Optional: {type: 'FACT'|'PREFERENCE'|'EVENT'|'CONVERSATION', importance: 0-1, tags: [...], attributes: {...}}.", "additionalProperties": true}
    },
    "required": ["content", "namespace"]
  }
}
```
