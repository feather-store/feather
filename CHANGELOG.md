# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.7.0] — 2026-03-21

### Added — Self-Aligned Context Engine (Phase 1)

#### `feather_db.providers` — LLM Provider Abstraction
- **`LLMProvider`** (ABC) — minimal `complete(messages, max_tokens, temperature) -> str` interface shared by all providers
- **`ClaudeProvider`** — Anthropic Claude via `anthropic` SDK. Default: `claude-haiku-4-5-20251001`. Separates `system` from conversation messages per Anthropic API spec.
- **`OpenAIProvider`** — OpenAI Chat Completions API and any compatible endpoint (Groq, Mistral, Together AI, vLLM, LM Studio). `json_mode=True` sets `response_format=json_object`.
- **`OllamaProvider`** — Ollama local server. Subclass of `OpenAIProvider`; default `base_url="http://localhost:11434/v1"`. No API key required.
- **`GeminiProvider`** — Google Gemini via `google-genai` SDK. Uses `response_mime_type="application/json"` for native JSON mode.
- All providers default to `temperature=0.0` for deterministic JSON output.

#### `feather_db.engine` — ContextEngine
- **`ContextEngine`** — self-aligned ingestion pipeline that wraps `DB` with LLM-powered classification.
- **`ingest(text, hint) -> int`** — 10-step pipeline: embed → sample context → LLM classify → apply hint → store → link → episode → watch triggers → contradiction check → auto-save.
- **`ingest_batch(texts, hints) -> list[int]`** — batch ingestion helper.
- LLM JSON schema: `{entity_type, importance, confidence, ttl, namespace, episode_id, suggested_links}`
- **`_heuristic_classify(text)`** — built-in keyword-based fallback; activated when `provider=None` or LLM fails. Fully offline, zero latency.
- 5-stage JSON extraction — direct parse → strip code fences → balanced brace scan → trailing comma repair → heuristic fallback. Robust on small/local models.
- Node ID: SHA-256 of `text[:200] + timestamp + PID` mod 2^50.
- Integrates with `WatchManager`, `EpisodeManager`, `ContradictionDetector` (all optional).

#### New exports in `feather_db`
- `LLMProvider`, `ClaudeProvider`, `OpenAIProvider`, `OllamaProvider`, `GeminiProvider`, `ContextEngine`

#### New example
- `examples/context_engine_demo.py` — auto-detects provider from env vars (Claude → OpenAI → Gemini → Ollama → heuristic), ingests 6 records, runs semantic search and context chain.

### Changed
- `feather_db.__version__` → `"0.7.0"`
- `README.md` — updated to v0.7.0 architecture; added Self-Aligned Context Engine, LLM agent connectors, MCP server, LangChain/LlamaIndex sections.

---

## [0.6.1] — 2026-03-14

### Fixed
- `setup.py`: `-undefined dynamic_lookup` linker flag now only applied on macOS — fixes build from source on Linux (HuggingFace Spaces, manylinux, Ubuntu)

## [0.6.0] — 2026-03-13

### Added — Complete Living Context Engine

#### C++ Core (file format v6)
- **`Metadata.ttl`** (int64_t) — time-to-live in seconds from `timestamp`. `0` = never expires. Enables working/session memory that auto-evicts.
- **`Metadata.confidence`** (float, default 1.0) — epistemic certainty about the stored fact. Separate from `importance` (relevance weight).
- **`DB.forget(id)`** — soft-delete: `markDelete` in HNSW (exits all searches), blanks `content`, sets `importance=0`. Graph shell preserved so incoming edges remain traversable.
- **`DB.purge(namespace_id)`** — hard-delete all nodes in a namespace. Removes from HNSW, `metadata_store_`, reverse index, and outgoing edges of surviving nodes. Returns count.
- **`DB.forget_expired()`** — scans all nodes, calls `forget()` on any where `ttl > 0 && now > timestamp + ttl`. Returns count of expired nodes.
- File format bumped to **v6**. v5, v4, v3 files load transparently via end-of-stream guards.

#### `feather_db.memory` — MemoryManager
- **`why_retrieved(db, id, query_vec)`** — full score breakdown: similarity, stickiness, effective_age_days, recency, importance, confidence, final_score, and human-readable formula string.
- **`health_report(db)`** — orphan nodes, hot/warm/cold tier counts, expired-TTL count, recall histogram, avg_importance, avg_confidence.
- **`search_mmr(db, query_vec, k, diversity)`** — Maximal Marginal Relevance post-processing. Balances relevance vs diversity. `diversity=0` = pure similarity, `diversity=1` = maximum spread.
- **`assign_tiers(db)`** — classifies nodes as hot/warm/cold by recall_count + recency percentile. Optionally writes `tier` attribute back to DB.
- **`consolidate(db, namespace, since_hours, llm_fn)`** — greedy union-find clustering by cosine similarity, generates summary nodes, links originals with `consolidated_into` edge, lowers original importance to 0.3.

#### `feather_db.triggers` — WatchManager + ContradictionDetector
- **`WatchManager.watch(db, query_text, threshold, callback, embed_fn)`** — register a semantic watch that fires `callback(node_id, similarity)` when a new node matches.
- **`WatchManager.check_triggers(db, new_node_id)`** — check all watches against a newly added node. Returns list of match records.
- **`ContradictionDetector.check(db, new_node_id, threshold, auto_link)`** — detect high-similarity nodes from a different source; auto-create `contradicts` edges.
- **`ContradictionDetector.scan_all(db)`** — full-DB contradiction scan (expensive; use sparingly).

#### `feather_db.episodes` — EpisodeManager
- **`begin_episode(db, episode_id, description)`** — create an episode header node; returns deterministic header ID.
- **`add_to_episode(db, node_id, episode_id)`** — tag node with episode_id attribute + link from header via `episode_contains` edge.
- **`get_episode(db, episode_id)`** — ordered member nodes by timestamp.
- **`close_episode(db, episode_id)`** — seals episode, adds `episode_end` edge, writes `episode_status=closed`.
- **`list_episodes(db)`** — all open and closed episodes.

#### `feather_db.merge`
- **`merge(target_db, source_path, conflict_policy)`** — merge two Feather DBs. Policies: `keep_target` (default), `keep_source`, `merge` (union attrs, higher importance/confidence wins). Returns `{merged, skipped, conflicts, edges_added}`.

#### `feather_db.integrations.langchain_compat`
- **`FeatherVectorStore`** — LangChain `VectorStore`: `add_texts()`, `similarity_search()`, `similarity_search_with_score()`, `from_texts()`, `from_documents()`.
- **`FeatherMemory`** — LangChain `BaseMemory`: semantic history retrieval with adaptive decay scoring, `save_context()`, `clear()`.
- **`FeatherRetriever`** — LangChain `BaseRetriever`: wraps `context_chain()` for graph-expanded retrieval.

#### `feather_db.integrations.llamaindex_compat`
- **`FeatherVectorStoreIndex`** — LlamaIndex `VectorStore`: `add()` (BaseNode), `query()` (VectorStoreQuery → VectorStoreQueryResult), `delete()`.
- **`FeatherReader`** — LlamaIndex `BaseReader`: `load_data(db_path, dim)` → list of LlamaIndex Documents.

#### 7 New Agent Tools (all providers — total 14 tools)
- **`feather_forget`** — soft-delete a node by ID
- **`feather_health`** — knowledge graph health report
- **`feather_why`** — score breakdown explaining a retrieval
- **`feather_mmr_search`** — diversity-aware semantic search
- **`feather_consolidate`** — cluster + merge similar nodes
- **`feather_episode_get`** — ordered nodes in a named episode
- **`feather_expire`** — scan and expire all TTL-exceeded nodes

#### `feather_db.integrations.mcp_server` — MCP Server
- **`feather-serve`** CLI entry point — `feather-serve --db my.feather --dim 3072`
- Full MCP server exposing all **14 Feather tools** as first-class MCP tool definitions
- Works with Claude Desktop, Cursor, and any MCP-compatible agent — zero code required
- Claude Desktop config: `{"mcpServers": {"feather": {"command": "feather-serve", "args": ["--db", "my.feather"]}}}`
- `asyncio.run_in_executor` wrapping for non-blocking tool dispatch
- Resource endpoint: `feather://db/info` returns DB stats

### Changed
- `feather_db.__version__` → `"0.6.0"`
- `pyproject.toml`: optional dependency groups `langchain`, `llamaindex`, `mcp`, `all`; `feather-serve` console script entry point
- `feather_db.integrations.__init__`: exports LangChain + LlamaIndex adapters (graceful fallback if deps absent)

---

### Added — LLM Agent Connectors (`feather_db.integrations`)

- **`feather_db/integrations/`** package — production-ready connectors that expose Feather DB as tool-use / function-calling tools for any LLM provider.
- **`ClaudeConnector`** — Feather DB tools in Anthropic `input_schema` format. Drop-in for `client.messages.create(tools=...)`. Includes `run_loop()` for fully autonomous multi-turn agent execution.
- **`OpenAIConnector`** — OpenAI function-calling format. Works with OpenAI (`gpt-4o`), Azure OpenAI, Groq (`llama-3.3-70b-versatile`), Mistral, Together AI, Ollama. Pass `base_url=` for any OpenAI-compatible endpoint.
- **`GeminiConnector`** — Gemini `FunctionDeclaration` + `types.Tool` format. Works with `client.chats.create(config=conn.chat_config())` and `google-genai` SDK.
- **`GeminiEmbedder`** — Multimodal embedder wrapping `models/gemini-embedding-2-preview` (3072-dim). Supports `embed_text()`, `embed_image()`, `embed_video_transcript()`, `embed_any()`. Mock mode (offline) built in.
- **`FeatherTools`** base class — 7 built-in tools available to all connectors:
  - `feather_search` — semantic search with namespace/entity/product filtering
  - `feather_context_chain` — vector search + BFS graph expansion (n hops)
  - `feather_get_node` — full metadata + edge inspection by ID
  - `feather_get_related` — graph neighbour traversal (in/out/both directions)
  - `feather_add_intel` — agent-ingestible new intelligence nodes
  - `feather_link_nodes` — typed weighted edge creation from agent output
  - `feather_timeline` — chronological node list by product or entity type
- All connectors share a single `TOOL_SPECS` definition (single source of truth) — adding a new tool propagates automatically to all three providers.
- Mock embedder fallback — all connectors work fully offline without an API key (useful for testing, CI, local dev).
- Top-level exports: `from feather_db import ClaudeConnector, OpenAIConnector, GeminiConnector, GeminiEmbedder`.
- **Example**: `examples/agent_connectors_demo.py` — complete multi-connector demo with context chain, timeline, write tools, and agent loop for all four providers (Claude, OpenAI, Groq, Gemini). Runs fully offline in mock mode.

---

## [0.5.0] — 2026-02-28

### Added — Context Graph & Living Context Engine

- **`Edge` struct** (C++): `{target_id, rel_type, weight}` — replaces flat `links` list. Backward-compat: old `links` IDs promoted to `Edge(rel="related_to", w=1.0)` on load. Python `meta.links` property still works.
- **Typed, weighted `db.link(from, to, rel_type, weight)`** — 9 built-in relationship types: `related_to`, `derived_from`, `caused_by`, `contradicts`, `supports`, `precedes`, `part_of`, `references`, `multimodal_of`. Free-form strings also accepted.
- **Reverse index** (`IncomingEdge`): in-memory index rebuilt on load; query via `db.get_incoming(id)`.
- **`db.get_edges(id)`** — outgoing edges with type + weight.
- **`db.auto_link(modality, threshold, rel_type, candidates)`** — auto-creates edges between records whose vector similarity exceeds `threshold`. Semantic relationship discovery without manual labeling.
- **`db.context_chain(query, k, hops, modality)`** — vector search + n-hop BFS graph expansion. Returns `ContextChainResult` with scored `ContextNode` list and `ContextEdge` list. Score = `similarity × hop_decay × importance × stickiness`.
- **`db.export_graph_json(namespace_filter, entity_filter)`** — D3/Cytoscape-compatible JSON. Dangling-edge safe: only emits edges where both source and target are in the exported node set.
- **`db.get_vector(id, modality)`** — retrieve raw vector for a given ID and modality.
- **`db.get_all_ids(modality)`** — list all IDs present in a modality index.
- **`feather_db.graph` module**: `export_graph()` → Python dict; `visualize()` → self-contained interactive D3 force-graph HTML (D3 inlined, no CDN dependency, works offline).
- **`feather_db.RelType`** — constant class with all 9 standard relationship type strings.
- **File format v5**: edges appended after attributes section; v3/v4 files load transparently.
- **Interactive Inspector** (`examples/feather_inspector.py`): local HTTP server with force graph, PCA embedding scatter view, top-8 similarity panel, edit and delete via REST API.
- **Example**: `examples/context_graph_demo.py` — full Nike campaign knowledge graph demo with auto-link, context_chain, multimodal pocket, HTML export.

### Changed
- `feather_db.__version__` bumped to `0.5.0`.
- `db.size()` method added — returns total record count.
- `visualize()` now inlines D3.js (280KB) — fully offline, no CDN needed.
- `export_graph_json` filters dangling edges (edges to nodes excluded by namespace filter).
- Force-graph simulation tuned for large graphs (>100 nodes): nodes pre-spread, edge labels hidden, faster alpha decay.

---

## [0.4.0] — 2026-02-20

### Added — Generic Living Context Engine (Namespace + Entity + Attributes)

- **Namespace + Entity + Attributes** (`Metadata` fields): `namespace_id` (partition key — brand, org, tenant), `entity_id` (subject key — user, customer, product), `attributes` (domain-specific KV map `map<string,string>`). C++ core remains fully domain-agnostic.
- **File format v4**: new fields appended after `last_recalled_at` in the binary layout; v3 files load transparently (missing fields default to empty).
- **`DB.update_metadata(id, meta)`**: replace the full metadata record for an existing ID without touching the HNSW index.
- **`DB.update_importance(id, importance)`**: targeted importance update for signal-feedback loops.
- **`meta.set_attribute(key, value)` / `meta.get_attribute(key, default)`**: safe KV helpers that bypass pybind11 map copy gotcha (`meta.attributes['k'] = v` silently does nothing; use these methods instead).
- **Filter fields**: `SearchFilter.namespace_id`, `SearchFilter.entity_id`, `SearchFilter.attributes_match` (all-KV-match semantics).
- **`FilterBuilder` methods**: `.namespace(ns)`, `.entity(eid)`, `.attribute(key, value)` (chainable, accumulates KV pairs).
- **`DomainProfile`** (Python base class): generic typed helpers `set_namespace()`, `set_entity()`, `set_attr()`, `get_attr()`, `to_metadata()`.
- **`MarketingProfile(DomainProfile)`**: digital marketing adapter — `set_brand()`, `set_user()`, `set_channel()`, `set_campaign()`, `set_ctr()`, `set_roas()`, `set_platform()` plus read-only properties.
- **Example**: `examples/marketing_living_context.py` — end-to-end multi-brand demo with namespace/entity/attribute filtering and importance feedback.

### Changed
- `feather_db.__version__` bumped to `0.4.0`.
- `feather_db.__all__` extended with `DomainProfile`, `MarketingProfile`.

---

## [0.3.0] — 2026-02-15

### Added — Multimodal Pockets + Contextual Graph + Living Context

- **Multimodal Pockets**: Support for storing multiple named signals (text, visual, audio) under a single Entity ID. Each modality gets its own independent HNSW index with its own dimensionality.
- **Contextual Graph**: Native variable-length edge lists (`links`) in Metadata for graph relationships.
- **Living Context / Sticky Memory**: Adaptive decay based on `recall_count` — frequently accessed records resist temporal decay. Formula: `stickiness = 1 + log(1 + recall_count)`, `effective_age = age / stickiness`.
- **API**: New `db.link(from, to)` and `db.search(..., modality="visual")` methods.
- **CLI**: Updates to `feather-cli` for multimodal ingestion and linking.

---

## [0.2.1] — 2026-01-20

### Added — Context Engine + Time-Decay Scoring

- **Context Engine**: Structured metadata attributes (type, source, creation_time).
- **Time-Decay Scoring**: `Scorer` class with half-life decay formula.
- **Filtered Search**: `FilterBuilder` and pre-filtering logic in HNSW.
- **Python Bindings**: `feather_db.Metadata` and `feather_db.ContextType` objects.

---

## [0.1.0] — 2025-11-16

### Added — Initial Release

- Core C++17 HNSW implementation with Python (pybind11) and Rust (CLI) bindings.
- Binary `.feather` file format with magic number validation.
- L2 (Euclidean) distance metric with SIMD optimizations (AVX512/AVX/SSE).
- Add rate: 2,000–5,000 vectors/second. Search time: 0.5–1.5ms per query (k=10).

---

[0.5.0]: https://github.com/feather-store/feather/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/feather-store/feather/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/feather-store/feather/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/feather-store/feather/compare/v0.1.0...v0.2.1
[0.1.0]: https://github.com/feather-store/feather/releases/tag/v0.1.0
