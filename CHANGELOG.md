# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
