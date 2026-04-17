# Feather DB: An Embedded Living Context Engine for AI-Native Applications

**Version 1.0 — April 2026**
**Hawky.ai — hello@hawky.ai — https://www.getfeather.store**

---

## Executive Summary

AI applications have a memory problem. Large language models are stateless — every call starts from zero. Existing solutions bolt on external vector databases that require dedicated servers, network round-trips, and complex DevOps. The result is architectures that are expensive to run, slow to query, and architecturally mismatched to the problem.

**Feather DB** is an embedded vector database and living context engine designed from first principles for AI-native applications. It runs in-process, stores everything in a single file, and delivers sub-millisecond approximate nearest neighbour (ANN) search — with no server, no network, and no operational overhead. More importantly, it introduces the concept of **living context**: memory that adapts over time, prioritising what is frequently recalled and allowing dormant knowledge to naturally fade, exactly as human memory works.

This white paper explains the problem, the architecture, the benchmark results, and the positioning of Feather DB as the foundational memory layer for AI products.

---

## 1. The Problem: AI Applications Have No Long-Term Memory

### 1.1 The Stateless LLM Problem

Every call to a large language model (LLM) is independent. The model has no memory of prior interactions, no awareness of accumulated user context, and no ability to learn from past queries. This forces application developers to solve memory externally — and current solutions are inadequate.

### 1.2 Limitations of Existing Vector Databases

| System | Deployment | Latency | Operational Cost | Context-Aware |
|--------|-----------|---------|-----------------|---------------|
| Pinecone | Cloud-only | ~50–200ms (network) | High ($$$) | No |
| Weaviate | Server | ~5–20ms (local) | Medium | No |
| Chroma | Embedded | ~2–10ms | Low | No |
| Qdrant | Server/Cloud | ~5–30ms | Medium | No |
| FAISS | Library | ~0.5ms | Zero | No |
| **Feather DB** | **Embedded** | **~0.1ms** | **Zero** | **Yes** |

The core problem is that existing systems treat memory as **static storage** — they store and retrieve vectors, but have no model of *relevance over time*, *relational context*, or *adaptive prioritisation*.

### 1.3 What AI Applications Actually Need

A memory layer for AI applications needs to answer four questions simultaneously:

1. **Semantic similarity**: What do I know that is conceptually close to this query?
2. **Recency relevance**: Is this memory still current, or has it aged out?
3. **Relational context**: What else is connected to this memory?
4. **Salience**: Has this memory proven useful before?

No existing embedded system answers all four. Feather DB does.

---

## 2. Architecture

### 2.1 Core Design Principles

Feather DB is built around four architectural decisions:

**Embedded-first**: The database runs as a library within the application process. There is no server to deploy, no network to cross, and no serialisation overhead. Queries touch the index directly in memory.

**File-based persistence**: All data — vectors, metadata, graph edges, BM25 index — persists in a single `.feather` binary file. This makes backup, replication, and deployment trivial.

**Shared-metadata, multi-index**: Each *entity* (e.g., a user, a document, a product) has one `Metadata` object shared across all its modality indices (text, image, audio). This prevents metadata duplication while allowing heterogeneous vector dimensions.

**Living context as a first-class primitive**: Relevance is not static. The scoring layer is not a post-processing step — it is built into the search pipeline.

### 2.2 HNSW Index

Feather DB uses a hierarchical navigable small world (HNSW) graph index for approximate nearest neighbour search, with parameters:

- `M = 16` — number of bidirectional links per node
- `ef_construction = 200` — beam width during index construction
- `ef = 10` — beam width during search (tunable)
- `max_elements = 1,000,000` per modality index

HNSW provides `O(log n)` query complexity and sub-linear construction time, making it practical for both small (< 1k) and large (1M+) vector collections.

### 2.3 Multimodal Pockets

Each named *modality* (e.g., "text", "visual", "audio") receives an independent HNSW index. Modality indices are created on-demand at first insertion. This means:

- Text and image vectors for the same entity can have different dimensions (768 vs 512).
- Search is always within a single modality, preventing dimension mismatch.
- A single metadata record anchors all modalities for an entity.

### 2.4 The Living Context Engine

The living context engine is the core differentiator. It implements **adaptive temporal decay**: a scoring formula that combines vector similarity, recency, and access frequency into a single relevance score.

```
stickiness    = 1 + log(1 + recall_count)
effective_age = age_days / stickiness
recency       = 0.5 ^ (effective_age / half_life_days)
final_score   = ((1 - w) × similarity + w × recency) × importance
```

**Key insight**: Records that are frequently recalled resist decay. A memory accessed 20 times ages at 1/4 the rate of a memory never recalled. This mirrors episodic memory consolidation in human cognition — frequently activated memories become long-term.

The `ScoringConfig` struct exposes three parameters:
- `half_life` (default 30 days): how fast unaccessed memories decay
- `weight` (default 0.3): blend between similarity and recency
- `min_weight` (default 0.0): floor on the recency multiplier

### 2.5 BM25 Inverted Index

Feather DB embeds a full Okapi BM25 inverted index over record content fields, enabling exact keyword and phrase search. The BM25 parameters are:

- `k1 = 1.2` — term frequency saturation
- `b = 0.75` — document length normalisation
- IDF formula: `log((N - n_t + 0.5)/(n_t + 0.5) + 1)`

The inverted index is:
- Built in-memory from `metadata_store_` content on every `open()`
- Updated incrementally on every `add()` and `update_metadata()` call
- Filtered by the same `SearchFilter` as vector search

No additional storage format is required — BM25 is reconstructed from the existing metadata on load, maintaining full backward compatibility with all `.feather` file versions.

### 2.6 Hybrid Search via Reciprocal Rank Fusion

Hybrid search merges BM25 keyword rankings and HNSW vector rankings using Reciprocal Rank Fusion (RRF):

```
score(d) = Σ  1 / (rrf_k + rank_i(d) + 1)
```

where `rrf_k = 60` (default) dampens the influence of very high ranks. RRF is rank-based, not score-based, making it robust to score scale differences between BM25 and cosine similarity.

**Precision improvement**: In our 3,000-record benchmark across 10 topic categories, hybrid search achieved 70% topic precision vs 44% for keyword-only, while vector search achieved 100% on well-separated topic clusters. Hybrid search is the recommended default for production use.

### 2.7 Context Graph

Every record can participate in a typed, weighted knowledge graph via `Edge` structs:

```cpp
struct Edge {
    uint64_t    target_id;  // destination record
    std::string rel_type;   // "caused_by", "references", "similar_to", ...
    float       weight;     // [0.0, 1.0]
};
```

The graph supports:
- `link(from, to, rel_type, weight)` — manual edge creation
- `auto_link(threshold)` — automatic edge creation for vectors above similarity threshold
- `get_edges(id)` / `get_incoming(id)` — outgoing and incoming edge traversal
- `context_chain(query, k, hops)` — vector search + n-hop BFS expansion

The `context_chain` operation is particularly powerful for AI applications: it finds the semantically closest records, then expands outward along graph edges, returning a subgraph of related context. This is the building block for knowledge retrieval, citation chains, and causal reasoning.

### 2.8 Production-Hardening (v0.8.0)

| Feature | Implementation |
|---------|---------------|
| Thread safety | `std::mutex` per DB instance; `std::lock_guard` on all public methods |
| Atomic saves | Write to `path.tmp`, `std::rename()` (POSIX atomic), clear WAL |
| Write-ahead log | Binary append-only `.wal` file; replayed on `open()`; cleared on `save()` |
| Soft delete + compact | `importance=0.0` + `_deleted=true`; `compact()` rebuilds HNSW |
| Cursor pagination | `GET /records?limit=N&after=ID` for efficient large dataset browsing |

### 2.9 File Format (v5)

The `.feather` binary format is self-contained and version-tagged:

```
[magic: FEAT] [version: 5]
[metadata section: count, per-record serialized structs]
[modality section: count, per-modality name + dim + (id, vector) pairs]
```

Format versions 3 and 4 load transparently via guarded field reads. This ensures that upgrading Feather DB never requires a migration step.

---

## 3. Benchmark Results

All benchmarks were run on Apple M-series hardware. Dataset: 3,050 records, 256-dimensional vectors, 3 namespaces, 10 topic categories.

### 3.1 Ingestion Throughput

| Operation | Time |
|-----------|------|
| 3,000 records ingested | 1,151ms |
| **Throughput** | **2,606 records/sec** |
| Save to disk | 23ms |
| Cold open (3,050 records) | 1,023ms |

### 3.2 Search Latency

| Search Type | Latency (p50) |
|-------------|---------------|
| Vector (HNSW, k=10) | **0.10ms** |
| Keyword (BM25, k=5) | **0.06ms** |
| Hybrid (RRF, k=10) | **0.24ms** |
| Context chain (k=3, hops=2) | **0.10ms** |

### 3.3 Search Quality (Topic Precision @ k=5)

| Query | Vector | Keyword | Hybrid |
|-------|--------|---------|--------|
| "invoice billing payment" | 5/5 | 0/5 | 2/5 |
| "AES-256 encryption TLS audit" | 5/5 | 5/5 | 5/5 |
| "HNSW benchmark millisecond" | 5/5 | 1/5 | 2/5 |
| "dark mode notification digest" | 5/5 | 5/5 | 5/5 |
| "SSO SAML OAuth login error" | 5/5 | 0/5 | 3/5 |
| **Average** | **100%** | **44%** | **70%** |

*Note: Vector search dominates when embeddings are available. Hybrid is the recommended default when both query vectors and keyword terms are present.*

### 3.4 Concurrency

| Metric | Value |
|--------|-------|
| Threads | 8 |
| Queries per thread | 50 |
| Total time | 26ms |
| **Queries/sec** | **15,119** |
| Thread errors | **0** |

### 3.5 Reliability

| Test | Result |
|------|--------|
| WAL crash recovery (50 records, no save) | **100%** |
| Atomic save (no corruption on kill) | **Pass** |
| compact() (100 soft-deleted records) | **100 removed** |
| Filter correctness (namespace / entity / attribute) | **100%** |

---

## 4. Use Cases

### 4.1 AI Chatbot Long-Term Memory

Store every conversation turn as a vector + metadata. On each new message, run hybrid search to retrieve relevant past context, inject it into the LLM prompt, and update recall counts. The adaptive decay ensures that frequently cited memories persist while irrelevant history fades naturally.

```python
db = feather_db.DB.open("user_123.feather", dim=768)
# On message receive:
relevant = db.hybrid_search(embed(user_message), user_message, k=5)
context  = [r.metadata.content for r in relevant]
# Inject context into prompt, then store new turn:
meta = feather_db.Metadata()
meta.content = user_message
meta.type    = feather_db.ContextType.CONVERSATION
db.add(id=next_id(), vec=embed(user_message), meta=meta)
```

### 4.2 Enterprise Knowledge Base

Index documentation, FAQs, runbooks, and Slack threads. Use namespace isolation per department. Link related articles via `db.link()`. Use `context_chain()` to retrieve not just the most similar article but its entire dependency tree.

### 4.3 Personalized Recommendation Engine

Store user interactions as vectors with `entity_id = user_id`. Filter searches by `namespace_id = tenant` and `entity_id = user`. Use adaptive decay to naturally de-prioritise stale preferences without manual cleanup.

### 4.4 Agent Memory for Multi-Step Tasks

AI agents executing long-horizon tasks need to recall prior decisions, tool outputs, and intermediate results. Feather DB's graph layer lets agents link cause-to-effect chains and traverse them during planning.

### 4.5 Multi-Tenant SaaS Platform

The namespace/entity/attributes system maps directly to SaaS tenancy. Each customer gets an isolated namespace. Queries never cross namespace boundaries. The FastAPI wrapper (`feather-api`) provides a ready-to-deploy REST layer with per-namespace authentication.

---

## 5. Integration

### Python (pip)
```bash
pip install feather-db
```
```python
import feather_db
db = feather_db.DB.open("context.feather", dim=768)
```

### REST API (Docker)
```bash
docker-compose -f feather-api/docker-compose.yml up
# POST /v1/{namespace}/vectors
# POST /v1/{namespace}/hybrid_search
```

### Rust CLI
```bash
cargo install feather-db-cli
feather add --db my.feather --id 1 --vec "0.1,0.2,0.3"
feather search --db my.feather --vec "0.1,0.2,0.3" --k 5
```

---

## 6. Roadmap

| Phase | Status | Features |
|-------|--------|---------|
| 1–3 | Done | HNSW, metadata, decay, multimodal, graph |
| 4 | Done | Namespace/entity/attributes, FastAPI wrapper |
| 5 | Done | Typed edges, context_chain, D3 visualizer |
| 6 | Done | BM25, hybrid search, thread safety, WAL, compact |
| 7 | Planned | SIMD acceleration, parallel ingestion, quantisation |
| 8 | Planned | Distributed replication, Raft consensus |
| 9 | Planned | Native LLM integrations (OpenAI, Anthropic, Gemini) |

---

## 7. Conclusion

Feather DB is not another vector database. It is a **living context engine** — a system where memory is dynamic, relational, and adaptive. By combining HNSW ANN search, BM25 keyword retrieval, Reciprocal Rank Fusion, a typed knowledge graph, and adaptive temporal decay in a single embedded library with zero operational overhead, Feather DB provides everything an AI application needs to build persistent, intelligent memory.

The 0.10ms search latency, 15,000+ queries/sec concurrency, 100% WAL crash recovery, and sub-4MB file size for 3,000 records demonstrate that this is not a research prototype — it is production-ready infrastructure.

**Feather DB is available now:**
- PyPI: `pip install feather-db`
- Crates.io: `cargo install feather-db-cli`
- GitHub: github.com/feather-store/feather
- Demo: huggingface.co/spaces/Sri-Vigneshwar-DJ/feather-db

---

*© 2026 Hawky.ai. MIT License. Contributions welcome.*
