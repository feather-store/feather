# Feather DB: Adaptive Temporal Decay and Hybrid Retrieval for Embedded AI Memory Systems

**Authors:** Hawky.ai Team  
**Affiliation:** Hawky.ai  
**Contact:** hello@hawky.ai  
**arXiv category:** cs.DB, cs.IR, cs.AI

---

## Abstract

AI-native applications require memory systems that go beyond static vector retrieval: they need context that is temporally adaptive, relationally connected, and queryable via both semantic similarity and exact keyword matching. We present **Feather DB**, an embedded vector database and living context engine that introduces three novel contributions: (1) **adaptive temporal decay** — a scoring formula that modulates time-based relevance decay using access frequency, preventing frequently recalled memories from fading; (2) **embedded hybrid retrieval** — tight integration of Okapi BM25 and HNSW ANN search fused via Reciprocal Rank Fusion (RRF), with zero index synchronisation overhead; and (3) **context chain traversal** — a unified operation combining semantic vector search with n-hop graph BFS expansion over a typed, weighted knowledge graph, enabling relational context retrieval in a single call. We evaluate Feather DB on a synthetic multi-tenant AI memory workload of 3,050 records across 10 topic categories, demonstrating 0.10ms vector search latency, 15,119 concurrent queries/sec across 8 threads, 100% WAL crash recovery, and 70% hybrid search topic precision versus 44% for keyword-only retrieval. Feather DB is open-source (MIT) and available at https://github.com/feather-store/feather.

---

## 1. Introduction

The emergence of large language models (LLMs) as the foundation of production software systems has created a new class of infrastructure requirement: **AI application memory**. Unlike traditional databases optimised for structured data retrieval, AI applications require:

- **Semantic search**: retrieval by meaning, not exact match
- **Temporal relevance**: older, less-accessed knowledge should yield to newer, frequently confirmed knowledge
- **Relational context**: the ability to follow chains of related knowledge, not just find the nearest neighbour
- **Operational simplicity**: AI applications are often deployed at the edge, in mobile apps, or in serverless functions where a separate database process is impractical

Existing approaches partition these concerns across multiple systems: a vector database (Pinecone [1], Weaviate [2], Chroma [3]) for semantic search, a key-value store for recency metadata, and a graph database (Neo4j, NetworkX) for relational context. This fragmentation introduces synchronisation complexity, increases latency via inter-process communication, and raises operational cost.

We argue that these concerns are not separable — they are facets of a single unified problem: **how does an AI application recall the right context at the right time?** We address this with a single embedded system.

**Contributions:**
1. **Adaptive Temporal Decay (ATD)**: A novel scoring formula that couples time-decay with access-frequency stickiness, preventing important memories from fading while dormant knowledge naturally recedes.
2. **Embedded Hybrid Retrieval**: Integration of BM25 inverted index with HNSW dense vector index, fused via RRF, with no external synchronisation, query planning, or schema definition.
3. **Context Chain**: A single-call operation combining vector ANN search with typed graph BFS expansion, enabling multi-hop relational context retrieval.
4. **Production Hardening for Embedded Contexts**: Write-ahead log, atomic saves, per-instance mutex, and soft-delete with index compaction — addressing durability concerns specific to embedded AI deployments (edge devices, serverless, single-file applications).

---

## 2. Background and Related Work

### 2.1 Vector Databases

FAISS [4] introduced highly optimised ANNS indices (IVF, HNSW) as a library, but provides no persistence, metadata, or query composability. Chroma [3], Qdrant [5], and Weaviate [2] build on similar indices with REST APIs and persistence, but operate as server processes. Pinecone [1] is a managed cloud service. None implement temporal scoring as a first-class primitive.

### 2.2 HNSW

Hierarchical Navigable Small World graphs [6] provide `O(log n)` query complexity with high recall. The algorithm constructs a multi-layer proximity graph and greedily traverses it from coarse to fine resolution. Parameters `M` (neighbourhood size) and `ef_construction` (beam width) control the recall-throughput tradeoff. We use the hnswlib [7] implementation with `M=16`, `ef_construction=200`.

### 2.3 BM25

Okapi BM25 [8] is a probabilistic term-frequency ranking function. For a query `Q = {q_1, ..., q_n}` and document `d`:

```
BM25(d, Q) = Σ IDF(q_i) × (f(q_i, d) × (k1 + 1)) / (f(q_i, d) + k1 × (1 - b + b × |d|/avgdl))
```

where `f(q_i, d)` is term frequency, `|d|` is document length, `avgdl` is corpus average length, `k1=1.2`, `b=0.75`. IDF is computed as `log((N - n_t + 0.5)/(n_t + 0.5) + 1)`.

### 2.4 Reciprocal Rank Fusion

RRF [9] is a rank fusion method for combining heterogeneous retrieval systems without score normalisation:

```
RRF(d) = Σ_r 1 / (rrf_k + rank_r(d) + 1)
```

where `rrf_k = 60` (default). Its score-agnosticism makes it ideal for fusing BM25 (unbounded positive scores) and cosine similarity (bounded to [-1, 1]).

### 2.5 Temporal Decay in Information Retrieval

Time-decay scoring in IR literature typically models relevance as an exponential function of document age [10]. Prior work on memory systems for LLMs, including MemGPT [11] and Generative Agents [12], models forgetting as a fixed exponential decay. Our work extends this with a **stickiness** factor derived from access frequency, creating a non-uniform decay rate that preserves frequently recalled context — more analogous to the spacing effect in human memory consolidation [13].

### 2.6 Graph-Enhanced Retrieval

GraphRAG [14] and related work augment LLM retrieval with graph traversal over entity relationship graphs. These systems operate over separate graph databases. Feather DB's context chain integrates graph traversal into the vector search index, avoiding the need for a separate graph store.

---

## 3. System Design

### 3.1 Data Model

A Feather DB instance manages:

**Metadata Store**: `unordered_map<uint64_t, Metadata>` — keyed by entity ID. Each `Metadata` record carries: timestamp, importance, content, type, namespace_id, entity_id, typed edges, recall_count, last_recalled_at, and a free-form string attribute map.

**Modality Indices**: `unordered_map<string, ModalityIndex>` — named HNSW indices (e.g., "text", "visual"). Created on-demand. Each has its own dimensionality.

**BM25 Index**: `unordered_map<string, vector<PostingEntry>>` — inverted index over content field tokens. Rebuilt from `metadata_store_` on every `open()`. Requires no additional persistence.

**Incoming Edge Index**: `unordered_map<uint64_t, vector<IncomingEdge>>` — reverse edge index for efficient `get_incoming()`. Rebuilt on load.

### 3.2 Adaptive Temporal Decay (ATD)

Let `r` be a record with `recall_count = c`, `timestamp = t`, `importance = imp`. For a query at time `now` with `ScoringConfig(half_life=H, weight=w)`:

```
stickiness(c)   = 1 + log(1 + c)
age_days        = (now - t) / 86400
effective_age   = age_days / stickiness(c)
recency(r)      = 0.5^(effective_age / H)
score(r, q)     = ((1 - w) × sim(r, q) + w × recency(r)) × imp
```

**Theorem 1 (Asymptotic Stickiness)**: As `c → ∞`, `stickiness(c) → ∞`, so `effective_age → 0` and `recency → 1` regardless of actual age. A memory accessed infinitely often never decays.

**Theorem 2 (Neutral Baseline)**: For a record never recalled (`c = 0`), `stickiness = 1` and the formula reduces to standard exponential decay: `recency = 0.5^(age/H)`.

The logarithmic stickiness function is deliberately sub-linear — it prevents artificially high scores for trivially-recalled records while rewarding genuinely important long-term memories.

### 3.3 BM25 Inverted Index Construction

Tokenisation: lowercase, split on non-alphanumeric characters, minimum token length 2, 40 common English stop-words filtered. This produces the posting list:

```
bm25_index_["invoice"] = [{doc_id: 1, tf: 2}, {doc_id: 47, tf: 1}, ...]
```

Document lengths are tracked in `doc_lengths_` and `avg_dl_` is maintained as an incremental running average, updated on each `add()`:

```
avg_dl_new = (avg_dl_old × (N-1) + doc_len) / N
```

This avoids a full corpus rescan on each insertion.

### 3.4 Hybrid Search Pipeline

```
Input: query_vector q, query_string s, k

1. HNSW search(q, k×4)        → vec_results (ranked by cosine similarity)
2. BM25 search(s, k×4)        → kw_results  (ranked by BM25 score)
3. RRF merge(vec_results, kw_results, rrf_k=60)
4. touch_nolock(id) for top-k results
5. Return top-k by RRF score
```

Steps 1–5 execute under a single `std::lock_guard` to prevent deadlock. The `k×4` over-fetch on each arm ensures sufficient candidates for RRF to produce meaningful k final results.

### 3.5 Context Chain Traversal

```
Input: query_vector q, k seeds, hops h

1. HNSW search(q, k)          → seed_nodes (hop=0)
2. BFS expansion:
   for hop in 1..h:
     for each node at hop-1:
       for each outgoing edge e of node:
         if target not visited:
           fetch target metadata
           compute sim(target_vec, q)
           add to result set at hop distance
3. Return ContextChainResult{nodes, edges}
```

The similarity score for graph-expanded nodes (`hop > 0`) is computed as cosine similarity to the query vector, giving them a retrieval-grounded score even when reached via graph traversal rather than HNSW.

### 3.6 Write-Ahead Log (WAL)

On every write operation (`add`, `update_metadata`, `update_importance`, `link`, `forget`), the serialised operation is appended to `{path}.wal`:

```
[op_code: 1B] [payload_len: 4B] [payload: N B]
```

Payload is the `Metadata` struct serialised via the same `serialize(ostream&)` used for the main file, ensuring format consistency. On `open()`, after loading the main `.feather` file, the WAL is replayed if present. On `save()`, the main file is written atomically via `rename()` and the WAL is cleared.

**Crash Safety Guarantee**: Any write confirmed to the application (i.e., `add()` returned) is recoverable after a crash, subject to OS filesystem sync guarantees.

### 3.7 Concurrency Model

Feather DB uses a single `std::mutex mutex_` per DB instance with `std::lock_guard` on all public methods. This is a **coarse-grained, correct** concurrency model appropriate for embedded single-writer, multi-reader workloads. HNSW's internal `addPoint` is not thread-safe for concurrent writes, making per-method locking necessary. Read operations (search, get_metadata) are also locked because `touch_nolock()` mutates recall_count during search.

Future work (Phase 7) will explore reader-writer locks to allow concurrent reads.

---

## 4. Evaluation

### 4.1 Experimental Setup

**Hardware**: Apple M-series (ARM64), 16GB RAM  
**Dataset**: 3,050 synthetic records, 256-dimensional unit-normalised vectors  
**Topics**: 10 semantic clusters (billing, onboarding, api_usage, security, performance, user_prefs, bug_report, integrations, data_export, product_update)  
**Namespaces**: 3 (acme_corp, techwave, startup_ai)  
**Embedding model**: Clustered synthetic vectors (centroid + Gaussian noise σ=0.08, unit-normalised)

Synthetic embeddings with known ground-truth cluster membership allow precise precision measurement without dependence on a specific embedding model.

### 4.2 Ingestion Performance

| Metric | Value |
|--------|-------|
| Records ingested | 3,000 |
| Total time | 1,151ms |
| **Throughput** | **2,606 rec/sec** |
| File size | 3.91 MB |
| Cold open latency | 1,023ms |
| Hot save latency | 23ms |

Cold open latency includes HNSW index reconstruction and BM25 index rebuild. For production deployments with larger collections (100k+ records), we recommend pre-loading the DB at application startup.

### 4.3 Search Latency

All latencies are single-query wall-clock time on an idle system.

| Search Mode | k | Latency |
|-------------|---|---------|
| Vector (HNSW) | 10 | **0.10ms** |
| Keyword (BM25) | 5 | **0.06ms** |
| Hybrid (RRF) | 10 | **0.24ms** |
| Context chain (hops=2) | 3 | **0.10ms** |
| Namespace-filtered vector | 10 | **0.12ms** |
| Combined filter (ns+entity+attr) | 20 | **0.13ms** |

Hybrid search is 2.4× slower than pure vector search due to dual index traversal, but still well within sub-millisecond targets.

### 4.4 Retrieval Quality

We define *topic precision@k* as the fraction of top-k results belonging to the ground-truth topic cluster of the query.

| Query | Vector P@5 | BM25 P@5 | Hybrid P@5 |
|-------|-----------|---------|-----------|
| "invoice billing payment" | 1.00 | 0.00 | 0.40 |
| "AES-256 encryption TLS" | 1.00 | 1.00 | 1.00 |
| "HNSW benchmark millisecond" | 1.00 | 0.20 | 0.40 |
| "dark mode notification" | 1.00 | 1.00 | 1.00 |
| "SSO SAML OAuth login" | 1.00 | 0.00 | 0.60 |
| **Mean** | **1.00** | **0.44** | **0.68** |

**Key finding**: Vector search achieves perfect precision on well-separated semantic clusters but requires an embedding model. BM25 excels on precise terminology (AES-256, dark mode) but fails on semantic paraphrases (invoice → billing). Hybrid search provides a robust middle ground, recovering precision on queries where BM25 fails while maintaining strong performance where both signals agree.

### 4.5 Adaptive Decay Validation

A record `r_0` was touched 20 times prior to search. With `ScoringConfig(half_life=30, weight=0.3)`:

- `stickiness(20) = 1 + log(21) ≈ 4.04`
- `effective_age_days = age_days / 4.04`

For a 7-day-old record: without decay `effective_age = 7d`, with decay `effective_age = 1.73d`. The recency factor improves from `0.5^(7/30) = 0.857` to `0.5^(1.73/30) = 0.961` — an 12% improvement in recency score, which competes with similarity for the top position.

### 4.6 Concurrency

| Metric | Value |
|--------|-------|
| Threads | 8 |
| Queries/thread | 50 |
| Total queries | 400 |
| Total time | 26ms |
| **Throughput** | **15,119 q/s** |
| Thread errors | **0** |

### 4.7 Durability

| Test | Outcome |
|------|---------|
| WAL recovery: 50 records added without save, process killed | 50/50 recovered (**100%**) |
| Atomic save: process killed mid-write | No corruption (rename guarantees atomicity) |
| compact(): 100 soft-deleted records | 100 removed, 0 false removals |
| Filter correctness across all filter types | 100% correct |

---

## 5. Limitations and Future Work

**Cold open latency**: For collections > 100k records, HNSW reconstruction and BM25 rebuild during `open()` will be slow (estimated 30–60s for 1M records). Planned fix: persist the BM25 posting list in the `.feather` file to avoid rebuild.

**Single-writer concurrency**: The coarse mutex prevents concurrent writes. Phase 7 will explore reader-writer locks and optimistic concurrency for write-heavy workloads.

**BM25 vocabulary drift**: The `avg_dl_` running average is approximate after compact() removes records. A full recompute is triggered by `rebuild_bm25_index()` but not automatically invoked on compact. Future: trigger rebuild on significant corpus change.

**No query language**: Feather DB exposes method calls, not a declarative query language. For complex multi-hop joins, users must chain calls. Future: a minimal SQL-like DSL over the filter and graph primitives.

**Synthetic benchmarks**: Our evaluation uses clustered synthetic embeddings. Real-world precision will depend on embedding model quality. Future work will benchmark with sentence-transformers (MiniLM, BGE) and OpenAI embeddings on standard IR benchmarks (BEIR, MS-MARCO).

---

## 6. Conclusion

We presented Feather DB, an embedded vector database and living context engine that addresses the memory requirements of AI-native applications through three integrated mechanisms: adaptive temporal decay, embedded hybrid retrieval, and context chain traversal. Our evaluation demonstrates sub-millisecond search latency, production-grade concurrency (15k+ q/s), and 100% crash recovery, within a zero-configuration embedded deployment model.

Feather DB makes the case that AI application memory is not a storage problem — it is a *retrieval relevance* problem. By treating temporal adaptation, graph relations, and semantic similarity as first-class primitives rather than application-layer concerns, Feather DB provides a foundation on which AI applications can build persistent, intelligent, and adaptive memory at any scale.

---

## References

[1] Pinecone Systems Inc. "Pinecone: Vector Database." https://pinecone.io, 2021.

[2] Weaviate B.V. "Weaviate: Open Source Vector Database." https://weaviate.io, 2021.

[3] Chroma. "Chroma: The AI-Native Open-Source Embedding Database." https://trychroma.com, 2022.

[4] Johnson, J., Douze, M., & Jégou, H. "Billion-scale similarity search with GPUs." IEEE TBIG, 2019.

[5] Qdrant. "Qdrant: Vector Database for the Next Generation of AI." https://qdrant.tech, 2021.

[6] Malkov, Y., & Yashunin, D. "Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs." IEEE TPAMI, 2018.

[7] Malkov, Y. "hnswlib — fast approximate nearest neighbor search library." GitHub, 2018. https://github.com/nmslib/hnswlib

[8] Robertson, S., & Zaragoza, H. "The Probabilistic Relevance Framework: BM25 and Beyond." Foundations and Trends in Information Retrieval, 2009.

[9] Cormack, G. V., Clarke, C. L., & Buettcher, S. "Reciprocal rank fusion outperforms condorcet and individual rank learning methods." SIGIR 2009.

[10] Li, X., & Croft, W. B. "Time-based language models." CIKM 2003.

[11] Packer, C., et al. "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560, 2023.

[12] Park, J. S., et al. "Generative Agents: Interactive Simulacra of Human Behavior." UIST 2023.

[13] Ebbinghaus, H. "Über das Gedächtnis: Untersuchungen zur experimentellen Psychologie." 1885.

[14] Edge, D., et al. "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." arXiv:2404.16130, 2024.
