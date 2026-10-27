# How Feather DB Works — End to End

> A walkthrough of the whole system: what happens to a piece of data from the moment it enters Feather DB until it is retrieved, persisted, recovered after a crash, and served to an LLM agent.
>
> Written against the code at **v0.18.2**. Where older docs (`CLAUDE.md`, `docs/architecture/*`) disagree with the code, this document follows the code. File references are `path:line`.

---

## Table of contents

1. [The mental model in one minute](#1-the-mental-model-in-one-minute)
2. [The layers](#2-the-layers)
3. [Repository map](#3-repository-map)
4. [The data model](#4-the-data-model)
5. [The core engine (C++)](#5-the-core-engine-c)
   - 5.1 Opening a DB
   - 5.2 The write path: `add` / `add_batch`
   - 5.3 The read paths: vector, filtered, BM25, hybrid, context chain
   - 5.4 Scoring and adaptive decay ("living context")
   - 5.5 The context graph
   - 5.6 Deleting: forget / purge / expire / compact
   - 5.7 Persistence: `save()`, the `.feather` file, the WAL
   - 5.8 Concurrency
   - 5.9 Performance knobs
6. [Bindings: how Python and Rust reach the engine](#6-bindings-how-python-and-rust-reach-the-engine)
7. [The Python layer (`feather_db/`)](#7-the-python-layer-feather_db)
8. [Agent integrations and the MCP server](#8-agent-integrations-and-the-mcp-server)
9. [The Cloud API (`feather-api/`)](#9-the-cloud-api-feather-api)
10. [End-to-end traces](#10-end-to-end-traces)
11. [Build, packaging, CI, tests, benchmarks](#11-build-packaging-ci-tests-benchmarks)
12. [Environment variables](#12-environment-variables)
13. [Known gaps and gotchas](#13-known-gaps-and-gotchas)
14. [Where to change what](#14-where-to-change-what)

---

## 1. The mental model in one minute

Feather DB is an **embedded** database, like SQLite. It is a library that you load into your own process, not a server you connect to. A database is **one file** (`my.feather`) plus a write-ahead log beside it (`my.feather.wal`).

Each **record** has:

- a `uint64` **id**
- one or more **vectors**, at most one per *modality* (`"text"`, `"visual"`, `"audio"`, …), each held in its own HNSW index
- one **Metadata** object shared by all of that record's vectors: content, timestamps, importance, namespace/entity/attributes, recall counters, and **typed weighted edges** to other records

A record can be retrieved in four ways:

| Method | Mechanism |
|---|---|
| Vector similarity | HNSW approximate nearest neighbour, or an exact scan when a selective filter applies |
| Keywords | BM25 inverted index over `content` |
| Hybrid | Vector and BM25 results fused with Reciprocal Rank Fusion |
| Graph | Vector search to find seeds, then a BFS over edges (`context_chain`) |

Every retrieval increments the hit's `recall_count`. The scorer uses that count to slow the record's time decay, so memories that get used stay fresh and unused ones fade. This is the **"living context"** idea.

On top of the engine sits a Python layer. It can turn raw text into facts, entities and relationships using LLMs, and it exposes all of this as tools to agents such as Claude, over MCP or through function calling. A FastAPI service wraps the same engine as a multi-namespace REST server.

---

## 2. The layers

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Consumers                                                                 │
│  Claude Desktop/Code (MCP) · your agent · LangChain · LlamaIndex · curl    │
└──────────────┬───────────────────────────────┬─────────────────────────────┘
               │                               │ HTTP (X-API-Key)
┌──────────────▼──────────────┐   ┌────────────▼───────────────────────────┐
│ feather_db.integrations     │   │ feather-api/  (FastAPI, 1 worker)      │
│  mcp_server  (feather-serve)│──►│  one .feather file per namespace       │
│  mcp_remote  (REST client)  │   │  embedding service, admin SPA, metrics │
│  FeatherTools / connectors  │   └────────────┬───────────────────────────┘
└──────────────┬──────────────┘                │
┌──────────────▼───────────────────────────────▼─────────────────────────────┐
│ feather_db (pure Python)                                                   │
│  IngestPipeline → extractors (facts, entities, temporal, ontology,         │
│  contradictions) · reason (planner/executor) · ContextEngine · providers   │
│  MemoryManager · Episodes · Triggers · merge · feedback · graph viz        │
└──────────────┬─────────────────────────────────────────────────────────────┘
               │ pybind11  (feather_db.core, GIL released on hot paths)
┌──────────────▼─────────────────────────────────────────────────────────────┐
│ C++17 engine — include/feather.h  (class feather::DB)                      │
│  HNSW per modality · metadata store · reverse edge index                   │
│  secondary indexes (ns/entity/attr) · BM25 index · WAL · .feather v9       │
└──────────────▲─────────────────────────────────────────────────────────────┘
               │ extern "C" ABI (src/feather_core.cpp)
        ┌──────┴───────┐
        │ Rust CLI     │  feather-cli/ (vendored copy of include/ + src/)
        └──────────────┘
```

---

## 3. Repository map

```
feather/
├── include/                 C++ engine — nearly all logic is header-only
│   ├── feather.h            ★ class feather::DB — THE database (≈2.2k lines)
│   ├── metadata.h           Metadata, Edge, ContextType
│   ├── scoring.h            ScoringConfig + Scorer (adaptive decay formula)
│   ├── filter.h             SearchFilter + matches()
│   ├── hnswalg.h/hnswlib.h  HNSW (hnswlib fork: resize, save/load stream, filters)
│   ├── space_l2.h           L2 distance (SSE/AVX runtime-dispatched) + Int8L2Space
│   ├── space_ip.h, bruteforce.h, visited_list_pool.h, stop_condition.h
├── src/
│   ├── metadata.cpp         Metadata::serialize / deserialize (the on-disk record)
│   ├── feather_core.cpp     extern "C" API for the Rust CLI
│   └── filter.cpp, scoring.cpp  (thin)
├── bindings/feather.cpp     pybind11 module → feather_db.core
├── feather_db/              Python package (see §7, §8)
├── feather-api/             FastAPI Cloud server + admin SPA (see §9)
├── feather-cli/             Rust CLI crate; cpp/ is a synced copy of include/ + src/
├── tests/                   pytest suite
├── test_*.py (root)         feature scripts (auto-compact, int8, persist-graph, …)
├── bench/, benchmarks/      ANN + LongMemEval benchmarking
├── examples/                runnable demos
├── hf-space/                old Gradio demo (pinned to 0.6.1)
├── scripts/                 sync-cpp.sh, verify.sh (pre-release checklist)
└── setup.py / pyproject.toml
```

---

## 4. The data model

### 4.1 `Metadata` (`include/metadata.h`)

| Field | Type | Purpose |
|---|---|---|
| `timestamp` | int64 | Creation or "valid at" time, in Unix seconds. Drives recency decay. |
| `importance` | float, default 1.0 | Multiplies the final score. Set to 0 on forget. |
| `confidence` | float, default 1.0 | Epistemic certainty. Stored but not used by the scorer. |
| `type` | `ContextType` | `FACT=0`, `PREFERENCE=1`, `EVENT=2`, `CONVERSATION=3` |
| `source` | string | Where the record came from. `"_forgotten"` marks a soft-deleted record. |
| `content` | string | Human-readable text. This is what BM25 indexes. |
| `tags_json` | string | A raw JSON string. The tag filter does a substring match on it. |
| `namespace_id` | string | Partition or tenant key. Indexed. |
| `entity_id` | string | Subject key. Indexed. |
| `attributes` | map<str,str> | Domain key/value pairs. Each pair is indexed. `_deleted=true` marks a dead record. |
| `edges` | vector<Edge> | Outgoing `{target_id, rel_type, weight}` |
| `recall_count`, `last_recalled_at` | `mutable std::atomic` | Salience counters, updated on every retrieval hit |
| `ttl` | int64 | Seconds to live, counted from `timestamp`. 0 means the record never expires. |

`Metadata` is copyable even though it holds atomics: `assign_from` copies the counter values (`metadata.h:81-110`).

> ⚠️ **Python gotcha:** `meta.attributes["k"] = v` silently does nothing, because pybind11 hands you a copy of the map. Use `meta.set_attribute(k, v)` and `meta.get_attribute(k, default)` instead. Likewise, `db.get_metadata(id)` returns a copy: after editing it, call `db.update_metadata(id, meta)`.

### 4.2 Ids

Ids are plain `uint64`, and the engine never generates them. Each layer uses its own convention:

| Layer | Id scheme |
|---|---|
| `IngestPipeline` | `1e9+n` for source records, `2e9+n` for facts, `3e9+n` for entities (per-instance counter) |
| `ContextEngine`, `MemoryManager.consolidate`, episodes | `sha256(...) % 2**50` |
| `FeatherTools.feather_add_intel` | Sequential from 90001, per process |
| Cloud API | Random in `[1, 2^53)`, so JavaScript can represent it |
| `mcp_remote` | 53-bit `sha1(text|time_ns)` |

`add()` with an existing id is an **upsert**. It replaces the metadata but keeps the old edges if the new metadata has none.

### 4.3 Multimodal "pockets"

Each modality name gets its own HNSW index with its own dimension, created lazily on the first `add` to that modality. The same id can hold a 768-d text vector and a 512-d visual vector while sharing a single `Metadata`. Search always targets one modality at a time.

---

## 5. The core engine (C++)

The whole database is `class feather::DB` in `include/feather.h`. Its private state:

```cpp
unordered_map<string, ModalityIndex> modality_indices_;   // HNSW per modality
unordered_map<uint64_t, Metadata>    metadata_store_;     // THE source of truth
unordered_map<uint64_t, vector<IncomingEdge>> reverse_index_;   // derived
ns_index_, entity_index_, attr_index_                     // derived (live records only)
bm25_index_ (term → postings), doc_lengths_, avg_dl_      // derived
std::shared_mutex mutex_;  std::mutex save_mutex_;        // concurrency
FILE* wal_file_;                                          // write-ahead log
```

Only two things are persisted: `metadata_store_` and the vectors (or HNSW graphs). The reverse edge index, secondary indexes and BM25 index are **derived**. They are rebuilt on load and then maintained incrementally on every mutation.

### 5.1 Opening a DB — `DB::open(path, default_dim=768)`

```
open(path)
 ├─ path_ = path, wal_path_ = path + ".wal"
 ├─ acquire_file_lock()             exclusive OS lock on <path>.lock (flock / LockFileEx)
 │                                  → a second open() anywhere fails here, before any read
 └─ load_vectors()
     ├─ read_base_file()            parse .feather (if it exists)
     │   ├─ check magic "FEAT", read version (2 … 9 supported)
     │   ├─ metadata section → metadata_store_
     │   │     (meta_count bounded by remaining bytes → corrupt file = error, not hang)
     │   └─ for each modality:
     │         read name, dim (guard: 0 < dim ≤ 2^20), flags
     │         if persist_graph (v9):  loadIndexStream()  ← no rebuild, fast
     │         else: read all vectors, dequantize if on-disk int8,
     │               parallel_add() to rebuild the HNSW graph on a thread pool
     ├─ replay_wal(<wal>.old)       only exists if a checkpoint was interrupted
     ├─ replay_wal(<wal>)           apply any mutations logged since the last save
     ├─ build_reverse_index()
     ├─ build_secondary_indexes()
     ├─ rebuild_bm25_index()
     └─ load_complete_ = true       ← MUST be the last statement
```

Key properties:

- **The WAL is replayed even when no base file exists.** A namespace that has never been saved can hold all of its data in the WAL.
- **Nothing is written back after a failed load.** If loading throws, `load_complete_` stays false, so the destructor skips its checkpoint and `write_snapshot()` refuses to run. The alternative would be writing a half-parsed fragment over the original file and deleting the WAL that could rebuild it. See `~DB()` and `tests/test_open_failure_safety.py`.
- **`open()` creates no index.** The `"text"` index is created on the first `add`. That lets `set_int8_ram()` take effect beforehand, and avoids allocating RAM for an empty index. `dim()` reports `default_dim_` until an index exists.
- **One writer process per file (0.20).** A second *process* opening the same path for writing is refused with the holder's pid; `DB.open(p, read_only=True)` takes a shared lock so many reader processes coexist. The lock is reentrant within one process, so a second `open()` in the same process is admitted (and remains a footgun: use one handle). Release a handle with `db.close()` or a `with DB.open(...) as db:` block. Because Python never destroys a `DB` (`py::nodelete`), dropping the reference does *not* release it. The OS drops the lock if the process dies. Set `FEATHER_LOCK=0` to disable it, e.g. on NFS where `flock` is unreliable.

### 5.2 The write path — `add()`

```
db.add(id, vec, meta, modality="text")
 0. encode the WAL payload [modality, dim, vector, serialized meta]   (no lock held)
 1. take mutex_ EXCLUSIVELY
 2. dimension check against the modality's index → throw BEFORE anything is logged
 3. WAL: append ADD record + CRC32, fflush → returns this record's LSN
 4. get_or_create_index(modality, dim)
       new index: L2Space (or Int8L2Space), capacity 4096, M=16, ef_construction=200,
       ef=50, allow_replace_deleted=true
 5. reserve(): if full, resizeIndex(max(needed, 2×capacity))
 6. add_point(): (quantize to int8 if in-RAM int8) → hnsw.addPoint(vec, id, reuse_slot)
       a NEW id takes over the slot of a vector forget()/purge() deleted, so
       tombstones don't accumulate; an existing id is updated in place
 7. metadata upsert:
       existing id → deindex old ns/entity/attrs, remember old content,
                     keep old edges if the new meta has none (else re-point the
                     reverse edge index at the new edges)
 8. index_meta()  → ns_index_ / entity_index_ / attr_index_ (unless dead)
 9. add_to_bm25_index(content) → retire old postings, tokenize, add postings
                                  (dead records are not indexed)
10. release mutex_                  ← readers resume here
11. wal_wait_durable(LSN)           group commit: one leader fsyncs everything appended so
                                    far; add() returns only once its record is on disk
```

**Group commit (0.19).** The fsync used to happen at step 3, inside the exclusive lock. Every search then waited for a disk flush on every write (measured: −68% reads with one writer), and writers queued for one fsync each (throughput stayed flat at ~255 writes/s whatever the thread count).

Now the append happens inside the lock, so WAL order still equals apply order. The fsync happens after the lock is released, and one fsync releases every writer whose record it covers. **Trade-off:** a concurrent reader can see a record a few milliseconds before it is durable. The writer is still only acknowledged after the fsync. `FEATHER_WAL_STRICT=1` restores "never visible before durable": concurrent writers still share fsyncs, but the fsync happens inside the lock again.

**`add_batch(ids, vecs, metas, modality)`** has the same per-item semantics, with three differences:
- It **fsyncs the WAL once for the whole batch**, not once per record.
- It builds the HNSW graph in **parallel**: a thread pool claims items through an atomic counter. This is roughly 3.4× faster than calling `add()` in a loop.
- **It works in chunks** (`FEATHER_BATCH_CHUNK`, default `max(64, 2 × cores)`) and releases the exclusive lock between them. Previously a 1,000-vector batch held the lock through its whole graph build, which cut reads by 99%. Readers can now observe a partially applied batch. The call still returns only once the whole batch is durable.

`reserve()` runs *before* the parallel build, because `resizeIndex` is not thread-safe. The Python binding releases the GIL around the whole call.

**Tokenizer (for BM25):** splits on non-alphanumeric characters, lowercases, drops tokens shorter than 2 characters, and drops ~60 English stop words (`feather.h:440-466`).

### 5.3 The read paths

All reads take `mutex_` in **shared** mode, so any number of them run concurrently.

#### a) `search(q, k, filter, scoring, modality, record_salience=true, with_cosine=false)`

The first step is to decide whether the filter touches an **indexed field**: `namespace_id`, `entity_id` or an attribute.

```
              filter constrains ns / entity / attribute?
                 │ yes                              │ no (or no filter)
                 ▼                                  ▼
  PRE-FILTERED PATH                        HNSW PATH
  candidates = intersect(secondary         FilterWrapper → hnsw.searchKnn(
     index sets, smallest first)              q, k (or 3k if scoring), filter)
  cand × sel² > 10·k and                   for each hit:
  ef ≈ 2k/selectivity ≤ 4096?                 score = scoring ? Scorer(...)
     yes → filtered HNSW with that ef                  : 1/(1+dist)
           (exact scan if it comes up short)
     no  → exact scan: SIMD distance on the
           stored vector (no copy), skip dead /
           non-matching predicates
  partial-sort (score, id), copy metadata  partial-sort (score, id), copy metadata
  for the top k only                       for the top k only
                 └──────────────┬───────────────┘
                                ▼
            if record_salience: touch() each RETURNED hit
            (recall_count++, last_recalled_at = now — atomic)
            if with_cosine: SearchResult.cosine = exact cosine(query, stored vector)
```

Why two paths? HNSW's filtered traversal is limited by `ef`. When a filter is selective, say one namespace out of many, the traversal quietly returns far fewer than `k` results. The exact path returns a **complete** top-k and costs O(matches), so a selective filter is also fast.

For a *large*, unselective candidate set, the scan becomes the slow option, so such sets go through the graph instead. The route is chosen by comparing costs. The scan costs about `candidates`. A filtered walk costs about `k / selectivity²`, because it needs a wider beam *and* more of the nodes it visits fail the filter. Measured at 100k × 768-d:

| Share of records matching the filter | Scan | Walk |
|---|---|---|
| 2% | 0.55 ms | 45 ms |
| 20% | 9.1 ms | 1.1 ms |

The walk is taken when `candidates × sel² > C × k`, with C = 10 calibrated at 128-d and 768-d. `ef` is raised for that one call only (`searchKnnEf`, so concurrent searches are unaffected), and the exact scan is the fallback whenever the walk returns fewer than `k` results.

In every path, only the `k` winners get their `Metadata` copied. Before 0.19 the pre-filtered path copied every candidate's content, attributes and edges before sorting. A query whose dimension doesn't match the index now raises instead of reading past the end of the query buffer.

`dist` is the squared L2 distance from hnswlib's `L2Space`, and the raw similarity is `1/(1+dist)`. Hits are touched only *after* truncation to `k`, so "recalled" means "returned", not "examined". Pass `record_salience=False` for evaluations, monitoring and internal self-queries (§13).

#### b) `keyword_search(query, k, filter)` — BM25 (`feather.h:1766`)

This is standard BM25, with `k1 = 1.2` and `b = 0.75`:

```
idf(t)   = ln( (N − n_t + 0.5) / (n_t + 0.5) + 1 )
score(d) = Σ_t idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·dl/avgdl))
```

Dead records are skipped, and the filter is applied per posting. Hits are touched.

#### c) `hybrid_search(vec, query, k, rrf_k=60, filter, scoring, modality)` (`feather.h:1828`)

1. Run a vector search for `3k` candidates, or `9k` when scoring is on.
2. Run a BM25 search for `3k` candidates.
3. Fuse the two lists with **Reciprocal Rank Fusion**: `rrf(id) = Σ_lists 1/(rrf_k + rank + 1)`.
4. Keep the top `k`. The returned `score` is the RRF score, not a similarity.

Hybrid search **does not** touch salience.

Note that the vector half of hybrid search always uses the HNSW-filtered path, not the pre-filtered exact path.

#### d) `context_chain(query, k=5, hops=2, modality)` — GraphRAG (`feather.h:1328`)

```
1. seeds = HNSW top-k   (sim = 1/(1+dist), seeds are touched)
2. BFS up to `hops`, following BOTH outgoing edges (metadata.edges)
   and incoming edges (reverse_index_); record every edge seen
3. score each visited node:
       base       = sim            if hop == 0
                  = 1/(1+hop)      otherwise
       score      = base × importance × (1 + ln(1 + recall_count))
4. dedupe edges, sort nodes by score → {nodes[{id,score,similarity,hop,metadata}], edges[]}
```

### 5.4 Scoring and adaptive decay (`include/scoring.h`)

When a `ScoringConfig(half_life=30, weight=0.3, min=0)` is passed, each hit is scored as follows:

```
similarity      = 1 / (1 + dist)
stickiness      = 1 + ln(1 + recall_count)          # 0 recalls → 1.0, 10 → 3.4, 100 → 5.6
effective_age   = age_days / stickiness             # frequently-used memories age slower
recency         = max(min, 0.5 ^ (effective_age / half_life))
final           = ((1 − weight)·similarity + weight·recency) · importance
```

The loop that closes over this is:

1. A search returns a record.
2. The record's `recall_count` goes up.
3. Its stickiness goes up.
4. It decays more slowly.
5. It keeps ranking higher.

`MemoryManager.why_retrieved()` re-implements this formula in Python so you can explain any single score (`memory.py:43-93`).

> Recall counters are **not** WAL-logged. They reach disk only when `save()` runs. A crash loses salience updates made since the last save, but never records.

### 5.5 The context graph

- **`link(from, to, rel_type="related_to", weight=1.0)`** appends an `Edge` to `from`'s metadata and adds an `IncomingEdge` to `reverse_index_[to]`. It is idempotent per `(from, to, rel_type)`, WAL-logged as `LINK`, and does nothing if `from` does not exist.
- **`get_edges(id)`** returns the outgoing edges. **`get_incoming(id)`** returns the incoming ones, from the reverse index.
- **`auto_link(modality, threshold=0.8, rel_type, candidates=15)`** runs a kNN search for every live vector and links pairs whose `1/(1+dist) ≥ threshold`. Edge weight is the similarity. The kNN pass runs under the **shared** lock, so queries continue meanwhile, and only applying the edges is exclusive. Every edge is WAL-logged as `LINK`, with one fsync for the whole pass.
- **`export_graph_json(ns, entity)`** returns D3/Cytoscape-style `{nodes, edges}` and filters out dangling edges. `feather_db.graph.visualize()` turns that into a self-contained offline HTML force graph, with D3 inlined from `feather_db/d3.min.js`.

Relationship types are free-form strings. These are the ones in common use:

| Defined in | Types |
|---|---|
| `graph.RelType` | related_to, derived_from, caused_by, contradicts, supports, precedes, part_of, references, multimodal_of |
| Phase 9 pipeline | extracted_from, refers_to, supersedes, uses_creative, targets_segment, … |
| Episodes | episode_contains, episode_end |
| Consolidation | consolidated_into |

### 5.6 Deleting: forget / purge / expire / compact

| Operation | What happens | Reversible? |
|---|---|---|
| `forget(id)` | WAL `FORGET`, then `markDelete` in every HNSW. Removes the record from the secondary and BM25 indexes, blanks `content`, sets `source="_forgotten"` and `importance=0`. **The node shell and its edges stay**, so the graph can still be traversed through it. | No |
| `forget_expired()` | Applies `forget` to every record with `ttl>0 && now > timestamp+ttl`. One fsync for the whole sweep. | No |
| `purge(namespace)` | Hard delete. Marks the vectors deleted, **erases** the metadata, removes the ids from every index, and prunes edges pointing at purged ids. WAL-logged as `PURGE` since 0.19. | No |
| `compact()` | Rebuilds every modality index from live survivors only. Drops dead metadata and reclaims marked-deleted and orphaned vectors. It is orphan-safe: purged ids are never resurrected. See below for how it runs. | – |
| `set_auto_compact(ratio)` | After a forget, purge or expire, if any modality's `deleted/total ≥ ratio`, **requests** a compaction on a background thread. The triggering call returns immediately. `wait_for_compaction(timeout)` blocks until it is done. 0 (the default) turns this off. | – |

`save()` never writes dead records, so a forgotten record disappears from the file on the next save, whether or not `compact()` ran.

**How `compact()` avoids freezing the DB (0.19).** It runs in three phases:
1. A short exclusive section snapshots the live vectors (raw storage bytes) and starts a change log.
2. The new indexes are built in parallel with **no lock held**. Readers keep using the old index, and writers keep changing it, recording their vector-level changes in the log.
3. A short exclusive section replays the log onto the new indexes, swaps them in, and erases metadata that is still dead.

Before 0.19 the whole rebuild ran under the exclusive lock: 40 s at 100k×128 and 86 s at 50k×768 with no reads or writes. Transient RAM during a compaction is one copy of the live vectors plus the new index.

**Deleted-slot reuse.** Indexes are created with `allow_replace_deleted`, so `add()` of a *new* id takes over a slot freed by `forget()`/`purge()` rather than growing the index. Tombstones therefore stop piling up, and compaction is needed much less often. Reuse happens only on single-threaded insert paths under the exclusive lock, because the parallel batch build never reuses slots. The vendored hnswlib was patched so that re-adding a forgotten id revives its own node rather than throwing.

### 5.7 Persistence

#### `save()`

```
lock save_mutex_ (one saver at a time)
  mutex_ SHARED (queries keep running; writers wait only for this part)
    write_snapshot(path + ".tmp")  refuse if !load_complete_; throw on any write error
    wal_rotate()                   fsync + close the WAL, rename <wal> → <wal>.old
  release mutex_                   ← writers continue into a fresh <wal>
  fsync(.tmp) → atomic replace(.tmp → path) → fsync(dir)
  delete <wal>.old                 ← its records are now in the durable base file
```

Recovery handles a crash at any step. The old base file plus `<wal>.old` plus `<wal>` replays to the right state, and replay is idempotent, so a crash after the rename but before the delete is also fine. If a previous checkpoint left `<wal>.old` behind, `save()` falls back to a fully synchronous checkpoint and removes both WAL files.

Two bugs in this path were fixed in 0.19:
- The new file used to be **renamed into place without an fsync** and the WAL deleted straight after. A power loss could leave an empty base file and no WAL.
- A write error such as a full disk was never checked. A truncated snapshot could be renamed over the good file, and the WAL then deleted.

**`close(save=True)`** checkpoints, closes the WAL, releases the file lock and makes the handle unusable. `with DB.open(p) as db:` calls it for you. The destructor `~DB()` also checkpoints, but only if the load completed. In Python the `DB` object is bound with `py::nodelete`, so **the destructor never runs from Python**. Use `close()`, and rely on the WAL for crash safety in between.

#### The `.feather` file, format v9

```
[magic "FEAT" 0x46454154 : u32] [version = 9 : u32]

── metadata section ──────────────────────────────────────────────────
[meta_count : u32]                         (live records only)
repeat: [id : u64] [Metadata record — see below]

── modality section ──────────────────────────────────────────────────
[modal_count : u32]
repeat per modality:
  [name_len : u16][name]
  [dim : u32]
  [quantized : u8]                         v7+: on-disk int8
  [int8_ram : u8] ([scale : f32] if set)   v8+: in-RAM int8
  [persist_graph : u8]                     v9+
  if persist_graph:
      [hnswlib index stream: header + level-0 data (vectors) + link lists]
  else:
      [element_count : u32]
      repeat: [id : u64] then
              quantized=0 → [dim × f32]
              quantized=1 → [scale : f32][dim × i8]
```

A **Metadata record** (`src/metadata.cpp`) is laid out like this:

```
timestamp i64 · importance f32 · type u8
source   [u16 len][bytes]
content  [u32 len][bytes]
tags     [u16 len][bytes]
legacy links_count u16 (always 0)
recall_count u32 · last_recalled_at u64
namespace [u16][bytes] · entity [u16][bytes]
attributes [u16 count] × ([u16 klen][k][u32 vlen][v])
edges      [u16 count] × ([u64 target][u8 rlen][rel][f32 weight])
ttl i64 · confidence f32
```

The HNSW graph is only persisted on the fast path, which requires that the modality holds **exactly** the live set (no pending forget or purge) **and** is not on-disk-quantized. Otherwise the vectors are written out and the graph is rebuilt, in parallel, on load. Persisting the graph makes the file about 25% bigger and makes cold loads 5–25× faster. Running `compact()` puts a modality with pending deletes back on the fast path.

Files in formats v2 through v8 load without conversion, since each newer field is read only when `version >=` its introduction.

#### The WAL, format v2

```
header: ["FWAL" u32][version = 2 u32]
record: [op u8][id u64][payload_len u32][payload][crc32 u32]
        crc32 covers op + id + len + payload
ops:    ADD=1 (modality, dim, vector, meta) · UPDATE=2 (meta) · UIMP=3 (importance)
        LINK=4 (to, rel, weight) · FORGET=5 · PURGE=6 (namespace)  ← 0.19
```

A pre-0.19 build replaying a WAL that contains `PURGE` skips that record, because unknown ops fall through replay's if/else chain.

The WAL obeys these rules:

- **Log before mutate, acknowledge after durable.** Every mutation appends its WAL record, inside the exclusive lock, before touching memory. It returns to the caller only after `wal_wait_durable()` has seen that record fsynced. Concurrent writers share fsyncs (group commit, §5.2). Set `FEATHER_WAL_SYNC=0` to skip fsync and gain throughput at the cost of durability. Set `FEATHER_WAL_STRICT=1` to keep the fsync inside the lock.
- **Verify before applying.** Replay stops at the first record that has a bad CRC, a length larger than the bytes remaining, or a payload over 256 MB. Everything before that point is recovered, and a torn tail is discarded cleanly.
- **Backward compatible.** A v1 WAL (≤0.17.0, no header, no CRC) still replays. The `'F'` byte can never be a v1 opcode, so detecting the header is unambiguous.
- The WAL file handle stays open for the life of the DB. `save()` rotates it to `<wal>.old` and deletes that once the base file holds everything.

**Crash recovery, end to end:** the process dies after some `add`s and before `save()`. On the next `open()`, the old base file loads, the WAL replays on top of it, and the derived indexes rebuild over the combined state. No acknowledged write is lost (`tests/test_wal_recovery.py`).

### 5.8 Concurrency

- There is one readers-writer lock per DB, `mutex_`, and it is a **phase-fair `FairSharedMutex`** (0.19).
  - **Shared lock:** search, keyword and hybrid search, `context_chain`, the const accessors, `touch`, the snapshot half of `save`, and the kNN pass of `auto_link`.
  - **Exclusive lock:** anything that changes the metadata store, a derived index or an HNSW graph. That includes anything that can call `resizeIndex`.
- **Why phase-fair?** libstdc++'s `std::shared_mutex` is glibc's reader-preferring rwlock. Under a steady read load a writer waited until *no* reader held the lock at all. Measured: 8 readers at 768-d cut a writer from 191 to 0.6 writes/s, and hybrid-search readers blocked writes for the entire test.
  - The fair lock makes new readers queue once a writer is waiting.
  - When that writer finishes, every reader that queued during its turn goes before the next writer. Neither side can starve.
  - Building with `-DFEATHER_STD_SHARED_MUTEX` restores the old lock, for A/B benchmarking.
  - The lock is **not recursive**. A thread holding it shared must never take it again, or a waiting writer will deadlock it. No engine method does this.
  - **Reader fast path.** While no writer is waiting or active, `lock_shared` and `unlock_shared` are a single atomic CAS and a single `fetch_sub` on one word. The internal mutex is used only once a writer is involved.
  - HNSW's visited-list pool hands each thread its own list through a per-thread atomic slot.
  - Both are covered by `tests/cpp/lock_stress.cpp`, which is also run under ThreadSanitizer.
- Salience updates are the only writes allowed under the shared lock, and they are safe because the counters are `mutable std::atomic`.
- **Lock order:** `save_mutex_` → `mutex_` → `wal_mutex_`. `save_mutex_` serializes concurrent `save()` calls, because they share the `.tmp` path. `wal_mutex_` guards the WAL file, the LSNs and the group-commit leader flag. The WAL fsync itself runs with **no** lock held.
- **Background compactor:** a lazily-started thread that runs `compact()` when auto-compaction asks for it. `close()` and `~DB()` stop and join it, and never do so while holding `mutex_`.
- pybind11 releases the GIL around `add`, `add_batch`, `search`, `keyword_search`, `hybrid_search`, `context_chain`, `link`, `save`, `compact`, `forget` and `purge`. Python threads therefore really do search in parallel.

### 5.9 Performance knobs

| Knob | Effect |
|---|---|
| `set_ef(ef, modality="")` | HNSW search beam width. The default is **50**, which gives ~0.97–0.99 recall@10. The hnswlib default of 10 gave only ~0.2 at 50k+ vectors. |
| `set_quantized(modality, True)` | On-disk int8 with a per-vector scale (format v7). The file is ~3× smaller. Vectors are dequantized on load, so RAM use is unchanged. |
| `set_int8_ram(modality, max_abs)` | In-RAM int8 under one global scale, `max_abs/127`, using `Int8L2Space`. RAM drops ~1.7×. It is lossy and **must be called before the first add** to that modality. |
| `FEATHER_LOAD_THREADS` | Caps the thread pool used for parallel graph builds (load, `add_batch`, `compact`). |
| `FEATHER_BATCH_CHUNK` | `add_batch` chunk size between lock releases. Default `max(64, 2 × cores)`. |
| `FEATHER_WAL_STRICT` | `1` = fsync inside the exclusive lock, so no read ever sees a not-yet-durable record. Slower under mixed load. |
| Distance kernels (0.19) | Chosen at **runtime** from CPUID (`include/feather_simd.h`): AVX-512F, AVX2+FMA, SSE2 or scalar on x86-64, NEON on arm64. Each wide kernel is compiled with a per-function `target` attribute, so one portable binary (including the PyPI wheels, which used to be SSE-only) uses the best ISA the machine has. `feather_db.core.simd_info()` reports the choice. |
| `FEATHER_SIMD` (build time) | `auto` (default: portable, runtime dispatch), `native` (`-march=native`, not portable, never for wheels), `none` (scalar). The legacy values `sse`, `avx` and `avx512` mean `auto`. |
| `FEATHER_SIMD_RUNTIME` | Caps the runtime level (`scalar\|sse\|avx2\|avx512`). Useful for A/B tests; it can never enable an ISA the CPU lacks. |
| `FEATHER_PREFILTER_MODE` / `FEATHER_PREFILTER_C` | Filtered-search routing: `auto` (default: walk when `candidates × sel² > C × k`, C = 10), `exact` or `hnsw` forces a route. |
| Adaptive capacity | Each index starts with room for 4096 elements and doubles on demand. There is no fixed maximum. |

---

## 6. Bindings: how Python and Rust reach the engine

### Python — `bindings/feather.cpp` → `feather_db.core`

This module exposes `DB`, `Metadata`, `ContextType`, `Edge`, `IncomingEdge`, `ScoringConfig`, `SearchFilter`, `SearchResult` and the `ContextChain*` types.

- NumPy arrays are copied into `std::vector<float>`, forced to C-contiguous float32 so strided views are read correctly. `add_batch` takes an N×dim array.
- `DB` is held with `py::nodelete`: Python garbage collection never destroys it, so the destructor's auto-save never runs. **Call `close()`**, which checkpoints and releases the inter-process file lock, or use `with DB.open(p) as db:`.
- New in 0.19:
  - `close(save=True)`, `closed`, context-manager support
  - `wait_for_compaction(timeout)` and `is_compacting()`
  - `wal_size()`, which drives a caller-side checkpoint policy, and `wal_fsync_count()`
  - `search(..., with_cosine=True)`, which fills `SearchResult.cosine`
  - `SearchResult.to_dict(include_metadata)` and `Metadata.to_dict()`, which build JSON-ready dicts in C++
- `feather_db/__init__.py` re-exports the core types along with the Python helpers (`FilterBuilder`, `MemoryManager`, `ContextEngine`, providers, connectors, …).

`FilterBuilder` (`feather_db/filter.py`) is a fluent wrapper around `SearchFilter`:

```python
f = (FilterBuilder().namespace("acme").entity("user_1")
     .attribute("channel", "instagram").types([ContextType.FACT])
     .after(ts).min_importance(0.5).build())
```

### Rust — `src/feather_core.cpp` + `feather-cli/`

- `feather_core.cpp` exports a C ABI:
  - `feather_open`, `feather_add`, `feather_add_with_meta`, `feather_link`, `feather_touch`
  - `feather_search`, `feather_search_with_filter`, `feather_save`, `feather_close`
  - `feather_forget`, `feather_purge`, `feather_forget_expired`
- `feather-cli/build.rs` compiles a **vendored copy** of the C++ in `feather-cli/cpp/`, which `scripts/sync-cpp.sh` generates because `cargo package` can't reach outside the crate. CI fails if the copy drifts.
- CLI commands (`src/main.rs`):
  - `feather new <path> --dim N`
  - `feather add <db> <id> -n vec.npy [--importance --context-type --source --content --modality]`
  - `feather link <db> <from> <to>`
  - `feather search <db> -n q.npy [--k --type-filter --source-filter --modality]`
- The Rust side has no namespace, entity, attribute, graph, BM25 or hybrid features. Those exist only in Python and the API.

---

## 7. The Python layer (`feather_db/`)

Every module here drives the engine through the core `DB` API. Nothing below the pybind layer knows about LLMs.

### 7.1 LLM providers (`providers.py`)

```python
class LLMProvider(ABC):
    def complete(self, messages, max_tokens=512, temperature=0.0) -> str: ...
```

Every LLM-backed component calls only `provider.complete(...)`, so any object with that method can be plugged in.

| Provider | Default model | Notes |
|---|---|---|
| `ClaudeProvider` | `claude-haiku-4-5-20251001` | Reads `ANTHROPIC_API_KEY` |
| `OpenAIProvider` | `gpt-4o-mini` | `base_url` works for Azure, Groq, vLLM, … |
| `OllamaProvider` | `llama3.1:8b` | Talks to localhost:11434 |
| `GeminiProvider` | `gemini-2.0-flash` | Always requests a JSON response type |

Separately, `integrations/llm.py::make_chat()` is a small urllib chat helper used by the benchmarks. It is not an `LLMProvider`.

### 7.2 The Phase 9 ingest pipeline — raw text → knowledge graph

This is the main write path for "agentic context". It lives in `pipelines/ingest.py` and `extractors/`.

```
IngestRecord(content, source_id, timestamp, metadata)
      │
      ▼
IngestPipeline._ingest_one
 1. SOURCE record   id 1e9+n   kind=source_record, content=raw text,
                    entity_id=source_id, namespace=pipeline ns → db.add(embed(text))
 2. FactExtractor   (LLM) → [Fact(subject, predicate, object, confidence, valid_at)]
 3. EntityResolver  (LLM) → canonical ids for all subjects/objects
                    e.g. "Acme" → brand::acme,  unknown → unknown::<slug>
 4. TemporalParser  (rules, no LLM) → dates   [currently only counted in stats]
 5. for each fact:
    a. (phase2) ContradictionResolver: filtered search for same subject+predicate
       in this namespace → rule-based + optional LLM severity
    b. FACT record  id 2e9+n  content "s p o", importance = confidence,
                    timestamp = valid_at or source ts, attrs kind=fact,subject,predicate,object
    c. link fact ──extracted_from──► source
    d. ENTITY record id 3e9+n (deduped by canonical id)
       link fact ──refers_to (weight=entity conf)──► entity
    e. link new fact ──contradicts (1.0 / 0.7 / 0.4 by severity)──► old fact
 6. (phase2) OntologyLinker (LLM) over this record's facts → typed fact↔fact edges
    (caused_by, supports, supersedes, part_of, …; contradicts/supersedes need a rationale)
      │
      ▼
IngestStats(records, facts, entities, failures, contradictions_by_severity, …)
```

The graph that results looks like this:

```
             ┌──refers_to──► [ENTITY brand::acme]
[FACT] ──────┼──refers_to──► [ENTITY campaign::summer_sale]
   │         └──extracted_from──► [SOURCE memo-001 (raw text)]
   └──contradicts / supersedes / supports──► [older FACT]
```

The extractors:

| Extractor | File | Uses LLM? | Output |
|---|---|---|---|
| `FactExtractor(provider, max_facts_per_call=20, min_confidence=0.5)` | `facts.py` | yes | `Fact` triples |
| `EntityResolver(provider, known_entities)` | `entities.py` | yes | `Entity(surface_form, canonical_id, kind, confidence, aliases)`, always one per input |
| `TemporalParser(anchor, date_format="us")` | `temporal.py` | no | `ExtractedTimestamp` (ISO, M/D/Y, "March 2024", "Q3 2025", "3 weeks ago", "last month", …) |
| `OntologyLinker(provider, allowed_relations)` | `ontology.py` | yes | `OntologyEdge`. Drops hallucinated ids, self-loops, and unknown relations. |
| `ContradictionResolver(provider=None, numeric_tolerance=0.02)` | `contradictions.py` | optional | `ContradictionFinding(severity, suggested_resolution)`. Detects only; never deletes. |

All LLM extractors share a pattern:
1. Call `temperature=0` and ask for JSON output.
2. Retry up to 2 times with backoff.
3. Parse robustly with `_jsonparse.extract_json`, which tries fenced blocks, then balanced braces, then a trailing-comma repair.

The contradiction rules work like this:
- Subject and predicate must match (after normalization) and the objects must differ.
- Numeric objects within 2% of each other → `possible`/merge.
- Numeric objects further apart → `probable`.
- Non-numeric objects → `definite`.
- If both facts carry `valid_at`, the newer one becomes `possible`/supersedes.

**The pipeline does not call `save()`.** The caller must.

### 7.3 The query side — `reason/`

- `QueryPlanner(db).plan(query, context)` currently always emits a single `hybrid_search` step. The `provider` argument is accepted but not yet used.
- `PlanExecutor(db, embedder).execute(plan, query)` embeds the query and runs `db.hybrid_search`, falling back to `db.search` on error. It returns `PlanResult(results, step_traces, warnings, elapsed)`.
- Other planned step kinds are stubs that only emit a warning: `vector_search` works, while `attribute_scan`, `expand_graph`, `rerank` and `hierarchy_walk` do not.
- **Feather never writes the final answer.** The calling agent or LLM composes it from the retrieved records.

### 7.4 `ContextEngine` (`engine.py`) — the v0.7 "classify then store" path

This is a simpler alternative to the pipeline: one text in, one record out. `engine.ingest(text, hint)` runs these steps:

1. Embed the text.
2. Sample the 10 most similar existing records as context.
3. Ask the LLM to classify the text as JSON: `entity_type`, `importance`, `confidence`, `ttl`, `namespace`, `episode_id`, and up to 5 `suggested_links`. If the LLM fails, keyword heuristics are used instead.
4. Apply any `hint` overrides.
5. `db.add` the record, using a hashed id and `entity_id = entity_type`.
6. `db.link` the suggested links.
7. Attach the record to an episode.
8. Fire watch triggers.
9. Run the contradiction detector.
10. `save()`.

It has no query method; query `engine.db` directly.

### 7.5 Memory utilities

| Module | What it does |
|---|---|
| `memory.py` — `MemoryManager` | `why_retrieved` explains a score. `health_report` counts hot, warm, cold, orphan and expired records. `search_mmr` gives diverse results via Maximal Marginal Relevance. `assign_tiers` writes the `tier` attribute. `consolidate` clusters similar recent records, creates a summary node linked with `consolidated_into`, and lowers the originals' importance to 0.3. |
| `episodes.py` — `EpisodeManager` | Episode header nodes (EVENT, `entity_id="__episode__"`, deterministic id) linked to members with `episode_contains`. Supports begin, add, get, close and list. |
| `triggers.py` — `WatchManager` | In-memory "alert me when something similar to X arrives" callbacks, based on cosine threshold. |
| `triggers.py` — `ContradictionDetector` | Structural check: a new record with similarity ≥ 0.9 to an existing record from a *different source* gets a `contradicts` edge. |
| `merge.py` — `merge(target_db, source_path, conflict_policy)` | Copies records and edges from another `.feather` file. Policy is `keep_target`, `keep_source` or `merge`. |
| `hierarchy.py` | In-memory Brand → Channel → Campaign → AdSet → Ad → Creative tree helper. Not wired into anything yet. |
| `feedback/` | `FeedbackLog` is an append-only JSONL log of events: fact corrected, endorsed, retracted, thumbs up/down, … `feedback_decay_modifier` turns a record's event history into a score multiplier. Nothing applies this modifier to search yet. |
| `domain_profiles.py` | `MarketingProfile` maps brand → namespace and user → entity, and stores channel, CTR, ROAS, … as attributes. |
| `graph.py` | `RelType` constants, `export_graph()` and `visualize()` (offline D3 HTML). |

---

## 8. Agent integrations and the MCP server

### 8.1 `FeatherTools` (`integrations/base.py`) — the tool surface

`FeatherTools(db_path, dim=3072, embedder=callable, system_provider=None, namespace)` opens the DB and implements every tool. `TOOL_SPECS` is the single source of truth for the tool schemas, and `handle(name, input)` dispatches calls. If you pass `system_provider`, it builds an `IngestPipeline` with `FactExtractor` and `EntityResolver`. Phase 2 (contradictions and ontology) is **off** there.

| Tool | Does |
|---|---|
| `feather_search` | Vector search with namespace, entity and product filters |
| `feather_context_chain` | Vector search plus graph BFS |
| `feather_get_node`, `feather_get_related` | Inspect a node and its edges |
| `feather_add_intel` | Store one record directly |
| `feather_link_nodes` | Add an edge |
| `feather_timeline` | Records sorted by time |
| `feather_forget`, `feather_expire` | Deletion |
| `feather_health`, `feather_why`, `feather_mmr_search`, `feather_consolidate` | `MemoryManager` wrappers |
| `feather_episode_get` | Read an episode |
| `feather_ingest` | Run the Phase 9 pipeline, or store raw text if no provider is configured |
| `feather_recall` | Hybrid search (BM25 + vector + decay scoring), namespace-filtered |

Connectors wrap the same tools for each vendor's function-calling format:

| Connector | Details |
|---|---|
| `ClaudeConnector` | `tools()` in Anthropic format, `run_loop(client, messages)` |
| `OpenAIConnector` | OpenAI function-calling format |
| `GeminiConnector` / `GeminiEmbedder` | Gemini function calling and embeddings |

### 8.2 The MCP server (`integrations/mcp_server.py`, console script `feather-serve`)

```
feather-serve --db acme.feather --dim 768 --namespace acme \
              --embed-provider gemini --system-provider claude
       │
       ├─ local backend  (--db):      FeatherTools, 16 tools, full Phase 9
       └─ remote backend (--api-url): RemoteFeatherTools → Cloud REST API, 8 tools
                                      (ingest, recall, keyword_recall, context_chain,
                                       get_record, link, stats, list_namespaces)
```

- **Embeddings for the MCP server** come from `integrations/embedders.py::make_embedder`. It supports `gemini | openai | voyage | cohere | ollama | hash` using urllib only, and pads or truncates vectors to `--dim`. `hash` is a deterministic embedder that needs no key, and it is the default. It's fine for tests but gives poor semantic recall.
- `call_tool` runs `tools.handle` on a thread and returns JSON text. There is one resource, `feather://db/info`.

> Fixed in 0.19: `main()` used to call `asyncio.run(stdio_server(server))`, which passed the server object as `stdin`. `stdio_server` is an async context manager. It now runs `async with stdio_server() as (r, w): await server.run(r, w, ...)`.

### 8.3 Framework adapters

| Adapter | Details |
|---|---|
| `langchain_compat` | `FeatherVectorStore`, `FeatherMemory` (decay-scored conversational memory; `clear()` purges the namespace), `FeatherRetriever` (context chain) |
| `llamaindex_compat` | `FeatherVectorStore`, `FeatherReader`. Import them from `feather_db.integrations.llamaindex_compat` directly; see §13. |

---

## 9. The Cloud API (`feather-api/`)

A FastAPI app that loads the `feather_db` engine into its own process. Each process is **single-worker** and is the single owner of its namespaces. To use more cores, run several processes behind `feather-gateway` (§9.6).

### 9.1 Storage and lifecycle (`app/db_manager.py`)

- Each namespace is one file, `{FEATHER_DATA_DIR}/{ns}.feather` (default `/data`), with names sanitized to `[A-Za-z0-9_-]`.
- At startup every existing `.feather` file this process owns is opened. A corrupt file, or one held by another process, is logged and skipped.
- `get(ns, create=True)` creates a namespace lazily on the first write. Read routes return 404 for unknown namespaces.
- A namespace's dimension is fixed by its first vector.
- Locking and threads:
  - Writes take a per-namespace Python lock and run on a **separate bounded write pool** (`FEATHER_WRITE_THREADS`, default 8). A burst of blocking writes can therefore no longer fill the 40-thread pool that searches run on. Measured before 0.19: 64 ingest clients cut search throughput 85%.
  - Reads take no Python lock; they rely on the engine's shared lock and on the GIL being released.
- Saves (0.19):
  - Mutations do **not** save. Every mutation is already durable in the WAL.
  - A **background checkpointer** saves a namespace once its WAL passes `FEATHER_CHECKPOINT_WAL_MB` (64) or has been dirty for `FEATHER_CHECKPOINT_MAX_AGE_S` (300 s). It checks every `FEATHER_CHECKPOINT_POLL_S` (5 s).
  - Before 0.19 a single `DELETE` rewrote the whole namespace file: measured 15 deletes/s.
  - `flush`, `save`, `compact`, `quantize` and create still save explicitly.
  - `close_all()` runs on shutdown and releases every file lock.
- Deletes prune incoming edges through the engine's reverse index, in O(in-degree). Previously they scanned every record's metadata in Python and missed records stored under a non-`text` modality.
- **Upload (`adopt`)** replaces a namespace with an uploaded file:
  1. Validate the magic number and version.
  2. Back up the current file to `.bak`.
  3. Close the live handle without saving, then delete the stale `.wal`, `.wal.old` and `.tmp`.
  4. Swap the new file in with an atomic `os.replace`.
  5. Reopen it, restoring the backup if the open fails.

### 9.2 Auth

- There is one server-wide key, `FEATHER_API_KEY`, sent in the `X-API-Key` header. Every `/v1/*` route requires it.
- If the key is missing, startup **fails closed** unless `FEATHER_DEV_MODE=1`.
- There is **no per-namespace authorization**: anyone with the key can access every namespace.

### 9.3 Embeddings (`app/embedding.py`)

The embedding provider is pluggable: `openai | azure_openai | gemini | voyage | cohere | ollama | none`, plus `mock` in dev mode only.

- **Transport (0.19):** pooled `httpx` clients (no TLS handshake per call), and provider-side batching of up to 96 texts per request.
- **Caching:** a small LRU cache, `FEATHER_EMBED_CACHE` (default 2048 entries).
- **`/ingest_text` is async.** It awaits an `EmbedBatcher` that merges concurrent ingests into one provider call, controlled by `FEATHER_EMBED_BATCH_WINDOW_MS` (5) and `FEATHER_EMBED_BATCH_MAX` (64). No thread is held while the provider responds.
- **Configuration:** env vars, or at runtime with `PUT /v1/admin/embedding_config`. Runtime changes are held in memory and lost on restart.
- **Dimensions:** vectors are **never padded or truncated**. OpenAI/Azure `text-embedding-3-*` and Gemini are asked for `FEATHER_EMBED_DIM` natively, because Matryoshka shortening is done by the provider. Any other mismatch returns a 400 that names both numbers. Before 0.19, a 1536-d model with the default dim of 768 was silently cut in half.

### 9.4 Routes, grouped

| Group | Routes |
|---|---|
| Health / meta | `GET /health` (no auth), `GET/POST /v1/namespaces`, `DELETE /v1/namespaces/{ns}`, `GET /v1/namespaces/{ns}/stats`, `GET /v1/{ns}/schema`, `/hierarchy`, `/top_recalled` |
| Write | `POST /v1/{ns}/vectors` (you supply the vector), `POST /v1/{ns}/ingest_text` (server embeds), `POST /v1/{ns}/import` (bulk; `add_batch`; returns 200, 207 or 400), `POST /v1/{ns}/flush`, `/save`, `/seed` |
| Records | `GET /v1/{ns}/records` (cursor pagination), `GET/PUT/DELETE /v1/{ns}/records/{id}`, `PUT …/importance`, `POST …/link`, `DELETE …/link/{to}`, `POST /v1/{ns}/records/batch_delete`, `POST /v1/{ns}/purge`, `/compact` |
| Search | Bodies of the hot routes (search, keyword, hybrid, context_chain, vectors, import) are decoded with msgspec (`app/fastparse.py`); the pydantic models remain the documented schemas. `POST /v1/{ns}/search` (vector; `track=false` skips salience; `raw_score` returns the true cosine, computed in the engine), `/keyword_search`, `/hybrid_search`. All take `include_metadata=false` for id/score-only hits; vector routes accept `vector_b64` (base64 little-endian float32) instead of a JSON float list. Responses are serialized with orjson straight from C++-built dicts. |
| Graph | `GET /v1/{ns}/records/{id}/edges`, `POST /v1/{ns}/context_chain`, `GET /v1/{ns}/graph` |
| Admin | `GET /v1/admin/overview`, `/metrics` (p50/p95/p99), `/activity`, `/ops_timeseries`, `/connection_info`, `GET/PUT /v1/admin/embedding_config`, `/embedding_models`, `POST /v1/admin/upload`, `GET /v1/{ns}/admin/index_stats`, `PUT …/auto_compact`, `PUT …/quantize` |
| UI | `/admin/` is a single-file Alpine, Tailwind and D3 SPA that loads from CDNs. `/` and `/dashboard` redirect there. |

> Search routes take a **vector**, not text. The client embeds the query itself. Only the write path (`ingest_text`, `import`) embeds on the server.

The metrics middleware (pure ASGI since 0.19) keeps the last 2000 requests in an in-memory ring buffer; they are not persisted.

### 9.5 Deploying

`feather-api/Dockerfile` is a multi-stage build that compiles the wheel, then serves it with uvicorn on port 8000 (1 worker) with a volume at `/data`. The build context must be the repo root. Deploy configs are provided for `docker-compose.yml`, Fly, Render and Railway. CI publishes the image to `ghcr.io/<owner>/feather-api` on `v*` tags.

### 9.6 Scaling out: shards + gateway

- Run N API processes with `FEATHER_SHARD_COUNT=N` and `FEATHER_SHARD_INDEX=0..N-1`. Each serves only the namespaces with `crc32(ns) % N == index`, and answers **421** (with `owner_shard`) for the rest.
- `feather-gateway/` is a small async proxy with the same owner rule. It routes `/v1/{ns}/…` to the owning shard, fans out `GET /v1/namespaces` and the admin overview, and broadcasts the embedding config.
- The engine's file lock makes a routing mistake fail loudly instead of corrupting a file.
- See `docs/deploy-sharding.md` and `feather-api/docker-compose.sharded.yml`.

---

## 10. End-to-end traces

### Trace A — Library: store, search, crash, recover

```python
import feather_db as fdb, numpy as np, time

db = fdb.DB.open("ctx.feather", dim=768)          # read file (none yet) + replay WAL (none)

m = fdb.Metadata()
m.content = "User prefers dark mode"; m.timestamp = int(time.time())
m.namespace_id = "acme"; m.entity_id = "user_1"; m.set_attribute("channel", "web")
db.add(id=1, vec=embed(m.content), meta=m)
#   → WAL ADD + fsync → HNSW("text") created, point inserted
#   → metadata_store_[1]; ns/entity/attr indexes; BM25 postings {user, prefers, dark, mode}

db.link(1, 2, "supports", 0.8)                   # WAL LINK; edge + reverse index

f = fdb.FilterBuilder().namespace("acme").build()
hits = db.search(embed("theme settings"), k=5, filter=f,
                 scoring=fdb.ScoringConfig(half_life=30, weight=0.3))
#   → namespace is indexed → exact scan over acme's ids → Scorer → top 5
#   → each returned hit: recall_count += 1 (atomic)

# 💥 process killed here — no save()
db = fdb.DB.open("ctx.feather", dim=768)
#   → no base file → replay WAL: ADD 1, LINK 1→2 → rebuild indexes → record 1 is back
#     (its recall_count from the search is lost: salience isn't WAL-logged)
db.save()                                         # write .tmp → rename → delete WAL
```

### Trace B — Agent: Claude → MCP → Phase 9 pipeline → recall

1. The user configures Claude Desktop to run `feather-serve --db acme.feather --dim 768 --embed-provider gemini --system-provider claude --namespace acme`.
2. Claude calls **`feather_ingest`** with `content="Acme launched the Summer Sale on March 15, 2024. CTR averaged 4.5% in week one."` and `source_id="memo-001"`.
3. `FeatherTools.feather_ingest` wraps the input in an `IngestRecord` and calls `IngestPipeline.ingest`, which:
   - stores the **source** record `1000000001` with the Gemini embedding of the raw text,
   - calls the Claude Haiku **FactExtractor**, which returns `(Acme, launched, Summer Sale, 2024-03-15)` and `(Summer Sale, had_average_CTR, 4.5%)`,
   - calls the **EntityResolver**, which returns `brand::acme` and `campaign::summer_sale…`,
   - stores the facts as `2000000001` and `2000000002`, each embedded as `"s p o"` and with `importance = confidence`,
   - stores the entities as `3000000001` and up, deduplicated by canonical id,
   - adds the edges `fact ─extracted_from→ source` and `fact ─refers_to→ entity`,
   - and finally calls `db.save()` (FeatherTools does this).
4. Later, Claude calls **`feather_recall(query="How did the Summer Sale perform?")`**. This:
   - embeds the query,
   - runs `db.hybrid_search(vec, query, k=16, scoring=ScoringConfig(30, 0.3))`, where BM25 matches "summer" and "sale", HNSW matches the meaning, and RRF fuses the two,
   - filters the results to namespace `acme` and returns the top k.
5. Claude can call `feather_get_related(2000000002, direction="both")` or `feather_context_chain` to walk to the raw memo and the entities, for provenance.
6. **Claude writes the answer**, using the returned records as grounded context.

### Trace C — Cloud: text in over HTTP

```
POST /v1/acme/ingest_text  {"text": "...", "metadata": {...}}   X-API-Key: …
 → verify key → EmbeddingService.embed(text) (e.g. OpenAI) → pad/trim to FEATHER_EMBED_DIM
 → manager.get("acme")  (auto-create /data/acme.feather)
 → dim check → random JS-safe id → Metadata (namespace=acme, content=text, ts=now)
 → under ns lock: db.add()  (WAL + fsync)  → throttled save (≤ every 30 s)
 ← {id, namespace, embedded: true, dim}

POST /v1/acme/hybrid_search {"vector": [...client-embedded...], "query": "...", "k": 10}
 → db.hybrid_search (shared lock, GIL released) → JSON hits
```

---

## 11. Build, packaging, CI, tests, benchmarks

### Build

```bash
python setup.py build_ext --inplace      # compiles feather_db/core.*  (-O3 -std=c++17)
pip install -e ".[dev]"                  # extras: dev, examples, langchain, llamaindex, mcp, all
cd feather-cli && cargo build --release  # Rust CLI
```

- `setup.py` declares `depends=include/*.h`, so editing a header forces a rebuild. If a change seems not to take effect, run `rm -rf build` and rebuild.
- After recompiling, start a fresh Python process. `importlib.reload` does not reload a compiled extension.
- The version string must match in four places: `pyproject.toml`, `setup.py`, `feather_db/__init__.py` and `feather-cli/Cargo.toml`. `scripts/verify.sh` checks this.

### CI (`.github/workflows/`)

| Workflow | Trigger | Does |
|---|---|---|
| `test.yml` | push/PR | ubuntu+macOS × Py3.9–3.13: build, pytest, 6 root feature scripts, cpp-sync check |
| `wheels.yml` | `v*` tag | cibuildwheel cp38–313 for manylinux x86_64 and macOS x86_64/arm64, then PyPI publish (no Windows wheels) |
| `publish-rust.yml` | `v*` tag | `cargo test` + publish `feather-db-cli` |
| `release.yml` | `v*` tag | CLI binaries + GitHub Release |
| `docker.yml` | `v*` tag | Multi-arch `feather-api` image → GHCR |

### Tests

```bash
python setup.py build_ext --inplace
pytest tests -q
python test_persist_graph.py          # root feature scripts, one by one
scripts/verify.sh [quick]             # full pre-release checklist
```

- `tests/conftest.py` provides a deterministic hash embedder and fixtures for a `db` and a `populated_db`. LLM tests use mock providers, and live-provider tests are skipped unless API keys are set.
- Coverage spans:
  - **Core:** core, filter, graph, BM25/hybrid, recall@10 vs brute force
  - **Durability:** concurrency, WAL recovery, WAL durability, open-failure safety
  - **Python layer:** engine, memory, hierarchy, feedback, reason, all extractors, pipelines, Phase 9 smoke, MCP Phase 9 tools
  - **API:** auth, records
- The root `test_*.py` scripts glob for `feather_db/core*.so`, so on Windows (where the extension is a `.pyd`) they need adapting.

### Benchmarks

- `python -m bench run vector_ann --dataset synthetic --n 10000 --dim 128 --k 10` runs a reproducible ANN benchmark and writes JSON to `bench/results/`. `python -m bench report` renders the results.
- On SIFT 500k × 128, recall@10 is 0.972 at ef=50 with p50 ≈ 0.19 ms (≈5.4k QPS), and 0.991 at ef=100.
- The weak spot is 50k × 768 synthetic data, which reached recall@10 of only 0.58 at the time of the report.
- LongMemEval results:

  | Setup | Metric | Result |
  |---|---|---|
  | `benchmarks/longmemeval.py`, BM25 only, no API key | Retrieval recall@5 / @10 | 0.974 / 0.986 |
  | `bench` QA harness, GPT-4o answerer (`bench/results/`) | QA accuracy | ≈0.69 overall |

  Retrieval recall and QA accuracy are different metrics and cannot be compared with each other.

---

## 12. Environment variables

| Variable | Where | Default | Meaning |
|---|---|---|---|
| `FEATHER_WAL_SYNC` | engine | on | `0` skips fsync on WAL writes and checkpoints |
| `FEATHER_WAL_STRICT` | engine | off | `1` = fsync inside the exclusive lock (no read sees a not-yet-durable write) |
| `FEATHER_LOCK` | engine | on | `0` disables the inter-process file lock (e.g. NFS) |
| `FEATHER_BATCH_CHUNK` | engine | max(64, 2×cores) | `add_batch` records per exclusive section |
| `FEATHER_LOAD_THREADS` | engine | #cores | Thread cap for parallel graph builds |
| `FEATHER_SIMD` | build | `auto` | `auto\|native\|none` (runtime kernel dispatch; legacy sse/avx/avx512 = auto) |
| `FEATHER_SIMD_RUNTIME` | engine | detected | Cap the distance-kernel ISA: `scalar\|sse\|avx2\|avx512` |
| `FEATHER_PREFILTER_MODE` / `_C` | engine | auto / 10 | Filtered-search route: `auto\|exact\|hnsw`; C is the cost-model constant |
| `FEATHER_API_KEY` | API | — (required) | Shared API key |
| `FEATHER_DEV_MODE` | API | off | Allow running with no key |
| `FEATHER_DATA_DIR` | API | `/data` | Where namespace files live |
| `FEATHER_DB_DIM` | API | 768 | Reported dimension for empty namespaces |
| `FEATHER_CHECKPOINT_WAL_MB` / `_MAX_AGE_S` / `_POLL_S` | API | 64 / 300 / 5 | Background checkpoint policy |
| `FEATHER_WRITE_THREADS` | API | 8 | Write-route thread pool (kept apart from the search pool) |
| `FEATHER_SHARD_COUNT` / `FEATHER_SHARD_INDEX` | API | 1 / 0 | Namespace sharding (§9.6) |
| `FEATHER_SHARDS` | gateway | — | Shard base URLs, in shard-index order |
| `FEATHER_EMBED_CONCURRENCY` | API | 8 | Parallel provider requests during import |
| `FEATHER_EMBED_CACHE` / `_BATCH_WINDOW_MS` / `_BATCH_MAX` / `_TIMEOUT_S` | API | 2048 / 5 / 64 / 30 | Embedding cache and micro-batching |
| `FEATHER_EMBED_MOCK_DELAY_MS` | API | 0 | Latency of the dev-only `mock` provider |
| `FEATHER_MAX_UPLOAD_MB` | API | 256 | Upload limit |
| `FEATHER_GRAPH_NODE_LIMIT` | API | 4000 | Node cap for graph export |
| `FEATHER_EMBED_PROVIDER / _MODEL / _API_KEY / _BASE_URL / _DEPLOYMENT / _API_VERSION / _DIM` | API | `none`, …, 768 | Server-side embedding |
| `FEATHER_API_URL`, `FEATHER_API_KEY` | MCP remote | — | Remote backend for `feather-serve` |
| `FEATHER_EMBED_API_KEY` | MCP embedders | — | Embedding key (falls back to the provider's own env var) |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY` | providers | — | LLM keys |

---

## 13. Known gaps and gotchas

These were verified in the code at the time of writing. They're worth fixing or at least knowing about.

**Fixed in the 0.19 concurrency work.** For measurements, see `CONCURRENCY_BASELINE.md` (before) and `CONCURRENCY_RESULTS.md` (after).
- **Durability:**
  - Unsynced checkpoints and unchecked write errors (§5.7).
  - `auto_link` and `purge` not WAL-logged.
  - Wrong-dim `add()` logged before validation.
- **Concurrency:**
  - Writer starvation (§5.8).
  - fsync held under the exclusive lock (§5.2).
  - `add_batch`, compaction and `update_metadata` stalling every reader.
- **Correctness:**
  - `add()` with edges not updating the reverse index.
  - Internal searches inflating recall counts: ContextEngine sampling, ContradictionDetector, pipeline candidate lookup, MMR over-fetch and tool over-fetch now pass `record_salience=False` and touch only what they return.
  - `IngestPipeline` restarting its id counters per instance, which overwrote records after a restart. It now continues after the highest stored id and re-learns stored entities.
- **Integrations:**
  - The broken LlamaIndex export.
  - `feather-serve` never starting (§8.2).
- **Cloud API:**
  - Silent embedding pad/truncate.
  - The single process, now scaled out via shards (§9.6).
  - The per-delete full save and O(N) edge pruning.
  - Thread-pool starvation.

**Engine**
- **Salience counters are persisted only by `save()`.** Neither `touch` nor recall updates are WAL-logged.
- **Hybrid search never touches salience, while keyword search always does.** Vector search touches unless `record_salience=False`.
- **The tag filter matches substrings in `tags_json`.** A filter for `"ad"` also matches `"adult"`.
- **Searches don't scale past ~4 Python threads for cheap (128-d) queries.** The GIL-held parts of each call (argument conversion, result objects) dominate. `tests/test_concurrency.py::test_concurrent_search_scales` has failed on this since before 0.19. Use processes (shards) for more throughput.

**Python layer**
- **Parsed dates are discarded.** `TemporalParser` output is counted in stats and then thrown away.
- **Unwired modules.** `Hierarchy`, the feedback decay modifier, and the planner's LLM `provider` are not connected to anything yet.
- **`EntityResolver` doesn't see stored entities.** The pipeline never passes it `known_entities`, so it can't link new mentions to entities already in the DB.
- **`ContradictionDetector` links one way.** It adds a single one-directional `contradicts` edge, even though its docstring says bidirectional.
- **JSON mode may conflict with the extractors.** OpenAI's `json_mode` and Gemini's JSON response type force a JSON *object*, while the fact extractor asks for an *array*.

**Cloud API**
- **No text search.** Search routes need a vector from the client.
- **The admin SPA's vector and hybrid search is not real.** It sends a seeded pseudo-random vector instead of embedding the typed text.
- **No tenant isolation.** One global API key covers every namespace.

**Tooling and docs**
- **The Rust CLI is behind.** It has no namespace, graph or hybrid features, and `feather-cli/README.md` documents a different syntax from `main.rs`.
- **`hf-space/` is stale.** It pins feather-db 0.6.1.
- **Parts of `CLAUDE.md` are out of date.** The version, the Metadata byte layout, and the "Rust CLI v0.12" note all predate the current code. `docs/architecture/phase9-plan.md` describes APIs that don't exist yet, such as `Synthesizer` and `feather_db.consolidation`.

---

## 14. Where to change what

| I want to… | Touch |
|---|---|
| Add a field to every record | `include/metadata.h` → `src/metadata.cpp` (serialize/deserialize, appended at the end with a read guard for older files) → `bindings/feather.cpp` → `scripts/sync-cpp.sh` |
| Add a DB method | `include/feather.h` (pick shared vs. exclusive lock; WAL-log it if it mutates; fsync once per bulk op) → `bindings/feather.cpp` (release the GIL) → `src/feather_core.cpp` + `feather-cli/src/lib.rs` if the CLI needs it |
| Change the file format | Bump `version` in `save_vectors`, add an `if (version >= N)` branch in `read_base_file`, bound every count you read against the bytes remaining, keep `load_complete_ = true` as the last statement of `load_vectors` |
| Change ranking | `include/scoring.h` (and keep `MemoryManager.why_retrieved` in sync) or the RRF block in `hybrid_search` |
| Add an extractor or relation | `feather_db/extractors/`, register it in `IngestPipeline`, and add the relation to `OntologyLinker.DEFAULT_RELATIONS` |
| Add an agent tool | `TOOL_SPECS` + a method in `integrations/base.py` (it appears automatically in MCP, Claude and OpenAI connectors) |
| Add an API route | `feather-api/app/main.py` + `models.py`, with auth via `Depends(verify_api_key)` |
| Ship a release | Bump the version in 4 places → `scripts/verify.sh` → push a `v*` tag (wheels, crate, CLI binaries and Docker image publish automatically) |
