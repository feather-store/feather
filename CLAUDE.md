# Feather DB — Technical Reference for AI Assistants

> This document is the single authoritative guide for understanding, modifying, and extending Feather DB. Read it fully before making any changes.

---

## 1. Project Overview

**Feather DB** is a lightweight, embedded vector database and living context engine written in C++17 with Python and Rust bindings — zero-server, file-based, designed for AI context and embeddings.

**Core Value Proposition:**
- Sub-millisecond ANN (Approximate Nearest Neighbour) search via HNSW
- Native multimodal support (text, image, audio vectors per entity)
- Embedded Contextual Graph — typed+weighted edges, reverse index, auto-link by similarity
- Adaptive Decay / Living Context — frequently accessed items resist temporal decay
- Namespace + Entity + Attributes — generic partition + subject + KV metadata for any domain
- Graph visualizer — self-contained D3 force-graph HTML, fully offline
- Single file persistence (`.feather` binary format **v10**: append-only commits behind two checksummed headers, so a save writes only what changed; persisted HNSW graph for fast cold load, optional int8 on-disk and in-RAM; v3–v9 files load transparently)

**Version:** `0.21.0` (Phase 9 — agent memory: pocket tiering, typed agent store, context packets, MCP over the real protocol, inter-process locking)

> **Phase 9 (v0.19–v0.20) additions** — the agent-memory layer. Four modules,
> all Python, sitting on the unchanged engine:
> - **`feather_db/pocket.py`** — per-agent working memory with HOT / WARM /
>   CACHE tiers. `hot()` returns a budget-fitted working set ranked by *heat*
>   (`recency × use_weight × inherit_weight × importance`, pinned ⇒ 1.0);
>   `recall()` reaches warm storage the budget could not carry. Scopes are
>   dotted and **inherit**: `org.brand.creative` reads its own scope and every
>   ancestor, decayed by `INHERIT_DECAY` per hop. `hot()` is memoised on
>   `(budget, write-generation, second)` — it is O(namespace) cold (171 ms at
>   10k) and 0.001 ms warm. `recall()` **invalidates** it, because `search()`
>   increments `recall_count`: a read that is really a write.
> - **`integrations/langgraph_store.py`** — `FeatherStore`, a LangGraph
>   `BaseStore` (long-term, cross-thread memory, *not* `BaseCheckpointSaver`).
>   Implements the two abstract methods, `batch`/`abatch`. Beyond the interface
>   it adds `scan()` / `count()` / `newest()`, because `search()` has no
>   ordering and no cheap count.
> - **`integrations/agent_memory.py`** — `AgentMemory`, the five kinds an agent
>   stores, with the lifetime of each enforced. See §5.5.
> - **`integrations/mcp_agent.py`** — `feather-agent`, the MCP server shaped
>   like agent memory (5 tools, 3 prompts, 2 resources). Replaces
>   `mcp_server.py`, which is **broken against SDK 2.0** (`Server.list_tools`
>   was removed, so `create_server()` raises on import).
> - **Inter-process locking** — see §9 "File locking". `db.close()` is now
>   required to release a file.

> **Phase 8 (v0.13–v0.15) additions** — see `include/feather.h`, `feather_db/integrations/`:
> - **Parallel HNSW load** (`parallel_add`): graph rebuilt across a thread pool on
>   open — ~4.7× faster. `FEATHER_LOAD_THREADS` caps it. Lazy `open()` (the modality
>   index is created on first `add()`, not eagerly — `dim()` falls back to `default_dim_`).
> - **Parallel batch ingest**: `add_batch(ids, vecs, metas)` builds the graph in
>   parallel with the GIL released (~3.4× faster bulk insert).
> - **SIMD on x86**: SSE/AVX L2 kernels compiled on x86_64 (runtime-dispatched in
>   `space_l2.h`); arm64 unchanged. `setup.py` is platform-conditional; `FEATHER_SIMD`.
> - **In-RAM int8** (`Int8L2Space`): `set_int8_ram(modality, max_abs)` stores int8 in
>   memory (global scale, integer-L2) → ~1.7× less RAM, **file format v8** (int8-ram
>   flag + scale persisted). `is_int8_ram()`. Lossy; best for embeddings.
> - **MCP connector for Claude** (`feather_db.integrations.mcp_server` / `mcp_remote`):
>   `feather-serve` exposes Feather to Claude Desktop/Code — local `--db` OR remote
>   `--api-url` (Cloud API). `--embed-provider gemini|openai|…` for real client-side
>   embeddings (`integrations/embedders.py`). Cloud API gained `/admin/index_stats`,
>   `/admin/auto_compact`, `/admin/quantize`; `/import` uses `add_batch`.
>
> **Phase 7 (v0.11–v0.12) additions** — see `include/feather.h`:
> - **Secondary metadata indexes** (`ns_index_`/`entity_index_`/`attr_index_`):
>   O(matches) lookups via `ids_in_namespace`/`ids_for_entity`/`ids_with_attribute`/
>   `namespace_size`/`list_namespaces`. Rebuilt on load, maintained on mutation.
> - **Pre-filtered ANN search**: `search()` ranks exactly over the indexed
>   candidate set for namespace/entity/attribute filters → complete top-k.
> - **Incremental auto-compaction**: `set_auto_compact(ratio)`; `compact()` now
>   reclaims forgotten records and is orphan-safe (won't resurrect purged ids).
> - **On-disk int8 quantization**: `set_quantized(modality)` → file format v7,
>   ~3× smaller `.feather`, dequantized to float32 on load (RAM index unchanged).

> **What changed since this doc was first written (v0.5.0 era):** the
> `feather-api/` package was rewritten in v0.10. The Gradio dashboard is gone;
> a custom HTML/Tailwind/Alpine.js SPA at `/admin/` replaces it. A pluggable
> embedding service supports OpenAI / Azure OpenAI / Gemini / Voyage / Cohere
> / Ollama. New modules: `app/embedding.py`, `app/metrics.py`. New endpoints:
> create/delete namespace, schema discovery, hierarchy, top-recalled, graph
> export, context_chain, purge, compact, ingest_text, import, ops_timeseries,
> connection_info, embedding_config. The C++ `save_vectors()` was fixed so
> `forget()`/`purge()` survive save+reload. Phase 9.1 Agentic Context Engine
> (FactExtractor, EntityResolver, OntologyLinker, ContradictionResolver,
> FeedbackLog, IngestPipeline, hierarchy module) shipped in v0.9.0 — see
> `feather_db.extractors`, `.feedback`, `.hierarchy`, `.pipelines`, `.reason`.

---

## 2. Repository Structure

```
feather/
├── include/                 # C++ headers (core logic lives here)
│   ├── feather.h            # ← MAIN DB CLASS (read this first)
│   ├── metadata.h           # Metadata struct + ContextType enum + Edge struct
│   ├── scoring.h            # Scorer + ScoringConfig (decay logic)
│   ├── filter.h             # SearchFilter struct
│   ├── hnswalg.h            # HNSW index algorithm (hnswlib fork)
│   ├── hnswlib.h            # hnswlib base interfaces
│   ├── space_l2.h           # L2/Euclidean distance (SIMD optimized)
│   ├── space_ip.h           # Inner Product distance space
│   ├── bruteforce.h         # Brute force fallback index
│   ├── visited_list_pool.h  # Visited node pool for HNSW search
│   └── stop_condition.h     # Search stop condition interface
├── src/
│   ├── feather_core.cpp     # C-compatible extern "C" wrappers (for Rust CLI)
│   ├── metadata.cpp         # Metadata serialize/deserialize
│   ├── filter.cpp           # Filter logic
│   └── scoring.cpp          # Scorer (thin wrapper; logic in .h)
├── bindings/
│   └── feather.cpp          # pybind11 Python bindings (the Python API bridge)
├── feather_db/
│   ├── __init__.py          # Python package exports (DB, Metadata, RelType, etc.)
│   ├── filter.py            # Python FilterBuilder helper class
│   ├── pocket.py            # ← Pocket: per-agent hot/warm/cache memory, scoped
│   ├── domain_profiles.py   # DomainProfile base + MarketingProfile adapter
│   ├── graph.py             # export_graph(), visualize(), RelType constants
│   ├── d3.min.js            # D3.js v7.9.0 inlined for offline visualization
│   └── integrations/
│       ├── langgraph_store.py  # ← FeatherStore: LangGraph BaseStore
│       ├── agent_memory.py     # ← AgentMemory: the five kinds (§5.5)
│       ├── mcp_agent.py        # ← feather-agent: MCP server for SDK 2.0
│       ├── mcp_server.py       #   BROKEN on SDK 2.0 — superseded, do not use
│       ├── mcp_remote.py       #   MCP against the Cloud API instead of a file
│       └── embedders.py        #   make_embedder() + default_embedder() (env)
├── feather-cli/             # Rust CLI crate (feather-db-cli)
│   ├── src/main.rs          # CLI entry point
│   ├── src/lib.rs           # CLI command implementations
│   ├── build.rs             # Rust build script (links C++ core)
│   └── Cargo.toml           # Rust package manifest (v0.20.0)
├── feather-api/             # FastAPI Cloud wrapper (v0.10 rewrite)
│   ├── app/main.py          # FastAPI app + all /v1/* routes
│   ├── app/db_manager.py    # DB lifecycle management + delete()
│   ├── app/models.py        # Pydantic request/response models
│   ├── app/embedding.py     # Pluggable embeddings (OpenAI/Azure/Gemini/Voyage/Cohere/Ollama)
│   ├── app/metrics.py       # In-memory ring buffer for ops + latency
│   ├── static/admin/        # Atlas-style admin SPA (mounted at /admin/)
│   │   └── index.html       # Single-page app (Tailwind + Alpine.js + D3, no build step)
│   ├── Dockerfile           # Multi-stage Docker build
│   └── docker-compose.yml   # Local compose stack
├── examples/                # Working Python usage examples
│   ├── context_graph_demo.py        # Full context graph demo
│   ├── marketing_living_context.py  # Namespace/entity/attribute filtering
│   └── feather_inspector.py         # Local HTTP inspector app
├── benchmarks/              # Performance benchmarks
│   └── stress_test.py       # 100k vector stress test
├── real_data/               # Real dataset files (not committed)
├── p-test/                  # Rust CLI integration tests
├── setup.py                 # Python build (compiles C++ extension)
├── pyproject.toml           # PEP 517 metadata (version: 0.21.0)
├── MANIFEST.in              # Source distribution manifest
└── CHANGELOG.md             # Version history
```

---

## 3. Core Architecture

### 3.1 The `feather::DB` Class (`include/feather.h`)

This is the single most important file. **All logic flows through here.**

```cpp
namespace feather {
class DB {
private:
    // Each named modality (e.g., "text", "visual", "audio") has its own HNSW index.
    // This is the "Multimodal Pocket" design.
    std::unordered_map<std::string, ModalityIndex> modality_indices_;
    std::string path_;
    // Centralized metadata store keyed by entity ID (uint64_t).
    // A record can span multiple modality indices but shares ONE Metadata entry.
    std::unordered_map<uint64_t, Metadata> metadata_store_;
    // Reverse edge index (built in-memory on load)
    std::unordered_map<uint64_t, std::vector<IncomingEdge>> incoming_index_;
};
}
```

**Key Design Decisions:**
- **Multimodal via multiple HNSW indices**: each call to `add(id, vec, meta, modality)` routes to the correct named `ModalityIndex`. New modalities are created on-demand.
- **Shared metadata by ID**: a single `Metadata` object tracks all cross-modal data (edges, recall_count, importance, timestamps) for a given entity ID.
- **HNSW params**: `M=16`, `ef_construction=200`. `max_elements` is **adaptive** (v0.15.3): each modality index starts at `INITIAL_MAX_ELEMENTS = 4096` and doubles via `resizeIndex()` on demand (`reserve()` is called before every insert), so RAM tracks the working set instead of preallocating 1M elements per index.
- **`ef` (search beam width)** defaults to `10`. Higher = more accurate but slower.
- **Reverse edge index**: rebuilt from `metadata_store_` edges on every `load()`. Not persisted separately.

### 3.2 File Format (`.feather` binary v10, append-only)

Codec: `include/format_v10.h`. Writer/loader: the "v10 commit" / "v10 load"
blocks of `include/feather.h`. All little-endian.

```
[0,    4096)  header slot 0     each: 64-byte header, rest zero
[4096, 8192)  header slot 1
[8192, end)   sections, appended by successive commits

header: magic "FEAT" | version 10 | 0xFFFFFFFF | slot | generation u64 |
        committed end u64 | data start u64 (8192) | unix time u64 |
        reserved u64 | xxh64 of bytes [0,56)
section: [type u32][flags u32][payload_len u64][payload][xxh64 u64]
         checksum = xxh64(payload, seed = xxh64(the 16 header bytes))
```

| type | payload |
|---|---|
| `MODALITY` (1) | name, dim, int8_ram + scale, quantized, M, ef_construction, data/link sizes. Starts a full image of that modality: the loader recreates the index empty. |
| `GRAPH` (2) | name, vec encoding (0 raw storage bytes / 1 per-vector int8), slot count u32, entry point, max level, then records `[slot u32][kind u8][level u8]` + (if DATA) `[label u64][vector]` + level-0 links + upper links. Kind LINKS-only = a neighbour whose links an insert rewired (~140 B at M=16). |
| `META` (3) | `[op u8: 1 put, 2 del][id u64][len u32][Metadata::serialize bytes]`. Dead (forgotten/`_deleted`) records are written as deletes. |
| `SALIENCE` (4) | `[id u64][recall_count u32][last_recalled_at u64]` — a search hit costs 20 bytes, not the record. |
| `VECTORS` (5) | `[name][count u32][stride u32][pad u32][pad][count × stride]`: one modality's vectors in slot order, starting on a 64 KiB file boundary. Written by a full rewrite; that image's GRAPH records then use vec encoding 2 (`VEC_IMAGE`, no vector bytes). |
| `VERSIONS` (7, optional) | `[clock u64]` + `[id u64][version u64]` for every live record this commit's META put. **Written after META**: a META put resets the version it is loaded into. |
| `IDEMPOTENCY` (8, optional) | `[key str16][fingerprint str16][created i64][n u32][(id, version) × n][note str32]` per keyed write batch. A full rewrite keeps only unexpired keys. |
| `FULLVEC` (6, optional) | `[name][dim u32][count u32]` + `[slot u32][float32 × dim]`: the originals of an in-RAM int8 modality with `refine > 0`. Written after the same commit's GRAPH records; a GRAPH data record voids the slot's earlier FULLVEC. Only offsets are kept in RAM (`fv_off`). |

**In-RAM int8 originals (refine).** Search on an int8 modality walks with
`ef × refine`, fetches `k × refine` by int8 distance (the wider beam is
needed: without it recall plateaus at ~0.98 however many candidates are
re-ranked), and re-scores them by float L2 against originals
read with `pread` (`full_vector`, `fv_file_`). An original is in exactly one
place: the file (`fv_off[slot]`), the WAL record that logged it
(`fv_pending`, holding a `shared_ptr` to that WAL generation's descriptor so
rotation and removal don't invalidate it), or a RAM copy (only after
converting a float modality, until the next save). Rules if you touch it:
- **A new path that writes a slot's vector calls `keep_full()` after the
  insert** (with its `WalPos`), as `add`/`add_batch`/replay do. Otherwise the
  slot has no original and silently falls back to its int8 distance.
- **Offsets are published by `commit_install()`** under the exclusive lock,
  still inside `save_mutex_`, after `commit_finish()`. A full rewrite swaps
  `fv_file_` to the new inode at the same moment. Anything that renumbers
  slots must hold `save_mutex_` (compaction does) and carry the floats across
  (`remap_full_vectors`). A pending entry is released only if its `seq` is
  the one the commit wrote; a write landing during the fsync is newer.
- `fv_file_` and every WAL pin are opened **before** the stream that parses
  the file, and kept only if the path still names their inode.
- The MODALITY payload has trailing `[refine u8][int8_train u8]`: older
  loaders ignore trailing bytes, so new trailing fields are compatible.

Unknown type → load fails ("newer Feather") unless the section's flags have
bit 0 (`SEC_OPTIONAL`) set. Sections are chunked at 8 MiB so neither a commit
nor a load buffers more.

**A commit** (`save()`): under the SHARED lock, `commit_begin()` appends the
dirty slots / records past the committed end (no fsync) and rotates the WAL;
the lock is released; `commit_finish()` fsyncs the sections, writes the header
with `generation+1` into slot `(generation+1) % 2`, fsyncs it, and
`<wal>.old` is deleted. Nothing ever rewrites a byte a valid header covers.

**What changed** is tracked at the source, never inferred: hnswlib marks every
slot write in `HierarchicalNSW::dirty_` (`DIRTY_LINKS` / `DIRTY_DATA`, set in
`addPoint`, `updatePoint`, `mutuallyConnectNewElement`, mark/unmark delete,
`setExternalLabel`); every metadata mutation calls `mark_meta_dirty(id)`
(`dirty_meta_`, guarded by `mutex_`); a query's salience touch queues the
record once in `salience_dirty_` via `Metadata::recall_unsaved`. **A new code
path that writes a slot or a record without marking it is a change that never
reaches disk** — and the WAL holding it is deleted at the next save.
`tests/test_format_v10.py::TestRoundTrip` and the crash-reopen stress catch it.

**A full rewrite** (`write_full_v10` → `<path>.tmp` → fsync → rename, both
slots get the same header) happens when the file is still v3–v9, after
`purge()` / `compact()` / a `set_quantized()` change (`vacuum_requested_`),
after a commit that failed once its dirty sets were consumed
(`commit_failed_`), on `vacuum()`, or once `committed_end` exceeds
`FEATHER_VACUUM_RATIO` (default 2) × the live estimate + `FEATHER_VACUUM_MIN_BYTES`
(default 4 MiB). A writer removes a stale `<path>.tmp` on open.

**Load** picks the newest valid header (checksum, version, `end <= file size`)
and replays its sections. A section failing its checksum or running past the
end → fall back to the other header, **only if `<wal>.old` exists** (a crash
mid-commit; the WAL redoes it). Without it the bytes rotted after a successful
commit: refuse, leave the file untouched, and let `FEATHER_RECOVER_OLDER=1`
fall back deliberately.
Anything else (unknown type, structural error under a good checksum) throws
rather than fall back — falling back would discard data that is there, and the
writer would then truncate it. A writer then cuts bytes past the end and
overwrites the abandoned header slot. Every count is bounded by the bytes that
would back it; `restoreFinish()` rejects links to missing slots.

**Read-only handles map the vector image** (`load_vectors_section`): the
VECTORS bytes are mmapped `MAP_PRIVATE` into an address-space reservation
(4× the image, for growth), so N reader processes share one copy through the
page cache; WAL replay and later commits write into it copy-on-write (private
pages, the file is never written). Writers read the image into the heap — a
long-lived writer must not pin a vacuumed inode. Only on-disk-quantized
modalities (`VEC_INT8`) are not mappable. `FEATHER_MMAP=0` disables it;
Windows falls back to reading. `storage_stats()["mapped_bytes"]` reports it.
Vectors appended after the image are private to every reader, so a save also
rewrites the file once they exceed `FEATHER_IMAGE_RATIO` (default 1) × the
image (doubling rule). This works because hnswlib's slots were split: level 0
holds `[links][label]`, vectors live in `HierarchicalNSW::vectors_` (see the
"split layout" block); `saveIndexStream`/`loadIndexStream` convert to and from
upstream's interleaved slots, so v9 files are byte-identical. **Never truncate
or rewrite a `.feather` in place** (`cp` over it, `truncate`): a mapped reader
gets SIGBUS. Feather itself only ever truncates bytes past the committed end,
and an image is only ever written by a full rewrite (new inode via rename).

Measured, 100k×384, 10 concurrent readers (`benchmarks/readers_memory.py`):
vectors-only DB 2,268 MB → 891 MB (737 private + 154 shared); with ~300-char
agent text 3,547 MB → 2,283 MB. The rest is heap built at open — metadata,
BM25 postings (rebuilt from content, now with a per-open stem cache: open
2.1 s → 0.72 s), secondary indexes — which mapping does not reach.

**Backward compatibility**: v3–v9 files load transparently (`read_base_file`
falls through to the old reader); the first `save()` rewrites them as v10.
Older builds refuse a v10 file: they read the `0xFFFFFFFF` at offset 8 as the
metadata count and reject it against the file size. `FEATHER_FORMAT=9` keeps
writing v9 whole-file snapshots (rollback path).

**Every dead record's vectors are marked deleted** (`retire_if_dead_nolock`
on add / add_batch / update_metadata / replay; `forget_nolock`; purge), and
`retire_orphan_nodes()` enforces it on open. That invariant is what makes it
safe to save the graph with deleted nodes in it — both writers do, v9 included
(it used to fall back to bare vectors after one `forget()`: 0.15 s → 53 s
opens). Saves drop dead records, so a live node without one would be a
ghost hit with empty metadata.

**v9 layout** (still read): `[magic][version=9][meta_count][records…][modal_count]`
then per modality `[name][dim][quantized][int8_ram (+scale)][persist_graph]` and
either the hnswlib graph blob or `[element_count][id + float32/int8 vector…]`.

**Trade-offs, measured.** 100k×384: 176.5 MB → 0.42 MB written per save after
100 adds + 100 updates, 105 ms → 0.9 ms, writer stall during a save 117 ms →
8 ms (`benchmarks/save_cost.py`). Cold load 75 → 99 ms at 50k×384 (every byte
checksummed, every link validated). An on-disk-quantized file is ~2× smaller
than float at 128-d instead of >2.5×, because v10 persists its links too — and
loads without the rebuild v9 needed. Superseded bytes, including earlier
versions of forgotten records, stay in the file until a vacuum.

---

## 4. Data Structures

### 4.1 `Metadata` (`include/metadata.h`)

```cpp
struct Edge {
    uint64_t    target_id;
    std::string rel_type;  // e.g. "caused_by", "supports", free-form strings ok
    float       weight;    // [0.0-1.0]
};

struct IncomingEdge {
    uint64_t    source_id;
    std::string rel_type;
    float       weight;
};

struct Metadata {
    int64_t  timestamp;       // Unix timestamp of creation
    float    importance;      // Relevance weight [0.0–1.0], default 1.0
    ContextType type;         // FACT | PREFERENCE | EVENT | CONVERSATION
    std::string source;       // Origin identifier (e.g., "gpt-4o", "user")
    std::string content;      // Human-readable text content
    std::string tags_json;    // JSON array string: '["tag1","tag2"]'

    // Context Graph (v0.5.0)
    std::vector<Edge> edges;  // Typed weighted outgoing edges (replaces flat links)

    // Living Context
    uint32_t recall_count;        // Incremented on every search hit (via touch())
    uint64_t last_recalled_at;    // Unix timestamp of last retrieval

    // Namespace / Entity / Attributes (v0.4.0)
    std::string namespace_id;                            // partition key
    std::string entity_id;                               // subject key
    std::unordered_map<std::string,std::string> attributes; // domain KV pairs
};
```

**Native key** (`Metadata.key`, 0.22): unique among the live records of its
`namespace_id` (`key_index_`, maintained by `index_meta`/`deindex_meta`;
`check_key_free_nolock` refuses a second holder before anything is logged).
Serialized as the reserved attribute `"\0key"`, NOT a new field — see the
`version` note in §9 for why a trailing field breaks v9 readers. Never derive
an id from a key (the 53-bit SHA-1 ids collided and overwrote each other):
address keyed records through `db.upsert()` / `id_for()` / `WriteBatch` key
refs, and from Python through `feather_db.keys`, which also finds and adopts
records written at the old hashed ids.

**Valid time** (`Metadata.valid_at`, 0.22): when what a record says
happened or became true, where `timestamp` is when it was recorded. 0 = not
given; `valid_time()` then falls back to the timestamp. Serialized as the
reserved attribute `"\0valid_at"` (decimal) for the same reason as `key`, so
no format, WAL or downgrade change. `valid_index_` orders each namespace by
`(valid_time(), id)` beside `time_index_` (maintained in `index_meta` /
`deindex_meta`); `newest(..., by_valid=True)` walks it, and `SearchFilter`
has `valid_after` / `valid_before`. A new filter field must also go into
`namespace_only()`, or a namespace graph walk skips it. `FeatherStore` maps a
numeric `valid_at` in an item's value onto it; `AgentMemory` writes one for
episodes (event time) and for dated facts and preferences.

**CRITICAL GOTCHA**: `meta.attributes['k'] = v` silently does nothing in Python (pybind11 returns a copy of the map). Always use `meta.set_attribute(key, value)` and `meta.get_attribute(key)`.

**Backward-compat**: Python `meta.links` property still works (returns list of target IDs from edges).

### 4.2 `ContextType` Enum

| Value          | int | Meaning                       |
|----------------|-----|-------------------------------|
| `FACT`         | 0   | Static knowledge               |
| `PREFERENCE`   | 1   | User preference or setting     |
| `EVENT`        | 2   | Time-bound occurrence          |
| `CONVERSATION` | 3   | Dialog turn or message         |

### 4.3 `SearchFilter` (`include/filter.h`)

All fields are `std::optional` — only set fields are evaluated.

```cpp
struct SearchFilter {
    optional<vector<ContextType>> types;
    optional<string>              source;
    optional<string>              source_prefix;
    optional<int64_t>             timestamp_after;
    optional<int64_t>             timestamp_before;
    optional<float>               importance_gte;
    optional<vector<string>>      tags_contains;
    // v0.4.0 additions:
    optional<string>              namespace_id;       // exact namespace match
    optional<string>              entity_id;          // exact entity match
    optional<unordered_map<string,string>> attributes_match; // all KV must match
};
```

### 4.4 `ScoringConfig` + `Scorer` (`include/scoring.h`)

The **Adaptive Decay** formula:

```
stickiness      = 1 + log(1 + recall_count)    # grows with access frequency
effective_age   = age_in_days / stickiness      # sticky items age slower
recency         = 0.5 ^ (effective_age / half_life_days)
final_score     = ((1 - time_weight) * similarity + time_weight * recency) * importance
```

**Default config**: `half_life=30 days`, `time_weight=0.3`, `min_weight=0.0`

---

## 5. Python API (`feather_db`)

### Install
```bash
pip install feather-db  # v0.21.0 on PyPI (binary wheels, cp39-cp314)
# or from source:
python setup.py build_ext --inplace
```

### Python Package Exports

```python
from feather_db import (
    DB, ContextType, Metadata, ScoringConfig,
    Edge, IncomingEdge,
    ContextNode, ContextEdge, ContextChainResult,
    FilterBuilder,
    DomainProfile, MarketingProfile,
    visualize, export_graph, RelType,
)
```

### Quick Reference

```python
import feather_db, numpy as np, time

db = feather_db.DB.open("my_context.feather", dim=768)

# --- Metadata ---
meta = feather_db.Metadata()
meta.timestamp = int(time.time())
meta.importance = 0.9
meta.type = feather_db.ContextType.FACT
meta.source = "pipeline-v1"
meta.content = "User prefers dark mode"
meta.namespace_id = "acme"
meta.entity_id = "user_123"
meta.set_attribute("channel", "instagram")  # ← use this, NOT meta.attributes['k'] = v

db.add(id=42, vec=np.random.rand(768).astype(np.float32), meta=meta)

# --- Multimodal ---
db.add(id=42, vec=np.random.rand(512).astype(np.float32), modality="visual")

# --- Search ---
results = db.search(query_vec, k=10)
results = db.search(query_vec, k=5, modality="visual")

# --- Filtered search ---
from feather_db import FilterBuilder
f = FilterBuilder().namespace("acme").entity("user_123").attribute("channel", "instagram").build()
results = db.search(query_vec, k=10, filter=f)

# --- Scored search (adaptive decay) ---
cfg = feather_db.ScoringConfig(half_life=30.0, weight=0.3, min=0.0)
results = db.search(query_vec, k=10, scoring=cfg)

# SearchResult fields: r.id, r.score, r.metadata
for r in results:
    print(r.id, r.score, r.metadata.content)

# --- Typed graph edges ---
db.link(from_id=1, to_id=2, rel_type="caused_by", weight=0.9)
edges    = db.get_edges(1)      # list[Edge]
incoming = db.get_incoming(2)   # list[IncomingEdge]

# --- Auto-link by similarity ---
db.auto_link(modality="text", threshold=0.85, rel_type="related_to")

# --- Context chain (vector search + BFS graph expansion) ---
result = db.context_chain(query=query_vec, k=5, hops=2, modality="text")
for node in result.nodes:
    print(node.id, node.score, node.hop_distance)

# --- Export / import ---
json_str = db.export_graph_json(namespace_filter="acme", entity_filter="")
vec      = db.get_vector(id=42, modality="text")   # returns np.ndarray
ids      = db.get_all_ids(modality="visual")        # returns list[int]

# --- Metadata-only updates (no HNSW touch) ---
db.update_metadata(id=42, meta=new_meta)
db.update_importance(id=42, importance=0.95)

# --- Salience ---
db.touch(id=42)           # manual boost; called automatically on search hits
meta = db.get_metadata(42)

# --- Compare-and-set + atomic batches ---
m = db.get_metadata(42)                 # m.version: changes on every write, never reused
db.put_if(42, m, m.version)             # raises VersionConflictError if it changed since
b = feather_db.WriteBatch()
b.put(8, vec2, entity_meta).put(7, vec, fact_meta, expected_version=0).link(7, 8, "about")
r = db.write(b, idempotency_key="req-123", fingerprint=h)   # all or none; a retry -> r.replayed

db.save()                 # v10: appends only what changed since the last save
db.vacuum()               # rewrite now, dropping superseded bytes
db.storage_stats()        # {'format', 'generation', 'file_bytes', 'last_commit_bytes', ...}
```

### 5.5 Agent memory — `Pocket`, `FeatherStore`, `AgentMemory`

Three layers, each usable alone:

```python
# ── Pocket: one agent's working set, budget-fitted ───────────────────────
from feather_db.pocket import pocket
pkt = pocket(db, ("org", "brand", "creative"), budget_tokens=4000)
pkt.remember("tone", "captions stay lowercase", pinned=True)
pkt.hot()            # what to carry now, ranked by heat, trimmed to budget
pkt.hot_text()       # the same, as a prompt-ready string
pkt.recall("tone")   # reaches WARM storage the budget could not carry
pkt.stats()          # hot/warm/cache counts, token usage, inherited scopes

# ── FeatherStore: LangGraph long-term memory ─────────────────────────────
from feather_db.integrations.langgraph_store import FeatherStore
store = FeatherStore("agent.feather", dim=768)       # embed= defaults to env
graph = builder.compile(store=store)                 # ← the whole integration

# ── AgentMemory: the five kinds, lifetimes enforced ─────────────────────
from feather_db.integrations import AgentMemory
mem = AgentMemory(store, org="hawky", user="user_842", agent="creative", brand="nike")
mem.preferences.set("comms", "prefers written async")   # revised in place
mem.episodes.record("report shipped late", outcome="miss")  # append-only
mem.facts.state("positioning", "premium")               # corrected, keeps history
mem.procedures.define("weekly", ["pull", "send"])       # versioned as name@N
mem.entities.enrich("nike", {"industry": "apparel"})    # merged, org-wide
mem.observations.note("style", "decisive", evidence=["comms"])
mem.about()                                             # all kinds, one prefix read
mem.recall("how to contact", kinds=["preferences"])
```

```python
# ── PacketBuilder: required context guaranteed, the rest budgeted ───────
from feather_db import PacketBuilder, RequiredRule
builder = PacketBuilder(pkt, budget_tokens=4000,
                        resolve_required=mongo_rule_lookup)   # optional
packet = builder.build(required=["compliance"], query="which hook")
if not packet.may_mutate:
    raise PolicyBlocked(packet.why_blocked())     # fail closed
prompt = packet.text
packet.manifest()    # refs + versions + omissions, for replay
```

**Packet refs are versioned by `Metadata.version`**, never `recall_count`
(every read moves that, and an edit does not). `PacketBuilder.revalidate()`
replays a stored manifest against current versions; a required ref that is
missing or changed blocks the mutation.

**`hot()` is not a policy path.** It fits a budget by heat, so a binding rule
can be crowded out by ordinary notes — silently. Anything that gates a mutation
must go through `PacketBuilder`, which puts required refs first, judges the
required set **whole**, and raises `RequiredContextUnavailable` rather than
returning a partial constraint set. `allow_degraded=True` is for read-only help
only.

**`resolve_required=` makes an external store authoritative, with no pocket
fallback.** That is deliberate: Feather is derived retrieval and indexing is
asynchronous, so a pocket fallback would (a) turn index lag into "rule missing"
and block every mutation, and (b) let a stale Feather copy of a just-changed
rule be enforced as binding. A key the resolver does not return is missing.

| kind | write semantics | get it wrong and… |
|---|---|---|
| `preferences` | upsert by key | six contradictory answers, none current |
| `episodes` | append-only, key embeds the event time | the history that made it an episode is gone |
| `facts` | upsert, previous kept in `supersedes` | cannot answer "since when" |
| `procedures` | new version each write, `name@N` retained | a regression cannot be diffed against what worked |
| `entities` | merge into existing attributes | additions overwrite instead of accumulating |
| `observations` | cites `evidence` keys | a drawn conclusion cannot be traced or retracted |

**Namespace ordering is load-bearing.** Search takes a *prefix*, so
`(org, subject, kind)` makes "everything about this user" a prefix read while
`(org, kind, subject)` makes it a filter over every user in the org. Same data;
the second does not scale. `AgentMemory` owns the ordering so a caller cannot
invert it.

**Never use `store.search(ns, query=None, limit=n)` to mean "the newest n".**
`search` has no ordering, so it returns an arbitrary `n`; sorting those is
sorting the wrong set. At 300 episodes it returned 299, 298, 296, 295, 293 —
silently skipping two. Use `store.newest(ns, n, key=…)`, and `store.count(ns)`
rather than `len(search(...))`. `count` cannot use the engine's
`namespace_size()`, which counts tombstones until compaction.

### Domain Profiles

```python
from feather_db import MarketingProfile

p = MarketingProfile()
p.set_brand("nike")
p.set_user("user_8821")
p.set_channel("instagram")
p.set_ctr(0.045)
p.set_roas(3.2)
meta = p.to_metadata()
```

### Graph Visualization

```python
from feather_db.graph import visualize, export_graph

visualize(db, output_path="/tmp/graph.html")  # self-contained D3 HTML
data = export_graph(db, namespace_filter="nike")  # Python dict
```

---

## 6. Rust CLI (`feather-db-cli`)

**Crate on Crates.io:** `feather-db-cli` v0.21.0

```bash
feather add    --db my.feather --id 1 --vec "0.1,0.2,0.3" --modality text
feather search --db my.feather --vec "0.1,0.2,0.3" --k 5
feather link   --db my.feather --from 1 --to 2
feather save   --db my.feather
```

The Rust CLI calls the `extern "C"` C ABI from `src/feather_core.cpp` via the FFI bridge in `build.rs`. **No exception may cross that ABI** (unwinding through an `extern "C"` frame aborts): a new entry point wraps its body in `guarded(...)` and returns `0`/`-1` (`nullptr`, `SIZE_MAX`), with the message in `feather_last_error()`; the Rust wrapper turns that into `Result<_, String>`. The CLI does **not** yet expose v0.4.0/v0.5.0 graph features (namespace, attributes, context_chain) — those are Python-only for now.

---

## 7. pybind11 Bindings (`bindings/feather.cpp`)

Key points:
- `DB` is owned by Python (default `std::unique_ptr` holder). It was `py::nodelete` until 0.22, so nothing a handle allocated was ever freed (~50 MB per open/close of a 39 MB file). Dropping the last reference now runs `~DB()`: a never-closed writer checkpoints and unlocks; a reader never writes. `close()` frees the memory immediately and every later call raises (`ensure_open()` is in every public method that takes `mutex_` — add it to new ones).
- **Anything that can block runs without the GIL**: `open`, every method that takes `mutex_`, `save_mutex_` or `wal_mutex_` (a reader queues behind a writer — the lock is phase-fair), and `~DB` (`py::release_gil_before_calling_cpp_dtor`, pybind11 ≥ 3 only, behind `#if`). A new binding that takes one of those locks gets `py::call_guard<py::gil_scoped_release>()`; a lambda converts its Python arguments first, releases around the C++ call only, and builds the Python result after. `tests/test_gil_release.py` fails on a regression (the stall equals the call). Losing the GIL also lost an accidental guarantee: opens were never concurrent. The engine now serialises writer opens per path (`open_mutex`) and holds `registry_mutex()` across the whole of `acquire_lock`; `release_lock` unlocks the flock before the registry entry disappears.
- `add()` accepts `py::array_t<float>` (NumPy arrays) and copies data into `std::vector<float>`.
- `search()` accepts optional raw pointers to `SearchFilter` and `ScoringConfig`.
- `get_vector()` returns `py::array_t<float>`.
- `meta.attributes` map mutation via `meta.attributes['k'] = v` silently does nothing — pybind11 returns a copy. Use `meta.set_attribute(k, v)`.

---

## 8. Build System

### Python Extension
```bash
# Development build (inplace, fastest)
python setup.py build_ext --inplace

# Production wheel
python setup.py sdist bdist_wheel
```

`setup.py` compiles:
- `bindings/feather.cpp` (pybind11 entry point)
- `src/filter.cpp`, `src/metadata.cpp`, `src/scoring.cpp`
- Flags: `-O3 -std=c++17`

### SIMD (runtime-dispatched since 0.19)
Distance kernels live in `include/feather_simd.h` and are picked at RUNTIME from
CPUID: AVX-512F / AVX2+FMA / SSE2 / scalar on x86-64, NEON on arm64. Wide
kernels use per-function `FEATHER_TARGET("avx2,fma")` attributes — never add a
global `-mavx`/`-march` to the default build (it makes wheels crash on older
CPUs). Build modes: `FEATHER_SIMD=auto` (default) | `native` | `none`;
`FEATHER_SIMD_RUNTIME=scalar|sse|avx2|avx512` caps the level for A/B tests;
`feather_db.core.simd_info()` reports it. A new kernel needs a numpy
equivalence case in `tests/test_simd.py` (it runs every level the CPU has).

### Rust CLI
`build.rs` declares `rerun-if-changed=cpp`: the `cc` crate's own
`rerun-if-env-changed` lines otherwise switch off Cargo's source tracking, and
a build after `sync-cpp.sh` silently linked the old engine.
`tests/test_cli_v10.py` runs the binary against Python-written files and back.
```bash
cd feather-cli
cargo build --release
cargo publish  # requires cargo login
```

---

## 9. Common Patterns & Gotchas

### Multimodal pockets have independent dims
Each modality gets its own HNSW index. You cannot search text vectors against the visual index.

```python
db.add(id=1, vec=np.rand(768), modality="text")    # 768-dim
db.add(id=1, vec=np.rand(512), modality="visual")  # 512-dim, independent
```

### `meta.attributes` pybind11 copy gotcha
```python
# WRONG — silently does nothing
meta.attributes["channel"] = "instagram"

# CORRECT
meta.set_attribute("channel", "instagram")
value = meta.get_attribute("channel", default="")
```

### `touch()` is called automatically on search
Every `search()` increments `recall_count` for all returned records. Call `touch()` manually only to boost salience outside of search.

### Import cache when recompiling C++ `.so`
`importlib.reload()` does NOT reload compiled `.so` files. Start a fresh Python process to pick up recompiled bindings.

### `load_complete_` — why the destructor is guarded
`~DB()` only checkpoints when `load_complete_` is true, and `save_vectors()`
throws `std::logic_error` otherwise. `DB::open()` holds a `unique_ptr`, so a
throw inside `load_vectors()` unwinds directly into the destructor — without the
flag, a file that failed to parse got a half-parsed fragment written back over
it and its WAL cleared. If you add an early return or a new throw site in the
load path, `load_complete_ = true` must stay the **last** statement of
`load_vectors()`; anything after it is code that ran on a DB already declared
loaded.

Corollary for new format-reading code: every count or length read off disk is
a loop trip count or an allocation size. Rules, all enforced by
`tests/test_corrupt_files.py` (a seeded fuzz; CI also runs it on a rotating
seed — `FEATHER_FUZZ_SEED` reproduces a failure):

- **Bound it by the bytes actually remaining before you loop or allocate.**
  `meta_count`, `modal_count`, `element_count` in `read_base_file()`; the
  base layer and every link list in `loadIndexStream()`; v10 payloads go
  through `v10::Reader`, which checks every read.
- **Validate the factors before you multiply them.** `count * slot_size` was
  once checked while `slot_size` was still unchecked: a forged pair wrapped
  the product under the guard and the base layer overflowed the heap.
- **Check every read.** An unchecked `f.read` leaves the field holding stack
  garbage, which then sizes an index.
- **Strings grow as their bytes arrive** (`get_bytes` in `metadata.cpp`) —
  never `resize(len)` from an on-disk length first.
- **Counts of graph slots are 32-bit**, like the internal ids they count
  (`tableint`); v10 stores them as u32 and v9's 64-bit field is rejected
  above 2^32.
- **A graph is validated before search can follow it** (`checkGraph` in
  `hnswalg.h`): links to missing nodes, links into a layer the target does
  not have, and nodes above the entry point are all corruption.

A checksum is not a bounds check: it only proves the bytes are the ones
written. The fuzz recomputes v10 checksums and WAL CRCs so it reaches the
parsers.

### WAL durability — what the guarantee actually is
The WAL is format **v2**: an 8-byte header (`FWAL` + version) then records of
`[op:1][id:8][plen:4][payload][crc32:4]`. Two rules if you touch it:

- **Verify before applying.** `replay_wal()` checks each record's CRC and stops
  at the first mismatch. A length check alone only catches a short tail; it
  cannot see a record that is the right size with the wrong bytes, and replaying
  one writes garbage into the next checkpoint.
- **Append inside the lock, wait for durability outside it (group commit, 0.19).**
  `wal_append()` writes + flushes and returns the record's LSN; call it while
  holding `mutex_` exclusively so WAL order == apply order. After releasing
  `mutex_`, call `wal_wait_durable(lsn)` — a leader fsyncs everything appended
  so far and releases every writer it covers. A new mutation follows this shape:

  ```cpp
  const std::string payload = encode_...(...);          // no lock
  uint64_t lsn = 0;
  {
      std::unique_lock<RWMutex> lock(mutex_);
      ensure_open();
      /* validate BEFORE logging */
      const uint64_t v = next_version_nolock();
      lsn = wal_append(WalOp::X, id, payload, nullptr, v);   // + its STAMP, one write
      /* apply in memory */
      stamp_nolock(id, v);                               // LAST: an apply may copy a
                                                         //   caller's Metadata over it
      if (wal_strict()) wal_wait_durable(lsn);           // FEATHER_WAL_STRICT=1
  }
  wal_wait_durable(lsn);                                // never inside mutex_ otherwise
  ```

  **Versions** (`Metadata::version`, compare-and-set) are one clock per DB:
  every change to a record takes `next_version_nolock()`, logs it in a
  `STAMP` record and stamps the record last. A new mutation that skips this
  leaves the record's version where it was, so a stale `put_if` succeeds over
  it. `version` is deliberately NOT in `Metadata::serialize()`: the v9 record
  stream has no length prefix, so a new trailing field would break 0.21 readers
  of a `FEATHER_FORMAT=9` file. Replay applies a data record with a provisional
  `++clock` and the STAMP that follows sets the exact value. The clock never
  moves backwards, and a file without one gets a `time << 20` floor
  (`seed_version_clock`).

  **Atomic groups** (`write_batch`): `BEGIN`, the data records, `COMMIT`, all
  built into ONE `wal_append_group()` write. `replay_wal()` buffers a group
  and applies nothing until its `COMMIT`; `good_end` stays at the `BEGIN`, so
  a torn group is cut off whole. Every check runs before logging: nothing may
  throw between the WAL write and the end of the apply except allocation.
  Builds before these ops skip `STAMP`/`BEGIN`/`COMMIT` and still apply each
  data record, so a downgrade loses atomicity, never data. Keep it that way:
  never put a data change in a new op an old build would skip.

  Bulk mutations append all records and wait once on the last LSN. Never fsync
  (or call `wal_wait_durable`) inside the exclusive lock outside strict mode:
  it blocks every reader for the disk flush (measured −68% reads, 1 writer).

v1 WALs (≤0.17.0, no header, no CRC) still replay — the header sniff is exact
because `'F'` cannot be a v1 opcode. Don't remove that path while any deployed
build can still leave a v1 WAL behind. `FEATHER_WAL_SYNC=0` disables fsync.

**A failed WAL write or fsync poisons the handle** (`poisoned_`, PostgreSQL's
PANIC-on-fsync-error rule): that call raises, every later write and save
raises, `close()` releases without a checkpoint, and only a reopen (which
replays the WAL and cuts a torn tail) accepts writes again. Never retry, and
never swallow the error: a record appended after a torn one is unreachable,
and a retried fsync can succeed without the pages that failed. So:

- A new sync on a write path goes through `sync_or_fail(...)`; a new write
  error that may have left bytes in a file calls `fail_writes(why)`. Only
  `fsio::SyncFailed` (an fsync that ran and failed) poisons. Failing to *open*
  a file is an ordinary, retryable error, and "fsync unsupported"
  (`EINVAL`/`ENOTSUP`/`ENOSYS`) counts as synced, as it always did. Don't
  tighten either: a service on a FUSE/network mount, or one briefly at
  EMFILE, would lose every write until restart. Load-time header repair stays
  best-effort.
- `wal_append()` must stay **before** the in-memory apply, and a bulk
  mutation appends a chunk's records before applying any of them, so a throw
  leaves memory untouched.
- The WAL handle is unbuffered (`_IONBF`), one `fwrite` per record: a
  buffered tail of a failed record would be completed later by `fclose()`.
- Tests inject fsync failures with `feather_db.core._fail_fsyncs_after(n)`
  (`tests/test_write_failures.py`); a real short write uses `RLIMIT_FSIZE` in
  a subprocess.
- Refusals raise `feather::WriteFailed` → `feather_db.WriteFailedError`
  (a `RuntimeError`). The Cloud API maps it to a `503` and `DBManager` reopens
  the namespace (`_recover_locked`: new handle first, old one closed only if
  that worked; throttled by `FEATHER_RECOVER_RETRY_S`).

**macOS:** plain `fsync()` does not flush the drive's write cache, so by default
a WAL write survives a process or kernel crash but not a power cut.
`FEATHER_WAL_FULLFSYNC=1` uses `F_FULLFSYNC` (WAL and checkpoints) for real
durability at ~150× the cost per sync (25 µs → 3.7 ms measured). Opt-in, as
SQLite's `fullfsync` is; a no-op on other platforms. It is also why
`test_concurrent_writers_share_fsyncs` sets it: a 25 µs fsync is too short for
writers to overlap, so group commit is unobservable without it.

**Checkpoints (0.19, append-only since v10).** `save()` appends the commit
under the SHARED lock, then `wal_rotate()` renames `<wal>` → `<wal>.old`; the
fsync and header switch happen after the lock is released (§3.2);
`<wal>.old` is deleted last.
`load_vectors()` replays `<wal>.old` then `<wal>`. Replay must stay idempotent
(a crash after the rename replays `<wal>.old` over a base that already has it).
Every new WalOp must be replayable twice without changing the result.

### Locking model — which lock does your new method need?
`mutex_` is a `FairSharedMutex` (phase-fair, atomic reader fast path; alias
`RWMutex`; any change to it must pass `tests/cpp/lock_stress.cpp`, including
under `-fsanitize=thread`) — glibc's
`std::shared_mutex` prefers readers and starved writers under read load. A
writer phase runs the writers already queued back to back for at most
`WRITER_PHASE` (100 µs): one writer per phase made 4 cheap writers slower than
one, and a count bound instead of a time bound let 16 queued `add()`s shut
readers out (9.8k → 0.7k reads/s). Benchmark a change to either with
alternating runs; single runs on a laptop swing ±2×. It is
**not recursive**: never take it shared while already holding it shared (a
waiting writer would deadlock you). Lock order is `save_mutex_` → `mutex_` →
`wal_mutex_`; never join the compactor thread while holding `mutex_`.
Take `std::shared_lock<RWMutex>` if the method only
reads (all const accessors, `search`, `keyword_search`, `hybrid_search`,
`context_chain`, `save`'s snapshot); take `std::unique_lock<RWMutex>` if it mutates
`metadata_store_`, any derived index, or the HNSW graph — including anything
that can trigger `resizeIndex`, which is not thread-safe. The one exception is
salience: `Metadata::recall_count` / `last_recalled_at` are `mutable
std::atomic`, so `touch_nolock()` is `const` and legal under a shared lock.
Don't mutate anything else from a shared section.

### Editing headers? The build may not notice (fixed, but know why)
Nearly the whole engine is in `include/*.h`. setuptools only stat-checks the
listed `.cpp` sources, so before `depends=glob("include/*.h")` was added to
`setup.py`, an incremental `build_ext` after a header edit reused stale objects
and produced a build that looked fresh but contained none of the changes. If you
ever see a change "not take effect", `rm -rf build` and rebuild.

### Filtered search: complete always, exact below FEATHER_NS_ANN_MIN
An indexed filter (namespace / namespace_prefix / entity / attribute) returns
a complete top-k (never short when k match), which `Pocket.recall` depends on.
It ranks exactly over the candidate set, EXCEPT inside a namespace holding
≥ `FEATHER_NS_ANN_MIN` (4096) records of the modality: once that namespace's
graph is built (0.22), the search walks it — approximate like unfiltered
search, recall 0.995–1.0 measured, falling back to the scan if the walk comes
up short. `FEATHER_PREFILTER_MODE=exact` / `set_namespace_ann_min(0)` restore
exact everywhere.

Namespace graphs (`NsGraph`, `SlotRefSpace`) hold SLOT numbers of the
modality's index, not vectors. Rules if you touch them:
- **Anything that renumbers slots or changes the space** (compaction's swap,
  int8 conversion, a new index) calls `drop_ns_graphs_nolock()`, which bumps
  `ann_gen`. A graph used against renumbered slots reads other records'
  vectors — wrong results, no crash to tell you.
- **A write that gives a record a vector or a namespace** reaches
  `ns_ann_note_nolock()`; one that kills it or moves it out reaches
  `ns_ann_unnote_nolock()` (via `deindex_meta`). A namespace-only search walks
  WITHOUT a filter callback, trusting exactly that membership.
- The builder reads slots only under the shared lock, chunk by chunk, and
  rechecks `ann_gen` before each chunk; `close()`/`~DB` stop it before
  freeing (never join it while holding `mutex_`).
`tests/test_namespace_search.py` fails on any of these.

`FEATHER_PREFILTER_MODE=auto` separately enables the cost model that walks
the GLOBAL graph for large unselective non-namespace sets (recall@10 ≈ 0.86 on
a 50% filter). Don't make `auto` the default; it silently changes results.

### Auto-compaction is asynchronous
`set_auto_compact()` compacts on a background thread. `size()` counts
tombstones until it finishes, so read it after `wait_for_compaction()` if you
need the post-compaction number.

### Max elements per index (adaptive since v0.15.3)
Each modality index starts at `INITIAL_MAX_ELEMENTS = 4096` and grows on demand: `reserve()` calls `resizeIndex()` (doubling) before any insert exceeds capacity. There is no hard ceiling — indices grow to fit whatever you store. `resizeIndex` is **not thread-safe**, so `reserve()` runs before `parallel_add` for the full batch size, never from inside it.

### File locking — `close()` is now required (v0.20.0)
`DB.open()` takes an **exclusive** advisory `flock` on `<path>.lock` for the
handle's lifetime. A second *process* is refused, naming the holder's pid.
`DB.open(..., read_only=True)` takes **no lock at all** and is never blocked,
even while a writer holds the file. Every mutation on that handle raises at the
call site.

That is safe, not sloppy: a v10 commit appends past the committed end and only
then switches the header, and a reader parses up to the end of the header it
chose, so it reads one whole commit (a torn header fails its checksum; the
other slot is used). A full rewrite still goes through `<path>.tmp` +
`rename()`. Either way a reader keeps its own consistent snapshot until it
reopens. WAL records carry a CRC32 and replay stops at the first mismatch, so
replaying a log the writer is appending to yields a clean prefix; a read-only
handle never calls `save_vectors()`, so it never clears the writer's WAL. What a
reader gives up is **freshness, not integrity** — reopen to advance.

A shared lock was the first implementation and was wrong: `LOCK_SH` conflicts
with the writer's `LOCK_EX`, so every reader was refused while the writer merely
held the handle open. That blocks the one topology this is for — a single writer
service plus many agent processes reading the same file.

**What this does NOT do:** let two processes write one file. It cannot. Each DB
holds the whole dataset in RAM and `save_vectors()` rewrites the file from that
view, so serialising the saves would only decide *whose* records vanish
(measured before the lock: 20 of 20 lost, both processes exiting 0). Concurrent
writing needs merge-on-save or a paged store — a format change.

Three consequences you will hit:

1. **Close handles deterministically.** `db.close()` (or `with DB.open(...) as
   db:`) checkpoints, releases the lock and frees the memory at a moment you
   choose. Dropping the last reference does the same through `~DB()`, but at
   GC time (and, built with pybind11 < 3, with the GIL held for the checkpoint). A test that wants the WAL
   left behind must say so — `close(save=False)`, or `os._exit(0)` in a child —
   since a discarded writer now checkpoints it away.
2. **The lock is reentrant within one process.** A strict lock would be a
   breaking change for exactly the reason above — it refused programs their own
   file, and broke 26 tests + 34 errors. Same-process double-open therefore
   remains the documented footgun: use **one handle** and separate agents by
   namespace/scope, or `mcp_agent.build(db=...)`.
3. **`FEATHER_LOCK=0`** disables it (flock is unreliable on NFS and some overlay
   mounts). The caller then owns the consequence.

Keep the registry key in `lock_key()` on `realpath(dirname) + basename`, never
on the file: `realpath` only resolves an existing file, and on macOS rewrites
`/var/…` → `/private/var/…`, so keying on the file gave a different key before
and after the first save — reentrancy missed and the handle reported *itself* as
"another process (pid \<ourselves\>)".

`close(save=False)` releases the handle WITHOUT checkpointing, so the WAL is
kept. That is how a test simulates a crash. `close()` also stops the background
compactor first, so call it before handing a file to another process.

### Cloud API: a namespace name IS its file name
`DBManager` keys handles by the name as given, so any mapping that sends two
names to one file gives that file two handles, and each checkpoint overwrites
the other's acknowledged writes. Until 0.22, `_namespace_path` stripped
characters (`ni.ke` → `nike.feather`); a `DELETE` of the alias deleted the real
namespace. Now `validate_namespace()` rejects instead (`InvalidNamespaceError`
→ 400; same allowed set as before, so existing files stay valid), and
`_claim_file()` compares the inode of `<path>.lock` against every open
namespace (`NamespaceAliasError` → 409). That catches case and Unicode folding
on macOS and Windows without merging `Nike`/`nike` on Linux. Never add a
normalisation step such as lowercasing: it is the same bug again.
`tests/test_api_namespace_names.py`.

### Cloud API: errors are typed, overload is a 429, the loop never blocks
Every error body is `{"detail", "error", "retryable", ...}` (`app/errors.py`).
Raise `errors.APIError(status, detail, "<type>", **extra)` when the status
alone does not say what happened; plain `HTTPException` gets the default type
for its status. `retryable` is true only for 429/503: a 5xx a retry cannot fix
must not invite one. Upstream (embedding provider) failures are
`UpstreamError`: transient → 503, refused → 502, never the client's 4xx.

`app/admission.py` bounds requests in flight per class (read / write /
upload) and answers 429 + Retry-After past it. A new read-only POST route goes
in its `_READ_POSTS` pattern, or it is counted (and refused) as a write.

**Nothing blocking runs on the event loop.** An `async def` route awaits
I/O only; disk work, a DB open, a save, an engine call and a large decode go
through `run_in_threadpool` / the write pool. `DBManager` loads an adopted
file outside `_global_lock` (the namespace is in `_loading` meanwhile, and
answers 503 `namespace_loading`) — don't do a long load under that lock: every
namespace's open, recovery and delete waits on it.
`tests/test_api_backpressure.py` fails on either regression.

### Cloud API sharding and namespace lifetime
- **Ownership** is `app/placement.py`: override table, else rendezvous hash
  over shard ids. The gateway carries a copy (`tests/test_gateway.py` checks
  they agree). Never route by position in a list: ids are what is stable.
- **Namespaces open lazily and close when idle** (`DBManager.get` /
  `close_namespace` / `sweep`). Open and close happen OUTSIDE `_global_lock`,
  with the name held in `_transit` so nobody opens a second handle on the
  same file meanwhile (the flock is reentrant in-process: nothing else would
  stop it). A close takes the namespace's write lock and re-checks
  `_last_used` under `_global_lock`; `get()` stamps `_last_used` BEFORE reading
  `_dbs`.
- **Anything that can close a handle under a request** relies on the engine
  raising `ClosedError` (`HandleClosed`) BEFORE the call does anything —
  `ensure_open()` and `check_writable()` both — and on `_retry_closed` in
  `main.py` running the route again. A new engine entry point must check
  `closed_` first, with that exception type.
- **Routes that list namespaces** must not open them: `list_namespaces()` is
  every known one, `open_namespaces()` the open ones, `stats_of()` the counts
  of closed ones.

### File saved on close
`feather::DB::~DB()` calls `save()`. Call `db.save()` explicitly in long-running
processes, and `db.close()` when done with the file.

### Dangling edges in `export_graph_json`
If a record exists in the edge list but not in the metadata store (e.g., added without metadata), `export_graph_json` filters those dangling edges automatically via the `exported_ids` set.

---

## 9b. The contracts

`CONTRACTS.md` states what Feather promises — exact results by default; one
writer, many readers; read-only opens are consistent snapshots; one feature
set in every mode — and names the tests that enforce each. Read it before
changing defaults, the WAL, or the Cloud API. In particular:
- **A new public `DB` method needs a Cloud API route and a `RemoteDB` method**,
  mapped in `feather_db/remote.py` `PARITY` (or a stated reason it has none).
  `tests/test_parity.py` fails otherwise; `tests/test_contracts.py` runs the
  shared tests against both modes.
- **Every WAL file starts with its `WALGEN` record**, written with the header
  (`wal_open_for_append`); a rotation passes the generation being committed
  to `wal_rotate()`. Readers (`read_snapshot`, `chain_holds`) depend on it.
- **An approximate route is opt-in.** Never make one a default.

## 10. Adding New Features — Checklist

When adding a new feature to Feather DB, touch these files **in order**:

1. **`include/metadata.h`** — Add new field to `Metadata` struct
2. **`src/metadata.cpp`** — Update `serialize()` and `deserialize()`
3. **`include/feather.h`** — Add new method to `DB` class
4. **`src/feather_core.cpp`** — Add `extern "C"` wrapper for Rust/FFI
5. **`bindings/feather.cpp`** — Expose to Python via pybind11
6. **`feather_db/__init__.py`** — Export from Python package
7. **`feather-cli/src/lib.rs`** — Add CLI command in Rust
8. **`examples/`** — Add a usage example
9. **`CHANGELOG.md`** — Document the change
10. **`scripts/sync-cpp.sh`** — run it if you touched `include/` or `src/`; CI
    fails on drift between `feather-cli/cpp/` and the originals
11. **`feather-api/app/main.py` + `feather_db/remote.py`** — the route and the
    `RemoteDB` method, mapped in `PARITY` (CONTRACTS.md §4)
12. **`./scripts/verify.sh`** — the gate. Nothing ships red.

If the feature is agent-facing, it probably belongs in `feather_db/pocket.py`,
`integrations/agent_memory.py` or `integrations/mcp_agent.py` rather than on
`DB` — the engine stays generic and the memory model lives above it.

---

## 11. Known Issues & Limitations

| Issue | Details |
|-------|---------|
| ~~Two processes destroy each other's writes~~ | **Fixed in v0.20.0**: enforced single-writer / many-reader via `flock`. Still NOT concurrent multi-process *writing* — the second writer is refused, not queued. `db.close()` required to release. |
| Writes are serialized (reads are not) | `mutex_` is a `std::shared_mutex`: retrieval + const accessors take it **shared** (concurrent), mutations take it **exclusively** (one writer at a time, and no reader runs alongside). Salience (`recall_count`/`last_recalled_at`) is `mutable std::atomic` so a query can record its hit under the shared lock. |
| Soft deletes reclaimed on compaction | `forget()`/`purge()` mark vectors deleted; space is reclaimed by `compact()` or `set_auto_compact(ratio)` (Phase 7) |
| int8 quantization (two modes) | `set_quantized()` shrinks the file; `set_int8_ram()` shrinks RAM (~1.7×, opt-in) via `Int8L2Space`. Its scale is trained from the data, and `refine` (default 2) re-ranks by float originals kept in the file, so recall matches float. Each re-ranked candidate costs a `pread`: on a cold page cache that is a disk read. A trained scale is fixed once set; data that later drifts beyond the clip is clamped. |
| Versions and idempotency keys need v10 | `FEATHER_FORMAT=9` files carry neither: on reopen versions restart above a `time << 20` floor (no reuse) and keys are gone, so a retry is applied again. The Rust CLI exposes neither. |
| `tags_json` is a raw string | Tag filtering uses substring search, not JSON parsing |
| Superseded bytes stay until a vacuum (v10) | append-only: an updated or forgotten record's earlier versions remain in the file until `vacuum()`, `purge()`/`compact()` + save, or the automatic vacuum (`FEATHER_VACUUM_RATIO`). Vectors of forgotten records stay as deleted graph nodes until `compact()`. |
| Crash replay can leave a dangling edge | pre-existing (v9 too): `purge()` in memory counts records forgotten before the last save (their shells keep the namespace); the file has none, so after a crash the WAL's PURGE does not prune edges pointing at them. Fix needs the purged id list in the WAL record. |
| ~~Max 1M vectors per modality~~ | **Fixed in v0.15.3**: index capacity is adaptive (starts 4096, grows via `resizeIndex`); no hard cap |
| `meta.attributes['k'] = v` no-op | pybind11 map copy; use `set_attribute()` |
| Load time for large attribute DBs | v4/v5 attribute map deserialization is O(n * attrs); namespace/attribute *lookups* are O(matches) via secondary indexes (Phase 7) |
| Rust CLI missing v0.5.0 features | namespace/entity/context_chain are Python-only for now |
| Rust CLI now participates in locking | if a long-running Python process holds a DB, `feather add --db that.feather` is refused rather than silently discarding one side |
| `mcp_server.py` is broken on SDK 2.0 | `Server.list_tools` was removed; `create_server()` raises on import. Use `mcp_agent.py` (`feather-agent`) |
| BM25 is Porter-stemmed, not semantic | plurals and verb forms now match (`creative`/`creatives`, `running`/`run`, `policy`/`policies`), but Porter does not unify every derivational pair — `allergy` still does not find `allergic` (it stems to `allergi` vs `allerg`). A paraphrase with no shared root still returns nothing, indistinguishable from "never stored" — set `FEATHER_EMBED_PROVIDER` for anything relying on paraphrase. `FEATHER_STEM=0` restores exact matching. Tokens over 64 chars are left unstemmed — a blob or id is not a word, and folding it kept a superlinear term on the ingest and load paths |
| Namespace graphs are rebuilt after every open | not persisted: the first filtered search on a large namespace after a reopen queues a background build (~6 s at 50k × 384) and scans until it finishes |
| ~~BM25 cannot match a record by its KEY~~ | **Fixed in 0.22**: native keys (`Metadata.key`) are BM25-indexed with the content (`bm25_text()`) |
| `newest(key=...)` still scans | a custom key reads and sorts the namespace. Episodes no longer need one: they order by valid time (`by_valid=True`), except that a namespace holding episodes written before 0.22 (no `valid_at`) falls back to the key scan |
| `all()` ties within one second | the engine timestamp is whole seconds, so items written in the same second order arbitrarily. `_Episodes` overrides this using the event time in the key; sub-second ordering needs format v10's second timestamp |

---

## 12. Testing

```bash
# One command, the pre-release gate — build, suite, format scripts, the four
# data-loss regressions, MCP over the protocol, vendored C++ sync, version
# consistency, release prereqs, live API contract. Exits non-zero on anything.
./scripts/verify.sh            # or `./scripts/verify.sh quick`

pytest tests -q --deselect tests/test_engine.py::TestProviderInterface::test_provider_str
# 553 passed as of v0.21.0. The deselected test needs `openai` installed.

# The agent-memory suites specifically:
pytest tests/test_pocket.py tests/test_agent_memory.py \
       tests/test_langgraph_store.py tests/test_file_locking.py -q
# MCP over a real ClientSession (not direct calls) — must RUN, not skip:
pytest tests/test_mcp_protocol.py tests/test_mcp_personas.py -q

source repro_venv/bin/activate

python3 examples/langgraph_agent_memory.py   # memory added to an agent in 1 line
python3 examples/context_graph_demo.py
python3 examples/marketing_living_context.py
python3 examples/feather_inspector.py   # local inspector at http://localhost:7777

python3 benchmarks/stress_test.py

cd p-test && ./run_tests.sh   # Rust CLI tests
```

---

## 13. Phase Roadmap

| Phase | Status | Highlights |
|-------|--------|-----------|
| Phase 1 | Done | Basic HNSW, L2 search, `.feather` file format |
| Phase 2 | Done | Context Engine: metadata, types, time-decay scoring, filtered search |
| Phase 3 | Done | Multimodal pockets, graph links, adaptive decay |
| Phase 4a | Done | Generic namespace/entity/attributes (v0.4.0) |
| Phase 4b | Done | FastAPI + Docker cloud wrapper in `feather-api/` |
| Phase 5 | Done | Typed edges, reverse index, auto_link, context_chain, D3 visualizer (v0.5.0) |
| Phase 6 | Done | LLM connectors, MCP, LangChain/LlamaIndex, ContextEngine (v0.6–v0.9) |
| Cloud | Done | FastAPI admin SPA + pluggable embeddings (v0.10 Cloud Edition) |
| Phase 7 | Done | Secondary metadata indexes, pre-filtered ANN, auto-compaction (v0.11.0), on-disk int8 quantization / format v7 (v0.12.0) |
| Phase 8 | Done | Parallel load + `add_batch`, SIMD-on-x86 (v0.13.0), in-RAM int8 / format v8 (v0.15.0), Claude MCP connector + real embedders (v0.14–v0.15), adaptive index capacity (v0.15.3), persisted HNSW graph / format v9 (v0.16.0) |
| Phase 9 | Done | Agent memory: `Pocket` hot/warm/cache tiering, `FeatherStore` (LangGraph `BaseStore`), `AgentMemory`'s five kinds, `feather-agent` MCP server for SDK 2.0 tested over the real protocol, inter-process file locking (v0.19–v0.20) |
| Phase 10 | In progress | Format v10 **done**: append-only commits, two checksummed headers, typed length-prefixed sections, incremental saves, vacuum. Valid time **done** (0.22, `Metadata.valid_at`, indexed + filterable; a reserved attribute, no format change). Still planned: metric tag, typed attributes for real comparisons, sub-second timestamps. Plus multi-tenant auth, in-RAM int8 SIMD distance |
