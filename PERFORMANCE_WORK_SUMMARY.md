# Feather DB Concurrency & Performance Work: The Full Story

> **What this is:** a start-to-finish account of the concurrency and performance work on Feather DB. It covers what we set out to understand, what turned out to be wrong, how each problem was fixed, and what the measurements say. Where a fix did not help, this document says so.
>
> **Where the work lives:** branch `perf/concurrency` (not committed yet), based on v0.18.2.
>
> **Deeper detail:**
> - [HOW_IT_WORKS.md](HOW_IT_WORKS.md): how the system works end to end
> - [CONCURRENCY.md](CONCURRENCY.md): the original problem analysis
> - [CONCURRENCY_FIX_PLAN.md](CONCURRENCY_FIX_PLAN.md): the fix plan
> - [CONCURRENCY_BASELINE.md](CONCURRENCY_BASELINE.md): the "before" numbers
> - [CONCURRENCY_RESULTS.md](CONCURRENCY_RESULTS.md): the "after" numbers
>
> **Raw benchmark data:** [benchmarks/runs/2026-09-24-concurrency/](benchmarks/runs/2026-09-24-concurrency/)

---

## Contents

1. [Where we started](#1-where-we-started)
2. [The question: does it handle high concurrency?](#2-the-question-does-it-handle-high-concurrency)
3. [Measuring before fixing](#3-measuring-before-fixing)
4. [What the benchmark revealed](#4-what-the-benchmark-revealed)
5. [The fixes](#5-the-fixes)
6. [Results: before vs after](#6-results-before-vs-after)
7. [Second round: search speed](#7-second-round-search-speed)
8. [What did not work, or was over-predicted](#8-what-did-not-work-or-was-over-predicted)
9. [Behaviour changes to know before merging](#9-behaviour-changes-to-know-before-merging)
10. [What is still open](#10-what-is-still-open)
11. [How to verify all of this yourself](#11-how-to-verify-all-of-this-yourself)

---

## 1. Where we started

Feather DB is an embedded vector database, like SQLite for vectors. Its layers are:

- **Engine:** a C++ core with HNSW search, a metadata store, a BM25 keyword index and a context graph.
- **Storage:** a write-ahead log (WAL) plus one `.feather` file per database.
- **Python layer:** LLM-driven ingestion (facts, entities, contradictions) and agent tools, including the MCP server.
- **Cloud API:** a FastAPI server with one `.feather` file per namespace.

The first step was reading the whole codebase and writing it up in [HOW_IT_WORKS.md](HOW_IT_WORKS.md). That pass also turned up a list of correctness bugs, most of which were fixed later:
- `feather-serve` could not start.
- The LlamaIndex adapters were never exported.
- Pipeline records were overwritten after a restart.
- The API silently padded or truncated embeddings.

---

## 2. The question: does it handle high concurrency?

The short answer from reading the code: **reads scaled, writes did not, and the API could not scale out.** We identified five suspected problems ([CONCURRENCY.md](CONCURRENCY.md)):

| # | Suspected problem | Where |
|---|---|---|
| 1 | Every write held the database's exclusive lock **while waiting for the disk flush (fsync)**, so all searches stopped during every write. | `include/feather.h`, `add()` and other mutations |
| 2 | Compaction rebuilt every index while holding the exclusive lock, freezing the database. | `compact()` |
| 3 | `save()` rewrote the whole file while writers waited, and the API saved after every delete. | `save()`, `feather-api/app/main.py` |
| 4 | `update_metadata` scanned the entire graph on every call. | `update_metadata()` |
| 5 | The API was one Python process with no protection against a second process opening the same file, which would corrupt it. | `feather-api/Dockerfile`, `db_manager.py` |

There were also two API-level issues: Python overhead around a very fast engine call, and slow embedding calls tying up the thread pool.

**Rust or C++?** A Rust rewrite would not fix these: the engine is already native C++. The problems were in locking design and process architecture, not the language.

---

## 3. Measuring before fixing

Rather than fix from theory, we wrote two benchmarks and measured first.

- **[benchmarks/concurrency_bench.py](benchmarks/concurrency_bench.py)** measures the engine: how much searches slow down while writes, batch loads, saves, compactions or metadata updates run on the same database. It runs with fsync on and off.
- **[benchmarks/api_concurrency_bench.py](benchmarks/api_concurrency_bench.py)** measures the real API server under mixed traffic: searches alongside writes, ingests (against a fake embedding server with an adjustable delay), deletes and bulk imports.

**Environment:** WSL2 Ubuntu on an i9-13900HX, because Windows had no compiler for the extension. The numbers vary up to about 2× between runs on this laptop, so comparisons are made within a run. The full baseline is in [CONCURRENCY_BASELINE.md](CONCURRENCY_BASELINE.md).

---

## 4. What the benchmark revealed

The baseline confirmed every suspected problem. It also found three we hadn't predicted:

| Finding | Measured (before) | Predicted? |
|---|---|---|
| **Writers starved by readers.** Linux's `std::shared_mutex` prefers readers, so under steady reads a writer never gets in. | 8 readers at 768-d cut one writer from **191 to 0.6 writes/s**. With hybrid-search readers, writes were blocked for the whole test. | **No, new** |
| Writes hold the lock through the fsync | 1 writer cut searches by **68%**; 4 writers cut them by **83%** | Yes |
| Adding writers doesn't add throughput | 1, 4 or 16 writers all gave **~255 writes/s**, with waits up to 7 s | Partly |
| **`add_batch` froze readers** while building its graph | Searches fell **99%** during bulk loads | **No, new** |
| Compaction froze everything | **40 s** (100k×128), **86 s** (50k×768) | Yes |
| The `forget()` that triggered auto-compaction | Blocked its caller for **35–57 s** | Yes |
| `update_metadata` cost grew with the graph | 0.4 → 13.2 ms as edges went 10k → 500k | Yes |
| API capped at one core | **~490 searches/s**, whatever the number of clients | Yes |
| **Any ingest traffic starved API search, even with a fast embedder** | Search dropped **85%** with a 5 ms embedder | **Partly new** |
| API deletes | **15 deletes/s** | Yes |

---

## 5. The fixes

Each fix below lists the problem, the change, and where it lives.

### 5.1 Safety first

**`save()` was not actually durable.**
- **Problem:** the new file was renamed into place without being flushed to disk, and then the WAL was deleted. A power loss could leave an empty file and no WAL. A full disk was never detected, so a truncated file could replace the good one.
- **Fix:** fsync the new file, atomically replace the old one, fsync the directory, and only then remove the WAL. Every write error is checked. *(`include/feather.h`: `write_snapshot`, `publish_snapshot`)*

**Two processes could open the same file.**
- **Problem:** two owners would both replay and append to the same WAL and overwrite each other's saves.
- **Fix:** `DB.open()` takes an exclusive OS lock on `<path>.lock` (`flock` on Linux, `LockFileEx` on Windows). A second open anywhere, including in the same process, raises an error. We added `db.close()` and `with DB.open(...) as db:` to release it. The OS drops the lock if the process crashes. *(`include/feather.h`, `bindings/feather.cpp`, and ~20 tests and scripts migrated)*

### 5.2 Readers and writers stop blocking each other

**Writer starvation.**
- **Fix:** a new `FairSharedMutex` that is **phase-fair**. Once a writer waits, new readers queue behind it. When the writer finishes, every reader that queued goes before the next writer, so neither side can starve.
- **Follow-up:** a "long reader" mode for checkpoints lets queries keep flowing while a save holds the lock. An intermediate version without it regressed reads to 56/s during saves; that was caught by the benchmark and fixed.

**fsync under the lock.**
- **Fix: group commit.** Writes append to the WAL inside the lock, so WAL order still matches memory order. The fsync happens *after* the lock is released, and one fsync covers every writer queued behind it. Readers no longer wait for the disk.
- **Trade-off:** a reader can see a record a few milliseconds before it is durable. The writer is still acknowledged only after the flush. `FEATHER_WAL_STRICT=1` restores the old behaviour.

**`add_batch` froze readers.**
- **Fix:** batches are applied in small chunks, releasing the lock between them.

**Compaction froze the database.**
- **Fix:** compaction takes a quick snapshot, builds new indexes **without holding the lock** while reads and writes continue (writes are logged), then briefly locks to replay the log and swap indexes in. Auto-compaction runs on a background thread, with `wait_for_compaction()` to wait for it.
- **Also:** HNSW now **reuses the slots of deleted vectors**, so tombstones stop piling up. The vendored hnswlib was patched so re-adding a forgotten id works.

**Saves blocked writers.**
- **Fix:** `save()` writes its snapshot, rotates the WAL to `<wal>.old`, and releases the lock. The slow part (flush, rename, delete the old WAL) happens outside it. Recovery replays `<wal>.old` then `<wal>`.

**`update_metadata` scanned everything.**
- **Fix:** it now touches only the record's own edges.

**Smaller engine fixes:**
- `purge` and `auto_link` are now in the WAL.
- A wrong-dimension `add()` is rejected before it's logged.
- `add()` with edges keeps the reverse edge index correct.
- Strided NumPy arrays are read correctly.

### 5.3 The Cloud API

| Problem | Fix |
|---|---|
| Every single delete rewrote the whole file | Mutations no longer save. A **background checkpointer** saves when the WAL grows past 64 MB or 5 minutes. |
| Deleting scanned every record in Python, and missed non-`text` records | Edge pruning uses the engine's reverse index (O(in-degree)) |
| Writes and ingests filled the thread pool that searches use | Writes run on a **separate write pool** |
| Ingest held a thread while waiting for the embedding provider | `ingest_text` is **async**, with a micro-batcher that merges concurrent ingests into one provider call, pooled HTTP connections, and a cache |
| Embeddings silently padded or truncated to 768 dims | Never resized. OpenAI and Gemini are asked for the right size natively; any other mismatch returns 400. |
| Response building was slow | Results are serialized straight from C++-built dicts with orjson. `raw_score` cosine is computed in C++. `include_metadata=false` and binary `vector_b64` input are new options. |
| One process could use only one core | **Sharding:** run N API processes (`FEATHER_SHARD_COUNT`/`FEATHER_SHARD_INDEX`) behind the new **`feather-gateway`**. A shard answers 421 for namespaces it doesn't own. |

### 5.4 Python-layer bugs fixed along the way

- **`feather-serve` (MCP server) could not start.** It called the MCP SDK's stdio server incorrectly.
- **`IngestPipeline` overwrote records after a restart.** Its id counters restarted at 1; they now continue after the highest stored id and re-learn stored entities.
- **Internal lookups inflated recall counts,** skewing the "living context" decay. Contradiction checks, MMR over-fetch and tool over-fetch now count only what they return.
- **The LlamaIndex adapters were never exported.** An import typo was silently swallowed.

---

## 6. Results: before vs after

The same benchmarks were run on the same machine. Full tables are in [CONCURRENCY_RESULTS.md](CONCURRENCY_RESULTS.md).

| Problem | Before | After |
|---|---|---|
| Writer starvation (768-d, 8 readers, 1 writer) | **0.6 writes/s** | **202 writes/s** |
| Writer starvation (hybrid-search readers + 4 writers) | **0.5 writes/s** | **399 writes/s** |
| Searches slowed by 1 writer (128-d) | −68%, p99 8.4 ms | −21%, p99 1.4 ms |
| Write throughput, 16 writers | 260/s | **2,267/s** |
| Searches during bulk load | p50 **62 ms** | p50 **4.5 ms** |
| Compaction freeze | **40–67 s** | longest wait **0.6–0.7 s** |
| `forget()` triggering auto-compaction | **35–57 s** | **43–53 ms** |
| `update_metadata` at 500k edges | 13.2 ms | **0.015 ms** |
| API searches/s (1 process) | 488 | **1,678** |
| API search during slow-embedder ingest | **52/s** | **1,181/s** |
| API deletes/s | 15 | **187** |

**One deliberate trade-off:** at 768-d under heavy concurrent writing, searches are now slower than before. Previously they only *looked* fast because writes were locked out. With the fair lock, writers get their turn.

---

## 7. Second round: search speed

After the concurrency work, we targeted raw search speed.

### 7.1 SIMD distance kernels chosen at runtime: worked, modestly

- **Problem:** PyPI wheels were built SSE-only, so pip users never got the AVX kernels. ARM (Apple Silicon, Graviton) had no SIMD distance code at all.
- **Fix:** a new [include/feather_simd.h](include/feather_simd.h). Each fast kernel (AVX-512, AVX2+FMA, SSE2, NEON) is compiled for its own instruction set, and the best one is chosen at startup. One wheel runs on every CPU. `feather_db.core.simd_info()` shows the choice.
- **Result:** the kernels are much faster in isolation (AVX2 is 5.5× faster than scalar), but unfiltered search is only **~1.2× faster** end to end. Graph search mostly waits on memory, not arithmetic.

### 7.2 Filtered search: worked, dramatically

- **Problem:** filtered searches (by namespace, entity or attribute) copied every candidate's vector *and full metadata*, then computed distances with a scalar loop.
- **Fix:**
  - The exact scan uses the SIMD kernel directly on the stored vector, with no copying.
  - Only the top k results get their metadata copied.
  - Large, broad filters use the HNSW graph with a wider search for that one query.
- **A mistake along the way:** the first rule for choosing the graph route used candidate *count*. A sweep showed that *selectivity* is what matters: at 2% selectivity the graph took 45 ms against a 0.55 ms scan. The rule became a cost comparison, `candidates × selectivity² > 10 × k`.

| Filtered search (vs old wheel build) | 128-d | 768-d | 1536-d |
|---|---|---|---|
| 500-record namespace | 8.8× | 6.3× | 6.5× |
| 60,000-record namespace | 68× (141 → 2.1 ms) | 82× (227 → 2.8 ms) | 130× (295 → 2.3 ms) |

Recall stayed at 1.0 (one cell at 0.995).

### 7.3 int8 SIMD: worked

- **Problem:** int8 vectors were 4× smaller than float but not faster.
- **Fix:**
  - A faster int8 AVX2 kernel processes 32 bytes per step, with exact results.
  - A new float-query × int8-row kernel speeds up the int8 filtered scan.
- **Result:** int8 search is now **1.43× faster than float at 768-d and 1.82× at 1536-d**, as well as 4× smaller. int8 filtered search went from 110–135 ms to about 1 ms.

---

## 8. What did not work, or was over-predicted

These predictions from this work turned out wrong or overstated. The measurements are in [CONCURRENCY_RESULTS.md](CONCURRENCY_RESULTS.md) §7–§8.

| Prediction | Reality | Why |
|---|---|---|
| AVX2 would make search 1.5–2× faster | **~1.2×** end to end | Graph search is limited by memory latency, not arithmetic |
| A lock-free read path and visited lists: +20–40% at 16+ threads | **No measurable change** (noise between repeats: 19k–31k qps for every build) | A search is ~60 µs of work; the locks cost nanoseconds. Memory bandwidth is the limit. |
| msgspec parsing: 5–10× faster requests | Search **within noise** (−7% to +7%); bulk import **+10–20%** | Pydantic v2 already validates in Rust. The remaining per-request cost is the HTTP stack. |
| A size threshold for the filtered-search route | Wrong: it made some searches 6–20× slower | Selectivity, not size, drives the graph route's cost. Replaced with a cost model. |
| First filtered-search benchmark numbers | Distorted | The benchmark computed ground truth between timed queries, which flushed the CPU cache. Fixed and re-run. |

The lock and visited-list changes were kept because they are correct and ThreadSanitizer-clean. The recommendation for msgspec is to **keep it on `/import` only and take it back off the search routes**, where it duplicates the schema for no gain.

---

## 9. Behaviour changes to know before merging

1. **One writer process per file.** This lock shipped upstream in 0.20.0 and this branch now uses it: a second *process* opening a `.feather` for writing is refused, `read_only=True` opens share the file, and a second open in the same process is admitted. Use `db.close()` or a `with` block; `del db` does *not* release it. `FEATHER_LOCK=0` disables the lock.
2. **Group commit visibility.** Readers can see a write a few ms before it's durable. The writer only returns after it is. `FEATHER_WAL_STRICT=1` restores the old behaviour.
3. **Auto-compaction is background.** Use `wait_for_compaction()` to wait for it.
4. **`add_batch` is chunked.** Readers can see a partially applied batch.
5. **The API rejects embedding-dimension mismatches** with a 400 instead of padding or truncating.
6. **The API no longer saves on each mutation.** Durability comes from the WAL, and a background checkpointer folds it into the file.
7. **New WAL operation `PURGE`.** Builds older than this change skip it on replay.
8. **`search()` rejects wrong-dimension queries** instead of reading past the end of its buffer.

---

## 10. What is still open

- **Not validated on this machine:**
  - AVX-512 (this CPU has none) and ARM NEON (no toolchain). Both are compile-only here, and CI covers them on capable runners.
  - The Windows file-lock and fsync code (no MSVC compiler).
  - A Rust CLI build with `cargo`. Its C++ and C interface were tested directly.
- **Not load-tested:** the sharding gateway. It is functionally tested only.
- **Deferred:**
  - Idle-namespace eviction, which needs reference counting to be safe.
  - Binary vectors in `mcp_remote`, which older servers would reject.
  - A Windows CI job.
- **Still failing:** `test_concurrent_search_scales`. It failed before this work and still fails: Python threads can't scale cheap searches past ~4 threads because of the GIL.
- **Next biggest wins, if wanted:**
  - A **batch search API**: many queries per call, with no per-query Python overhead.
  - **Hybrid search without metadata copies** for discarded candidates.
  - **Binary quantization** for 768-d and larger embeddings.
  - **Load-testing the gateway** to confirm multi-process scaling.

---

## 11. How to verify all of this yourself

```bash
# Linux / WSL, from feather/ (not /mnt/c)
python setup.py build_ext --inplace
pytest tests -q                                  # 366 passed; test_concurrent_search_scales fails as it did on v0.18.2
for t in test_secondary_index test_prefiltered_search test_auto_compact test_quantization \
         test_batch_ingest test_persist_graph test_int8_ram test_parallel_load; do python $t.py; done

# lock + visited-list stress test, also under ThreadSanitizer
g++ -std=c++17 -O1 -g -fsanitize=thread -Iinclude tests/cpp/lock_stress.cpp -o /tmp/ls -lpthread && /tmp/ls

# the benchmarks behind every number above
python benchmarks/concurrency_bench.py orchestrate --n 100000 --dim 128 --seconds 8 --out runs
python benchmarks/api_concurrency_bench.py --n 20000 --dim 128 --out runs/api_128.json
python benchmarks/search_simd_bench.py --n 100000 --dim 768 --out runs/simd_768.json
python benchmarks/search_simd_bench.py --int8 --n 100000 --dim 768 --out runs/int8_768.json
```
