# Concurrency Fixes: Before vs After

> These are the measured results of implementing [CONCURRENCY_FIX_PLAN.md](CONCURRENCY_FIX_PLAN.md), Phases 0–6, on branch `perf/concurrency`.
>
> **Before:** [CONCURRENCY_BASELINE.md](CONCURRENCY_BASELINE.md), run on v0.18.2. **After:** the same two benchmark scripts with the same parameters, run on the same machine. Raw data is in [benchmarks/runs/2026-09-24-concurrency/](benchmarks/runs/2026-09-24-concurrency/), with the after runs in `after/`.
>
> **Test suite:** 346 passed. The only failure, `test_concurrent_search_scales`, was already failing on v0.18.2 (see [§5](#5-what-did-not-change-or-got-worse)). All 8 root feature scripts pass. The benchmark scripts are unchanged except for one extra API scenario (`search_fast_c16`).

**How to read these tables.** Absolute numbers vary up to about 2× between runs on this WSL2 laptop with mixed P-cores and E-cores. Readers-only throughput alone differed by that much between baseline runs. For each scenario, the most reliable comparison is the **drop relative to the same run's readers-only number**, plus the **write, stall and latency figures**, which changed by far more than the noise.

---

## 1. Headline

| Problem (from the baseline) | Before | After |
|---|---|---|
| **Writer starvation.** 768-d, 8 readers, 1 writer | **0.6 writes/s**, p50 1.3 s | **202 writes/s**, p50 4.7 ms |
| **Writer starvation.** Hybrid-search readers + 4 writers (128-d) | **0.5 writes/s** (blocked for the whole window) | **399 writes/s** |
| **Reads while 1 writer runs** (128-d, fsync on) | −68% reads, read p99 **8.4 ms** | −21% reads, read p99 **1.4 ms** |
| **Write throughput**, 16 writers (fsync on) | **260/s** (same as 1 writer), p99 656 ms | **2,267/s**, p99 13.6 ms |
| **Reads while `add_batch(1000)` loops** (128-d) | 148/s, read p50 **62 ms** | 1,986/s, read p50 **4.5 ms** |
| **Compaction** (100k×128, 10% dead) | DB frozen **39.9 s** | longest read **0.72 s**, read p99 12.5 ms, done in 6.8 s |
| **Compaction** (50k×768) | DB frozen **66.8 s** | longest read **0.63 s**, done in 8.6 s |
| **The `forget()` that triggers auto-compaction** | **35.0 s** (768-d: 56.7 s) | **43 ms** (768-d: 53 ms) |
| **`update_metadata`** at 500k edges | **13.2 ms** | **0.015 ms** (~880×) |
| **Reads while `update_metadata` loops** | −95% (997/s, p50 13 ms) | −12% (25.4k/s, p50 0.29 ms) |
| **Writes while `save()` loops** (128-d, fsync on) | 17/s, p99 **2.1 s** | 62/s, p99 **0.44 s** |
| **API search, 1 process, 16 clients** (128-d) | **488 req/s**, p99 54 ms | **1,678 req/s**, p99 14 ms |
| **API search while 64 ingests run on a 300 ms embedder** | **52 req/s**, p99 468 ms | **1,181 req/s**, p99 26 ms |
| **API deletes/s** (128-d) | **15/s**, p50 135 ms | **187/s**, p50 10.6 ms |

---

## 2. Engine: reads and writes together (8 readers)

### 100k × 128-d, fsync on (the default)

| Background writes | Reads/s before (Δ vs alone) | Reads/s after (Δ vs alone) | Read p99 before → after | Writes/s before → after | Write p99 / max before → after |
|---|---|---|---|---|---|
| none | 16,652 | 23,939 | 1.56 → 0.87 ms | — | — |
| 1 × `add` | 5,296 (−68%) | 18,826 (−21%) | 8.4 → 1.4 ms | 189 → 354 | 13 / 44 → 5.4 / 8.2 ms |
| 4 × `add` | 2,845 (−83%) | 11,411 (−52%) | 10.5 → 1.8 ms | 116 → 1,160 | 749 / 2,865 → 7.3 / 11.7 ms |
| 16 × `add` | 2,932 (−82%) | 8,932 (−63%) | 10.2 → 2.1 ms | 116 → 1,117 | 2,939 / 7,042 → 29 / 54 ms |
| 4 × `link` | 3,131 (−81%) | 20,958 (−12%) | 9.7 → 1.1 ms | 123 → 1,206 | 588 / 1,309 → 5.7 / 9.7 ms |
| 1 × `add_batch(1000)` | 148 (−99%) | 1,986 (−92%) | 117 → 9.5 ms | 14.5 → 10.8 batches | — |

Writes alone, by thread count: **255 / 254 / 260 writes/s → 475 / 1,551 / 2,267** for 1 / 4 / 16 threads. Write p99 went from 6.6 / 161 / 656 ms to 3.6 / 3.7 / 13.6 ms.

- **Before:** throughput stayed flat because every writer paid its own fsync while holding the lock.
- **After:** throughput scales with writers because concurrent writers share one fsync (group commit), and the fsync no longer blocks readers.

### 50k × 768-d, fsync on: writer starvation

| Background writes | Writes/s before (alone → with readers) | Writes/s after (alone → with readers) | Write p50 with readers, before → after | Reads/s before → after |
|---|---|---|---|---|
| 1 × `add` | 191 → **0.6** | 278 → **202** | 1,270 → 4.7 ms | 11,983 → 9,489 |
| 4 × `add` | 182 → **0.9** | 710 → **472** | 3,465 → 8.3 ms | 11,844 → 3,797 |
| 16 × `add` | 198 → **2.5** | 732 → **462** | 8,024 → 34 ms | 13,569 → 3,686 |
| 4 × `link` | **3.2** | **1,037** | 417 → 3.7 ms | 12,680 → 11,224 |
| 1 × `add_batch(1000)` | 4.7 batches/s | 4.3 batches/s | — | **44 → 1,031** |
| hybrid readers + 4 × `add` | **0.6** | **434** | 8,012 → 9.0 ms | 8,928 → 3,491 |

**This is a deliberate trade-off.** At 768-d the old lock looked good for reads only because writes almost never got in. Under the phase-fair lock, writers get their turn. Reads therefore now yield to writes, and under a heavy concurrent write load at 768-d reads drop. That is the intended fairness. To favour reads over write latency, use fewer writer threads or `add_batch`.

### Checkpoint (`save()`) running in a loop, with 2 writers and 8 readers

| | Reads/s · max, before → after | Writes/s · p99, before → after |
|---|---|---|
| 128-d fsync on | 14,633 · 147 ms → **18,873 · 6.6 ms** | 17 · 2,083 ms → **62 · 442 ms** |
| 128-d fsync off | 16,815 · 5.6 ms → **20,939 · 81 ms** | 47 · 1,665 ms → **65 · 696 ms** |
| 768-d fsync on | 10,867 · 35 ms → **13,587 · 12 ms** | 0.6 → **56** |

The snapshot is still written while holding the shared lock, so writers wait for that part (the p99 above). The slow part (fsyncing the new file, the rename, and deleting the rotated WAL) now happens after the lock is released.

An intermediate version of this work caused a regression here: with the plain fair lock, a writer queued behind a checkpoint made new readers wait for the whole save (reads fell to 56/s, p99 2.2 s). Checkpoints now take the lock in a *long-shared* mode that keeps admitting readers, and `tests/test_fair_lock.py` guards against this.

---

## 3. Engine: exclusive sections (fsync off)

| | 100k × 128 before | 100k × 128 after | 50k × 768 before | 50k × 768 after |
|---|---|---|---|---|
| `compact()` duration | 39.95 s | **6.8 s** | 85.6 s | **8.6 s** |
| Longest read during compaction | 39.9 s | **0.72 s** | 66.8 s | **0.63 s** |
| Read p99 during compaction | — (frozen) | 12.5 ms | — (frozen) | 35 ms |
| Longest write during compaction | 39.95 s | 0.72 s | 66.9 s | 0.63 s |
| The `forget()` that crosses the auto-compact threshold | 34,954 ms | **43 ms** | 56,708 ms | **53 ms** |
| `update_metadata` @ 10k / 100k / 500k edges | 0.41 / 3.55 / 13.19 ms | **0.015 / 0.015 / 0.015 ms** | — | 0.014 / 0.012 / 0.014 ms |
| Reads/s while 1 thread loops `update_metadata` (500k edges) | 997 (−95%) | **25,447 (−12%)** | 1,532 | 34,372 |

Compaction still has two short exclusive sections: the snapshot at the start and the catch-up/swap at the end. Those account for the remaining ~0.6–0.7 s longest read. The build in between runs on half the cores (`FEATHER_COMPACT_THREADS`) while queries continue.

---

## 4. Cloud API (1 process, 20k records)

| Scenario | 128-d before | 128-d after | 768-d before | 768-d after |
|---|---|---|---|---|
| 1 client | 337 req/s · p50 2.9 ms | **1,039 · 0.91 ms** | 249 · 4.0 ms | **682 · 1.4 ms** |
| 16 clients | 488 · p99 54 ms | **1,678 · p99 14 ms** | 417 · 62 ms | **1,267 · 20 ms** |
| 64 clients | 497 · p99 181 ms | **1,790 · p99 49 ms** | 415 · 207 ms | **1,294 · 64 ms** |
| 16 clients, `vector_b64` + `include_metadata=false` | — | **1,812 · p99 13 ms** | — | **1,488 · p99 19 ms** |
| 16 search + 8 `POST /vectors` | search 264 · writes 144/s | **1,061 · 482/s** | 205 · 113/s | **669 · 274/s** |
| 16 search + 64 `ingest_text`, 5 ms embedder | search **75** · p99 264 ms | **673 · 36 ms** | 54 · 389 ms | **425 · 60 ms** |
| 16 search + 64 `ingest_text`, 300 ms embedder | search **52** · p99 468 ms | **1,181 · 26 ms** | 48 · 673 ms | **675 · 50 ms** |
| Ingests/s during those runs (5 ms / 300 ms embedder) | 189 / 128 | **447 / 179** | 136 / 119 | **266 / 161** |
| 16 search + 2 `DELETE` clients | deletes **15/s** · p50 135 ms, search 271 | **187/s · 10.6 ms, search 1,421** | 9/s · 221 ms, 301 | **143/s · 13.7 ms, 1,006** |
| 16 search + 1 `/import` client | search 300 · p99 125, 4.3k rec/s | **770 · p99 75, 7.2k rec/s** | 276 · 328, 0.7k rec/s | **626 · 198, 1.3k rec/s** |

**Where the gains come from:**
- **Single-process search is about 3× faster.** Hits are serialized with orjson from dicts built in C++, the `response_model` validation pass is skipped, the metrics middleware is pure ASGI, and `raw_score` is computed in the engine. With `include_metadata=false`, a search moves only ids and scores.
- **Ingest no longer starves search.** `ingest_text` awaits its embedding without holding a thread, and engine writes run on a separate write pool. Search therefore keeps its thread pool: during slow-embedder ingest it now gets **1,181 req/s instead of 52**. The micro-batcher coalesced about 3–4 texts per provider call (e.g. 4,465 ingests in 1,207 calls).
- **Deletes are about 12× faster.** There is no full-file save per delete anymore; the WAL covers durability and the background checkpointer folds it in. Edge pruning goes through the reverse index instead of reading every record's metadata in Python.
- **Scale-out:** everything above is a single process. Beyond this, use shards behind `feather-gateway` ([docs/deploy-sharding.md](docs/deploy-sharding.md)). Throughput then multiplies by the shard count, since shards share nothing but the disk. The gateway was verified functionally (`tests/test_gateway.py`: routing, fan-out, 421 ownership) but was not load-tested in this run.

---

## 5. What did not change, or got worse

- **Reads under a heavy concurrent write load at 768-d** are lower than on v0.18.2, as §2 explains. That is the cost of writers no longer starving. `FEATHER_STD_SHARED_MUTEX` rebuilds the old reader-preferring lock for comparison; it is not recommended.
- **Search throughput from many Python threads** still peaks at about 4 threads for cheap 128-d queries. The GIL-held parts of each call (argument conversion, result objects) dominate. `test_concurrent_search_scales` fails exactly as it did on v0.18.2. Use processes (shards) for more throughput.
- **`add_batch` still costs readers** while a bulk load runs (−92% at 128-d with a writer looping back-to-back 1,000-record batches). Each exclusive section is now one small chunk instead of the whole batch, so the stall per read dropped from ~62 ms to ~4.5 ms. Throughput is still shared.
- **Group commit changes visibility.** A reader can see a record a few ms before its fsync completes. The writer is still acknowledged only after it is durable. Set `FEATHER_WAL_STRICT=1` if you need the old behaviour.
- **Not validated here:**
  - The Windows code paths (`LockFileEx`, `MoveFileEx`, `FlushFileBuffers`). There is no MSVC toolchain on this machine; they compile only on Windows.
  - The Rust CLI build with `cargo`: no Rust toolchain. Its vendored C++ was compiled and its C ABI was exercised with a C harness instead: open, add, search, save, close.

---

## 7. Search path: runtime SIMD dispatch and a faster filtered search

These are measured with [benchmarks/search_simd_bench.py](benchmarks/search_simd_bench.py): clustered data, single-thread p50, k=10, `record_salience=False`. Timing and recall scoring are done in separate passes, because computing ground truth between timed searches evicts the caches and distorts the result. Three builds were measured on the same machine:
- **old:** untouched v0.18.2, built the way the PyPI wheel is (`FEATHER_SIMD=sse`)
- **new:** the default runtime-dispatch build, which picked **avx2+fma** here
- **new-sse:** the new build forced to the SSE2 kernel, which separates the kernel effect from the filtered-path rewrite

Raw data is in [benchmarks/runs/2026-09-24-concurrency/simd/](benchmarks/runs/2026-09-24-concurrency/simd/).

### Filtered search (namespace filter, 100k records; 80k at 1536-d)

| Namespace size | 128-d old → new | 768-d old → new | 1536-d old → new | Route (new) |
|---|---|---|---|---|
| 500 | 0.66 → **0.075 ms** (8.8×) | 1.53 → **0.24 ms** (6.3×) | 2.86 → **0.44 ms** (6.5×) | exact scan |
| 5,000 | 10.0 → **1.18 ms** (8.5×) | 20.3 → **3.07 ms** (6.6×) | 32.0 → **6.0–8.1 ms** (~4–5×) | exact scan |
| 25,000 | 53.0 → **1.51 ms** (35×) | 98.9 → **1.76 ms** (56×) | 154.7 → **2.12 ms** (73×) | filtered HNSW |
| 60,000 | 141.7 → **2.09 ms** (68×) | 227.5 → **2.77 ms** (82×) | 295.3 → **2.28 ms** (130×) | filtered HNSW |

Recall@10 is 1.0 in every cell except 128-d with a 60k namespace (0.995, on the graph route).

**Where the gain comes from:**
- **Small sets (the exact scan).** Most of the gain is not SIMD width.
  - The old path copied every candidate's vector and full `Metadata` (content, attributes, edges) before sorting.
  - It also rebuilt the candidate set as a hash set.
  - The new path reads the stored vector in place, ranks on `(score, id)`, and copies metadata for the top k only.
- **Large sets.** These now take the filtered graph walk, with `ef` raised for that one call.
- **Choosing the route.** This first shipped as a fixed-size threshold, which a sweep showed to be wrong: at 2% selectivity a walk costs 45 ms against a 0.55 ms scan. The route is now a cost model, walking when `candidates × selectivity² > 10 × k`.

### Unfiltered search

| | 128-d | 768-d | 1536-d |
|---|---|---|---|
| 1-thread p50, old → new | 0.094 → **0.074 ms** (1.27×) | 0.250 → **0.206 ms** (1.22×) | 0.411 → **0.337 ms** (1.22×) |
| 8-thread QPS, old → new | 15.4k → 16.2k | 14.6k → 12.9k | 9.3k → 9.1k |

This is a smaller gain than the 1.5–2× I originally predicted.
- **The kernels themselves are much faster.** Measured in cache with a C++ harness, AVX2+FMA takes 5.6 / 35 / 75 ns per distance at 128 / 768 / 1536-d, against 30.9 / 261 / 553 ns for scalar and 9.6 / 47 / 89 ns for SSE2.
- **But graph search spends most of its time waiting on memory.** Each hop fetches a vector that isn't in cache. So a faster kernel only moves the end-to-end number by about 20%.
- **The 8-thread numbers** are dominated by GIL hand-offs and memory, and vary between runs.
- **Bigger unfiltered gains need something else:** smaller vectors (in-RAM int8 now has an AVX2 kernel, or binary quantization) and less Python per call (a batch search API).

**What pip users get.** The same wheel now uses AVX2+FMA or AVX-512 when the CPU has them, and falls back to SSE2 or scalar otherwise. arm64 wheels get NEON distance kernels, where before there was no SIMD at all. `feather_db.core.simd_info()` shows the choice.

**Not validated on this machine:**
- The **AVX-512** kernel: this i9-13900HX has no AVX-512, so it was only compile-checked here. `tests/test_simd.py` covers it on CI runners that have AVX-512.
- The **NEON** kernels: there's no arm64 toolchain here. They're covered by the macOS arm64 CI and wheel legs.

---

## 8. Read-path locks, int8 SIMD, msgspec parsing

These three changes were measured individually. Only one of them paid off.

### Read-path lock fast path + lock-free visited lists: no measurable gain

**What changed:**
- **`FairSharedMutex`** now has an atomic reader fast path. With no writer waiting, `lock_shared` and `unlock_shared` are one CAS and one `fetch_sub` on a single word, and the internal mutex is only used once a writer is involved.
- **HNSW's `VisitedListPool`** gives each thread a slot that it swaps in and out with one atomic exchange. Upstream took a mutex twice per search.

**Validation:** both are covered by [tests/cpp/lock_stress.cpp](tests/cpp/lock_stress.cpp). It passes at `-O3` and under ThreadSanitizer: 11.3M reads and 10.9k writes in 3 s, up to 11 concurrent readers, no torn reads, no lost updates, and no visited list handed to two threads.

**Measurement:** `DB::search` called from C++ threads, with no GIL, in three builds of the same harness. "Previous" is the mutex-based fair lock plus the upstream pool.

| 128-d, qps | 1 thread | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| previous | 16,976 | 53,310 | 87,470 | 108,968 | 143,605 |
| **current** | 15,957 | 54,363 | 92,345 | 110,528 | 141,070 |
| glibc `std::shared_mutex` | 18,058 | 60,587 | 105,288 | 120,803 | 153,607 |

At 768-d with 32 threads, three alternating repeats of each build ranged from **19k to 31k qps with no consistent winner**. The differences are within run-to-run noise.

**Why there's no gain:** a search costs about 60 µs of work and the locks cost tens of nanoseconds. Scaling is limited by memory bandwidth and this hybrid-core CPU, not by locks. The predicted +20–40% did not happen. The change is kept because it is correct, tested, and removes per-read mutex traffic that could matter on larger servers. It can be reverted without loss here.

### int8 SIMD: int8 is now faster than float, not just smaller

**What changed:**
- **The int8×int8 AVX2 kernel** now processes 32 bytes per step. It computes the absolute difference exactly as uint8 via bias plus saturating subtracts, then squares with madd. It returns identical results to the scalar kernel and is up to 1.45× faster than the 16-byte version.
- **A new float-query × int8-row kernel** (AVX2 and NEON) replaces the scalar loop in the int8 filtered scan.

| In cache, ns per distance (AVX2) | 128-d | 768-d | 1536-d |
|---|---|---|---|
| float | 6.0 | 43.3 | 89.9 |
| int8 | 5.6 | **28.8** | **48.4** |
| float × int8 row (scalar → AVX2) | 71.5 → 9.6 | 413 → 43 | 792 → 106 |

End to end (100k records at 768-d, 80k at 1536-d, [benchmarks/search_simd_bench.py](benchmarks/search_simd_bench.py) `--int8`):

| | 768-d | 1536-d |
|---|---|---|
| int8 unfiltered p50: old wheel build → new | 0.183 → **0.104 ms** (1.76×) | 0.215 → **0.116 ms** (1.85×) |
| new int8 vs new float, unfiltered p50 | **0.104 vs 0.149 ms** (1.43× faster) | **0.116 vs 0.211 ms** (1.82× faster) |
| 8-thread qps, int8 vs float | 21.0k vs 17.9k | **25.0k vs 10.8k** |
| int8 filtered, 60k namespace: old → new | 109.6 → **1.2 ms** | 134.8 → **0.92 ms** |

Recall@10 for int8 is 0.93–0.985 on both the old and new builds. That is quantization loss, not a regression.

### msgspec request parsing: small gain, mostly on bulk import

**What changed:** search, keyword, hybrid, context_chain, `/vectors` and `/import` decode their bodies with msgspec Structs ([feather-api/app/fastparse.py](feather-api/app/fastparse.py)). Metadata stays validated by the strict pydantic `MetadataIn`, and the pydantic models stay the documented OpenAPI schemas.

| API, 1 process | pydantic (§4) | msgspec |
|---|---|---|
| 128-d search: 1 / 16 / 64 clients (req/s) | 1,039 / 1,678 / 1,790 | 966 / 1,757 / 1,809 |
| 768-d search: 1 / 16 / 64 clients | 682 / 1,267 / 1,294 | 729 / 1,361 / 1,316 |
| `/import` of 1,000 records, 128-d (batches/s) | 7.2 | 7.9 (+10%) |
| `/import` of 1,000 records, 768-d, p50 | 841 ms | **683 ms** (−19%) |

**Search moved by −7% to +7%, which is noise.** Pydantic v2 already validates in Rust, so body parsing was not the dominant per-request cost. The predicted 5–10× parsing win does not show up end to end. What remains per request is the HTTP stack itself: uvicorn/Starlette plus the thread-pool hop, about 0.5–1 ms per request on one core. The lever there is more processes (shards, §4), not a faster parser.

**Recommendation:** keep msgspec on `/import`, where bodies are large and the gain is real. **Consider reverting it on the search routes:** it duplicates the request schema, which risks drift, for no measurable benefit.

---

## 6. Reproduce

```bash
# Linux / WSL, from feather/ (not /mnt/c)
python setup.py build_ext --inplace
python benchmarks/concurrency_bench.py orchestrate --n 100000 --dim 128 --seconds 8 --out runs
python benchmarks/concurrency_bench.py orchestrate --n 50000  --dim 768 --seconds 8 --out runs
python benchmarks/api_concurrency_bench.py --n 20000 --dim 128 --out runs/api_128.json
python benchmarks/api_concurrency_bench.py --n 20000 --dim 768 --out runs/api_768.json

# §7 search path: same script against any build; cap the kernel for A/B
python benchmarks/search_simd_bench.py --n 100000 --dim 768 --out runs/simd_768.json
FEATHER_SIMD_RUNTIME=sse python benchmarks/search_simd_bench.py --n 100000 --dim 768 --out runs/simd_768_sse.json
FEATHER_BENCH_PATH=/path/to/old/build python benchmarks/search_simd_bench.py --n 100000 --dim 768 --out runs/old_768.json

# A/B the lock: rebuild with the old reader-preferring std::shared_mutex
python setup.py build_ext --inplace -D FEATHER_STD_SHARED_MUTEX
```
