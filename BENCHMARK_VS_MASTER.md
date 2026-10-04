# Benchmark: `perf/concurrency` vs GitHub master (v0.20.0)

A head-to-head run of the `perf/concurrency` branch against the code currently on GitHub master. It answers one question: which build is faster, at what, and by how much.

**Short answer.** The branch is faster on almost everything, and by large factors on concurrent writes, compaction, metadata updates and filtered search. Pure read throughput is a tie. Three things are slightly slower or unchanged, and they are listed in [section 5](#5-where-the-branch-is-not-faster).

---

## 1. What was compared, and how

| | Master | Branch |
|---|---|---|
| Code | `origin/master` at commit `9448dd8` | `perf/concurrency` working tree, synced onto that same commit |
| Version reported | 0.20.0 | 0.20.0 |
| Distance kernel | built in, no runtime dispatch | runtime dispatch, picked AVX2+FMA |

- **Date:** 2026-10-04.
- **Machine:** Intel i9-13900HX laptop, 32 logical cores, WSL2 Ubuntu, native Linux filesystem.
- **Method:** both builds compiled from source with the same command, then the same benchmark scripts run on each, one after the other so they never competed for CPU.
- **Scripts:** `benchmarks/concurrency_bench.py` and `benchmarks/search_simd_bench.py`, unchanged, copied into the master tree so both builds ran identical code.
- **Raw data:** [benchmarks/runs/2026-10-04-vs-master/](benchmarks/runs/2026-10-04-vs-master/).

**How much to trust the numbers.** Each scenario is a single run with a 5-second window. On this laptop, run-to-run noise is up to about 2x. Treat any ratio under about 1.5x as a tie. The large ratios are far outside the noise.

In every table, "Change" is how many times better the branch is. For throughput, higher is better. For latency and duration, lower is better.

---

## 2. Reads and writes together

100,000 records, 128 dimensions, fsync on (the default).

### Writers only

| Scenario | Master | Branch | Change |
|---|---|---|---|
| 1 writer, writes/s | 250 | 315 | 1.3x |
| 4 writers, writes/s | 234 | 879 | 3.8x |
| 16 writers, writes/s | 224 | 1,371 | 6.1x |
| 4 writers, write p99 | 169 ms | 6.8 ms | 25x |
| 16 writers, write p99 | 913 ms | 22 ms | 42x |

On master, adding writers does not add throughput: every writer pays its own fsync while holding the lock. On the branch, concurrent writers share one fsync (group commit), so throughput scales with writers.

### 8 readers plus writers

| Scenario | Master | Branch | Change |
|---|---|---|---|
| + 1 `add` writer, reads/s | 5,162 | 13,741 | 2.7x |
| + 1 `add` writer, read p99 | 5.0 ms | 1.9 ms | 2.6x |
| + 4 `add` writers, writes/s | 196 | 821 | 4.2x |
| + 4 `add` writers, write p99 | 236 ms | 8.7 ms | 27x |
| + 16 `add` writers, writes/s | 224 | 733 | 3.3x |
| + 16 `add` writers, write p99 | 2,001 ms | 37 ms | 54x |
| + 4 `link` writers, reads/s | 6,322 | 14,779 | 2.3x |
| + 4 `link` writers, writes/s | 265 | 971 | 3.7x |
| + 4 `link` writers, write p99 | 444 ms | 7.1 ms | 63x |
| + `add_batch(1000)` loop, reads/s | 150 | 1,636 | 11x |
| + `add_batch(1000)` loop, read p50 | 56 ms | 2.4 ms | 23x |

### Hybrid-search readers plus 4 writers

| Scenario | Master | Branch | Change |
|---|---|---|---|
| writes/s | 0.8 | 774 | 967x |
| write p99 | 5,012 ms | 9.6 ms | 523x |

This is the largest single fix. On master, writers are starved almost completely while hybrid searches run.

### Writes while `save()` runs in a loop (8 readers, 2 writers)

| Scenario | Master | Branch | Change |
|---|---|---|---|
| writes/s | 23 | 78 | 3.4x |
| write p99 | 1,981 ms | 347 ms | 5.7x |
| reads/s | 14,015 | 14,105 | tie |

---

## 3. Compaction and metadata updates

100,000 records, 128 dimensions, fsync off.

| Scenario | Master | Branch | Change |
|---|---|---|---|
| `compact()` duration | 40.5 s | 4.2 s | 9.6x |
| Longest read stall during compaction | 40.5 s | 0.52 s | 77x |
| Reads/s during compaction | 137 | 1,236 | 9.0x |
| The `forget()` that triggers auto-compaction | 35.5 s | 44 ms | 807x |
| Time to forget 20,000 records | 68.4 s | 6.0 s | 11x |
| `update_metadata` at 10,000 edges | 0.43 ms | 0.026 ms | 17x |
| `update_metadata` at 100,000 edges | 5.8 ms | 0.025 ms | 231x |
| `update_metadata` at 500,000 edges | 15.4 ms | 0.029 ms | 532x |
| Reads/s while `update_metadata` loops | 854 | 15,234 | 18x |
| `update_metadata` calls/s in that loop | 61 | 2,247 | 37x |

Master holds the exclusive lock for the whole compaction, so the database is frozen for 40 seconds. The branch builds the new index without holding the lock and only takes it briefly at the start and end.

One number looks worse for the branch and is not: read p99 during compaction is 2.2 ms on master and 19.6 ms on the branch. Master's figure only counts the few reads that completed outside the freeze. Its real worst case is the 40.5-second stall above.

On master, `update_metadata` cost grows with the number of edges in the graph. On the branch it is flat.

---

## 4. Search latency

100,000 records, 768 dimensions, single thread, k=10. Recall@10 is 1.0 on both builds in every filtered row.

| Scenario | Master | Branch | Change |
|---|---|---|---|
| Unfiltered, p50 | 0.264 ms | 0.146 ms | 1.8x |
| Unfiltered, p99 | 0.453 ms | 0.339 ms | 1.3x |
| Unfiltered, 8-thread queries/s | 14,006 | 16,923 | 1.2x |
| Filtered, namespace of 500, p50 | 1.55 ms | 0.16 ms | 9.6x |
| Filtered, namespace of 5,000, p50 | 19.7 ms | 2.0 ms | 9.8x |
| Filtered, namespace of 25,000, p50 | 93 ms | 0.94 ms | 100x |
| Filtered, namespace of 60,000, p50 | 123 ms | 1.1 ms | 113x |

- **Small namespaces** use an exact scan. The gain comes from reading vectors in place and copying metadata only for the top k, where master copies every candidate's vector and full metadata.
- **Large namespaces** use the filtered graph walk, chosen by a cost model.
- **Unfiltered search** gains from the AVX2+FMA kernel. The gain is modest because graph search spends most of its time waiting on memory, not computing distances.

---

## 5. Where the branch is not faster

| Scenario | Master | Branch | Result |
|---|---|---|---|
| Readers only, 1 thread, reads/s | 7,370 | 7,953 | tie |
| Readers only, 4 threads, reads/s | 19,724 | 19,432 | tie |
| Readers only, 8 threads, reads/s | 16,139 | 16,877 | tie |
| `add_batch(1000)` with 8 readers, batches/s | 15.4 | 10.1 | 0.7x |
| Hybrid readers + 4 writers, reads/s | 7,354 | 6,071 | 0.8x |
| Index build, 100k x 768 | 9.4 s | 11.6 s | within noise |

- **Batch inserts with readers running are slower.** Batches are now applied in chunks so readers get turns between chunks. Readers gain 11x in the same scenario.
- **Hybrid reads drop slightly when 4 writers run.** Master's higher read figure exists only because its writers got 0.8 writes/s. This is a deliberate fairness trade-off.
- **Search from many Python threads** stops scaling at about 4 threads on both builds, because of the GIL. `tests/test_concurrent_search_scales` fails on both for that reason.

---

## 6. Not measured in this run

- **The cloud API.** `benchmarks/api_concurrency_bench.py` was not rerun against v0.20.0. The earlier run against v0.18.2 showed about 3x on single-process search requests per second and about 12x on deletes. See [CONCURRENCY_RESULTS.md](CONCURRENCY_RESULTS.md) section 4.
- **768-dimension contention.** Only 128 dimensions were rerun for the read/write tables. The earlier 768-dimension results are in [CONCURRENCY_RESULTS.md](CONCURRENCY_RESULTS.md) section 2.
- **int8 storage.** Earlier results are in [CONCURRENCY_RESULTS.md](CONCURRENCY_RESULTS.md) section 8.
- **Windows, AVX-512 and ARM NEON.** This machine has no MSVC toolchain, no AVX-512 and no arm64 toolchain.
- **The sharding gateway under load.** It is functionally tested only.

---

## 7. Reproduce

Run on Linux or WSL from a native filesystem, not `/mnt/c`. Build each tree, copy the two benchmark scripts into the master tree, then run in each tree:

```bash
python setup.py build_ext --inplace

python benchmarks/concurrency_bench.py run --suite contention --sync 1 \
    --n 100000 --dim 128 --seconds 5 --out contention.json

python benchmarks/concurrency_bench.py run --suite exclusive --sync 0 \
    --n 100000 --dim 128 --seconds 5 --out exclusive.json

FEATHER_BENCH_PATH=$PWD python benchmarks/search_simd_bench.py \
    --n 100000 --dim 768 --out simd.json
```

Run the two builds one after the other, never in parallel.
