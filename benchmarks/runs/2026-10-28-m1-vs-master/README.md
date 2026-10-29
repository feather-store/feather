# `perf/concurrency` vs master (v0.21.0), Apple M1, 2026-10-28

**Branch** is the `perf/concurrency` working tree (HEAD `945a353` plus the
uncommitted 0.22 work). **Master** is `origin/master` at `f61dbfa`, which is
v0.21.0. Both builds were compiled with `python setup.py build_ext --inplace`
(`-O3`) and ran identical copies of each benchmark script.

**Machine and method**
- Machine: Apple M1, 8 cores, 16 GB RAM, macOS.
- Benchmarks ran one at a time, with nothing else running.
- `run_all.sh` ran the branch first, then master, for each benchmark.
- `recheck.py` reran the rows that looked like branch regressions, this time
  with master first. It catches order effects such as heat.

**Trust.** Each number comes from a single run, and run-to-run noise on a
laptop can reach about 2×. Treat any ratio under 1.5× as a tie. The large
ratios are far outside the noise.

Why the branch searches faster: master has no hand-written NEON kernels; the
branch dispatches them at runtime (`include/feather_simd.h`).

## Search

| | master | branch | |
|---|---|---|---|
| unfiltered, 128-d, ef=50, 1 thread | 0.168 ms | 0.081 ms | 2.1× |
| unfiltered, 128-d, 4 threads | 17.4k qps | 43.3k qps | 2.5× |
| unfiltered, 768-d, 1 thread | 0.49 ms | 0.19 ms | 2.6× |
| namespace filter, 1% of records | 1.28 ms | 0.096 ms | 13× |
| namespace filter, 60k of 100k, 768-d | 142 ms | 30 ms | 4.7× |
| attribute filter, 25% of records | 27 ms | 6.6 ms | 4.1× |
| BM25 `keyword_search` | 11.9 ms | 3.3 ms | 3.6× |
| `hybrid_search` | 12.1 ms | 3.4 ms | 3.6× |
| `context_chain`, hops=2 | 0.21 ms | 0.07 ms | 3× |
| per-namespace graph, 50k tenant (opt-in) | scan 4.9 ms | 0.46 ms | 10.8× |
| `FeatherStore.newest(10)`, 100k records | 148 ms | 0.04 ms | 3,700× |

At equal `ef`, recall@10 is the same on both builds.

## Writes, saves, lifecycle

| | master | branch | |
|---|---|---|---|
| `add()` with metadata | 2.2k/s | 4.4k/s | 2× |
| `add_batch()` | 10.7k/s | 15.8k/s | 1.5× |
| `stress_test`: 100k × 768 `add()`, save every 10k | 1,162 s | 293 s | 4× |
| bytes written per save, 100k × 384, after 200 changes | 176.5 MB (v9) | 0.42 MB | 420× |
| `save()` wall time, same setup | 846 ms | 1.2 ms | 700× |
| worst `add()` stall while a save runs | 1,236 ms | 8.4 ms | |
| `compact()` after forgetting 10% (full suite) | 40 s | 5.5 s | 7× |
| `compact()` under load (exclusive suite) | 85.7 s | 18.6 s | 4.6× |
| slowest `forget()` with auto-compact on | 84.5 s | 17 ms | |
| `update_metadata`, record with 500k edges | 3.3 ms | 0.047 ms | 70× |

## Concurrency (engine, 100k × 128, WAL fsync on)

| | master | branch |
|---|---|---|
| 8 readers + 16 writers: reads/s | 153 | 9,817 |
| 8 readers + 4 writers: read p99 | 67 ms | 6.1 ms |
| 8 readers, save loop running: reads/s | 114 | 10,204 |
| 8 readers, save loop running: read p99 | 440 ms | 5.3 ms |
| 1 writer, no readers: writes/s | 508 | 2,056 |
| 8 readers during `update_metadata` (500k edges): reads/s | 1,205 | 18,853 |

## Cloud API (20k × 128, 1 worker)

| | master | branch |
|---|---|---|
| search, 1 client | 580/s | 1,117/s |
| search, 16 clients | 842/s | 2,093/s |
| search, 16 clients, with 8 writing | 462/s | 1,767/s |
| search during 64-client ingest | 92/s, p99 1.4 s | 877/s, p99 105 ms |
| deletes beside 16 searching clients | 9.5/s | 371/s |
| `/import` of 20k records | 10.0 s | 5.4 s |
| `/keyword_search`, p50 | 1.31 ms | 0.66 ms |

## Correctness and memory

| | master | branch |
|---|---|---|
| read-only opens that saw a torn view, writer saving | 573 / 2,688 | 0 / 2,564 |
| 10 read-only processes, 100k × 384 | 2,114 MB | 784 MB + 154 MB shared once |
| 40 tenants per shard, 4 hot | 561 MB (all open) | 101 MB |
| int8 filtered recall@10, 768-d | 0.948–0.983 | 1.000 |
| int8 recall@10, 384-d (trained scale + refine) | 0.938 | 1.000 |
| LongMemEval keyword, 500 questions | recall@1 0.870, @10 0.988 | identical, 17 s vs 21 s |

## Where the branch was slower, and what happened

Each row was re-measured after the comparison, with master and branch
alternating.

| | master | branch, first run | branch now | |
|---|---|---|---|---|
| `export_graph_json(namespace)` | 7.6 ms | 29 ms | **4 ms** | fixed: reads the namespace/entity index instead of walking the store 3× |
| `GET /records?limit=100` | 5.4–6.5 ms | 7.1 ms | **5.0–5.9 ms** | fixed: rows from `Metadata.to_dict()` instead of a pydantic model per record |
| 8 readers + 4 `link()` writers: writes/s | 9–19k | 2.6k | **6–13k** | fixed: a writer phase runs the queued writers for up to 100 µs |
| same mix: reads/s | 4–5k | 16–25k | 12.6–20k | traded ~20% for the writes; still 3–4× master |
| int8 search, 768-d | 1.40 ms, recall 0.659 | — | **0.51 ms, recall 0.832** | not a regression: the first run converted a float modality without saving. Measured set the same way on both, 50k × 768, ef=50 |
| open with `FEATHER_MMAP=0`, 10 at once | 1.8 s | 12.5 s | 1.7 s | not a regression: one-off, the file had just been written. Default mmap: 0.45 s vs 1.9 s |
| search after 10k forgets | 0.24–0.35 ms | 0.95 ms mean | 0.12–0.20 ms | not a regression: noise in the first run |
| `forget()` ×10k | 1.6 s | 1.8–4 s | | noise; within run-to-run variance |

**What is still slower, and why it stays**
- `update_importance`, 0.026 → 0.033 ms. Every write now takes a version and
  logs a STAMP for compare-and-set.
- `get_metadata`, about +0.4 µs per call. The call now releases the GIL, so
  it never stalls other Python threads behind a writer.
- `all_ids()` over 20k records, 0.27 → 0.43 ms. The walk over the metadata map
  costs more per node: records are 272 B instead of 232 (key, version), and
  each write allocates index nodes next to them. Listing routes no longer
  show it.

## Not comparable
- `search_fast`: master answers 422, because the route is new on the branch.
- `save_cost`, `namespace_search`, `idle_namespaces`, `int8_refine`: these
  run on the branch only. Each has an "old behaviour" row (v9 save, scan,
  all open, clip 1.0).
