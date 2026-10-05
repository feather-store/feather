# Feather DB — Full End-to-End Benchmark Report

**Version:** feather-db 0.18.2 (file format v9) · **Run date:** 2026-09-24 · **Total wall time:** ~75 min

Every test and benchmark in the repo was run, plus two new suites: [benchmarks/full_suite.py](benchmarks/full_suite.py) times every public `DB` operation, and [benchmarks/api_bench.py](benchmarks/api_bench.py) is an HTTP benchmark of the Cloud API. Raw logs and JSON for every step are in [benchmarks/runs/2026-09-24/](benchmarks/runs/2026-09-24/).

---

## 0. Environment

| | |
|---|---|
| CPU | Intel Core i9-13900HX, 32 logical CPUs, 36 MB L3 |
| OS | WSL2 Ubuntu 24.04 (kernel 6.18.33.2-microsoft-standard) on Windows 11, 7.6 GB RAM allotted |
| Disk | ext4 inside the WSL VHDX (repo copied to `~/fbench`, **not** `/mnt/c`, so I/O isn't skewed by the 9P bridge) |
| Toolchain | Python 3.12.3, g++ 13.3, `-O3 -std=c++17 -DUSE_SSE -DUSE_AVX -mavx` (AVX kernels on) |
| Durability | Default `FEATHER_WAL_SYNC` (fsync on every mutation) unless marked **nosync** (`FEATHER_WAL_SYNC=0`) |

> Why WSL: there is no MSVC on this machine, and the root `test_*.py` scripts use Linux-only APIs (`resource`, `core*.so`). Build-essential, python3-dev and python3-venv were installed into the WSL Ubuntu distro to compile the extension.

---

## 1. Headline numbers

| Area | Result |
|---|---|
| **Correctness** | pytest **303 passed / 1 failed / 7 skipped** (the skips need LLM API keys). Root feature tests: **7 of 8 all-pass**; the one failure is a broken RAM measurement, not a product bug (§9) |
| **ANN search, SIFT 500k** | **0.14 ms p50 @ recall@10 0.972** (ef=50) · 0.24 ms @ 0.991 (ef=100) · 15.4k QPS single thread (ef=10) |
| **vs v0.8.0 baseline, SIFT 500k** | Search **24–25 % faster** at ef=50/100/200 (≈20 % at ef=10), same recall. Build **27 % faster** (nosync). File **+24 %** (persisted graph) |
| **Bulk ingest** (`add_batch`, 100k×128, full metadata) | **59–64k vec/s** |
| **Serial `add()`** (100k×128) | **488 vec/s** with fsync (default) vs **10,000 vec/s** nosync: **fsync costs about 20×** |
| **Cold load** (100k×128, persisted graph) | **0.26 s** open + first search (≈390k records/s) |
| **Filtered search** | Entity: **13 µs** · namespace at 1 %: **0.77 ms** · attribute at 25 %: **38 ms** (⚠ §3) |
| **BM25 / hybrid** (100k docs) | **21 ms / 22 ms** p50 |
| **LongMemEval-S retrieval** (BM25, 500 q, no embedder) | **recall@1 0.874 · @5 0.974 · @10 0.986** |
| **HTTP API** (20k records, 1 uvicorn worker) | `/search` **2.6 ms p50**, ~370 req/s per client, peaks at **~500 req/s** |

---

## 2. Findings that need action

Ordered by impact.

1. **The default per-mutation WAL fsync dominates every write path.** Each `add` / `link` / `forget` / `update_*` costs about 1.5–2.5 ms.
   - Serial `add()`: 488 vec/s → 10,000 vec/s with `FEATHER_WAL_SYNC=0`.
   - `link()`: 569/s → **742,800/s** (1,300×).
   - `forget()`: 532/s → 10,760/s.
   - Serial-ingest builds in the `bench` harness are 6–8× slower than the v0.8.0 baseline (10k×128: 32.6 s vs 5.3 s). The legacy 100k×768 stress test took **35 min** to ingest (48 vec/s).
   - `add_batch` is unaffected (one fsync per batch), which is why [test_batch_ingest.py](test_batch_ingest.py) now shows **212×** instead of the ~3.4× in the docs.
   - *Suggested fix:* group commit (fsync every N ms or N ops), a `link_batch` / `forget_batch`, and a prominent note on `FEATHER_WAL_SYNC`.
2. **Bug: adding a vector to a second modality without `meta` wipes the record's existing metadata.** `db.add(id=1, vec=v_img, modality="visual")` (or `add_batch(..., metas=None, modality="visual")`) resets `namespace_id`, `content`, attributes and the BM25 entry to empty. The record silently drops out of namespace filters and keyword search. Reproduced in isolation; the snippet is in §10.
3. **Broad attribute filters are slow.** The pre-filtered exact path ranks every candidate, so `attributes_match={"color":"red"}` at 25 % selectivity costs **38 ms** at 100k×128. That is about 600× unfiltered search and 50× a namespace filter at 1 %. *Suggested fix:* above a selectivity threshold, switch to filtered HNSW traversal.
4. **Search doesn't scale past ~4 threads for cheap queries.** At 128-d with ef=50 it peaks at **~46k QPS on 4 threads**, then drops (8 threads: 39k, 32 threads: 37k). [tests/test_concurrency.py](tests/test_concurrency.py) fails for the same reason (8t = 1.35× 1t, it needs 1.5×).
   - Heavy queries do scale: 768-d with ef=400 goes from 799 to 4,504 QPS (5.6×).
   - So the ceiling is per-call binding work done while holding the GIL (numpy → `std::vector` copy, building result objects), not the C++ lock.
5. **BM25 latency grows with posting-list size.** 21 ms p50 at 100k docs and 8 ms at 50k, on a Zipf-distributed vocabulary. Hybrid search inherits the cost.
6. **`update_metadata` costs 1.15 ms even with nosync** (4.8 ms with fsync), about 1,000× `update_importance`. It looks like a full re-serialise or re-index per call.
7. **On-disk int8 quantized files load 4–20× slower** (100k×128: 0.97 s vs 0.26 s; 50k×768: 4.1 s vs 0.20 s). `persist_graph` is forced off for quantized modalities, so load always rebuilds the graph.
8. **Cloud API:** `GET /namespaces/{ns}/stats` takes **28 ms** at 20k records (an O(N) scan) and `GET /records?limit=100` takes 9 ms. One uvicorn worker tops out around 500 req/s whatever the client concurrency.
9. **[test_int8_ram.py](test_int8_ram.py) fails on Linux** because of its `ru_maxrss` measurement: it reports +0 MB for both indexes. Measured properly with `/proc/self/statm`, int8 **does** work: a 60k×768 index drops from **296 MB to 184 MB (1.6× less)**.

---

## 3. Vector search

### 3.1 Real data: SIFT (TexMex, ground truth shipped with the dataset)

| Dataset | ef | recall@10 | p50 ms | p95 ms | v0.8.0 recall | v0.8.0 p50 ms |
|---|---|---|---|---|---|---|
| SIFT1M (first 500k) | 10 | 0.765 | 0.057 | 0.083 | 0.774 | 0.074 |
| | 50 | **0.972** | **0.14** | 0.20 | 0.972 | 0.187 |
| | 100 | 0.991 | 0.24 | 0.30 | 0.991 | 0.321 |
| | 200 | 0.998 | 0.43 | 0.54 | 0.998 | 0.562 |
| SIFTsmall (10k) | 10 | 0.934 | 0.037 | 0.076 | — | — |
| | 50 | 0.996 | 0.06 | 0.08 | 0.996 | 0.114 |
| | 100 | 0.998 | 0.08 | 0.09 | — | — |
| | 200 | 1.000 | 0.13 | 0.16 | — | — |

SIFT 500k build: **211.9 s (2,360 vec/s, serial `add`, nosync)** vs v0.8.0's 291.3 s (1,716 vec/s). File 345 MB (724 B/vec) vs 278 MB (583 B/vec). Peak RSS 1,070 MB (same as before).

### 3.2 Synthetic data: `bench` harness, uniform random

| N × dim | recall@10 | p50 ms | p99 ms | QPS | build (fsync) | v0.8.0 p50 / build |
|---|---|---|---|---|---|---|
| 10k × 128 | 0.991 | 0.067 | 0.202 | 12,737 | 32.6 s (306/s) | 0.133 ms / 5.3 s |
| 50k × 768 | 0.563 | 0.475 | 0.875 | 1,933 | 258 s (193/s) | 1.653 ms / 394 s |

(50k×768 uniform random is a pathological case for any HNSW at ef=50. Recall matches v0.8.0's 0.576.)

### 3.3 Full suite: clustered synthetic data (500 tight clusters, deliberately hard)

| ef | 100k×128 recall | p50 ms | QPS/thread | 50k×768 recall | p50 ms | QPS/thread |
|---|---|---|---|---|---|---|
| 10 | 0.312 | 0.033 | 29,000 | 0.344 | 0.109 | 8,000 |
| 50 | 0.655 | 0.062 | 14,930 | 0.701 | 0.186 | 5,154 |
| 100 | 0.800 | 0.080 | 12,220 | 0.833 | 0.292 | 3,365 |
| 200 | 0.920 | 0.133 | 7,395 | 0.944 | 0.521 | 1,802 |
| 400 | **0.982** | 0.241 | 4,102 | **0.990** | 1.012 | 930 |

The in-cluster neighbours are nearly equidistant, so this data needs ef ≈ 400 for 0.98+. Use SIFT (§3.1) as the representative figure.

**Other knobs (100k×128, ef=50):**

| Variant | p50 ms |
|---|---|
| k=1 | 0.038 |
| k=50 | 0.082 |
| k=100 | 0.159 |
| `record_salience=True` (default, updates recall counters) | 0.047 |
| 2nd modality `visual` (dim 64, 10k) | 0.031 |

### 3.4 Concurrent search throughput (GIL released in C++)

| Threads | 100k×128 QPS | 100k×128 nosync | 50k×768 QPS | probe: 20k×768, ef=400 |
|---|---|---|---|---|
| 1 | 19,690 | 19,770 | 4,978 | 799 |
| 2 | 33,110 | 32,640 | 9,692 | 1,480 |
| 4 | **45,940** | 43,990 | 16,940 | 2,666 |
| 8 | 39,210 | 38,590 | **22,380** | 4,053 |
| 16 | 39,120 | 38,100 | 15,690 | **4,504** |
| 32 | 36,540 | 37,130 | 16,460 | — |

---

## 4. Filtered, scored, keyword and hybrid search

p50 latency in ms, k=10.

| Query | 100k × 128 | 50k × 768 |
|---|---|---|
| Unfiltered (ef=50) | 0.062 | 0.186 |
| `entity_id` (0.02 % selectivity, indexed) | **0.0125** | 0.018 |
| `namespace_id` (1 %, indexed, exact) | 0.766 | 0.796 |
| `attributes_match` color=red (25 %, indexed, exact) | ⚠ **37.8** | ⚠ **29.6** |
| `importance_gte` 0.9 (10 %, scan filter) | 0.349 | 1.40 |
| `types=[FACT]` (100 %) | 0.064 | 0.212 |
| Adaptive decay `ScoringConfig(30 d, 0.3)` | 0.061 | 0.187 |
| **BM25** `keyword_search` | 21.2 | 8.0 |
| BM25 + namespace filter | 7.1 | 1.8 |
| **Hybrid** (BM25 + HNSW, RRF) | 22.5 | 8.1 |

- The namespace pre-filter returned the full top-k on **100 %** of queries (the completeness guarantee holds).
- [test_prefiltered_search.py](test_prefiltered_search.py): filtered results match brute force exactly (**ALL PASS**).
- Adaptive-decay scoring adds no measurable latency over plain search.

---

## 5. Ingestion

| Operation | 100k×128 fsync | 100k×128 nosync | 50k×768 fsync |
|---|---|---|---|
| `add()` serial, no metadata (20k) | 488 /s | **10,000 /s** | 387 /s |
| `add()` serial, full metadata (20k) | 246 /s | 9,440 /s | 353 /s |
| `add_batch()` parallel, full metadata | **59,430 /s** | **63,640 /s** | **9,057 /s** |
| `add_batch()` 2nd modality (10k, dim 64) | 8,295 /s | 8,486 /s | 14,360 /s |
| RSS growth for the index incl. metadata | +393 MB | — | +694 MB |

Root and legacy scripts:

- **[test_batch_ingest.py](test_batch_ingest.py)** (40k×128): serial 79.8 s vs `add_batch` 0.377 s, **211.6× faster**. Batch recall@10 is 1.000. ALL PASS.
- **[phase3_benchmark.py](benchmarks/phase3_benchmark.py)** (10k multimodal, fsync):
  - ingest 397 vec/s
  - 20k links at 503/s
  - search plus graph walk 0.07 ms avg
- **[stress_test.py](benchmarks/stress_test.py)** (100k×768, uniform random, fsync):
  - ingest **48 vec/s** (2,100 s)
  - search **P50 0.287 ms / P99 0.476 ms**
  - links at 450/s
  - RSS 446 MB

---

## 6. Graph / context engine

Measured at 100k×128 with 50k random links.

| Operation | fsync | nosync |
|---|---|---|
| `link()` | 569 /s (1.76 ms each) | **742,800 /s** |
| `get_edges(id)` | 1.0 µs | — |
| `get_incoming(id)` | 0.85 µs | — |
| `context_chain` k=5, hops = 1 / 2 / 3 | 0.063 / 0.057 / 0.062 ms | — |
| `context_chain` on a **dense** graph (5k nodes, ~40 edges/node), hops = 1 / 2 / 3 | 0.37 / **8.7** / **54.5** ms | — |
| `export_graph_json(namespace)` (1k nodes) | 17.7 ms, 0.21 MB | — |
| `auto_link` over 20k records (15 candidates) | 0.375 s, 30,253 edges (53k rec/s) | — |
| `auto_link` at 768-d (20k) | 1.63 s, 26,055 edges | — |

`context_chain` cost is driven by fan-out: on a sparse graph it's as cheap as a search, but on a dense graph it grows about 6× per hop.

---

## 7. Metadata and secondary indexes

Measured at 100k×128.

| Operation | Latency | Throughput |
|---|---|---|
| `get_metadata(id)` | 0.85 µs | 1.0 M/s |
| `get_vector(id)` | 1.1 µs | 0.8 M/s |
| `touch(id)` | 0.31 µs | 2.7 M/s |
| `update_importance` (fsync / nosync) | 1.5 ms / 0.8 µs | 620 /s / 1.09 M/s |
| `update_metadata` (fsync / nosync) | 4.8 ms / ⚠ **1.15 ms** | 211 /s / 854 /s |
| `ids_in_namespace` (1k hits) | 33 µs | 15k /s |
| `ids_for_entity` (20 hits) | 1.1 µs | 845k /s |
| `ids_with_attribute` (25k hits) | 1.09 ms | 836 /s |
| `namespace_size` | 0.12 µs | 5.8 M/s |
| `list_namespaces` | 1.6 µs | 585k /s |

[test_secondary_index.py](test_secondary_index.py): ALL PASS.

---

## 8. Persistence, quantization and memory

| Scenario | 100k × 128 | 50k × 768 |
|---|---|---|
| `save()` float32 + persisted graph (v9) | 0.096 s, **80.4 MB** (843 B/rec incl. metadata) | 0.129 s, 164 MB |
| Cold `open()` + first search | **0.258 s** | **0.203 s** |
| Recall after reload | identical (0.685 → 0.685) | identical |
| On-disk int8 (v7): file size | **19.4 MB (4.2× smaller)** | **40.2 MB (4.1× smaller)** |
| On-disk int8: load time | ⚠ 0.97 s | ⚠ 4.11 s |
| On-disk int8: recall@10 (vs float) | 0.675 (vs 0.685) | 0.662 (vs 0.700) |
| In-RAM int8: vector memory | see note | **83 MB vs 177 MB (2.1× less)** |
| In-RAM int8: recall@10 | 0.656 | 0.648 |

Note: the 128-d in-process RSS delta is unreliable, because earlier allocations in the same process were not returned to the OS. The isolated per-process measurement of in-RAM int8 at 60k×768 was **296 MB → 184 MB (1.6×)**.

Root feature tests:

- **[test_parallel_load.py](test_parallel_load.py)** (40k×128, 28.9 MB):
  - persisted-graph load **46.6 ms**
  - rebuild fallback: serial 2,839 ms vs parallel 485 ms (**5.86×**)
  - recall 1.000 on every path
  - ALL PASS
- **[test_persist_graph.py](test_persist_graph.py)** (20k×128): reload in **14 ms**, 50/50 queries identical, fallback and quantized paths OK. ALL PASS.
- **[test_quantization.py](test_quantization.py)**: 3.57× smaller file, max element error 0.39 %, top-10 overlap 9/10. ALL PASS.
- **[test_int8_ram.py](test_int8_ram.py)**: int8 recall 0.882 vs float 1.000, dequant error 0.46 %, round-trip OK. **FAIL only on the RAM check** (measurement bug, §2.9).

---

## 9. Deletion lifecycle

| Operation | 100k × 128 | 50k × 768 |
|---|---|---|
| `forget()` ×10 % (fsync) | 532 /s (18.8 s) | 550 /s |
| `forget()` nosync | 10,760 /s | — |
| Search with 10 % tombstones (mean / p50) | 0.111 / 0.059 ms | 0.281 / 0.234 ms |
| `purge(namespace)` | 84 ms (1,000 recs) | 27 ms (500 recs) |
| `compact()` after 10 % forget | **11.3 s** (9,889 removed) | **20.3 s** |
| Search after compact (mean / p50) | **0.058 / 0.054 ms** | 0.176 / 0.167 ms |
| File after compact + save | 80.4 → 71.6 MB | 164 → 146 MB |

Tombstones roughly double mean latency until `compact()` runs. [test_auto_compact.py](test_auto_compact.py) (threshold trigger, orphan safety, persistence): ALL PASS.

---

## 10. Cloud HTTP API (`feather-api`, uvicorn, 1 worker)

Measured at 20k×128 over real HTTP on localhost, with `track=false` on reads.

| Endpoint | p50 ms | p99 ms | req/s (1 client) |
|---|---|---|---|
| `POST /import` (1,000/batch) | — | — | **2,659 records/s** |
| `POST /vectors` (single add) | 6.6 | 8.5 | 153 |
| `POST /search` k=10 | **2.6** | 4.0 | 368 |
| `POST /search` track=true | 2.8 | 4.0 | 350 |
| `POST /keyword_search` | 2.5 | 3.5 | 392 |
| `POST /hybrid_search` | 3.1 | 3.9 | 318 |
| `GET /records/{id}` | 1.9 | 2.7 | 524 |
| `POST /records/{id}/link` | 4.6 | 7.1 | 208 |
| `GET /records/{id}/edges` | 1.9 | 2.8 | 520 |
| `POST /context_chain` hops=2 | 2.6 | 3.5 | 378 |
| `GET /records?limit=100` | 9.3 | 12.6 | 105 |
| `GET /namespaces/{ns}/stats` | ⚠ **28.1** | 31.1 | 35 |

Concurrent `/search` throughput:

| Clients | req/s |
|---|---|
| 1 | 366 |
| 4 | 472 |
| 16 | **496** |
| 32 | 456 |

HTTP and JSON add about 2.5 ms per call on top of the ~0.06 ms engine time. Run multiple workers for throughput.

Repro for the multimodal metadata bug (§2.2):

```python
m = fc.Metadata(); m.content = "hello"; m.namespace_id = "nsA"
db.add(id=1, vec=v8, meta=m)
db.add(id=1, vec=v4, modality="visual")      # no meta
db.get_metadata(1).namespace_id               # -> ''  (was 'nsA')
```

---

## 11. LongMemEval (ICLR 2025): long-term memory retrieval

| Run | Questions | Result |
|---|---|---|
| **BM25 retrieval, LongMemEval-S** (~50 sessions per question, [benchmarks/longmemeval.py](benchmarks/longmemeval.py)) | 500 | **R@1 0.874 · R@3 0.942 · R@5 0.974 · R@10 0.986**, 102 s total (~0.2 s per question including ingest) |
| By question type, S (R@10) | | knowledge-update 1.000 · single-session-user 1.000 · single-session-assistant 1.000 · multi-session 0.985 · temporal 0.985 · **preference 0.900** |
| BM25 on the oracle variant (evidence only, sanity check) | 500 | 1.000 at every k |
| `bench` harness, oracle, deterministic hash embedder + substring judge | 500 | 0.432 overall. **Not a quality number:** the hash embedder has no semantics. It only proves the pipeline runs (5 ms per question, 0 failures) |
| QA pipeline dry run ([longmemeval_qa.py](benchmarks/longmemeval_qa.py)) | 3 | Prompts assembled OK, no API calls |

---

## 12. Test suite detail

| Suite | Result |
|---|---|
| `pytest tests/` (29 files, 311 tests, 33 s) | **303 passed, 1 failed, 7 skipped** |
| └ failed | `test_concurrency.py::test_concurrent_search_scales` (1t = 28,853 QPS, 8t = 36,489 QPS; needs 1.5×). See §2.4 |
| └ skipped | 7 LLM-provider tests (no `ANTHROPIC` / `OPENAI` / `GEMINI` keys, no Ollama) |
| Root `test_*.py` (8 scripts) | 7 ALL PASS · `test_int8_ram` 1 failing check (measurement bug) |

---

## 13. Not covered in this run

- **LLM-dependent benchmarks:** LongMemEval QA accuracy, dense and hybrid LongMemEval with real embeddings, `longmemeval_phase9` (fact extraction), and the 7 skipped provider tests. These need API keys and cost money; run them with `GOOGLE_API_KEY` / `ANTHROPIC_API_KEY` set.
- **Rust CLI (`feather-cli`, `p-test/`):** no Rust toolchain on this machine.
- **Full SIFT1M (1M vectors):** it sits exactly at HNSW `max_elements`, so it was subset to 500k to match the v0.8.0 baseline.
- **Native Windows (MSVC) build:** not possible here (no cl.exe). All numbers are Linux under WSL2, and WSL2 fsync latency may differ from bare-metal NVMe.

---

## 14. Reproduce

```bash
# inside Linux / WSL, from feather/
python -m venv venv && . venv/bin/activate
pip install numpy pybind11 setuptools pytest pytest-asyncio httpx psutil openai -r feather-api/requirements.txt
python setup.py build_ext --inplace
bash benchmarks/runs/2026-09-24/run_all.sh     # the exact sequential runner used (~75 min)
```

Individual pieces:

```bash
python benchmarks/full_suite.py --n 100000 --dim 128 --out suite.json
FEATHER_WAL_SYNC=0 python benchmarks/full_suite.py --n 100000 --dim 128
python benchmarks/api_bench.py --n 20000 --dim 128
python -m bench run vector_ann_real --dataset sift1m --n 500000 --queries 1000 --ef-sweep 10,50,100,200
python benchmarks/longmemeval.py --data ~/.cache/feather/longmemeval/longmemeval_s_cleaned.json --mode keyword
```
