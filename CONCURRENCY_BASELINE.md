# Concurrency Baseline: Measured Numbers (Phase 0)

> The measured "before" numbers for [CONCURRENCY.md](CONCURRENCY.md) and [CONCURRENCY_FIX_PLAN.md](CONCURRENCY_FIX_PLAN.md). The earlier suites time each operation in isolation ([BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)). This run measures **interference**: how much reads slow down while writes, batch loads, saves, compactions, metadata updates or API ingest traffic are running.
>
> **Run date:** 2026-09-24 · **Version:** feather-db 0.18.2 · **Raw data:** [benchmarks/runs/2026-09-24-concurrency/](benchmarks/runs/2026-09-24-concurrency/)

---

## TL;DR

| # | Finding | Worst measured effect | In the earlier docs? |
|---|---|---|---|
| **1** | **Writers starve under read load.** The Linux `std::shared_mutex` prefers readers, so a writer waits until no reader holds the lock at all. | 8 readers at 768-d: a single writer drops from **191 → 0.6 writes/s** (p50 1.3 s). With hybrid-search readers, writes stall for the **entire 8 s window**. | **No, new** |
| 2 | Each fsync-backed write holds the exclusive lock through its fsync | 1 writer cuts read throughput by **68%** (p99 1.6 → 8.4 ms). 4 writers cut it by **83%**. With `FEATHER_WAL_SYNC=0` the loss is only 10–21%. | Yes (#1) |
| 3 | Write throughput does not scale with writers, and waiting writers queue unfairly | 1, 4 or 16 writers all give **~255 writes/s** with fsync. Writer p99 rises from 7 ms to **656 ms**, max **1.3 s**. | Partly |
| **4** | **`add_batch` builds the HNSW graph under the exclusive lock** | One writer looping 1,000-record batches cuts reads by **99%**: 16.7k → 148/s, p50 0.44 → 62 ms | **No, new** |
| 5 | Compaction freezes the DB | 100k×128: **40 s** with no reads and no writes. 50k×768: **86 s**. The `forget()` that triggers auto-compaction blocks its caller for **35–57 s**. | Yes (#2) |
| 6 | `update_metadata` cost grows linearly with total edges | 0.41 → 3.5 → **13.2 ms** at 10k → 100k → 500k edges. One updater thread cuts concurrent reads by **95%**. | Yes (#4) |
| 7 | `save()` blocks writers for its whole duration | With saves looping, writes drop to **17/s** and writer p99 reaches **2.1 s** | Yes (#3) |
| 8 | The API is capped at one core | `/search` tops out at **~490 req/s** (128-d) whether there are 16 or 64 clients, with the server at ~113% CPU | Yes (#5, #6) |
| **9** | **Any `ingest_text` traffic starves search, even with a fast embedder** | 64 ingest clients with a **5 ms** embedder cut search by **85%** (488 → 75/s, p50 215 ms). With a 300 ms embedder the cut is 89%. | Partly (#7 blamed only slow embedders) |
| 10 | A single API delete is expensive | **15 deletes/s** at p50 135 ms, and 2 delete clients halve search throughput | Yes (plan 2.3, 2.4) |

Items **1, 4 and 9** are new. They change the fix plan; see [§5](#5-what-this-changes-in-the-plan).

---

## 1. Setup and method

| | |
|---|---|
| Machine | Intel i9-13900HX (8 P-cores + 16 E-cores, 32 threads), Windows 11 |
| OS | WSL2 Ubuntu 24.04, kernel 6.18, 7.6 GB RAM allotted |
| Disk | ext4 inside the WSL virtual disk. The repo was copied to `~/featherbench`, not run from `/mnt/c`. |
| Build | g++ 13.3, `-O3 -std=c++17 -DUSE_SSE -DUSE_AVX -mavx`, Python 3.12.3 |
| Engine data | 100k × 128-d and 50k × 768-d. Random vectors; each record has 12-word content, a namespace, an entity and one attribute. Built with `add_batch` and checkpointed before every scenario. |
| API data | 20k records, 128-d and 768-d, uvicorn with **1 worker** (the shipped config) |

**How the engine benchmark works** ([benchmarks/concurrency_bench.py](benchmarks/concurrency_bench.py)):
- Reader threads loop `search(k=10, record_salience=False)`. Writer threads loop `add`, `link` or `add_batch` into the **same** DB.
- Each scenario runs for 8 s, and latency is timed per call.
- Every scenario runs twice, with `FEATHER_WAL_SYNC=1` (the default) and `=0`. Each mode runs in its own process, because the engine reads that variable once.
- The "exclusive-section" tests (compaction, auto-compaction, `update_metadata`) run with fsync off, so fsync cost doesn't hide the CPU cost.

**How the API benchmark works** ([benchmarks/api_concurrency_bench.py](benchmarks/api_concurrency_bench.py)):
- It starts the real `feather-api`.
- Load comes from several client **processes** of up to 8 threads each, so the client's own GIL isn't the bottleneck.
- A fake Ollama-compatible embedding server with an adjustable delay stands in for the provider, so no product code was changed.
- Each scenario runs for 10 s.

**Limitations:**
- **Absolute numbers drift up to 2× between runs** on this hybrid-core laptop under WSL. The same readers-only scenario measured 33.5k/s in one pass and 16.7k/s in the next, probably depending on whether threads landed on P-cores or E-cores. **Compare numbers within one run, not across runs.** The first 128-d pass also overlapped a smoke test, so it was discarded. The canonical 128-d data comes from a clean re-run; the noisy pass is kept as `engine_128_first_pass_noisy.log`.
- **Latency is timed from Python**, so it includes GIL hand-off time. That overhead is real for any Python caller, but it isn't pure engine time.
- **WSL2 fsync** takes about 2–4 ms here. Cloud block storage is often slower, and local NVMe on bare metal is faster. So the fsync-related results (2, 3, 7) are representative of cloud disks, not a best case.
- **Durations and reproducibility.** Compaction here took 40 s for 100k×128. The earlier [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) measured 11.3 s for a similar operation. The difference is not explained yet; core placement is a likely factor. Either way, the DB is **fully frozen** for the whole compaction.

---

## 2. Engine results

### 2.1 Reads alone (the reference point)

| | 100k×128 fsync on | 100k×128 fsync off | 50k×768 fsync on | 50k×768 fsync off |
|---|---|---|---|---|
| 1 reader | 7,470/s · p99 0.21 ms | 7,656/s · 0.20 | 2,034/s · 0.89 | 2,260/s · 0.72 |
| 4 readers | 21,352/s · p99 0.47 | 20,740/s · 0.50 | 7,945/s · 1.19 | 9,020/s · 0.78 |
| 8 readers | 16,652/s · p99 1.56 | 16,113/s · 1.69 | 12,821/s · 1.43 | 13,356/s · 1.29 |

For cheap queries, throughput peaks at 4 threads and falls at 8. The earlier report found the same thing: the Python-side GIL hand-off dominates at 128-d.

### 2.2 Writes alone: no scaling, unfair queueing

`add()`, one record per call, with full metadata:

| Writers | fsync on (128-d) | fsync off (128-d) | fsync on (768-d) | fsync off (768-d) |
|---|---|---|---|---|
| 1 | **255/s** · p99 6.6 ms · max 12 | 2,209/s · p99 1.0 | 191/s · p99 8.3 | 557/s · p99 3.0 |
| 4 | **254/s** · p99 **161** · max 409 | 1,808/s · p99 25 | 182/s · p99 161 | 542/s · p99 66 |
| 16 | **260/s** · p99 **656** · max **1,325** | 1,719/s · p99 89 | 198/s · p99 573 · max 1,328 | 530/s · p99 367 |

- **Throughput with fsync on is flat at about 255/s, whatever the number of writers.** Each `add` pays its own fsync while holding the lock, so writers take turns. Group commit (plan Phase 3) targets exactly this.
- With fsync off, adding writers *lowers* throughput, because they contend on the lock.
- The median barely moves while p99 and max explode, which means some writers wait far longer than others. The lock hands off unfairly.

### 2.3 Reads while writes run (8 readers)

**100k × 128-d** (clean re-run):

| Background writes | fsync on: reads/s (Δ) · read p99 | fsync on: writes/s · write p99 / max | fsync off: reads/s (Δ) · read p99 | fsync off: writes/s · write p99 |
|---|---|---|---|---|
| none | 16,652 · 1.56 ms | — | 16,113 · 1.69 ms | — |
| 1 × `add` | 5,296 (**−68%**) · **8.4 ms** | 189 · 13 / 44 ms | 14,583 (−9%) · 1.5 | 762 · 3.6 |
| 4 × `add` | 2,845 (**−83%**) · **10.5 ms** | 116 · **749 / 2,865** ms | 12,789 (−21%) · 1.6 | 804 · 36 |
| 16 × `add` | 2,932 (**−82%**) · 10.2 ms | 116 · **2,939 / 7,042** ms | 11,437 (−29%) · 1.8 | 729 · 194 |
| 4 × `link` | 3,131 (**−81%**) · 9.7 ms | 123 · 588 / 1,309 ms | 14,756 (−8%) · 1.8 | 1,914 · 6.5 |
| 1 × `add_batch(1000)` | **148 (−99%)** · p50 **62** / p99 117 ms | 14.5 batches · 116 ms | **217 (−99%)** · p50 41 / p99 74 | 21.9 batches · 76 |

**50k × 768-d**, where the writer-starvation effect shows:

| Background writes | fsync on: reads/s · p99 | fsync on: writes/s (alone → with readers) · write p50 / max | fsync off: writes/s (alone → with readers) |
|---|---|---|---|
| none | 12,821 · 1.43 ms | — | — |
| 1 × `add` | 11,983 · 1.50 | **191 → 0.6/s** · p50 **1,270 ms** / max 3.3 s | **557 → 1.2/s** |
| 4 × `add` | 11,844 · 1.46 | **182 → 0.9/s** · p50 3.5 s / max **8.0 s** | 542 → 3.0/s |
| 16 × `add` | 13,569 · 1.19 | **198 → 2.5/s** · p50 **8.0 s** | 530 → 5.9/s |
| 4 × `link` | 12,680 · 1.46 | 3.2/s · p50 417 ms / max 7.2 s | 5.1/s |
| 1 × `add_batch(1000)` | **44 (−99.7%)** · p50 **188 ms** | 4.7 batches/s | reads 342/s · p99 255 ms |

**Reading these two tables together.** The two finding types trade places depending on how long a read holds the lock:

- **Short reads (128-d, ~0.4 ms each).** Brief gaps with no reader appear, so writers get in. Each writer then holds the lock **through its fsync**, and readers lose 68–83% of their throughput (finding 2). With fsync off, the loss shrinks to 9–29%, which confirms that the fsync *inside* the lock is the cause.
- **Longer reads (768-d, or hybrid search).** Eight overlapping readers almost never all release at once. Glibc's reader-preferring rwlock lets new readers in while a writer waits, so **the writer starves** (finding 1). Reads look healthy, but writes are close to zero, and turning fsync off doesn't help.
- **`add_batch` hurts in both cases.** The whole parallel HNSW build of 1,000 vectors runs while holding the exclusive lock ([feather.h:1160](include/feather.h#L1160), [feather.h:1209](include/feather.h#L1209)), so every batch freezes readers for 40–190 ms (finding 4).

### 2.4 Hybrid-search readers starve writers completely

| 8 × `hybrid_search` readers + 4 × `add` writers | reads/s | writes/s | write p50 |
|---|---|---|---|
| 100k×128 fsync on | 7,198 (unchanged from 7,144 with no writers) | **0.5** | **8,019 ms** (the whole window) |
| 100k×128 fsync off | 4,864 | **0.5** | 8,007 ms |
| 50k×768 fsync on | 8,928 | **0.6** | 8,012 ms |

The only writes that finished were the ones that got in after the readers stopped at the end of the window. In practice, while a read-heavy workload of hybrid searches is running, **new data cannot be written at all.**

### 2.5 `save()` during a write load

8 readers, 2 writers and a thread calling `save()` in a loop:

| | reads/s · max | writes/s · p99 / max |
|---|---|---|
| 128-d fsync on | 14,633 · max **147 ms** | **17/s** · **2,083 / 3,236 ms** |
| 128-d fsync off | 16,815 · max 5.6 ms | 46.7/s · 1,665 / 1,994 ms |
| 768-d fsync on | 10,867 · max 35 ms | **0.6/s** · p50 978 ms / max 7.7 s |

As designed, reads keep flowing during a save. Writers, however, are blocked for every full-file rewrite. The API does this rewrite on every delete, and at least every 30 s during imports.

### 2.6 Compaction and auto-compaction (fsync off)

| | 100k × 128-d | 50k × 768-d |
|---|---|---|
| Records removed | 10,000 (10%) | 5,000 (10%) |
| `compact()` duration | **39.95 s** | **85.6 s** |
| Longest read wait during it | **39.9 s** | **66.8 s** |
| Longest write wait during it | **39.95 s** | **66.9 s** |
| Reads before → reads/s averaged over the window | 15,243 → 150 | 13,239 → 2,845 |
| **Auto-compaction:** the one `forget()` that crossed the 10% threshold | **34,954 ms** (median `forget` 0.011 ms) | **56,708 ms** |

Compaction is a complete freeze of both reads and writes. With `set_auto_compact` on, some unlucky caller's `forget()` absorbs the whole cost.

### 2.7 `update_metadata` versus graph size (100k records, fsync off)

| Total edges in DB | `update_metadata` mean / p99 | `update_importance` mean (reference) |
|---|---|---|
| 10,000 | 0.41 / 0.60 ms | 0.008 ms |
| 100,000 | 3.55 / 4.86 ms | 0.017 ms |
| 500,000 | **13.19 / 18.30 ms** | 0.040 ms |

Cost grows linearly with **total** edges, not with the updated record's own edges, confirming the full scan of the reverse index ([feather.h:1518-1523](include/feather.h#L1518-L1523)).

With 500k edges and **one** thread looping `update_metadata`, 8 readers drop from **18,833/s to 997/s (−95%)**, and read p50 goes from **0.39 ms to 13.0 ms**.

---

## 3. Cloud API results

These use uvicorn with 1 worker, `track=false` on searches, and 20k records. Each cell shows req/s, then p50 / p99 in ms.

| Scenario | 128-d search | 768-d search | Background traffic achieved | Server CPU |
|---|---|---|---|---|
| 1 client | 337 · 2.9 / 3.7 | 249 · 4.0 / 5.2 | — | 79% |
| 16 clients | **488** · 32 / 54 | **417** · 38 / 62 | — | 114% |
| 64 clients | 497 · 127 / **181** | 415 · 153 / 207 | — | 113% |
| 16 search + 8 `POST /vectors` | 264 (**−46%**) · 60 / 86 | 205 (−51%) · 77 / 109 | 144 writes/s (128-d) | 102% |
| 16 search + 64 `ingest_text`, **5 ms** embedder | **75 (−85%)** · **215 / 264** | 54 (−87%) · 296 / 389 | 189 ingests/s | 106% |
| 16 search + 64 `ingest_text`, **300 ms** embedder | **52 (−89%)** · **310 / 468** | 48 (−89%) · 337 / 673 | 128 ingests/s | **72%** |
| 16 search + 2 `DELETE /records/{id}` | 271 (−45%) · 61 / 96 | 301 (−28%) · 51 / 89 | **15 deletes/s** · p50 135 ms (768-d: 9/s, 221 ms) | 154% |
| 16 search + 1 `/import` (1,000/batch) | 300 · 39 / **125** | 276 · 42 / **328** | 4,300 records/s (768-d: 700/s) | 282% |

What these show:
- **The single-core ceiling.** Past 16 clients, adding clients only adds queueing: p99 grows from 54 to 181 ms while throughput stays around 490/s. Server CPU sits at about 113%, meaning one core plus a little native work. The engine itself answers a search in well under 0.5 ms, so almost all of the ~2.9 ms single-client latency is HTTP, validation and JSON.
- **Ingest starves search, and not only because of slow embedders.** A 5 ms embedder hurts almost as much as a 300 ms one, so the embedding wait isn't the only cause. Every sync route shares one 40-thread pool. Ingest requests take threads, then queue on the per-namespace Python write lock while each `db.add` does a fsync that holds the engine lock. Searches wait for free threads behind them. With the slow embedder, CPU drops to 72% while search latency is at its worst, which confirms that threads are sitting blocked rather than working.
- **Deletes.** Each one does a `forget` with fsync, then an O(N) scan in Python over every record to prune edges, then a **full file save**. That caps deletes at 9–15/s and halves search throughput.

---

## 4. Baseline table for the fix plan

These are the "Phase 0 (before)" values for the table in [CONCURRENCY_FIX_PLAN.md → Release plan](CONCURRENCY_FIX_PLAN.md#release-plan). All engine rows are 100k×128 except where noted.

| Metric | Phase 0 (before) |
|---|---|
| Reader p99, 8 readers, no writers | 1.56 ms (16.7k reads/s) |
| Reader p99, 8 readers + 4 writers, fsync on | **10.5 ms** (2.8k reads/s, −83%) |
| Write throughput, 16 writers, fsync on | **260/s** (the same as 1 writer) |
| Writer throughput with 8 readers, 768-d | **0.6–2.5/s** (starved) |
| Reads/s while `add_batch(1000)` loops | **148/s** (−99%) |
| Max read stall during `compact()` | **39.9 s** (768-d: 66.8 s) |
| The `forget()` that triggers auto-compaction | **35 s** (768-d: 57 s) |
| `update_metadata` @ 500k edges | **13.2 ms** |
| API search, 1 process, 16 clients | **488 req/s**, p99 54 ms |
| API search p99 with 64 slow ingests | **468 ms** (52 req/s) |
| API deletes/s | **15/s** |

---

## 5. What this changes in the plan

### New item: fix writer starvation (add to plan Phase 2)

- **Where:** `mutable std::shared_mutex mutex_;` at [include/feather.h:69](include/feather.h#L69). With libstdc++, this is glibc's `pthread_rwlock_t` with default attributes, which **prefer readers**.
- **Fix:** replace it with a small **writer-preferring** (or phase-fair) shared mutex that has the same `lock`, `unlock`, `lock_shared` and `unlock_shared` interface, so every `std::unique_lock` and `std::shared_lock` in the file keeps working. Once a writer is waiting, new readers block until it is done.

  ```cpp
  class FairSharedMutex {
      std::mutex m_; std::condition_variable readers_cv_, writers_cv_;
      int active_readers_ = 0, waiting_writers_ = 0; bool writer_ = false;
  public:
      void lock_shared()   { std::unique_lock l(m_);
                             readers_cv_.wait(l, [&]{ return !writer_ && waiting_writers_ == 0; });
                             ++active_readers_; }
      void unlock_shared() { std::unique_lock l(m_);
                             if (--active_readers_ == 0) writers_cv_.notify_one(); }
      void lock()          { std::unique_lock l(m_); ++waiting_writers_;
                             writers_cv_.wait(l, [&]{ return !writer_ && active_readers_ == 0; });
                             --waiting_writers_; writer_ = true; }
      void unlock()        { std::unique_lock l(m_); writer_ = false;
                             if (waiting_writers_) writers_cv_.notify_one(); else readers_cv_.notify_all(); }
  };
  ```

- **Watch out:** preferring writers means readers now wait behind a waiting writer. **Until group commit (Phase 3) moves the fsync out of the lock, that writer holds the lock for 2–4 ms**, so read latency would get worse. Ship this together with Phase 3, or behind a flag until then. Its test is §2.4 of this report: hybrid readers plus writers must show writes/s close to the writers-alone number.

### New item: take `add_batch`'s graph build out of the exclusive lock (add to plan Phase 5)

- **Where:** `add_batch` holds `std::unique_lock` from [feather.h:1160](include/feather.h#L1160) through `parallel_add` at [feather.h:1209](include/feather.h#L1209).
- **Fix:** same idea as background compaction.
  1. Under the lock: WAL, metadata and `reserve()`.
  2. Release the lock and build the new graph nodes. hnswlib's `addPoint` is internally locked per node. **This needs verification**: is `searchKnn` safe to run concurrently with `addPoint` in our fork?
  3. Briefly re-lock to publish.
- **Fallback if concurrent add + search proves unsafe:** split big batches into chunks of about 64 records, releasing the lock between chunks. Each stall then drops from 40–190 ms to a few ms.

### Changes in priority

- **Phase 3 (group commit) moves up next to Phase 2.** It drives finding 2 (−68 to −83% reads), finding 3 (flat 255 writes/s) and part of finding 9 (API ingest). It is also a prerequisite for the writer-starvation fix.
- **Plan item 4.1 (async ingest) should also move ingest off the shared thread pool.** Use a separate bounded executor for the engine write. The 5 ms-embedder result shows that ingest starves search even with a fast provider.
- **Items 2.3 and 2.4 (cheap deletes, no per-delete save) are confirmed.** 15 deletes/s is low enough to be a user-visible problem.

---

## 6. Reproduce

```bash
# Linux / WSL (not /mnt/c), from feather/
python3 -m venv .venv && . .venv/bin/activate
pip install setuptools wheel pybind11 numpy fastapi httpx python-multipart uvicorn psutil
python setup.py build_ext --inplace

# engine: fsync on, fsync off, exclusive sections (≈15 min per size)
python benchmarks/concurrency_bench.py orchestrate --n 100000 --dim 128 --seconds 8 --out runs
python benchmarks/concurrency_bench.py orchestrate --n 50000  --dim 768 --seconds 8 --out runs

# API: 8 scenarios × 10 s (≈3 min per size)
python benchmarks/api_concurrency_bench.py --n 20000 --dim 128 --out runs/api_128.json
python benchmarks/api_concurrency_bench.py --n 20000 --dim 768 --out runs/api_768.json
```

Run on an otherwise idle machine, and compare numbers **within** one run. Re-run this exact suite after each fix-plan phase, and add the results as new columns in §4.
