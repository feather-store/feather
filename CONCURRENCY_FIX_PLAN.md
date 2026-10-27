# Concurrency Fix Plan: End to End

> **What this is:** the implementation plan for every problem in [CONCURRENCY.md](CONCURRENCY.md). For each fix it names the exact files and functions to change, sketches the code, lists the tests to add, and defines "done". The work is split into phases, and every phase ships on its own as a separately releasable PR.
>
> **Status: Phases 0–6 are implemented** on branch `perf/concurrency`. The measured results are in [CONCURRENCY_RESULTS.md](CONCURRENCY_RESULTS.md). Differences from the plan as written:
> - **Two new fixes from the Phase 0 benchmark:**
>   - a phase-fair lock against writer starvation, with a long-shared mode for checkpoints
>   - chunked `add_batch`
> - **`save()` is non-blocking for writers.** It uses WAL rotation (`<wal>.old`) rather than an LSN-truncated WAL.
> - **Deferred:**
>   - idle-namespace eviction (6.1), because a handle-in-use race needs reference counting first
>   - `vector_b64` in `mcp_remote` (4.2a), because older servers would reject it
>   - the Windows CI leg (1), because no MSVC toolchain was available to validate it
> - **Phase 7 (native server):** the decision gate is not met. Sharding plus the cheaper request path covers it.
>
> The code sketches below show intent; the shipped code is in the files named. Line numbers refer to v0.18.2.

---

## Contents

- [0. Ground rules](#0-ground-rules)
- [Phase 0 — Baseline benchmark](#phase-0--baseline-benchmark)
- [Phase 1 — Safety: durable save + single-owner file lock](#phase-1--safety-durable-save--single-owner-file-lock)
- [Phase 2 — Shorten exclusive sections (quick wins)](#phase-2--shorten-exclusive-sections-quick-wins)
- [Phase 3 — Group commit: fsync outside the data lock](#phase-3--group-commit-fsync-outside-the-data-lock)
- [Phase 4 — API throughput](#phase-4--api-throughput)
- [Phase 5 — Compaction without stalls](#phase-5--compaction-without-stalls)
- [Phase 6 — Scale out: shard namespaces across processes](#phase-6--scale-out-shard-namespaces-across-processes)
- [Phase 7 — Decision gate: a native server?](#phase-7--decision-gate-a-native-server)
- [Release plan](#release-plan)
- [Risks](#risks)
- [Master checklist of files touched](#master-checklist-of-files-touched)

---

## 0. Ground rules

These apply to every phase.

1. **One phase per PR.** Each PR runs the Phase 0 benchmark before and after, and pastes both results into the PR description.
2. **After any change to `include/` or `src/`**, run `scripts/sync-cpp.sh`. CI fails otherwise, because `feather-cli/cpp/` is a vendored copy.
3. **Clean rebuild after header edits:** `rm -rf build && python setup.py build_ext --inplace`. Most of the engine lives in headers.
4. **Don't break existing files.** No phase changes the `.feather` file format. Phase 2 adds one WAL opcode, which older builds skip (see 2.4).
5. **Keep durability.** A write that has returned to the caller must survive a crash. The one change to *visibility* is in Phase 3, and it is documented there.
6. **Lock order** is fixed and must never be inverted: `save_mutex_` → `mutex_` → `wal_mutex_` (the last one is new in Phase 3).
7. **Every PR updates** `CHANGELOG.md` and any doc section it invalidates: `HOW_IT_WORKS.md`, `CONCURRENCY.md` and `CLAUDE.md`.
8. **Before merging**, run `scripts/verify.sh`.

---

## Phase 0 — Baseline benchmark

**Goal:** have numbers before touching code, so every later phase can prove it helped.

### Add

**`bench/scenarios/concurrency.py`**, registered in `bench/__main__.py` as `python -m bench run concurrency`. It is the script from [CONCURRENCY.md §9](CONCURRENCY.md#9-measure-first), extended to cover:

| Scenario | Measures |
|---|---|
| `readers_only` | Search p50, p99 and max with 8 reader threads |
| `readers_plus_writers` | The same, with 1 and 4 writer threads running `add()` |
| `readers_plus_batch` | The same, with 1 thread running `add_batch(1000)` in a loop |
| `compaction_stall` | Forget ⅓ of the records, call `compact()` in a background thread, and record the longest single search latency during it |
| `update_metadata_cost` | Time per `update_metadata` call at 10k, 100k and 500k edges |
| `save_stall` | Longest `add()` latency while a `save()` runs in parallel |

Each scenario runs twice, with `FEATHER_WAL_SYNC=1` and `=0`. Results are written to `bench/results/concurrency__*.json`, like the other benches.

**`bench/api_load.md`** is a recipe for loading the API with `oha`:
- `POST /search` alone
- `POST /search` combined with `POST /ingest_text` against a **mock embedding provider that sleeps 300 ms**

To get the mock, add a `mock` provider to `embedding.py` that is enabled only when `FEATHER_DEV_MODE=1`.

### Done when

- The baseline JSON files are committed.
- The numbers are copied into the table in [Release plan](#release-plan) as the "before" column.

---

## Phase 1 — Safety: durable save + single-owner file lock

> **Superseded in part (0.20.0).** The lock described below was replaced by the
> upstream inter-process lock: reentrant within one process, shared for
> `read_only=True`, disabled with `FEATHER_LOCK=0` (not `FEATHER_FILE_LOCK=off`).
> The rest of this phase (durable save, `close()`, context manager) stands.

These are correctness fixes, and they come first because later phases build on them. Phase 3 depends on 1.1, and Phase 6 depends on 1.2.

### 1.1 Make `save()` actually durable

**Problem.** [feather.h:931-936](include/feather.h#L931-L936) renames `.tmp` over the real file and then deletes the WAL, but it never fsyncs the new file. It also never checks whether the write succeeded. On a full disk this renames a truncated file into place and then deletes the WAL.

**Change** (`include/feather.h`):

```cpp
// ── new private helpers ──────────────────────────────────────────
static void fsync_file(const std::string& p) {
#if defined(_WIN32)
    HANDLE h = CreateFileA(p.c_str(), GENERIC_WRITE,
                           FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("fsync open failed: " + p);
    BOOL ok = FlushFileBuffers(h); CloseHandle(h);
    if (!ok) throw std::runtime_error("FlushFileBuffers failed: " + p);
#else
    int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("fsync open failed: " + p);
    int rc = ::fsync(fd); ::close(fd);
    if (rc != 0) throw std::runtime_error("fsync failed: " + p);
#endif
}
static void fsync_parent_dir(const std::string& p) {
#if !defined(_WIN32)
    auto dir = std::filesystem::path(p).parent_path();
    int fd = ::open(dir.empty() ? "." : dir.c_str(), O_RDONLY);
    if (fd >= 0) { ::fsync(fd); ::close(fd); }
#endif  // Windows: MOVEFILE_WRITE_THROUGH below covers the metadata
}
static void atomic_replace(const std::string& from, const std::string& to) {
#if defined(_WIN32)
    if (!MoveFileExA(from.c_str(), to.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("MoveFileEx failed: " + from + " → " + to);
#else
    if (std::rename(from.c_str(), to.c_str()) != 0)
        throw std::runtime_error("Atomic rename failed: " + from + " → " + to);
#endif
}
```

At the end of `save_vectors()`:

```cpp
f.flush();
if (!f) throw std::runtime_error("write failed (disk full?): " + tmp_path);
f.close();
if (!durable_checkpoints_disabled()) {  // honour FEATHER_WAL_SYNC=0 here too
    fsync_file(tmp_path);                // 1. data on disk
}
atomic_replace(tmp_path, path_);         // 2. swap
if (!durable_checkpoints_disabled())
    fsync_parent_dir(path_);             // 3. the rename itself on disk
wal_clear();                             // 4. only now is the WAL redundant
```

Supporting changes:
- **Includes.** Add `<filesystem>`, `<fcntl.h>` on POSIX, and on Windows `#define NOMINMAX` + `#include <windows.h>`. `NOMINMAX` is required because `feather.h` uses `std::min` and `std::max`.
- **Env var.** `durable_checkpoints_disabled()` reuses the `FEATHER_WAL_SYNC` switch, so there is only one knob.
- **Windows behaviour change.** `MoveFileExA` also fixes the fact that `std::rename` fails on Windows when the target file already exists.

**Tests** (`tests/test_wal_durability.py`):
- `test_save_checks_write_errors`: point the DB at a path whose `.tmp` cannot be written (a directory with the same name) and assert that `save()` raises **and** that the WAL still exists afterwards.
- `test_save_then_reopen_roundtrip`: a regression test for the new rename path, on every platform in CI.

### 1.2 One process per file: an OS lock, plus `close()`

**Problem.** Nothing stops two processes, or two `DB` objects in one process, from opening the same `.feather` file. They would both replay and append to the same WAL, and each `save()` would overwrite the other's. See [CONCURRENCY.md §5](CONCURRENCY.md#5-the-api-is-a-single-process-with-no-guard-against-a-second-one).

**Change** (`include/feather.h`):

1. **Lock file.** `DB::open()` acquires an exclusive, non-blocking lock on `path + ".lock"`:

   ```cpp
   // members
   #if defined(_WIN32)
       HANDLE lock_handle_ = INVALID_HANDLE_VALUE;
   #else
       int    lock_fd_ = -1;
   #endif
       bool closed_ = false;

   void acquire_file_lock() {
       if (lock_mode() == LockMode::Off) return;            // FEATHER_FILE_LOCK=off
       const std::string lp = path_ + ".lock";
   #if defined(_WIN32)
       lock_handle_ = CreateFileA(lp.c_str(), GENERIC_READ | GENERIC_WRITE,
                                  FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                  OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
       OVERLAPPED ov{};
       if (lock_handle_ == INVALID_HANDLE_VALUE ||
           !LockFileEx(lock_handle_, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
                       0, 1, 0, &ov))
           throw std::runtime_error("database is already open elsewhere: " + path_);
   #else
       lock_fd_ = ::open(lp.c_str(), O_RDWR | O_CREAT, 0644);
       if (lock_fd_ < 0 || ::flock(lock_fd_, LOCK_EX | LOCK_NB) != 0)
           throw std::runtime_error("database is already open elsewhere: " + path_);
   #endif
   }
   void release_file_lock();   // unlock + close handle; idempotent
   ```

   It is called in `open()` **before** `load_vectors()`, so a second opener fails before it can read or replay anything.

   `flock` is per open file description, and `LockFileEx` is per handle. So a *second `open()` in the same process* also fails, not just a second process. That is intended: two in-process instances corrupt the file exactly like two processes do.

2. **`close()`.** Add a public `close(bool save = true)`:
   - take `save_mutex_` and `mutex_` exclusively
   - `save_vectors()` if requested and `load_complete_` is true
   - `wal_close()`, then `release_file_lock()`, then set `closed_ = true`

   Every public mutator and `save()` starts with `ensure_open()`, which throws `std::logic_error("DB is closed")` when `closed_` is set.

   `~DB()` becomes: `if (!closed_) { existing checkpoint logic }` followed by `release_file_lock()`.

3. **Escape hatch.** `FEATHER_FILE_LOCK=off` disables the lock, for read-only tooling on network filesystems where `flock` is unreliable (NFS). Document this in `HOW_IT_WORKS.md`.

**Bindings** (`bindings/feather.cpp`):

```cpp
.def("close", &feather::DB::close, py::arg("save") = true,
     py::call_guard<py::gil_scoped_release>())
.def("__enter__", [](feather::DB& db) -> feather::DB& { return db; },
     py::return_value_policy::reference)
.def("__exit__", [](feather::DB& db, py::object, py::object, py::object) { db.close(true); })
.def_property_readonly("closed", &feather::DB::is_closed)
```

`py::nodelete` stays as it is. `close()` is now the supported way to release a DB from Python, and `with fdb.DB.open(p) as db:` works.

**C ABI** (`src/feather_core.cpp`): `feather_close` already deletes the handle, which runs `~DB()` and releases the lock. No Rust change is needed beyond the sync script.

**API** (`feather-api/app/db_manager.py`):
- **`adopt()`** ([db_manager.py:142-145](feather-api/app/db_manager.py#L142-L145)): before `self._dbs.pop(namespace)`, call `old.close(save=False)`. The uploaded file replaces the old state, so saving it would be wrong. Without the close, reopening the adopted file would fail on the lock.
- **`delete()`**: `db.close(save=False)` before removing the files. Also remove `.lock`.
- **`_load_existing()`**: a namespace that fails with "already open elsewhere" is logged as a **conflict** (another process owns it) and is not treated as corrupt.

**Migration: tests and examples that reopen without closing.** For example, [test_persist_graph.py:35-88](test_persist_graph.py#L35-L88) calls `db2 = fc.DB.open(p)` while `db` is still alive. Each such place must call `db.close()` first. Find them with:

```bash
grep -rn "DB.open(" tests/ test_*.py examples/ bench/ benchmarks/ feather_db/
```

`feather_db/merge.py` opens the *source* DB. It must `close(save=False)` it when done.

**Tests** (new `tests/test_file_lock.py`):

| Test | Asserts |
|---|---|
| `test_second_open_same_process_raises` | A second `open()` raises |
| `test_close_releases_lock` | `open` → `close` → `open` succeeds |
| `test_second_process_raises` | A `multiprocessing` child opening the same path gets the error |
| `test_lock_released_on_crash` | A child that holds the DB and is killed (SIGKILL / `TerminateProcess`) leaves no stale lock; the parent can open |
| `test_closed_db_rejects_writes` | `add()` after `close()` raises |
| `test_env_off_allows_double_open` | `FEATHER_FILE_LOCK=off` allows a double open |
| `test_context_manager_saves` | `with DB.open(p) as db: db.add(...)`, then reopen: the record is present |

### Phase 1 done when

- All the new tests pass on Linux, macOS **and Windows**. Add a `windows-latest` leg to `test.yml` for these two test files at least.
- `scripts/verify.sh` is green.
- The API can re-upload a namespace, and delete and recreate it, without lock errors (add both cases to `tests/test_api_records.py`).

---

## Phase 2 — Shorten exclusive sections (quick wins)

These are small, local changes that cut the time spent holding `mutex_` exclusively.

### 2.1 `update_metadata`: O(edges of this record), not O(all edges)

**Where:** [feather.h:1518-1523](include/feather.h#L1518-L1523).

**Change:** replace the full `reverse_index_` sweep with a targeted removal driven by the **old** record's edges:

```cpp
auto old = metadata_store_.find(id);
if (old != metadata_store_.end()) {
    deindex_meta(id, old->second);
    prev_content = old->second.content;
    had_prev     = true;
    for (const auto& e : old->second.edges) {           // ← only this record's targets
        auto rit = reverse_index_.find(e.target_id);
        if (rit == reverse_index_.end()) continue;
        auto& v = rit->second;
        v.erase(std::remove_if(v.begin(), v.end(),
                [id](const IncomingEdge& ie) { return ie.source_id == id; }), v.end());
        if (v.empty()) reverse_index_.erase(rit);
    }
}
metadata_store_[id] = meta;
...
for (const auto& e : meta.edges)
    reverse_index_[e.target_id].push_back({id, e.rel_type, e.weight});
```

**Why this is safe:** `reverse_index_` only ever contains entries that were derived from some record's `edges`. So the old record's `edges` list names every target that could hold an entry sourced from `id`.

**Test** (`tests/test_graph.py`): add edges A→B and A→C, then `update_metadata(A)` with edges A→C and A→D. Assert that `get_incoming(B)` no longer contains A, that C and D each contain A exactly once, and that the result equals a rebuild after reopening.

### 2.2 `compact_nolock`: stop rebuilding indexes that are already correct, and build in parallel

**Where:** [feather.h:381-425](include/feather.h#L381-L425).

**Change:**
1. Replace the serial loop at lines 416-417 with `parallel_add(m_idx, survivors)`, the same routine the load path uses.
2. Delete `build_secondary_indexes()` and `rebuild_bm25_index()` (lines 422-423). Dead records are already removed from both indexes at the moment they die:
   - `forget`: [feather.h:1949-1952](include/feather.h#L1949-L1952)
   - `purge`: [1979-1982](include/feather.h#L1979-L1982)
   - `forget_expired`: [2031-2032](include/feather.h#L2031-L2032)
   - `update_metadata` into a dead state: [1517](include/feather.h#L1517), [1527-1528](include/feather.h#L1527-L1528)
3. Replace `build_reverse_index()` (line 421) with an incremental cleanup over `dead`:

   ```cpp
   for (uint64_t id : dead) reverse_index_.erase(id);          // edges INTO dead ids
   for (auto& [tgt, v] : reverse_index_)                       // edges FROM dead ids
       v.erase(std::remove_if(v.begin(), v.end(),
               [&](const IncomingEdge& ie){ return dead.count(ie.source_id); }), v.end());
   ```

   The second loop is still O(edges), but it runs once per compaction, not once per record.

**Safety net:** in debug builds (`#ifndef NDEBUG`), after compacting, rebuild all derived indexes into temporaries and `assert` they equal the live ones. Add `tests/test_compact_consistency.py`, which forgets, purges, expires and marks `_deleted` via `update_metadata`, then compacts. It asserts that `keyword_search`, `ids_in_namespace` and `get_incoming` give the same results as a freshly reopened DB.

### 2.3 API: cheaper deletes

**Where:** `_prune_edges_to` at [feather-api/app/main.py:418-436](feather-api/app/main.py#L418-L436). It calls `get_metadata()` on **every record** in Python to find edges pointing at the deleted id. It also only iterates the `"text"` modality ids, which is a bug: records with no text vector are missed.

**Change:** use the reverse index the engine already maintains.

```python
def _prune_edges_to(db, dead_id: int) -> int:
    removed = 0
    for inc in db.get_incoming(dead_id):                 # O(in-degree)
        meta = db.get_metadata(inc.source_id)
        if meta is None:
            continue
        kept = [e for e in meta.edges if e.target_id != dead_id]
        if len(kept) != len(meta.edges):
            removed += len(meta.edges) - len(kept)
            meta.edges = kept
            db.update_metadata(inc.source_id, meta)
    return removed
```

`_prune_edges_to_set` (line 439) gets the same treatment: take the union of `get_incoming(d)` over all dead ids.

**Test** (`tests/test_api_records.py`): delete a record that has incoming edges from records with only a `"visual"` vector, and assert the edges are pruned. This covers the old modality bug.

### 2.4 WAL-log the two unlogged mutations, then drop per-request saves

**Problem.** `purge` and `auto_link` are not written to the WAL, so the API has to call `save()` after them to be crash-safe. Once they are logged, most API `save()` calls become unnecessary.

**Engine change** (`include/feather.h`):
- Add `WalOp::PURGE = 0x06`, whose payload is `[u16 len][namespace]`. `purge()` appends it and syncs, and `replay_wal()` re-runs the purge logic. Factor that logic into `purge_nolock(ns)` so replay and the live call share it.
- `auto_link()` appends one `WalOp::LINK` record per edge it creates, then does **one** `wal_sync()` at the end.
- Keep `WAL_VERSION = 2`. A pre-0.19 build replaying a WAL that contains op 6 skips it, because unknown ops fall through the `if`/`else` chain. That is an acceptable downgrade behaviour; note it in the CHANGELOG.

**API change** (`feather-api/app/main.py`): remove `db.save()` from single-record handlers, which are now WAL-covered. The current call sites are 377, 820, 866, 892, 909, 931, 987, 1135 and 1617.

| Line | Route | Action |
|---|---|---|
| 820 | `DELETE /records/{id}` | remove |
| 866 | `batch_delete` | remove |
| 892 | unlink | remove |
| 909 | `purge` | remove (WAL-logged after 2.4) |
| 931 | `compact` | **keep**: a checkpoint right after compaction is the point |
| 987 | `quantize` | **keep**: this setting only takes effect on save |
| 377, 1135, 1617 | explicit `create`, `/save` and `/flush` routes | **keep** |

**Background checkpointer** (new, in `main.py` `lifespan`): replaces `_throttled_save` calls that happen on the request path.

```python
async def _checkpointer():
    while True:
        await asyncio.sleep(CHECKPOINT_POLL_S)                 # default 5 s
        for ns in manager.list_namespaces():
            db = manager.peek(ns)                              # no auto-create
            if db is None:
                continue
            if db.wal_size() >= CHECKPOINT_WAL_BYTES or \
               (db.wal_size() > 0 and time.time() - _last_save.get(ns, 0) >= CHECKPOINT_MAX_AGE_S):
                await run_in_threadpool(db.save)               # NOT under manager.lock
                _last_save[ns] = time.time()
```

- New env vars: `FEATHER_CHECKPOINT_WAL_MB` (default 64), `FEATHER_CHECKPOINT_MAX_AGE_S` (default 300) and `FEATHER_CHECKPOINT_POLL_S` (default 5).
- It needs a small engine accessor, `size_t wal_size() const`, which returns `std::filesystem::file_size(wal_path_)` or 0 if there is no WAL. Bind it in `bindings/feather.cpp`.
- Taking `save()` out from under `manager.lock` is safe. The engine's own `save_mutex_` and shared `mutex_` already give `save()` a consistent snapshot, so the Python lock never protected anything there.

**Tests:**
- `tests/test_wal_recovery.py`: purge, then kill -9, then reopen. The purged namespace must stay purged. Do the same for auto_link: the links must survive.
- `tests/test_api_records.py`: delete a record without any save, simulate a crash by closing without saving (`close(save=False)`) and reopening, and assert the record is still gone.

### Phase 2 done when

- `update_metadata_cost` is flat as the edge count grows from 10k to 500k.
- `compaction_stall` shows a lower maximum stall than Phase 0. Expect several times lower, from the parallel build and the skipped BM25 rebuild.
- An API `DELETE` no longer rewrites the namespace file (confirm with the file's mtime in a test).

---

## Phase 3 — Group commit: fsync outside the data lock

This phase fixes [CONCURRENCY.md §1](CONCURRENCY.md#1-writes-hold-the-exclusive-lock-through-an-fsync) and is the largest read-latency win under write load. It depends on Phase 1.1: checkpoints must be durable before the WAL can be trimmed safely.

### Design

| | Before | After |
|---|---|---|
| Holding `mutex_` exclusively | serialize payload + fwrite + **fsync** + apply | fwrite + fflush + apply |
| After releasing `mutex_` | — | wait until the WAL is fsynced past this record's LSN |
| Concurrent writers | one fsync each, serialized | **one fsync covers all of them** |
| `add()` returns once | the record is durable | the record is durable (unchanged) |
| Readers can see the record once | it is durable | it is applied in memory, possibly *before* it is durable |

**WAL order still matches apply order**, because the append still happens inside the exclusive section. Replay therefore reproduces exactly the in-memory history.

**Visibility change.** In the gap between "applied" and "fsynced", a concurrent reader can see a record that a machine crash would then lose. The writer is never told it succeeded, so the writer's durability guarantee is unchanged. This is the standard group-commit trade-off. For users who need "never visible before durable", keep a strict mode: `FEATHER_WAL_STRICT=1` keeps the fsync inside the lock (the current behaviour), but still batches concurrent writers.

### Engine change (`include/feather.h`)

New members:

```cpp
mutable std::mutex              wal_mutex_;      // guards wal_file_, LSNs, syncing flag
mutable std::condition_variable wal_cv_;
mutable uint64_t wal_written_lsn_ = 0;           // appended + fflush'd
mutable uint64_t wal_synced_lsn_  = 0;           // on stable storage (or checkpointed)
mutable bool     wal_syncing_     = false;
```

`wal_append` returns an LSN:

```cpp
uint64_t wal_append(WalOp op, uint64_t id, const std::string& payload) {
    if (wal_path_.empty()) return 0;
    std::lock_guard<std::mutex> g(wal_mutex_);
    if (!wal_open_for_append()) return 0;
    /* ... existing fwrite of op/id/len/payload/crc + fflush ... */
    return ++wal_written_lsn_;
}
```

The sync is split into a leader/follower wait:

```cpp
void wal_wait_durable(uint64_t lsn) const {
    if (lsn == 0 || !wal_sync_enabled()) return;
    std::unique_lock<std::mutex> lk(wal_mutex_);
    while (wal_synced_lsn_ < lsn) {
        if (wal_syncing_) { wal_cv_.wait(lk); continue; }     // follower
        wal_syncing_ = true;                                   // leader
        const uint64_t target = wal_written_lsn_;
        std::FILE* fh = wal_file_;
        lk.unlock();
        if (fh) { std::fflush(fh); platform_fsync(fh); }       // _commit / fsync
        lk.lock();
        wal_syncing_    = false;
        wal_synced_lsn_ = std::max(wal_synced_lsn_, target);
        wal_cv_.notify_all();
    }
}
```

`wal_clear()` and `wal_close()` coordinate with an in-flight sync:

```cpp
void wal_clear() const {
    if (wal_path_.empty()) return;
    std::unique_lock<std::mutex> lk(wal_mutex_);
    wal_cv_.wait(lk, [&]{ return !wal_syncing_; });   // never fclose under a running fsync
    if (wal_file_) { std::fclose(wal_file_); wal_file_ = nullptr; }
    std::remove(wal_path_.c_str());
    wal_synced_lsn_ = wal_written_lsn_;   // checkpoint (Phase 1.1: fsync'd) covers them
    wal_cv_.notify_all();                 // release any waiter whose record is now in the base file
}
```

Every mutator follows the same pattern. `add()` is shown here; apply it identically to `add_batch`, `update_metadata`, `update_importance`, `link`, `forget`, `forget_expired`, `purge` and `auto_link`:

```cpp
void add(uint64_t id, const std::vector<float>& vec, const Metadata& meta, const std::string& modality) {
    ensure_open();
    const std::string payload = encode_add_payload(modality, vec, meta);   // no lock
    uint64_t lsn;
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        lsn = wal_append(WalOp::ADD, id, payload);
        if (wal_strict()) wal_wait_durable(lsn);        // FEATHER_WAL_STRICT=1 path
        /* ... existing apply: index, metadata, secondary, BM25 — unchanged ... */
    }
    wal_wait_durable(lsn);                              // returns immediately if already synced
}
```

In strict mode, `wal_wait_durable` runs while `mutex_` is held. That is the one place `wal_mutex_` is taken under `mutex_`, which matches the lock order in the ground rules.

Other changes in this phase:
- **Payload encoding helpers.** `encode_add_payload`, `encode_link_payload` and so on move the `ostringstream` work out of the lock. They are pure functions of their arguments.
- **`add_batch`.** It keeps one fsync per batch, but the fsync now also runs outside the lock: append all records under the lock, remember the last LSN, release the lock, then wait on that LSN.
- **Checkpoints during a batch.** A `save()` that starts while a writer is in `wal_wait_durable` is safe. The writer's record is already applied and its WAL bytes are flushed, so `save()` captures the record in the base file. `wal_clear()` then advances `wal_synced_lsn_` past it and wakes the writer.

### Tests

- **`tests/test_group_commit.py`** (new):

  | Test | Asserts |
  |---|---|
  | `test_many_writers_fewer_fsyncs` | 16 threads × 200 `add()` calls produce far fewer fsyncs than 3200. Add a test-only counter `wal_sync_count()`, compiled in with `-DFEATHER_TEST_HOOKS`, or exposed always since it is cheap. |
  | `test_readers_not_blocked_by_fsync` | Wrap `platform_fsync` in a test hook that sleeps 20 ms. Run 4 writers; reader p99 must stay well under 20 ms. |
  | `test_strict_mode_serializes` | With `FEATHER_WAL_STRICT=1`, reader p99 goes back up, confirming the fsync happens under the lock |
  | `test_save_during_group_commit` | Writers plus a looping `save()` for 5 s, then reopen: every acknowledged id is present |

- **Existing tests must pass unchanged:** `tests/test_wal_recovery.py` (kill -9 between checkpoints), `tests/test_wal_durability.py` and `tests/test_concurrency.py`.
- **ThreadSanitizer run** (new CI job, Linux only, non-blocking at first): build with `-fsanitize=thread` and run `test_concurrency.py`, `test_group_commit.py` and the Phase 0 bench in a reduced mode.

### Phase 3 done when

- `readers_plus_writers` (4 writers, `FEATHER_WAL_SYNC=1`): reader p99 is at most 2× the `readers_only` p99. Today it is expected to be far worse; Phase 0 will tell us by how much.
- Write throughput with 16 concurrent writers is at least 5× the Phase 0 number on a disk where fsync is slow.
- There are no TSan reports in the engine.
- `HOW_IT_WORKS.md` §5.7 and §5.8 are updated to describe group commit, LSNs and strict mode.

---

## Phase 4 — API throughput

This phase fixes [CONCURRENCY.md §6](CONCURRENCY.md#6-python-overhead-around-a-02-ms-search) and [§7](CONCURRENCY.md#7-blocking-embedding-calls-can-starve-the-thread-pool). Nothing in the engine changes, except 4.2c.

### 4.1 Embeddings: async, pooled, batched, honest about dimensions

**Files:** `feather-api/app/embedding.py`, `feather-api/app/main.py`, `feather-api/requirements.txt` (add `httpx`).

1. **Pooled HTTP.** Replace `urllib` in `_http_post_json` ([embedding.py:157-167](feather-api/app/embedding.py#L157-L167)) with two long-lived clients:
   - a module-level `httpx.Client(timeout=30, limits=Limits(max_keepalive_connections=20))` for the sync `/import` path
   - an `httpx.AsyncClient` for the async path

   This removes the TCP and TLS handshake from every call. Close both in `lifespan`.

2. **Async embed.** Add `async def aembed(text)` and `async def aembed_many(texts)` next to the sync versions. Each provider driver gets an async twin that shares its request-building code, with only the transport differing.

3. **Async `ingest_text`** ([main.py:1422-1453](feather-api/app/main.py#L1422-L1453)):

   ```python
   @app.post("/v1/{namespace}/ingest_text", ...)
   async def ingest_text(namespace: str, req: IngestTextRequest):
       try:
           vec = await EMBED_BATCHER.embed(req.text)          # no thread held while waiting
       except RuntimeError as e:
           raise HTTPException(400, str(e))
       return await run_in_threadpool(_store_ingested, namespace, req, vec)   # engine work in a thread
   ```

   `_store_ingested` holds the rest of the current body: the dim check, id, `Metadata`, `with manager.lock(ns): db.add(...)`. It does **not** call `_throttled_save`, because the Phase 2.4 checkpointer handles saving.

4. **Micro-batcher** (new class `EmbedBatcher` in `embedding.py`):
   - It holds an `asyncio.Queue` of `(text, future)` pairs.
   - A background task drains up to `FEATHER_EMBED_BATCH_MAX` items (default 64), or whatever arrived within `FEATHER_EMBED_BATCH_WINDOW_MS` (default 8 ms). It sends them as **one** provider request and resolves each future.
   - Batch endpoints: OpenAI and Azure (`input: [...]`), Voyage, Cohere, Gemini (`batchEmbedContents`) and Ollama (`/api/embed` with a list).
   - Any provider without batch support falls back to `asyncio.gather` over single calls.
   - Errors are delivered to every future in the failed batch.

5. **Stop silent padding.** Remove the pad/truncate block in `embed()` ([embedding.py:146-153](feather-api/app/embedding.py#L146-L153)). Raise `RuntimeError(f"model returned {len(vec)} dims, config says {dim}")` instead. This matches the "reject, don't pad" intent already written in `ingest_text` and `/import`.

   This is a **behaviour change**. Anyone whose model dimension differs from `FEATHER_EMBED_DIM` now gets a 400 instead of corrupted vectors. Put it in the CHANGELOG under "Breaking".

6. **LRU cache** (optional flag `FEATHER_EMBED_CACHE=2048`): map `(provider, model, sha256(text))` to a vector, and serve hits without a network call.

**Tests** (`tests/test_api_embedding.py`, new; uses the Phase 0 `mock` provider that sleeps 300 ms):

| Test | Asserts |
|---|---|
| `test_ingest_does_not_starve_search` | 100 concurrent `ingest_text` calls plus 200 `search` calls: search p99 stays under 50 ms |
| `test_batcher_coalesces` | 50 concurrent ingests produce at most 3 provider calls (the mock counts its calls) |
| `test_dim_mismatch_rejected` | A mock that returns 1536 dims with a 768 config gets a 400, not a padded vector |

### 4.2 Search fast path

**Files:** `feather-api/app/models.py`, `feather-api/app/main.py`, `bindings/feather.cpp`, `include/feather.h`.

**a) Binary query vectors.** In `SearchRequest`, `HybridSearchRequest`, `ContextChainRequest` and `AddVectorRequest`, add `vector_b64: Optional[str]`, containing little-endian float32 bytes encoded as base64. Make `vector` optional, and validate that exactly one of the two is set:

```python
@model_validator(mode="after")
def _one_vector(self):
    if (self.vector is None) == (self.vector_b64 is None):
        raise ValueError("provide exactly one of `vector` or `vector_b64`")
    return self

def query_array(self) -> np.ndarray:
    if self.vector_b64 is not None:
        return np.frombuffer(base64.b64decode(self.vector_b64), dtype="<f4")
    return np.asarray(self.vector, dtype=np.float32)
```

Update `mcp_remote.py` and the connection-info snippets to send `vector_b64`.

**b) Skip Pydantic on the way out.** The search routes return an `ORJSONResponse` built from plain dicts. FastAPI skips `response_model` validation when a `Response` object is returned. Keep `response_model=` on the decorator so the OpenAPI docs are unchanged.

To make building the dicts cheap, add a C++-side converter in the bindings:

```cpp
.def("to_dict", [](const feather::DB::SearchResult& r) { /* build py::dict directly */ })
```

Add `orjson` to `requirements.txt`.

**c) Cosine computed in C++.** Add a `bool with_cosine = false` parameter to `DB::search` and a `float cosine = NAN` field to `SearchResult`. When the flag is set, compute the cosine against the stored vector, which the pre-filtered path already reads and the HNSW path reads through `read_vector_label`, only for the `k` returned hits.

The API's `raw_score` then passes `with_cosine=True`, and the per-hit `get_vector` and numpy loop at [main.py:692-706](feather-api/app/main.py#L692-L706) is deleted. For int8-RAM modalities the cosine is computed on dequantized values; document this.

**d) Optional metadata.** Add `include_metadata: bool = True` to the search requests. When it is false, skip building metadata entirely and return only `{id, score}`.

**e) ASGI metrics middleware.** Replace `@app.middleware("http")` ([main.py:96-111](feather-api/app/main.py#L96-L111)) with a pure ASGI middleware class. It wraps `send` to capture the status code and times the call, without going through `BaseHTTPMiddleware`.

**Tests:**
- `tests/test_api_records.py`:
  - `vector` and `vector_b64` return identical results.
  - `raw_score` values match the old numpy computation to 1e-5.
  - `include_metadata=false` returns only ids and scores.
- `scripts/verify.sh` leg 8 already checks `raw_score` and `track=false` behaviour; keep it green.

### Phase 4 done when

- Under `oha`, **search-only** requests per second on one process are at least 2× the Phase 0 number, using `vector_b64` and `include_metadata=false`.
- With 300 ms mock embeddings, **mixed search + ingest** keeps search p99 under 50 ms.

---

## Phase 5 — Compaction without stalls

This phase fixes the rest of [CONCURRENCY.md §2](CONCURRENCY.md#2-compaction-freezes-the-db).

### 5.1 Reuse deleted HNSW slots

Our hnswlib fork already supports this: [hnswalg.h:83-99](include/hnswalg.h#L83-L99) (constructor flag) and [hnswalg.h:1074-1110](include/hnswalg.h#L1074-L1110) (`addPoint(..., replace_deleted)`).

**Change** (`include/feather.h`):
- Every `HierarchicalNSW` constructor call passes `/*random_seed=*/100, /*allow_replace_deleted=*/true`. The call sites are:
  - `get_or_create_index` ([feather.h:158](include/feather.h#L158))
  - `set_int8_ram` ([feather.h:2098](include/feather.h#L2098))
  - `compact_nolock` ([feather.h:409](include/feather.h#L409))
- `add_point()` ([feather.h:180-192](include/feather.h#L180-L192)) calls `addPoint(data, id, /*replace_deleted=*/true)` **only from the serial path**. `parallel_add` passes `false`, because the fork's replace path says *"we assume that there are no concurrent operations on deleted element"* ([hnswalg.h:1100](include/hnswalg.h#L1100)). Enabling replacement inside `parallel_add` would need a stress test that proves it is safe, and that is left as a follow-up.
- **The load path**: `loadIndexStream` must repopulate `deleted_elements` when the flag is on. [hnswalg.h:939](include/hnswalg.h#L939) suggests it does; verify this with a test.

**Effect:** a steady stream of forget + add stops growing the index. The deleted-to-total ratio stays low, so auto-compaction rarely triggers. Persisting the HNSW graph (v9, which requires live == total) is also possible more often.

**Tests** (`tests/test_slot_reuse.py`):
- Add 10k records, forget 5k, add 5k new. `index_stats` must show a total element count of about 10k, not 15k. Recall@10 against brute force must stay at 0.95 or higher (use `tests/test_recall.py` helpers).
- A forgotten id that is later re-added with a new vector is searchable with that new vector.

### 5.2 Background compaction with a short swap

**Change** (`include/feather.h`):

1. **State**:
   ```cpp
   struct CompactionDelta {
       std::vector<std::pair<uint64_t, std::vector<float>>> adds;   // per modality
       std::unordered_set<uint64_t> removes;
   };
   bool compacting_ = false;
   std::unordered_map<std::string, CompactionDelta> compaction_delta_;
   std::thread compactor_;  std::mutex compactor_mx_;  std::condition_variable compactor_cv_;
   bool compaction_requested_ = false, stop_compactor_ = false;
   ```

2. **Writers record deltas.** While `compacting_` is true, `add`, `add_batch`, `forget`, `purge` and `forget_expired` also record into `compaction_delta_`. They already hold the exclusive lock, so no extra locking is needed.

3. **`compact()` becomes three steps:**
   1. **Shared lock:** snapshot the live `(id, vector)` pairs for each modality, set `compacting_ = true`, and clear the delta. Setting the flag needs a brief exclusive lock, or an atomic that writers check under their own exclusive lock.
   2. **No lock:** build the new indexes with `parallel_add`, preserving each modality's storage type (float or int8).
   3. **Exclusive lock, short:**
      - replay the delta onto the new indexes (`adds` → `add_point`, `removes` → `markDelete`)
      - swap the `index` and `space` pointers
      - erase dead metadata
      - do the incremental reverse-index cleanup from 2.2
      - set `compacting_ = false`

   While step 2 runs, readers keep using the **old** index and writers keep writing to it. The delta makes sure nothing is lost at the swap.

4. **Auto-compaction goes to the background.** `maybe_auto_compact_nolock()` no longer compacts inline. It sets `compaction_requested_` and notifies `compactor_cv_`. A worker thread started in `open()` waits on the condition variable and runs `compact()`. `close()` and `~DB()` set `stop_compactor_`, notify, and `join()` the thread.

   Manual `compact()` still runs synchronously for callers who want to wait for it, but it now uses the three-step form.

5. **Only one compaction at a time.** Guard with `compacting_`. A second request while one is running is a no-op that returns 0.

**Tests** (`tests/test_background_compaction.py`):

| Test | Asserts |
|---|---|
| `test_writes_during_compaction_are_kept` | Start compaction on 100k records, add 1k and forget 1k during it. Afterwards every added id is searchable and every forgotten id is gone. |
| `test_reads_during_compaction` | Search latency max during compaction stays under 50 ms (compare with Phase 0 `compaction_stall`) |
| `test_auto_compact_is_async` | With `set_auto_compact(0.2)`, the `forget()` that crosses the threshold returns in under 10 ms |
| `test_close_during_compaction` | `close()` joins the worker cleanly; reopen shows a consistent state |

### Phase 5 done when

- `compaction_stall` max read stall is under 50 ms at 500k records.
- Auto-compaction never blocks the caller of `forget()`.

---

## Phase 6 — Scale out: shard namespaces across processes

This phase fixes the scale half of [CONCURRENCY.md §5](CONCURRENCY.md#5-the-api-is-a-single-process-with-no-guard-against-a-second-one). The safety half was fixed in Phase 1.2. There are no engine changes.

### Design

```
                 ┌──────────────┐
  clients ──────►│   gateway    │  owner(ns) = crc32(ns) % N
                 └──┬───┬───┬───┘  cross-ns routes: fan out + merge
                    │   │   │
              ┌─────▼┐ ┌▼────┐ ┌▼─────┐
              │ api 0│ │api 1│ │api N-1│   FEATHER_SHARD_INDEX=i, FEATHER_SHARD_COUNT=N
              └──┬───┘ └──┬──┘ └──┬────┘
                 └────────┴───────┘
                 shared /data volume (or per-shard volumes)
```

### Changes

1. **`feather-api/app/db_manager.py`:**
   - Add `owner(ns) = zlib.crc32(ns.encode()) % SHARD_COUNT`. Use `zlib.crc32`, not Python's `hash()`: `hash()` is randomized per process and would route inconsistently.
   - `_load_existing()` only opens namespaces this shard owns.
   - `get(ns)` raises `NotOwned` for others, and `main.py` maps that to **HTTP 421 Misdirected Request** with the owning shard index in the body.
   - Add **idle eviction**: `close(save=True)` any namespace untouched for `FEATHER_NS_IDLE_CLOSE_S` (default 900 s). This bounds memory and makes reassigning a namespace cheap.

2. **New `feather-gateway/`**, a small async proxy built on FastAPI + httpx, or Envoy with a Lua filter:
   - It routes `/v1/{ns}/…` to shard `owner(ns)`, using the **same** `owner()` function. Put that function in a tiny shared module or keep a byte-for-byte copy, and add a test to keep the two in sync.
   - It fans out `GET /v1/namespaces`, `GET /v1/admin/overview`, `/v1/admin/metrics` and `/v1/admin/activity` to every shard and merges the results.
   - `POST /v1/namespaces` and `/v1/admin/upload` are routed by the namespace name in the body.
   - `PUT /v1/admin/embedding_config` is broadcast to all shards.
   - It adds `X-Feather-Shard` to responses to help debugging.

3. **`feather-api/docker-compose.sharded.yml`**: a gateway plus N API containers sharing one `/data` volume. This is safe because of the Phase 1.2 lock and the ownership check. Alternatively, give each shard its own volume.

4. **Resharding (changing N)** is a documented procedure in `docs/deploy-sharding.md`:
   1. Drain the gateway.
   2. Stop all shards. `close()` saves and releases every file.
   3. Change `N`.
   4. Start the shards.
   5. Resume.

   Namespace files do not move with a shared volume. Online resharding is out of scope.

**Tests:**
- `tests/test_sharding.py`:
  - `owner()` is stable across processes
  - a non-owned namespace gets a 421
  - two shards pointed at the same data directory never both open the same namespace
- Gateway integration test: spin up 2 shards and the gateway in-process on different ports. Create 20 namespaces, then check that `GET /v1/namespaces` through the gateway lists all 20 and that search works for each.

### Phase 6 done when

- Search-only requests per second through the gateway with N = 4 on a 4-core box are at least 3× a single shard.
- No namespace is ever open in two processes. The lock errors counted in logs during the soak test must be 0.

---

## Phase 7 — Decision gate: a native server?

Only start this if Phases 1–6 are done and **measured**, and one of these still holds:

- a single shard's search requests per second are CPU-bound in Python at under ~50% of what the C++ engine can serve (profile with `py-spy`), **and**
- sharding cannot add cores cheaply enough, for example because of per-shard memory cost.

If the gate is met:

- Write an async Rust server (axum + tokio) that calls the engine through the **existing C ABI**.
- First extend `src/feather_core.cpp` with the calls it lacks: namespace, entity and attribute filters; `keyword_search`; `hybrid_search`; `context_chain`; `get_metadata`; `update_metadata`; `add_batch`; `compact`; `close`.
- **Keep the engine in C++.** Rewriting it in Rust fixes nothing on this list.
- Keep the same REST contract, so the gateway, the SPA and `mcp_remote` work unchanged.

If the gate is not met, stop here. The Python API with sharding is the answer.

---

## Release plan

| Release | Contents | Compatibility notes |
|---|---|---|
| **0.19.0** | Phase 1 (durable save, file lock, `close()`, context manager) + Phase 2 | **Behaviour change:** a second open of the same file now raises; set `FEATHER_FILE_LOCK=off` to restore the old behaviour. New WAL op `PURGE` (older builds skip it). API deletes no longer rewrite the file. |
| **0.20.0** | Phase 3 group commit | **Visibility change:** a record can be visible to readers before it is durable; set `FEATHER_WAL_STRICT=1` to restore the old behaviour. |
| **0.21.0** | Phase 4 API throughput | **Breaking (API):** an embedding dimension mismatch now returns 400 instead of silently padding or truncating. New optional request fields; old clients are unaffected. |
| **0.22.0** | Phase 5 compaction | Needs no user action. `compact()` semantics are unchanged. |
| deploy-only | Phase 6 sharding | New `feather-gateway`, compose file and docs. The single-process deployment still works. |

Fill in this table from the benchmark runs:

| Metric | Phase 0 (before) | After P2 | After P3 | After P4 | After P5 | After P6 |
|---|---|---|---|---|---|---|
| Reader p99, 0 writers | | | | | | |
| Reader p99, 4 writers (sync on) | | | | | | |
| Write throughput, 16 writers | | | | | | |
| Max read stall during `compact()` | | | | | | |
| `update_metadata` @ 500k edges | | | | | | |
| API search RPS (1 process) | | | | | | |
| API search p99 with slow ingest | | | | | | |
| API search RPS (gateway, N=4) | | | | | | |

---

## Risks

| Risk | Where | Mitigation |
|---|---|---|
| The file lock breaks users who reopen without closing | Phase 1.2 | Clear error message naming `close()` and `FEATHER_FILE_LOCK=off`; migrate all in-repo callers in the same PR; CHANGELOG entry |
| `flock` is unreliable on NFS or SMB | Phase 1.2 | Document it; `FEATHER_FILE_LOCK=off` escape hatch; recommend local disks |
| A subtle race in group commit (fclose during fsync, lost wakeups) | Phase 3 | `wal_clear` waits for `!wal_syncing_`; TSan CI job; kill -9 recovery tests; strict mode as a fallback |
| Visible-before-durable surprises someone | Phase 3 | Strict-mode flag; documented in `HOW_IT_WORKS.md` and the CHANGELOG |
| Delta replay misses an operation type during background compaction | Phase 5.2 | Every mutator goes through one `record_delta()` helper; a test runs each mutator during compaction |
| `replace_deleted` degrades HNSW recall over time | Phase 5.1 | Recall test in CI; background compaction still rebuilds periodically |
| The gateway and the shards disagree on `owner()` | Phase 6 | Shared function plus a cross-process stability test; shards return 421 instead of serving the wrong namespace |
| Windows differences (`LockFileEx`, `MoveFileEx`, `_commit`) | Phases 1, 3 | Add a `windows-latest` CI leg. Root `test_*.py` scripts glob for `.so` only, so fix them to accept `.pyd` too. |

---

## Master checklist of files touched

| File | Phases |
|---|---|
| `include/feather.h` | 1.1, 1.2, 2.1, 2.2, 2.4, 3, 4.2c, 5.1, 5.2 |
| `bindings/feather.cpp` | 1.2 (`close`, `__enter__`/`__exit__`, `closed`), 2.4 (`wal_size`), 4.2 (`to_dict`, `with_cosine`) |
| `src/feather_core.cpp` | 1.2 (verify `feather_close`), 7 (only if the gate is met) |
| `feather-cli/cpp/**` | Every C++ phase, via `scripts/sync-cpp.sh` |
| `feather_db/merge.py` | 1.2 (close the source DB) |
| `feather_db/integrations/mcp_remote.py` | 4.2a (`vector_b64`) |
| `feather-api/app/db_manager.py` | 1.2 (close on adopt and delete, lock conflicts), 6 (ownership, idle close) |
| `feather-api/app/main.py` | 2.3, 2.4 (remove saves, checkpointer), 4.1, 4.2, 6 (421) |
| `feather-api/app/embedding.py` | 4.1 (httpx, async, batcher, no padding, cache, mock) |
| `feather-api/app/models.py` | 4.2 (`vector_b64`, `include_metadata`) |
| `feather-api/requirements.txt` | 4 (`httpx`, `orjson`) |
| `feather-gateway/` (new) | 6 |
| `feather-api/docker-compose.sharded.yml` (new) | 6 |
| `bench/scenarios/concurrency.py` (new), `bench/__main__.py`, `bench/api_load.md` (new) | 0 |
| `tests/test_file_lock.py`, `test_compact_consistency.py`, `test_group_commit.py`, `test_api_embedding.py`, `test_slot_reuse.py`, `test_background_compaction.py`, `test_sharding.py` (all new) | 1–6 |
| `tests/test_wal_durability.py`, `test_wal_recovery.py`, `test_graph.py`, `test_api_records.py` | 1–4 |
| `test_persist_graph.py` and other root scripts; `examples/*` | 1.2 migration (`close()` before reopen; `.pyd` glob) |
| `.github/workflows/test.yml` | 1 (Windows leg), 3 (TSan job) |
| `CHANGELOG.md`, `HOW_IT_WORKS.md`, `CONCURRENCY.md`, `CLAUDE.md`, `docs/deploy-sharding.md` (new) | Every phase |
