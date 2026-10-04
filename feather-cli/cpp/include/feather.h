#pragma once
#include <vector>
#include <algorithm>
#include <string>
#include <tuple>
#include <memory>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <queue>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <condition_variable>
#include <filesystem>
#include <limits>
#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX       // feather.h uses std::min / std::max
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>     // LockFileEx, FlushFileBuffers, MoveFileEx
#  include <io.h>          // _commit / _fileno for the WAL fsync
#else
#  include <unistd.h>      // fsync
#  include <fcntl.h>       // open
#  include <sys/file.h>    // flock
#endif
#include "hnswlib.h"
#include "metadata.h"
#include "filter.h"
#include "scoring.h"
#include <optional>

namespace feather {

// ── Phase-fair readers-writer lock ─────────────────────────────────────
// libstdc++'s std::shared_mutex is glibc's pthread_rwlock with default
// attributes, which PREFER READERS: a writer waits until no reader holds the
// lock at all. Under a steady read load that moment never comes — measured,
// 8 readers at 768-d cut a single writer from 191 to 0.6 writes/s, and
// hybrid-search readers blocked writes for the whole measurement window
// (CONCURRENCY_BASELINE.md §2.3–2.4).
//
// This lock is phase-fair: once a writer is waiting, newly arriving readers
// queue behind it; when that writer releases, every reader that queued during
// its turn is admitted before the next writer. Neither side can starve the
// other. Same interface as std::shared_mutex, so std::shared_lock /
// std::unique_lock work unchanged. NOT recursive: a thread holding the lock
// shared must never take it again (a waiting writer would deadlock it).
class FairSharedMutex {
    // state_: [W_ACTIVE | W_WAITING | reader count (30 bits)].
    // Readers take a lock-free fast path whenever no writer is waiting or
    // active: lock_shared/unlock_shared are one CAS / one fetch_sub on this
    // word. The mutex + condition variables are only used once a writer is
    // involved. (The first version took the mutex twice per read.)
    static constexpr uint32_t W_ACTIVE  = 1u << 31;
    static constexpr uint32_t W_WAITING = 1u << 30;
    static constexpr uint32_t READERS   = W_WAITING - 1u;
    std::atomic<uint32_t> state_{0};

    std::mutex m_;                          // guards the fields below
    std::condition_variable readers_cv_, writers_cv_;
    size_t waiting_readers_ = 0;
    size_t waiting_writers_ = 0;
    size_t reader_pass_     = 0;            // readers admitted ahead of waiting writers
    size_t long_readers_    = 0;            // active snapshot-style readers (save, auto_link)

    // Slow path, caller holds m_: may a reader enter now?
    bool reader_may_enter(uint32_t s) const {
        return !(s & W_ACTIVE) &&
               (!(s & W_WAITING) || reader_pass_ > 0 || long_readers_ > 0);
    }
public:
    void lock_shared() {
        uint32_t s = state_.load(std::memory_order_relaxed);
        while (!(s & (W_ACTIVE | W_WAITING))) {           // fast path: no writer around
            if (state_.compare_exchange_weak(s, s + 1, std::memory_order_acquire,
                                             std::memory_order_relaxed))
                return;
        }
        std::unique_lock<std::mutex> l(m_);                // a writer is waiting or active
        ++waiting_readers_;
        readers_cv_.wait(l, [&] { return reader_may_enter(state_.load(std::memory_order_acquire)); });
        --waiting_readers_;
        if (reader_pass_ > 0) --reader_pass_;
        state_.fetch_add(1, std::memory_order_acquire);
    }
    void unlock_shared() {
        const uint32_t s = state_.fetch_sub(1, std::memory_order_release) - 1;
        if ((s & READERS) == 0 && (s & W_WAITING)) {
            // Last reader out while a writer waits. Taking m_ before notifying
            // closes the gap between the writer's predicate check and its wait.
            std::lock_guard<std::mutex> l(m_);
            writers_cv_.notify_one();
        }
    }
    // Shared lock for long, read-only sections. Readers keep being admitted
    // while it is held; writers wait for it (and for regular readers) as usual.
    void lock_shared_long() {
        lock_shared();
        std::lock_guard<std::mutex> l(m_);
        ++long_readers_;
        readers_cv_.notify_all();          // release readers queued behind a writer
    }
    void unlock_shared_long() {
        {
            std::lock_guard<std::mutex> l(m_);
            --long_readers_;
        }
        unlock_shared();
    }
    void lock() {
        std::unique_lock<std::mutex> l(m_);
        ++waiting_writers_;
        state_.fetch_or(W_WAITING, std::memory_order_relaxed);   // closes the reader fast path
        writers_cv_.wait(l, [&] {
            const uint32_t s = state_.load(std::memory_order_acquire);
            return !(s & W_ACTIVE) && (s & READERS) == 0 && reader_pass_ == 0;
        });
        --waiting_writers_;
        // No reader can enter meanwhile: the fast path is closed by W_WAITING,
        // and the slow path needs m_, which we hold.
        state_.store(W_ACTIVE | (waiting_writers_ ? W_WAITING : 0u), std::memory_order_relaxed);
    }
    void unlock() {
        std::lock_guard<std::mutex> l(m_);
        state_.store(waiting_writers_ ? W_WAITING : 0u, std::memory_order_release);
        if (waiting_readers_ > 0) {        // readers' turn (phase-fair)
            if (waiting_writers_) reader_pass_ = waiting_readers_;
            readers_cv_.notify_all();
        } else if (waiting_writers_ > 0) {
            writers_cv_.notify_one();
        }
    }
};

// Build with -DFEATHER_STD_SHARED_MUTEX to fall back to std::shared_mutex
// (reader-preferring on Linux) — kept for A/B benchmarking only.
#ifdef FEATHER_STD_SHARED_MUTEX
using RWMutex = std::shared_mutex;
#else
using RWMutex = FairSharedMutex;
#endif

// RAII shared lock for long read-only sections (checkpoint snapshots, the
// auto_link kNN pass): other readers keep flowing while it is held.
class LongSharedLock {
    RWMutex& m_;
public:
    explicit LongSharedLock(RWMutex& m) : m_(m) {
#ifdef FEATHER_STD_SHARED_MUTEX
        m_.lock_shared();
#else
        m_.lock_shared_long();
#endif
    }
    ~LongSharedLock() {
#ifdef FEATHER_STD_SHARED_MUTEX
        m_.unlock_shared();
#else
        m_.unlock_shared_long();
#endif
    }
    LongSharedLock(const LongSharedLock&) = delete;
    LongSharedLock& operator=(const LongSharedLock&) = delete;
};

// ── Platform file helpers (durable checkpoints, single-owner lock) ─────
namespace fsio {
inline void fsync_path(const std::string& p) {
#if defined(_WIN32)
    HANDLE h = CreateFileA(p.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                           nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("fsync: cannot open " + p);
    BOOL ok = FlushFileBuffers(h);
    CloseHandle(h);
    if (!ok) throw std::runtime_error("fsync: FlushFileBuffers failed for " + p);
#else
    int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("fsync: cannot open " + p);
    int rc = ::fsync(fd);
    ::close(fd);
    if (rc != 0) throw std::runtime_error("fsync failed for " + p);
#endif
}
// Make a rename durable: on POSIX the directory entry must be synced too.
inline void fsync_parent_dir(const std::string& p) {
#if !defined(_WIN32)
    auto dir = std::filesystem::path(p).parent_path();
    std::string d = dir.empty() ? std::string(".") : dir.string();
    int fd = ::open(d.c_str(), O_RDONLY);
    if (fd >= 0) { ::fsync(fd); ::close(fd); }
#else
    (void)p;   // MOVEFILE_WRITE_THROUGH already flushed the rename
#endif
}
// Atomically replace `to` with `from`. std::rename fails on Windows when the
// target exists, and doesn't wait for the metadata write — MoveFileEx does both.
inline void atomic_replace(const std::string& from, const std::string& to) {
#if defined(_WIN32)
    if (!MoveFileExA(from.c_str(), to.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("atomic replace failed: " + from + " -> " + to);
#else
    if (std::rename(from.c_str(), to.c_str()) != 0)
        throw std::runtime_error("atomic rename failed: " + from + " -> " + to);
#endif
}
inline void fsync_file(std::FILE* f) {
#if defined(_WIN32)
    _commit(_fileno(f));
#else
    ::fsync(fileno(f));
#endif
}
inline bool exists(const std::string& p) {
    std::error_code ec;
    return std::filesystem::exists(p, ec);
}
inline uint64_t size_or_zero(const std::string& p) {
    std::error_code ec;
    auto s = std::filesystem::file_size(p, ec);
    return ec ? 0 : static_cast<uint64_t>(s);
}
} // namespace fsio

// ── Reverse-index entry: who points to a given node ──────────────
struct IncomingEdge {
    uint64_t    source_id;
    std::string rel_type;
    float       weight;
};

class DB {
private:
    struct ModalityIndex {
        std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
        std::unique_ptr<hnswlib::SpaceInterface<float>> space;  // L2Space or Int8L2Space
        size_t dim;
        bool  int8  = false;   // in-RAM int8 storage (4x smaller)
        float scale = 0.0f;    // global quant scale when int8 (= max_abs / 127)
    };

    std::unordered_map<std::string, ModalityIndex> modality_indices_;
    std::string path_;
    std::string wal_path_;

    // ── Inter-process locking ────────────────────────────────────────────
    // Held for the whole lifetime of the handle, on a sidecar `<path>.lock`.
    //
    // What this does and does not buy, because the distinction matters: it does
    // NOT make two processes able to write one file concurrently. It cannot.
    // Each DB holds the entire dataset in RAM and save_vectors() rewrites the
    // file from that in-memory view, so even a mutex around save() would have
    // process B write B's view — which never contained A's records. A's writes
    // vanish, both processes exit 0. Measured before this change: 20 of 20
    // records lost. Concurrent writing needs merge-on-save or a paged storage
    // engine; that is a format change, not a lock.
    //
    // What it does buy is ENFORCEMENT of the model that is actually safe:
    // one writer, many readers. A second writer is refused with the holder's
    // pid instead of silently destroying data, and a reader takes a shared
    // lock so it cannot be running while a writer rewrites the file underneath
    // it. Loud and recoverable beats silent and not.
    std::string lock_path_;
    int  lock_fd_   = -1;
    bool read_only_ = false;
    bool closed_    = false;
    std::string lock_key_;        // canonical path, key into the registry
    bool reentrant_ = false;      // admitted against a lock this process holds

    // Set as the very last act of load_vectors(). Guards the destructor and
    // save_vectors() so a partially-read file is never written back. See ~DB().
    bool load_complete_ = false;
    size_t default_dim_ = 768;   // dim reported before any modality index exists
    std::unordered_map<uint64_t, Metadata> metadata_store_;

    // Thread safety — one readers-writer lock per DB instance.
    // Retrieval (search / keyword_search / hybrid_search) and every const
    // accessor take it in SHARED mode, so N queries run concurrently; hnswlib's
    // searchKnn is const and its visited-list pool is internally mutex-guarded,
    // so concurrent traversal is safe. Anything that mutates the store, the
    // derived indexes, or the HNSW graph (including resizeIndex) takes it
    // EXCLUSIVELY. Retrieval's only mutation is the salience touch, which goes
    // through Metadata's mutable atomic counters — see touch_nolock().
    mutable RWMutex mutex_;
    // Serialises save() against save(): concurrent savers would collide on the
    // shared ".tmp" path and the atomic rename. Always taken BEFORE mutex_.
    mutable std::mutex save_mutex_;

    // Reverse index: target_id → list of (source_id, rel_type, weight)
    std::unordered_map<uint64_t, std::vector<IncomingEdge>> reverse_index_;

    // ── Secondary metadata indexes ───────────────────────────────────
    // Inverted indexes for O(matches) filtered lookup instead of O(n) scans.
    // Rebuilt on load (like reverse_index_), maintained incrementally on
    // mutation. Only LIVE records are indexed (forgotten/deleted excluded),
    // so the id sets double as candidate sets for filtered search.
    std::unordered_map<std::string, std::unordered_set<uint64_t>> ns_index_;     // namespace_id → ids
    std::unordered_map<std::string, std::unordered_set<uint64_t>> entity_index_; // entity_id    → ids
    std::unordered_map<std::string, std::unordered_set<uint64_t>> attr_index_;   // "key\x1fval" → ids

    // ── Auto-compaction ──────────────────────────────────────────────
    // When a modality index's deleted/total ratio crosses this threshold after
    // a forget/purge/expire, the index is rebuilt to reclaim the dead vectors.
    // 0.0 disables it (default) — compaction stays manual via compact().
    float auto_compact_ratio_ = 0.0f;

    // ── On-disk int8 quantization ────────────────────────────────────
    // Modalities whose vectors are persisted as int8 + per-vector scale (file
    // format v7) — ~4x smaller on disk and faster to load. The in-memory HNSW
    // index stays float32; vectors are dequantized on load. Opt-in per modality.
    std::unordered_set<std::string> quantized_modalities_;

    // ── In-RAM int8 quantization ─────────────────────────────────────
    // Modalities whose HNSW index stores int8[dim] vectors (4x less RAM) under a
    // global scale = max_abs/127. Must be configured before the modality's index
    // is created (i.e. before the first add). Persisted in file format v8.
    std::unordered_map<std::string, float> int8_ram_scale_;

    // ── BM25 Inverted Index ──────────────────────────────────────────
    // doc_len rides in the padding after term_freq (still 16 bytes). A doc's
    // postings are always written and retired together with its length, so
    // the copy never goes stale, and scoring needs no doc_lengths_ lookup.
    struct PostingEntry { uint64_t doc_id; uint32_t term_freq; uint32_t doc_len; };
    std::unordered_map<std::string, std::vector<PostingEntry>> bm25_index_;
    std::unordered_map<uint64_t, uint32_t> doc_lengths_;
    // Running sum of doc_lengths_ so avg_dl_ is an O(1) update per document.
    // Re-summing the whole map on every insert made indexing O(n^2): a cold
    // load of a text-bearing DB rebuilds the index document by document, so
    // load time grew quadratically (~4x per doubling) with record count.
    double total_dl_ = 0.0;
    double avg_dl_ = 0.0;
    static constexpr float BM25_K1 = 1.2f;
    static constexpr float BM25_B  = 0.75f;

    // ── WAL op codes ─────────────────────────────────────────────────
    // PURGE (0.19) is new; a pre-0.19 build replaying a WAL that contains it
    // skips the record (unknown ops fall through replay's if/else chain).
    enum class WalOp : uint8_t {
        ADD    = 0x01,
        UPDATE = 0x02,
        UIMP   = 0x03,
        LINK   = 0x04,
        FORGET = 0x05,
        PURGE  = 0x06,
    };

    // ── WAL group commit ─────────────────────────────────────────────
    // Records are appended (fwrite+fflush) inside the exclusive data lock, so
    // WAL order == apply order. The fsync happens AFTER the data lock is
    // released: a writer waits until the WAL is durable past its own record's
    // LSN, and one fsync by a "leader" covers every record appended before it
    // started. Readers are no longer blocked for the duration of an fsync, and
    // concurrent writers share fsyncs instead of queueing one each.
    // wal_mutex_ is always taken AFTER mutex_ (never the reverse).
    mutable std::mutex              wal_mutex_;
    mutable std::condition_variable wal_cv_;
    mutable uint64_t wal_written_lsn_ = 0;   // appended + flushed to the OS
    mutable uint64_t wal_synced_lsn_  = 0;   // on stable storage (or checkpointed)
    mutable bool     wal_syncing_     = false;
    mutable uint64_t wal_fsyncs_      = 0;   // stat: number of WAL fsyncs issued

    // ── Background compaction ────────────────────────────────────────
    // While a compaction builds its new indexes (without holding mutex_),
    // writers append their vector-level changes here so they can be replayed
    // onto the new indexes at the swap. Guarded by mutex_.
    struct CompactionOp {
        bool remove;                 // false = add/upsert, true = markDelete
        std::string modality;        // add only
        uint64_t id;
        std::vector<float> vec;      // add only
    };
    bool compacting_ = false;
    std::vector<CompactionOp> compaction_log_;
    std::mutex compact_run_mx_;      // one compaction at a time
    std::thread compactor_;          // started lazily by auto-compaction
    std::mutex compactor_mx_;
    std::condition_variable compactor_cv_;
    bool compaction_requested_ = false;
    bool compactor_busy_       = false;
    bool stop_compactor_       = false;

    // ── Helpers ─────────────────────────────────────────────────────

    // Default HNSW search beam width. hnswlib's built-in default is 10,
    // which gives poor recall (~0.2) at 50k+ vectors in high-dimensional
    // spaces. 50 trades ~5x more work per query for near-exact recall and
    // still leaves p99 well under 10ms in our benchmarks.
    static constexpr size_t DEFAULT_EF = 50;
    // Adaptive index capacity: start small, grow on demand via resizeIndex().
    // Old behaviour preallocated 1M elements per modality index (~hundreds of MB
    // of link_list_locks_ + data_level0_memory_ touched per index regardless of
    // how many vectors were actually stored). We now start at INITIAL_MAX and
    // double as needed, so RAM tracks the real working set, not the worst case.
    static constexpr size_t INITIAL_MAX_ELEMENTS = 4096;

    // Ensure the index can hold at least `target` elements. NOT thread-safe
    // (resizeIndex reallocs every backing buffer) — call before any add, and
    // before parallel_add for the full batch size, never from inside it.
    static void reserve(ModalityIndex& m_idx, size_t target) {
        size_t cap = m_idx.index->getMaxElements();
        if (target > cap)
            m_idx.index->resizeIndex(std::max(target, cap * 2));
    }

    // Every index is created with allow_replace_deleted: a new insert reuses the
    // slot of a vector that forget()/purge() marked deleted instead of growing
    // the index, so tombstones stop accumulating and compaction is needed far
    // less often.
    static std::unique_ptr<hnswlib::HierarchicalNSW<float>>
    make_hnsw(hnswlib::SpaceInterface<float>* space, size_t capacity) {
        auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space, capacity, 16, 200, /*random_seed=*/100, /*allow_replace_deleted=*/true);
        index->setEf(DEFAULT_EF);
        return index;
    }

    ModalityIndex& get_or_create_index(const std::string& modality, size_t dim) {
        auto it = modality_indices_.find(modality);
        if (it == modality_indices_.end()) {
            auto cfg = int8_ram_scale_.find(modality);
            bool int8 = cfg != int8_ram_scale_.end();
            float scale = int8 ? cfg->second : 0.0f;
            std::unique_ptr<hnswlib::SpaceInterface<float>> space;
            if (int8) space = std::make_unique<hnswlib::Int8L2Space>(dim, scale);
            else      space = std::make_unique<hnswlib::L2Space>(dim);
            auto index = make_hnsw(space.get(), INITIAL_MAX_ELEMENTS);
            modality_indices_[modality] = {std::move(index), std::move(space), dim, int8, scale};
            return modality_indices_[modality];
        }
        return it->second;
    }

    // Build the int8 storage payload for one vector under a global scale.
    static void quantize_global(const float* v, size_t dim, float scale, int8_t* out) {
        float inv = (scale > 0.0f) ? 1.0f / scale : 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            long q = std::lround(v[i] * inv);
            if (q >  127) q =  127;
            if (q < -127) q = -127;
            out[i] = static_cast<int8_t>(q);
        }
    }

    // Insert raw storage-format bytes. `reuse_slot` lets a NEW label take over a
    // deleted slot. It is only ever passed from single-threaded paths under the
    // exclusive lock: hnswlib's replace path assumes no concurrent operation on
    // the slot being reused, so parallel_add never reuses. An existing label is
    // always updated in place (reusing a slot for it would orphan its old node
    // and return it twice from search).
    static void insert_raw(ModalityIndex& m_idx, uint64_t id, const void* data, bool reuse_slot) {
        auto& idx = *m_idx.index;
        bool reuse = reuse_slot && idx.getDeletedCount() > 0 &&
                     idx.label_lookup_.find(id) == idx.label_lookup_.end();
        idx.addPoint(data, id, reuse);
    }

    // Insert a float vector into a modality index, quantizing to int8 first if
    // the modality is in-RAM int8. Centralises the float-vs-int8 store decision.
    static void add_point(ModalityIndex& m_idx, uint64_t id, const float* vec,
                          bool reuse_slot = false) {
        if (m_idx.int8) {
            // Reusable per-thread buffer — a fresh std::vector per call across the
            // parallel insert pool churns the allocator and inflates RSS by ~MBs.
            static thread_local std::vector<int8_t> q;
            q.resize(m_idx.dim);
            quantize_global(vec, m_idx.dim, m_idx.scale, q.data());
            insert_raw(m_idx, id, q.data(), reuse_slot);
        } else {
            insert_raw(m_idx, id, vec, reuse_slot);
        }
    }

    // Encode a query in the modality's storage format (int8 blob or float bytes)
    // so it can be passed straight to searchKnn / the distance function.
    static std::vector<char> encode_query(const ModalityIndex& m_idx, const float* q) {
        if (m_idx.int8) {
            std::vector<char> blob(m_idx.dim);
            quantize_global(q, m_idx.dim, m_idx.scale,
                            reinterpret_cast<int8_t*>(blob.data()));
            return blob;
        }
        const char* p = reinterpret_cast<const char*>(q);
        return std::vector<char>(p, p + m_idx.dim * sizeof(float));
    }

    // Read a stored vector back as float32 (dequantizing if the modality is int8).
    static std::vector<float> read_vector_internal(const ModalityIndex& m_idx, size_t internal_id) {
        const char* raw = m_idx.index->getDataByInternalId(internal_id);
        std::vector<float> out(m_idx.dim);
        if (m_idx.int8) {
            const int8_t* q = reinterpret_cast<const int8_t*>(raw);
            for (size_t i = 0; i < m_idx.dim; ++i) out[i] = static_cast<float>(q[i]) * m_idx.scale;
        } else {
            const float* f = reinterpret_cast<const float*>(raw);
            for (size_t i = 0; i < m_idx.dim; ++i) out[i] = f[i];
        }
        return out;
    }

    // Read a stored vector back as float32 by external id. Throws if absent.
    static std::vector<float> read_vector_label(const ModalityIndex& m_idx, uint64_t id) {
        if (m_idx.int8) {
            auto q = m_idx.index->template getDataByLabel<int8_t>(id);  // dim int8s
            std::vector<float> out(q.size());
            for (size_t i = 0; i < q.size(); ++i)
                out[i] = static_cast<float>(q[i]) * m_idx.scale;
            return out;
        }
        return m_idx.index->template getDataByLabel<float>(id);
    }

    void build_reverse_index() {
        reverse_index_.clear();
        for (const auto& [id, meta] : metadata_store_) {
            for (const auto& e : meta.edges) {
                reverse_index_[e.target_id].push_back({id, e.rel_type, e.weight});
            }
        }
    }

    // ── int8 quantization helpers (per-vector symmetric scalar) ──────
    // scale = max|v| / 127; q_i = round(v_i / scale) clamped to [-127,127].
    // Returns the scale needed to reconstruct. Lossy but typically retains
    // very high recall for embedding-shaped vectors.
    static float quantize_vec(const float* v, size_t dim, int8_t* out) {
        float amax = 0.0f;
        for (size_t i = 0; i < dim; ++i) amax = std::max(amax, std::fabs(v[i]));
        float scale = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        float inv   = 1.0f / scale;
        for (size_t i = 0; i < dim; ++i) {
            long q = std::lround(v[i] * inv);
            if (q >  127) q =  127;
            if (q < -127) q = -127;
            out[i] = static_cast<int8_t>(q);
        }
        return scale;
    }
    static void dequantize_vec(const int8_t* q, size_t dim, float scale, float* out) {
        for (size_t i = 0; i < dim; ++i) out[i] = static_cast<float>(q[i]) * scale;
    }

    // Insert many (id, vector) pairs into a modality index using a thread pool.
    // hnswlib addPoint is thread-safe (per-node link locks + label/lookup locks),
    // so the graph is built concurrently. Caller must guarantee exclusive
    // structural access (no concurrent resize) — true at load and batch-ingest,
    // and we never exceed max_elements here so no resize is triggered.
    // Thread count for a parallel build of n items: every core, but at least two
    // items per thread. Chunked add_batch relies on the full width to keep each
    // exclusive section short.
    static size_t build_threads(size_t n) {
        unsigned hw = std::thread::hardware_concurrency();
        size_t nthreads = std::min<size_t>(hw ? hw : 4, std::max<size_t>(1, n / 2));
        if (const char* env = std::getenv("FEATHER_LOAD_THREADS")) {
            long v = std::atol(env);                // override / cap thread count
            if (v >= 1) nthreads = std::min<size_t>(static_cast<size_t>(v), nthreads);
        }
        return nthreads;
    }

    template <class Fn>
    static void run_parallel(size_t n, Fn&& fn, size_t max_threads = 0) {
        size_t nthreads = build_threads(n);
        if (max_threads > 0) nthreads = std::min(nthreads, max_threads);
        if (nthreads <= 1) { for (size_t i = 0; i < n; ++i) fn(i); return; }
        std::atomic<size_t> next{0};
        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        for (size_t t = 0; t < nthreads; ++t) {
            pool.emplace_back([&]() {
                size_t i;
                while ((i = next.fetch_add(1)) < n) fn(i);
            });
        }
        for (auto& th : pool) th.join();
    }

    static void parallel_add(ModalityIndex& m_idx,
                             std::vector<std::pair<uint64_t, std::vector<float>>>& items) {
        run_parallel(items.size(), [&](size_t i) {
            add_point(m_idx, items[i].first, items[i].second.data());
        });
    }

    // Same, from vectors already in the index's storage format (float or int8
    // bytes, `stride` bytes each) — used by compaction to rebuild without a
    // dequantize/requantize round trip.
    static void parallel_add_raw(ModalityIndex& m_idx, const std::vector<uint64_t>& ids,
                                 const std::vector<char>& data, size_t stride,
                                 size_t max_threads = 0) {
        run_parallel(ids.size(), [&](size_t i) {
            m_idx.index->addPoint(data.data() + i * stride, ids[i], false);
        }, max_threads);
    }

    // Background compaction builds on at most half the cores by default
    // (FEATHER_COMPACT_THREADS): with every core busy rebuilding, concurrent
    // queries were measured at p50 12 ms instead of 0.5 ms at 768-d.
    static size_t compact_threads() {
        static const size_t c = [] {
            if (const char* e = std::getenv("FEATHER_COMPACT_THREADS")) {
                long v = std::atol(e);
                if (v >= 1) return static_cast<size_t>(v);
            }
            unsigned hw = std::thread::hardware_concurrency();
            return std::max<size_t>(1, (hw ? hw : 4) / 2);
        }();
        return c;
    }

    // ── Secondary index helpers ──────────────────────────────────────
    static std::string attr_key(const std::string& k, const std::string& v) {
        // Unit separator (0x1f) can't appear in normal keys/values, so this
        // composite key is collision-free across different (k,v) splits.
        return k + std::string(1, '\x1f') + v;
    }

    static bool is_dead_meta(const Metadata& m) {
        if (m.source == "_forgotten") return true;
        auto it = m.attributes.find("_deleted");
        return it != m.attributes.end() && it->second == "true";
    }

    void index_meta(uint64_t id, const Metadata& m) {
        if (!m.namespace_id.empty()) ns_index_[m.namespace_id].insert(id);
        if (!m.entity_id.empty())    entity_index_[m.entity_id].insert(id);
        for (const auto& [k, v] : m.attributes) attr_index_[attr_key(k, v)].insert(id);
    }

    void deindex_meta(uint64_t id, const Metadata& m) {
        auto drop = [id](std::unordered_map<std::string, std::unordered_set<uint64_t>>& idx,
                         const std::string& key) {
            auto it = idx.find(key);
            if (it == idx.end()) return;
            it->second.erase(id);
            if (it->second.empty()) idx.erase(it);
        };
        if (!m.namespace_id.empty()) drop(ns_index_, m.namespace_id);
        if (!m.entity_id.empty())    drop(entity_index_, m.entity_id);
        for (const auto& [k, v] : m.attributes) drop(attr_index_, attr_key(k, v));
    }

    void build_secondary_indexes() {
        ns_index_.clear();
        entity_index_.clear();
        attr_index_.clear();
        for (const auto& [id, meta] : metadata_store_) {
            if (is_dead_meta(meta)) continue;   // candidate sets are live-only
            index_meta(id, meta);
        }
    }

    // Candidate ids for a filter's INDEXED fields (namespace/entity/attributes),
    // computed as the intersection of the relevant secondary-index sets.
    // Sets `indexed` = true if the filter constrained at least one indexed field
    // (so the caller knows the result is an authoritative candidate set rather
    // than "no constraint"). An empty return with indexed=true means the filter
    // genuinely matches nothing.
    // Returns a plain vector: a single-set filter (e.g. one namespace) is a
    // linear copy instead of re-hashing every id into a new unordered_set, and
    // a multi-set filter probes the larger sets for each id of the smallest.
    std::vector<uint64_t>
    candidates_for_filter(const SearchFilter& f, bool& indexed) const {
        indexed = false;
        std::vector<const std::unordered_set<uint64_t>*> sets;
        auto pick = [&](const std::unordered_map<std::string, std::unordered_set<uint64_t>>& idx,
                        const std::string& key) -> bool {
            indexed = true;
            auto it = idx.find(key);
            if (it == idx.end()) return false;   // signals empty intersection
            sets.push_back(&it->second);
            return true;
        };
        if (f.namespace_id && !pick(ns_index_, *f.namespace_id))     return {};
        if (f.entity_id    && !pick(entity_index_, *f.entity_id))    return {};
        if (f.attributes_match)
            for (const auto& [k, v] : *f.attributes_match)
                if (!pick(attr_index_, attr_key(k, v)))              return {};

        if (!indexed) return {};                 // no indexed constraint at all

        // Intersect smallest-first to minimise work.
        std::sort(sets.begin(), sets.end(),
                  [](auto* a, auto* b) { return a->size() < b->size(); });
        std::vector<uint64_t> result;
        result.reserve(sets[0]->size());
        for (uint64_t id : *sets[0]) {
            bool all = true;
            for (size_t i = 1; i < sets.size() && all; ++i) all = sets[i]->count(id) > 0;
            if (all) result.push_back(id);
        }
        return result;
    }

    // ── Compaction support ───────────────────────────────────────────
    // Caller holds mutex_ exclusively. Record a vector-level change made while a
    // background compaction is building its new indexes, so the swap can replay
    // it (see compact()).
    void log_compaction_add(const std::string& modality, uint64_t id, const float* v, size_t dim) {
        if (!compacting_) return;
        compaction_log_.push_back({false, modality, id, std::vector<float>(v, v + dim)});
    }
    void log_compaction_remove(uint64_t id) {
        if (!compacting_) return;
        compaction_log_.push_back({true, std::string(), id, {}});
    }

    // Drop reverse-index entries into and out of `gone` (ids whose metadata was
    // erased). One pass over the reverse index instead of a full rebuild.
    void prune_reverse_index(const std::unordered_set<uint64_t>& gone) {
        if (gone.empty()) return;
        for (uint64_t id : gone) reverse_index_.erase(id);
        for (auto it = reverse_index_.begin(); it != reverse_index_.end(); ) {
            auto& v = it->second;
            v.erase(std::remove_if(v.begin(), v.end(),
                    [&](const IncomingEdge& ie) { return gone.count(ie.source_id) > 0; }), v.end());
            if (v.empty()) it = reverse_index_.erase(it); else ++it;
        }
    }

    // Caller holds mutex_. If any modality's deleted/total ratio has crossed the
    // configured threshold, ask the background compactor to rebuild. This used
    // to compact INLINE, so whichever forget() crossed the threshold blocked for
    // the whole rebuild (measured: 35 s at 100k x 128) with the DB frozen.
    void maybe_auto_compact_nolock() {
        if (auto_compact_ratio_ <= 0.0f || compacting_ || closed_) return;
        for (const auto& [name, m_idx] : modality_indices_) {
            size_t total = m_idx.index->getCurrentElementCount();
            if (total == 0) continue;
            float ratio = static_cast<float>(m_idx.index->getDeletedCount())
                        / static_cast<float>(total);
            if (ratio >= auto_compact_ratio_) { request_compaction(); return; }
        }
    }

    void request_compaction() {
        std::lock_guard<std::mutex> g(compactor_mx_);
        if (stop_compactor_) return;
        compaction_requested_ = true;
        if (!compactor_.joinable())
            compactor_ = std::thread([this] { compactor_loop(); });
        compactor_cv_.notify_one();
    }

    void compactor_loop() {
        std::unique_lock<std::mutex> g(compactor_mx_);
        for (;;) {
            compactor_cv_.wait(g, [&] { return compaction_requested_ || stop_compactor_; });
            if (stop_compactor_) return;
            compaction_requested_ = false;
            compactor_busy_ = true;
            g.unlock();
            try { compact(); } catch (...) {}
            g.lock();
            compactor_busy_ = false;
            compactor_cv_.notify_all();   // wake wait_for_compaction()
        }
    }

    // Stop and join the background compactor. Must be called WITHOUT mutex_
    // held (the worker may be waiting for it inside compact()).
    void stop_compactor() {
        {
            std::lock_guard<std::mutex> g(compactor_mx_);
            stop_compactor_ = true;
        }
        compactor_cv_.notify_all();
        if (compactor_.joinable() && compactor_.get_id() != std::this_thread::get_id())
            compactor_.join();
    }

    static const std::unordered_set<std::string>& stop_words() {
        static const std::unordered_set<std::string> sw = {
            "a","an","the","and","or","but","in","on","at","to","for",
            "of","with","by","from","is","are","was","were","be","been",
            "have","has","had","do","does","did","will","would","could",
            "should","may","might","shall","can","not","no","it","its",
            "this","that","these","those","i","me","my","we","us","our",
            "you","your","he","him","his","she","her","they","them","their"
        };
        return sw;
    }

    static std::vector<std::string> tokenize(const std::string& text) {
        std::vector<std::string> tokens;
        std::string tok;
        const auto& sw = stop_words();
        for (unsigned char c : text) {
            if (std::isalnum(c)) {
                tok += static_cast<char>(std::tolower(c));
            } else {
                if (tok.size() >= 2 && sw.find(tok) == sw.end())
                    tokens.push_back(tok);
                tok.clear();
            }
        }
        if (tok.size() >= 2 && sw.find(tok) == sw.end())
            tokens.push_back(tok);
        return tokens;
    }

    void recompute_avg_dl() {
        avg_dl_ = doc_lengths_.empty()
                ? 1.0
                : total_dl_ / static_cast<double>(doc_lengths_.size());
    }

    // Drop a document from the BM25 index. `old_content` must be the content
    // that was indexed for this id: its tokens name exactly which posting lists
    // to touch, making removal O(terms in doc). Pass nullptr only when the old
    // content genuinely isn't known — that falls back to a scan of the entire
    // vocabulary, which is O(corpus) and should stay off any hot path.
    void remove_from_bm25_index(uint64_t id, const std::string* old_content) {
        auto len_it = doc_lengths_.find(id);
        if (len_it == doc_lengths_.end()) return;     // not indexed
        total_dl_ -= static_cast<double>(len_it->second);
        doc_lengths_.erase(len_it);

        auto drop_from = [id](std::vector<PostingEntry>& postings) {
            postings.erase(
                std::remove_if(postings.begin(), postings.end(),
                    [id](const PostingEntry& p) { return p.doc_id == id; }),
                postings.end());
        };

        if (old_content) {
            std::unordered_set<std::string> terms;
            for (auto& t : tokenize(*old_content)) terms.insert(std::move(t));
            for (const auto& t : terms) {
                auto pit = bm25_index_.find(t);
                if (pit == bm25_index_.end()) continue;
                drop_from(pit->second);
                if (pit->second.empty()) bm25_index_.erase(pit);
            }
        } else {
            for (auto pit = bm25_index_.begin(); pit != bm25_index_.end(); ) {
                drop_from(pit->second);
                if (pit->second.empty()) pit = bm25_index_.erase(pit);
                else                     ++pit;
            }
        }
        recompute_avg_dl();
    }

    // Index (or re-index) one document. `old_content` is the previously indexed
    // content when this id is being replaced — supplying it keeps re-indexing
    // proportional to the document rather than to the corpus.
    void add_to_bm25_index(uint64_t id, const std::string& content,
                           const std::string* old_content = nullptr) {
        // Re-indexing an existing doc: retire its old postings first. Blanked
        // content (forget()) therefore also removes the doc, as it should.
        if (doc_lengths_.count(id)) remove_from_bm25_index(id, old_content);
        if (content.empty()) return;
        auto tokens = tokenize(content);
        if (tokens.empty()) return;

        // Count term frequencies
        std::unordered_map<std::string, uint32_t> tf;
        for (const auto& t : tokens) tf[t]++;

        // Update doc length and posting lists
        doc_lengths_[id] = static_cast<uint32_t>(tokens.size());
        total_dl_ += static_cast<double>(tokens.size());
        for (const auto& [term, freq] : tf)
            bm25_index_[term].push_back({id, freq, static_cast<uint32_t>(tokens.size())});

        recompute_avg_dl();   // O(1) — running total, not a scan
    }

    void rebuild_bm25_index() {
        bm25_index_.clear();
        doc_lengths_.clear();
        total_dl_ = 0.0;
        avg_dl_   = 0.0;
        for (const auto& [id, meta] : metadata_store_) {
            if (is_dead_meta(meta)) continue;   // forgotten/deleted stay unsearchable
            add_to_bm25_index(id, meta.content);
        }
    }

    // ── Touch — call from within already-locked methods ─────────────
    // const, and safe under the SHARED lock: Metadata's salience counters are
    // mutable atomics, so concurrent queries can record their hits without
    // serialising on the writer lock. Writers are excluded meanwhile, so the
    // map itself can't rehash underneath us.
    void touch_nolock(uint64_t id) const {
        auto it = metadata_store_.find(id);
        if (it != metadata_store_.end())
            it->second.note_recall(static_cast<uint64_t>(std::time(nullptr)));
    }

    // ── WAL helpers ──────────────────────────────────────────────────
    // A single WAL record's payload can't plausibly exceed this. The bound is
    // what stops a corrupt/torn length field from being taken at face value —
    // see the guard in replay_wal().
    static constexpr uint32_t MAX_WAL_PAYLOAD = 256u * 1024 * 1024;

    // WAL on-disk format. v1 (<=0.17.0) was a bare record stream with no header
    // and no checksum; v2 prefixes WAL_MAGIC + version and appends a CRC32 to
    // every record. replay_wal() sniffs the header so a WAL left behind by an
    // older build still recovers — the crash it exists for must not be made
    // worse by the upgrade that fixes it.
    //   v2 record: [op:1][id:8][plen:4][payload:plen][crc32:4]
    //   crc32 covers op + id + plen + payload.
    static constexpr uint32_t WAL_MAGIC   = 0x4C415746u; // "FWAL" little-endian
    static constexpr uint32_t WAL_VERSION = 2u;

    // CRC-32 (IEEE 802.3, reflected). A length guard only catches a *short*
    // tail; it cannot see a record whose bytes were mangled in place. Without a
    // checksum such a record replays as plausible garbage — a corrupt vector or
    // a metadata blob that deserializes into nonsense — and is then written
    // into the next checkpoint as if it were real. The checksum turns silent
    // corruption into a clean stop at the last known-good record.
    static uint32_t crc32_of(const void* data, size_t len, uint32_t crc = 0xFFFFFFFFu) {
        static const auto table = [] {
            std::array<uint32_t, 256> t{};
            for (uint32_t i = 0; i < 256; ++i) {
                uint32_t c = i;
                for (int k = 0; k < 8; ++k)
                    c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
                t[i] = c;
            }
            return t;
        }();
        const auto* p = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < len; ++i)
            crc = table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
        return crc;
    }

    // The WAL handle is held open for the life of the DB. Reopening the file for
    // every single append cost an open+close syscall pair per record, which is
    // paid once per item inside add_batch — i.e. on the bulk-ingest hot path.
    // A C stdio handle rather than std::ofstream because durability needs the
    // underlying descriptor: fflush() only pushes into the kernel, and there is
    // no portable way to reach fileno() through a std::ofstream.
    mutable std::FILE* wal_file_ = nullptr;

    // fsync policy. Default on: a WAL whose writes are only flush()ed survives
    // the process dying but NOT the machine or container dying, which is most
    // of what a write-ahead log is for. FEATHER_WAL_SYNC=0 trades that back for
    // throughput on hosts where the caller accepts the risk.
    static bool wal_sync_enabled() {
        static const bool on = [] {
            const char* e = std::getenv("FEATHER_WAL_SYNC");
            return !(e && (e[0] == '0' || e[0] == 'n' || e[0] == 'N'));
        }();
        return on;
    }

    // FEATHER_WAL_STRICT=1: keep the pre-0.19 visibility guarantee — a write is
    // fsynced BEFORE the exclusive lock is released, so no reader can ever see
    // a record that a machine crash could still lose. Concurrent writers still
    // share fsyncs (group commit), but readers wait for the fsync again.
    static bool wal_strict() {
        static const bool on = [] {
            const char* e = std::getenv("FEATHER_WAL_STRICT");
            return e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y' || e[0] == 't' || e[0] == 'T');
        }();
        return on;
    }

    std::string wal_old_path() const { return wal_path_ + ".old"; }

    // Caller holds wal_mutex_.
    bool wal_open_for_append() const {
        if (wal_file_) return true;
        bool fresh = true;
        {   // a zero-length or missing file needs the v2 header written first
            std::ifstream probe(wal_path_, std::ios::binary | std::ios::ate);
            if (probe && probe.tellg() > 0) fresh = false;
        }
        wal_file_ = std::fopen(wal_path_.c_str(), "ab");
        if (!wal_file_) return false;
        if (fresh) {
            uint32_t magic = WAL_MAGIC, ver = WAL_VERSION;
            std::fwrite(&magic, 4, 1, wal_file_);
            std::fwrite(&ver,   4, 1, wal_file_);
        }
        return true;
    }

    // Append one record and flush it to the OS. Returns its LSN (0 when there
    // is no WAL). Called with mutex_ held exclusively, so WAL order is exactly
    // the order mutations are applied in memory. Durability is established
    // separately by wal_wait_durable(), normally after mutex_ is released.
    uint64_t wal_append(WalOp op, uint64_t id, const std::string& payload) {
        if (wal_path_.empty()) return 0;
        std::lock_guard<std::mutex> g(wal_mutex_);
        if (!wal_open_for_append()) return 0;

        auto op_b = static_cast<uint8_t>(op);
        uint32_t plen = static_cast<uint32_t>(payload.size());

        uint32_t crc = 0xFFFFFFFFu;
        crc = crc32_of(&op_b, 1, crc);
        crc = crc32_of(&id,   8, crc);
        crc = crc32_of(&plen, 4, crc);
        if (plen > 0) crc = crc32_of(payload.data(), plen, crc);
        crc ^= 0xFFFFFFFFu;

        std::fwrite(&op_b, 1, 1, wal_file_);
        std::fwrite(&id,   8, 1, wal_file_);
        std::fwrite(&plen, 4, 1, wal_file_);
        if (plen > 0) std::fwrite(payload.data(), 1, plen, wal_file_);
        std::fwrite(&crc,  4, 1, wal_file_);
        std::fflush(wal_file_);
        return ++wal_written_lsn_;
    }

    // Group commit. Block until every WAL record up to `lsn` is on stable
    // storage. The first waiter becomes the leader and fsyncs everything
    // appended so far — outside wal_mutex_, so other writers keep appending —
    // while the rest wait on the condition variable and are released by that
    // single fsync. N concurrent writers pay ~1 fsync instead of N serial ones.
    void wal_wait_durable(uint64_t lsn) const {
        if (lsn == 0 || !wal_sync_enabled()) return;
        std::unique_lock<std::mutex> lk(wal_mutex_);
        while (wal_synced_lsn_ < lsn) {
            if (wal_syncing_) { wal_cv_.wait(lk); continue; }
            if (!wal_file_) {                      // checkpointed away meanwhile
                wal_synced_lsn_ = std::max(wal_synced_lsn_, wal_written_lsn_);
                break;
            }
            wal_syncing_ = true;
            const uint64_t target = wal_written_lsn_;
            std::FILE* fh = wal_file_;             // stays open: close/rotate wait
            lk.unlock();                           //   for !wal_syncing_
            fsio::fsync_file(fh);
            lk.lock();
            ++wal_fsyncs_;
            wal_syncing_    = false;
            wal_synced_lsn_ = std::max(wal_synced_lsn_, target);
            wal_cv_.notify_all();
        }
    }

    // Wait until no fsync is in flight. Caller holds `lk` on wal_mutex_.
    void wal_quiesce(std::unique_lock<std::mutex>& lk) const {
        wal_cv_.wait(lk, [&] { return !wal_syncing_; });
    }

    // ── Advisory file lock ───────────────────────────────────────────────

    // ── In-process lock registry ─────────────────────────────────────────
    // flock() conflicts between two open file descriptions even inside ONE
    // process, which would make the lock a breaking change: the Python binding
    // holds DB with py::nodelete, so `del db` never runs ~DB(), and any program
    // that reopened its own file — including most of our own tests — would be
    // refused by a lock it already held.
    //
    // So the lock is REENTRANT per process. The first handle takes the flock;
    // further handles on the same path in the same process are admitted against
    // a refcount. The guarantee that matters is unchanged, because the hazard
    // this fixes is cross-process: two processes, each with a full in-memory
    // copy, each rewriting the whole file (measured: 20 of 20 records lost,
    // both exiting 0). Same-process double-open remains the documented footgun
    // it already was — use one handle and separate agents by namespace, or
    // build(db=...) — and it is the caller's own state, not a silent collision
    // with a process they cannot see.
    static std::mutex& registry_mutex() {
        static std::mutex m;
        return m;
    }
    static std::unordered_map<std::string, int>& registry() {
        static std::unordered_map<std::string, int> held;
        return held;
    }

    /// Resolve to a canonical key so "./a.feather" and "a.feather" are one entry.
    ///
    /// Canonicalises the DIRECTORY and appends the basename, never the file
    /// itself. realpath() on the file only succeeds once the file exists, and on
    /// macOS it rewrites /var/folders/... to /private/var/folders/... — so
    /// keying on the file gave one key before the first save and a different one
    /// after. Reentrancy then missed, the handle took a second flock against one
    /// this same process already held, and the failure was reported as "another
    /// process (pid <ourselves>)". The directory exists in both cases, so this
    /// key is stable across the file's creation.
    static std::string lock_key(const std::string& path) {
#ifndef _WIN32
        size_t slash = path.find_last_of('/');
        std::string dir  = (slash == std::string::npos) ? "." : path.substr(0, slash);
        std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
        if (dir.empty()) dir = "/";
        char buf[4096];
        if (::realpath(dir.c_str(), buf)) return std::string(buf) + "/" + base;
#endif
        return path;
    }

    static bool locking_enabled() {
        const char* e = std::getenv("FEATHER_LOCK");
        // Opt-out exists because flock() is unreliable on NFS and some
        // container overlay mounts, where a spurious failure to lock would be
        // worse than no lock at all.
        return !(e && (e[0] == '0' || e[0] == 'n' || e[0] == 'N'));
    }

    /// Read the pid recorded by whoever holds the lock, for the error message.
    /// Best-effort: an empty or unreadable file just yields "unknown".
    std::string lock_holder() const {
        std::ifstream f(lock_path_);
        std::string pid;
        if (f && std::getline(f, pid) && !pid.empty()) return pid;
        return "unknown";
    }

    void acquire_lock(bool exclusive) {
        if (!locking_enabled()) return;
        lock_path_ = path_ + ".lock";
        lock_key_  = lock_key(path_);
        {
            std::lock_guard<std::mutex> g(registry_mutex());
            auto it = registry().find(lock_key_);
            if (it != registry().end() && it->second > 0) {
                it->second += 1;        // this process already holds it
                reentrant_ = true;
                return;
            }
        }
#ifndef _WIN32
        int fd = ::open(lock_path_.c_str(), O_RDWR | O_CREAT, 0644);
        if (fd < 0) return;        // unwritable dir: proceed rather than refuse
        if (::flock(fd, (exclusive ? LOCK_EX : LOCK_SH) | LOCK_NB) != 0) {
            std::string holder = lock_holder();
            ::close(fd);
            // flock() conflicts between two open file descriptions even in ONE
            // process, and that is the mistake people actually make — so say so
            // rather than reporting "another process (pid <yourself>)".
            // Backstop for the case the registry missed: if the lock file
            // names our own pid we already hold this lock, so admit the handle
            // reentrantly instead of reporting ourselves as another process.
            if (holder == std::to_string(static_cast<long>(::getpid()))) {
                std::lock_guard<std::mutex> g(registry_mutex());
                registry()[lock_key_] += 1;
                reentrant_ = true;
                return;
            }
            throw std::runtime_error(
                std::string(exclusive
                    ? "Cannot open '" + path_ + "' for writing: another process "
                      "(pid " + holder + ") holds the write lock.\n"
                    : "Cannot open '" + path_ + "' for reading: a writer "
                      "(pid " + holder + ") holds it exclusively.\n") +
                "Feather is single-writer: one process writes, many may read. "
                "Set FEATHER_LOCK=0 to disable this check (you then own the "
                "consequence: concurrent writers silently discard each other's "
                "records).");
        }
        lock_fd_ = fd;
        {
            std::lock_guard<std::mutex> g(registry_mutex());
            registry()[lock_key_] = 1;
        }
        if (exclusive) {           // record who holds it, for the next caller
            if (::ftruncate(fd, 0) == 0) {
                std::string pid = std::to_string(static_cast<long>(::getpid())) + "\n";
                ssize_t n = ::write(fd, pid.c_str(), pid.size());
                (void)n;           // advisory only; a short write costs nothing
            }
        }
#else
        // Windows: LockFileEx on a sidecar handle. Same semantics, different API.
        HANDLE h = CreateFileA(lock_path_.c_str(), GENERIC_READ | GENERIC_WRITE,
                               FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                               OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (h == INVALID_HANDLE_VALUE) return;
        OVERLAPPED ov = {};
        DWORD flags = LOCKFILE_FAIL_IMMEDIATELY | (exclusive ? LOCKFILE_EXCLUSIVE_LOCK : 0);
        if (!LockFileEx(h, flags, 0, 1, 0, &ov)) {
            CloseHandle(h);
            throw std::runtime_error("Cannot open '" + path_ +
                "': another process holds the lock. Feather is single-writer.");
        }
        lock_fd_ = static_cast<int>(reinterpret_cast<intptr_t>(h));
#endif
    }

    /// Refuse a mutation on a read-only handle.
    ///
    /// Checked at the mutation rather than at save() because a read-only handle
    /// that accepts add() and only complains at checkpoint time has already lied
    /// to the caller: the record looked stored, the process exits, nothing is
    /// there. Failing at the call site makes the mistake findable.
    void check_writable(const char* what) const {
        if (closed_)
            throw std::runtime_error(
                std::string("Cannot ") + what + ": '" + path_ + "' is closed.");
        if (read_only_)
            throw std::runtime_error(
                std::string("Cannot ") + what + ": '" + path_ +
                "' was opened read-only. Reopen with read_only=false (and make "
                "sure no other process holds the write lock).");
    }

public:
    bool is_read_only() const { return read_only_; }
    bool is_closed()    const { return closed_; }

private:
    void release_lock() {
        if (!lock_key_.empty()) {
            std::lock_guard<std::mutex> g(registry_mutex());
            auto it = registry().find(lock_key_);
            if (it != registry().end() && --it->second > 0) {
                lock_key_.clear();
                return;             // other handles in this process still open
            }
            if (it != registry().end()) registry().erase(it);
            lock_key_.clear();
        }
        if (reentrant_ || lock_fd_ < 0) return;
#ifndef _WIN32
        ::flock(lock_fd_, LOCK_UN);
        ::close(lock_fd_);
#else
        HANDLE h = reinterpret_cast<HANDLE>(static_cast<intptr_t>(lock_fd_));
        OVERLAPPED ov = {};
        UnlockFileEx(h, 0, 1, 0, &ov);
        CloseHandle(h);
#endif
        lock_fd_ = -1;
        // The .lock file itself is left in place deliberately. flock state
        // lives on the descriptor, not the inode, so unlinking it races with a
        // waiter that has already opened it and would hand two processes
        // independent locks on two different inodes.
    }

    void wal_close() const {
        std::unique_lock<std::mutex> lk(wal_mutex_);
        wal_quiesce(lk);
        if (wal_file_) { std::fclose(wal_file_); wal_file_ = nullptr; }
    }

    // Checkpoint completed: every record so far is in the (durable) base file.
    // Caller must hold mutex_ so no append can race the removal.
    void wal_clear() const {
        if (wal_path_.empty()) return;
        std::unique_lock<std::mutex> lk(wal_mutex_);
        wal_quiesce(lk);
        if (wal_file_) { std::fclose(wal_file_); wal_file_ = nullptr; }
        std::remove(wal_path_.c_str());
        std::remove(wal_old_path().c_str());
        wal_synced_lsn_ = wal_written_lsn_;
        wal_cv_.notify_all();
    }

    // Start of a non-blocking checkpoint (caller holds mutex_, so no appends
    // race it). Make the current WAL durable, then rename it to <wal>.old so new
    // writes go to a fresh WAL while the base file is being made durable
    // outside the lock. Recovery replays <wal>.old then <wal> on top of
    // whichever base file survived — replay is idempotent, so both orders of
    // "crash before/after the base-file rename" recover correctly.
    // Returns false if a previous checkpoint left a <wal>.old behind; the caller
    // then does a fully synchronous checkpoint instead.
    bool wal_rotate() const {
        if (wal_path_.empty()) return true;
        std::unique_lock<std::mutex> lk(wal_mutex_);
        wal_quiesce(lk);
        if (fsio::exists(wal_old_path())) return false;
        if (wal_file_) {
            std::fflush(wal_file_);
            if (wal_sync_enabled()) { fsio::fsync_file(wal_file_); ++wal_fsyncs_; }
            std::fclose(wal_file_);
            wal_file_ = nullptr;
        }
        wal_synced_lsn_ = wal_written_lsn_;     // all records now durable
        wal_cv_.notify_all();
        if (fsio::exists(wal_path_))
            fsio::atomic_replace(wal_path_, wal_old_path());
        return true;
    }

    void replay_wal(const std::string& path) {
        if (path.empty()) return;
        std::ifstream wf(path, std::ios::binary);
        if (!wf) return;

        // Size the file once so each record's declared length can be sanity
        // checked against what's actually left to read.
        wf.seekg(0, std::ios::end);
        const std::streamoff wal_size = wf.tellg();
        wf.seekg(0, std::ios::beg);

        // Sniff the format. v2 opens with WAL_MAGIC + version and carries a
        // CRC32 per record; anything else is a v1 WAL from <=0.17.0, which is
        // read exactly as before. Upgrading must not strand a WAL that a crash
        // already left on disk. 'F' (0x46) can never be a v1 opcode, so the
        // discrimination is unambiguous rather than heuristic.
        bool has_crc = false;
        {
            uint32_t magic = 0;
            if (wal_size >= 8 && wf.read(reinterpret_cast<char*>(&magic), 4) &&
                magic == WAL_MAGIC) {
                uint32_t ver = 0;
                wf.read(reinterpret_cast<char*>(&ver), 4);
                if (ver > WAL_VERSION) return;   // written by a newer build
                has_crc = true;
            } else {
                wf.clear();
                wf.seekg(0, std::ios::beg);      // v1: no header, rewind
            }
        }

        while (true) {
            uint8_t op_b; uint64_t id; uint32_t plen;
            if (!wf.read(reinterpret_cast<char*>(&op_b), 1)) break;
            if (!wf.read(reinterpret_cast<char*>(&id),   8)) break;
            if (!wf.read(reinterpret_cast<char*>(&plen), 4)) break;
            // Guard BEFORE allocating. A torn or corrupt length field otherwise
            // becomes a multi-GB std::string: a 13-byte truncated WAL declaring
            // a 4.29 GB payload drove peak RSS to 1.5 GB and would OOM a
            // memory-capped container on startup. The base-file loader already
            // guards dim/element_count exactly this way; the WAL did not.
            // A bad length means the tail is garbage, so stop replaying here and
            // keep everything recovered so far.
            const std::streamoff remaining = wal_size - wf.tellg();
            const std::streamoff need = static_cast<std::streamoff>(plen) + (has_crc ? 4 : 0);
            if (plen > MAX_WAL_PAYLOAD || need > remaining)
                break;
            std::string payload(plen, '\0');
            if (plen > 0 && !wf.read(&payload[0], plen)) break;

            // Verify before applying. A mangled record must not be replayed and
            // then persisted into the next checkpoint as though it were real —
            // stop at the last record that still checks out.
            if (has_crc) {
                uint32_t stored = 0;
                if (!wf.read(reinterpret_cast<char*>(&stored), 4)) break;
                uint32_t crc = 0xFFFFFFFFu;
                crc = crc32_of(&op_b, 1, crc);
                crc = crc32_of(&id,   8, crc);
                crc = crc32_of(&plen, 4, crc);
                if (plen > 0) crc = crc32_of(payload.data(), plen, crc);
                crc ^= 0xFFFFFFFFu;
                if (crc != stored) break;
            }

            std::istringstream ss(payload);
            auto op = static_cast<WalOp>(op_b);

            if (op == WalOp::ADD) {
                uint16_t mod_len = 0;
                ss.read(reinterpret_cast<char*>(&mod_len), 2);
                std::string modality(mod_len, '\0');
                if (mod_len > 0) ss.read(&modality[0], mod_len);
                uint32_t dim32 = 0;
                ss.read(reinterpret_cast<char*>(&dim32), 4);
                std::vector<float> vec(dim32);
                ss.read(reinterpret_cast<char*>(vec.data()), dim32 * 4);
                Metadata meta = Metadata::deserialize(ss);
                auto& m_idx = get_or_create_index(modality, dim32);
                if (dim32 != m_idx.dim) continue;   // corrupt / mismatched record
                reserve(m_idx, m_idx.index->getCurrentElementCount() + 1);
                try { add_point(m_idx, id, vec.data(), /*reuse_slot=*/true); } catch (...) {}
                metadata_store_[id] = std::move(meta);

            } else if (op == WalOp::UPDATE) {
                Metadata meta = Metadata::deserialize(ss);
                metadata_store_[id] = std::move(meta);

            } else if (op == WalOp::UIMP) {
                float imp = 0.0f;
                ss.read(reinterpret_cast<char*>(&imp), 4);
                auto it = metadata_store_.find(id);
                if (it != metadata_store_.end()) it->second.importance = imp;

            } else if (op == WalOp::LINK) {
                uint64_t to_id = 0;
                ss.read(reinterpret_cast<char*>(&to_id), 8);
                uint8_t rel_len = 0;
                ss.read(reinterpret_cast<char*>(&rel_len), 1);
                std::string rel_type(rel_len, '\0');
                if (rel_len > 0) ss.read(&rel_type[0], rel_len);
                float weight = 1.0f;
                ss.read(reinterpret_cast<char*>(&weight), 4);
                auto it = metadata_store_.find(id);
                if (it != metadata_store_.end()) {
                    bool exists = false;
                    for (const auto& e : it->second.edges)
                        if (e.target_id == to_id && e.rel_type == rel_type) { exists = true; break; }
                    if (!exists) {
                        it->second.edges.push_back({to_id, rel_type, weight});
                        reverse_index_[to_id].push_back({id, rel_type, weight});
                    }
                }

            } else if (op == WalOp::FORGET) {
                for (auto& [name, m_idx] : modality_indices_) {
                    try { m_idx.index->markDelete(id); } catch (...) {}
                }
                auto it = metadata_store_.find(id);
                if (it != metadata_store_.end()) {
                    it->second.content    = "";
                    it->second.source     = "_forgotten";
                    it->second.importance = 0.0f;
                    it->second.ttl        = 0;
                }

            } else if (op == WalOp::PURGE) {
                uint16_t ns_len = 0;
                ss.read(reinterpret_cast<char*>(&ns_len), 2);
                std::string ns(ns_len, '\0');
                if (ns_len > 0) ss.read(&ns[0], ns_len);
                purge_core(ns, /*maintain_derived=*/false);
            }
        }
        // Derived indexes are built once by load_vectors() after this returns —
        // rebuilding them here too would do the expensive BM25 pass twice.
    }

    static std::string escape_json(const std::string& s) {
        std::string out;
        out.reserve(s.size() + 4);
        for (unsigned char c : s) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:
                    if (c < 0x20) {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                        out += buf;
                    } else {
                        out += c;
                    }
            }
        }
        return out;
    }

    // ── Persistence ─────────────────────────────────────────────────

    // Should checkpoints fsync the new base file? Shares the FEATHER_WAL_SYNC
    // switch: with it off, the caller has already accepted flush-only writes.
    static bool checkpoint_sync_enabled() { return wal_sync_enabled(); }

    // Make a fully-written .tmp snapshot the durable base file.
    //   1. fsync the snapshot  2. atomically replace  3. fsync the directory
    // Before 0.19 none of these syncs happened: after a power loss the renamed
    // file could come back empty while the WAL had already been deleted.
    void publish_snapshot(const std::string& tmp_path) const {
        if (checkpoint_sync_enabled()) fsio::fsync_path(tmp_path);
        fsio::atomic_replace(tmp_path, path_);
        if (checkpoint_sync_enabled()) fsio::fsync_parent_dir(path_);
    }

    // Fully synchronous checkpoint (caller holds mutex_, shared or exclusive):
    // snapshot, publish, then drop both WAL files.
    void save_vectors() const {
        check_writable("save");
        write_snapshot(path_ + ".tmp");
        publish_snapshot(path_ + ".tmp");
        wal_clear();
    }

    // Serialize the whole DB to `tmp_path`. Caller holds mutex_ (shared is
    // enough). Throws on any write error — including a full disk — so a
    // truncated snapshot can never be published.
    void write_snapshot(const std::string& tmp_path) const {
        // A DB whose load threw holds a fragment of the file, not its contents.
        // Writing that back is data loss, so refuse loudly rather than silently
        // truncating. The destructor checks the same flag before calling in.
        if (!load_complete_)
            throw std::logic_error(
                "refusing to save a DB whose load did not complete — "
                "this would overwrite " + path_ + " with a partial read");

        // Atomic save: write to .tmp, then rename — prevents corruption on crash
        std::ofstream f(tmp_path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot save to temp file: " + tmp_path);

        uint32_t magic   = 0x46454154; // "FEAT"
        uint32_t version = 9;          // v7: on-disk int8; v8: in-RAM int8 flag+scale; v9: persisted HNSW graph
        f.write((char*)&magic,   4);
        f.write((char*)&version, 4);

        // Build the set of valid IDs — exclude _forgotten and _deleted.
        // This makes forget()/purge() actually persist across save+reload.
        auto is_dead = [](const Metadata& m) -> bool {
            if (m.source == "_forgotten") return true;
            auto it = m.attributes.find("_deleted");
            return it != m.attributes.end() && it->second == "true";
        };
        std::unordered_set<uint64_t> valid_ids;
        valid_ids.reserve(metadata_store_.size());
        for (const auto& [id, meta] : metadata_store_) {
            if (!is_dead(meta)) valid_ids.insert(id);
        }

        // Metadata section — only write live records
        uint32_t meta_count = static_cast<uint32_t>(valid_ids.size());
        f.write((char*)&meta_count, 4);
        for (const auto& [id, meta] : metadata_store_) {
            if (!valid_ids.count(id)) continue;
            f.write((char*)&id, 8);
            meta.serialize(f);
        }

        // Modality indices section — only write vectors whose ID is live
        uint32_t modal_count = static_cast<uint32_t>(modality_indices_.size());
        f.write((char*)&modal_count, 4);
        for (const auto& [name, m_idx] : modality_indices_) {
            uint16_t name_len = static_cast<uint16_t>(name.size());
            f.write((char*)&name_len, 2);
            f.write(name.data(), name_len);
            uint32_t dim32 = static_cast<uint32_t>(m_idx.dim);
            f.write((char*)&dim32, 4);

            uint8_t quant = quantized_modalities_.count(name) ? 1 : 0;
            f.write((char*)&quant, 1);
            // v8: persist in-RAM int8 mode + its global scale so reload restores it
            uint8_t int8ram = m_idx.int8 ? 1 : 0;
            f.write((char*)&int8ram, 1);
            if (int8ram) f.write((char*)&m_idx.scale, 4);

            size_t total = m_idx.index->cur_element_count;
            uint32_t live_count = 0;
            for (size_t i = 0; i < total; ++i) {
                uint64_t id = m_idx.index->getExternalLabel(i);
                if (valid_ids.count(id)) live_count++;
            }

            // v9 fast path: persist the prebuilt HNSW graph verbatim so load()
            // restores it instead of rebuilding from vectors (~10x faster cold
            // load, and keeps the higher-quality serial-build graph). Only safe
            // when the graph holds exactly the live set (no forgotten/purged
            // nodes to filter) and the modality isn't on-disk-quantized (which
            // re-encodes vectors to int8, incompatible with the graph blob).
            uint8_t persist_graph = (live_count == total && !quant) ? 1 : 0;
            f.write((char*)&persist_graph, 1);
            if (persist_graph) {
                m_idx.index->saveIndexStream(f);
                continue;
            }

            f.write((char*)&live_count, 4);
            std::vector<int8_t> qbuf(quant ? m_idx.dim : 0);
            for (size_t i = 0; i < total; ++i) {
                uint64_t id = m_idx.index->getExternalLabel(i);
                if (!valid_ids.count(id)) continue;
                // dequantize int8 nodes to float; on-disk quant is independent
                std::vector<float> data = read_vector_internal(m_idx, i);
                f.write((char*)&id, 8);
                if (quant) {
                    float scale = quantize_vec(data.data(), m_idx.dim, qbuf.data());
                    f.write((char*)&scale, 4);
                    f.write((char*)qbuf.data(), m_idx.dim);   // dim bytes
                } else {
                    f.write((char*)data.data(), m_idx.dim * sizeof(float));
                }
            }
        }
        f.flush();
        if (!f) {
            f.close();
            std::remove(tmp_path.c_str());
            throw std::runtime_error("write failed (disk full?) while saving " + tmp_path);
        }
        f.close();
        if (!f) {
            std::remove(tmp_path.c_str());
            throw std::runtime_error("close failed while saving " + tmp_path);
        }
    }

    // Open a namespace: read the base .feather (if any), then replay the WAL on
    // top, then build the derived indexes once over the combined result.
    //
    // The base file being absent or unreadable is NOT a reason to skip WAL
    // replay. A namespace that has never been save()d has no base file at all,
    // yet its WAL may hold every write it has ever received — which is exactly
    // the state the throttled-import path leaves a fresh namespace in for up to
    // FEATHER_IMPORT_SAVE_INTERVAL_S. Returning early there silently discarded
    // a complete, on-disk WAL and lost every record in it.
    void load_vectors() {
        read_base_file();
        // Crash recovery — runs even with no base file. <wal>.old exists only if
        // a checkpoint was interrupted between rotating the WAL and deleting
        // the old one; its records are older than everything in <wal>.
        replay_wal(wal_old_path());
        replay_wal(wal_path_);
        build_reverse_index();
        build_secondary_indexes();
        rebuild_bm25_index();
        load_complete_ = true; // last statement: a throw anywhere above leaves
                               // this false and the destructor won't checkpoint
    }

    void read_base_file() {
        std::ifstream f(path_, std::ios::binary);
        if (!f) return;

        uint32_t magic, version;
        f.read((char*)&magic,   4);
        f.read((char*)&version, 4);
        if (magic != 0x46454154) return;

        if (version == 2) {
            // v2: single "text" index, metadata interleaved with vectors
            uint32_t dim32;
            f.read((char*)&dim32, 4);
            auto& m_idx = get_or_create_index("text", dim32);
            uint64_t id;
            std::vector<float> vec(dim32);
            while (f.read((char*)&id, 8)) {
                Metadata meta = Metadata::deserialize(f);
                f.read((char*)vec.data(), dim32 * sizeof(float));
                reserve(m_idx, m_idx.index->getCurrentElementCount() + 1);
                add_point(m_idx, id, vec.data());
                metadata_store_[id] = std::move(meta);
            }
        } else if (version >= 3) {
            // v3/v4/v5: separate metadata section then modality indices
            uint32_t meta_count;
            f.read((char*)&meta_count, 4);
            // A corrupt count is a loop trip count, so an absurd value means
            // meta_count iterations of deserialize() against a stream that ran
            // out — each one allocating from garbage lengths. Observed: a forged
            // count of 0xFFFFFF00 left the process spinning for >10 minutes
            // instead of failing. The smallest possible record is id(8) plus a
            // minimal Metadata, so anything claiming more records than there are
            // bytes left is corrupt by arithmetic, not by heuristic.
            {
                const std::streamoff here = f.tellg();
                f.seekg(0, std::ios::end);
                const std::streamoff end = f.tellg();
                f.seekg(here, std::ios::beg);
                if (static_cast<std::streamoff>(meta_count) > (end - here) / 8)
                    throw std::runtime_error(
                        "corrupt .feather: metadata count " + std::to_string(meta_count) +
                        " exceeds remaining file size");
            }
            for (uint32_t i = 0; i < meta_count; ++i) {
                uint64_t id;
                if (!f.read((char*)&id, 8))
                    throw std::runtime_error("corrupt .feather: truncated metadata section");
                metadata_store_[id] = Metadata::deserialize(f);
            }
            uint32_t modal_count;
            f.read((char*)&modal_count, 4);
            for (uint32_t m = 0; m < modal_count; ++m) {
                uint16_t name_len;
                f.read((char*)&name_len, 2);
                std::string name(name_len, ' ');
                f.read(&name[0], name_len);
                uint32_t dim32, element_count;
                f.read((char*)&dim32, 4);
                // Guard: a corrupt/forged header with an absurd dim would make
                // index creation (and per-vector buffers) allocate gigabytes.
                // No real embedding approaches 2^20 dims.
                if (dim32 == 0 || dim32 > (1u << 20))
                    throw std::runtime_error("corrupt .feather: implausible vector dim "
                                             + std::to_string(dim32));
                uint8_t quant = 0;
                if (version >= 7) f.read((char*)&quant, 1);
                uint8_t int8ram = 0;
                float   int8scale = 0.0f;
                if (version >= 8) {
                    f.read((char*)&int8ram, 1);
                    if (int8ram) f.read((char*)&int8scale, 4);
                }
                uint8_t persist_graph = 0;
                if (version >= 9) f.read((char*)&persist_graph, 1);
                // configure int8-RAM BEFORE the index is created so it is built
                // as an int8 index; vectors below are re-quantized via add_point.
                if (int8ram) int8_ram_scale_[name] = int8scale;
                auto& m_idx = get_or_create_index(name, dim32);
                if (quant) quantized_modalities_.insert(name);

                if (persist_graph) {
                    // v9: restore the prebuilt HNSW graph verbatim — no rebuild.
                    // The blob carries the base layer (vectors) + link lists; the
                    // space matches (Int8L2Space if int8ram, else L2Space).
                    m_idx.index->loadIndexStream(f, m_idx.space.get(), 0);
                    m_idx.index->setEf(DEFAULT_EF);
                    continue;
                }

                // Read all vectors serially (sequential I/O), then build the
                // HNSW graph in parallel — graph construction dominates load.
                f.read((char*)&element_count, 4);
                // Guard: reject an element_count the file is too small to back,
                // BEFORE reserving/allocating for it. Each on-disk element is at
                // least id(8B) + (quant ? scale(4B)+dim : dim*4) bytes.
                {
                    std::streampos cur = f.tellg();
                    f.seekg(0, std::ios::end);
                    std::streamoff remaining = (f.tellg() >= cur) ? (f.tellg() - cur) : -1;
                    f.seekg(cur);
                    size_t min_elem = 8 + (quant ? ((size_t)dim32 + 4) : ((size_t)dim32 * 4));
                    if (remaining >= 0 &&
                        (uint64_t)element_count > (uint64_t)remaining / std::max<size_t>(min_elem, 1))
                        throw std::runtime_error("corrupt .feather: element_count "
                            + std::to_string(element_count) + " exceeds file size");
                }
                std::vector<std::pair<uint64_t, std::vector<float>>> items;
                items.reserve(element_count);
                std::vector<int8_t> qbuf(quant ? dim32 : 0);
                for (uint32_t i = 0; i < element_count; ++i) {
                    uint64_t id;
                    f.read((char*)&id, 8);
                    std::vector<float> vec(dim32);
                    if (quant) {
                        float scale = 1.0f;
                        f.read((char*)&scale, 4);
                        f.read((char*)qbuf.data(), dim32);   // dim bytes
                        dequantize_vec(qbuf.data(), dim32, scale, vec.data());
                    } else {
                        f.read((char*)vec.data(), dim32 * sizeof(float));
                    }
                    items.emplace_back(id, std::move(vec));
                }
                reserve(m_idx, m_idx.index->getCurrentElementCount() + items.size());
                parallel_add(m_idx, items);
            }
        }

    }

public:
    // ─────────────────────────────────────────────────────────────────
    // Factory
    // ─────────────────────────────────────────────────────────────────
private:
    void ensure_open() const {
        if (closed_) throw std::logic_error("DB is closed: " + path_);
    }

public:
    /// Open a database. `read_only` takes a SHARED lock, so any number of
    /// readers may hold the file at once; the default takes an EXCLUSIVE one and
    /// refuses if another process already has it. See the lock_path_ comment for
    /// why this is enforcement of single-writer rather than concurrent writing.
    static std::unique_ptr<DB> open(const std::string& path, size_t default_dim = 768,
                                    bool read_only = false) {
        auto db = std::make_unique<DB>();
        db->path_        = path;
        db->wal_path_    = path + ".wal";
        db->default_dim_ = default_dim;
        db->read_only_   = read_only;
        // Before load_vectors(), for two reasons: a refused handle should not
        // have parsed a 2 GB file first, and replay_wal() WRITES (it applies the
        // log and then clears it), so it must not run on a file another process
        // is mid-save on.
        db->acquire_lock(!read_only);
        db->load_vectors();
        // Intentionally do NOT pre-create the "text" index. An empty HNSW index
        // preallocates ~70MB (1M-element link locks etc.); pre-creating it forced
        // set_int8_ram()/set_quantized() to build a *second* index, doubling RAM.
        // The modality is created lazily on the first add() — by which point
        // set_int8_ram() has taken effect — and dim() falls back to default_dim_.
        return db;
    }

    // ─────────────────────────────────────────────────────────────────
    // Ingestion
    // ─────────────────────────────────────────────────────────────────
private:
    static std::string encode_add(const std::string& modality,
                                  const std::vector<float>& vec, const Metadata& meta) {
        std::ostringstream ws;
        uint16_t mod_len = static_cast<uint16_t>(modality.size());
        ws.write(reinterpret_cast<const char*>(&mod_len), 2);
        ws.write(modality.data(), mod_len);
        uint32_t dim32 = static_cast<uint32_t>(vec.size());
        ws.write(reinterpret_cast<const char*>(&dim32), 4);
        ws.write(reinterpret_cast<const char*>(vec.data()), vec.size() * 4);
        meta.serialize(ws);
        return ws.str();
    }

    // Validate a vector against its modality BEFORE anything is logged: a WAL
    // record with the wrong dim would be replayed into the index on the next
    // open and read past the end of its vector.
    void check_dim_nolock(const std::string& modality, size_t dim) const {
        if (dim == 0) throw std::runtime_error("empty vector for modality " + modality);
        auto it = modality_indices_.find(modality);
        if (it != modality_indices_.end() && it->second.dim != dim)
            throw std::runtime_error("Dimension mismatch for modality " + modality +
                                     ": got " + std::to_string(dim) + ", index has " +
                                     std::to_string(it->second.dim));
    }

    // Metadata half of an upsert (caller holds mutex_ exclusively).
    void apply_meta_upsert_nolock(uint64_t id, const Metadata& meta) {
        std::string prev_content;
        bool had_prev = false;
        auto it = metadata_store_.find(id);
        if (it != metadata_store_.end()) {
            deindex_meta(id, it->second);   // drop stale secondary-index entries
            prev_content = it->second.content;   // needed to retire old BM25 postings
            had_prev     = true;
            Metadata combined = meta;
            if (combined.edges.empty() && !it->second.edges.empty()) {
                combined.edges = it->second.edges;   // edges kept: reverse index unchanged
            } else if (!meta.edges.empty()) {
                // Edges replaced: keep the reverse index exact (it used to go
                // stale until the next reload, so get_incoming() missed them).
                for (const auto& e : it->second.edges) {
                    auto rit = reverse_index_.find(e.target_id);
                    if (rit == reverse_index_.end()) continue;
                    auto& v = rit->second;
                    v.erase(std::remove_if(v.begin(), v.end(),
                            [id](const IncomingEdge& ie) { return ie.source_id == id; }), v.end());
                    if (v.empty()) reverse_index_.erase(rit);
                }
                for (const auto& e : meta.edges)
                    reverse_index_[e.target_id].push_back({id, e.rel_type, e.weight});
            }
            it->second = std::move(combined);
        } else {
            it = metadata_store_.emplace(id, meta).first;
            for (const auto& e : meta.edges)
                reverse_index_[e.target_id].push_back({id, e.rel_type, e.weight});
        }
        const bool dead = is_dead_meta(it->second);
        if (!dead) index_meta(id, it->second);
        // A dead record must not be keyword-searchable either.
        add_to_bm25_index(id, dead ? std::string() : meta.content,
                          had_prev ? &prev_content : nullptr);
    }

    // add_batch builds the graph in chunks and releases the exclusive lock
    // between them, so readers get a turn at least every chunk. One 1,000-vector
    // batch used to hold the lock for the whole parallel build: measured, reads
    // fell 99% while batches ran (CONCURRENCY_BASELINE.md section 2.3).
    static size_t batch_chunk() {
        static const size_t c = [] {
            if (const char* e = std::getenv("FEATHER_BATCH_CHUNK")) {
                long v = std::atol(e);
                if (v >= 1) return static_cast<size_t>(v);
            }
            unsigned hw = std::thread::hardware_concurrency();
            return std::max<size_t>(64, 2 * static_cast<size_t>(hw ? hw : 4));
        }();
        return c;
    }

public:
    void add(uint64_t id, const std::vector<float>& vec,
             const Metadata& meta = Metadata(),
             const std::string& modality = "text") {
        check_writable("add a record");
        const std::string payload = encode_add(modality, vec, meta);   // no lock needed
        uint64_t lsn = 0;
        {
            std::unique_lock<RWMutex> lock(mutex_);
            ensure_open();
            check_dim_nolock(modality, vec.size());
            // WAL: log before mutating in-memory state
            lsn = wal_append(WalOp::ADD, id, payload);

            auto& m_idx = get_or_create_index(modality, vec.size());
            reserve(m_idx, m_idx.index->getCurrentElementCount() + 1);
            add_point(m_idx, id, vec.data(), /*reuse_slot=*/true);
            log_compaction_add(modality, id, vec.data(), vec.size());
            apply_meta_upsert_nolock(id, meta);
            if (wal_strict()) wal_wait_durable(lsn);
        }
        wal_wait_durable(lsn);   // group commit: fsync outside the data lock
    }

    // Bulk insert. Same per-item semantics as add(), but the HNSW graph (the
    // expensive part) is built in PARALLEL, much faster for bulk ingestion.
    // `metas` may be empty (default Metadata for all) or must match ids.size().
    // The batch is applied in chunks (FEATHER_BATCH_CHUNK, default
    // max(64, 2 x cores)); readers may observe a partially-applied batch. The
    // call returns once the whole batch is durable (one fsync, not one per record).
    void add_batch(const std::vector<uint64_t>& ids,
                   const std::vector<std::vector<float>>& vecs,
                   const std::vector<Metadata>& metas,
                   const std::string& modality = "text") {
        check_writable("add a batch");
        const size_t n = ids.size();
        if (n == 0) return;
        if (vecs.size() != n)
            throw std::runtime_error("add_batch: ids and vecs size mismatch");
        if (!metas.empty() && metas.size() != n)
            throw std::runtime_error("add_batch: metas size mismatch");
        const size_t dim = vecs[0].size();
        for (const auto& v : vecs)
            if (v.size() != dim)
                throw std::runtime_error("Dimension mismatch for modality " + modality);
        {
            std::shared_lock<RWMutex> lock(mutex_);
            ensure_open();
            check_dim_nolock(modality, dim);
        }
        static const Metadata kDefault;
        const size_t chunk = batch_chunk();
        uint64_t last_lsn = 0;

        for (size_t s = 0; s < n; s += chunk) {
            const size_t e = std::min(n, s + chunk);
            std::vector<std::string> payloads;
            payloads.reserve(e - s);
            for (size_t i = s; i < e; ++i)
                payloads.push_back(encode_add(modality, vecs[i], metas.empty() ? kDefault : metas[i]));

            std::unique_lock<RWMutex> lock(mutex_);
            ensure_open();
            check_dim_nolock(modality, dim);
            for (size_t i = s; i < e; ++i) {
                last_lsn = wal_append(WalOp::ADD, ids[i], payloads[i - s]);
                apply_meta_upsert_nolock(ids[i], metas.empty() ? kDefault : metas[i]);
                log_compaction_add(modality, ids[i], vecs[i].data(), dim);
            }
            auto& m_idx = get_or_create_index(modality, dim);
            reserve(m_idx, m_idx.index->getCurrentElementCount() + (e - s));
            run_parallel(e - s, [&](size_t k) {          // concurrent graph construction
                add_point(m_idx, ids[s + k], vecs[s + k].data());
            });
            if (wal_strict()) wal_wait_durable(last_lsn);
        }
        wal_wait_durable(last_lsn);   // one fsync for the whole batch
    }

    // ─────────────────────────────────────────────────────────────────
    // Salience
    // ─────────────────────────────────────────────────────────────────
    void touch(uint64_t id) {
        std::shared_lock<RWMutex> lock(mutex_);   // atomic counters
        touch_nolock(id);
    }

    // ─────────────────────────────────────────────────────────────────
    // Graph: link
    // ─────────────────────────────────────────────────────────────────
    static std::string encode_link(uint64_t to_id, const std::string& rel_type, float weight) {
        std::ostringstream ws;
        ws.write(reinterpret_cast<const char*>(&to_id), 8);
        auto rel_len = static_cast<uint8_t>(std::min(rel_type.size(), size_t(255)));
        ws.write(reinterpret_cast<const char*>(&rel_len), 1);
        ws.write(rel_type.data(), rel_len);
        ws.write(reinterpret_cast<const char*>(&weight), 4);
        return ws.str();
    }

    void link(uint64_t from_id, uint64_t to_id,
              const std::string& rel_type = "related_to",
              float weight = 1.0f) {
        check_writable("create an edge");
        const std::string payload = encode_link(to_id, rel_type, weight);
        uint64_t lsn = 0;
        {
            std::unique_lock<RWMutex> lock(mutex_);
            ensure_open();
            auto it = metadata_store_.find(from_id);
            if (it == metadata_store_.end()) return;

            for (const auto& e : it->second.edges)
                if (e.target_id == to_id && e.rel_type == rel_type) return;

            lsn = wal_append(WalOp::LINK, from_id, payload);
            it->second.edges.push_back({to_id, rel_type, weight});
            reverse_index_[to_id].push_back({from_id, rel_type, weight});
            if (wal_strict()) wal_wait_durable(lsn);
        }
        wal_wait_durable(lsn);
    }

    // ─────────────────────────────────────────────────────────────────
    // Graph: query edges
    // ─────────────────────────────────────────────────────────────────
    std::vector<Edge> get_edges(uint64_t id) const {
        std::shared_lock<RWMutex> lock(mutex_);
        auto it = metadata_store_.find(id);
        if (it == metadata_store_.end()) return {};
        return it->second.edges;
    }

    std::vector<IncomingEdge> get_incoming(uint64_t id) const {
        std::shared_lock<RWMutex> lock(mutex_);
        auto it = reverse_index_.find(id);
        if (it == reverse_index_.end()) return {};
        return it->second;
    }

    // ─────────────────────────────────────────────────────────────────
    // Graph: auto-link by vector similarity
    // ─────────────────────────────────────────────────────────────────
    // `threshold` is a COSINE similarity in [-1, 1], and it now means that.
    //
    // It used to be compared against 1/(1+L2_squared), which is not a similarity
    // at all: the documented default of 0.80 actually required cosine >= 0.875,
    // and on unnormalised embeddings it was unreachable — measured, 20 documents
    // at true cosine 0.9971 produced ZERO links, silently. auto_link is the
    // primitive the clustering and entity work sits on, so a threshold that
    // quietly means something else makes the whole feature look broken.
    //
    // The cosine is computed from the vectors rather than derived from the L2
    // distance, because the shortcut cos = 1 - L2sq/2 only holds for unit
    // vectors and nothing in Feather enforces normalisation. The cost is one
    // dot product per candidate inside a loop that was already O(n*candidates)
    // under the exclusive lock.
    //
    // The kNN pass (the expensive part) runs under the SHARED lock, so queries
    // continue meanwhile; only applying the new edges takes the exclusive lock.
    // Every edge is WAL-logged (it used to be in-memory only until the next
    // save()), with one fsync for the whole pass.
    size_t auto_link(const std::string& modality = "text",
                     float threshold = 0.80f,
                     const std::string& rel_type = "related_to",
                     size_t candidates = 15) {
        struct Cand { uint64_t from, to; float sim; };
        std::vector<Cand> found;

        auto cosine = [&](const std::vector<float>& a, const std::vector<float>& b) -> float {
            if (a.size() != b.size() || a.empty()) return 0.0f;
            double dot = 0.0, na = 0.0, nb = 0.0;
            for (size_t d = 0; d < a.size(); ++d) {
                dot += static_cast<double>(a[d]) * b[d];
                na  += static_cast<double>(a[d]) * a[d];
                nb  += static_cast<double>(b[d]) * b[d];
            }
            if (na <= 0.0 || nb <= 0.0) return 0.0f;
            return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
        };

        {
            LongSharedLock lock(mutex_);   // long pass: queries keep flowing
            ensure_open();
            auto m_it = modality_indices_.find(modality);
            if (m_it == modality_indices_.end()) return 0;
            auto& m_idx = m_it->second;
            size_t n = m_idx.index->cur_element_count;
            for (size_t i = 0; i < n; ++i) {
                if (m_idx.index->isMarkedDeleted(i)) continue;
                uint64_t from_id = m_idx.index->getExternalLabel(i);
                auto mit = metadata_store_.find(from_id);
                if (mit == metadata_store_.end() || is_dead_meta(mit->second)) continue;
                // stored data is already in the index's storage format (float or
                // int8), so it can be used directly as the query.
                const void* qdata = m_idx.index->getDataByInternalId(i);
                std::vector<float> from_vec;
                try { from_vec = read_vector_label(m_idx, from_id); } catch (...) { continue; }
                auto res = m_idx.index->searchKnn(qdata, candidates + 1);
                while (!res.empty()) {
                    auto [dist, to_id] = res.top(); res.pop();
                    if (to_id == from_id) continue;
                    std::vector<float> to_vec;
                    try { to_vec = read_vector_label(m_idx, to_id); } catch (...) { continue; }
                    float sim = cosine(from_vec, to_vec);   // true cosine, not 1/(1+L2)
                    if (sim >= threshold) found.push_back({from_id, to_id, sim});
                }
            }
        }
        size_t links_created = 0;
        uint64_t last_lsn = 0;
        {
            std::unique_lock<RWMutex> lock(mutex_);
            ensure_open();
            for (const auto& c : found) {
                auto mit = metadata_store_.find(c.from);   // may have changed meanwhile
                if (mit == metadata_store_.end() || is_dead_meta(mit->second)) continue;
                auto& meta = mit->second;
                bool exists = false;
                for (const auto& e : meta.edges)
                    if (e.target_id == c.to && e.rel_type == rel_type) { exists = true; break; }
                if (exists) continue;
                last_lsn = wal_append(WalOp::LINK, c.from, encode_link(c.to, rel_type, c.sim));
                meta.edges.push_back({c.to, rel_type, c.sim});
                reverse_index_[c.to].push_back({c.from, rel_type, c.sim});
                ++links_created;
            }
            if (wal_strict()) wal_wait_durable(last_lsn);
        }
        wal_wait_durable(last_lsn);
        return links_created;
    }

    // ─────────────────────────────────────────────────────────────────
    // Context Chain: vector search + n-hop graph expansion
    // ─────────────────────────────────────────────────────────────────
    struct ContextNode {
        uint64_t id;
        float    score;
        float    similarity;  // 0 if reached via graph expansion
        int      hop;         // 0 = direct search hit, 1+ = graph hops
        Metadata metadata;
    };

    struct ContextEdge {
        uint64_t    source;
        uint64_t    target;
        std::string rel_type;
        float       weight;
    };

    struct ContextChainResult {
        std::vector<ContextNode> nodes;
        std::vector<ContextEdge> edges;
    };

    ContextChainResult context_chain(const std::vector<float>& query,
                                     size_t k = 5,
                                     int hops = 2,
                                     const std::string& modality = "text") {
        // Read-only apart from the salience touch (atomic), so chains run
        // concurrently with queries and with each other.
        std::shared_lock<RWMutex> lock(mutex_);
        auto m_it = modality_indices_.find(modality);
        if (m_it == modality_indices_.end()) return {};
        auto& m_idx = m_it->second;
        check_query_dim(m_idx, query.size(), modality);

        // Step 1: vector search → seed nodes (encode query to storage format)
        auto qbytes = encode_query(m_idx, query.data());
        auto raw = m_idx.index->searchKnn(qbytes.data(), k);
        std::unordered_map<uint64_t, float> sim_scores;
        while (!raw.empty()) {
            auto [dist, id] = raw.top(); raw.pop();
            float sim = 1.0f / (1.0f + dist);
            sim_scores[id] = sim;
            touch_nolock(id);
        }

        // Step 2: BFS expansion over edges (outgoing + incoming)
        std::unordered_map<uint64_t, int> visited;   // id → best hop
        std::queue<std::pair<uint64_t, int>> bfs;
        for (const auto& [id, _] : sim_scores) {
            visited[id] = 0;
            bfs.push({id, 0});
        }

        std::vector<ContextEdge> collected_edges;

        while (!bfs.empty()) {
            auto [cur_id, cur_hop] = bfs.front(); bfs.pop();
            if (cur_hop >= hops) continue;

            // Outgoing edges
            auto it = metadata_store_.find(cur_id);
            if (it != metadata_store_.end()) {
                for (const auto& e : it->second.edges) {
                    collected_edges.push_back({cur_id, e.target_id, e.rel_type, e.weight});
                    if (visited.find(e.target_id) == visited.end()) {
                        visited[e.target_id] = cur_hop + 1;
                        bfs.push({e.target_id, cur_hop + 1});
                    }
                }
            }
            // Incoming edges
            auto rit = reverse_index_.find(cur_id);
            if (rit != reverse_index_.end()) {
                for (const auto& ie : rit->second) {
                    collected_edges.push_back({ie.source_id, cur_id, ie.rel_type, ie.weight});
                    if (visited.find(ie.source_id) == visited.end()) {
                        visited[ie.source_id] = cur_hop + 1;
                        bfs.push({ie.source_id, cur_hop + 1});
                    }
                }
            }
        }

        // Step 3: build result nodes with scores
        ContextChainResult result;
        double now_ts = static_cast<double>(std::time(nullptr));

        for (const auto& [id, hop] : visited) {
            auto mit = metadata_store_.find(id);
            Metadata meta = (mit != metadata_store_.end()) ? mit->second : Metadata();

            float sim = 0.0f;
            auto sit = sim_scores.find(id);
            if (sit != sim_scores.end()) sim = sit->second;

            // Score: similarity decays by hop, modulated by importance + stickiness
            float stickiness = 1.0f + std::log(1.0f + static_cast<float>(meta.recalls()));
            float hop_decay  = 1.0f / (1.0f + static_cast<float>(hop));
            float base       = (hop == 0) ? sim : hop_decay;
            float score      = base * meta.importance * stickiness;

            result.nodes.push_back({id, score, sim, hop, std::move(meta)});
        }

        // Deduplicate edges
        std::sort(collected_edges.begin(), collected_edges.end(),
            [](const ContextEdge& a, const ContextEdge& b) {
                return std::tie(a.source, a.target, a.rel_type) <
                       std::tie(b.source, b.target, b.rel_type);
            });
        collected_edges.erase(std::unique(collected_edges.begin(), collected_edges.end(),
            [](const ContextEdge& a, const ContextEdge& b) {
                return a.source == b.source && a.target == b.target &&
                       a.rel_type == b.rel_type;
            }), collected_edges.end());
        result.edges = std::move(collected_edges);

        // Sort nodes by score descending
        std::sort(result.nodes.begin(), result.nodes.end(),
            [](const ContextNode& a, const ContextNode& b) { return a.score > b.score; });

        return result;
    }

    // ─────────────────────────────────────────────────────────────────
    // Graph export: D3 / Cytoscape-compatible JSON
    // ─────────────────────────────────────────────────────────────────
    std::string export_graph_json(const std::string& ns_filter   = "",
                                  const std::string& eid_filter  = "") const {
        std::shared_lock<RWMutex> lock(mutex_);
        std::ostringstream oss;
        oss << "{\"nodes\":[";
        bool first = true;
        for (const auto& [id, meta] : metadata_store_) {
            if (!ns_filter.empty()  && meta.namespace_id != ns_filter)  continue;
            if (!eid_filter.empty() && meta.entity_id    != eid_filter) continue;
            if (!first) oss << ","; first = false;

            oss << "{\"id\":"         << id;
            oss << ",\"label\":\""    << escape_json(meta.content.substr(0, 60)) << "\"";
            oss << ",\"namespace_id\":\"" << escape_json(meta.namespace_id)      << "\"";
            oss << ",\"entity_id\":\"" << escape_json(meta.entity_id)            << "\"";
            oss << ",\"type\":"       << static_cast<int>(meta.type);
            oss << ",\"source\":\""   << escape_json(meta.source)                << "\"";
            oss << ",\"importance\":" << meta.importance;
            oss << ",\"recall_count\":" << meta.recalls();
            oss << ",\"timestamp\":"  << meta.timestamp;
            oss << ",\"attributes\":{";
            bool fa = true;
            for (const auto& [k,v] : meta.attributes) {
                if (!fa) oss << ","; fa = false;
                oss << "\"" << escape_json(k) << "\":\"" << escape_json(v) << "\"";
            }
            oss << "}}";
        }

        // Build set of exported node IDs so we can filter dangling edges
        std::unordered_set<uint64_t> exported_ids;
        for (const auto& [id, meta] : metadata_store_) {
            if (!ns_filter.empty()  && meta.namespace_id != ns_filter)  continue;
            if (!eid_filter.empty() && meta.entity_id    != eid_filter) continue;
            exported_ids.insert(id);
        }

        oss << "],\"edges\":[";
        first = true;
        for (const auto& [from_id, meta] : metadata_store_) {
            if (!ns_filter.empty()  && meta.namespace_id != ns_filter)  continue;
            if (!eid_filter.empty() && meta.entity_id    != eid_filter) continue;
            for (const auto& e : meta.edges) {
                // Only emit edge if target also exists in the exported node set
                if (exported_ids.find(e.target_id) == exported_ids.end()) continue;
                if (!first) oss << ","; first = false;
                oss << "{\"source\":"      << from_id;
                oss << ",\"target\":"      << e.target_id;
                oss << ",\"rel_type\":\"" << escape_json(e.rel_type) << "\"";
                oss << ",\"weight\":"      << e.weight;
                oss << "}";
            }
        }
        oss << "]}";
        return oss.str();
    }

    // ─────────────────────────────────────────────────────────────────
    // Metadata CRUD
    // ─────────────────────────────────────────────────────────────────
    std::optional<Metadata> get_metadata(uint64_t id) const {
        std::shared_lock<RWMutex> lock(mutex_);
        auto it = metadata_store_.find(id);
        if (it != metadata_store_.end()) return it->second;
        return std::nullopt;
    }

    void update_metadata(uint64_t id, const Metadata& meta) {
        check_writable("update metadata");
        std::string payload;
        {
            std::ostringstream ws;
            meta.serialize(ws);
            payload = ws.str();
        }
        uint64_t lsn = 0;
        {
            std::unique_lock<RWMutex> lock(mutex_);
            ensure_open();
            lsn = wal_append(WalOp::UPDATE, id, payload);
            std::string prev_content;
            bool had_prev = false;
            auto old = metadata_store_.find(id);
            if (old != metadata_store_.end()) {
                deindex_meta(id, old->second);
                prev_content = old->second.content;
                had_prev     = true;
                // Retire this record's reverse-index entries using its OWN old
                // edge list: O(edges of this record). This used to sweep the
                // whole reverse index on every update: measured 13 ms per call
                // at 500k edges, all of it under the exclusive lock.
                for (const auto& e : old->second.edges) {
                    auto rit = reverse_index_.find(e.target_id);
                    if (rit == reverse_index_.end()) continue;
                    auto& v = rit->second;
                    v.erase(std::remove_if(v.begin(), v.end(),
                            [id](const IncomingEdge& ie) { return ie.source_id == id; }), v.end());
                    if (v.empty()) reverse_index_.erase(rit);
                }
                old->second = meta;
            } else {
                metadata_store_.emplace(id, meta);
            }
            if (!is_dead_meta(meta)) index_meta(id, meta);
            for (const auto& e : meta.edges)
                reverse_index_[e.target_id].push_back({id, e.rel_type, e.weight});
            // A record updated into a dead state must leave the keyword index too.
            add_to_bm25_index(id, is_dead_meta(meta) ? std::string() : meta.content,
                              had_prev ? &prev_content : nullptr);
            if (wal_strict()) wal_wait_durable(lsn);
        }
        wal_wait_durable(lsn);
    }

    void update_importance(uint64_t id, float importance) {
        std::string payload(reinterpret_cast<const char*>(&importance), 4);
        uint64_t lsn = 0;
        {
            std::unique_lock<RWMutex> lock(mutex_);
            ensure_open();
            lsn = wal_append(WalOp::UIMP, id, payload);
            auto it = metadata_store_.find(id);
            if (it != metadata_store_.end()) it->second.importance = importance;
            if (wal_strict()) wal_wait_durable(lsn);
        }
        wal_wait_durable(lsn);
    }

    // Get raw vector for a given id and modality (empty if not found)
    std::vector<float> get_vector(uint64_t id, const std::string& modality = "text") const {
        std::shared_lock<RWMutex> lock(mutex_);
        auto it = modality_indices_.find(modality);
        if (it == modality_indices_.end()) return {};
        try {
            return read_vector_label(it->second, id);   // dequantizes if int8
        } catch (...) {
            return {};
        }
    }

    // Get all IDs present in a modality index
    std::vector<uint64_t> get_all_ids(const std::string& modality = "text") const {
        std::shared_lock<RWMutex> lock(mutex_);
        auto it = modality_indices_.find(modality);
        if (it == modality_indices_.end()) return {};
        const auto& m_idx = it->second;
        std::vector<uint64_t> ids;
        ids.reserve(std::min(metadata_store_.size(),
                             m_idx.index->getCurrentElementCount()));
        // Membership test straight against the label table. This used to call
        // read_vector_label() in a try/catch — so every record NOT in this
        // modality cost a thrown-and-caught C++ exception, and every record that
        // WAS in it cost a full dim-length vector copy that was immediately
        // discarded. Same result set (getDataByLabel throws on exactly these two
        // conditions), without the exceptions or the copies.
        for (const auto& [id, _] : metadata_store_) {
            auto lit = m_idx.index->label_lookup_.find(id);
            if (lit == m_idx.index->label_lookup_.end())    continue;
            if (m_idx.index->isMarkedDeleted(lit->second))  continue;
            ids.push_back(id);
        }
        return ids;
    }

    // Every id that has metadata, regardless of which modality holds its
    // vector(s). Records browsing/counting should use this so a DB whose
    // vectors live under a non-"text" modality still lists its records.
    std::vector<uint64_t> all_ids() const {
        std::shared_lock<RWMutex> lock(mutex_);
        std::vector<uint64_t> ids;
        ids.reserve(metadata_store_.size());
        for (const auto& [id, _] : metadata_store_) ids.push_back(id);
        return ids;
    }

    // The actual modality index names present in this DB (e.g. "text",
    // "visual", or whatever an external pipeline named them).
    std::vector<std::string> modality_names() const {
        std::shared_lock<RWMutex> lock(mutex_);
        std::vector<std::string> names;
        names.reserve(modality_indices_.size());
        for (const auto& [name, _] : modality_indices_) names.push_back(name);
        return names;
    }

    // ─────────────────────────────────────────────────────────────────
    // Secondary-index queries — O(matches), LIVE records only.
    // Back the API's namespace/entity/attribute scans and feed feature A's
    // pre-filtered search with ready-made candidate sets.
    // ─────────────────────────────────────────────────────────────────
    std::vector<uint64_t> ids_in_namespace(const std::string& ns) const {
        std::shared_lock<RWMutex> lock(mutex_);
        auto it = ns_index_.find(ns);
        if (it == ns_index_.end()) return {};
        return {it->second.begin(), it->second.end()};
    }

    std::vector<uint64_t> ids_for_entity(const std::string& eid) const {
        std::shared_lock<RWMutex> lock(mutex_);
        auto it = entity_index_.find(eid);
        if (it == entity_index_.end()) return {};
        return {it->second.begin(), it->second.end()};
    }

    std::vector<uint64_t> ids_with_attribute(const std::string& key,
                                             const std::string& val) const {
        std::shared_lock<RWMutex> lock(mutex_);
        auto it = attr_index_.find(attr_key(key, val));
        if (it == attr_index_.end()) return {};
        return {it->second.begin(), it->second.end()};
    }

    size_t namespace_size(const std::string& ns) const {
        std::shared_lock<RWMutex> lock(mutex_);
        auto it = ns_index_.find(ns);
        return it == ns_index_.end() ? 0 : it->second.size();
    }

    std::vector<std::string> list_namespaces() const {
        std::shared_lock<RWMutex> lock(mutex_);
        std::vector<std::string> out;
        out.reserve(ns_index_.size());
        for (const auto& [ns, _] : ns_index_) out.push_back(ns);
        return out;
    }

    // Every distance kernel reads exactly dim floats from the query pointer and
    // trusts the caller for the length. A short query is therefore an
    // out-of-bounds READ, not a wrong answer: measured, a 256-dim query against
    // a 512-dim index read 1 KB past the end of the buffer and returned a
    // confident-looking score. A long query is merely wrong — it silently
    // compares a truncated prefix. The Cloud API guarded this at the HTTP edge;
    // anyone using Feather embedded, which is the whole point of the product,
    // had no guard at all.
    void check_query_dim(const ModalityIndex& m_idx, size_t got,
                         const std::string& modality) const {
        if (got != m_idx.dim)
            throw std::invalid_argument(
                "query vector has " + std::to_string(got) + " dimensions but "
                "modality '" + modality + "' is " + std::to_string(m_idx.dim) +
                " — a mismatched query reads past the end of the buffer");
    }

    // ─────────────────────────────────────────────────────────────────
    // Search
    // ─────────────────────────────────────────────────────────────────
    struct SearchResult {
        uint64_t id;
        float    score;
        Metadata metadata;
        // Exact cosine(query, stored vector) when search(..., with_cosine=true);
        // NaN otherwise. `score` is a ranking value (1/(1+L2^2), optionally
        // decay-weighted), not a similarity you can threshold.
        float    cosine = std::numeric_limits<float>::quiet_NaN();
    };

private:
    // Fill SearchResult::cosine for the (already truncated) result set.
    static void fill_cosines(const ModalityIndex& m_idx, const std::vector<float>& q,
                             std::vector<SearchResult>& results) {
        double qn = 0.0;
        for (float x : q) qn += static_cast<double>(x) * x;
        qn = std::sqrt(qn);
        for (auto& r : results) {
            std::vector<float> v;
            try { v = read_vector_label(m_idx, r.id); } catch (...) { continue; }
            if (v.size() != q.size()) continue;
            double dot = 0.0, vn = 0.0;
            for (size_t i = 0; i < v.size(); ++i) {
                dot += static_cast<double>(q[i]) * v[i];
                vn  += static_cast<double>(v[i]) * v[i];
            }
            vn = std::sqrt(vn);
            if (qn > 0.0 && vn > 0.0) r.cosine = static_cast<float>(dot / (qn * vn));
        }
    }

public:

    // `record_salience` controls whether this query counts as a recall for the
    // records it touches. Default true (a real user query is evidence of value),
    // but evaluation, monitoring and the engine's own internal self-queries must
    // pass false: a read that mutates the state it measures makes benchmarking
    // meaningless and, worse, permanently corrupts the decay signal — only the
    // inflated counter is persisted, so the original is unrecoverable.
    std::vector<SearchResult> search(const std::vector<float>& q, size_t k = 5,
                                     const SearchFilter*   filter  = nullptr,
                                     const ScoringConfig*  scoring = nullptr,
                                     const std::string&    modality = "text",
                                     bool record_salience = true,
                                     bool with_cosine = false) {
        // SHARED lock: queries run concurrently with each other. The only
        // mutation here is the salience touch, which goes through Metadata's
        // atomic counters.
        std::shared_lock<RWMutex> lock(mutex_);
        auto m_it = modality_indices_.find(modality);
        if (m_it == modality_indices_.end()) return {};
        auto& m_idx = m_it->second;
        check_query_dim(m_idx, q.size(), modality);

        const double now_ts = static_cast<double>(std::time(nullptr));
        static const Metadata kNoMeta;
        auto score_of = [&](float dist, const Metadata& meta) {
            return scoring ? Scorer::calculate_score(dist, meta, *scoring, now_ts)
                           : 1.0f / (1.0f + dist);
        };
        // Rank on (score, id) only; copy metadata for the k winners alone. The
        // pre-filtered path used to copy the full Metadata (content, attributes,
        // edges) of EVERY candidate before sorting — 20k string copies to
        // return 10.
        auto finish = [&](std::vector<std::pair<float, uint64_t>>& scored) {
            const size_t kk = std::min(k, scored.size());
            std::partial_sort(scored.begin(), scored.begin() + kk, scored.end(),
                [](const std::pair<float, uint64_t>& a, const std::pair<float, uint64_t>& b) {
                    return a.first > b.first || (a.first == b.first && a.second < b.second);
                });
            std::vector<SearchResult> results;
            results.reserve(kk);
            for (size_t i = 0; i < kk; ++i) {
                auto it = metadata_store_.find(scored[i].second);
                results.push_back({scored[i].second, scored[i].first,
                                   it != metadata_store_.end() ? it->second : kNoMeta});
            }
            // Touch AFTER truncation: a recall means "this record was returned",
            // not "this record was examined" (touching every scanned candidate
            // made stickiness a constant and collapsed adaptive decay).
            if (record_salience)
                for (const auto& r : results) touch_nolock(r.id);
            if (with_cosine) fill_cosines(m_idx, q, results);
            return results;
        };

        struct FilterWrapper : public hnswlib::BaseFilterFunctor {
            const SearchFilter* filter_;
            const std::unordered_map<uint64_t, Metadata>& store_;
            FilterWrapper(const SearchFilter* f,
                          const std::unordered_map<uint64_t, Metadata>& s)
                : filter_(f), store_(s) {}
            bool operator()(hnswlib::labeltype id) override {
                if (!filter_) return true;
                auto it = store_.find(id);
                if (it == store_.end()) return false;
                if (is_dead_meta(it->second)) return false;   // same rule as the exact scan
                return filter_->matches(it->second);
            }
        };
        const size_t want = scoring ? k * 3 : k;          // scoring re-ranks a wider pool
        auto qbytes = encode_query(m_idx, q.data());      // float bytes or int8 blob

        // Graph traversal, optionally with a per-call ef. Returns (score, id).
        auto hnsw = [&](const SearchFilter* f, size_t ef_override) {
            FilterWrapper fw(f, metadata_store_);
            auto res = m_idx.index->searchKnnEf(qbytes.data(), want, ef_override,
                                                 f ? &fw : nullptr);
            std::vector<std::pair<float, uint64_t>> scored;
            scored.reserve(res.size());
            while (!res.empty()) {
                auto [dist, id] = res.top(); res.pop();
                auto it = metadata_store_.find(id);
                scored.push_back({score_of(dist, it != metadata_store_.end() ? it->second : kNoMeta), id});
            }
            return scored;
        };

        // ── Pre-filtered path ────────────────────────────────────────
        // When the filter constrains an indexed field (namespace/entity/
        // attribute), the secondary indexes give the exact candidate set.
        //  * Small or very selective sets: exact scan over just those vectors,
        //    with the index's own SIMD distance kernel on the stored data (no
        //    per-candidate copy). Complete top-k whenever >= k records match,
        //    which ef-bounded filtered HNSW cannot promise for selective filters.
        //  * Large, unselective sets: filtered graph traversal with ef raised
        //    to ~2k/selectivity for this call, falling back to the exact scan
        //    if the walk comes up short.
        //  The route is a cost comparison, not a size threshold. The scan costs
        //  ~candidates; a filtered walk costs ~k/selectivity^2 (a wider beam AND
        //  more visited nodes failing the filter). Measured at 100k x 768:
        //  sel 2% -> scan 0.55 ms vs walk 45 ms; sel 20% -> scan 9.1 ms vs walk
        //  1.1 ms. Take the walk when candidates * sel^2 > C * k (C=10,
        //  calibrated at 128-d and 768-d; FEATHER_PREFILTER_C overrides).
        if (filter) {
            bool indexed = false;
            auto cand = candidates_for_filter(*filter, indexed);
            if (indexed) {
                const size_t live = m_idx.index->getCurrentElementCount() -
                                    m_idx.index->getDeletedCount();
                const int mode = prefilter_mode();
                if (mode != PREFILTER_EXACT && live > 0 && !cand.empty()) {
                    const double c   = static_cast<double>(cand.size());
                    const double sel = std::min(1.0, c / static_cast<double>(live));
                    const double need = 2.0 * static_cast<double>(want) / sel;
                    const bool walk = need <= static_cast<double>(PREFILTER_MAX_EF) &&
                        (mode == PREFILTER_HNSW ||
                         (c >= 2000.0 && c * sel * sel > prefilter_c() * static_cast<double>(want)));
                    if (walk) {
                        const size_t ef = std::max<size_t>(m_idx.index->ef_,
                                                           static_cast<size_t>(need));
                        auto scored = hnsw(filter, ef);
                        if (scored.size() >= std::min(want, cand.size()))
                            return finish(scored);
                    }
                }
                std::vector<std::pair<float, uint64_t>> scored;
                scored.reserve(cand.size());
                auto& idx = *m_idx.index;
                const auto dist_fn = m_idx.space->get_dist_func();
                void* dist_param   = m_idx.space->get_dist_func_param();
                const auto f32i8   = feather_simd::f32_i8_l2_fn();
                for (uint64_t id : cand) {
                    auto it = metadata_store_.find(id);
                    if (it == metadata_store_.end() || is_dead_meta(it->second)) continue;
                    if (!filter->matches(it->second)) continue;   // non-indexed predicates
                    // label_lookup_ is only mutated under the exclusive lock, so
                    // reading it under our shared lock needs no extra mutex.
                    auto lit = idx.label_lookup_.find(id);
                    if (lit == idx.label_lookup_.end()) continue; // not in this modality
                    if (idx.isMarkedDeleted(lit->second)) continue;
                    const char* data = idx.getDataByInternalId(lit->second);
                    float dist;
                    if (m_idx.int8) {
                        // Asymmetric: float query vs int8 row dequantized inside
                        // the (SIMD) kernel, as exact as before, without the loop.
                        dist = f32i8(q.data(), reinterpret_cast<const int8_t*>(data),
                                     m_idx.scale, m_idx.dim);
                    } else {
                        dist = dist_fn(q.data(), data, dist_param);   // SIMD kernel
                    }
                    scored.push_back({score_of(dist, it->second), id});
                }
                return finish(scored);
            }
        }

        // ── Graph path (no filter, or only non-indexed predicates) ────
        auto scored = hnsw(filter, 0);
        return finish(scored);
    }

private:
    // Filtered-search routing (see the cost model in search()).
    static constexpr size_t PREFILTER_MAX_EF = 4096;
    enum { PREFILTER_AUTO = 0, PREFILTER_EXACT = 1, PREFILTER_HNSW = 2 };
    // FEATHER_PREFILTER_MODE=auto|exact|hnsw forces a route (tests, A/B).
    static int prefilter_mode() {
        static const int m = [] {
            const char* e = std::getenv("FEATHER_PREFILTER_MODE");
            if (e && !std::strcmp(e, "exact")) return static_cast<int>(PREFILTER_EXACT);
            if (e && !std::strcmp(e, "hnsw"))  return static_cast<int>(PREFILTER_HNSW);
            return static_cast<int>(PREFILTER_AUTO);
        }();
        return m;
    }
    static double prefilter_c() {
        static const double c = [] {
            if (const char* e = std::getenv("FEATHER_PREFILTER_C")) {
                double v = std::atof(e);
                if (v > 0.0) return v;
            }
            return 10.0;
        }();
        return c;
    }

public:

    // ─────────────────────────────────────────────────────────────────
    // BM25 keyword search
    // ─────────────────────────────────────────────────────────────────
private:
    using ScoredId = std::pair<float, uint64_t>;

    // Best first; ties go to the smaller id (same order as search()).
    static bool better_scored(const ScoredId& a, const ScoredId& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    }

    // The best `limit` BM25 hits for `query` as (score, id), best first.
    // Caller holds mutex_ (shared is enough). Returns ids and scores only, so
    // callers copy Metadata just for the hits they actually return.
    std::vector<ScoredId> bm25_top_nolock(const std::string& query, size_t limit,
                                          const SearchFilter* filter) const {
        std::vector<ScoredId> ranked;
        if (limit == 0 || doc_lengths_.empty()) return ranked;
        auto terms = tokenize(query);
        if (terms.empty()) return ranked;

        std::unordered_set<std::string> unique_terms(terms.begin(), terms.end());
        std::vector<const std::vector<PostingEntry>*> lists;
        size_t total_postings = 0;
        for (const auto& term : unique_terms) {
            auto it = bm25_index_.find(term);
            if (it == bm25_index_.end()) continue;
            lists.push_back(&it->second);
            total_postings += it->second.size();
        }
        if (lists.empty()) return ranked;

        // Never return a posting whose record is gone or dead: it would
        // surface as a hit with empty metadata.
        auto eligible = [&](uint64_t id) {
            auto mit = metadata_store_.find(id);
            return mit != metadata_store_.end() && !is_dead_meta(mit->second) &&
                   (!filter || filter->matches(mit->second));
        };

        // Each posting carries its document's length, so scoring needs no
        // doc_lengths_ lookup. A filter is applied per posting: it usually
        // rejects most documents, and keeping them out of `scores` is cheaper
        // than scoring them. Without one, forget()/purge() already drop
        // postings, so eligibility is only checked on the winners below.
        const double N = static_cast<double>(doc_lengths_.size());
        const double avdl = avg_dl_ > 0.0 ? avg_dl_ : 1.0;
        std::unordered_map<uint64_t, float> scores;
        if (!filter) scores.reserve(total_postings);
        for (const auto* postings : lists) {
            const double n_t = static_cast<double>(postings->size());
            // IDF (BM25+): log((N - n_t + 0.5) / (n_t + 0.5) + 1)
            const double idf = std::log((N - n_t + 0.5) / (n_t + 0.5) + 1.0);
            for (const auto& p : *postings) {
                if (filter && !eligible(p.doc_id)) continue;
                const double tf = static_cast<double>(p.term_freq);
                const double tf_norm = (tf * (BM25_K1 + 1.0)) /
                    (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * static_cast<double>(p.doc_len) / avdl));
                scores[p.doc_id] += static_cast<float>(idf * tf_norm);
            }
        }

        ranked.reserve(scores.size());
        for (const auto& [id, sc] : scores) ranked.emplace_back(sc, id);
        // A stale posting that reaches the top is dropped and the next best
        // takes its place. With a filter every entry is already eligible, so
        // this loop runs once.
        for (;;) {
            const auto head_end = ranked.begin() + std::min(limit, ranked.size());
            std::partial_sort(ranked.begin(), head_end, ranked.end(), better_scored);
            const auto kept_end = std::remove_if(ranked.begin(), head_end,
                [&](const ScoredId& e) { return !eligible(e.second); });
            if (kept_end == head_end) {
                ranked.erase(head_end, ranked.end());
                return ranked;
            }
            ranked.erase(kept_end, head_end);
        }
    }

    // Caller holds mutex_. Copies metadata for the final hits only.
    std::vector<SearchResult> to_results_nolock(const std::vector<ScoredId>& ranked,
                                                bool record_hit) const {
        std::vector<SearchResult> results;
        results.reserve(ranked.size());
        for (const auto& [sc, id] : ranked) {
            if (record_hit) touch_nolock(id);
            auto mit = metadata_store_.find(id);
            results.push_back({id, sc, mit != metadata_store_.end() ? mit->second : Metadata()});
        }
        return results;
    }

public:
    std::vector<SearchResult> keyword_search(const std::string& query, size_t k = 10,
                                             const SearchFilter* filter = nullptr) {
        std::shared_lock<RWMutex> lock(mutex_);
        return to_results_nolock(bm25_top_nolock(query, k, filter), /*record_hit=*/true);
    }

    // ─────────────────────────────────────────────────────────────────
    // Hybrid search: BM25 + vector via Reciprocal Rank Fusion (RRF)
    // ─────────────────────────────────────────────────────────────────
    std::vector<SearchResult> hybrid_search(const std::vector<float>& vec,
                                            const std::string& query,
                                            size_t k = 10,
                                            size_t rrf_k = 60,
                                            const SearchFilter* filter = nullptr,
                                            const ScoringConfig* scoring = nullptr,
                                            const std::string& modality = "text") {
        // Read-only end to end (RRF fusion never touches salience), so this
        // runs fully concurrently with other queries. Both rankings and the
        // fusion work on (score, id); metadata is copied for the final k only.
        std::shared_lock<RWMutex> lock(mutex_);
        const size_t candidates = k * 3;

        // ── Inline vector search (no re-lock) ─────────────────────────
        std::vector<ScoredId> vec_ranked;
        {
            auto m_it = modality_indices_.find(modality);
            if (m_it != modality_indices_.end()) {
                auto& m_idx = m_it->second;
                check_query_dim(m_idx, vec.size(), modality);
                struct FW : public hnswlib::BaseFilterFunctor {
                    const SearchFilter* f_; const std::unordered_map<uint64_t,Metadata>& s_;
                    FW(const SearchFilter* f, const std::unordered_map<uint64_t,Metadata>& s): f_(f),s_(s){}
                    bool operator()(hnswlib::labeltype id) override {
                        if (!f_) return true;
                        auto it = s_.find(id); return it!=s_.end() && f_->matches(it->second);
                    }
                } fw(filter, metadata_store_);
                size_t cands = scoring ? candidates * 3 : candidates;
                auto qbytes = encode_query(m_idx, vec.data());
                auto res = m_idx.index->searchKnn(qbytes.data(), cands, filter ? &fw : nullptr);
                const double now_ts = static_cast<double>(std::time(nullptr));
                static const Metadata kNoMeta;
                vec_ranked.reserve(res.size());
                while (!res.empty()) {
                    auto [dist, id] = res.top(); res.pop();
                    float score;
                    if (scoring) {
                        auto it = metadata_store_.find(id);
                        score = Scorer::calculate_score(
                            dist, it != metadata_store_.end() ? it->second : kNoMeta,
                            *scoring, now_ts);
                    } else {
                        score = 1.0f / (1.0f + dist);
                    }
                    vec_ranked.emplace_back(score, id);
                }
                const auto head_end = vec_ranked.begin() + std::min(candidates, vec_ranked.size());
                std::partial_sort(vec_ranked.begin(), head_end, vec_ranked.end(), better_scored);
                vec_ranked.erase(head_end, vec_ranked.end());
            }
        }

        // ── Inline BM25 search (no re-lock) ──────────────────────────
        const auto kw_ranked = bm25_top_nolock(query, candidates, filter);

        // ── RRF merge ────────────────────────────────────────────────
        std::unordered_map<uint64_t, double> rrf_scores;
        rrf_scores.reserve(vec_ranked.size() + kw_ranked.size());
        for (size_t rank = 0; rank < vec_ranked.size(); ++rank)
            rrf_scores[vec_ranked[rank].second] += 1.0 / (static_cast<double>(rrf_k) + rank + 1);
        for (size_t rank = 0; rank < kw_ranked.size(); ++rank)
            rrf_scores[kw_ranked[rank].second] += 1.0 / (static_cast<double>(rrf_k) + rank + 1);

        std::vector<std::pair<double, uint64_t>> fused;
        fused.reserve(rrf_scores.size());
        for (const auto& [id, sc] : rrf_scores) fused.emplace_back(sc, id);
        const auto top_end = fused.begin() + std::min(k, fused.size());
        std::partial_sort(fused.begin(), top_end, fused.end(),
            [](const std::pair<double, uint64_t>& a, const std::pair<double, uint64_t>& b) {
                return a.first > b.first || (a.first == b.first && a.second < b.second);
            });

        std::vector<ScoredId> ranked;
        ranked.reserve(top_end - fused.begin());
        for (auto it = fused.begin(); it != top_end; ++it)
            ranked.emplace_back(static_cast<float>(it->first), it->second);
        return to_results_nolock(ranked, /*record_hit=*/false);
    }

    // ─────────────────────────────────────────────────────────────────
    // Memory lifecycle: forget / purge / expire
    // ─────────────────────────────────────────────────────────────────

    // Soft-delete: mark-deleted in HNSW (exits search), blank content,
    // set importance=0. The node shell remains so graph edges stay traversable.
private:
    // Caller holds mutex_ exclusively.
    void forget_nolock(uint64_t id) {
        for (auto& [name, m_idx] : modality_indices_) {
            try { m_idx.index->markDelete(id); } catch (...) {}
        }
        log_compaction_remove(id);
        auto it = metadata_store_.find(id);
        if (it != metadata_store_.end()) {
            deindex_meta(id, it->second);   // forgotten records leave candidate sets
            // ...and leave the keyword index, otherwise a forgotten record stays
            // fully retrievable via keyword_search/hybrid_search.
            remove_from_bm25_index(id, &it->second.content);
            it->second.content    = "";
            it->second.source     = "_forgotten";
            it->second.importance = 0.0f;
            it->second.ttl        = 0;
        }
    }

    // Hard-delete every record in a namespace. Shared by purge() and WAL
    // replay; replay passes maintain_derived=false because load_vectors()
    // rebuilds the derived indexes once afterwards.
    size_t purge_core(const std::string& ns_id, bool maintain_derived) {
        std::unordered_set<uint64_t> to_purge;
        for (const auto& [id, meta] : metadata_store_)
            if (meta.namespace_id == ns_id) to_purge.insert(id);
        if (to_purge.empty()) return 0;

        for (auto& [name, m_idx] : modality_indices_)
            for (uint64_t id : to_purge) {
                try { m_idx.index->markDelete(id); } catch (...) {}
            }
        for (uint64_t id : to_purge) {
            log_compaction_remove(id);
            auto it = metadata_store_.find(id);
            if (it == metadata_store_.end()) continue;
            if (maintain_derived) {
                deindex_meta(id, it->second);
                // Purged ids must leave the keyword index too: a stale posting
                // resolves to no metadata and surfaces as an empty ghost hit.
                remove_from_bm25_index(id, &it->second.content);
            }
            metadata_store_.erase(it);
        }
        if (maintain_derived) prune_reverse_index(to_purge);
        // Prune edges in surviving nodes that pointed to purged targets
        for (auto& [id, meta] : metadata_store_) {
            meta.edges.erase(
                std::remove_if(meta.edges.begin(), meta.edges.end(),
                    [&to_purge](const Edge& e) { return to_purge.count(e.target_id) > 0; }),
                meta.edges.end());
        }
        return to_purge.size();
    }

public:
    void forget(uint64_t id) {
        check_writable("forget a record");
        uint64_t lsn = 0;
        {
            std::unique_lock<RWMutex> lock(mutex_);
            ensure_open();
            lsn = wal_append(WalOp::FORGET, id, "");
            forget_nolock(id);
            maybe_auto_compact_nolock();
            if (wal_strict()) wal_wait_durable(lsn);
        }
        wal_wait_durable(lsn);
    }

    // Hard-delete: remove all nodes in namespace_id from indices +
    // metadata store + reverse index. Returns count of removed nodes.
    // WAL-logged since 0.19 (it used to survive a crash only via save()).
    size_t purge(const std::string& ns_id) {
        std::string payload;
        {
            uint16_t len = static_cast<uint16_t>(std::min<size_t>(ns_id.size(), 65535));
            payload.append(reinterpret_cast<const char*>(&len), 2);
            payload.append(ns_id.data(), len);
        }
        uint64_t lsn = 0;
        size_t n = 0;
        {
            std::unique_lock<RWMutex> lock(mutex_);
            ensure_open();
            lsn = wal_append(WalOp::PURGE, 0, payload);
            n = purge_core(ns_id, /*maintain_derived=*/true);
            maybe_auto_compact_nolock();
            if (wal_strict()) wal_wait_durable(lsn);
        }
        wal_wait_durable(lsn);
        return n;
    }

    // Scan all nodes and soft-delete any with ttl>0 where now > timestamp+ttl.
    // Returns count of nodes forgotten.
    size_t forget_expired() {
        uint64_t last_lsn = 0;
        size_t count = 0;
        {
            std::unique_lock<RWMutex> lock(mutex_);
            ensure_open();
            int64_t now = static_cast<int64_t>(std::time(nullptr));
            std::vector<uint64_t> expired;
            for (const auto& [id, meta] : metadata_store_) {
                if (meta.ttl > 0 && now > meta.timestamp + meta.ttl)
                    expired.push_back(id);
            }
            for (uint64_t id : expired) {
                last_lsn = wal_append(WalOp::FORGET, id, "");
                forget_nolock(id);
                ++count;
            }
            maybe_auto_compact_nolock();
            if (wal_strict()) wal_wait_durable(last_lsn);
        }
        wal_wait_durable(last_lsn);   // one fsync for the whole sweep
        return count;
    }

    // ─────────────────────────────────────────────────────────────────
    // compact(): rebuild HNSW indices without soft-deleted records
    // ─────────────────────────────────────────────────────────────────
    // Three phases, so the DB is never frozen for the rebuild itself:
    //   1. exclusive lock (short): snapshot the live vectors of every modality
    //      in their storage format and start logging vector-level changes;
    //   2. NO lock: build the new indexes in parallel from the snapshot, while
    //      readers keep using (and writers keep changing) the old ones;
    //   3. exclusive lock (short): replay the changes logged during phase 2
    //      onto the new indexes, swap them in, and erase dead metadata.
    // Measured before this change: 40 s (100k x 128) to 86 s (50k x 768) with
    // no reads or writes at all. Transient RAM: one copy of the live vectors
    // plus the new index while phase 2 runs. Returns the number of dead records
    // erased.
    size_t compact() {
        check_writable("compact");
        std::lock_guard<std::mutex> one_at_a_time(compact_run_mx_);
        struct Snap {
            std::string name;
            size_t dim; bool int8; float scale; size_t ef; size_t stride;
            std::vector<uint64_t> ids;
            std::vector<char> data;
            std::unique_ptr<hnswlib::SpaceInterface<float>> space;
            std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
        };
        std::vector<Snap> snaps;
        std::unordered_set<uint64_t> dead;

        // ── Phase 1: snapshot ────────────────────────────────────────
        {
            std::unique_lock<RWMutex> lock(mutex_);
            ensure_open();
            for (const auto& [id, meta] : metadata_store_)
                if (is_dead_meta(meta)) dead.insert(id);
            bool work = !dead.empty();
            if (!work)
                for (auto& [name, m_idx] : modality_indices_)
                    if (m_idx.index->getDeletedCount() > 0) { work = true; break; }
            if (!work) return 0;

            for (auto& [name, m_idx] : modality_indices_) {
                Snap sn;
                sn.name   = name;
                sn.dim    = m_idx.dim;
                sn.int8   = m_idx.int8;
                sn.scale  = m_idx.scale;
                sn.ef     = m_idx.index->ef_;
                sn.stride = m_idx.space->get_data_size();
                const size_t n = m_idx.index->cur_element_count;
                sn.ids.reserve(n);
                sn.data.reserve(n * sn.stride);
                for (size_t i = 0; i < n; ++i) {
                    if (m_idx.index->isMarkedDeleted(i)) continue;
                    uint64_t id = m_idx.index->getExternalLabel(i);
                    auto mit = metadata_store_.find(id);
                    if (mit == metadata_store_.end()) continue;  // purged / orphaned
                    if (is_dead_meta(mit->second))      continue;  // forgotten / _deleted
                    sn.ids.push_back(id);
                    const char* raw = m_idx.index->getDataByInternalId(i);
                    sn.data.insert(sn.data.end(), raw, raw + sn.stride);
                }
                snaps.push_back(std::move(sn));
            }
            compacting_ = true;
            compaction_log_.clear();
        }

        // ── Phase 2: build without holding the data lock ─────────────
        try {
            for (auto& sn : snaps) {
                if (sn.int8) sn.space = std::make_unique<hnswlib::Int8L2Space>(sn.dim, sn.scale);
                else         sn.space = std::make_unique<hnswlib::L2Space>(sn.dim);
                sn.index = make_hnsw(sn.space.get(),
                                     std::max(INITIAL_MAX_ELEMENTS, sn.ids.size() + sn.ids.size() / 8 + 64));
                ModalityIndex tmp{std::move(sn.index), nullptr, sn.dim, sn.int8, sn.scale};
                parallel_add_raw(tmp, sn.ids, sn.data, sn.stride, compact_threads());
                sn.index = std::move(tmp.index);
                std::vector<char>().swap(sn.data);    // release the snapshot early
            }
        } catch (...) {
            std::unique_lock<RWMutex> lock(mutex_);
            compacting_ = false;
            compaction_log_.clear();
            throw;
        }

        // ── Phase 3: catch up and swap ───────────────────────────────
        std::unique_lock<RWMutex> lock(mutex_);
        compacting_ = false;
        if (closed_) { compaction_log_.clear(); return 0; }
        std::unordered_map<std::string, Snap*> by_name;
        for (auto& sn : snaps) by_name[sn.name] = &sn;
        for (const auto& op : compaction_log_) {
            if (op.remove) {
                for (auto& sn : snaps) {
                    try { sn.index->markDelete(op.id); } catch (...) {}
                }
                continue;
            }
            auto it = by_name.find(op.modality);
            if (it == by_name.end()) continue;            // modality created meanwhile
            Snap& sn = *it->second;
            ModalityIndex tmp{std::move(sn.index), nullptr, sn.dim, sn.int8, sn.scale};
            reserve(tmp, tmp.index->getCurrentElementCount() + 1);
            add_point(tmp, op.id, op.vec.data(), /*reuse_slot=*/true);
            sn.index = std::move(tmp.index);
        }
        compaction_log_.clear();
        compaction_log_.shrink_to_fit();

        for (auto& sn : snaps) {
            auto mit = modality_indices_.find(sn.name);
            if (mit == modality_indices_.end()) continue;
            sn.index->setEf(sn.ef);
            mit->second.index = std::move(sn.index);   // old index freed here
            mit->second.space = std::move(sn.space);
        }

        // Erase metadata that is STILL dead (an id may have been re-added
        // during phase 2). Dead records are already out of the secondary and
        // BM25 indexes (forget/purge/update remove them), so only the reverse
        // index needs pruning — no full rebuild.
        std::unordered_set<uint64_t> erased;
        for (uint64_t id : dead) {
            auto it = metadata_store_.find(id);
            if (it == metadata_store_.end() || !is_dead_meta(it->second)) continue;
            remove_from_bm25_index(id, &it->second.content);   // no-op unless add() of a dead meta slipped in
            metadata_store_.erase(it);
            erased.insert(id);
        }
        prune_reverse_index(erased);
        return erased.size();
    }

    // Configure auto-compaction. ratio in (0,1] triggers a rebuild of a modality
    // index once its deleted/total ratio crosses `ratio` after a forget/purge/
    // expire. 0 disables it. e.g. set_auto_compact(0.2) -> rebuild at 20% dead.
    // The rebuild runs on a background thread (it used to run inline inside the
    // forget() that crossed the threshold).
    void set_auto_compact(float ratio) {
        std::unique_lock<RWMutex> lock(mutex_);
        auto_compact_ratio_ = ratio;
    }
    float get_auto_compact() const {
        std::shared_lock<RWMutex> lock(mutex_);
        return auto_compact_ratio_;
    }

    // True while a compaction is between its snapshot and its swap.
    bool is_compacting() const {
        std::shared_lock<RWMutex> lock(mutex_);
        return compacting_;
    }

    // Block until no background (auto-)compaction is requested or running.
    // Returns false if `timeout_s` > 0 elapsed first.
    bool wait_for_compaction(double timeout_s = 0.0) {
        std::unique_lock<std::mutex> g(compactor_mx_);
        auto idle = [&] { return stop_compactor_ || (!compaction_requested_ && !compactor_busy_); };
        if (timeout_s <= 0.0) { compactor_cv_.wait(g, idle); return true; }
        return compactor_cv_.wait_for(g, std::chrono::duration<double>(timeout_s), idle);
    }

    // Persist a modality's vectors as int8 + per-vector scale (file format v7):
    // ~4x smaller on disk, dequantized to float32 on load. Takes effect on the
    // next save(). The in-memory index is unchanged. Opt-in; default off.
    void set_quantized(const std::string& modality, bool on) {
        std::unique_lock<RWMutex> lock(mutex_);
        if (on) quantized_modalities_.insert(modality);
        else    quantized_modalities_.erase(modality);
    }
    bool is_quantized(const std::string& modality) const {
        std::shared_lock<RWMutex> lock(mutex_);
        return quantized_modalities_.count(modality) > 0;
    }

    // Store this modality's vectors as int8[dim] IN MEMORY (4x less RAM) using a
    // global scale = max_abs/127. Must be called BEFORE the first vector is added
    // to the modality (it sets the index storage type). `max_abs` should bound
    // the largest |component| in your vectors (values beyond it are clamped);
    // for unit-norm embeddings a small value like 0.3-1.0 is typical.
    void set_int8_ram(const std::string& modality, float max_abs = 1.0f) {
        std::unique_lock<RWMutex> lock(mutex_);
        auto it = modality_indices_.find(modality);
        if (it != modality_indices_.end() &&
            it->second.index->getCurrentElementCount() > 0)
            throw std::runtime_error(
                "set_int8_ram('" + modality + "') must be called before any "
                "vectors are added to that modality");
        if (max_abs <= 0.0f) max_abs = 1.0f;
        float scale = max_abs / 127.0f;
        int8_ram_scale_[modality] = scale;
        // An existing (empty) index for this modality is rebuilt as int8 in place.
        if (it != modality_indices_.end()) {
            size_t dim = it->second.dim;
            auto space = std::make_unique<hnswlib::Int8L2Space>(dim, scale);
            auto index = make_hnsw(space.get(), INITIAL_MAX_ELEMENTS);
            it->second = {std::move(index), std::move(space), dim, true, scale};
        }
    }
    bool is_int8_ram(const std::string& modality) const {
        std::shared_lock<RWMutex> lock(mutex_);
        return int8_ram_scale_.count(modality) > 0;
    }

    // ─────────────────────────────────────────────────────────────────
    // Persistence & info
    // ─────────────────────────────────────────────────────────────────
    // Checkpoint. The snapshot is written under the SHARED lock (queries keep
    // running), and the slow part — fsyncing the new base file and the rename —
    // happens after the lock is released: the WAL is rotated to <wal>.old at
    // the snapshot point so writers can proceed into a fresh WAL immediately.
    // Before 0.19, writers were blocked for the entire save (measured: writer
    // p99 2.1 s while saves looped).
    void save() {
        check_writable("save");
        std::lock_guard<std::mutex> save_lock(save_mutex_);
        const std::string tmp = path_ + ".tmp";
        bool rotated = false;
        {
            LongSharedLock lock(mutex_);   // long: queries keep flowing during the snapshot
            ensure_open();
            write_snapshot(tmp);
            rotated = wal_rotate();
            if (!rotated) {
                // A previous checkpoint left <wal>.old behind (it failed after
                // rotating). Finish synchronously so the two WALs never need
                // merging: publish, then drop both.
                publish_snapshot(tmp);
                wal_clear();
                return;
            }
        }
        publish_snapshot(tmp);                       // fsync + rename + dir fsync
        std::remove(wal_old_path().c_str());         // its records are in the base now
    }

    // Close the DB: checkpoint (unless save=false), release the WAL and the
    // inter-process file lock. Further calls on this handle throw. Required
    // rather than decorative: Python never runs ~DB() (the handle is held with
    // py::nodelete), so without close() — or a `with DB.open(...) as db:`
    // block — the lock lives until process exit. Idempotent. A read-only
    // handle holds only a SHARED lock and never rewrites the file.
    void close(bool do_save = true) {
        stop_compactor();                            // never join while holding mutex_
        std::lock_guard<std::mutex> save_lock(save_mutex_);
        std::unique_lock<RWMutex> lock(mutex_);
        if (closed_) return;
        if (do_save && load_complete_ && !read_only_) save_vectors();
        wal_close();
        release_lock();
        closed_ = true;
    }

    ~DB() {
        stop_compactor();
        // Only checkpoint a DB whose load actually completed. If load_vectors()
        // threw, `this` holds a HALF-PARSED view of the file — some records
        // read, the rest never reached — and save_vectors() would serialise
        // that fragment over the very file it failed to read, then wal_clear()
        // the log that could have rebuilt it. DB::open() builds a unique_ptr,
        // so a throw inside load unwinds straight into this destructor: the
        // failure path and the destroy-the-evidence path were the same path.
        //
        // Reproduced: a .feather truncated to 60% left a 16 KB
        // `<path>.tmp` behind — the destructor had begun rewriting the file
        // from the fragment and only crashed before reaching std::rename.
        // Whether the original survived was down to where the crash landed.
        // read_only_ is the second guard: a reader holds only a SHARED lock, so
        // a checkpoint from here would rewrite the file while other readers —
        // and possibly a writer waiting on it — are using it.
        if (!closed_ && load_complete_ && !read_only_) {
            try { save_vectors(); } catch (...) {}
        }
        wal_close();   // save_vectors() clears the WAL on success; on failure
                       // the handle still has to be released
        release_lock();
    }

    // Bytes currently held in the WAL files (<wal> + <wal>.old). A caller-side
    // checkpoint policy ("save once the WAL passes N MB") uses this instead of
    // saving after every mutation.
    uint64_t wal_size() const {
        return fsio::size_or_zero(wal_path_) + fsio::size_or_zero(wal_old_path());
    }

    // Number of WAL fsyncs issued so far (group-commit effectiveness).
    uint64_t wal_fsync_count() const {
        std::lock_guard<std::mutex> g(wal_mutex_);
        return wal_fsyncs_;
    }

    size_t dim(const std::string& modality = "text") const {
        std::shared_lock<RWMutex> lock(mutex_);
        auto it = modality_indices_.find(modality);
        if (it != modality_indices_.end()) return it->second.dim;
        return default_dim_;   // modality not created yet → report the open() default
    }

    size_t size() const {
        std::shared_lock<RWMutex> lock(mutex_);
        return metadata_store_.size();
    }

    // ─────────────────────────────────────────────────────────────────
    // Tuning: HNSW search beam width
    // ─────────────────────────────────────────────────────────────────
    // Higher ef = better recall, slower search. Default is DEFAULT_EF (50).
    // Pass modality = "" (default) to apply to all modalities.
    void set_ef(size_t ef, const std::string& modality = "") {
        std::unique_lock<RWMutex> lock(mutex_);
        if (modality.empty()) {
            for (auto& [_name, mi] : modality_indices_) {
                mi.index->setEf(ef);
            }
        } else {
            auto it = modality_indices_.find(modality);
            if (it == modality_indices_.end())
                throw std::runtime_error("unknown modality: " + modality);
            it->second.index->setEf(ef);
        }
    }

    size_t get_ef(const std::string& modality = "text") const {
        std::shared_lock<RWMutex> lock(mutex_);
        auto it = modality_indices_.find(modality);
        if (it == modality_indices_.end())
            throw std::runtime_error("unknown modality: " + modality);
        return it->second.index->ef_;
    }
};

} // namespace feather
