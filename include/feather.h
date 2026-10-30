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
#include <cerrno>
#include <array>
#include <chrono>
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
#  include <sys/stat.h>    // stat: exists / size (std::filesystem needs macOS 10.15+)
#  include <sys/mman.h>    // mmap: read-only handles share the vector image
#endif
#include "hnswlib.h"
#include "metadata.h"
#include "filter.h"
#include "scoring.h"
#include "format_v10.h"
#include <deque>
#include <map>
#include <optional>
#include <set>

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
    size_t writer_pass_     = 0;            // writers left in this writer phase
    std::chrono::steady_clock::time_point phase_start_;   // when this writer phase began
    // A writer phase hands over to the next queued writer only while it is
    // younger than this; past it, the readers waiting get their turn.
    static constexpr std::chrono::microseconds WRITER_PHASE{100};
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
        // A writer phase may run the writers queued when it began back to
        // back, up to WRITER_PHASE of time, before the readers' next turn.
        // One writer per phase made N writers SLOWER than one: each write paid
        // a full reader phase (wake every reader, wait out their searches) —
        // measured, 8 readers + 4 link() writers did 2.6k writes/s against
        // 9.8k with one writer. The time bound is what keeps it fair to
        // readers: by count alone, 16 queued add()s (~0.5 ms each) held them
        // off for ms and cut reads 9.8k -> 0.7k/s. Cheap writes now share a
        // phase; an expensive one still gets one to itself.
        if (writer_pass_ > 0) {
            --writer_pass_;
        } else {
            writer_pass_ = waiting_writers_;
            phase_start_ = std::chrono::steady_clock::now();
        }
        // No reader can enter meanwhile: the fast path is closed by W_WAITING,
        // and the slow path needs m_, which we hold.
        state_.store(W_ACTIVE | (waiting_writers_ ? W_WAITING : 0u), std::memory_order_relaxed);
    }
    void unlock() {
        std::lock_guard<std::mutex> l(m_);
        state_.store(waiting_writers_ ? W_WAITING : 0u, std::memory_order_release);
        if (writer_pass_ > 0 && waiting_writers_ > 0 &&   // the rest of this writer phase
            (waiting_readers_ == 0 ||
             std::chrono::steady_clock::now() - phase_start_ < WRITER_PHASE)) {
            writers_cv_.notify_one();
            return;
        }
        writer_pass_ = 0;
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
// Test hook (feather_db.core._fail_fsyncs_after): let `n` more syncs succeed,
// then fail every one after that until it is reset with -1. It stays failing
// on purpose: a disk that returned an fsync error is not trusted again either.
inline std::atomic<long>& sync_faults_after() { static std::atomic<long> n{-1}; return n; }
inline std::atomic<int>&  sync_fault_errno()  { static std::atomic<int>  e{EIO}; return e; }
inline bool sync_fault() {
    auto& n = sync_faults_after();
    long v = n.load();
    while (v > 0 && !n.compare_exchange_weak(v, v - 1)) {}
    return v == 0;
}
#if !defined(_WIN32)
// macOS fsync() hands data to the drive without flushing its write cache, so
// it survives a process or kernel crash but not a power cut. F_FULLFSYNC is the
// durable form and costs ~150x (measured 25us -> 3.7ms per WAL write), so it is
// opt-in, as SQLite's fullfsync is: FEATHER_WAL_FULLFSYNC=1. A no-op elsewhere,
// and a filesystem that refuses F_FULLFSYNC gets plain fsync.
inline bool full_fsync_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("FEATHER_WAL_FULLFSYNC");
        return e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y' || e[0] == 't' || e[0] == 'T');
    }();
    return on;
}
// An fsync that ran and failed: the only failure that poisons a DB handle.
// Not being able to OPEN a file to sync it (EMFILE, ENOENT, ...) is an ordinary
// error — nothing was lost, and a retry may work.
struct SyncFailed : std::runtime_error { using std::runtime_error::runtime_error; };

// fsync, retried on EINTR. A filesystem that cannot sync at all (EINVAL,
// ENOTSUP, ENOSYS: special files, some FUSE and network mounts) counts as
// synced, as it always did here: refusing every write on such a mount would
// take down a service that has run on it until now.
inline int fsync_eintr(int fd) {
    int rc;
    do { rc = ::fsync(fd); } while (rc != 0 && errno == EINTR);
    return rc;
}
inline bool sync_unsupported(int err) {
    return err == EINVAL || err == ENOTSUP || err == EOPNOTSUPP || err == ENOSYS;
}
inline int sync_fd(int fd) {
    int rc;
    if (sync_fault()) { errno = sync_fault_errno().load(); rc = -1; }
#if defined(__APPLE__)
    else if (full_fsync_enabled() && ::fcntl(fd, F_FULLFSYNC) == 0) rc = 0;
#endif
    else rc = fsync_eintr(fd);
    return rc != 0 && sync_unsupported(errno) ? 0 : rc;
}
#endif
#if defined(_WIN32)
struct SyncFailed : std::runtime_error { using std::runtime_error::runtime_error; };
#endif
inline void fsync_path(const std::string& p) {
#if defined(_WIN32)
    HANDLE h = CreateFileA(p.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                           nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("fsync: cannot open " + p);
    BOOL ok = !sync_fault() && FlushFileBuffers(h);
    CloseHandle(h);
    if (!ok) throw SyncFailed("fsync: FlushFileBuffers failed for " + p);
#else
    int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("fsync: cannot open " + p + ": " + std::strerror(errno));
    int rc = sync_fd(fd);
    const int err = errno;
    ::close(fd);
    if (rc != 0) throw SyncFailed("fsync failed for " + p + ": " + std::strerror(err));
#endif
}
// Make a rename durable: on POSIX the directory entry must be synced too.
// Best effort where it always was: a directory we cannot open, or a filesystem
// that cannot sync directories (EBADF too, as PostgreSQL allows), is skipped.
// Only an fsync that ran and failed throws.
inline void fsync_parent_dir(const std::string& p) {
#if !defined(_WIN32)
    const size_t slash = p.find_last_of('/');
    const std::string d = slash == std::string::npos ? std::string(".")
                        : slash == 0 ? std::string("/") : p.substr(0, slash);
    int fd = ::open(d.c_str(), O_RDONLY);
    if (fd < 0) return;
    const int rc  = sync_fault() ? (errno = sync_fault_errno().load(), -1) : fsync_eintr(fd);
    const int err = errno;
    ::close(fd);
    if (rc != 0 && !sync_unsupported(err) && err != EBADF)
        throw SyncFailed("fsync failed for directory " + d + ": " + std::strerror(err));
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
// Throws on failure. The caller must not retry: after a failed fsync the
// kernel may have dropped the pages it could not write and marked them clean,
// so a second fsync can succeed without them (PostgreSQL, "fsyncgate" 2018).
inline void fsync_file(std::FILE* f, const std::string& name = "file") {
#if defined(_WIN32)
    const bool ok = !sync_fault() && _commit(_fileno(f)) == 0;
#else
    const bool ok = sync_fd(fileno(f)) == 0;
#endif
    if (!ok) throw SyncFailed("fsync failed for " + name + ": " + std::strerror(errno));
}
// POSIX, not std::filesystem, outside Windows: the macOS wheels and CI build
// for a deployment target below 10.15, where libc++ marks std::filesystem
// unavailable and the extension fails to compile.
inline bool exists(const std::string& p) {
#if defined(_WIN32)
    std::error_code ec;
    return std::filesystem::exists(p, ec);
#else
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
#endif
}
inline uint64_t size_or_zero(const std::string& p) {
#if defined(_WIN32)
    std::error_code ec;
    auto s = std::filesystem::file_size(p, ec);
    return ec ? 0 : static_cast<uint64_t>(s);
#else
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 ? static_cast<uint64_t>(st.st_size) : 0;
#endif
}
// 64-bit positioned stdio for the v10 append handle.
inline int seek64(std::FILE* f, uint64_t off) {
#if defined(_WIN32)
    return _fseeki64(f, static_cast<__int64>(off), SEEK_SET);
#else
    return ::fseeko(f, static_cast<off_t>(off), SEEK_SET);
#endif
}
inline bool truncate_file(std::FILE* f, uint64_t size) {
    std::fflush(f);
#if defined(_WIN32)
    return _chsize_s(_fileno(f), static_cast<__int64>(size)) == 0;
#else
    return ::ftruncate(fileno(f), static_cast<off_t>(size)) == 0;
#endif
}
// Reserve `reserve` bytes of address space and map `len` bytes of `path`,
// starting at `offset`, copy-on-write over the start of it. The file is never
// written: a store into a mapped page gives this process a private copy of
// that page, and the reserved tail is ordinary zero-filled memory for growth.
// {nullptr, 0} when unsupported (Windows) or refused; the caller then reads
// into the heap instead.
struct Mapping { char* base = nullptr; size_t bytes = 0; };
inline Mapping map_private(const std::string& path, uint64_t offset, size_t len, size_t reserve) {
#if defined(_WIN32)
    (void)path; (void)offset; (void)len; (void)reserve;
    return {};
#else
    const size_t pg = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    if (pg == 0 || offset % pg) return {};
    auto round = [pg](size_t n) { return (n + pg - 1) / pg * pg; };
    const size_t bytes = round(std::max(reserve, len));
    int flags = MAP_PRIVATE | MAP_ANON;
#if defined(MAP_NORESERVE)
    flags |= MAP_NORESERVE;          // address space, not committed memory
#endif
    void* base = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (base == MAP_FAILED) return {};
    if (len) {
        int fd = ::open(path.c_str(), O_RDONLY);
        void* m = fd < 0 ? MAP_FAILED
                         : ::mmap(base, round(len), PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE | MAP_FIXED, fd, static_cast<off_t>(offset));
        if (fd >= 0) ::close(fd);
        if (m == MAP_FAILED) { ::munmap(base, bytes); return {}; }
    }
    return {static_cast<char*>(base), bytes};
#endif
}
inline void truncate_to(const std::string& p, uint64_t size) {
#if defined(_WIN32)
    std::error_code ec;
    std::filesystem::resize_file(p, static_cast<uintmax_t>(size), ec);
#else
    if (::truncate(p.c_str(), static_cast<off_t>(size)) != 0) { /* best effort */ }
#endif
}
// Positioned reads from one open file, safe from many threads at once (no
// shared file position). Holds the inode it opened: after a rename replaces
// the path, reads keep returning the old file's bytes until reopen().
class RandomReader {
#if defined(_WIN32)
    HANDLE h_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
public:
    RandomReader() = default;
    RandomReader(const RandomReader&) = delete;
    RandomReader& operator=(const RandomReader&) = delete;
    ~RandomReader() { close(); }
    bool is_open() const {
#if defined(_WIN32)
        return h_ != INVALID_HANDLE_VALUE;
#else
        return fd_ >= 0;
#endif
    }
    // Replace the open file with `path`. False (and closed) if it cannot be opened.
    bool reopen(const std::string& path) {
        close();
#if defined(_WIN32)
        // FILE_SHARE_DELETE: a full rewrite renames over this file while it is open.
        h_ = CreateFileA(path.c_str(), GENERIC_READ,
                         FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                         nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
#endif
        return is_open();
    }
    void close() {
#if defined(_WIN32)
        if (h_ != INVALID_HANDLE_VALUE) CloseHandle(h_);
        h_ = INVALID_HANDLE_VALUE;
#else
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
#endif
    }
    // Whether `path` still names the file this reader holds.
    bool same_file(const std::string& path) const {
#if defined(_WIN32)
        (void)path;
        return is_open();
#else
        struct stat a, b;
        return fd_ >= 0 && ::fstat(fd_, &a) == 0 && ::stat(path.c_str(), &b) == 0 &&
               a.st_dev == b.st_dev && a.st_ino == b.st_ino;
#endif
    }
    void read_at(uint64_t off, void* buf, size_t n) const {
        char* p = static_cast<char*>(buf);
        while (n > 0) {
#if defined(_WIN32)
            OVERLAPPED ov{};
            ov.Offset     = static_cast<DWORD>(off);
            ov.OffsetHigh = static_cast<DWORD>(off >> 32);
            DWORD got = 0;
            const DWORD want = static_cast<DWORD>(std::min<size_t>(n, 1u << 30));
            if (!ReadFile(h_, p, want, &got, &ov) || got == 0)
                throw std::runtime_error("read failed in .feather at offset " + std::to_string(off));
            const size_t r = got;
#else
            const ssize_t r = ::pread(fd_, p, n, static_cast<off_t>(off));
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0)
                throw std::runtime_error("read failed in .feather at offset " + std::to_string(off) +
                                         (r < 0 ? std::string(": ") + std::strerror(errno) : ""));
#endif
            p += r; off += static_cast<uint64_t>(r); n -= static_cast<size_t>(r);
        }
    }
};
} // namespace fsio

// ── Reverse-index entry: who points to a given node ──────────────
struct IncomingEdge {
    uint64_t    source_id;
    std::string rel_type;
    float       weight;
};

// A WAL write or an fsync failed, now or earlier on this handle: the handle
// refuses writes until the file is reopened. Python: feather_db.WriteFailedError
// (a RuntimeError).
struct WriteFailed : std::runtime_error { using std::runtime_error::runtime_error; };

// A call on a closed handle. Raised before the call does anything, so a
// caller that closed a handle under a request (a server closing an idle
// namespace) can safely run the request again on a fresh one.
struct HandleClosed : std::logic_error { using std::logic_error::logic_error; };

// A conditional write whose expected version was not the record's current
// one. Nothing of its batch was logged or applied. `actual` is 0 when the
// record does not exist (or was forgotten).
struct VersionConflict : std::runtime_error {
    uint64_t id, expected, actual;
    VersionConflict(uint64_t i, uint64_t e, uint64_t a)
        : std::runtime_error("version conflict on record " + std::to_string(i) +
                             ": expected version " + std::to_string(e) +
                             ", record is at " + std::to_string(a)),
          id(i), expected(e), actual(a) {}
};

// A write that would give a (namespace, key) to a second live record. Keys
// are unique: the write is refused before anything is logged.
struct DuplicateKey : std::invalid_argument {
    std::string ns, key;
    uint64_t holder;
    DuplicateKey(const std::string& n, const std::string& k, uint64_t h)
        : std::invalid_argument("key '" + k + "' in namespace '" + n + "' already belongs to record " +
                                std::to_string(h)),
          ns(n), key(k), holder(h) {}
};

// A record addressed by its key instead of its id.
struct KeyRef {
    std::string ns, key;
};

// An idempotency key presented again with a different request. Applying it
// would make the key mean two writes; returning the first result would
// answer a request that was never made.
struct IdempotencyKeyReused : std::runtime_error { using std::runtime_error::runtime_error; };

// One operation of DB::write_batch(). Every kind has the semantics of the
// single-record call it is named after.
struct BatchOp {
    enum Kind : uint8_t {
        PUT    = 1,   // add(): vector + metadata, metadata without edges keeps the old ones
        UPDATE = 2,   // update_metadata(): replaces the record, edges included
        LINK   = 3,   // link(): both ends must be live records after the ops before it
        FORGET = 4,   // forget()
        UPSERT = 5,   // the record whose (meta.namespace_id, meta.key) this is, created
                      // with a fresh id if there is none; PUT with `vec`, else UPDATE
                      // keeping the record's edges when `meta` has none (add()'s rule)
    };
    Kind kind = PUT;
    uint64_t id = 0;
    std::optional<KeyRef> ref;           // address the record by key instead of `id`
    std::optional<KeyRef> to_ref;        // LINK: the target by key instead of `to_id`
    // Compare-and-set: the record's version before the batch must equal this
    // (0 = the record must not exist). Unset = unconditional.
    std::optional<uint64_t> expected;
    Metadata meta;                       // PUT, UPDATE
    std::vector<float> vec;              // PUT
    std::string modality = "text";       // PUT
    uint64_t to_id = 0;                  // LINK
    std::string rel_type = "related_to"; // LINK
    float weight = 1.0f;                 // LINK
    // LINK: the target must be live too. False gives link()'s rule (a
    // forward reference is allowed) — what a single-edge request means.
    bool require_target = true;
};

// What a batch did: per op, the record it changed and that record's version
// afterwards. `replayed` = the idempotency key had been seen, and this is the
// first attempt's result; nothing was applied this time.
struct BatchResult {
    std::vector<uint64_t> ids;
    std::vector<uint64_t> versions;
    bool replayed = false;
    std::string note;   // what the caller stored with the key (a response body, say)
};

// The distance space of a per-namespace graph. Its "vectors" are 8-byte
// references into the modality's own index, so a namespace graph adds links,
// not a second copy of the data: a reference is a slot of that index, or —
// top bit set — a pointer to the query's bytes, already in the index's
// storage format. Distances are the index's own kernel on the same bytes,
// so they rank exactly as the modality's do (int8 included).
class SlotRefSpace : public hnswlib::SpaceInterface<float> {
public:
    static constexpr uint64_t QUERY = uint64_t{1} << 63;
    struct Param {
        const hnswlib::HierarchicalNSW<float>* base;
        hnswlib::DISTFUNC<float> fn;
        void* fn_param;
    };
    explicit SlotRefSpace(Param p) : p_(p) {}
    static uint64_t query_ref(const void* bytes) {
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(bytes)) | QUERY;
    }
    size_t get_data_size() override { return sizeof(uint64_t); }
    hnswlib::DISTFUNC<float> get_dist_func() override { return &dist; }
    void* get_dist_func_param() override { return &p_; }

private:
    Param p_;
    static const void* deref(const void* ref, const Param* p) {
        uint64_t v;
        std::memcpy(&v, ref, sizeof v);
        if (v & QUERY) return reinterpret_cast<const void*>(static_cast<uintptr_t>(v & ~QUERY));
        return p->base->getDataByInternalId(static_cast<hnswlib::tableint>(v));
    }
    static float dist(const void* a, const void* b, const void* param) {
        const auto* p = static_cast<const Param*>(param);
        return p->fn(deref(a, p), deref(b, p), p->fn_param);
    }
};

class DB {
private:
    struct ModalityIndex {
        std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
        std::unique_ptr<hnswlib::SpaceInterface<float>> space;  // L2Space or Int8L2Space
        size_t dim;
        bool  int8  = false;   // in-RAM int8 storage (4x smaller)
        float scale = 0.0f;    // global quant scale when int8 (= max_abs / 127)
        // The v10 file holds an image of THIS index, so a commit can append
        // only its dirty slots. False for an index the file has never seen —
        // new, rebuilt by compaction or set_int8_ram, or read from v3-v9 —
        // whose next commit writes it whole.
        bool  persisted = false;
        // Slots whose vectors sit in the file's VECTORS image (format v10),
        // and whether this process maps them rather than holding a copy.
        size_t image_count = 0;
        bool   mapped      = false;
        // ── Full precision for an int8 modality (refine > 0) ──────────
        // int8 search re-ranks its top k x refine candidates by exact float
        // L2. The floats are not held in RAM: a slot's lives in the file at
        // fv_off[slot] (a FULLVEC record; 0 = none), or — until a commit puts
        // it there — in the WAL record that logged it (fv_pending; `wal` keeps
        // that WAL generation open across its rotation and removal). Only a
        // conversion of existing float vectors holds copies in `v`.
        // `seq` tells a commit's install whether the entry it wrote is still
        // the newest: a write landing between commit_begin() and the install
        // replaces it, and must not be released.
        struct PendingFull {
            uint64_t seq = 0;
            std::shared_ptr<const fsio::RandomReader> wal;
            uint64_t off = 0;               // of the floats in `wal`
            std::vector<float> v;           // when there is no `wal`
        };
        uint32_t refine = 0;
        std::vector<uint64_t> fv_off;
        std::unordered_map<uint32_t, PendingFull> fv_pending;
        // Bumped whenever the index is replaced in place (int8 conversion),
        // so a compaction built from the old one does not swap over it.
        uint64_t epoch = 0;
        // Bumped whenever the slots or the space change (compaction's swap,
        // int8 conversion): every namespace graph refers to slots, and one
        // built against the old ones is dropped.
        uint64_t ann_gen = 0;
    };

    // ── Per-namespace ANN ────────────────────────────────────────────
    // A filtered search inside a namespace ranks exactly over its records —
    // complete, but O(namespace): 3.8 ms for a 50k-record tenant against
    // 0.39 ms for an unfiltered walk (100k x 384). A namespace with at least
    // FEATHER_NS_ANN_MIN records in a modality gets its own HNSW graph over
    // the modality's slots (SlotRefSpace: ~160 B a node, no vector copies).
    // Built in the background when a search first needs it — that search,
    // and any until it is ready, scans as before — then kept current by
    // every write. Not persisted: rebuilt on demand after a reopen.
    struct NsGraph {
        std::unique_ptr<SlotRefSpace> space;
        std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
        uint64_t gen = 0;   // ModalityIndex::ann_gen it was built against
    };
    using NsKey = std::pair<std::string, std::string>;   // (modality, namespace)
    std::map<NsKey, std::unique_ptr<NsGraph>> ns_graphs_;  // guarded by mutex_
    std::thread ns_builder_;
    std::mutex ns_mx_;                       // guards the queue below, not mutex_ state
    std::condition_variable ns_cv_;
    std::deque<NsKey> ns_queue_;
    std::set<NsKey> ns_pending_;
    bool ns_building_    = false;
    bool stop_ns_builder_ = false;

    std::unordered_map<std::string, ModalityIndex> modality_indices_;
    std::string path_;
    std::string wal_path_;

    // ── Inter-process locking ────────────────────────────────────────────
    // Held for the whole lifetime of the handle, on a sidecar `<path>.lock`.
    //
    // What this does and does not buy, because the distinction matters: it does
    // NOT make two processes able to write one file concurrently. It cannot.
    // Each DB holds the entire dataset in RAM and commits from that in-memory
    // view, so even a mutex around save() would have process B write B's view
    // — which never contained A's records. A's writes vanish, both processes
    // exit 0. Measured before this change: 20 of 20 records lost. (v10's
    // append-only commits do not change this: B's commit describes B's graph
    // slots and B's header supersedes A's.) Concurrent writing needs
    // merge-on-commit or a paged storage engine, not a lock.
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
    // Serialises save() against save(): concurrent commits would interleave
    // their appends and header switches (and collide on ".tmp" for a full
    // rewrite). Also guards the v10 commit state below. Always taken BEFORE mutex_.
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
    std::unordered_map<std::string, uint64_t> key_index_;   // key_ref(ns, key) → THE live record
    // The namespaces of ns_index_, sorted: a namespace_prefix filter is a range.
    std::set<std::string> ns_names_;
    // Live records of each namespace ("" included) ordered by (timestamp, id):
    // "the newest n" is a walk from the end, not a scan and a sort.
    using TimeKey = std::pair<int64_t, uint64_t>;
    std::unordered_map<std::string, std::set<TimeKey>> time_index_;
    // The same, ordered by (valid_time(), id): "what happened most recently"
    // where a record's event time differs from when it was written.
    std::unordered_map<std::string, std::set<TimeKey>> valid_index_;

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
    // global scale = clip/127. Persisted in file format v8.
    std::unordered_map<std::string, float> int8_ram_scale_;
    // Modalities waiting for INT8_TRAIN_MIN vectors to train their scale on;
    // they stay float until then (set_int8_ram without max_abs).
    std::unordered_set<std::string> int8_train_;
    // Re-rank factor per int8 modality; > 0 keeps float originals in the file.
    std::unordered_map<std::string, uint32_t> int8_refine_;
    // The file the FULLVEC offsets point into. Opened with the load (same
    // inode as the one parsed) and swapped with the offsets by a commit that
    // rewrites the file. Guarded by mutex_.
    fsio::RandomReader fv_file_;
    uint64_t fv_seq_ = 0;          // PendingFull::seq source; under mutex_ (exclusive)

    // ── BM25 Inverted Index ──────────────────────────────────────────
    // A posting names its document by a u32 doc number, not the u64 id, and
    // the length lives once per doc in bm25_doc_len_: 8 bytes a posting
    // instead of 16, at ~50 postings a document. Numbers are handed out in
    // increasing order and never reused until the next rebuild, so a stale
    // posting can only point at the doc that wrote it (same as when it held
    // the id), and a list stays in number order for sequential lookups.
    struct PostingEntry { uint32_t doc; uint32_t term_freq; };
    std::unordered_map<std::string, std::vector<PostingEntry>> bm25_index_;
    std::unordered_map<uint64_t, uint32_t> doc_lengths_;   // id -> doc number
    std::vector<uint64_t> bm25_doc_id_;                    // doc number -> id
    std::vector<uint32_t> bm25_doc_len_;                   // doc number -> tokens
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
        // Versioning and atomic groups (0.22). A build that predates them
        // skips unknown ops, so it still replays every data record: what it
        // loses on a downgrade is versions, keys and all-or-nothing.
        STAMP  = 0x07,   // [version u64]: the version of the record just before it
        BEGIN  = 0x08,   // [count u32][key str16][fingerprint str16][created i64][note str32]
        COMMIT = 0x09,   // [count u32][version u64 x count]
                         //   [nres u32][(id u64, version u64) x nres] — the batch's result
        // [generation u64]: the first record of every WAL file — the base
        // file's commit generation its operations follow. A read-only open
        // checks base -> <wal>.old -> <wal> forms an unbroken chain, and
        // retries when a commit finished between its reads (see
        // read_snapshot()). Written in the same write as the header.
        WALGEN = 0x0A,
    };
    // What replay_wal() saw of a WAL file: whether it held any record, and
    // the generation its WALGEN record named (none: written before them).
    struct WalSeen {
        bool present = false;
        std::optional<uint64_t> gen;
    };
    // Where an appended WAL record sits: its generation of the WAL, held open
    // (it outlives the rotation to <wal>.old and the removal), and the record's
    // offset. `file` is null when there is no readable WAL to point into.
    struct WalPos {
        std::shared_ptr<const fsio::RandomReader> file;
        uint64_t off = 0;
    };
    // Offset of the vector inside an ADD record at `rec`: [op u8][id u64]
    // [plen u32] then encode_add()'s [mod_len u16][modality][dim u32][floats].
    static uint64_t wal_vector_offset(uint64_t rec, const std::string& modality) {
        return rec + 13 + 2 + modality.size() + 4;
    }

    // ── WAL group commit ─────────────────────────────────────────────
    // Records are appended (one unbuffered fwrite) inside the exclusive data lock, so
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

    // ── Failed writes are fatal to the handle ────────────────────────
    // Set by the first WAL write error or fsync error and never cleared:
    // every later write and save on this handle is refused until the file is
    // reopened, as PostgreSQL PANICs on an fsync error. Retrying is what loses
    // data — a WAL with a torn record in the middle replays only up to it, and
    // an fsync retried after a failure can succeed without the pages that
    // failed. Reopening replays the WAL that IS on disk and cuts a torn tail.
    // poison_reason_ is written once, under poison_mx_ (a leaf lock: nothing
    // else is ever taken while holding it), before poisoned_ is set.
    mutable std::atomic<bool> poisoned_{false};
    mutable std::mutex        poison_mx_;
    mutable std::string       poison_reason_;

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

    // ── File format v10: append-only commits ─────────────────────────
    // A commit appends what changed since the previous one and then switches
    // the header (format_v10.h), so a save costs the size of the change rather
    // than the size of the DB — 182 MB per save at 100k x 384 before.
    //
    // What changed is tracked at the source: hnswlib marks dirty slots itself
    // (HierarchicalNSW::dirty_), every metadata mutation calls
    // mark_meta_dirty(), and a query's salience touch queues its record in
    // salience_dirty_. All three are consumed by commit_begin().
    //
    // Guarded by save_mutex_ (every commit holds it):
    bool       file_v10_          = false;   // path_ holds a v10 file we may append to
    uint32_t   base_version_      = 0;       // format of the file as opened (stats)
    uint64_t   committed_end_     = 0;
    uint64_t   generation_        = 0;
    bool       commit_failed_     = false;   // dirty sets were lost: next commit is full
    std::FILE* data_fh_           = nullptr; // writer's append handle, opened lazily
    uint64_t   last_commit_bytes_ = 0;
    bool       last_commit_full_  = false;
    uint64_t   commits_           = 0;
    uint64_t   vacuums_           = 0;
    // Guarded by mutex_: written by mutations (exclusive) and consumed by a
    // commit, which holds it shared — no writer runs — under save_mutex_.
    std::unordered_set<uint64_t> dirty_meta_;
    // Compare-and-set versions: every change to a record takes ++version_clock_
    // (Metadata::version). One clock per DB, persisted in VERSIONS sections, so
    // a version is never reused, not even by a record re-created after a
    // delete. clock_loaded_: the file carried one (v3-v9 files do not, and get
    // a time-based floor instead, see seed_version_clock()).
    uint64_t version_clock_ = 0;
    bool     clock_loaded_  = false;
    // Idempotency keys of write_batch(): key -> the first attempt's result.
    struct IdemEntry {
        std::string fingerprint;
        int64_t     created = 0;
        std::vector<std::pair<uint64_t, uint64_t>> results;   // (id, version) per op
        std::string note;
    };
    static constexpr size_t MAX_IDEM_NOTE = 64 * 1024;
    static void put_str32(std::string& out, const std::string& s) {
        v10::put<uint32_t>(out, static_cast<uint32_t>(s.size()));
        out.append(s);
    }
    static std::string get_str32(v10::Reader& r) {
        const uint32_t n = r.get<uint32_t>();
        return std::string(r.take(n), n);
    }
    std::unordered_map<std::string, IdemEntry> idem_;
    std::deque<std::pair<int64_t, std::string>> idem_order_;  // insertion order, for expiry
    std::unordered_set<std::string> dirty_idem_;               // keys the next commit writes
    bool     vacuum_requested_ = false;      // purge/compact/set_quantized: rewrite next
    uint64_t meta_rec_bytes_   = 0;          // META record sizes seen, for the live
    uint64_t meta_rec_count_   = 0;          //   estimate that triggers a vacuum
    // Queries record salience under the SHARED lock, so they cannot touch
    // dirty_meta_; they queue here instead (once per record per commit, via
    // Metadata::recall_unsaved).
    mutable std::mutex            salience_mx_;
    mutable std::vector<uint64_t> salience_dirty_;

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
    // set_int8_ram's default re-rank factor: candidates fetched per result.
    static constexpr uint32_t DEFAULT_INT8_REFINE = 2;

    // Ensure the index can hold at least `target` elements. NOT thread-safe
    // (resizeIndex reallocs every backing buffer) — call before any add, and
    // before parallel_add for the full batch size, never from inside it.
    static void reserve(ModalityIndex& m_idx, size_t target) {
        size_t cap = m_idx.index->getMaxElements();
        if (target > cap)
            m_idx.index->resizeIndex(std::max(target, cap * 2));
    }

    // Deleted slots are NOT reused: a new insert always takes a fresh slot and
    // tombstones stay as routing nodes until compact(). hnswlib's replace path
    // (updatePoint on the reused slot) re-prunes every old neighbour of the
    // slot and can drop another record's last incoming link, leaving it
    // unreachable even to its own vector (0.2-0.3% of records under
    // forget+add churn at ef=400). Reclaim space with set_auto_compact().
    static std::unique_ptr<hnswlib::HierarchicalNSW<float>>
    make_hnsw(hnswlib::SpaceInterface<float>* space, size_t capacity,
              size_t M = 16, size_t ef_construction = 200) {
        auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space, capacity, M, ef_construction, /*random_seed=*/100, /*allow_replace_deleted=*/false);
        index->setEf(DEFAULT_EF);
        return index;
    }

    // (Re)create a modality's index, empty, honouring its in-RAM int8 config.
    ModalityIndex& create_index(const std::string& modality, size_t dim,
                                size_t M = 16, size_t ef_construction = 200) {
        auto cfg = int8_ram_scale_.find(modality);
        bool int8 = cfg != int8_ram_scale_.end();
        float scale = int8 ? cfg->second : 0.0f;
        std::unique_ptr<hnswlib::SpaceInterface<float>> space;
        if (int8) space = std::make_unique<hnswlib::Int8L2Space>(dim, scale);
        else      space = std::make_unique<hnswlib::L2Space>(dim);
        auto index = make_hnsw(space.get(), INITIAL_MAX_ELEMENTS, M, ef_construction);
        auto& slot = modality_indices_[modality];
        const uint64_t epoch = slot.epoch + 1;
        slot = {std::move(index), std::move(space), dim, int8, scale};
        slot.epoch = epoch;
        if (int8) {
            auto r = int8_refine_.find(modality);
            slot.refine = r != int8_refine_.end() ? r->second : 0;
        }
        return slot;
    }

    ModalityIndex& get_or_create_index(const std::string& modality, size_t dim) {
        auto it = modality_indices_.find(modality);
        if (it != modality_indices_.end()) return it->second;
        return create_index(modality, dim);
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

    // Insert raw storage-format bytes. A new label takes a fresh slot (see
    // make_hnsw); an existing label, a forgotten one included, is updated in
    // place.
    static void insert_raw(ModalityIndex& m_idx, uint64_t id, const void* data) {
        m_idx.index->addPoint(data, id);
    }

    // Insert a float vector into a modality index, quantizing to int8 first if
    // the modality is in-RAM int8. Centralises the float-vs-int8 store decision.
    static void add_point(ModalityIndex& m_idx, uint64_t id, const float* vec) {
        if (m_idx.int8) {
            // Reusable per-thread buffer — a fresh std::vector per call across the
            // parallel insert pool churns the allocator and inflates RSS by ~MBs.
            static thread_local std::vector<int8_t> q;
            q.resize(m_idx.dim);
            quantize_global(vec, m_idx.dim, m_idx.scale, q.data());
            insert_raw(m_idx, id, q.data());
        } else {
            insert_raw(m_idx, id, vec);
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

    // ── Full precision for in-RAM int8 (refine) ──────────────────────
    // Remember the float original of `id`'s vector until a commit writes it
    // to the file. Called after the insert, so it lands under the slot the
    // insert chose. Caller holds mutex_ exclusively.
    // `at` is where the WAL record of this write holds the floats; without
    // one (a reader that could not pin the WAL it replays) they are copied.
    void keep_full(ModalityIndex& m, uint64_t id, const float* vec, const WalPos& at) {
        if (!m.int8 || !m.refine) return;
        auto it = m.index->label_lookup_.find(id);
        if (it == m.index->label_lookup_.end()) return;
        auto& p = m.fv_pending[static_cast<uint32_t>(it->second)];
        p.seq = ++fv_seq_;
        p.wal = at.file;
        p.off = at.off;
        if (at.file) std::vector<float>().swap(p.v);
        else         p.v.assign(vec, vec + m.dim);
    }

    // The float original of `slot` into `out` (dim floats): pending, else read
    // from the file. False when there is none (written before refine was on,
    // or by a build without it) — the caller keeps the int8 distance.
    bool full_vector(const ModalityIndex& m, uint32_t slot, float* out) const {
        auto p = m.fv_pending.find(slot);
        if (p != m.fv_pending.end()) {
            if (p->second.wal) p->second.wal->read_at(p->second.off, out, m.dim * sizeof(float));
            else               std::memcpy(out, p->second.v.data(), m.dim * sizeof(float));
            return true;
        }
        if (slot < m.fv_off.size() && m.fv_off[slot] && fv_file_.is_open()) {
            fv_file_.read_at(m.fv_off[slot], out, m.dim * sizeof(float));
            return true;
        }
        return false;
    }

    // A stored vector as float32: the original when the modality keeps one,
    // else read_vector_label() (dequantized for int8).
    std::vector<float> read_vector_best(const ModalityIndex& m, uint64_t id) const {
        if (m.int8 && m.refine) {
            auto it = m.index->label_lookup_.find(id);
            if (it != m.index->label_lookup_.end() && !m.index->isMarkedDeleted(it->second)) {
                std::vector<float> out(m.dim);
                if (full_vector(m, static_cast<uint32_t>(it->second), out.data())) return out;
            }
        }
        return read_vector_label(m, id);
    }

    // Vectors a modality must hold before set_int8_ram() without max_abs
    // trains its scale (FEATHER_INT8_TRAIN_MIN). 1,000 at 384-d gave a clip
    // within 1-4% of one trained on all 20,000.
    static size_t int8_train_min() {
        if (const char* e = std::getenv("FEATHER_INT8_TRAIN_MIN")) {
            long v = std::atol(e);
            if (v >= 1) return static_cast<size_t>(v);
        }
        return 1000;
    }

    // The int8 clip (scale = clip / 127) trained on a float modality: the one
    // minimising the mean squared quantization error over a sample of its
    // components. A larger clip costs rounding error on every component, a
    // smaller one clamps the few beyond it. Unit-norm embeddings rarely have
    // a component above 0.25, so the old fixed max_abs = 1.0 left ~2 of the 8
    // bits unused; max|x| wastes levels on a few outlier dimensions. Measured
    // at 384-d, exact scan, recall@10: 0.906 at 1.0, 0.972 at max|x|, 0.978
    // here; with outlier dimensions 0.941 / 0.949 / 0.956.
    static float train_int8_clip(const ModalityIndex& m) {
        const auto& ix = *m.index;
        const size_t n = ix.cur_element_count, dim = m.dim;
        constexpr size_t SAMPLE = size_t{1} << 18;            // components
        const size_t vecs = std::max<size_t>(1, SAMPLE / dim);
        const size_t step = std::max<size_t>(1, n / vecs);
        std::vector<float> a;
        a.reserve((n / step + 1) * dim);
        float hi = 0.0f;
        for (size_t i = 0; i < n; i += step) {
            const auto id = static_cast<hnswlib::tableint>(i);
            if (ix.isMarkedDeleted(id)) continue;
            const float* v = reinterpret_cast<const float*>(ix.getDataByInternalId(id));
            for (size_t d = 0; d < dim; ++d) {
                const float x = std::fabs(v[d]);
                if (!std::isfinite(x)) continue;
                a.push_back(x);
                hi = std::max(hi, x);
            }
        }
        if (!(hi > 0.0f)) return 1.0f;
        auto err = [&](double c) {
            const double s = c / 127.0, inv = 1.0 / s;
            double e = 0.0;
            for (float x : a) {
                const double q = std::min(std::nearbyint(x * inv), 127.0) * s;
                e += (x - q) * (x - q);
            }
            return e;
        };
        // A coarse grid over (hi/64, hi], then a fine one around its best.
        double best_c = hi, best_e = err(hi);
        auto scan = [&](double lo, double up, int points) {
            for (int g = 0; g < points; ++g) {
                const double c = lo + (up - lo) * g / (points - 1);
                const double e = err(c);
                if (e < best_e) { best_e = e; best_c = c; }
            }
        };
        const double lo = hi / 64.0, step_c = (hi - lo) / 127.0;
        scan(lo, hi, 128);
        scan(std::max(lo, best_c - step_c), std::min<double>(hi, best_c + step_c), 33);
        return static_cast<float>(best_c);
    }

    // Turn float modality `name` into an in-RAM int8 one under `scale`,
    // keeping its graph: links, levels, labels and delete marks are copied
    // slot for slot through the restore path a load uses; only the vectors are
    // re-encoded. With refine on, every live float original becomes pending —
    // the next save() writes them, and until then they are what the float
    // index held. Caller holds mutex_ exclusively.
    void convert_to_int8_nolock(const std::string& name, float scale) {
        auto& m = modality_indices_.at(name);
        auto& old = *m.index;
        const size_t n = old.cur_element_count;
        auto space = std::make_unique<hnswlib::Int8L2Space>(m.dim, scale);
        auto index = make_hnsw(space.get(), std::max(INITIAL_MAX_ELEMENTS, n),
                               old.M_, old.ef_construction_);
        index->restoreBegin(n);
        std::vector<int8_t> q(m.dim);
        for (size_t i = 0; i < n; ++i) {
            const auto id = static_cast<hnswlib::tableint>(i);
            const int level = old.element_levels_[i];
            const hnswlib::labeltype label = old.getExternalLabel(id);
            quantize_global(reinterpret_cast<const float*>(old.getDataByInternalId(id)),
                            m.dim, scale, q.data());
            index->restoreElement(id, level, reinterpret_cast<const char*>(old.get_linklist0(id)),
                                  level > 0 ? old.linkLists_[i] : nullptr, &label,
                                  reinterpret_cast<const char*>(q.data()));
        }
        index->restoreFinish(n ? static_cast<hnswlib::tableint>(old.enterpoint_node_) : 0,
                             n ? old.maxlevel_ : -1);
        index->setEf(old.ef_);

        auto r = int8_refine_.find(name);
        const uint32_t refine = r != int8_refine_.end() ? r->second : 0;
        std::unordered_map<uint32_t, ModalityIndex::PendingFull> pending;
        if (refine) {
            pending.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                const auto id = static_cast<hnswlib::tableint>(i);
                if (old.isMarkedDeleted(id)) continue;
                const float* v = reinterpret_cast<const float*>(old.getDataByInternalId(id));
                pending[static_cast<uint32_t>(i)] = {++fv_seq_, nullptr, 0, std::vector<float>(v, v + m.dim)};
            }
        }
        int8_ram_scale_[name] = scale;
        int8_train_.erase(name);
        m.index       = std::move(index);     // frees the float index
        m.space       = std::move(space);
        m.int8        = true;
        m.scale       = scale;
        m.persisted   = false;
        m.image_count = 0;
        m.mapped      = false;
        m.refine      = refine;
        m.fv_off.clear();
        m.fv_pending  = std::move(pending);
        m.epoch      += 1;
        drop_ns_graphs_nolock(name, m);
        // The float image in the file is now garbage four times the size of
        // what replaces it: rewrite rather than append next to it.
        vacuum_requested_ = true;
    }

    // Train and convert a modality waiting on set_int8_ram() once it holds
    // enough vectors. Caller holds mutex_ exclusively.
    void maybe_train_int8_nolock(const std::string& name) {
        if (int8_train_.empty() || !int8_train_.count(name)) return;
        auto it = modality_indices_.find(name);
        if (it == modality_indices_.end() || it->second.int8) return;
        const auto& ix = *it->second.index;
        if (ix.cur_element_count - ix.num_deleted_.load() < int8_train_min()) return;
        convert_to_int8_nolock(name, train_int8_clip(it->second) / 127.0f);
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

    // Length-prefixed, so no (namespace, key) split can produce another's string.
    static std::string key_ref(const std::string& ns, const std::string& key) {
        return std::to_string(ns.size()) + ':' + ns + key;
    }

    // What BM25 indexes for a record: its content, and its key, so a search
    // for the name a memory was stored under finds it.
    static std::string bm25_text(const Metadata& m) {
        return m.key.empty() ? m.content : m.content + "\n" + m.key;
    }

    // Refuse a write that would give `m`'s key to a second live record.
    void check_key_free_nolock(uint64_t id, const Metadata& m) const {
        if (m.key.empty() || is_dead_meta(m)) return;
        auto it = key_index_.find(key_ref(m.namespace_id, m.key));
        if (it != key_index_.end() && it->second != id) throw DuplicateKey(m.namespace_id, m.key, it->second);
    }

    void index_meta(uint64_t id, const Metadata& m) {
        time_index_[m.namespace_id].insert({m.timestamp, id});
        valid_index_[m.namespace_id].insert({m.valid_time(), id});
        if (!m.key.empty()) key_index_[key_ref(m.namespace_id, m.key)] = id;
        if (!m.namespace_id.empty()) {
            auto& set = ns_index_[m.namespace_id];
            if (set.empty()) ns_names_.insert(m.namespace_id);
            set.insert(id);
        }
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
        if (auto t = time_index_.find(m.namespace_id); t != time_index_.end()) {
            t->second.erase({m.timestamp, id});
            if (t->second.empty()) time_index_.erase(t);
        }
        if (auto t = valid_index_.find(m.namespace_id); t != valid_index_.end()) {
            t->second.erase({m.valid_time(), id});
            if (t->second.empty()) valid_index_.erase(t);
        }
        if (!m.key.empty()) {
            auto k = key_index_.find(key_ref(m.namespace_id, m.key));
            if (k != key_index_.end() && k->second == id) key_index_.erase(k);
        }
        if (!m.namespace_id.empty()) {
            drop(ns_index_, m.namespace_id);
            if (!ns_index_.count(m.namespace_id)) ns_names_.erase(m.namespace_id);
            ns_ann_unnote_nolock(id, m.namespace_id);
        }
        if (!m.entity_id.empty())    drop(entity_index_, m.entity_id);
        for (const auto& [k, v] : m.attributes) drop(attr_index_, attr_key(k, v));
    }

    void build_secondary_indexes() {
        ns_index_.clear();
        entity_index_.clear();
        attr_index_.clear();
        key_index_.clear();
        ns_names_.clear();
        time_index_.clear();
        valid_index_.clear();
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
    // Every namespace that is `prefix` or below it, from the sorted names:
    // the exact name, then the range ["prefix.", "prefix/") — '/' is '.' + 1.
    template <class F>
    void for_each_ns_under(const std::string& prefix, F&& f) const {
        if (ns_index_.count(prefix)) f(prefix);
        const std::string lo = prefix + ".", hi = prefix + "/";
        for (auto it = ns_names_.lower_bound(lo); it != ns_names_.end() && *it < hi; ++it) f(*it);
    }

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
        // A prefix is the union of the namespaces under it: one set, or a
        // merged copy of several. (The empty prefix constrains nothing.)
        std::unordered_set<uint64_t> merged;
        if (f.namespace_prefix && !f.namespace_prefix->empty()) {
            indexed = true;
            std::vector<const std::unordered_set<uint64_t>*> under;
            for_each_ns_under(*f.namespace_prefix,
                              [&](const std::string& ns) { under.push_back(&ns_index_.at(ns)); });
            if (under.empty()) return {};
            if (under.size() == 1) {
                sets.push_back(under[0]);
            } else {
                size_t n = 0;
                for (auto* u : under) n += u->size();
                merged.reserve(n);
                for (auto* u : under) merged.insert(u->begin(), u->end());
                sets.push_back(&merged);
            }
        }
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

    // ── Per-namespace ANN: building and upkeep ──────────────────────
    // Off unless asked for: filtered results are exact by default (CONTRACTS.md).
    static size_t ns_ann_min_default() {
        static const size_t v = [] {
            // FEATHER_PREFILTER_MODE=exact asks for exact filtered search
            // everywhere: no namespace graphs either.
            const char* mode = std::getenv("FEATHER_PREFILTER_MODE");
            if (mode && !std::strcmp(mode, "exact")) return size_t{0};
            const char* e = std::getenv("FEATHER_NS_ANN_MIN");
            if (!e) return size_t{0};
            long long x = std::atoll(e);
            return x > 0 ? static_cast<size_t>(x) : size_t{0};     // 0 = off
        }();
        return v;
    }
    // Atomic: read by searches under the shared lock, set by set_namespace_ann_min().
    std::atomic<size_t> ns_ann_min_{ns_ann_min_default()};
    size_t ns_ann_min() const { return ns_ann_min_.load(std::memory_order_relaxed); }
    static constexpr size_t NS_ANN_EF_CONSTRUCTION = 100;
    static constexpr size_t NS_ANN_CHUNK = 1024;   // inserts per shared-lock hold

    // The graph of (modality, ns) if it is built and current. Caller holds mutex_.
    const NsGraph* ns_graph_nolock(const std::string& modality, const ModalityIndex& m,
                                   const std::string& ns) const {
        auto it = ns_graphs_.find({modality, ns});
        if (it == ns_graphs_.end() || it->second->gen != m.ann_gen) return nullptr;
        return it->second.get();
    }

    // Ask the builder for a graph. Safe under the shared lock: it touches only
    // the queue, which has its own mutex.
    void request_ns_graph(const std::string& modality, const std::string& ns) const {
        auto* self = const_cast<DB*>(this);
        std::lock_guard<std::mutex> g(self->ns_mx_);
        if (self->stop_ns_builder_ || !self->ns_pending_.insert({modality, ns}).second) return;
        self->ns_queue_.push_back({modality, ns});
        if (!self->ns_builder_.joinable())
            self->ns_builder_ = std::thread([self] { self->ns_builder_loop(); });
        self->ns_cv_.notify_one();
    }

    void ns_builder_loop() {
        std::unique_lock<std::mutex> g(ns_mx_);
        for (;;) {
            ns_cv_.wait(g, [&] { return stop_ns_builder_ || !ns_queue_.empty(); });
            if (stop_ns_builder_) return;
            NsKey key = ns_queue_.front();
            ns_queue_.pop_front();
            ns_building_ = true;
            g.unlock();
            try { build_ns_graph(key); } catch (...) {}   // a failed build just means "scan"
            g.lock();
            ns_building_ = false;
            ns_pending_.erase(key);
            ns_cv_.notify_all();          // wake wait_for_namespace_graphs()
        }
    }

    void stop_ns_builder() {
        {
            std::lock_guard<std::mutex> g(ns_mx_);
            stop_ns_builder_ = true;
        }
        ns_cv_.notify_all();
        if (ns_builder_.joinable() && ns_builder_.get_id() != std::this_thread::get_id())
            ns_builder_.join();
    }

    bool ns_build_aborted(const std::string& modality, uint64_t gen) const {
        if (closed_) return true;
        {
            std::lock_guard<std::mutex> g(const_cast<std::mutex&>(ns_mx_));
            if (stop_ns_builder_) return true;
        }
        auto mit = modality_indices_.find(modality);
        return mit == modality_indices_.end() || mit->second.ann_gen != gen;
    }

    // Build the graph of (modality, ns) without holding up writers: inserts
    // run in chunks under the SHARED lock (they read the modality's vectors,
    // which only an exclusive holder moves), the graph is private until it is
    // installed, and the install — under the exclusive lock — adds whatever
    // the namespace gained meanwhile. A compaction or int8 conversion during
    // the build (ann_gen moves) abandons it.
    void build_ns_graph(const NsKey& key) {
        const auto& [modality, ns] = key;
        auto g = std::make_unique<NsGraph>();
        std::vector<std::pair<uint64_t, uint32_t>> members;     // (record, slot)
        {
            std::shared_lock<RWMutex> lock(mutex_);
            if (closed_) return;
            auto mit = modality_indices_.find(modality);
            auto sit = ns_index_.find(ns);
            if (mit == modality_indices_.end() || sit == ns_index_.end()) return;
            const auto& m = mit->second;
            if (ns_graph_nolock(modality, m, ns)) return;
            const auto& idx = *m.index;
            for (uint64_t id : sit->second) {
                auto lit = idx.label_lookup_.find(id);
                if (lit != idx.label_lookup_.end() && !idx.isMarkedDeleted(lit->second))
                    members.emplace_back(id, lit->second);
            }
            if (members.size() < std::max<size_t>(ns_ann_min(), 1)) return;
            g->gen = m.ann_gen;
            g->space = std::make_unique<SlotRefSpace>(SlotRefSpace::Param{
                m.index.get(), m.space->get_dist_func(), m.space->get_dist_func_param()});
            g->index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                g->space.get(), members.size() + members.size() / 4 + 64, 16, NS_ANN_EF_CONSTRUCTION);
        }
        for (size_t s = 0; s < members.size(); s += NS_ANN_CHUNK) {
            std::shared_lock<RWMutex> lock(mutex_);
            if (ns_build_aborted(modality, g->gen)) return;
            const auto& idx = *modality_indices_.at(modality).index;
            const size_t e = std::min(members.size(), s + NS_ANN_CHUNK);
            run_parallel(e - s, [&](size_t k) {
                const auto [id, slot] = members[s + k];
                // A slot can be deleted (or relabelled by compaction's swap)
                // since the snapshot; the install picks it up.
                if (idx.getExternalLabel(slot) != id || idx.isMarkedDeleted(slot)) return;
                const uint64_t ref = slot;
                g->index->addPoint(&ref, id);
            });
        }
        std::unique_lock<RWMutex> lock(mutex_);
        if (ns_build_aborted(modality, g->gen)) return;
        auto sit = ns_index_.find(ns);
        if (sit != ns_index_.end()) {
            const auto& idx = *modality_indices_.at(modality).index;
            for (uint64_t id : sit->second) ns_graph_put(*g, idx, id, /*vector_changed=*/false);
        }
        // ...and what left the namespace, or died, while it was being built.
        for (const auto& [id, slot] : g->index->label_lookup_) {
            auto it = metadata_store_.find(id);
            if (it == metadata_store_.end() || is_dead_meta(it->second) || it->second.namespace_id != ns)
                if (!g->index->isMarkedDeleted(slot)) g->index->markDeletedInternal(slot);
        }
        ns_graphs_[key] = std::move(g);
    }

    // Make record `id` current in graph `g`: added if missing, re-linked if its
    // slot or (vector_changed) its vector changed. Caller holds mutex_
    // exclusively, or owns `g`.
    static void ns_graph_put(NsGraph& g, const hnswlib::HierarchicalNSW<float>& idx,
                             uint64_t id, bool vector_changed) {
        auto lit = idx.label_lookup_.find(id);
        if (lit == idx.label_lookup_.end() || idx.isMarkedDeleted(lit->second)) return;
        const uint64_t ref = lit->second;
        auto& gi = *g.index;
        auto gl = gi.label_lookup_.find(id);
        if (gl != gi.label_lookup_.end()) {
            uint64_t have;
            std::memcpy(&have, gi.getDataByInternalId(gl->second), sizeof have);
            if (have == ref && !vector_changed) {
                // Back in the namespace (or a metadata-only write): its links
                // are still right, only the membership mark changes.
                if (gi.isMarkedDeleted(gl->second)) gi.unmarkDelete(id);
                return;
            }
        } else if (gi.cur_element_count >= gi.getMaxElements()) {
            gi.resizeIndex(gi.getMaxElements() * 2);
        }
        gi.addPoint(&ref, id);           // an existing label is updated, re-linked and unmarked
    }

    // Record `id` left namespace `ns` or died: mark it in that namespace's
    // graphs, so a graph's live nodes are exactly the namespace's live
    // records. That is what lets a namespace-only search walk with no filter
    // callback (hnswlib skips marked nodes itself). Caller holds mutex_
    // exclusively.
    void ns_ann_unnote_nolock(uint64_t id, const std::string& ns) {
        if (ns_graphs_.empty() || ns.empty()) return;
        for (auto& [name, m] : modality_indices_) {
            auto g = ns_graphs_.find({name, ns});
            if (g == ns_graphs_.end() || g->second->gen != m.ann_gen) continue;
            auto& gi = *g->second->index;
            auto gl = gi.label_lookup_.find(id);
            if (gl != gi.label_lookup_.end() && !gi.isMarkedDeleted(gl->second)) gi.markDelete(id);
        }
    }

    // Keep the graphs current after a write to record `id`. A record that left
    // a namespace stays in its old graph, where the search filter skips it.
    // Caller holds mutex_ exclusively.
    void ns_ann_note_nolock(uint64_t id, bool vector_changed) {
        if (ns_graphs_.empty()) return;
        auto it = metadata_store_.find(id);
        if (it == metadata_store_.end() || is_dead_meta(it->second) || it->second.namespace_id.empty()) return;
        for (auto& [name, m] : modality_indices_) {
            auto g = ns_graphs_.find({name, it->second.namespace_id});
            if (g == ns_graphs_.end() || g->second->gen != m.ann_gen) continue;
            ns_graph_put(*g->second, *m.index, id, vector_changed);
        }
    }

    // The slots of `m` changed meaning: its namespace graphs go. Caller holds
    // mutex_ exclusively.
    void drop_ns_graphs_nolock(const std::string& modality, ModalityIndex& m) {
        m.ann_gen += 1;
        for (auto it = ns_graphs_.begin(); it != ns_graphs_.end(); )
            it = it->first.first == modality ? ns_graphs_.erase(it) : std::next(it);
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

    // ── Porter stemmer ───────────────────────────────────────────────────
    //
    // Without it, BM25 matched only exact surface forms: a memory whose content
    // said "allergic" was not found by searching "allergy", and "creative"
    // missed "creatives". For an agent memory that failure is worse than it
    // looks — the caller cannot tell "no such memory" from "stored, but you
    // spelled it differently", so it reads as the agent having forgotten.
    //
    // Safe to change without a format bump: bm25_index_ is never serialised,
    // it is rebuilt from `content` on every load() and after compact(), so an
    // existing .feather re-indexes under the new rules the next time it opens.
    // Both indexing and querying go through tokenize(), so they can never
    // disagree about how a word is folded.
    //
    // FEATHER_STEM=0 restores exact-form matching.
    static bool stemming_enabled() {
        static const bool on = [] {
            const char* e = std::getenv("FEATHER_STEM");
            return !(e && (e[0] == '0' || e[0] == 'n' || e[0] == 'N'));
        }();
        return on;
    }

    static bool stem_vowel(char c) {
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
    }
    /// Porter's consonant test. 'y' is a consonant exactly when the character
    /// before it is not, which makes a run of y's alternate.
    ///
    /// Resolved ARITHMETICALLY, not by recursing. The recursive form was one
    /// frame deep per preceding 'y', so `stem_measure` became O(n^2) and the
    /// stack depth equalled the token length. `tokenize()` splits only on
    /// non-alphanumerics, so a single long run of y's — a base64 blob or an id
    /// run will do, it need not be adversarial — is ONE token. Measured before
    /// this fix: 20k chars 0.84 s, 50k chars 3.27 s, and because
    /// rebuild_bm25_index() runs on every load(), a single 30 KB record cost
    /// 0.58 s on EVERY open of that file, permanently.
    static bool stem_is_consonant(const std::string& w, size_t i) {
        char c = w[i];
        if (stem_vowel(c)) return false;
        if (c != 'y')      return true;
        size_t j = i;
        while (j > 0 && w[j - 1] == 'y') --j;      // start of this run of y's
        // Nothing before the run reads as a vowel, so the first 'y' is a
        // consonant; each further 'y' flips.
        bool base_is_consonant = (j > 0) && !stem_vowel(w[j - 1]);
        bool first_y_consonant = !base_is_consonant;
        return (((i - j) & 1u) == 0) ? first_y_consonant : !first_y_consonant;
    }
    /// Porter's `m`: how many vowel-consonant sequences the stem contains.
    static int stem_measure(const std::string& w) {
        int m = 0; size_t i = 0, n = w.size();
        while (i < n && stem_is_consonant(w, i)) ++i;
        while (i < n) {
            while (i < n && !stem_is_consonant(w, i)) ++i;
            if (i >= n) break;
            ++m;
            while (i < n && stem_is_consonant(w, i)) ++i;
        }
        return m;
    }
    static bool stem_has_vowel(const std::string& w) {
        for (size_t i = 0; i < w.size(); ++i) if (!stem_is_consonant(w, i)) return true;
        return false;
    }
    static bool stem_double_consonant(const std::string& w) {
        size_t n = w.size();
        return n >= 2 && w[n-1] == w[n-2] && stem_is_consonant(w, n-1);
    }
    /// consonant-vowel-consonant where the last is not w, x or y
    static bool stem_cvc(const std::string& w) {
        size_t n = w.size();
        if (n < 3) return false;
        if (!stem_is_consonant(w, n-3) || stem_is_consonant(w, n-2) ||
            !stem_is_consonant(w, n-1)) return false;
        char c = w[n-1];
        return c != 'w' && c != 'x' && c != 'y';
    }
    static bool stem_ends(const std::string& w, const char* suf, size_t& stem_len) {
        size_t sl = std::strlen(suf), n = w.size();
        if (n < sl || w.compare(n - sl, sl, suf) != 0) return false;
        stem_len = n - sl;
        return true;
    }
    /// Replace suffix `suf` with `rep` when the remaining stem has measure > `min_m`.
    static bool stem_replace(std::string& w, const char* suf, const char* rep, int min_m) {
        size_t sl;
        if (!stem_ends(w, suf, sl)) return false;
        std::string stem = w.substr(0, sl);
        if (stem_measure(stem) > min_m) w = stem + rep;
        return true;   // the suffix matched, so no later rule in this step applies
    }

    // The longest word in English is 45 letters. A token past this is a blob,
    // an id or a hash, and folding it cannot help a search — while the scan
    // back over a run of y's inside stem_is_consonant is linear, so leaving it
    // unbounded keeps a superlinear term on the ingest AND load paths.
    static constexpr size_t MAX_STEM_LEN = 64;

    static std::string stem(const std::string& word) {
        if (word.size() <= 2 || word.size() > MAX_STEM_LEN) return word;
        std::string w = word;
        size_t sl;

        // ── step 1a: plurals ──
        if (stem_ends(w, "sses", sl) || stem_ends(w, "ies", sl)) {
            w = w.substr(0, sl) + (w[w.size()-3] == 'i' ? "i" : "ss");
        } else if (w.back() == 's' && w.size() > 1 && w[w.size()-2] != 's') {
            w.pop_back();
        }

        // ── step 1b: -eed / -ed / -ing ──
        bool step1b_hit = false;
        if (stem_ends(w, "eed", sl)) {
            if (stem_measure(w.substr(0, sl)) > 0) w.pop_back();
        } else if ((stem_ends(w, "ed", sl) || stem_ends(w, "ing", sl)) &&
                   stem_has_vowel(w.substr(0, sl))) {
            w = w.substr(0, sl);
            step1b_hit = true;
        }
        if (step1b_hit) {
            if (stem_ends(w, "at", sl) || stem_ends(w, "bl", sl) || stem_ends(w, "iz", sl)) {
                w += "e";
            } else if (stem_double_consonant(w) &&
                       w.back() != 'l' && w.back() != 's' && w.back() != 'z') {
                w.pop_back();
            } else if (stem_measure(w) == 1 && stem_cvc(w)) {
                w += "e";
            }
        }

        // ── step 1c: y → i ──
        if (w.size() > 1 && w.back() == 'y' && stem_has_vowel(w.substr(0, w.size()-1)))
            w.back() = 'i';

        // ── step 2 ──
        stem_replace(w, "ational", "ate", 0) || stem_replace(w, "tional", "tion", 0) ||
        stem_replace(w, "enci", "ence", 0)   || stem_replace(w, "anci", "ance", 0) ||
        stem_replace(w, "izer", "ize", 0)    ||
        // The paper says ABLI; Porter's reference C and Snowball both use BLI,
        // and that is what collapses -bly/-ble (possibly/possible,
        // terribly/terrible). ABLI alone leaves those split.
        stem_replace(w, "bli", "ble", 0)     ||
        stem_replace(w, "alli", "al", 0)     || stem_replace(w, "entli", "ent", 0) ||
        // Porter's own reference implementation carries this as a documented
        // addition to the 1980 paper. Without it the whole -ology family splits:
        // technology -> technologi but technological -> technolog.
        stem_replace(w, "logi", "log", 0)    ||
        stem_replace(w, "eli", "e", 0)       || stem_replace(w, "ousli", "ous", 0) ||
        stem_replace(w, "ization", "ize", 0) || stem_replace(w, "ation", "ate", 0) ||
        stem_replace(w, "ator", "ate", 0)    || stem_replace(w, "alism", "al", 0) ||
        stem_replace(w, "iveness", "ive", 0) || stem_replace(w, "fulness", "ful", 0) ||
        stem_replace(w, "ousness", "ous", 0) || stem_replace(w, "aliti", "al", 0) ||
        stem_replace(w, "iviti", "ive", 0)   || stem_replace(w, "biliti", "ble", 0);

        // ── step 3 ──
        stem_replace(w, "icate", "ic", 0) || stem_replace(w, "ative", "", 0) ||
        stem_replace(w, "alize", "al", 0) || stem_replace(w, "iciti", "ic", 0) ||
        stem_replace(w, "ical", "ic", 0)  || stem_replace(w, "ful", "", 0) ||
        stem_replace(w, "ness", "", 0);

        // ── step 4 ──
        bool s4 =
            stem_replace(w, "al", "", 1)    || stem_replace(w, "ance", "", 1) ||
            stem_replace(w, "ence", "", 1)  || stem_replace(w, "er", "", 1) ||
            stem_replace(w, "ic", "", 1)    || stem_replace(w, "able", "", 1) ||
            stem_replace(w, "ible", "", 1)  || stem_replace(w, "ant", "", 1) ||
            stem_replace(w, "ement", "", 1) || stem_replace(w, "ment", "", 1) ||
            stem_replace(w, "ent", "", 1);
        if (!s4) {
            if (stem_ends(w, "ion", sl)) {
                std::string st = w.substr(0, sl);
                if (stem_measure(st) > 1 && !st.empty() &&
                    (st.back() == 's' || st.back() == 't')) w = st;
            } else {
                stem_replace(w, "ou", "", 1)   || stem_replace(w, "ism", "", 1) ||
                stem_replace(w, "ate", "", 1)  || stem_replace(w, "iti", "", 1) ||
                stem_replace(w, "ous", "", 1)  || stem_replace(w, "ive", "", 1) ||
                stem_replace(w, "ize", "", 1);
            }
        }

        // ── step 5 ──
        if (w.size() > 1 && w.back() == 'e') {
            std::string st = w.substr(0, w.size() - 1);
            int m = stem_measure(st);
            if (m > 1 || (m == 1 && !stem_cvc(st))) w = st;
        }
        if (stem_measure(w) > 1 && stem_double_consonant(w) && w.back() == 'l')
            w.pop_back();

        return w.empty() ? word : w;
    }

    // word -> stem, for a pass that tokenizes a whole corpus. Stemming every
    // occurrence was 1.4 s of a 2.1 s open at 100k records (4.5M words); words
    // repeat (Zipf), so a pass stems each distinct word once. Only for
    // single-threaded passes: query-time tokenize() runs concurrently and
    // passes no cache.
    using StemCache = std::unordered_map<std::string, std::string>;

    static std::vector<std::string> tokenize(const std::string& text, StemCache* cache = nullptr) {
        std::vector<std::string> tokens;
        std::string tok;
        const auto& sw = stop_words();
        const bool stemming = stemming_enabled();
        auto emit = [&] {
            if (tok.size() < 2 || sw.find(tok) != sw.end()) return;
            if (!stemming)  { tokens.push_back(tok); return; }
            if (!cache)     { tokens.push_back(stem(tok)); return; }
            auto it = cache->find(tok);
            if (it == cache->end()) it = cache->emplace(tok, stem(tok)).first;
            tokens.push_back(it->second);
        };
        for (unsigned char c : text) {
            if (std::isalnum(c)) {
                tok += static_cast<char>(std::tolower(c));
            } else {
                emit();
                tok.clear();
            }
        }
        emit();
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
        const uint32_t doc = len_it->second;
        total_dl_ -= static_cast<double>(bm25_doc_len_[doc]);
        doc_lengths_.erase(len_it);

        auto drop_from = [doc](std::vector<PostingEntry>& postings) {
            postings.erase(
                std::remove_if(postings.begin(), postings.end(),
                    [doc](const PostingEntry& p) { return p.doc == doc; }),
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
                           const std::string* old_content = nullptr,
                           StemCache* cache = nullptr) {
        // Re-indexing an existing doc: retire its old postings first. Blanked
        // content (forget()) therefore also removes the doc, as it should.
        if (doc_lengths_.count(id)) remove_from_bm25_index(id, old_content);
        if (content.empty()) return;
        auto tokens = tokenize(content, cache);
        if (tokens.empty()) return;

        // Count term frequencies
        std::unordered_map<std::string, uint32_t> tf;
        for (const auto& t : tokens) tf[t]++;

        // Every re-index takes a fresh number; a writer that has used them
        // all (4G re-indexes since the last rebuild) renumbers from scratch.
        if (bm25_doc_id_.size() >= std::numeric_limits<uint32_t>::max()) {
            rebuild_bm25_index();
            if (doc_lengths_.count(id)) return;        // the rebuild indexed it
        }
        const uint32_t doc = static_cast<uint32_t>(bm25_doc_id_.size());
        bm25_doc_id_.push_back(id);
        bm25_doc_len_.push_back(static_cast<uint32_t>(tokens.size()));
        doc_lengths_[id] = doc;
        total_dl_ += static_cast<double>(tokens.size());
        for (const auto& [term, freq] : tf)
            bm25_index_[term].push_back({doc, freq});

        recompute_avg_dl();   // O(1) — running total, not a scan
    }

    void rebuild_bm25_index() {
        bm25_index_.clear();
        doc_lengths_.clear();
        bm25_doc_id_.clear();
        bm25_doc_len_.clear();
        total_dl_ = 0.0;
        avg_dl_   = 0.0;
        StemCache cache;
        for (const auto& [id, meta] : metadata_store_) {
            if (is_dead_meta(meta)) continue;   // forgotten/deleted stay unsearchable
            add_to_bm25_index(id, bm25_text(meta), nullptr, &cache);
        }
    }

    // ── Touch — call from within already-locked methods ─────────────
    // const, and safe under the SHARED lock: Metadata's salience counters are
    // mutable atomics, so concurrent queries can record their hits without
    // serialising on the writer lock. Writers are excluded meanwhile, so the
    // map itself can't rehash underneath us.
    // Caller holds mutex_ exclusively (or is the single-threaded load path).
    void mark_meta_dirty(uint64_t id) { dirty_meta_.insert(id); }

    void touch_nolock(uint64_t id) const {
        auto it = metadata_store_.find(id);
        if (it != metadata_store_.end() &&
            it->second.note_recall(static_cast<uint64_t>(std::time(nullptr)))) {
            std::lock_guard<std::mutex> g(salience_mx_);
            salience_dirty_.push_back(id);
        }
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
    // The same WAL generation opened for reading (pending refine originals
    // point into it), and its size: the offset of the next record.
    mutable std::shared_ptr<fsio::RandomReader> wal_reader_;
    mutable uint64_t wal_end_ = 0;
    // The generation the next fresh WAL file follows (its WALGEN record):
    // the loaded base's, then each commit's as it rotates the WAL.
    mutable uint64_t wal_gen_ = 0;

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

    // Caller holds wal_mutex_. Throws when the WAL cannot be opened: carrying
    // on without it would acknowledge writes no crash keeps. That alone does
    // not poison the handle — nothing reached the file, memory is untouched,
    // and the next write retries the open (EMFILE is often transient). A
    // header that failed half-way does poison: those bytes are in the file.
    void wal_open_for_append() const {
        if (wal_file_) return;
        bool fresh = true;
        {   // a zero-length or missing file needs the v2 header written first
            std::ifstream probe(wal_path_, std::ios::binary | std::ios::ate);
            if (probe && probe.tellg() > 0) fresh = false;
        }
        wal_file_ = std::fopen(wal_path_.c_str(), "ab");
        if (!wal_file_)
            throw std::runtime_error("cannot open the WAL " + wal_path_ + ": " +
                                     std::strerror(errno) + " (the write was not applied)");
        // Unbuffered: every append is one whole record handed to the kernel,
        // and nothing of a failed one stays behind to be flushed at fclose().
        std::setvbuf(wal_file_, nullptr, _IONBF, 0);
        wal_end_ = fresh ? 0 : fsio::size_or_zero(wal_path_);
        {
            auto rd = std::make_shared<fsio::RandomReader>();
            wal_reader_ = rd->reopen(wal_path_) ? std::move(rd) : nullptr;
        }
        if (fresh) {
            // Header and WALGEN record in ONE write: a reader that finds the
            // file sees either nothing (no records: as if absent) or both.
            const uint32_t hdr[2] = {WAL_MAGIC, WAL_VERSION};
            std::string buf(reinterpret_cast<const char*>(hdr), sizeof hdr);
            const std::string gen = encode_u64(wal_gen_);
            buf += encode_wal_record(WalOp::WALGEN, 0, gen);
            if (std::fwrite(buf.data(), 1, buf.size(), wal_file_) != buf.size())
                fail_writes("WAL header write failed for " + wal_path_ + ": " + std::strerror(errno));
            wal_end_ = buf.size();
            // A new file's directory entry is not durable until the directory
            // is synced: records fsynced into it could vanish with the name.
            if (wal_sync_enabled())
                sync_or_fail([&] { fsio::fsync_parent_dir(wal_path_); });
        }
    }

    // One whole record: [op][id][plen][payload][crc32].
    static std::string encode_wal_record(WalOp op, uint64_t id, const std::string& payload) {
        const auto op_b = static_cast<uint8_t>(op);
        const auto plen = static_cast<uint32_t>(payload.size());
        std::string rec(13 + payload.size() + 4, '\0');
        std::memcpy(&rec[0], &op_b, 1);
        std::memcpy(&rec[1], &id,   8);
        std::memcpy(&rec[9], &plen, 4);
        if (plen > 0) std::memcpy(&rec[13], payload.data(), plen);
        const uint32_t crc = crc32_of(rec.data(), 13 + plen, 0xFFFFFFFFu) ^ 0xFFFFFFFFu;
        std::memcpy(&rec[13 + plen], &crc, 4);
        return rec;
    }

    // One record of a WAL group, as wal_append_group() writes it.
    struct WalRec {
        WalOp op;
        uint64_t id;
        const std::string* payload;
    };

    static std::string encode_u64(uint64_t v) {
        return std::string(reinterpret_cast<const char*>(&v), 8);
    }

    // Append records and hand them to the OS in ONE write. Returns the LSN
    // (0 when there is no WAL). Called with mutex_ held exclusively, so WAL
    // order is exactly the order mutations are applied in memory, and BEFORE
    // the mutations are applied: a throw here leaves memory untouched.
    // Durability is established separately by wal_wait_durable(), normally
    // after mutex_ is released. `at`, if given, receives each record's
    // position. One write is what makes a group torn only at its tail, which
    // replay_wal() then drops whole.
    uint64_t wal_append_group(const std::vector<WalRec>& recs, std::vector<WalPos>* at = nullptr) {
        if (wal_path_.empty()) return 0;
        size_t total = 0;
        for (const auto& r : recs) {
            if (r.payload->size() > MAX_WAL_PAYLOAD)
                throw std::runtime_error("WAL record of " + std::to_string(r.payload->size()) +
                                         " bytes exceeds the limit (the write was not applied)");
            total += 13 + r.payload->size() + 4;
        }
        std::string buf(total, '\0');
        std::vector<uint64_t> offs;
        offs.reserve(recs.size());
        size_t p = 0;
        for (const auto& r : recs) {
            const auto op_b = static_cast<uint8_t>(r.op);
            const auto plen = static_cast<uint32_t>(r.payload->size());
            offs.push_back(p);
            std::memcpy(&buf[p],     &op_b, 1);
            std::memcpy(&buf[p + 1], &r.id, 8);
            std::memcpy(&buf[p + 9], &plen, 4);
            if (plen > 0) std::memcpy(&buf[p + 13], r.payload->data(), plen);
            const uint32_t crc = crc32_of(&buf[p], 13 + plen, 0xFFFFFFFFu) ^ 0xFFFFFFFFu;
            std::memcpy(&buf[p + 13 + plen], &crc, 4);
            p += 13 + plen + 4;
        }

        std::lock_guard<std::mutex> g(wal_mutex_);
        // Checked here, under the lock, not only in check_writable(): a record
        // appended after a torn one would never be replayed.
        if (poisoned_.load()) throw WriteFailed(refusal());
        wal_open_for_append();
        // One contiguous write: the handle is unbuffered (see
        // wal_open_for_append), so a failed write leaves no tail in a stdio
        // buffer for a later flush to complete behind the caller's back.
        if (std::fwrite(buf.data(), 1, buf.size(), wal_file_) != buf.size())
            fail_writes("WAL write failed for " + wal_path_ + ": " + std::strerror(errno));
        if (at) {
            at->clear();
            for (uint64_t o : offs) at->push_back({wal_reader_, wal_end_ + o});
        }
        wal_end_ += buf.size();
        return ++wal_written_lsn_;
    }

    // One record, followed by its STAMP when `version` is set.
    uint64_t wal_append(WalOp op, uint64_t id, const std::string& payload, WalPos* at = nullptr,
                        uint64_t version = 0) {
        const std::string stamp = encode_u64(version);
        std::vector<WalRec> recs{{op, id, &payload}};
        if (version) recs.push_back({WalOp::STAMP, id, &stamp});
        if (!at) return wal_append_group(recs);
        std::vector<WalPos> pos;
        const uint64_t lsn = wal_append_group(recs, &pos);
        *at = pos.empty() ? WalPos{} : pos[0];
        return lsn;
    }

    // Group commit. Block until every WAL record up to `lsn` is on stable
    // storage. The first waiter becomes the leader and fsyncs everything
    // appended so far — outside wal_mutex_, so other writers keep appending —
    // while the rest wait on the condition variable and are released by that
    // single fsync. N concurrent writers pay ~1 fsync instead of N serial ones.
    //
    // A failed fsync is not retried. It fails this writer, every writer it
    // covered, and every write after it (see poisoned_). A record already
    // durable before the failure still returns normally.
    void wal_wait_durable(uint64_t lsn) const {
        if (lsn == 0 || !wal_sync_enabled()) return;
        std::unique_lock<std::mutex> lk(wal_mutex_);
        while (wal_synced_lsn_ < lsn) {
            if (poisoned_.load()) throw WriteFailed(refusal());
            if (wal_syncing_) { wal_cv_.wait(lk); continue; }
            if (!wal_file_) {                      // checkpointed away meanwhile
                wal_synced_lsn_ = std::max(wal_synced_lsn_, wal_written_lsn_);
                break;
            }
            wal_syncing_ = true;
            const uint64_t target = wal_written_lsn_;
            std::FILE* fh = wal_file_;             // stays open: close/rotate wait
            lk.unlock();                           //   for !wal_syncing_
            std::string err;
            try { fsio::fsync_file(fh, wal_path_); }
            catch (const std::exception& e) { err = e.what(); }
            lk.lock();
            ++wal_fsyncs_;
            wal_syncing_ = false;
            if (err.empty()) wal_synced_lsn_ = std::max(wal_synced_lsn_, target);
            else             poison(err);          // the loop's next pass throws
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
    // The flock lives on ONE descriptor per process, owned by the registry
    // entry rather than by the handle that took it: when that handle closed
    // first, a later close of the reentrant handle had no descriptor to
    // release, and the lock stayed held until the process exited.
    struct HeldLock { int count = 0; int fd = -1; };
    static std::unordered_map<std::string, HeldLock>& registry() {
        static std::unordered_map<std::string, HeldLock> held;
        return held;
    }
    // One mutex per path ever opened for writing (see open()). Never freed:
    // a handful of bytes per distinct path, and no static-destruction order
    // to get wrong against a thread still opening at exit.
    static std::mutex& open_mutex(const std::string& key) {
        static auto* guard = new std::mutex;
        static auto* per_path = new std::unordered_map<std::string, std::unique_ptr<std::mutex>>;
        std::lock_guard<std::mutex> g(*guard);
        auto& m = (*per_path)[key];
        if (!m) m = std::make_unique<std::mutex>();
        return *m;
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

        // READERS TAKE NO LOCK, deliberately.
        //
        // A shared lock would buy nothing and cost the architecture that needs
        // this most: one writer service plus many agent processes reading the
        // same brand's file. LOCK_SH conflicts with the writer's LOCK_EX, so
        // every reader was refused while the writer simply held the handle
        // open — not even while saving.
        //
        // Nothing is given up. A v10 commit never rewrites a byte that a valid
        // header covers: it appends past the committed end and only then
        // switches the header, and a reader parses up to the end of the
        // header it chose, so it sees one whole commit — a torn header fails
        // its checksum and the other slot is used. A full rewrite (first save
        // of a v3-v9 file, vacuum) still goes to `<path>.tmp` and is renamed
        // over the original, so a reader holds the complete old inode or the
        // complete new one. Either way a reader keeps its consistent snapshot
        // until it reopens. The WAL is safe for the same kind of reason: every record
        // carries a CRC32 and replay stops at the first mismatch, so a reader
        // replaying a log the writer is appending to sees a clean prefix. A
        // read-only handle never calls save_vectors(), so it never clears the
        // writer's WAL either.
        //
        // What a reader gives up is freshness, not integrity: it sees the file
        // as of its open. Reopen (or go through the writer) to advance.
        if (!exclusive) return;

        lock_path_ = path_ + ".lock";
        lock_key_  = lock_key(path_);
        // One critical section from the lookup to the registry insert. The
        // Python binding used to hold the GIL across open(), which serialised
        // this by accident; without it, two threads could both miss the
        // entry, both flock() — and the loser reported "another process".
        // Everything inside is non-blocking (LOCK_NB).
        std::lock_guard<std::mutex> g(registry_mutex());
        {
            auto it = registry().find(lock_key_);
            if (it != registry().end() && it->second.count > 0) {
                it->second.count += 1;  // this process already holds it
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
                registry()[lock_key_].count += 1;
                reentrant_ = true;
                return;
            }
            throw std::runtime_error(
                "Cannot open '" + path_ + "' for writing: another process (pid "
                + holder + ") holds the write lock.\n"
                "Feather is single-writer, many-reader. Readers are never "
                "blocked — open with read_only=true, which takes no lock and "
                "gives a consistent snapshot as of the open. If you need a "
                "second WRITER, route its writes through the process that owns "
                "the file. Set FEATHER_LOCK=0 to disable this check (you then "
                "own the consequence: concurrent writers silently discard each "
                "other's records).");
        }
        lock_fd_ = fd;
        registry()[lock_key_] = HeldLock{1, fd};
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
            throw HandleClosed(std::string("Cannot ") + what + ": '" + path_ + "' is closed.");
        if (read_only_)
            throw std::runtime_error(
                std::string("Cannot ") + what + ": '" + path_ +
                "' was opened read-only. Reopen with read_only=false (and make "
                "sure no other process holds the write lock).");
        if (poisoned_.load())
            throw WriteFailed(std::string("Cannot ") + what + ": " + refusal());
    }

    // Record the first write failure. Idempotent; later causes are dropped.
    void poison(const std::string& why) const {
        std::lock_guard<std::mutex> g(poison_mx_);
        if (poisoned_.load()) return;
        poison_reason_ = why;
        poisoned_.store(true);
    }

    std::string refusal() const {
        std::lock_guard<std::mutex> g(poison_mx_);
        return "'" + path_ + "' stopped accepting writes after a failed write (" +
               poison_reason_ + "). Close it and reopen the file: the reopen "
               "replays the WAL that reached the disk.";
    }

    // A WAL write or an fsync failed: poison the handle and fail this call.
    [[noreturn]] void fail_writes(const std::string& why) const {
        poison(why);
        throw WriteFailed(refusal());
    }

    // Run a sync that a write path depends on. An fsync that failed poisons
    // the handle and is never retried; any other error (the file could not be
    // opened, say) propagates unchanged, as it always did, and may be retried.
    template <class Sync> void sync_or_fail(Sync&& sync) const {
        try { sync(); }
        catch (const fsio::SyncFailed& e) { fail_writes(e.what()); }
    }

public:
    bool is_read_only() const { return read_only_; }
    bool is_closed()    const { return closed_; }
    // Why this handle refuses writes; empty while it accepts them.
    std::string write_error() const {
        std::lock_guard<std::mutex> g(poison_mx_);
        return poisoned_.load() ? poison_reason_ : std::string();
    }

private:
    void release_lock() {
        if (!lock_key_.empty()) {
            int last_fd = -1;
            bool registered = false;
            {
                std::lock_guard<std::mutex> g(registry_mutex());
                auto it = registry().find(lock_key_);
                if (it != registry().end()) {
                    registered = true;
                    if (--it->second.count > 0) {
                        lock_key_.clear();
                        lock_fd_ = -1;  // the entry still owns the descriptor
                        return;         // other handles in this process still open
                    }
                    last_fd = it->second.fd;
                    registry().erase(it);
                }
                lock_key_.clear();
#ifndef _WIN32
                // Unlock before the entry's absence is visible: an open racing
                // this close would otherwise find no entry, hit our still-held
                // flock, and admit itself reentrantly against an entry nobody
                // holds the lock for.
                if (registered && last_fd >= 0) { ::flock(last_fd, LOCK_UN); ::close(last_fd); }
#endif
            }
#ifndef _WIN32
            if (registered) {           // last handle in this process: released
                lock_fd_ = -1;
                return;
            }
#else
            (void)registered; (void)last_fd;
#endif
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
        if (!wal_file_) return;
        // A writer may still be waiting outside mutex_ for its record to be
        // synced (close(save=False), or a failed checkpoint). Sync for it:
        // once the file is gone it would otherwise take its record as durable.
        if (wal_sync_enabled() && wal_synced_lsn_ < wal_written_lsn_ && !poisoned_.load()) {
            try {
                fsio::fsync_file(wal_file_, wal_path_);
                ++wal_fsyncs_;
                wal_synced_lsn_ = wal_written_lsn_;
            } catch (const std::exception& e) { poison(e.what()); }
        }
        std::fclose(wal_file_);
        wal_file_ = nullptr;
        wal_reader_.reset();
        wal_cv_.notify_all();
    }

    // Checkpoint completed: every record so far is in the (durable) base file.
    // Caller must hold mutex_ so no append can race the removal.
    void wal_clear() const {
        if (wal_path_.empty()) return;
        std::unique_lock<std::mutex> lk(wal_mutex_);
        wal_quiesce(lk);
        if (wal_file_) { std::fclose(wal_file_); wal_file_ = nullptr; }
        wal_reader_.reset();
        std::remove(wal_path_.c_str());
        std::remove(wal_old_path().c_str());
        wal_synced_lsn_ = wal_written_lsn_;
        wal_gen_ = generation_;                 // everything so far is in the base
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
    bool wal_rotate(uint64_t next_gen) const {
        if (wal_path_.empty()) return true;
        std::unique_lock<std::mutex> lk(wal_mutex_);
        wal_quiesce(lk);
        if (poisoned_.load()) throw WriteFailed(refusal());
        if (fsio::exists(wal_old_path())) return false;
        if (wal_file_) {
            if (std::fflush(wal_file_) != 0)
                fail_writes("WAL write failed for " + wal_path_ + ": " + std::strerror(errno));
            if (wal_sync_enabled()) {
                ++wal_fsyncs_;
                sync_or_fail([&] { fsio::fsync_file(wal_file_, wal_path_); });
            }
            std::fclose(wal_file_);
            wal_file_ = nullptr;
        }
        wal_reader_.reset();                    // pending entries keep theirs
        wal_synced_lsn_ = wal_written_lsn_;     // all records now durable
        wal_cv_.notify_all();
        if (fsio::exists(wal_path_))
            fsio::atomic_replace(wal_path_, wal_old_path());
        wal_gen_ = next_gen;                    // the next WAL follows this commit
        return true;
    }

    // Returns the offset just past the last record that checked out, or -1
    // when there is nothing replayable to vouch for (no file, newer format).
    std::streamoff replay_wal(const std::string& path, WalSeen* seen = nullptr) {
        if (path.empty()) return -1;
        // Pending refine originals point into this WAL rather than being
        // copied: pin the inode the stream reads (opened first, kept only if
        // the path still names it — the inode read_base_file uses).
        auto pin = std::make_shared<fsio::RandomReader>();
        pin->reopen(path);
        std::ifstream wf(path, std::ios::binary);
        if (!wf) return -1;
        std::shared_ptr<const fsio::RandomReader> wal_pin;
        if (pin->same_file(path)) wal_pin = std::move(pin);

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
                if (ver > WAL_VERSION) return -1;   // written by a newer build
                has_crc = true;
            } else {
                wf.clear();
                wf.seekg(0, std::ios::beg);      // v1: no header, rewind
            }
        }
        std::streamoff good_end = wf.tellg();
        struct GroupRec { uint8_t op; uint64_t id; std::string payload; std::streamoff start; };
        struct Group {
            std::streamoff start = 0;
            uint32_t count = 0;
            std::string key, fingerprint, note;
            int64_t created = 0;
            std::vector<GroupRec> recs;
        };
        std::optional<Group> group;
        std::streamoff pos = good_end;

        while (true) {
            const std::streamoff rec_start = pos;
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
            pos = wf.tellg();
            const std::streamoff rec_end = pos;
            const auto op = static_cast<WalOp>(op_b);

            // An atomic group: BEGIN, its records, COMMIT, written by one
            // write. Nothing of it is applied until its COMMIT has been read,
            // and good_end stays at the BEGIN until then, so a torn group is
            // dropped whole and cut off the log.
            if (op == WalOp::BEGIN) {
                if (group) break;                       // nested: corrupt from here
                group.emplace();
                group->start = rec_start;
                wal_decode(path, rec_start, payload, [&](v10::Reader& r) {
                    group->count       = r.get<uint32_t>();
                    group->key         = r.str16();
                    group->fingerprint = r.str16();
                    group->created     = r.get<int64_t>();
                    group->note        = get_str32(r);
                });
                continue;
            }
            if (group) {
                if (op != WalOp::COMMIT) {
                    if (group->recs.size() >= group->count) break;
                    group->recs.push_back({op_b, id, std::move(payload), rec_start});
                    continue;
                }
                std::vector<uint64_t> versions;
                std::vector<std::pair<uint64_t, uint64_t>> results;
                wal_decode(path, rec_start, payload, [&](v10::Reader& r) {
                    const uint32_t n = r.get<uint32_t>();
                    r.need(uint64_t{n} * 8);
                    for (uint32_t i = 0; i < n; ++i) versions.push_back(r.get<uint64_t>());
                    const uint32_t m = r.get<uint32_t>();
                    r.need(uint64_t{m} * 16);
                    for (uint32_t i = 0; i < m; ++i) {
                        const uint64_t rid = r.get<uint64_t>();
                        results.emplace_back(rid, r.get<uint64_t>());
                    }
                });
                if (versions.size() != group->recs.size() || group->recs.size() != group->count)
                    break;                              // incomplete: drop the group
                for (size_t i = 0; i < group->recs.size(); ++i) {
                    const auto& g = group->recs[i];
                    replay_record(path, g.op, g.id, g.payload, g.start, wal_pin, versions[i]);
                }
                if (!group->key.empty())
                    idem_put_nolock(group->key, {group->fingerprint, group->created, std::move(results),
                                                 std::move(group->note)});
                group.reset();
                good_end = rec_end;
                continue;
            }
            good_end = rec_end;
            if (seen) seen->present = true;
            if (op == WalOp::WALGEN) {
                uint64_t g = 0;
                wal_decode(path, rec_start, payload, [&](v10::Reader& r) { g = r.get<uint64_t>(); });
                if (seen && !seen->gen) seen->gen = g;
                continue;
            }
            if (op == WalOp::STAMP) {
                uint64_t v = 0;
                wal_decode(path, rec_start, payload, [&](v10::Reader& r) { v = r.get<uint64_t>(); });
                stamp_nolock(id, v);
                continue;
            }
            if (op == WalOp::COMMIT) continue;          // without a BEGIN: nothing to commit
            replay_record(path, op_b, id, payload, rec_start, wal_pin, 0);
        }
        // Derived indexes are built once by load_vectors() after this returns —
        // rebuilding them here too would do the expensive BM25 pass twice.
        return good_end;
    }

    // Decode a WAL payload; a CRC-valid record that does not decode was
    // crafted or written by a bug, and fails the open rather than replaying.
    template <class F>
    static void wal_decode(const std::string& path, std::streamoff at, const std::string& payload, F&& f) {
        try {
            v10::Reader r(payload.data(), payload.size());
            f(r);
        } catch (const std::exception&) {
            throw std::runtime_error("corrupt WAL '" + path + "': record at offset " +
                                     std::to_string(at) + " does not decode");
        }
    }

    // Apply one data record of the WAL, and stamp the record it changed with
    // `version` (0: the next clock value — a record logged before versions
    // existed, or one whose STAMP follows and overrides it).
    void replay_record(const std::string& path, uint8_t op_b, uint64_t id, const std::string& payload,
                       std::streamoff rec_start,
                       const std::shared_ptr<const fsio::RandomReader>& wal_pin, uint64_t version) {
        const uint32_t plen = static_cast<uint32_t>(payload.size());
        std::istringstream ss(payload);
        auto op = static_cast<WalOp>(op_b);
        // The CRC proves these are the bytes that were written, not that
        // they decode: a record that passes it and is still malformed was
        // crafted or written by a bug, and is refused rather than applied.
        auto rd = [&](void* p, size_t n) {
            if (!ss.read(static_cast<char*>(p), static_cast<std::streamsize>(n)))
                throw std::runtime_error("corrupt WAL '" + path + "': record at offset " +
                                         std::to_string(rec_start) + " is truncated");
        };

        if (op == WalOp::ADD) {
            uint16_t mod_len = 0;
            rd(&mod_len, 2);
            std::string modality(mod_len, '\0');
            if (mod_len > 0) rd(&modality[0], mod_len);
            uint32_t dim32 = 0;
            rd(&dim32, 4);
            if (dim32 == 0 || dim32 > (1u << 20) || uint64_t{dim32} * 4 > plen)
                throw std::runtime_error("corrupt WAL '" + path + "': implausible vector dim " +
                                         std::to_string(dim32));
            std::vector<float> vec(dim32);
            rd(vec.data(), dim32 * 4);
            Metadata meta = Metadata::deserialize(ss);
            auto& m_idx = get_or_create_index(modality, dim32);
            if (dim32 != m_idx.dim) return;   // corrupt / mismatched record
            reserve(m_idx, m_idx.index->getCurrentElementCount() + 1);
            try {
                add_point(m_idx, id, vec.data());
                keep_full(m_idx, id, vec.data(),
                          {wal_pin, wal_vector_offset(static_cast<uint64_t>(rec_start), modality)});
            } catch (...) {}
            replay_meta_upsert(id, std::move(meta));
            retire_if_dead_nolock(id);
            maybe_train_int8_nolock(modality);

        } else if (op == WalOp::UPDATE) {
            // update_metadata() REPLACES the record, edges included. Replay
            // used the add rule (keep edges when none are given), so an
            // edge the caller had dropped came back after a crash.
            mark_meta_dirty(id);
            metadata_store_[id] = Metadata::deserialize(ss);
            retire_if_dead_nolock(id);

        } else if (op == WalOp::UIMP) {
            float imp = 0.0f;
            rd(&imp, 4);
            auto it = metadata_store_.find(id);
            if (it != metadata_store_.end()) { it->second.importance = imp; mark_meta_dirty(id); }

        } else if (op == WalOp::LINK) {
            uint64_t to_id = 0;
            rd(&to_id, 8);
            uint8_t rel_len = 0;
            rd(&rel_len, 1);
            std::string rel_type(rel_len, '\0');
            if (rel_len > 0) rd(&rel_type[0], rel_len);
            float weight = 1.0f;
            rd(&weight, 4);
            auto it = metadata_store_.find(id);
            if (it != metadata_store_.end() && !is_dead_meta(it->second)) {
                bool exists = false;
                for (const auto& e : it->second.edges)
                    if (e.target_id == to_id && e.rel_type == rel_type) { exists = true; break; }
                if (!exists) {
                    it->second.edges.push_back({to_id, rel_type, weight});
                    reverse_index_[to_id].push_back({id, rel_type, weight});
                    mark_meta_dirty(id);
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
                mark_meta_dirty(id);
            }

        } else if (op == WalOp::PURGE) {
            uint16_t ns_len = 0;
            rd(&ns_len, 2);
            std::string ns(ns_len, '\0');
            if (ns_len > 0) rd(&ns[0], ns_len);
            uint64_t base = 0;                // since 0.22: the survivors' version base
            if (ss.peek() != std::char_traits<char>::eof()) rd(&base, 8);
            purge_core(ns, /*maintain_derived=*/false, base);
            return;                           // stamps its survivors itself
        } else {
            return;                           // a newer build's op: skipped, as before
        }
        stamp_nolock(id, version);
    }

    // ── Versions ─────────────────────────────────────────────────────
    // Caller holds mutex_ exclusively (or is the single-threaded load path).
    uint64_t next_version_nolock() { return ++version_clock_; }

    // Set a record's version: `v`, or the next clock value when v == 0.
    // The clock never moves backwards, so a replayed STAMP older than a
    // provisional one cannot make a later write reuse a version.
    uint64_t stamp_nolock(uint64_t id, uint64_t v = 0) {
        if (v == 0) v = next_version_nolock();
        else        version_clock_ = std::max(version_clock_, v);
        auto it = metadata_store_.find(id);
        if (it != metadata_store_.end()) it->second.version = v;
        return v;
    }

    // What compare-and-set compares: 0 for a record that does not exist or
    // was forgotten, which is what every reload sees for it too.
    uint64_t current_version_nolock(uint64_t id) const {
        auto it = metadata_store_.find(id);
        if (it == metadata_store_.end() || is_dead_meta(it->second)) return 0;
        return it->second.version;
    }

    // ── Idempotency keys ─────────────────────────────────────────────
    static int64_t idem_ttl_s() {
        static const int64_t v = [] {
            const char* e = std::getenv("FEATHER_IDEMPOTENCY_TTL_S");
            long long x = e ? std::atoll(e) : 0;
            return static_cast<int64_t>(x > 0 ? x : 24 * 3600);
        }();
        return v;
    }
    static size_t idem_max() {
        static const size_t v = [] {
            const char* e = std::getenv("FEATHER_IDEMPOTENCY_MAX");
            long long x = e ? std::atoll(e) : 0;
            return static_cast<size_t>(x > 0 ? x : 100000);
        }();
        return v;
    }
    bool idem_expired(int64_t created, int64_t now) const { return now - created >= idem_ttl_s(); }

    // Record a key's result, then drop expired keys and, past
    // FEATHER_IDEMPOTENCY_MAX, the oldest ones. A dropped key's retry is
    // applied again: the bound is what keeps a key per request from growing
    // without limit.
    void idem_put_nolock(const std::string& key, IdemEntry e) {
        const int64_t now = static_cast<int64_t>(std::time(nullptr));
        if (idem_expired(e.created, now)) return;
        idem_order_.emplace_back(e.created, key);
        idem_[key] = std::move(e);
        dirty_idem_.insert(key);
        while (!idem_order_.empty()) {
            const auto& [created, k] = idem_order_.front();
            auto it = idem_.find(k);
            const bool stale = it == idem_.end() || it->second.created != created;
            if (!stale && !idem_expired(created, now) && idem_.size() <= idem_max()) break;
            if (!stale) { idem_.erase(it); dirty_idem_.erase(k); }
            idem_order_.pop_front();
        }
    }

    std::optional<BatchResult> idem_result_nolock(const std::string& key, const std::string& fingerprint) const {
        const IdemEntry* e = idem_find_nolock(key);
        if (!e) return std::nullopt;
        if (e->fingerprint != fingerprint)
            throw IdempotencyKeyReused("idempotency key '" + key + "' was already used for a different request");
        BatchResult res;
        for (const auto& [id, v] : e->results) {
            res.ids.push_back(id);
            res.versions.push_back(v);
        }
        res.replayed = true;
        res.note = e->note;
        return res;
    }

    const IdemEntry* idem_find_nolock(const std::string& key) const {
        auto it = idem_.find(key);
        if (it == idem_.end()) return nullptr;
        if (idem_expired(it->second.created, static_cast<int64_t>(std::time(nullptr)))) return nullptr;
        return &it->second;
    }

    static void put_idem(std::string& buf, const std::string& key, const IdemEntry& e) {
        v10::put_str16(buf, key);
        v10::put_str16(buf, e.fingerprint);
        v10::put<int64_t>(buf, e.created);
        v10::put<uint32_t>(buf, static_cast<uint32_t>(e.results.size()));
        for (const auto& [id, v] : e.results) { v10::put<uint64_t>(buf, id); v10::put<uint64_t>(buf, v); }
        put_str32(buf, e.note);
    }

    // A file that carried no clock (v3-v9, or a FEATHER_FORMAT=9 rollback)
    // restarts its versions. Starting the clock at the time keeps them above
    // any version handed out before, as long as the process issued fewer
    // than ~1M writes per second on average (2^20 per second of wall time).
    void seed_version_clock() {
        if (clock_loaded_) return;
        version_clock_ = std::max(version_clock_, static_cast<uint64_t>(std::time(nullptr)) << 20);
    }

    // The upsert rule of apply_meta_upsert_nolock(): metadata without edges
    // keeps the record's existing edges. Replay used to overwrite them, so a
    // re-add after link() lost the link on recovery, and replaying <wal>.old
    // over a base that already held it was not idempotent. The reverse index
    // is rebuilt from metadata after replay, so only the edges matter here.
    void replay_meta_upsert(uint64_t id, Metadata meta) {
        mark_meta_dirty(id);
        auto it = metadata_store_.find(id);
        if (it == metadata_store_.end()) { metadata_store_.emplace(id, std::move(meta)); return; }
        if (meta.edges.empty() && !is_dead_meta(it->second)) meta.edges = std::move(it->second.edges);
        it->second = std::move(meta);
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
    // A failed sync poisons the handle like a WAL one: the WAL is kept, and
    // the reopen replays it over whichever base file reached the disk.
    void publish_snapshot(const std::string& tmp_path) const {
        if (checkpoint_sync_enabled()) sync_or_fail([&] { fsio::fsync_path(tmp_path); });
        fsio::atomic_replace(tmp_path, path_);
        if (checkpoint_sync_enabled()) sync_or_fail([&] { fsio::fsync_parent_dir(path_); });
    }

    // Fully synchronous checkpoint (close / destructor; caller holds mutex_
    // exclusively or nothing else can run): commit, then drop both WAL files.
    void save_vectors() {
        check_writable("save");
        if (legacy_format_requested()) {
            write_snapshot(path_ + ".tmp");
            take_salience();
            commit_clear_dirty(/*full=*/true);
            publish_legacy(path_ + ".tmp");
            wal_clear();
            return;
        }
        Commit c = commit_begin(/*force_full=*/false);
        commit_finish(c);
        commit_install(c);
        wal_clear();
    }

    // FEATHER_FORMAT=9 keeps writing the v9 whole-file snapshot, for a fleet
    // that still has to be read by a pre-v10 build. Read on every save so it
    // can be flipped without reopening.
    static bool legacy_format_requested() {
        const char* e = std::getenv("FEATHER_FORMAT");
        return e && std::strcmp(e, "9") == 0;
    }

    void publish_legacy(const std::string& tmp) {
        close_data_fh();               // the inode it points at is being replaced
        publish_snapshot(tmp);
        file_v10_      = false;
        base_version_  = 9;
        committed_end_ = 0;
    }

    void close_data_fh() {
        if (data_fh_) { std::fclose(data_fh_); data_fh_ = nullptr; }
    }

    static double env_double(const char* name, double dflt) {
        const char* e = std::getenv(name);
        if (!e || !*e) return dflt;
        char* end = nullptr;
        double v = std::strtod(e, &end);
        return end != e ? v : dflt;
    }

    // ── v10 commit ───────────────────────────────────────────────────
    struct Commit {
        bool     skip  = false;   // nothing changed: no bytes, no header
        bool     full  = false;   // whole file written to <path>.tmp
        uint64_t gen   = 0;
        uint64_t end   = 0;
        uint64_t bytes = 0;       // bytes this commit wrote
        std::vector<std::pair<std::string, size_t>> images;   // full: VECTORS image per modality
        // FULLVEC records written, published by commit_install(): offsets in
        // the file this commit leaves at path_, and the pending entry (seq)
        // each one releases (0: it came from the file, nothing to release).
        struct FullRec { uint32_t slot; uint64_t off; uint64_t seq; };
        struct FullSet { std::string name; uint64_t epoch; std::vector<FullRec> recs; };
        std::vector<FullSet> full_vecs;
    };

    // Writes sections at the stream's current position.
    struct SectionOut {
        std::FILE* fh      = nullptr;
        uint64_t   pos     = 0;   // file offset of the next byte
        uint64_t   written = 0;
        void write(uint32_t type, const std::string& payload, uint32_t flags = 0) {
            char hdr[v10::SECTION_HDR];
            const uint64_t len   = payload.size();
            std::memcpy(hdr, &type, 4);
            std::memcpy(hdr + 4, &flags, 4);
            std::memcpy(hdr + 8, &len, 8);
            const uint64_t sum = v10::section_checksum(hdr, payload.data(), len);
            if (std::fwrite(hdr, 1, sizeof(hdr), fh) != sizeof(hdr) ||
                (len && std::fwrite(payload.data(), 1, len, fh) != len) ||
                std::fwrite(&sum, 1, 8, fh) != 8)
                throw std::runtime_error("write failed (disk full?) while committing");
            const uint64_t n = v10::SECTION_HDR + len + v10::SECTION_TAIL;
            pos += n;
            written += n;
        }
        // Same section, payload given in pieces and hashed as it streams —
        // for a VECTORS image, which must not be copied into one buffer.
        void write_parts(uint32_t type, const std::vector<std::pair<const char*, size_t>>& parts) {
            uint64_t len = 0;
            for (const auto& pp : parts) len += pp.second;
            char hdr[v10::SECTION_HDR];
            const uint32_t flags = 0;
            std::memcpy(hdr, &type, 4);
            std::memcpy(hdr + 4, &flags, 4);
            std::memcpy(hdr + 8, &len, 8);
            v10::Xxh64Stream h(v10::xxh64(hdr, sizeof(hdr), 0));
            if (std::fwrite(hdr, 1, sizeof(hdr), fh) != sizeof(hdr))
                throw std::runtime_error("write failed (disk full?) while committing");
            for (const auto& pp : parts) {
                h.update(pp.first, pp.second);
                if (pp.second && std::fwrite(pp.first, 1, pp.second, fh) != pp.second)
                    throw std::runtime_error("write failed (disk full?) while committing");
            }
            const uint64_t sum = h.digest();
            if (std::fwrite(&sum, 1, 8, fh) != 8)
                throw std::runtime_error("write failed (disk full?) while committing");
            const uint64_t n = v10::SECTION_HDR + len + v10::SECTION_TAIL;
            pos += n;
            written += n;
        }
    };

    // One modality's vectors as a VECTORS section, padded so they start on a
    // v10::MAP_ALIGN boundary of the file.
    void write_vectors_section(SectionOut& out, const std::string& name, const ModalityIndex& m) const {
        const auto& ix = *m.index;
        const uint64_t n = ix.cur_element_count;
        std::string prefix;
        v10::put_str16(prefix, name);
        v10::put<uint32_t>(prefix, v10::count32(n));
        v10::put<uint32_t>(prefix, static_cast<uint32_t>(ix.data_size_));
        const uint64_t start = out.pos + v10::SECTION_HDR + prefix.size() + 4;
        const auto pad = static_cast<uint32_t>((v10::MAP_ALIGN - start % v10::MAP_ALIGN) % v10::MAP_ALIGN);
        v10::put<uint32_t>(prefix, pad);
        static const std::array<char, v10::MAP_ALIGN> zeros{};
        out.write_parts(v10::SEC_VECTORS, {
            {prefix.data(), prefix.size()},
            {zeros.data(), pad},
            {n ? ix.getDataByInternalId(0) : zeros.data(), static_cast<size_t>(n * ix.data_size_)},
        });
    }

    uint8_t vec_encoding(const std::string& name, const ModalityIndex& m) const {
        return (!m.int8 && quantized_modalities_.count(name)) ? v10::VEC_INT8 : v10::VEC_RAW;
    }

    std::string modality_payload(const std::string& name, const ModalityIndex& m) const {
        const auto& ix = *m.index;
        std::string p;
        v10::put_str16(p, name);
        v10::put<uint32_t>(p, static_cast<uint32_t>(m.dim));
        v10::put<uint8_t>(p, m.int8 ? 1 : 0);
        v10::put<float>(p, m.scale);
        v10::put<uint8_t>(p, quantized_modalities_.count(name) ? 1 : 0);
        v10::put<uint32_t>(p, static_cast<uint32_t>(ix.M_));
        v10::put<uint32_t>(p, static_cast<uint32_t>(ix.ef_construction_));
        v10::put<uint32_t>(p, static_cast<uint32_t>(ix.data_size_));
        v10::put<uint32_t>(p, static_cast<uint32_t>(ix.size_links_level0_));
        v10::put<uint32_t>(p, static_cast<uint32_t>(ix.size_links_per_element_));
        // Trailing fields (format_v10.h): skipped by builds that predate them.
        auto r = int8_refine_.find(name);
        v10::put<uint8_t>(p, static_cast<uint8_t>(r != int8_refine_.end() ? std::min<uint32_t>(r->second, 255) : 0));
        v10::put<uint8_t>(p, int8_train_.count(name) ? 1 : 0);
        return p;
    }

    // FULLVEC sections for one int8 modality with refine on: every slot when
    // `all`, else the slots whose data this commit's GRAPH records carry (a
    // GRAPH data record voids the slot's earlier FULLVEC on load, so each one
    // needs a new FULLVEC even when the float has not changed). The float
    // comes from fv_pending, else from the file it is in now. Offsets go to
    // `c.full_vecs` for commit_install(). Deleted slots are skipped.
    void write_fullvec(SectionOut& out, const std::string& name, const ModalityIndex& m,
                       bool all, Commit& c) const {
        using HNSW = hnswlib::HierarchicalNSW<float>;
        if (!m.int8 || !m.refine) return;
        const auto& ix = *m.index;
        const size_t n = ix.cur_element_count;
        const size_t vec_bytes = m.dim * sizeof(float);
        std::string prefix;
        v10::put_str16(prefix, name);
        v10::put<uint32_t>(prefix, static_cast<uint32_t>(m.dim));
        const size_t count_off = prefix.size();
        v10::put<uint32_t>(prefix, 0);

        Commit::FullSet set{name, m.epoch, {}};
        std::string buf = prefix;
        uint32_t records = 0;
        size_t first = 0;                  // set.recs index of this chunk's first record
        auto flush = [&] {
            if (!records) return;
            std::memcpy(&buf[count_off], &records, 4);
            const uint64_t base = out.pos + v10::SECTION_HDR;   // payload start in the file
            for (size_t k = first; k < set.recs.size(); ++k) set.recs[k].off += base;
            out.write(v10::SEC_FULLVEC, buf, v10::SEC_OPTIONAL);
            buf = prefix;
            records = 0;
            first = set.recs.size();
        };
        std::vector<float> tmp(m.dim);
        for (size_t i = 0; i < n; ++i) {
            const auto id = static_cast<hnswlib::tableint>(i);
            if (!all && !(ix.dirtyBits(id) & HNSW::DIRTY_DATA)) continue;
            if (ix.isMarkedDeleted(id)) continue;
            const auto slot = static_cast<uint32_t>(i);
            auto p = m.fv_pending.find(slot);
            const uint64_t seq = p != m.fv_pending.end() ? p->second.seq : 0;
            if (!full_vector(m, slot, tmp.data())) continue;   // no original to keep
            const float* src = tmp.data();
            v10::put<uint32_t>(buf, slot);
            set.recs.push_back({slot, buf.size(), seq});   // made absolute by flush()
            buf.append(reinterpret_cast<const char*>(src), vec_bytes);
            ++records;
            if (buf.size() >= v10::CHUNK) flush();
        }
        flush();
        c.full_vecs.push_back(std::move(set));
    }

    // Publish what a commit wrote to FULLVEC: offsets into the file now at
    // path_, and the release of every pending float it carried that has not
    // been replaced since. After a full rewrite every offset is new and so is
    // the file. Caller holds mutex_ exclusively (readers use both), after
    // commit_finish() succeeded.
    void commit_install(Commit& c) {
        if (c.skip) return;
        if (c.full) {
            for (auto& [name, m] : modality_indices_) m.fv_off.clear();
            // The inode the rename put at path_. A failed open only costs the
            // re-rank (int8 distances); the floats stay in the file.
            fv_file_.reopen(path_);
        }
        for (auto& set : c.full_vecs) {
            auto it = modality_indices_.find(set.name);
            if (it == modality_indices_.end() || it->second.epoch != set.epoch) continue;
            auto& m = it->second;
            if (!fv_file_.is_open()) fv_file_.reopen(path_);
            for (const auto& r : set.recs) {
                if (r.slot >= m.fv_off.size()) m.fv_off.resize(m.index->cur_element_count, 0);
                if (r.slot < m.fv_off.size()) m.fv_off[r.slot] = r.off;
                auto p = m.fv_pending.find(r.slot);
                if (r.seq && p != m.fv_pending.end() && p->second.seq == r.seq) m.fv_pending.erase(p);
            }
        }
        c.full_vecs.clear();
    }

    // GRAPH sections for one modality: every slot when `all`, else the slots
    // hnswlib marked dirty. A slot whose vector, label or level changed gets
    // a full record; one whose links merely changed — an insert rewires ~M
    // neighbours — gets its link lists only (~140 bytes instead of ~1.7 KB at
    // 384-d). Each chunk repeats the header so it applies on its own.
    // `image`: the vectors were just written as a VECTORS section, so the
    // records carry none (VEC_IMAGE).
    void write_graph(SectionOut& out, const std::string& name, const ModalityIndex& m, bool all,
                     bool image = false) const {
        using HNSW = hnswlib::HierarchicalNSW<float>;
        const auto& ix = *m.index;
        const size_t n = ix.cur_element_count;
        const uint8_t enc = image ? v10::VEC_IMAGE : vec_encoding(name, m);
        std::string prefix;
        v10::put_str16(prefix, name);
        v10::put<uint8_t>(prefix, enc);
        v10::put<uint32_t>(prefix, v10::count32(n));
        v10::put<uint32_t>(prefix, static_cast<uint32_t>(ix.enterpoint_node_));
        v10::put<int32_t>(prefix, static_cast<int32_t>(ix.maxlevel_));
        const size_t count_off = prefix.size();
        v10::put<uint32_t>(prefix, 0);

        std::string buf = prefix;
        uint32_t records = 0;
        auto flush = [&] {
            if (!records) return;
            std::memcpy(&buf[count_off], &records, 4);
            out.write(v10::SEC_GRAPH, buf);
            buf = prefix;
            records = 0;
        };
        std::vector<int8_t> qbuf(enc == v10::VEC_INT8 ? m.dim : 0);
        for (size_t i = 0; i < n; ++i) {
            const auto id = static_cast<hnswlib::tableint>(i);
            const uint8_t bits = all ? (HNSW::DIRTY_DATA | HNSW::DIRTY_LINKS) : ix.dirtyBits(id);
            if (!bits) continue;
            const int level = ix.element_levels_[i];
            if (level < 0 || level > 255)
                throw std::runtime_error("graph node level out of range for format v10");
            const bool data = (bits & HNSW::DIRTY_DATA) != 0;
            v10::put<uint32_t>(buf, static_cast<uint32_t>(i));
            v10::put<uint8_t>(buf, v10::REC_LINKS | (data ? v10::REC_DATA : 0));
            v10::put<uint8_t>(buf, static_cast<uint8_t>(level));
            if (data) {
                v10::put<uint64_t>(buf, static_cast<uint64_t>(ix.getExternalLabel(id)));
                if (enc == v10::VEC_IMAGE) {
                    // in the VECTORS section
                } else if (enc == v10::VEC_INT8) {
                    std::vector<float> v = read_vector_internal(m, i);
                    float sc = quantize_vec(v.data(), m.dim, qbuf.data());
                    v10::put<float>(buf, sc);
                    buf.append(reinterpret_cast<const char*>(qbuf.data()), m.dim);
                } else {
                    buf.append(ix.getDataByInternalId(id), ix.data_size_);
                }
            }
            buf.append(reinterpret_cast<const char*>(ix.get_linklist0(id)), ix.size_links_level0_);
            if (level > 0)
                buf.append(ix.linkLists_[i], ix.size_links_per_element_ * static_cast<size_t>(level));
            ++records;
            if (buf.size() >= v10::CHUNK) flush();
        }
        flush();
    }

    // One META record: a put carries the serialized Metadata, a delete only
    // the id. Dead records (forgotten / _deleted) are written as deletes, as
    // v9 dropped them from every snapshot.
    void put_meta(std::string& buf, uint64_t id, const Metadata* m) {
        v10::put<uint8_t>(buf, m ? v10::META_PUT : v10::META_DEL);
        v10::put<uint64_t>(buf, id);
        const size_t len_off = buf.size();
        v10::put<uint32_t>(buf, 0);
        if (!m) return;
        const size_t start = buf.size();
        {
            v10::StringSink sink(buf);
            std::ostream os(&sink);
            m->serialize(os);
        }
        const auto len = static_cast<uint32_t>(buf.size() - start);
        std::memcpy(&buf[len_off], &len, 4);
        meta_rec_bytes_ += len;
        meta_rec_count_ += 1;
    }

    // Every byte a full rewrite would produce, approximately. The file is
    // vacuumed once it exceeds this by FEATHER_VACUUM_RATIO (default 2: the
    // garbage left by superseded records outweighs the live data).
    uint64_t live_bytes_estimate() const {
        uint64_t live = v10::DATA_START;
        for (const auto& [name, m] : modality_indices_) {
            const auto& ix = *m.index;
            const size_t n = ix.cur_element_count;
            const size_t vec = vec_encoding(name, m) == v10::VEC_INT8 ? 4 + m.dim : ix.data_size_;
            live += static_cast<uint64_t>(n) * (6 + 8 + vec + ix.size_links_level0_);
            uint64_t upper = 0;
            for (size_t i = 0; i < n; ++i) upper += static_cast<uint64_t>(ix.element_levels_[i]);
            live += upper * ix.size_links_per_element_;
            // A rewrite pads the VECTORS image to a MAP_ALIGN boundary; left
            // out, a small file's own fresh rewrite looked like garbage and
            // every save rewrote it again.
            if (n > 0 && vec_encoding(name, m) == v10::VEC_RAW) live += v10::MAP_ALIGN;
            // The float originals an int8 modality keeps for refine.
            if (m.int8 && m.refine)
                live += static_cast<uint64_t>(n - std::min<size_t>(n, ix.num_deleted_.load())) * (4 + m.dim * sizeof(float));
        }
        const uint64_t avg = meta_rec_count_ ? meta_rec_bytes_ / meta_rec_count_ : 128;
        live += static_cast<uint64_t>(metadata_store_.size()) * (13 + avg);
        return live;
    }

    bool vacuum_due() const {
        const double ratio = env_double("FEATHER_VACUUM_RATIO", 2.0);
        if (ratio <= 0.0) return false;
        const double slack = env_double("FEATHER_VACUUM_MIN_BYTES", 4.0 * 1024 * 1024);
        if (static_cast<double>(committed_end_) >
            ratio * static_cast<double>(live_bytes_estimate()) + slack)
            return true;
        // Vectors added since the last full rewrite are outside the VECTORS
        // image, so every read-only process holds its own copy of them. Once
        // they outgrow the image (FEATHER_IMAGE_RATIO, default 1), rewrite: a
        // doubling rule, so the rewrites cost O(1) per vector added.
        const double image_ratio = env_double("FEATHER_IMAGE_RATIO", 1.0);
        if (image_ratio <= 0.0) return false;
        double in_image = 0, outside = 0;
        for (const auto& [name, m] : modality_indices_) {
            if (vec_encoding(name, m) != v10::VEC_RAW) continue;
            const size_t n = m.index->cur_element_count;
            const size_t img = std::min(n, m.image_count);
            in_image += static_cast<double>(img) * m.index->data_size_;
            outside  += static_cast<double>(n - img) * m.index->data_size_;
        }
        return outside > image_ratio * in_image + slack;
    }

    static bool mmap_enabled() {
        const char* e = std::getenv("FEATHER_MMAP");
        return !(e && (e[0] == '0' || e[0] == 'n' || e[0] == 'N'));
    }

    // Hand the queued salience records to this commit, clearing each one's
    // flag BEFORE its counters are read (see Metadata::note_recall).
    std::vector<uint64_t> take_salience() {
        std::vector<uint64_t> ids;
        {
            std::lock_guard<std::mutex> g(salience_mx_);
            ids.swap(salience_dirty_);
        }
        for (uint64_t id : ids) {
            auto it = metadata_store_.find(id);
            if (it != metadata_store_.end()) it->second.clear_recall_unsaved();
        }
        return ids;
    }

    bool has_changes(const std::vector<uint64_t>& salience) const {
        if (!dirty_meta_.empty() || !salience.empty() || !dirty_idem_.empty()) return true;
        for (const auto& [name, m] : modality_indices_) {
            if (!m.persisted) return true;
            const auto& ix = *m.index;
            for (size_t i = 0, n = ix.cur_element_count; i < n; ++i)
                if (ix.dirtyBits(static_cast<hnswlib::tableint>(i))) return true;
        }
        return false;
    }

    void commit_clear_dirty(bool full) {
        for (auto& [name, m] : modality_indices_) {
            m.index->clearDirty();
            m.persisted = true;
        }
        dirty_meta_.clear();
        dirty_idem_.clear();
        if (full) vacuum_requested_ = false;
    }

    // VERSIONS sections: the clock, then the version of every live record put
    // by this commit's META sections (all of them, or the dirty ones). Always
    // after META: a META put resets the version it is loaded into.
    void write_versions(SectionOut& out, bool all) const {
        std::string buf;
        auto flush = [&] {
            out.write(v10::SEC_VERSIONS, buf, v10::SEC_OPTIONAL);
            buf.clear();
            v10::put<uint64_t>(buf, version_clock_);
        };
        v10::put<uint64_t>(buf, version_clock_);
        auto put = [&](uint64_t id, const Metadata& m) {
            v10::put<uint64_t>(buf, id);
            v10::put<uint64_t>(buf, m.version);
            if (buf.size() >= v10::CHUNK) flush();
        };
        if (all) {
            for (const auto& [id, m] : metadata_store_)
                if (!is_dead_meta(m)) put(id, m);
        } else {
            for (uint64_t id : dirty_meta_) {
                auto it = metadata_store_.find(id);
                if (it != metadata_store_.end() && !is_dead_meta(it->second)) put(id, it->second);
            }
        }
        out.write(v10::SEC_VERSIONS, buf, v10::SEC_OPTIONAL);
    }

    void write_idempotency(SectionOut& out, bool all) const {
        std::string buf;
        const int64_t now = static_cast<int64_t>(std::time(nullptr));
        auto put = [&](const std::string& key, const IdemEntry& e) {
            if (idem_expired(e.created, now)) return;
            put_idem(buf, key, e);
            if (buf.size() >= v10::CHUNK) { out.write(v10::SEC_IDEMPOTENCY, buf, v10::SEC_OPTIONAL); buf.clear(); }
        };
        if (all) {
            for (const auto& [key, e] : idem_) put(key, e);
        } else {
            for (const auto& key : dirty_idem_) {
                auto it = idem_.find(key);
                if (it != idem_.end()) put(key, it->second);
            }
        }
        if (!buf.empty()) out.write(v10::SEC_IDEMPOTENCY, buf, v10::SEC_OPTIONAL);
    }

    // The whole DB as a fresh v10 file at `tmp`: an image of every modality,
    // every live record, and the same header in both slots.
    void write_full_v10(const std::string& tmp, Commit& c) {
        std::FILE* fh = std::fopen(tmp.c_str(), "wb");
        if (!fh) throw std::runtime_error("Cannot save to temp file: " + tmp);
        std::vector<char> iobuf(1u << 20);
        std::setvbuf(fh, iobuf.data(), _IOFBF, iobuf.size());
        try {
            static const std::array<char, v10::DATA_START> zeros{};
            if (std::fwrite(zeros.data(), 1, zeros.size(), fh) != zeros.size())
                throw std::runtime_error("write failed (disk full?) while saving " + tmp);
            SectionOut out{fh, v10::DATA_START, v10::DATA_START};
            for (const auto& [name, m] : modality_indices_) {
                out.write(v10::SEC_MODALITY, modality_payload(name, m));
                const bool image = vec_encoding(name, m) == v10::VEC_RAW && m.index->cur_element_count > 0;
                if (image) write_vectors_section(out, name, m);
                write_graph(out, name, m, /*all=*/true, image);
                write_fullvec(out, name, m, /*all=*/true, c);
                c.images.emplace_back(name, image ? static_cast<size_t>(m.index->cur_element_count) : size_t(0));
            }
            std::string buf;
            for (const auto& [id, meta] : metadata_store_) {
                if (is_dead_meta(meta)) continue;
                put_meta(buf, id, &meta);
                if (buf.size() >= v10::CHUNK) { out.write(v10::SEC_META, buf); buf.clear(); }
            }
            if (!buf.empty()) out.write(v10::SEC_META, buf);
            write_versions(out, /*all=*/true);
            write_idempotency(out, /*all=*/true);

            c.gen   = generation_ + 1;
            c.end   = out.pos;
            c.bytes = out.pos;
            v10::Header h;
            h.generation = c.gen;
            h.end        = c.end;
            h.time       = static_cast<uint64_t>(std::time(nullptr));
            for (uint32_t slot = 0; slot < 2; ++slot) {
                h.slot = slot;
                auto b = v10::encode_header(h);
                if (fsio::seek64(fh, slot * v10::SLOT_SIZE) != 0 ||
                    std::fwrite(b.data(), 1, b.size(), fh) != b.size())
                    throw std::runtime_error("write failed (disk full?) while saving " + tmp);
            }
            if (std::fflush(fh) != 0 || std::ferror(fh))
                throw std::runtime_error("write failed (disk full?) while saving " + tmp);
        } catch (...) {
            std::fclose(fh);
            std::remove(tmp.c_str());
            throw;
        }
        if (std::fclose(fh) != 0) {
            std::remove(tmp.c_str());
            throw std::runtime_error("close failed while saving " + tmp);
        }
    }

    // Append what changed to path_ past the committed end. No fsync and no
    // header yet: until commit_finish() switches the header, these bytes are
    // invisible to every reader and to recovery.
    void append_delta(Commit& c, const std::vector<uint64_t>& salience) {
        if (!data_fh_) {
            data_fh_ = std::fopen(path_.c_str(), "r+b");
            if (!data_fh_) throw std::runtime_error("Cannot open " + path_ + " to commit");
        }
        // A commit that failed part-way left bytes past the end. Unreferenced,
        // but cut them so the file is exactly what the header describes.
        if (fsio::size_or_zero(path_) > committed_end_)
            fsio::truncate_file(data_fh_, committed_end_);
        if (fsio::seek64(data_fh_, committed_end_) != 0)
            throw std::runtime_error("seek failed while committing " + path_);

        SectionOut out{data_fh_, committed_end_, 0};
        for (const auto& [name, m] : modality_indices_) {
            if (!m.persisted) out.write(v10::SEC_MODALITY, modality_payload(name, m));
            write_graph(out, name, m, /*all=*/!m.persisted);
            write_fullvec(out, name, m, /*all=*/!m.persisted, c);
        }
        std::string buf;
        for (uint64_t id : dirty_meta_) {
            auto it = metadata_store_.find(id);
            const bool live = it != metadata_store_.end() && !is_dead_meta(it->second);
            put_meta(buf, id, live ? &it->second : nullptr);
            if (buf.size() >= v10::CHUNK) { out.write(v10::SEC_META, buf); buf.clear(); }
        }
        if (!buf.empty()) { out.write(v10::SEC_META, buf); buf.clear(); }
        if (!dirty_meta_.empty()) write_versions(out, /*all=*/false);
        write_idempotency(out, /*all=*/false);
        // A recall changes two counters; writing the whole record (content and
        // all) for every search hit would make read-heavy saves heavy.
        for (uint64_t id : salience) {
            if (dirty_meta_.count(id)) continue;          // its META put has the counters
            auto it = metadata_store_.find(id);
            if (it == metadata_store_.end() || is_dead_meta(it->second)) continue;
            v10::put<uint64_t>(buf, id);
            v10::put<uint32_t>(buf, it->second.recalls());
            v10::put<uint64_t>(buf, it->second.recalled_at());
            if (buf.size() >= v10::CHUNK) { out.write(v10::SEC_SALIENCE, buf); buf.clear(); }
        }
        if (!buf.empty()) out.write(v10::SEC_SALIENCE, buf);

        if (std::fflush(data_fh_) != 0 || std::ferror(data_fh_))
            throw std::runtime_error("write failed (disk full?) while committing " + path_);
        c.skip  = out.written == 0;
        c.gen   = generation_ + 1;
        c.end   = out.pos;
        c.bytes = out.written;
    }

    // First half of a commit. Caller holds save_mutex_ and mutex_ (shared is
    // enough: no writer runs, and readers touch none of the dirty state). Cost
    // is the size of the change — this is all the time writers wait. On a
    // throw nothing has been consumed and the next commit retries.
    Commit commit_begin(bool force_full) {
        // A DB whose load threw holds a fragment of the file, not its
        // contents. Writing that back is data loss. See ~DB().
        if (!load_complete_)
            throw std::logic_error(
                "refusing to save a DB whose load did not complete — "
                "this would overwrite " + path_ + " with a partial read");
        std::vector<uint64_t> salience = take_salience();
        Commit c;
        try {
            c.full = force_full || commit_failed_ || !file_v10_ || vacuum_requested_ || vacuum_due();
            if (c.full)                          write_full_v10(path_ + ".tmp", c);
            else if (!has_changes(salience))     c.skip = true;
            else                                 append_delta(c, salience);
            for (const auto& [name, count] : c.images) modality_indices_.at(name).image_count = count;
        } catch (...) {
            // Their flags are already cleared; a full META put still carries
            // the counters.
            for (uint64_t id : salience) mark_meta_dirty(id);
            throw;
        }
        commit_clear_dirty(c.full);
        return c;
    }

    // Second half: make the commit durable and visible. Needs save_mutex_
    // only, so it runs after the data lock is released. A throw here leaves
    // the dirty sets already consumed, so the next commit is a full rewrite.
    void commit_finish(Commit& c) {
        if (c.skip) return;
        try {
            if (c.full) {
                close_data_fh();             // the inode it points at is being replaced
                publish_snapshot(path_ + ".tmp");
                ++vacuums_;
            } else {
                // Sections first, header second: a header must never reach
                // the disk ahead of the bytes it vouches for.
                if (checkpoint_sync_enabled())
                    sync_or_fail([&] { fsio::fsync_file(data_fh_, path_); });
                v10::Header h;
                h.generation = c.gen;
                h.end        = c.end;
                h.slot       = static_cast<uint32_t>(c.gen % 2);
                h.time       = static_cast<uint64_t>(std::time(nullptr));
                auto b = v10::encode_header(h);
                if (fsio::seek64(data_fh_, h.slot * v10::SLOT_SIZE) != 0 ||
                    std::fwrite(b.data(), 1, b.size(), data_fh_) != b.size() ||
                    std::fflush(data_fh_) != 0)
                    throw std::runtime_error("header write failed while committing " + path_);
                if (checkpoint_sync_enabled())
                    sync_or_fail([&] { fsio::fsync_file(data_fh_, path_); });
            }
        } catch (...) {
            commit_failed_ = true;
            throw;
        }
        file_v10_          = true;
        base_version_      = v10::VERSION;
        generation_        = c.gen;
        committed_end_     = c.end;
        commit_failed_     = false;
        last_commit_bytes_ = c.bytes;
        last_commit_full_  = c.full;
        ++commits_;
    }

    // ── v10 load ─────────────────────────────────────────────────────
    // A section that fails its checksum, or runs past the header's end, is a
    // commit that never fully reached the disk: the load falls back to the
    // previous header. Anything else — an unknown section type, a structural
    // error in bytes whose checksum is fine — is NOT a reason to fall back:
    // silently loading an older generation would discard data that is there,
    // and the writer would then truncate it.
    struct TornSection : std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    struct RestoreState {
        uint32_t          ep       = static_cast<uint32_t>(-1);
        int32_t           maxlevel = -1;
        bool              graph    = false;
        size_t            image    = 0;    // slots whose vectors a VECTORS section holds
        std::vector<bool> seen;    // slot has had a full record
    };

    void reset_loaded_state() {
        metadata_store_.clear();
        modality_indices_.clear();
        quantized_modalities_.clear();
        int8_ram_scale_.clear();
        int8_train_.clear();
        int8_refine_.clear();
        ns_graphs_.clear();
        meta_rec_bytes_ = meta_rec_count_ = 0;
        version_clock_ = 0;
        clock_loaded_  = false;
        idem_.clear();
        idem_order_.clear();
    }

    // Returns false when the file is not v10, for the v2-v9 reader.
    bool read_v10(std::ifstream& f, uint64_t file_size) {
        char s0[v10::HEADER_LEN] = {}, s1[v10::HEADER_LEN] = {};
        f.read(s0, sizeof(s0));
        const size_t n0 = static_cast<size_t>(f.gcount());
        f.clear();
        size_t n1 = 0;
        if (file_size >= v10::SLOT_SIZE + v10::HEADER_LEN) {
            f.seekg(static_cast<std::streamoff>(v10::SLOT_SIZE));
            f.read(s1, sizeof(s1));
            n1 = static_cast<size_t>(f.gcount());
            f.clear();
        }
        uint32_t magic = 0, version = 0;
        if (n0 >= 8) { std::memcpy(&magic, s0, 4); std::memcpy(&version, s0 + 4, 4); }
        // v2-v9 open with "FEAT" and their version; slot 1 of such a file is
        // ordinary data. Decided on the raw bytes, before any checksum.
        if (magic == v10::MAGIC && version >= 2 && version < v10::VERSION) return false;

        v10::Header h[2];
        const v10::HeaderState st[2] = {
            v10::decode_header(s0, n0, file_size, h[0]),
            v10::decode_header(s1, n1, file_size, h[1]),
        };
        if (st[0] == v10::HeaderState::NEWER || st[1] == v10::HeaderState::NEWER)
            throw std::runtime_error("'" + path_ + "' was written by a newer Feather "
                                     "(format > 10); upgrade to open it");
        std::vector<v10::Header> cands;
        for (int i = 0; i < 2; ++i)
            if (st[i] == v10::HeaderState::VALID) cands.push_back(h[i]);
        if (cands.empty()) {
            if (st[0] == v10::HeaderState::NOT_FEATHER && st[1] == v10::HeaderState::NOT_FEATHER)
                return false;                        // not a Feather file at all
            throw std::runtime_error(
                "corrupt .feather: neither header is valid (both fail their checksum "
                "or claim more bytes than the file holds)");
        }
        std::sort(cands.begin(), cands.end(),
                  [](const v10::Header& a, const v10::Header& b) { return a.generation > b.generation; });
        if (cands.size() == 2 && cands[0].generation == cands[1].generation &&
            cands[0].end == cands[1].end)
            cands.pop_back();                        // a full write puts one header in both

        v10::Header chosen;
        for (size_t k = 0; k < cands.size(); ++k) {
            try {
                reset_loaded_state();
                parse_v10(f, cands[k].end, file_size);
                chosen = cands[k];
                break;
            } catch (const TornSection& e) {
                if (k + 1 == cands.size())
                    throw std::runtime_error(std::string("corrupt .feather: ") + e.what());
                // Falling back is only lossless when the WAL can redo the
                // abandoned commit. A commit deletes <wal>.old after its
                // header is durable, so a commit torn by a crash always has
                // one beside it. Without it these bytes went bad AFTER the
                // commit succeeded: falling back would silently drop
                // committed data, and a writer would then cut it off.
                const char* force = std::getenv("FEATHER_RECOVER_OLDER");
                if (!fsio::exists(wal_old_path()) && !(force && force[0] == '1'))
                    throw std::runtime_error(
                        "corrupt .feather: commit " + std::to_string(cands[k].generation) +
                        " of '" + path_ + "' is damaged (" + e.what() + ") and no WAL can "
                        "rebuild it. The file has not been modified. Set "
                        "FEATHER_RECOVER_OLDER=1 to open the previous commit (" +
                        std::to_string(cands[k + 1].generation) + ") instead; a writer "
                        "will then discard the damaged one.");
            }
        }
        generation_    = chosen.generation;
        committed_end_ = chosen.end;
        file_v10_      = true;
        base_version_  = v10::VERSION;
        if (read_only_) return true;
        // Bytes past the committed end are a commit that never switched the
        // header. Cut them (a reader never looks past the end; it is the next
        // append that must not land after them).
        if (file_size > committed_end_)
            fsio::truncate_to(path_, committed_end_);
        // And retire the header of a commit we fell back from. Left in place
        // it would come back to life once later appends reached its end, and
        // vouch for a commit it never saw.
        const uint32_t other = static_cast<uint32_t>((chosen.generation + 1) % 2);
        if (st[other] != v10::HeaderState::VALID || h[other].generation > chosen.generation)
            write_header_slot(other, chosen);
        return true;
    }

    // Overwrite one header slot in place, durably. Load-time repair only.
    void write_header_slot(uint32_t slot, v10::Header h) {
        h.slot = slot;
        auto b = v10::encode_header(h);
        std::FILE* fh = std::fopen(path_.c_str(), "r+b");
        if (!fh) throw std::runtime_error("Cannot open " + path_ + " to repair its header");
        const bool ok = fsio::seek64(fh, slot * v10::SLOT_SIZE) == 0 &&
                        std::fwrite(b.data(), 1, b.size(), fh) == b.size() &&
                        std::fflush(fh) == 0;
        // Best effort, as before: failing open() over it would make a file
        // that opened yesterday unopenable. If this sync did not stick, the
        // next open finds the stale slot again and repeats the repair.
        if (ok && checkpoint_sync_enabled()) {
            try { fsio::fsync_file(fh, path_); } catch (const fsio::SyncFailed&) {}
        }
        std::fclose(fh);
        if (!ok) throw std::runtime_error("Cannot repair the header of " + path_);
    }

    void parse_v10(std::ifstream& f, uint64_t end, uint64_t file_size) {
        std::unordered_map<std::string, RestoreState> rs;
        std::vector<char> payload;
        uint64_t pos = v10::DATA_START;
        f.clear();
        f.seekg(static_cast<std::streamoff>(pos));
        while (pos < end) {
            if (end - pos < v10::SECTION_HDR + v10::SECTION_TAIL)
                throw TornSection("section header runs past the committed end");
            char hdr[v10::SECTION_HDR];
            if (!f.read(hdr, sizeof(hdr))) throw TornSection("short read in section header");
            uint32_t type, flags;
            uint64_t len;
            std::memcpy(&type, hdr, 4);
            std::memcpy(&flags, hdr + 4, 4);
            std::memcpy(&len, hdr + 8, 8);
            if (len > end - pos - v10::SECTION_HDR - v10::SECTION_TAIL)
                throw TornSection("section runs past the committed end");
            if (type == v10::SEC_VECTORS || type == v10::SEC_FULLVEC) {
                if (type == v10::SEC_VECTORS) load_vectors_section(f, pos, len, hdr, rs);
                else                          load_fullvec_section(f, pos, len, hdr);
                pos += v10::SECTION_HDR + len + v10::SECTION_TAIL;
                continue;
            }
            payload.resize(static_cast<size_t>(len));
            uint64_t sum = 0;
            if ((len && !f.read(payload.data(), static_cast<std::streamsize>(len))) ||
                !f.read(reinterpret_cast<char*>(&sum), 8))
                throw TornSection("short read in section");
            if (v10::section_checksum(hdr, payload.data(), len) != sum)
                throw TornSection("section checksum mismatch at offset " + std::to_string(pos));

            v10::Reader r(payload.data(), payload.size());
            switch (type) {
                case v10::SEC_MODALITY: load_modality_section(r, rs);            break;
                case v10::SEC_GRAPH:    load_graph_section(r, rs, file_size);    break;
                case v10::SEC_META:     load_meta_section(r);                    break;
                case v10::SEC_SALIENCE: load_salience_section(r);                break;
                case v10::SEC_VERSIONS: load_versions_section(r);                break;
                case v10::SEC_IDEMPOTENCY: load_idempotency_section(r);          break;
                default:
                    if (!(flags & v10::SEC_OPTIONAL))
                        throw std::runtime_error("'" + path_ + "' holds section type " +
                            std::to_string(type) + ", written by a newer Feather; upgrade to open it");
            }
            pos += v10::SECTION_HDR + len + v10::SECTION_TAIL;
        }
        for (auto& [name, st] : rs) {
            auto& m = modality_indices_.at(name);
            if (st.graph) {
                for (size_t i = 0; i < st.seen.size(); ++i)
                    if (!st.seen[i])
                        throw std::runtime_error("corrupt .feather: graph slot " + std::to_string(i) +
                                                 " of '" + name + "' was never written");
                m.index->restoreFinish(st.ep, st.maxlevel);
            }
            m.index->setEf(DEFAULT_EF);
            m.persisted = true;
        }
    }

    // A VECTORS section. Read-only handles map the vector bytes (shared with
    // every other process that maps them); writers, and any platform or file
    // where mapping fails, read them straight into place. The checksum is
    // computed over the bytes either way.
    void load_vectors_section(std::ifstream& f, uint64_t pos, uint64_t len, const char* hdr,
                              std::unordered_map<std::string, RestoreState>& rs) {
        v10::Xxh64Stream h(v10::xxh64(hdr, v10::SECTION_HDR, 0));
        uint64_t left = len;
        auto take = [&](void* p, size_t n) {
            if (left < n || !f.read(static_cast<char*>(p), static_cast<std::streamsize>(n)))
                throw TornSection("short VECTORS section");
            h.update(p, n);
            left -= n;
        };
        uint16_t name_len = 0;
        take(&name_len, 2);
        std::string name(name_len, '\0');
        if (name_len) take(&name[0], name_len);
        uint32_t count = 0;
        uint32_t stride = 0, pad = 0;
        take(&count, 4);
        take(&stride, 4);
        take(&pad, 4);
        if (pad > left) throw TornSection("VECTORS padding runs past the section");
        std::vector<char> scratch(std::min<uint64_t>(pad, v10::MAP_ALIGN));
        for (uint64_t done = 0; done < pad; ) {
            const size_t k = static_cast<size_t>(std::min<uint64_t>(pad - done, scratch.size()));
            take(scratch.data(), k);
            done += k;
        }
        const uint64_t data_off = pos + v10::SECTION_HDR + (len - left);
        const uint64_t data_len = left;

        auto mit = modality_indices_.find(name);
        const bool valid = mit != modality_indices_.end() && rs.count(name) &&
                           mit->second.index->cur_element_count == 0 &&
                           stride == mit->second.index->data_size_ &&
                           static_cast<uint64_t>(count) * stride == data_len;
        auto finish = [&] {
            uint64_t sum = 0;
            if (!f.read(reinterpret_cast<char*>(&sum), 8)) throw TornSection("short VECTORS section");
            if (h.digest() != sum)
                throw TornSection("section checksum mismatch at offset " + std::to_string(pos));
        };
        if (!valid) {
            // Hash through it first: torn bytes are a fallback, not corruption.
            std::vector<char> buf(1u << 20);
            for (uint64_t done = 0; done < data_len; ) {
                const size_t k = static_cast<size_t>(std::min<uint64_t>(data_len - done, buf.size()));
                take(buf.data(), k);
                done += k;
            }
            finish();
            throw std::runtime_error("corrupt .feather: VECTORS section for '" + name +
                                     "' does not match its modality");
        }
        auto& m  = mit->second;
        auto& ix = *m.index;
        bool mapped = false;
        if (read_only_ && mmap_enabled() && data_off % v10::MAP_ALIGN == 0 && data_len > 0) {
            // Room to grow in place: WAL replay and later commits add slots.
            const size_t reserve = static_cast<size_t>(
                std::max<uint64_t>(uint64_t{count} * 4, uint64_t{count} + 65536) * stride);
            auto mp = fsio::map_private(path_, data_off, static_cast<size_t>(data_len), reserve);
            if (mp.base) {
                ix.attachVectorMapping(mp.base, mp.bytes);
                h.update(mp.base, static_cast<size_t>(data_len));   // shared page cache, not a copy
                f.seekg(static_cast<std::streamoff>(data_off + data_len));
                mapped = true;
            }
        }
        ix.restoreBegin(static_cast<size_t>(count));
        if (!mapped) {
            char* dst = ix.getDataByInternalId(0);
            for (uint64_t done = 0; done < data_len; ) {
                const size_t k = static_cast<size_t>(std::min<uint64_t>(data_len - done, 8u << 20));
                take(dst + done, k);
                done += k;
            }
        }
        finish();
        m.mapped      = mapped;
        m.image_count = static_cast<size_t>(count);
        rs[name].image = static_cast<size_t>(count);
    }

    void load_modality_section(v10::Reader& r, std::unordered_map<std::string, RestoreState>& rs) {
        const std::string name = r.str16();
        const uint32_t dim     = r.get<uint32_t>();
        const uint8_t  int8    = r.get<uint8_t>();
        const float    scale   = r.get<float>();
        const uint8_t  quant   = r.get<uint8_t>();
        const uint32_t M       = r.get<uint32_t>();
        const uint32_t efc     = r.get<uint32_t>();
        const uint32_t dsize   = r.get<uint32_t>();
        const uint32_t links0  = r.get<uint32_t>();
        const uint32_t linksN  = r.get<uint32_t>();
        // Same guard as v9: an absurd dim would allocate gigabytes.
        if (dim == 0 || dim > (1u << 20))
            throw std::runtime_error("corrupt .feather: implausible vector dim " + std::to_string(dim));
        if (int8 && !(std::isfinite(scale) && scale > 0.0f))
            throw std::runtime_error("corrupt .feather: bad int8 scale for '" + name + "'");
        if (M < 2 || M > 10000)
            throw std::runtime_error("corrupt .feather: implausible HNSW M " + std::to_string(M));
        // Trailing fields; absent from files written before them.
        const uint8_t refine = r.left() >= 1 ? r.get<uint8_t>() : 0;
        const uint8_t train  = r.left() >= 1 ? r.get<uint8_t>() : 0;
        if (int8) int8_ram_scale_[name] = scale; else int8_ram_scale_.erase(name);
        if (refine) int8_refine_[name] = refine; else int8_refine_.erase(name);
        if (train && !int8) int8_train_.insert(name); else int8_train_.erase(name);
        if (quant) quantized_modalities_.insert(name); else quantized_modalities_.erase(name);
        modality_indices_.erase(name);
        auto& m = create_index(name, dim, M, efc);
        const auto& ix = *m.index;
        if (ix.data_size_ != dsize || ix.size_links_level0_ != links0 ||
            ix.size_links_per_element_ != linksN)
            throw std::runtime_error("'" + path_ + "': graph layout of '" + name +
                                     "' does not match this build");
        rs[name] = RestoreState{};
    }

    void load_graph_section(v10::Reader& r, std::unordered_map<std::string, RestoreState>& rs,
                            uint64_t file_size) {
        const std::string name = r.str16();
        const uint8_t  enc      = r.get<uint8_t>();
        const uint32_t n        = r.get<uint32_t>();
        const uint32_t ep       = r.get<uint32_t>();
        const int32_t  maxlevel = r.get<int32_t>();
        const uint32_t count    = r.get<uint32_t>();
        auto mit = modality_indices_.find(name);
        if (mit == modality_indices_.end() || !rs.count(name))
            throw std::runtime_error("corrupt .feather: graph section for undeclared modality '" + name + "'");
        auto& m  = mit->second;
        auto& ix = *m.index;
        if (enc > v10::VEC_IMAGE || (enc == v10::VEC_INT8 && m.int8))
            throw std::runtime_error("corrupt .feather: unknown vector encoding " + std::to_string(enc));
        const size_t image = rs[name].image;
        if (enc == v10::VEC_IMAGE && image == 0)
            throw std::runtime_error("corrupt .feather: graph records refer to a missing VECTORS section");
        // Both counts are loop trip counts / allocation sizes: bound them by
        // the bytes that would have to exist to back them.
        const size_t vec_bytes = enc == v10::VEC_INT8 ? 4 + m.dim
                               : enc == v10::VEC_IMAGE ? 0 : ix.data_size_;
        const size_t min_full  = 6 + 8 + vec_bytes + ix.size_links_level0_;
        if (n > file_size / min_full)
            throw std::runtime_error("corrupt .feather: graph slot count " + std::to_string(n) +
                                     " exceeds file size");
        if (count > r.left() / (6 + ix.size_links_level0_))
            throw std::runtime_error("corrupt .feather: graph record count exceeds its section");
        ix.restoreBegin(static_cast<size_t>(n));
        auto& st = rs[name];
        st.seen.resize(static_cast<size_t>(n), false);
        st.graph    = true;
        st.ep       = ep;
        st.maxlevel = maxlevel;

        std::vector<float> fbuf(enc == v10::VEC_INT8 ? m.dim : 0);
        for (uint32_t k = 0; k < count; ++k) {
            const uint32_t id    = r.get<uint32_t>();
            const uint8_t  kind  = r.get<uint8_t>();
            const int      level = r.get<uint8_t>();
            if (id >= n || !(kind & v10::REC_LINKS) || (kind & ~(v10::REC_LINKS | v10::REC_DATA)))
                throw std::runtime_error("corrupt .feather: bad graph record");
            hnswlib::labeltype label = 0;
            const char* data = nullptr;
            if (kind & v10::REC_DATA) {
                label = static_cast<hnswlib::labeltype>(r.get<uint64_t>());
                if (enc == v10::VEC_IMAGE) {
                    if (id >= image)
                        throw std::runtime_error("corrupt .feather: graph slot outside its VECTORS image");
                } else if (enc == v10::VEC_INT8) {
                    const float sc = r.get<float>();
                    dequantize_vec(reinterpret_cast<const int8_t*>(r.take(m.dim)), m.dim, sc, fbuf.data());
                    data = reinterpret_cast<const char*>(fbuf.data());
                } else {
                    data = r.take(ix.data_size_);
                }
            } else if (!st.seen[id] || level != ix.element_levels_[id]) {
                throw std::runtime_error("corrupt .feather: link record for a graph slot never written");
            }
            const char* l0 = r.take(ix.size_links_level0_);
            const char* up = level > 0 ? r.take(ix.size_links_per_element_ * static_cast<size_t>(level)) : nullptr;
            const bool full = (kind & v10::REC_DATA) != 0;
            ix.restoreElement(id, level, l0, up, full ? &label : nullptr, data);
            if (full) st.seen[id] = true;
            // New data voids the slot's float original; a FULLVEC after this
            // record in the same commit restores it. One written by a build
            // without FULLVEC never does, and its stale float is never used.
            if (full && id < m.fv_off.size()) m.fv_off[id] = 0;
        }
    }

    // FULLVEC: where each slot's float original is. Only offsets are kept;
    // the floats are hashed through a small buffer and never held, so a
    // section costs no 8 MiB payload buffer (measured: it left 8.7 MB of
    // freed-but-resident heap per open at 100k x 384). As with VECTORS, the
    // whole section is hashed before anything is judged, so torn bytes fall
    // back and only a section whose checksum holds can be called corrupt.
    // Only an int8 modality whose graph holds the slot can own one.
    void load_fullvec_section(std::ifstream& f, uint64_t pos, uint64_t len, const char* hdr) {
        v10::Xxh64Stream h(v10::xxh64(hdr, v10::SECTION_HDR, 0));
        uint64_t left = len;
        auto take = [&](void* p, size_t n) {
            if (left < n || !f.read(static_cast<char*>(p), static_cast<std::streamsize>(n)))
                throw TornSection("short FULLVEC section");
            h.update(p, n);
            left -= n;
        };
        std::vector<char> scratch(64u << 10);
        auto skip = [&](uint64_t n) {
            while (n) {
                const size_t k = static_cast<size_t>(std::min<uint64_t>(n, scratch.size()));
                take(scratch.data(), k);
                n -= k;
            }
        };
        std::string why;                      // first structural fault; judged after the checksum
        uint16_t name_len = 0;
        take(&name_len, 2);
        std::string name(name_len, '\0');
        if (name_len) take(&name[0], name_len);
        uint32_t dim = 0, count = 0;
        take(&dim, 4);
        take(&count, 4);
        auto mit = modality_indices_.find(name);
        ModalityIndex* m = mit != modality_indices_.end() ? &mit->second : nullptr;
        const uint64_t rec = 4 + uint64_t{dim} * sizeof(float);
        if (!m || !m->int8 || dim != m->dim)
            why = "full-precision vectors for '" + name + "' do not match its modality";
        else if (uint64_t{count} * rec != left)
            why = "full-precision record count does not match its section";
        std::vector<std::pair<uint32_t, uint64_t>> found;
        if (why.empty()) {
            found.reserve(count);
            const size_t n = m->index->cur_element_count;
            for (uint32_t k = 0; k < count; ++k) {
                uint32_t slot = 0;
                take(&slot, 4);
                found.emplace_back(slot, pos + v10::SECTION_HDR + (len - left));
                skip(rec - 4);
                if (slot >= n && why.empty()) why = "full-precision vector for a missing graph slot";
            }
        }
        skip(left);
        uint64_t sum = 0;
        if (!f.read(reinterpret_cast<char*>(&sum), 8)) throw TornSection("short FULLVEC section");
        if (h.digest() != sum)
            throw TornSection("section checksum mismatch at offset " + std::to_string(pos));
        if (!why.empty()) throw std::runtime_error("corrupt .feather: " + why);
        if (m->fv_off.size() < m->index->cur_element_count)
            m->fv_off.resize(m->index->cur_element_count, 0);
        for (const auto& [slot, off] : found) m->fv_off[slot] = off;
    }

    void load_meta_section(v10::Reader& r) {
        while (r.left()) {
            const uint8_t  op  = r.get<uint8_t>();
            const uint64_t id  = r.get<uint64_t>();
            const uint32_t len = r.get<uint32_t>();
            const char*    b   = r.take(len);
            if (op == v10::META_PUT) {
                v10::MemSource src(b, len);
                std::istream is(&src);
                metadata_store_[id] = Metadata::deserialize(is);
                meta_rec_bytes_ += len;
                meta_rec_count_ += 1;
            } else if (op == v10::META_DEL) {
                metadata_store_.erase(id);
            } else {
                throw std::runtime_error("corrupt .feather: bad metadata record");
            }
        }
    }

    void load_versions_section(v10::Reader& r) {
        version_clock_ = std::max(version_clock_, r.get<uint64_t>());
        clock_loaded_ = true;
        while (r.left()) {
            const uint64_t id = r.get<uint64_t>();
            const uint64_t v  = r.get<uint64_t>();
            if (v > version_clock_)
                throw std::runtime_error("corrupt .feather: record version above the version clock");
            auto it = metadata_store_.find(id);
            if (it != metadata_store_.end()) it->second.version = v;
        }
    }

    // Loaded entries are already in the file: not dirty, unlike a replayed one.
    void load_idempotency_section(v10::Reader& r) {
        while (r.left()) {
            std::string key = r.str16();
            IdemEntry e;
            e.fingerprint = r.str16();
            e.created     = r.get<int64_t>();
            const uint32_t n = r.get<uint32_t>();
            r.need(uint64_t{n} * 16);
            e.results.reserve(n);
            for (uint32_t i = 0; i < n; ++i) {
                const uint64_t id = r.get<uint64_t>();
                e.results.emplace_back(id, r.get<uint64_t>());
            }
            e.note = get_str32(r);
            idem_order_.emplace_back(e.created, key);
            idem_[std::move(key)] = std::move(e);
        }
    }

    void load_salience_section(v10::Reader& r) {
        while (r.left()) {
            const uint64_t id = r.get<uint64_t>();
            const uint32_t rc = r.get<uint32_t>();
            const uint64_t lr = r.get<uint64_t>();
            auto it = metadata_store_.find(id);
            if (it == metadata_store_.end()) continue;
            it->second.recall_count.store(rc, std::memory_order_relaxed);
            it->second.last_recalled_at.store(lr, std::memory_order_relaxed);
        }
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
            // restores it instead of rebuilding from vectors, and keeps the
            // higher-quality serial-build graph. Deleted nodes are kept, with
            // their delete marks — the loader has always restored those. This
            // used to require live_count == total, so ONE forget() sent every
            // later open down the rebuild path: measured 0.15 s -> 53 s at
            // 100k x 384. Safe since dead records always carry a delete mark
            // (retire_if_dead_nolock): no live node outlives its metadata.
            // On-disk-quantized modalities still store vectors only (int8
            // re-encoding is incompatible with the graph blob).
            uint8_t persist_graph = quant ? 0 : 1;
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
    // Base file, then <wal>.old, then <wal>. A writer reads them alone. A
    // read-only open races a live writer: a commit can finish between reading
    // the base and reading the WALs, deleting the <wal>.old whose operations
    // the base it read does not hold yet — and the view comes out torn (the
    // base, then later operations, missing the ones between). Measured before
    // this check: 187 of 7395 opens under a writer saving every 7 adds. So a
    // reader checks the WALGEN generations chain onto the base it read, and
    // starts over when they do not.
    std::streamoff read_snapshot() {
        std::optional<std::tuple<uint64_t, std::optional<uint64_t>, std::optional<uint64_t>>> last;
        for (int attempt = 0;; ++attempt) {
            read_base_file();
            if (base_version_ != 0) seed_version_clock();
            // Crash recovery — runs even with no base file. <wal>.old exists
            // only if a checkpoint was interrupted between rotating the WAL and
            // deleting the old one; its records are older than everything in
            // <wal>.
            WalSeen old_seen, cur_seen;
            replay_wal(wal_old_path(), &old_seen);
            const std::streamoff good_end = replay_wal(wal_path_, &cur_seen);
            // The same broken chain twice running is not a race — a race
            // moves the base on (its commit switched the header before
            // deleting <wal>.old). It is the files as they are (a crash, a
            // WAL removed by hand), and a reader loads what a writer would.
            const auto seen = std::make_tuple(generation_,
                old_seen.present ? old_seen.gen : std::nullopt,
                cur_seen.present ? cur_seen.gen : std::nullopt);
            if (!read_only_ || chain_holds(generation_, old_seen, cur_seen) || seen == last) {
                wal_gen_ = cur_seen.gen ? *cur_seen.gen : generation_;
                return good_end;
            }
            last = seen;
            if (attempt >= 200)
                throw std::runtime_error("could not read a consistent snapshot of '" + path_ +
                                         "': its writer kept committing during every attempt");
            reset_loaded_state();
            generation_ = 0; base_version_ = 0; file_v10_ = false; committed_end_ = 0;
            dirty_meta_.clear();
            dirty_idem_.clear();
            std::this_thread::sleep_for(std::chrono::microseconds(std::min(50000, 100 << std::min(attempt, 9))));
        }
    }

    // Do the WALs hold exactly the operations after base generation `b`?
    //   <wal> at b                  — yes (a <wal>.old is already in the base)
    //   <wal> at b+1, <wal>.old at b — a commit not finished yet: yes
    //   no WAL with records          — the base alone: an older snapshot, whole
    // Anything else skipped operations: a commit finished between the reads.
    // A WAL with records but no WALGEN predates the record: not judged.
    static bool chain_holds(uint64_t b, const WalSeen& old_wal, const WalSeen& cur) {
        if ((old_wal.present && !old_wal.gen) || (cur.present && !cur.gen)) return true;
        if (cur.present) {
            if (*cur.gen == b) return true;
            return *cur.gen == b + 1 && old_wal.present && *old_wal.gen == b;
        }
        return !old_wal.present || *old_wal.gen <= b;
    }

    void load_vectors() {
        // A full rewrite that crashed before its rename leaves <path>.tmp.
        // Nothing reads it; only a writer may remove it.
        if (!read_only_) std::remove((path_ + ".tmp").c_str());
        const std::streamoff good_end = read_snapshot();
        // Cut a torn or corrupt tail. Replay stops there, and the next append
        // would land BEHIND it: acknowledged writes the next recovery can never
        // reach. A read-only handle shares the writer's WAL and must not touch it.
        if (!read_only_ && good_end >= 0 &&
            static_cast<uint64_t>(good_end) < fsio::size_or_zero(wal_path_)) {
            fsio::truncate_to(wal_path_, static_cast<uint64_t>(good_end));
        }
        // A modality saved while waiting to train may hold enough by now.
        for (const auto& name : std::vector<std::string>(int8_train_.begin(), int8_train_.end()))
            maybe_train_int8_nolock(name);
        // Records no file or WAL gave a version (written before versions
        // existed) take fresh ones: above every version handed out before.
        for (auto& [id, m] : metadata_store_)
            if (m.version == 0) m.version = next_version_nolock();
        std::sort(idem_order_.begin(), idem_order_.end());
        retire_orphan_nodes();
        build_reverse_index();
        build_secondary_indexes();
        rebuild_bm25_index();
        load_complete_ = true; // last statement: a throw anywhere above leaves
                               // this false and the destructor won't checkpoint
    }

    void read_base_file() {
        // FULLVEC offsets must point into the inode that is parsed, and a
        // writer's full rewrite may rename a new file over path_ at any time.
        // So the reader is opened first, and kept only if path_ still names
        // its inode once the stream is open: a rewrite always renames a NEW
        // file in, so the path cannot have left that inode and come back.
        std::ifstream f;
        bool same = false;
        for (int attempt = 0; attempt < 3 && !same; ++attempt) {
            fv_file_.reopen(path_);
            f = std::ifstream(path_, std::ios::binary);
            if (!f) break;
            same = fv_file_.same_file(path_);
        }
        if (!f) { fv_file_.close(); return; }
        if (!same) fv_file_.close();                        // re-rank falls back to int8
        if (read_v10(f, fsio::size_or_zero(path_))) return;
        fv_file_.close();                                   // v3-v9 hold no originals
        f.clear();
        f.seekg(0, std::ios::beg);

        // Every read is checked: a short read used to leave the field holding
        // whatever was on the stack (dim32, element_count), which then sized
        // an index. Counts are bounded by the bytes left before they loop.
        auto rd = [&](void* p, size_t n) {
            if (!f.read(static_cast<char*>(p), static_cast<std::streamsize>(n)))
                throw std::runtime_error("corrupt .feather: truncated");
        };
        const std::streamoff file_end = static_cast<std::streamoff>(fsio::size_or_zero(path_));
        auto left = [&]() -> uint64_t {
            const std::streamoff at = f.tellg();
            return at < 0 || at > file_end ? 0 : static_cast<uint64_t>(file_end - at);
        };
        auto check_dim = [](uint32_t dim32) {
            // An absurd dim would make index creation (and per-vector
            // buffers) allocate gigabytes. No real embedding approaches 2^20.
            if (dim32 == 0 || dim32 > (1u << 20))
                throw std::runtime_error("corrupt .feather: implausible vector dim "
                                         + std::to_string(dim32));
        };

        uint32_t magic = 0, version = 0;
        if (!f.read((char*)&magic, 4) || !f.read((char*)&version, 4)) return;
        if (magic != 0x46454154) return;
        base_version_ = version;   // the first save rewrites it as v10

        if (version == 2) {
            // v2: single "text" index, metadata interleaved with vectors
            uint32_t dim32 = 0;
            rd(&dim32, 4);
            check_dim(dim32);
            auto& m_idx = get_or_create_index("text", dim32);
            uint64_t id;
            std::vector<float> vec(dim32);
            while (f.read((char*)&id, 8)) {
                Metadata meta = Metadata::deserialize(f);
                rd(vec.data(), dim32 * sizeof(float));
                reserve(m_idx, m_idx.index->getCurrentElementCount() + 1);
                add_point(m_idx, id, vec.data());
                metadata_store_[id] = std::move(meta);
            }
        } else if (version >= 3) {
            // v3/v4/v5: separate metadata section then modality indices
            uint32_t meta_count = 0;
            rd(&meta_count, 4);
            // A corrupt count is a loop trip count, so an absurd value means
            // meta_count iterations of deserialize() against a stream that ran
            // out — each one allocating from garbage lengths. Observed: a forged
            // count of 0xFFFFFF00 left the process spinning for >10 minutes
            // instead of failing. The smallest possible record is id(8) plus a
            // minimal Metadata, so anything claiming more records than there are
            // bytes left is corrupt by arithmetic, not by heuristic.
            if (meta_count > left() / 8)
                throw std::runtime_error(
                    "corrupt .feather: metadata count " + std::to_string(meta_count) +
                    " exceeds remaining file size");
            for (uint32_t i = 0; i < meta_count; ++i) {
                uint64_t id;
                if (!f.read((char*)&id, 8))
                    throw std::runtime_error("corrupt .feather: truncated metadata section");
                metadata_store_[id] = Metadata::deserialize(f);
            }
            uint32_t modal_count = 0;
            rd(&modal_count, 4);
            // name_len(2) + dim(4) is the least a modality can take.
            if (modal_count > left() / 6)
                throw std::runtime_error("corrupt .feather: modality count " +
                                         std::to_string(modal_count) + " exceeds file size");
            for (uint32_t m = 0; m < modal_count; ++m) {
                uint16_t name_len = 0;
                rd(&name_len, 2);
                std::string name(name_len, ' ');
                if (name_len) rd(&name[0], name_len);
                uint32_t dim32 = 0, element_count = 0;
                rd(&dim32, 4);
                check_dim(dim32);
                uint8_t quant = 0;
                if (version >= 7) rd(&quant, 1);
                uint8_t int8ram = 0;
                float   int8scale = 0.0f;
                if (version >= 8) {
                    rd(&int8ram, 1);
                    if (int8ram) rd(&int8scale, 4);
                    if (int8ram && !(std::isfinite(int8scale) && int8scale > 0.0f))
                        throw std::runtime_error("corrupt .feather: bad int8 scale for '" + name + "'");
                }
                uint8_t persist_graph = 0;
                if (version >= 9) rd(&persist_graph, 1);
                if (modality_indices_.count(name))
                    throw std::runtime_error("corrupt .feather: modality '" + name + "' appears twice");
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
                rd(&element_count, 4);
                // Reject an element_count the file is too small to back, BEFORE
                // reserving/allocating for it. Each on-disk element is
                // id(8B) + (quant ? scale(4B)+dim : dim*4) bytes.
                {
                    const size_t min_elem = 8 + (quant ? ((size_t)dim32 + 4) : ((size_t)dim32 * 4));
                    if (element_count > left() / min_elem)
                        throw std::runtime_error("corrupt .feather: element_count "
                            + std::to_string(element_count) + " exceeds file size");
                }
                std::vector<std::pair<uint64_t, std::vector<float>>> items;
                items.reserve(element_count);
                std::vector<int8_t> qbuf(quant ? dim32 : 0);
                for (uint32_t i = 0; i < element_count; ++i) {
                    uint64_t id = 0;
                    rd(&id, 8);
                    std::vector<float> vec(dim32);
                    if (quant) {
                        float scale = 1.0f;
                        rd(&scale, 4);
                        rd(qbuf.data(), dim32);   // dim bytes
                        dequantize_vec(qbuf.data(), dim32, scale, vec.data());
                    } else {
                        rd(vec.data(), dim32 * sizeof(float));
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
        if (closed_) throw HandleClosed("DB is closed: " + path_);
    }

public:
    /// Open a database. `read_only` takes a SHARED lock, so any number of
    /// readers may hold the file at once; the default takes an EXCLUSIVE one and
    /// refuses if another process already has it. See the lock_path_ comment for
    /// why this is enforcement of single-writer rather than concurrent writing.
    static std::unique_ptr<DB> open(const std::string& path, size_t default_dim = 768,
                                    bool read_only = false) {
        // Writer opens of one path take turns within the process. They always
        // did — the Python binding held the GIL across open() — and the load
        // depends on it: a writer's load cuts a torn WAL tail, repairs a header
        // slot and removes a stale .tmp, which two loads of one file must not
        // do at once. A reader's load writes nothing, so readers never wait.
        // Declared before `db`, so a failed load is torn down inside it.
        std::unique_lock<std::mutex> opening;
        if (!read_only) opening = std::unique_lock<std::mutex>(open_mutex(lock_key(path)));
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
        mark_meta_dirty(id);
        std::string prev_content;
        bool had_prev = false;
        auto it = metadata_store_.find(id);
        if (it != metadata_store_.end()) {
            deindex_meta(id, it->second);   // drop stale secondary-index entries
            prev_content = bm25_text(it->second);   // needed to retire old BM25 postings
            had_prev     = true;
            // Re-adding a forgotten id starts a new record. Every save drops
            // dead records, so after a reload — or a crash replay — the old
            // shell and its edges are gone; inheriting them here made the
            // in-memory state disagree with every recovered one.
            const bool was_dead = is_dead_meta(it->second);
            Metadata combined = meta;
            if (combined.edges.empty() && !it->second.edges.empty() && !was_dead) {
                combined.edges = it->second.edges;   // edges kept: reverse index unchanged
            } else if (!meta.edges.empty() || was_dead) {
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
        add_to_bm25_index(id, dead ? std::string() : bm25_text(it->second),
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

    // The in-memory half of add(), after its record is logged at `at`.
    void apply_add_nolock(uint64_t id, const float* vec, size_t dim, const Metadata& meta,
                          const std::string& modality, WalPos at) {
        at.off = wal_vector_offset(at.off, modality);
        auto& m_idx = get_or_create_index(modality, dim);
        reserve(m_idx, m_idx.index->getCurrentElementCount() + 1);
        add_point(m_idx, id, vec);
        keep_full(m_idx, id, vec, at);
        log_compaction_add(modality, id, vec, dim);
        apply_meta_upsert_nolock(id, meta);
        retire_if_dead_nolock(id);
        maybe_train_int8_nolock(modality);
        ns_ann_note_nolock(id, /*vector_changed=*/true);
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
            check_key_free_nolock(id, meta);
            // WAL: log before mutating in-memory state
            const uint64_t v = next_version_nolock();
            WalPos at;
            lsn = wal_append(WalOp::ADD, id, payload, &at, v);
            apply_add_nolock(id, vec.data(), vec.size(), meta, modality, at);
            stamp_nolock(id, v);
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
            if (!metas.empty()) {
                // Keys must stay unique, within the chunk as well as against
                // the DB. (Chunks commit separately, as add_batch always has.)
                std::unordered_map<std::string, uint64_t> seen;
                for (size_t i = s; i < e; ++i) {
                    const Metadata& m = metas[i];
                    check_key_free_nolock(ids[i], m);
                    if (m.key.empty() || is_dead_meta(m)) continue;
                    auto [it, fresh] = seen.emplace(key_ref(m.namespace_id, m.key), ids[i]);
                    if (!fresh && it->second != ids[i]) throw DuplicateKey(m.namespace_id, m.key, it->second);
                }
            }
            // Log the whole chunk before applying any of it: a failed append
            // throws with this chunk's records absent from memory entirely.
            std::vector<WalPos> at(e - s);
            std::vector<uint64_t> vers(e - s);
            for (size_t i = s; i < e; ++i) {
                vers[i - s] = next_version_nolock();
                last_lsn = wal_append(WalOp::ADD, ids[i], payloads[i - s], &at[i - s], vers[i - s]);
                at[i - s].off = wal_vector_offset(at[i - s].off, modality);
            }
            for (size_t i = s; i < e; ++i) {
                apply_meta_upsert_nolock(ids[i], metas.empty() ? kDefault : metas[i]);
                log_compaction_add(modality, ids[i], vecs[i].data(), dim);
            }
            auto& m_idx = get_or_create_index(modality, dim);
            reserve(m_idx, m_idx.index->getCurrentElementCount() + (e - s));
            run_parallel(e - s, [&](size_t k) {          // concurrent graph construction
                add_point(m_idx, ids[s + k], vecs[s + k].data());
            });
            for (size_t i = s; i < e; ++i) keep_full(m_idx, ids[i], vecs[i].data(), at[i - s]);
            for (size_t i = s; i < e; ++i) retire_if_dead_nolock(ids[i]);
            for (size_t i = s; i < e; ++i) stamp_nolock(ids[i], vers[i - s]);
            for (size_t i = s; i < e; ++i) ns_ann_note_nolock(ids[i], /*vector_changed=*/true);
            // Converts within the chunk that reaches the threshold, so a bulk
            // load never builds more than ~INT8_TRAIN_MIN vectors as float.
            maybe_train_int8_nolock(modality);
            if (wal_strict()) wal_wait_durable(last_lsn);
        }
        wal_wait_durable(last_lsn);   // one fsync for the whole batch
    }

    // ─────────────────────────────────────────────────────────────────
    // Salience
    // ─────────────────────────────────────────────────────────────────
    void touch(uint64_t id) {
        std::shared_lock<RWMutex> lock(mutex_);   // atomic counters
        ensure_open();
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

private:
    // Add the edge unless the record already has it. Caller holds mutex_
    // exclusively and has checked that `from_id` is a live record.
    void apply_link_nolock(uint64_t from_id, uint64_t to_id, const std::string& rel_type, float weight) {
        auto& meta = metadata_store_.at(from_id);
        for (const auto& e : meta.edges)
            if (e.target_id == to_id && e.rel_type == rel_type) return;
        meta.edges.push_back({to_id, rel_type, weight});
        mark_meta_dirty(from_id);
        reverse_index_[to_id].push_back({from_id, rel_type, weight});
    }

public:

    // Whether the edge exists afterwards: false — and nothing written — when
    // `from_id` is not a live record (an edge from a forgotten one would
    // vanish on the next reload). The target may not exist yet: a forward
    // reference, which merge and ingest order depend on. (A WriteBatch link
    // is stricter: both ends live, so a batch never leaves an edge to nothing.)
    bool link(uint64_t from_id, uint64_t to_id,
              const std::string& rel_type = "related_to",
              float weight = 1.0f) {
        check_writable("create an edge");
        const std::string payload = encode_link(to_id, rel_type, weight);
        uint64_t lsn = 0;
        {
            std::unique_lock<RWMutex> lock(mutex_);
            ensure_open();
            auto it = metadata_store_.find(from_id);
            if (it == metadata_store_.end() || is_dead_meta(it->second)) return false;

            for (const auto& e : it->second.edges)
                if (e.target_id == to_id && e.rel_type == rel_type) return true;

            const uint64_t v = next_version_nolock();
            lsn = wal_append(WalOp::LINK, from_id, payload, nullptr, v);
            apply_link_nolock(from_id, to_id, rel_type, weight);
            stamp_nolock(from_id, v);
            if (wal_strict()) wal_wait_durable(lsn);
        }
        wal_wait_durable(lsn);
        return true;
    }

    // ─────────────────────────────────────────────────────────────────
    // Graph: query edges
    // ─────────────────────────────────────────────────────────────────
    std::vector<Edge> get_edges(uint64_t id) const {
        std::shared_lock<RWMutex> lock(mutex_);

        ensure_open();
        auto it = metadata_store_.find(id);
        if (it == metadata_store_.end()) return {};
        return it->second.edges;
    }

    std::vector<IncomingEdge> get_incoming(uint64_t id) const {
        std::shared_lock<RWMutex> lock(mutex_);

        ensure_open();
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
        check_writable("auto-link");
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
                // The kNN pass ran under the shared lock; the target may have
                // been purged or forgotten since. An edge to it would dangle.
                auto tit = metadata_store_.find(c.to);
                if (tit == metadata_store_.end() || is_dead_meta(tit->second)) continue;
                auto& meta = mit->second;
                bool exists = false;
                for (const auto& e : meta.edges)
                    if (e.target_id == c.to && e.rel_type == rel_type) { exists = true; break; }
                if (exists) continue;
                const uint64_t v = next_version_nolock();
                last_lsn = wal_append(WalOp::LINK, c.from, encode_link(c.to, rel_type, c.sim), nullptr, v);
                meta.edges.push_back({c.to, rel_type, c.sim});
                mark_meta_dirty(c.from);
                reverse_index_[c.to].push_back({c.from, rel_type, c.sim});
                meta.version = v;
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
        ensure_open();
        auto m_it = modality_indices_.find(modality);
        if (m_it == modality_indices_.end()) return {};
        auto& m_idx = m_it->second;
        check_query_dim(m_idx, query.size(), modality);

        // Step 1: vector search → seed nodes (encode query to storage format)
        auto qbytes = encode_query(m_idx, query.data());
        std::unordered_map<uint64_t, float> sim_scores;
        for (const auto& [dist, id] : knn_nolock(m_idx, query.data(), qbytes, k, 0, nullptr)) {
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

        ensure_open();
        // The records to export: from the namespace / entity index when a
        // filter names one (O(matches); this used to walk the whole store
        // three times for one namespace), else every live record. Forgotten
        // records are gone here as everywhere (the indexes never hold them).
        std::vector<std::pair<uint64_t, const Metadata*>> picked;
        auto take = [&](uint64_t id) {
            auto it = metadata_store_.find(id);
            if (it == metadata_store_.end() || is_dead_meta(it->second)) return;
            const Metadata& m = it->second;
            if (!ns_filter.empty()  && m.namespace_id != ns_filter)  return;
            if (!eid_filter.empty() && m.entity_id    != eid_filter) return;
            picked.emplace_back(id, &m);
        };
        const std::unordered_set<uint64_t>* from = nullptr;
        bool none = false;
        for (const auto& [idx, key] : {std::pair{&ns_index_, &ns_filter}, std::pair{&entity_index_, &eid_filter}}) {
            if (key->empty()) continue;
            auto it = idx->find(*key);
            if (it == idx->end()) { none = true; from = nullptr; break; }
            if (!from || it->second.size() < from->size()) from = &it->second;
        }
        if (from) {
            picked.reserve(from->size());
            for (uint64_t id : *from) take(id);
        } else if (!none) {
            picked.reserve(metadata_store_.size());
            for (const auto& kv : metadata_store_) take(kv.first);
        }
        std::sort(picked.begin(), picked.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        std::ostringstream oss;
        oss << "{\"nodes\":[";
        bool first = true;
        for (const auto& [id, mp] : picked) {
            const Metadata& meta = *mp;
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
        exported_ids.reserve(picked.size());
        for (const auto& p : picked) exported_ids.insert(p.first);

        oss << "],\"edges\":[";
        first = true;
        for (const auto& [from_id, mp] : picked) {
            for (const auto& e : mp->edges) {
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
    // A forgotten record is gone: None, as after any save or reload (which
    // drop dead records). It used to come back as a "_forgotten" shell until
    // the next save — so the same call answered differently before and after
    // a reopen, and differently from the Cloud API (404). `include_dead` is
    // for callers that inspect tombstones on purpose (recovery tests, a
    // listing of deletions).
    std::optional<Metadata> get_metadata(uint64_t id, bool include_dead = false) const {
        std::shared_lock<RWMutex> lock(mutex_);

        ensure_open();
        auto it = metadata_store_.find(id);
        if (it == metadata_store_.end() || (!include_dead && is_dead_meta(it->second))) return std::nullopt;
        return it->second;
    }

private:
    // The in-memory half of update_metadata(): replace the record, edges
    // included. Caller holds mutex_ exclusively and has logged the change.
    void apply_update_nolock(uint64_t id, const Metadata& meta) {
        mark_meta_dirty(id);
        std::string prev_content;
        bool had_prev = false;
        auto old = metadata_store_.find(id);
        if (old != metadata_store_.end()) {
            deindex_meta(id, old->second);
            prev_content = bm25_text(old->second);
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
        retire_if_dead_nolock(id);
        for (const auto& e : meta.edges)
            reverse_index_[e.target_id].push_back({id, e.rel_type, e.weight});
        // A record updated into a dead state must leave the keyword index too.
        add_to_bm25_index(id, is_dead_meta(meta) ? std::string() : bm25_text(meta),
                          had_prev ? &prev_content : nullptr);
        ns_ann_note_nolock(id, /*vector_changed=*/false);   // its namespace may have changed
    }

public:
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
            check_key_free_nolock(id, meta);
            const uint64_t v = next_version_nolock();
            lsn = wal_append(WalOp::UPDATE, id, payload, nullptr, v);
            apply_update_nolock(id, meta);
            stamp_nolock(id, v);
            if (wal_strict()) wal_wait_durable(lsn);
        }
        wal_wait_durable(lsn);
    }

    // ─────────────────────────────────────────────────────────────────
    // Versions, atomic batches, idempotency
    // ─────────────────────────────────────────────────────────────────
    // A record's compare-and-set version: 0 when it does not exist or was
    // forgotten. Every change to the record moves it to a value it never
    // had before (one clock per DB, never reused).
    uint64_t version(uint64_t id) const {
        std::shared_lock<RWMutex> lock(mutex_);
        ensure_open();
        return current_version_nolock(id);
    }

    // Apply `ops` as one unit: every op, or none.
    //   * Checked first, against the state before the batch: every
    //     `expected` version (VersionConflict), dims, and that both ends of a
    //     LINK are live records once the ops before it are applied. A failed
    //     check logs and applies nothing.
    //   * Logged as one WAL group in one write (BEGIN, records, COMMIT), so a
    //     crash part-way through drops the whole group on replay.
    //   * Applied under one exclusive lock: no reader or save sees part of it.
    // `idempotency_key`: a batch presented again with the same key returns the
    // first result (`replayed`) without applying anything; with a different
    // `fingerprint` it raises IdempotencyKeyReused. Keys are logged and saved
    // with the batch, so a retry after a crash is recognised too; they expire
    // after FEATHER_IDEMPOTENCY_TTL_S (default 24 h), and past
    // FEATHER_IDEMPOTENCY_MAX keys (default 100,000) the oldest are dropped.
    BatchResult write_batch(std::vector<BatchOp> ops,
                            const std::string& idempotency_key = "",
                            const std::string& fingerprint = "",
                            const std::string& note = "") {
        check_writable("write a batch");
        if (ops.empty()) return {};
        if (idempotency_key.size() > 1024 || fingerprint.size() > 1024)
            throw std::invalid_argument("idempotency key and fingerprint are limited to 1024 bytes");
        if (note.size() > MAX_IDEM_NOTE)
            throw std::invalid_argument("idempotency note is limited to 64 KiB");
        auto bad = [](size_t i, const std::string& why) {
            return std::invalid_argument("batch op " + std::to_string(i) + ": " + why);
        };
        // An op that names records by key is resolved, and encoded, under the
        // lock; the rest are encoded here, without it.
        auto by_key = [](const BatchOp& op) {
            return op.kind == BatchOp::UPSERT || op.ref || op.to_ref;
        };
        auto encode = [&](size_t i) -> std::string {
            const auto& op = ops[i];
            switch (op.kind) {
                case BatchOp::PUT:    return encode_add(op.modality, op.vec, op.meta);
                case BatchOp::UPDATE: {
                    std::ostringstream ws;
                    op.meta.serialize(ws);
                    return ws.str();
                }
                case BatchOp::LINK:   return encode_link(op.to_id, op.rel_type, op.weight);
                default:              return std::string();
            }
        };
        std::vector<std::string> payloads(ops.size());
        for (size_t i = 0; i < ops.size(); ++i) {
            const auto& op = ops[i];
            switch (op.kind) {
                case BatchOp::PUT:
                    if (op.vec.empty()) throw bad(i, "put needs a vector");
                    break;
                case BatchOp::UPSERT:
                    if (op.meta.key.empty()) throw bad(i, "upsert needs meta.key");
                    break;
                case BatchOp::LINK:
                    // encode_link() keeps 255 bytes: a longer type would be one
                    // edge in memory and another after a replay.
                    if (op.rel_type.size() > 255) throw bad(i, "rel_type longer than 255 bytes");
                    break;
                case BatchOp::UPDATE:
                case BatchOp::FORGET:
                    break;
                default:
                    throw bad(i, "unknown kind");
            }
            if (!by_key(op)) payloads[i] = encode(i);
        }

        BatchResult res;
        uint64_t lsn = 0;
        {
            std::unique_lock<RWMutex> lock(mutex_);
            ensure_open();
            if (!idempotency_key.empty()) {
                if (auto prior = idem_result_nolock(idempotency_key, fingerprint)) return *prior;
            }

            // Check everything, and resolve every key, before logging anything.
            // Keys resolve against the state the ops before them leave: an
            // overlay over key_index_ of what this batch binds and frees.
            constexpr uint64_t NONE = std::numeric_limits<uint64_t>::max();
            std::unordered_map<std::string, uint64_t> key_ov;    // key_ref -> id (NONE: freed)
            std::unordered_map<uint64_t, std::string> idkey_ov;  // id -> key_ref ("" = none)
            std::unordered_map<uint64_t, bool> live_after;       // records this batch wrote
            std::unordered_map<std::string, size_t> dims;
            auto id_of_key = [&](const std::string& kr) -> uint64_t {
                auto o = key_ov.find(kr);
                if (o != key_ov.end()) return o->second;
                auto k = key_index_.find(kr);
                return k == key_index_.end() ? NONE : k->second;
            };
            auto key_of_id = [&](uint64_t id) -> std::string {
                auto o = idkey_ov.find(id);
                if (o != idkey_ov.end()) return o->second;
                auto m = metadata_store_.find(id);
                if (m == metadata_store_.end() || is_dead_meta(m->second) || m->second.key.empty()) return "";
                return key_ref(m->second.namespace_id, m->second.key);
            };
            auto set_key = [&](uint64_t id, const std::string& kr) {
                const std::string old = key_of_id(id);
                if (!old.empty() && old != kr) key_ov[old] = NONE;
                if (!kr.empty()) key_ov[kr] = id;
                idkey_ov[id] = kr;
            };
            auto live = [&](uint64_t id) {
                auto it = live_after.find(id);
                return it != live_after.end() ? it->second : current_version_nolock(id) != 0;
            };
            auto resolve = [&](size_t i, const KeyRef& r) {
                const uint64_t id = id_of_key(key_ref(r.ns, r.key));
                if (id == NONE) throw bad(i, "no live record has key '" + r.key + "' in namespace '" + r.ns + "'");
                return id;
            };
            for (size_t i = 0; i < ops.size(); ++i) {
                auto& op = ops[i];
                if (op.kind == BatchOp::UPSERT) {
                    const uint64_t found = id_of_key(key_ref(op.meta.namespace_id, op.meta.key));
                    op.id = found != NONE ? found : allocate_id_nolock(live_after);
                    if (!op.vec.empty()) {
                        op.kind = BatchOp::PUT;
                    } else {
                        op.kind = BatchOp::UPDATE;
                        // add()'s rule — metadata without edges keeps the
                        // record's — applied here, so the log holds exactly
                        // what is applied and replay needs no rule of its own.
                        auto m = metadata_store_.find(op.id);
                        if (op.meta.edges.empty() && m != metadata_store_.end() && !is_dead_meta(m->second))
                            op.meta.edges = m->second.edges;
                    }
                } else if (op.ref) {
                    op.id = resolve(i, *op.ref);
                }
                if (op.to_ref) op.to_id = resolve(i, *op.to_ref);
                if (op.expected) {
                    const uint64_t cur = current_version_nolock(op.id);
                    if (cur != *op.expected) throw VersionConflict(op.id, *op.expected, cur);
                }
                switch (op.kind) {
                    case BatchOp::PUT:
                    case BatchOp::UPDATE: {
                        if (op.kind == BatchOp::PUT) {
                            check_dim_nolock(op.modality, op.vec.size());
                            auto [it, fresh] = dims.emplace(op.modality, op.vec.size());
                            if (!fresh && it->second != op.vec.size())
                                throw std::runtime_error("Dimension mismatch for modality " + op.modality +
                                                         " within the batch");
                        }
                        const bool alive = !is_dead_meta(op.meta);
                        const std::string kr = alive && !op.meta.key.empty()
                            ? key_ref(op.meta.namespace_id, op.meta.key) : std::string();
                        if (!kr.empty()) {
                            const uint64_t holder = id_of_key(kr);
                            if (holder != NONE && holder != op.id)
                                throw DuplicateKey(op.meta.namespace_id, op.meta.key, holder);
                        }
                        set_key(op.id, kr);
                        live_after[op.id] = alive;
                        break;
                    }
                    case BatchOp::LINK:
                        if (!live(op.id) || (op.require_target && !live(op.to_id)))
                            throw bad(i, "link " + std::to_string(op.id) + " -> " + std::to_string(op.to_id) +
                                         " needs both ends to be live records");
                        break;
                    case BatchOp::FORGET:
                        set_key(op.id, "");
                        live_after[op.id] = false;
                        break;
                    default:
                        break;
                }
            }
            for (size_t i = 0; i < ops.size(); ++i)
                if (payloads[i].empty() && ops[i].kind != BatchOp::FORGET) payloads[i] = encode(i);

            static constexpr WalOp kOp[] = {WalOp::ADD, WalOp::ADD, WalOp::UPDATE, WalOp::LINK, WalOp::FORGET};
            std::vector<uint64_t> vers(ops.size());
            for (auto& v : vers) v = next_version_nolock();
            const int64_t now = static_cast<int64_t>(std::time(nullptr));
            std::vector<std::pair<uint64_t, uint64_t>> results;
            results.reserve(ops.size());
            for (size_t i = 0; i < ops.size(); ++i) results.emplace_back(ops[i].id, vers[i]);

            // One op without a key needs no group: a record and its STAMP.
            const bool grouped = ops.size() > 1 || !idempotency_key.empty();
            std::string begin, commit;
            std::vector<WalRec> recs;
            recs.reserve(ops.size() + 2);
            if (grouped) {
                v10::put<uint32_t>(begin, static_cast<uint32_t>(ops.size()));
                v10::put_str16(begin, idempotency_key);
                v10::put_str16(begin, fingerprint);
                v10::put<int64_t>(begin, now);
                put_str32(begin, idempotency_key.empty() ? std::string() : note);
                v10::put<uint32_t>(commit, static_cast<uint32_t>(ops.size()));
                for (uint64_t v : vers) v10::put<uint64_t>(commit, v);
                const bool keyed = !idempotency_key.empty();
                v10::put<uint32_t>(commit, keyed ? static_cast<uint32_t>(results.size()) : 0u);
                if (keyed)
                    for (const auto& [id, v] : results) { v10::put<uint64_t>(commit, id); v10::put<uint64_t>(commit, v); }
                recs.push_back({WalOp::BEGIN, 0, &begin});
            }
            for (size_t i = 0; i < ops.size(); ++i) recs.push_back({kOp[ops[i].kind], ops[i].id, &payloads[i]});
            const std::string stamp = encode_u64(vers[0]);
            if (grouped) recs.push_back({WalOp::COMMIT, 0, &commit});
            else         recs.push_back({WalOp::STAMP, ops[0].id, &stamp});
            std::vector<WalPos> at;
            lsn = wal_append_group(recs, &at);

            bool forgot = false;
            for (size_t i = 0; i < ops.size(); ++i) {
                const auto& op = ops[i];
                const size_t r = i + (grouped ? 1 : 0);
                switch (op.kind) {
                    case BatchOp::PUT:
                        apply_add_nolock(op.id, op.vec.data(), op.vec.size(), op.meta, op.modality,
                                         r < at.size() ? at[r] : WalPos{});
                        break;
                    case BatchOp::UPDATE: apply_update_nolock(op.id, op.meta);                        break;
                    case BatchOp::LINK:   apply_link_nolock(op.id, op.to_id, op.rel_type, op.weight); break;
                    case BatchOp::FORGET: forget_nolock(op.id); forgot = true;                        break;
                    default:              break;
                }
                stamp_nolock(op.id, vers[i]);
            }
            for (const auto& [id, v] : results) {
                res.ids.push_back(id);
                res.versions.push_back(v);
            }
            if (!idempotency_key.empty())
                idem_put_nolock(idempotency_key, {fingerprint, now, std::move(results), note});
            if (forgot) maybe_auto_compact_nolock();
            if (wal_strict()) wal_wait_durable(lsn);
        }
        wal_wait_durable(lsn);
        return res;
    }

    // ── Native keys ─────────────────────────────────────────────────
    // The live record holding `key` in `ns`, if any.
    std::optional<uint64_t> id_for(const std::string& key, const std::string& ns = "") const {
        std::shared_lock<RWMutex> lock(mutex_);
        ensure_open();
        auto it = key_index_.find(key_ref(ns, key));
        if (it == key_index_.end()) return std::nullopt;
        return it->second;
    }

    std::optional<Metadata> get_by_key(const std::string& key, const std::string& ns = "") const {
        std::shared_lock<RWMutex> lock(mutex_);
        ensure_open();
        auto it = key_index_.find(key_ref(ns, key));
        if (it == key_index_.end()) return std::nullopt;
        return metadata_store_.at(it->second);
    }

    // Write the record keyed (meta.namespace_id, meta.key): the existing one,
    // or a new one with a fresh id. With `vec` as add(); without, the
    // metadata replaces the record's, keeping its edges when `meta` has none.
    // Returns (id, version).
    std::pair<uint64_t, uint64_t> upsert(const Metadata& meta, const std::vector<float>& vec = {},
                                         const std::string& modality = "text",
                                         std::optional<uint64_t> expected = std::nullopt) {
        BatchOp op;
        op.kind = BatchOp::UPSERT;
        op.meta = meta;
        op.vec = vec;
        op.modality = modality;
        op.expected = expected;
        const BatchResult r = write_batch({std::move(op)});
        return {r.ids.at(0), r.versions.at(0)};
    }

private:
    // Ids of keyed records come from the version clock, which is persisted and
    // never moves back, so an id is never handed out twice — not even after
    // the record that had it is deleted and dropped (an edge still pointing at
    // the old id must not start pointing at a stranger). Offset by 2^52, above
    // the small explicit ids callers pick and still below 2^53 (exact in JSON).
    static constexpr uint64_t KEY_ID_BASE = uint64_t{1} << 52;
    uint64_t allocate_id_nolock(const std::unordered_map<uint64_t, bool>& taken) {
        for (;;) {
            const uint64_t id = KEY_ID_BASE + next_version_nolock();
            if (!metadata_store_.count(id) && !taken.count(id)) return id;
        }
    }

public:
    // The stored result of an earlier batch with this key, if there is one
    // (replayed = true); raises IdempotencyKeyReused for another fingerprint.
    // For a caller whose own checks would answer a retry differently — a
    // delete retried after it succeeded finds no record — so it can ask
    // first. write_batch() checks again, atomically, either way.
    std::optional<BatchResult> idempotent_result(const std::string& key, const std::string& fingerprint) const {
        std::shared_lock<RWMutex> lock(mutex_);
        ensure_open();
        return idem_result_nolock(key, fingerprint);
    }

    // The ids of the `limit` newest live records in namespace `ns` (with
    // `prefix`: in it and every namespace below it), newest first by
    // (timestamp, id). `before` pages: only records strictly older than that
    // (timestamp, id). A walk from the end of an ordered index — O(log n +
    // limit), a k-way merge for a prefix — where a caller used to read the
    // whole namespace and sort it. `by_valid`: by (valid_time(), id) instead,
    // newest event first, and `before` is then a valid-time cursor.
    std::vector<uint64_t> newest(const std::string& ns, size_t limit, bool prefix = false,
                                 std::optional<std::pair<int64_t, uint64_t>> before = std::nullopt,
                                 bool by_valid = false) const {
        std::shared_lock<RWMutex> lock(mutex_);
        ensure_open();
        const auto& index = by_valid ? valid_index_ : time_index_;
        std::vector<const std::set<TimeKey>*> sets;
        auto add = [&](const std::string& name) {
            auto it = index.find(name);
            if (it != index.end()) sets.push_back(&it->second);
        };
        if (prefix && !ns.empty()) for_each_ns_under(ns, add);
        else if (prefix) for (const auto& [name, set] : index) sets.push_back(&set);
        else add(ns);
        using RIt = std::set<TimeKey>::const_reverse_iterator;
        struct Head { RIt it, end; };
        auto newer = [](const Head& a, const Head& b) { return *a.it < *b.it; };   // max-heap
        std::vector<Head> heap;
        for (const auto* set : sets) {
            auto start = before ? RIt(set->lower_bound(*before)) : set->rbegin();
            if (start != set->rend()) heap.push_back({start, set->rend()});
        }
        std::make_heap(heap.begin(), heap.end(), newer);
        std::vector<uint64_t> out;
        out.reserve(std::min<size_t>(limit, 1024));
        while (!heap.empty() && out.size() < limit) {
            std::pop_heap(heap.begin(), heap.end(), newer);
            Head& h = heap.back();
            out.push_back(h.it->second);
            if (++h.it == h.end) heap.pop_back();
            else std::push_heap(heap.begin(), heap.end(), newer);
        }
        return out;
    }

    // The namespace size from which a filtered search walks a namespace graph
    // instead of scanning. 0 — the default unless FEATHER_NS_ANN_MIN is set —
    // keeps every filtered search an exact scan. 4096 is the measured point
    // where a graph pays (benchmarks/namespace_search.py).
    void set_namespace_ann_min(size_t n) {
        std::unique_lock<RWMutex> lock(mutex_);
        ensure_open();
        ns_ann_min_.store(n, std::memory_order_relaxed);
        if (n == 0) ns_graphs_.clear();
    }

    // Per-namespace graphs: how many, how many nodes, and how many are queued.
    struct NsAnnInfo { size_t graphs = 0, nodes = 0, pending = 0, min_records = 0; };
    NsAnnInfo namespace_ann_info() const {
        NsAnnInfo info;
        info.min_records = ns_ann_min();
        {
            std::shared_lock<RWMutex> lock(mutex_);
            ensure_open();
            for (const auto& [key, g] : ns_graphs_) {
                auto mit = modality_indices_.find(key.first);
                if (mit == modality_indices_.end() || g->gen != mit->second.ann_gen) continue;
                info.graphs += 1;
                info.nodes  += g->index->cur_element_count;
            }
        }
        std::lock_guard<std::mutex> g(const_cast<std::mutex&>(ns_mx_));
        info.pending = ns_pending_.size();
        return info;
    }

    // Block until no namespace graph is queued or being built (or timeout).
    bool wait_for_namespace_graphs(double timeout_s = 0.0) {
        std::unique_lock<std::mutex> g(ns_mx_);
        auto idle = [&] { return stop_ns_builder_ || (ns_pending_.empty() && !ns_building_); };
        if (timeout_s <= 0.0) { ns_cv_.wait(g, idle); return true; }
        return ns_cv_.wait_for(g, std::chrono::duration<double>(timeout_s), idle);
    }

    // Compare-and-set: replace record `id` with `meta` (edges included, as
    // update_metadata() does) only if its version is still `expected` (0 = it
    // must not exist yet). With `vec`, the vector is replaced in the same
    // atomic write. Returns the new version; throws VersionConflict.
    uint64_t put_if(uint64_t id, const Metadata& meta, uint64_t expected,
                    const std::vector<float>& vec = {}, const std::string& modality = "text") {
        std::vector<BatchOp> ops;
        if (!vec.empty()) {
            BatchOp put;
            put.kind = BatchOp::PUT;
            put.id = id;
            put.meta = meta;
            put.vec = vec;
            put.modality = modality;
            ops.push_back(std::move(put));
        }
        BatchOp up;
        up.kind = BatchOp::UPDATE;
        up.id = id;
        up.meta = meta;
        ops.push_back(std::move(up));
        ops.front().expected = expected;
        return write_batch(ops).versions.back();
    }

    void update_importance(uint64_t id, float importance) {
        check_writable("update importance");
        std::string payload(reinterpret_cast<const char*>(&importance), 4);
        uint64_t lsn = 0;
        {
            std::unique_lock<RWMutex> lock(mutex_);
            ensure_open();
            const uint64_t v = next_version_nolock();
            lsn = wal_append(WalOp::UIMP, id, payload, nullptr, v);
            auto it = metadata_store_.find(id);
            if (it != metadata_store_.end()) { it->second.importance = importance; mark_meta_dirty(id); }
            stamp_nolock(id, v);
            if (wal_strict()) wal_wait_durable(lsn);
        }
        wal_wait_durable(lsn);
    }

    // Get raw vector for a given id and modality (empty if not found)
    std::vector<float> get_vector(uint64_t id, const std::string& modality = "text") const {
        std::shared_lock<RWMutex> lock(mutex_);

        ensure_open();
        auto it = modality_indices_.find(modality);
        if (it == modality_indices_.end()) return {};
        try {
            return read_vector_best(it->second, id);    // the float original, else dequantized
        } catch (...) {
            return {};
        }
    }

    // Get all IDs present in a modality index
    std::vector<uint64_t> get_all_ids(const std::string& modality = "text") const {
        std::shared_lock<RWMutex> lock(mutex_);

        ensure_open();
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

        ensure_open();
        std::vector<uint64_t> ids;
        ids.reserve(metadata_store_.size());
        for (const auto& [id, _] : metadata_store_) ids.push_back(id);
        return ids;
    }

    // The actual modality index names present in this DB (e.g. "text",
    // "visual", or whatever an external pipeline named them).
    std::vector<std::string> modality_names() const {
        std::shared_lock<RWMutex> lock(mutex_);

        ensure_open();
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

        ensure_open();
        auto it = ns_index_.find(ns);
        if (it == ns_index_.end()) return {};
        return {it->second.begin(), it->second.end()};
    }

    std::vector<uint64_t> ids_for_entity(const std::string& eid) const {
        std::shared_lock<RWMutex> lock(mutex_);

        ensure_open();
        auto it = entity_index_.find(eid);
        if (it == entity_index_.end()) return {};
        return {it->second.begin(), it->second.end()};
    }

    std::vector<uint64_t> ids_with_attribute(const std::string& key,
                                             const std::string& val) const {
        std::shared_lock<RWMutex> lock(mutex_);

        ensure_open();
        auto it = attr_index_.find(attr_key(key, val));
        if (it == attr_index_.end()) return {};
        return {it->second.begin(), it->second.end()};
    }

    size_t namespace_size(const std::string& ns) const {
        std::shared_lock<RWMutex> lock(mutex_);

        ensure_open();
        auto it = ns_index_.find(ns);
        return it == ns_index_.end() ? 0 : it->second.size();
    }

    std::vector<std::string> list_namespaces() const {
        std::shared_lock<RWMutex> lock(mutex_);

        ensure_open();
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
    // Refine: give the `n` nearest of `cands` (int8 distance, id) their exact
    // float32 L2, read from each one's original (fv_pending or the file).
    // Candidates without an original keep their int8 distance. Measured at
    // 384-d, exact scan: re-ranking the top 2k lifts recall@10 to >= 0.9997
    // from 0.906-0.978 depending on the scale. Caller holds mutex_ (shared).
    void refine_nolock(const ModalityIndex& m, const float* q,
                       std::vector<std::pair<float, uint64_t>>& cands, size_t n) const {
        n = std::min(n, cands.size());
        if (n == 0) return;
        if (n < cands.size())
            std::partial_sort(cands.begin(), cands.begin() + n, cands.end());
        const auto l2 = feather_simd::l2_fn();
        const auto& idx = *m.index;
        std::vector<float> v(m.dim);
        for (size_t i = 0; i < n; ++i) {
            auto lit = idx.label_lookup_.find(cands[i].second);
            if (lit == idx.label_lookup_.end()) continue;
            if (full_vector(m, static_cast<uint32_t>(lit->second), v.data()))
                cands[i].first = l2(q, v.data(), m.dim);
        }
    }

    // The k nearest (distance, id), nearest first — the graph walk every
    // vector query goes through. For an int8 modality with refine on it walks
    // with ef x refine and fetches k x refine, re-ranked by exact float L2.
    // The wider beam is not optional: int8 distances misjudge which neighbour
    // to expand, so true neighbours never enter a beam of the float width,
    // and no re-rank can recover them. Measured at 100k x 384, ef 50:
    // re-ranking alone plateaued at recall 0.980-0.985 for refine 2..8 (float
    // 0.995); with the beam widened, 0.995. The walk's fault, not the graph's:
    // a float-built graph walked in int8 lost the same.
    std::vector<std::pair<float, uint64_t>>
    knn_nolock(const ModalityIndex& m, const float* q, const std::vector<char>& qbytes,
               size_t k, size_t ef_override, hnswlib::BaseFilterFunctor* f) const {
        const size_t refine = (m.int8 && m.refine) ? m.refine : 0;
        const size_t ef = refine ? (ef_override ? ef_override : m.index->ef_) * refine
                                 : ef_override;
        auto res = m.index->searchKnnEf(qbytes.data(), refine ? k * refine : k, ef, f);
        std::vector<std::pair<float, uint64_t>> out;
        out.reserve(res.size());
        while (!res.empty()) { out.push_back(res.top()); res.pop(); }
        if (refine) refine_nolock(m, q, out, out.size());
        std::sort(out.begin(), out.end());
        if (out.size() > k) out.resize(k);
        return out;
    }

    // The namespaces a filter confines a search to: its namespace_id, or every
    // namespace under its namespace_prefix. Empty when it confines to none.
    std::vector<std::string> filter_namespaces(const SearchFilter& f) const {
        std::vector<std::string> out;
        if (f.namespace_id) {
            if (ns_index_.count(*f.namespace_id) &&
                (!f.namespace_prefix || SearchFilter::under(*f.namespace_id, *f.namespace_prefix)))
                out.push_back(*f.namespace_id);
        } else if (f.namespace_prefix && !f.namespace_prefix->empty()) {
            for_each_ns_under(*f.namespace_prefix, [&](const std::string& ns) { out.push_back(ns); });
        }
        return out;
    }

    // Whether a filter constrains nothing but namespaces.
    static bool namespace_only(const SearchFilter& f) {
        return !f.types && !f.source && !f.source_prefix && !f.timestamp_after &&
               !f.timestamp_before && !f.valid_after && !f.valid_before &&
               !f.importance_gte && !f.confidence_gte &&
               !f.tags_contains && !f.entity_id && !f.attributes_match;
    }

    // A namespace graph's walk returns only live records still in that
    // namespace that pass the rest of the filter.
    struct NsFilter : public hnswlib::BaseFilterFunctor {
        const SearchFilter* filter_;
        const std::string& ns_;
        const std::unordered_map<uint64_t, Metadata>& store_;
        NsFilter(const SearchFilter* f, const std::string& ns,
                 const std::unordered_map<uint64_t, Metadata>& s) : filter_(f), ns_(ns), store_(s) {}
        bool operator()(hnswlib::labeltype id) override {
            auto it = store_.find(id);
            if (it == store_.end() || is_dead_meta(it->second)) return false;
            if (it->second.namespace_id != ns_) return false;
            return !filter_ || filter_->matches(it->second);
        }
    };

    // Fill SearchResult::cosine for the (already truncated) result set.
    void fill_cosines(const ModalityIndex& m_idx, const std::vector<float>& q,
                      std::vector<SearchResult>& results) const {
        double qn = 0.0;
        for (float x : q) qn += static_cast<double>(x) * x;
        qn = std::sqrt(qn);
        for (auto& r : results) {
            std::vector<float> v;
            try { v = read_vector_best(m_idx, r.id); } catch (...) { continue; }
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
        ensure_open();
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

        // In-RAM int8 with refine on: candidates re-ranked by exact float L2
        // (knn_nolock on the graph path, refine_nolock on the exact scan).
        const size_t refine = (m_idx.int8 && m_idx.refine) ? m_idx.refine : 0;

        // Graph traversal, optionally with a per-call ef. Returns (score, id).
        auto hnsw = [&](const SearchFilter* f, size_t ef_override) {
            FilterWrapper fw(f, metadata_store_);
            const auto dists = knn_nolock(m_idx, q.data(), qbytes, want, ef_override, f ? &fw : nullptr);
            std::vector<std::pair<float, uint64_t>> scored;
            scored.reserve(dists.size());
            for (const auto& [dist, id] : dists) {
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
            // Confined to namespaces holding at least FEATHER_NS_ANN_MIN records:
            // each namespace with a graph is walked, the rest scanned. A filter
            // on namespaces alone is routed by their sizes, without copying
            // its candidates (50k ids for a 50k-record tenant).
            const std::vector<std::string> spaces = filter_namespaces(*filter);
            size_t ns_total = 0;
            for (const auto& ns : spaces) ns_total += ns_index_.at(ns).size();
            const bool graphs_fit = ns_ann_min() && ns_total >= ns_ann_min();
            const bool sized_only = graphs_fit && namespace_only(*filter);
            bool indexed = sized_only;
            std::vector<uint64_t> cand;
            if (!sized_only) cand = candidates_for_filter(*filter, indexed);
            const bool use_graphs = sized_only || (graphs_fit && cand.size() >= ns_ann_min());
            if (indexed) {
                const size_t live = m_idx.index->getCurrentElementCount() -
                                    m_idx.index->getDeletedCount();
                const int mode = prefilter_mode();
                if (!use_graphs && mode != PREFILTER_EXACT && live > 0 && !cand.empty()) {
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
                // (distance, id) for everything examined, scored at the end.
                std::vector<std::pair<float, uint64_t>> scored;
                auto& idx = *m_idx.index;
                const auto dist_fn = m_idx.space->get_dist_func();
                void* dist_param   = m_idx.space->get_dist_func_param();
                const auto f32i8   = feather_simd::f32_i8_l2_fn();
                auto scan = [&](uint64_t id) {
                    auto it = metadata_store_.find(id);
                    if (it == metadata_store_.end() || is_dead_meta(it->second)) return;
                    if (!filter->matches(it->second)) return;     // non-indexed predicates
                    // label_lookup_ is only mutated under the exclusive lock, so
                    // reading it under our shared lock needs no extra mutex.
                    auto lit = idx.label_lookup_.find(id);
                    if (lit == idx.label_lookup_.end()) return;   // not in this modality
                    if (idx.isMarkedDeleted(lit->second)) return;
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
                    scored.push_back({dist, id});
                };

                if (use_graphs) {
                    const size_t kk = refine ? want * refine : want;
                    // Twice the modality's beam: this walk replaces an exact
                    // scan, so it buys recall back (0.93 -> 0.99 on 32-d
                    // Gaussian data) at a cost still a fraction of the scan's.
                    const size_t ef = 2 * std::max(m_idx.index->ef_, want) * (refine ? refine : 1);
                    const uint64_t qref = SlotRefSpace::query_ref(qbytes.data());
                    for (const auto& ns : spaces) {
                        const auto& members = ns_index_.at(ns);
                        const NsGraph* g = members.size() >= ns_ann_min()
                                               ? ns_graph_nolock(modality, m_idx, ns) : nullptr;
                        if (g) {
                            // Membership is exact (marks), so a filter that only
                            // names namespaces needs no per-node callback.
                            NsFilter nf(filter, ns, metadata_store_);
                            auto res = g->index->searchKnnEf(&qref, kk, ef,
                                                             namespace_only(*filter) ? nullptr : &nf);
                            // Short: the other predicates leave too few in this
                            // namespace for a walk to find. Scan it instead.
                            if (res.size() >= std::min(kk, members.size())) {
                                for (; !res.empty(); res.pop()) scored.push_back(res.top());
                                continue;
                            }
                        } else if (members.size() >= ns_ann_min()) {
                            request_ns_graph(modality, ns);   // scanned until it is ready
                        }
                        for (uint64_t id : members) scan(id);
                    }
                } else {
                    scored.reserve(cand.size());
                    for (uint64_t id : cand) scan(id);
                }
                // The nearest want x refine on their exact distance, then scores.
                if (refine) refine_nolock(m_idx, q.data(), scored, want * refine);
                for (auto& [dist, id] : scored) dist = score_of(dist, metadata_store_.find(id)->second);
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
    // FEATHER_PREFILTER_MODE=exact|auto|hnsw. EXACT is the default: an indexed
    // filter (namespace / entity / attribute) has always meant an exact,
    // complete top-k, and callers such as Pocket.recall rely on it. `auto`
    // opts in to the cost model, which walks the graph for large unselective
    // sets — faster, but approximate (measured recall@10 0.86 on a 50% filter,
    // and with scoring= it only re-ranks the nearest candidates).
    static int prefilter_mode() {
        static const int m = [] {
            const char* e = std::getenv("FEATHER_PREFILTER_MODE");
            if (e && !std::strcmp(e, "auto")) return static_cast<int>(PREFILTER_AUTO);
            if (e && !std::strcmp(e, "hnsw")) return static_cast<int>(PREFILTER_HNSW);
            return static_cast<int>(PREFILTER_EXACT);
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

    // Best first; ties go to the HIGHER id -- the order std::greater<pair<score,
    // id>> gave keyword_search before this comparator existed, which callers may
    // depend on. Note this is the opposite of search()'s finish lambda, which
    // breaks ties toward the smaller id; the two paths are deliberately not
    // unified here because changing either reorders results callers have seen.
    static bool better_scored(const ScoredId& a, const ScoredId& b) {
        return a.first > b.first || (a.first == b.first && a.second > b.second);
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

        // Scores accumulate by doc number and become ids only for the ranking:
        // a list is in number order, so its length and id lookups are
        // sequential reads. A filter is applied per posting: it usually
        // rejects most documents, and keeping them out of `scores` is cheaper
        // than scoring them. Without one, forget()/purge() already drop
        // postings, so eligibility is only checked on the winners below.
        const double N = static_cast<double>(doc_lengths_.size());
        const double avdl = avg_dl_ > 0.0 ? avg_dl_ : 1.0;
        std::unordered_map<uint32_t, float> scores;
        if (!filter) scores.reserve(total_postings);
        for (const auto* postings : lists) {
            const double n_t = static_cast<double>(postings->size());
            // IDF (BM25+): log((N - n_t + 0.5) / (n_t + 0.5) + 1)
            const double idf = std::log((N - n_t + 0.5) / (n_t + 0.5) + 1.0);
            for (const auto& p : *postings) {
                if (filter && !eligible(bm25_doc_id_[p.doc])) continue;
                const double tf = static_cast<double>(p.term_freq);
                const double tf_norm = (tf * (BM25_K1 + 1.0)) /
                    (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * static_cast<double>(bm25_doc_len_[p.doc]) / avdl));
                scores[p.doc] += static_cast<float>(idf * tf_norm);
            }
        }

        ranked.reserve(scores.size());
        for (const auto& [doc, sc] : scores) ranked.emplace_back(sc, bm25_doc_id_[doc]);
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

        ensure_open();
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
        ensure_open();
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
                        auto it = s_.find(id);
                        if (it == s_.end()) return false;
                        // Same rule as search()'s FilterWrapper and bm25_top_nolock's
                        // eligible(). Without it the BM25 half of the fusion drops a
                        // tombstone and the vector half puts it back through RRF: a
                        // soft-deleted record (_deleted=true) is not markDelete'd in
                        // HNSW, so only this check keeps it out.
                        if (is_dead_meta(it->second)) return false;
                        return f_->matches(it->second);
                    }
                } fw(filter, metadata_store_);
                size_t cands = scoring ? candidates * 3 : candidates;
                auto qbytes = encode_query(m_idx, vec.data());
                const auto res = knn_nolock(m_idx, vec.data(), qbytes, cands, 0, filter ? &fw : nullptr);
                const double now_ts = static_cast<double>(std::time(nullptr));
                static const Metadata kNoMeta;
                vec_ranked.reserve(res.size());
                for (const auto& [dist, id] : res) {
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
                return a.first > b.first || (a.first == b.first && a.second > b.second);
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
    // A dead record (forgotten, or `_deleted` set through add/update) must not
    // be reachable by vector search. forget() always did this; a `_deleted`
    // record kept a live node, so it was returned by unfiltered search — and
    // once saved with the graph (v10, which keeps deleted marks) it came back
    // after a reload as a hit with no metadata. Caller holds mutex_
    // exclusively, after the record's vectors are in place.
    void retire_if_dead_nolock(uint64_t id) {
        auto it = metadata_store_.find(id);
        if (it == metadata_store_.end() || !is_dead_meta(it->second)) return;
        for (auto& [name, m_idx] : modality_indices_) {
            try { m_idx.index->markDelete(id); } catch (...) {}   // absent / already deleted
        }
        log_compaction_remove(id);
    }

    // Load-time repair. Every live vector has a live record — add() and
    // add_batch() always store metadata, and every path that kills or erases
    // a record also marks its node deleted. A file saved before that last rule
    // (a `_deleted` record in a v10 file) can still hold a live node with no
    // record behind it; search would return it as an empty hit. One hash
    // lookup per node; on a sound file it changes nothing.
    void retire_orphan_nodes() {
        for (auto& [name, m_idx] : modality_indices_) {
            auto& ix = *m_idx.index;
            for (size_t i = 0, n = ix.cur_element_count; i < n; ++i) {
                const auto iid = static_cast<hnswlib::tableint>(i);
                if (ix.isMarkedDeleted(iid)) continue;
                auto it = metadata_store_.find(static_cast<uint64_t>(ix.getExternalLabel(iid)));
                if (it == metadata_store_.end() || is_dead_meta(it->second))
                    ix.markDeletedInternal(iid);
            }
        }
    }

    // Caller holds mutex_ exclusively.
    void forget_nolock(uint64_t id) {
        for (auto& [name, m_idx] : modality_indices_) {
            try { m_idx.index->markDelete(id); } catch (...) {}
        }
        log_compaction_remove(id);
        auto it = metadata_store_.find(id);
        if (it != metadata_store_.end()) {
            mark_meta_dirty(id);
            deindex_meta(id, it->second);   // forgotten records leave candidate sets
            // ...and leave the keyword index, otherwise a forgotten record stays
            // fully retrievable via keyword_search/hybrid_search.
            const std::string old_text = bm25_text(it->second);
            remove_from_bm25_index(id, &old_text);
            it->second.content    = "";
            it->second.source     = "_forgotten";
            it->second.importance = 0.0f;
            it->second.ttl        = 0;
        }
    }

    // Hard-delete every record in a namespace. Shared by purge() and WAL
    // replay; replay passes maintain_derived=false because load_vectors()
    // rebuilds the derived indexes once afterwards.
    // Survivors whose edges into the namespace are pruned get versions
    // version_base+1, +2, ... in id order: the same in the live purge and in
    // its replay (0: from the clock, for a PURGE logged before 0.22).
    size_t purge_core(const std::string& ns_id, bool maintain_derived, uint64_t version_base = 0) {
        std::unordered_set<uint64_t> to_purge;
        for (const auto& [id, meta] : metadata_store_)
            if (meta.namespace_id == ns_id) to_purge.insert(id);
        if (to_purge.empty()) return 0;

        for (auto& [name, m_idx] : modality_indices_)
            for (uint64_t id : to_purge) {
                try { m_idx.index->markDelete(id); } catch (...) {}
            }
        // A purge is a hard delete, so the next save rewrites the file rather
        // than appending: the purged records' bytes must not survive in it.
        vacuum_requested_ = true;
        for (uint64_t id : to_purge) {
            log_compaction_remove(id);
            mark_meta_dirty(id);
            auto it = metadata_store_.find(id);
            if (it == metadata_store_.end()) continue;
            if (maintain_derived) {
                deindex_meta(id, it->second);
                // Purged ids must leave the keyword index too: a stale posting
                // resolves to no metadata and surfaces as an empty ghost hit.
                const std::string old_text = bm25_text(it->second);
                remove_from_bm25_index(id, &old_text);
            }
            metadata_store_.erase(it);
        }
        if (maintain_derived) prune_reverse_index(to_purge);
        // Prune edges in surviving nodes that pointed to purged targets
        std::vector<uint64_t> changed;
        for (auto& [id, meta] : metadata_store_) {
            const size_t before = meta.edges.size();
            meta.edges.erase(
                std::remove_if(meta.edges.begin(), meta.edges.end(),
                    [&to_purge](const Edge& e) { return to_purge.count(e.target_id) > 0; }),
                meta.edges.end());
            if (meta.edges.size() != before) { mark_meta_dirty(id); changed.push_back(id); }
        }
        std::sort(changed.begin(), changed.end());
        for (size_t k = 0; k < changed.size(); ++k)
            stamp_nolock(changed[k], version_base ? version_base + 1 + k : 0);
        return to_purge.size();
    }

public:
    void forget(uint64_t id) {
        check_writable("forget a record");
        uint64_t lsn = 0;
        {
            std::unique_lock<RWMutex> lock(mutex_);
            ensure_open();
            const uint64_t v = next_version_nolock();
            lsn = wal_append(WalOp::FORGET, id, "", nullptr, v);
            forget_nolock(id);
            stamp_nolock(id, v);
            maybe_auto_compact_nolock();
            if (wal_strict()) wal_wait_durable(lsn);
        }
        wal_wait_durable(lsn);
    }

    // Hard-delete: remove all nodes in namespace_id from indices +
    // metadata store + reverse index. Returns count of removed nodes.
    // WAL-logged since 0.19 (it used to survive a crash only via save()).
    size_t purge(const std::string& ns_id) {
        check_writable("purge a namespace");
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
            // A FORGET per record first: a build without WalOp::PURGE (0.21.0
            // and earlier) skips that op on replay, and after a crash would
            // bring every purged record back. It replays the FORGETs instead,
            // so a rollback leaves them deleted. Replaying both is idempotent.
            for (const auto& [id, meta] : metadata_store_)
                if (meta.namespace_id == ns_id) wal_append(WalOp::FORGET, id, "");
            // The version base rides at the end of the payload: a build
            // before it reads the namespace and ignores the rest.
            const uint64_t base = version_clock_;
            lsn = wal_append(WalOp::PURGE, 0, payload + encode_u64(base));
            n = purge_core(ns_id, /*maintain_derived=*/true, base);
            maybe_auto_compact_nolock();
            if (wal_strict()) wal_wait_durable(lsn);
        }
        wal_wait_durable(lsn);
        return n;
    }

    // Scan all nodes and soft-delete any with ttl>0 where now > timestamp+ttl.
    // Returns count of nodes forgotten.
    size_t forget_expired() {
        check_writable("forget expired records");
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
                const uint64_t v = next_version_nolock();
                last_lsn = wal_append(WalOp::FORGET, id, "", nullptr, v);
                forget_nolock(id);
                stamp_nolock(id, v);
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
    // A compaction renumbers slots: carry each live label's float original
    // (pending, or its place in the file) from the old index's slot to the
    // new one's. The old index holds the newest vector of every label — the
    // writes logged during the rebuild were applied to it too.
    static void remap_full_vectors(ModalityIndex& m, const hnswlib::HierarchicalNSW<float>& fresh) {
        if (!m.int8 || !m.refine) return;
        const auto& old = *m.index;
        const size_t n = fresh.cur_element_count;
        std::vector<uint64_t> off(n, 0);
        std::unordered_map<uint32_t, ModalityIndex::PendingFull> pending;
        for (size_t i = 0; i < n; ++i) {
            const auto id = static_cast<hnswlib::tableint>(i);
            if (fresh.isMarkedDeleted(id)) continue;
            auto lit = old.label_lookup_.find(fresh.getExternalLabel(id));
            if (lit == old.label_lookup_.end()) continue;
            const auto from = static_cast<uint32_t>(lit->second);
            auto p = m.fv_pending.find(from);
            if (p != m.fv_pending.end())   pending[static_cast<uint32_t>(i)] = std::move(p->second);
            else if (from < m.fv_off.size()) off[i] = m.fv_off[from];
        }
        m.fv_off     = std::move(off);
        m.fv_pending = std::move(pending);
    }

    size_t compact() {
        check_writable("compact");
        std::lock_guard<std::mutex> one_at_a_time(compact_run_mx_);
        struct Snap {
            std::string name;
            size_t dim; bool int8; float scale; size_t ef; size_t stride;
            uint64_t epoch;
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
                sn.epoch  = m_idx.epoch;
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
        // save_mutex_ first (lock order): a swap renumbers slots, so it must
        // not land between a commit's FULLVEC writes and commit_install().
        std::lock_guard<std::mutex> save_lock(save_mutex_);
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
            add_point(tmp, op.id, op.vec.data());
            sn.index = std::move(tmp.index);
        }
        compaction_log_.clear();
        compaction_log_.shrink_to_fit();

        for (auto& sn : snaps) {
            auto mit = modality_indices_.find(sn.name);
            if (mit == modality_indices_.end()) continue;
            // Converted to int8 (or recreated) meanwhile: the new index was not
            // built from what this one holds. It keeps its tombstones.
            if (mit->second.epoch != sn.epoch) continue;
            // The live index's ef, not the phase-1 snapshot: a set_ef() made
            // while compaction ran would otherwise be silently undone here.
            sn.index->setEf(mit->second.index->ef_);
            remap_full_vectors(mit->second, *sn.index);
            drop_ns_graphs_nolock(sn.name, mit->second);    // slots renumbered
            mit->second.index = std::move(sn.index);   // old index freed here
            mit->second.space = std::move(sn.space);
            mit->second.persisted = false;
        }
        // Compaction exists to reclaim space, so the next save rewrites the
        // file instead of appending a second copy of every graph.
        vacuum_requested_ = true;

        // Erase metadata that is STILL dead (an id may have been re-added
        // during phase 2). Dead records are already out of the secondary and
        // BM25 indexes (forget/purge/update remove them), so only the reverse
        // index needs pruning — no full rebuild.
        std::unordered_set<uint64_t> erased;
        for (uint64_t id : dead) {
            auto it = metadata_store_.find(id);
            if (it == metadata_store_.end() || !is_dead_meta(it->second)) continue;
            const std::string old_text = bm25_text(it->second);
            remove_from_bm25_index(id, &old_text);   // no-op unless add() of a dead meta slipped in
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

        ensure_open();
        auto_compact_ratio_ = ratio;
    }
    float get_auto_compact() const {
        std::shared_lock<RWMutex> lock(mutex_);

        ensure_open();
        return auto_compact_ratio_;
    }

    // True while a compaction is between its snapshot and its swap.
    bool is_compacting() const {
        std::shared_lock<RWMutex> lock(mutex_);

        ensure_open();
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

        ensure_open();
        const bool was = quantized_modalities_.count(modality) > 0;
        if (on) quantized_modalities_.insert(modality);
        else    quantized_modalities_.erase(modality);
        // Re-encode every stored vector, not just the ones the next commit
        // happens to touch: the point is a smaller file.
        if (was != on) vacuum_requested_ = true;
    }
    bool is_quantized(const std::string& modality) const {
        std::shared_lock<RWMutex> lock(mutex_);

        ensure_open();
        return quantized_modalities_.count(modality) > 0;
    }

    // Store this modality's vectors as int8[dim] IN MEMORY (~4x less RAM)
    // under one global scale = clip / 127.
    //
    // max_abs <= 0 (default): the clip is TRAINED from the data — the value
    // minimising mean squared quantization error over a sample of components
    // (train_int8_clip). A modality that already holds INT8_TRAIN_MIN live
    // vectors (FEATHER_INT8_TRAIN_MIN, default 1000) is converted now; one
    // with fewer stays float32 until it does, then converts on that write.
    // max_abs > 0: a fixed clip; components beyond it are clamped.
    //
    // refine > 0 (default 2) keeps every vector's float32 original in the
    // file — read back from disk, never held in RAM once saved — and search
    // re-ranks its top k x refine int8 candidates by exact distance. 0 keeps
    // no originals: smallest file, int8 recall.
    //
    // Converting a non-empty float modality keeps its graph and re-encodes
    // only the vectors. Its originals stay in RAM until the next save().
    // An int8 modality's scale is fixed: calling again only changes refine.
    void set_int8_ram(const std::string& modality, float max_abs = 0.0f,
                      uint32_t refine = DEFAULT_INT8_REFINE) {
        std::unique_lock<RWMutex> lock(mutex_);
        ensure_open();
        if (refine > 255)
            throw std::invalid_argument("set_int8_ram: refine must be 0-255");
        if (!std::isfinite(max_abs))
            throw std::invalid_argument("set_int8_ram: max_abs must be finite");
        auto it = modality_indices_.find(modality);
        const bool exists = it != modality_indices_.end();
        const bool empty  = !exists || it->second.index->getCurrentElementCount() == 0;
        // A change to a modality the file already describes: rewrite it at
        // the next save so the MODALITY section carries the new settings.
        if (exists && it->second.persisted) vacuum_requested_ = true;
        if (refine) int8_refine_[modality] = refine; else int8_refine_.erase(modality);

        if (exists && it->second.int8 && !empty) {
            auto& m = it->second;
            if (max_abs > 0.0f && std::fabs(max_abs / 127.0f - m.scale) > 1e-6f * m.scale)
                throw std::runtime_error(
                    "set_int8_ram('" + modality + "'): already int8 with clip " +
                    std::to_string(m.scale * 127.0f) + "; its scale cannot change");
            m.refine = refine;
            if (!refine) {               // drop the originals; a rewrite drops them from the file
                decltype(m.fv_pending)().swap(m.fv_pending);
                decltype(m.fv_off)().swap(m.fv_off);
            }
            return;
        }
        if (max_abs > 0.0f) {
            const float scale = max_abs / 127.0f;
            int8_train_.erase(modality);
            if (!empty) { convert_to_int8_nolock(modality, scale); return; }
            int8_ram_scale_[modality] = scale;
            // An existing (empty) index for this modality is rebuilt as int8 in place.
            if (exists) create_index(modality, it->second.dim, it->second.index->M_,
                                     it->second.index->ef_construction_);
            return;
        }
        // Trained. An empty int8 index goes back to float until there is data.
        int8_train_.insert(modality);
        if (int8_ram_scale_.erase(modality) && exists)
            create_index(modality, it->second.dim, it->second.index->M_,
                         it->second.index->ef_construction_);
        maybe_train_int8_nolock(modality);
    }
    bool is_int8_ram(const std::string& modality) const {
        std::shared_lock<RWMutex> lock(mutex_);

        ensure_open();
        return int8_ram_scale_.count(modality) > 0;
    }

    struct Int8Info {
        bool     int8;              // the index stores int8 now
        bool     training;          // waiting for INT8_TRAIN_MIN vectors to train on
        float    clip;              // scale x 127 (0 when not int8)
        uint32_t refine;            // re-rank factor; > 0 keeps float originals
        uint64_t originals_on_disk; // slots whose original is in the file
        uint64_t originals_in_wal;  // ...still only in the WAL record that logged it
        uint64_t originals_in_ram;  // ...copied in RAM (a conversion), until save()
        uint32_t train_min;         // vectors a trained modality waits for
    };
    Int8Info int8_info(const std::string& modality) const {
        std::shared_lock<RWMutex> lock(mutex_);
        ensure_open();
        Int8Info info{false, int8_train_.count(modality) > 0, 0.0f, 0, 0, 0, 0,
                      static_cast<uint32_t>(int8_train_min())};
        auto r = int8_refine_.find(modality);
        if (r != int8_refine_.end()) info.refine = r->second;
        auto sc = int8_ram_scale_.find(modality);
        if (sc != int8_ram_scale_.end()) { info.int8 = true; info.clip = sc->second * 127.0f; }
        auto it = modality_indices_.find(modality);
        if (it != modality_indices_.end() && it->second.int8) {
            const auto& m = it->second;
            info.refine = m.refine;
            for (size_t i = 0; i < m.fv_off.size(); ++i)
                if (m.fv_off[i] && !m.fv_pending.count(static_cast<uint32_t>(i))) ++info.originals_on_disk;
            for (const auto& [slot, p] : m.fv_pending) ++(p.wal ? info.originals_in_wal : info.originals_in_ram);
        }
        return info;
    }

    // ─────────────────────────────────────────────────────────────────
    // Persistence & info
    // ─────────────────────────────────────────────────────────────────
    // Checkpoint: one v10 commit. Under the SHARED lock it appends only what
    // changed since the last commit (queries keep running; writers wait for
    // that much and no more) and rotates the WAL to <wal>.old. The fsync of
    // the new sections and the header switch that publishes them happen after
    // the lock is released. Before v10 every save rewrote the whole file under
    // the lock — 182 MB at 100k x 384, with writers paused up to 101 ms.
    //
    // A save rewrites the whole file instead (to <path>.tmp, then rename) when
    // the file is still v3-v9, after purge()/compact()/set_quantized(), or once
    // superseded records outweigh live data (FEATHER_VACUUM_RATIO).
    void save() { checkpoint(/*force_full=*/false); }

    // Rewrite the file now, dropping every superseded record — including the
    // earlier versions of updated or forgotten records, which an append-only
    // file keeps until it is vacuumed.
    void vacuum() { checkpoint(/*force_full=*/true); }

    struct StorageStats {
        uint32_t format;              // on-disk format of the file (0: none yet)
        uint64_t generation;          // v10 commit generation
        uint64_t file_bytes;          // committed bytes
        uint64_t live_bytes_estimate; // what a vacuum would leave
        uint64_t last_commit_bytes;   // bytes the last save wrote
        bool     last_commit_full;    // ...as a whole-file rewrite
        uint64_t commits;             // saves that wrote something, this handle
        uint64_t vacuums;             // of which whole-file rewrites
        uint64_t mapped_bytes;        // vector bytes this process maps from the file
    };
    StorageStats storage_stats() const {
        std::lock_guard<std::mutex> save_lock(save_mutex_);
        std::shared_lock<RWMutex> lock(mutex_);
        ensure_open();
        uint64_t mapped = 0;
        for (const auto& [name, m] : modality_indices_)
            if (m.mapped) mapped += static_cast<uint64_t>(m.image_count) * m.index->data_size_;
        return {base_version_, generation_,
                file_v10_ ? committed_end_ : fsio::size_or_zero(path_),
                live_bytes_estimate(), last_commit_bytes_, last_commit_full_,
                commits_, vacuums_, mapped};
    }

private:
    void checkpoint(bool force_full) {
        check_writable("save");
        std::lock_guard<std::mutex> save_lock(save_mutex_);
        if (legacy_format_requested()) { save_legacy(); return; }
        Commit c;
        bool rotated = false;
        {
            LongSharedLock lock(mutex_);   // long: queries keep flowing
            ensure_open();
            c = commit_begin(force_full);
            try { rotated = wal_rotate(c.skip ? generation_ : c.gen); }
            catch (...) { commit_failed_ = true; throw; }   // dirty sets already consumed
            if (!rotated) {
                // A previous checkpoint left <wal>.old behind (it failed after
                // rotating). Finish synchronously so the two WALs never need
                // merging: publish, then drop both.
                commit_finish(c);
                wal_clear();
            }
        }
        if (rotated) commit_finish(c);               // fsync + header switch
        {
            // Still under save_mutex_: nothing that renumbers slots (compact)
            // can run between the commit and the publication of its offsets.
            std::unique_lock<RWMutex> lock(mutex_);
            if (!closed_) commit_install(c);
        }
        if (rotated) std::remove(wal_old_path().c_str());   // its records are in the file now
    }

    // FEATHER_FORMAT=9: the pre-v10 whole-file snapshot.
    void save_legacy() {
        const std::string tmp = path_ + ".tmp";
        {
            LongSharedLock lock(mutex_);
            ensure_open();
            write_snapshot(tmp);
            take_salience();
            commit_clear_dirty(/*full=*/true);
            if (!wal_rotate(generation_)) {
                publish_legacy(tmp);
                wal_clear();
                return;
            }
        }
        publish_legacy(tmp);
        std::remove(wal_old_path().c_str());
    }

public:
    // Close the DB: checkpoint (unless save=false), release the WAL and the
    // inter-process file lock. Further calls on this handle throw. Required
    // rather than decorative: Python never runs ~DB() (the handle is held with
    // py::nodelete), so without close() — or a `with DB.open(...) as db:`
    // block — the lock lives until process exit. Idempotent. A read-only
    // handle holds only a SHARED lock and never rewrites the file.
    void close(bool do_save = true) {
        stop_compactor();                            // never join while holding mutex_
        stop_ns_builder();
        std::lock_guard<std::mutex> save_lock(save_mutex_);
        std::unique_lock<RWMutex> lock(mutex_);
        if (closed_) return;
        // A poisoned handle is released without a checkpoint; the WAL stays
        // for the reopen. Not an error here: the write that failed already
        // raised, and close() runs from __exit__ while that is propagating.
        if (do_save && load_complete_ && !read_only_ && !poisoned_.load()) save_vectors();
        close_data_fh();
        fv_file_.close();
        wal_close();
        release_lock();
        closed_ = true;
        release_memory_nolock();
    }

private:
    // Drop everything a closed handle can no longer use: the graphs (or their
    // mappings), metadata, and every derived index. close() used to leave all
    // of it allocated, and since Python never destroyed the DB object
    // (py::nodelete) the memory lived until process exit — measured ~50 MB per
    // open/close of a 39 MB file, so the API and MCP servers grew without
    // bound. swap() with empties, not clear(): clear() keeps bucket arrays.
    void release_memory_nolock() {
        decltype(modality_indices_)().swap(modality_indices_);
        decltype(metadata_store_)().swap(metadata_store_);
        decltype(reverse_index_)().swap(reverse_index_);
        decltype(ns_index_)().swap(ns_index_);
        decltype(entity_index_)().swap(entity_index_);
        decltype(attr_index_)().swap(attr_index_);
        decltype(key_index_)().swap(key_index_);
        decltype(ns_names_)().swap(ns_names_);
        decltype(time_index_)().swap(time_index_);
        decltype(valid_index_)().swap(valid_index_);
        decltype(ns_graphs_)().swap(ns_graphs_);
        decltype(bm25_index_)().swap(bm25_index_);
        decltype(doc_lengths_)().swap(doc_lengths_);
        decltype(bm25_doc_id_)().swap(bm25_doc_id_);
        decltype(bm25_doc_len_)().swap(bm25_doc_len_);
        decltype(dirty_meta_)().swap(dirty_meta_);
        decltype(idem_)().swap(idem_);
        decltype(idem_order_)().swap(idem_order_);
        decltype(dirty_idem_)().swap(dirty_idem_);
        decltype(compaction_log_)().swap(compaction_log_);
        {
            std::lock_guard<std::mutex> g(salience_mx_);
            decltype(salience_dirty_)().swap(salience_dirty_);
        }
        total_dl_ = avg_dl_ = 0.0;
    }

public:

    ~DB() {
        stop_compactor();
        stop_ns_builder();
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
        if (!closed_ && load_complete_ && !read_only_ && !poisoned_.load()) {
            try { save_vectors(); } catch (...) {}
        }
        close_data_fh();
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

        ensure_open();
        auto it = modality_indices_.find(modality);
        if (it != modality_indices_.end()) return it->second.dim;
        return default_dim_;   // modality not created yet → report the open() default
    }

    size_t size() const {
        std::shared_lock<RWMutex> lock(mutex_);

        ensure_open();
        return metadata_store_.size();
    }

    // ─────────────────────────────────────────────────────────────────
    // Tuning: HNSW search beam width
    // ─────────────────────────────────────────────────────────────────
    // Higher ef = better recall, slower search. Default is DEFAULT_EF (50).
    // Pass modality = "" (default) to apply to all modalities.
    void set_ef(size_t ef, const std::string& modality = "") {
        std::unique_lock<RWMutex> lock(mutex_);

        ensure_open();
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

        ensure_open();
        auto it = modality_indices_.find(modality);
        if (it == modality_indices_.end())
            throw std::runtime_error("unknown modality: " + modality);
        return it->second.index->ef_;
    }
};

} // namespace feather
