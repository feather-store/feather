#pragma once
#include <atomic>
#include <cmath>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace feather {

enum class ContextType : uint8_t {
    FACT = 0,
    PREFERENCE = 1,
    EVENT = 2,
    CONVERSATION = 3
};

// Phase 5: Typed, weighted graph edge
struct Edge {
    uint64_t    target_id;
    std::string rel_type;   // "related_to", "derived_from", "caused_by", etc.
    float       weight;     // relationship strength [0.0–1.0]

    Edge() : target_id(0), rel_type("related_to"), weight(1.0f) {}
    Edge(uint64_t t, const std::string& r, float w)
        : target_id(t), rel_type(r), weight(w) {}
};

// recall_count is a moving count of recent recalls, not a lifetime total. Each
// recall first fades the count so far by the time since the previous recall,
// then adds one:  n <- round(n * 0.5^(gap / RECALL_HALF_LIFE_DAYS)) + 1.
// Scorer::effective_recalls then fades it from last_recalled_at to now. The
// earlier rule kept a lifetime count and faded all of it from the LAST recall,
// as if every recall had happened then: 100 recalls a year ago plus one
// yesterday counted ~99. Now that is 1, which is right (exact: 1.02).
//
// Rounding loses at most half a recall per gap. Over short gaps the count
// still goes up by exactly one per recall: a gap fades nothing until
// n * (1 - 0.5^(gap/30d)) reaches 0.5, which is n ~22 for daily gaps and
// n ~1.9M for one-second gaps. Under regular use it levels off
// near half the exact moving count. Daily recalls for a year give 22 against
// an exact 44 (the old rule gave 365).
//
// Fixed, not per query: a moving count can only fade at one rate. History
// before the last recall fades at 30 days whatever half-life a query or a
// Pocket uses; the time since the last recall fades at the caller's.
constexpr double RECALL_HALF_LIFE_DAYS = 30.0;

// note_recall() reads and writes both counters, and runs under the DB's SHARED
// lock, so two queries can recall one record at once. A stripe keeps the pair
// consistent without a mutex in every record.
inline std::mutex& recall_stripe(const void* p) {
    static std::mutex stripes[64];
    return stripes[(reinterpret_cast<uintptr_t>(p) * 0x9E3779B97F4A7C15ull) >> 58];
}

struct Metadata {
    int64_t timestamp;
    float importance;
    ContextType type;
    std::string source;
    std::string content;
    std::string tags_json;

    // Phase 3: Salience.
    // `mutable` + atomic: recording a retrieval is the ONLY mutation a query
    // performs, and queries run under the DB's shared (read) lock so they can
    // execute concurrently. Writers are excluded while any query is in flight,
    // so no map rehash can race these; the atomics make the concurrent
    // increment itself well-defined. Relaxed ordering is sufficient — these are
    // independent counters, not synchronisation for other state.
    mutable std::atomic<uint32_t> recall_count;
    mutable std::atomic<uint64_t> last_recalled_at;
    // Counters changed since the last commit (file format v10). Not copied by
    // assign_from(): it describes this stored object, not the value.
    mutable std::atomic<bool> recall_unsaved{false};

    // Convenience for the (many) places that just want the value.
    uint32_t recalls() const { return recall_count.load(std::memory_order_relaxed); }
    uint64_t recalled_at() const { return last_recalled_at.load(std::memory_order_relaxed); }
    // Both counters as one value, for anything that persists them: a recall
    // landing between two plain loads would save a count with the previous
    // recall's time, and the count would then fade from the wrong moment.
    std::pair<uint32_t, uint64_t> recall_state() const {
        std::lock_guard<std::mutex> g(recall_stripe(this));
        return {recalls(), recalled_at()};
    }

    // Record one retrieval. Safe to call under a shared lock. Returns true for
    // the first retrieval since the counters were last committed, so the DB
    // queues this record for the next commit exactly once.
    bool note_recall(uint64_t now) const {
        {
            std::lock_guard<std::mutex> g(recall_stripe(this));
            const uint32_t n = recalls();
            uint32_t next = 1;
            if (n) {
                // A count with no recall time (copied by hand) fades from the
                // record's own timestamp, as in effective_recalls().
                const uint64_t at = recalled_at();
                const uint64_t from = at ? at : static_cast<uint64_t>(
                                                    timestamp > 0 ? timestamp : 0);
                const double gap_days = now > from ? (now - from) / 86400.0 : 0.0;
                const double kept = std::floor(
                    n * std::pow(0.5, gap_days / RECALL_HALF_LIFE_DAYS) + 0.5);
                next = kept >= 4294967295.0 ? UINT32_MAX
                                            : static_cast<uint32_t>(kept) + 1;
            }
            recall_count.store(next, std::memory_order_relaxed);
            last_recalled_at.store(now, std::memory_order_relaxed);
        }
        // acq_rel pairs with clear_recall_unsaved(): a commit that clears the
        // flag after this exchange also sees the increment above, and one that
        // cleared it before makes this call return true and queue it again.
        return !recall_unsaved.exchange(true, std::memory_order_acq_rel);
    }
    void clear_recall_unsaved() const {
        recall_unsaved.exchange(false, std::memory_order_acq_rel);
    }

    // Phase 4: Namespace + Entity + Attributes
    std::string namespace_id;
    std::string entity_id;
    std::unordered_map<std::string, std::string> attributes;

    // Phase 5: Typed, weighted context graph edges (replaces plain `links`)
    std::vector<Edge> edges;

    // Phase 6: Working memory + epistemic confidence
    int64_t ttl;          // seconds-to-live from timestamp; 0 = never expires
    float   confidence;   // certainty about this fact [0.0–1.0]; default 1.0

    // Compare-and-set version, assigned by the DB on every change to the
    // record and never reused within a file (one clock per DB). 0 = never
    // written through a DB. NOT part of serialize(): the v9 record stream has
    // no length prefix, so a new trailing field would break 0.21 readers of a
    // FEATHER_FORMAT=9 file. v10 keeps it in the VERSIONS section, the WAL in
    // STAMP / COMMIT records.
    uint64_t version = 0;

    // Native key: unique among the live records of its namespace_id, so a
    // record is addressed by (namespace_id, key) instead of an id derived
    // from a hash of it (a collision silently overwrote another record).
    // Serialized as the reserved attribute KEY_ATTR, not a new field: every
    // format and the WAL carry attributes already, and a build that predates
    // keys keeps it as an ordinary attribute and writes it back intact.
    std::string key;
    static const std::string& key_attr() {
        static const std::string k("\0key", 4);
        return k;
    }

    // Valid time: when what the record says happened or became true, as
    // opposed to `timestamp`, when it was recorded (transaction time). A
    // session on 2023/05/14 saying "last Saturday I went to a wedding" is
    // recorded 05/14 and valid 05/06. 0 = not given: valid_time() is then
    // the timestamp. Serialized as the reserved attribute VALID_ATTR
    // (decimal), for the reasons `key` is.
    int64_t valid_at = 0;
    static const std::string& valid_attr() {
        static const std::string k("\0valid_at", 9);
        return k;
    }
    int64_t valid_time() const { return valid_at ? valid_at : timestamp; }

    Metadata() : timestamp(0), importance(1.0f), type(ContextType::FACT),
                 recall_count(0), last_recalled_at(0),
                 ttl(0), confidence(1.0f) {}

    // std::atomic is neither copyable nor movable, so the compiler-generated
    // copy/move for Metadata is deleted — these restore it by transferring the
    // counters' values. Everything else moves/copies as usual.
    Metadata(const Metadata& o) { assign_from(o); }
    Metadata& operator=(const Metadata& o) {
        if (this != &o) assign_from(o);
        return *this;
    }
    Metadata(Metadata&& o) noexcept { assign_from(std::move(o)); }
    Metadata& operator=(Metadata&& o) noexcept {
        if (this != &o) assign_from(std::move(o));
        return *this;
    }

    void serialize(std::ostream& os) const;
    static Metadata deserialize(std::istream& is);

private:
    void assign_from(const Metadata& o) {
        timestamp = o.timestamp; importance = o.importance; type = o.type;
        source = o.source; content = o.content; tags_json = o.tags_json;
        namespace_id = o.namespace_id; entity_id = o.entity_id;
        attributes = o.attributes; edges = o.edges;
        ttl = o.ttl; confidence = o.confidence; version = o.version; key = o.key;
        valid_at = o.valid_at;
        recall_count.store(o.recalls(), std::memory_order_relaxed);
        last_recalled_at.store(o.recalled_at(), std::memory_order_relaxed);
    }
    void assign_from(Metadata&& o) {
        timestamp = o.timestamp; importance = o.importance; type = o.type;
        source = std::move(o.source); content = std::move(o.content);
        tags_json = std::move(o.tags_json);
        namespace_id = std::move(o.namespace_id); entity_id = std::move(o.entity_id);
        attributes = std::move(o.attributes); edges = std::move(o.edges);
        ttl = o.ttl; confidence = o.confidence; version = o.version; key = std::move(o.key);
        valid_at = o.valid_at;
        recall_count.store(o.recalls(), std::memory_order_relaxed);
        last_recalled_at.store(o.recalled_at(), std::memory_order_relaxed);
    }
};

struct ContextRecord {
    uint64_t id;
    Metadata metadata;
    // The vector is still managed by DB class/index
};

} // namespace feather
