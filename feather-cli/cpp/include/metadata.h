#pragma once
#include <atomic>
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

    // Convenience for the (many) places that just want the value.
    uint32_t recalls() const { return recall_count.load(std::memory_order_relaxed); }
    uint64_t recalled_at() const { return last_recalled_at.load(std::memory_order_relaxed); }

    // Record one retrieval. Safe to call under a shared lock.
    void note_recall(uint64_t now) const {
        recall_count.fetch_add(1, std::memory_order_relaxed);
        last_recalled_at.store(now, std::memory_order_relaxed);
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
        ttl = o.ttl; confidence = o.confidence;
        recall_count.store(o.recalls(), std::memory_order_relaxed);
        last_recalled_at.store(o.recalled_at(), std::memory_order_relaxed);
    }
    void assign_from(Metadata&& o) {
        timestamp = o.timestamp; importance = o.importance; type = o.type;
        source = std::move(o.source); content = std::move(o.content);
        tags_json = std::move(o.tags_json);
        namespace_id = std::move(o.namespace_id); entity_id = std::move(o.entity_id);
        attributes = std::move(o.attributes); edges = std::move(o.edges);
        ttl = o.ttl; confidence = o.confidence;
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
