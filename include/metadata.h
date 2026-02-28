#pragma once
#include <string>
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

    // Phase 3: Salience
    uint32_t recall_count;
    uint64_t last_recalled_at;

    // Phase 4: Namespace + Entity + Attributes
    std::string namespace_id;
    std::string entity_id;
    std::unordered_map<std::string, std::string> attributes;

    // Phase 5: Typed, weighted context graph edges (replaces plain `links`)
    std::vector<Edge> edges;

    Metadata() : timestamp(0), importance(1.0f), type(ContextType::FACT),
                 recall_count(0), last_recalled_at(0) {}

    void serialize(std::ostream& os) const;
    static Metadata deserialize(std::istream& is);
};

struct ContextRecord {
    uint64_t id;
    Metadata metadata;
    // The vector is still managed by DB class/index
};

} // namespace feather
