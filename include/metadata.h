#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace feather {

enum class ContextType : uint8_t {
    FACT = 0,
    PREFERENCE = 1,
    EVENT = 2,
    CONVERSATION = 3
};

struct Metadata {
    int64_t timestamp;
    float importance;
    ContextType type;
    std::string source;
    std::string content;
    std::string tags_json; // JSON array of tags
    
    // Phase 3 Features: Graph & Salience
    std::vector<uint64_t> links;      // Record IDs this record points to
    uint32_t recall_count;            // How many times this was returned in search
    uint64_t last_recalled_at;        // Timestamp of last recall

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
