#pragma once
#include "metadata.h"
#include <vector>
#include <string>
#include <optional>
#include <unordered_map>
#include <algorithm>

namespace feather {

struct SearchFilter {
    std::optional<std::vector<ContextType>> types;
    std::optional<std::string> source;
    std::optional<std::string> source_prefix;
    std::optional<int64_t> timestamp_after;
    std::optional<int64_t> timestamp_before;
    std::optional<float> importance_gte;
    std::optional<std::vector<std::string>> tags_contains;

    // Phase 4 filters: namespace, entity, attributes
    std::optional<std::string> namespace_id;
    std::optional<std::string> entity_id;
    std::optional<std::unordered_map<std::string, std::string>> attributes_match;

    bool matches(const Metadata& meta) const {
        if (types) {
            bool found = false;
            for (auto t : *types) {
                if (meta.type == t) { found = true; break; }
            }
            if (!found) return false;
        }

        if (source && meta.source != *source) return false;
        if (source_prefix && meta.source.find(*source_prefix) != 0) return false;
        if (timestamp_after && meta.timestamp < *timestamp_after) return false;
        if (timestamp_before && meta.timestamp > *timestamp_before) return false;
        if (importance_gte && meta.importance < *importance_gte) return false;

        if (tags_contains) {
            for (const auto& tag : *tags_contains) {
                if (meta.tags_json.find(tag) == std::string::npos) return false;
            }
        }

        if (namespace_id && meta.namespace_id != *namespace_id) return false;
        if (entity_id && meta.entity_id != *entity_id) return false;

        if (attributes_match) {
            for (const auto& [key, val] : *attributes_match) {
                auto it = meta.attributes.find(key);
                if (it == meta.attributes.end() || it->second != val) return false;
            }
        }

        return true;
    }
};

} // namespace feather
