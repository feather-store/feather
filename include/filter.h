#pragma once
#include "metadata.h"
#include <vector>
#include <string>
#include <optional>
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

    bool matches(const Metadata& meta) const {
        if (types) {
            bool found = false;
            for (auto t : *types) {
                if (meta.type == t) {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }

        if (source && meta.source != *source) return false;
        if (source_prefix && meta.source.find(*source_prefix) != 0) return false;
        if (timestamp_after && meta.timestamp < *timestamp_after) return false;
        if (timestamp_before && meta.timestamp > *timestamp_before) return false;
        if (importance_gte && meta.importance < *importance_gte) return false;

        if (tags_contains) {
            // This is a simple check, assuming metadata contains the tags in some form.
            // For now, let's just check if the tags_json contains the strings.
            // In a real implementation, we might want to parse the JSON.
            for (const auto& tag : *tags_contains) {
                if (meta.tags_json.find(tag) == std::string::npos) return false;
            }
        }

        return true;
    }
};

} // namespace feather
