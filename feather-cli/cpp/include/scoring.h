#pragma once
#include "metadata.h"
#include <cmath>
#include <ctime>

namespace feather {

struct ScoringConfig {
    float decay_half_life_days;
    float time_weight;
    float min_weight;

    ScoringConfig(float half_life = 30.0f, float weight = 0.3f, float min = 0.0f)
        : decay_half_life_days(half_life), time_weight(weight), min_weight(min) {}
};

class Scorer {
public:
    static float calculate_score(float distance, const Metadata& meta, const ScoringConfig& config, double now_ts) {
        // Convert distance to similarity (0-1)
        // For L2 distance, similarity = 1 / (1 + distance)
        float similarity = 1.0f / (1.0f + distance);

        // Temporal decay (exponential)
        double age_seconds = now_ts - static_cast<double>(meta.timestamp);
        if (age_seconds < 0) age_seconds = 0;
        
        double age_days = age_seconds / 86400.0;
        float recency = std::pow(0.5f, static_cast<float>(age_days / config.decay_half_life_days));
        
        // Apply min_weight floor if needed
        if (recency < config.min_weight) recency = config.min_weight;

        // Combined score
        float final_score = ((1.0f - config.time_weight) * similarity + config.time_weight * recency) * meta.importance;
        
        return final_score;
    }
};

} // namespace feather
