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

        // Temporal decay (exponential) with Adaptive "Stickiness"
        // "Core Memories" (high recall_count) decay slower.
        
        double age_seconds = now_ts - static_cast<double>(meta.timestamp);
        if (age_seconds < 0) age_seconds = 0;
        
        // Stickiness factor: 1.0 (default) -> grows with recall_count
        // e.g., recall=0 -> factor=1.0
        //       recall=10 -> factor=3.4
        //       recall=100 -> factor=5.6
        float stickiness = 1.0f + std::log(1.0f + meta.recall_count);
        
        double age_days = age_seconds / 86400.0;
        
        // Effective Age is reduced by stickiness
        float effective_age_days = static_cast<float>(age_days) / stickiness;
        
        float recency = std::pow(0.5f, effective_age_days / config.decay_half_life_days);
        
        // Apply min_weight floor if needed
        if (recency < config.min_weight) recency = config.min_weight;

        // Combined score
        float final_score = ((1.0f - config.time_weight) * similarity + config.time_weight * recency) * meta.importance;
        
        return final_score;
    }
};

} // namespace feather
