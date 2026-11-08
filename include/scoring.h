#pragma once
#include "metadata.h"
#include <cmath>
#include <ctime>

namespace feather {

struct ScoringConfig {
    float decay_half_life_days;
    float time_weight;
    float min_weight;
    // The cosine at which a record counts as unrelated to the query; relevance
    // is measured from it (Scorer::relevance). < 0, the default: the
    // modality's background cosine, estimated from its own vectors
    // (DB::background_cosine). 0: plain cosine. In (0, 1): that value.
    float relevance_floor;

    ScoringConfig(float half_life = 30.0f, float weight = 0.3f, float min = 0.0f,
                  float floor = -1.0f)
        : decay_half_life_days(half_life), time_weight(weight), min_weight(min),
          relevance_floor(floor) {}
};

class Scorer {
public:
    // A record's recall count as of now: n * 0.5^(days since last recall / H),
    // H = RECALL_HALF_LIFE_DAYS. A recall is evidence the record mattered
    // THEN. Counted raw, ln(1+n) only ever grew, so a record recalled 100
    // times a year ago stayed ~1000x ahead of an unused one of the same age
    // for good. n is a moving count as of its last recall (note_recall fades
    // the history at each recall at the same H), so this fades the rest.
    //
    // One half-life for salience everywhere, not the caller's: scored search
    // (30 d by default), Pocket.heat (7 d) and context_chain used to credit the
    // same record's use differently. Their RECENCY half-lives still differ on
    // purpose; how much a record has been used does not depend on who asks.
    // A count carried without a recall time (copied by hand) fades from the
    // record's own timestamp. Pocket's `effective_recalls` mirrors this.
    static double effective_recalls(const Metadata& meta, double now_ts) {
        const uint32_t n = meta.recalls();
        if (n == 0) return 0.0;
        constexpr double half_life_days = RECALL_HALF_LIFE_DAYS;
        const uint64_t at = meta.recalled_at();
        double since = now_ts - static_cast<double>(at ? at : static_cast<uint64_t>(
                                   meta.timestamp > 0 ? meta.timestamp : 0));
        if (since < 0) since = 0;
        return static_cast<double>(n) * std::pow(0.5, since / 86400.0 / half_life_days);
    }

    // How related a record is, from its cosine: (cos - floor) / (1 - floor).
    //
    // Raw cosine is not on a scale freshness can multiply. An embedding model
    // gives unrelated text a cosine well above 0 (gemini-embedding-001: random
    // pairs ~0.51, the evidence for a question 0.6-0.8), so "freshness costs at
    // most w of the score" let a fresh unrelated record overtake an old
    // relevant one: on LongMemEval knowledge-update (vector search, w=0.3) both
    // evidence sessions left the top 5 in 17 of 62 questions. Measured from
    // the model's background cosine, none did (62/62, as with scoring off), and
    // the newest evidence still ranked first in 40 (23 off). The floor is
    // estimated per modality from its own vectors, so no model needs a constant.
    static float relevance(float cosine, float floor) {
        if (!(floor > 0.0f)) return cosine;
        return (cosine - floor) / (1.0f - floor);
    }

    // score = relevance x freshness x importance, freshness = (1-w) + w*recency.
    //
    // Recency SCALES relevance; it is not added to it. The old score was
    // (1-w)*1/(1+L2^2) + w*recency. 1/(1+L2^2) is 1/(3-2cos) for unit vectors,
    // a compressed similarity, so at the defaults a record at cos 0.5 written
    // today tied one at cos 0.8 from 30 days ago (0.65 each); for embeddings
    // that are not unit length it is ~0 and recency ranked alone. Now freshness
    // costs at most w of a record's relevance: at w=0.3 a record beats every
    // record less than 0.7 as relevant, however new.
    //
    // A record at or below the floor is unrelated to the query: it keeps its
    // (non-positive) relevance, so no recency or importance lifts it above a
    // related one. `floor` is the resolved relevance floor (0 = plain cosine).
    static float calculate_score(float cosine, const Metadata& meta,
                                 const ScoringConfig& config, double now_ts,
                                 float floor = 0.0f) {
        const float rel = relevance(cosine, floor);
        if (!(rel > 0.0f)) return rel;

        // Age from when it happened or became true (valid_at), not when it was
        // written: a 2023 fact imported today is not fresh. valid_time() is
        // the timestamp when no valid time was given.
        double age_days = (now_ts - static_cast<double>(meta.valid_time())) / 86400.0;
        if (age_days < 0) age_days = 0;
        // "Core memories" decay slower: recall=0 -> 1.0, 10 -> 3.4, 100 -> 5.6,
        // counted as of now (effective_recalls), not as of their peak.
        const double stickiness = 1.0 + std::log(1.0 + effective_recalls(meta, now_ts));
        double recency = std::pow(0.5, age_days / stickiness / config.decay_half_life_days);
        if (recency < config.min_weight) recency = config.min_weight;

        const double w = config.time_weight;
        const double freshness = (1.0 - w) + w * recency;
        return static_cast<float>(rel * freshness * meta.importance);
    }

    // Cosine from a squared L2 distance and the two squared norms:
    // |q-x|^2 = |q|^2 + |x|^2 - 2 q.x. Exact for float vectors of any length.
    static float cosine_from_l2(float dist2, double qq, double xx) {
        if (!(qq > 0.0) || !(xx > 0.0)) return 0.0f;
        const double c = (qq + xx - static_cast<double>(dist2)) / (2.0 * std::sqrt(qq * xx));
        return static_cast<float>(c > 1.0 ? 1.0 : (c < -1.0 ? -1.0 : c));
    }
};

} // namespace feather
