#pragma once
// Pocket.hot() in the engine: the working set of an agent's scopes, chosen
// by heat under a token budget, without building a Python object per record.
//
// feather_db/pocket.py is the specification. Pocket.heat(), display_text()
// and the shadowing and budget loop of Pocket.hot() are mirrored here
// operation for operation, so the two agree to the bit: same double
// arithmetic, same libm calls, same evaluation order, no fused multiply-add
// (an FMA rounds once where Python rounds twice, and a 1-ulp difference can
// swap two records of near-equal heat). tests/test_hot_select.py compares
// them on random states. A change to either is a change to both.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "metadata.h"

#if defined(__GNUC__) && !defined(__clang__)
#define FEATHER_NO_FMA __attribute__((optimize("fp-contract=off")))
#else
#define FEATHER_NO_FMA
#endif

namespace feather {

struct HeatQuery {
    std::vector<std::string> namespaces;   // deepest first: the index is the depth
    double now = 0.0;
    double half_life_days = 7.0;           // Pocket.half_life
    double usage_half_life = 3.0;          // USAGE_HALF_LIFE
    double unused_floor = 0.4;             // UNUSED_FLOOR
    double inherit_decay = 0.7;            // INHERIT_DECAY
    double min_heat = 0.01;
    double budget = 4000.0;
    int64_t chars_per_token = 4;
    std::string pinned_attr = "_pocket_pinned";
    std::string due_attr = "_pocket_due_at";
    std::string origin_attr = "_origin";
    std::string text_attr = "_lg_value";   // a JSON value whose "text" is shown
    std::vector<std::string> origins;      // the known origins
    std::string default_origin = "user";   // an unlabelled record's
    std::string unknown_origin = "external";
    std::unordered_set<std::string> trusted;
    std::unordered_set<std::string> exclude_sources;   // not memory (_NOT_MEMORY)
};

struct HeatPick {
    uint64_t id;
    uint32_t depth;
    double heat;
    int64_t tokens;
};

namespace heat {

// Python's len() of a str decoded from UTF-8: one per code point.
inline int64_t utf8_chars(const char* p, size_t n) {
    int64_t c = 0;
    for (size_t i = 0; i < n; ++i)
        c += (static_cast<unsigned char>(p[i]) & 0xC0) != 0x80;
    return c;
}

// A validating JSON reader that accepts exactly what CPython's json.loads
// accepts (its C scanner: strict strings, NaN/Infinity, last duplicate key
// wins) and measures one top-level string field in code points.
class Json {
public:
    Json(const std::string& s, const std::string& field) : s_(s), field_(field) {}

    // len(json.loads(s).get(field)) when that is a non-empty str, else -1:
    // malformed JSON, a top level that is not an object, a field that is
    // missing, empty or not a string.
    int64_t field_chars() {
        ws();
        if (i_ >= s_.size() || s_[i_] != '{') return -1;    // .get on a non-dict
        int64_t found = -1;
        if (!object(0, &found)) return -1;
        ws();
        if (i_ != s_.size()) return -1;            // "Extra data"
        return found > 0 ? found : -1;
    }

private:
    static constexpr int kMaxDepth = 512;
    const std::string& s_;
    const std::string& field_;
    size_t i_ = 0;

    void ws() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' ||
                                  s_[i_] == '\r'))
            ++i_;
    }
    bool lit(const char* w) {
        size_t n = std::strlen(w);
        if (s_.compare(i_, n, w) != 0) return false;
        i_ += n;
        return true;
    }
    static int hex(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }
    bool u4(size_t at, unsigned* out) const {
        if (at + 4 > s_.size()) return false;
        unsigned v = 0;
        for (size_t k = 0; k < 4; ++k) {
            int h = hex(s_[at + k]);
            if (h < 0) return false;
            v = v * 16 + static_cast<unsigned>(h);
        }
        *out = v;
        return true;
    }
    // At the opening quote. Leaves the decoded code point count in *chars
    // and the raw bytes between the quotes in [*b, *e).
    bool string(int64_t* chars, size_t* b = nullptr, size_t* e = nullptr) {
        ++i_;
        const size_t start = i_;
        int64_t n = 0;
        while (true) {
            if (i_ >= s_.size()) return false;                 // unterminated
            const unsigned char c = static_cast<unsigned char>(s_[i_]);
            if (c == '"') break;
            if (c < 0x20) return false;                        // strict: control character
            if (c != '\\') {
                n += (c & 0xC0) != 0x80;
                ++i_;
                continue;
            }
            if (i_ + 1 >= s_.size()) return false;
            const char esc = s_[i_ + 1];
            if (esc == 'u') {
                unsigned v;
                if (!u4(i_ + 2, &v)) return false;
                i_ += 6;
                // A high surrogate followed by an escaped low one is one character.
                if (v >= 0xD800 && v <= 0xDBFF && i_ + 1 < s_.size() && s_[i_] == '\\' &&
                    s_[i_ + 1] == 'u') {
                    unsigned w;
                    if (u4(i_ + 2, &w) && w >= 0xDC00 && w <= 0xDFFF) i_ += 6;
                }
                ++n;
                continue;
            }
            if (esc != '"' && esc != '\\' && esc != '/' && esc != 'b' && esc != 'f' &&
                esc != 'n' && esc != 'r' && esc != 't')
                return false;
            i_ += 2;
            ++n;
        }
        if (b) *b = start;
        if (e) *e = i_;
        ++i_;
        *chars = n;
        return true;
    }
    bool digits() {
        const size_t b = i_;
        while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
        return i_ > b;
    }
    bool number() {
        const size_t b = i_;
        if (i_ < s_.size() && s_[i_] == '-') ++i_;
        if (i_ >= s_.size()) { i_ = b; return false; }
        if (s_[i_] == '0') ++i_;
        else if (s_[i_] >= '1' && s_[i_] <= '9') digits();
        else { i_ = b; return false; }
        // A fraction or exponent is taken only when complete ("1." is 1 then ".").
        if (i_ + 1 < s_.size() && s_[i_] == '.' && s_[i_ + 1] >= '0' && s_[i_ + 1] <= '9') {
            ++i_;
            digits();
        }
        if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
            size_t j = i_ + 1;
            if (j < s_.size() && (s_[j] == '+' || s_[j] == '-')) ++j;
            if (j < s_.size() && s_[j] >= '0' && s_[j] <= '9') {
                i_ = j;
                digits();
            }
        }
        return true;
    }
    bool value(int depth) {
        if (depth > kMaxDepth || i_ >= s_.size()) return false;
        const char c = s_[i_];
        int64_t n;
        if (c == '"') return string(&n);
        if (c == '{') return object(depth + 1, nullptr);
        if (c == '[') return array(depth + 1);
        if (c == 'n') return lit("null");
        if (c == 't') return lit("true");
        if (c == 'f') return lit("false");
        if (number()) return true;
        if (c == 'N') return lit("NaN");
        if (c == 'I') return lit("Infinity");
        if (c == '-') return lit("-Infinity");
        return false;
    }
    bool array(int depth) {
        ++i_;
        ws();
        if (i_ < s_.size() && s_[i_] == ']') { ++i_; return true; }
        while (true) {
            ws();
            if (!value(depth)) return false;
            ws();
            if (i_ >= s_.size()) return false;
            if (s_[i_] == ']') { ++i_; return true; }
            if (s_[i_] != ',') return false;
            ++i_;
        }
    }
    // `found`: set to the chars of the last top-level `field_` (a string),
    // or -1 when its last value is not one.
    bool object(int depth, int64_t* found) {
        if (depth > kMaxDepth) return false;
        ++i_;
        ws();
        if (i_ < s_.size() && s_[i_] == '}') { ++i_; return true; }
        while (true) {
            ws();
            if (i_ >= s_.size() || s_[i_] != '"') return false;
            int64_t kn;
            size_t kb, ke;
            if (!string(&kn, &kb, &ke)) return false;
            // Compare the key as decoded only when it can be the field: a key
            // with escapes is decoded by comparing escape-free bytes (the
            // field names used here never need escapes).
            const bool is_field = found && ke - kb == field_.size() &&
                                  s_.compare(kb, ke - kb, field_) == 0;
            ws();
            if (i_ >= s_.size() || s_[i_] != ':') return false;
            ++i_;
            ws();
            if (is_field && i_ < s_.size() && s_[i_] == '"') {
                int64_t vn;
                if (!string(&vn)) return false;
                *found = vn;
            } else {
                if (is_field) *found = -1;
                if (!value(depth)) return false;
            }
            ws();
            if (i_ >= s_.size()) return false;
            if (s_[i_] == '}') { ++i_; return true; }
            if (s_[i_] != ',') return false;
            ++i_;
        }
    }
};

// display_text(meta) in code points: the value's text, else the content.
inline int64_t display_chars(const Metadata& m, const std::string& text_attr) {
    auto it = m.attributes.find(text_attr);
    if (it != m.attributes.end() && !it->second.empty()) {
        int64_t n = Json(it->second, "text").field_chars();
        if (n > 0) return n;
    }
    return utf8_chars(m.content.data(), m.content.size());
}

// due_at(meta): float(raw) into *out, false when it is unset or not a
// number. ("nan" is a number to Python: a record due at NaN is never due.)
inline bool due_of(const Metadata& m, const std::string& due_attr, double* out) {
    auto it = m.attributes.find(due_attr);
    if (it == m.attributes.end() || it->second.empty()) return false;
    const std::string& s = it->second;
    if (s.find_first_of("xX_(") != std::string::npos) return false;   // strtod-only forms
    const char* b = s.c_str();
    char* end = nullptr;
    const double v = std::strtod(b, &end);
    if (end == b) return false;
    while (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r') ++end;
    if (*end != '\0') return false;
    *out = v;
    return true;
}

inline const std::string* attr(const Metadata& m, const std::string& k) {
    auto it = m.attributes.find(k);
    return it == m.attributes.end() ? nullptr : &it->second;
}

// Pocket.heat(meta, now, depth). `inherit_weight` is pow(q.inherit_decay,
// depth), computed once per depth by the caller (the same bits).
FEATHER_NO_FMA inline double heat_of(const Metadata& m, const HeatQuery& q, bool trusted,
                                     double inherit_weight) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const std::string* pin = attr(m, q.pinned_attr);
    if (trusted && pin && *pin == "true") return 1.0;
    double due;
    if (trusted && due_of(m, q.due_attr, &due)) return q.now >= due ? 1.0 : 0.0;
    double age_days = (q.now - static_cast<double>(m.valid_time())) / 86400.0;
    age_days = age_days > 0.0 ? age_days : 0.0;
    double recalls = 0.0;
    if (trusted) {
        const auto [n, last] = m.recall_state();
        if (n) {
            const int64_t at = last ? static_cast<int64_t>(last)
                                    : (m.timestamp > 0 ? m.timestamp : 0);
            double since = q.now - static_cast<double>(at);
            since = since > 0.0 ? since : 0.0;
            const double since_days = since / 86400.0;
            recalls = static_cast<double>(n) * std::pow(0.5, since_days / RECALL_HALF_LIFE_DAYS);
        }
    }
    // No recalls: log(1) and pow(0.5, 0) are exactly 0 and 1, so stickiness
    // is 1 and use_weight the floor, to the bit, without the two calls.
    double stickiness = 1.0, use_weight;
    if (recalls == 0.0) {
        const double spread = (1.0 - q.unused_floor) * 0.0;
        use_weight = q.unused_floor + spread;
    } else {
        stickiness = 1.0 + std::log(1.0 + recalls);
        const double usage = 1.0 - std::pow(0.5, recalls / q.usage_half_life);
        const double spread = (1.0 - q.unused_floor) * usage;
        use_weight = q.unused_floor + spread;
    }
    const double recency = std::pow(0.5, age_days / stickiness / q.half_life_days);
    double imp = static_cast<double>(m.importance);
    imp = imp < 1.0 ? imp : 1.0;
    imp = imp > 0.0 ? imp : 0.0;
    const double a = recency * use_weight;
    const double b = a * inherit_weight;
    return b * imp;
}

}  // namespace heat
}  // namespace feather
