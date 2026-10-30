#include <cstdlib>
#include "../include/metadata.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace feather {

void Metadata::serialize(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&timestamp), 8);
    os.write(reinterpret_cast<const char*>(&importance), 4);
    uint8_t type_val = static_cast<uint8_t>(type);
    os.write(reinterpret_cast<const char*>(&type_val), 1);

    uint16_t source_len = static_cast<uint16_t>(source.size());
    os.write(reinterpret_cast<const char*>(&source_len), 2);
    os.write(source.data(), source_len);

    uint32_t content_len = static_cast<uint32_t>(content.size());
    os.write(reinterpret_cast<const char*>(&content_len), 4);
    os.write(content.data(), content_len);

    uint16_t tags_len = static_cast<uint16_t>(tags_json.size());
    os.write(reinterpret_cast<const char*>(&tags_len), 2);
    os.write(tags_json.data(), tags_len);

    // Phase 3: Write 0 links (legacy slot — edges field replaces this in v5)
    // We write links_count=0 so v3/v4 readers see no plain links and don't crash.
    uint16_t links_count = 0;
    os.write(reinterpret_cast<const char*>(&links_count), 2);
    // Salience counters are atomic — snapshot them into plain values so the
    // on-disk layout is byte-identical to previous versions.
    uint32_t rc = recalls();
    uint64_t lr = recalled_at();
    os.write(reinterpret_cast<const char*>(&rc), 4);
    os.write(reinterpret_cast<const char*>(&lr), 8);

    // Phase 4: namespace_id, entity_id, attributes
    uint16_t ns_len = static_cast<uint16_t>(namespace_id.size());
    os.write(reinterpret_cast<const char*>(&ns_len), 2);
    os.write(namespace_id.data(), ns_len);

    uint16_t eid_len = static_cast<uint16_t>(entity_id.size());
    os.write(reinterpret_cast<const char*>(&eid_len), 2);
    os.write(entity_id.data(), eid_len);

    // The native key travels as the first attribute, under a name no caller
    // can mean (it starts with NUL): see Metadata::key.
    auto put_attr = [&os](const std::string& k, const std::string& v) {
        uint16_t key_len = static_cast<uint16_t>(k.size());
        os.write(reinterpret_cast<const char*>(&key_len), 2);
        os.write(k.data(), key_len);
        uint32_t val_len = static_cast<uint32_t>(v.size());
        os.write(reinterpret_cast<const char*>(&val_len), 4);
        os.write(v.data(), val_len);
    };
    uint16_t attr_count = static_cast<uint16_t>(attributes.size() + (key.empty() ? 0 : 1)
                                                + (valid_at ? 1 : 0));
    os.write(reinterpret_cast<const char*>(&attr_count), 2);
    if (!key.empty()) put_attr(key_attr(), key);
    if (valid_at) put_attr(valid_attr(), std::to_string(valid_at));
    for (const auto& [k, val] : attributes) put_attr(k, val);

    // Phase 5: typed, weighted edges
    uint16_t edge_count = static_cast<uint16_t>(edges.size());
    os.write(reinterpret_cast<const char*>(&edge_count), 2);
    for (const auto& e : edges) {
        os.write(reinterpret_cast<const char*>(&e.target_id), 8);
        uint8_t rt_len = static_cast<uint8_t>(std::min(e.rel_type.size(), size_t(255)));
        os.write(reinterpret_cast<const char*>(&rt_len), 1);
        os.write(e.rel_type.data(), rt_len);
        os.write(reinterpret_cast<const char*>(&e.weight), 4);
    }

    // Phase 6: ttl + confidence
    os.write(reinterpret_cast<const char*>(&ttl),        8);
    os.write(reinterpret_cast<const char*>(&confidence), 4);
}

namespace {

// Every length below comes off disk. A record is read from a stream that
// does not know where the record ends, so a length is checked by reading:
// strings grow a chunk at a time as bytes actually arrive, so a corrupt
// 4 GB length costs at most one chunk past the end of the stream instead of a
// 4 GB zero-filled allocation, and a short read is an error rather than a
// field silently left half-filled.
[[noreturn]] void truncated() {
    throw std::runtime_error("corrupt metadata record: truncated");
}

template <class T> void get(std::istream& is, T& v) {
    if (!is.read(reinterpret_cast<char*>(&v), sizeof(T))) truncated();
}

// A trailing field older writers did not emit: false at a clean end of the
// record, an error if the field is cut part way.
template <class T> bool get_opt(std::istream& is, T& v) {
    is.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (is.gcount() == static_cast<std::streamsize>(sizeof(T))) return true;
    if (is.gcount() == 0) return false;
    truncated();
}

void get_bytes(std::istream& is, std::string& s, size_t n) {
    constexpr size_t CHUNK = 64 * 1024;
    s.clear();
    while (s.size() < n) {
        const size_t at = s.size();
        const size_t k = std::min(CHUNK, n - at);
        s.resize(at + k);
        if (!is.read(&s[at], static_cast<std::streamsize>(k))) truncated();
    }
}

template <class Len> std::string get_str(std::istream& is) {
    Len n = 0;
    get(is, n);
    std::string s;
    get_bytes(is, s, n);
    return s;
}

} // namespace

Metadata Metadata::deserialize(std::istream& is) {
    Metadata m;
    get(is, m.timestamp);
    get(is, m.importance);
    uint8_t type_val = 0;
    get(is, type_val);
    m.type = static_cast<ContextType>(type_val);
    m.source    = get_str<uint16_t>(is);
    m.content   = get_str<uint32_t>(is);
    m.tags_json = get_str<uint16_t>(is);

    // Phase 3: legacy links slot (v3/v4 used this; v5 writes 0 here but reads edges below)
    uint16_t links_count = 0;
    if (!get_opt(is, links_count)) return m;
    for (uint16_t i = 0; i < links_count; ++i) {
        // Old v3/v4 plain link IDs — promote to edges with default type/weight
        uint64_t target = 0;
        get(is, target);
        m.edges.push_back({target, "related_to", 1.0f});
    }
    uint32_t rc = 0;
    get(is, rc);
    m.recall_count.store(rc, std::memory_order_relaxed);
    uint64_t lr = 0;
    get(is, lr);
    m.last_recalled_at.store(lr, std::memory_order_relaxed);

    // Phase 4: namespace_id, entity_id, attributes
    uint16_t ns_len = 0;
    if (!get_opt(is, ns_len)) return m;
    get_bytes(is, m.namespace_id, ns_len);
    m.entity_id = get_str<uint16_t>(is);

    uint16_t attr_count = 0;
    get(is, attr_count);
    for (uint16_t i = 0; i < attr_count; ++i) {
        std::string key = get_str<uint16_t>(is);
        if (key == Metadata::key_attr()) m.key = get_str<uint32_t>(is);
        else if (key == Metadata::valid_attr()) {
            // Not a number: keep it as the attribute it was, rather than
            // lose it or throw on a file an older build wrote back.
            std::string v = get_str<uint32_t>(is);
            char* end = nullptr;
            long long t = std::strtoll(v.c_str(), &end, 10);
            if (!v.empty() && end && *end == '\0') m.valid_at = t;
            else m.attributes[std::move(key)] = std::move(v);
        }
        else                             m.attributes[std::move(key)] = get_str<uint32_t>(is);
    }

    // Phase 5: typed, weighted edges
    uint16_t edge_count = 0;
    if (!get_opt(is, edge_count)) return m;
    for (uint16_t i = 0; i < edge_count; ++i) {
        Edge e;
        get(is, e.target_id);
        e.rel_type = get_str<uint8_t>(is);
        get(is, e.weight);
        m.edges.push_back(std::move(e));
    }

    // Phase 6: ttl + confidence (guarded — v5 files default to 0 / 1.0)
    int64_t ttl_val = 0;
    if (!get_opt(is, ttl_val)) return m;
    m.ttl = ttl_val;
    float conf_val = 1.0f;
    if (!get_opt(is, conf_val)) return m;
    m.confidence = conf_val;

    return m;
}

} // namespace feather
