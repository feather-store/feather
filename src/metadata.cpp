#include "../include/metadata.h"
#include <iostream>

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
    os.write(reinterpret_cast<const char*>(&recall_count), 4);
    os.write(reinterpret_cast<const char*>(&last_recalled_at), 8);

    // Phase 4: namespace_id, entity_id, attributes
    uint16_t ns_len = static_cast<uint16_t>(namespace_id.size());
    os.write(reinterpret_cast<const char*>(&ns_len), 2);
    os.write(namespace_id.data(), ns_len);

    uint16_t eid_len = static_cast<uint16_t>(entity_id.size());
    os.write(reinterpret_cast<const char*>(&eid_len), 2);
    os.write(entity_id.data(), eid_len);

    uint16_t attr_count = static_cast<uint16_t>(attributes.size());
    os.write(reinterpret_cast<const char*>(&attr_count), 2);
    for (const auto& [key, val] : attributes) {
        uint16_t key_len = static_cast<uint16_t>(key.size());
        os.write(reinterpret_cast<const char*>(&key_len), 2);
        os.write(key.data(), key_len);
        uint32_t val_len = static_cast<uint32_t>(val.size());
        os.write(reinterpret_cast<const char*>(&val_len), 4);
        os.write(val.data(), val_len);
    }

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
}

Metadata Metadata::deserialize(std::istream& is) {
    Metadata m;
    is.read(reinterpret_cast<char*>(&m.timestamp), 8);
    is.read(reinterpret_cast<char*>(&m.importance), 4);
    uint8_t type_val;
    is.read(reinterpret_cast<char*>(&type_val), 1);
    m.type = static_cast<ContextType>(type_val);

    uint16_t source_len;
    is.read(reinterpret_cast<char*>(&source_len), 2);
    m.source.resize(source_len);
    is.read(&m.source[0], source_len);

    uint32_t content_len;
    is.read(reinterpret_cast<char*>(&content_len), 4);
    m.content.resize(content_len);
    is.read(&m.content[0], content_len);

    uint16_t tags_len;
    is.read(reinterpret_cast<char*>(&tags_len), 2);
    m.tags_json.resize(tags_len);
    is.read(&m.tags_json[0], tags_len);

    // Phase 3: legacy links slot (v3/v4 used this; v5 writes 0 here but reads edges below)
    uint16_t links_count = 0;
    if (!is.read(reinterpret_cast<char*>(&links_count), 2)) return m;
    if (links_count > 0) {
        // Old v3/v4 plain link IDs — promote to edges with default type/weight
        for (uint16_t i = 0; i < links_count; ++i) {
            uint64_t target;
            is.read(reinterpret_cast<char*>(&target), 8);
            m.edges.push_back({target, "related_to", 1.0f});
        }
    }
    is.read(reinterpret_cast<char*>(&m.recall_count), 4);
    is.read(reinterpret_cast<char*>(&m.last_recalled_at), 8);

    // Phase 4: namespace_id, entity_id, attributes
    uint16_t ns_len = 0;
    if (!is.read(reinterpret_cast<char*>(&ns_len), 2)) return m;
    m.namespace_id.resize(ns_len);
    if (ns_len > 0) is.read(&m.namespace_id[0], ns_len);

    uint16_t eid_len = 0;
    is.read(reinterpret_cast<char*>(&eid_len), 2);
    m.entity_id.resize(eid_len);
    if (eid_len > 0) is.read(&m.entity_id[0], eid_len);

    uint16_t attr_count = 0;
    is.read(reinterpret_cast<char*>(&attr_count), 2);
    for (uint16_t i = 0; i < attr_count; ++i) {
        uint16_t key_len = 0;
        is.read(reinterpret_cast<char*>(&key_len), 2);
        std::string key(key_len, '\0');
        if (key_len > 0) is.read(&key[0], key_len);
        uint32_t val_len = 0;
        is.read(reinterpret_cast<char*>(&val_len), 4);
        std::string val(val_len, '\0');
        if (val_len > 0) is.read(&val[0], val_len);
        m.attributes[key] = val;
    }

    // Phase 5: typed, weighted edges
    uint16_t edge_count = 0;
    if (!is.read(reinterpret_cast<char*>(&edge_count), 2)) return m;
    for (uint16_t i = 0; i < edge_count; ++i) {
        Edge e;
        is.read(reinterpret_cast<char*>(&e.target_id), 8);
        uint8_t rt_len = 0;
        is.read(reinterpret_cast<char*>(&rt_len), 1);
        e.rel_type.resize(rt_len);
        if (rt_len > 0) is.read(&e.rel_type[0], rt_len);
        is.read(reinterpret_cast<char*>(&e.weight), 4);
        m.edges.push_back(std::move(e));
    }

    return m;
}

} // namespace feather
