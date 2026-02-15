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

    // Phase 3 serialization
    uint16_t links_count = static_cast<uint16_t>(links.size());
    os.write(reinterpret_cast<const char*>(&links_count), 2);
    if (links_count > 0) {
        os.write(reinterpret_cast<const char*>(links.data()), links_count * 8);
    }
    os.write(reinterpret_cast<const char*>(&recall_count), 4);
    os.write(reinterpret_cast<const char*>(&last_recalled_at), 8);
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

    // Phase 3 deserialization
    uint16_t links_count = 0;
    if (is.read(reinterpret_cast<char*>(&links_count), 2)) {
        m.links.resize(links_count);
        if (links_count > 0) {
            is.read(reinterpret_cast<char*>(m.links.data()), links_count * 8);
        }
        is.read(reinterpret_cast<char*>(&m.recall_count), 4);
        is.read(reinterpret_cast<char*>(&m.last_recalled_at), 8);
    }

    return m;
}

} // namespace feather
