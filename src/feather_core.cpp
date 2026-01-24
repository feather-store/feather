#include "../include/feather.h"
#include <vector>
#include <memory>

extern "C" {
    void* feather_open(const char* path, size_t dim) {
        try {
            auto db = feather::DB::open(path, dim);
            return new std::unique_ptr<feather::DB>(std::move(db));
        } catch (...) { return nullptr; }
    }

    void feather_add(void* db_ptr, uint64_t id, const float* vec, size_t len) {
        if (!db_ptr) return;
        auto& db = *static_cast<std::unique_ptr<feather::DB>*>(db_ptr);
        db->add(id, std::vector<float>(vec, vec + len));
    }

    void feather_add_with_meta(void* db_ptr, uint64_t id, const float* vec, size_t len,
                               int64_t timestamp, float importance, uint8_t type,
                               const char* source, const char* content) {
        if (!db_ptr) return;
        auto& db = *static_cast<std::unique_ptr<feather::DB>*>(db_ptr);
        
        feather::Metadata meta;
        meta.timestamp = timestamp;
        meta.importance = importance;
        meta.type = static_cast<feather::ContextType>(type);
        if (source) meta.source = source;
        if (content) meta.content = content;
        
        db->add(id, std::vector<float>(vec, vec + len), meta);
    }

    void feather_search(void* db_ptr, const float* query, size_t len, size_t k,
                        uint64_t* out_ids, float* out_dists) {
        if (!db_ptr) return;
        auto& db = *static_cast<std::unique_ptr<feather::DB>*>(db_ptr);
        auto results = db->search(std::vector<float>(query, query + len), k);
        for (size_t i = 0; i < results.size() && i < k; ++i) {
            out_ids[i] = results[i].id;
            out_dists[i] = results[i].score;
        }
    }

    void feather_search_with_filter(void* db_ptr, const float* query, size_t len, size_t k,
                                    uint8_t type_filter, const char* source_filter,
                                    uint64_t* out_ids, float* out_dists) {
        if (!db_ptr) return;
        auto& db = *static_cast<std::unique_ptr<feather::DB>*>(db_ptr);
        
        feather::SearchFilter filter;
        if (type_filter != 255) { // 255 = no filter
            filter.types = std::vector<feather::ContextType>{static_cast<feather::ContextType>(type_filter)};
        }
        if (source_filter && strlen(source_filter) > 0) {
            filter.source = source_filter;
        }

        auto results = db->search(std::vector<float>(query, query + len), k, &filter);
        for (size_t i = 0; i < results.size() && i < k; ++i) {
            out_ids[i] = results[i].id;
            out_dists[i] = results[i].score;
        }
    }

    void feather_save(void* db_ptr) {
        if (!db_ptr) return;
        auto& db = *static_cast<std::unique_ptr<feather::DB>*>(db_ptr);
        db->save();
    }

    void feather_close(void* db_ptr) {
        if (db_ptr) delete static_cast<std::unique_ptr<feather::DB>*>(db_ptr);
    }
}
