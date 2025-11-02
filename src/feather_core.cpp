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
        db->add(id, std::vector<float>(vec, vec + len));  // ← FIXED: db is unique_ptr*
    }

    void feather_search(void* db_ptr, const float* query, size_t len, size_t k,
                        uint64_t* out_ids, float* out_dists) {
        if (!db_ptr) return;
        auto& db = *static_cast<std::unique_ptr<feather::DB>*>(db_ptr);
        auto results = db->search(std::vector<float>(query, query + len), k);  // ← FIXED
        for (size_t i = 0; i < results.size() && i < k; ++i) {
            out_ids[i] = std::get<0>(results[i]);
            out_dists[i] = std::get<1>(results[i]);
        }
    }

    void feather_save(void* db_ptr) {
        if (!db_ptr) return;
        auto& db = *static_cast<std::unique_ptr<feather::DB>*>(db_ptr);
        db->save();  // ← FIXED
    }

    void feather_close(void* db_ptr) {
        if (db_ptr) delete static_cast<std::unique_ptr<feather::DB>*>(db_ptr);
    }
}
