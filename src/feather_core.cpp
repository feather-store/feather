#include "../include/feather.h"
#include <vector>
#include <memory>

// The C ABI the Rust CLI links against. No exception may cross it: unwinding
// through an extern "C" frame is undefined behaviour (std::terminate in
// practice). Every entry point catches, records the message for
// feather_last_error(), and reports failure through its return value:
// 0 / -1, nullptr from feather_open, SIZE_MAX from the size_t counters.
namespace {
thread_local std::string g_last_error;

template <class F> int guarded(F&& f) {
    try { f(); g_last_error.clear(); return 0; }
    catch (const std::exception& e) { g_last_error = e.what(); }
    catch (...) { g_last_error = "unknown error"; }
    return -1;
}

feather::DB& db_of(void* p) { return **static_cast<std::unique_ptr<feather::DB>*>(p); }
}

extern "C" {
    // Message of the last failed call on this thread ("" after a success).
    const char* feather_last_error() { return g_last_error.c_str(); }

    void* feather_open(const char* path, size_t dim) {
        void* out = nullptr;
        guarded([&] {
            auto db = feather::DB::open(path, dim);
            out = new std::unique_ptr<feather::DB>(std::move(db));
        });
        return out;
    }

    int feather_add(void* db_ptr, uint64_t id, const float* vec, size_t len) {
        if (!db_ptr) return -1;
        return guarded([&] {
            db_of(db_ptr).add(id, std::vector<float>(vec, vec + len), feather::Metadata(), "text");
        });
    }

    int feather_add_with_meta(void* db_ptr, uint64_t id, const float* vec, size_t len,
                              int64_t timestamp, float importance, uint8_t type,
                              const char* source, const char* content, const char* modality) {
        if (!db_ptr) return -1;
        return guarded([&] {
            feather::Metadata meta;
            meta.timestamp = timestamp;
            meta.importance = importance;
            meta.type = static_cast<feather::ContextType>(type);
            if (source) meta.source = source;
            if (content) meta.content = content;

            std::string mod = modality ? modality : "text";
            db_of(db_ptr).add(id, std::vector<float>(vec, vec + len), meta, mod);
        });
    }

    int feather_link(void* db_ptr, uint64_t from_id, uint64_t to_id) {
        if (!db_ptr) return -1;
        return guarded([&] { db_of(db_ptr).link(from_id, to_id); });
    }

    int feather_touch(void* db_ptr, uint64_t id) {
        if (!db_ptr) return -1;
        return guarded([&] { db_of(db_ptr).touch(id); });
    }

    int feather_search(void* db_ptr, const float* query, size_t len, size_t k,
                       uint64_t* out_ids, float* out_dists, const char* modality) {
        if (!db_ptr) return -1;
        return guarded([&] {
            std::string mod = modality ? modality : "text";
            auto results = db_of(db_ptr).search(std::vector<float>(query, query + len), k, nullptr, nullptr, mod);
            for (size_t i = 0; i < results.size() && i < k; ++i) {
                out_ids[i] = results[i].id;
                out_dists[i] = results[i].score;
            }
        });
    }

    int feather_search_with_filter(void* db_ptr, const float* query, size_t len, size_t k,
                                   uint8_t type_filter, const char* source_filter,
                                   uint64_t* out_ids, float* out_dists, const char* modality) {
        if (!db_ptr) return -1;
        return guarded([&] {
            feather::SearchFilter filter;
            if (type_filter != 255) { // 255 = no filter
                filter.types = std::vector<feather::ContextType>{static_cast<feather::ContextType>(type_filter)};
            }
            if (source_filter && strlen(source_filter) > 0) {
                filter.source = source_filter;
            }

            std::string mod = modality ? modality : "text";
            auto results = db_of(db_ptr).search(std::vector<float>(query, query + len), k, &filter, nullptr, mod);
            for (size_t i = 0; i < results.size() && i < k; ++i) {
                out_ids[i] = results[i].id;
                out_dists[i] = results[i].score;
            }
        });
    }

    int feather_save(void* db_ptr) {
        if (!db_ptr) return -1;
        return guarded([&] { db_of(db_ptr).save(); });
    }

    // ~DB() never throws (a failed checkpoint there is swallowed), so callers
    // that need to know whether the final save worked call feather_save first.
    void feather_close(void* db_ptr) {
        if (db_ptr) delete static_cast<std::unique_ptr<feather::DB>*>(db_ptr);
    }

    // Phase 6: memory lifecycle
    int feather_forget(void* db_ptr, uint64_t id) {
        if (!db_ptr) return -1;
        return guarded([&] { db_of(db_ptr).forget(id); });
    }

    size_t feather_purge(void* db_ptr, const char* namespace_id) {
        if (!db_ptr || !namespace_id) return 0;
        size_t n = 0;
        return guarded([&] { n = db_of(db_ptr).purge(namespace_id); }) == 0 ? n : SIZE_MAX;
    }

    size_t feather_forget_expired(void* db_ptr) {
        if (!db_ptr) return 0;
        size_t n = 0;
        return guarded([&] { n = db_of(db_ptr).forget_expired(); }) == 0 ? n : SIZE_MAX;
    }
}
