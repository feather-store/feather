#include "../include/feather.h"
#include <cstdlib>
#include <cstring>
#include <vector>
#include <memory>

// A byte slice from the caller; p == nullptr means "not given".
struct FeatherStr { const char* p; size_t n; };

// The C ABI the Rust CLI links against. No exception may cross it: unwinding
// through an extern "C" frame is undefined behaviour (std::terminate in
// practice). Every entry point catches, records the message for
// feather_last_error(), and reports failure through its return value:
// 0 / -1, nullptr from feather_open, SIZE_MAX from the size_t counters.
namespace {
thread_local std::string g_last_error;

// What kind of failure the last call on this thread was, so a caller (the
// HTTP server) can answer with the right status without parsing messages.
enum ErrorKind : int {
    FE_OK = 0,
    FE_OTHER = 1,             // anything else: a server error
    FE_INVALID = 2,           // std::invalid_argument, a dimension mismatch: the request's fault
    FE_CLOSED = 3,            // HandleClosed / SnapshotErased: reopen and run the call again
    FE_WRITE_FAILED = 4,      // WriteFailed: the handle refuses writes until reopened
    FE_VERSION_CONFLICT = 5,  // VersionConflict: feather_last_conflict() has the versions
    FE_DUPLICATE_KEY = 6,     // DuplicateKey
    FE_IDEMPOTENCY_REUSED = 7 // IdempotencyKeyReused
};
thread_local int g_last_kind = FE_OK;
thread_local uint64_t g_conflict[3] = {0, 0, 0};   // id, expected, actual

template <class F> int guarded(F&& f) {
    try { f(); g_last_error.clear(); g_last_kind = FE_OK; return 0; }
    catch (const feather::VersionConflict& e) {
        g_last_error = e.what(); g_last_kind = FE_VERSION_CONFLICT;
        g_conflict[0] = e.id; g_conflict[1] = e.expected; g_conflict[2] = e.actual;
    }
    catch (const feather::IdempotencyKeyReused& e) { g_last_error = e.what(); g_last_kind = FE_IDEMPOTENCY_REUSED; }
    catch (const feather::WriteFailed& e)          { g_last_error = e.what(); g_last_kind = FE_WRITE_FAILED; }
    catch (const feather::DuplicateKey& e)         { g_last_error = e.what(); g_last_kind = FE_DUPLICATE_KEY; }
    catch (const feather::HandleClosed& e)         { g_last_error = e.what(); g_last_kind = FE_CLOSED; }
    catch (const std::invalid_argument& e)         { g_last_error = e.what(); g_last_kind = FE_INVALID; }
    catch (const std::exception& e) {
        g_last_error = e.what();
        g_last_kind = std::strstr(e.what(), "Dimension mismatch") ? FE_INVALID : FE_OTHER;
    }
    catch (...) { g_last_error = "unknown error"; g_last_kind = FE_OTHER; }
    return -1;
}

feather::DB& db_of(void* p) { return **static_cast<std::unique_ptr<feather::DB>*>(p); }

// ── Packed records for feather_*_bin ────────────────────────────────────
// Little-endian, no padding. str = [u32 len][bytes]. A metadata record:
//   [i64 timestamp][f32 importance][u8 type][str source][str content]
//   [str tags_json][str namespace_id][str entity_id]
//   [u32 n_attrs]{[str key][str value]}*
//   [u32 recall_count][u64 last_recalled_at]
//   [u32 n_edges]{[u64 target][str rel_type][f32 weight]}*
//   [i64 ttl][f32 confidence][str key][u64 version][i64 valid_at]
// The reader (feather-server/src/packed.rs) must change with it.
struct Packer {
    std::string s;
    template <class T> void raw(T v) { s.append(reinterpret_cast<const char*>(&v), sizeof v); }
    void u8(uint8_t v) { raw(v); }
    void u32(uint32_t v) { raw(v); }
    void u64(uint64_t v) { raw(v); }
    void i64(int64_t v) { raw(v); }
    void f32(float v) { raw(v); }
    void str(const std::string& v) {
        if (v.size() > UINT32_MAX) throw std::length_error("string too long to pack");
        u32(static_cast<uint32_t>(v.size()));
        s.append(v);
    }
    void meta(const feather::Metadata& m) {
        i64(m.timestamp);
        f32(m.importance);
        u8(static_cast<uint8_t>(m.type));
        str(m.source); str(m.content); str(m.tags_json);
        str(m.namespace_id); str(m.entity_id);
        u32(static_cast<uint32_t>(m.attributes.size()));
        for (const auto& [k, v] : m.attributes) { str(k); str(v); }
        const auto [n, at] = m.recall_state();
        u32(n); u64(at);
        u32(static_cast<uint32_t>(m.edges.size()));
        for (const auto& e : m.edges) { u64(e.target_id); str(e.rel_type); f32(e.weight); }
        i64(m.ttl);
        f32(m.confidence);
        str(m.key);
        u64(m.version);
        i64(m.valid_at);
    }
    // [u32 count]{[u64 id][f32 score][f32 cosine, NaN = none][u8 has_meta][meta]}*
    void results(const std::vector<feather::DB::SearchResult>& rs, bool include_metadata) {
        u32(static_cast<uint32_t>(rs.size()));
        for (const auto& r : rs) {
            u64(r.id); f32(r.score); f32(r.cosine);
            u8(include_metadata ? 1 : 0);
            if (include_metadata) meta(r.metadata);
        }
    }
    void release(uint8_t** out, size_t* out_len) {
        auto* p = static_cast<uint8_t*>(std::malloc(s.empty() ? 1 : s.size()));
        if (!p) throw std::bad_alloc();
        std::memcpy(p, s.data(), s.size());
        *out = p;
        *out_len = s.size();
    }
};

inline bool given(FeatherStr s) { return s.p != nullptr; }
inline std::string str_of(FeatherStr s) { return s.p ? std::string(s.p, s.n) : std::string(); }
}

// The filter a search takes. A field left at its "unset" value is not
// evaluated: a null str, a zero count, or its bit clear in `has`.
struct feather_filter {
    FeatherStr namespace_id, namespace_prefix, entity_id, source, source_prefix;
    const uint8_t* types; size_t n_types;
    const FeatherStr* attr_keys; const FeatherStr* attr_vals; size_t n_attrs;
    const FeatherStr* tags; size_t n_tags;
    uint32_t has;   // FF_* bits below
    float importance_gte, confidence_gte;
    int64_t timestamp_after, timestamp_before, valid_after, valid_before;
};
enum : uint32_t {
    FF_IMPORTANCE_GTE = 1, FF_CONFIDENCE_GTE = 2, FF_TIMESTAMP_AFTER = 4,
    FF_TIMESTAMP_BEFORE = 8, FF_VALID_AFTER = 16, FF_VALID_BEFORE = 32,
};

struct feather_scoring { float half_life, weight, min, relevance_floor; };

// The metadata a write gives. Every str is required except `key` (null =
// not given). `has` bits say whether recall_count / last_recalled_at are given.
struct feather_meta_in {
    int64_t timestamp; float importance; uint8_t type;
    FeatherStr source, content, tags_json, namespace_id, entity_id, key;
    float confidence; int64_t ttl; int64_t valid_at;
    uint32_t has;   // 1 recall_count, 2 last_recalled_at
    uint32_t recall_count; uint64_t last_recalled_at;
    const FeatherStr* attr_keys; const FeatherStr* attr_vals; size_t n_attrs;
    const uint64_t* edge_targets; const FeatherStr* edge_types; const float* edge_weights; size_t n_edges;
};


// DB.heat_select's arguments (Pocket.hot() in the engine).
struct feather_heat_query {
    const FeatherStr* namespaces; size_t n_namespaces;   // deepest first
    double now, budget, half_life_days, min_heat, usage_half_life, unused_floor, inherit_decay;
    int64_t chars_per_token;
    FeatherStr pinned_attr, due_attr, origin_attr, text_attr, default_origin, unknown_origin;
    const FeatherStr* trusted; size_t n_trusted;
    const FeatherStr* exclude_sources; size_t n_exclude;
    const FeatherStr* origins; size_t n_origins;
};

// One op of POST /batch (feather-api writes.Op). `meta` null on a PUT
// writes the vector only: a live record keeps its metadata, a new one gets an
// empty record in the batch's default namespace.
struct feather_batch_op {
    uint8_t kind;                        // feather::BatchOp::Kind
    uint32_t flags;                      // FB_* bits below
    uint64_t id;
    FeatherStr ref_ns, ref_key;          // FB_REF: the record by key
    FeatherStr to_ref_ns, to_ref_key;    // FB_TO_REF: the link target by key
    uint64_t expected;                   // FB_EXPECTED
    const feather_meta_in* meta;
    const float* vec; size_t vec_len;    // vec null: no vector
    FeatherStr modality;
    uint64_t to_id;
    FeatherStr rel_type;
    float weight;
};
enum : uint32_t {
    FB_REF = 1, FB_TO_REF = 2, FB_EXPECTED = 4, FB_REQUIRE_TARGET = 8,
    // UPDATE (_keep_identity): the request left edges / key / valid_at out,
    // so a live record keeps its own.
    FB_KEEP_EDGES = 16, FB_KEEP_KEY = 32, FB_KEEP_VALID_AT = 64,
};

namespace {
std::optional<feather::SearchFilter> filter_of(const feather_filter* f) {
    if (!f) return std::nullopt;
    feather::SearchFilter sf;
    bool any = false;
    auto str = [&](FeatherStr s, std::optional<std::string>& dst) {
        if (given(s)) { dst = str_of(s); any = true; }
    };
    str(f->namespace_id, sf.namespace_id);
    str(f->namespace_prefix, sf.namespace_prefix);
    str(f->entity_id, sf.entity_id);
    str(f->source, sf.source);
    str(f->source_prefix, sf.source_prefix);
    if (f->n_types) {
        std::vector<feather::ContextType> t;
        for (size_t i = 0; i < f->n_types; ++i) t.push_back(static_cast<feather::ContextType>(f->types[i]));
        sf.types = std::move(t); any = true;
    }
    if (f->n_attrs) {
        std::unordered_map<std::string, std::string> m;
        for (size_t i = 0; i < f->n_attrs; ++i) m[str_of(f->attr_keys[i])] = str_of(f->attr_vals[i]);
        sf.attributes_match = std::move(m); any = true;
    }
    if (f->n_tags) {
        std::vector<std::string> t;
        for (size_t i = 0; i < f->n_tags; ++i) t.push_back(str_of(f->tags[i]));
        sf.tags_contains = std::move(t); any = true;
    }
    if (f->has & FF_IMPORTANCE_GTE)   { sf.importance_gte = f->importance_gte; any = true; }
    if (f->has & FF_CONFIDENCE_GTE)   { sf.confidence_gte = f->confidence_gte; any = true; }
    if (f->has & FF_TIMESTAMP_AFTER)  { sf.timestamp_after = f->timestamp_after; any = true; }
    if (f->has & FF_TIMESTAMP_BEFORE) { sf.timestamp_before = f->timestamp_before; any = true; }
    if (f->has & FF_VALID_AFTER)      { sf.valid_after = f->valid_after; any = true; }
    if (f->has & FF_VALID_BEFORE)     { sf.valid_before = f->valid_before; any = true; }
    if (!any) return std::nullopt;
    return sf;
}

std::optional<feather::ScoringConfig> scoring_of(const feather_scoring* s) {
    if (!s) return std::nullopt;
    return feather::ScoringConfig(s->half_life, s->weight, s->min, s->relevance_floor);
}

std::vector<std::string> strs_of(const FeatherStr* p, size_t n) {
    std::vector<std::string> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) out.push_back(str_of(p[i]));
    return out;
}

// feather-api's _is_dead_meta.
bool is_dead(const feather::Metadata& m) {
    if (m.source == "_forgotten") return true;
    auto it = m.attributes.find("_deleted");
    return it != m.attributes.end() && it->second == "true";
}

void pack_ids(Packer& p, const std::vector<uint64_t>& ids) {
    p.u32(static_cast<uint32_t>(ids.size()));
    for (uint64_t id : ids) p.u64(id);
}

feather::Metadata meta_of(const feather_meta_in& in) {
    feather::Metadata m;
    m.timestamp = in.timestamp;
    m.importance = in.importance;
    m.type = static_cast<feather::ContextType>(in.type);
    m.source = str_of(in.source);
    m.content = str_of(in.content);
    m.tags_json = str_of(in.tags_json);
    m.namespace_id = str_of(in.namespace_id);
    m.entity_id = str_of(in.entity_id);
    m.confidence = in.confidence;
    m.ttl = in.ttl;
    m.valid_at = in.valid_at;
    if (given(in.key) && in.key.n) m.key = str_of(in.key);
    if (in.has & 1) m.recall_count.store(in.recall_count, std::memory_order_relaxed);
    if (in.has & 2) m.last_recalled_at.store(in.last_recalled_at, std::memory_order_relaxed);
    for (size_t i = 0; i < in.n_edges; ++i)
        m.edges.push_back({in.edge_targets[i], str_of(in.edge_types[i]), in.edge_weights[i]});
    for (size_t i = 0; i < in.n_attrs; ++i)
        m.attributes[str_of(in.attr_keys[i])] = str_of(in.attr_vals[i]);
    return m;
}
}

extern "C" {
    // Message of the last failed call on this thread ("" after a success).
    const char* feather_last_error() { return g_last_error.c_str(); }

    // ErrorKind of the last failed call on this thread (0 after a success).
    int feather_last_error_kind() { return g_last_kind; }

    // The versions of the last FE_VERSION_CONFLICT on this thread.
    void feather_last_conflict(uint64_t* id, uint64_t* expected, uint64_t* actual) {
        *id = g_conflict[0]; *expected = g_conflict[1]; *actual = g_conflict[2];
    }

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
            db_of(db_ptr).add_vector(id, std::vector<float>(vec, vec + len), "text");
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

    // erase(): the receipt as a JSON string the caller frees with
    // feather_free_string(), or nullptr (message in feather_last_error()).
    // `ids` may be null when n_ids is 0; `namespace_id` and `entity_id` may be
    // null or "" for "not given".
    char* feather_erase(void* db_ptr, const char* namespace_id, int prefix,
                        const char* entity_id, const uint64_t* ids, size_t n_ids,
                        int dry_run) {
        if (!db_ptr) return nullptr;
        char* out = nullptr;
        guarded([&] {
            feather::DB::EraseRequest rq;
            if (namespace_id) rq.namespace_id = namespace_id;
            if (entity_id)    rq.entity_id    = entity_id;
            rq.namespace_prefix = prefix != 0;
            if (ids && n_ids) rq.ids.assign(ids, ids + n_ids);
            rq.dry_run = dry_run != 0;
            const std::string json = feather::DB::erase_receipt_json(db_of(db_ptr).erase(rq));
            out = static_cast<char*>(std::malloc(json.size() + 1));
            if (!out) throw std::bad_alloc();
            std::memcpy(out, json.c_str(), json.size() + 1);
        });
        return out;
    }

    void feather_free_string(char* s) { std::free(s); }

    size_t feather_forget_expired(void* db_ptr) {
        if (!db_ptr) return 0;
        size_t n = 0;
        return guarded([&] { n = db_of(db_ptr).forget_expired(); }) == 0 ? n : SIZE_MAX;
    }

    // ── For feather-server (the Rust HTTP server) ───────────────────────
    // Reads return packed records (see Packer) in a malloc'd buffer the
    // caller frees with feather_free_buf(). Return 0 on success, -1 on error
    // (feather_last_error / feather_last_error_kind), 1 for "no such record".

    void feather_free_buf(uint8_t* p) { std::free(p); }

    int feather_search_bin(void* db_ptr, const float* q, size_t len, size_t k,
                           const feather_filter* filter, const feather_scoring* scoring,
                           FeatherStr modality, int record_salience, int with_cosine,
                           int include_metadata, uint8_t** out, size_t* out_len) {
        if (!db_ptr) return -1;
        return guarded([&] {
            const auto sf = filter_of(filter);
            const auto sc = scoring_of(scoring);
            const auto rs = db_of(db_ptr).search(std::vector<float>(q, q + len), k,
                                                 sf ? &*sf : nullptr, sc ? &*sc : nullptr,
                                                 str_of(modality), record_salience != 0,
                                                 with_cosine != 0);
            Packer p;
            p.results(rs, include_metadata != 0);
            p.release(out, out_len);
        });
    }

    int feather_keyword_search_bin(void* db_ptr, FeatherStr query, size_t k,
                                   const feather_filter* filter, int record_salience,
                                   int include_metadata, uint8_t** out, size_t* out_len) {
        if (!db_ptr) return -1;
        return guarded([&] {
            const auto sf = filter_of(filter);
            const auto rs = db_of(db_ptr).keyword_search(str_of(query), k, sf ? &*sf : nullptr,
                                                         record_salience != 0);
            Packer p;
            p.results(rs, include_metadata != 0);
            p.release(out, out_len);
        });
    }

    int feather_hybrid_search_bin(void* db_ptr, const float* q, size_t len, FeatherStr query,
                                  size_t k, size_t rrf_k, const feather_filter* filter,
                                  const feather_scoring* scoring, FeatherStr modality,
                                  FeatherStr fusion, int include_metadata,
                                  uint8_t** out, size_t* out_len) {
        if (!db_ptr) return -1;
        return guarded([&] {
            const auto sf = filter_of(filter);
            const auto sc = scoring_of(scoring);
            const auto rs = db_of(db_ptr).hybrid_search(std::vector<float>(q, q + len), str_of(query),
                                                        k, rrf_k, sf ? &*sf : nullptr,
                                                        sc ? &*sc : nullptr, str_of(modality),
                                                        str_of(fusion));
            Packer p;
            p.results(rs, include_metadata != 0);
            p.release(out, out_len);
        });
    }

    // One record's packed metadata; 1 when there is none (or it is dead and
    // include_dead is 0).
    int feather_get_bin(void* db_ptr, uint64_t id, int include_dead, uint8_t** out, size_t* out_len) {
        if (!db_ptr) return -1;
        bool found = false;
        const int rc = guarded([&] {
            const auto m = db_of(db_ptr).get_metadata(id, include_dead != 0);
            if (!m) return;
            found = true;
            Packer p;
            p.meta(*m);
            p.release(out, out_len);
        });
        return rc != 0 ? rc : (found ? 0 : 1);
    }

    // [u32 n]{[u8 present][meta if present]}*, in the order of `ids`.
    int feather_get_many_bin(void* db_ptr, const uint64_t* ids, size_t n, int include_dead,
                             uint8_t** out, size_t* out_len) {
        if (!db_ptr) return -1;
        return guarded([&] {
            const auto ms = db_of(db_ptr).get_metadata_many(std::vector<uint64_t>(ids, ids + n),
                                                            include_dead != 0);
            Packer p;
            p.u32(static_cast<uint32_t>(ms.size()));
            for (const auto& m : ms) {
                p.u8(m ? 1 : 0);
                if (m) p.meta(*m);
            }
            p.release(out, out_len);
        });
    }

    // The stored vector as float32 (empty when the record has none in this
    // modality); free with feather_free_buf().
    int feather_get_vector(void* db_ptr, uint64_t id, FeatherStr modality, float** out, size_t* n) {
        if (!db_ptr) return -1;
        return guarded([&] {
            const auto v = db_of(db_ptr).get_vector(id, str_of(modality));
            auto* p = static_cast<float*>(std::malloc(v.empty() ? 1 : v.size() * sizeof(float)));
            if (!p) throw std::bad_alloc();
            if (!v.empty()) std::memcpy(p, v.data(), v.size() * sizeof(float));
            *out = p;
            *n = v.size();
        });
    }

    // 1 and the dim when `modality` has an index; 0 when it has none yet
    // (the first vector will set its dim).
    int feather_modality_dim(void* db_ptr, FeatherStr modality, size_t* out_dim) {
        if (!db_ptr) return -1;
        int has = 0;
        const int rc = guarded([&] {
            auto& db = db_of(db_ptr);
            const std::string mod = str_of(modality);
            for (const auto& name : db.modality_names()) {
                if (name == mod) { *out_dim = db.dim(mod); has = 1; return; }
            }
        });
        return rc != 0 ? rc : has;
    }

    // Why the handle refuses writes (a malloc'd string to free with
    // feather_free_buf), or nullptr when it accepts them.
    char* feather_write_error(void* db_ptr) {
        if (!db_ptr) return nullptr;
        char* out = nullptr;
        guarded([&] {
            const std::string why = db_of(db_ptr).write_error();
            if (why.empty()) return;
            out = static_cast<char*>(std::malloc(why.size() + 1));
            if (!out) throw std::bad_alloc();
            std::memcpy(out, why.c_str(), why.size() + 1);
        });
        return out;
    }

    // Bytes in the WAL (current + rotated); UINT64_MAX on error.
    uint64_t feather_wal_size(void* db_ptr) {
        if (!db_ptr) return UINT64_MAX;
        uint64_t n = 0;
        return guarded([&] { n = db_of(db_ptr).wal_size(); }) == 0 ? n : UINT64_MAX;
    }

    // Close, checkpointing only if `save` (a handle whose write failed must
    // not checkpoint: close(save=false) keeps its WAL for the reopen).
    int feather_close_ex(void* db_ptr, int save) {
        if (!db_ptr) return 0;
        auto* h = static_cast<std::unique_ptr<feather::DB>*>(db_ptr);
        const int rc = guarded([&] { (*h)->close(save != 0); });
        delete h;
        return rc;
    }

    // An idempotency key's stored result, if any: 1 with the version of its
    // last op and the stored note (malloc'd, feather_free_buf), 0 if none.
    int feather_idempotent_result(void* db_ptr, FeatherStr key, FeatherStr fingerprint,
                                  uint64_t* out_version, char** out_note, size_t* out_note_len) {
        if (!db_ptr) return -1;
        int found = 0;
        const int rc = guarded([&] {
            const auto r = db_of(db_ptr).idempotent_result(str_of(key), str_of(fingerprint));
            if (!r) return;
            found = 1;
            *out_version = r->versions.empty() ? 0 : r->versions.back();
            auto* p = static_cast<char*>(std::malloc(r->note.size() + 1));
            if (!p) throw std::bad_alloc();
            std::memcpy(p, r->note.c_str(), r->note.size() + 1);
            *out_note = p;
            *out_note_len = r->note.size();
        });
        return rc != 0 ? rc : found;
    }

    // One PUT through write_batch (atomic, versioned, idempotent with a key).
    // `meta` null = write the vector only: a live record keeps its metadata,
    // a new one gets an empty record in `default_ns`. Writes the record's new
    // version, whether the key replayed an earlier attempt, and that
    // attempt's note (malloc'd or null; feather_free_buf).
    int feather_put(void* db_ptr, uint64_t id, const float* vec, size_t len,
                    const feather_meta_in* meta, FeatherStr modality, FeatherStr default_ns,
                    FeatherStr idem_key, FeatherStr fingerprint, FeatherStr note,
                    uint64_t* out_version, int* out_replayed, char** out_note, size_t* out_note_len) {
        if (!db_ptr) return -1;
        return guarded([&] {
            auto& db = db_of(db_ptr);
            feather::BatchOp op;
            op.kind = feather::BatchOp::PUT;
            op.id = id;
            op.vec.assign(vec, vec + len);
            op.modality = str_of(modality);
            if (meta) {
                op.meta = meta_of(*meta);
            } else if (auto existing = db.get_metadata(id)) {
                op.meta = std::move(*existing);
            } else {
                op.meta.namespace_id = str_of(default_ns);
            }
            std::vector<feather::BatchOp> ops;
            ops.push_back(std::move(op));
            const auto r = db.write_batch(std::move(ops), str_of(idem_key), str_of(fingerprint),
                                          str_of(note));
            *out_version = r.versions.empty() ? 0 : r.versions.back();
            *out_replayed = r.replayed ? 1 : 0;
            *out_note = nullptr;
            *out_note_len = 0;
            if (r.replayed && !r.note.empty()) {
                auto* p = static_cast<char*>(std::malloc(r.note.size() + 1));
                if (!p) throw std::bad_alloc();
                std::memcpy(p, r.note.c_str(), r.note.size() + 1);
                *out_note = p;
                *out_note_len = r.note.size();
            }
        });
    }
    // ── Agent-memory routes of feather-server ───────────────────────────
    // What an agent session sends besides search and get (measured: setup,
    // every turn, close). Id lists pack as [u32 n][u64 id]*.

    int feather_dim(void* db_ptr, FeatherStr modality, size_t* out) {
        if (!db_ptr) return -1;
        return guarded([&] { *out = db_of(db_ptr).dim(str_of(modality)); });
    }

    // The live record holding `key` in `ns`: its id and packed metadata; 1 if none.
    int feather_lookup_key_bin(void* db_ptr, FeatherStr key, FeatherStr ns, uint64_t* out_id,
                               uint8_t** out, size_t* out_len) {
        if (!db_ptr) return -1;
        bool found = false;
        const int rc = guarded([&] {
            const auto r = db_of(db_ptr).lookup_key(str_of(key), str_of(ns));
            if (!r) return;
            found = true;
            *out_id = r->first;
            Packer p;
            p.meta(r->second);
            p.release(out, out_len);
        });
        return rc != 0 ? rc : (found ? 0 : 1);
    }

    // [u32 n]{[u8 found][u64 id]}*, in the order of the refs.
    int feather_ids_for_many_bin(void* db_ptr, const FeatherStr* nss, const FeatherStr* keys, size_t n,
                                 uint8_t** out, size_t* out_len) {
        if (!db_ptr) return -1;
        return guarded([&] {
            std::vector<std::pair<std::string, std::string>> refs;
            refs.reserve(n);
            for (size_t i = 0; i < n; ++i) refs.emplace_back(str_of(nss[i]), str_of(keys[i]));
            const auto ids = db_of(db_ptr).ids_for_many(refs);
            Packer p;
            p.u32(static_cast<uint32_t>(ids.size()));
            for (const auto& id : ids) { p.u8(id ? 1 : 0); p.u64(id ? *id : 0); }
            p.release(out, out_len);
        });
    }

    int feather_newest_bin(void* db_ptr, FeatherStr ns, size_t limit, int prefix, int has_before,
                           int64_t before_ts, uint64_t before_id, int by_valid,
                           uint8_t** out, size_t* out_len) {
        if (!db_ptr) return -1;
        return guarded([&] {
            std::optional<std::pair<int64_t, uint64_t>> before;
            if (has_before) before = std::make_pair(before_ts, before_id);
            Packer p;
            pack_ids(p, db_of(db_ptr).newest(str_of(ns), limit, prefix != 0, before, by_valid != 0));
            p.release(out, out_len);
        });
    }

    // GET /ids: which = 0 every record, 1 in namespace `a` (prefix), 2 for
    // entity `a`, 3 with attribute a=b, 4 with a vector in modality `a`.
    int feather_ids_bin(void* db_ptr, int which, FeatherStr a, FeatherStr b, int prefix,
                        uint8_t** out, size_t* out_len) {
        if (!db_ptr) return -1;
        return guarded([&] {
            auto& db = db_of(db_ptr);
            std::vector<uint64_t> ids;
            switch (which) {
                case 1: ids = db.ids_in_namespace(str_of(a), prefix != 0); break;
                case 2: ids = db.ids_for_entity(str_of(a)); break;
                case 3: ids = db.ids_with_attribute(str_of(a), str_of(b)); break;
                case 4: ids = db.get_all_ids(str_of(a)); break;
                default: ids = db.all_ids(); break;
            }
            Packer p;
            pack_ids(p, ids);
            p.release(out, out_len);
        });
    }

    // [u32 n]{[str namespace][u64 size]}*
    int feather_metadata_namespaces_bin(void* db_ptr, uint8_t** out, size_t* out_len) {
        if (!db_ptr) return -1;
        return guarded([&] {
            auto& db = db_of(db_ptr);
            const auto names = db.list_namespaces();
            Packer p;
            p.u32(static_cast<uint32_t>(names.size()));
            for (const auto& n : names) { p.str(n); p.u64(db.namespace_size(n)); }
            p.release(out, out_len);
        });
    }

    int feather_count_live(void* db_ptr, FeatherStr ns, int prefix, const FeatherStr* exclude,
                           size_t n_exclude, uint64_t* out) {
        if (!db_ptr) return -1;
        return guarded([&] {
            *out = db_of(db_ptr).count_live(str_of(ns), prefix != 0, strs_of(exclude, n_exclude));
        });
    }

    // [u32 n]{[u64 id][u32 depth][f64 heat][i64 tokens]}*, in rank order.
    int feather_heat_select_bin(void* db_ptr, const feather_heat_query* in, uint8_t** out, size_t* out_len) {
        if (!db_ptr || !in) return -1;
        return guarded([&] {
            feather::HeatQuery q;
            q.namespaces = strs_of(in->namespaces, in->n_namespaces);
            q.now = in->now; q.budget = in->budget; q.half_life_days = in->half_life_days;
            q.min_heat = in->min_heat; q.usage_half_life = in->usage_half_life;
            q.unused_floor = in->unused_floor; q.inherit_decay = in->inherit_decay;
            q.chars_per_token = in->chars_per_token;
            q.pinned_attr = str_of(in->pinned_attr); q.due_attr = str_of(in->due_attr);
            q.origin_attr = str_of(in->origin_attr); q.text_attr = str_of(in->text_attr);
            q.default_origin = str_of(in->default_origin); q.unknown_origin = str_of(in->unknown_origin);
            for (auto& t : strs_of(in->trusted, in->n_trusted)) q.trusted.insert(std::move(t));
            for (auto& e : strs_of(in->exclude_sources, in->n_exclude)) q.exclude_sources.insert(std::move(e));
            q.origins = strs_of(in->origins, in->n_origins);
            const auto picks = db_of(db_ptr).heat_select(q);
            Packer p;
            p.u32(static_cast<uint32_t>(picks.size()));
            for (const auto& k : picks) { p.u64(k.id); p.u32(k.depth); p.raw(k.heat); p.i64(k.tokens); }
            p.release(out, out_len);
        });
    }

    int feather_touch_many(void* db_ptr, const uint64_t* ids, size_t n) {
        if (!db_ptr) return -1;
        return guarded([&] { db_of(db_ptr).touch_many(std::vector<uint64_t>(ids, ids + n)); });
    }

    int feather_normalize_vectors(void* db_ptr, FeatherStr modality, uint64_t* out) {
        if (!db_ptr) return -1;
        return guarded([&] { *out = db_of(db_ptr).normalize_vectors(str_of(modality)); });
    }

    // GET /admin/engine, packed:
    //   [u64 size][u64 dim][f32 auto_compact][u8 compacting]
    //   [u64 graphs][u64 nodes][u64 pending][u64 min_records]
    //   [u32 format][u64 generation][u64 file_bytes][u64 live_bytes_estimate]
    //   [u64 last_commit_bytes][u8 last_commit_full][u64 commits][u64 vacuums]
    //   [u64 mapped_bytes][u64 history_files][u64 history_bytes]
    //   [u64 wal_size][u64 wal_fsync_count]
    //   [u32 n]{[str name][u64 dim][u64 ef][u8 quantized][u8 int8_ram]
    //           [u8 int8][u8 training][f32 clip][u32 refine][u64 on_disk]
    //           [u64 in_wal][u64 in_ram][u32 train_min][f32 background_cosine]}*
    int feather_engine_state_bin(void* db_ptr, uint8_t** out, size_t* out_len) {
        if (!db_ptr) return -1;
        return guarded([&] {
            auto& db = db_of(db_ptr);
            Packer p;
            p.u64(db.size()); p.u64(db.dim()); p.f32(db.get_auto_compact()); p.u8(db.is_compacting());
            const auto a = db.namespace_ann_info();
            p.u64(a.graphs); p.u64(a.nodes); p.u64(a.pending); p.u64(a.min_records);
            const auto s = db.storage_stats();
            p.u32(s.format); p.u64(s.generation); p.u64(s.file_bytes); p.u64(s.live_bytes_estimate);
            p.u64(s.last_commit_bytes); p.u8(s.last_commit_full); p.u64(s.commits); p.u64(s.vacuums);
            p.u64(s.mapped_bytes); p.u64(s.history_files); p.u64(s.history_bytes);
            p.u64(db.wal_size()); p.u64(db.wal_fsync_count());
            const auto names = db.modality_names();
            p.u32(static_cast<uint32_t>(names.size()));
            for (const auto& n : names) {
                p.str(n); p.u64(db.dim(n)); p.u64(db.get_ef(n));
                p.u8(db.is_quantized(n)); p.u8(db.is_int8_ram(n));
                const auto i = db.int8_info(n);
                p.u8(i.int8); p.u8(i.training); p.f32(i.clip); p.u32(i.refine);
                p.u64(i.originals_on_disk); p.u64(i.originals_in_wal); p.u64(i.originals_in_ram);
                p.u32(i.train_min);
                p.f32(db.background_cosine(n));
            }
            p.release(out, out_len);
        });
    }

    // POST /batch: one atomic, versioned write (feather-api write_batch +
    // writes.apply). A stored result for the idempotency key answers first,
    // as writes.prior does. Packed result:
    //   [u8 replayed][str note][u32 n]{[u64 id][u64 version]}*
    int feather_write_batch_bin(void* db_ptr, const feather_batch_op* in, size_t n, FeatherStr default_ns,
                                FeatherStr idem_key, FeatherStr fingerprint, FeatherStr note,
                                uint8_t** out, size_t* out_len) {
        if (!db_ptr) return -1;
        return guarded([&] {
            auto& db = db_of(db_ptr);
            const std::string key = str_of(idem_key), fp = str_of(fingerprint);
            std::optional<feather::BatchResult> r;
            if (!key.empty()) r = db.idempotent_result(key, fp);
            if (!r) {
                std::vector<feather::BatchOp> ops;
                ops.reserve(n);
                for (size_t i = 0; i < n; ++i) {
                    const auto& o = in[i];
                    feather::BatchOp op;
                    op.kind = static_cast<feather::BatchOp::Kind>(o.kind);
                    op.id = o.id;
                    if (o.flags & FB_REF) op.ref = feather::KeyRef{str_of(o.ref_ns), str_of(o.ref_key)};
                    if (o.flags & FB_TO_REF) op.to_ref = feather::KeyRef{str_of(o.to_ref_ns), str_of(o.to_ref_key)};
                    if (o.flags & FB_EXPECTED) op.expected = o.expected;
                    if (o.meta) {
                        op.meta = meta_of(*o.meta);
                    } else if (op.kind == feather::BatchOp::PUT) {
                        // _kept_metadata: the vector only.
                        if (auto existing = db.get_metadata(o.id)) op.meta = std::move(*existing);
                        else op.meta.namespace_id = str_of(default_ns);
                    }   // LINK / FORGET carry no metadata (WriteBatch.link / .forget)
                    if (op.kind == feather::BatchOp::UPDATE && o.meta) {
                        // _keep_identity: what the request left out, a live record keeps.
                        std::optional<uint64_t> rid;
                        if (op.ref) rid = db.id_for(op.ref->key, op.ref->ns);
                        else rid = o.id;
                        const auto old = rid ? db.get_metadata(*rid) : std::nullopt;
                        if (old && !is_dead(*old)) {
                            if (o.flags & FB_KEEP_EDGES) op.meta.edges = old->edges;
                            if (o.flags & FB_KEEP_KEY) op.meta.key = old->key;
                            if (o.flags & FB_KEEP_VALID_AT) op.meta.valid_at = old->valid_at;
                        }
                    }
                    if (o.vec) op.vec.assign(o.vec, o.vec + o.vec_len);
                    op.modality = str_of(o.modality);
                    op.to_id = o.to_id;
                    op.rel_type = str_of(o.rel_type);
                    op.weight = o.weight;
                    op.require_target = (o.flags & FB_REQUIRE_TARGET) != 0;
                    ops.push_back(std::move(op));
                }
                r = db.write_batch(std::move(ops), key, fp, str_of(note));
            }
            Packer p;
            p.u8(r->replayed ? 1 : 0);
            p.str(r->note);
            p.u32(static_cast<uint32_t>(r->ids.size()));
            for (size_t i = 0; i < r->ids.size(); ++i) { p.u64(r->ids[i]); p.u64(r->versions[i]); }
            p.release(out, out_len);
        });
    }

    // Edges INTO `id`: [u32 n]{[u64 source][str rel_type][f32 weight]}*.
    // DELETE /records/{id} prunes them in the forget's batch (_prune_ops).
    int feather_incoming_bin(void* db_ptr, uint64_t id, uint8_t** out, size_t* out_len) {
        if (!db_ptr) return -1;
        return guarded([&] {
            const auto in = db_of(db_ptr).get_incoming(id);
            Packer p;
            p.u32(static_cast<uint32_t>(in.size()));
            for (const auto& e : in) { p.u64(e.source_id); p.str(e.rel_type); p.f32(e.weight); }
            p.release(out, out_len);
        });
    }

    // POST /erase: erase() with every argument it takes, the receipt as JSON
    // (DB::erase_receipt_json). has_cites / has_feeds = 0 keeps the engine's
    // default edge kinds; 1 with n = 0 means none.
    int feather_erase_json(void* db_ptr, FeatherStr ns, int prefix, FeatherStr entity,
                           const uint64_t* ids, size_t n_ids,
                           int has_cites, const FeatherStr* cites, size_t n_cites,
                           int has_feeds, const FeatherStr* feeds, size_t n_feeds,
                           int dry_run, uint8_t** out, size_t* out_len) {
        if (!db_ptr) return -1;
        return guarded([&] {
            feather::DB::EraseRequest rq;
            rq.namespace_id = str_of(ns);
            rq.namespace_prefix = prefix != 0;
            rq.entity_id = str_of(entity);
            if (ids && n_ids) rq.ids.assign(ids, ids + n_ids);
            if (has_cites) rq.cites = strs_of(cites, n_cites);
            if (has_feeds) rq.feeds = strs_of(feeds, n_feeds);
            rq.dry_run = dry_run != 0;
            Packer p;
            p.s = feather::DB::erase_receipt_json(db_of(db_ptr).erase(rq));
            p.release(out, out_len);
        });
    }
}
