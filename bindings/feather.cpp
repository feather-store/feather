#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../include/feather.h"

namespace py = pybind11;

namespace {
// The ops of DB.write(), collected in Python. Vectors are copied in here, so
// the write itself runs without the GIL.
struct WriteBatch {
    std::vector<feather::BatchOp> ops;
};

std::vector<float> to_vec(py::array_t<float, py::array::c_style | py::array::forcecast> a) {
    auto buf = a.request();
    const float* p = static_cast<const float*>(buf.ptr);
    return std::vector<float>(p, p + buf.size);
}

// A record named in a WriteBatch op: an int id, a key (str, in namespace ""),
// or a (namespace, key) tuple.
void set_target(const py::handle& h, uint64_t& id, std::optional<feather::KeyRef>& ref) {
    if (py::isinstance<py::str>(h)) {
        ref = feather::KeyRef{"", h.cast<std::string>()};
    } else if (py::isinstance<py::tuple>(h)) {
        auto t = h.cast<py::tuple>();
        if (t.size() != 2) throw py::value_error("a key reference is (namespace, key)");
        ref = feather::KeyRef{t[0].cast<std::string>(), t[1].cast<std::string>()};
    } else {
        id = h.cast<uint64_t>();
    }
}
} // namespace

// Build the API's JSON-ready dict for a Metadata straight from C++. The Cloud
// API used to go Metadata -> pydantic model -> dict per hit; this is one pass.
static py::dict metadata_to_dict(const feather::Metadata& m) {
    py::dict d;
    d["timestamp"]        = m.timestamp;
    d["importance"]       = m.importance;
    d["type"]             = static_cast<int>(m.type);
    d["source"]           = m.source;
    d["content"]          = m.content;
    d["tags_json"]        = m.tags_json;
    d["namespace_id"]     = m.namespace_id;
    d["entity_id"]        = m.entity_id;
    py::dict attrs;
    for (const auto& [k, v] : m.attributes) attrs[py::str(k)] = v;
    d["attributes"]       = attrs;
    d["recall_count"]     = m.recalls();
    d["last_recalled_at"] = m.recalled_at();
    py::list links, edges;
    for (const auto& e : m.edges) {
        links.append(e.target_id);
        py::dict ed;
        ed["target_id"] = e.target_id;
        ed["rel_type"]  = e.rel_type;
        ed["weight"]    = e.weight;
        edges.append(ed);
    }
    d["links"]            = links;
    d["edges"]            = edges;     // typed: what `links` drops (rel_type, weight)
    d["ttl"]              = m.ttl;
    d["confidence"]       = m.confidence;
    d["key"]              = m.key;
    d["version"]          = m.version;
    d["valid_at"]         = m.valid_at;
    return d;
}

PYBIND11_MODULE(core, m) {
    m.doc() = "Feather: Embedded Vector Database + Living Context Engine";

    // ── SIMD introspection + kernel test hooks ──────────────────────────
    m.def("simd_info", []() {
        py::dict d;
        d["active"]   = feather_simd::level_name(feather_simd::level());
        d["detected"] = feather_simd::level_name(feather_simd::detected_level());
        return d;
    }, "Distance-kernel ISA in use ('active', after any FEATHER_SIMD_RUNTIME cap) "
       "and the best one this CPU supports ('detected').");

    // Levels usable on this machine, lowest first (for tests / benchmarks).
    m.def("_simd_levels", []() {
        std::vector<std::string> out{"scalar"};
        const int det = feather_simd::detected_level();
        if (det == feather_simd::NEON) { out.push_back("neon"); return out; }
        if (det >= feather_simd::SSE)    out.push_back("sse2");
        if (det >= feather_simd::AVX2)   out.push_back("avx2+fma");
        if (det >= feather_simd::AVX512) out.push_back("avx512f");
        return out;
    });

    py::register_exception<feather::WriteFailed>(m, "WriteFailedError", PyExc_RuntimeError);
    py::register_exception<feather::IdempotencyKeyReused>(m, "IdempotencyKeyReusedError", PyExc_ValueError);
    py::register_exception<feather::DuplicateKey>(m, "DuplicateKeyError", PyExc_ValueError);
    py::register_exception<feather::HandleClosed>(m, "ClosedError", PyExc_RuntimeError);
    // VersionConflictError carries .id, .expected and .actual, so a caller can
    // re-read and retry without parsing the message.
    static PyObject* version_conflict =
        py::register_exception<feather::VersionConflict>(m, "VersionConflictError", PyExc_RuntimeError).ptr();
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const feather::VersionConflict& e) {
            py::object err = py::reinterpret_borrow<py::object>(version_conflict)(e.what());
            err.attr("id")       = e.id;
            err.attr("expected") = e.expected;
            err.attr("actual")   = e.actual;
            PyErr_SetObject(version_conflict, err.ptr());
        }
    });

    py::class_<feather::BatchResult>(m, "BatchResult")
        .def_readonly("ids",      &feather::BatchResult::ids,
             "Per op, the record it wrote.")
        .def_readonly("versions", &feather::BatchResult::versions,
             "Per op, that record's version after the op.")
        .def_readonly("note",     &feather::BatchResult::note,
             "The note stored with the idempotency key by the first attempt.")
        .def_readonly("replayed", &feather::BatchResult::replayed,
             "True when the idempotency key had been seen: this is the first attempt's "
             "result, and nothing was applied this time.")
        .def("__repr__", [](const feather::BatchResult& r) {
            return "<BatchResult ops=" + std::to_string(r.ids.size()) +
                   (r.replayed ? " replayed>" : ">");
        });

    py::class_<WriteBatch>(m, "WriteBatch",
        "Records to write in one atomic DB.write(): every op is applied, or none. "
        "Each op takes expected_version= for compare-and-set (0 = the record must "
        "not exist yet); every check runs before anything is written.")
        .def(py::init<>())
        .def("put", [](WriteBatch& b, uint64_t id,
                       py::array_t<float, py::array::c_style | py::array::forcecast> vec,
                       const std::optional<feather::Metadata>& meta, const std::string& modality,
                       std::optional<uint64_t> expected_version) -> WriteBatch& {
                 feather::BatchOp op;
                 op.kind = feather::BatchOp::PUT;
                 op.id = id;
                 op.vec = to_vec(vec);
                 if (meta) op.meta = *meta;
                 op.modality = modality;
                 op.expected = expected_version;
                 b.ops.push_back(std::move(op));
                 return b;
             }, py::arg("id"), py::arg("vec"), py::arg("meta") = std::nullopt,
             py::arg("modality") = "text", py::arg("expected_version") = std::nullopt,
             py::return_value_policy::reference_internal,
             "As DB.add(): vector + metadata; metadata without edges keeps the record's edges.")
        .def("upsert", [](WriteBatch& b, const feather::Metadata& meta,
                          std::optional<py::array_t<float, py::array::c_style | py::array::forcecast>> vec,
                          const std::string& modality, std::optional<uint64_t> expected_version) -> WriteBatch& {
                 feather::BatchOp op;
                 op.kind = feather::BatchOp::UPSERT;
                 op.meta = meta;
                 if (vec) op.vec = to_vec(*vec);
                 op.modality = modality;
                 op.expected = expected_version;
                 b.ops.push_back(std::move(op));
                 return b;
             }, py::arg("meta"), py::arg("vec") = std::nullopt, py::arg("modality") = "text",
             py::arg("expected_version") = std::nullopt,
             py::return_value_policy::reference_internal,
             "The record keyed (meta.namespace_id, meta.key), created with a fresh id if "
             "there is none. Ops after it can name it by that key.")
        .def("update", [](WriteBatch& b, py::object target, const feather::Metadata& meta,
                          std::optional<uint64_t> expected_version) -> WriteBatch& {
                 feather::BatchOp op;
                 op.kind = feather::BatchOp::UPDATE;
                 set_target(target, op.id, op.ref);
                 op.meta = meta;
                 op.expected = expected_version;
                 b.ops.push_back(std::move(op));
                 return b;
             }, py::arg("id"), py::arg("meta"), py::arg("expected_version") = std::nullopt,
             py::return_value_policy::reference_internal,
             "As DB.update_metadata(): replaces the record, edges included. `id` may be a "
             "key: a str, or a (namespace, key) tuple.")
        .def("link", [](WriteBatch& b, py::object from_id, py::object to_id, const std::string& rel_type,
                        float weight, std::optional<uint64_t> expected_version,
                        bool require_target) -> WriteBatch& {
                 feather::BatchOp op;
                 op.kind = feather::BatchOp::LINK;
                 op.require_target = require_target;
                 set_target(from_id, op.id, op.ref);
                 set_target(to_id, op.to_id, op.to_ref);
                 op.rel_type = rel_type;
                 op.weight = weight;
                 op.expected = expected_version;
                 b.ops.push_back(std::move(op));
                 return b;
             }, py::arg("from_id"), py::arg("to_id"), py::arg("rel_type") = "related_to",
             py::arg("weight") = 1.0f, py::arg("expected_version") = std::nullopt,
             py::arg("require_target") = true,
             py::return_value_policy::reference_internal,
             "As DB.link(), but both ends must be live records once the ops before it "
             "are applied: a batch never leaves an edge to nothing. Either end may be a "
             "key: a str, or a (namespace, key) tuple.")
        .def("forget", [](WriteBatch& b, py::object id, std::optional<uint64_t> expected_version) -> WriteBatch& {
                 feather::BatchOp op;
                 op.kind = feather::BatchOp::FORGET;
                 set_target(id, op.id, op.ref);
                 op.expected = expected_version;
                 b.ops.push_back(std::move(op));
                 return b;
             }, py::arg("id"), py::arg("expected_version") = std::nullopt,
             py::return_value_policy::reference_internal)
        .def("__len__", [](const WriteBatch& b) { return b.ops.size(); })
        .def("ops", [](const WriteBatch& b) {
                 static const char* kinds[] = {"", "put", "update", "link", "forget", "upsert"};
                 py::list out;
                 for (const auto& op : b.ops) {
                     py::dict d;
                     d["kind"] = kinds[op.kind];
                     d["id"] = op.id;
                     d["ref"] = op.ref ? py::object(py::make_tuple(op.ref->ns, op.ref->key)) : py::object(py::none());
                     d["to_id"] = op.to_id;
                     d["to_ref"] = op.to_ref ? py::object(py::make_tuple(op.to_ref->ns, op.to_ref->key)) : py::object(py::none());
                     d["expected"] = op.expected ? py::object(py::int_(*op.expected)) : py::object(py::none());
                     d["meta"] = op.meta;
                     d["vec"] = op.vec.empty() ? py::object(py::none())
                                               : py::object(py::array_t<float>(op.vec.size(), op.vec.data()));
                     d["modality"] = op.modality;
                     d["rel_type"] = op.rel_type;
                     d["weight"] = op.weight;
                     d["require_target"] = op.require_target;
                     out.append(d);
                 }
                 return out;
             }, "The ops, as dicts — for a client that sends the batch elsewhere (RemoteDB).");

    // Fault injection for the write-failure tests: let `n` more fsyncs
    // succeed, then fail every one with `errno` until reset with -1.
    m.def("_fail_fsyncs_after", [](long n, int err) {
              feather::fsio::sync_fault_errno().store(err);
              feather::fsio::sync_faults_after().store(n);
          }, py::arg("n"), py::arg("errno") = EIO);

    auto level_from = [](const std::string& name) -> int {
        const int det = feather_simd::detected_level();
        int l;
        if      (name == "scalar")   l = feather_simd::SCALAR;
        else if (name == "sse2")     l = feather_simd::SSE;
        else if (name == "avx2+fma") l = feather_simd::AVX2;
        else if (name == "avx512f")  l = feather_simd::AVX512;
        else if (name == "neon")     l = feather_simd::NEON;
        else throw std::invalid_argument("unknown SIMD level " + name);
        bool ok = (l == feather_simd::SCALAR) ||
                  (det == feather_simd::NEON ? l == feather_simd::NEON
                                             : (l != feather_simd::NEON && l <= det));
        if (!ok) throw std::invalid_argument("SIMD level " + name + " not supported on this CPU");
        return l;
    };

    m.def("_l2sqr", [level_from](py::array_t<float, py::array::c_style | py::array::forcecast> a,
                                 py::array_t<float, py::array::c_style | py::array::forcecast> b,
                                 const std::string& level) {
        if (a.size() != b.size()) throw std::invalid_argument("length mismatch");
        auto fn = level.empty() ? feather_simd::l2_fn() : feather_simd::l2_fn_for(level_from(level));
        return fn(a.data(), b.data(), static_cast<size_t>(a.size()));
    }, py::arg("a"), py::arg("b"), py::arg("level") = "",
       "Squared L2 with a given kernel (test hook).");

    m.def("_int8_l2sqr", [level_from](py::array_t<int8_t, py::array::c_style | py::array::forcecast> a,
                                      py::array_t<int8_t, py::array::c_style | py::array::forcecast> b,
                                      const std::string& level) {
        if (a.size() != b.size()) throw std::invalid_argument("length mismatch");
        auto fn = level.empty() ? feather_simd::i8_l2_fn() : feather_simd::i8_l2_fn_for(level_from(level));
        return static_cast<long long>(fn(a.data(), b.data(), static_cast<size_t>(a.size())));
    }, py::arg("a"), py::arg("b"), py::arg("level") = "",
       "Integer squared L2 of two int8 vectors with a given kernel (test hook).");

    m.def("_f32_i8_l2sqr", [level_from](py::array_t<float, py::array::c_style | py::array::forcecast> q,
                                        py::array_t<int8_t, py::array::c_style | py::array::forcecast> v,
                                        float scale, const std::string& level) {
        if (q.size() != v.size()) throw std::invalid_argument("length mismatch");
        auto fn = level.empty() ? feather_simd::f32_i8_l2_fn()
                                : feather_simd::f32_i8_l2_fn_for(level_from(level));
        return fn(q.data(), v.data(), scale, static_cast<size_t>(q.size()));
    }, py::arg("q"), py::arg("v"), py::arg("scale"), py::arg("level") = "",
       "Squared L2 between a float query and an int8 row x scale (test hook).");

    // ── ContextType ──────────────────────────────────────────────────
    py::enum_<feather::ContextType>(m, "ContextType")
        .value("FACT",         feather::ContextType::FACT)
        .value("PREFERENCE",   feather::ContextType::PREFERENCE)
        .value("EVENT",        feather::ContextType::EVENT)
        .value("CONVERSATION", feather::ContextType::CONVERSATION)
        .export_values();

    // ── Edge ─────────────────────────────────────────────────────────
    py::class_<feather::Edge>(m, "Edge")
        .def(py::init<>())
        .def(py::init<uint64_t, const std::string&, float>(),
             py::arg("target_id"), py::arg("rel_type") = "related_to", py::arg("weight") = 1.0f)
        .def_readwrite("target_id", &feather::Edge::target_id)
        .def_readwrite("rel_type",  &feather::Edge::rel_type)
        .def_readwrite("weight",    &feather::Edge::weight)
        .def("__repr__", [](const feather::Edge& e) {
            return "<Edge target=" + std::to_string(e.target_id) +
                   " rel=" + e.rel_type +
                   " w=" + std::to_string(e.weight) + ">";
        });

    // ── IncomingEdge ─────────────────────────────────────────────────
    py::class_<feather::IncomingEdge>(m, "IncomingEdge")
        .def_readonly("source_id", &feather::IncomingEdge::source_id)
        .def_readonly("rel_type",  &feather::IncomingEdge::rel_type)
        .def_readonly("weight",    &feather::IncomingEdge::weight)
        .def("__repr__", [](const feather::IncomingEdge& ie) {
            return "<IncomingEdge source=" + std::to_string(ie.source_id) +
                   " rel=" + ie.rel_type +
                   " w=" + std::to_string(ie.weight) + ">";
        });

    // ── Metadata ─────────────────────────────────────────────────────
    py::class_<feather::Metadata>(m, "Metadata")
        .def(py::init<>())
        .def_readwrite("key",             &feather::Metadata::key,
             "Native key: unique among the live records of this namespace_id. Write "
             "with DB.upsert() / WriteBatch.upsert(), read with DB.get_by_key(). "
             "BM25 indexes it with the content.")
        .def_readwrite("version",         &feather::Metadata::version,
             "Compare-and-set version: changes on every write to the record and is "
             "never reused. Pass it to DB.put_if() or a WriteBatch op's expected_version. "
             "Set by the DB: a value assigned here is ignored when the record is written.")
        .def_readwrite("timestamp",       &feather::Metadata::timestamp)
        .def_readwrite("valid_at",        &feather::Metadata::valid_at,
             "Valid time (unix seconds): when what this record says happened or became "
             "true, where that differs from `timestamp` (when it was recorded). 0 = not "
             "given. Filter with valid_after / valid_before; order with "
             "DB.newest(by_valid=True).")
        .def_property_readonly("valid_time", &feather::Metadata::valid_time,
             "valid_at, or the timestamp when valid_at is 0.")
        .def_readwrite("importance",      &feather::Metadata::importance)
        .def_readwrite("type",            &feather::Metadata::type)
        .def_readwrite("source",          &feather::Metadata::source)
        .def_readwrite("content",         &feather::Metadata::content)
        .def_readwrite("tags_json",       &feather::Metadata::tags_json)
        // Atomic members: expose as properties (def_readwrite cannot bind an
        // std::atomic, which is neither copyable nor assignable from Python).
        .def_property("recall_count",
            [](const feather::Metadata& m) { return m.recalls(); },
            [](feather::Metadata& m, uint32_t v) {
                m.recall_count.store(v, std::memory_order_relaxed); })
        .def_property("last_recalled_at",
            [](const feather::Metadata& m) { return m.recalled_at(); },
            [](feather::Metadata& m, uint64_t v) {
                m.last_recalled_at.store(v, std::memory_order_relaxed); })
        .def_readwrite("namespace_id",    &feather::Metadata::namespace_id)
        .def_readwrite("entity_id",       &feather::Metadata::entity_id)
        .def_readwrite("attributes",      &feather::Metadata::attributes)
        .def_readwrite("edges",           &feather::Metadata::edges)
        // Backward compat: read-only `links` property returns target IDs
        .def_property_readonly("links", [](const feather::Metadata& m) {
            std::vector<uint64_t> ids;
            ids.reserve(m.edges.size());
            for (const auto& e : m.edges) ids.push_back(e.target_id);
            return ids;
        })
        // Phase 6 fields
        .def_readwrite("ttl",        &feather::Metadata::ttl)
        .def_readwrite("confidence", &feather::Metadata::confidence)
        // Helpers to avoid the pybind11 map/dict copy gotcha
        .def("set_attribute", [](feather::Metadata& m,
                                  const std::string& key,
                                  const std::string& value) {
            m.attributes[key] = value;
        }, py::arg("key"), py::arg("value"),
        "Set a single attribute key-value pair.")
        .def("get_attribute", [](const feather::Metadata& m,
                                  const std::string& key,
                                  const std::string& default_val) {
            auto it = m.attributes.find(key);
            return it != m.attributes.end() ? it->second : default_val;
        }, py::arg("key"), py::arg("default") = "",
        "Get an attribute value by key, or default if absent.")
        .def("to_dict", &metadata_to_dict,
             "All fields as a plain dict (links = edge target ids).");

    // ── ScoringConfig ────────────────────────────────────────────────
    py::class_<feather::ScoringConfig>(m, "ScoringConfig")
        .def(py::init<float, float, float>(),
             py::arg("half_life") = 30.0f,
             py::arg("weight")    = 0.3f,
             py::arg("min")       = 0.0f)
        .def_readwrite("decay_half_life_days", &feather::ScoringConfig::decay_half_life_days)
        .def_readwrite("time_weight",          &feather::ScoringConfig::time_weight)
        .def_readwrite("min_weight",           &feather::ScoringConfig::min_weight);

    // ── SearchFilter ─────────────────────────────────────────────────
    py::class_<feather::SearchFilter>(m, "SearchFilter")
        .def(py::init<>())
        .def_readwrite("types",            &feather::SearchFilter::types)
        .def_readwrite("source",           &feather::SearchFilter::source)
        .def_readwrite("source_prefix",    &feather::SearchFilter::source_prefix)
        .def_readwrite("timestamp_after",  &feather::SearchFilter::timestamp_after)
        .def_readwrite("timestamp_before", &feather::SearchFilter::timestamp_before)
        .def_readwrite("valid_after",      &feather::SearchFilter::valid_after)
        .def_readwrite("valid_before",     &feather::SearchFilter::valid_before)
        .def_readwrite("importance_gte",   &feather::SearchFilter::importance_gte)
        .def_readwrite("confidence_gte",   &feather::SearchFilter::confidence_gte)
        .def_readwrite("tags_contains",    &feather::SearchFilter::tags_contains)
        .def_readwrite("namespace_id",     &feather::SearchFilter::namespace_id)
        .def_readwrite("namespace_prefix", &feather::SearchFilter::namespace_prefix)
        .def_readwrite("entity_id",        &feather::SearchFilter::entity_id)
        .def_readwrite("attributes_match", &feather::SearchFilter::attributes_match);

    // ── SearchResult ─────────────────────────────────────────────────
    py::class_<feather::DB::SearchResult>(m, "SearchResult")
        .def_readonly("id",       &feather::DB::SearchResult::id)
        .def_readonly("score",    &feather::DB::SearchResult::score)
        .def_readonly("metadata", &feather::DB::SearchResult::metadata)
        .def_property_readonly("cosine", [](const feather::DB::SearchResult& r) -> py::object {
            if (std::isnan(r.cosine)) return py::none();
            return py::float_(r.cosine);
        }, "Exact cosine(query, stored vector) if searched with with_cosine=True, else None.")
        .def("to_dict", [](const feather::DB::SearchResult& r, bool include_metadata) {
            py::dict d;
            d["id"]    = r.id;
            d["score"] = r.score;
            d["cosine"] = std::isnan(r.cosine) ? py::object(py::none()) : py::object(py::float_(r.cosine));
            if (include_metadata) d["metadata"] = metadata_to_dict(r.metadata);
            return d;
        }, py::arg("include_metadata") = true);

    // ── ContextNode / ContextEdge / ContextChainResult ───────────────
    py::class_<feather::DB::ContextNode>(m, "ContextNode")
        .def_readonly("id",         &feather::DB::ContextNode::id)
        .def_readonly("score",      &feather::DB::ContextNode::score)
        .def_readonly("similarity", &feather::DB::ContextNode::similarity)
        .def_readonly("hop",        &feather::DB::ContextNode::hop)
        .def_readonly("metadata",   &feather::DB::ContextNode::metadata);

    py::class_<feather::DB::ContextEdge>(m, "ContextEdge")
        .def_readonly("source",   &feather::DB::ContextEdge::source)
        .def_readonly("target",   &feather::DB::ContextEdge::target)
        .def_readonly("rel_type", &feather::DB::ContextEdge::rel_type)
        .def_readonly("weight",   &feather::DB::ContextEdge::weight);

    py::class_<feather::DB::ContextChainResult>(m, "ContextChainResult")
        .def_readonly("nodes", &feather::DB::ContextChainResult::nodes)
        .def_readonly("edges", &feather::DB::ContextChainResult::edges);

    // ── DB ───────────────────────────────────────────────────────────
    // Owned by Python: when the last reference goes, ~DB() runs — checkpoint
    // (a writer that was never closed), release the lock, free the memory.
    // This used to be py::nodelete, so nothing a DB held was ever freed: every
    // open of a 39 MB file cost ~50 MB until the process exited.
    //
    // The GIL is released around everything that can block: the open (a load
    // can take seconds), every call that waits on the data lock — a reader
    // queues behind a writer, and the lock is phase-fair — and, on pybind11 3,
    // the destructor's checkpoint. Holding it there froze every Python thread
    // in the process for the duration. Nothing released touches a Python
    // object: arguments are converted before, results after.
    py::class_<feather::DB, std::unique_ptr<feather::DB>>(m, "DB"
#if PYBIND11_VERSION_MAJOR >= 3
        , py::release_gil_before_calling_cpp_dtor()
#endif
        )
        .def_static("open", &feather::DB::open,
                    py::arg("path"), py::arg("dim") = 768,
                    py::arg("read_only") = false, py::call_guard<py::gil_scoped_release>(),
                    "Open a .feather file.\n\n"
                    "Takes an exclusive inter-process write lock; a second "
                    "writer is refused with the holder's pid. read_only=True "
                    "takes a shared lock instead, so many readers may hold the "
                    "file at once, and every mutation raises.\n\n"
                    "Feather is single-writer by design: for several agents in "
                    "ONE process, share this handle and separate them by "
                    "namespace, rather than opening the file twice. Set "
                    "FEATHER_LOCK=0 to disable locking (concurrent writers then "
                    "silently discard each other's records).")

        // -- Ingestion --
        .def("add", [](feather::DB& db, uint64_t id,
                        py::array_t<float, py::array::c_style | py::array::forcecast> vec,
                        const std::optional<feather::Metadata>& meta,
                        const std::string& modality) {
            auto buf = vec.request();
            const float* ptr = static_cast<const float*>(buf.ptr);
            std::vector<float> v(ptr, ptr + buf.size);
            py::gil_scoped_release rel;   // C++ insert runs without the GIL
            db.add(id, v, meta ? *meta : feather::Metadata(), modality);
        }, py::arg("id"), py::arg("vec"),
           py::arg("meta") = std::nullopt,
           py::arg("modality") = "text")

        // -- Bulk ingestion: parallel HNSW build --
        .def("add_batch", [](feather::DB& db,
                              const std::vector<uint64_t>& ids,
                              py::array_t<float, py::array::c_style | py::array::forcecast> vecs,
                              const std::optional<std::vector<feather::Metadata>>& metas,
                              const std::string& modality) {
            auto buf = vecs.request();
            if (buf.ndim != 2)
                throw std::runtime_error("add_batch: vecs must be a 2D array (N x dim)");
            size_t n = static_cast<size_t>(buf.shape[0]);
            size_t dim = static_cast<size_t>(buf.shape[1]);
            if (ids.size() != n)
                throw std::runtime_error("add_batch: len(ids) != vecs.shape[0]");
            const float* ptr = static_cast<const float*>(buf.ptr);
            std::vector<std::vector<float>> vv(n);
            for (size_t i = 0; i < n; ++i)
                vv[i].assign(ptr + i * dim, ptr + (i + 1) * dim);
            std::vector<feather::Metadata> mv =
                metas ? *metas : std::vector<feather::Metadata>();
            {
                py::gil_scoped_release rel;   // run the parallel insert without the GIL
                db.add_batch(ids, vv, mv, modality);
            }
        }, py::arg("ids"), py::arg("vecs"), py::arg("metas") = std::nullopt,
           py::arg("modality") = "text",
           "Bulk-insert N records (vecs: N x dim float32). HNSW graph built in parallel.")

        // -- Search --
        .def("search", [](feather::DB& db,
                           py::array_t<float, py::array::c_style | py::array::forcecast> q,
                           size_t k,
                           const feather::SearchFilter* filter,
                           const feather::ScoringConfig* scoring,
                           const std::string& modality,
                           bool record_salience,
                           bool with_cosine) {
            auto buf = q.request();
            const float* ptr = static_cast<const float*>(buf.ptr);
            std::vector<float> query(ptr, ptr + buf.size);
            // Drop the GIL for the C++ query. DB::search takes the data lock in
            // shared mode, so N server threads genuinely search in parallel;
            // holding the GIL here capped the whole process at one core.
            py::gil_scoped_release rel;
            return db.search(query, k, filter, scoring, modality, record_salience, with_cosine);
        }, py::arg("q"), py::arg("k") = 5,
           py::arg("filter") = nullptr, py::arg("scoring") = nullptr,
           py::arg("modality") = "text", py::arg("record_salience") = true,
           py::arg("with_cosine") = false)

        // -- Graph --
        .def("link", &feather::DB::link,
             py::arg("from_id"), py::arg("to_id"),
             py::arg("rel_type") = "related_to",
             py::arg("weight") = 1.0f, py::call_guard<py::gil_scoped_release>())

        .def("get_edges",    &feather::DB::get_edges,    py::arg("id"), py::call_guard<py::gil_scoped_release>())
        .def("get_incoming", &feather::DB::get_incoming, py::arg("id"), py::call_guard<py::gil_scoped_release>())

        .def("auto_link", &feather::DB::auto_link,
             py::arg("modality")   = "text",
             py::arg("threshold")  = 0.80f,
             py::arg("rel_type")   = "related_to",
             py::arg("candidates") = 15,
             "Auto-create edges between records whose vector similarity exceeds threshold.",
             py::call_guard<py::gil_scoped_release>())

        .def("context_chain", [](feather::DB& db, py::array_t<float, py::array::c_style | py::array::forcecast> q,
                                  size_t k, int hops, const std::string& modality) {
            auto buf = q.request();
            const float* ptr = static_cast<const float*>(buf.ptr);
            std::vector<float> query(ptr, ptr + buf.size);
            py::gil_scoped_release rel;
            return db.context_chain(query, k, hops, modality);
        }, py::arg("q"), py::arg("k") = 5, py::arg("hops") = 2,
           py::arg("modality") = "text",
           "Vector search + n-hop graph expansion. Returns ContextChainResult.")

        .def("export_graph_json", &feather::DB::export_graph_json,
             py::arg("namespace_filter") = "",
             py::arg("entity_filter")    = "",
             "Export graph as D3/Cytoscape-compatible JSON string.",
             py::call_guard<py::gil_scoped_release>())

        // -- Metadata --
        .def("touch",             &feather::DB::touch,             py::arg("id"), py::call_guard<py::gil_scoped_release>())
        .def("get_metadata",      &feather::DB::get_metadata,      py::arg("id"), py::arg("include_dead") = false,
             py::call_guard<py::gil_scoped_release>(),
             "The record's metadata, or None if it does not exist or was forgotten "
             "(include_dead=True returns a forgotten record's tombstone until the next save).")
        .def("update_metadata",   &feather::DB::update_metadata,   py::arg("id"), py::arg("meta"), py::call_guard<py::gil_scoped_release>())
        .def("update_importance", &feather::DB::update_importance, py::arg("id"), py::arg("importance"), py::call_guard<py::gil_scoped_release>())
        .def("id_for", &feather::DB::id_for, py::arg("key"), py::arg("namespace") = "",
             py::call_guard<py::gil_scoped_release>(),
             "The id of the live record holding `key` in `namespace`, or None.")
        .def("get_by_key", &feather::DB::get_by_key, py::arg("key"), py::arg("namespace") = "",
             py::call_guard<py::gil_scoped_release>(),
             "The metadata of the live record holding `key` in `namespace`, or None.")
        .def("upsert", [](feather::DB& db, const feather::Metadata& meta,
                          std::optional<py::array_t<float, py::array::c_style | py::array::forcecast>> vec,
                          const std::string& modality, std::optional<uint64_t> expected_version) {
                 std::vector<float> v;
                 if (vec) v = to_vec(*vec);
                 py::gil_scoped_release rel;
                 return db.upsert(meta, v, modality, expected_version);
             }, py::arg("meta"), py::arg("vec") = std::nullopt, py::arg("modality") = "text",
             py::arg("expected_version") = std::nullopt,
             "Write the record keyed (meta.namespace_id, meta.key): the existing one, or a "
             "new one with a fresh id — never a hash of the key, so two keys never share a "
             "record. With vec as add(); without, the metadata replaces the record's, "
             "keeping its edges when meta has none. Returns (id, version).")
        .def("newest", [](const feather::DB& db, const std::string& ns, size_t limit, bool prefix,
                          py::object before, bool by_valid) {
                 std::optional<std::pair<int64_t, uint64_t>> cursor;
                 if (py::isinstance<py::tuple>(before)) {
                     auto t = before.cast<py::tuple>();
                     cursor = std::make_pair(t[0].cast<int64_t>(), t[1].cast<uint64_t>());
                 } else if (!before.is_none()) {
                     cursor = std::make_pair(before.cast<int64_t>(), uint64_t{0});   // older than ts
                 }
                 py::gil_scoped_release rel;
                 return db.newest(ns, limit, prefix, cursor, by_valid);
             }, py::arg("namespace") = "", py::arg("limit") = 10, py::arg("prefix") = false,
             py::arg("before") = py::none(), py::arg("by_valid") = false,
             "Ids of the `limit` newest live records in `namespace` (prefix=True: and every "
             "namespace below it), newest first by (timestamp, id). `before` pages: a "
             "timestamp, or the (timestamp, id) of the last record of the previous page. "
             "by_valid=True orders by (valid_time, id) instead — when it happened, not "
             "when it was written — and `before` is then a valid time. "
             "An ordered index walk, not a scan and sort.")
        .def("namespace_ann_info", [](const feather::DB& db) {
                 feather::DB::NsAnnInfo i;
                 { py::gil_scoped_release rel; i = db.namespace_ann_info(); }
                 py::dict d;
                 d["graphs"] = i.graphs; d["nodes"] = i.nodes; d["pending"] = i.pending;
                 d["min_records"] = i.min_records;
                 return d;
             }, "Per-namespace ANN graphs: how many are built, their nodes, how many are "
                "queued, and the namespace size (FEATHER_NS_ANN_MIN) that gets one.")
        .def("set_namespace_ann_min", &feather::DB::set_namespace_ann_min, py::arg("n"),
             py::call_guard<py::gil_scoped_release>(),
             "Namespaces with at least n records (in a modality) get their own ANN graph, "
             "so a filtered search inside them walks it instead of scanning every record. "
             "0 turns it off: every filtered search is an exact scan.")
        .def("wait_for_namespace_graphs", &feather::DB::wait_for_namespace_graphs,
             py::arg("timeout") = 0.0, py::call_guard<py::gil_scoped_release>(),
             "Block until no namespace graph is queued or building (timeout seconds, 0 = none).")
        .def("version", &feather::DB::version, py::arg("id"), py::call_guard<py::gil_scoped_release>(),
             "The record's compare-and-set version; 0 if it does not exist or was forgotten.")
        .def("put_if", [](feather::DB& db, uint64_t id, const feather::Metadata& meta,
                          uint64_t expected_version,
                          std::optional<py::array_t<float, py::array::c_style | py::array::forcecast>> vec,
                          const std::string& modality) {
                 std::vector<float> v;
                 if (vec) v = to_vec(*vec);
                 py::gil_scoped_release rel;
                 return db.put_if(id, meta, expected_version, v, modality);
             }, py::arg("id"), py::arg("meta"), py::arg("expected_version"),
             py::arg("vec") = std::nullopt, py::arg("modality") = "text",
             "Replace the record (edges included) only if its version is still "
             "expected_version (0 = it must not exist). With vec, the vector too, atomically. "
             "Returns the new version; raises VersionConflictError.")
        .def("write", [](feather::DB& db, const WriteBatch& b,
                         std::optional<std::string> idempotency_key, const std::string& fingerprint,
                         const std::string& note) {
                 py::gil_scoped_release rel;
                 return db.write_batch(b.ops, idempotency_key.value_or(""), fingerprint, note);
             }, py::arg("batch"), py::arg("idempotency_key") = std::nullopt, py::arg("fingerprint") = "",
             py::arg("note") = "",
             "Apply a WriteBatch atomically: every op or none, one WAL write, one lock. "
             "With idempotency_key, a repeat returns the first result (replayed=True, with "
             "the `note` the first attempt stored, up to 64 KiB) instead of applying again; "
             "a different fingerprint under the same key raises IdempotencyKeyReusedError.")
        .def("idempotent_result", &feather::DB::idempotent_result, py::arg("key"), py::arg("fingerprint") = "",
             py::call_guard<py::gil_scoped_release>(),
             "The stored BatchResult of an earlier write with this key, or None. For a caller "
             "whose own checks would answer a retry differently (a delete retried after it "
             "succeeded finds no record); write() checks again atomically either way.")
        .def("get_vector", [](feather::DB& db, uint64_t id, const std::string& modality) {
            std::vector<float> vec;
            {
                py::gil_scoped_release rel;
                vec = db.get_vector(id, modality);
            }
            return py::array_t<float>(vec.size(), vec.data());
        }, py::arg("id"), py::arg("modality") = "text")
        .def("get_all_ids", &feather::DB::get_all_ids, py::arg("modality") = "text", py::call_guard<py::gil_scoped_release>())
        .def("all_ids", &feather::DB::all_ids,
             "Every id that has metadata, across all modalities.",
             py::call_guard<py::gil_scoped_release>())
        .def("modality_names", &feather::DB::modality_names,
             "Names of the modality indices present in this DB.",
             py::call_guard<py::gil_scoped_release>())

        // -- Secondary metadata indexes (O(matches) lookups) --
        .def("ids_in_namespace", &feather::DB::ids_in_namespace, py::arg("namespace_id"),
             "Live record ids in a namespace (O(matches) via secondary index).",
             py::call_guard<py::gil_scoped_release>())
        .def("ids_for_entity", &feather::DB::ids_for_entity, py::arg("entity_id"),
             "Live record ids for an entity (O(matches) via secondary index).",
             py::call_guard<py::gil_scoped_release>())
        .def("ids_with_attribute", &feather::DB::ids_with_attribute,
             py::arg("key"), py::arg("value"),
             "Live record ids carrying attribute key=value (O(matches)).",
             py::call_guard<py::gil_scoped_release>())
        .def("namespace_size", &feather::DB::namespace_size, py::arg("namespace_id"),
             "Live record count in a namespace (O(1)).",
             py::call_guard<py::gil_scoped_release>())
        .def("list_namespaces", &feather::DB::list_namespaces,
             "All namespace ids that currently have at least one live record.",
             py::call_guard<py::gil_scoped_release>())

        // -- BM25 keyword search --
        .def("keyword_search", [](feather::DB& db, const std::string& query, size_t k,
                                   feather::SearchFilter* filter) {
            py::gil_scoped_release rel;
            return db.keyword_search(query, k, filter);
        }, py::arg("query"), py::arg("k") = 10, py::arg("filter") = nullptr,
           "BM25 keyword search over content field. Returns list of SearchResult.")

        // -- Hybrid search: BM25 + vector via RRF --
        .def("hybrid_search", [](feather::DB& db, py::array_t<float, py::array::c_style | py::array::forcecast> q,
                                   const std::string& query, size_t k,
                                   size_t rrf_k,
                                   feather::SearchFilter* filter,
                                   feather::ScoringConfig* scoring,
                                   const std::string& modality) {
            py::buffer_info buf = q.request();
            std::vector<float> query_vec(static_cast<float*>(buf.ptr),
                                         static_cast<float*>(buf.ptr) + buf.size);
            py::gil_scoped_release rel;
            return db.hybrid_search(query_vec, query, k, rrf_k, filter, scoring, modality);
        }, py::arg("vec"), py::arg("query"), py::arg("k") = 10,
           py::arg("rrf_k") = 60, py::arg("filter") = nullptr,
           py::arg("scoring") = nullptr, py::arg("modality") = "text",
           "Hybrid search: BM25 + dense vector merged via Reciprocal Rank Fusion (RRF).")

        // -- Compact (rebuild HNSW without dead records) --
        .def("compact", &feather::DB::compact,
             "Rebuild HNSW indices removing dead (forgotten/_deleted) records and "
             "orphaned vectors. Returns count of records removed.",
             py::call_guard<py::gil_scoped_release>())
        .def("set_auto_compact", &feather::DB::set_auto_compact, py::arg("ratio"),
             "Auto-rebuild a modality index when its deleted/total ratio crosses "
             "`ratio` after forget/purge/expire. 0 disables (default).",
             py::call_guard<py::gil_scoped_release>())
        .def("get_auto_compact", &feather::DB::get_auto_compact,
             "Current auto-compaction threshold (0 = disabled).",
             py::call_guard<py::gil_scoped_release>())

        // -- On-disk int8 quantization (file format v7) --
        .def("set_quantized", &feather::DB::set_quantized,
             py::arg("modality"), py::arg("on") = true,
             "Persist a modality's vectors as int8 + per-vector scale (~4x "
             "smaller on disk). Takes effect on next save(); dequantized on load.",
             py::call_guard<py::gil_scoped_release>())
        .def("is_quantized", &feather::DB::is_quantized, py::arg("modality"),
             "Whether a modality is persisted with int8 quantization.",
             py::call_guard<py::gil_scoped_release>())

        // -- In-RAM int8 quantization (4x less memory) --
        .def("set_int8_ram", [](feather::DB& db, const std::string& modality,
                                py::object max_abs, uint32_t refine) {
                 const float clip = max_abs.is_none() ? 0.0f : max_abs.cast<float>();
                 py::gil_scoped_release rel;
                 db.set_int8_ram(modality, clip, refine);
             },
             py::arg("modality"), py::arg("max_abs") = py::none(), py::arg("refine") = 2,
             "Store this modality's vectors as int8 IN MEMORY (~4x less RAM) under "
             "one global scale = clip/127. max_abs=None (default) trains the clip "
             "from the data: now if the modality holds >= 1000 vectors "
             "(FEATHER_INT8_TRAIN_MIN), else once it does — float until then. "
             "A number fixes the clip; components beyond it are clamped. "
             "refine > 0 keeps float32 originals in the file (read from disk, not "
             "RAM, once saved) and re-ranks the top k*refine int8 candidates by "
             "exact distance; 0 keeps none. Converting a non-empty float modality "
             "keeps its graph. Once int8, calling again only changes refine.")
        .def("is_int8_ram", &feather::DB::is_int8_ram, py::arg("modality"),
             "Whether a modality's in-memory index stores int8 vectors.",
             py::call_guard<py::gil_scoped_release>())
        .def("int8_info", [](const feather::DB& db, const std::string& modality) {
            feather::DB::Int8Info i;
            {
                py::gil_scoped_release rel;
                i = db.int8_info(modality);
            }
            py::dict d;
            d["int8"]              = i.int8;
            d["training"]          = i.training;
            d["clip"]              = i.clip;
            d["refine"]            = i.refine;
            d["originals_on_disk"] = i.originals_on_disk;
            d["originals_in_wal"]  = i.originals_in_wal;
            d["originals_in_ram"]  = i.originals_in_ram;
            d["train_min"]         = i.train_min;
            return d;
        }, py::arg("modality"),
           "In-RAM int8 state of a modality: whether it is int8 or still waiting "
           "to train, its clip (scale x 127), the refine factor, and where its float "
           "originals are: in the file, still only in the WAL, or copied in RAM "
           "(after converting a float modality) until the next save().")

        // -- Memory lifecycle --
        .def("forget",         &feather::DB::forget,         py::arg("id"),
             "Soft-delete: remove from search + blank content. Graph shell preserved.",
             py::call_guard<py::gil_scoped_release>())
        .def("purge",          &feather::DB::purge,          py::arg("namespace_id"),
             "Hard-delete all nodes in namespace_id. Returns count of removed nodes.",
             py::call_guard<py::gil_scoped_release>())
        .def("forget_expired", &feather::DB::forget_expired,
             "Soft-delete all nodes where ttl>0 and now > timestamp+ttl. Returns count.",
             py::call_guard<py::gil_scoped_release>())

        // -- Persistence & info --
        .def("save", &feather::DB::save, py::call_guard<py::gil_scoped_release>(),
             "Commit what changed since the last save. File format v10 appends "
             "only the change; the whole file is rewritten only when it is still "
             "v3-v9, after purge()/compact()/set_quantized(), or once superseded "
             "records outweigh live data.")
        .def("vacuum", &feather::DB::vacuum, py::call_guard<py::gil_scoped_release>(),
             "Rewrite the file now, dropping every superseded record (including "
             "earlier versions of updated and forgotten records).")
        .def("storage_stats", [](const feather::DB& db) {
            feather::DB::StorageStats s;
            {
                py::gil_scoped_release rel;   // waits out a save in progress
                s = db.storage_stats();
            }
            py::dict d;
            d["format"]              = s.format;
            d["generation"]          = s.generation;
            d["file_bytes"]          = s.file_bytes;
            d["live_bytes_estimate"] = s.live_bytes_estimate;
            d["last_commit_bytes"]   = s.last_commit_bytes;
            d["last_commit_full"]    = s.last_commit_full;
            d["commits"]             = s.commits;
            d["vacuums"]             = s.vacuums;
            d["mapped_bytes"]        = s.mapped_bytes;
            return d;
        }, "On-disk format, commit generation, committed bytes, and what the last save wrote.")
        .def("close", &feather::DB::close, py::arg("save") = true,
             "Checkpoint (unless save=False), release the WAL and the inter-process "
             "file lock, and free the handle's memory. Any later call on it raises. "
             "Dropping the last reference does the same (with a checkpoint), but "
             "close() does it at a moment you choose. Idempotent.",
             py::call_guard<py::gil_scoped_release>())
        .def("is_read_only", &feather::DB::is_read_only)
        .def("is_closed",    &feather::DB::is_closed)
        .def("write_error", [](const feather::DB& db) -> py::object {
            std::string e = db.write_error();
            return e.empty() ? py::object(py::none()) : py::object(py::str(e));
        }, "None while the handle accepts writes. After a WAL write or fsync "
           "failure: the cause. Every later write and save then raises until "
           "the file is reopened; close() releases it without a checkpoint.")
        .def_property_readonly("closed", &feather::DB::is_closed)
        .def("__enter__", [](feather::DB& db) -> feather::DB& { return db; },
             py::return_value_policy::reference)
        .def("__exit__", [](feather::DB& db, py::object, py::object, py::object) {
            py::gil_scoped_release rel;
            db.close(true);
        })
        .def("wal_size", &feather::DB::wal_size,
             "Bytes currently in the WAL (+ <wal>.old). Use it to checkpoint on "
             "WAL size instead of saving after every mutation.",
             py::call_guard<py::gil_scoped_release>())
        .def("wal_fsync_count", &feather::DB::wal_fsync_count,
             "WAL fsyncs issued so far (group commit shares one across writers).",
             py::call_guard<py::gil_scoped_release>())
        .def("is_compacting", &feather::DB::is_compacting,
             "True while a compaction is building its new indexes.",
             py::call_guard<py::gil_scoped_release>())
        .def("wait_for_compaction", &feather::DB::wait_for_compaction,
             py::arg("timeout") = 0.0,
             "Block until background auto-compaction is idle. timeout<=0 waits "
             "forever; returns False if the timeout elapsed first.",
             py::call_guard<py::gil_scoped_release>())
        .def("size", &feather::DB::size, py::call_guard<py::gil_scoped_release>())
        .def("dim",  &feather::DB::dim, py::arg("modality") = "text", py::call_guard<py::gil_scoped_release>())

        // -- HNSW tuning --
        .def("set_ef", &feather::DB::set_ef,
             py::arg("ef"), py::arg("modality") = "",
             "Set HNSW search beam width. Higher ef -> better recall, "
             "slower search. Default is 50. Pass modality=\"\" to apply "
             "to all modalities.",
             py::call_guard<py::gil_scoped_release>())
        .def("get_ef", &feather::DB::get_ef,
             py::arg("modality") = "text",
             "Current HNSW search beam width for the given modality.",
             py::call_guard<py::gil_scoped_release>());
}
