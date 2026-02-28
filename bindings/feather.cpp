#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../include/feather.h"

namespace py = pybind11;

PYBIND11_MODULE(core, m) {
    m.doc() = "Feather: SQLite for Vectors — Living Context Engine";

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
        .def_readwrite("timestamp",       &feather::Metadata::timestamp)
        .def_readwrite("importance",      &feather::Metadata::importance)
        .def_readwrite("type",            &feather::Metadata::type)
        .def_readwrite("source",          &feather::Metadata::source)
        .def_readwrite("content",         &feather::Metadata::content)
        .def_readwrite("tags_json",       &feather::Metadata::tags_json)
        .def_readwrite("recall_count",    &feather::Metadata::recall_count)
        .def_readwrite("last_recalled_at",&feather::Metadata::last_recalled_at)
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
        "Get an attribute value by key, or default if absent.");

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
        .def_readwrite("importance_gte",   &feather::SearchFilter::importance_gte)
        .def_readwrite("tags_contains",    &feather::SearchFilter::tags_contains)
        .def_readwrite("namespace_id",     &feather::SearchFilter::namespace_id)
        .def_readwrite("entity_id",        &feather::SearchFilter::entity_id)
        .def_readwrite("attributes_match", &feather::SearchFilter::attributes_match);

    // ── SearchResult ─────────────────────────────────────────────────
    py::class_<feather::DB::SearchResult>(m, "SearchResult")
        .def_readonly("id",       &feather::DB::SearchResult::id)
        .def_readonly("score",    &feather::DB::SearchResult::score)
        .def_readonly("metadata", &feather::DB::SearchResult::metadata);

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
    py::class_<feather::DB, std::unique_ptr<feather::DB, py::nodelete>>(m, "DB")
        .def_static("open", &feather::DB::open,
                    py::arg("path"), py::arg("dim") = 768)

        // -- Ingestion --
        .def("add", [](feather::DB& db, uint64_t id,
                        py::array_t<float> vec,
                        const std::optional<feather::Metadata>& meta,
                        const std::string& modality) {
            auto buf = vec.request();
            const float* ptr = static_cast<const float*>(buf.ptr);
            std::vector<float> v(ptr, ptr + buf.size);
            db.add(id, v, meta ? *meta : feather::Metadata(), modality);
        }, py::arg("id"), py::arg("vec"),
           py::arg("meta") = std::nullopt,
           py::arg("modality") = "text")

        // -- Search --
        .def("search", [](feather::DB& db, py::array_t<float> q, size_t k,
                           const feather::SearchFilter* filter,
                           const feather::ScoringConfig* scoring,
                           const std::string& modality) {
            auto buf = q.request();
            const float* ptr = static_cast<const float*>(buf.ptr);
            std::vector<float> query(ptr, ptr + buf.size);
            return db.search(query, k, filter, scoring, modality);
        }, py::arg("q"), py::arg("k") = 5,
           py::arg("filter") = nullptr, py::arg("scoring") = nullptr,
           py::arg("modality") = "text")

        // -- Graph --
        .def("link", &feather::DB::link,
             py::arg("from_id"), py::arg("to_id"),
             py::arg("rel_type") = "related_to",
             py::arg("weight") = 1.0f)

        .def("get_edges",    &feather::DB::get_edges,    py::arg("id"))
        .def("get_incoming", &feather::DB::get_incoming, py::arg("id"))

        .def("auto_link", &feather::DB::auto_link,
             py::arg("modality")   = "text",
             py::arg("threshold")  = 0.80f,
             py::arg("rel_type")   = "related_to",
             py::arg("candidates") = 15,
             "Auto-create edges between records whose vector similarity exceeds threshold.")

        .def("context_chain", [](feather::DB& db, py::array_t<float> q,
                                  size_t k, int hops, const std::string& modality) {
            auto buf = q.request();
            const float* ptr = static_cast<const float*>(buf.ptr);
            std::vector<float> query(ptr, ptr + buf.size);
            return db.context_chain(query, k, hops, modality);
        }, py::arg("q"), py::arg("k") = 5, py::arg("hops") = 2,
           py::arg("modality") = "text",
           "Vector search + n-hop graph expansion. Returns ContextChainResult.")

        .def("export_graph_json", &feather::DB::export_graph_json,
             py::arg("namespace_filter") = "",
             py::arg("entity_filter")    = "",
             "Export graph as D3/Cytoscape-compatible JSON string.")

        // -- Metadata --
        .def("touch",             &feather::DB::touch,             py::arg("id"))
        .def("get_metadata",      &feather::DB::get_metadata,      py::arg("id"))
        .def("update_metadata",   &feather::DB::update_metadata,   py::arg("id"), py::arg("meta"))
        .def("update_importance", &feather::DB::update_importance, py::arg("id"), py::arg("importance"))
        .def("get_vector", [](feather::DB& db, uint64_t id, const std::string& modality) {
            auto vec = db.get_vector(id, modality);
            return py::array_t<float>(vec.size(), vec.data());
        }, py::arg("id"), py::arg("modality") = "text")
        .def("get_all_ids", &feather::DB::get_all_ids, py::arg("modality") = "text")

        // -- Persistence & info --
        .def("save", &feather::DB::save)
        .def("size", &feather::DB::size)
        .def("dim",  &feather::DB::dim, py::arg("modality") = "text");
}
