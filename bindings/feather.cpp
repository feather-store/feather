#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../include/feather.h"

namespace py = pybind11;

PYBIND11_MODULE(core, m) {
    m.doc() = "Feather: SQLite for Vectors";

    py::enum_<feather::ContextType>(m, "ContextType")
        .value("FACT", feather::ContextType::FACT)
        .value("PREFERENCE", feather::ContextType::PREFERENCE)
        .value("EVENT", feather::ContextType::EVENT)
        .value("CONVERSATION", feather::ContextType::CONVERSATION)
        .export_values();

    py::class_<feather::Metadata>(m, "Metadata")
        .def(py::init<>())
        .def_readwrite("timestamp", &feather::Metadata::timestamp)
        .def_readwrite("importance", &feather::Metadata::importance)
        .def_readwrite("type", &feather::Metadata::type)
        .def_readwrite("source", &feather::Metadata::source)
        .def_readwrite("content", &feather::Metadata::content)
        .def_readwrite("tags_json", &feather::Metadata::tags_json);

    py::class_<feather::ScoringConfig>(m, "ScoringConfig")
        .def(py::init<float, float, float>(), py::arg("half_life") = 30.0f, py::arg("weight") = 0.3f, py::arg("min") = 0.0f)
        .def_readwrite("decay_half_life_days", &feather::ScoringConfig::decay_half_life_days)
        .def_readwrite("time_weight", &feather::ScoringConfig::time_weight)
        .def_readwrite("min_weight", &feather::ScoringConfig::min_weight);

    py::class_<feather::SearchFilter>(m, "SearchFilter")
        .def(py::init<>())
        .def_readwrite("types", &feather::SearchFilter::types)
        .def_readwrite("source", &feather::SearchFilter::source)
        .def_readwrite("source_prefix", &feather::SearchFilter::source_prefix)
        .def_readwrite("timestamp_after", &feather::SearchFilter::timestamp_after)
        .def_readwrite("timestamp_before", &feather::SearchFilter::timestamp_before)
        .def_readwrite("importance_gte", &feather::SearchFilter::importance_gte)
        .def_readwrite("tags_contains", &feather::SearchFilter::tags_contains);

    py::class_<feather::DB::SearchResult>(m, "SearchResult")
        .def_readonly("id", &feather::DB::SearchResult::id)
        .def_readonly("score", &feather::DB::SearchResult::score)
        .def_readonly("metadata", &feather::DB::SearchResult::metadata);

    py::class_<feather::DB, std::unique_ptr<feather::DB, py::nodelete>>(m, "DB")
        .def_static("open", &feather::DB::open, py::arg("path"), py::arg("dim") = 768)

        .def("add", [](feather::DB& db, uint64_t id, py::array_t<float> vec, const std::optional<feather::Metadata>& meta) {
            auto buf = vec.request();
            if (buf.size != db.dim()) throw std::runtime_error("Dimension mismatch");
            const float* ptr = static_cast<const float*>(buf.ptr);
            std::vector<float> vec_copy(ptr, ptr + buf.size);
            db.add(id, vec_copy, meta ? *meta : feather::Metadata());
        }, py::arg("id"), py::arg("vec"), py::arg("meta") = std::nullopt)

        .def("search", [](const feather::DB& db, py::array_t<float> q, size_t k = 5,
                          const feather::SearchFilter* filter = nullptr,
                          const feather::ScoringConfig* scoring = nullptr) {
            auto buf = q.request();
            if (buf.size != db.dim()) throw std::runtime_error("Query dimension mismatch");
            const float* ptr = static_cast<const float*>(buf.ptr);
            std::vector<float> query(ptr, ptr + buf.size);
            return db.search(query, k, filter, scoring);
        }, py::arg("q"), py::arg("k") = 5, py::arg("filter") = nullptr, py::arg("scoring") = nullptr)

        .def("get_metadata", &feather::DB::get_metadata, py::arg("id"))
        .def("save", &feather::DB::save)
        .def("dim", &feather::DB::dim);
}
