#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../include/feather.h"

namespace py = pybind11;

PYBIND11_MODULE(core, m) {
    m.doc() = "Feather: SQLite for Vectors";

    py::class_<feather::DB, std::unique_ptr<feather::DB, py::nodelete>>(m, "DB")
        .def_static("open", &feather::DB::open, py::arg("path"), py::arg("dim") = 768)

        .def("add", [](feather::DB& db, uint64_t id, py::array_t<float> vec) {
            auto buf = vec.request();
            if (buf.size != db.dim()) throw std::runtime_error("Dimension mismatch");
            const float* ptr = static_cast<const float*>(buf.ptr);
            std::vector<float> vec_copy(ptr, ptr + buf.size);
            db.add(id, vec_copy);
        }, py::arg("id"), py::arg("vec"))
        .def("search", [](const feather::DB& db, py::array_t<float> q, size_t k = 5) {
            auto buf = q.request();
            if (buf.size != db.dim()) throw std::runtime_error("Query dimension mismatch");
            const float* ptr = static_cast<const float*>(buf.ptr);
            std::vector<float> query(ptr, ptr + buf.size);
            auto results = db.search(query, k);

            py::array_t<uint64_t> ids(results.size());
            py::array_t<float> distances(results.size());
            auto ids_ptr = ids.mutable_data();
            auto dist_ptr = distances.mutable_data();

            for (size_t i = 0; i < results.size(); ++i) {
                auto [id, dist] = results[i];
                ids_ptr[i] = id;
                dist_ptr[i] = dist;
            }
            return py::make_tuple(ids, distances);
        }, py::arg("q"), py::arg("k") = 5)
        .def("save", &feather::DB::save)
        .def("dim", &feather::DB::dim);
}
