#pragma once
#include <vector>
#include <algorithm>
#include <string>
#include <tuple>
#include <memory>
#include <stdexcept>
#include <fstream>
#include "hnswlib.h"

namespace feather {
class DB {
private:
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
    size_t dim_;
    std::string path_;

    void save_vectors() const {
        std::ofstream f(path_, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot save file");

        uint32_t magic = 0x46454154; // "FEAT"
        uint32_t version = 1;
        uint32_t dim32 = dim_;
        f.write((char*)&magic, 4);
        f.write((char*)&version, 4);
        f.write((char*)&dim32, 4);

        for (size_t i = 0; i < index_->cur_element_count; ++i) {
            uint64_t id = index_->getExternalLabel(i);
            const float* data = reinterpret_cast<const float*>(index_->getDataByInternalId(i));
            f.write((char*)&id, 8);
            f.write((char*)data, dim_ * sizeof(float));
        }
    }

    void load_vectors() {
        std::ifstream f(path_, std::ios::binary);
        if (!f) return;

        uint32_t magic, version, dim32;
        f.read((char*)&magic, 4);
        f.read((char*)&version, 4);
        f.read((char*)&dim32, 4);
        if (magic != 0x46454154 || version != 1 || dim32 != dim_) return;

        uint64_t id;
        std::vector<float> vec(dim_);
        while (f.read((char*)&id, 8)) {
            f.read((char*)vec.data(), dim_ * sizeof(float));
            index_->addPoint(vec.data(), id);
        }
    }

public:
    static std::unique_ptr<DB> open(const std::string& path, size_t dim = 768) {
        auto db = std::make_unique<DB>();
        db->path_ = path;
        db->dim_ = dim;
        auto* space = new hnswlib::L2Space(dim);
        db->index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space, 1'000'000, 16, 200);
        db->load_vectors();
        return db;
    }

    void add(uint64_t id, const std::vector<float>& vec) {
        if (vec.size() != dim_) throw std::runtime_error("Dimension mismatch");
        index_->addPoint(vec.data(), id);
    }

    auto search(const std::vector<float>& q, size_t k = 5) const {  // ← const
        auto res = index_->searchKnn(q.data(), k);
        std::vector<std::tuple<uint64_t, float>> out;
        while (!res.empty()) {
            auto [dist, id] = res.top();
            out.emplace_back(id, dist);
            res.pop();
        }
        std::reverse(out.begin(), out.end());
        return out;
    }

    void save() { save_vectors(); }
    ~DB() { save(); }

    // ← PUBLIC GETTER
    size_t dim() const { return dim_; }
};
}  // namespace feather
