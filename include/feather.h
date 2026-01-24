#pragma once
#include <vector>
#include <algorithm>
#include <string>
#include <tuple>
#include <memory>
#include <stdexcept>
#include <fstream>
#include <unordered_map>
#include "hnswlib.h"
#include "metadata.h"
#include "filter.h"
#include "scoring.h"
#include <optional>

namespace feather {
class DB {
private:
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
    size_t dim_;
    std::string path_;
    std::unordered_map<uint64_t, Metadata> metadata_store_;

    void save_vectors() const {
        std::ofstream f(path_, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot save file");

        uint32_t magic = 0x46454154; // "FEAT"
        uint32_t version = 2;
        uint32_t dim32 = dim_;
        f.write((char*)&magic, 4);
        f.write((char*)&version, 4);
        f.write((char*)&dim32, 4);

        for (size_t i = 0; i < index_->cur_element_count; ++i) {
            uint64_t id = index_->getExternalLabel(i);
            const float* data = reinterpret_cast<const float*>(index_->getDataByInternalId(i));
            f.write((char*)&id, 8);
            
            // Save metadata
            auto it = metadata_store_.find(id);
            if (it != metadata_store_.end()) {
                it->second.serialize(f);
            } else {
                Metadata().serialize(f); // Save default metadata if missing
            }

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
        
        if (magic != 0x46454154 || dim32 != dim_) return;

        if (version == 1) {
            uint64_t id;
            std::vector<float> vec(dim_);
            while (f.read((char*)&id, 8)) {
                f.read((char*)vec.data(), dim_ * sizeof(float));
                index_->addPoint(vec.data(), id);
                metadata_store_[id] = Metadata(); // Default metadata for v1
            }
        } else if (version == 2) {
            uint64_t id;
            std::vector<float> vec(dim_);
            while (f.read((char*)&id, 8)) {
                Metadata meta = Metadata::deserialize(f);
                f.read((char*)vec.data(), dim_ * sizeof(float));
                index_->addPoint(vec.data(), id);
                metadata_store_[id] = std::move(meta);
            }
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

    void add(uint64_t id, const std::vector<float>& vec, const Metadata& meta = Metadata()) {
        if (vec.size() != dim_) throw std::runtime_error("Dimension mismatch");
        index_->addPoint(vec.data(), id);
        metadata_store_[id] = meta;
    }

    struct SearchResult {
        uint64_t id;
        float score;
        Metadata metadata;
    };

    std::vector<SearchResult> search(const std::vector<float>& q, size_t k = 5,
                                     const SearchFilter* filter = nullptr,
                                     const ScoringConfig* scoring = nullptr) const {
        
        struct FilterWrapper : public hnswlib::BaseFilterFunctor {
            const SearchFilter* filter_;
            const std::unordered_map<uint64_t, Metadata>& metadata_store_;
            FilterWrapper(const SearchFilter* f, const std::unordered_map<uint64_t, Metadata>& m)
                : filter_(f), metadata_store_(m) {}
            bool operator()(hnswlib::labeltype id) override {
                if (!filter_) return true;
                auto it = metadata_store_.find(id);
                if (it == metadata_store_.end()) return false;
                return filter_->matches(it->second);
            }
        };

        FilterWrapper hnsw_filter(filter, metadata_store_);
        size_t candidates_to_search = (scoring) ? k * 3 : k;
        auto res = index_->searchKnn(q.data(), candidates_to_search, filter ? &hnsw_filter : nullptr);

        std::vector<SearchResult> results;
        double now_ts = static_cast<double>(std::time(nullptr));

        while (!res.empty()) {
            auto [dist, id] = res.top();
            res.pop();

            auto it = metadata_store_.find(id);
            Metadata meta = (it != metadata_store_.end()) ? it->second : Metadata();

            float final_score;
            if (scoring) {
                final_score = Scorer::calculate_score(dist, meta, *scoring, now_ts);
            } else {
                final_score = 1.0f / (1.0f + dist); // Default similarity score
            }

            results.push_back({id, final_score, std::move(meta)});
        }

        // Sort by score descending
        std::sort(results.begin(), results.end(), [](const SearchResult& a, const SearchResult& b) {
            return a.score > b.score;
        });

        // Limit to k
        if (results.size() > k) {
            results.resize(k);
        }

        return results;
    }

    std::optional<Metadata> get_metadata(uint64_t id) const {
        auto it = metadata_store_.find(id);
        if (it != metadata_store_.end()) return it->second;
        return std::nullopt;
    }

    void save() { save_vectors(); }
    ~DB() { save(); }

    // ← PUBLIC GETTER
    size_t dim() const { return dim_; }
};
}  // namespace feather
