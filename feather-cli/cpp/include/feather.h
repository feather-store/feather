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
    struct ModalityIndex {
        std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
        std::unique_ptr<hnswlib::L2Space> space;
        size_t dim;
    };

    std::unordered_map<std::string, ModalityIndex> modality_indices_;
    std::string path_;
    std::unordered_map<uint64_t, Metadata> metadata_store_;

    ModalityIndex& get_or_create_index(const std::string& modality, size_t dim) {
        auto it = modality_indices_.find(modality);
        if (it == modality_indices_.end()) {
            auto space = std::make_unique<hnswlib::L2Space>(dim);
            auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), 1'000'000, 16, 200);
            modality_indices_[modality] = {std::move(index), std::move(space), dim};
            return modality_indices_[modality];
        }
        return it->second;
    }

    void save_vectors() const {
        std::ofstream f(path_, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot save file");

        uint32_t magic = 0x46454154; // "FEAT"
        uint32_t version = 3;
        f.write((char*)&magic, 4);
        f.write((char*)&version, 4);

        // Save Metadata Store
        uint32_t meta_count = static_cast<uint32_t>(metadata_store_.size());
        f.write((char*)&meta_count, 4);
        for (const auto& [id, meta] : metadata_store_) {
            f.write((char*)&id, 8);
            meta.serialize(f);
        }

        // Save Modality Indices
        uint32_t modal_count = static_cast<uint32_t>(modality_indices_.size());
        f.write((char*)&modal_count, 4);
        for (const auto& [name, m_idx] : modality_indices_) {
            uint16_t name_len = static_cast<uint16_t>(name.size());
            f.write((char*)&name_len, 2);
            f.write(name.data(), name_len);
            
            uint32_t dim32 = static_cast<uint32_t>(m_idx.dim);
            f.write((char*)&dim32, 4);
            
            uint32_t element_count = static_cast<uint32_t>(m_idx.index->cur_element_count);
            f.write((char*)&element_count, 4);
            
            for (size_t i = 0; i < element_count; ++i) {
                uint64_t id = m_idx.index->getExternalLabel(i);
                const float* data = reinterpret_cast<const float*>(m_idx.index->getDataByInternalId(i));
                f.write((char*)&id, 8);
                f.write((char*)data, m_idx.dim * sizeof(float));
            }
        }
    }

    void load_vectors() {
        std::ifstream f(path_, std::ios::binary);
        if (!f) return;

        uint32_t magic, version;
        f.read((char*)&magic, 4);
        f.read((char*)&version, 4);
        
        if (magic != 0x46454154) return;

        if (version == 2) {
            // Backward compatibility for v2 (single default "text" index)
            uint32_t dim32;
            f.read((char*)&dim32, 4);
            auto& m_idx = get_or_create_index("text", dim32);
            uint64_t id;
            std::vector<float> vec(dim32);
            while (f.read((char*)&id, 8)) {
                Metadata meta = Metadata::deserialize(f);
                f.read((char*)vec.data(), dim32 * sizeof(float));
                m_idx.index->addPoint(vec.data(), id);
                metadata_store_[id] = std::move(meta);
            }
        } else if (version == 3) {
            uint32_t meta_count;
            f.read((char*)&meta_count, 4);
            for (uint32_t i = 0; i < meta_count; ++i) {
                uint64_t id;
                f.read((char*)&id, 8);
                metadata_store_[id] = Metadata::deserialize(f);
            }

            uint32_t modal_count;
            f.read((char*)&modal_count, 4);
            for (uint32_t m = 0; m < modal_count; ++m) {
                uint16_t name_len;
                f.read((char*)&name_len, 2);
                std::string name(name_len, ' ');
                f.read(&name[0], name_len);
                
                uint32_t dim32;
                f.read((char*)&dim32, 4);
                
                uint32_t element_count;
                f.read((char*)&element_count, 4);
                
                auto& m_idx = get_or_create_index(name, dim32);
                std::vector<float> vec(dim32);
                for (uint32_t i = 0; i < element_count; ++i) {
                    uint64_t id;
                    f.read((char*)&id, 8);
                    f.read((char*)vec.data(), dim32 * sizeof(float));
                    m_idx.index->addPoint(vec.data(), id);
                }
            }
        }
    }

public:
    static std::unique_ptr<DB> open(const std::string& path, size_t default_dim = 768) {
        auto db = std::make_unique<DB>();
        db->path_ = path;
        db->load_vectors();
        
        // If no index was loaded, create a default "text" index
        if (db->modality_indices_.empty()) {
            db->get_or_create_index("text", default_dim);
        }
        return db;
    }

    void add(uint64_t id, const std::vector<float>& vec, const Metadata& meta = Metadata(), const std::string& modality = "text") {
        auto& m_idx = get_or_create_index(modality, vec.size());
        if (vec.size() != m_idx.dim) throw std::runtime_error("Dimension mismatch for modality " + modality);
        
        m_idx.index->addPoint(vec.data(), id);
        
        // Merge metadata if record already exists
        auto it = metadata_store_.find(id);
        if (it != metadata_store_.end()) {
            // Keep existing metadata, maybe update some fields?
            // For now, we prefer the new metadata if provided, but keep links
            Metadata combined = meta;
            if (combined.links.empty() && !it->second.links.empty()) {
                combined.links = it->second.links;
            }
            metadata_store_[id] = combined;
        } else {
            metadata_store_[id] = meta;
        }
    }

    void touch(uint64_t id) {
        auto it = metadata_store_.find(id);
        if (it != metadata_store_.end()) {
            it->second.recall_count++;
            it->second.last_recalled_at = static_cast<uint64_t>(std::time(nullptr));
        }
    }

    void link(uint64_t from_id, uint64_t to_id) {
        auto it = metadata_store_.find(from_id);
        if (it != metadata_store_.end()) {
            if (std::find(it->second.links.begin(), it->second.links.end(), to_id) == it->second.links.end()) {
                it->second.links.push_back(to_id);
            }
        }
    }

    struct SearchResult {
        uint64_t id;
        float score;
        Metadata metadata;
    };

    std::vector<SearchResult> search(const std::vector<float>& q, size_t k = 5,
                                     const SearchFilter* filter = nullptr,
                                     const ScoringConfig* scoring = nullptr,
                                     const std::string& modality = "text") {
        
        auto m_it = modality_indices_.find(modality);
        if (m_it == modality_indices_.end()) return {}; // Modality not found
        auto& m_idx = m_it->second;

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
        auto res = m_idx.index->searchKnn(q.data(), candidates_to_search, filter ? &hnsw_filter : nullptr);

        std::vector<SearchResult> results;
        double now_ts = static_cast<double>(std::time(nullptr));

        while (!res.empty()) {
            auto [dist, id] = res.top();
            res.pop();

            // Update recall metrics (Salience)
            touch(id);

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
    size_t dim(const std::string& modality = "text") const { 
        auto it = modality_indices_.find(modality);
        if (it != modality_indices_.end()) return it->second.dim;
        return 0;
    }
};
}  // namespace feather
