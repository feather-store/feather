#pragma once
#include <vector>
#include <algorithm>
#include <string>
#include <tuple>
#include <memory>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <queue>
#include "hnswlib.h"
#include "metadata.h"
#include "filter.h"
#include "scoring.h"
#include <optional>

namespace feather {

// ── Reverse-index entry: who points to a given node ──────────────
struct IncomingEdge {
    uint64_t    source_id;
    std::string rel_type;
    float       weight;
};

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

    // Reverse index: target_id → list of (source_id, rel_type, weight)
    std::unordered_map<uint64_t, std::vector<IncomingEdge>> reverse_index_;

    // ── Helpers ─────────────────────────────────────────────────────

    ModalityIndex& get_or_create_index(const std::string& modality, size_t dim) {
        auto it = modality_indices_.find(modality);
        if (it == modality_indices_.end()) {
            auto space = std::make_unique<hnswlib::L2Space>(dim);
            auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                space.get(), 1'000'000, 16, 200);
            modality_indices_[modality] = {std::move(index), std::move(space), dim};
            return modality_indices_[modality];
        }
        return it->second;
    }

    void build_reverse_index() {
        reverse_index_.clear();
        for (const auto& [id, meta] : metadata_store_) {
            for (const auto& e : meta.edges) {
                reverse_index_[e.target_id].push_back({id, e.rel_type, e.weight});
            }
        }
    }

    static std::string escape_json(const std::string& s) {
        std::string out;
        out.reserve(s.size() + 4);
        for (unsigned char c : s) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:
                    if (c < 0x20) {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                        out += buf;
                    } else {
                        out += c;
                    }
            }
        }
        return out;
    }

    // ── Persistence ─────────────────────────────────────────────────

    void save_vectors() const {
        std::ofstream f(path_, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot save file");

        uint32_t magic   = 0x46454154; // "FEAT"
        uint32_t version = 5;
        f.write((char*)&magic,   4);
        f.write((char*)&version, 4);

        // Metadata section
        uint32_t meta_count = static_cast<uint32_t>(metadata_store_.size());
        f.write((char*)&meta_count, 4);
        for (const auto& [id, meta] : metadata_store_) {
            f.write((char*)&id, 8);
            meta.serialize(f);
        }

        // Modality indices section
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
                const float* data = reinterpret_cast<const float*>(
                    m_idx.index->getDataByInternalId(i));
                f.write((char*)&id, 8);
                f.write((char*)data, m_idx.dim * sizeof(float));
            }
        }
    }

    void load_vectors() {
        std::ifstream f(path_, std::ios::binary);
        if (!f) return;

        uint32_t magic, version;
        f.read((char*)&magic,   4);
        f.read((char*)&version, 4);
        if (magic != 0x46454154) return;

        if (version == 2) {
            // v2: single "text" index, metadata interleaved with vectors
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
        } else if (version >= 3) {
            // v3/v4/v5: separate metadata section then modality indices
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
                uint32_t dim32, element_count;
                f.read((char*)&dim32, 4);
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

        build_reverse_index();
    }

public:
    // ─────────────────────────────────────────────────────────────────
    // Factory
    // ─────────────────────────────────────────────────────────────────
    static std::unique_ptr<DB> open(const std::string& path, size_t default_dim = 768) {
        auto db = std::make_unique<DB>();
        db->path_ = path;
        db->load_vectors();
        if (db->modality_indices_.empty())
            db->get_or_create_index("text", default_dim);
        return db;
    }

    // ─────────────────────────────────────────────────────────────────
    // Ingestion
    // ─────────────────────────────────────────────────────────────────
    void add(uint64_t id, const std::vector<float>& vec,
             const Metadata& meta = Metadata(),
             const std::string& modality = "text") {
        auto& m_idx = get_or_create_index(modality, vec.size());
        if (vec.size() != m_idx.dim)
            throw std::runtime_error("Dimension mismatch for modality " + modality);
        m_idx.index->addPoint(vec.data(), id);

        auto it = metadata_store_.find(id);
        if (it != metadata_store_.end()) {
            Metadata combined = meta;
            // Preserve existing edges
            if (combined.edges.empty() && !it->second.edges.empty())
                combined.edges = it->second.edges;
            metadata_store_[id] = combined;
        } else {
            metadata_store_[id] = meta;
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Salience
    // ─────────────────────────────────────────────────────────────────
    void touch(uint64_t id) {
        auto it = metadata_store_.find(id);
        if (it != metadata_store_.end()) {
            it->second.recall_count++;
            it->second.last_recalled_at = static_cast<uint64_t>(std::time(nullptr));
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Graph: link
    // ─────────────────────────────────────────────────────────────────
    void link(uint64_t from_id, uint64_t to_id,
              const std::string& rel_type = "related_to",
              float weight = 1.0f) {
        auto it = metadata_store_.find(from_id);
        if (it == metadata_store_.end()) return;

        // Avoid duplicate (same target + rel_type)
        for (const auto& e : it->second.edges)
            if (e.target_id == to_id && e.rel_type == rel_type) return;

        it->second.edges.push_back({to_id, rel_type, weight});
        reverse_index_[to_id].push_back({from_id, rel_type, weight});
    }

    // ─────────────────────────────────────────────────────────────────
    // Graph: query edges
    // ─────────────────────────────────────────────────────────────────
    std::vector<Edge> get_edges(uint64_t id) const {
        auto it = metadata_store_.find(id);
        if (it == metadata_store_.end()) return {};
        return it->second.edges;
    }

    std::vector<IncomingEdge> get_incoming(uint64_t id) const {
        auto it = reverse_index_.find(id);
        if (it == reverse_index_.end()) return {};
        return it->second;
    }

    // ─────────────────────────────────────────────────────────────────
    // Graph: auto-link by vector similarity
    // ─────────────────────────────────────────────────────────────────
    size_t auto_link(const std::string& modality = "text",
                     float threshold = 0.80f,
                     const std::string& rel_type = "related_to",
                     size_t candidates = 15) {
        auto m_it = modality_indices_.find(modality);
        if (m_it == modality_indices_.end()) return 0;
        auto& m_idx = m_it->second;
        size_t n = m_idx.index->cur_element_count;
        size_t links_created = 0;

        for (size_t i = 0; i < n; ++i) {
            uint64_t from_id = m_idx.index->getExternalLabel(i);
            const float* raw = reinterpret_cast<const float*>(
                m_idx.index->getDataByInternalId(i));
            std::vector<float> query(raw, raw + m_idx.dim);

            auto res = m_idx.index->searchKnn(query.data(), candidates + 1);
            while (!res.empty()) {
                auto [dist, to_id] = res.top(); res.pop();
                if (to_id == from_id) continue;
                float sim = 1.0f / (1.0f + dist);
                if (sim < threshold) continue;
                auto& meta = metadata_store_[from_id];
                bool exists = false;
                for (const auto& e : meta.edges)
                    if (e.target_id == to_id && e.rel_type == rel_type) { exists = true; break; }
                if (!exists) {
                    meta.edges.push_back({to_id, rel_type, sim});
                    reverse_index_[to_id].push_back({from_id, rel_type, sim});
                    ++links_created;
                }
            }
        }
        return links_created;
    }

    // ─────────────────────────────────────────────────────────────────
    // Context Chain: vector search + n-hop graph expansion
    // ─────────────────────────────────────────────────────────────────
    struct ContextNode {
        uint64_t id;
        float    score;
        float    similarity;  // 0 if reached via graph expansion
        int      hop;         // 0 = direct search hit, 1+ = graph hops
        Metadata metadata;
    };

    struct ContextEdge {
        uint64_t    source;
        uint64_t    target;
        std::string rel_type;
        float       weight;
    };

    struct ContextChainResult {
        std::vector<ContextNode> nodes;
        std::vector<ContextEdge> edges;
    };

    ContextChainResult context_chain(const std::vector<float>& query,
                                     size_t k = 5,
                                     int hops = 2,
                                     const std::string& modality = "text") {
        auto m_it = modality_indices_.find(modality);
        if (m_it == modality_indices_.end()) return {};
        auto& m_idx = m_it->second;

        // Step 1: vector search → seed nodes
        auto raw = m_idx.index->searchKnn(query.data(), k);
        std::unordered_map<uint64_t, float> sim_scores;
        while (!raw.empty()) {
            auto [dist, id] = raw.top(); raw.pop();
            float sim = 1.0f / (1.0f + dist);
            sim_scores[id] = sim;
            touch(id);
        }

        // Step 2: BFS expansion over edges (outgoing + incoming)
        std::unordered_map<uint64_t, int> visited;   // id → best hop
        std::queue<std::pair<uint64_t, int>> bfs;
        for (const auto& [id, _] : sim_scores) {
            visited[id] = 0;
            bfs.push({id, 0});
        }

        std::vector<ContextEdge> collected_edges;

        while (!bfs.empty()) {
            auto [cur_id, cur_hop] = bfs.front(); bfs.pop();
            if (cur_hop >= hops) continue;

            // Outgoing edges
            auto it = metadata_store_.find(cur_id);
            if (it != metadata_store_.end()) {
                for (const auto& e : it->second.edges) {
                    collected_edges.push_back({cur_id, e.target_id, e.rel_type, e.weight});
                    if (visited.find(e.target_id) == visited.end()) {
                        visited[e.target_id] = cur_hop + 1;
                        bfs.push({e.target_id, cur_hop + 1});
                    }
                }
            }
            // Incoming edges
            auto rit = reverse_index_.find(cur_id);
            if (rit != reverse_index_.end()) {
                for (const auto& ie : rit->second) {
                    collected_edges.push_back({ie.source_id, cur_id, ie.rel_type, ie.weight});
                    if (visited.find(ie.source_id) == visited.end()) {
                        visited[ie.source_id] = cur_hop + 1;
                        bfs.push({ie.source_id, cur_hop + 1});
                    }
                }
            }
        }

        // Step 3: build result nodes with scores
        ContextChainResult result;
        double now_ts = static_cast<double>(std::time(nullptr));

        for (const auto& [id, hop] : visited) {
            auto mit = metadata_store_.find(id);
            Metadata meta = (mit != metadata_store_.end()) ? mit->second : Metadata();

            float sim = 0.0f;
            auto sit = sim_scores.find(id);
            if (sit != sim_scores.end()) sim = sit->second;

            // Score: similarity decays by hop, modulated by importance + stickiness
            float stickiness = 1.0f + std::log(1.0f + static_cast<float>(meta.recall_count));
            float hop_decay  = 1.0f / (1.0f + static_cast<float>(hop));
            float base       = (hop == 0) ? sim : hop_decay;
            float score      = base * meta.importance * stickiness;

            result.nodes.push_back({id, score, sim, hop, std::move(meta)});
        }

        // Deduplicate edges
        std::sort(collected_edges.begin(), collected_edges.end(),
            [](const ContextEdge& a, const ContextEdge& b) {
                return std::tie(a.source, a.target, a.rel_type) <
                       std::tie(b.source, b.target, b.rel_type);
            });
        collected_edges.erase(std::unique(collected_edges.begin(), collected_edges.end(),
            [](const ContextEdge& a, const ContextEdge& b) {
                return a.source == b.source && a.target == b.target &&
                       a.rel_type == b.rel_type;
            }), collected_edges.end());
        result.edges = std::move(collected_edges);

        // Sort nodes by score descending
        std::sort(result.nodes.begin(), result.nodes.end(),
            [](const ContextNode& a, const ContextNode& b) { return a.score > b.score; });

        return result;
    }

    // ─────────────────────────────────────────────────────────────────
    // Graph export: D3 / Cytoscape-compatible JSON
    // ─────────────────────────────────────────────────────────────────
    std::string export_graph_json(const std::string& ns_filter   = "",
                                  const std::string& eid_filter  = "") const {
        std::ostringstream oss;
        oss << "{\"nodes\":[";
        bool first = true;
        for (const auto& [id, meta] : metadata_store_) {
            if (!ns_filter.empty()  && meta.namespace_id != ns_filter)  continue;
            if (!eid_filter.empty() && meta.entity_id    != eid_filter) continue;
            if (!first) oss << ","; first = false;

            oss << "{\"id\":"         << id;
            oss << ",\"label\":\""    << escape_json(meta.content.substr(0, 60)) << "\"";
            oss << ",\"namespace_id\":\"" << escape_json(meta.namespace_id)      << "\"";
            oss << ",\"entity_id\":\"" << escape_json(meta.entity_id)            << "\"";
            oss << ",\"type\":"       << static_cast<int>(meta.type);
            oss << ",\"source\":\""   << escape_json(meta.source)                << "\"";
            oss << ",\"importance\":" << meta.importance;
            oss << ",\"recall_count\":" << meta.recall_count;
            oss << ",\"timestamp\":"  << meta.timestamp;
            oss << ",\"attributes\":{";
            bool fa = true;
            for (const auto& [k,v] : meta.attributes) {
                if (!fa) oss << ","; fa = false;
                oss << "\"" << escape_json(k) << "\":\"" << escape_json(v) << "\"";
            }
            oss << "}}";
        }

        // Build set of exported node IDs so we can filter dangling edges
        std::unordered_set<uint64_t> exported_ids;
        for (const auto& [id, meta] : metadata_store_) {
            if (!ns_filter.empty()  && meta.namespace_id != ns_filter)  continue;
            if (!eid_filter.empty() && meta.entity_id    != eid_filter) continue;
            exported_ids.insert(id);
        }

        oss << "],\"edges\":[";
        first = true;
        for (const auto& [from_id, meta] : metadata_store_) {
            if (!ns_filter.empty()  && meta.namespace_id != ns_filter)  continue;
            if (!eid_filter.empty() && meta.entity_id    != eid_filter) continue;
            for (const auto& e : meta.edges) {
                // Only emit edge if target also exists in the exported node set
                if (exported_ids.find(e.target_id) == exported_ids.end()) continue;
                if (!first) oss << ","; first = false;
                oss << "{\"source\":"      << from_id;
                oss << ",\"target\":"      << e.target_id;
                oss << ",\"rel_type\":\"" << escape_json(e.rel_type) << "\"";
                oss << ",\"weight\":"      << e.weight;
                oss << "}";
            }
        }
        oss << "]}";
        return oss.str();
    }

    // ─────────────────────────────────────────────────────────────────
    // Metadata CRUD
    // ─────────────────────────────────────────────────────────────────
    std::optional<Metadata> get_metadata(uint64_t id) const {
        auto it = metadata_store_.find(id);
        if (it != metadata_store_.end()) return it->second;
        return std::nullopt;
    }

    void update_metadata(uint64_t id, const Metadata& meta) {
        metadata_store_[id] = meta;
        // Rebuild reverse index entries for this node
        for (auto& [target, incoming_list] : reverse_index_) {
            incoming_list.erase(
                std::remove_if(incoming_list.begin(), incoming_list.end(),
                    [id](const IncomingEdge& ie) { return ie.source_id == id; }),
                incoming_list.end());
        }
        for (const auto& e : meta.edges)
            reverse_index_[e.target_id].push_back({id, e.rel_type, e.weight});
    }

    void update_importance(uint64_t id, float importance) {
        auto it = metadata_store_.find(id);
        if (it != metadata_store_.end()) it->second.importance = importance;
    }

    // Get raw vector for a given id and modality (empty if not found)
    std::vector<float> get_vector(uint64_t id, const std::string& modality = "text") const {
        auto it = modality_indices_.find(modality);
        if (it == modality_indices_.end()) return {};
        try {
            return it->second.index->template getDataByLabel<float>(id);
        } catch (...) {
            return {};
        }
    }

    // Get all IDs present in a modality index
    std::vector<uint64_t> get_all_ids(const std::string& modality = "text") const {
        auto it = modality_indices_.find(modality);
        if (it == modality_indices_.end()) return {};
        std::vector<uint64_t> ids;
        ids.reserve(it->second.index->getCurrentElementCount());
        for (const auto& [id, _] : metadata_store_) {
            try {
                it->second.index->template getDataByLabel<float>(id);
                ids.push_back(id);
            } catch (...) {}
        }
        return ids;
    }

    // ─────────────────────────────────────────────────────────────────
    // Search
    // ─────────────────────────────────────────────────────────────────
    struct SearchResult {
        uint64_t id;
        float    score;
        Metadata metadata;
    };

    std::vector<SearchResult> search(const std::vector<float>& q, size_t k = 5,
                                     const SearchFilter*   filter  = nullptr,
                                     const ScoringConfig*  scoring = nullptr,
                                     const std::string&    modality = "text") {
        auto m_it = modality_indices_.find(modality);
        if (m_it == modality_indices_.end()) return {};
        auto& m_idx = m_it->second;

        struct FilterWrapper : public hnswlib::BaseFilterFunctor {
            const SearchFilter* filter_;
            const std::unordered_map<uint64_t, Metadata>& store_;
            FilterWrapper(const SearchFilter* f,
                          const std::unordered_map<uint64_t, Metadata>& s)
                : filter_(f), store_(s) {}
            bool operator()(hnswlib::labeltype id) override {
                if (!filter_) return true;
                auto it = store_.find(id);
                if (it == store_.end()) return false;
                return filter_->matches(it->second);
            }
        };

        FilterWrapper hnsw_filter(filter, metadata_store_);
        size_t candidates = (scoring) ? k * 3 : k;
        auto res = m_idx.index->searchKnn(q.data(), candidates,
                                          filter ? &hnsw_filter : nullptr);

        std::vector<SearchResult> results;
        double now_ts = static_cast<double>(std::time(nullptr));
        (void)now_ts; // used conditionally when scoring != nullptr

        while (!res.empty()) {
            auto [dist, id] = res.top(); res.pop();
            touch(id);
            auto it = metadata_store_.find(id);
            Metadata meta = (it != metadata_store_.end()) ? it->second : Metadata();
            float score = scoring
                ? Scorer::calculate_score(dist, meta, *scoring, now_ts)
                : 1.0f / (1.0f + dist);
            results.push_back({id, score, std::move(meta)});
        }

        std::sort(results.begin(), results.end(),
            [](const SearchResult& a, const SearchResult& b) { return a.score > b.score; });
        if (results.size() > k) results.resize(k);
        return results;
    }

    // ─────────────────────────────────────────────────────────────────
    // Persistence & info
    // ─────────────────────────────────────────────────────────────────
    void save() { save_vectors(); }
    ~DB() { save(); }

    size_t dim(const std::string& modality = "text") const {
        auto it = modality_indices_.find(modality);
        return (it != modality_indices_.end()) ? it->second.dim : 0;
    }

    size_t size() const { return metadata_store_.size(); }
};

} // namespace feather
