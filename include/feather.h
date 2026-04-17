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
#include <mutex>
#include <cstdio>
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
    std::string wal_path_;
    std::unordered_map<uint64_t, Metadata> metadata_store_;

    // Thread safety — one mutex per DB instance
    mutable std::mutex mutex_;

    // Reverse index: target_id → list of (source_id, rel_type, weight)
    std::unordered_map<uint64_t, std::vector<IncomingEdge>> reverse_index_;

    // ── BM25 Inverted Index ──────────────────────────────────────────
    struct PostingEntry { uint64_t doc_id; uint32_t term_freq; };
    std::unordered_map<std::string, std::vector<PostingEntry>> bm25_index_;
    std::unordered_map<uint64_t, uint32_t> doc_lengths_;
    double avg_dl_ = 0.0;
    static constexpr float BM25_K1 = 1.2f;
    static constexpr float BM25_B  = 0.75f;

    // ── WAL op codes ─────────────────────────────────────────────────
    enum class WalOp : uint8_t {
        ADD    = 0x01,
        UPDATE = 0x02,
        UIMP   = 0x03,
        LINK   = 0x04,
        FORGET = 0x05,
    };

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

    static const std::unordered_set<std::string>& stop_words() {
        static const std::unordered_set<std::string> sw = {
            "a","an","the","and","or","but","in","on","at","to","for",
            "of","with","by","from","is","are","was","were","be","been",
            "have","has","had","do","does","did","will","would","could",
            "should","may","might","shall","can","not","no","it","its",
            "this","that","these","those","i","me","my","we","us","our",
            "you","your","he","him","his","she","her","they","them","their"
        };
        return sw;
    }

    static std::vector<std::string> tokenize(const std::string& text) {
        std::vector<std::string> tokens;
        std::string tok;
        const auto& sw = stop_words();
        for (unsigned char c : text) {
            if (std::isalnum(c)) {
                tok += static_cast<char>(std::tolower(c));
            } else {
                if (tok.size() >= 2 && sw.find(tok) == sw.end())
                    tokens.push_back(tok);
                tok.clear();
            }
        }
        if (tok.size() >= 2 && sw.find(tok) == sw.end())
            tokens.push_back(tok);
        return tokens;
    }

    void add_to_bm25_index(uint64_t id, const std::string& content) {
        if (content.empty()) return;
        auto tokens = tokenize(content);
        if (tokens.empty()) return;

        // Remove old posting entries for this doc (handles updates)
        auto old_it = doc_lengths_.find(id);
        if (old_it != doc_lengths_.end()) {
            for (auto& [term, postings] : bm25_index_) {
                postings.erase(
                    std::remove_if(postings.begin(), postings.end(),
                        [id](const PostingEntry& p) { return p.doc_id == id; }),
                    postings.end());
            }
        }

        // Count term frequencies
        std::unordered_map<std::string, uint32_t> tf;
        for (const auto& t : tokens) tf[t]++;

        // Update doc length and posting lists
        doc_lengths_[id] = static_cast<uint32_t>(tokens.size());
        for (const auto& [term, freq] : tf)
            bm25_index_[term].push_back({id, freq});

        // Recompute avg_dl
        double total = 0.0;
        for (const auto& [_, len] : doc_lengths_) total += len;
        avg_dl_ = doc_lengths_.empty() ? 1.0 : total / static_cast<double>(doc_lengths_.size());
    }

    void rebuild_bm25_index() {
        bm25_index_.clear();
        doc_lengths_.clear();
        avg_dl_ = 0.0;
        for (const auto& [id, meta] : metadata_store_)
            add_to_bm25_index(id, meta.content);
    }

    // ── Touch (no lock) — call from within already-locked methods ────
    void touch_nolock(uint64_t id) {
        auto it = metadata_store_.find(id);
        if (it != metadata_store_.end()) {
            it->second.recall_count++;
            it->second.last_recalled_at = static_cast<uint64_t>(std::time(nullptr));
        }
    }

    // ── WAL helpers ──────────────────────────────────────────────────
    void wal_append(WalOp op, uint64_t id, const std::string& payload) {
        if (wal_path_.empty()) return;
        std::ofstream wf(wal_path_, std::ios::binary | std::ios::app);
        if (!wf) return;
        auto op_b = static_cast<uint8_t>(op);
        uint32_t plen = static_cast<uint32_t>(payload.size());
        wf.write(reinterpret_cast<const char*>(&op_b), 1);
        wf.write(reinterpret_cast<const char*>(&id),   8);
        wf.write(reinterpret_cast<const char*>(&plen), 4);
        if (plen > 0) wf.write(payload.data(), plen);
    }

    void wal_clear() const {
        if (!wal_path_.empty()) std::remove(wal_path_.c_str());
    }

    void replay_wal() {
        if (wal_path_.empty()) return;
        std::ifstream wf(wal_path_, std::ios::binary);
        if (!wf) return;

        while (true) {
            uint8_t op_b; uint64_t id; uint32_t plen;
            if (!wf.read(reinterpret_cast<char*>(&op_b), 1)) break;
            if (!wf.read(reinterpret_cast<char*>(&id),   8)) break;
            if (!wf.read(reinterpret_cast<char*>(&plen), 4)) break;
            std::string payload(plen, '\0');
            if (plen > 0 && !wf.read(&payload[0], plen)) break;

            std::istringstream ss(payload);
            auto op = static_cast<WalOp>(op_b);

            if (op == WalOp::ADD) {
                uint16_t mod_len = 0;
                ss.read(reinterpret_cast<char*>(&mod_len), 2);
                std::string modality(mod_len, '\0');
                if (mod_len > 0) ss.read(&modality[0], mod_len);
                uint32_t dim32 = 0;
                ss.read(reinterpret_cast<char*>(&dim32), 4);
                std::vector<float> vec(dim32);
                ss.read(reinterpret_cast<char*>(vec.data()), dim32 * 4);
                Metadata meta = Metadata::deserialize(ss);
                auto& m_idx = get_or_create_index(modality, dim32);
                try { m_idx.index->addPoint(vec.data(), id); } catch (...) {}
                metadata_store_[id] = std::move(meta);

            } else if (op == WalOp::UPDATE) {
                Metadata meta = Metadata::deserialize(ss);
                metadata_store_[id] = std::move(meta);

            } else if (op == WalOp::UIMP) {
                float imp = 0.0f;
                ss.read(reinterpret_cast<char*>(&imp), 4);
                auto it = metadata_store_.find(id);
                if (it != metadata_store_.end()) it->second.importance = imp;

            } else if (op == WalOp::LINK) {
                uint64_t to_id = 0;
                ss.read(reinterpret_cast<char*>(&to_id), 8);
                uint8_t rel_len = 0;
                ss.read(reinterpret_cast<char*>(&rel_len), 1);
                std::string rel_type(rel_len, '\0');
                if (rel_len > 0) ss.read(&rel_type[0], rel_len);
                float weight = 1.0f;
                ss.read(reinterpret_cast<char*>(&weight), 4);
                auto it = metadata_store_.find(id);
                if (it != metadata_store_.end()) {
                    bool exists = false;
                    for (const auto& e : it->second.edges)
                        if (e.target_id == to_id && e.rel_type == rel_type) { exists = true; break; }
                    if (!exists) {
                        it->second.edges.push_back({to_id, rel_type, weight});
                        reverse_index_[to_id].push_back({id, rel_type, weight});
                    }
                }

            } else if (op == WalOp::FORGET) {
                for (auto& [name, m_idx] : modality_indices_) {
                    try { m_idx.index->markDelete(id); } catch (...) {}
                }
                auto it = metadata_store_.find(id);
                if (it != metadata_store_.end()) {
                    it->second.content    = "";
                    it->second.source     = "_forgotten";
                    it->second.importance = 0.0f;
                    it->second.ttl        = 0;
                }
            }
        }
        build_reverse_index();
        rebuild_bm25_index();
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
        // Atomic save: write to .tmp, then rename — prevents corruption on crash
        std::string tmp_path = path_ + ".tmp";
        std::ofstream f(tmp_path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot save to temp file: " + tmp_path);

        uint32_t magic   = 0x46454154; // "FEAT"
        uint32_t version = 6;
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
        f.close();
        // Atomic rename: tmp → real path (POSIX atomic)
        if (std::rename(tmp_path.c_str(), path_.c_str()) != 0)
            throw std::runtime_error("Atomic rename failed: " + tmp_path + " → " + path_);
        // Checkpoint: clear WAL now that the full state is on disk
        wal_clear();
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
        rebuild_bm25_index();
        // Replay any uncommitted WAL entries (crash recovery)
        replay_wal();
    }

public:
    // ─────────────────────────────────────────────────────────────────
    // Factory
    // ─────────────────────────────────────────────────────────────────
    static std::unique_ptr<DB> open(const std::string& path, size_t default_dim = 768) {
        auto db = std::make_unique<DB>();
        db->path_     = path;
        db->wal_path_ = path + ".wal";
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
        std::lock_guard<std::mutex> lock(mutex_);

        // WAL: log before mutating in-memory state
        {
            std::ostringstream ws;
            uint16_t mod_len = static_cast<uint16_t>(modality.size());
            ws.write(reinterpret_cast<const char*>(&mod_len), 2);
            ws.write(modality.data(), mod_len);
            uint32_t dim32 = static_cast<uint32_t>(vec.size());
            ws.write(reinterpret_cast<const char*>(&dim32), 4);
            ws.write(reinterpret_cast<const char*>(vec.data()), vec.size() * 4);
            meta.serialize(ws);
            wal_append(WalOp::ADD, id, ws.str());
        }

        auto& m_idx = get_or_create_index(modality, vec.size());
        if (vec.size() != m_idx.dim)
            throw std::runtime_error("Dimension mismatch for modality " + modality);
        m_idx.index->addPoint(vec.data(), id);

        auto it = metadata_store_.find(id);
        if (it != metadata_store_.end()) {
            Metadata combined = meta;
            if (combined.edges.empty() && !it->second.edges.empty())
                combined.edges = it->second.edges;
            metadata_store_[id] = combined;
        } else {
            metadata_store_[id] = meta;
        }
        add_to_bm25_index(id, meta.content);
    }

    // ─────────────────────────────────────────────────────────────────
    // Salience
    // ─────────────────────────────────────────────────────────────────
    void touch(uint64_t id) {
        std::lock_guard<std::mutex> lock(mutex_);
        touch_nolock(id);
    }

    // ─────────────────────────────────────────────────────────────────
    // Graph: link
    // ─────────────────────────────────────────────────────────────────
    void link(uint64_t from_id, uint64_t to_id,
              const std::string& rel_type = "related_to",
              float weight = 1.0f) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = metadata_store_.find(from_id);
        if (it == metadata_store_.end()) return;

        for (const auto& e : it->second.edges)
            if (e.target_id == to_id && e.rel_type == rel_type) return;

        // WAL
        {
            std::ostringstream ws;
            ws.write(reinterpret_cast<const char*>(&to_id), 8);
            auto rel_len = static_cast<uint8_t>(std::min(rel_type.size(), size_t(255)));
            ws.write(reinterpret_cast<const char*>(&rel_len), 1);
            ws.write(rel_type.data(), rel_len);
            ws.write(reinterpret_cast<const char*>(&weight), 4);
            wal_append(WalOp::LINK, from_id, ws.str());
        }

        it->second.edges.push_back({to_id, rel_type, weight});
        reverse_index_[to_id].push_back({from_id, rel_type, weight});
    }

    // ─────────────────────────────────────────────────────────────────
    // Graph: query edges
    // ─────────────────────────────────────────────────────────────────
    std::vector<Edge> get_edges(uint64_t id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = metadata_store_.find(id);
        if (it == metadata_store_.end()) return {};
        return it->second.edges;
    }

    std::vector<IncomingEdge> get_incoming(uint64_t id) const {
        std::lock_guard<std::mutex> lock(mutex_);
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
        std::lock_guard<std::mutex> lock(mutex_);
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
        std::lock_guard<std::mutex> lock(mutex_);
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
            touch_nolock(id);
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
        std::lock_guard<std::mutex> lock(mutex_);
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
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = metadata_store_.find(id);
        if (it != metadata_store_.end()) return it->second;
        return std::nullopt;
    }

    void update_metadata(uint64_t id, const Metadata& meta) {
        std::lock_guard<std::mutex> lock(mutex_);
        // WAL
        {
            std::ostringstream ws;
            meta.serialize(ws);
            wal_append(WalOp::UPDATE, id, ws.str());
        }
        metadata_store_[id] = meta;
        for (auto& [target, incoming_list] : reverse_index_) {
            incoming_list.erase(
                std::remove_if(incoming_list.begin(), incoming_list.end(),
                    [id](const IncomingEdge& ie) { return ie.source_id == id; }),
                incoming_list.end());
        }
        for (const auto& e : meta.edges)
            reverse_index_[e.target_id].push_back({id, e.rel_type, e.weight});
        add_to_bm25_index(id, meta.content);
    }

    void update_importance(uint64_t id, float importance) {
        std::lock_guard<std::mutex> lock(mutex_);
        // WAL
        {
            std::ostringstream ws;
            ws.write(reinterpret_cast<const char*>(&importance), 4);
            wal_append(WalOp::UIMP, id, ws.str());
        }
        auto it = metadata_store_.find(id);
        if (it != metadata_store_.end()) it->second.importance = importance;
    }

    // Get raw vector for a given id and modality (empty if not found)
    std::vector<float> get_vector(uint64_t id, const std::string& modality = "text") const {
        std::lock_guard<std::mutex> lock(mutex_);
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
        std::lock_guard<std::mutex> lock(mutex_);
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
        std::lock_guard<std::mutex> lock(mutex_);
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
            touch_nolock(id);
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
    // BM25 keyword search
    // ─────────────────────────────────────────────────────────────────
    std::vector<SearchResult> keyword_search(const std::string& query, size_t k = 10,
                                             const SearchFilter* filter = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto terms = tokenize(query);
        if (terms.empty() || doc_lengths_.empty()) return {};

        size_t N = doc_lengths_.size();
        double avdl = avg_dl_ > 0.0 ? avg_dl_ : 1.0;

        // Unique query terms
        std::unordered_set<std::string> unique_terms(terms.begin(), terms.end());

        std::unordered_map<uint64_t, float> scores;

        for (const auto& term : unique_terms) {
            auto it = bm25_index_.find(term);
            if (it == bm25_index_.end()) continue;
            const auto& postings = it->second;
            size_t n_t = postings.size();

            // IDF (BM25+): log((N - n_t + 0.5) / (n_t + 0.5) + 1)
            double idf = std::log(
                (static_cast<double>(N) - static_cast<double>(n_t) + 0.5) /
                (static_cast<double>(n_t) + 0.5) + 1.0);

            for (const auto& p : postings) {
                if (filter) {
                    auto mit = metadata_store_.find(p.doc_id);
                    if (mit == metadata_store_.end() || !filter->matches(mit->second))
                        continue;
                }
                auto dl_it = doc_lengths_.find(p.doc_id);
                uint32_t dl = (dl_it != doc_lengths_.end()) ? dl_it->second : 1;
                double tf_norm =
                    (static_cast<double>(p.term_freq) * (BM25_K1 + 1.0)) /
                    (static_cast<double>(p.term_freq) +
                     BM25_K1 * (1.0 - BM25_B + BM25_B * static_cast<double>(dl) / avdl));
                scores[p.doc_id] += static_cast<float>(idf * tf_norm);
            }
        }

        // Sort by score descending
        std::vector<std::pair<float, uint64_t>> ranked;
        ranked.reserve(scores.size());
        for (const auto& [id, sc] : scores) ranked.push_back({sc, id});
        std::sort(ranked.begin(), ranked.end(), std::greater<std::pair<float,uint64_t>>());
        if (ranked.size() > k) ranked.resize(k);

        std::vector<SearchResult> results;
        results.reserve(ranked.size());
        for (const auto& [sc, id] : ranked) {
            touch_nolock(id);
            auto mit = metadata_store_.find(id);
            Metadata meta = (mit != metadata_store_.end()) ? mit->second : Metadata();
            results.push_back({id, sc, std::move(meta)});
        }
        return results;
    }

    // ─────────────────────────────────────────────────────────────────
    // Hybrid search: BM25 + vector via Reciprocal Rank Fusion (RRF)
    // ─────────────────────────────────────────────────────────────────
    std::vector<SearchResult> hybrid_search(const std::vector<float>& vec,
                                            const std::string& query,
                                            size_t k = 10,
                                            size_t rrf_k = 60,
                                            const SearchFilter* filter = nullptr,
                                            const ScoringConfig* scoring = nullptr,
                                            const std::string& modality = "text") {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t candidates = k * 3;

        // ── Inline vector search (no re-lock) ─────────────────────────
        std::vector<SearchResult> vec_results;
        {
            auto m_it = modality_indices_.find(modality);
            if (m_it != modality_indices_.end()) {
                auto& m_idx = m_it->second;
                struct FW : public hnswlib::BaseFilterFunctor {
                    const SearchFilter* f_; const std::unordered_map<uint64_t,Metadata>& s_;
                    FW(const SearchFilter* f, const std::unordered_map<uint64_t,Metadata>& s): f_(f),s_(s){}
                    bool operator()(hnswlib::labeltype id) override {
                        if (!f_) return true;
                        auto it = s_.find(id); return it!=s_.end() && f_->matches(it->second);
                    }
                } fw(filter, metadata_store_);
                size_t cands = scoring ? candidates * 3 : candidates;
                auto res = m_idx.index->searchKnn(vec.data(), cands, filter ? &fw : nullptr);
                double now_ts = static_cast<double>(std::time(nullptr));
                while (!res.empty()) {
                    auto [dist, id] = res.top(); res.pop();
                    auto it = metadata_store_.find(id);
                    Metadata meta = (it != metadata_store_.end()) ? it->second : Metadata();
                    float score = scoring
                        ? Scorer::calculate_score(dist, meta, *scoring, now_ts)
                        : 1.0f / (1.0f + dist);
                    vec_results.push_back({id, score, std::move(meta)});
                }
                std::sort(vec_results.begin(), vec_results.end(),
                    [](const SearchResult& a, const SearchResult& b){ return a.score > b.score; });
                if (vec_results.size() > candidates) vec_results.resize(candidates);
            }
        }

        // ── Inline BM25 search (no re-lock) ──────────────────────────
        std::vector<SearchResult> kw_results;
        {
            auto terms = tokenize(query);
            if (!terms.empty() && !doc_lengths_.empty()) {
                size_t N = doc_lengths_.size();
                double avdl = avg_dl_ > 0.0 ? avg_dl_ : 1.0;
                std::unordered_set<std::string> uterms(terms.begin(), terms.end());
                std::unordered_map<uint64_t, float> scores;
                for (const auto& term : uterms) {
                    auto it = bm25_index_.find(term);
                    if (it == bm25_index_.end()) continue;
                    size_t n_t = it->second.size();
                    double idf = std::log((static_cast<double>(N)-n_t+0.5)/(n_t+0.5)+1.0);
                    for (const auto& p : it->second) {
                        if (filter) {
                            auto mit = metadata_store_.find(p.doc_id);
                            if (mit==metadata_store_.end()||!filter->matches(mit->second)) continue;
                        }
                        auto dl_it = doc_lengths_.find(p.doc_id);
                        uint32_t dl = dl_it!=doc_lengths_.end() ? dl_it->second : 1;
                        double tf_norm = (p.term_freq*(BM25_K1+1.0)) /
                            (p.term_freq + BM25_K1*(1.0-BM25_B+BM25_B*dl/avdl));
                        scores[p.doc_id] += static_cast<float>(idf * tf_norm);
                    }
                }
                std::vector<std::pair<float,uint64_t>> ranked;
                ranked.reserve(scores.size());
                for (const auto& [id, sc] : scores) ranked.push_back({sc, id});
                std::sort(ranked.begin(), ranked.end(), std::greater<std::pair<float,uint64_t>>());
                if (ranked.size() > candidates) ranked.resize(candidates);
                for (const auto& [sc, id] : ranked) {
                    auto mit = metadata_store_.find(id);
                    Metadata meta = (mit != metadata_store_.end()) ? mit->second : Metadata();
                    kw_results.push_back({id, sc, std::move(meta)});
                }
            }
        }

        // ── RRF merge ────────────────────────────────────────────────
        std::unordered_map<uint64_t, double> rrf_scores;
        for (size_t rank = 0; rank < vec_results.size(); ++rank)
            rrf_scores[vec_results[rank].id] += 1.0 / (static_cast<double>(rrf_k) + rank + 1);
        for (size_t rank = 0; rank < kw_results.size(); ++rank)
            rrf_scores[kw_results[rank].id] += 1.0 / (static_cast<double>(rrf_k) + rank + 1);

        std::vector<std::pair<double, uint64_t>> ranked;
        ranked.reserve(rrf_scores.size());
        for (const auto& [id, sc] : rrf_scores) ranked.push_back({sc, id});
        std::sort(ranked.begin(), ranked.end(), std::greater<std::pair<double,uint64_t>>());
        if (ranked.size() > k) ranked.resize(k);

        std::vector<SearchResult> results;
        results.reserve(ranked.size());
        for (const auto& [sc, id] : ranked) {
            auto mit = metadata_store_.find(id);
            Metadata meta = (mit != metadata_store_.end()) ? mit->second : Metadata();
            results.push_back({id, static_cast<float>(sc), std::move(meta)});
        }
        return results;
    }

    // ─────────────────────────────────────────────────────────────────
    // Memory lifecycle: forget / purge / expire
    // ─────────────────────────────────────────────────────────────────

    // Soft-delete: mark-deleted in HNSW (exits search), blank content,
    // set importance=0. The node shell remains so graph edges stay traversable.
    void forget(uint64_t id) {
        std::lock_guard<std::mutex> lock(mutex_);
        wal_append(WalOp::FORGET, id, "");
        for (auto& [name, m_idx] : modality_indices_) {
            try { m_idx.index->markDelete(id); } catch (...) {}
        }
        auto it = metadata_store_.find(id);
        if (it != metadata_store_.end()) {
            it->second.content    = "";
            it->second.source     = "_forgotten";
            it->second.importance = 0.0f;
            it->second.ttl        = 0;
        }
    }

    // Hard-delete: remove all nodes in namespace_id from indices +
    // metadata store + reverse index. Returns count of removed nodes.
    size_t purge(const std::string& ns_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::unordered_set<uint64_t> to_purge;
        for (const auto& [id, meta] : metadata_store_)
            if (meta.namespace_id == ns_id) to_purge.insert(id);

        // Mark-delete in every modality
        for (auto& [name, m_idx] : modality_indices_) {
            for (uint64_t id : to_purge) {
                try { m_idx.index->markDelete(id); } catch (...) {}
            }
        }
        // Erase metadata
        for (uint64_t id : to_purge) metadata_store_.erase(id);

        // Clean reverse index: remove entries sourced from purged nodes
        for (auto& [target, incoming] : reverse_index_) {
            incoming.erase(
                std::remove_if(incoming.begin(), incoming.end(),
                    [&to_purge](const IncomingEdge& ie) {
                        return to_purge.count(ie.source_id) > 0;
                    }),
                incoming.end());
        }
        // Remove reverse index entries for purged target keys
        for (uint64_t id : to_purge) reverse_index_.erase(id);

        // Prune edges in surviving nodes that pointed to purged targets
        for (auto& [id, meta] : metadata_store_) {
            meta.edges.erase(
                std::remove_if(meta.edges.begin(), meta.edges.end(),
                    [&to_purge](const Edge& e) {
                        return to_purge.count(e.target_id) > 0;
                    }),
                meta.edges.end());
        }

        return to_purge.size();
    }

    // Scan all nodes and soft-delete any with ttl>0 where now > timestamp+ttl.
    // Returns count of nodes forgotten.
    size_t forget_expired() {
        std::lock_guard<std::mutex> lock(mutex_);
        int64_t now = static_cast<int64_t>(std::time(nullptr));
        size_t  count = 0;
        std::vector<uint64_t> expired;
        for (const auto& [id, meta] : metadata_store_) {
            if (meta.ttl > 0 && now > meta.timestamp + meta.ttl)
                expired.push_back(id);
        }
        for (uint64_t id : expired) {
            wal_append(WalOp::FORGET, id, "");
            for (auto& [name, m_idx] : modality_indices_) {
                try { m_idx.index->markDelete(id); } catch (...) {}
            }
            auto it = metadata_store_.find(id);
            if (it != metadata_store_.end()) {
                it->second.content    = "";
                it->second.source     = "_forgotten";
                it->second.importance = 0.0f;
                it->second.ttl        = 0;
            }
            ++count;
        }
        return count;
    }

    // ─────────────────────────────────────────────────────────────────
    // 7d: compact() — rebuild HNSW indices without soft-deleted records
    // ─────────────────────────────────────────────────────────────────
    size_t compact() {
        std::lock_guard<std::mutex> lock(mutex_);

        // Collect IDs marked as soft-deleted (_deleted="true" + importance==0)
        std::unordered_set<uint64_t> to_remove;
        for (const auto& [id, meta] : metadata_store_) {
            bool deleted_flag = false;
            auto attr_it = meta.attributes.find("_deleted");
            if (attr_it != meta.attributes.end() && attr_it->second == "true")
                deleted_flag = true;
            if (deleted_flag && meta.importance == 0.0f)
                to_remove.insert(id);
        }
        if (to_remove.empty()) return 0;

        // Rebuild each modality index without the deleted IDs
        for (auto& [name, m_idx] : modality_indices_) {
            size_t n = m_idx.index->cur_element_count;
            std::vector<std::pair<uint64_t, std::vector<float>>> survivors;
            survivors.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                uint64_t id = m_idx.index->getExternalLabel(i);
                if (to_remove.count(id)) continue;
                const float* data = reinterpret_cast<const float*>(
                    m_idx.index->getDataByInternalId(i));
                survivors.push_back({id, std::vector<float>(data, data + m_idx.dim)});
            }
            auto space     = std::make_unique<hnswlib::L2Space>(m_idx.dim);
            auto new_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                space.get(), 1'000'000, 16, 200);
            for (const auto& [id, vec] : survivors)
                new_index->addPoint(vec.data(), id);
            m_idx.index = std::move(new_index);
            m_idx.space  = std::move(space);
        }

        // Remove metadata
        for (uint64_t id : to_remove) metadata_store_.erase(id);

        // Rebuild derived structures
        build_reverse_index();
        rebuild_bm25_index();

        return to_remove.size();
    }

    // ─────────────────────────────────────────────────────────────────
    // Persistence & info
    // ─────────────────────────────────────────────────────────────────
    void save() {
        std::lock_guard<std::mutex> lock(mutex_);
        save_vectors();
    }
    ~DB() {
        // save() acquires mutex — call save_vectors() directly in destructor
        // (no other threads should be using the DB at destruction time)
        try { save_vectors(); } catch (...) {}
    }

    size_t dim(const std::string& modality = "text") const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = modality_indices_.find(modality);
        return (it != modality_indices_.end()) ? it->second.dim : 0;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return metadata_store_.size();
    }
};

} // namespace feather
