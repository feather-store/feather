// The engine under concurrent use, for ThreadSanitizer: readers (search,
// keyword and hybrid search, context_chain, metadata reads) alongside
// writers (add, update, link, forget), a checkpointer (save) and background
// auto-compaction, all on one DB. lock_stress.cpp covers the lock alone;
// this covers what the lock protects. Then reopen and check every
// acknowledged write is there.
//
//   g++ -std=c++17 -O1 -g -fsanitize=thread -Iinclude tests/cpp/engine_stress.cpp \
//       src/metadata.cpp src/filter.cpp src/scoring.cpp -o /tmp/es -lpthread && /tmp/es
#include "feather.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <random>
#include <thread>
#include <unistd.h>
#include <unordered_set>
#include <vector>

static constexpr size_t DIM = 16;

static std::vector<float> vec_of(std::mt19937_64& rng) {
    std::normal_distribution<float> n;
    std::vector<float> v(DIM);
    for (auto& x : v) x = n(rng);
    return v;
}

static feather::Metadata meta_of(uint64_t id) {
    feather::Metadata m;
    m.timestamp = 1700000000 + static_cast<int64_t>(id);
    m.content = "record " + std::to_string(id) + " alpha beta";
    m.namespace_id = id % 2 ? "ns.a" : "ns.b";
    m.entity_id = "e" + std::to_string(id % 7);
    m.attributes["k"] = std::to_string(id % 3);
    return m;
}

int main() {
    const std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
                             "/feather-engine-stress-" + std::to_string(getpid()) + ".feather";
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> next_id{1};
    std::atomic<long> reads{0}, failures{0};
    std::mutex forgot_mx;
    std::unordered_set<uint64_t> forgotten;   // ids a writer forgot: may or may not reopen
    {
        auto db = feather::DB::open(path, DIM);
        db->set_auto_compact(0.2f);
        std::mt19937_64 seed_rng(1);
        for (uint64_t i = 0; i < 200; ++i) db->add(next_id++, vec_of(seed_rng), meta_of(i), "text");

        std::vector<std::thread> ts;
        for (int r = 0; r < 4; ++r) {
            ts.emplace_back([&, r] {
                std::mt19937_64 rng(100 + r);
                while (!stop) {
                    try {
                        auto q = vec_of(rng);
                        feather::SearchFilter f;
                        f.namespace_id = "ns.a";
                        db->search(q, 5);
                        db->search(q, 5, &f);
                        db->keyword_search("alpha", 5);
                        db->hybrid_search(q, "beta", 5);
                        db->context_chain(q, 3, 1);
                        db->get_metadata(rng() % next_id.load() + 1);
                        ++reads;
                    } catch (const std::exception& e) {
                        std::fprintf(stderr, "reader: %s\n", e.what());
                        ++failures;
                    }
                }
            });
        }
        for (int w = 0; w < 2; ++w) {
            ts.emplace_back([&, w] {
                std::mt19937_64 rng(200 + w);
                while (!stop) {
                    try {
                        const uint64_t id = next_id++;
                        db->add(id, vec_of(rng), meta_of(id), "text");
                        const uint64_t other = rng() % id + 1;
                        if (id % 5 == 0) db->update_metadata(other, meta_of(other));
                        if (id % 7 == 0) db->link(id, other, "rel", 0.5f);
                        if (id % 11 == 0) {
                            {
                                std::lock_guard<std::mutex> g(forgot_mx);
                                forgotten.insert(other);
                            }
                            db->forget(other);
                        }
                    } catch (const std::exception& e) {
                        std::fprintf(stderr, "writer: %s\n", e.what());
                        ++failures;
                    }
                }
            });
        }
        ts.emplace_back([&] {
            while (!stop) {
                try {
                    db->save();
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "save: %s\n", e.what());
                    ++failures;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        });
        const char* secs = std::getenv("STRESS_SECONDS");
        std::this_thread::sleep_for(std::chrono::seconds(secs ? std::atoi(secs) : 5));
        stop = true;
        for (auto& t : ts) t.join();
        db->close();
    }
    // Every acknowledged write that was never forgotten must reopen.
    const uint64_t written = next_id.load() - 1;
    size_t kept = 0, lost = 0;
    {
        auto db = feather::DB::open(path, DIM, /*read_only=*/true);
        for (uint64_t id = 1; id <= written; ++id) {
            if (forgotten.count(id)) continue;
            if (db->get_metadata(id)) ++kept;
            else if (++lost <= 5) std::fprintf(stderr, "lost record %llu\n", static_cast<unsigned long long>(id));
        }
    }
    for (const char* sfx : {"", ".wal", ".wal.old", ".lock", ".tmp", ".erasures"})
        std::remove((path + sfx).c_str());
    std::printf("engine stress: %ld read rounds, %llu records written, %zu kept, %zu lost, %ld failures\n",
                reads.load(), static_cast<unsigned long long>(written), kept, lost, failures.load());
    if (failures != 0 || lost != 0 || kept == 0) {
        std::fprintf(stderr, "FAIL\n");
        return 1;
    }
    return 0;
}
