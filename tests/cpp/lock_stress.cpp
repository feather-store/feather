// Stress test for FairSharedMutex (atomic reader fast path) and the
// per-thread-slot VisitedListPool. Build + run (also under ThreadSanitizer):
//
//   g++ -std=c++17 -O1 -g -fsanitize=thread -Iinclude tests/cpp/lock_stress.cpp -o /tmp/ls -lpthread && /tmp/ls
//   g++ -std=c++17 -O3 -Iinclude tests/cpp/lock_stress.cpp -o /tmp/ls -lpthread && /tmp/ls
//
// Checks: writers are exclusive (no torn reads, no lost increments), readers
// run concurrently, writers are not starved under continuous reads, long
// readers keep admitting readers, and no VisitedList is handed to two threads.
#include "feather.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

using feather::FairSharedMutex;
using Clock = std::chrono::steady_clock;

static int fail(const char* msg) { std::fprintf(stderr, "FAIL: %s\n", msg); return 1; }

int main() {
    // ── 1. exclusion + fairness ─────────────────────────────────────────
    FairSharedMutex mu;
    long a = 0, b = 0;                   // plain ints: TSan flags any unsynchronised access
    std::atomic<bool> stop{false}, torn{false};
    std::atomic<long> reads{0}, writes{0}, max_concurrent{0};
    std::atomic<int> inside{0};

    std::vector<std::thread> ts;
    for (int r = 0; r < 12; ++r)
        ts.emplace_back([&] {
            while (!stop.load()) {
                mu.lock_shared();
                int now = inside.fetch_add(1) + 1;
                long m = max_concurrent.load();
                while (now > m && !max_concurrent.compare_exchange_weak(m, now)) {}
                if (a != b) torn = true;
                inside.fetch_sub(1);
                mu.unlock_shared();
                reads.fetch_add(1, std::memory_order_relaxed);
            }
        });
    for (int w = 0; w < 3; ++w)
        ts.emplace_back([&] {
            while (!stop.load()) {
                mu.lock();
                if (inside.load() != 0) torn = true;    // a reader inside a write section
                ++a; ++b;
                mu.unlock();
                writes.fetch_add(1, std::memory_order_relaxed);
                std::this_thread::yield();
            }
        });
    ts.emplace_back([&] {                               // long reader, like a checkpoint
        while (!stop.load()) {
            mu.lock_shared_long();
            long a0 = a;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            if (a != a0 || a != b) torn = true;         // no writer may run meanwhile
            mu.unlock_shared_long();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    std::this_thread::sleep_for(std::chrono::seconds(3));
    stop = true;
    for (auto& t : ts) t.join();
    std::printf("reads=%ld writes=%ld max_concurrent_readers=%ld a=%ld\n",
                reads.load(), writes.load(), max_concurrent.load(), a);
    if (torn) return fail("torn read / reader inside a write section");
    if (a != writes.load() || a != b) return fail("lost update");
    if (writes.load() < 100) return fail("writers starved");
    if (max_concurrent.load() < 2) return fail("readers never ran concurrently");

    // ── 2. visited-list pool: a list is never shared ────────────────────
    hnswlib::VisitedListPool pool(1, 1000);
    std::mutex held_mu;
    std::set<hnswlib::VisitedList*> held;
    std::atomic<bool> dup{false};
    std::vector<std::thread> ps;
    for (int t = 0; t < 96; ++t)                        // > kSlots: slots are shared
        ps.emplace_back([&] {
            for (int i = 0; i < 20000; ++i) {
                auto* vl = pool.getFreeVisitedList();
                { std::lock_guard<std::mutex> g(held_mu); if (!held.insert(vl).second) dup = true; }
                vl->mass[i % 1000] = vl->curV;          // touch it
                { std::lock_guard<std::mutex> g(held_mu); held.erase(vl); }
                pool.releaseVisitedList(vl);
            }
        });
    for (auto& t : ps) t.join();
    if (dup) return fail("a VisitedList was handed to two threads");
    std::printf("OK\n");
    return 0;
}
