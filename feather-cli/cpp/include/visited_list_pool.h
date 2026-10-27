#pragma once

#include <atomic>
#include <mutex>
#include <string.h>
#include <deque>

namespace hnswlib {
typedef unsigned short int vl_type;

class VisitedList {
 public:
    vl_type curV;
    vl_type *mass;
    unsigned int numelements;

    VisitedList(int numelements1) {
        curV = -1;
        numelements = numelements1;
        mass = new vl_type[numelements];
    }

    void reset() {
        curV++;
        if (curV == 0) {
            memset(mass, 0, sizeof(vl_type) * numelements);
            curV++;
        }
    }

    ~VisitedList() { delete[] mass; }
};
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
    // Feather: per-thread slots. Every search takes a list and gives it back;
    // upstream did both under one mutex, which every concurrent search
    // contended on. Each thread now owns a slot and swaps its list in and out
    // with one atomic exchange, so the mutex path below is only hit when a
    // thread's slot is empty (first use, or >kSlots threads sharing slots).
    static constexpr size_t kSlots = 64;
    std::atomic<VisitedList *> slots_[kSlots];

    std::deque<VisitedList *> pool;
    std::mutex poolguard;
    int numelements;

    static size_t thread_slot() {
        static std::atomic<size_t> next{0};
        thread_local const size_t slot = next.fetch_add(1, std::memory_order_relaxed) % kSlots;
        return slot;
    }

 public:
    VisitedListPool(int initmaxpools, int numelements1) {
        numelements = numelements1;
        for (auto &s : slots_) s.store(nullptr, std::memory_order_relaxed);
        for (int i = 0; i < initmaxpools; i++)
            pool.push_front(new VisitedList(numelements));
    }

    VisitedList *getFreeVisitedList() {
        VisitedList *rez = slots_[thread_slot()].exchange(nullptr, std::memory_order_acquire);
        if (!rez) {
            std::unique_lock <std::mutex> lock(poolguard);
            if (pool.size() > 0) {
                rez = pool.front();
                pool.pop_front();
            } else {
                rez = new VisitedList(numelements);
            }
        }
        rez->reset();
        return rez;
    }

    void releaseVisitedList(VisitedList *vl) {
        VisitedList *expected = nullptr;
        if (slots_[thread_slot()].compare_exchange_strong(expected, vl, std::memory_order_release,
                                                          std::memory_order_relaxed))
            return;
        std::unique_lock <std::mutex> lock(poolguard);
        pool.push_front(vl);
    }

    ~VisitedListPool() {
        for (auto &s : slots_) delete s.exchange(nullptr);
        while (pool.size()) {
            VisitedList *rez = pool.front();
            pool.pop_front();
            delete rez;
        }
    }
};
}  // namespace hnswlib
