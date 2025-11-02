#include <iostream>
#include <random>
#include "include/feather.h"

int main() {
    auto db = feather::DB::open("demo.feather", 128);  // ← now returns unique_ptr

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (uint64_t i = 0; i < 1000; ++i) {
        std::vector<float> vec(128);
        for (auto& v : vec) v = dist(gen);
        db->add(i, vec);
    }

    std::vector<float> query(128, 0.1f);
    auto results = db->search(query, 5);

    std::cout << "Top-5 nearest neighbors:\n";
    for (auto [id, distance] : results) {
        std::cout << "  ID: " << id << "  distance: " << distance << "\n";
    }

    db->save();  // Explicit save
    return 0;
}
