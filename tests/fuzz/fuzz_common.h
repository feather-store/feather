// Shared by the libFuzzer targets: write the input where a file or WAL goes,
// open it read-only, and use what loaded as the seeded fuzz's probe does
// (tests/test_corrupt_files.py), so corrupt graph links and lengths are
// followed, not just parsed. An exception is the expected answer to a corrupt
// input; a crash, a sanitizer report or a huge allocation is a bug.
#pragma once
#include "feather.h"

#include <cstdio>
#include <string>
#include <unistd.h>

namespace fuzz {

inline std::string scratch_dir() {
    const char* t = std::getenv("TMPDIR");
    std::string d = (t && *t) ? t : "/tmp";
    return d + "/feather-fuzz-" + std::to_string(getpid());
}

inline void write_file(const std::string& path, const uint8_t* data, size_t size) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) std::abort();
    if (size && std::fwrite(data, 1, size, f) != size) std::abort();
    std::fclose(f);
}

inline void exercise(const std::string& path) {
    try {
        auto db = feather::DB::open(path, 8, /*read_only=*/true);
        for (const auto& mod : db->modality_names()) {
            const size_t dim = db->dim(mod);
            if (dim == 0 || dim > 4096) continue;
            std::vector<float> q(dim, 0.5f);
            db->search(q, 5, nullptr, nullptr, mod);
            db->context_chain(q, 3, 2, mod);
        }
        db->keyword_search("alpha", 5);
        for (uint64_t id : db->all_ids()) {
            db->get_metadata(id, true);
            db->get_edges(id);
            db->get_incoming(id);
        }
        db->export_graph_json("", "");
    } catch (const std::exception&) {
        // refused: the right answer to a corrupt input
    }
}

}  // namespace fuzz
