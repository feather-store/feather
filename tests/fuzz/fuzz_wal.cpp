// libFuzzer target: WAL replay (v1 and v2 records, CRCs, atomic groups,
// STAMP/WALGEN, every op's payload decoder), on top of a fixed valid base
// file. FEATHER_FUZZ_BASE names that file: the base the seed WALs were
// written against (tests/fuzz/make_corpus.py writes both), so the seeds
// replay fully and mutations reach every op.
#include "fuzz_common.h"

#include <sys/stat.h>

#include <fstream>
#include <sstream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    static const std::string dir = [] {
        std::string d = fuzz::scratch_dir();
        mkdir(d.c_str(), 0700);
        return d;
    }();
    static const std::string base = [] {
        const char* b = std::getenv("FEATHER_FUZZ_BASE");
        if (!b || !*b) {
            std::fprintf(stderr, "set FEATHER_FUZZ_BASE to the base .feather (tests/fuzz/make_corpus.py)\n");
            std::abort();
        }
        std::ifstream in(b, std::ios::binary);
        std::stringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }();
    const std::string path = dir + "/w.feather";
    // A fresh copy each time: a read-only open never writes the file, but the
    // input must not depend on what an earlier one left behind.
    fuzz::write_file(path, reinterpret_cast<const uint8_t*>(base.data()), base.size());
    fuzz::write_file(path + ".wal", data, size);
    fuzz::exercise(path);
    return 0;
}
