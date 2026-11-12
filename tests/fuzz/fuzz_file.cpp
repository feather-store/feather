// libFuzzer target: the .feather loader (v10 sections and headers, the
// v3-v9 reader, the hnswlib graph stream, Metadata::deserialize).
// CI runs both targets (.github/workflows/test.yml, job `fuzz`). Locally,
// with a clang that has libFuzzer (not Apple's; a Linux container works):
//
//   python tests/fuzz/make_corpus.py corpus
//   clang++ -std=c++17 -O1 -g -fsanitize=fuzzer,address,undefined -Iinclude \
//     tests/fuzz/fuzz_file.cpp src/metadata.cpp src/filter.cpp src/scoring.cpp -o fuzz_file
//   ./fuzz_file corpus/file -max_total_time=300 -malloc_limit_mb=1024
//   (fuzz_wal: the same with FEATHER_FUZZ_BASE=corpus/base.feather and corpus/wal)
//
// A crash writes crash-<hash>; `./fuzz_file crash-<hash>` reproduces it.
#include "fuzz_common.h"

#include <sys/stat.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    static const std::string dir = [] {
        std::string d = fuzz::scratch_dir();
        mkdir(d.c_str(), 0700);
        return d;
    }();
    const std::string path = dir + "/f.feather";
    std::remove((path + ".wal").c_str());
    fuzz::write_file(path, data, size);
    fuzz::exercise(path);
    return 0;
}
