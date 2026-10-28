fn main() {
    // `cc` prints rerun-if-env-changed lines, and once a build script prints
    // any rerun-if-* line Cargo stops rebuilding on source changes. Without
    // this, editing the vendored engine (scripts/sync-cpp.sh) left the CLI
    // linked against the old C++ — a build that looked fresh and wasn't.
    println!("cargo:rerun-if-changed=cpp");
    println!("cargo:rerun-if-changed=build.rs");
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("cpp/src/feather_core.cpp")
        .file("cpp/src/metadata.cpp")
        .file("cpp/src/filter.cpp")
        .file("cpp/src/scoring.cpp")
        .include("cpp/include")
        .compile("feather");
}
