fn main() {
    // Built from the canonical engine, not a vendored copy: this crate is not
    // published (publish = false), so it can reach outside its own root.
    // rerun-if-changed must name the directories: once a build script prints
    // any rerun-if-* line, Cargo stops tracking sources by itself (see
    // feather-cli/build.rs for the bug that caused).
    println!("cargo:rerun-if-changed=../include");
    println!("cargo:rerun-if-changed=../src");
    println!("cargo:rerun-if-changed=build.rs");
    let mut b = cc::Build::new();
    b.cpp(true)
        .std("c++17")
        // The flags the Python extension is built with (setup.py: -O3,
        // and Python's own CFLAGS carry -DNDEBUG), so the two servers run
        // the same engine code. FEATHER_SIMD=native adds -march=native, as
        // setup.py does.
        .opt_level(3)
        .define("NDEBUG", None)
        .file("../src/feather_core.cpp")
        .file("../src/metadata.cpp")
        .file("../src/filter.cpp")
        .file("../src/scoring.cpp")
        .include("../include")
        .warnings(false);
    println!("cargo:rerun-if-env-changed=FEATHER_SIMD");
    match std::env::var("FEATHER_SIMD").unwrap_or_default().to_lowercase().as_str() {
        "none" => {
            b.define("NO_MANUAL_VECTORIZATION", None);
        }
        "native" => {
            b.flag("-march=native");
        }
        _ => {}
    }
    b.compile("feather");
}
