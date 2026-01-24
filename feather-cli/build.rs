fn main() {
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
