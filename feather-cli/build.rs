fn main() {
    cc::Build::new()
        .cpp(true)
        .file("cpp/src/feather_core.cpp")
        .include("cpp/include")
        .compile("feather");
}
