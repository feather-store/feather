fn main() {
    cc::Build::new()
        .cpp(true)
        .file("../src/feather_core.cpp")
        .include("../include")
        .flag("-std=c++17")
        .compile("feather");

    println!("cargo:rustc-link-search=native=..");
    println!("cargo:rustc-link-lib=static=feather");
}
