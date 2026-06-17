from setuptools import setup, Extension
import pybind11
import sys
import os
import platform

# -undefined dynamic_lookup is macOS-only; Linux doesn't need it
extra_link_args = ["-undefined", "dynamic_lookup"] if sys.platform == "darwin" else []

# SIMD: space_l2.h ships hand-written SSE/AVX/AVX-512 L2 kernels gated behind
# USE_* macros with RUNTIME CPU dispatch. They are x86 intrinsics, so we only
# enable them on x86_64. SSE2 is baseline on all x86-64; AVX is selected at
# runtime via AVXCapable(). On arm64/aarch64 we rely on -O3 NEON auto-vectorization.
# Override with FEATHER_SIMD=none|sse|avx|avx512 (avx512 only if your build AND
# run hosts both support it).
_machine = platform.machine().lower()
_simd_args = []
if _machine in ("x86_64", "amd64"):
    _mode = os.getenv("FEATHER_SIMD", "avx").lower()
    if _mode != "none":
        _simd_args += ["-DUSE_SSE"]
        if _mode in ("avx", "avx512"):
            _simd_args += ["-DUSE_AVX", "-mavx"]
        if _mode == "avx512":
            _simd_args += ["-DUSE_AVX512", "-mavx512f", "-mavx512dq"]

ext_modules = [
    Extension(
        "feather_db.core",
        ["bindings/feather.cpp", "src/metadata.cpp", "src/filter.cpp", "src/scoring.cpp"],
        include_dirs=[pybind11.get_include(), "include"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"] + _simd_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="feather-db",
    version="0.15.3",
    packages=["feather_db", "feather_db.integrations"],
    package_data={"feather_db": ["d3.min.js"]},
    ext_modules=ext_modules,
    python_requires=">=3.8",
)
