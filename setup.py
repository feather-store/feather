from setuptools import setup, Extension, find_packages
import pybind11
import glob
import sys
import os
import platform

# -undefined dynamic_lookup is macOS-only; Linux doesn't need it
extra_link_args = ["-undefined", "dynamic_lookup"] if sys.platform == "darwin" else []

# SIMD: distance kernels are selected at RUNTIME (include/feather_simd.h).
# The AVX2+FMA and AVX-512 kernels are compiled with per-function
# __attribute__((target(...))), so the default build needs no global -mavx and
# the same binary runs on any x86-64 CPU while still using AVX2/AVX-512 where
# the CPU has it. arm64 uses NEON. Published wheels therefore get the fast
# kernels too (before 0.19 they were built SSE-only).
#
# FEATHER_SIMD:
#   auto    (default) portable build, runtime dispatch
#   native  -march=native: also lets the compiler auto-vectorise the rest of the
#           engine for THIS machine's CPU. Not portable, so never use it for wheels.
#   none    scalar distance kernels only (debugging)
#   sse | avx | avx512  legacy values, treated as auto (dispatch picks the ISA)
# FEATHER_SIMD_RUNTIME=scalar|sse|avx2|avx512 caps the level at run time.
#
# platform.machine() is the HOST architecture. When a wheel is cross-compiled —
# cibuildwheel building x86_64 on an arm64 runner, or universal2 — the host and
# the TARGET differ, and gating x86 intrinsics on the host pulls immintrin.h into
# an arm64 build:
#   immintrin.h:14: error: "This header is only meant to be used on x86 and x64"
# ARCHFLAGS and _PYTHON_HOST_PLATFORM are what the build tooling sets to say what
# it is actually targeting, so prefer those and fall back to the host.
def _target_machine() -> str:
    archflags = os.getenv("ARCHFLAGS", "")
    if "-arch" in archflags:
        arches = [a for a in archflags.split() if a not in ("-arch",)]
        # A universal2 build compiles for both, so only the intersection is safe.
        if arches and all(a in ("arm64", "aarch64") for a in arches):
            return "arm64"
        if len(set(arches)) > 1:
            return "universal"          # no arch-specific flags are valid
        if arches:
            return arches[0].lower()
    host_plat = os.getenv("_PYTHON_HOST_PLATFORM", "")
    if host_plat:
        if "arm64" in host_plat or "aarch64" in host_plat:
            return "arm64"
        if "x86_64" in host_plat:
            return "x86_64"
        if "universal" in host_plat:
            return "universal"
    return platform.machine().lower()


_machine = _target_machine()
_simd_args = []
_mode = os.getenv("FEATHER_SIMD", "auto").lower()
if _mode == "none":
    _simd_args += ["-DNO_MANUAL_VECTORIZATION"]
elif _mode == "native":
    if _machine == "universal":
        print("FEATHER_SIMD=native ignored for a universal2 build", file=sys.stderr)
    elif sys.platform != "win32":
        _simd_args += ["-march=native"]
elif _mode not in ("auto", "sse", "avx", "avx512", ""):
    raise SystemExit(f"FEATHER_SIMD={_mode!r}: expected auto, native or none")

# Nearly all of the engine lives in headers (include/feather.h alone is the DB).
# setuptools only stat-checks the listed sources, so without `depends` an
# incremental `build_ext` silently reuses stale objects after a header edit —
# you get a build that looks fresh but contains none of your changes.
_headers = sorted(glob.glob("include/*.h"))

ext_modules = [
    Extension(
        "feather_db.core",
        ["bindings/feather.cpp", "src/metadata.cpp", "src/filter.cpp", "src/scoring.cpp"],
        include_dirs=[pybind11.get_include(), "include"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"] + _simd_args,
        extra_link_args=extra_link_args,
        depends=_headers,
    ),
]

setup(
    name="feather-db",
    version="0.21.0",
    # auto-discover all subpackages (extractors/feedback/pipelines/reason/
    # integrations) — a hardcoded list silently shipped broken wheels missing
    # the Phase 9.1 subpackages added after 0.8.0.
    packages=find_packages(include=["feather_db", "feather_db.*"]),
    package_data={"feather_db": ["d3.min.js"]},
    ext_modules=ext_modules,
    python_requires=">=3.8",
)
