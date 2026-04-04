from setuptools import setup, Extension
import pybind11
import sys

# -undefined dynamic_lookup is macOS-only; Linux doesn't need it
extra_link_args = ["-undefined", "dynamic_lookup"] if sys.platform == "darwin" else []

ext_modules = [
    Extension(
        "feather_db.core",
        ["bindings/feather.cpp", "src/metadata.cpp", "src/filter.cpp", "src/scoring.cpp"],
        include_dirs=[pybind11.get_include(), "include"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="feather-db",
    version="0.7.0",
    packages=["feather_db", "feather_db.integrations"],
    package_data={"feather_db": ["d3.min.js"]},
    ext_modules=ext_modules,
    python_requires=">=3.8",
)
