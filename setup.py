from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "feather_db.core",
        ["bindings/feather.cpp", "src/metadata.cpp", "src/filter.cpp", "src/scoring.cpp"],
        include_dirs=[pybind11.get_include(), "include"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
        extra_link_args=["-undefined", "dynamic_lookup"],
    ),
]

setup(
    name="feather-db",
    version="0.2.1",
    packages=["feather_db"],
    ext_modules=ext_modules,
    python_requires=">=3.8",
)
