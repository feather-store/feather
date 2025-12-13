from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "feather.core",
        ["bindings/feather.cpp"],
        include_dirs=[pybind11.get_include(), "include"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
        extra_link_args=["-undefined", "dynamic_lookup"],
    ),
]

setup(
    name="feather-db",
    version="0.1.0",
    packages=["feather"],
    ext_modules=ext_modules,
    python_requires=">=3.8",
)
