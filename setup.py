import sys

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

# Floating-point contraction off: a product and the sum it feeds are rounded
# one after the other, never fused into one instruction. MSVC does not take
# the option.
extra_compile_args = [] if sys.platform == "win32" else ["-ffp-contract=off"]

setup(
    ext_modules=[
        Pybind11Extension(
            "hmmcraft._hmmc",
            ["ext/_hmmc.cpp"],
            cxx_std=11,
            extra_compile_args=extra_compile_args,
        ),
    ],
)
