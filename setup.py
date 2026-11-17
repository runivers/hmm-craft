import sys

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

# The emission loops are compared with numpy expressions as bytes:
# floating-point contraction is off for their module, so that a product and
# the sum it feeds are rounded one after the other, never fused into one
# instruction. MSVC does not take the option.
no_contraction = [] if sys.platform == "win32" else ["-ffp-contract=off"]

setup(
    ext_modules=[
        Pybind11Extension("hmmcraft._hmmc", ["ext/_hmmc.cpp"], cxx_std=11),
        Pybind11Extension(
            "hmmcraft._emissionsc",
            ["ext/_emissionsc.cpp"],
            cxx_std=11,
            extra_compile_args=no_contraction,
        ),
    ],
)
