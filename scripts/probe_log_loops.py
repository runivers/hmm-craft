"""
Probe whether numpy's float64 ``log`` gives a strided input a result that
depends on where the output lands in memory.

    python -I scripts/probe_log_loops.py

On a CPU with AVX-512, numpy before 2.0.2 has two float64 ``log`` loops, a
vector loop and a scalar one. For a strided input it takes the scalar loop
when the output starts after the input's first element but no more than one
stride times the length past it, even if the two do not overlap; numpy 2.0.2
measures the input up to its last element. When the output is the input
itself, it takes the vector loop. The two loops are different code, so their
last bits can differ; that is read from the code, and this probe is what
shows it on a machine. The full and tied log-densities take the log of the
diagonal of a Cholesky factor: hmmlearn 0.3.3 of the strided view, hmm-craft
in place on a contiguous copy.

The probe draws seeded positive values, places them with the stride of the
diagonal of a 3x3 matrix, writes their log right after the last one, and
counts the values whose log differs, as bytes, from the log taken in place
on a contiguous copy. It prints the numpy version, the AVX-512 dispatch
targets numpy has enabled, and the count. As read from numpy's code, the
count can be above 0 only with an AVX-512 target on and numpy before 2.0.2;
numpy's ``NPY_DISABLE_CPU_FEATURES`` can turn those targets off. The probe
is a report, not a check: the exit status is always 0.
"""

import sys

import numpy as np

N_VALUES = 100_000
STRIDE = 4  # elements: the diagonal of a 3x3 matrix


def cpu_tables():
    """Return numpy's dispatch targets and its CPU feature table."""
    try:
        from numpy._core import _multiarray_umath as umath
    except ImportError:  # numpy 1.x
        from numpy.core import _multiarray_umath as umath
    return umath.__cpu_dispatch__, umath.__cpu_features__


def avx512_targets():
    """Return the AVX-512 dispatch targets numpy has enabled."""
    dispatch, features = cpu_tables()
    return [
        name
        for name in dispatch
        if (name.startswith("AVX512") or name == "X86_V4")
        and features.get(name)
    ]


def count_differing(n=N_VALUES, stride=STRIDE):
    """Count values whose strided log differs from the in-place log."""
    values = np.random.RandomState(0).lognormal(0.0, 3.0, n)
    end = stride * (n - 1) + 1
    buf = np.ones(end + n)
    strided = buf[:end:stride]
    strided[:] = values
    out = buf[end:]
    np.log(strided, out=out)
    contiguous = strided.copy()
    np.log(contiguous, out=contiguous)
    return int(
        np.count_nonzero(out.view(np.int64) != contiguous.view(np.int64))
    ), n


def main():
    print(f"numpy {np.__version__}")
    targets = avx512_targets()
    print("AVX-512 dispatch targets enabled:", " ".join(targets) or "none")
    differing, n = count_differing()
    print(
        f"float64 log of a strided input against one in place: "
        f"{differing} of {n} values differ"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
