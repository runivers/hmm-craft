"""
Probe where a product is fused into the sum it feeds, and in which order
numpy sums.

    python -I scripts/probe_contraction.py
    python -I scripts/probe_contraction.py --package hmmcraft

A compiler may turn ``a * b + c`` into one fused multiply-add, which rounds
once where the two operations round twice; the last bit of the result can
then differ. Without ``--package`` the probe reports on numpy, with seeded
synthetic data:

* ``numpy.einsum('ij,ik,il->jkl', P, X, X)``, the statistic of the ``full``
  and ``tied`` covariances, against a loop over the samples in which every
  product and every sum is a numpy call of its own, so each is rounded on
  its own;
* the two sums over the features in the ``diag`` log-density,
  ``numpy.log(covars).sum(axis=-1)`` and
  ``((X[:, None, :] - means) ** 2 / covars).sum(axis=-1)``, against numpy's
  pairwise summation written out one addition at a time, as read from
  numpy's code: fewer than 8 elements in order, up to 128 in eight running
  sums, more than 128 by halves.

The shapes cross those two limits. Each report ends with a control that
must differ: the loop over the samples in reverse, and the features summed
in order.

With ``--package`` the probe reports on that package's C++ kernel instead:
the three sums of products of the scaling functions, each on values for
which two roundings give 0.75 and one fused multiply-add gives
0.75 + 2**-53. ``hmmlearn`` has the same functions.

The probe is a report, not a check: the exit status is 0 whatever it
finds, and not 0 only when the probe itself fails.
"""

import argparse
import importlib
import platform
import sys

import numpy as np

N_COMPONENTS = 3
N_SAMPLES = (1, 2, 7, 8, 9, 300)
N_FEATURES = (1, 3, 7, 8, 9, 128, 129, 257)

STEP = 2.0**-53
A = 0.25 + STEP
B = 1.0 - STEP
# Each case leaves 0.5 in a sum and adds A * B to it. A * B is
# 0.25 + 2**-54 + 2**-55 - 2**-106: rounded on its own it is 0.25 + 2**-54,
# and 0.75 + 2**-54, a tie, rounds to 0.75. Rounded once, the sum is
# 0.75 + 2**-53. name -> (arguments, where the sum is in the result).
KERNEL_CASES = {
    "forward_scaling": (
        (
            np.array([0.5, A, 0.25 - STEP]),
            np.array([[1.0, 0.5, 0.0], [B, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            np.ones((2, 3)),
        ),
        lambda result: result[1][1, 0],
    ),
    "backward_scaling": (
        (
            np.array([0.5, 0.5]),
            np.array([[0.5 + STEP, A], [0.0, 0.0]]),
            np.ones((2, 2)),
            np.array([1.0, B]),
        ),
        lambda result: result[0, 0],
    ),
    "compute_scaling_xi_sum": (
        (
            np.array([[0.5], [A], [0.0]]),
            np.array([[1.0]]),
            np.array([[0.0], [1.0], [B]]),
            np.ones((3, 1)),
        ),
        lambda result: result[0, 0],
    ),
}


def same(a, b):
    """Whether two arrays are equal as bytes."""
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def synthetic(n_samples, n_features):
    """Return seeded weights, samples, means and variances."""
    rs = np.random.RandomState(0)
    weights = rs.dirichlet(np.ones(N_COMPONENTS), size=n_samples)
    X = rs.normal(size=(n_samples, n_features))
    means = rs.normal(size=(N_COMPONENTS, n_features))
    covars = rs.lognormal(size=(N_COMPONENTS, n_features))
    return weights, X, means, covars


def outer_sum(weights, X, reverse=False):
    """Sum ``weights[i] * X[i] * X[i]`` over the samples, in separate calls."""
    out = np.zeros((weights.shape[1], X.shape[1], X.shape[1]))
    order = range(len(X))
    for i in reversed(order) if reverse else order:
        out += (weights[i][:, None, None] * X[i][:, None]) * X[i]
    return out


def pairwise_sum(a):
    """Sum over the last axis in numpy's pairwise order, in separate calls."""
    n = a.shape[-1]
    if n < 8:
        total = np.full(a.shape[:-1], -0.0)
        for k in range(n):
            total = total + a[..., k]
        return total
    if n <= 128:
        r = [a[..., k] for k in range(8)]
        whole = n - n % 8
        for i in range(8, whole, 8):
            for k in range(8):
                r[k] = r[k] + a[..., i + k]
        total = ((r[0] + r[1]) + (r[2] + r[3])) + (
            (r[4] + r[5]) + (r[6] + r[7])
        )
        for i in range(whole, n):
            total = total + a[..., i]
        return total
    half = n // 2
    half -= half % 8
    return pairwise_sum(a[..., :half]) + pairwise_sum(a[..., half:])


def ordered_sum(a):
    """Sum over the last axis, one element after the other."""
    total = np.zeros(a.shape[:-1])
    for k in range(a.shape[-1]):
        total = total + a[..., k]
    return total


def word(equal):
    """Return the word the reports print for a comparison."""
    return "equal" if equal else "differs"


def report_numpy():
    """Print the einsum and the feature-sum reports."""
    print(f"numpy {np.__version__}, {platform.system()} {platform.machine()}")
    shapes = [(ns, nf) for ns in N_SAMPLES for nf in N_FEATURES]

    print("einsum 'ij,ik,il->jkl' against one rounding per product and sum")
    print("samples features result")
    n_equal = 0
    for ns, nf in shapes:
        weights, X, _, _ = synthetic(ns, nf)
        equal = same(
            np.einsum("ij,ik,il->jkl", weights, X, X), outer_sum(weights, X)
        )
        n_equal += equal
        print(f"{ns:7d} {nf:8d} {word(equal)}")
    control = same(
        np.einsum("ij,ik,il->jkl", weights, X, X),
        outer_sum(weights, X, reverse=True),
    )
    print(f"control, samples in reverse (must differ): {word(control)}")
    print(f"einsum: {n_equal} of {len(shapes)} shapes equal")

    print("feature sums of the diag log-density against the pairwise order")
    print("samples features log(covars) squares")
    n_equal = 0
    for ns, nf in shapes:
        _, X, means, covars = synthetic(ns, nf)
        logs = np.log(covars)
        squares = (X[:, None, :] - means) ** 2 / covars
        equal_logs = same(logs.sum(axis=-1), pairwise_sum(logs))
        equal_squares = same(squares.sum(axis=-1), pairwise_sum(squares))
        n_equal += equal_logs and equal_squares
        print(f"{ns:7d} {nf:8d} {word(equal_logs):11s} {word(equal_squares)}")
    control = same(squares.sum(axis=-1), ordered_sum(squares))
    print(f"control, features in order (must differ): {word(control)}")
    print(f"feature sums: {n_equal} of {len(shapes)} shapes equal")


def report_kernel(package):
    """Print, for each sum of products in the kernel, how it is rounded."""
    try:
        kernel = importlib.import_module(f"{package}._hmmc")
    except ImportError as error:
        print(f"kernel of {package}: not importable ({error})")
        return
    print(
        f"kernel of {package}, {platform.system()} {platform.machine()} "
        "(0.75: two roundings; 0.75 + 2**-53: fused)"
    )
    for name, (arguments, pick) in KERNEL_CASES.items():
        value = float(pick(getattr(kernel, name)(*arguments)))
        if value == 0.75:
            how = "two roundings"
        elif value == 0.75 + STEP:
            how = "fused"
        else:
            how = f"neither ({value.hex()})"
        print(f"{name}: {how}")


def main(argv=None):
    """Print the report that the arguments ask for and return 0."""
    summary = " ".join(__doc__.split("\n\n")[0].split())
    parser = argparse.ArgumentParser(description=summary)
    parser.add_argument("--package", choices=["hmmcraft", "hmmlearn"])
    args = parser.parse_args(argv)
    if args.package:
        report_kernel(args.package)
    else:
        report_numpy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
