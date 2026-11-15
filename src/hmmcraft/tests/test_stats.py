import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal

from hmmcraft import stats

N_COMPONENTS = 3


def _make_inputs(covariance_type, n_features):
    """Seeded samples, means and SPD covariances of the requested shape."""
    rs = np.random.RandomState(0)
    X = rs.standard_normal((10, n_features))
    means = rs.standard_normal((N_COMPONENTS, n_features))
    n_covars = 1 if covariance_type == "tied" else N_COMPONENTS
    a = rs.standard_normal((n_covars, n_features, n_features))
    covars = a @ a.transpose(0, 2, 1) + n_features * np.eye(n_features)
    if covariance_type == "tied":
        covars = covars[0]
    return X, means, covars


def _same_memory(a, b):
    """True if two arrays cover exactly the same bytes in the same order."""
    return (a.ctypes.data == b.ctypes.data and a.shape == b.shape
            and a.strides == b.strides)


@pytest.fixture
def array_logs_seen_by_stats(monkeypatch):
    """Record, for each call of ``np.log`` made from ``hmmcraft.stats`` on an
    array, whether its input is C-contiguous and its output is its input."""
    real_log = np.log
    seen = []

    def log(x, *args, **kwargs):
        out = real_log(x, *args, **kwargs)
        caller = sys._getframe(1).f_globals.get("__name__")
        if caller == stats.__name__ and np.ndim(x) > 0:
            x = np.asarray(x)
            seen.append((x.flags.c_contiguous, _same_memory(x, out)))
        return out

    monkeypatch.setattr(np, "log", log)
    return seen


@pytest.mark.parametrize("n_features", [1, 3])
@pytest.mark.parametrize("covariance_type", ["full", "tied"])
def test_full_and_tied_log_density_take_log_in_place_on_contiguous_arrays(
        array_logs_seen_by_stats, covariance_type, n_features):
    """The full and tied log-densities take numpy's ``log`` of an array only
    in place, on a contiguous array.

    On some CPUs, numpy chooses its float64 ``log`` loop for an array by
    where the output lies near the input in memory, unless the two are the
    same memory; in place, the choice does not depend on the allocator. One
    feature gives a 1x1 Cholesky factor, whose diagonal is a contiguous view
    already.
    """
    X, means, covars = _make_inputs(covariance_type, n_features)
    stats.log_multivariate_normal_density(X, means, covars, covariance_type)
    assert array_logs_seen_by_stats, "no np.log of an array was recorded"
    assert all(contiguous and in_place
               for contiguous, in_place in array_logs_seen_by_stats), \
        array_logs_seen_by_stats


@pytest.mark.parametrize("n_features", [1, 3])
@pytest.mark.parametrize("covariance_type", ["full", "tied"])
def test_full_and_tied_log_density_match_scipy(covariance_type, n_features):
    """The full and tied log-densities equal scipy's multivariate normal
    log-pdf, component by component."""
    X, means, covars = _make_inputs(covariance_type, n_features)
    got = stats.log_multivariate_normal_density(
        X, means, covars, covariance_type)
    if covariance_type == "tied":
        covars = np.broadcast_to(covars, (N_COMPONENTS,) + covars.shape)
    expected = np.column_stack([
        multivariate_normal(mean=mu, cov=cv).logpdf(X)
        for mu, cv in zip(means, covars)])
    assert got.shape == (len(X), N_COMPONENTS)
    assert_allclose(got, expected, rtol=1e-12)
