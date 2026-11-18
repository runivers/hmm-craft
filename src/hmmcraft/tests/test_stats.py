import sys
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg
from scipy.stats import multivariate_normal

from hmmcraft import _emissionsc, stats
from hmmcraft.hmm import GMMHMM, GaussianHMM

N_COMPONENTS = 3


def _make_inputs(covariance_type, n_features, n_samples=10):
    """Seeded samples, means and SPD covariances of the requested shape."""
    rs = np.random.RandomState(0)
    X = rs.standard_normal((n_samples, n_features))
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


# The full and tied log-densities with factors held for several calls.


def _tied_as_written(X, means, covars):
    """Compute Gaussian log-density at X for a tied model."""
    nc, nf = means.shape
    cv = np.broadcast_to(covars, (nc, nf, nf))
    return _full_as_written(X, means, cv)


def _full_as_written(X, means, covars, min_covar=1.e-7):
    """The full log-density as the package wrote it before it could hold
    factors; with ``_tied_as_written``, the reference the held factors must
    reproduce as bytes, with their warnings and errors. It is not edited
    with the package; it calls ``linalg`` through the module, so a test that
    replaces ``linalg.cholesky`` replaces it here too."""
    _, nf = means.shape
    log_prob = []
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            try:
                cv_chol = linalg.cholesky(cv + min_covar * np.eye(nf),
                                          lower=True)
            except linalg.LinAlgError:
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")

        # In place on a copy: numpy < 2.0.2 with AVX-512 picks its log loop by
        # where the output lies near the input, unless the two coincide.
        cv_diag = np.diagonal(cv_chol).copy()
        cv_log_det = 2 * np.sum(np.log(cv_diag, out=cv_diag))
        cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob.append(-.5 * (nf * np.log(2 * np.pi)
                               + (cv_sol ** 2).sum(axis=1)
                               + cv_log_det))

    return np.transpose(log_prob)


_AS_WRITTEN = {"full": _full_as_written, "tied": _tied_as_written}
_HELD = {"full": stats._log_multivariate_normal_density_full,
         "tied": stats._log_multivariate_normal_density_tied}


def _with_held_factors(covariance_type):
    """The density as a block of the E-step computes it: the factors are
    computed once (``_covar_factors``), then given to the density."""

    def density(X, means, covars):
        factors = stats._covar_factors(means, covars, covariance_type)
        return _HELD[covariance_type](X, means, covars, factors=factors)

    return density


FACTOR_LAYOUTS = ["as scipy gives it", "C order", "F order"]


def _factor_layout(monkeypatch, layout):
    """Make ``linalg.cholesky`` return its factor in ``layout``."""
    if layout == "as scipy gives it":
        return
    order = {"C order": np.ascontiguousarray,
             "F order": np.asfortranarray}[layout]
    real = linalg.cholesky

    def cholesky(*args, **kwargs):
        return order(real(*args, **kwargs))

    monkeypatch.setattr(linalg, "cholesky", cholesky)


# numpy's default error state; this test package sets every report to "warn".
_DEFAULT_STATE = {"divide": "warn", "over": "warn", "under": "ignore",
                  "invalid": "warn"}
FACTOR_STATES = [_DEFAULT_STATE, {"all": "warn"}, {"all": "raise"},
                 {"all": "ignore"}, {"all": "call"}]
FACTOR_STATE_IDS = ["default", "all=warn", "all=raise", "all=ignore",
                    "all=call"]


def _spd(n_samples, n_features, scale=1.0):
    """Seeded SPD inputs; X times ``scale``."""
    def build(t):
        X, means, covars = _make_inputs(t, n_features, n_samples)
        return X * scale, means, covars
    return build


def _covar(value):
    """State 1's covariance (tied: the covariance) set to ``value``."""
    def change(t):
        X, means, covars = _make_inputs(t, 2)
        covars = covars.copy()
        if t == "tied":
            covars[...] = value
        else:
            covars[1] = value
        return X, means, covars
    return change


def _x_row(value, mean=None):
    """Row 3 of X set to ``value``, and the means to ``mean`` if given."""
    def change(t):
        X, means, covars = _make_inputs(t, 2)
        X = X.copy()
        X[3] = value
        if mean is not None:
            means = np.full_like(means, mean)
        return X, means, covars
    return change


def _residual_1e_200(t):
    X, means, covars = _make_inputs(t, 2)
    return X * 1e-200, means * 1e-200, covars


def _reports_then_raises(t):
    """State 0's arithmetic reports an overflow; state 1 cannot be
    factored."""
    X, means, covars = _covar([[1.0, 2.0], [2.0, 1.0]])(t)
    X[3] = 1e200
    return X, means, covars


def _screened(change):
    def build(t):
        X, means, covars = _make_inputs(t, 2)
        return change(t, X, means, covars)
    return build


def _one_state_fewer(t, X, means, covars):
    return X, means, covars[:-1]


def _tied_as_3d(t, X, means, covars):
    return X, means, covars[None]


def _integer_means(t, X, means, covars):
    return X, np.rint(3 * means).astype(np.int64), covars


def _non_contiguous(t, X, means, covars):
    return X, means, np.repeat(covars, 2, axis=-1)[..., ::2]


# name: (inputs for a covariance type, whether factors are held,
#        the covariance types it applies to)
FACTOR_CASES = {
    "1 row, 1 feature": (_spd(1, 1), True, ("full", "tied")),
    "10 rows, 3 features": (_spd(10, 3), True, ("full", "tied")),
    "1 row, 8 features": (_spd(1, 8), True, ("full", "tied")),
    "300 rows, 8 features": (_spd(300, 8), True, ("full", "tied")),
    # One row far from the means: the sum of squares is most of the value,
    # so the last bits of the solve, which differ between the two memory
    # orders of one factor for a one-row sequence, reach the result.
    "1 row, 8 features, X times 20": (_spd(1, 8, 20), True, ("full", "tied")),
    "1 row, 8 features, X times 30": (_spd(1, 8, 30), True, ("full", "tied")),
    "1 row, 8 features, X times 50": (_spd(1, 8, 50), True, ("full", "tied")),
    "covariance of ones": (_covar(1.0), True, ("full", "tied")),
    "covariance of zeros": (_covar(0.0), True, ("full", "tied")),
    "not positive definite": (
        _covar([[1.0, 2.0], [2.0, 1.0]]), False, ("full", "tied")),
    "NaN covariance": (_covar(np.nan), False, ("full", "tied")),
    "inf covariance": (_covar(np.inf), False, ("full", "tied")),
    "inf row in X": (_x_row(np.inf), True, ("full", "tied")),
    "NaN row in X": (_x_row(np.nan), True, ("full", "tied")),
    "1e308 against -1e308": (_x_row(1e308, -1e308), True, ("full", "tied")),
    "row of 1e200": (_x_row(1e200), True, ("full", "tied")),
    "residual of 1e-200": (_residual_1e_200, True, ("full", "tied")),
    "state 0 reports, state 1 raises": (
        _reports_then_raises, False, ("full",)),
    "F-ordered covars": (
        _screened(lambda t, X, m, c: (X, m, np.asfortranarray(c))), False,
        ("full", "tied")),
    "non-contiguous covars": (
        _screened(_non_contiguous), False, ("full", "tied")),
    "float32 covars": (
        _screened(lambda t, X, m, c: (X, m, c.astype(np.float32))), False,
        ("full", "tied")),
    "longdouble covars": (
        _screened(lambda t, X, m, c: (X, m, c.astype(np.longdouble))),
        False, ("full", "tied")),
    "integer covars": (
        _screened(lambda t, X, m, c: (X, m, np.rint(c).astype(np.int64))),
        False, ("full", "tied")),
    "one state fewer than the means": (
        _screened(_one_state_fewer), False, ("full",)),
    "tied covariance of shape (1, nf, nf)": (
        _screened(_tied_as_3d), False, ("tied",)),
    "integer means": (_screened(_integer_means), True, ("full", "tied")),
}
FACTOR_CASE_ROWS = [(name, t) for name, (_, _, types) in FACTOR_CASES.items()
                    for t in types]
FACTOR_CASE_IDS = [f"{t}-{name}" for name, t in FACTOR_CASE_ROWS]


def _factor_inputs(name, covariance_type):
    build, held, _ = FACTOR_CASES[name]
    with np.errstate(all="ignore"):
        X, means, covars = build(covariance_type)
    return X, means, covars, held


def _outcome_in_state(state, function, *args):
    """``_outcome`` under the numpy error ``state``, and the error handler's
    calls."""
    handled = []
    with np.errstate(call=lambda kind, flag: handled.append(kind), **state):
        outcome = _outcome(function, *args)
    return outcome, handled


@pytest.mark.parametrize("state", FACTOR_STATES, ids=FACTOR_STATE_IDS)
@pytest.mark.parametrize("layout", FACTOR_LAYOUTS)
@pytest.mark.parametrize("name, covariance_type", FACTOR_CASE_ROWS,
                         ids=FACTOR_CASE_IDS)
def test_densities_with_held_factors_equal_the_lines_as_written(
        monkeypatch, name, covariance_type, layout, state):
    """The full and tied log-densities, given the factors ``_covar_factors``
    holds, give the values, the error and its text, the warnings and the
    error handler's calls of the lines they replace, in every error state
    and whatever the memory order of the factor scipy returns."""
    _factor_layout(monkeypatch, layout)
    X, means, covars, _ = _factor_inputs(name, covariance_type)
    got = _outcome_in_state(state, _with_held_factors(covariance_type),
                            X, means, covars)
    want = _outcome_in_state(state, _AS_WRITTEN[covariance_type],
                             X, means, covars)
    assert got == want


@pytest.mark.parametrize("state", FACTOR_STATES, ids=FACTOR_STATE_IDS)
@pytest.mark.parametrize("name, covariance_type", FACTOR_CASE_ROWS,
                         ids=FACTOR_CASE_IDS)
def test_factors_are_held_only_for_plain_covariances_that_factor(
        name, covariance_type, state):
    """``_covar_factors`` gives one pair per state for plain C-ordered
    float64 covariances that factor, with or without ``min_covar``, and
    None for the others; it warns, raises and reports nothing."""
    X, means, covars, held = _factor_inputs(name, covariance_type)
    handled = []
    with (warnings.catch_warnings(record=True) as caught,
          np.errstate(call=lambda kind, flag: handled.append(kind),
                      **state)):
        warnings.simplefilter("always")
        factors = stats._covar_factors(means, covars, covariance_type)
    assert caught == [] and handled == []
    if held:
        assert isinstance(factors, list) and len(factors) == len(means)
    else:
        assert factors is None


def _factor_calls(monkeypatch):
    """Record the factors ``linalg.cholesky`` returns, with their bytes, and
    the factor each ``linalg.solve_triangular`` call is given."""
    returned, solved = [], []
    real_cholesky = linalg.cholesky
    real_solve = linalg.solve_triangular

    def cholesky(*args, **kwargs):
        factor = real_cholesky(*args, **kwargs)
        returned.append((factor, factor.tobytes()))
        return factor

    def solve_triangular(a, *args, **kwargs):
        solved.append(a)
        return real_solve(a, *args, **kwargs)

    monkeypatch.setattr(linalg, "cholesky", cholesky)
    monkeypatch.setattr(linalg, "solve_triangular", solve_triangular)
    return returned, solved


@pytest.mark.parametrize("layout", FACTOR_LAYOUTS)
@pytest.mark.parametrize("n_features", [1, 3, 8])
@pytest.mark.parametrize("covariance_type", ["full", "tied"])
def test_a_held_factor_is_the_array_cholesky_returned(
        monkeypatch, covariance_type, n_features, layout):
    """Each solve with held factors is given the very array
    ``linalg.cholesky`` returned for its state (tied: the one array), whose
    memory order picks ``solve_triangular``'s LAPACK call, and the solves
    leave its bytes unchanged."""
    _factor_layout(monkeypatch, layout)
    returned, solved = _factor_calls(monkeypatch)
    X, means, covars = _make_inputs(covariance_type, n_features)
    factors = stats._covar_factors(means, covars, covariance_type)
    n_factored = 1 if covariance_type == "tied" else N_COMPONENTS
    assert len(returned) == n_factored
    for _ in range(2):
        _HELD[covariance_type](X, means, covars, factors=factors)
    assert len(returned) == n_factored
    expected = [returned[0 if covariance_type == "tied" else c][0]
                for c in range(N_COMPONENTS)] * 2
    assert len(solved) == len(expected)
    assert all(a is b for a, b in zip(solved, expected))
    assert all(factor.tobytes() == before for factor, before in returned)


# The diag and spherical log-densities: a C++ loop for the arrays it is proven
# for, numpy's expression for the rest.


def _numpy_diag(X, means, covars):
    # X: (ns, nf); means: (nc, nf); covars: (nc, nf) -> (ns, nc)
    _, nf = means.shape
    # Avoid 0 log 0 = nan in degenerate covariance case.
    covars = np.maximum(covars, np.finfo(float).tiny)
    with np.errstate(over="ignore"):
        return -0.5 * (nf * np.log(2 * np.pi)
                       + np.log(covars).sum(axis=-1)
                       + ((X[:, None, :] - means) ** 2 / covars).sum(axis=-1))


def _numpy_spherical(X, means, covars):
    nc, nf = means.shape
    if covars.ndim == 1:
        covars = covars[:, np.newaxis]
    covars = np.broadcast_to(covars, (nc, nf))
    return _numpy_diag(X, means, covars)


def _numpy_log_density(X, means, covars, covariance_type):
    """hmmlearn 0.3.3's diag and spherical log-density, as it was written.

    The reference the package must reproduce as bytes, with its warnings and
    errors; it is not edited with the package.
    """
    function = {"diag": _numpy_diag, "spherical": _numpy_spherical}
    return function[covariance_type](X, means, covars)


def _value_bytes(a):
    """The bytes of an array's values.

    An x87 extended float (``np.longdouble`` on x86) holds its value in the
    first 10 bytes of each element; the rest is padding whose content numpy
    does not define, so two equal results can differ there.
    """
    width = a.dtype.itemsize
    if (a.dtype.kind == "f" and np.finfo(a.dtype).nmant == 63
            and width > 10):
        elements = np.ascontiguousarray(a).view(np.uint8).reshape(-1, width)
        return elements[:, :10].tobytes()
    return a.tobytes()


def test_value_bytes_ignore_the_padding_of_an_extended_float():
    """Two arrays of the same values give the same value bytes, whatever
    the padding an x87 extended float leaves in its storage holds."""
    a = np.arange(4, dtype=np.longdouble) / 3
    b = a.copy()
    width = a.dtype.itemsize
    if np.finfo(np.longdouble).nmant == 63 and width > 10:
        b.view(np.uint8).reshape(-1, width)[:, 10:] = 0xA5
        assert a.tobytes() != b.tobytes()
        assert _value_bytes(a) == _value_bytes(b)
    else:
        assert _value_bytes(a) == a.tobytes()
    assert _value_bytes(a) != _value_bytes(a + 1)
    assert _value_bytes(np.arange(3.0)) == np.arange(3.0).tobytes()


def _outcome(function, *args):
    """What one call gives, everything that a caller can see recorded.

    ``("array", type, dtype, shape, flags, bytes)`` or ``("error", type,
    text)``, and the list of warnings as ``(category, message)``.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = function(*args)
        except (ValueError, TypeError, AttributeError, IndexError,
                FloatingPointError) as error:
            outcome = ("error", type(error).__name__, str(error))
        else:
            outcome = ("array", type(result).__name__, result.dtype.str,
                       result.shape, result.flags.c_contiguous,
                       result.flags.owndata, result.flags.writeable,
                       result.flags.aligned, _value_bytes(result))
    return outcome, [(w.category.__name__, str(w.message)) for w in caught]


@pytest.fixture
def loop_results(monkeypatch):
    """Record each call of the C++ loop: "kept" for an array, "none" for
    None (the package then computes numpy's expression).

    The test runs in numpy's default error state, the one a caller has
    unless they change it; this test package sets every report to "warn".
    """
    real = _emissionsc.log_density_diag
    results = []

    def recording(*args):
        log_prob = real(*args)
        results.append("none" if log_prob is None else "kept")
        return log_prob

    monkeypatch.setattr(_emissionsc, "log_density_diag", recording)
    with np.errstate(divide="warn", over="warn", under="ignore",
                     invalid="warn"):
        yield results


def _assert_numpy_outcome(X, means, covars, covariance_type="diag"):
    got = _outcome(stats.log_multivariate_normal_density,
                   X, means, covars, covariance_type)
    want = _outcome(_numpy_log_density, X, means, covars, covariance_type)
    assert got == want


def _random_inputs(n_samples, n_components, n_features, covariance_type,
                   seed=0):
    rs = np.random.RandomState(seed)
    X = rs.standard_normal((n_samples, n_features))
    means = rs.standard_normal((n_components, n_features))
    shape = ((n_components,) if covariance_type == "spherical"
             else (n_components, n_features))
    covars = rs.uniform(0.5, 2.0, shape)
    return X, means, covars


@pytest.mark.parametrize("covariance_type", ["diag", "spherical"])
@pytest.mark.parametrize("n_samples, n_components",
                         [(1, 1), (2, 3), (9, 2), (40, 5)])
@pytest.mark.parametrize("n_features",
                         [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 128, 129, 300])
def test_diag_and_spherical_log_density_equal_the_numpy_expression_as_bytes(
        loop_results, n_features, n_samples, n_components, covariance_type):
    """The C++ loop gives the bytes of numpy's expression.

    The widths cross numpy's three ways of adding a contiguous row: in order
    below 8 terms, in eight running sums up to 128, by halves above.
    """
    _assert_numpy_outcome(
        *_random_inputs(n_samples, n_components, n_features,
                        covariance_type),
        covariance_type)
    assert loop_results == ["kept"]


@pytest.mark.parametrize("n_features", [1, 9, 129])
@pytest.mark.parametrize("scale_x, scale_v, negative", [
    (1e-160, 1.0, False),    # subnormal squares
    (1.0, 1e300, False),     # subnormal quotients
    (1e-300, 1e-300, False),
    (1e-155, 1e-5, False),
    (1e150, 1.0, False),     # sums near the largest float
    (1e3, 0.0, False),       # zero variances, clamped
    (1.0, 5e-324, False),    # the smallest variance: quotients overflow
    (1e-160, 1.0, True),     # negative variances, clamped
])
def test_log_density_keeps_numpy_bytes_across_magnitudes(
        loop_results, n_features, scale_x, scale_v, negative):
    """Values far from 1 give numpy's bytes, whichever path computes them."""
    X, means, covars = _random_inputs(4, 3, n_features, "diag")
    with np.errstate(all="ignore"):
        X, means, covars = X * scale_x, means * scale_x, covars * scale_v
    if negative:
        covars[1, 0] = -covars[1, 0]
    _assert_numpy_outcome(X, means, covars)
    assert len(loop_results) == 1


def _spherical_covars(shape, covars):
    if shape == "1-D":
        return covars
    if shape == "column":
        return covars[:, None]
    return np.broadcast_to(covars[:, None], (len(covars), 9))


@pytest.mark.parametrize("shape", ["1-D", "column", "broadcast view"])
def test_spherical_covars_reach_the_loop_in_every_shape_they_come_in(
        loop_results, shape):
    """The clamp makes a new C-ordered float64 array of any spherical
    covariances; the loop computes it."""
    X, means, covars = _random_inputs(6, 3, 9, "spherical")
    _assert_numpy_outcome(X, means, _spherical_covars(shape, covars),
                          "spherical")
    assert loop_results == ["kept"]


def _in_fortran_order(a):
    out = np.empty(a.shape, order="F")
    out[...] = a
    return out


def _read_only(a):
    a = a.copy()
    a.setflags(write=False)
    return a


TAKEN = {
    "read-only X": lambda X, m, c: (_read_only(X), m, c),
    "row slice of a longer X": lambda X, m, c: (np.vstack([X, X])[2:8], m, c),
    "one-row X created in F order":
        lambda X, m, c: (_in_fortran_order(X[:1]), m, c),
    "one-state means and covars created in F order":
        lambda X, m, c: (X, _in_fortran_order(m[:1]),
                         _in_fortran_order(c[:1])),
    "integer covars": lambda X, m, c: (X, m, np.ones(c.shape, dtype=int)),
    "list covars": lambda X, m, c: (X, m, c.tolist()),
}


@pytest.mark.parametrize("name", list(TAKEN))
def test_arrays_numpy_treats_alike_are_taken(loop_results, name):
    """Inputs that reach the loop as C-ordered float64 arrays, as they are
    or after the clamp, are computed by it with numpy's bytes."""
    X, means, covars = TAKEN[name](*_random_inputs(6, 3, 9, "diag"))
    _assert_numpy_outcome(X, means, covars)
    assert loop_results == ["kept"]


class _Subclass(np.ndarray):
    pass


def _unaligned(a):
    raw = np.frombuffer(bytearray(8 * a.size + 1), dtype=np.float64,
                        offset=1).reshape(a.shape)
    raw[...] = a
    assert not raw.flags.aligned
    return raw


def _matrix(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        return np.matrix(a)


REFUSED = {
    "float32 X": lambda X, m, c: (X.astype(np.float32), m, c),
    "float32 means": lambda X, m, c: (X, m.astype(np.float32), c),
    "longdouble X": lambda X, m, c: (X.astype(np.longdouble), m, c),
    "integer X": lambda X, m, c: ((X * 10).astype(np.int64), m, c),
    "byte-swapped X": lambda X, m, c: (X.astype(">f8"), m, c),
    "F-ordered X": lambda X, m, c: (np.asfortranarray(X), m, c),
    "F-ordered means": lambda X, m, c: (X, np.asfortranarray(m), c),
    "F-ordered covars": lambda X, m, c: (X, m, np.asfortranarray(c)),
    "stepped X": lambda X, m, c: (np.repeat(X, 2, axis=1)[:, ::2], m, c),
    "reversed X": lambda X, m, c: (X[::-1], m, c),
    "unaligned X": lambda X, m, c: (_unaligned(X), m, c),
    "subclass X": lambda X, m, c: (X.view(_Subclass), m, c),
    "subclass means": lambda X, m, c: (X, m.view(_Subclass), c),
    "subclass covars": lambda X, m, c: (X, m, c.view(_Subclass)),
    "matrix X": lambda X, m, c: (_matrix(X), m, c),
    "masked X": lambda X, m, c: (np.ma.masked_array(X), m, c),
    "list X": lambda X, m, c: (X.tolist(), m, c),
    "list means": lambda X, m, c: (X, m.tolist(), c),
    "1-D X": lambda X, m, c: (X[0], m, c),
    "3-D X": lambda X, m, c: (X[None], m, c),
    "1-D means": lambda X, m, c: (X, m[0], c),
    "empty X": lambda X, m, c: (X[:0], m, c),
    "no states": lambda X, m, c: (X, m[:0], c[:0]),
    "no features": lambda X, m, c: (X[:, :0], m[:, :0], c[:, :0]),
    "X of one column": lambda X, m, c: (X[:, :1].copy(), m, c),
    "covars of one row": lambda X, m, c: (X, m, c[:1]),
    "covars of one column": lambda X, m, c: (X, m, c[:, :1].copy()),
    "1-D covars": lambda X, m, c: (X, m, c[0]),
    "scalar covars": lambda X, m, c: (X, m, 2.0),
    "X too wide": lambda X, m, c: (np.hstack([X, X]), m, c),
    "covars of a wrong shape": lambda X, m, c: (X, m, c[:, :4]),
}


@pytest.mark.parametrize("name", list(REFUSED))
def test_inputs_the_loop_is_not_proven_for_take_the_numpy_expression(
        loop_results, name):
    """Another type, dtype, layout or shape never reaches the loop: numpy's
    expression gives its values, exception or warnings."""
    X, means, covars = REFUSED[name](*_random_inputs(6, 3, 9, "diag"))
    _assert_numpy_outcome(X, means, covars)
    assert loop_results == []


def _put(which, index, value):
    def change(X, m, c):
        arrays = {"X": X.copy(), "m": m.copy(), "c": c.copy()}
        arrays[which][index] = value
        return arrays["X"], arrays["m"], arrays["c"]
    return change


def _inf_in_X_and_means(X, m, c):
    X, m = X.copy(), m.copy()
    X[1, 0] = m[0, 0] = np.inf
    return X, m, c


NOT_FINITE = {
    "NaN in X": _put("X", (2, 4), np.nan),
    "inf in X": _put("X", (1, 0), np.inf),
    "inf in X and in the means": _inf_in_X_and_means,
    "NaN in covars": _put("c", (1, 1), np.nan),
    "inf in covars": _put("c", (1, 1), np.inf),
    "overflow in the difference": lambda X, m, c: (X * 1e308, -m * 1e308, c),
    "overflow in the square": lambda X, m, c: (X * 1e200, m, c),
    "overflow in the quotient": lambda X, m, c: (X * 1e100, m, c * 1e-250),
    "overflow in the sum": lambda X, m, c: (
        np.full((1, 300), 1.3e154), np.zeros((1, 300)), np.ones((1, 300))),
}


@pytest.mark.parametrize("error_state", [
    {}, {"invalid": "raise"}, {"over": "raise"}, {"divide": "raise"}],
    ids=["default", "invalid=raise", "over=raise", "divide=raise"])
@pytest.mark.parametrize("name", list(NOT_FINITE))
def test_a_result_that_is_not_finite_is_recomputed_by_numpy(
        loop_results, name, error_state):
    """When the loop gives an inf or NaN, numpy's expression runs: its
    values, warnings and FloatingPointError are the caller's."""
    with np.errstate(all="ignore"):
        X, means, covars = NOT_FINITE[name](
            *_random_inputs(6, 3, 9, "diag"))
    with np.errstate(**error_state):
        _assert_numpy_outcome(X, means, covars)
    assert loop_results == ["none"]


@pytest.mark.parametrize("error_state", [
    {"under": "warn"}, {"under": "raise"}, {"all": "raise"}, {"all": "warn"}],
    ids=["under=warn", "under=raise", "all=raise", "all=warn"])
def test_an_underflow_report_is_left_to_numpy(loop_results, error_state):
    """With underflow reported, numpy computes: the loop reports none."""
    X, means, covars = _random_inputs(6, 3, 9, "diag")
    X, means = X * 1e-160, means * 1e-160
    with np.errstate(**error_state):
        _assert_numpy_outcome(X, means, covars)
    assert loop_results == []
    _assert_numpy_outcome(X, means, covars)
    assert loop_results == ["kept"]


@pytest.mark.parametrize("n_features, route", [
    (16, ["kept"]), (17, []), (35, [])])
def test_a_row_longer_than_numpys_buffer_is_left_to_numpy(
        loop_results, n_features, route):
    """numpy can add a row longer than its buffer in pieces of the buffer's
    size; such rows are computed by numpy."""
    X, means, covars = _random_inputs(5, 3, n_features, "diag")
    default = np.getbufsize()
    np.setbufsize(16)
    try:
        _assert_numpy_outcome(X, means, covars)
    finally:
        np.setbufsize(default)
    assert loop_results == route


def _model(model_class, covariance_type, implementation, X):
    """A model with explicit seeded parameters and no random start; the
    means are rows of ``X``."""
    rs = np.random.RandomState(0)
    n_components, n_mix = 3, 2
    kwargs = {"n_mix": n_mix} if model_class is GMMHMM else {}
    model = model_class(n_components, covariance_type=covariance_type,
                        implementation=implementation, init_params="",
                        n_iter=5, tol=0.0, random_state=0, **kwargs)
    model.startprob_ = np.full(n_components, 1 / n_components)
    model.transmat_ = np.full((n_components, n_components),
                              1 / n_components)
    mix = (n_mix,) if model_class is GMMHMM else ()
    model.means_ = X[rs.randint(len(X), size=(n_components,) + mix)]
    per_feature = () if covariance_type == "spherical" else (X.shape[1],)
    model.covars_ = rs.uniform(0.5, 2.0, (n_components,) + mix
                               + per_feature)
    if model_class is GMMHMM:
        model.weights_ = np.full((n_components, n_mix), 1 / n_mix)
    return model


def _fit_outcome(model, X, lengths):
    """Every fitted array, the score and history, both decodes and the
    state probabilities, as bytes."""
    model.fit(X, lengths)
    out = {name: np.asarray(getattr(model, name)).tobytes()
           for name in ("startprob_", "transmat_", "means_", "covars_",
                        "weights_")
           if hasattr(model, name)}
    out["score"] = np.float64(model.score(X, lengths)).tobytes()
    out["history"] = np.asarray(model.monitor_.history).tobytes()
    for algorithm in ("viterbi", "map"):
        log_prob, states = model.decode(X, lengths, algorithm=algorithm)
        out[algorithm] = np.float64(log_prob).tobytes() + states.tobytes()
    out["proba"] = model.predict_proba(X, lengths).tobytes()
    return out


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("covariance_type", ["diag", "spherical"])
@pytest.mark.parametrize("model_class", [GaussianHMM, GMMHMM])
def test_fits_are_the_same_with_and_without_the_loop(
        loop_results, monkeypatch, model_class, covariance_type,
        implementation):
    """A fit, its score, decodes and state probabilities have the same bytes
    whether the C++ loop or numpy's expression computes the densities."""
    rs = np.random.RandomState(0)
    X = np.concatenate([rs.standard_normal((40, 3)) + shift
                        for shift in (-3, 0, 3)])
    rs.shuffle(X)
    lengths = [70, 50]
    with_loop = _fit_outcome(
        _model(model_class, covariance_type, implementation, X), X, lengths)
    assert "kept" in loop_results and "none" not in loop_results
    monkeypatch.setattr(_emissionsc, "log_density_diag",
                        lambda *args: None)
    without_loop = _fit_outcome(
        _model(model_class, covariance_type, implementation, X), X, lengths)
    assert with_loop == without_loop
