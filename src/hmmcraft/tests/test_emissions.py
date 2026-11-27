"""The emission hooks' shorter paths, ``hmmcraft._emissions``.

On the machines where it is used, a C++ loop computes the statistic of the
full and tied covariances for the arrays it is proven for; numpy's einsum
computes it for the rest. The Poisson and multinomial hooks make fewer
``scipy.stats`` calls for the arrays they are proven for, with the bytes,
warnings and errors of one call per state. In an E-step, ``GMMHMM``'s
statistics read the mixture densities its likelihood computed for the same
sequence, with the bytes, warnings and errors of computing them again.
"""

import copy
import os
import platform
import sys
import warnings

import numpy as np
import pytest
import scipy.stats
from scipy import special

from hmmcraft import _emissions, _emissionsc
from hmmcraft.base import BaseHMM
from hmmcraft.hmm import (
    GMMHMM,
    GaussianHMM,
    MultinomialHMM,
    PoissonHMM,
)
from hmmcraft.utils import log_normalize
from hmmcraft.vhmm import VariationalGaussianHMM

from .test_emissionsc import _one_rounding_per_operation
from .test_stats import (
    _in_fortran_order,
    _matrix,
    _outcome,
    _read_only,
    _Subclass,
    _unaligned,
)

ES = "ij,ik,il->jkl"


@pytest.fixture
def loop_calls(monkeypatch):
    """Record each call of the C++ loop: "kept" for an array, "none" for
    None (numpy then computes the statistic).

    No error state is set: the route does not depend on one.
    """
    real = _emissionsc.weighted_outer_sum
    calls = []

    def recording(*args):
        stat = real(*args)
        calls.append("none" if stat is None else "kept")
        return stat

    monkeypatch.setattr(_emissionsc, "weighted_outer_sum", recording)
    return calls


def _route(taken):
    """The calls of the loop expected here: ``taken`` where the loop is
    used, none elsewhere."""
    return taken if _emissions._STATISTIC_LOOP else []


def _same_as_einsum(weights, X):
    return _outcome(_emissions._weighted_outer_sum, weights, X) == _outcome(
        np.einsum, ES, weights, X, X
    )


def _random_inputs(n_samples, n_components, n_features):
    """Seeded weights, each row summing to one, and samples."""
    rs = np.random.RandomState(0)
    weights = rs.dirichlet(np.ones(n_components), n_samples)
    X = rs.standard_normal((n_samples, n_features))
    return weights, X


@pytest.mark.parametrize(
    "machine, maxsize, used",
    [
        ("x86_64", 2**63 - 1, True),
        ("AMD64", 2**63 - 1, True),
        ("arm64", 2**63 - 1, False),
        ("aarch64", 2**63 - 1, False),
        ("ARM64", 2**63 - 1, False),
        ("amd64", 2**63 - 1, False),
        ("ppc64le", 2**63 - 1, False),
        ("", 2**63 - 1, False),
        ("x86_64", 2**31 - 1, False),
        ("AMD64", 2**31 - 1, False),
    ],
)
def test_the_loop_is_used_on_64_bit_x86_only(machine, maxsize, used):
    """The loop computes the statistic only for a 64-bit interpreter on
    x86_64 or AMD64."""
    assert _emissions._loop_platform(machine, maxsize) is used


# The machine names of the platforms wheels are built for.
WHEEL_PLATFORMS = {
    "x86_64": True,
    "AMD64": True,
    "arm64": False,
    "aarch64": False,
}


def test_the_statistic_takes_the_route_of_this_platform(loop_calls):
    """On this machine the statistic goes to the loop if, and only if, the
    machine is 64-bit x86; every other machine keeps numpy's einsum."""
    used = (
        WHEEL_PLATFORMS.get(platform.machine(), False) and sys.maxsize > 2**32
    )
    assert _emissions._STATISTIC_LOOP is used
    _emissions._weighted_outer_sum(*_random_inputs(5, 2, 3))
    assert loop_calls == (["kept"] if used else [])


@pytest.mark.parametrize(
    "n_samples, n_components, n_features",
    [
        (n_samples, n_components, n_features)
        for n_samples, n_components in [(1, 1), (2, 3), (9, 2), (40, 5)]
        for n_features in [1, 2, 3, 8, 9, 17]
    ]
    + [
        # Sample counts and entries per sample on either side of 8192.
        (8193, 2, 1),
        (16385, 1, 2),
        (3, 1, 91),
        (2, 3, 64),
    ],
)
def test_the_statistic_equals_numpy_einsum_as_bytes(
    loop_calls, n_samples, n_components, n_features
):
    """The statistic has the bytes of numpy.einsum's, and the loop computes
    it where it is used, unless there is a single entry."""
    assert _same_as_einsum(
        *_random_inputs(n_samples, n_components, n_features)
    )
    single = n_components * n_features == 1
    assert loop_calls == ([] if single else _route(["kept"]))


def test_one_state_with_one_feature_is_left_to_numpy(loop_calls):
    """numpy adds a single output entry in pieces of 8192 samples, another
    order than the loop's: one state with one feature never reaches it."""
    assert _same_as_einsum(*_random_inputs(16385, 1, 1))
    assert loop_calls == []


def _scaled(weights_scale, x_scale):
    def make(weights, X):
        return weights * weights_scale, X * x_scale

    return make


def _signed_zeros(weights, X):
    weights, X = weights.copy(), X.copy()
    weights[1] = 0.0
    X[0, 1] = X[2, 0] = -0.0
    X[3, 2] = 0.0
    return weights, X


MAGNITUDES = {
    "subnormal products": _scaled(1.0, 1e-160),
    "products that underflow to zero": _scaled(1.0, 1e-170),
    "subnormal weights": _scaled(1e-310, 1.0),
    "values near the largest float": _scaled(1.0, 1e150),
    "zeros and negative zeros": _signed_zeros,
}


@pytest.mark.parametrize("name", list(MAGNITUDES))
def test_the_statistic_keeps_numpy_bytes_across_magnitudes(loop_calls, name):
    """Values far from 1 give numpy's bytes, whichever path computes them."""
    with np.errstate(all="ignore"):
        weights, X = MAGNITUDES[name](*_random_inputs(4, 3, 3))
    assert _same_as_einsum(weights, X)
    assert loop_calls == _route(["kept"])


TAKEN = {
    "read-only arrays": lambda w, X: (_read_only(w), _read_only(X)),
    "row slices of longer arrays": lambda w, X: (
        np.vstack([w, w])[2:8],
        np.vstack([X, X])[2:8],
    ),
    "a piece of np.split": lambda w, X: (
        np.split(w, [2])[1],
        np.split(X, [2])[1],
    ),
    "one row created in F order": lambda w, X: (
        _in_fortran_order(w[:1]),
        _in_fortran_order(X[:1]),
    ),
    "one state": lambda w, X: (_in_fortran_order(w[:, :1]), X),
    "one feature": lambda w, X: (w, _in_fortran_order(X[:, :1])),
    "one sample": lambda w, X: (w[:1].copy(), X[:1].copy()),
}


@pytest.mark.parametrize("name", list(TAKEN))
def test_arrays_numpy_treats_alike_reach_the_statistic_loop(loop_calls, name):
    """Inputs that are C-ordered float64 arrays as they are reach the loop
    and give numpy's bytes."""
    assert _same_as_einsum(*TAKEN[name](*_random_inputs(6, 3, 4)))
    assert loop_calls == _route(["kept"])


REFUSED = {
    "float32 X": lambda w, X: (w, X.astype(np.float32)),
    "float32 weights": lambda w, X: (w.astype(np.float32), X),
    "longdouble X": lambda w, X: (w, X.astype(np.longdouble)),
    "integer X": lambda w, X: (w, (X * 10).astype(np.int64)),
    "byte-swapped X": lambda w, X: (w, X.astype(">f8")),
    "F-ordered X": lambda w, X: (w, np.asfortranarray(X)),
    "F-ordered weights": lambda w, X: (np.asfortranarray(w), X),
    "stepped X": lambda w, X: (w, np.repeat(X, 2, axis=1)[:, ::2]),
    "reversed X": lambda w, X: (w, X[::-1]),
    "reversed weights": lambda w, X: (w[::-1], X),
    "unaligned X": lambda w, X: (w, _unaligned(X)),
    "unaligned weights": lambda w, X: (_unaligned(w), X),
    "subclass X": lambda w, X: (w, X.view(_Subclass)),
    "subclass weights": lambda w, X: (w.view(_Subclass), X),
    "matrix X": lambda w, X: (w, _matrix(X)),
    "masked X": lambda w, X: (w, np.ma.masked_array(X)),
    "list X": lambda w, X: (w, X.tolist()),
    "list weights": lambda w, X: (w.tolist(), X),
    "1-D X": lambda w, X: (w, X[0]),
    "3-D X": lambda w, X: (w, X[None]),
    "1-D weights": lambda w, X: (w[0], X),
    "no samples": lambda w, X: (w[:0], X[:0]),
    "no states": lambda w, X: (w[:, :0], X),
    "no features": lambda w, X: (w, X[:, :0]),
    "X one row short": lambda w, X: (w, X[:-1]),
    "one row of weights for every sample": lambda w, X: (w[:1], X),
    "one row of X for every sample": lambda w, X: (w, X[:1]),
    "boolean weights": lambda w, X: (w > 0.3, X),
}


@pytest.mark.parametrize("name", list(REFUSED))
def test_inputs_the_statistic_loop_is_not_proven_for_take_numpy_einsum(
    loop_calls, name
):
    """Another type, dtype, layout or shape never reaches the loop:
    numpy.einsum gives its values or its exception."""
    assert _same_as_einsum(*REFUSED[name](*_random_inputs(6, 3, 4)))
    assert loop_calls == []


def _put(which, index, value):
    def change(weights, X):
        arrays = {"w": weights.copy(), "X": X.copy()}
        arrays[which][index] = value
        return arrays["w"], arrays["X"]

    return change


def _inf_under_a_zero_weight(weights, X):
    weights, X = weights.copy(), X.copy()
    weights[2, 1] = 0.0
    X[2, 3] = np.inf
    return weights, X


NOT_FINITE = {
    "NaN in X": _put("X", (2, 1), np.nan),
    "inf in X": _put("X", (1, 0), np.inf),
    "NaN weight": _put("w", (4, 2), np.nan),
    "overflow in the product": lambda w, X: (w, X * 1e160),
    "overflow in the sum": lambda w, X: (
        np.ones_like(w),
        np.full_like(X, 1e154),
    ),
    "inf under a zero weight": _inf_under_a_zero_weight,
}


@pytest.mark.parametrize("name", list(NOT_FINITE))
def test_a_statistic_that_is_not_finite_is_recomputed_by_numpy(
    loop_calls, name
):
    """When the loop gives an inf or NaN, numpy.einsum runs: the NaN bytes
    are numpy's."""
    assert _same_as_einsum(*NOT_FINITE[name](*_random_inputs(6, 3, 4)))
    assert loop_calls == _route(["none"])


ERROR_DATA = {
    "plain": (1.0, ["kept"]),
    "products that underflow": (1e-170, ["kept"]),
    "an overflow": (1e160, ["none"]),
}


@pytest.mark.parametrize(
    "error_state",
    [{}, {"all": "warn"}, {"all": "raise"}],
    ids=["default", "all=warn", "all=raise"],
)
@pytest.mark.parametrize("name", list(ERROR_DATA))
def test_the_statistic_reports_nothing_in_any_error_state(
    loop_calls, name, error_state
):
    """numpy.einsum reports no floating-point error, and neither does the
    loop: no warning, no FloatingPointError, and the same route whatever
    the error state."""
    scale, taken = ERROR_DATA[name]
    weights, X = _random_inputs(6, 3, 4)
    X = X * scale
    with np.errstate(
        divide="warn", over="warn", under="ignore", invalid="warn"
    ):
        _emissions._weighted_outer_sum(weights, X)
    assert loop_calls == _route(taken)
    del loop_calls[:]
    with np.errstate(**error_state):
        got = _outcome(_emissions._weighted_outer_sum, weights, X)
        assert got == _outcome(np.einsum, ES, weights, X, X)
    outcome, caught = got
    assert outcome[0] == "array" and caught == []
    assert loop_calls == _route(taken)


def test_the_loop_route_rounds_every_operation_on_any_platform(
    loop_calls, monkeypatch
):
    """Where the loop is used, it gives the bytes of every product and sum
    rounded on its own, on any machine."""
    monkeypatch.setattr(_emissions, "_STATISTIC_LOOP", True)
    weights, X = _random_inputs(9, 2, 3)
    got = _emissions._weighted_outer_sum(weights, X)
    assert got.tobytes() == _one_rounding_per_operation(weights, X).tobytes()
    assert loop_calls == ["kept"]


def test_the_numpy_route_is_numpy_einsum(loop_calls, monkeypatch):
    """Where the loop is not used, the statistic is numpy.einsum's."""
    monkeypatch.setattr(_emissions, "_STATISTIC_LOOP", False)
    assert _same_as_einsum(*_random_inputs(9, 2, 3))
    assert loop_calls == []


def _fit_data():
    rs = np.random.RandomState(0)
    X = np.concatenate(
        [rs.standard_normal((40, 3)) + shift for shift in (-3, 0, 3)]
    )
    rs.shuffle(X)
    return X, [70, 50]


def _continue(model, X, lengths):
    """Five more iterations from where ``model`` stopped; every fitted
    array, the score and history, both decodes and the state
    probabilities, as bytes."""
    model.init_params = ""
    model.n_iter = 5
    model.fit(X, lengths)
    out = {
        name: value.tobytes()
        for name, value in vars(model).items()
        if isinstance(value, np.ndarray)
    }
    out["score"] = np.float64(model.score(X, lengths)).tobytes()
    out["history"] = np.asarray(model.monitor_.history).tobytes()
    for algorithm in ("viterbi", "map"):
        log_prob, states = model.decode(X, lengths, algorithm=algorithm)
        out[algorithm] = np.float64(log_prob).tobytes() + states.tobytes()
    out["proba"] = model.predict_proba(X, lengths).tobytes()
    return out


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("covariance_type", ["full", "tied"])
@pytest.mark.parametrize("model_class", [GaussianHMM, VariationalGaussianHMM])
def test_full_and_tied_fits_are_the_same_with_and_without_the_statistic_loop(
    loop_calls, monkeypatch, model_class, covariance_type, implementation
):
    """A fit, its score, decodes and state probabilities have the same bytes
    whether the C++ loop or numpy.einsum computes the statistic."""
    X, lengths = _fit_data()
    model = model_class(
        3,
        covariance_type=covariance_type,
        implementation=implementation,
        n_iter=1,
        tol=0.0,
        random_state=0,
    )
    model.fit(X, lengths)
    with_loop = _continue(copy.deepcopy(model), X, lengths)
    if _emissions._STATISTIC_LOOP:
        assert "kept" in loop_calls and "none" not in loop_calls
    else:
        assert loop_calls == []
    monkeypatch.setattr(_emissions, "_STATISTIC_LOOP", False)
    without_loop = _continue(copy.deepcopy(model), X, lengths)
    assert with_loop.keys() == without_loop.keys()
    assert with_loop == without_loop


# The Poisson and multinomial hooks: fewer scipy.stats calls for the arrays
# they are proven for, one call per state (hmmlearn 0.3.3's lines) for the
# rest, for every input with a report and for every state holding NaN.


class _Recorded:
    """A scipy.stats distribution whose ``logpmf`` and ``pmf`` calls are
    recorded as ``(name, number of dimensions of x)``."""

    def __init__(self, name, distribution, calls):
        self._name = name
        self._distribution = distribution
        self._calls = calls

    def logpmf(self, x, *args, **kwargs):
        self._calls.append((f"{self._name}.logpmf", np.ndim(x)))
        return self._distribution.logpmf(x, *args, **kwargs)

    def pmf(self, x, *args, **kwargs):
        self._calls.append((f"{self._name}.pmf", np.ndim(x)))
        return self._distribution.pmf(x, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._distribution, name)


@pytest.fixture
def scipy_calls(monkeypatch):
    """Record each scipy.stats Poisson and multinomial call made through
    ``hmmcraft._emissions``: a call for one state has a 2-D ``x``, a
    multinomial call for every state a 3-D one.

    The test runs in numpy's default error state, the one a caller has
    unless they change it; this test package sets every report to "warn".
    """
    calls = []
    monkeypatch.setattr(
        _emissions, "poisson", _Recorded("poisson", scipy.stats.poisson, calls)
    )
    monkeypatch.setattr(
        _emissions,
        "multinomial",
        _Recorded("multinomial", scipy.stats.multinomial, calls),
    )
    with np.errstate(
        divide="warn", over="warn", under="ignore", invalid="warn"
    ):
        yield calls


def _poisson_per_state(X, lambdas, n_components, hook):
    """hmmlearn 0.3.3's Poisson hooks, as written: one scipy.stats call per
    state. The reference the package must reproduce as bytes, with its
    warnings and errors; it is not edited with the package."""
    out = np.empty((len(X), n_components))
    for c in range(n_components):
        if hook == "log":
            out[:, c] = _emissions.poisson.logpmf(X, lambdas[c]).sum(axis=1)
        else:
            out[:, c] = _emissions.poisson.pmf(X, lambdas[c]).prod(axis=1)
    return out


def _multinomial_per_state(X, emissionprob, n_components, hook):
    """hmmlearn 0.3.3's multinomial hooks, as written: one scipy.stats call
    per state."""
    out = np.empty((len(X), n_components))
    n_trials = X.sum(axis=1)
    for c in range(n_components):
        if hook == "log":
            out[:, c] = _emissions.multinomial.logpmf(
                X, n=n_trials, p=emissionprob[c, :]
            )
        else:
            out[:, c] = _emissions.multinomial.pmf(
                X, n=n_trials, p=emissionprob[c, :]
            )
    return out


FAMILIES = {
    "poisson": (PoissonHMM, "lambdas_", _poisson_per_state),
    "multinomial": (MultinomialHMM, "emissionprob_", _multinomial_per_state),
}
HOOKS = ["log", "pmf"]


def _model(family, params, n_components):
    """A model built by hand, its emission parameters stored as they are,
    past the conversion of the attribute's setter."""
    model_class, name, _ = FAMILIES[family]
    model = model_class(n_components)
    vars(model)[name] = params
    return model


def _hook(model, hook):
    return {
        "log": model._compute_log_likelihood,
        "pmf": model._compute_likelihood,
    }[hook]


def _per_state_call(family, hook):
    """The scipy.stats call that 0.3.3 makes once per state."""
    return (f"{family}.{'logpmf' if hook == 'log' else 'pmf'}", 2)


def _compare(calls, family, hook, X, params, n_components=None):
    """Assert that the hook gives the outcome of one scipy.stats call per
    state, and return that outcome and the calls made for it; ``calls``
    then holds the hook's own calls."""
    if n_components is None:
        n_components = len(params)
    per_state = FAMILIES[family][2]
    want = _outcome(per_state, X, params, n_components, hook)
    per_state_calls = calls[:]
    del calls[:]
    got = _outcome(_hook(_model(family, params, n_components), hook), X)
    assert got == want
    return want, per_state_calls


def _calls_left_to_scipy_stats(outcome, per_state_calls):
    """The calls of one scipy.stats call per state that the hook makes too:
    all of them when they report a floating-point error, else those of the
    states whose values hold NaN."""
    result, caught = outcome
    if caught or result[:2] == ("error", "FloatingPointError"):
        return per_state_calls
    values = np.frombuffer(result[-1]).reshape(result[3])
    has_nan = np.isnan(values).any(axis=0)
    return [
        call
        for call, nan in zip(per_state_calls, has_nan, strict=False)
        if nan
    ]


def _poisson_inputs(n_samples, n_components, n_features, dtype=np.int64):
    """Seeded rates, and counts drawn from them."""
    rs = np.random.RandomState(0)
    lambdas = 5 * rs.rand(n_components, n_features)
    states = rs.randint(n_components, size=n_samples)
    X = rs.poisson(lambdas[states]).astype(dtype)
    return X, lambdas


def _multinomial_inputs(n_samples, n_components, n_features, totals=5):
    """Seeded probabilities, and counts drawn from them: ``totals`` trials
    in every row, or a number of its own in each ("per row"), the first
    row all zeros."""
    rs = np.random.RandomState(0)
    emissionprob = rs.dirichlet(np.ones(n_features), n_components)
    states = rs.randint(n_components, size=n_samples)
    if totals == "per row":
        totals = rs.randint(0, 9, size=n_samples)
        totals[0] = 0
    else:
        totals = np.full(n_samples, totals)
    X = np.array(
        [
            rs.multinomial(total, emissionprob[state])
            for total, state in zip(totals, states, strict=False)
        ],
        dtype=np.int64,
    )
    return X.reshape(n_samples, n_features), emissionprob


INPUTS = {"poisson": _poisson_inputs, "multinomial": _multinomial_inputs}
SHAPES = [
    (n_samples, n_components, n_features)
    for n_samples, n_components in [(1, 1), (2, 3), (9, 2), (40, 5)]
    for n_features in [1, 2, 3, 8, 9, 17]
]


def _with(a, index, value, dtype=None):
    """A copy of ``a``, in ``dtype`` if given, with ``value`` at
    ``index``."""
    a = a.astype(dtype or a.dtype)
    a[index] = value
    return a


def _unaligned_copy(a):
    """A copy of ``a`` whose data start one byte past an aligned
    address."""
    raw = np.frombuffer(
        bytearray(a.nbytes + 1), dtype=a.dtype, offset=1
    ).reshape(a.shape)
    raw[...] = a
    assert not raw.flags.aligned
    return raw


def test_poisson_emissions_of_a_hand_written_case(scipy_calls):
    """Two zero counts at rates 1 and 2: the log-probability is exactly -3,
    the probability exp(-1) * exp(-2), in a new C-ordered float64 array;
    scipy.stats is not called."""
    model = _model("poisson", np.array([[1.0, 2.0]]), 1)
    X = np.array([[0, 0]], dtype=np.int64)
    for hook, value in [("log", -3.0), ("pmf", np.exp(-1.0) * np.exp(-2.0))]:
        got = _hook(model, hook)(X)
        assert got.dtype == np.float64 and got.shape == (1, 1)
        assert got.flags.c_contiguous and got.flags.writeable
        assert got.tobytes() == np.array([[value]]).tobytes()
    assert scipy_calls == []


@pytest.mark.parametrize("hook", HOOKS)
@pytest.mark.parametrize(
    "dtype", [np.int64, np.float64], ids=["int64", "float64"]
)
@pytest.mark.parametrize("n_samples, n_components, n_features", SHAPES)
def test_poisson_emissions_equal_scipy_stats_per_state_as_bytes(
    scipy_calls, n_samples, n_components, n_features, dtype, hook
):
    """Plain int64 or float64 counts and float64 rates give the bytes of one
    scipy.stats call per state, without a scipy.stats call."""
    _compare(
        scipy_calls,
        "poisson",
        hook,
        *_poisson_inputs(n_samples, n_components, n_features, dtype),
    )
    assert scipy_calls == []


POISSON_WITHOUT_REPORT = {
    "a zero rate under a positive count": (
        [[2, 0, 5], [4, 0, 3], [1, 6, 0], [3, 4, 0]],
        np.int64,
        [[3.1, 0.0, 4.5], [1.6, 5.3, 0.0]],
    ),
    "a zero rate under a zero count": (
        [[0, 3]],
        np.int64,
        [[0.0, 2.0], [1.0, 1.0]],
    ),
    "counts of 2**53": ([[2**53, 1]], np.int64, [[1.0, 2.0], [3.0, 4.0]]),
    "float counts of 2**53": (
        [[2.0**53, 1.0]],
        np.float64,
        [[1.0, 2.0], [3.0, 4.0]],
    ),
    "a negative zero count": (
        [[-0.0, 1.0]],
        np.float64,
        [[1.0, 2.0], [3.0, 4.0]],
    ),
    "an infinite rate under a zero count": (
        [[0, 1]],
        np.int64,
        [[np.inf, 1.0], [2.0, 3.0]],
    ),
}


@pytest.mark.parametrize("hook", HOOKS)
@pytest.mark.parametrize("name", list(POISSON_WITHOUT_REPORT))
def test_poisson_values_scipy_gives_without_a_report_stay_direct(
    scipy_calls, name, hook
):
    """A count or rate at a limit that scipy.stats evaluates without a
    report, ``-inf`` included, keeps the shorter path and scipy's bytes."""
    X, dtype, lambdas = POISSON_WITHOUT_REPORT[name]
    want, _ = _compare(
        scipy_calls,
        "poisson",
        hook,
        np.array(X, dtype=dtype),
        np.array(lambdas),
    )
    assert want[1] == []
    assert scipy_calls == []


# name: (family, X, its dtype, parameters, hooks, error state)
REPORTS = {
    "an infinite rate under a positive count": (
        "poisson",
        [[1, 0]],
        np.int64,
        [[1.0, 2.0], [np.inf, 2.0], [0.5, 1.0]],
        HOOKS,
        {},
    ),
    "two rates whose sum overflows": (
        "poisson",
        [[0, 0]],
        np.int64,
        [[1e308, 1e308], [1.0, 1.0], [2.0, 2.0]],
        ["log"],
        {},
    ),
    "a log that overflows to a probability of 0": (
        "poisson",
        [[2.5e305]],
        np.float64,
        [[1e-10], [1.0], [2.0]],
        HOOKS,
        {},
    ),
    "a probability that underflows, reported": (
        "poisson",
        [[1000]],
        np.int64,
        [[1.0], [2.0], [3.0]],
        ["pmf"],
        {"under": "warn"},
    ),
    "a negative count over a zero probability": (
        "multinomial",
        [[-1, 1, 1], [1, 1, 1]],
        np.int64,
        [[0.0, 0.4, 0.6], [0.2, 0.3, 0.5], [0.5, 0.25, 0.25]],
        HOOKS,
        {},
    ),
}
REPORT_CASES = [
    (name, hook) for name, case in REPORTS.items() for hook in case[4]
]


def _report_case(name):
    family, X, dtype, params, _, state = REPORTS[name]
    return family, np.array(X, dtype=dtype), np.array(params), state


@pytest.mark.parametrize("name, hook", REPORT_CASES)
def test_a_report_sends_every_state_to_scipy_stats(scipy_calls, name, hook):
    """A floating-point report the caller asks for, here a warning, comes
    from scipy.stats: every state is computed by one call per state, with
    its warnings."""
    family, X, params, state = _report_case(name)
    with np.errstate(**state):
        want, per_state_calls = _compare(scipy_calls, family, hook, X, params)
    assert want[0][0] == "array" and want[1] != []
    assert per_state_calls == [_per_state_call(family, hook)] * 3
    assert [call for call in scipy_calls if call[1] == 2] == per_state_calls


@pytest.mark.parametrize("name, hook", REPORT_CASES)
def test_a_raised_report_is_the_error_of_one_call_per_state(
    scipy_calls, name, hook
):
    """Under ``all="raise"`` the hook raises the FloatingPointError of one
    scipy.stats call per state, with its text, not chained to an error of
    the shorter path."""
    family, X, params, _ = _report_case(name)
    with np.errstate(all="raise"):
        want, _ = _compare(scipy_calls, family, hook, X, params)
        with pytest.raises(FloatingPointError) as raised:
            _hook(_model(family, params, 3), hook)(X)
    assert want[0] == ("error", "FloatingPointError", str(raised.value))
    assert raised.value.__context__ is None
    assert raised.value.__cause__ is None


@pytest.mark.parametrize("name, hook", REPORT_CASES)
def test_what_the_caller_ignores_is_not_a_report(scipy_calls, name, hook):
    """Under ``all="ignore"`` the same inputs keep the shorter path; only a
    state whose values hold NaN is computed again by scipy.stats."""
    family, X, params, _ = _report_case(name)
    with np.errstate(all="ignore"):
        want, per_state_calls = _compare(scipy_calls, family, hook, X, params)
    assert want[1] == []
    assert [
        call for call in scipy_calls if call[1] == 2
    ] == _calls_left_to_scipy_stats(want, per_state_calls)


@pytest.mark.parametrize("hook", HOOKS)
def test_a_poisson_state_holding_nan_is_computed_again_by_scipy_stats(
    scipy_calls, hook
):
    """An infinite count: ``-inf`` at rate 0.5, kept; at rate 1 a NaN that
    nothing reports, so that state alone is computed by scipy.stats."""
    want, per_state_calls = _compare(
        scipy_calls,
        "poisson",
        hook,
        np.array([[np.inf]]),
        np.array([[0.5], [1.0]]),
    )
    assert want[1] == []
    assert scipy_calls == per_state_calls[1:]


POISSON_REFUSED = {
    "a fractional count": lambda X, L: (_with(X, (1, 2), 0.5, np.float64), L),
    "a negative count": lambda X, L: (_with(X, (1, 2), -1), L),
    "a negative float count": lambda X, L: (
        _with(X, (1, 2), -1.0, np.float64),
        L,
    ),
    "a NaN count": lambda X, L: (_with(X, (1, 2), np.nan, np.float64), L),
    "uint8 X holding 255": lambda X, L: (_with(X, (1, 2), 255, np.uint8), L),
    "int8 X holding 127": lambda X, L: (_with(X, (1, 2), 127, np.int8), L),
    "int32 X": lambda X, L: (X.astype(np.int32), L),
    "uint64 X": lambda X, L: (X.astype(np.uint64), L),
    "boolean X": lambda X, L: (X > 2, L),
    "object X": lambda X, L: (X.astype(object), L),
    "float32 X": lambda X, L: (X.astype(np.float32), L),
    "longdouble X": lambda X, L: (X.astype(np.longdouble), L),
    "byte-swapped X": lambda X, L: (X.astype(">i8"), L),
    "F-ordered X": lambda X, L: (np.asfortranarray(X), L),
    "stepped X": lambda X, L: (np.repeat(X, 2, axis=1)[:, ::2], L),
    "reversed X": lambda X, L: (X[::-1], L),
    "unaligned X": lambda X, L: (_unaligned_copy(X), L),
    "subclass X": lambda X, L: (X.view(_Subclass), L),
    "matrix X": lambda X, L: (_matrix(X), L),
    "masked X": lambda X, L: (np.ma.masked_array(X), L),
    "list X": lambda X, L: (X.tolist(), L),
    "1-D X": lambda X, L: (X[0], L),
    "3-D X": lambda X, L: (X[None], L),
    "no samples": lambda X, L: (X[:0], L),
    "no features": lambda X, L: (X[:, :0], L),
    "integer rates": lambda X, L: (X, np.round(L).astype(np.int64)),
    "float32 rates": lambda X, L: (X, L.astype(np.float32)),
    "F-ordered rates": lambda X, L: (X, np.asfortranarray(L)),
    "stepped rates": lambda X, L: (X, np.repeat(L, 2, axis=1)[:, ::2]),
    "one column of rates": lambda X, L: (X, np.ascontiguousarray(L[:, :1])),
    "a row of rates short": lambda X, L: (X, L[:-1], 3),
    "a row of rates too many": lambda X, L: (X, np.vstack([L, L[:1]]), 3),
    "a negative rate": lambda X, L: (X, _with(L, (1, 2), -1.0)),
    "a NaN rate": lambda X, L: (X, _with(L, (1, 2), np.nan)),
    "subclass rates": lambda X, L: (X, L.view(_Subclass)),
    "list rates": lambda X, L: (X, L.tolist()),
}


@pytest.mark.parametrize("hook", HOOKS)
@pytest.mark.parametrize("name", list(POISSON_REFUSED))
def test_inputs_the_direct_poisson_form_is_not_proven_for_take_scipy_stats(
    scipy_calls, name, hook
):
    """Another type, dtype, layout, shape or value of the counts or the
    rates is scored by one scipy.stats call per state: its values, its
    warnings or its exception."""
    _, per_state_calls = _compare(
        scipy_calls,
        "poisson",
        hook,
        *POISSON_REFUSED[name](*_poisson_inputs(6, 3, 4)),
    )
    assert scipy_calls == per_state_calls


POISSON_TAKEN = {
    "read-only arrays": lambda X, L: (_read_only(X), _read_only(L)),
    "a row slice": lambda X, L: (np.vstack([X, X])[2:8], L),
    "a piece of np.split": lambda X, L: (np.split(X, [2])[1], L),
    "one state": lambda X, L: (X, L[:1]),
    "one feature": lambda X, L: (
        np.asfortranarray(X[:, :1]),
        np.asfortranarray(L[:, :1]),
    ),
    "one sample": lambda X, L: (X[:1], L),
}


@pytest.mark.parametrize("hook", HOOKS)
@pytest.mark.parametrize(
    "dtype", [np.int64, np.float64], ids=["int64", "float64"]
)
@pytest.mark.parametrize("name", list(POISSON_TAKEN))
def test_arrays_numpy_treats_alike_reach_the_direct_poisson_form(
    scipy_calls, name, dtype, hook
):
    """Counts and rates that are C-ordered arrays of the dtypes proven, as
    they are, take the shorter path with scipy's bytes."""
    _compare(
        scipy_calls,
        "poisson",
        hook,
        *POISSON_TAKEN[name](*_poisson_inputs(6, 3, 4, dtype)),
    )
    assert scipy_calls == []


def _shorter_path_calls(family, hook):
    """The scipy.stats calls of the shorter path for a few rows."""
    return [] if family == "poisson" else [("multinomial.logpmf", 3)]


ERROR_STATES = {
    "default": {},
    "all=warn": {"all": "warn"},
    "all=raise": {"all": "raise"},
    "all=call": {"all": "call"},
}


@pytest.mark.parametrize("hook", HOOKS)
@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("state", list(ERROR_STATES))
def test_count_emissions_take_the_shorter_path_in_every_error_state(
    scipy_calls, state, family, hook
):
    """Data with nothing to report take the shorter path whatever the error
    state: no warning, no error, no call of an error handler."""
    handled = []
    X, params = INPUTS[family](9, 2, 3)
    with np.errstate(
        call=lambda kind, flag: handled.append(kind), **ERROR_STATES[state]
    ):
        want, _ = _compare(scipy_calls, family, hook, X, params)
    assert want[0][0] == "array" and want[1] == [] and handled == []
    assert scipy_calls == _shorter_path_calls(family, hook)


@pytest.mark.parametrize("hook", HOOKS)
@pytest.mark.parametrize("family", list(FAMILIES))
def test_scipy_special_reports_leave_the_count_emissions_to_scipy_stats(
    scipy_calls, family, hook
):
    """While scipy.special reports errors of its own, every state is
    computed by one scipy.stats call per state, with those reports."""
    X, params = INPUTS[family](6, 3, 4)
    X[0] = [1, 1, 1, 2]
    params[0] = [0.0, 0.5, 0.25, 0.25]
    with special.errstate(singular="warn"):
        want, per_state_calls = _compare(scipy_calls, family, hook, X, params)
    assert any(category == "SpecialFunctionWarning" for category, _ in want[1])
    assert scipy_calls == per_state_calls


def test_multinomial_emissions_of_a_hand_written_case(scipy_calls):
    """One state that always emits the first symbol: probability 1 for one
    draw of it and 0 for one of the second, from one call for every
    state."""
    model = _model("multinomial", np.array([[1.0, 0.0]]), 1)
    X = np.array([[1, 0], [0, 1]], dtype=np.int64)
    for hook, values in [("log", [[0.0], [-np.inf]]), ("pmf", [[1.0], [0.0]])]:
        got = _hook(model, hook)(X)
        assert got.dtype == np.float64 and got.shape == (2, 1)
        assert got.flags.c_contiguous and got.flags.writeable
        assert got.tobytes() == np.array(values).tobytes()
    assert scipy_calls == [("multinomial.logpmf", 3)] * 2


@pytest.mark.parametrize("hook", HOOKS)
@pytest.mark.parametrize("totals", [1, 5, "per row"])
@pytest.mark.parametrize("n_samples, n_components, n_features", SHAPES)
def test_multinomial_emissions_equal_scipy_stats_per_state_as_bytes(
    scipy_calls, n_samples, n_components, n_features, totals, hook
):
    """Plain int64 counts and float64 probabilities give the bytes of one
    scipy.stats call per state from one call for every state."""
    _compare(
        scipy_calls,
        "multinomial",
        hook,
        *_multinomial_inputs(n_samples, n_components, n_features, totals),
    )
    assert scipy_calls == [("multinomial.logpmf", 3)]


@pytest.mark.parametrize("hook", HOOKS)
@pytest.mark.parametrize("rows, n_calls", [(1, 9), (2, 5), (7, 2)])
def test_multinomial_rows_are_scored_in_pieces_with_the_same_bytes(
    scipy_calls, monkeypatch, rows, n_calls, hook
):
    """The rows are scored a piece at a time, as many rows per call as the
    bound on the entries allows; the bytes do not change."""
    X, emissionprob = _multinomial_inputs(9, 3, 4, "per row")
    monkeypatch.setattr(
        _emissions, "_MULTINOMIAL_ENTRIES", rows * emissionprob.size
    )
    _compare(scipy_calls, "multinomial", hook, X, emissionprob)
    assert scipy_calls == [("multinomial.logpmf", 3)] * n_calls


@pytest.mark.parametrize("hook", HOOKS)
def test_a_row_over_the_bound_is_scored_by_one_call_per_state(
    scipy_calls, monkeypatch, hook
):
    """When one row has more entries than the bound, every state is scored
    by its own call."""
    X, emissionprob = _multinomial_inputs(9, 3, 4, "per row")
    monkeypatch.setattr(
        _emissions, "_MULTINOMIAL_ENTRIES", emissionprob.size - 1
    )
    _, per_state_calls = _compare(
        scipy_calls, "multinomial", hook, X, emissionprob
    )
    assert scipy_calls == per_state_calls
    assert per_state_calls == [_per_state_call("multinomial", hook)] * 3


MULTINOMIAL_VALUES = {
    "a negative count over a zero probability": (
        [[-1, 1, 1], [1, 1, 1]],
        [[0.0, 0.4, 0.6], [0.2, 0.3, 0.5]],
    ),
    "a negative probability": (
        [[1, 2, 3], [0, 0, 1]],
        [[0.2, 0.3, 0.5], [-0.1, 0.6, 0.5]],
    ),
    "a NaN probability": (
        [[1, 2, 3], [0, 0, 1]],
        [[np.nan, 0.5, 0.5], [0.2, 0.3, 0.5]],
    ),
    "a row summing to 0.999": (
        [[1, 2, 3], [0, 0, 1]],
        [[0.2, 0.3, 0.5], [0.2, 0.3, 0.499]],
    ),
    "a total of the largest int64": (
        [[np.iinfo(np.int64).max, 0, 0], [1, 1, 1]],
        [[0.2, 0.3, 0.5], [0.5, 0.25, 0.25]],
    ),
}


@pytest.mark.parametrize("hook", HOOKS)
@pytest.mark.parametrize("name", list(MULTINOMIAL_VALUES))
def test_a_report_or_a_nan_sends_multinomial_states_to_scipy_stats(
    scipy_calls, name, hook
):
    """After the call for every state, a report sends every state to one
    call per state and a NaN its own state; the bytes, warnings and errors
    are those of one call per state, in either scipy version."""
    X, emissionprob = MULTINOMIAL_VALUES[name]
    want, per_state_calls = _compare(
        scipy_calls,
        "multinomial",
        hook,
        np.array(X, dtype=np.int64),
        np.array(emissionprob),
    )
    assert [call for call in scipy_calls if call[1] == 3] == [
        ("multinomial.logpmf", 3)
    ]
    assert [
        call for call in scipy_calls if call[1] == 2
    ] == _calls_left_to_scipy_stats(want, per_state_calls)


MULTINOMIAL_REFUSED = {
    "whole float64 counts": lambda X, P: (X.astype(np.float64), P),
    "a fractional count": lambda X, P: (_with(X, (1, 2), 0.5, np.float64), P),
    "int32 X": lambda X, P: (X.astype(np.int32), P),
    "uint8 X": lambda X, P: (X.astype(np.uint8), P),
    "boolean X": lambda X, P: (X > 0, P),
    "F-ordered X": lambda X, P: (np.asfortranarray(X), P),
    "stepped X": lambda X, P: (np.repeat(X, 2, axis=1)[:, ::2], P),
    "reversed X": lambda X, P: (X[::-1], P),
    "unaligned X": lambda X, P: (_unaligned_copy(X), P),
    "subclass X": lambda X, P: (X.view(_Subclass), P),
    "list X": lambda X, P: (X.tolist(), P),
    "1-D X": lambda X, P: (X[0], P),
    "no samples": lambda X, P: (X[:0], P),
    "float32 probabilities": lambda X, P: (X, P.astype(np.float32)),
    "F-ordered probabilities": lambda X, P: (X, np.asfortranarray(P)),
    "stepped probabilities": lambda X, P: (X, np.repeat(P, 2, axis=1)[:, ::2]),
    "a row of probabilities short": lambda X, P: (X, P[:-1], 3),
    "object probabilities": lambda X, P: (X, P.astype(object)),
    "list probabilities": lambda X, P: (X, P.tolist()),
}


@pytest.mark.parametrize("hook", HOOKS)
@pytest.mark.parametrize("name", list(MULTINOMIAL_REFUSED))
def test_inputs_the_one_call_form_is_not_proven_for_take_a_call_per_state(
    scipy_calls, name, hook
):
    """Another type, dtype, layout or shape of the counts or the
    probabilities is scored by one scipy.stats call per state."""
    _, per_state_calls = _compare(
        scipy_calls,
        "multinomial",
        hook,
        *MULTINOMIAL_REFUSED[name](*_multinomial_inputs(6, 3, 4, "per row")),
    )
    assert scipy_calls == per_state_calls


class _PoissonPerState(PoissonHMM):
    """PoissonHMM with hmmlearn 0.3.3's hooks."""

    def _compute_log_likelihood(self, X):
        return _poisson_per_state(X, self.lambdas_, self.n_components, "log")

    def _compute_likelihood(self, X):
        return _poisson_per_state(X, self.lambdas_, self.n_components, "pmf")


class _MultinomialPerState(MultinomialHMM):
    """MultinomialHMM with hmmlearn 0.3.3's hooks."""

    def _compute_log_likelihood(self, X):
        return _multinomial_per_state(
            X, self.emissionprob_, self.n_components, "log"
        )

    def _compute_likelihood(self, X):
        return _multinomial_per_state(
            X, self.emissionprob_, self.n_components, "pmf"
        )


def _count_fit_data(family, dtype):
    """Two sequences sampled from upstream's test models."""
    if family == "poisson":
        model = PoissonHMM(2)
        model.startprob_ = np.array([0.6, 0.4])
        model.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]])
        model.lambdas_ = np.array([[3.1, 1.4, 4.5], [1.6, 5.3, 0.1]])
    else:
        model = MultinomialHMM(2, n_trials=5)
        model.startprob_ = np.array([0.6, 0.4])
        model.transmat_ = np.array([[0.8, 0.2], [0.2, 0.8]])
        model.emissionprob_ = np.array(
            [[0.5, 0.3, 0.1, 0.1], [0.1, 0.1, 0.4, 0.4]]
        )
    X, _ = model.sample(100, random_state=np.random.RandomState(0))
    return X.astype(dtype), [60, 40]


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize(
    "family, dtype",
    [
        ("poisson", np.int64),
        ("poisson", np.float64),
        ("multinomial", np.int64),
    ],
    ids=["poisson-int64", "poisson-float64", "multinomial"],
)
def test_count_fits_are_the_same_as_with_one_scipy_call_per_state(
    scipy_calls, family, dtype, implementation
):
    """A fit, its score, decodes and state probabilities have the same bytes
    with the package's hooks as with one scipy.stats call per state."""
    X, lengths = _count_fit_data(family, dtype)
    model_class, per_state_class = {
        "poisson": (PoissonHMM, _PoissonPerState),
        "multinomial": (MultinomialHMM, _MultinomialPerState),
    }[family]
    kwargs = {"n_trials": 5} if family == "multinomial" else {}
    model = per_state_class(
        2,
        implementation=implementation,
        n_iter=1,
        tol=0.0,
        random_state=0,
        **kwargs,
    )
    model.fit(X, lengths)
    shorter = copy.deepcopy(model)
    shorter.__class__ = model_class
    del scipy_calls[:]
    with_package_hooks = _continue(shorter, X, lengths)
    if family == "poisson":
        assert scipy_calls == []
    else:
        assert scipy_calls
        assert all(ndim == 3 for _, ndim in scipy_calls)
    per_state = _continue(copy.deepcopy(model), X, lengths)
    assert with_package_hooks.keys() == per_state.keys()
    assert with_package_hooks == per_state


# GMMHMM's E-step: the mixture densities _compute_log_likelihood computed for
# a sequence are read back by the statistics hook, when computing them again
# would give the same values and report nothing.


@pytest.fixture
def density_calls(monkeypatch):
    """Record each Gaussian log-density call made through
    ``hmmcraft._emissions`` (one per state, mixture hook and sequence)."""
    real = _emissions.log_multivariate_normal_density
    calls = []

    def recording(*args, **kwargs):
        calls.append(len(args[0]))
        return real(*args, **kwargs)

    monkeypatch.setattr(
        _emissions, "log_multivariate_normal_density", recording
    )
    return calls


# numpy's default error state; this test package sets every report to "warn".
DEFAULT_STATE = {
    "divide": "warn",
    "over": "warn",
    "under": "ignore",
    "invalid": "warn",
}


class _GMMUpstreamStatistics(GMMHMM):
    """GMMHMM with hmmlearn 0.3.3's statistics hook: it computes every
    state's mixture densities again."""

    def _accumulate_sufficient_statistics(
        self, stats, X, lattice, post_comp, fwdlattice, bwdlattice
    ):
        BaseHMM._accumulate_sufficient_statistics(
            self, stats, X, lattice, post_comp, fwdlattice, bwdlattice
        )

        n_samples, _ = X.shape

        post_mix = np.zeros((n_samples, self.n_components, self.n_mix))
        for p in range(self.n_components):
            log_denses = self._compute_log_weighted_gaussian_densities(X, p)
            log_normalize(log_denses, axis=-1)
            with np.errstate(under="ignore"):
                post_mix[:, p, :] = np.exp(log_denses)

        with np.errstate(under="ignore"):
            post_comp_mix = post_comp[:, :, None] * post_mix

        stats["post_mix_sum"] += post_comp_mix.sum(axis=0)
        stats["post_sum"] += post_comp.sum(axis=0)

        if "m" in self.params:  # means stats
            stats["m_n"] += np.einsum("ijk,il->jkl", post_comp_mix, X)

        if "c" in self.params:  # covariance stats
            centered = X[:, None, None, :] - self.means_

            def outer_f(x):  # Outer product over features.
                return x[..., :, None] * x[..., None, :]

            if self.covariance_type == "full":
                centered_dots = outer_f(centered)
                c_n = np.einsum(
                    "ijk,ijklm->jklm", post_comp_mix, centered_dots
                )
            elif self.covariance_type == "diag":
                centered2 = np.square(centered, out=centered)  # reuse
                c_n = np.einsum("ijk,ijkl->jkl", post_comp_mix, centered2)
            elif self.covariance_type == "spherical":
                # Faster than (x**2).sum(-1).
                centered_norm2 = np.einsum("...i,...i", centered, centered)
                c_n = np.einsum("ijk,ijk->jk", post_comp_mix, centered_norm2)
            elif self.covariance_type == "tied":
                centered_dots = outer_f(centered)
                c_n = np.einsum("ijk,ijklm->jlm", post_comp_mix, centered_dots)

            stats["c_n"] += c_n


def _gmm_far_data():
    """``_fit_data()``'s samples as three sequences, with one sample of the
    second far from every state: it falls back under scaling."""
    X, _ = _fit_data()
    X = X.copy()
    X[50] = 1000.0
    return X, [40, 40, 40]


def _gmm_start(covariance_type, n_mix, implementation="log"):
    """A two-state GMMHMM after one iteration on ``_fit_data()``."""
    X, lengths = _fit_data()
    with np.errstate(**DEFAULT_STATE):
        model = GMMHMM(
            2,
            n_mix=n_mix,
            covariance_type=covariance_type,
            implementation=implementation,
            n_iter=1,
            tol=0.0,
            random_state=0,
        ).fit(X, lengths)
    model.init_params = ""
    return model


def _as(model, cls):
    """A copy of ``model`` as an instance of ``cls``."""
    other = copy.deepcopy(model)
    other.__class__ = cls
    return other


def _estep(model, X, lengths, state):
    """What one E-step gives under ``state``: the statistics and the
    log-likelihood as bytes, or the error (type, text, whether it has a
    context); the warnings (category, text, file name) and the calls of an
    error handler."""
    handled = []
    with (
        warnings.catch_warnings(record=True) as caught,
        np.errstate(call=lambda kind, flag: handled.append(kind), **state),
    ):
        warnings.simplefilter("always")
        try:
            stats, logprob = model._do_estep(X, lengths)
        except (FloatingPointError, ValueError) as error:
            got = (
                "error",
                type(error).__name__,
                str(error),
                error.__context__ is None,
            )
        else:
            got = {
                name: np.asarray(value).tobytes()
                for name, value in stats.items()
            }
            got["logprob"] = np.float64(logprob).tobytes()
    reports = [
        (w.category.__name__, str(w.message), os.path.basename(w.filename))
        for w in caught
    ]
    return got, reports, handled


COVARIANCE_TYPES = ["spherical", "diag", "full", "tied"]


@pytest.mark.parametrize("n_mix", [1, 2])
@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPES)
def test_an_e_step_computes_each_mixture_density_once_per_sequence(
    density_calls, covariance_type, implementation, n_mix
):
    """In numpy's default error state, one iteration computes each state's
    mixture densities once per sequence, for one mixture and for several."""
    X, lengths = _fit_data()
    model = _gmm_start(covariance_type, n_mix, implementation)
    del density_calls[:]
    with np.errstate(**DEFAULT_STATE):
        model.fit(X, lengths)
    assert len(density_calls) == model.n_components * len(lengths)
    assert "_estep_values" not in vars(model)


@pytest.mark.parametrize("n_mix", [1, 2])
@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPES)
def test_a_zero_weight_computes_the_densities_again(
    density_calls, covariance_type, implementation, n_mix
):
    """A zero mixture weight in one state is a report in numpy's default
    error state: every state's densities are computed again, as by
    hmmlearn 0.3.3's hooks, and the warnings come as often."""
    X, lengths = _fit_data()
    model = _gmm_start(covariance_type, n_mix, implementation)
    model.weights_ = model.weights_.copy()
    model.weights_[1] = np.eye(n_mix)[0] if n_mix > 1 else 0.0
    upstream = _as(model, _GMMUpstreamStatistics)
    del density_calls[:]
    got = _estep(model, X, lengths, DEFAULT_STATE)
    assert len(density_calls) == 2 * model.n_components * len(lengths)
    assert got[1] != []
    assert got == _estep(upstream, X, lengths, DEFAULT_STATE)


@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPES)
def test_a_sequence_that_falls_back_computes_its_densities_for_each_pass_only(
    density_calls, caplog, covariance_type
):
    """Of three sequences under scaling, the one that falls back computes
    its densities in the scaling pass and in the log pass, and its
    statistics read the second: four calls per state in all."""
    X, lengths = _gmm_far_data()
    model = _gmm_start(covariance_type, 2, "scaling")
    del density_calls[:]
    with (
        np.errstate(**DEFAULT_STATE),
        caplog.at_level("WARNING", logger="hmmcraft.base"),
    ):
        model._do_estep(X, lengths)
    assert any("1 of 3 sequences" in r.getMessage() for r in caplog.records)
    assert len(density_calls) == model.n_components * 4


@pytest.mark.parametrize("n_mix", [1, 2])
@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPES)
def test_scaling_computes_the_densities_again_when_underflow_is_reported(
    density_calls, covariance_type, n_mix
):
    """The scaling pass computes the densities with underflow ignored: when
    the caller asks for underflow reports, the statistics compute them
    again in the caller's state, and log keeps them."""
    X, lengths = _fit_data()
    calls = {}
    for implementation in ["log", "scaling"]:
        model = _gmm_start(covariance_type, n_mix, implementation)
        del density_calls[:]
        with np.errstate(under="warn"):
            model._do_estep(X, lengths)
        calls[implementation] = len(density_calls)
    assert calls["log"] == model.n_components * len(lengths)
    assert calls["scaling"] == 2 * calls["log"]


@pytest.mark.parametrize(
    "state", [DEFAULT_STATE, {"all": "warn"}], ids=["default", "all=warn"]
)
@pytest.mark.parametrize("data", ["two", "three-with-a-fallback"])
@pytest.mark.parametrize("n_mix", [1, 2, 3])
@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPES)
def test_e_step_statistics_are_those_of_hmmlearn_statistics_hook_as_bytes(
    covariance_type, implementation, n_mix, data, state
):
    """Every statistic of an E-step and its log-likelihood have the bytes,
    warnings and handler calls of hmmlearn 0.3.3's statistics hook."""
    X, lengths = _fit_data() if data == "two" else _gmm_far_data()
    model = _gmm_start(covariance_type, n_mix, implementation)
    upstream = _as(model, _GMMUpstreamStatistics)
    got = _estep(model, X, lengths, state)
    assert isinstance(got[0], dict)
    assert got == _estep(upstream, X, lengths, state)


@pytest.mark.parametrize("n_mix", [1, 2])
@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPES)
def test_gmm_fits_are_the_same_as_with_hmmlearn_statistics_hook(
    covariance_type, implementation, n_mix
):
    """A fit, its score, decodes and state probabilities have the same bytes
    as with hmmlearn 0.3.3's statistics hook."""
    X, lengths = _fit_data()
    model = _gmm_start(covariance_type, n_mix, implementation)
    with np.errstate(**DEFAULT_STATE):
        once = _continue(_as(model, GMMHMM), X, lengths)
        upstream = _continue(_as(model, _GMMUpstreamStatistics), X, lengths)
    assert once.keys() == upstream.keys()
    assert once == upstream


def _zero_weight(model):
    model.weights_ = model.weights_.copy()
    model.weights_[0] = np.eye(model.n_mix)[0]


def _tiny_variances(model):
    model.covars_ = model.covars_.copy()
    model.covars_[0] = np.finfo(float).tiny


GMM_REPORTS = {
    "a zero weight": _zero_weight,
    "variances of the smallest normal float": _tiny_variances,
}


@pytest.mark.parametrize(
    "state",
    [{"all": "warn"}, {"all": "call"}, {"all": "raise"}],
    ids=["all=warn", "all=call", "all=raise"],
)
@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("name", list(GMM_REPORTS))
def test_a_report_in_the_densities_is_given_as_often_as_by_hmmlearn_hooks(
    name, implementation, state
):
    """A floating-point report in the densities comes as often, from the
    same file, as with hmmlearn 0.3.3's hooks; a raised one is the same
    error with no context; the statistics have the same bytes."""
    X, lengths = _fit_data()
    model = _gmm_start("diag", 2, implementation)
    GMM_REPORTS[name](model)
    got = _estep(model, X, lengths, state)
    assert got[1] != [] or got[2] != [] or got[0][0] == "error"
    assert got == _estep(_as(model, _GMMUpstreamStatistics), X, lengths, state)


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize(
    "fitted_n_mix, n_mix",
    [(3, 1), (1, 3)],
    ids=["three-wide-n_mix-1", "one-wide-n_mix-3"],
)
def test_densities_wider_or_narrower_than_n_mix_give_hmmlearn_outcome(
    fitted_n_mix, n_mix, implementation
):
    """Mixture parameters whose width is not ``n_mix`` give the outcome of
    hmmlearn 0.3.3's hooks: what is read back follows the densities'
    width, not ``n_mix``."""
    X, lengths = _fit_data()
    model = _gmm_start("diag", fitted_n_mix, implementation)
    model.n_mix = n_mix
    got = _estep(model, X, lengths, DEFAULT_STATE)
    assert got == _estep(
        _as(model, _GMMUpstreamStatistics), X, lengths, DEFAULT_STATE
    )


class _OtherWeightsForTheLikelihood:
    """Scores with equal mixture weights, and keeps its own for the rest."""

    def _compute_log_likelihood(self, X):
        weights = self.weights_
        self.weights_ = np.full_like(weights, 1 / self.n_mix)
        try:
            return super()._compute_log_likelihood(X)
        finally:
            self.weights_ = weights


class _CountedDensities:
    """Counts the calls of the weighted densities hook."""

    counted = 0

    def _compute_log_weighted_gaussian_densities(self, X, i_comp):
        type(self).counted += 1
        return super()._compute_log_weighted_gaussian_densities(X, i_comp)


class _StatisticsOfACopy:
    """Accumulates the statistics of a copy of each sequence."""

    def _accumulate_sufficient_statistics(self, stats, X, *args):
        super()._accumulate_sufficient_statistics(stats, X.copy(), *args)


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize(
    "override",
    [_OtherWeightsForTheLikelihood, _CountedDensities, _StatisticsOfACopy],
    ids=["likelihood", "densities", "statistics"],
)
def test_a_subclass_replacing_a_gmm_hook_gets_hmmlearn_statistics(
    override, implementation
):
    """A subclass that replaces one of the three mixture hooks gets the
    statistics of the same subclass of hmmlearn 0.3.3's hooks: the
    densities are computed again, twice per state and sequence."""
    X, lengths = _gmm_far_data()
    model = _gmm_start("diag", 2, implementation)
    outcomes, counts = [], []
    for base in (GMMHMM, _GMMUpstreamStatistics):
        cls = type(
            f"{override.__name__}{base.__name__}",
            (override, base),
            {"counted": 0},
        )
        outcomes.append(_estep(_as(model, cls), X, lengths, DEFAULT_STATE))
        counts.append(cls.counted)
    assert outcomes[0] == outcomes[1]
    assert counts[0] == counts[1]
    if override is _CountedDensities:
        assert counts[0] >= 2 * model.n_components * len(lengths)
