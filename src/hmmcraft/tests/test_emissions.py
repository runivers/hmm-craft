"""The statistic of the full and tied covariances, ``hmmcraft._emissions``.

On the machines where it is used, a C++ loop computes the statistic for the
arrays it is proven for; numpy's einsum computes it for the rest.
"""

import copy
import platform
import sys

import numpy as np
import pytest

from hmmcraft import _emissions, _emissionsc
from hmmcraft.hmm import GaussianHMM
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
    return (_outcome(_emissions._weighted_outer_sum, weights, X)
            == _outcome(np.einsum, ES, weights, X, X))


def _random_inputs(n_samples, n_components, n_features):
    """Seeded weights, each row summing to one, and samples."""
    rs = np.random.RandomState(0)
    weights = rs.dirichlet(np.ones(n_components), n_samples)
    X = rs.standard_normal((n_samples, n_features))
    return weights, X


@pytest.mark.parametrize("machine, maxsize, used", [
    ("x86_64", 2 ** 63 - 1, True),
    ("AMD64", 2 ** 63 - 1, True),
    ("arm64", 2 ** 63 - 1, False),
    ("aarch64", 2 ** 63 - 1, False),
    ("ARM64", 2 ** 63 - 1, False),
    ("amd64", 2 ** 63 - 1, False),
    ("ppc64le", 2 ** 63 - 1, False),
    ("", 2 ** 63 - 1, False),
    ("x86_64", 2 ** 31 - 1, False),
    ("AMD64", 2 ** 31 - 1, False),
])
def test_the_loop_is_used_on_64_bit_x86_only(machine, maxsize, used):
    """The loop computes the statistic only for a 64-bit interpreter on
    x86_64 or AMD64."""
    assert _emissions._loop_platform(machine, maxsize) is used


# The machine names of the platforms wheels are built for.
WHEEL_PLATFORMS = {"x86_64": True, "AMD64": True,
                   "arm64": False, "aarch64": False}


def test_the_statistic_takes_the_route_of_this_platform(loop_calls):
    """On this machine the statistic goes to the loop if, and only if, the
    machine is 64-bit x86; every other machine keeps numpy's einsum."""
    used = (WHEEL_PLATFORMS.get(platform.machine(), False)
            and sys.maxsize > 2 ** 32)
    assert _emissions._STATISTIC_LOOP is used
    _emissions._weighted_outer_sum(*_random_inputs(5, 2, 3))
    assert loop_calls == (["kept"] if used else [])


@pytest.mark.parametrize("n_samples, n_components, n_features", [
    (n_samples, n_components, n_features)
    for n_samples, n_components in [(1, 1), (2, 3), (9, 2), (40, 5)]
    for n_features in [1, 2, 3, 8, 9, 17]
] + [
    # Sample counts and entries per sample on either side of 8192.
    (8193, 2, 1), (16385, 1, 2), (3, 1, 91), (2, 3, 64),
])
def test_the_statistic_equals_numpy_einsum_as_bytes(
        loop_calls, n_samples, n_components, n_features):
    """The statistic has the bytes of numpy.einsum's, and the loop computes
    it where it is used, unless there is a single entry."""
    assert _same_as_einsum(
        *_random_inputs(n_samples, n_components, n_features))
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
        np.vstack([w, w])[2:8], np.vstack([X, X])[2:8]),
    "a piece of np.split": lambda w, X: (
        np.split(w, [2])[1], np.split(X, [2])[1]),
    "one row created in F order": lambda w, X: (
        _in_fortran_order(w[:1]), _in_fortran_order(X[:1])),
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
        loop_calls, name):
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
    "overflow in the sum": lambda w, X: (np.ones_like(w),
                                         np.full_like(X, 1e154)),
    "inf under a zero weight": _inf_under_a_zero_weight,
}


@pytest.mark.parametrize("name", list(NOT_FINITE))
def test_a_statistic_that_is_not_finite_is_recomputed_by_numpy(
        loop_calls, name):
    """When the loop gives an inf or NaN, numpy.einsum runs: the NaN bytes
    are numpy's."""
    assert _same_as_einsum(*NOT_FINITE[name](*_random_inputs(6, 3, 4)))
    assert loop_calls == _route(["none"])


ERROR_DATA = {
    "plain": (1.0, ["kept"]),
    "products that underflow": (1e-170, ["kept"]),
    "an overflow": (1e160, ["none"]),
}


@pytest.mark.parametrize("error_state", [{}, {"all": "warn"},
                                         {"all": "raise"}],
                         ids=["default", "all=warn", "all=raise"])
@pytest.mark.parametrize("name", list(ERROR_DATA))
def test_the_statistic_reports_nothing_in_any_error_state(
        loop_calls, name, error_state):
    """numpy.einsum reports no floating-point error, and neither does the
    loop: no warning, no FloatingPointError, and the same route whatever
    the error state."""
    scale, taken = ERROR_DATA[name]
    weights, X = _random_inputs(6, 3, 4)
    X = X * scale
    with np.errstate(divide="warn", over="warn", under="ignore",
                     invalid="warn"):
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
        loop_calls, monkeypatch):
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
    X = np.concatenate([rs.standard_normal((40, 3)) + shift
                        for shift in (-3, 0, 3)])
    rs.shuffle(X)
    return X, [70, 50]


def _continue(model, X, lengths):
    """Five more iterations from where ``model`` stopped; every fitted
    array, the score and history, both decodes and the state
    probabilities, as bytes."""
    model.init_params = ""
    model.n_iter = 5
    model.fit(X, lengths)
    out = {name: value.tobytes() for name, value in vars(model).items()
           if isinstance(value, np.ndarray)}
    out["score"] = np.float64(model.score(X, lengths)).tobytes()
    out["history"] = np.asarray(model.monitor_.history).tobytes()
    for algorithm in ("viterbi", "map"):
        log_prob, states = model.decode(X, lengths, algorithm=algorithm)
        out[algorithm] = np.float64(log_prob).tobytes() + states.tobytes()
    out["proba"] = model.predict_proba(X, lengths).tobytes()
    return out


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("covariance_type", ["full", "tied"])
@pytest.mark.parametrize("model_class", [GaussianHMM,
                                         VariationalGaussianHMM])
def test_full_and_tied_fits_are_the_same_with_and_without_the_statistic_loop(
        loop_calls, monkeypatch, model_class, covariance_type,
        implementation):
    """A fit, its score, decodes and state probabilities have the same bytes
    whether the C++ loop or numpy.einsum computes the statistic."""
    X, lengths = _fit_data()
    model = model_class(3, covariance_type=covariance_type,
                        implementation=implementation, n_iter=1, tol=0.0,
                        random_state=0)
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
