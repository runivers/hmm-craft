"""Input checks and values of the emission loops, ``hmmcraft._emissionsc``.

``tests/test_stats.py`` compares the loop with the numpy expression it
stands for; this file calls the module directly.
"""

import numpy as np
import pytest

from hmmcraft import _emissionsc

N_SAMPLES, N_COMPONENTS, N_FEATURES = 3, 2, 4


def _arguments():
    """Seeded samples, means, variances and head the loop accepts."""
    rs = np.random.RandomState(0)
    X = rs.standard_normal((N_SAMPLES, N_FEATURES))
    means = rs.standard_normal((N_COMPONENTS, N_FEATURES))
    covars = rs.uniform(0.5, 2.0, (N_COMPONENTS, N_FEATURES))
    head = rs.standard_normal(N_COMPONENTS)
    return {"X": X, "means": means, "covars": covars, "head": head}


def _unaligned(a):
    """A copy of ``a`` whose data starts one byte off a double's alignment."""
    raw = np.frombuffer(bytearray(8 * a.size + 1), dtype=np.float64,
                        offset=1).reshape(a.shape)
    raw[...] = a
    assert not raw.flags.aligned
    return raw


def _call(arguments):
    return _emissionsc.log_density_diag(
        arguments["X"], arguments["means"], arguments["covars"],
        arguments["head"])


def test_log_density_diag_of_a_hand_written_case():
    """-0.5 * (head + sum of (x - mean)**2 / variance), in a new float64
    array of shape (n_samples, n_components) that the caller may write."""
    got = _emissionsc.log_density_diag(
        np.array([[1.0, 2.0]]), np.array([[0.0, 0.0]]),
        np.array([[1.0, 4.0]]), np.array([0.5]))
    # -0.5 * (0.5 + 1 / 1 + 4 / 4)
    assert type(got) is np.ndarray
    assert got.dtype == np.float64 and got.shape == (1, 1)
    assert got.flags.c_contiguous and got.flags.writeable
    assert got[0, 0] == -1.25


NOT_FLOAT64 = "expected a float64 numpy array"
NOT_C_ALIGNED = "expected a C-contiguous, aligned array"
SHAPE = "shape mismatch"
EMPTY = "empty array"

REFUSED = {
    "float32 X": ("X", lambda a: a["X"].astype(np.float32), NOT_FLOAT64),
    "integer means": ("means", lambda a: a["means"].astype(np.int64),
                      NOT_FLOAT64),
    "byte-swapped covars": ("covars", lambda a: a["covars"].astype(">f8"),
                            NOT_FLOAT64),
    "list X": ("X", lambda a: a["X"].tolist(), NOT_FLOAT64),
    "None head": ("head", lambda a: None, NOT_FLOAT64),
    "F-ordered X": ("X", lambda a: np.asfortranarray(a["X"]), NOT_C_ALIGNED),
    "every second column of X": (
        "X", lambda a: np.repeat(a["X"], 2, axis=1)[:, ::2], NOT_C_ALIGNED),
    "unaligned X": ("X", lambda a: _unaligned(a["X"]), NOT_C_ALIGNED),
    "F-ordered covars": ("covars", lambda a: np.asfortranarray(a["covars"]),
                         NOT_C_ALIGNED),
    "1-D X": ("X", lambda a: a["X"][0].copy(), SHAPE),
    "2-D head": ("head", lambda a: a["head"][None, :].copy(), SHAPE),
    "X one column wider": (
        "X", lambda a: np.hstack([a["X"], a["X"][:, :1]]), SHAPE),
    "covars one row short": ("covars", lambda a: a["covars"][:-1].copy(),
                             SHAPE),
    "covars one column short": (
        "covars", lambda a: a["covars"][:, :-1].copy(), SHAPE),
    "head one short": ("head", lambda a: a["head"][:-1].copy(), SHAPE),
}


@pytest.mark.parametrize("name", list(REFUSED))
def test_log_density_diag_refuses_an_array_it_would_have_to_convert(name):
    """An argument that is not a float64 array, not read as it is, or not of
    the matching shape raises ValueError; nothing is converted."""
    key, make, text = REFUSED[name]
    arguments = _arguments()
    arguments[key] = make(arguments)
    with pytest.raises(ValueError) as error:
        _call(arguments)
    assert type(error.value) is ValueError
    assert str(error.value) == text


@pytest.mark.parametrize("name, n_samples, n_components, n_features", [
    ("no sample", 0, N_COMPONENTS, N_FEATURES),
    ("no state", N_SAMPLES, 0, N_FEATURES),
    ("no feature", N_SAMPLES, N_COMPONENTS, 0),
])
def test_log_density_diag_refuses_an_empty_dimension(
        name, n_samples, n_components, n_features):
    """A dimension of length zero raises ValueError before any loop runs."""
    with pytest.raises(ValueError) as error:
        _emissionsc.log_density_diag(
            np.zeros((n_samples, n_features)),
            np.zeros((n_components, n_features)),
            np.ones((n_components, n_features)), np.zeros(n_components))
    assert type(error.value) is ValueError
    assert str(error.value) == EMPTY


def _set(key, index, value):
    def change(arguments):
        arguments[key] = arguments[key].copy()
        arguments[key][index] = value
    return change


def _scale_X(arguments):
    arguments["X"] = arguments["X"] * 1e200


NOT_FINITE = {
    "NaN in the first sample": _set("X", (0, 1), np.nan),
    "NaN in the last sample": _set("X", (-1, -1), np.nan),
    "infinite mean": _set("means", (1, 0), np.inf),
    "zero variance under a non-zero difference": _set("covars", (0, 2), 0.0),
    "infinite head": _set("head", 1, np.inf),
    "NaN head": _set("head", 0, np.nan),
    "a square that overflows": _scale_X,
}


@pytest.mark.parametrize("name", list(NOT_FINITE))
def test_log_density_diag_returns_none_when_a_value_is_not_finite(name):
    """None, not an array, when any of the log-densities is inf or NaN."""
    arguments = _arguments()
    NOT_FINITE[name](arguments)
    assert _call(arguments) is None


# The statistic of the full and tied covariances.


def _one_rounding_per_operation(weights, X):
    """For each state, the sum over the samples of (weight * x[k]) * x[l],
    from zeros, sample after sample.

    Every product and every sum is a numpy call of its own, so each is
    rounded on its own on every platform.
    """
    out = np.zeros((weights.shape[1], X.shape[1], X.shape[1]))
    for w, x in zip(weights, X):
        out = (w[:, None, None] * x[:, None]) * x + out
    return out


def _statistic_arguments(n_samples=N_SAMPLES, n_components=N_COMPONENTS,
                         n_features=N_FEATURES):
    """Seeded weights, each row summing to one, and samples."""
    rs = np.random.RandomState(0)
    weights = rs.dirichlet(np.ones(n_components), n_samples)
    X = rs.standard_normal((n_samples, n_features))
    return {"weights": weights, "X": X}


def _statistic(arguments):
    return _emissionsc.weighted_outer_sum(
        arguments["weights"], arguments["X"])


def test_weighted_outer_sum_of_a_hand_written_case():
    """For each state, the sum over the samples of weight * x * x', in a new
    float64 array of shape (n_components, n_features, n_features) that the
    caller may write."""
    got = _emissionsc.weighted_outer_sum(
        np.array([[1.0, 0.5], [2.0, 0.0]]), np.array([[1.0, 2.0],
                                                      [3.0, 0.0]]))
    # State 0: 1 * [[1, 2], [2, 4]] + 2 * [[9, 0], [0, 0]]
    # State 1: 0.5 * [[1, 2], [2, 4]] + 0 * [[9, 0], [0, 0]]
    assert type(got) is np.ndarray
    assert got.dtype == np.float64 and got.shape == (2, 2, 2)
    assert got.flags.c_contiguous and got.flags.writeable
    assert got.tolist() == [[[19.0, 2.0], [2.0, 4.0]],
                            [[0.5, 1.0], [1.0, 2.0]]]


@pytest.mark.parametrize("n_samples, n_components",
                         [(1, 1), (2, 3), (9, 2), (40, 5)])
@pytest.mark.parametrize("n_features", [1, 2, 3, 8, 9, 17])
def test_weighted_outer_sum_rounds_every_product_and_sum_on_its_own(
        n_features, n_samples, n_components):
    """The loop gives the bytes of the same operations done one numpy call
    at a time: entry [c, k, l] gets (weight * x[k]) * x[l] added to it."""
    arguments = _statistic_arguments(n_samples, n_components, n_features)
    got = _statistic(arguments)
    want = _one_rounding_per_operation(arguments["weights"], arguments["X"])
    assert got.shape == want.shape
    assert got.tobytes() == want.tobytes()


def test_weighted_outer_sum_does_not_fuse_the_product_into_the_sum():
    """The second product is rounded before it is added.

    Entry [0, 0, 1] is 0.5 + (A * 1) * B. The product rounded on its own is
    0.25 + 2**-54, and the sum, a tie, rounds to 0.75; a fused multiply-add
    gives 0.75 + 2**-53.
    """
    step = 2.0 ** -53
    a, b = 0.25 + step, 1.0 - step
    got = _emissionsc.weighted_outer_sum(
        np.array([[0.5], [a]]), np.array([[1.0, 1.0], [1.0, b]]))
    assert got[0, 0, 1] == 0.75
    assert got.tolist() == [[[0.75 + step, 0.75], [0.75, 0.75]]]


def test_weighted_outer_sum_adds_to_positive_zero():
    """The sums start from +0.0: products of -0.0 leave every entry +0.0."""
    got = _emissionsc.weighted_outer_sum(
        np.array([[1.0]]), np.array([[-0.0, 1.0]]))
    assert got.tolist() == [[[0.0, 0.0], [0.0, 1.0]]]
    assert not np.signbit(got).any()


STATISTIC_REFUSED = {
    "float32 X": ("X", lambda a: a["X"].astype(np.float32), NOT_FLOAT64),
    "integer weights": ("weights", lambda a: a["weights"].astype(np.int64),
                        NOT_FLOAT64),
    "byte-swapped X": ("X", lambda a: a["X"].astype(">f8"), NOT_FLOAT64),
    "list X": ("X", lambda a: a["X"].tolist(), NOT_FLOAT64),
    "None weights": ("weights", lambda a: None, NOT_FLOAT64),
    "F-ordered X": ("X", lambda a: np.asfortranarray(a["X"]), NOT_C_ALIGNED),
    "every second column of X": (
        "X", lambda a: np.repeat(a["X"], 2, axis=1)[:, ::2], NOT_C_ALIGNED),
    "unaligned X": ("X", lambda a: _unaligned(a["X"]), NOT_C_ALIGNED),
    "F-ordered weights": (
        "weights", lambda a: np.asfortranarray(a["weights"]), NOT_C_ALIGNED),
    "1-D X": ("X", lambda a: a["X"][0].copy(), SHAPE),
    "3-D weights": ("weights", lambda a: a["weights"][None].copy(), SHAPE),
    "X one row short": ("X", lambda a: a["X"][:-1].copy(), SHAPE),
    "weights one row short": (
        "weights", lambda a: a["weights"][:-1].copy(), SHAPE),
}


@pytest.mark.parametrize("name", list(STATISTIC_REFUSED))
def test_weighted_outer_sum_refuses_an_array_it_would_have_to_convert(name):
    """An argument that is not a float64 array, not read as it is, or not of
    the matching shape raises ValueError; nothing is converted."""
    key, make, text = STATISTIC_REFUSED[name]
    arguments = _statistic_arguments()
    arguments[key] = make(arguments)
    with pytest.raises(ValueError) as error:
        _statistic(arguments)
    assert type(error.value) is ValueError
    assert str(error.value) == text


@pytest.mark.parametrize("name, n_samples, n_components, n_features", [
    ("no sample", 0, N_COMPONENTS, N_FEATURES),
    ("no state", N_SAMPLES, 0, N_FEATURES),
    ("no feature", N_SAMPLES, N_COMPONENTS, 0),
])
def test_weighted_outer_sum_refuses_an_empty_dimension(
        name, n_samples, n_components, n_features):
    """A dimension of length zero raises ValueError before any loop runs."""
    with pytest.raises(ValueError) as error:
        _emissionsc.weighted_outer_sum(
            np.ones((n_samples, n_components)),
            np.zeros((n_samples, n_features)))
    assert type(error.value) is ValueError
    assert str(error.value) == EMPTY


def _scale(key, factor):
    def change(arguments):
        arguments[key] = arguments[key] * factor
    return change


def _large_samples(arguments):
    arguments["weights"] = np.ones((2, 1))
    arguments["X"] = np.full((2, 1), 1e154)


def _inf_under_a_zero_weight(arguments):
    _set("weights", (1, 0), 0.0)(arguments)
    _set("X", (1, 2), np.inf)(arguments)


STATISTIC_NOT_FINITE = {
    "NaN in the first sample": _set("X", (0, 1), np.nan),
    "NaN in the last sample": _set("X", (-1, -1), np.nan),
    "NaN weight": _set("weights", (1, 1), np.nan),
    "infinite X": _set("X", (2, 0), np.inf),
    "a product that overflows": _scale("X", 1e160),
    "a sum that overflows": _large_samples,
    "inf under a zero weight": _inf_under_a_zero_weight,
}


@pytest.mark.parametrize("name", list(STATISTIC_NOT_FINITE))
def test_weighted_outer_sum_returns_none_when_an_entry_is_not_finite(name):
    """None, not an array, when any entry of the statistic is inf or NaN."""
    arguments = _statistic_arguments()
    STATISTIC_NOT_FINITE[name](arguments)
    assert _statistic(arguments) is None
