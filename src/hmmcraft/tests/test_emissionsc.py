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
