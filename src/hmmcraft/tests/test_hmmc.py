"""Input checks of the C++ kernel, ``hmmcraft._hmmc``."""

import numpy as np
import pytest

from hmmcraft import _hmmc
from hmmcraft.hmm import CategoricalHMM


def _arguments(name, n_samples, n_components):
    """Return valid-looking arguments of kernel function ``name``.

    The matrices have ``n_samples`` rows and ``n_components`` columns.
    """
    start = np.full(n_components, 1 / max(n_components, 1))
    trans = np.full((n_components, n_components), 1 / max(n_components, 1))
    frame = np.full((n_samples, n_components), 0.5)
    lattice = np.zeros((n_samples, n_components))
    return {
        "forward_log": (start, trans, frame),
        "forward_scaling": (start, trans, frame),
        "backward_log": (start, trans, frame),
        "backward_scaling": (start, trans, frame, np.ones(n_samples)),
        "compute_log_xi_sum": (lattice, trans, lattice, frame),
        "compute_scaling_xi_sum": (lattice, trans, lattice, frame),
        "viterbi": (start, trans, frame),
    }[name]


KERNEL_FUNCTIONS = [
    "forward_log", "forward_scaling", "backward_log", "backward_scaling",
    "compute_log_xi_sum", "compute_scaling_xi_sum", "viterbi",
]


@pytest.mark.parametrize("name", KERNEL_FUNCTIONS)
def test_kernel_refuses_a_sequence_without_samples(name):
    """A matrix with no rows raises before any row is read or written."""
    with pytest.raises(ValueError, match="at least one sample"):
        getattr(_hmmc, name)(*_arguments(name, 0, 2))


@pytest.mark.parametrize("name", KERNEL_FUNCTIONS)
def test_kernel_refuses_a_model_without_states(name):
    """A model with no states raises before any column is read."""
    with pytest.raises(ValueError, match="at least one state"):
        getattr(_hmmc, name)(*_arguments(name, 3, 0))


@pytest.mark.parametrize("name", KERNEL_FUNCTIONS)
def test_kernel_accepts_a_single_sample(name):
    """One sample is the smallest sequence and runs."""
    getattr(_hmmc, name)(*_arguments(name, 1, 2))


class _NoRowsCategoricalHMM(CategoricalHMM):
    """An emission hook that returns no rows, whatever ``X`` holds."""

    def _compute_likelihood(self, X):
        return super()._compute_likelihood(X)[:0]

    def _compute_log_likelihood(self, X):
        return super()._compute_log_likelihood(X)[:0]


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("method", ["score", "decode", "predict_proba"])
def test_emission_hook_without_rows_raises(implementation, method):
    """A subclass whose emissions have no rows gets an error, not a value."""
    h = _NoRowsCategoricalHMM(2, n_features=2, init_params="",
                              implementation=implementation)
    h.startprob_ = np.array([0.5, 0.5])
    h.transmat_ = np.array([[0.5, 0.5], [0.5, 0.5]])
    h.emissionprob_ = np.array([[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError, match="at least one sample"):
        getattr(h, method)(np.array([[0], [1]]))
