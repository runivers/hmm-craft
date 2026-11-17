"""Input checks of the C++ kernel, ``hmmcraft._hmmc``."""

import numpy as np
import pytest

from hmmcraft import _hmmc, base
from hmmcraft.hmm import CategoricalHMM

UNDERFLOW_TEXT = (
    "forward pass failed with underflow; consider using "
    "implementation='log' instead"
)


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


class _OneColumnCategoricalHMM(CategoricalHMM):
    """An emission hook that returns one state's column of two."""

    def _compute_likelihood(self, X):
        return super()._compute_likelihood(X)[:, :1]

    def _compute_log_likelihood(self, X):
        return super()._compute_log_likelihood(X)[:, :1]


def _hooked(cls, implementation):
    h = cls(2, n_features=2, init_params="", implementation=implementation)
    h.startprob_ = np.array([0.5, 0.5])
    h.transmat_ = np.array([[0.5, 0.5], [0.5, 0.5]])
    h.emissionprob_ = np.array([[0.5, 0.5], [0.5, 0.5]])
    return h


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("method", ["score", "decode", "predict_proba", "fit"])
def test_emission_hook_without_rows_raises(implementation, method):
    """A subclass whose emissions have no rows gets an error, not a value."""
    h = _hooked(_NoRowsCategoricalHMM, implementation)
    with pytest.raises(ValueError, match="at least one sample"):
        getattr(h, method)(np.array([[0], [1]]))


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("method", ["score", "decode", "predict_proba", "fit"])
def test_emission_hook_with_too_few_columns_raises(implementation, method):
    """
    A subclass whose emissions do not have one column per state gets the
    kernel's shape error under either implementation, not a fallback.
    """
    h = _hooked(_OneColumnCategoricalHMM, implementation)
    with pytest.raises(ValueError, match="^shape mismatch$"):
        getattr(h, method)(np.array([[0], [1]]))


@pytest.mark.parametrize("row", [0, 2], ids=["first-row", "later-row"])
def test_forward_scaling_underflow_text_starts_with_the_constant(row):
    """
    A sample whose probabilities sum to zero over the states raises
    ``ValueError`` with the kernel's full text, which the package recognises
    by its start.
    """
    start, trans, frame = _arguments("forward_scaling", 4, 2)
    frame[row] = 0
    with pytest.raises(ValueError) as info:
        _hmmc.forward_scaling(start, trans, frame)
    assert type(info.value) is ValueError
    assert str(info.value) == UNDERFLOW_TEXT
    assert str(info.value).startswith(base._SCALING_UNDERFLOW)
    assert base._is_scaling_underflow(info.value)


@pytest.mark.parametrize(
    "n_samples, n_components, frame_columns",
    [
        (0, 2, 2),
        (3, 0, 0),
        (3, 2, 3),
    ],
    ids=["no-sample", "no-state", "shape-mismatch"],
)
def test_other_kernel_errors_are_not_taken_for_underflow(
    n_samples, n_components, frame_columns
):
    """The kernel's input errors never match the underflow test."""
    start, trans, _ = _arguments("forward_scaling", n_samples, n_components)
    frame = np.full((n_samples, frame_columns), 0.5)
    with pytest.raises(ValueError) as info:
        _hmmc.forward_scaling(start, trans, frame)
    assert not base._is_scaling_underflow(info.value)
