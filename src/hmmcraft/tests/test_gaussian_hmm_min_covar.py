from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_allclose

from .. import hmm

COVARIANCE_TYPES = ["spherical", "diag", "full", "tied"]


class RecordingGaussianHMM(hmm.GaussianHMM):
    """A GaussianHMM that keeps its covariances after every M-step."""

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        if not hasattr(self, "recorded"):
            self.recorded = []
        self.recorded.append(np.array(self._covars_))


def _rounded_sequences(n_sequences=5, length=400, scale=0.5, seed=0):
    # Rounded Gaussian values: about two thirds are exactly zero, so one
    # state can explain the zeros with a vanishing variance.
    rng = np.random.RandomState(seed)
    X = np.round(rng.normal(0, scale, (n_sequences, length)))
    return X.reshape(-1, 1), [length] * n_sequences


def _duplicated_features(seed=1):
    # The second feature is an exact copy of the first: every covariance
    # matrix is singular without a floor, and each variance can collapse.
    X, lengths = _rounded_sequences(seed=seed)
    return np.column_stack([X[:, 0], X[:, 0]]), lengths


def _smallest(covars, covariance_type):
    """Smallest variance (spherical, diag) or eigenvalue (full, tied)."""
    if covariance_type in ("spherical", "diag"):
        return covars.min()
    return min(np.linalg.eigvalsh(cv).min()
               for cv in covars.reshape(-1, *covars.shape[-2:]))


@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPES)
def test_covars_floor_holds_after_every_m_step(covariance_type):
    X, lengths = _duplicated_features()
    h = RecordingGaussianHMM(2, covariance_type, covars_floor=1e-3,
                             n_iter=30, tol=-np.inf, random_state=0)
    h.fit(X, lengths)
    smallest = [_smallest(cv, covariance_type) for cv in h.recorded]
    assert len(smallest) == 30
    assert min(smallest) >= 1e-3 * (1 - 1e-9)
    # The floor binds somewhere.
    assert min(smallest) <= 1e-3 * (1 + 1e-9)


def _fixed_start(covariance_type, variances=(0.1, 1.0), **kwargs):
    """A model with explicit starting parameters, so a fit does not depend
    on the k-means start."""
    h = hmm.GaussianHMM(2, covariance_type, init_params="", random_state=0,
                        **kwargs)
    h.startprob_ = np.array([0.5, 0.5])
    h.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])
    h.means_ = np.array([[0.0], [1.5]])
    variances = np.array(variances)
    h.covars_ = {"spherical": variances,
                 "diag": variances[:, None],
                 "full": variances[:, None, None],
                 "tied": variances[:1, None]}[covariance_type]
    return h


@pytest.mark.parametrize("covariance_type", ["spherical", "diag"])
def test_covars_floor_off_by_default(covariance_type):
    X, lengths = _rounded_sequences()
    kwargs = {"n_iter": 100, "tol": 1e-4}
    default = _fixed_start(covariance_type, **kwargs).fit(X, lengths)
    explicit = _fixed_start(
        covariance_type, covars_floor=None, **kwargs).fit(X, lengths)
    # Without a floor the variance of the "zeros" state ends far below
    # min_covar.
    assert default._covars_.min() < 1e-3
    assert default.covars_floor is None
    for name in ("_covars_", "means_", "transmat_", "startprob_"):
        assert getattr(default, name).tobytes() == (
            getattr(explicit, name).tobytes())


@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPES)
def test_covars_floor_leaves_an_explicit_start_below_it_alone(
        covariance_type):
    X, lengths = _rounded_sequences()
    starts = []
    for floor in (None, 1e-3):
        h = _fixed_start(covariance_type, variances=(1e-5, 2e-5),
                         covars_floor=floor, n_iter=0)
        starts.append(h.fit(X, lengths)._covars_)
    assert starts[0].tobytes() == starts[1].tobytes()
    assert starts[1].max() < 1e-3


@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPES)
def test_covars_floor_leaves_a_computed_start_below_it_alone(covariance_type):
    # init_params with "c": the start covariance is computed from the data
    # (here far below the floor), and the floor applies from the first M-step.
    X = np.random.RandomState(3).normal(0, 1e-3, (200, 2))
    starts = []
    for floor in (None, 1e-3):
        h = hmm.GaussianHMM(2, covariance_type, covars_floor=floor,
                            min_covar=1e-9, init_params="c", n_iter=0,
                            random_state=0)
        h.startprob_ = np.array([0.5, 0.5])
        h.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])
        h.means_ = np.zeros((2, 2))
        starts.append(h.fit(X)._covars_)
    assert starts[0].tobytes() == starts[1].tobytes()
    assert starts[1].max() < 1e-3


@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPES)
def test_inactive_covars_floor_changes_nothing(covariance_type):
    # From the same starting point, a floor that never binds gives exactly
    # the fit without a floor.
    rng = np.random.RandomState(2)
    X = np.concatenate([rng.normal(0, 1, (500, 2)),
                        rng.normal(4, 2, (500, 2))])
    start = hmm.GaussianHMM(2, covariance_type, random_state=0)
    start._init(X)
    fits = []
    for floor in (1e-3, None):
        h = hmm.GaussianHMM(2, covariance_type, covars_floor=floor,
                            n_iter=20, init_params="", random_state=0)
        h.startprob_ = start.startprob_
        h.transmat_ = start.transmat_
        h.means_ = start.means_
        h.covars_ = start._covars_
        fits.append(h.fit(X))
    assert np.linalg.eigvalsh(fits[0].covars_).min() > 1e-2
    assert_allclose(fits[0].means_, fits[1].means_, rtol=0, atol=0)
    assert_allclose(fits[0].covars_, fits[1].covars_, rtol=0, atol=0)


@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPES)
def test_monotone_em_with_covars_floor(covariance_type):
    # The floored update is the exact M-step under the constraint, so the
    # log-likelihood still never decreases.
    X, lengths = _rounded_sequences()
    h = hmm.GaussianHMM(2, covariance_type, covars_floor=1e-2, n_iter=30,
                        tol=-np.inf, random_state=0)
    h.fit(X, lengths)
    history = np.asarray(h.monitor_.history)
    assert (np.diff(history) >= -1e-8 * np.abs(history[1:])).all()


@pytest.mark.parametrize(
    "floor", [0, -1, np.nan, np.inf, True, False, np.True_, "1e-3", [1e-3],
              10 ** 400],
    ids=["zero", "negative", "nan", "inf", "True", "False", "np.True_",
         "string", "list", "int_beyond_float"])
def test_covars_floor_rejects_non_positive_and_non_numeric(floor):
    X, lengths = _rounded_sequences()
    h = hmm.GaussianHMM(2, covars_floor=floor, random_state=0)
    with pytest.raises(ValueError, match="covars_floor"):
        h.fit(X, lengths)


def test_covars_floor_accepts_any_real_number_as_its_float_value():
    # A Fraction is a real number: it floors exactly as its float does.
    X, lengths = _rounded_sequences()
    fits = [_fixed_start("diag", covars_floor=floor, n_iter=10).fit(X, lengths)
            for floor in (Fraction(1, 1000), 1e-3)]
    assert fits[0]._covars_.min() >= 1e-3
    assert fits[0]._covars_.tobytes() == fits[1]._covars_.tobytes()


def test_model_pickled_without_covars_floor_loads():
    X, lengths = _rounded_sequences()
    fitted = hmm.GaussianHMM(2, random_state=0).fit(X, lengths)
    state = fitted.__getstate__()
    del state["covars_floor"]
    loaded = hmm.GaussianHMM.__new__(hmm.GaussianHMM)
    loaded.__setstate__(state)
    assert loaded.get_params()["covars_floor"] is None
    assert_allclose(loaded.predict(X, lengths), fitted.predict(X, lengths))
