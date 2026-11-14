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


@pytest.mark.parametrize("covariance_type", ["spherical", "diag"])
def test_covars_floor_off_by_default(covariance_type):
    X, lengths = _rounded_sequences()
    kwargs = {"n_iter": 100, "tol": 1e-4, "random_state": 0}
    default = hmm.GaussianHMM(2, covariance_type, **kwargs).fit(X, lengths)
    explicit = hmm.GaussianHMM(
        2, covariance_type, covars_floor=None, **kwargs).fit(X, lengths)
    # Without a floor the variance of the "zeros" state ends far below
    # min_covar.
    assert default._covars_.min() < 1e-3
    assert default.covars_floor is None
    for name in ("_covars_", "means_", "transmat_", "startprob_"):
        assert getattr(default, name).tobytes() == (
            getattr(explicit, name).tobytes())


@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPES)
def test_covars_floor_leaves_the_start_alone(covariance_type):
    X, lengths = _rounded_sequences()
    starts = []
    for floor in (None, 1e-3):
        h = hmm.GaussianHMM(covariance_type=covariance_type, n_components=2,
                            covars_floor=floor, n_iter=0, random_state=0)
        starts.append(h.fit(X, lengths)._covars_.tobytes())
    assert starts[0] == starts[1]


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


@pytest.mark.parametrize("floor", [0, -1, np.nan, np.inf])
def test_covars_floor_rejects_non_positive(floor):
    X, lengths = _rounded_sequences()
    h = hmm.GaussianHMM(2, covars_floor=floor, random_state=0)
    with pytest.raises(ValueError, match="covars_floor"):
        h.fit(X, lengths)


def test_model_pickled_without_covars_floor_loads():
    X, lengths = _rounded_sequences()
    fitted = hmm.GaussianHMM(2, random_state=0).fit(X, lengths)
    state = fitted.__getstate__()
    del state["covars_floor"]
    loaded = hmm.GaussianHMM.__new__(hmm.GaussianHMM)
    loaded.__setstate__(state)
    assert loaded.get_params()["covars_floor"] is None
    assert_allclose(loaded.predict(X, lengths), fitted.predict(X, lengths))
