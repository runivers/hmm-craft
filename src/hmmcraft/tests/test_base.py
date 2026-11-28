import copy
import decimal
import itertools
import re
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import special

from hmmcraft import _hmmc, base, hmm, vhmm
from hmmcraft.base import BaseHMM, ConvergenceMonitor

from .test_sequence_lengths import new_hmm


class TestMonitor:
    def test_converged_by_iterations(self):
        m = ConvergenceMonitor(tol=1e-3, n_iter=2, verbose=False)
        assert not m.converged
        m.report(-0.01)
        assert not m.converged
        m.report(-0.1)
        assert m.converged

    def test_converged_by_log_prob(self):
        m = ConvergenceMonitor(tol=1e-3, n_iter=10, verbose=False)
        for log_prob in [-0.03, -0.02, -0.01]:
            m.report(log_prob)
            assert not m.converged

        m.report(-0.0101)
        assert m.converged

    def test_reset(self):
        m = ConvergenceMonitor(tol=1e-3, n_iter=10, verbose=False)
        m.iter = 1
        m.history.append(-0.01)
        m._reset()
        assert m.iter == 0
        assert not m.history

    def test_report_first_iteration(self, capsys):
        m = ConvergenceMonitor(tol=1e-3, n_iter=10, verbose=True)
        m.report(-0.01)
        out, err = capsys.readouterr()
        assert not out
        expected = m._template.format(iter=1, log_prob=-0.01, delta=np.nan)
        assert err.splitlines() == [expected]

    def test_report(self, capsys):
        n_iter = 10
        m = ConvergenceMonitor(tol=1e-3, n_iter=n_iter, verbose=True)
        for i in reversed(range(n_iter)):
            m.report(-0.01 * i)

        out, err = capsys.readouterr()
        assert not out
        assert len(err.splitlines()) == n_iter
        assert len(m.history) == n_iter


class StubHMM(BaseHMM):
    """An HMM with hardcoded observation probabilities."""

    def _compute_log_likelihood(self, X):
        return self.log_frameprob


class TestBaseAgainstWikipedia:
    def setup_method(self, method):
        # Example from http://en.wikipedia.org/wiki/Forward-backward_algorithm
        self.frameprob = np.asarray(
            [[0.9, 0.2], [0.9, 0.2], [0.1, 0.8], [0.9, 0.2], [0.9, 0.2]]
        )
        self.log_frameprob = np.log(self.frameprob)
        h = StubHMM(2)
        h.transmat_ = [[0.7, 0.3], [0.3, 0.7]]
        h.startprob_ = [0.5, 0.5]
        h.log_frameprob = self.log_frameprob
        h.frameprob = self.frameprob
        self.hmm = h

    def test_do_forward_scaling_pass(self):
        log_prob, fwdlattice, _scaling_factors = _hmmc.forward_scaling(
            self.hmm.startprob_, self.hmm.transmat_, self.frameprob
        )
        ref_log_prob = -3.3725
        assert round(log_prob, 4) == ref_log_prob
        reffwdlattice = np.exp(
            [
                [0.4500, 0.1000],
                [0.3105, 0.0410],
                [0.0230, 0.0975],
                [0.0408, 0.0150],
                [0.0298, 0.0046],
            ]
        )
        assert_allclose(np.exp(fwdlattice), reffwdlattice, 4)

    def test_do_forward_pass(self):
        log_prob, fwdlattice = _hmmc.forward_log(
            self.hmm.startprob_, self.hmm.transmat_, self.log_frameprob
        )

        ref_log_prob = -3.3725
        assert round(log_prob, 4) == ref_log_prob
        reffwdlattice = np.array(
            [
                [0.4500, 0.1000],
                [0.3105, 0.0410],
                [0.0230, 0.0975],
                [0.0408, 0.0150],
                [0.0298, 0.0046],
            ]
        )
        assert_allclose(np.exp(fwdlattice), reffwdlattice, 4)

    def test_do_backward_scaling_pass(self):
        _log_prob, _fwdlattice, scaling_factors = _hmmc.forward_scaling(
            self.hmm.startprob_, self.hmm.transmat_, self.frameprob
        )
        bwdlattice = _hmmc.backward_scaling(
            self.hmm.startprob_,
            self.hmm.transmat_,
            self.frameprob,
            scaling_factors,
        )
        refbwdlattice = np.array(
            [
                [0.0661, 0.0455],
                [0.0906, 0.1503],
                [0.4593, 0.2437],
                [0.6900, 0.4100],
                [1.0000, 1.0000],
            ]
        )
        scaling_factors = np.cumprod(scaling_factors[::-1])[::-1]
        bwdlattice_scaled = bwdlattice / scaling_factors[:, None]
        # Answer will be equivalent when the scaling factor is accounted for
        assert_allclose(bwdlattice_scaled, refbwdlattice, 4)

    def test_do_backward_log_pass(self):
        bwdlattice = _hmmc.backward_log(
            self.hmm.startprob_, self.hmm.transmat_, self.log_frameprob
        )
        refbwdlattice = np.array(
            [
                [0.0661, 0.0455],
                [0.0906, 0.1503],
                [0.4593, 0.2437],
                [0.6900, 0.4100],
                [1.0000, 1.0000],
            ]
        )
        assert_allclose(np.exp(bwdlattice), refbwdlattice, 4)

    def test_do_viterbi_pass(self):
        log_prob, state_sequence = _hmmc.viterbi(
            self.hmm.startprob_, self.hmm.transmat_, self.log_frameprob
        )
        refstate_sequence = [0, 0, 1, 0, 0]
        assert_allclose(state_sequence, refstate_sequence)
        ref_log_prob = -4.4590
        assert round(log_prob, 4) == ref_log_prob

    def test_score_samples(self):
        # ``StubHMM` ignores the values in ```X``, so we just pass in an
        # array of the appropriate shape.
        log_prob, posteriors = self.hmm.score_samples(self.log_frameprob)
        assert_allclose(posteriors.sum(axis=1), np.ones(len(posteriors)))
        ref_log_prob = -3.3725
        assert round(log_prob, 4) == ref_log_prob
        refposteriors = np.array(
            [
                [0.8673, 0.1327],
                [0.8204, 0.1796],
                [0.3075, 0.6925],
                [0.8204, 0.1796],
                [0.8673, 0.1327],
            ]
        )
        assert_allclose(posteriors, refposteriors, atol=1e-4)

    def test_generate_samples(self):
        _X0, Z0 = self.hmm.sample(n_samples=10)
        _X, Z = self.hmm.sample(n_samples=10, currstate=Z0[-1])
        assert len(Z0) == len(Z) == 10 and Z[0] == Z0[-1]

    @pytest.mark.parametrize("n_samples", [0, -1, np.array(0), 0.5, False])
    def test_sample_refuses_fewer_than_one(self, n_samples):
        """``sample`` draws at least one sample."""
        with pytest.raises(ValueError, match="n_samples must be at least 1"):
            self.hmm.sample(n_samples)

    def test_sample_float_keeps_type_error(self):
        """A float count of at least 1 fails where it did before."""
        with pytest.raises(TypeError):
            self.hmm.sample(2.0)

    @pytest.mark.parametrize(
        "n_samples, length",
        [
            (1, 1),
            (True, 1),
            (np.int64(1), 1),
            (np.array(1), 1),
            (np.array(True), 1),
            (np.True_, 1),
            (np.array(2, dtype=object), 2),
            (np.array(3), 3),
        ],
    )
    def test_sample_count_as_given(self, n_samples, length):
        """
        A count of at least 1 draws what the integer draws, whatever type
        holds it.
        """
        X, Z = self.hmm.sample(n_samples, random_state=0)
        X_int, Z_int = self.hmm.sample(length, random_state=0)
        assert len(X) == len(Z) == length
        assert X.tobytes() == X_int.tobytes()
        assert Z.tobytes() == Z_int.tobytes()


class TestBaseConsistentWithGMM:
    def setup_method(self, method):
        n_components = 8
        n_samples = 10

        self.log_frameprob = np.log(
            np.random.random((n_samples, n_components))
        )

        h = StubHMM(n_components)
        h.log_frameprob = self.log_frameprob

        # If startprob and transmat are uniform across all states (the
        # default), the transitions are uninformative - the model
        # reduces to a GMM with uniform mixing weights (in terms of
        # posteriors, not likelihoods).
        h.startprob_ = np.ones(n_components) / n_components
        h.transmat_ = np.ones((n_components, n_components)) / n_components

        self.hmm = h

    def test_score_samples(self):
        _log_prob, hmmposteriors = self.hmm.score_samples(self.log_frameprob)

        n_samples, n_components = self.log_frameprob.shape
        assert_allclose(hmmposteriors.sum(axis=1), np.ones(n_samples))

        norm = special.logsumexp(self.log_frameprob, axis=1)[:, np.newaxis]
        gmmposteriors = np.exp(
            self.log_frameprob - np.tile(norm, (1, n_components))
        )
        assert_allclose(hmmposteriors, gmmposteriors)

    def test_decode(self):
        _log_prob, state_sequence = self.hmm.decode(self.log_frameprob)

        _n_samples, n_components = self.log_frameprob.shape
        norm = special.logsumexp(self.log_frameprob, axis=1)[:, np.newaxis]
        gmmposteriors = np.exp(
            self.log_frameprob - np.tile(norm, (1, n_components))
        )
        gmmstate_sequence = gmmposteriors.argmax(axis=1)
        assert_allclose(state_sequence, gmmstate_sequence)


def test_base_hmm_attributes():
    n_components = 20
    startprob = np.random.random(n_components)
    startprob /= startprob.sum()
    transmat = np.random.random((n_components, n_components))
    transmat /= np.tile(transmat.sum(axis=1)[:, np.newaxis], (1, n_components))

    h = StubHMM(n_components)

    assert h.n_components == n_components

    h.startprob_ = startprob
    assert_allclose(h.startprob_, startprob)

    with pytest.raises(ValueError):
        h.startprob_ = 2 * startprob
        h._check()
    with pytest.raises(ValueError):
        h.startprob_ = []
        h._check()
    with pytest.raises(ValueError):
        h.startprob_ = np.zeros((n_components - 2, 2))
        h._check()

    h.startprob_ = startprob
    h.transmat_ = transmat
    assert_allclose(h.transmat_, transmat)
    with pytest.raises(ValueError):
        h.transmat_ = 2 * transmat
        h._check()
    with pytest.raises(ValueError):
        h.transmat_ = []
        h._check()
    with pytest.raises(ValueError):
        h.transmat_ = np.zeros((n_components - 2, n_components))
        h._check()

    # Rows that sum to 1 with a negative entry are not distributions.
    h.transmat_ = transmat
    negative = np.zeros(n_components)
    negative[:2] = [1.5, -0.5]
    h.startprob_ = negative
    with pytest.raises(ValueError, match="startprob_"):
        h._check()
    h.startprob_ = startprob
    h.transmat_ = transmat.copy()
    h.transmat_[0] = 0
    h.transmat_[0, :2] = [1.2, -0.2]
    with pytest.raises(ValueError, match="transmat_"):
        h._check()


def _sum_check_models():
    rs = np.random.RandomState(7)
    continuous = rs.normal(size=(40, 2)) + 3 * rs.randint(0, 2, (40, 1))
    symbols = rs.randint(0, 3, (40, 1))
    counts = rs.poisson(4, (40, 3))
    multinomial = rs.multinomial(5, [0.2, 0.3, 0.5], size=40)
    cases = [
        (
            "GaussianHMM",
            lambda: hmm.GaussianHMM(2, n_iter=4, random_state=0),
            continuous,
        ),
        (
            "CategoricalHMM",
            lambda: hmm.CategoricalHMM(2, n_iter=4, random_state=0),
            symbols,
        ),
        (
            "MultinomialHMM",
            lambda: hmm.MultinomialHMM(2, 5, n_iter=4, random_state=0),
            multinomial,
        ),
        (
            "PoissonHMM",
            lambda: hmm.PoissonHMM(2, n_iter=4, random_state=0),
            counts,
        ),
    ]
    cases += [
        (
            f"GMMHMM-{cov}",
            lambda cov=cov: hmm.GMMHMM(
                2, 2, covariance_type=cov, n_iter=4, random_state=0
            ),
            continuous,
        )
        for cov in ("spherical", "diag", "full", "tied")
    ]
    return cases


SUM_CHECK_MODELS = _sum_check_models()


@pytest.mark.parametrize(
    "factory, X",
    [c[1:] for c in SUM_CHECK_MODELS],
    ids=[c[0] for c in SUM_CHECK_MODELS],
)
def test_check_of_float64_sums_does_not_call_allclose(monkeypatch, factory, X):
    """
    The check of a fitted model, whose distributions are float64, tests
    their sums without ``np.allclose``.
    """
    model = factory().fit(X)
    calls = []
    allclose = np.allclose

    def spy(*args, **kwargs):
        calls.append(args)
        return allclose(*args, **kwargs)

    monkeypatch.setattr(np, "allclose", spy)
    model._check()
    assert calls == []


def _sum_sweep():
    """
    100,001 consecutive doubles centred on each edge of ``np.allclose``'s
    default tolerance around 1, then NaN, the infinities, the two zeros, 1
    and 2.
    """
    k = np.arange(-50_000, 50_001)
    edges = [1 - (1e-08 + 1e-05), 1 + (1e-08 + 1e-05)]
    return np.concatenate(
        [edge + k * np.spacing(edge) for edge in edges]
        + [[np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, 2.0]]
    )


def test_float64_sums_to_1_equals_allclose():
    """
    On float64 sums, the sum test gives ``np.allclose(s, 1)``'s answer for
    each value of the sweep, as a scalar and as a one-element array, and for
    seeded rows of them, without a floating-point error.
    """
    values = _sum_sweep()
    assert values.size == 200_009
    expected = np.isclose(values, 1)
    with np.errstate(all="raise"):
        scalars = [base._sums_to_1(value) for value in values]
        rows = [base._sums_to_1(values[i : i + 1]) for i in range(values.size)]
    assert_array_equal(scalars, expected)
    assert_array_equal(rows, expected)
    rng = np.random.default_rng(0)
    differ = []
    for n in (1, 2, 3, 4, 7, 8, 9, 16, 17, 33):
        for _ in range(200):
            row = rng.choice(values, size=n)
            if rng.random() < 0.5:
                row[rng.random(n) < 0.7] = 1.0
            with np.errstate(all="raise"):
                got = base._sums_to_1(row)
            if got != np.allclose(row, 1):
                differ.append(row)
    assert differ == []


def _matrix(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        return np.matrix(a)


@pytest.mark.parametrize(
    "rows",
    [
        np.array([[0.5, 0.5], [0.2, 0.8]], dtype=object),
        np.array([[0.5, 0.5], [0.2, 0.7]], dtype=object),
        np.array([[2**62, 2**62, 1]]),
        np.array([[0.5, 0.5], [0.2, 0.8]], dtype=np.float32),
        np.array([decimal.Decimal("0.5")] * 2, dtype=object),
        _matrix([[0.5, 0.5], [0.2, 0.7]]),
        np.ma.array([[0.5, 0.5], [0.2, 0.7]], mask=[[0, 0], [1, 1]]),
    ],
    ids=[
        "object",
        "object-not-1",
        "int64-wrapping",
        "float32",
        "Decimal",
        "matrix",
        "masked",
    ],
)
def test_other_sums_to_1_are_allclose(rows):
    """
    A sum that is not a float64 scalar or array gets the answer, or the
    exception, of ``np.allclose(s, 1)`` under the running numpy.
    """
    s = rows.sum(axis=-1)

    def outcome(test):
        try:
            return bool(test(s))
        except (TypeError, ValueError) as exc:
            return type(exc)

    assert outcome(base._sums_to_1) == outcome(lambda s: np.allclose(s, 1))


LEFT_RIGHT_START = np.array([1.0, 0.0, 0.0])
LEFT_RIGHT_TRANSMAT = np.array(
    [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]
)


def _with(array, index, value):
    array = array.copy()
    array[index] = value
    return array


@pytest.mark.parametrize("implementation", ["scaling", "log"])
@pytest.mark.parametrize(
    "startprob, transmat",
    [
        (LEFT_RIGHT_START, LEFT_RIGHT_TRANSMAT),
        (
            np.array([1.0, -0.0, -0.0]),
            np.where(LEFT_RIGHT_TRANSMAT == 0, -0.0, LEFT_RIGHT_TRANSMAT),
        ),
        (LEFT_RIGHT_START, _with(LEFT_RIGHT_TRANSMAT, (0, 2), 5e-324)),
        (np.array([1 + 1e-9, 0.0, 0.0]), LEFT_RIGHT_TRANSMAT),
    ],
    ids=["zeros", "negative-zeros", "denormal", "start-above-1"],
)
def test_structural_zeros_fit_and_score(startprob, transmat, implementation):
    """
    A left-right model is valid: zeros (``-0.0`` too), a denormal entry and
    a start probability above 1 within the sum's tolerance are accepted;
    ``score`` is the log of the sum over every state path, and a fit keeps
    every zero at zero.
    """
    emissionprob = np.array(
        [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]]
    )
    X = np.array([[0], [2], [1]])
    # Every state is visited and left, so no row of the fit is empty.
    X_fit = np.array([[0], [0], [1], [1], [2], [2]])

    def brute_force_log_likelihood(h):
        total = 0.0
        for path in itertools.product(range(3), repeat=len(X)):
            p = h.startprob_[path[0]] * h.emissionprob_[path[0], X[0, 0]]
            for t in range(1, len(X)):
                p *= (
                    h.transmat_[path[t - 1], path[t]]
                    * h.emissionprob_[path[t], X[t, 0]]
                )
            total += p
        return np.log(total)

    h = hmm.CategoricalHMM(
        3, init_params="", n_iter=5, implementation=implementation
    )
    h.startprob_ = startprob
    h.transmat_ = transmat
    h.emissionprob_ = emissionprob
    assert_allclose(h.score(X), brute_force_log_likelihood(h), rtol=1e-12)

    h.fit(X_fit)
    assert np.all(h.startprob_[startprob == 0] == 0)
    assert np.all(h.transmat_[transmat == 0] == 0)
    assert_allclose(h.score(X), brute_force_log_likelihood(h), rtol=1e-12)


def test_stationary_distribution():
    n_components = 10
    h = StubHMM(n_components)
    transmat = np.random.random((n_components, n_components))
    transmat /= np.tile(transmat.sum(axis=1)[:, np.newaxis], (1, n_components))
    h.transmat_ = transmat
    stationary = h.get_stationary_distribution()
    assert stationary.dtype == float
    assert h.get_stationary_distribution().T @ h.transmat_ == pytest.approx(
        stationary
    )


class EqualEmissionsHMM(BaseHMM):
    """An HMM whose states emit every observation with the same probability."""

    def _compute_log_likelihood(self, X):
        return np.zeros((len(X), self.n_components))


@pytest.mark.parametrize("lengths", [None, [3, 2]])
@pytest.mark.parametrize("implementation", ["scaling", "log"])
def test_map_log_prob_is_the_log_of_the_chosen_posteriors(
    implementation, lengths
):
    h = EqualEmissionsHMM(2, implementation=implementation)
    h.startprob_ = np.full(2, 0.5)
    h.transmat_ = np.full((2, 2), 0.5)

    log_prob, _ = h.decode(np.zeros((5, 1)), lengths, algorithm="map")

    # Every posterior is one half.
    assert log_prob <= 0
    assert_allclose(log_prob, 5 * np.log(0.5), rtol=1e-12)


@pytest.mark.parametrize("implementation", ["scaling", "log"])
def test_map_log_prob_sums_the_log_of_the_largest_posteriors(implementation):
    rs = np.random.RandomState(1)
    lengths = [6, 4]
    X = rs.normal(size=(10, 2)) + 2 * rs.randint(0, 2, (10, 1))
    h = hmm.GaussianHMM(
        2, "diag", init_params="", implementation=implementation
    )
    h.startprob_ = np.array([0.6, 0.4])
    h.transmat_ = np.array([[0.7, 0.3], [0.2, 0.8]])
    h.means_ = np.array([[0.0, 0.0], [2.0, 2.0]])
    h.covars_ = np.ones((2, 2))

    log_prob, states = h.decode(X, lengths, algorithm="map")

    posteriors = np.split(h.predict_proba(X, lengths), [6])
    expected = sum(np.log(p.max(axis=1)).sum() for p in posteriors)
    assert_allclose(log_prob, expected, rtol=1e-12)
    assert log_prob < 0
    assert_allclose(states, h.predict_proba(X, lengths).argmax(axis=1))


def _occupancy_cases():
    """(id, factory, X, statistic of the class holding the state mass)."""
    rs = np.random.RandomState(7)
    continuous = rs.normal(size=(40, 2)) + 3 * rs.randint(0, 2, (40, 1))
    symbols = rs.randint(0, 3, (40, 1))
    counts = rs.poisson(4, (40, 3))
    multinomial = rs.multinomial(5, [0.2, 0.3, 0.5], size=40)
    cases = [
        (
            f"GaussianHMM-{cov}",
            lambda cls=hmm.GaussianHMM, cov=cov: cls(
                2, cov, n_iter=4, random_state=0
            ),
            continuous,
            "post",
        )
        for cov in ("spherical", "diag", "full", "tied")
    ]
    cases += [
        (
            "GMMHMM",
            lambda: hmm.GMMHMM(2, 2, n_iter=4, random_state=0),
            continuous,
            "post_sum",
        ),
        (
            "PoissonHMM",
            lambda: hmm.PoissonHMM(2, n_iter=4, random_state=0),
            counts,
            "post",
        ),
        (
            "VariationalGaussianHMM",
            lambda: vhmm.VariationalGaussianHMM(2, n_iter=4, random_state=0),
            continuous,
            "post",
        ),
        (
            "CategoricalHMM",
            lambda: hmm.CategoricalHMM(2, n_iter=4, random_state=0),
            symbols,
            None,
        ),
        (
            "MultinomialHMM",
            lambda: hmm.MultinomialHMM(2, 5, n_iter=4, random_state=0),
            multinomial,
            None,
        ),
        (
            "VariationalCategoricalHMM",
            lambda: vhmm.VariationalCategoricalHMM(
                2, n_iter=4, random_state=0
            ),
            symbols,
            None,
        ),
    ]
    return cases


OCCUPANCY_CASES = _occupancy_cases()
OCCUPANCY_IDS = [case[0] for case in OCCUPANCY_CASES]
LENGTHS = [25, 15]


def _recording(model, statistic):
    """
    Make ``model`` keep what its last M-step was given: the class's own state
    mass (``statistic``) and a sum of the posteriors of the E-step.
    """
    cls = type(model)

    class Recorder(cls):
        def _accumulate_sufficient_statistics(
            self, stats, X, lattice, posteriors, fwdlattice, bwdlattice
        ):
            if getattr(self, "recorded_sum", None) is None:
                self.recorded_sum = np.zeros(self.n_components)
            self.recorded_sum += posteriors.sum(axis=0)
            super()._accumulate_sufficient_statistics(
                stats, X, lattice, posteriors, fwdlattice, bwdlattice
            )

        def _do_mstep(self, stats):
            self.recorded_stat = (
                None if statistic is None else (stats[statistic].copy())
            )
            self.recorded_posterior_sum, self.recorded_sum = (
                self.recorded_sum,
                None,
            )
            super()._do_mstep(stats)

    model.__class__ = Recorder
    return model


def _fitted_arrays(model):
    return {
        k: np.asarray(v)
        for k, v in vars(model).items()
        if k.endswith("_") and k not in ("monitor_", "state_occupancy_")
    }


@pytest.mark.parametrize(
    "case",
    [c for c in OCCUPANCY_CASES if c[3]],
    ids=[c[0] for c in OCCUPANCY_CASES if c[3]],
)
def test_state_occupancy_equals_the_last_m_step_statistic(case):
    _, factory, X, statistic = case
    model = _recording(factory(), statistic).fit(X, LENGTHS)
    assert model.state_occupancy_.tobytes() == (model.recorded_stat.tobytes())


@pytest.mark.parametrize("case", OCCUPANCY_CASES, ids=OCCUPANCY_IDS)
def test_state_occupancy_is_the_posterior_sum_for_every_model(case):
    _, factory, X, statistic = case
    model = _recording(factory(), statistic).fit(X, LENGTHS)
    assert model.state_occupancy_.shape == (2,)
    assert model.state_occupancy_.tobytes() == (
        model.recorded_posterior_sum.tobytes()
    )
    assert_allclose(model.state_occupancy_.sum(), len(X), rtol=1e-9)


@pytest.mark.parametrize("case", OCCUPANCY_CASES, ids=OCCUPANCY_IDS)
def test_state_occupancy_leaves_the_fit_unchanged(case):
    _, factory, X, statistic = case
    plain = factory().fit(X, LENGTHS)
    recorded = _recording(factory(), statistic).fit(X, LENGTHS)
    for name, value in _fitted_arrays(plain).items():
        assert value.tobytes() == _fitted_arrays(recorded)[name].tobytes()


def test_state_occupancy_absent_after_a_fit_without_iterations():
    _, factory, X, _ = OCCUPANCY_CASES[1]
    model = factory().fit(X, LENGTHS)
    assert hasattr(model, "state_occupancy_")
    model.n_iter = 0
    model.fit(X, LENGTHS)
    assert not hasattr(model, "state_occupancy_")


def _model_cases():
    """(id, factory taking n_components, X) for every model class."""
    rs = np.random.RandomState(10)
    continuous = rs.normal(size=(20, 2)) + 3 * rs.randint(0, 2, (20, 1))
    symbols = rs.randint(0, 3, (20, 1))
    counts = rs.poisson(4, (20, 3))
    multinomial = rs.multinomial(5, [0.2, 0.3, 0.5], size=20)
    return [
        (
            "GaussianHMM",
            lambda n: hmm.GaussianHMM(n, random_state=0),
            continuous,
        ),
        ("GMMHMM", lambda n: hmm.GMMHMM(n, random_state=0), continuous),
        (
            "CategoricalHMM",
            lambda n: hmm.CategoricalHMM(n, random_state=0),
            symbols,
        ),
        (
            "MultinomialHMM",
            lambda n: hmm.MultinomialHMM(n, 5, random_state=0),
            multinomial,
        ),
        ("PoissonHMM", lambda n: hmm.PoissonHMM(n, random_state=0), counts),
        (
            "VariationalCategoricalHMM",
            lambda n: vhmm.VariationalCategoricalHMM(n, random_state=0),
            symbols,
        ),
        (
            "VariationalGaussianHMM",
            lambda n: vhmm.VariationalGaussianHMM(n, random_state=0),
            continuous,
        ),
    ]


MODEL_CASES = _model_cases()
MODEL_IDS = [case[0] for case in MODEL_CASES]


@pytest.mark.parametrize("n_components", [0, -1, 0.5])
@pytest.mark.parametrize("case", MODEL_CASES, ids=MODEL_IDS)
def test_fit_refuses_n_components_below_one(case, n_components):
    """
    ``fit`` refuses an ``n_components`` less than 1 before it starts, so
    the model keeps what it held.
    """
    _, factory, X = case
    model = factory(n_components)
    before = dict(vars(model))
    with pytest.raises(ValueError, match="n_components must be at least 1"):
        model.fit(X)
    assert vars(model).keys() == before.keys()
    assert all(vars(model)[k] is v for k, v in before.items())


@pytest.mark.parametrize("n_components", [1.5, 2.0, True])
@pytest.mark.parametrize("case", MODEL_CASES, ids=MODEL_IDS)
def test_fit_n_components_not_integer_keeps_type_error(case, n_components):
    """A float or boolean number of states fails where it did before."""
    _, factory, X = case
    with pytest.raises(TypeError):
        factory(n_components).fit(X)


@pytest.mark.parametrize("case", MODEL_CASES, ids=MODEL_IDS)
def test_one_state_model_fits(case):
    """A model with one state fits and scores a finite log-likelihood."""
    _, factory, X = case
    assert np.isfinite(factory(1).fit(X).score(X))


@pytest.mark.parametrize("name", ["CategoricalHMM", "PoissonHMM"])
def test_n_components_as_zero_dim_array_fits(name):
    """
    An ``n_components`` given as a 0-d integer array fits as the integer
    does.
    """
    _, factory, X = MODEL_CASES[MODEL_IDS.index(name)]
    expected = factory(2).fit(X)
    model = factory(np.array(2)).fit(X)
    assert model.monitor_.history == expected.monitor_.history
    assert model.transmat_.tobytes() == expected.transmat_.tobytes()


@pytest.mark.parametrize(
    "call",
    [
        lambda h, X: h.fit(X),
        lambda h, X: h.score(X),
        lambda h, X: h.score_samples(X),
        lambda h, X: h.predict_proba(X),
        lambda h, X: h.decode(X, algorithm="map"),
    ],
    ids=["fit", "score", "score_samples", "predict_proba", "decode-map"],
)
def test_unknown_implementation_raises(call):
    """Every path that reads ``implementation`` names it when it is wrong."""
    h = new_hmm("foo")
    with pytest.raises(ValueError, match="implementation"):
        call(h, np.array([[0], [1]]))


def test_unknown_implementation_raises_in_variational_fit():
    """The variational models read ``implementation`` through one place."""
    model = vhmm.VariationalGaussianHMM(
        2, implementation="foo", random_state=0
    )
    X = np.random.RandomState(10).normal(size=(20, 2))
    with pytest.raises(ValueError, match="implementation"):
        model.fit(X)


def _hand_built(name):
    """
    A two-state model with every parameter set and no ``n_features``
    (``None`` for a ``VariationalCategoricalHMM``, whose constructor sets
    it).
    """
    rs = np.random.RandomState(10)
    if name == "VariationalCategoricalHMM":
        h = vhmm.VariationalCategoricalHMM(2, init_params="")
        h.emissionprob_prior_ = np.full((2, 3), 1 / 3)
        h.emissionprob_posterior_ = np.array(
            [[2.0, 3.0, 5.0], [5.0, 3.0, 2.0]]
        )
        h.emissionprob_ = np.array([[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]])
        X = rs.randint(0, 3, (20, 1))
    elif name == "VariationalGaussianHMM":
        h = vhmm.VariationalGaussianHMM(2, "diag", init_params="")
        h.means_prior_ = np.zeros((2, 2))
        h.means_posterior_ = np.array([[0.0, 0.0], [3.0, 3.0]])
        h.beta_prior_ = np.ones(2)
        h.beta_posterior_ = np.array([10.0, 10.0])
        h.dof_prior_ = np.full(2, 2.0)
        h.dof_posterior_ = np.array([12.0, 12.0])
        h.scale_prior_ = np.full((2, 2), 1e-3)
        h.scale_posterior_ = np.full((2, 2), 12.0)
        h.covars_ = np.ones((2, 2))
        X = rs.normal(size=(20, 2))
    if name.startswith("Variational"):
        h.startprob_prior_ = np.full(2, 0.5)
        h.startprob_posterior_ = np.array([6.0, 4.0])
        h.transmat_prior_ = np.full((2, 2), 0.5)
        h.transmat_posterior_ = np.array([[7.0, 3.0], [4.0, 6.0]])
    elif name == "GaussianHMM":
        h = hmm.GaussianHMM(2, init_params="")
        h.means_ = np.array([[0.0, 0.0], [3.0, 3.0]])
        h.covars_ = np.ones((2, 2))
        X = rs.normal(size=(20, 2))
    elif name == "GMMHMM":
        h = hmm.GMMHMM(2, n_mix=2, covariance_type="diag", init_params="")
        h.weights_ = np.array([[0.5, 0.5], [0.3, 0.7]])
        h.means_ = np.array(
            [[[0.0, 0.0], [1.0, 1.0]], [[3.0, 3.0], [4.0, 4.0]]]
        )
        h.covars_ = np.ones((2, 2, 2))
        X = rs.normal(size=(20, 2))
    elif name == "CategoricalHMM":
        h = hmm.CategoricalHMM(2, init_params="")
        h.emissionprob_ = np.array([[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]])
        X = rs.randint(0, 3, (20, 1))
    elif name == "MultinomialHMM":
        h = hmm.MultinomialHMM(2, 5, init_params="")
        h.emissionprob_ = np.array([[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]])
        X = rs.multinomial(5, [0.2, 0.3, 0.5], size=20)
    else:
        h = hmm.PoissonHMM(2, init_params="")
        h.lambdas_ = np.array([[1.0, 2.0], [4.0, 3.0]])
        X = rs.poisson(3, (20, 2))
    h.startprob_ = np.array([0.6, 0.4])
    h.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]])
    return h, X


@pytest.mark.parametrize("criterion", ["aic", "bic"])
@pytest.mark.parametrize(
    "name", ["GaussianHMM", "CategoricalHMM", "PoissonHMM"]
)
def test_aic_bic_before_n_features(name, criterion):
    """
    ``aic`` and ``bic`` of a model built by hand take the number of
    features from its emission parameter: the value equals the one after a
    ``score``, and the ``score`` leaves ``n_features`` unset.
    """
    h, X = _hand_built(name)
    before = getattr(h, criterion)(X)
    h.score(X)
    assert getattr(h, "n_features", None) is None
    assert before == getattr(h, criterion)(X)


@pytest.mark.parametrize("name", ["MultinomialHMM", "PoissonHMM", "GMMHMM"])
def test_n_features_none_is_taken_from_emissions(name):
    """
    An ``n_features`` set to ``None`` is read from the emission parameter,
    as when it is not set: the same score, and ``n_features`` stays
    ``None``.
    """
    h, X = _hand_built(name)
    expected = h.score(X)
    h, X = _hand_built(name)
    h.n_features = None
    assert h.score(X) == expected
    assert h.n_features is None


@pytest.mark.parametrize(
    "name",
    [
        "GaussianHMM",
        "GMMHMM",
        "MultinomialHMM",
        "PoissonHMM",
        "VariationalGaussianHMM",
    ],
)
def test_fit_reads_n_features_none_as_unset(name):
    """
    ``fit`` reads an ``n_features`` set to ``None`` as not set: the model
    fits, and the fit and the number of features it stores equal, byte for
    byte, those of the same model without the attribute.
    """

    def build():
        if name == "VariationalGaussianHMM":
            _, factory, X = MODEL_CASES[MODEL_IDS.index(name)]
            return factory(2), X
        h, X = _hand_built(name)
        if name == "GMMHMM":
            # A mixture component of this start loses its last weight by
            # the tenth iteration, which divides by zero.
            h.n_iter = 5
        return h, X

    expected, X = build()
    expected.fit(X)
    h, X = build()
    h.n_features = None
    h.fit(X)
    assert _fit_state(h) == _fit_state(expected)
    assert h.n_features == expected.n_features
    assert type(h.n_features) is type(expected.n_features)


def _refused_X(name, X):
    """``X`` changed so that the model fitted on ``X`` refuses it."""
    if name in ("CategoricalHMM", "VariationalCategoricalHMM"):
        return X + 3
    if name in ("MultinomialHMM", "PoissonHMM"):
        return -1 - X
    return X[:, :1]


def _refused_then_fitted(name):
    """
    ``(model, wrong, right)``: a model with ``init_params="st"`` whose
    emission parameters have three features, 50 rows of ``X`` with four
    columns, which it refuses, and 50 rows with three, which it fits.
    """
    rs = np.random.RandomState(0)
    if name == "MultinomialHMM":
        h = hmm.MultinomialHMM(
            n_components=2, n_trials=5, init_params="st", random_state=0
        )
        h.emissionprob_ = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
        return (
            h,
            rs.multinomial(5, [0.25] * 4, size=50),
            rs.multinomial(5, [0.2, 0.3, 0.5], size=50),
        )
    if name == "PoissonHMM":
        h = hmm.PoissonHMM(n_components=2, init_params="st", random_state=0)
        h.lambdas_ = np.array([[3.1, 1.4, 4.5], [1.6, 5.3, 0.1]])
        return h, rs.poisson(3, size=(50, 4)), rs.poisson(3, size=(50, 3))
    if name == "GMMHMM":
        h = hmm.GMMHMM(
            n_components=2, n_mix=2, init_params="st", random_state=0
        )
        h.weights_ = np.array([[0.5, 0.5], [0.5, 0.5]])
        h.means_ = rs.normal(size=(2, 2, 3))
        h.covars_ = np.ones((2, 2, 3))
    else:
        h = hmm.GaussianHMM(n_components=2, init_params="st", random_state=0)
        h.means_ = rs.normal(size=(2, 3))
        h.covars_ = np.ones((2, 3))
    return h, rs.normal(size=(50, 4)), rs.normal(size=(50, 3))


GMM_PRIORS = [
    "weights_prior",
    "means_prior",
    "means_weight",
    "covars_prior",
    "covars_weight",
]


@pytest.mark.parametrize("case", MODEL_CASES, ids=MODEL_IDS)
def test_X_check_writes_nothing(case):
    """
    The check of ``X`` against a fitted model reads it and writes nothing,
    whether it passes ``X`` or refuses it: every attribute keeps its object
    and every array its bytes.
    """
    name, factory, X = case
    model = factory(2).fit(X)
    before = dict(vars(model))
    arrays = {
        k: v.tobytes() for k, v in before.items() if isinstance(v, np.ndarray)
    }
    model._check_X(X)
    with pytest.raises(ValueError, match="X must"):
        model._check_X(_refused_X(name, X))
    assert vars(model).keys() == before.keys()
    assert all(vars(model)[k] is v for k, v in before.items())
    assert all(vars(model)[k].tobytes() == b for k, b in arrays.items())


@pytest.mark.parametrize("before", ["absent", None], ids=["absent", "None"])
@pytest.mark.parametrize(
    "name", ["MultinomialHMM", "PoissonHMM", "GMMHMM", "GaussianHMM"]
)
def test_fit_after_a_refused_fit_equals_a_fresh_fit(name, before):
    """
    A ``fit`` refused on ``X`` of another width leaves ``n_features`` as it
    was before the call, absent or ``None`` (a ``GaussianHMM`` keeps the
    width of its ``means_``), and the priors of a ``GMMHMM`` as they were,
    a covariance prior filled in from ``None`` as its number. A ``fit`` on
    ``X`` of the model's width then equals a new model's fit, byte for
    byte.
    """
    model, wrong, right = _refused_then_fitted(name)
    if before is None:
        model.n_features = None
    priors = {p: getattr(model, p) for p in GMM_PRIORS if name == "GMMHMM"}
    with pytest.raises(ValueError):
        model.fit(wrong)
    if name == "GaussianHMM":
        assert model.n_features == 3
    elif before is None:
        assert model.n_features is None
    else:
        assert "n_features" not in vars(model)
    if name == "GMMHMM":
        for p in ["weights_prior", "means_prior", "means_weight"]:
            assert getattr(model, p) is priors[p]
        # the defaults of a "diag" model
        assert type(model.covars_prior) is float
        assert model.covars_prior == -1.5
        assert type(model.covars_weight) is float
        assert model.covars_weight == 0.0
    model.fit(right)
    fresh, _, fresh_right = _refused_then_fitted(name)
    fresh.fit(fresh_right)
    assert _fit_state(model) == _fit_state(fresh)
    assert model.n_features == fresh.n_features
    for p in priors:
        assert _as_bytes(getattr(model, p)) == _as_bytes(getattr(fresh, p))


EM_NAMES = [
    "GaussianHMM",
    "GMMHMM",
    "CategoricalHMM",
    "MultinomialHMM",
    "PoissonHMM",
]
# The parameters each model converts to an array when they are assigned.
CONVERTED = {
    "GaussianHMM": ["startprob_", "transmat_", "means_"],
    "GMMHMM": ["startprob_", "transmat_", "weights_", "means_", "covars_"],
    "CategoricalHMM": ["startprob_", "transmat_", "emissionprob_"],
    "MultinomialHMM": ["startprob_", "transmat_", "emissionprob_"],
    "PoissonHMM": ["startprob_", "transmat_", "lambdas_"],
}
VARIATIONAL_NAMES = ["VariationalCategoricalHMM", "VariationalGaussianHMM"]
_VARIATIONAL_TRANSITIONS = [
    "startprob_prior_",
    "startprob_posterior_",
    "transmat_prior_",
    "transmat_posterior_",
    "startprob_",
    "transmat_",
]
CONVERTED.update(
    {
        "VariationalCategoricalHMM": [
            *_VARIATIONAL_TRANSITIONS,
            "emissionprob_prior_",
            "emissionprob_posterior_",
            "emissionprob_",
        ],
        "VariationalGaussianHMM": [
            *_VARIATIONAL_TRANSITIONS,
            "means_prior_",
            "means_posterior_",
            "beta_prior_",
            "beta_posterior_",
            "dof_prior_",
            "dof_posterior_",
            "scale_prior_",
            "scale_posterior_",
        ],
    }
)
CONVERTED_CASES = [
    (name, attr)
    for name in EM_NAMES + VARIATIONAL_NAMES
    for attr in CONVERTED[name]
]
CONVERTED_IDS = [f"{name}-{attr}" for name, attr in CONVERTED_CASES]
PREDICTIONS = {
    "score": lambda h, X: h.score(X),
    "score_samples": lambda h, X: h.score_samples(X),
    "decode-viterbi": lambda h, X: h.decode(X, algorithm="viterbi"),
    "decode-map": lambda h, X: h.decode(X, algorithm="map"),
    "predict": lambda h, X: h.predict(X),
    "predict_proba": lambda h, X: h.predict_proba(X),
    "sample": lambda h, X: h.sample(5, random_state=0),
    "aic": lambda h, X: h.aic(X),
    "bic": lambda h, X: h.bic(X),
}


def _predictions(name):
    """The predictions of the model *name*: no ``aic`` or ``bic`` for a
    variational model."""
    if name in VARIATIONAL_NAMES:
        return {
            k: v for k, v in PREDICTIONS.items() if k not in ("aic", "bic")
        }
    return PREDICTIONS


PREDICTION_CASES = [
    (name, key)
    for name in EM_NAMES + VARIATIONAL_NAMES
    for key in _predictions(name)
]


def _built(name, build):
    """
    A two-state model built by hand from arrays or from lists, or fitted,
    and an ``X`` it takes.
    """
    if build == "fitted":
        _, factory, X = MODEL_CASES[MODEL_IDS.index(name)]
        return factory(2).fit(X), X
    h, X = _hand_built(name)
    if build == "lists":
        for attr in CONVERTED[name]:
            setattr(h, attr, getattr(h, attr).tolist())
    return h, X


def _record_assignments(model):
    """
    Give *model* a subclass of its class that records the name of every
    attribute assigned or deleted; return the list it records into.
    """
    names = []
    cls = type(model)

    def __setattr__(self, name, value):
        names.append(name)
        super(recording, self).__setattr__(name, value)

    def __delattr__(self, name):
        names.append(name)
        super(recording, self).__delattr__(name)

    recording = type(
        cls.__name__,
        (cls,),
        {"__setattr__": __setattr__, "__delattr__": __delattr__},
    )
    model.__class__ = recording
    return names


def _snapshot(model):
    """The model's attributes, and the bytes of those that are arrays."""
    state = dict(vars(model))
    return state, {
        k: v.tobytes() for k, v in state.items() if isinstance(v, np.ndarray)
    }


def _assert_unchanged(model, snapshot):
    """Same names in the same order, same objects, same bytes."""
    state, arrays = snapshot
    assert list(vars(model)) == list(state)
    assert all(vars(model)[k] is v for k, v in state.items())
    assert all(vars(model)[k].tobytes() == b for k, b in arrays.items())


def _as_bytes(result):
    """A prediction's result as bytes, with its dtype and shape."""
    if isinstance(result, tuple):
        return tuple(_as_bytes(r) for r in result)
    result = np.asarray(result)
    return result.dtype.str, result.shape, result.tobytes()


@pytest.mark.parametrize("build", ["arrays", "lists", "fitted"])
@pytest.mark.parametrize(
    "name, key",
    PREDICTION_CASES,
    ids=[f"{n}-{k}" for n, k in PREDICTION_CASES],
)
def test_prediction_assigns_nothing(name, key, build):
    """
    A prediction reads the model: it assigns and deletes no attribute,
    and every attribute, ``n_features`` included, keeps its object, its
    place and its bytes, on a model built by hand from arrays or from
    lists and on a fitted one.
    """
    call = PREDICTIONS[key]
    h, X = _built(name, build)
    snapshot = _snapshot(h)
    assigned = _record_assignments(h)
    call(h, X)
    assert assigned == []
    _assert_unchanged(h, snapshot)


@pytest.mark.parametrize("call", PREDICTIONS.values(), ids=PREDICTIONS.keys())
@pytest.mark.parametrize("implementation", ["log", "scaling"])
def test_prediction_on_the_fitted_data_leaves_a_mixture_model_unchanged(
    implementation, call
):
    """
    A prediction on the very array a mixture model was fitted on assigns
    nothing and leaves every attribute as it was.
    """
    _, _, X = MODEL_CASES[MODEL_IDS.index("GMMHMM")]
    h = hmm.GMMHMM(
        2, n_mix=2, implementation=implementation, random_state=0
    ).fit(X)
    snapshot = _snapshot(h)
    assigned = _record_assignments(h)
    call(h, X)
    assert assigned == []
    _assert_unchanged(h, snapshot)


REFUSED_PARAMETERS = [
    (
        "GaussianHMM",
        "startprob_",
        [1.2, -0.2],
        "startprob_ must not have negative entries",
    ),
    (
        "GaussianHMM",
        "transmat_",
        [[0.7, 0.3], [1.2, -0.2]],
        "transmat_ must not have negative entries",
    ),
    (
        "GaussianHMM",
        "means_",
        [[0.0, np.nan], [3.0, 3.0]],
        "means_ must contain only finite values",
    ),
    (
        "GMMHMM",
        "weights_",
        [[0.5, 0.6], [0.3, 0.7]],
        "weights_ rows must sum to 1",
    ),
    (
        "GMMHMM",
        "means_",
        [[0.0, 0.0], [3.0, 3.0]],
        r"means_ must have shape \(n_components, n_mix, n_features\)",
    ),
    (
        "CategoricalHMM",
        "emissionprob_",
        [[1.2, -0.1, -0.1], [0.5, 0.3, 0.2]],
        "emissionprob_ must not have negative entries",
    ),
    (
        "CategoricalHMM",
        "emissionprob_",
        [[np.nan, 0.5, 0.5], [0.5, 0.3, 0.2]],
        "emissionprob_ rows must sum to 1",
    ),
    (
        "MultinomialHMM",
        "emissionprob_",
        [[1.2, -0.1, -0.1], [0.5, 0.3, 0.2]],
        "emissionprob_ must not have negative entries",
    ),
    (
        "MultinomialHMM",
        "emissionprob_",
        [[np.nan, 0.5, 0.5], [0.5, 0.3, 0.2]],
        "emissionprob_ rows must sum to 1",
    ),
    (
        "MultinomialHMM",
        "emissionprob_",
        [[0.3, 0.3, 0.3], [0.5, 0.3, 0.2]],
        "emissionprob_ rows must sum to 1",
    ),
    (
        "PoissonHMM",
        "lambdas_",
        [[1.0, -2.0], [4.0, 3.0]],
        "lambdas_ must not have negative entries",
    ),
    (
        "PoissonHMM",
        "lambdas_",
        [[1.0, np.nan], [4.0, 3.0]],
        "lambdas_ must contain only finite values",
    ),
    (
        "PoissonHMM",
        "lambdas_",
        [[1.0, np.inf], [4.0, 3.0]],
        "lambdas_ must contain only finite values",
    ),
    (
        "VariationalCategoricalHMM",
        "transmat_",
        [[1.2, -0.2], [0.4, 0.6]],
        "transmat_ must not have negative entries",
    ),
    (
        "VariationalCategoricalHMM",
        "startprob_",
        [np.inf, 0.4],
        "startprob_ must contain only finite values",
    ),
    (
        "VariationalCategoricalHMM",
        "emissionprob_",
        [[np.nan, 0.5, 0.5], [0.5, 0.3, 0.2]],
        "emissionprob_ must contain only finite values",
    ),
    (
        "VariationalCategoricalHMM",
        "startprob_prior_",
        0.5,
        "startprob_prior_ must have length n_components",
    ),
    (
        "VariationalCategoricalHMM",
        "startprob_posterior_",
        [6.0],
        "startprob_posterior_ must have length n_components",
    ),
    (
        "VariationalCategoricalHMM",
        "transmat_prior_",
        [1.0, 2.0],
        r"transmat_prior_ must have shape \(n_components, n_components\)",
    ),
    (
        "VariationalCategoricalHMM",
        "emissionprob_",
        [[1.2, -0.1, -0.1], [0.5, 0.3, 0.2]],
        "emissionprob_ must not have negative entries",
    ),
    (
        "VariationalGaussianHMM",
        "startprob_",
        [1.2, -0.2],
        "startprob_ must not have negative entries",
    ),
    (
        "VariationalGaussianHMM",
        "transmat_",
        [[0.7, 0.3], [np.nan, 0.6]],
        "transmat_ must contain only finite values",
    ),
    (
        "VariationalGaussianHMM",
        "means_posterior_",
        [[0.0, np.nan], [3.0, 3.0]],
        "means_posterior_ must not contain NaN",
    ),
    (
        "VariationalGaussianHMM",
        "covars_",
        [[1.0, np.nan], [1.0, 1.0]],
        "covars_ must not contain NaN",
    ),
]
REFUSED_IDS = [
    "GaussianHMM-negative-startprob_",
    "GaussianHMM-negative-transmat_",
    "GaussianHMM-nan-means_",
    "GMMHMM-unnormalized-weights_",
    "GMMHMM-2-D-means_",
    "CategoricalHMM-negative-emissionprob_",
    "CategoricalHMM-nan-emissionprob_",
    "MultinomialHMM-negative-emissionprob_",
    "MultinomialHMM-nan-emissionprob_",
    "MultinomialHMM-unnormalized-emissionprob_",
    "PoissonHMM-negative-lambdas_",
    "PoissonHMM-nan-lambdas_",
    "PoissonHMM-inf-lambdas_",
    "VariationalCategoricalHMM-negative-transmat_",
    "VariationalCategoricalHMM-inf-startprob_",
    "VariationalCategoricalHMM-nan-emissionprob_",
    "VariationalCategoricalHMM-scalar-startprob_prior_",
    "VariationalCategoricalHMM-short-startprob_posterior_",
    "VariationalCategoricalHMM-row-transmat_prior_",
    "VariationalCategoricalHMM-negative-emissionprob_",
    "VariationalGaussianHMM-negative-startprob_",
    "VariationalGaussianHMM-nan-transmat_",
    "VariationalGaussianHMM-nan-means_posterior_",
    "VariationalGaussianHMM-nan-covars_",
]


@pytest.mark.parametrize(
    "name, attr, value, match", REFUSED_PARAMETERS, ids=REFUSED_IDS
)
@pytest.mark.parametrize("method", ["score", "decode", "sample"])
def test_refused_prediction_assigns_nothing(name, attr, value, match, method):
    """
    A prediction refuses an invalid parameter, naming it, and raises before
    it has assigned anything.
    """
    h, X = _hand_built(name)
    setattr(h, attr, np.array(value))
    snapshot = _snapshot(h)
    assigned = _record_assignments(h)
    with pytest.raises(ValueError, match=match):
        if method == "sample":
            h.sample(5, random_state=0)
        else:
            getattr(h, method)(X)
    assert assigned == []
    _assert_unchanged(h, snapshot)


FIT_ONLY_VALUES = [
    ("VariationalCategoricalHMM", "startprob_prior_", [np.nan, 0.5]),
    (
        "VariationalCategoricalHMM",
        "transmat_posterior_",
        [[7.0, -3.0], [4.0, 6.0]],
    ),
    (
        "VariationalCategoricalHMM",
        "emissionprob_posterior_",
        [[np.inf, 3.0, 5.0], [5.0, 3.0, 2.0]],
    ),
    ("VariationalGaussianHMM", "dof_prior_", [-0.3, 2.0]),
    ("VariationalGaussianHMM", "beta_posterior_", [np.nan, 10.0]),
    ("VariationalGaussianHMM", "beta_posterior_", [np.inf, 10.0]),
    ("VariationalGaussianHMM", "scale_prior_", [[np.nan, 1e-3], [1e-3, 1e-3]]),
    ("VariationalGaussianHMM", "startprob_prior_", 0.5),
    ("VariationalGaussianHMM", "startprob_prior_", [0.5]),
    ("VariationalGaussianHMM", "startprob_posterior_", [6.0]),
    ("VariationalGaussianHMM", "transmat_prior_", [1.0, 2.0]),
    ("VariationalGaussianHMM", "transmat_prior_", [[1.0], [2.0]]),
    ("VariationalGaussianHMM", "transmat_posterior_", [1.0, 1.0]),
    ("VariationalCategoricalHMM", "startprob_prior_", [[0.5], [0.5]]),
    ("VariationalCategoricalHMM", "startprob_posterior_", [[6.0], [4.0]]),
]


def _shape_id(attr, value):
    """
    ``-shape-<dims>`` for a start or transition prior or posterior not of
    its two-state shape, ``-shape-scalar`` for a number; ``""`` otherwise.
    """
    documented = {"startprob": (2,), "transmat": (2, 2)}
    shape = np.shape(value)
    if documented.get(attr.split("_")[0], shape) == shape:
        return ""
    return "-shape-" + ("x".join(map(str, shape)) or "scalar")


FIT_ONLY_IDS = [
    f"{name}-{attr}"
    + ("-inf" if np.isinf(np.asarray(value, dtype=float)).any() else "")
    + _shape_id(attr, value)
    for name, attr, value in FIT_ONLY_VALUES
]


@pytest.mark.parametrize(
    "name, attr, value", FIT_ONLY_VALUES, ids=FIT_ONLY_IDS
)
def test_prediction_does_not_test_what_only_fit_reads(name, attr, value):
    """
    A prediction does not read the priors and posteriors of a variational
    model, so an invalid value there is not refused: every prediction has
    the bytes of the same model without it.
    """
    expected, X = _hand_built(name)
    h, _ = _hand_built(name)
    setattr(h, attr, value)
    for call in _predictions(name).values():
        assert _as_bytes(call(h, X)) == _as_bytes(call(expected, X))


@pytest.mark.parametrize("covariance_type", ["spherical", "diag"])
@pytest.mark.parametrize("attr", ["means_posterior_", "covars_"])
def test_infinite_gaussian_state_is_not_refused_on_prediction(
    attr, covariance_type
):
    """
    A ``spherical`` or ``diag`` ``VariationalGaussianHMM`` with an infinite
    mean or variance in one state scores and decodes: that state emits
    nothing, so the score is finite and every state is the other one.
    """
    h, X = _hand_built("VariationalGaussianHMM")
    if covariance_type == "spherical":
        h.covariance_type = "spherical"
        h.scale_prior_ = np.full(2, 1e-3)
        h.scale_posterior_ = np.full(2, 12.0)
        h.covars_ = np.ones(2)
    if attr == "means_posterior_":
        h.means_posterior_ = np.array([[0.0, np.inf], [3.0, 3.0]])
    else:
        covars = h._covars_.copy()
        covars[0] = np.inf
        h._covars_ = covars
    assert np.isfinite(h.score(X))
    assert (h.predict(X) == 1).all()
    assert np.isfinite(h.predict_proba(X)).all()


FIT_REFUSED = [
    (
        "VariationalCategoricalHMM",
        "transmat_prior_",
        [[np.nan, 0.5], [0.5, 0.5]],
        "transmat_prior_ must contain only finite",
    ),
    (
        "VariationalCategoricalHMM",
        "startprob_posterior_",
        [np.inf, 4.0],
        "startprob_posterior_ must contain only finite",
    ),
    (
        "VariationalCategoricalHMM",
        "emissionprob_posterior_",
        [[np.inf, 3.0, 5.0], [5.0, 3.0, 2.0]],
        "emissionprob_posterior_ must contain only finite",
    ),
    (
        "VariationalGaussianHMM",
        "scale_prior_",
        [[np.nan, 1e-3], [1e-3, 1e-3]],
        "scale_prior_ must contain only finite",
    ),
    (
        "VariationalGaussianHMM",
        "means_prior_",
        [[0.0, np.inf], [0.0, 0.0]],
        "means_prior_ must contain only finite",
    ),
    (
        "VariationalGaussianHMM",
        "means_posterior_",
        [[0.0, -np.inf], [3.0, 3.0]],
        "means_posterior_ must contain only finite",
    ),
    (
        "VariationalGaussianHMM",
        "dof_posterior_",
        [np.nan, 12.0],
        "dof_posterior_ must contain only finite",
    ),
    (
        "VariationalGaussianHMM",
        "beta_posterior_",
        [np.nan, 10.0],
        "beta_posterior_ must not contain NaN",
    ),
]


def _wrong_shape(name, attr, value):
    """
    The row of ``FIT_REFUSED_SHAPES`` for a start or transition prior or
    posterior of a two-state model given with another shape: the message
    of the shape test ``fit`` runs, or the one of the length or shape test
    a ``VariationalCategoricalHMM`` also runs on every prediction.
    """
    shape = np.shape(value)
    documented = (2,) if attr.startswith("startprob") else (2, 2)
    message = f"{attr} must have shape {documented}, actual shape: {shape}"
    if name == "VariationalCategoricalHMM":
        if documented == (2, 2):
            message = f"{attr} must have shape (n_components, n_components)"
        elif not shape or shape[0] != 2:
            message = f"{attr} must have length n_components"
    return name, attr, value, f"^{re.escape(message)}$"


FIT_REFUSED_SHAPES = [
    _wrong_shape(name, attr, value)
    for name in VARIATIONAL_NAMES
    for attr, value in [
        ("startprob_prior_", 0.5),
        ("startprob_prior_", [0.5]),
        ("startprob_prior_", [[0.5], [0.5]]),
        ("startprob_posterior_", 6.0),
        ("startprob_posterior_", [6.0]),
        ("startprob_posterior_", [[6.0], [4.0]]),
        ("transmat_prior_", [1.0, 2.0]),
        ("transmat_prior_", [[1.0], [2.0]]),
        ("transmat_prior_", [[1.0, 2.0]]),
        ("transmat_posterior_", [1.0, 1.0]),
    ]
]


@pytest.mark.parametrize(
    "name, attr, value, match",
    FIT_REFUSED + FIT_REFUSED_SHAPES,
    ids=[f"{n}-{a}" for n, a, _, _ in FIT_REFUSED]
    + [f"{n}-{a}{_shape_id(a, v)}" for n, a, v, _ in FIT_REFUSED_SHAPES],
)
def test_variational_fit_refuses_an_invalid_prior_or_posterior(
    name, attr, value, match
):
    """
    ``fit`` refuses a NaN or infinite prior or posterior of a variational
    model, or a start or transition one of another shape, naming it, before
    its first iteration. The model keeps the ``n_features`` the call set,
    as after any refused fit whose kept emission parameters give that
    number: two features, three symbols.
    """
    h, X = _hand_built(name)
    setattr(h, attr, value)
    with pytest.raises(ValueError, match=match):
        h.fit(X)
    expected = 2 if name == "VariationalGaussianHMM" else 3
    assert h.n_features == expected
    assert h._n_features_from_emissions() == expected
    assert list(h.monitor_.history) == []


@pytest.mark.parametrize(
    "attr, form",
    [("startprob_prior_", 1.0), ("transmat_prior_", [1.0])],
    ids=["startprob_prior_-scalar", "transmat_prior_-shape-1"],
)
def test_variational_gaussian_fit_accepts_the_one_state_forms(attr, form):
    """
    With one state, a ``VariationalGaussianHMM`` fits a
    ``startprob_prior_`` given as a number and a ``transmat_prior_`` of
    shape ``(1,)`` as it fits the documented shapes: every fitted array,
    the history and the score have the same bytes.
    """
    _, X = _hand_built("VariationalGaussianHMM")
    fitted = vhmm.VariationalGaussianHMM(1, n_iter=3, random_state=0).fit(X)
    fitted.init_params = ""
    documented = copy.deepcopy(fitted)
    given = copy.deepcopy(fitted)
    # The default prior, 1 / n_components, at the documented shape.
    assert getattr(documented, attr).tolist() == (
        [1.0] if attr == "startprob_prior_" else [[1.0]]
    )
    setattr(given, attr, form)
    documented.fit(X)
    given.fit(X)
    assert np.shape(getattr(given, attr)) == np.shape(form)
    expected = _fit_state(documented)
    del expected[attr]
    state = _fit_state(given)
    del state[attr]
    assert state == expected
    assert _as_bytes(given.score(X)) == _as_bytes(documented.score(X))


@pytest.mark.parametrize("name", VARIATIONAL_NAMES)
def test_variational_fit_expands_a_scalar_constructor_prior(name):
    """
    The constructor's ``startprob_prior`` and ``transmat_prior`` given as
    numbers are expanded by ``fit``'s start to the documented shapes: the
    fit has the bytes of the one given the same priors as full arrays.
    """
    _, X = _hand_built(name)
    cls = getattr(vhmm, name)
    numbers = cls(
        2, startprob_prior=2.0, transmat_prior=3.0, n_iter=3, random_state=0
    ).fit(X)
    arrays = cls(
        2,
        startprob_prior=np.full(2, 2.0),
        transmat_prior=np.full((2, 2), 3.0),
        n_iter=3,
        random_state=0,
    ).fit(X)
    assert numbers.startprob_prior_.shape == (2,)
    assert numbers.transmat_prior_.shape == (2, 2)
    assert _fit_state(numbers) == _fit_state(arrays)
    assert _as_bytes(numbers.score(X)) == _as_bytes(arrays.score(X))


def _variational_gaussian(covariance_type):
    """
    The hand-built ``VariationalGaussianHMM`` with the given covariance
    type: the ``diag`` model as built, or, for another type, the same model
    without its degrees of freedom and scales, which ``fit`` computes for
    that type from the data.
    """
    h, X = _hand_built("VariationalGaussianHMM")
    assert isinstance(h, vhmm.VariationalGaussianHMM)
    if covariance_type != "diag":
        h.covariance_type = covariance_type
        for name in (
            "dof_prior_",
            "dof_posterior_",
            "scale_prior_",
            "scale_posterior_",
        ):
            delattr(h, name)
        h.random_state = 0
    return h, X


@pytest.mark.parametrize(
    "covariance_type", ["spherical", "diag", "full", "tied"]
)
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_beta_posterior_is_not_refused_by_fit(value, covariance_type):
    """
    ``fit`` with ``"m"`` in ``params`` replaces ``beta_posterior_`` in its
    first M-step: an infinite one is not refused; only the first bound,
    computed before that step, is NaN, and the fit gives finite parameters
    and later bounds.
    """
    h, X = _variational_gaussian(covariance_type)
    h.beta_posterior_ = np.array([value, 10.0])
    h.n_iter = 3
    # The first bound takes the log-determinant of a NaN matrix. LAPACK's
    # sign for it depends on the platform: where it is negative,
    # ``_utils.logdet`` warns as well as returning NaN.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "invalid value encountered in log", RuntimeWarning
        )
        h.fit(X)
    assert np.isnan(h.monitor_.history[0])
    assert np.isfinite(list(h.monitor_.history)[1:]).all()
    assert np.isfinite(h.means_posterior_).all()
    assert np.isfinite(h.beta_posterior_).all()


KEPT_BETA_POSTERIOR = [
    (covariance_type, params, [np.inf, 10.0], "must contain only finite")
    for covariance_type in ["spherical", "diag", "full", "tied"]
    for params in ["stc", "st"]
] + [("diag", "st", [np.nan, 10.0], "must not contain NaN")]


@pytest.mark.parametrize(
    "covariance_type, params, value, match",
    KEPT_BETA_POSTERIOR,
    ids=[
        f"{c}-{p}-{'nan' if np.isnan(v).any() else 'inf'}"
        for c, p, v, _ in KEPT_BETA_POSTERIOR
    ],
)
def test_variational_fit_refuses_an_infinite_beta_posterior_it_keeps(
    covariance_type, params, value, match
):
    """
    Without ``"m"`` in ``params`` the fit keeps ``beta_posterior_``, so an
    infinite entry is refused, naming it, before the first iteration, for
    every covariance type; a NaN is refused as NaN.
    """
    h, X = _variational_gaussian(covariance_type)
    h.params = params
    h.beta_posterior_ = np.array(value)
    with pytest.raises(ValueError, match=f"beta_posterior_ {match}"):
        h.fit(X)
    assert list(h.monitor_.history) == []


@pytest.mark.parametrize("name, attr", CONVERTED_CASES, ids=CONVERTED_IDS)
def test_assignment_converts_to_an_array(name, attr):
    """
    A parameter is converted when it is assigned and kept under its own
    name: a list or a tuple reads back as the array of the same bytes; an
    array is kept as the same object (a copy for ``GMMHMM``'s emission
    parameters); it is missing before it is assigned and after ``del``.
    """
    h, _ = _hand_built(name)
    array = getattr(h, attr)
    for given in (array.tolist(), tuple(array.tolist())):
        setattr(h, attr, given)
        stored = getattr(h, attr)
        assert type(stored) is np.ndarray
        assert _as_bytes(stored) == _as_bytes(array)
        assert vars(h)[attr] is stored
    setattr(h, attr, array)
    copied = name == "GMMHMM" and attr not in ("startprob_", "transmat_")
    assert (getattr(h, attr) is array) is not copied
    assert _as_bytes(getattr(h, attr)) == _as_bytes(array)
    del h.__dict__[attr]
    assert not hasattr(h, attr)
    setattr(h, attr, array)
    delattr(h, attr)
    assert not hasattr(h, attr)
    assert attr not in vars(h)
    assert not hasattr(type(h)(2), attr)
    assert isinstance(getattr(type(h), attr), base._ConvertedAttribute)


@pytest.mark.parametrize("name, attr", CONVERTED_CASES, ids=CONVERTED_IDS)
def test_ragged_list_is_refused_at_assignment(name, attr):
    """
    A list that does not make an array is refused when it is assigned,
    and the parameter keeps its value.
    """
    h, _ = _hand_built(name)
    array = getattr(h, attr)
    with pytest.raises(ValueError, match="inhomogeneous"):
        setattr(h, attr, [[0.5, 0.5], [1.0]])
    assert vars(h)[attr] is array


ONE_STATE_CASES = [
    (
        "CategoricalHMM",
        "emissionprob_",
        lambda: hmm.CategoricalHMM(1),
        np.array([[0], [2], [1], [1], [0]]),
    ),
    (
        "MultinomialHMM",
        "emissionprob_",
        lambda: hmm.MultinomialHMM(1, 5),
        np.array([[1, 2, 2], [0, 0, 5], [3, 1, 1]]),
    ),
    (
        "PoissonHMM",
        "lambdas_",
        lambda: hmm.PoissonHMM(1),
        np.array([[1, 4, 0], [2, 3, 1], [0, 5, 2]]),
    ),
]


def _one_state(factory, attr, emission, implementation="log"):
    h = factory()
    h.implementation = implementation
    h.init_params = ""
    h.startprob_ = np.array([1.0])
    h.transmat_ = np.array([[1.0]])
    setattr(h, attr, emission)
    return h


@pytest.mark.parametrize(
    "name, attr, factory, X",
    ONE_STATE_CASES,
    ids=[c[0] for c in ONE_STATE_CASES],
)
def test_one_state_1d_emission_is_stored_as_one_row(name, attr, factory, X):
    """
    A one-state model given its emission parameter as one flat row stores
    it as the ``(1, n_features)`` array.
    """
    h = _one_state(factory, attr, [0.2, 0.3, 0.5])
    stored = getattr(h, attr)
    assert type(stored) is np.ndarray
    assert _as_bytes(stored) == _as_bytes(np.array([[0.2, 0.3, 0.5]]))


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize(
    "name, attr, factory, X",
    ONE_STATE_CASES,
    ids=[c[0] for c in ONE_STATE_CASES],
)
def test_one_state_1d_emission_predicts_as_one_row(
    name, attr, factory, X, implementation
):
    """
    A one-state model given its emission parameter as one flat list
    predicts as the model given the ``(1, n_features)`` array.
    """
    h = _one_state(factory, attr, [0.2, 0.3, 0.5], implementation)
    expected = _one_state(
        factory, attr, np.array([[0.2, 0.3, 0.5]]), implementation
    )
    for call in PREDICTIONS.values():
        assert _as_bytes(call(h, X)) == _as_bytes(call(expected, X))


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize(
    "name, attrs",
    [(name, (attr,)) for name, attr in CONVERTED_CASES]
    + [
        (name, tuple(CONVERTED[name])) for name in EM_NAMES + VARIATIONAL_NAMES
    ],
    ids=CONVERTED_IDS
    + [f"{name}-all" for name in EM_NAMES + VARIATIONAL_NAMES],
)
def test_list_parameters_predict_as_arrays(name, attrs, implementation):
    """
    A model given a parameter, or all of them, as lists predicts as the
    model given the arrays: every prediction has the same bytes.
    """
    expected, X = _hand_built(name)
    h, _ = _hand_built(name)
    for model in (expected, h):
        model.implementation = implementation
    for attr in attrs:
        setattr(h, attr, getattr(h, attr).tolist())
    for call in _predictions(name).values():
        assert _as_bytes(call(h, X)) == _as_bytes(call(expected, X))


@pytest.mark.parametrize("name", EM_NAMES + VARIATIONAL_NAMES)
def test_state_holding_lists_loads_as_arrays(name):
    """
    A pickled state that holds parameters as lists, as an older version
    could store them, loads with the parameters as arrays, no attribute
    added, and predicts as the model built from the arrays.
    """
    expected, X = _hand_built(name)
    state = dict(expected.__getstate__())
    for attr in CONVERTED[name]:
        state[attr] = state[attr].tolist()
    keys = [k for k in state if k != "_sklearn_version"]
    loaded = type(expected).__new__(type(expected))
    loaded.__setstate__(state)
    for attr in CONVERTED[name]:
        assert type(getattr(loaded, attr)) is np.ndarray
    assert list(vars(loaded)) == keys
    for call in _predictions(name).values():
        assert _as_bytes(call(loaded, X)) == _as_bytes(call(expected, X))


@pytest.mark.parametrize("name", EM_NAMES + VARIATIONAL_NAMES)
def test_fit_from_list_parameters_equals_fit_from_arrays(name):
    """
    ``fit`` with ``init_params=""`` from parameters given as lists gives
    the fit from the arrays: every fitted array and the score history.
    """
    expected, X = _hand_built(name)
    h, _ = _hand_built(name)
    for attr in CONVERTED[name]:
        setattr(h, attr, getattr(h, attr).tolist())
    for model in (expected, h):
        model.n_iter = 3
        model.fit(X)
    assert h.monitor_.history == expected.monitor_.history
    fitted, wanted = _fitted_arrays(h), _fitted_arrays(expected)
    assert list(fitted) == list(wanted)
    assert all(_as_bytes(fitted[k]) == _as_bytes(wanted[k]) for k in wanted)


def test_stationary_distribution_of_a_list_transmat():
    """
    ``get_stationary_distribution`` of a ``transmat_`` given as a list,
    before any prediction, equals that of the array.
    """
    transmat = np.array([[0.7, 0.3], [0.4, 0.6]])
    h = StubHMM(2)
    h.transmat_ = transmat.tolist()
    expected = StubHMM(2)
    expected.transmat_ = transmat
    assert _as_bytes(h.get_stationary_distribution()) == _as_bytes(
        expected.get_stationary_distribution()
    )


# A sequence that underflows under implementation="scaling" -------------------

UNDERFLOW_TEXT = (
    "forward pass failed with underflow; consider using "
    "implementation='log' instead"
)


def _far_gaussian():
    """
    A two-state ``GaussianHMM`` fitted on standard normal data, and the same
    data with one sample far from both states.
    """
    X = np.random.RandomState(10).normal(size=(100, 1))
    model = hmm.GaussianHMM(2, random_state=0).fit(X)
    X[50] = 50.0
    return model, X


@pytest.mark.parametrize(
    "call",
    [
        lambda h, X: h.score(X),
        lambda h, X: h.score_samples(X),
        lambda h, X: h.predict_proba(X),
        lambda h, X: h.decode(X, algorithm="map"),
        lambda h, X: h.decode(X, algorithm="viterbi"),
        lambda h, X: h.aic(X),
        lambda h, X: h.bic(X),
    ],
    ids=[
        "score",
        "score_samples",
        "predict_proba",
        "decode-map",
        "decode-viterbi",
        "aic",
        "bic",
    ],
)
def test_scaling_underflow_predicts_as_log(call):
    """
    A sequence whose scaled forward pass underflows is computed with the log
    functions: every prediction equals the ``"log"`` result byte for byte.
    """
    model, X = _far_gaussian()
    model.implementation = "log"
    expected = call(model, X)
    model.implementation = "scaling"
    assert _as_bytes(call(model, X)) == _as_bytes(expected)
    assert "_accumulate_as_log" not in vars(model)


def _three_sequences():
    rs = np.random.RandomState(10)
    A, B, C = (rs.normal(size=(n, 1)) for n in (40, 30, 50))
    B[7] = 50.0
    return A, B, C


def test_scaling_underflow_leaves_the_other_sequences_unchanged():
    """
    Of three sequences, only the one that underflows is computed with the log
    functions: the posteriors of the other two are those of a scaling run
    without it, and the log-likelihood adds the three in sequence order.
    """
    model, _ = _far_gaussian()
    A, B, C = _three_sequences()
    model.implementation = "log"
    log_b, post_b = model.score_samples(B)
    model.implementation = "scaling"
    log_a, post_a = model.score_samples(A)
    log_c, post_c = model.score_samples(C)
    _, post_ac = model.score_samples(np.concatenate([A, C]), [40, 50])
    log_prob, post = model.score_samples(
        np.concatenate([A, B, C]), [40, 30, 50]
    )
    assert _as_bytes(np.concatenate([post[:40], post[70:]])) == _as_bytes(
        post_ac
    )
    assert _as_bytes(post[:40]) == _as_bytes(post_a)
    assert _as_bytes(post[70:]) == _as_bytes(post_c)
    assert _as_bytes(post[40:70]) == _as_bytes(post_b)
    assert _as_bytes(log_prob) == _as_bytes(0 + log_a + log_b + log_c)
    assert _as_bytes(model.score(np.concatenate([A, B, C]), [40, 30, 50])) == (
        _as_bytes(log_prob)
    )


def _upstream_far_data():
    """The data of ``test_underflow_from_scaling``: one sample at 10000."""
    prng = np.random.RandomState(10)
    data = np.concatenate(
        [
            prng.normal(0, 1, 100),
            prng.normal(5, 1, 100),
            prng.normal(0, 1, 100),
            prng.normal(5, 1, 100),
        ]
    )
    data[40] = 10000
    return data[:, None]


def _fixed_gaussian(cov, implementation, n_iter=1):
    h = hmm.GaussianHMM(
        2,
        n_iter=n_iter,
        covariance_type=cov,
        implementation=implementation,
        init_params="",
    )
    h.startprob_ = [0.0, 1]
    h.transmat_ = [[0.4, 0.6], [0.6, 0.4]]
    h.means_ = [[0], [5]]
    h.covars_ = {
        "spherical": [1, 1],
        "diag": [[1], [1]],
        "full": [[[1]], [[1]]],
        "tied": [[1]],
    }[cov]
    return h


def _underflow_fit_cases():
    """(id, factory(implementation), X): one fit iteration that falls back."""
    gauss = _upstream_far_data()
    clean = np.random.RandomState(10).normal(size=(200, 2))
    clean[100:] += 4
    far = clean.copy()
    far[50] = 1000.0
    counts = np.random.RandomState(10).poisson([2.0, 5.0], size=(100, 2))
    counts[30] = 1000
    multi = np.random.RandomState(10).multinomial(5, [0.3, 0.3, 0.4], size=60)
    multi[20] = [1000, 0, 0]
    symbols = np.random.RandomState(10).randint(0, 3, size=(60, 1))

    def gmm(cov):
        start = hmm.GMMHMM(
            2, n_mix=2, n_iter=1, covariance_type=cov, random_state=0
        ).fit(clean)

        def factory(implementation):
            h = hmm.GMMHMM(
                2,
                n_mix=2,
                n_iter=1,
                covariance_type=cov,
                implementation=implementation,
                init_params="",
                random_state=0,
            )
            h.startprob_, h.transmat_ = start.startprob_, start.transmat_
            h.weights_, h.means_ = start.weights_, start.means_
            h.covars_ = start.covars_
            return h

        return factory

    def poisson(implementation):
        h = hmm.PoissonHMM(
            2, n_iter=1, implementation=implementation, init_params=""
        )
        h.startprob_ = [0.5, 0.5]
        h.transmat_ = [[0.7, 0.3], [0.3, 0.7]]
        h.lambdas_ = [[2.0, 5.0], [5.0, 2.0]]
        return h

    def multinomial(implementation):
        h = hmm.MultinomialHMM(
            2, n_iter=1, implementation=implementation, init_params=""
        )
        h.startprob_ = [0.5, 0.5]
        h.transmat_ = [[0.7, 0.3], [0.3, 0.7]]
        h.emissionprob_ = [[0.3, 0.3, 0.4], [0.4, 0.4, 0.2]]
        return h

    def categorical(implementation):
        # A probability below the kernel's threshold, above zero.
        h = hmm.CategoricalHMM(
            2, n_iter=1, implementation=implementation, init_params=""
        )
        h.startprob_ = [0.5, 0.5]
        h.transmat_ = [[0.7, 0.3], [0.3, 0.7]]
        h.emissionprob_ = [[0.5, 0.5, 1e-320], [0.4, 0.6, 1e-320]]
        return h

    def variational_gaussian(cov):
        def factory(implementation):
            m = vhmm.VariationalGaussianHMM(
                2,
                covariance_type=cov,
                implementation="log",
                n_iter=5,
                random_state=0,
            ).fit(clean)
            m.implementation, m.init_params, m.n_iter = implementation, "", 1
            return m

        return factory

    def variational_categorical(implementation):
        m = vhmm.VariationalCategoricalHMM(
            2, implementation="log", n_iter=5, random_state=0
        ).fit(symbols)
        m.implementation, m.init_params, m.n_iter = implementation, "", 1
        m.emissionprob_posterior_ = m.emissionprob_posterior_.copy()
        m.emissionprob_posterior_[:, 2] = 1e-5
        return m

    covs = ("spherical", "diag", "full", "tied")
    cases = [
        (
            f"GaussianHMM-{cov}",
            lambda impl, cov=cov: _fixed_gaussian(cov, impl),
            gauss,
        )
        for cov in covs
    ]
    cases += [(f"GMMHMM-{cov}", gmm(cov), far) for cov in covs]
    cases += [
        ("PoissonHMM", poisson, counts),
        ("MultinomialHMM", multinomial, multi),
        ("CategoricalHMM", categorical, symbols),
    ]
    cases += [
        (f"VariationalGaussianHMM-{cov}", variational_gaussian(cov), far)
        for cov in covs
    ]
    cases += [("VariationalCategoricalHMM", variational_categorical, symbols)]
    return cases


UNDERFLOW_FIT_CASES = _underflow_fit_cases()
UNDERFLOW_FIT_IDS = [case[0] for case in UNDERFLOW_FIT_CASES]


def _fit_state(model):
    """Every fitted array, ``state_occupancy_`` included, and the history."""
    state = {
        k: _as_bytes(v)
        for k, v in vars(model).items()
        if k.endswith("_") and k != "monitor_"
    }
    state["monitor_.history"] = list(model.monitor_.history)
    return state


@pytest.mark.parametrize("case", UNDERFLOW_FIT_CASES, ids=UNDERFLOW_FIT_IDS)
def test_scaling_underflow_fits_as_log(case):
    """
    An iteration of ``fit`` whose only sequence underflows computes it with
    the log functions, statistics included: the fit equals the ``"log"``
    fit byte for byte, in every model class.
    """
    _, factory, X = case
    expected = factory("log").fit(X)
    model = factory("scaling").fit(X)
    assert _fit_state(model) == _fit_state(expected)
    assert "_accumulate_as_log" not in vars(model)


def _accumulate_recorder(model, fail_when_log=False):
    """
    Make ``model`` record the arguments of each accumulation, with whether
    the log statistics were asked for.
    """
    cls = type(model)

    class Recorder(cls):
        def _accumulate_sufficient_statistics(
            self, stats, X, lattice, posteriors, fwdlattice, bwdlattice
        ):
            as_log = self._accumulate_as_log
            self.seen.append(
                (
                    _as_bytes(
                        (X, lattice, posteriors, fwdlattice, bwdlattice)
                    ),
                    as_log,
                )
            )
            if fail_when_log and as_log:
                raise RuntimeError("accumulation failed")
            super()._accumulate_sufficient_statistics(
                stats, X, lattice, posteriors, fwdlattice, bwdlattice
            )

    model.__class__ = Recorder
    model.seen = []
    return model


@pytest.mark.parametrize("cov", ["spherical", "diag", "full", "tied"])
def test_scaling_underflow_accumulates_only_that_sequence_as_log(cov):
    """
    In a fit over three sequences of which the second underflows, the first
    and third are accumulated with the arguments of a scaling fit without
    the second, the second with those of a ``"log"`` fit of it alone, and
    the log statistics are asked for during the second call only.
    """
    data = _upstream_far_data()
    first, second, third = data[:100].copy(), data[100:250].copy(), data[250:]
    first[40] = 0.3
    second[5] = 10000
    three = _accumulate_recorder(_fixed_gaussian(cov, "scaling")).fit(
        np.concatenate([first, second, third]), [100, 150, 150]
    )
    two = _accumulate_recorder(_fixed_gaussian(cov, "scaling")).fit(
        np.concatenate([first, third]), [100, 150]
    )
    alone = _accumulate_recorder(_fixed_gaussian(cov, "log")).fit(second)
    assert [as_log for _, as_log in three.seen] == [False, True, False]
    assert three.seen[0][0] == two.seen[0][0]
    assert three.seen[2][0] == two.seen[1][0]
    assert three.seen[1][0] == alone.seen[0][0]
    assert "_accumulate_as_log" not in vars(three)
    assert set(vars(three)) == set(vars(two))


def test_scaling_underflow_flag_is_cleared_when_accumulation_raises():
    """The private flag does not outlive an accumulation that raises."""
    model = _accumulate_recorder(
        _fixed_gaussian("diag", "scaling"), fail_when_log=True
    )
    with pytest.raises(RuntimeError, match="accumulation failed"):
        model.fit(_upstream_far_data())
    assert "_accumulate_as_log" not in vars(model)
    assert model._accumulate_as_log is False


def _impossible(implementation="scaling"):
    """A categorical model in which symbol 2 has probability zero."""
    h = hmm.CategoricalHMM(
        2, init_params="", implementation=implementation, n_iter=3
    )
    h.startprob_ = np.array([0.5, 0.5])
    h.transmat_ = np.array([[0.7, 0.3], [0.3, 0.7]])
    h.emissionprob_ = np.array([[0.5, 0.5, 0.0], [0.4, 0.6, 0.0]])
    return h


IMPOSSIBLE = np.array([[0], [1], [2], [1], [0]])


@pytest.mark.parametrize("lengths", [None, [2, 5]], ids=["alone", "second"])
@pytest.mark.parametrize(
    "call",
    [
        lambda h, X, n: h.score(X, n),
        lambda h, X, n: h.score_samples(X, n),
        lambda h, X, n: h.predict_proba(X, n),
        lambda h, X, n: h.decode(X, n, algorithm="map"),
        lambda h, X, n: h.aic(X, n),
        lambda h, X, n: h.fit(X, n),
    ],
    ids=[
        "score",
        "score_samples",
        "predict_proba",
        "decode-map",
        "aic",
        "fit",
    ],
)
def test_scaling_underflow_of_an_impossible_sequence_still_raises(
    call, lengths
):
    """
    A sequence the log functions give no finite log-likelihood for raises
    the kernel's underflow error as before, with no warning first.
    """
    X = (
        IMPOSSIBLE
        if lengths is None
        else np.concatenate([IMPOSSIBLE[:2], IMPOSSIBLE])
    )
    h = _impossible()
    with pytest.raises(ValueError) as info:
        call(h, X, lengths)
    assert str(info.value) == UNDERFLOW_TEXT
    assert "_accumulate_as_log" not in vars(h)


def _underflow_records(caplog):
    return [
        r
        for r in caplog.records
        if r.name == "hmmcraft.base" and "underflowed" in r.getMessage()
    ]


def test_scaling_underflow_logs_one_warning_per_call(caplog):
    """
    A scoring call in which a sequence falls back logs one warning naming
    how many of how many sequences did and the first index; a scaling call
    without underflow logs none.
    """
    model, _ = _far_gaussian()
    A, B, C = _three_sequences()
    model.implementation = "scaling"
    with caplog.at_level("WARNING", logger="hmmcraft.base"):
        model.score(np.concatenate([A, C]), [40, 50])
        assert _underflow_records(caplog) == []
        model.score(np.concatenate([A, B, C]), [40, 30, 50])
    records = _underflow_records(caplog)
    assert len(records) == 1
    assert records[0].levelname == "WARNING"
    assert "1 of 3 sequences" in records[0].getMessage()
    assert "index 1" in records[0].getMessage()


def test_scaling_underflow_logs_one_warning_per_e_step(caplog):
    """An E-step in which sequences fall back logs one warning."""
    data = _upstream_far_data()
    with caplog.at_level("WARNING", logger="hmmcraft.base"):
        _fixed_gaussian("diag", "scaling").fit(data, [200, 200])
        records = _underflow_records(caplog)
        assert len(records) == 1
        assert "1 of 2 sequences" in records[0].getMessage()
        caplog.clear()
        _fixed_gaussian("diag", "scaling").fit(data[200:])
        assert _underflow_records(caplog) == []


# The E-step's private list ---------------------------------------------------


def _sequence_recorder(model):
    """
    Make ``model`` record, at each accumulation, whether the E-step's
    private list holds the sequence being accumulated.
    """
    cls = type(model)

    class Recorder(cls):
        def _accumulate_sufficient_statistics(self, stats, X, *args):
            self.seen.append(vars(self)["_estep_values"][0] is X)
            super()._accumulate_sufficient_statistics(stats, X, *args)

    model.__class__ = Recorder
    model.seen = []
    return model


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("name", ["GaussianHMM", "GMMHMM"])
def test_e_step_values_hold_the_sequence_while_it_is_computed(
    name, implementation
):
    """
    While ``_do_estep`` computes a sequence, its private list on the
    instance holds that sequence first.
    """
    _, factory, X = MODEL_CASES[MODEL_IDS.index(name)]
    model = factory(2)
    model.implementation = implementation
    model.n_iter = 1
    _sequence_recorder(model).fit(X, [12, 8])
    assert model.seen == [True, True]
    assert "_estep_values" not in vars(model)


class _RaisingStatistics(hmm.GMMHMM):
    def _accumulate_sufficient_statistics(self, *args):
        raise RuntimeError("accumulation failed")


class _RaisingLikelihood(hmm.GMMHMM):
    def _compute_log_likelihood(self, X):
        raise RuntimeError("likelihood failed")


def _estep_values_cases():
    """(id, model, X, the error fit raises or None)."""
    cases = [(name, factory(2), X, None) for name, factory, X in MODEL_CASES]
    _, _, X = MODEL_CASES[MODEL_IDS.index("GMMHMM")]
    cases += [
        (
            "GMMHMM-statistics-raise",
            _RaisingStatistics(2, n_mix=2, random_state=0),
            X,
            RuntimeError,
        ),
        (
            "GMMHMM-likelihood-raise",
            _RaisingLikelihood(2, n_mix=2, random_state=0),
            X,
            RuntimeError,
        ),
        ("impossible-scaling", _impossible(), IMPOSSIBLE, ValueError),
    ]
    return cases


@pytest.mark.parametrize(
    "case",
    _estep_values_cases(),
    ids=[case[0] for case in _estep_values_cases()],
)
def test_e_step_values_are_removed_after_a_fit_and_when_a_hook_raises(case):
    """
    The E-step's private list does not outlive ``fit``, also when a hook
    raises; every name ``fit`` adds to the instance ends in an underscore,
    but ``n_features``, which the start sets.
    """
    _, model, X, error = case
    before = set(vars(model))
    if error is None:
        model.fit(X)
    else:
        with pytest.raises(error):
            model.fit(X)
    assert "_estep_values" not in vars(model)
    assert type(model)._estep_values is None
    added = {
        name for name in set(vars(model)) - before if not name.endswith("_")
    }
    assert added <= {"n_features"}
