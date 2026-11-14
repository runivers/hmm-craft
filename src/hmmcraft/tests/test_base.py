import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import special

from hmmcraft.base import BaseHMM, ConvergenceMonitor
from hmmcraft import _hmmc, hmm, vhmm


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
        self.frameprob = np.asarray([[0.9, 0.2],
                                     [0.9, 0.2],
                                     [0.1, 0.8],
                                     [0.9, 0.2],
                                     [0.9, 0.2]])
        self.log_frameprob = np.log(self.frameprob)
        h = StubHMM(2)
        h.transmat_ = [[0.7, 0.3], [0.3, 0.7]]
        h.startprob_ = [0.5, 0.5]
        h.log_frameprob = self.log_frameprob
        h.frameprob = self.frameprob
        self.hmm = h

    def test_do_forward_scaling_pass(self):
        log_prob, fwdlattice, scaling_factors = _hmmc.forward_scaling(
            self.hmm.startprob_, self.hmm.transmat_, self.frameprob)
        ref_log_prob = -3.3725
        assert round(log_prob, 4) == ref_log_prob
        reffwdlattice = np.exp([[0.4500, 0.1000],
                                [0.3105, 0.0410],
                                [0.0230, 0.0975],
                                [0.0408, 0.0150],
                                [0.0298, 0.0046]])
        assert_allclose(np.exp(fwdlattice), reffwdlattice, 4)

    def test_do_forward_pass(self):
        log_prob, fwdlattice = _hmmc.forward_log(
            self.hmm.startprob_, self.hmm.transmat_, self.log_frameprob)

        ref_log_prob = -3.3725
        assert round(log_prob, 4) == ref_log_prob
        reffwdlattice = np.array([[0.4500, 0.1000],
                                  [0.3105, 0.0410],
                                  [0.0230, 0.0975],
                                  [0.0408, 0.0150],
                                  [0.0298, 0.0046]])
        assert_allclose(np.exp(fwdlattice), reffwdlattice, 4)

    def test_do_backward_scaling_pass(self):
        log_prob, fwdlattice, scaling_factors = _hmmc.forward_scaling(
            self.hmm.startprob_, self.hmm.transmat_, self.frameprob)
        bwdlattice = _hmmc.backward_scaling(self.hmm.startprob_,
            self.hmm.transmat_, self.frameprob, scaling_factors)
        refbwdlattice = np.array([[0.0661, 0.0455],
                                  [0.0906, 0.1503],
                                  [0.4593, 0.2437],
                                  [0.6900, 0.4100],
                                  [1.0000, 1.0000]])
        scaling_factors = np.cumprod(scaling_factors[::-1])[::-1]
        bwdlattice_scaled = bwdlattice / scaling_factors[:, None]
        # Answer will be equivalent when the scaling factor is accounted for
        assert_allclose(bwdlattice_scaled, refbwdlattice, 4)

    def test_do_backward_log_pass(self):
        bwdlattice = _hmmc.backward_log(
            self.hmm.startprob_, self.hmm.transmat_, self.log_frameprob)
        refbwdlattice = np.array([[0.0661, 0.0455],
                                  [0.0906, 0.1503],
                                  [0.4593, 0.2437],
                                  [0.6900, 0.4100],
                                  [1.0000, 1.0000]])
        assert_allclose(np.exp(bwdlattice), refbwdlattice, 4)

    def test_do_viterbi_pass(self):
        log_prob, state_sequence = _hmmc.viterbi(
            self.hmm.startprob_, self.hmm.transmat_, self.log_frameprob)
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
        refposteriors = np.array([[0.8673, 0.1327],
                                  [0.8204, 0.1796],
                                  [0.3075, 0.6925],
                                  [0.8204, 0.1796],
                                  [0.8673, 0.1327]])
        assert_allclose(posteriors, refposteriors, atol=1e-4)

    def test_generate_samples(self):
        X0, Z0 = self.hmm.sample(n_samples=10)
        X, Z = self.hmm.sample(n_samples=10, currstate=Z0[-1])
        assert len(Z0) == len(Z) == 10 and Z[0] == Z0[-1]


class TestBaseConsistentWithGMM:
    def setup_method(self, method):
        n_components = 8
        n_samples = 10

        self.log_frameprob = np.log(
            np.random.random((n_samples, n_components)))

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
        log_prob, hmmposteriors = self.hmm.score_samples(self.log_frameprob)

        n_samples, n_components = self.log_frameprob.shape
        assert_allclose(hmmposteriors.sum(axis=1), np.ones(n_samples))

        norm = special.logsumexp(self.log_frameprob, axis=1)[:, np.newaxis]
        gmmposteriors = np.exp(self.log_frameprob
                               - np.tile(norm, (1, n_components)))
        assert_allclose(hmmposteriors, gmmposteriors)

    def test_decode(self):
        _log_prob, state_sequence = self.hmm.decode(self.log_frameprob)

        n_samples, n_components = self.log_frameprob.shape
        norm = special.logsumexp(self.log_frameprob, axis=1)[:, np.newaxis]
        gmmposteriors = np.exp(self.log_frameprob
                               - np.tile(norm, (1, n_components)))
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


def test_stationary_distribution():
    n_components = 10
    h = StubHMM(n_components)
    transmat = np.random.random((n_components, n_components))
    transmat /= np.tile(transmat.sum(axis=1)[:, np.newaxis], (1, n_components))
    h.transmat_ = transmat
    stationary = h.get_stationary_distribution()
    assert stationary.dtype == float
    assert (h.get_stationary_distribution().T @ h.transmat_
            == pytest.approx(stationary))


class EqualEmissionsHMM(BaseHMM):
    """An HMM whose states emit every observation with the same probability."""
    def _compute_log_likelihood(self, X):
        return np.zeros((len(X), self.n_components))


@pytest.mark.parametrize("lengths", [None, [3, 2]])
@pytest.mark.parametrize("implementation", ["scaling", "log"])
def test_map_log_prob_is_the_log_of_the_chosen_posteriors(
        implementation, lengths):
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
    h = hmm.GaussianHMM(2, "diag", init_params="",
                        implementation=implementation)
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
    cases = [(f"GaussianHMM-{cov}",
              lambda cls=hmm.GaussianHMM, cov=cov: cls(
                  2, cov, n_iter=4, random_state=0),
              continuous, "post")
             for cov in ("spherical", "diag", "full", "tied")]
    cases += [
        ("GMMHMM", lambda: hmm.GMMHMM(2, 2, n_iter=4, random_state=0),
         continuous, "post_sum"),
        ("PoissonHMM", lambda: hmm.PoissonHMM(2, n_iter=4, random_state=0),
         counts, "post"),
        ("VariationalGaussianHMM",
         lambda: vhmm.VariationalGaussianHMM(2, n_iter=4, random_state=0),
         continuous, "post"),
        ("CategoricalHMM",
         lambda: hmm.CategoricalHMM(2, n_iter=4, random_state=0),
         symbols, None),
        ("MultinomialHMM",
         lambda: hmm.MultinomialHMM(2, 5, n_iter=4, random_state=0),
         multinomial, None),
        ("VariationalCategoricalHMM",
         lambda: vhmm.VariationalCategoricalHMM(2, n_iter=4, random_state=0),
         symbols, None),
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
                self, stats, X, lattice, posteriors, fwdlattice, bwdlattice):
            if getattr(self, "recorded_sum", None) is None:
                self.recorded_sum = np.zeros(self.n_components)
            self.recorded_sum += posteriors.sum(axis=0)
            super()._accumulate_sufficient_statistics(
                stats, X, lattice, posteriors, fwdlattice, bwdlattice)

        def _do_mstep(self, stats):
            self.recorded_stat = None if statistic is None else (
                stats[statistic].copy())
            self.recorded_posterior_sum, self.recorded_sum = (
                self.recorded_sum, None)
            super()._do_mstep(stats)

    model.__class__ = Recorder
    return model


def _fitted_arrays(model):
    return {k: np.asarray(v) for k, v in vars(model).items()
            if k.endswith("_") and k not in (
                "monitor_", "state_occupancy_")}


@pytest.mark.parametrize("case", [c for c in OCCUPANCY_CASES if c[3]],
                         ids=[c[0] for c in OCCUPANCY_CASES if c[3]])
def test_state_occupancy_equals_the_last_m_step_statistic(case):
    _, factory, X, statistic = case
    model = _recording(factory(), statistic).fit(X, LENGTHS)
    assert model.state_occupancy_.tobytes() == (
        model.recorded_stat.tobytes())


@pytest.mark.parametrize("case", OCCUPANCY_CASES, ids=OCCUPANCY_IDS)
def test_state_occupancy_is_the_posterior_sum_for_every_model(case):
    _, factory, X, statistic = case
    model = _recording(factory(), statistic).fit(X, LENGTHS)
    assert model.state_occupancy_.shape == (2,)
    assert model.state_occupancy_.tobytes() == (
        model.recorded_posterior_sum.tobytes())
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
