import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import special

from hmmcraft import _hmmc, hmm, vhmm
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
        log_prob, fwdlattice, _scaling_factors = _hmmc.forward_scaling(
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
        _log_prob, _fwdlattice, scaling_factors = _hmmc.forward_scaling(
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
        _X0, Z0 = self.hmm.sample(n_samples=10)
        _X, Z = self.hmm.sample(n_samples=10, currstate=Z0[-1])
        assert len(Z0) == len(Z) == 10 and Z[0] == Z0[-1]

    @pytest.mark.parametrize(
        "n_samples", [0, -1, np.array(0), 0.5, False])
    def test_sample_refuses_fewer_than_one(self, n_samples):
        """``sample`` draws at least one sample."""
        with pytest.raises(ValueError, match="n_samples must be at least 1"):
            self.hmm.sample(n_samples)

    def test_sample_float_keeps_type_error(self):
        """A float count of at least 1 fails where it did before."""
        with pytest.raises(TypeError):
            self.hmm.sample(2.0)

    @pytest.mark.parametrize("n_samples, length", [
        (1, 1), (True, 1), (np.int64(1), 1), (np.array(1), 1),
        (np.array(True), 1), (np.True_, 1),
        (np.array(2, dtype=object), 2), (np.array(3), 3),
    ])
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
        _log_prob, hmmposteriors = self.hmm.score_samples(self.log_frameprob)

        n_samples, n_components = self.log_frameprob.shape
        assert_allclose(hmmposteriors.sum(axis=1), np.ones(n_samples))

        norm = special.logsumexp(self.log_frameprob, axis=1)[:, np.newaxis]
        gmmposteriors = np.exp(self.log_frameprob
                               - np.tile(norm, (1, n_components)))
        assert_allclose(hmmposteriors, gmmposteriors)

    def test_decode(self):
        _log_prob, state_sequence = self.hmm.decode(self.log_frameprob)

        _n_samples, n_components = self.log_frameprob.shape
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


LEFT_RIGHT_START = np.array([1.0, 0.0, 0.0])
LEFT_RIGHT_TRANSMAT = np.array([[0.5, 0.5, 0.0],
                                [0.0, 0.5, 0.5],
                                [0.0, 0.0, 1.0]])


def _with(array, index, value):
    array = array.copy()
    array[index] = value
    return array


@pytest.mark.parametrize("implementation", ["scaling", "log"])
@pytest.mark.parametrize("startprob, transmat", [
    (LEFT_RIGHT_START, LEFT_RIGHT_TRANSMAT),
    (np.array([1.0, -0.0, -0.0]),
     np.where(LEFT_RIGHT_TRANSMAT == 0, -0.0, LEFT_RIGHT_TRANSMAT)),
    (LEFT_RIGHT_START, _with(LEFT_RIGHT_TRANSMAT, (0, 2), 5e-324)),
    (np.array([1 + 1e-9, 0.0, 0.0]), LEFT_RIGHT_TRANSMAT),
], ids=["zeros", "negative-zeros", "denormal", "start-above-1"])
def test_structural_zeros_fit_and_score(startprob, transmat, implementation):
    """
    A left-right model is valid: zeros (``-0.0`` too), a denormal entry and
    a start probability above 1 within the sum's tolerance are accepted;
    ``score`` is the log of the sum over every state path, and a fit keeps
    every zero at zero.
    """
    emissionprob = np.array([[0.7, 0.2, 0.1],
                             [0.1, 0.7, 0.2],
                             [0.2, 0.1, 0.7]])
    X = np.array([[0], [2], [1]])
    # Every state is visited and left, so no row of the fit is empty.
    X_fit = np.array([[0], [0], [1], [1], [2], [2]])

    def brute_force_log_likelihood(h):
        total = 0.0
        for path in itertools.product(range(3), repeat=len(X)):
            p = h.startprob_[path[0]] * h.emissionprob_[path[0], X[0, 0]]
            for t in range(1, len(X)):
                p *= (h.transmat_[path[t - 1], path[t]]
                      * h.emissionprob_[path[t], X[t, 0]])
            total += p
        return np.log(total)

    h = hmm.CategoricalHMM(3, init_params="", n_iter=5,
                           implementation=implementation)
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


def _model_cases():
    """(id, factory taking n_components, X) for every model class."""
    rs = np.random.RandomState(10)
    continuous = rs.normal(size=(20, 2)) + 3 * rs.randint(0, 2, (20, 1))
    symbols = rs.randint(0, 3, (20, 1))
    counts = rs.poisson(4, (20, 3))
    multinomial = rs.multinomial(5, [0.2, 0.3, 0.5], size=20)
    return [
        ("GaussianHMM", lambda n: hmm.GaussianHMM(n, random_state=0),
         continuous),
        ("GMMHMM", lambda n: hmm.GMMHMM(n, random_state=0), continuous),
        ("CategoricalHMM", lambda n: hmm.CategoricalHMM(n, random_state=0),
         symbols),
        ("MultinomialHMM",
         lambda n: hmm.MultinomialHMM(n, 5, random_state=0), multinomial),
        ("PoissonHMM", lambda n: hmm.PoissonHMM(n, random_state=0), counts),
        ("VariationalCategoricalHMM",
         lambda n: vhmm.VariationalCategoricalHMM(n, random_state=0),
         symbols),
        ("VariationalGaussianHMM",
         lambda n: vhmm.VariationalGaussianHMM(n, random_state=0),
         continuous),
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


@pytest.mark.parametrize("call", [
    lambda h, X: h.fit(X),
    lambda h, X: h.score(X),
    lambda h, X: h.score_samples(X),
    lambda h, X: h.predict_proba(X),
    lambda h, X: h.decode(X, algorithm="map"),
], ids=["fit", "score", "score_samples", "predict_proba", "decode-map"])
def test_unknown_implementation_raises(call):
    """Every path that reads ``implementation`` names it when it is wrong."""
    h = new_hmm("foo")
    with pytest.raises(ValueError, match="implementation"):
        call(h, np.array([[0], [1]]))


def test_unknown_implementation_raises_in_variational_fit():
    """The variational models read ``implementation`` through one place."""
    model = vhmm.VariationalGaussianHMM(2, implementation="foo",
                                        random_state=0)
    X = np.random.RandomState(10).normal(size=(20, 2))
    with pytest.raises(ValueError, match="implementation"):
        model.fit(X)


def _hand_built(name):
    """A two-state model with every parameter set and no ``n_features``."""
    rs = np.random.RandomState(10)
    if name == "GaussianHMM":
        h = hmm.GaussianHMM(2, init_params="")
        h.means_ = np.array([[0.0, 0.0], [3.0, 3.0]])
        h.covars_ = np.ones((2, 2))
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
@pytest.mark.parametrize("name",
                         ["GaussianHMM", "CategoricalHMM", "PoissonHMM"])
def test_aic_bic_before_n_features(name, criterion):
    """
    ``aic`` and ``bic`` of a model built by hand take the number of
    features from its emission parameter: the value equals the one after a
    ``score`` has set ``n_features``.
    """
    h, X = _hand_built(name)
    before = getattr(h, criterion)(X)
    h.score(X)
    assert getattr(h, "n_features", None) is not None
    assert before == getattr(h, criterion)(X)


@pytest.mark.parametrize("name", ["MultinomialHMM", "PoissonHMM"])
def test_n_features_none_is_taken_from_emissions(name):
    """
    An ``n_features`` set to ``None`` is read from the emission parameter,
    as when it is not set: the same score, and the count is stored.
    """
    h, X = _hand_built(name)
    expected = h.score(X)
    h, X = _hand_built(name)
    h.n_features = None
    assert h.score(X) == expected
    assert h.n_features == (3 if name == "MultinomialHMM" else 2)


def _refused_X(name, X):
    """``X`` changed so that the model fitted on ``X`` refuses it."""
    if name in ("CategoricalHMM", "VariationalCategoricalHMM"):
        return X + 3
    if name in ("MultinomialHMM", "PoissonHMM"):
        return -1 - X
    return X[:, :1]


@pytest.mark.parametrize("case", MODEL_CASES, ids=MODEL_IDS)
def test_X_check_writes_nothing(case):
    """
    The check of ``X`` against a fitted model reads it and writes nothing,
    whether it passes ``X`` or refuses it: every attribute keeps its object
    and every array its bytes.
    """
    name, factory, X = case
    model = factory(2).fit(X)
    model._check()
    before = dict(vars(model))
    arrays = {k: v.tobytes() for k, v in before.items()
              if isinstance(v, np.ndarray)}
    model._check_X(X)
    with pytest.raises(ValueError, match="X must"):
        model._check_X(_refused_X(name, X))
    assert vars(model).keys() == before.keys()
    assert all(vars(model)[k] is v for k, v in before.items())
    assert all(vars(model)[k].tobytes() == b for k, b in arrays.items())
