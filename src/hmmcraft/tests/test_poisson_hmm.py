import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import special
from scipy.stats import poisson
from sklearn.utils import check_random_state

from hmmcraft import hmm

from . import assert_log_likelihood_increasing, normalized


class TestPoissonHMM:
    n_components = 2
    n_features = 3

    def new_hmm(self, impl):
        h = hmm.PoissonHMM(
            self.n_components, implementation=impl, random_state=0
        )
        h.startprob_ = np.array([0.6, 0.4])
        h.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]])
        h.lambdas_ = np.array([[3.1, 1.4, 4.5], [1.6, 5.3, 0.1]])
        return h

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_attributes(self, implementation):
        with pytest.raises(ValueError):
            h = self.new_hmm(implementation)
            h.lambdas_ = []
            h._check()
        with pytest.raises(ValueError):
            h.lambdas_ = np.zeros((self.n_components - 2, self.n_features))
            h._check()
        for value in (np.nan, np.inf, -1.4):
            h = self.new_hmm(implementation)
            h.lambdas_ = np.array([[3.1, value, 4.5], [1.6, 5.3, 0.1]])
            with pytest.raises(ValueError, match="lambdas_"):
                h._check()

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_zero_lambdas_score(self, implementation):
        """
        A rate of exactly 0 is valid: the score equals a forward pass over
        the Poisson log-probabilities, where a count above 0 is impossible.
        """
        h = self.new_hmm(implementation)
        h.lambdas_ = np.array([[3.1, 0.0, 4.5], [1.6, 5.3, 0.0]])
        X = np.array([[2, 0, 5], [4, 0, 3], [1, 6, 0], [3, 4, 0]])
        log_frameprob = poisson.logpmf(X[:, None, :], h.lambdas_[None]).sum(
            axis=-1
        )
        log_alpha = np.log(h.startprob_) + log_frameprob[0]
        for t in range(1, len(X)):
            log_alpha = log_frameprob[t] + special.logsumexp(
                log_alpha[:, None] + np.log(h.transmat_), axis=0
            )
        assert_allclose(h.score(X), special.logsumexp(log_alpha), rtol=1e-12)

    def _counts(self):
        """200 counts sampled from the model of ``new_hmm``."""
        return self.new_hmm("log").sample(200, random_state=0)[0]

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    @pytest.mark.parametrize(
        "transform",
        [
            lambda counts: counts / 2,
            lambda counts: counts - 1,
        ],
        ids=["halved", "minus-one"],
    )
    def test_fit_refuses_X_that_is_not_counts(self, transform, implementation):
        """
        ``fit`` refuses counts that are not whole numbers or are negative,
        naming ``X``, instead of fitting every rate to NaN.
        """
        h = hmm.PoissonHMM(
            self.n_components,
            n_iter=5,
            random_state=0,
            implementation=implementation,
        )
        with (
            np.errstate(all="ignore"),
            pytest.raises(ValueError, match="X must contain counts"),
        ):
            h.fit(transform(self._counts()))

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    @pytest.mark.parametrize(
        "X, match",
        [
            ([[2, 1.5, 3], [1, 0, 4]], "X must contain counts"),
            ([[2, -1, 3], [1, 0, 4]], "X must contain counts"),
            ([[2], [1]], "X must have n_features=3"),
        ],
        ids=["fraction", "negative", "one-column"],
    )
    def test_score_refuses_X_that_is_not_counts(
        self, X, match, implementation
    ):
        """
        ``score`` refuses, naming ``X``, a count that is negative or not a
        whole number, and ``X`` of another number of columns.
        """
        h = self.new_hmm(implementation)
        with np.errstate(all="ignore"), pytest.raises(ValueError, match=match):
            h.score(np.array(X))

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_whole_float_counts_fit_as_integers(self, implementation):
        """
        Counts held as whole floats fit and score as the same counts held
        as integers.
        """
        counts = self._counts()

        def fitted(X):
            return hmm.PoissonHMM(
                self.n_components,
                n_iter=5,
                random_state=0,
                implementation=implementation,
            ).fit(X)

        expected, h = fitted(counts), fitted(counts * 1.0)
        assert_array_equal(h.lambdas_, expected.lambdas_)
        assert h.score(counts) == expected.score(counts)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_boolean_counts_fit_and_score(self, implementation):
        """Counts of 0 and 1 held as booleans fit and score finitely."""
        X = self._counts() > 2
        h = hmm.PoissonHMM(
            self.n_components,
            n_iter=5,
            random_state=0,
            implementation=implementation,
        ).fit(X)
        assert np.isfinite(h.score(X))

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_score_samples(self, implementation, n_samples=1000):
        h = self.new_hmm(implementation)
        X, state_sequence = h.sample(n_samples)
        assert X.ndim == 2
        assert len(X) == len(state_sequence) == n_samples

        _ll, posteriors = h.score_samples(X)
        assert posteriors.shape == (n_samples, self.n_components)
        assert_allclose(posteriors.sum(axis=1), np.ones(n_samples))

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit(self, implementation, params="stl", n_iter=5):

        h = self.new_hmm(implementation)
        h.params = params

        lengths = np.array([10] * 10)
        X, _state_sequence = h.sample(lengths.sum())

        # Mess up the parameters and see if we can re-learn them.
        np.random.seed(0)
        h.startprob_ = normalized(np.random.random(self.n_components))
        h.transmat_ = normalized(
            np.random.random((self.n_components, self.n_components)), axis=1
        )
        h.lambdas_ = np.random.gamma(
            shape=2, size=(self.n_components, self.n_features)
        )

        assert_log_likelihood_increasing(h, X, lengths, n_iter)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_lambdas(self, implementation):
        self.test_fit(implementation, "l")

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_with_init(self, implementation, params="stl", n_iter=5):
        lengths = [10] * 10
        h = self.new_hmm(implementation)
        X, _state_sequence = h.sample(sum(lengths))

        # use init_function to initialize paramerters
        h = hmm.PoissonHMM(
            self.n_components, params=params, init_params=params
        )
        h._init(X, lengths)

        assert_log_likelihood_increasing(h, X, lengths, n_iter)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_criterion(self, implementation):
        random_state = check_random_state(412)
        m1 = self.new_hmm(implementation)
        X, _ = m1.sample(2000, random_state=random_state)

        aic = []
        bic = []
        ns = [2, 3, 4]
        for n in ns:
            h = hmm.PoissonHMM(
                n,
                n_iter=500,
                random_state=random_state,
                implementation=implementation,
            )
            h.fit(X)
            aic.append(h.aic(X))
            bic.append(h.bic(X))

        assert np.all(aic) > 0
        assert np.all(bic) > 0
        # AIC / BIC pick the right model occasionally
        # assert ns[np.argmin(aic)] == 2
        # assert ns[np.argmin(bic)] == 2
