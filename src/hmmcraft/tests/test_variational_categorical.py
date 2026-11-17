import numpy as np
import pytest
from sklearn.utils import check_random_state

from hmmcraft import hmm, vhmm

from . import (
    assert_log_likelihood_increasing,
    compare_variational_and_em_models,
    vi_uniform_startprob_and_transmat,
)


class TestVariationalCategorical:

    @pytest.fixture(autouse=True)
    def setup(self):
        # We fix the random state here to demonstrate that the model will
        # successfully remove "unnecessary" states.  In practice,
        # one should not set the random_state, and perform multiple
        # training steps, and take the model with the best lower-bound
        self.n_components = 3
        self.implementations = ["scaling", "log"]

    @staticmethod
    def get_beal_models():
        m1 = hmm.CategoricalHMM(3, init_params="")
        m1.n_features = 3
        m1.startprob_ = np.array([1/3., 1/3., 1/3.])
        m1.transmat_ = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        m1.emissionprob_ = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]])
        m2 = hmm.CategoricalHMM(3)
        m2.n_features = 3
        m2.startprob_ = np.array([1/3., 1/3., 1/3.])
        m2.transmat_ = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        m2.emissionprob_ = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]])
        m3 = hmm.CategoricalHMM(1)
        m3.n_features = 3
        m3.startprob_ = np.array([1])
        m3.transmat_ = np.array([[1]])
        m3.emissionprob_ = np.array([[0.5, 0.5, 0]])
        return m1, m2, m3

    @classmethod
    def get_from_one_beal(cls, N, length, rs=None):
        # Just fit the first of the beal models
        model = cls.get_beal_models()[0]
        sequences = []
        lengths = []
        for i in range(N):
            sequences.append(
                model.sample(length, random_state=check_random_state(rs))[0])
            lengths.append(len(sequences[-1]))
        sequences = np.concatenate(sequences)
        return sequences, lengths

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_init_priors(self, implementation):
        sequences, lengths = self.get_from_one_beal(7, 100, None)
        model = vhmm.VariationalCategoricalHMM(
            4, n_iter=500, random_state=1984, init_params="",
            implementation=implementation)
        model.pi_prior_ = np.full((4,), .25)
        model.pi_posterior_ = np.full((4,), 7/4)
        model.transmat_prior_ = np.full((4, 4), .25)
        model.transmat_posterior_ = np.full((4, 4), 7/4)
        model.emissionprob_prior_ = np.full((4, 3), 1/3)
        model.emissionprob_posterior_ = np.asarray([[.3, .4, .3],
                                                    [.8, .1, .1],
                                                    [.2, .2, .6],
                                                    [.2, .6, .2]])
        assert_log_likelihood_increasing(model, sequences, lengths, 10)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_n_features(self, implementation):

        sequences, lengths = self.get_from_one_beal(7, 100, None)
        # Learn n_Features
        model = vhmm.VariationalCategoricalHMM(
            4, implementation=implementation)
        assert_log_likelihood_increasing(model, sequences, lengths, 10)
        assert model.n_features == 3
        # Respect n_features
        model = vhmm.VariationalCategoricalHMM(
            4, implementation=implementation, n_features=5)

        assert_log_likelihood_increasing(model, sequences, lengths, 10)
        assert model.n_features == 5

        # Too few features
        with pytest.raises(ValueError):
            model = vhmm.VariationalCategoricalHMM(
                4, n_iter=500, random_state=1984,
                implementation=implementation)
            model.n_features = 2
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # No Negative Values
        with pytest.raises(ValueError):
            model = vhmm.VariationalCategoricalHMM(
                4, n_iter=500, random_state=1984,
                implementation=implementation)
            sequences[0] = -1
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Must be integers
        with pytest.raises(ValueError):
            model = vhmm.VariationalCategoricalHMM(
                4, n_iter=500, random_state=1984,
                implementation=implementation)
            sequences = sequences.astype(float)
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_counts_symbols_from_a_kept_posterior(self, implementation):
        """
        Without ``n_features``, ``fit`` takes the number of symbols from a
        kept ``emissionprob_posterior_`` with more symbols than ``X``
        shows, and fits as the model given that number; a symbol the
        posterior does not emit is refused, naming it, and ``n_features``
        stays unset.
        """
        X = np.random.RandomState(10).randint(0, 2, (30, 1))
        expected = self._fitted(3, implementation)
        expected.init_params = ""
        expected.fit(X)
        model = self._fitted(3, implementation)
        model.init_params = ""
        model.n_features = None
        model.fit(X)
        assert model.n_features == 3
        assert model.monitor_.history == expected.monitor_.history
        assert vars(model).keys() == vars(expected).keys()
        for name, value in vars(expected).items():
            if name.endswith("_") and name != "monitor_":
                fitted = np.asarray(vars(model)[name])
                assert fitted.dtype == np.asarray(value).dtype
                assert fitted.tobytes() == np.asarray(value).tobytes()
        model = self._fitted(3, implementation)
        model.init_params = ""
        model.n_features = None
        with pytest.raises(ValueError, match="Largest symbol is 3"):
            model.fit(np.array([[0], [3], [1]]))
        assert model.n_features is None

    @staticmethod
    def _hand_built(implementation):
        """
        A two-state model built by hand for three symbols, ``n_features``
        and ``init_params`` left at their defaults.
        """
        model = vhmm.VariationalCategoricalHMM(
            2, n_iter=5, random_state=0, implementation=implementation)
        model.startprob_prior_ = np.full(2, 0.5)
        model.startprob_posterior_ = np.array([6.0, 4.0])
        model.transmat_prior_ = np.full((2, 2), 0.5)
        model.transmat_posterior_ = np.array([[7.0, 3.0], [4.0, 6.0]])
        model.emissionprob_prior_ = np.full((2, 3), 1 / 3)
        model.emissionprob_posterior_ = np.array([[2.0, 3.0, 5.0],
                                                  [5.0, 3.0, 2.0]])
        model.startprob_ = np.array([0.6, 0.4])
        model.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]])
        model.emissionprob_ = np.array([[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]])
        return model

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_after_a_prediction_equals_a_fit_alone(self, implementation):
        """
        A prediction does not set ``n_features``: with ``"e"`` in
        ``init_params``, ``fit`` after a ``score`` starts as many symbols
        as ``X`` shows, as ``fit`` alone does, byte for byte.
        """
        X = np.random.RandomState(10).randint(0, 2, (30, 1))
        expected = self._hand_built(implementation).fit(X)
        model = self._hand_built(implementation)
        model.score(X)
        assert model.n_features is None
        model.fit(X)
        assert model.n_features == expected.n_features == 2
        assert model.monitor_.history == expected.monitor_.history
        for name, value in vars(expected).items():
            if name.endswith("_") and name != "monitor_":
                fitted = np.asarray(vars(model)[name])
                assert fitted.dtype == np.asarray(value).dtype
                assert fitted.tobytes() == np.asarray(value).tobytes()

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_init_incorrect_priors(self, implementation):
        sequences, lengths = self.get_from_one_beal(7, 100, None)

        # Test startprob shape
        with pytest.raises(ValueError):
            model = vhmm.VariationalCategoricalHMM(
                4, n_iter=500, random_state=1984, init_params="te",
                implementation=implementation)
            model.startprob_prior_ = np.full((3,), .25)
            model.startprob_posterior_ = np.full((4,), 7/4)
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        with pytest.raises(ValueError):
            model = vhmm.VariationalCategoricalHMM(
                4, n_iter=500, random_state=1984, init_params="te",
                implementation=implementation)
            model.startprob_prior_ = np.full((4,), .25)
            model.startprob_posterior_ = np.full((3,), 7/4)
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Test transmat shape
        with pytest.raises(ValueError):
            model = vhmm.VariationalCategoricalHMM(
                4, n_iter=500, random_state=1984, init_params="se",
                implementation=implementation)
            model.transmat_prior_ = np.full((3, 3), .25)
            model.transmat_posterior_ = np.full((4, 4), .25)
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        with pytest.raises(ValueError):
            model = vhmm.VariationalCategoricalHMM(
                4, n_iter=500, random_state=1984, init_params="se",
                implementation=implementation)
            model.transmat_prior_ = np.full((4, 4), .25)
            model.transmat_posterior_ = np.full((3, 3), 7/4)
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Test emission shape
        with pytest.raises(ValueError):
            model = vhmm.VariationalCategoricalHMM(
                4, n_iter=500, random_state=1984, init_params="st",
                implementation=implementation)
            model.emissionprob_prior_ = np.full((3, 3), 1/3)
            model.emissionprob_posterior_ = np.asarray([[.3, .4, .3],
                                                        [.8, .1, .1],
                                                        [.2, .2, .6],
                                                        [.2, .6, .2]])
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Test too many n_features
        with pytest.raises(ValueError):
            model = vhmm.VariationalCategoricalHMM(
                4, n_iter=500, random_state=1984, init_params="se",
                implementation=implementation)
            model.emissionprob_prior_ = np.full((4, 4), 7/4)
            model.emissionprob_posterior_ = np.full((4, 4), .25)
            model.n_features = 10
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Too small n_features
        with pytest.raises(ValueError):
            model = vhmm.VariationalCategoricalHMM(
                4, n_iter=500, random_state=1984, init_params="se",
                implementation=implementation)
            model.emissionprob_prior_ = np.full((4, 4), 7/4)
            model.emissionprob_posterior_ = np.full((4, 4), .25)

            model.n_features = 1
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Test that setting the desired prior value works
        model = vhmm.VariationalCategoricalHMM(
            4, n_iter=500, random_state=1984, init_params="ste",
            implementation=implementation,
            startprob_prior=1, transmat_prior=2, emissionprob_prior=3)
        assert_log_likelihood_increasing(model, sequences, lengths, 10)
        assert np.all(model.startprob_prior_ == 1)
        assert np.all(model.transmat_prior_ == 2)
        assert np.all(model.emissionprob_prior_ == 3)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_beal(self, implementation):
        rs = check_random_state(1984)
        m1, m2, m3 = self.get_beal_models()
        sequences = []
        lengths = []
        for i in range(7):
            for m in [m1, m2, m3]:
                sequences.append(m.sample(39, random_state=rs)[0])
                lengths.append(len(sequences[-1]))
        sequences = np.concatenate(sequences)
        model = vhmm.VariationalCategoricalHMM(12, n_iter=500,
                                               implementation=implementation,
                                               tol=1e-6,
                                               random_state=rs,
                                               verbose=False)

        assert_log_likelihood_increasing(model, sequences, lengths, 100)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_and_compare_with_em(self, implementation):
        # Explicitly setting Random State to test that certain
        # model states will become "unused"
        sequences, lengths = self.get_from_one_beal(7, 100, 1984)
        model = vhmm.VariationalCategoricalHMM(
            4, n_iter=500, random_state=1984,
            init_params="e",
            implementation=implementation)
        vi_uniform_startprob_and_transmat(model, lengths)
        model.fit(sequences, lengths)

        # The 1st hidden state will be "unused"
        assert (model.transmat_posterior_[1, :]
                == pytest.approx(.25, rel=1e-3))
        assert (model.emissionprob_posterior_[1, :]
                == pytest.approx(.3333, rel=1e-3))

        # An EM Model should behave the same behavior as a Variational Model,
        # When initialized with the normalized probabilities of the mode of the
        # Variational MOdel.
        em_hmm = hmm.CategoricalHMM(n_components=4, init_params="")
        em_hmm.startprob_ = model.startprob_
        em_hmm.transmat_ = model.transmat_
        em_hmm.emissionprob_ = model.emissionprob_

        compare_variational_and_em_models(model, em_hmm, sequences, lengths)

    @staticmethod
    def _fitted(n_features, implementation):
        """A model fitted on 30 symbols drawn from ``range(n_features)``."""
        X = np.random.RandomState(10).randint(0, n_features, (30, 1))
        model = vhmm.VariationalCategoricalHMM(
            2, n_iter=5, random_state=0, implementation=implementation)
        model.fit(X)
        assert model.n_features == n_features
        return model

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    @pytest.mark.parametrize("method", ["score", "predict"])
    @pytest.mark.parametrize("symbol", [-1, 3])
    def test_prediction_refuses_symbols_outside_range(
            self, symbol, method, implementation):
        """
        A fitted model refuses a negative symbol and the symbol
        ``n_features``, naming ``X``.
        """
        model = self._fitted(3, implementation)
        with pytest.raises(ValueError, match="X must contain symbols"):
            getattr(model, method)(np.array([[0], [symbol], [2]]))

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    @pytest.mark.parametrize("method", ["score", "predict", "decode"])
    @pytest.mark.parametrize("n_features, X", [
        (2, [[True], [False]]),
        (3, [[True], [False], [True]]),
    ], ids=["2-symbols-TF", "3-symbols-TFT"])
    def test_prediction_refuses_boolean_X(
            self, n_features, X, method, implementation):
        """
        A boolean ``X`` is refused, naming ``X``, as ``fit`` refuses it:
        NumPy reads a boolean array as a mask over the symbols.
        """
        model = self._fitted(n_features, implementation)
        with pytest.raises(ValueError, match="X must .*not booleans"):
            getattr(model, method)(np.array(X))

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_length_1_sequences(self, implementation):
        sequences1, lengths1 = self.get_from_one_beal(7, 100, 1984)

        # Include some length 1 sequences
        sequences2, lengths2 = self.get_from_one_beal(1, 1, 1984)
        sequences = np.concatenate([sequences1, sequences2])
        lengths = np.concatenate([lengths1, lengths2])

        model = vhmm.VariationalCategoricalHMM(
            4, n_iter=500, random_state=1984,
            implementation=implementation)
        assert_log_likelihood_increasing(model, sequences, lengths, 10)
