import numbers

import numpy as np
import pytest
from sklearn.utils import check_random_state

from hmmcraft import hmm, vhmm

from . import (
    assert_log_likelihood_increasing,
    compare_variational_and_em_models,
    make_covar_matrix,
    normalized,
    vi_uniform_startprob_and_transmat,
)


def get_mcgrory_titterington():
    m1 = hmm.GaussianHMM(4, init_params="")
    m1.n_features = 4
    m1.startprob_ = np.array([1 / 4.0, 1 / 4.0, 1 / 4.0, 1 / 4.0])
    m1.transmat_ = np.array(
        [
            [0.2, 0.2, 0.3, 0.3],
            [0.3, 0.2, 0.2, 0.3],
            [0.2, 0.3, 0.3, 0.2],
            [0.3, 0.3, 0.2, 0.2],
        ]
    )
    m1.means_ = np.array([[-1.5], [0], [1.5], [3.0]])
    m1.covars_ = np.sqrt([[0.25], [0.25], [0.25], [0.25]])
    return m1


def get_sequences(length, N, model, rs=None):
    sequences = []
    lengths = []
    rs = check_random_state(rs)
    for _i in range(N):
        sequences.append(model.sample(length, random_state=rs)[0])
        lengths.append(len(sequences[-1]))

    sequences = np.concatenate(sequences)
    return sequences, lengths


class _TestGaussian:
    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_random_fit(
        self,
        implementation,
        params="stmc",
        n_features=3,
        n_components=3,
        **kwargs,
    ):
        h = hmm.GaussianHMM(
            n_components,
            self.covariance_type,
            implementation=implementation,
            init_params="",
        )
        rs = check_random_state(1)
        h.startprob_ = normalized(rs.rand(n_components))
        h.transmat_ = normalized(rs.rand(n_components, n_components), axis=1)
        h.means_ = rs.randint(-20, 20, (n_components, n_features))
        h.covars_ = make_covar_matrix(
            self.covariance_type, n_components, n_features, random_state=rs
        )
        lengths = [200] * 5
        X, _state_sequence = h.sample(sum(lengths), random_state=rs)
        # Now learn a model
        model = vhmm.VariationalGaussianHMM(
            n_components,
            n_iter=50,
            tol=1e-9,
            random_state=rs,
            covariance_type=self.covariance_type,
            implementation=implementation,
        )

        assert_log_likelihood_increasing(model, X, lengths, n_iter=10)

    @pytest.mark.parametrize("method", ["score", "predict"])
    @pytest.mark.parametrize("n_columns", [1, 3])
    def test_prediction_refuses_X_of_another_n_features(
        self, n_columns, method
    ):
        """
        A model fitted on two features refuses ``X`` with one column fewer
        or one more, naming ``X``.
        """
        rs = np.random.RandomState(10)
        X = rs.normal(size=(30, 2)) + 3 * rs.randint(0, 2, (30, 1))
        model = vhmm.VariationalGaussianHMM(
            2, self.covariance_type, n_iter=5, random_state=0
        ).fit(X)
        with (
            np.errstate(all="ignore"),
            pytest.raises(ValueError, match="X must have"),
        ):
            getattr(model, method)(rs.normal(size=(10, n_columns)))

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_mcgrory_titterington1d(self, implementation):
        random_state = check_random_state(234234)
        # Setup to assure convergence

        sequences, lengths = get_sequences(
            500, 1, model=get_mcgrory_titterington(), rs=random_state
        )
        model = vhmm.VariationalGaussianHMM(
            5,
            n_iter=1000,
            tol=1e-9,
            random_state=random_state,
            init_params="mc",
            covariance_type=self.covariance_type,
            implementation=implementation,
        )
        vi_uniform_startprob_and_transmat(model, lengths)
        model.fit(sequences, lengths)
        # Perform one check that we are converging to the right answer
        assert sorted(model.means_posterior_.ravel())[3] == pytest.approx(
            self.test_fit_mcgrory_titterington1d_mean
        ), model.means_posterior_

        em_hmm = hmm.GaussianHMM(
            n_components=model.n_components,
            implementation=implementation,
            covariance_type=self.covariance_type,
        )
        em_hmm.startprob_ = model.startprob_
        em_hmm.transmat_ = model.transmat_
        em_hmm.means_ = model.means_posterior_
        em_hmm.covars_ = model._covars_

        compare_variational_and_em_models(model, em_hmm, sequences, lengths)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_common_initialization(self, implementation):
        random_state = check_random_state(234234)
        sequences, lengths = get_sequences(
            50, 10, model=get_mcgrory_titterington(), rs=random_state
        )

        with pytest.raises(ValueError):
            model = vhmm.VariationalGaussianHMM(
                4,
                n_iter=500,
                tol=1e-9,
                random_state=random_state,
                covariance_type="incorrect",
                implementation=implementation,
            )
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        with pytest.raises(ValueError):
            model = vhmm.VariationalGaussianHMM(
                4,
                n_iter=500,
                tol=1e-9,
                random_state=random_state,
                covariance_type="incorrect",
                init_params="",
                implementation=implementation,
            )
            model.startprob_ = np.asarray([0.25, 0.25, 0.25, 0.25])
            model.score(sequences, lengths)

        # Manually setup means - should converge
        model = vhmm.VariationalGaussianHMM(
            4,
            n_iter=500,
            tol=1e-9,
            random_state=random_state,
            init_params="stc",
            covariance_type=self.covariance_type,
            implementation=implementation,
        )
        model.means_prior_ = [[1], [1], [1], [1]]
        model.means_posterior_ = [[2], [1], [3], [4]]
        model.beta_prior_ = [1, 1, 1, 1]
        model.beta_posterior_ = [1, 1, 1, 1]
        assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Means have wrong shape
        with pytest.raises(ValueError):
            model = vhmm.VariationalGaussianHMM(
                4,
                n_iter=500,
                tol=1e-9,
                random_state=random_state,
                init_params="stc",
                covariance_type=self.covariance_type,
                implementation=implementation,
            )
            model.means_prior_ = [[1], [1], [1]]
            model.means_posterior_ = [[1], [1], [1], [1]]
            model.beta_prior_ = [1, 1, 1, 1]
            model.beta_posterior_ = [1, 1, 1, 1]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        with pytest.raises(ValueError):
            model = vhmm.VariationalGaussianHMM(
                4,
                n_iter=500,
                tol=1e-9,
                random_state=random_state,
                init_params="stc",
                covariance_type=self.covariance_type,
                implementation=implementation,
            )
            model.means_prior_ = [[1], [1], [1], [1]]
            model.means_posterior_ = [[1], [1], [1]]
            model.beta_prior_ = [1, 1, 1, 1]
            model.beta_posterior_ = [1, 1, 1, 1]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # beta's have wrong shape
        with pytest.raises(ValueError):
            model = vhmm.VariationalGaussianHMM(
                4,
                n_iter=500,
                tol=1e-9,
                random_state=random_state,
                init_params="stc",
                covariance_type=self.covariance_type,
                implementation=implementation,
            )
            model.means_prior_ = [[1], [1], [1], [1]]
            model.means_posterior_ = [[2], [1], [3], [4]]
            model.beta_prior_ = [1, 1, 1]
            model.beta_posterior_ = [1, 1, 1, 1]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        with pytest.raises(ValueError):
            model = vhmm.VariationalGaussianHMM(
                4,
                n_iter=500,
                tol=1e-9,
                random_state=random_state,
                init_params="stc",
                covariance_type=self.covariance_type,
                implementation=implementation,
            )
            model.means_prior_ = [[1], [1], [1], [1]]
            model.means_posterior_ = [[2], [1], [3], [4]]
            model.beta_prior_ = [1, 1, 1, 1]
            model.beta_posterior_ = [1, 1, 1]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)


class TestFull(_TestGaussian):
    covariance_type = "full"
    test_fit_mcgrory_titterington1d_mean = 1.41058519

    def new_for_init(self, implementation):
        model = vhmm.VariationalGaussianHMM(
            4,
            n_iter=500,
            tol=1e-9,
            init_params="stm",
            covariance_type=self.covariance_type,
            implementation=implementation,
        )
        model.dof_prior_ = [1, 1, 1, 1]
        model.dof_posterior_ = [1, 1, 1, 1]
        model.scale_prior_ = [[[2.0]], [[2.0]], [[2.0]], [[2]]]
        model.scale_posterior_ = [[[2.0]], [[2.0]], [[2.0]], [[2.0]]]
        return model

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_initialization(self, implementation):
        random_state = check_random_state(234234)
        sequences, lengths = get_sequences(
            50, 10, model=get_mcgrory_titterington()
        )

        # dof's have wrong shape
        with pytest.raises(ValueError):
            model = self.new_for_init(implementation)
            model.dof_prior_ = [1, 1, 1]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        with pytest.raises(ValueError):
            model = self.new_for_init(implementation)
            model.dof_posterior_ = [1, 1, 1]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # scales's have wrong shape
        with pytest.raises(ValueError):
            model = self.new_for_init(implementation)
            model.scale_prior_ = [[[2.0]], [[2.0]], [[2.0]]]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        with pytest.raises(ValueError):
            model = self.new_for_init(implementation)
            model.scale_posterior_ = [[2.0]], [[2.0]], [[2.0]]  # this is wrong
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Manually setup covariance
        with pytest.raises(ValueError):
            model = vhmm.VariationalGaussianHMM(
                4,
                n_iter=500,
                tol=1e-9,
                init_params="stm",
                covariance_type="incorrect",
                implementation=implementation,
            )
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Set priors correctly via params
        model = vhmm.VariationalGaussianHMM(
            4,
            n_iter=500,
            tol=1e-9,
            random_state=random_state,
            covariance_type=self.covariance_type,
            implementation=implementation,
            means_prior=[[0.0], [0.0], [0.0], [0.0]],
            beta_prior=[2.0, 2.0, 2.0, 2.0],
            dof_prior=[2.0, 2.0, 2.0, 2.0],
            scale_prior=[[[2.0]], [[2.0]], [[2.0]], [[2.0]]],
        )
        assert_log_likelihood_increasing(model, sequences, lengths, 10)
        assert np.all(model.means_prior_ == 0)
        assert np.all(model.beta_prior_ == 2.0)
        assert np.all(model.dof_prior_ == 2.0)
        assert np.all(model.scale_prior_ == 2.0)

        # Manually set everything
        model = vhmm.VariationalGaussianHMM(
            4,
            n_iter=500,
            tol=1e-9,
            random_state=random_state,
            covariance_type=self.covariance_type,
            implementation=implementation,
            init_params="",
        )
        model.means_prior_ = [[0.0], [0.0], [0.0], [0.0]]
        model.means_posterior_ = [[2], [1], [3], [4]]
        model.beta_prior_ = [2.0, 2.0, 2.0, 2.0]
        model.beta_posterior_ = [1, 1, 1, 1]
        model.dof_prior_ = [2.0, 2.0, 2.0, 2.0]
        model.dof_posterior_ = [1, 1, 1, 1]
        model.scale_prior_ = [[[2.0]], [[2.0]], [[2.0]], [[2.0]]]
        model.scale_posterior_ = [[[2.0]], [[2.0]], [[2.0]], [[2.0]]]
        assert_log_likelihood_increasing(model, sequences, lengths, 10)


class TestTied(_TestGaussian):
    test_fit_mcgrory_titterington1d_mean = 1.4774254
    covariance_type = "tied"

    def new_for_init(self, implementation):
        model = vhmm.VariationalGaussianHMM(
            4,
            n_iter=500,
            tol=1e-9,
            init_params="stm",
            covariance_type=self.covariance_type,
            implementation=implementation,
        )
        model.dof_prior_ = 1
        model.dof_posterior_ = 1
        model.scale_prior_ = [[2]]
        model.scale_posterior_ = [[2]]
        return model

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_initialization(self, implementation):
        random_state = check_random_state(234234)
        sequences, lengths = get_sequences(
            50, 10, model=get_mcgrory_titterington()
        )

        # dof's have wrong shape
        with pytest.raises(ValueError):
            model = self.new_for_init(implementation)
            model.dof_prior_ = [1]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        with pytest.raises(ValueError):
            model = self.new_for_init(implementation)
            model.dof_posterior_ = [1]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # scales's have wrong shape
        with pytest.raises(ValueError):
            model = self.new_for_init(implementation)
            model.scale_prior_ = [[[2]]]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        with pytest.raises(ValueError):
            model = self.new_for_init(implementation)
            model.scale_posterior_ = [
                [[2.0]],
                [[2.0]],
                [[2.0]],
            ]  # this is wrong
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Manually setup covariance
        with pytest.raises(ValueError):
            model = vhmm.VariationalGaussianHMM(
                4,
                n_iter=500,
                tol=1e-9,
                init_params="stm",
                covariance_type="incorrect",
                implementation=implementation,
            )
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Set priors correctly via params
        model = vhmm.VariationalGaussianHMM(
            4,
            n_iter=500,
            tol=1e-9,
            random_state=random_state,
            covariance_type=self.covariance_type,
            implementation=implementation,
            means_prior=[[0.0], [0.0], [0.0], [0.0]],
            beta_prior=[2.0, 2.0, 2.0, 2.0],
            dof_prior=2,
            scale_prior=[[2]],
        )
        assert_log_likelihood_increasing(model, sequences, lengths, 10)
        assert np.all(model.means_prior_ == 0)
        assert np.all(model.beta_prior_ == 2.0)
        assert np.all(model.dof_prior_ == 2.0)
        assert np.all(model.scale_prior_ == 2.0)

        # Manually set everything
        model = vhmm.VariationalGaussianHMM(
            4,
            n_iter=500,
            tol=1e-9,
            random_state=random_state,
            covariance_type=self.covariance_type,
            implementation=implementation,
            init_params="",
        )
        model.means_prior_ = [[0.0], [0.0], [0.0], [0.0]]
        model.means_posterior_ = [[2], [1], [3], [4]]
        model.beta_prior_ = [2.0, 2.0, 2.0, 2.0]
        model.beta_posterior_ = [1, 1, 1, 1]
        model.dof_prior_ = 2
        model.dof_posterior_ = 1
        model.scale_prior_ = [[2]]
        model.scale_posterior_ = [[2]]
        assert_log_likelihood_increasing(model, sequences, lengths, 10)

    @pytest.mark.parametrize("attr", ["dof_prior_", "dof_posterior_"])
    def test_dof_stays_a_number(self, attr):
        """
        The degrees of freedom of a ``tied`` model are numbers: a number is
        kept as the object given, a fit leaves numbers, and a list is
        converted to an array, which ``fit`` refuses as not numeric.
        """
        sequences, lengths = get_sequences(
            20, 2, model=get_mcgrory_titterington(), rs=check_random_state(0)
        )
        model = self.new_for_init("log")
        number = np.int64(3)
        setattr(model, attr, number)
        assert getattr(model, attr) is number
        model.n_iter = 2
        model.fit(sequences, lengths)
        assert isinstance(model.dof_prior_, numbers.Number)
        assert isinstance(model.dof_posterior_, numbers.Number)
        model = self.new_for_init("log")
        setattr(model, attr, [1])
        assert type(getattr(model, attr)) is np.ndarray
        assert getattr(model, attr).dtype == float
        with pytest.raises(ValueError, match=f"{attr} should be numeric"):
            model.fit(sequences, lengths)


class TestSpherical(_TestGaussian):
    test_fit_mcgrory_titterington1d_mean = 1.4105851867634462
    covariance_type = "spherical"

    def new_for_init(self, implementation):
        model = vhmm.VariationalGaussianHMM(
            4,
            n_iter=500,
            tol=1e-9,
            init_params="stm",
            covariance_type=self.covariance_type,
            implementation=implementation,
        )
        model.dof_prior_ = [1, 1, 1, 1]
        model.dof_posterior_ = [1, 1, 1, 1]
        model.scale_prior_ = [2, 2, 2, 2]
        model.scale_posterior_ = [2, 2, 2, 2]
        return model

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_initialization(self, implementation):
        random_state = check_random_state(234234)
        sequences, lengths = get_sequences(
            50, 10, model=get_mcgrory_titterington()
        )

        # dof's have wrong shape
        with pytest.raises(ValueError):
            model = self.new_for_init(implementation)
            model.dof_prior_ = [1, 1, 1]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        with pytest.raises(ValueError):
            model = self.new_for_init(implementation)
            model.dof_posterior_ = [1, 1, 1]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # scales's have wrong shape
        with pytest.raises(ValueError):
            model = self.new_for_init(implementation)
            model.scale_prior_ = [2, 2, 2]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        with pytest.raises(ValueError):
            model = self.new_for_init(implementation)
            model.scale_posterior_ = [2, 2, 2]  # this is wrong
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Manually setup covariance
        with pytest.raises(ValueError):
            model = vhmm.VariationalGaussianHMM(
                4,
                n_iter=500,
                tol=1e-9,
                init_params="stm",
                covariance_type="incorrect",
                implementation=implementation,
            )
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Set priors correctly via params
        model = vhmm.VariationalGaussianHMM(
            4,
            n_iter=500,
            tol=1e-9,
            random_state=random_state,
            covariance_type=self.covariance_type,
            implementation=implementation,
            means_prior=[[0.0], [0.0], [0.0], [0.0]],
            beta_prior=[2.0, 2.0, 2.0, 2.0],
            dof_prior=[2.0, 2.0, 2.0, 2.0],
            scale_prior=[2, 2, 2, 2],
        )
        assert_log_likelihood_increasing(model, sequences, lengths, 10)
        assert np.all(model.means_prior_ == 0)
        assert np.all(model.beta_prior_ == 2.0)
        assert np.all(model.dof_prior_ == 2.0)
        assert np.all(model.scale_prior_ == 2.0)

        # Manually set everything
        model = vhmm.VariationalGaussianHMM(
            4,
            n_iter=500,
            tol=1e-9,
            random_state=random_state,
            covariance_type=self.covariance_type,
            implementation=implementation,
            init_params="",
        )
        model.means_prior_ = [[0.0], [0.0], [0.0], [0.0]]
        model.means_posterior_ = [[2], [1], [3], [4]]
        model.beta_prior_ = [2.0, 2.0, 2.0, 2.0]
        model.beta_posterior_ = [1, 1, 1, 1]
        model.dof_prior_ = [2.0, 2.0, 2.0, 2.0]
        model.dof_posterior_ = [1, 1, 1, 1]
        model.scale_prior_ = [2, 2, 2, 2]
        model.scale_posterior_ = [2, 2, 2, 2]
        assert_log_likelihood_increasing(model, sequences, lengths, 10)


class TestDiagonal(_TestGaussian):
    test_fit_mcgrory_titterington1d_mean = 1.410585186763446
    covariance_type = "diag"

    def new_for_init(self, implementation):
        model = vhmm.VariationalGaussianHMM(
            4,
            n_iter=500,
            tol=1e-9,
            init_params="stm",
            covariance_type=self.covariance_type,
            implementation=implementation,
        )
        model.dof_prior_ = [1, 1, 1, 1]
        model.dof_posterior_ = [1, 1, 1, 1]
        model.scale_prior_ = [[2], [2], [2], [2]]
        model.scale_posterior_ = [[2], [2], [2], [2]]
        return model

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_initialization(self, implementation):
        random_state = check_random_state(234234)
        sequences, lengths = get_sequences(
            50, 10, model=get_mcgrory_titterington()
        )

        # dof's have wrong shape
        with pytest.raises(ValueError):
            model = self.new_for_init(implementation)
            model.dof_prior_ = [1, 1, 1]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        with pytest.raises(ValueError):
            model = self.new_for_init(implementation)
            model.dof_posterior_ = [1, 1, 1]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # scales's have wrong shape
        with pytest.raises(ValueError):
            model = self.new_for_init(implementation)
            model.scale_prior_ = [[2], [2], [2]]
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        with pytest.raises(ValueError):
            model = vhmm.VariationalGaussianHMM(
                4,
                n_iter=500,
                tol=1e-9,
                init_params="stm",
                covariance_type=self.covariance_type,
                implementation=implementation,
            )
            model.dof_prior_ = [1, 1, 1, 1]
            model.dof_posterior_ = [1, 1, 1, 1]
            model.scale_prior_ = [[2], [2], [2], [2]]
            model.scale_posterior_ = [[2, 2, 2]]  # this is wrong
            assert_log_likelihood_increasing(model, sequences, lengths, 10)

        # Set priors correctly via params
        model = vhmm.VariationalGaussianHMM(
            4,
            n_iter=500,
            tol=1e-9,
            random_state=random_state,
            covariance_type=self.covariance_type,
            implementation=implementation,
            means_prior=[[0.0], [0.0], [0.0], [0.0]],
            beta_prior=[2.0, 2.0, 2.0, 2.0],
            dof_prior=[2.0, 2.0, 2.0, 2.0],
            scale_prior=[[2], [2], [2], [2]],
        )
        assert_log_likelihood_increasing(model, sequences, lengths, 10)
        assert np.all(model.means_prior_ == 0)
        assert np.all(model.beta_prior_ == 2.0)
        assert np.all(model.dof_prior_ == 2.0)
        assert np.all(model.scale_prior_ == 2.0)

        # Manually set everything
        model = vhmm.VariationalGaussianHMM(
            4,
            n_iter=500,
            tol=1e-9,
            random_state=random_state,
            covariance_type=self.covariance_type,
            implementation=implementation,
            init_params="",
        )
        model.means_prior_ = [[0.0], [0.0], [0.0], [0.0]]
        model.means_posterior_ = [[2], [1], [3], [4]]
        model.beta_prior_ = [2.0, 2.0, 2.0, 2.0]
        model.beta_posterior_ = [1, 1, 1, 1]
        model.dof_prior_ = [2.0, 2.0, 2.0, 2.0]
        model.dof_posterior_ = [1, 1, 1, 1]
        model.scale_prior_ = [[2], [2], [2], [2]]
        model.scale_posterior_ = [[2], [2], [2], [2]]
        assert_log_likelihood_increasing(model, sequences, lengths, 10)


def test_covars_before_n_features():
    """
    ``covars_`` of a ``diag`` model is read before ``n_features`` is known:
    the full matrices need no feature count.
    """
    model = vhmm.VariationalGaussianHMM(2, "diag")
    model.covars_ = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert not hasattr(model, "n_features")
    expected = np.array([np.diag([1.0, 2.0]), np.diag([3.0, 4.0])])
    assert model.covars_.tobytes() == expected.tobytes()


def _copy_fitted(source, model):
    """Give *model* every fitted attribute of *source* but the monitor."""
    for name, value in vars(source).items():
        if name.endswith("_") and name not in ("monitor_", "state_occupancy_"):
            setattr(model, name, value)


def _width_refusal(covariance_type="full"):
    """
    ``(model, wider, X4)``: *model* has the parameters of a fit on three
    features and no ``n_features``; *wider* is fitted on ``X4``, which has
    four.
    """
    rs = np.random.RandomState(0)
    X3 = rs.normal(size=(50, 3))
    X4 = rs.normal(size=(50, 4))
    narrow = vhmm.VariationalGaussianHMM(
        2, covariance_type, random_state=0, n_iter=1
    ).fit(X3)
    wider = vhmm.VariationalGaussianHMM(
        2, covariance_type, random_state=0, n_iter=1
    ).fit(X4)
    model = vhmm.VariationalGaussianHMM(
        2, covariance_type, init_params="", random_state=0
    )
    _copy_fitted(narrow, model)
    return model, wider, X4


def test_score_after_a_width_refusal_and_wider_posteriors():
    """
    After a ``fit`` refused on ``X`` of another width, the parameters of
    that width set by hand score ``X`` as the model they come from.
    """
    model, wider, X4 = _width_refusal()
    with pytest.raises(ValueError):
        model.fit(X4)
    _copy_fitted(wider, model)
    assert model.score(X4) == wider.score(X4)


def test_spherical_covars_after_a_refused_fit_is_as_before():
    """
    A ``fit`` refused on ``X`` of another width leaves the number of
    features as it was: ``covars_`` of a ``spherical`` model keeps the
    shape and values it had before the call.
    """
    model, _, X4 = _width_refusal("spherical")
    before = model.covars_
    assert before.shape == (2, 3, 3)
    with pytest.raises(ValueError):
        model.fit(X4)
    after = model.covars_
    assert after.shape == before.shape
    assert after.tobytes() == before.tobytes()


@pytest.mark.parametrize("means", [None, np.zeros(2)], ids=["unset", "1-D"])
def test_spherical_covars_name_means_posterior(means):
    """
    ``covars_`` of a ``spherical`` model with no feature count names
    ``means_posterior_``, the attribute to set, when that is unset or of
    the wrong shape.
    """
    model = vhmm.VariationalGaussianHMM(2, "spherical")
    model.covars_ = np.array([1.0, 2.0])
    if means is not None:
        model.means_posterior_ = means
    with pytest.raises(ValueError, match="means_posterior_"):
        _ = model.covars_


@pytest.mark.parametrize(
    "covariance_type", ["full", "diag", "spherical", "tied"]
)
def test_list_start_and_transition_posteriors_fit(covariance_type):
    """
    A ``startprob_posterior_`` and a ``transmat_posterior_`` given as lists
    are converted when assigned, so a refit from them equals the refit from
    the arrays.
    """
    X = np.random.RandomState(10).normal(size=(100, 2))
    fitted = vhmm.VariationalGaussianHMM(
        2, covariance_type, random_state=0, n_iter=5
    ).fit(X)
    from_arrays = vhmm.VariationalGaussianHMM(
        2, covariance_type, init_params="", random_state=0, n_iter=5
    )
    _copy_fitted(fitted, from_arrays)
    from_lists = vhmm.VariationalGaussianHMM(
        2, covariance_type, init_params="", random_state=0, n_iter=5
    )
    _copy_fitted(fitted, from_lists)
    from_lists.startprob_posterior_ = fitted.startprob_posterior_.tolist()
    from_lists.transmat_posterior_ = fitted.transmat_posterior_.tolist()
    from_arrays.fit(X)
    from_lists.fit(X)
    assert from_lists.monitor_.history == from_arrays.monitor_.history
    assert (
        from_lists.means_posterior_.tobytes()
        == from_arrays.means_posterior_.tobytes()
    )
