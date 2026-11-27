import itertools
import logging
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less
from scipy import linalg
from sklearn.utils import check_random_state

from .. import hmm
from ..hmm import GMMHMM
from . import assert_log_likelihood_increasing, normalized
from .test_gmm_hmm import create_random_gmm


def sample_from_parallelepiped(low, high, n_samples, random_state):
    (n_features,) = low.shape
    X = np.zeros((n_samples, n_features))
    for i in range(n_features):
        X[:, i] = random_state.uniform(low[i], high[i], n_samples)
    return X


def prep_params(
    n_comps, n_mix, n_features, covar_type, low, high, random_state
):
    # the idea is to generate ``n_comps`` bounding boxes and then
    # generate ``n_mix`` mixture means in each of them

    dim_lims = np.zeros((n_comps + 1, n_features))

    # this generates a sequence of coordinates, which are then used as
    # vertices of bounding boxes for mixtures
    dim_lims[1:] = np.cumsum(
        random_state.uniform(low, high, (n_comps, n_features)), axis=0
    )

    means = np.zeros((n_comps, n_mix, n_features))
    for i, (left, right) in enumerate(itertools.pairwise(dim_lims)):
        means[i] = sample_from_parallelepiped(left, right, n_mix, random_state)

    startprob = np.zeros(n_comps)
    startprob[0] = 1

    transmat = normalized(
        random_state.uniform(size=(n_comps, n_comps)), axis=1
    )

    if covar_type == "spherical":
        covs = random_state.uniform(0.1, 5, size=(n_comps, n_mix))
    elif covar_type == "diag":
        covs = random_state.uniform(0.1, 5, size=(n_comps, n_mix, n_features))
    elif covar_type == "tied":
        covs = np.zeros((n_comps, n_features, n_features))
        for i in range(n_comps):
            low = random_state.uniform(-2, 2, (n_features, n_features))
            covs[i] = low.T @ low
    elif covar_type == "full":
        covs = np.zeros((n_comps, n_mix, n_features, n_features))
        for i in range(n_comps):
            for j in range(n_mix):
                low = random_state.uniform(
                    -2, 2, size=(n_features, n_features)
                )
                covs[i, j] = low.T @ low

    weights = normalized(random_state.uniform(size=(n_comps, n_mix)), axis=1)

    return covs, means, startprob, transmat, weights


class GMMHMMTestMixin:
    n_components = 3
    n_mix = 2
    n_features = 2
    low, high = 10, 15

    def new_hmm(self, implementation):
        prng = np.random.RandomState(14)
        covars, means, startprob, transmat, weights = prep_params(
            self.n_components,
            self.n_mix,
            self.n_features,
            self.covariance_type,
            self.low,
            self.high,
            prng,
        )
        h = GMMHMM(
            n_components=self.n_components,
            n_mix=self.n_mix,
            covariance_type=self.covariance_type,
            random_state=prng,
            implementation=implementation,
        )
        h.startprob_ = startprob
        h.transmat_ = transmat
        h.weights_ = weights
        h.means_ = means
        h.covars_ = covars
        return h

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_check_bad_covariance_type(self, implementation):
        h = self.new_hmm(implementation)
        with pytest.raises(ValueError):
            h.covariance_type = "bad_covariance_type"
            h._check()

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_check_good_covariance_type(self, implementation):
        h = self.new_hmm(implementation)
        h._check()  # should not raise any errors

    @pytest.mark.parametrize(
        "means",
        [
            lambda m: m[:, 0],
            lambda m: m[:, 0, 0],
        ],
        ids=["2-D", "1-D"],
    )
    def test_means_of_wrong_ndim_raise(self, means):
        """
        Without ``n_features``, a ``means_`` that is not 3-D is refused with
        a message naming it, not read for the number of features.
        """
        h = self.new_hmm("log")
        h.means_ = means(h.means_)
        assert not hasattr(h, "n_features")
        with pytest.raises(ValueError, match="means_ must have shape"):
            h._check()

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    @pytest.mark.parametrize("name", ["means_", "covars_"])
    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_nonfinite_parameter_raises(self, name, implementation, bad):
        """
        A NaN or an infinity in ``means_`` or ``covars_`` is refused, naming
        it, also where hmmlearn 0.3.3 scored an infinite ``spherical`` or
        ``diag`` entry as a component that emits nothing.
        """
        h = self.new_hmm(implementation)
        X, _ = h.sample(10, random_state=0)
        value = getattr(h, name).astype(float)
        value.flat[0] = bad
        setattr(h, name, value)
        # Without the check the NaN arithmetic would warn; ignoring it lets
        # a missing refusal show as one.
        with np.errstate(all="ignore"), pytest.raises(ValueError, match=name):
            h.score(X)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_negative_weights_raise(self, implementation):
        """
        ``weights_`` rows that sum to 1 with a negative entry are refused,
        naming it.
        """
        h = self.new_hmm(implementation)
        X, _ = h.sample(10, random_state=0)
        h.weights_[0] = [1.5, -0.5]
        with (
            np.errstate(all="ignore"),
            pytest.raises(ValueError, match="weights_"),
        ):
            h.score(X)

    @pytest.mark.parametrize("method", ["score", "predict"])
    @pytest.mark.parametrize("n_columns", [1, 3])
    def test_prediction_refuses_X_of_another_n_features(
        self, n_columns, method
    ):
        """
        ``X`` with one column fewer or one more than the model's features is
        refused, naming ``X``.
        """
        h = self.new_hmm("log")
        X = np.random.RandomState(10).normal(size=(10, n_columns))
        with (
            np.errstate(all="ignore"),
            pytest.raises(ValueError, match="X must have"),
        ):
            getattr(h, method)(X)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_sample(self, implementation):
        n_samples = 1000
        h = self.new_hmm(implementation)
        X, states = h.sample(n_samples)
        assert X.shape == (n_samples, self.n_features)
        assert len(states) == n_samples

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_init(self, implementation):
        n_samples = 1000
        h = self.new_hmm(implementation)
        X, _states = h.sample(n_samples)
        h._init(X, [n_samples])
        h._check()  # should not raise any errors

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_score_samples_and_decode(self, implementation):
        n_samples = 1000
        h = self.new_hmm(implementation)
        X, states = h.sample(n_samples)

        _ll, posteriors = h.score_samples(X)
        assert_allclose(np.sum(posteriors, axis=1), np.ones(n_samples))

        _viterbi_ll, decoded_states = h.decode(X)
        assert_allclose(states, decoded_states)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit(self, implementation):
        n_iter = 5
        n_samples = 1000
        lengths = None
        h = self.new_hmm(implementation)
        X, _state_sequence = h.sample(n_samples)

        # Mess up the parameters and see if we can re-learn them.
        covs0, means0, priors0, trans0, weights0 = prep_params(
            self.n_components,
            self.n_mix,
            self.n_features,
            self.covariance_type,
            self.low,
            self.high,
            np.random.RandomState(15),
        )
        h.covars_ = covs0 * 100
        h.means_ = means0
        h.startprob_ = priors0
        h.transmat_ = trans0
        h.weights_ = weights0
        assert_log_likelihood_increasing(h, X, lengths, n_iter)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_sparse_data(self, implementation):
        n_samples = 1000
        h = self.new_hmm(implementation)
        h.means_ *= 1000  # this will put gaussians very far apart
        X, _states = h.sample(n_samples)

        # this should not raise
        # "ValueError: array must not contain infs or NaNs"
        h._init(X, [1000])
        h.fit(X)

    @pytest.mark.xfail
    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_fit_zero_variance(self, implementation):
        # Example from issue #2 on GitHub.
        # this data has singular covariance matrix
        X = np.asarray(
            [
                [7.15000000e02, 5.8500000e02, 0.00000000e00, 0.00000000e00],
                [7.15000000e02, 5.2000000e02, 1.04705811e00, -6.03696289e01],
                [7.15000000e02, 4.5500000e02, 7.20886230e-01, -5.27055664e01],
                [7.15000000e02, 3.9000000e02, -4.57946777e-01, -7.80605469e01],
                [7.15000000e02, 3.2500000e02, -6.43127441e00, -5.59954834e01],
                [7.15000000e02, 2.6000000e02, -2.90063477e00, -7.80220947e01],
                [7.15000000e02, 1.9500000e02, 8.45532227e00, -7.03294373e01],
                [7.15000000e02, 1.3000000e02, 4.09387207e00, -5.83621216e01],
                [7.15000000e02, 6.5000000e01, -1.21667480e00, -4.48131409e01],
            ]
        )

        h = self.new_hmm(implementation)
        h.fit(X)

    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_criterion(self, implementation):
        random_state = check_random_state(2013)
        m1 = self.new_hmm(implementation)
        # Spread the means out to make this easier
        m1.means_ *= 10

        X, _ = m1.sample(4000, random_state=random_state)
        aic = []
        bic = []
        ns = [2, 3, 4, 5]
        for n in ns:
            h = GMMHMM(
                n,
                n_mix=2,
                covariance_type=self.covariance_type,
                random_state=random_state,
                implementation=implementation,
            )
            h.fit(X)
            aic.append(h.aic(X))
            bic.append(h.bic(X))

        assert np.all(aic) > 0
        assert np.all(bic) > 0
        # AIC / BIC pick the right model occasionally
        # assert ns[np.argmin(aic)] == self.n_components
        # assert ns[np.argmin(bic)] == self.n_components


class TestGMMHMMWithSphericalCovars(GMMHMMTestMixin):
    covariance_type = "spherical"


class TestGMMHMMWithDiagCovars(GMMHMMTestMixin):
    covariance_type = "diag"


class TestGMMHMMWithTiedCovars(GMMHMMTestMixin):
    covariance_type = "tied"


class TestGMMHMMWithFullCovars(GMMHMMTestMixin):
    covariance_type = "full"


class TestGMMHMM_KmeansInit:
    @pytest.mark.parametrize("implementation", ["scaling", "log"])
    def test_kmeans(self, implementation):
        # Generate two isolated cluster.
        # The second cluster has no. of points less than n_mix.
        np.random.seed(0)
        data1 = np.random.uniform(low=0, high=1, size=(100, 2))
        data2 = np.random.uniform(low=5, high=6, size=(5, 2))
        data = np.r_[data1, data2]
        model = GMMHMM(
            n_components=2, n_mix=10, n_iter=5, implementation=implementation
        )
        model.fit(data)  # _init() should not fail here
        # test whether the means are bounded by the data lower- and upperbounds
        assert_array_less(0, model.means_)
        assert_array_less(model.means_, 6)


class TestGMMHMM_MultiSequence:
    @pytest.mark.parametrize("covtype", ["diag", "spherical", "tied", "full"])
    def test_chunked(sellf, covtype, init_params="mcw"):
        np.random.seed(0)
        gmm = create_random_gmm(3, 2, covariance_type=covtype, prng=0)
        gmm.covariances_ = gmm.covars_
        data = gmm.sample(n_samples=1000)[0]

        model1 = GMMHMM(
            n_components=3,
            n_mix=2,
            covariance_type=covtype,
            random_state=1,
            init_params=init_params,
        )
        model2 = GMMHMM(
            n_components=3,
            n_mix=2,
            covariance_type=covtype,
            random_state=1,
            init_params=init_params,
        )
        # don't use random parameters for testing
        init = 1.0 / model1.n_components
        for model in (model1, model2):
            model.startprob_ = np.full(model.n_components, init)
            model.transmat_ = np.full(
                (model.n_components, model.n_components), init
            )

        model1.fit(data)
        model2.fit(data, lengths=[200] * 5)

        assert_allclose(model1.means_, model2.means_, rtol=0, atol=1e-2)
        assert_allclose(model1.covars_, model2.covars_, rtol=0, atol=1e-3)
        assert_allclose(model1.weights_, model2.weights_, rtol=0, atol=1e-3)
        assert_allclose(model1.transmat_, model2.transmat_, rtol=0, atol=1e-2)


_test_log = logging.getLogger(__name__)
EPS = np.finfo(np.float64).eps
#: The stack decides a matrix of 3 features when its smallest eigenvalue is
#: above this multiple of its largest.
MARGIN_3 = 1024 * 3 * EPS


def _stack(covariance_type, replace, n_features=3):
    """
    Identity covariances of 2 states and 2 mixtures in ``covariance_type``'s
    layout, with ``replace``'s matrices at their positions in the flattened
    stack.
    """
    shape = {"tied": (2,), "full": (2, 2)}[covariance_type]
    covars = np.broadcast_to(
        np.eye(n_features), shape + (n_features, n_features)
    ).copy()
    flat = covars.reshape(-1, n_features, n_features)
    for position, matrix in replace.items():
        flat[position] = matrix
    return covars


def _gmm_with(covariance_type, covars):
    h = GMMHMM(2, n_mix=2, covariance_type=covariance_type, init_params="")
    h.startprob_ = np.array([0.5, 0.5])
    h.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]])
    h.weights_ = np.array([[0.5, 0.5], [0.3, 0.7]])
    h.means_ = np.zeros((2, 2, covars.shape[-1]))
    h.covars_ = covars
    return h


def _matrix_names(covariance_type):
    if covariance_type == "tied":
        return ["state #0", "state #1"]
    return [f"state #{i}, mixture #{j}" for i in range(2) for j in range(2)]


def _check_matrix_by_matrix(covariance_type, covars):
    """
    The covariance tests ``GMMHMM._check`` makes one matrix at a time: the
    same calls, in the same order, with the same errors and log messages.
    """
    n_features = covars.shape[-1]
    for name, covar in zip(
        _matrix_names(covariance_type),
        covars.reshape(-1, n_features, n_features),
        strict=False,
    ):
        if not np.allclose(covar, covar.T):
            raise ValueError(f"Covariance of {name} is not symmetric")
        min_eigvalsh = linalg.eigvalsh(covar).min()
        if min_eigvalsh < 0:
            raise ValueError(f"Covariance of {name} is not positive definite")
        if min_eigvalsh == 0:
            _test_log.warning("Covariance of %s has a null eigenvalue.", name)


def _outcome(call, caplog, **errstate):
    """The error, the warnings and the log messages of ``call()``."""
    caplog.clear()
    raised = None
    with (
        caplog.at_level("WARNING", logger="hmmcraft"),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        try:
            with np.errstate(**errstate):
                call()
        except (ValueError, FloatingPointError) as exc:
            raised = type(exc), str(exc)
    return (
        raised,
        [str(w.message) for w in caught],
        [r.getMessage() for r in caplog.records],
    )


def _same_as_matrix_by_matrix(covariance_type, covars, caplog, **errstate):
    h = _gmm_with(covariance_type, covars)
    got = _outcome(h._check, caplog, **errstate)
    expected = _outcome(
        lambda: _check_matrix_by_matrix(covariance_type, covars),
        caplog,
        **errstate,
    )
    return got, expected


def _fitted_gmm(covariance_type):
    rs = np.random.RandomState(7)
    X = rs.normal(size=(40, 2)) + 3 * rs.randint(0, 2, (40, 1))
    return GMMHMM(
        2, 2, covariance_type=covariance_type, n_iter=4, random_state=0
    ).fit(X)


@pytest.mark.parametrize("covariance_type", ["tied", "full"])
def test_check_of_fitted_covariances_does_not_call_scipy(
    monkeypatch, covariance_type
):
    """
    The check of a fitted model tests its float64 covariances on the whole
    stack, without a call to ``scipy.linalg.eigvalsh`` per matrix.
    """
    h = _fitted_gmm(covariance_type)
    calls = []
    eigvalsh = linalg.eigvalsh

    def spy(*args, **kwargs):
        calls.append(args)
        return eigvalsh(*args, **kwargs)

    monkeypatch.setattr(linalg, "eigvalsh", spy)
    h._check()
    assert calls == []


def _rank_2(shift):
    """A seeded rank-2 matrix of 3 features, plus ``shift`` times its
    largest eigenvalue on the diagonal."""
    b = np.random.default_rng(1).normal(size=(3, 2))
    a = b @ b.T
    a = (a + a.T) / 2
    return a + shift * np.abs(np.linalg.eigvalsh(a)).max() * np.eye(3)


# The expected outcome, written out where every eigenvalue routine is exact
# (diagonal matrices, the symmetry test); None: today's two calls decide.
COVARIANCE_CASES = {
    "null-diagonal": (np.diag([1.0, 1.0, 0.0]), "null"),
    "zeros": (np.zeros((3, 3)), "null"),
    "negative-diagonal": (np.diag([1.0, 1.0, -1.0]), "negative"),
    "half-the-margin": (np.diag([1.0, 1.0, MARGIN_3 / 2]), "pass"),
    "twice-the-margin": (np.diag([1.0, 1.0, 2 * MARGIN_3]), "pass"),
    "identity": (np.eye(3), "pass"),
    "identity-1e-320": (np.eye(3) * 1e-320, "pass"),
    "identity-1e300": (np.eye(3) * 1e300, "pass"),
    "asymmetric-within-tolerance": (
        2 * np.eye(3) + np.triu(np.full((3, 3), 1e-9), 1),
        "pass",
    ),
    "asymmetric-beyond-tolerance": (
        np.eye(3) + np.triu(np.full((3, 3), 1e-3), 1),
        "asymmetric",
    ),
    "rank-2": (_rank_2(0), None),
    "rank-2-minus-4-n-eps": (_rank_2(-4 * 3 * EPS), None),
    "rank-2-plus-4-n-eps": (_rank_2(4 * 3 * EPS), None),
    "rank-2-minus-1e-13": (_rank_2(-1e-13), None),
    "rank-2-plus-1e-13": (_rank_2(1e-13), None),
    "rank-2-plus-1e-9": (_rank_2(1e-9), None),
}


@pytest.mark.parametrize("covariance_type", ["tied", "full"])
@pytest.mark.parametrize("position", [0, -1], ids=["first", "last"])
@pytest.mark.parametrize("case", COVARIANCE_CASES)
def test_covariance_check_at_and_near_a_zero_eigenvalue(
    caplog, covariance_type, position, case
):
    """
    A covariance at, just below, just above or well above a zero eigenvalue,
    or not symmetric, among identities, is refused, warned about or passed
    as one matrix at a time decides, naming the matrix.
    """
    matrix, kind = COVARIANCE_CASES[case]
    covars = _stack(covariance_type, {position: matrix})
    got, expected = _same_as_matrix_by_matrix(covariance_type, covars, caplog)
    assert got == expected
    if kind is not None:
        name = _matrix_names(covariance_type)[position]
        refusal = {
            "negative": "not positive definite",
            "asymmetric": "not symmetric",
        }.get(kind)
        raised, _, logged = expected
        assert raised == (
            None
            if refusal is None
            else (ValueError, f"Covariance of {name} is {refusal}")
        )
        assert logged == (
            [f"Covariance of {name} has a null eigenvalue."]
            if kind == "null"
            else []
        )


@pytest.mark.parametrize("covariance_type", ["tied", "full"])
def test_covariance_check_warns_before_it_refuses(caplog, covariance_type):
    """
    A null eigenvalue in the first matrix and a negative one in the last:
    the first is warned about, then the last is refused.
    """
    covars = _stack(
        covariance_type,
        {0: np.diag([1.0, 1.0, 0.0]), -1: np.diag([1.0, 1.0, -1.0])},
    )
    names = _matrix_names(covariance_type)
    first, last = names[0], names[-1]
    got = _outcome(_gmm_with(covariance_type, covars)._check, caplog)
    assert got == (
        (ValueError, f"Covariance of {last} is not positive definite"),
        [],
        [f"Covariance of {first} has a null eigenvalue."],
    )


@pytest.mark.parametrize("covariance_type", ["tied", "full"])
@pytest.mark.parametrize(
    "factor, decided_on_the_stack",
    [(2, True), (0.5, False)],
    ids=["twice", "half"],
)
def test_covariance_margin(
    monkeypatch, covariance_type, factor, decided_on_the_stack
):
    """
    A smallest eigenvalue above ``1024 * n_features * eps`` times the
    largest is decided on the stack; below it, each matrix gets its own
    ``scipy.linalg.eigvalsh`` call.
    """
    covars = _stack(
        covariance_type, {-1: np.diag([1.0, 1.0, factor * MARGIN_3])}
    )
    calls = []
    eigvalsh = linalg.eigvalsh

    def spy(*args, **kwargs):
        calls.append(args)
        return eigvalsh(*args, **kwargs)

    monkeypatch.setattr(linalg, "eigvalsh", spy)
    _gmm_with(covariance_type, covars)._check()
    assert len(calls) == (
        0 if decided_on_the_stack else len(_matrix_names(covariance_type))
    )


def _singular_integer_matrices():
    """
    Seeded exactly singular matrices ``B @ B.T`` of 2 to 4 features, ``B``
    of small integers and one column fewer, times scales from 1e-300 to
    1e300.
    """
    scales = [
        1.0,
        2.0**-40,
        2.0**40,
        1e-150,
        1e150,
        1e-300,
        1e300,
        3.7e-7,
        9.1e11,
    ]
    rng = np.random.default_rng(0)
    for n_features in (2, 3, 4):
        for i in range(40):
            b = rng.integers(-5, 6, size=(n_features, n_features - 1))
            yield (b @ b.T).astype(float) * scales[i % len(scales)]


@pytest.mark.parametrize("covariance_type", ["tied", "full"])
def test_singular_covariances_are_checked_one_matrix_at_a_time(
    caplog, covariance_type
):
    """
    Singular covariances, whose smallest eigenvalue two routines may round
    to opposite signs, get the outcome one matrix at a time gives.
    """
    differ = []
    for matrix in _singular_integer_matrices():
        covars = _stack(covariance_type, {-1: matrix}, len(matrix))
        got, expected = _same_as_matrix_by_matrix(
            covariance_type, covars, caplog
        )
        if got != expected:
            differ.append((matrix, got, expected))
    assert differ == []


def _with_subnormal_pair():
    a = np.eye(3)
    a[0, 1] = a[1, 0] = 1e-310
    return a


def _asymmetric_overflowing():
    a = np.eye(3) * 1e308
    a[0, 1], a[1, 0] = 1e308, -1e308
    return a


# A matrix whose symmetry test underflows or overflows, after one that is
# refused or warned about, or after an identity.
FLAG_CASES = {
    "negative-then-subnormal-identity": (
        np.diag([1.0, 1.0, -1.0]),
        np.eye(3) * 1e-320,
    ),
    "null-then-subnormal-identity": (
        np.diag([1.0, 1.0, 0.0]),
        np.eye(3) * 1e-320,
    ),
    "negative-then-subnormal-entry": (
        np.diag([1.0, 1.0, -1.0]),
        _with_subnormal_pair(),
    ),
    "null-then-subnormal-entry": (
        np.diag([1.0, 1.0, 0.0]),
        _with_subnormal_pair(),
    ),
    "identity-then-subnormal-entry": (np.eye(3), _with_subnormal_pair()),
    "negative-then-overflowing": (
        np.diag([1.0, 1.0, -1.0]),
        _asymmetric_overflowing(),
    ),
}


@pytest.mark.parametrize(
    "errstate",
    [
        {
            "divide": "warn",
            "over": "warn",
            "under": "ignore",
            "invalid": "warn",
        },
        {"all": "warn"},
        {"under": "raise"},
        {"over": "raise"},
        {"all": "raise"},
    ],
    ids=[
        "numpy-default",
        "all-warn",
        "under-raise",
        "over-raise",
        "all-raise",
    ],
)
@pytest.mark.parametrize("covariance_type", ["tied", "full"])
@pytest.mark.parametrize("case", FLAG_CASES)
def test_covariance_check_reports_floating_point_errors_in_order(
    caplog, case, covariance_type, errstate
):
    """
    Under the caller's error state, a matrix whose symmetry test underflows
    or overflows reports it where, and only if, one matrix at a time
    reaches it: the same error, warnings and log messages.
    """
    first, last = FLAG_CASES[case]
    covars = _stack(covariance_type, {0: first, -1: last})
    got, expected = _same_as_matrix_by_matrix(
        covariance_type, covars, caplog, **errstate
    )
    assert got == expected


def test_stack_screen_sets_no_floating_point_flag():
    """
    Covariances with entries at the stack test's bounds pass it without a
    floating-point error.
    """
    covars = np.array(
        [
            np.eye(2) * 1e-280,
            np.eye(2) * 2.0**1022,
            [[1.0, 2.0**-1000], [2.0**-1000, 1.0]],
            [[2.0, -(2.0**-1000)], [-(2.0**-1000), 1.0]],
        ]
    ).reshape(2, 2, 2, 2)
    with np.errstate(all="raise"):
        assert hmm._covars_pass_on_stack(covars)
