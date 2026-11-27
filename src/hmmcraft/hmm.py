"""
The :mod:`hmmcraft.hmm` module implements hidden Markov models.
"""

import logging
import numbers

import numpy as np
from scipy import linalg
from sklearn import cluster
from sklearn.utils import check_random_state
from threadpoolctl import threadpool_limits

from . import _emissions, _utils
from .base import _ABSENT, BaseHMM, _block_estep, _ConvertedAttribute
from .stats import _covar_factors
from .utils import normalize

__all__ = [
    "GMMHMM",
    "CategoricalHMM",
    "GaussianHMM",
    "MultinomialHMM",
    "PoissonHMM",
]


_log = logging.getLogger(__name__)
COVARIANCE_TYPES = frozenset(("spherical", "diag", "full", "tied"))


def _covars_floor_value(covars_floor):
    """
    Return ``covars_floor`` as a float, or None; raise ValueError if it is
    neither None nor a finite real number above 0 (a boolean is not one).
    """
    if covars_floor is None:
        return None
    message = (
        "covars_floor must be None or a finite number above 0, "
        f"got {covars_floor!r}"
    )
    if not isinstance(covars_floor, numbers.Real) or isinstance(
        covars_floor, (bool, np.bool_)
    ):
        # ValueError for every invalid value, as the parameter documents.
        raise ValueError(message)  # noqa: TRY004
    try:
        value = float(covars_floor)
    except (TypeError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not (np.isfinite(value) and value > 0):
        raise ValueError(message)
    return value


def _covars_pass_on_stack(covars):  # shape and finiteness already tested
    """
    Return True when every covariance matrix in the float64 stack *covars*
    is symmetric and clearly positive definite; False leaves the decision
    to the tests made one matrix at a time.
    """
    if (
        type(covars) is not np.ndarray
        or covars.dtype != np.float64
        or covars.size == 0
    ):
        return False
    try:
        a = np.abs(covars)
        # Comparisons set no floating-point flag; inside these bounds
        # neither does the arithmetic below.
        if not ((a == 0) | ((a >= 2.0**-1000) & (a <= 2.0**1022))).all():
            return False
        ct = covars.swapaxes(-1, -2)
        # np.allclose(covar, covar.T)'s comparison, with its defaults.
        if not (
            np.abs(covars - ct) <= 1e-08 + 1e-05 * a.swapaxes(-1, -2)
        ).all():
            return False
        w = np.linalg.eigvalsh(covars)  # ascending
    except (np.linalg.LinAlgError, MemoryError):
        return False
    # numpy's and scipy's eigvalsh can disagree on the sign of an eigenvalue
    # near zero, so the stack decides only matrices well clear of it.
    c = 1024 * covars.shape[-1] * np.finfo(np.float64).eps
    lo = w[..., 0]
    return bool(
        (lo > np.finfo(np.float64).tiny / c).all()
        and (lo > c * w[..., -1]).all()
    )


def _floor_covars(covars, covariance_type, min_covar):
    """
    Apply the ``min_covar`` floor to covariances in their compact layout.

    For "spherical" and "diag" every variance is clipped at ``min_covar``.
    For "full" and "tied" every eigenvalue below ``min_covar`` is raised to
    it; a matrix whose eigenvalues are all above the floor is returned
    unchanged.  In both cases the result is the maximiser of the M-step
    objective over covariances whose eigenvalues are at least ``min_covar``.
    """
    if min_covar <= 0:
        return covars
    if covariance_type in ("spherical", "diag"):
        return np.maximum(covars, min_covar)

    def floor_matrix(cv):
        eigvals, eigvecs = linalg.eigh(cv)
        if eigvals.min() >= min_covar:
            return cv
        return (eigvecs * np.maximum(eigvals, min_covar)) @ eigvecs.T

    if covariance_type == "tied":
        return floor_matrix(covars)
    return np.array([floor_matrix(cv) for cv in covars])


@_block_estep
class CategoricalHMM(_emissions.BaseCategoricalHMM, BaseHMM):
    """
    Hidden Markov Model with categorical (discrete) emissions.

    Attributes
    ----------
    n_features : int
        Number of possible symbols emitted by the model (in the samples).

    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    state_occupancy_ : array, shape (n_components, )
        Expected number of samples in each state: the posteriors of the last
        E-step of :meth:`fit`, summed over all samples.

    emissionprob_ : array, shape (n_components, n_features)
        Probability of emitting a given symbol when in each state.

    Examples
    --------
    >>> from hmmcraft.hmm import CategoricalHMM
    >>> CategoricalHMM(n_components=2)
    CategoricalHMM(n_components=2)
    """

    def __init__(
        self,
        n_components=1,
        startprob_prior=1.0,
        transmat_prior=1.0,
        *,
        emissionprob_prior=1.0,
        n_features=None,
        algorithm="viterbi",
        random_state=None,
        n_iter=10,
        tol=1e-2,
        verbose=False,
        params="ste",
        init_params="ste",
        implementation="log",
        n_threads=1,
    ):
        """
        Parameters
        ----------
        n_components : int
            Number of states.

        startprob_prior : array, shape (n_components, ), optional
            Parameters of the Dirichlet prior distribution for
            :attr:`startprob_`.

        transmat_prior : array, shape (n_components, n_components), optional
            Parameters of the Dirichlet prior distribution for each row
            of the transition probabilities :attr:`transmat_`.

        emissionprob_prior : array, shape (n_components, n_features), optional
            Parameters of the Dirichlet prior distribution for
            :attr:`emissionprob_`.

        n_features: int, optional
            The number of categorical symbols in the HMM. If not set, taken
            from ``emissionprob_`` when it is kept (``"e"`` not in
            ``init_params``), otherwise from the data.

        algorithm : {"viterbi", "map"}, optional
            Decoder algorithm.

            - "viterbi": finds the most likely sequence of states, given all
              emissions.
            - "map" (also known as smoothing or forward-backward): finds the
              sequence of the individual most-likely states, given all
              emissions.

        random_state: RandomState or an int seed, optional
            A random number generator instance.

        n_iter : int, optional
            Maximum number of iterations to perform.

        tol : float, optional
            Convergence threshold. EM will stop if the gain in log-likelihood
            is below this value.

        verbose : bool, optional
            Whether per-iteration convergence reports are printed to
            :data:`sys.stderr`.  Convergence can also be diagnosed using the
            :attr:`monitor_` attribute.

        params, init_params : string, optional
            The parameters that get updated during (``params``) or initialized
            before (``init_params``) the training.  Can contain any
            combination of 's' for startprob, 't' for transmat, and 'e' for
            emissionprob.  Defaults to all parameters.

        implementation : string, optional
            Determines if the forward-backward algorithm is implemented with
            logarithms ("log"), or using scaling ("scaling").  The default is
            to use logarithms for backwards compatability.

        n_threads : int, optional
            Number of threads :meth:`fit` uses in each E-step for the
            forward and backward passes and the expected transition
            counts. Each sequence is computed on one thread, as it is
            with one thread, and the results are added in the order of
            the sequences, so the fitted model is the same at any value.
            A sequence is never split: one sequence gains nothing.
            Sequences computed at the same time are held in memory
            together, so memory use grows with ``n_threads`` when the
            sequences are long. It has no effect on a model whose E-step
            runs one sequence at a time, such as a subclass that replaces
            an E-step method, and prediction uses one thread. The
            default, 1, starts no thread. A value that is not an integer
            of at least 1 (a boolean is not one) raises ValueError in
            :meth:`fit`.
        """
        BaseHMM.__init__(
            self,
            n_components,
            startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
            params=params,
            init_params=init_params,
            implementation=implementation,
            n_threads=n_threads,
        )
        self.emissionprob_prior = emissionprob_prior
        self.n_features = n_features

    emissionprob_ = _ConvertedAttribute(np.atleast_2d)

    def _init(self, X, lengths=None):
        super()._init(X, lengths)

        self.random_state = check_random_state(self.random_state)

        if self._needs_init("e", "emissionprob_"):
            self.emissionprob_ = self.random_state.rand(
                self.n_components, self.n_features
            )
            normalize(self.emissionprob_, axis=1)

    def _check(self):
        super()._check()

        n_features = self._get_n_features()
        if self.emissionprob_.shape != (self.n_components, n_features):
            raise ValueError(
                f"emissionprob_ must have shape"
                f"({self.n_components}, {n_features})"
            )
        self._check_sum_1("emissionprob_")

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        if "e" in self.params:
            self.emissionprob_ = np.maximum(
                self.emissionprob_prior - 1 + stats["obs"], 0
            )
            normalize(self.emissionprob_, axis=1)


@_block_estep
class GaussianHMM(_emissions.BaseGaussianHMM, BaseHMM):
    """
    Hidden Markov Model with Gaussian emissions.

    Attributes
    ----------
    n_features : int
        Dimensionality of the Gaussian emissions.

    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    state_occupancy_ : array, shape (n_components, )
        Expected number of samples in each state: the posteriors of the last
        E-step of :meth:`fit`, summed over all samples.

    means_ : array, shape (n_components, n_features)
        Mean parameters for each state.

    covars_ : array
        Covariance parameters for each state.

        The shape depends on :attr:`covariance_type`:

        * (n_components, )                        if "spherical",
        * (n_components, n_features)              if "diag",
        * (n_components, n_features, n_features)  if "full",
        * (n_features, n_features)                if "tied".

    Examples
    --------
    >>> from hmmcraft.hmm import GaussianHMM
    >>> GaussianHMM(n_components=2)
    GaussianHMM(n_components=2)
    """

    def __init__(
        self,
        n_components=1,
        covariance_type="diag",
        min_covar=1e-3,
        startprob_prior=1.0,
        transmat_prior=1.0,
        means_prior=0,
        means_weight=0,
        covars_prior=1e-2,
        covars_weight=1,
        algorithm="viterbi",
        random_state=None,
        n_iter=10,
        tol=1e-2,
        verbose=False,
        params="stmc",
        init_params="stmc",
        implementation="log",
        covars_floor=None,
        n_threads=1,
    ):
        """
        Parameters
        ----------
        n_components : int
            Number of states.

        covariance_type : {"spherical", "diag", "full", "tied"}, optional
            The type of covariance parameters to use:

            * "spherical" --- each state uses a single variance value that
              applies to all features.
            * "diag" --- each state uses a diagonal covariance matrix
              (default).
            * "full" --- each state uses a full (i.e. unrestricted)
              covariance matrix.
            * "tied" --- all states use **the same** full covariance matrix.

        min_covar : float, optional
            Added to the diagonal of the covariance of the data that starts
            the fit. It is not a floor during the fit: see ``covars_floor``.
            Defaults to 1e-3.

        startprob_prior : array, shape (n_components, ), optional
            Parameters of the Dirichlet prior distribution for
            :attr:`startprob_`.

        transmat_prior : array, shape (n_components, n_components), optional
            Parameters of the Dirichlet prior distribution for each row
            of the transition probabilities :attr:`transmat_`.

        means_prior : float or array, optional
            Mean of the Normal prior distribution for :attr:`means_`.  A
            number, or an array that broadcasts to
            ``(n_components, n_features)``: shape ``(n_features, )`` gives
            one value per feature for all states, a column of shape
            ``(n_components, 1)`` one value per state.  An array of shape
            ``(n_components, )`` is matched to the features, not to the
            states.  Defaults to 0.

        means_weight : float or array, optional
            Weight of ``means_prior``: the prior mean counts as this many
            observations of each state.  With "spherical" or "diag", a number
            or an array of a shape ``means_prior`` accepts; with "full" or
            "tied", a number only.  The default, 0, puts no prior on the
            means.

        covars_prior : float or array, optional
            Parameters of the prior distribution for the covariance matrix
            :attr:`covars_`, with ``covars_weight``.  It is added to the
            posterior-weighted scatter of the observations around each
            state's mean (for "tied", summed over the states) before the
            division by the posterior mass.

            * "spherical", "diag": a number, or an array of a shape
              ``means_prior`` accepts.  For "spherical" the variances this
              gives per feature are averaged.
            * "full": a number, or a symmetric array of shape
              ``(n_features, n_features)`` or
              ``(n_components, n_features, n_features)``.
            * "tied": a number, or a symmetric array of shape
              ``(n_features, n_features)``.

            With "full" or "tied", a number is added to every entry of the
            matrix, not only to its diagonal.  Defaults to 1e-2.

            If :attr:`covariance_type` is "spherical" or "diag" the prior is
            the inverse gamma distribution, otherwise --- the inverse Wishart
            distribution.

        covars_weight : float, optional
            A number, not an array.  ``covars_weight - 1`` ("spherical",
            "diag") or ``covars_weight - n_features`` ("full", "tied") is
            added to the posterior mass the covariance update divides by,
            when it is positive, so the default, 1, has no effect.

        algorithm : {"viterbi", "map"}, optional
            Decoder algorithm.

            - "viterbi": finds the most likely sequence of states, given all
              emissions.
            - "map" (also known as smoothing or forward-backward): finds the
              sequence of the individual most-likely states, given all
              emissions.

        random_state: RandomState or an int seed, optional
            A random number generator instance.

        n_iter : int, optional
            Maximum number of iterations to perform.

        tol : float, optional
            Convergence threshold. EM will stop if the gain in log-likelihood
            is below this value.

        verbose : bool, optional
            Whether per-iteration convergence reports are printed to
            :data:`sys.stderr`.  Convergence can also be diagnosed using the
            :attr:`monitor_` attribute.

        params, init_params : string, optional
            The parameters that get updated during (``params``) or initialized
            before (``init_params``) the training.  Can contain any combination
            of 's' for startprob, 't' for transmat, 'm' for means, and 'c' for
            covars.  Defaults to all parameters.

        implementation : string, optional
            Determines if the forward-backward algorithm is implemented with
            logarithms ("log"), or using scaling ("scaling").  The default is
            to use logarithms for backwards compatability.

        covars_floor : float or None, optional
            If a positive number, the covariances are floored after every
            M-step that updates them: variances ("spherical", "diag") and
            eigenvalues ("full", "tied") below it are raised to it. A state
            that explains a few repeated values would otherwise shrink its
            variance towards zero. The default, None, applies no floor,
            and this changes a fit only if a floor binds. It is independent
            of ``min_covar``. Any other value (not a finite number above 0,
            or a boolean) raises ValueError when the model is checked.

        n_threads : int, optional
            Number of threads :meth:`fit` uses in each E-step for the
            forward and backward passes and the expected transition
            counts. Each sequence is computed on one thread, as it is
            with one thread, and the results are added in the order of
            the sequences, so the fitted model is the same at any value.
            A sequence is never split: one sequence gains nothing.
            Sequences computed at the same time are held in memory
            together, so memory use grows with ``n_threads`` when the
            sequences are long. It has no effect on a model whose E-step
            runs one sequence at a time, such as a subclass that replaces
            an E-step method, and prediction uses one thread. The
            default, 1, starts no thread. A value that is not an integer
            of at least 1 (a boolean is not one) raises ValueError in
            :meth:`fit`.
        """
        super().__init__(
            n_components,
            startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            params=params,
            verbose=verbose,
            init_params=init_params,
            implementation=implementation,
            n_threads=n_threads,
        )
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight
        self.covars_floor = covars_floor

    means_ = _ConvertedAttribute(np.asarray)

    def __setstate__(self, state):
        # A model pickled before covars_floor existed has no such entry.
        state.setdefault("covars_floor", None)
        super().__setstate__(state)

    def _compute_estep_factors(self):
        # The covariance array and, per state, the factor and
        # log-determinant the full and tied densities compute from it
        # alone; None for the other types and the inputs it is not proven
        # for (stats._covar_factors).
        covars = getattr(self, "_covars_", None)
        factors = _covar_factors(
            getattr(self, "means_", None), covars, self.covariance_type
        )
        return None if factors is None else (covars, factors)

    def _init(self, X, lengths=None):
        super()._init(X, lengths)

        if self._needs_init("m", "means_"):
            kmeans = cluster.KMeans(
                n_clusters=self.n_components,
                random_state=self.random_state,
                n_init=10,
            )  # sklearn <1.4 backcompat.
            # One thread: scikit-learn's k-means adds per-thread partial
            # sums, so a seeded start would depend on the thread count.
            with threadpool_limits(limits=1):
                kmeans.fit(X)
            self.means_ = kmeans.cluster_centers_
        if self._needs_init("c", "covars_"):
            cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
            if not cv.shape:
                cv = cv.reshape(1, 1)
            self.covars_ = (
                _utils.distribute_covar_matrix_to_match_covariance_type(
                    cv, self.covariance_type, self.n_components
                ).copy()
            )

    def _check(self):
        super()._check()

        # means_ gives the number of features, not a stale n_features, so
        # a model whose means_ and covars_ were both replaced for another
        # number of features keeps working. The hook refuses a means_ of
        # the wrong shape.
        self._n_features_from_emissions()
        self._check_finite("means_", self.means_)

        if self.covariance_type not in COVARIANCE_TYPES:
            raise ValueError(
                f"covariance_type must be one of {COVARIANCE_TYPES}"
            )
        _covars_floor_value(self.covars_floor)
        # Finiteness only, read where the M-step stores it: the setter's
        # positive-definiteness tests are not applied to a fitted value.
        covars = getattr(self, "_covars_", None)
        if covars is not None:
            self._check_finite("covars_", covars)

    def _store_checked(self):
        super()._store_checked()
        # fit leaves the number of features means_ gives.
        self.n_features = self._n_features_from_emissions()

    def _get_n_features(self):
        # A means_ of a valid shape gives the number of features: n_features
        # can be left from an earlier fit. Otherwise n_features, so that fit
        # can still replace a means_ of another shape (n_components changed,
        # "m" in init_params); without either, the means_ error stands.
        try:
            return self._n_features_from_emissions()
        except ValueError:
            return super()._get_n_features()

    def _needs_sufficient_statistics_for_mean(self):
        return "m" in self.params

    def _needs_sufficient_statistics_for_covars(self):
        return "c" in self.params

    def _do_mstep(self, stats):
        super()._do_mstep(stats)

        means_prior = self.means_prior
        means_weight = self.means_weight

        # TODO: find a proper reference for estimates for different
        #       covariance models.
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
        denom = stats["post"][:, None]
        if "m" in self.params:
            self.means_ = (means_weight * means_prior + stats["obs"]) / (
                means_weight + denom
            )

        if "c" in self.params:
            covars_prior = self.covars_prior
            covars_weight = self.covars_weight
            meandiff = self.means_ - means_prior

            if self.covariance_type in ("spherical", "diag"):
                c_n = (
                    means_weight * meandiff**2
                    + stats["obs**2"]
                    - 2 * self.means_ * stats["obs"]
                    + self.means_**2 * denom
                )
                c_d = max(covars_weight - 1, 0) + denom
                self._covars_ = (covars_prior + c_n) / np.maximum(c_d, 1e-5)
                if self.covariance_type == "spherical":
                    # One variance per state, shape (n_components,), as
                    # documented for ``covars_``.
                    self._covars_ = self._covars_.mean(1)
            elif self.covariance_type in ("tied", "full"):
                c_n = np.empty(
                    (self.n_components, self.n_features, self.n_features)
                )
                for c in range(self.n_components):
                    obsmean = np.outer(stats["obs"][c], self.means_[c])
                    c_n[c] = (
                        means_weight * np.outer(meandiff[c], meandiff[c])
                        + stats["obs*obs.T"][c]
                        - obsmean
                        - obsmean.T
                        + np.outer(self.means_[c], self.means_[c])
                        * stats["post"][c]
                    )
                cvweight = max(covars_weight - self.n_features, 0)
                if self.covariance_type == "tied":
                    self._covars_ = (covars_prior + c_n.sum(axis=0)) / (
                        cvweight + stats["post"].sum()
                    )
                elif self.covariance_type == "full":
                    self._covars_ = (covars_prior + c_n) / (
                        cvweight + stats["post"][:, None, None]
                    )

            # A state that captures exactly repeated values would otherwise
            # shrink its variance towards zero.
            floor = _covars_floor_value(self.covars_floor)
            if floor is not None:
                self._covars_ = _floor_covars(
                    self._covars_, self.covariance_type, floor
                )


@_block_estep
class GMMHMM(_emissions.BaseGMMHMM):
    """
    Hidden Markov Model with Gaussian mixture emissions.

    Attributes
    ----------
    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    state_occupancy_ : array, shape (n_components, )
        Expected number of samples in each state: the posteriors of the last
        E-step of :meth:`fit`, summed over all samples.

    weights_ : array, shape (n_components, n_mix)
        Mixture weights for each state.

    means_ : array, shape (n_components, n_mix, n_features)
        Mean parameters for each mixture component in each state.

    covars_ : array
        Covariance parameters for each mixture components in each state.

        The shape depends on :attr:`covariance_type`:

        * (n_components, n_mix)                          if "spherical",
        * (n_components, n_mix, n_features)              if "diag",
        * (n_components, n_mix, n_features, n_features)  if "full"
        * (n_components, n_features, n_features)         if "tied".
    """

    def __init__(
        self,
        n_components=1,
        n_mix=1,
        min_covar=1e-3,
        startprob_prior=1.0,
        transmat_prior=1.0,
        weights_prior=1.0,
        means_prior=0.0,
        means_weight=0.0,
        covars_prior=None,
        covars_weight=None,
        algorithm="viterbi",
        covariance_type="diag",
        random_state=None,
        n_iter=10,
        tol=1e-2,
        verbose=False,
        params="stmcw",
        init_params="stmcw",
        implementation="log",
        n_threads=1,
    ):
        """
        Parameters
        ----------
        n_components : int
            Number of states in the model.

        n_mix : int
            Number of states in the GMM.

        covariance_type : {"spherical", "diag", "full", "tied"}, optional
            The type of covariance parameters to use:

            * "spherical" --- each state uses a single variance value that
              applies to all features.
            * "diag" --- each state uses a diagonal covariance matrix
              (default).
            * "full" --- each state uses a full (i.e. unrestricted)
              covariance matrix.
            * "tied" --- all mixture components of each state use **the same**
              full covariance matrix (note that this is not the same as for
              `GaussianHMM`).

        min_covar : float, optional
            Floor on the diagonal of the covariance matrix to prevent
            overfitting. Defaults to 1e-3.

        startprob_prior : array, shape (n_components, ), optional
            Parameters of the Dirichlet prior distribution for
            :attr:`startprob_`.

        transmat_prior : array, shape (n_components, n_components), optional
            Parameters of the Dirichlet prior distribution for each row
            of the transition probabilities :attr:`transmat_`.

        weights_prior : float or array, optional
            Parameters of the Dirichlet prior distribution for
            :attr:`weights_`.  A number, or an array that broadcasts to
            ``(n_components, n_mix)``: shape ``(n_mix, )`` gives one value
            per mixture component for all states, a column of shape
            ``(n_components, 1)`` one value per state.  An array of shape
            ``(n_components, )`` is matched to the mixture components, not
            to the states.  The default, 1, puts no prior on the weights.

        means_prior : float or array, optional
            Mean of the Normal prior distribution for :attr:`means_`.  A
            number, or an array that broadcasts to
            ``(n_components, n_mix, n_features)``: shape ``(n_features, )``
            gives one value per feature, a column of shape ``(n_mix, 1)``
            one value per mixture component, and one of shape
            ``(n_components, 1, 1)`` one value per state.  An array of shape
            ``(n_mix, )`` is matched to the features, not to the mixture
            components.  Defaults to 0.

        means_weight : float or array, optional
            Weight of ``means_prior``: the prior mean counts as this many
            observations of each mixture component.  A number, or an array
            of a shape ``weights_prior`` accepts.  The default, 0, puts no
            prior on the means.

        covars_prior : float or array, optional
            Parameters of the prior distribution for the covariance matrix
            :attr:`covars_`, with ``covars_weight``.

            If :attr:`covariance_type` is "spherical" or "diag" the prior is
            the inverse gamma distribution, otherwise --- the inverse Wishart
            distribution.  ``covars_prior`` is the shape parameter of the
            inverse gamma distribution and the scale matrix of the inverse
            Wishart distribution.

            * "spherical": a number, or an array of a shape
              ``weights_prior`` accepts.
            * "diag": a number, or an array of a shape ``means_prior``
              accepts.
            * "full": a number, or a symmetric array of shape
              ``(n_features, n_features)``,
              ``(n_mix, n_features, n_features)`` or
              ``(n_components, n_mix, n_features, n_features)``.
            * "tied": a number, or a symmetric array of shape
              ``(n_features, n_features)`` or
              ``(n_components, n_features, n_features)``.

            With "full" or "tied", a number is added to every entry of the
            matrix, not only to its diagonal.

        covars_weight : float or array, optional
            Parameters of the prior distribution for :attr:`covars_`, with
            ``covars_prior``: the scale parameter of the inverse gamma
            distribution, or the degrees of freedom of the inverse Wishart
            distribution.

            * "spherical", "full": a number, or an array of a shape
              ``weights_prior`` accepts.
            * "diag": a number, or an array of a shape ``means_prior``
              accepts.
            * "tied": a number, or an array of shape ``(n_components, )``,
              one value per state.

        algorithm : {"viterbi", "map"}, optional
            Decoder algorithm.

            - "viterbi": finds the most likely sequence of states, given all
              emissions.
            - "map" (also known as smoothing or forward-backward): finds the
              sequence of the individual most-likely states, given all
              emissions.

        random_state: RandomState or an int seed, optional
            A random number generator instance.

        n_iter : int, optional
            Maximum number of iterations to perform.

        tol : float, optional
            Convergence threshold. EM will stop if the gain in log-likelihood
            is below this value.

        verbose : bool, optional
            Whether per-iteration convergence reports are printed to
            :data:`sys.stderr`.  Convergence can also be diagnosed using the
            :attr:`monitor_` attribute.

        params, init_params : string, optional
            The parameters that get updated during (``params``) or initialized
            before (``init_params``) the training.  Can contain any combination
            of 's' for startprob, 't' for transmat, 'm' for means, 'c'
            for covars, and 'w' for GMM mixing weights.  Defaults to all
            parameters.

        implementation : string, optional
            Determines if the forward-backward algorithm is implemented with
            logarithms ("log"), or using scaling ("scaling").  The default is
            to use logarithms for backwards compatability.

        n_threads : int, optional
            Number of threads :meth:`fit` uses in each E-step for the
            forward and backward passes and the expected transition
            counts. Each sequence is computed on one thread, as it is
            with one thread, and the results are added in the order of
            the sequences, so the fitted model is the same at any value.
            A sequence is never split: one sequence gains nothing.
            Sequences computed at the same time are held in memory
            together, so memory use grows with ``n_threads`` when the
            sequences are long. It has no effect on a model whose E-step
            runs one sequence at a time, such as a subclass that replaces
            an E-step method, and prediction uses one thread. The
            default, 1, starts no thread. A value that is not an integer
            of at least 1 (a boolean is not one) raises ValueError in
            :meth:`fit`.
        """
        BaseHMM.__init__(
            self,
            n_components,
            startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
            params=params,
            init_params=init_params,
            implementation=implementation,
            n_threads=n_threads,
        )
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.n_mix = n_mix
        self.weights_prior = weights_prior
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight

    weights_ = _ConvertedAttribute(np.array)
    means_ = _ConvertedAttribute(np.array)
    covars_ = _ConvertedAttribute(np.array)

    def _init(self, X, lengths=None):
        super()._init(X, lengths=None)
        nc = self.n_components
        nf = self.n_features
        nm = self.n_mix

        def compute_cv():
            return np.cov(X.T) + self.min_covar * np.eye(nf)

        # Default values for covariance prior parameters
        self._init_covar_priors()
        self._fix_priors_shape()

        main_kmeans = cluster.KMeans(
            n_clusters=nc, random_state=self.random_state, n_init=10
        )  # sklearn >=1.2 compat.
        cv = None  # covariance matrix
        # One thread, as in GaussianHMM._init. One block for all the k-means
        # calls of the start, since each entry into the limit has a cost.
        with threadpool_limits(limits=1):
            labels = main_kmeans.fit_predict(X)
            main_centroid = np.mean(main_kmeans.cluster_centers_, axis=0)
            means = []
            for label in range(nc):
                kmeans = cluster.KMeans(
                    n_clusters=nm, random_state=self.random_state, n_init=10
                )  # sklearn >=1.2 compat.
                X_cluster = X[np.where(labels == label)]
                if X_cluster.shape[0] >= nm:
                    kmeans.fit(X_cluster)
                    means.append(kmeans.cluster_centers_)
                else:
                    if cv is None:
                        cv = compute_cv()
                    m_cluster = np.random.multivariate_normal(
                        main_centroid, cov=cv, size=nm
                    )
                    means.append(m_cluster)

        if self._needs_init("w", "weights_"):
            self.weights_ = np.full((nc, nm), 1 / nm)

        if self._needs_init("m", "means_"):
            self.means_ = np.stack(means)

        if self._needs_init("c", "covars_"):
            if cv is None:
                cv = compute_cv()
            if not cv.shape:
                cv = cv.reshape(1, 1)
            if self.covariance_type == "tied":
                self.covars_ = np.zeros((nc, nf, nf))
                self.covars_[:] = cv
            elif self.covariance_type == "full":
                self.covars_ = np.zeros((nc, nm, nf, nf))
                self.covars_[:] = cv
            elif self.covariance_type == "diag":
                self.covars_ = np.zeros((nc, nm, nf))
                self.covars_[:] = np.diag(cv)
            elif self.covariance_type == "spherical":
                self.covars_ = np.zeros((nc, nm))
                self.covars_[:] = cv.mean()

    def _init_covar_priors(self):
        if self.covariance_type == "full":
            if self.covars_prior is None:
                self.covars_prior = 0.0
            if self.covars_weight is None:
                self.covars_weight = -(1.0 + self.n_features + 1.0)
        elif self.covariance_type == "tied":
            if self.covars_prior is None:
                self.covars_prior = 0.0
            if self.covars_weight is None:
                self.covars_weight = -(self.n_mix + self.n_features + 1.0)
        elif self.covariance_type == "diag":
            if self.covars_prior is None:
                self.covars_prior = -1.5
            if self.covars_weight is None:
                self.covars_weight = 0.0
        elif self.covariance_type == "spherical":
            if self.covars_prior is None:
                self.covars_prior = -(self.n_mix + 2.0) / 2.0
            if self.covars_weight is None:
                self.covars_weight = 0.0

    # The priors fit broadcasts to the width of X (_prior_shapes): keep the
    # two lists in step.
    _derived_from_X = (
        "n_features",
        "weights_prior",
        "means_prior",
        "means_weight",
        "covars_prior",
        "covars_weight",
    )

    def _prior_shapes(self, nf):
        """
        Return ``(name, shape)`` for each prior, in the order they are
        broadcast: the shape :meth:`fit` stores it with.
        """
        nc = self.n_components
        nm = self.n_mix
        shapes = [
            ("weights_prior", (nc, nm)),
            ("means_prior", (nc, nm, nf)),
            ("means_weight", (nc, nm)),
        ]
        covars = {
            "full": ((nc, nm, nf, nf), (nc, nm)),
            "tied": ((nc, nf, nf), nc),
            "diag": ((nc, nm, nf), (nc, nm, nf)),
            "spherical": ((nc, nm), (nc, nm)),
        }.get(self.covariance_type)
        if covars is not None:
            shapes += [
                ("covars_prior", covars[0]),
                ("covars_weight", covars[1]),
            ]
        return shapes

    def _fix_priors_shape(self):
        # If priors are numbers, this function will make them into a
        # matrix of proper shape
        for name, shape in self._prior_shapes(self.n_features):
            setattr(
                self, name, np.broadcast_to(getattr(self, name), shape).copy()
            )

    def _value_after_refused_fit(self, name, before, current):
        # _init_covar_priors fills a None covariance prior with a number;
        # the prior keeps that number, without the width of the refused X,
        # so a later fit reads the value it read before. A prior broadcast
        # to an empty shape (n_mix=0) holds no number and goes back to None.
        if (
            name in ("covars_prior", "covars_weight")
            and before is None
            and current is not None
            and current is not _ABSENT
            and np.size(current) > 0
        ):
            return float(np.asarray(current).flat[0])
        return super()._value_after_refused_fit(name, before, current)

    def _check(self):
        super()._check()
        nc = self.n_components
        nf = self._get_n_features()
        nm = self.n_mix

        # A prior must broadcast to the shape fit stores it with; nothing
        # is stored here.
        for name, shape in self._prior_shapes(nf):
            np.broadcast_to(getattr(self, name), shape)

        # Checking covariance type
        if self.covariance_type not in COVARIANCE_TYPES:
            raise ValueError(
                f"covariance_type must be one of {COVARIANCE_TYPES}"
            )

        # Checking mixture weights' shape
        if self.weights_.shape != (nc, nm):
            raise ValueError(
                f"weights_ must have shape (n_components, n_mix), "
                f"actual shape: {self.weights_.shape}"
            )

        # Checking mixture weights' mathematical correctness
        self._check_sum_1("weights_")

        # Checking means' shape
        if self.means_.shape != (nc, nm, nf):
            raise ValueError(
                f"means_ must have shape (n_components, n_mix, n_features), "
                f"actual shape: {self.means_.shape}"
            )
        self._check_finite("means_", self.means_)

        # Checking covariances' shape
        covars_shape = self.covars_.shape
        needed_shapes = {
            "spherical": (nc, nm),
            "tied": (nc, nf, nf),
            "diag": (nc, nm, nf),
            "full": (nc, nm, nf, nf),
        }
        needed_shape = needed_shapes[self.covariance_type]
        if covars_shape != needed_shape:
            raise ValueError(
                f"{self.covariance_type!r} mixture covars must have shape "
                f"{needed_shape}, actual shape: {covars_shape}"
            )
        self._check_finite("covars_", self.covars_)

        # Checking covariances' mathematical correctness
        if (
            self.covariance_type == "spherical"
            or self.covariance_type == "diag"
        ):
            if np.any(self.covars_ < 0):
                raise ValueError(
                    f"{self.covariance_type!r} mixture covars "
                    f"must be non-negative"
                )
            if np.any(self.covars_ == 0):
                _log.warning("Degenerate mixture covariance")
        elif _covars_pass_on_stack(self.covars_):
            pass  # every matrix symmetric and clearly positive definite
        elif self.covariance_type == "tied":
            for i, covar in enumerate(self.covars_):
                if not np.allclose(covar, covar.T):
                    raise ValueError(
                        f"Covariance of state #{i} is not symmetric"
                    )
                min_eigvalsh = linalg.eigvalsh(covar).min()
                if min_eigvalsh < 0:
                    raise ValueError(
                        f"Covariance of state #{i} is not positive definite"
                    )
                if min_eigvalsh == 0:
                    _log.warning(
                        "Covariance of state #%d has a null eigenvalue.", i
                    )
        elif self.covariance_type == "full":
            for i, mix_covars in enumerate(self.covars_):
                for j, covar in enumerate(mix_covars):
                    if not np.allclose(covar, covar.T):
                        raise ValueError(
                            f"Covariance of state #{i}, mixture #{j} is not "
                            f"symmetric"
                        )
                    min_eigvalsh = linalg.eigvalsh(covar).min()
                    if min_eigvalsh < 0:
                        raise ValueError(
                            f"Covariance of state #{i}, mixture #{j} is not "
                            f"positive definite"
                        )
                    if min_eigvalsh == 0:
                        _log.warning(
                            "Covariance of state #%d, mixture #%d "
                            "has a null eigenvalue.",
                            i,
                            j,
                        )

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        nf = self.n_features
        nm = self.n_mix

        # Maximizing weights
        if "w" in self.params:
            alphas_minus_one = self.weights_prior - 1
            w_n = stats["post_mix_sum"] + alphas_minus_one
            w_d = (stats["post_sum"] + alphas_minus_one.sum(axis=1))[:, None]
            self.weights_ = w_n / w_d

        # Maximizing means
        if "m" in self.params:
            m_n = stats["m_n"]
            m_d = stats["post_mix_sum"] + self.means_weight
            # If a componenent has zero weight, then replace nan (0/0?) means
            # by 0 (0/1).  The actual value is irrelevant as the component will
            # be unused.  This needs to be done before maximizing covariances
            # as nans would otherwise propagate to other components if
            # covariances are tied.
            m_d[(self.weights_ == 0) & (m_n == 0).all(axis=-1)] = 1
            self.means_ = m_n / m_d[:, :, None]

        # Maximizing covariances
        if "c" in self.params:
            lambdas, mus = self.means_weight, self.means_prior
            centered_means = self.means_ - mus

            def outer_f(x):  # Outer product over features.
                return x[..., :, None] * x[..., None, :]

            if self.covariance_type == "full":
                centered_means_dots = outer_f(centered_means)

                psis_t = np.transpose(self.covars_prior, axes=(0, 1, 3, 2))
                nus = self.covars_weight

                c_n = psis_t + lambdas[:, :, None, None] * centered_means_dots
                c_n += stats["c_n"]
                c_d = (stats["post_mix_sum"] + 1 + nus + nf + 1)[
                    :, :, None, None
                ]

            elif self.covariance_type == "diag":
                alphas = self.covars_prior
                betas = self.covars_weight
                centered_means2 = centered_means**2

                c_n = lambdas[:, :, None] * centered_means2 + 2 * betas
                c_n += stats["c_n"]
                c_d = stats["post_mix_sum"][:, :, None] + 1 + 2 * (alphas + 1)

            elif self.covariance_type == "spherical":
                centered_means_norm2 = np.einsum(  # Faster than (x**2).sum(-1)
                    "...i,...i", centered_means, centered_means
                )

                alphas = self.covars_prior
                betas = self.covars_weight

                c_n = lambdas * centered_means_norm2 + 2 * betas
                c_n += stats["c_n"]
                c_d = nf * (stats["post_mix_sum"] + 1) + 2 * (alphas + 1)

            elif self.covariance_type == "tied":
                centered_means_dots = outer_f(centered_means)

                psis_t = np.transpose(self.covars_prior, axes=(0, 2, 1))
                nus = self.covars_weight

                c_n = (
                    np.einsum("ij,ijkl->ikl", lambdas, centered_means_dots)
                    + psis_t
                )
                c_n += stats["c_n"]
                c_d = (stats["post_sum"] + nm + nus + nf + 1)[:, None, None]

            self.covars_ = c_n / c_d


@_block_estep
class MultinomialHMM(_emissions.BaseMultinomialHMM):
    """
    Hidden Markov Model with multinomial emissions.

    Each row of ``X`` holds the counts of the ``n_features`` symbols in
    ``n_trials`` draws; for one symbol per sample, use `CategoricalHMM`.

    Attributes
    ----------
    n_features : int
        Number of possible symbols emitted by the model (in the samples).

    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    state_occupancy_ : array, shape (n_components, )
        Expected number of samples in each state: the posteriors of the last
        E-step of :meth:`fit`, summed over all samples.

    emissionprob_ : array, shape (n_components, n_features)
        Probability of emitting a given symbol when in each state.

    Examples
    --------
    >>> from hmmcraft.hmm import MultinomialHMM
    """

    def __init__(
        self,
        n_components=1,
        n_trials=None,
        startprob_prior=1.0,
        transmat_prior=1.0,
        algorithm="viterbi",
        random_state=None,
        n_iter=10,
        tol=1e-2,
        verbose=False,
        params="ste",
        init_params="ste",
        implementation="log",
        n_threads=1,
    ):
        """
        Parameters
        ----------
        n_components : int
            Number of states.

        n_trials : int or array of int
            Number of trials (when sampling, all samples must have the same
            :attr:`n_trials`).

        startprob_prior : array, shape (n_components, ), optional
            Parameters of the Dirichlet prior distribution for
            :attr:`startprob_`.

        transmat_prior : array, shape (n_components, n_components), optional
            Parameters of the Dirichlet prior distribution for each row
            of the transition probabilities :attr:`transmat_`.

        algorithm : {"viterbi", "map"}, optional
            Decoder algorithm.

            - "viterbi": finds the most likely sequence of states, given all
              emissions.
            - "map" (also known as smoothing or forward-backward): finds the
              sequence of the individual most-likely states, given all
              emissions.

        random_state: RandomState or an int seed, optional
            A random number generator instance.

        n_iter : int, optional
            Maximum number of iterations to perform.

        tol : float, optional
            Convergence threshold. EM will stop if the gain in log-likelihood
            is below this value.

        verbose : bool, optional
            Whether per-iteration convergence reports are printed to
            :data:`sys.stderr`.  Convergence can also be diagnosed using the
            :attr:`monitor_` attribute.

        params, init_params : string, optional
            The parameters that get updated during (``params``) or initialized
            before (``init_params``) the training.  Can contain any
            combination of 's' for startprob, 't' for transmat, and 'e' for
            emissionprob.  Defaults to all parameters.

        implementation : string, optional
            Determines if the forward-backward algorithm is implemented with
            logarithms ("log"), or using scaling ("scaling").  The default is
            to use logarithms for backwards compatability.

        n_threads : int, optional
            Number of threads :meth:`fit` uses in each E-step for the
            forward and backward passes and the expected transition
            counts. Each sequence is computed on one thread, as it is
            with one thread, and the results are added in the order of
            the sequences, so the fitted model is the same at any value.
            A sequence is never split: one sequence gains nothing.
            Sequences computed at the same time are held in memory
            together, so memory use grows with ``n_threads`` when the
            sequences are long. It has no effect on a model whose E-step
            runs one sequence at a time, such as a subclass that replaces
            an E-step method, and prediction uses one thread. The
            default, 1, starts no thread. A value that is not an integer
            of at least 1 (a boolean is not one) raises ValueError in
            :meth:`fit`.
        """
        BaseHMM.__init__(
            self,
            n_components,
            startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
            params=params,
            init_params=init_params,
            implementation=implementation,
            n_threads=n_threads,
        )
        self.n_trials = n_trials

    emissionprob_ = _ConvertedAttribute(np.atleast_2d)

    def _init(self, X, lengths=None):
        super()._init(X, lengths=None)
        self.random_state = check_random_state(self.random_state)
        if self._needs_init("e", "emissionprob_"):
            self.emissionprob_ = self.random_state.rand(
                self.n_components, self.n_features
            )
            normalize(self.emissionprob_, axis=1)

    def _check(self):
        super()._check()
        n_features = self._get_n_features()
        if self.emissionprob_.shape != (self.n_components, n_features):
            raise ValueError(
                "emissionprob_ must have shape (n_components, n_features)"
            )
        # A float copy, not stored: an object array of numbers is tested by
        # its values, as the emission density reads it.
        self._check_sum_1(
            "emissionprob_", np.asarray(self.emissionprob_, dtype=float)
        )
        if self.n_trials is None:
            raise ValueError("n_trials must be set")

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        if "e" in self.params:
            self.emissionprob_ = stats["obs"] / stats["obs"].sum(
                axis=1, keepdims=True
            )


@_block_estep
class PoissonHMM(_emissions.BasePoissonHMM):
    """
    Hidden Markov Model with Poisson emissions.

    ``X`` holds counts, non-negative whole numbers (whole numbers stored as
    floats included), one column per feature.

    Attributes
    ----------
    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    state_occupancy_ : array, shape (n_components, )
        Expected number of samples in each state: the posteriors of the last
        E-step of :meth:`fit`, summed over all samples.

    lambdas_ : array, shape (n_components, n_features)
        The expectation value of the waiting time parameters for each
        feature in a given state.
    """

    def __init__(
        self,
        n_components=1,
        startprob_prior=1.0,
        transmat_prior=1.0,
        lambdas_prior=0.0,
        lambdas_weight=0.0,
        algorithm="viterbi",
        random_state=None,
        n_iter=10,
        tol=1e-2,
        verbose=False,
        params="stl",
        init_params="stl",
        implementation="log",
        n_threads=1,
    ):
        """
        Parameters
        ----------
        n_components : int
            Number of states.

        startprob_prior : array, shape (n_components, ), optional
            Parameters of the Dirichlet prior distribution for
            :attr:`startprob_`.

        transmat_prior : array, shape (n_components, n_components), optional
            Parameters of the Dirichlet prior distribution for each row
            of the transition probabilities :attr:`transmat_`.

        lambdas_prior, lambdas_weight : array, shape (n_components,), optional
            The gamma prior on the lambda values using alpha-beta notation,
            respectivley. If None, will be set based on the method of
            moments.

        algorithm : {"viterbi", "map"}, optional
            Decoder algorithm.

            - "viterbi": finds the most likely sequence of states, given all
              emissions.
            - "map" (also known as smoothing or forward-backward): finds the
              sequence of the individual most-likely states, given all
              emissions.

        random_state: RandomState or an int seed, optional
            A random number generator instance.

        n_iter : int, optional
            Maximum number of iterations to perform.

        tol : float, optional
            Convergence threshold. EM will stop if the gain in log-likelihood
            is below this value.

        verbose : bool, optional
            Whether per-iteration convergence reports are printed to
            :data:`sys.stderr`.  Convergence can also be diagnosed using the
            :attr:`monitor_` attribute.

        params, init_params : string, optional
            The parameters that get updated during (``params``) or initialized
            before (``init_params``) the training.  Can contain any
            combination of 's' for startprob, 't' for transmat, and 'l' for
            lambdas.  Defaults to all parameters.

        implementation : string, optional
            Determines if the forward-backward algorithm is implemented with
            logarithms ("log"), or using scaling ("scaling").  The default is
            to use logarithms for backwards compatability.

        n_threads : int, optional
            Number of threads :meth:`fit` uses in each E-step for the
            forward and backward passes and the expected transition
            counts. Each sequence is computed on one thread, as it is
            with one thread, and the results are added in the order of
            the sequences, so the fitted model is the same at any value.
            A sequence is never split: one sequence gains nothing.
            Sequences computed at the same time are held in memory
            together, so memory use grows with ``n_threads`` when the
            sequences are long. It has no effect on a model whose E-step
            runs one sequence at a time, such as a subclass that replaces
            an E-step method, and prediction uses one thread. The
            default, 1, starts no thread. A value that is not an integer
            of at least 1 (a boolean is not one) raises ValueError in
            :meth:`fit`.
        """
        BaseHMM.__init__(
            self,
            n_components,
            startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
            params=params,
            init_params=init_params,
            implementation=implementation,
            n_threads=n_threads,
        )
        self.lambdas_prior = lambdas_prior
        self.lambdas_weight = lambdas_weight

    lambdas_ = _ConvertedAttribute(np.atleast_2d)

    def _init(self, X, lengths=None):
        super()._init(X, lengths)
        self.random_state = check_random_state(self.random_state)

        mean_X = X.mean()
        var_X = X.var()

        if self._needs_init("l", "lambdas_"):
            # initialize with method of moments based on X
            self.lambdas_ = self.random_state.gamma(
                shape=mean_X**2 / var_X,
                scale=var_X / mean_X,  # numpy uses theta = 1 / beta
                size=(self.n_components, self.n_features),
            )

    def _check(self):
        super()._check()

        n_features = self._get_n_features()
        if self.lambdas_.shape != (self.n_components, n_features):
            raise ValueError(
                "lambdas_ must have shape (n_components, n_features)"
            )
        self._check_finite("lambdas_", self.lambdas_)
        self._check_nonnegative("lambdas_", self.lambdas_)

    def _do_mstep(self, stats):
        super()._do_mstep(stats)

        if "l" in self.params:
            # Based on: Hyvönen & Tolonen, "Bayesian Inference 2019"
            # section 3.2
            # https://vioshyvo.github.io/Bayesian_inference
            alphas, betas = self.lambdas_prior, self.lambdas_weight
            n = stats["post"].sum()
            y_bar = stats["obs"] / stats["post"][:, None]
            # the same as kappa notation (more intuitive) but avoids divide by
            # 0, where:
            # kappas = betas / (betas + n)
            # self.lambdas_ = kappas * (alphas / betas) + (1 - kappas) * y_bar
            self.lambdas_ = (alphas + n * y_bar) / (betas + n)
