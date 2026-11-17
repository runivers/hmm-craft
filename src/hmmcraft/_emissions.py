import functools
import inspect
import platform
import sys
import warnings

import numpy as np
from scipy import special
from scipy.stats import multinomial, poisson
from sklearn.utils import check_random_state

from . import _emissionsc, _utils
from .base import BaseHMM, _AbstractHMM, _reports_as_errors
from .stats import _read_as_is, log_multivariate_normal_density
from .utils import fill_covars, log_normalize

_CATEGORICALHMM_DOC_SUFFIX = """

Notes
-----
Unlike other HMM classes, `CategoricalHMM` ``X`` arrays have shape
``(n_samples, 1)`` (instead of ``(n_samples, n_features)``).  Consider using
`sklearn.preprocessing.LabelEncoder` to transform your input to the right
format.
"""


def _loop_platform(machine, maxsize):
    """Whether the C++ loop computes the full and tied statistic here."""
    # The loop rounds every product and every sum on its own. numpy.einsum
    # gives those bytes where numpy's own loop is compiled to do the same;
    # the tests compare the two on 64-bit x86, so the loop is used there and
    # numpy.einsum everywhere else.
    return machine in ("x86_64", "AMD64") and maxsize > 2 ** 32


_STATISTIC_LOOP = _loop_platform(platform.machine(), sys.maxsize)


def _weighted_outer_sum(posteriors, X):
    """Sum over the samples of posterior * x * x', one matrix per state."""
    # posteriors: (nt, nc); obs: (nt, nf); obs: (nt, nf)
    # -> (nc, nf, nf)
    # Where the tests compare them, the C++ loop gives the bytes of the
    # einsum below for plain float64 arrays in C order with one row per
    # sample. One state with one feature
    # is left to numpy: with a single output entry it adds the samples in
    # pieces, another order. numpy's einsum reports no floating-point error
    # in any error state, and neither does the loop.
    if (_STATISTIC_LOOP
            and _read_as_is(posteriors, 2) and _read_as_is(X, 2)
            and len(posteriors) == len(X)
            and posteriors.shape[1] * X.shape[1] > 1):
        stat = _emissionsc.weighted_outer_sum(posteriors, X)
        # None: an entry is not finite. numpy then gives its own values.
        if stat is not None:
            return stat
    return np.einsum('ij,ik,il->jkl', posteriors, X, X)


_INT64 = np.dtype(np.int64)
# The most entries of the (rows, n_components, n_features) arrays one
# multinomial call makes: two arrays of 2 MiB.
_MULTINOMIAL_ENTRIES = 2 ** 18
# The mixture hooks whose values GMMHMM's statistics may read back from its
# likelihood in an E-step: only while none of them is replaced.
_GMM_HOOKS = ("_compute_log_weighted_gaussian_densities",
              "_compute_log_likelihood", "_accumulate_sufficient_statistics")


def _special_reports_nothing():
    """Whether scipy.special is set to report no error of its functions."""
    return all(action == "ignore"
               for kind, action in special.geterr().items()
               if kind != "memory")


def _states_with_nan(values):
    """The states whose column of ``values`` holds a NaN."""
    return np.flatnonzero(np.isnan(values).any(axis=0))


def _poisson_direct(X, lambdas, n_components):
    """Whether xlogy and gammaln stand for scipy.stats.poisson here."""
    return ((_read_as_is(X, 2, _INT64) or _read_as_is(X, 2))
            and _read_as_is(lambdas, 2)
            and lambdas.shape == (n_components, X.shape[1])
            and _special_reports_nothing())


def _are_counts(X):
    """scipy.stats.poisson's test of a count, for every entry of X."""
    if X.dtype == _INT64:
        return X.min() >= 0
    return ((X >= 0) & (np.floor(X) == X)).all()


def _multinomial_together(X, emissionprob, n_components):
    """Whether one scipy.stats call scores X against every state here."""
    return (_read_as_is(X, 2, _INT64) and _read_as_is(emissionprob, 2)
            and emissionprob.shape == (n_components, X.shape[1])
            and emissionprob.size <= _MULTINOMIAL_ENTRIES
            and _special_reports_nothing())


def _multinomial_logpmf(X, n_trials, emissionprob):
    """multinomial.logpmf of each row for each state, in pieces of rows."""
    # scipy.stats computes each row and state on its own, so one call for
    # every state gives the values of one call per state. A piece of rows
    # per call keeps its (rows, n_components, n_features) arrays small.
    logprobs = np.empty((len(X), len(emissionprob)))
    rows = _MULTINOMIAL_ENTRIES // emissionprob.size
    for start in range(0, len(X), rows):
        stop = start + rows
        logprobs[start:stop] = multinomial.logpmf(
            X[start:stop, None, :], n=n_trials[start:stop, None],
            p=emissionprob[None, :, :])
    return logprobs


def _make_wrapper(func):
    return functools.wraps(func)(lambda *args, **kwargs: func(*args, **kwargs))


def _n_features_of(model, name):
    """
    Return the number of features the emission parameter *name* of *model*
    implies: its number of columns.
    """
    if not hasattr(model, name):
        raise ValueError(
            f"the number of features is unknown: set {name} or n_features")
    return np.atleast_2d(getattr(model, name)).shape[1]


def _check_X_n_features(X, n_features):
    """Check that ``X`` has one column per feature."""
    if X.shape[1] != n_features:
        raise ValueError(
            f"X must have n_features={n_features} columns, got {X.shape[1]}")


def _check_X_counts(X):
    """
    Check that every entry of ``X`` is a count: a non-negative whole
    number, of any dtype (``2.0`` is a count).
    """
    invalid = (X < 0) | (X % 1 != 0)
    if invalid.any():
        raise ValueError(
            f"X must contain counts, non-negative whole numbers, "
            f"got {X[invalid][0]}")


def _means_shape(model, name="means_"):
    """Return the shape of the means *name*; raise if it is not set."""
    means = getattr(model, name, None)
    if means is None:
        raise ValueError(
            f"the number of features is unknown: set {name} or n_features")
    return np.shape(means)


class BaseCategoricalHMM(_AbstractHMM):

    # A refused fit puts nothing back: n_features is tested with <=, so a
    # later fit with fewer symbols succeeds with the count left.
    _derived_from_X = ()
    # The emission parameter whose width is the number of symbols.
    _emissionprob_name = "emissionprob_"

    def __init_subclass__(cls):
        for name in [
                "decode",
                "fit",
                "predict",
                "predict_proba",
                "sample",
                "score",
                "score_samples",
        ]:
            meth = getattr(cls, name)
            doc = inspect.getdoc(meth)
            if doc is None or _CATEGORICALHMM_DOC_SUFFIX in doc:
                wrapper = meth
            else:
                wrapper = _make_wrapper(meth)
                wrapper.__doc__ = (
                    doc.replace("(n_samples, n_features)", "(n_samples, 1)")
                    + _CATEGORICALHMM_DOC_SUFFIX)
            setattr(cls, name, wrapper)

    def _n_features_from_emissions(self):
        return _n_features_of(self, self._emissionprob_name)

    def _kept_n_features(self):
        """
        Return the number of symbols of the emission parameter that the
        start of :meth:`fit` keeps, or ``None`` when it draws a new one or
        the parameter is not set.
        """
        # init_params is read directly: _needs_init logs.
        if "e" in self.init_params:
            return None
        try:
            return self._n_features_from_emissions()
        except ValueError:
            return None

    def _check_X(self, X):
        # No column test: the deprecated (n_samples, k) layout is read as
        # one sequence of symbols by _compute_likelihood.
        if X.dtype == bool:
            # NumPy reads a boolean array as a mask over the symbols, not as
            # the symbols 0 and 1; fit refuses it too.
            raise ValueError(
                "X must contain symbols as integers, not booleans")
        n_features = self._get_n_features()
        invalid = (X < 0) | (X >= n_features) | (X % 1 != 0)
        if invalid.any():
            raise ValueError(
                f"X must contain symbols, whole numbers from 0 to "
                f"{n_features - 1}, got {X[invalid][0]}")

    def _check_and_set_n_features(self, X):
        """
        Check if ``X`` is a sample from a categorical distribution, i.e. an
        array of non-negative integers.
        """
        if not np.issubdtype(X.dtype, np.integer):
            raise ValueError("Symbols should be integers")
        if X.min() < 0:
            raise ValueError("Symbols should be nonnegative")
        if self.n_features is not None:
            if self.n_features - 1 < X.max():
                raise ValueError(
                    f"Largest symbol is {X.max()} but the model only emits "
                    f"symbols up to {self.n_features - 1}")
        else:
            n_features = X.max() + 1
            kept = self._kept_n_features()
            if kept is not None and n_features > kept:
                raise ValueError(
                    f"Largest symbol is {X.max()} but the model only emits "
                    f"symbols up to {kept - 1}")
            # A kept emission parameter gives the number of symbols, which X
            # may show fewer of; when they are equal the count of X is
            # stored, of its type.
            self.n_features = (kept if kept is not None and kept > n_features
                               else n_features)

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self._get_n_features()
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "e": nc * (nf - 1),
        }

    def _takes_blocks(self, X):
        # The deprecated layout, more than one column, gives more emission
        # rows than samples and warns at each call.
        return X.shape[1] == 1 and super()._takes_blocks(X)

    def _lattice_by_rows(self, X):
        # Each row indexes the emission probabilities on its own.
        return True

    def _compute_likelihood(self, X):
        if X.shape[1] != 1:
            warnings.warn("Inputs of shape other than (n_samples, 1) are "
                          "deprecated.", DeprecationWarning)
            X = np.concatenate(X)[:, None]
        return self.emissionprob_[:, X.squeeze(1)].T

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(
            self, stats, X, lattice, posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(stats=stats, X=X,
                                                  lattice=lattice,
                                                  posteriors=posteriors,
                                                  fwdlattice=fwdlattice,
                                                  bwdlattice=bwdlattice)

        if 'e' in self.params:
            if X.shape[1] != 1:
                warnings.warn("Inputs of shape other than (n_samples, 1) are "
                              "deprecated.", DeprecationWarning)
                X = np.concatenate(X)[:, None]
            np.add.at(stats['obs'].T, X.squeeze(1), posteriors)

    def _generate_sample_from_state(self, state, random_state=None):
        cdf = np.cumsum(self.emissionprob_[state, :])
        random_state = check_random_state(random_state)
        return [(cdf > random_state.rand()).argmax()]


class BaseGaussianHMM(_AbstractHMM):

    # The attribute a caller sets for the means, named in errors.
    _means_name = "means_"

    @property
    def covars_(self):
        """Return covars as a full matrix."""
        # A missing _covars_ stays an AttributeError, so that hasattr()
        # tells whether covars_ is set.
        covars = self._covars_
        # Only the spherical layout needs the number of features.
        n_features = (self._get_n_features()
                      if self.covariance_type == "spherical" else None)
        return fill_covars(covars, self.covariance_type, self.n_components,
                           n_features)

    @covars_.setter
    def covars_(self, covars):
        covars = np.array(covars, copy=True)
        _utils._validate_covars(covars, self.covariance_type,
                                self.n_components)
        self._covars_ = covars

    def _n_features_from_emissions(self):
        name = self._means_name
        shape = _means_shape(self, name)
        if len(shape) != 2 or shape[0] != self.n_components:
            raise ValueError(
                f"{name} must have shape (n_components, n_features) with "
                f"n_components={self.n_components}, actual shape: {shape}")
        return shape[1]

    def _check_X(self, X):
        # The width of the means X is broadcast against, not n_features,
        # which an earlier fit may have left.
        _check_X_n_features(X, self._n_features_from_emissions())

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self._get_n_features()
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "m": nc * nf,
            "c": {
                "spherical": nc,
                "diag": nc * nf,
                "full": nc * nf * (nf + 1) // 2,
                "tied": nf * (nf + 1) // 2,
            }[self.covariance_type],
        }

    def _compute_log_likelihood(self, X):
        return log_multivariate_normal_density(
            X, self.means_, self._covars_, self.covariance_type)

    def _lattice_by_rows(self, X):
        # The C++ density loop and the numpy expression both compute row by
        # row for plain float64 arrays in C order; for other layouts numpy
        # can add the features of a row in another order.
        if self.covariance_type not in ("spherical", "diag"):
            return False
        means, covars = self.means_, self._covars_
        return (_read_as_is(X, 2) and _read_as_is(means, 2)
                and _read_as_is(covars, np.ndim(covars))
                and X.shape[1] == means.shape[1] <= np.getbufsize())

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        if self.covariance_type in ('tied', 'full'):
            stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
                                           self.n_features))
        return stats

    def _accumulate_sufficient_statistics(
            self, stats, X, lattice, posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(stats=stats, X=X,
                                                  lattice=lattice,
                                                  posteriors=posteriors,
                                                  fwdlattice=fwdlattice,
                                                  bwdlattice=bwdlattice)

        if self._needs_sufficient_statistics_for_mean() or \
           self._needs_sufficient_statistics_for_covars():
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += posteriors.T @ X

        if self._needs_sufficient_statistics_for_covars():
            if self.covariance_type in ('spherical', 'diag'):
                stats['obs**2'] += posteriors.T @ X**2
            elif self.covariance_type in ('tied', 'full'):
                stats['obs*obs.T'] += _weighted_outer_sum(posteriors, X)

    def _needs_sufficient_statistics_for_mean(self):
        """
        Whether the sufficient statistics needed to update the means are
        updated during calls to `fit`.
        """
        raise NotImplementedError("Must be overriden in subclass")

    def _needs_sufficient_statistics_for_covars(self):
        """
        Whhether the sufficient statistics needed to update the covariances are
        updated during calls to `fit`.
        """
        raise NotImplementedError("Must be overriden in subclass")

    def _generate_sample_from_state(self, state, random_state):
        return random_state.multivariate_normal(
            self.means_[state], self.covars_[state]
        )


class BaseGMMHMM(BaseHMM):

    def _n_features_from_emissions(self):
        shape = _means_shape(self)
        if len(shape) != 3:
            raise ValueError(
                f"means_ must have shape (n_components, n_mix, n_features), "
                f"actual shape: {shape}")
        return shape[2]

    def _check_X(self, X):
        _check_X_n_features(X, self._n_features_from_emissions())

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self._get_n_features()
        nm = self.n_mix
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "m": nc * nm * nf,
            "c": {
                "spherical": nc * nm,
                "diag": nc * nm * nf,
                "full": nc * nm * nf * (nf + 1) // 2,
                "tied": nc * nf * (nf + 1) // 2,
            }[self.covariance_type],
            "w": nm - 1,
        }

    def _compute_log_weighted_gaussian_densities(self, X, i_comp):
        cur_means = self.means_[i_comp]
        cur_covs = self.covars_[i_comp]
        if self.covariance_type == 'spherical':
            cur_covs = cur_covs[:, None]
        log_cur_weights = np.log(self.weights_[i_comp])

        return log_multivariate_normal_density(
            X, cur_means, cur_covs, self.covariance_type
        ) + log_cur_weights

    def _takes_blocks(self, X):
        # A zero weight reports at every call of the likelihood; such a
        # model is computed one sequence at a time, and this test reports
        # nothing.
        if not super()._takes_blocks(X):
            return False
        try:
            with np.errstate(**_reports_as_errors()):
                np.log(self.weights_)
        except FloatingPointError:
            return False
        return True

    def _hooks_are_own(self):
        """Whether none of the three mixture hooks is replaced here."""
        return all(getattr(self, name)
                   == getattr(BaseGMMHMM, name).__get__(self)
                   for name in _GMM_HOOKS)

    def _mixture_log_likelihoods(self, X, parts=None):
        """Each state's log-likelihood; with ``parts``, each state's
        weighted log-densities and their logsumexp appended to it."""
        logprobs = np.empty((len(X), self.n_components))
        for i in range(self.n_components):
            log_denses = self._compute_log_weighted_gaussian_densities(X, i)
            with np.errstate(under="ignore"):
                lse = special.logsumexp(log_denses, axis=1)
            logprobs[:, i] = lse
            if parts is not None:
                parts.append((log_denses, lse))
        return logprobs

    def _compute_log_likelihood(self, X):
        kept = self._estep_values
        if kept is not None and kept[0] is X:
            # A second call for the sequence (the log pass of a sequence
            # that falls back under scaling) replaces what the first kept.
            del kept[1:]
            if self._hooks_are_own():
                asked = _reports_as_errors()
                parts = []
                # Kept only when computing them again in this error state
                # would report nothing: each report it asks for raises here,
                # and the densities are then computed below in the
                # caller's state, with their reports.
                try:
                    with np.errstate(**_reports_as_errors()):
                        # A zero weight, the one report of an ordinary fit,
                        # found before the densities are computed.
                        np.log(self.weights_)
                        logprobs = self._mixture_log_likelihoods(X, parts)
                except FloatingPointError:
                    pass
                else:
                    widths = {log_denses.shape[-1] for log_denses, _ in parts}
                    if widths == {1}:
                        # log_normalize writes 0 over a single mixture, as
                        # computed: nothing is kept but the error state.
                        kept.append((asked, None))
                    elif 1 not in widths and np.isfinite(logprobs).all():
                        # Subtracting a finite logsumexp reports nothing.
                        kept.append((asked, parts))
                    return logprobs
        return self._mixture_log_likelihoods(X)

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['post_mix_sum'] = np.zeros((self.n_components, self.n_mix))
        stats['post_sum'] = np.zeros(self.n_components)

        if 'm' in self.params:
            lambdas, mus = self.means_weight, self.means_prior
            stats['m_n'] = lambdas[:, :, None] * mus
        if 'c' in self.params:
            stats['c_n'] = np.zeros_like(self.covars_)

        # These statistics are stored in arrays and updated in-place.
        # We accumulate chunks of data for multiple sequences (aka
        # multiple frames) during fitting. The fit(X, lengths) method
        # in the BaseHMM class will call
        # _accumulate_sufficient_statistics once per sequence in the
        # training samples. Data from all sequences needs to be
        # accumulated and fed into _do_mstep.
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, lattice,
                                          post_comp, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(
            stats, X, lattice, post_comp, fwdlattice, bwdlattice
        )

        n_samples, _ = X.shape

        # Statistics shapes:
        # post_comp_mix     (n_samples, n_components, n_mix)
        # samples           (n_samples, n_features)
        # centered          (n_samples, n_components, n_mix, n_features)

        kept = self._estep_values
        hit = (kept is not None and len(kept) == 2 and kept[0] is X
               and kept[1][0] == _reports_as_errors())
        if hit:
            # What _compute_log_likelihood kept for this sequence in this
            # error state, read once: computing it again would give the
            # same values and report nothing.
            _, parts = kept.pop()

        post_mix = np.zeros((n_samples, self.n_components, self.n_mix))
        for p in range(self.n_components):
            if hit and parts is None:
                # Densities one mixture wide: log_normalize writes 0, whose
                # exp is 1.
                post_mix[:, p, :] = 1.0
                continue
            if hit:
                log_denses, lse = parts[p]
                log_denses = log_denses - lse[:, None]
            else:
                log_denses = self._compute_log_weighted_gaussian_densities(
                    X, p)
                log_normalize(log_denses, axis=-1)
            with np.errstate(under="ignore"):
                post_mix[:, p, :] = np.exp(log_denses)
        if hit:
            # The kept arrays are freed before the larger ones below.
            parts = lse = None

        with np.errstate(under="ignore"):
            post_comp_mix = post_comp[:, :, None] * post_mix

        stats['post_mix_sum'] += post_comp_mix.sum(axis=0)
        stats['post_sum'] += post_comp.sum(axis=0)

        if 'm' in self.params:  # means stats
            stats['m_n'] += np.einsum('ijk,il->jkl', post_comp_mix, X)

        if 'c' in self.params:  # covariance stats
            centered = X[:, None, None, :] - self.means_

            def outer_f(x):  # Outer product over features.
                return x[..., :, None] * x[..., None, :]

            if self.covariance_type == 'full':
                centered_dots = outer_f(centered)
                c_n = np.einsum('ijk,ijklm->jklm', post_comp_mix,
                                centered_dots)
            elif self.covariance_type == 'diag':
                centered2 = np.square(centered, out=centered)  # reuse
                c_n = np.einsum('ijk,ijkl->jkl', post_comp_mix, centered2)
            elif self.covariance_type == 'spherical':
                # Faster than (x**2).sum(-1).
                centered_norm2 = np.einsum('...i,...i', centered, centered)
                c_n = np.einsum('ijk,ijk->jk', post_comp_mix, centered_norm2)
            elif self.covariance_type == 'tied':
                centered_dots = outer_f(centered)
                c_n = np.einsum('ijk,ijklm->jlm', post_comp_mix, centered_dots)

            stats['c_n'] += c_n

    def _generate_sample_from_state(self, state, random_state):
        cur_weights = self.weights_[state]
        i_gauss = random_state.choice(self.n_mix, p=cur_weights)
        if self.covariance_type == 'tied':
            # self.covars_.shape == (n_components, n_features, n_features)
            # shouldn't that be (n_mix, ...)?
            covs = self.covars_
        else:
            covs = self.covars_[:, i_gauss]
            covs = fill_covars(covs, self.covariance_type,
                               self.n_components, self._get_n_features())
        return random_state.multivariate_normal(
            self.means_[state, i_gauss], covs[state]
        )


class BaseMultinomialHMM(BaseHMM):

    def _n_features_from_emissions(self):
        return _n_features_of(self, "emissionprob_")

    def _check_X(self, X):
        # No test of the row totals: each row is scored with its own. No
        # dtype test either, so whole floats stay valid counts.
        _check_X_n_features(X, self._n_features_from_emissions())
        _check_X_counts(X)

    def _check_and_set_n_features(self, X):  # Also sets n_trials.
        """
        Check if ``X`` is a sample from a multinomial distribution, i.e. an
        array of non-negative integers, summing up to n_trials.
        """
        super()._check_and_set_n_features(X)
        if not np.issubdtype(X.dtype, np.integer) or X.min() < 0:
            raise ValueError("Symbol counts should be nonnegative integers")
        if self.n_trials is None:
            self.n_trials = X.sum(axis=1)
        elif not (X.sum(axis=1) == self.n_trials).all():
            raise ValueError("Total count for each sample should add up to "
                             "the number of trials")

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self._get_n_features()
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "e": nc * (nf - 1),
        }

    def _lattice_by_rows(self, X):
        # scipy.stats computes each row and state on its own.
        return _multinomial_together(X, self.emissionprob_, self.n_components)

    def _compute_likelihood(self, X):
        probs = np.empty((len(X), self.n_components))
        n_trials = X.sum(axis=1)
        states = range(self.n_components)
        if _multinomial_together(X, self.emissionprob_, self.n_components):
            # A report, or a NaN in a state, is left to scipy.stats: see
            # _reports_as_errors.
            try:
                with np.errstate(**_reports_as_errors()):
                    logprobs = _multinomial_logpmf(
                        X, n_trials, self.emissionprob_)
                    for c in states:
                        # numpy's exp is given what scipy.stats gives it:
                        # one state's values, contiguous.
                        probs[:, c] = np.exp(
                            np.ascontiguousarray(logprobs[:, c]))
                    states = _states_with_nan(probs)
            except FloatingPointError:
                pass
        for c in states:
            probs[:, c] = multinomial.pmf(
                X, n=n_trials, p=self.emissionprob_[c, :])
        return probs

    def _compute_log_likelihood(self, X):
        logprobs = np.empty((len(X), self.n_components))
        n_trials = X.sum(axis=1)
        states = range(self.n_components)
        if _multinomial_together(X, self.emissionprob_, self.n_components):
            # A report, or a NaN in a state, is left to scipy.stats: see
            # _reports_as_errors.
            try:
                with np.errstate(**_reports_as_errors()):
                    logprobs = _multinomial_logpmf(
                        X, n_trials, self.emissionprob_)
                    states = _states_with_nan(logprobs)
            except FloatingPointError:
                pass
        for c in states:
            logprobs[:, c] = multinomial.logpmf(
                X, n=n_trials, p=self.emissionprob_[c, :])
        return logprobs

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)
        if 'e' in self.params:
            stats['obs'] += posteriors.T @ X

    def _generate_sample_from_state(self, state, random_state):
        try:
            n_trials, = np.unique(self.n_trials)
        except ValueError:
            raise ValueError("For sampling, a single n_trials must be given")
        return multinomial.rvs(n=n_trials, p=self.emissionprob_[state, :],
                               random_state=random_state)


class BasePoissonHMM(BaseHMM):

    def _n_features_from_emissions(self):
        return _n_features_of(self, "lambdas_")

    def _check_X(self, X):
        _check_X_n_features(X, self._n_features_from_emissions())
        _check_X_counts(X)

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self._get_n_features()
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "l": nc * nf,
        }

    # scipy.stats.poisson tests every count (non-negative, whole) and every
    # rate (non-negative) and, for those that pass, evaluates
    # xlogy(k, mu) - gammaln(k + 1) - mu. When all pass, the lines below are
    # that expression with gammaln(k + 1) computed once for all states. They
    # are used for the arrays the tests compare with scipy.stats as bytes:
    # plain int64 or float64 counts and float64 rates in C order (scipy adds
    # 1 to other integer types in a type that depends on the numpy version),
    # while scipy.special reports no error of its own.

    def _lattice_by_rows(self, X):
        # Each row's terms are added over its own features, and
        # scipy.stats computes element by element.
        return _poisson_direct(X, self.lambdas_, self.n_components)

    def _compute_likelihood(self, X):
        probs = np.empty((len(X), self.n_components))
        states = range(self.n_components)
        if _poisson_direct(X, self.lambdas_, self.n_components):
            # A report, or a NaN in a state, is left to scipy.stats: see
            # _reports_as_errors.
            try:
                with np.errstate(**_reports_as_errors()):
                    if _are_counts(X) and (self.lambdas_ >= 0).all():
                        log_factorial = special.gammaln(X + 1)
                        for c in states:
                            rates = self.lambdas_[c]
                            probs[:, c] = np.clip(np.exp(
                                special.xlogy(X, rates) - log_factorial
                                - rates), 0, 1).prod(axis=1)
                        states = _states_with_nan(probs)
            except FloatingPointError:
                pass
        for c in states:
            probs[:, c] = poisson.pmf(X, self.lambdas_[c]).prod(axis=1)
        return probs

    def _compute_log_likelihood(self, X):
        logprobs = np.empty((len(X), self.n_components))
        states = range(self.n_components)
        if _poisson_direct(X, self.lambdas_, self.n_components):
            # A report, or a NaN in a state, is left to scipy.stats: see
            # _reports_as_errors.
            try:
                with np.errstate(**_reports_as_errors()):
                    if _are_counts(X) and (self.lambdas_ >= 0).all():
                        log_factorial = special.gammaln(X + 1)
                        for c in states:
                            rates = self.lambdas_[c]
                            logprobs[:, c] = (
                                special.xlogy(X, rates) - log_factorial
                                - rates).sum(axis=1)
                        states = _states_with_nan(logprobs)
            except FloatingPointError:
                pass
        for c in states:
            logprobs[:, c] = poisson.logpmf(X, self.lambdas_[c]).sum(axis=1)
        return logprobs

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, lattice,
                                          posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(
            stats, obs, lattice, posteriors, fwdlattice, bwdlattice)
        if 'l' in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += posteriors.T @ obs

    def _generate_sample_from_state(self, state, random_state):
        return random_state.poisson(self.lambdas_[state])
