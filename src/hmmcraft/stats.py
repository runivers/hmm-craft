import numpy as np
from scipy import linalg

from . import _emissionsc

_FLOAT64 = np.dtype(np.float64)


def log_multivariate_normal_density(X, means, covars, covariance_type='diag'):
    """
    Compute the log probability under a multivariate Gaussian distribution.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds to a
        single data point.

    means : array_like, shape (n_components, n_features)
        List of n_features-dimensional mean vectors for n_components Gaussians.
        Each row corresponds to a single mean vector.

    covars : array_like
        List of n_components covariance parameters for each Gaussian. The shape
        depends on `covariance_type`:

        * (n_components, )                        if "spherical",
        * (n_components, n_features)              if "diag",
        * (n_components, n_features, n_features)  if "full",
        * (n_features, n_features)                if "tied".

    covariance_type : {"spherical", "diag", "full", "tied"}, optional
        The type of the covariance parameters.  Defaults to 'diag'.

    Returns
    -------
    lpr : array_like, shape (n_samples, n_components)
        Array containing the log probabilities of each data point in
        X under each of the n_components multivariate Gaussian distributions.
    """
    log_multivariate_normal_density_dict = {
        'spherical': _log_multivariate_normal_density_spherical,
        'tied': _log_multivariate_normal_density_tied,
        'diag': _log_multivariate_normal_density_diag,
        'full': _log_multivariate_normal_density_full}
    return log_multivariate_normal_density_dict[covariance_type](
        X, means, covars
    )


def _read_as_is(a, ndim, dtype=_FLOAT64):
    """Whether ``a`` is read as it is: a plain C-ordered ``dtype`` array.

    The dtype must be ``dtype`` itself, not one that compares equal to it:
    where ``long double`` is ``double`` (macOS arm64, Windows),
    ``np.longdouble`` equals ``np.float64``, but numpy computes it with
    other loops, so its expression's bytes are not the ones the loop
    repeats.
    """
    return (type(a) is np.ndarray and a.dtype == dtype
            and a.dtype.char == dtype.char and a.ndim == ndim
            and a.flags.c_contiguous and a.flags.aligned and a.size > 0)


def _log_multivariate_normal_density_diag(X, means, covars):
    """Compute Gaussian log-density at X for a diagonal model."""
    # X: (ns, nf); means: (nc, nf); covars: (nc, nf) -> (ns, nc)
    _, nf = means.shape
    # Avoid 0 log 0 = nan in degenerate covariance case.
    covars = np.maximum(covars, np.finfo(float).tiny)
    # The C++ loop gives the values of the expression below without its
    # three (ns, nc, nf) temporaries, for the inputs the tests compare as
    # bytes: plain float64 arrays in C order (numpy adds the features in
    # another order for other layouts, and can add a row longer than its
    # buffer in pieces), with no report of underflow asked for (the loop
    # gives none). The term with numpy's log stays here.
    if (_read_as_is(X, 2) and _read_as_is(means, 2)
            and _read_as_is(covars, 2)
            and X.shape[1] == nf and covars.shape == means.shape
            and nf <= np.getbufsize()
            and np.geterr()["under"] == "ignore"):
        with np.errstate(all="ignore"):
            head = nf * np.log(2 * np.pi) + np.log(covars).sum(axis=-1)
        log_prob = _emissionsc.log_density_diag(X, means, covars, head)
        # None: a value is not finite. The expression below then gives
        # numpy's values, warnings and errors.
        if log_prob is not None:
            return log_prob
    with np.errstate(over="ignore"):
        return -0.5 * (nf * np.log(2 * np.pi)
                       + np.log(covars).sum(axis=-1)
                       + ((X[:, None, :] - means) ** 2 / covars).sum(axis=-1))


def _log_multivariate_normal_density_spherical(X, means, covars):
    """Compute Gaussian log-density at X for a spherical model."""
    nc, nf = means.shape
    if covars.ndim == 1:
        covars = covars[:, np.newaxis]
    covars = np.broadcast_to(covars, (nc, nf))
    return _log_multivariate_normal_density_diag(X, means, covars)


def _log_multivariate_normal_density_tied(X, means, covars, factors=None):
    """Compute Gaussian log-density at X for a tied model."""
    nc, nf = means.shape
    cv = np.broadcast_to(covars, (nc, nf, nf))
    return _log_multivariate_normal_density_full(X, means, cv,
                                                 factors=factors)


def _cholesky_log_det(cv, nf, min_covar):
    """The lower Cholesky factor of ``cv``, as ``linalg.cholesky`` returns
    it, and the log-determinant of ``cv``."""
    try:
        cv_chol = linalg.cholesky(cv, lower=True)
    except linalg.LinAlgError:
        # The model is most probably stuck in a component with too
        # few observations, we need to reinitialize this components
        try:
            cv_chol = linalg.cholesky(cv + min_covar * np.eye(nf),
                                      lower=True)
        except linalg.LinAlgError:
            raise ValueError("'covars' must be symmetric, "
                             "positive-definite")

    # In place on a copy: numpy < 2.0.2 with AVX-512 picks its log loop by
    # where the output lies near the input, unless the two coincide.
    cv_diag = np.diagonal(cv_chol).copy()
    cv_log_det = 2 * np.sum(np.log(cv_diag, out=cv_diag))
    return cv_chol, cv_log_det


def _log_multivariate_normal_density_full(X, means, covars, min_covar=1.e-7,
                                          factors=None):
    """Log probability for full covariance matrices.

    ``factors``, when given, holds ``_cholesky_log_det``'s pair for each
    state, computed from ``covars`` (``_covar_factors``).
    """
    _, nf = means.shape
    log_prob = []
    for c, (mu, cv) in enumerate(zip(means, covars)):
        if factors is not None:
            cv_chol, cv_log_det = factors[c]
        else:
            cv_chol, cv_log_det = _cholesky_log_det(cv, nf, min_covar)
        cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob.append(-.5 * (nf * np.log(2 * np.pi)
                               + (cv_sol ** 2).sum(axis=1)
                               + cv_log_det))

    return np.transpose(log_prob)


def _covar_factors(means, covars, covariance_type, min_covar=1.e-7):
    """
    The Cholesky factor and log-determinant of each state's covariance, as
    the ``full`` and ``tied`` log-densities compute them, or None.

    Return a list of ``_cholesky_log_det`` pairs, one per state (``tied``:
    the one pair for every state), to be given to the density with the
    same ``means`` and ``covars``. None for another covariance type, for
    covariances that are not a plain C-ordered float64 array of the shape
    the means imply, and when a covariance cannot be factored or a
    floating-point report is raised: the density then factors at each call
    and raises, warns and reports as it does without factors.
    """
    # Comparisons only: scipy's cholesky may warn at each call for another
    # dtype, and its factor's memory order, which picks solve_triangular's
    # LAPACK call, is only known for a plain float64 input.
    if not (type(means) is np.ndarray and means.ndim == 2):
        return None
    nc, nf = means.shape
    if covariance_type == "full":
        if not (_read_as_is(covars, 3) and covars.shape == (nc, nf, nf)):
            return None
        slices = covars
    elif covariance_type == "tied":
        if not (_read_as_is(covars, 2) and covars.shape == (nf, nf)):
            return None
        # State 0's slice as the tied density reads it: a read-only view
        # with the strides of covars.
        slices = np.broadcast_to(covars, (nc, nf, nf))[:1]
    else:
        return None
    try:
        with np.errstate(all="raise"):
            factors = [_cholesky_log_det(cv, nf, min_covar) for cv in slices]
    except (ValueError, FloatingPointError):
        return None
    if covariance_type == "tied":
        factors = factors * nc
    return factors
