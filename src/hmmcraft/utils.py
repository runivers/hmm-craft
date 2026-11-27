"""Helpers to normalize arrays and to expand covariance arrays."""

import numpy as np
from scipy import special


def normalize(a, axis=None):
    """
    Normalize the input array so that it sums to 1.

    Parameters
    ----------
    a : array
        Non-normalized input data.

    axis : int
        Dimension along which normalization is performed.

    Notes
    -----
    Modifies the input **inplace**.
    """
    a_sum = a.sum(axis)
    if axis is not None and a.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum = a_sum.reshape(shape)

    a /= a_sum


def log_normalize(a, axis=None):
    """
    Normalize the input array so that ``sum(exp(a)) == 1``.

    Parameters
    ----------
    a : array
        Non-normalized input data.

    axis : int
        Dimension along which normalization is performed.

    Notes
    -----
    Modifies the input **inplace**.
    """
    if axis is not None and a.shape[axis] == 1:
        # Handle single-state GMMHMM in the degenerate case normalizing a
        # single -inf to zero.
        a[:] = 0
    else:
        with np.errstate(under="ignore"):
            a_lse = special.logsumexp(a, axis, keepdims=True)
        a -= a_lse


def fill_covars(covars, covariance_type="full", n_components=1, n_features=1):
    """
    Expand covariances to one full matrix per state.

    Parameters
    ----------
    covars : array
        Covariances in the layout of `covariance_type`.
    covariance_type : {"full", "diag", "tied", "spherical"}
        Layout of `covars`.
    n_components : int
        Number of states, used by the ``tied`` type.
    n_features : int
        Number of features, used by the ``spherical`` type.

    Returns
    -------
    array, shape (n_components, n_features, n_features)
        The covariance matrix of each state.
    """
    if covariance_type == "full":
        return covars
    if covariance_type == "diag":
        return np.array(list(map(np.diag, covars)))
    if covariance_type == "tied":
        return np.tile(covars, (n_components, 1, 1))
    if covariance_type == "spherical":
        # A (n_components, n_features) array holds one value per state
        # repeated over features (the layout produced by initialisation);
        # reduce it to one variance per state before expanding, otherwise
        # the result has n_components * n_features matrices.
        covars = np.asarray(covars)
        if covars.ndim == 2:
            if (covars == covars[:, :1]).all():
                # The copies are equal: take one. Their mean can differ
                # from it in the last bit.
                covars = covars[:, 0]
            else:
                covars = covars.mean(axis=1)
        # Regardless of what is passed in, we flatten in
        # and then expand it to the correct shape
        covars = np.ravel(covars)
        eye = np.eye(n_features)[np.newaxis, :, :]
        covars = covars[:, np.newaxis, np.newaxis]
        return eye * covars
    return None
