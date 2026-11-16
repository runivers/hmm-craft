import numpy as np
import pytest
from numpy.testing import assert_array_equal

from .. import hmm
from . import make_covar_matrix, normalized

# n_components != n_features, as in test_gmm_hmm.py, so that an array of
# length n_components cannot broadcast along the features.
NC, NF = 3, 2
COVARS_SHAPES = {"spherical": (NC,), "diag": (NC, NF),
                 "full": (NC, NF, NF), "tied": (NF, NF)}


def _start(covariance_type):
    prng = np.random.RandomState(10)
    startprob = normalized(prng.rand(NC))
    transmat = normalized(prng.rand(NC, NC), axis=1)
    means = prng.randint(-20, 20, (NC, NF)).astype(float)
    covars = make_covar_matrix(covariance_type, NC, NF, random_state=prng)
    if covariance_type in ("spherical", "diag"):
        covars = np.maximum(covars, 0.1)
    return prng, startprob, transmat, means, covars


def _fit(covariance_type, **priors):
    """One EM iteration from a fixed start on a seeded sample."""
    prng, startprob, transmat, means, covars = _start(covariance_type)
    h = hmm.GaussianHMM(NC, covariance_type, init_params="", n_iter=1,
                        **priors)
    h.startprob_, h.transmat_ = startprob, transmat
    h.means_, h.covars_ = means, covars
    X, _ = h.sample(200, random_state=prng)
    return h.fit(X)


# (covariance_type, prior, shape of an array filled with one number)
FILLED = [
    *[(ct, "means_prior", shape)
      for ct in COVARS_SHAPES for shape in [(NC, NF), (NC, 1), (NF,)]],
    *[(ct, name, shape)
      for ct in ["spherical", "diag"]
      for name in ["means_weight", "covars_prior"]
      for shape in [(NC, NF), (NC, 1), (NF,)]],
    ("full", "covars_prior", (NF, NF)),
    ("full", "covars_prior", (NC, NF, NF)),
    ("tied", "covars_prior", (NF, NF)),
]
# the number each prior is filled with: means_weight as in
# test_fit_with_priors, covars_prior its default
NUMBERS = {"means_prior": 1.0, "means_weight": 2.0, "covars_prior": 1e-2}


@pytest.mark.parametrize("covariance_type, prior, shape", FILLED,
                         ids=lambda v: str(v).replace(" ", ""))
def test_documented_prior_array_fits_like_its_number(
        covariance_type, prior, shape):
    """
    Each documented array form of a prior fits, and an array filled with
    one number gives the fit of that number, so a number for "full" or
    "tied" covars_prior is added to every entry.
    """
    others = {"means_weight": 2.0} if prior == "means_prior" else {}
    number = NUMBERS[prior]
    ref = _fit(covariance_type, **others, **{prior: number})
    h = _fit(covariance_type, **others, **{prior: np.full(shape, number)})
    assert ref.means_.shape == (NC, NF)
    assert ref._covars_.shape == COVARS_SHAPES[covariance_type]
    assert_array_equal(h.means_, ref.means_)
    assert_array_equal(h._covars_, ref._covars_)


@pytest.mark.parametrize("covariance_type, prior", [
    *[(ct, name) for ct in ["spherical", "diag"]
      for name in ["means_prior", "means_weight", "covars_prior"]],
    ("full", "means_prior"),
    ("tied", "means_prior"),
])
def test_prior_column_gives_one_value_per_state(covariance_type, prior):
    """
    A column of shape (n_components, 1) applies its row k to state k: after
    one iteration from the same start, state k's parameters are those of
    the fit with the number in row k.
    """
    others = {"means_weight": 2.0} if prior == "means_prior" else {}
    column = {"means_prior": [0.0, 1.0, 2.0],
              "means_weight": [0.0, 2.0, 4.0],
              "covars_prior": [1e-2, 1.0, 2.0]}[prior]
    h = _fit(covariance_type, **others,
             **{prior: np.array(column)[:, None]})
    for k, number in enumerate(column):
        ref = _fit(covariance_type, **others, **{prior: number})
        assert_array_equal(h.means_[k], ref.means_[k])
        if covariance_type != "tied":  # one covariance for all states
            assert_array_equal(h._covars_[k], ref._covars_[k])


# (covariance_type, prior, value) that raise ValueError
RAISES = [
    *[(ct, name, np.full(NC, NUMBERS.get(name, 2.0)))
      for ct in COVARS_SHAPES
      for name in ["means_prior", "means_weight", "covars_prior",
                   "covars_weight"]],
    *[(ct, "means_weight", np.full(shape, 2.0))
      for ct in ["full", "tied"] for shape in [(NC, NF), (NC, 1)]],
    *[(ct, "covars_weight", np.full((NC, 1), 2.0)) for ct in COVARS_SHAPES],
]


@pytest.mark.parametrize(
    "covariance_type, prior, value", RAISES,
    ids=[f"{ct}-{name}-{value.shape}".replace(" ", "")
         for ct, name, value in RAISES])
def test_undocumented_prior_shape_raises(covariance_type, prior, value):
    """
    An array of shape (n_components,) is not one value per state: with
    n_components != n_features it raises ValueError, as do an array
    means_weight with "full" or "tied" and an array covars_weight.
    """
    with pytest.raises(ValueError):
        _fit(covariance_type, **{prior: value})


@pytest.mark.parametrize("covariance_type", list(COVARS_SHAPES))
def test_covars_weight_at_or_below_threshold_has_no_effect(covariance_type):
    """
    covars_weight enters as max(w - 1, 0) ("spherical", "diag") or
    max(w - n_features, 0) ("full", "tied"): the default 1, 0 and the
    threshold give the same fit, one above it does not.
    """
    threshold = 1 if covariance_type in ("spherical", "diag") else NF
    ref = _fit(covariance_type)
    for weight in [0, threshold]:
        h = _fit(covariance_type, covars_weight=weight)
        assert_array_equal(h._covars_, ref._covars_)
    h = _fit(covariance_type, covars_weight=threshold + 2)
    assert not np.array_equal(h._covars_, ref._covars_)
