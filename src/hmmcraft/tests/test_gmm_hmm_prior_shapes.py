import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ..hmm import GMMHMM
from .test_gmm_hmm_new import prep_params

# n_components and n_mix as in test_gmm_hmm_new.py, n_features as in the
# data of test_gmm_hmm_multisequence.py: pairwise different, so that an
# array sized for one of them cannot broadcast along another.
NC, NM, NF = 3, 2, 4
COVARIANCE_TYPES = ["spherical", "diag", "full", "tied"]
# the shape the M-step uses for each prior
FITTED_SHAPES = {
    "weights_prior": dict.fromkeys(COVARIANCE_TYPES, (NC, NM)),
    "means_prior": dict.fromkeys(COVARIANCE_TYPES, (NC, NM, NF)),
    "means_weight": dict.fromkeys(COVARIANCE_TYPES, (NC, NM)),
    "covars_prior": {
        "spherical": (NC, NM),
        "diag": (NC, NM, NF),
        "full": (NC, NM, NF, NF),
        "tied": (NC, NF, NF),
    },
    "covars_weight": {
        "spherical": (NC, NM),
        "diag": (NC, NM, NF),
        "full": (NC, NM),
        "tied": (NC,),
    },
}
# the number each prior is set to: covars_prior as in test_gmm_hmm.py, the
# weights as in test_fit_with_priors of test_gaussian_hmm.py
NUMBERS = {
    "weights_prior": 2.0,
    "means_prior": 1.0,
    "means_weight": 2.0,
    "covars_prior": 1.0,
    "covars_weight": 2.0,
}


def _model(covariance_type, **priors):
    """A model with the start of test_gmm_hmm_new.py and no k-means."""
    prng = np.random.RandomState(14)
    covars, means, startprob, transmat, weights = prep_params(
        NC, NM, NF, covariance_type, 10, 15, prng
    )
    h = GMMHMM(
        NC,
        NM,
        covariance_type=covariance_type,
        init_params="",
        n_iter=1,
        random_state=0,
        **priors,
    )
    h.startprob_, h.transmat_, h.weights_ = startprob, transmat, weights
    h.means_, h.covars_ = means, covars
    return h, prng


def _fit(covariance_type, **priors):
    """
    One EM iteration from a fixed start on a seeded sample, drawn by a
    model without priors: sample() broadcasts the priors too.
    """
    h, prng = _model(covariance_type)
    X, _ = h.sample(200, random_state=prng)
    h, _ = _model(covariance_type, **priors)
    return h.fit(X)


def _value(covariance_type, prior, shape):
    """
    The prior's number for shape (), else an array of that shape with a
    different value along each axis; for "full" and "tied" covars_prior,
    symmetric positive definite matrices.
    """
    number = NUMBERS[prior]
    if shape == ():
        return number
    if prior == "covars_prior" and covariance_type in ("full", "tied"):
        lead = shape[:-2]
        scale = number + np.arange(np.prod(lead)).reshape(lead) / 10
        return scale[..., None, None] * (np.eye(NF) + 0.5)
    return number + np.arange(np.prod(shape)).reshape(shape) / 10


def _others(prior):
    # means_prior has no effect while means_weight is 0, its default
    return (
        {"means_weight": NUMBERS["means_weight"]}
        if prior == "means_prior"
        else {}
    )


# (covariance_type, prior, shape) of each documented form; () is a number
PER_MIX = [(), (NM,), (NC, 1), (NC, NM)]
PER_FEATURE = [(), (NF,), (NM, 1), (NC, 1, 1), (NM, NF), (NC, NM, NF)]
DOCUMENTED = [
    *[
        (ct, prior, shape)
        for ct in COVARIANCE_TYPES
        for prior, shapes in [
            ("weights_prior", PER_MIX),
            ("means_prior", PER_FEATURE),
            ("means_weight", PER_MIX),
        ]
        for shape in shapes
    ],
    *[
        ("spherical", prior, shape)
        for prior in ["covars_prior", "covars_weight"]
        for shape in PER_MIX
    ],
    *[
        ("diag", prior, shape)
        for prior in ["covars_prior", "covars_weight"]
        for shape in PER_FEATURE
    ],
    *[
        ("full", "covars_prior", shape)
        for shape in [(), (NF, NF), (NM, NF, NF), (NC, NM, NF, NF)]
    ],
    *[("full", "covars_weight", shape) for shape in PER_MIX],
    *[
        ("tied", "covars_prior", shape)
        for shape in [(), (NF, NF), (NC, NF, NF)]
    ],
    *[("tied", "covars_weight", shape) for shape in [(), (NC,)]],
]


@pytest.mark.parametrize(
    "covariance_type, prior, shape",
    DOCUMENTED,
    ids=lambda v: str(v).replace(" ", ""),
)
def test_documented_prior_form_fits_as_its_broadcast(
    covariance_type, prior, shape
):
    """
    Each documented form of a prior fits, and gives the fit of the form
    broadcast to the shape the M-step uses, which the model accepts too.
    """
    value = _value(covariance_type, prior, shape)
    others = _others(prior)
    h = _fit(covariance_type, **others, **{prior: value})
    fitted_shape = FITTED_SHAPES[prior][covariance_type]
    ref = _fit(
        covariance_type,
        **others,
        **{prior: np.broadcast_to(value, fitted_shape)},
    )
    assert_array_equal(h.weights_, ref.weights_)
    assert_array_equal(h.means_, ref.means_)
    assert_array_equal(h.covars_, ref.covars_)


@pytest.mark.parametrize(
    "covariance_type, prior",
    [
        *[(ct, "means_weight") for ct in COVARIANCE_TYPES],
        ("spherical", "covars_prior"),
        ("spherical", "covars_weight"),
        ("full", "covars_weight"),
    ],
)
def test_n_mix_array_gives_one_value_per_mixture_component(
    covariance_type, prior
):
    """
    An array of shape (n_mix,) applies its entry j to mixture component j
    of every state: after one iteration from the same start, component j
    has the parameters of the fit with the number in entry j.
    """
    entries = NUMBERS[prior] + np.arange(NM)
    h = _fit(covariance_type, **{prior: entries})
    for j, number in enumerate(entries):
        ref = _fit(covariance_type, **{prior: number})
        assert_array_equal(h.means_[:, j], ref.means_[:, j])
        if covariance_type != "tied":  # one covariance for all components
            assert_array_equal(h.covars_[:, j], ref.covars_[:, j])


@pytest.mark.parametrize(
    "covariance_type, prior, shape",
    [
        *[
            ("diag", prior, (NC, 1))
            for prior in ["weights_prior", "means_weight"]
        ],
        ("diag", "means_prior", (NC, 1, 1)),
        ("spherical", "covars_prior", (NC, 1)),
        ("diag", "covars_prior", (NC, 1, 1)),
        ("spherical", "covars_weight", (NC, 1)),
        ("diag", "covars_weight", (NC, 1, 1)),
        ("full", "covars_weight", (NC, 1)),
        ("tied", "covars_weight", (NC,)),
    ],
    ids=lambda v: str(v).replace(" ", ""),
)
def test_prior_column_gives_one_value_per_state(covariance_type, prior, shape):
    """
    A column of shape (n_components, 1) or (n_components, 1, 1), or for
    "tied" covars_weight an array of shape (n_components,), applies its
    entry k to state k: after one iteration from the same start, state k
    has the parameters of the fit with the number in entry k.
    """
    entries = NUMBERS[prior] + np.arange(NC)
    h = _fit(
        covariance_type, **_others(prior), **{prior: entries.reshape(shape)}
    )
    for k, number in enumerate(entries):
        ref = _fit(covariance_type, **_others(prior), **{prior: number})
        assert_array_equal(h.weights_[k], ref.weights_[k])
        assert_array_equal(h.means_[k], ref.means_[k])
        assert_array_equal(h.covars_[k], ref.covars_[k])


# (covariance_type, prior, shape) of forms that raise ValueError
RAISES = [
    # (n_mix,), which the docstring gave every prior before
    *[(ct, "means_prior", (NM,)) for ct in COVARIANCE_TYPES],
    *[(ct, "covars_prior", (NM,)) for ct in ["diag", "full", "tied"]],
    *[(ct, "covars_weight", (NM,)) for ct in ["diag", "tied"]],
    # (n_components,) where the M-step shape is (n_components, n_mix)
    *[
        (ct, prior, (NC,))
        for ct in COVARIANCE_TYPES
        for prior in ["weights_prior", "means_weight"]
    ],
    ("spherical", "covars_prior", (NC,)),
    ("spherical", "covars_weight", (NC,)),
    ("full", "covars_weight", (NC,)),
]


@pytest.mark.parametrize(
    "covariance_type, prior, shape",
    RAISES,
    ids=lambda v: str(v).replace(" ", ""),
)
def test_undocumented_prior_shape_raises(covariance_type, prior, shape):
    """
    With n_components, n_mix and n_features pairwise different, an array
    of shape (n_mix,) raises ValueError for every prior whose M-step shape
    does not end in n_mix, and an array of shape (n_components,) for every
    prior whose M-step shape is (n_components, n_mix).
    """
    with pytest.raises(ValueError):
        _fit(covariance_type, **{prior: np.full(shape, NUMBERS[prior])})
