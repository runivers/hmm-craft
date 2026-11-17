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


PRIORS = list(NUMBERS)


def _identity(covariance_type, n_components, n_mix, n_features):
    """Identity covariances of a model, in the shape of its type."""
    eye = np.eye(n_features)
    return {
        "full": np.tile(eye, (n_components, n_mix, 1, 1)),
        "tied": np.tile(eye, (n_components, 1, 1)),
        "diag": np.ones((n_components, n_mix, n_features)),
        "spherical": np.ones((n_components, n_mix)),
    }[covariance_type]


def _started(covariance_type, means, **priors):
    """
    A model with ``init_params="st"``, equal mixture weights, *means* and
    identity covariances.
    """
    nc, nm, nf = means.shape
    h = GMMHMM(
        nc,
        nm,
        covariance_type=covariance_type,
        init_params="st",
        random_state=0,
        **priors,
    )
    h.weights_ = np.full((nc, nm), 1 / nm)
    h.means_ = means
    h.covars_ = _identity(covariance_type, nc, nm, nf)
    return h


def _fit_bytes(model, X):
    """The fitted arrays, the five priors and the score, as bytes."""
    arrays = {
        name: value
        for name, value in vars(model).items()
        if name.endswith("_") and name not in ("monitor_", "state_occupancy_")
    }
    arrays.update({name: getattr(model, name) for name in PRIORS})
    arrays["score"] = model.score(X)
    return {
        name: (np.asarray(v).dtype.str, np.shape(v), np.asarray(v).tobytes())
        for name, v in arrays.items()
    }


def test_type_change_after_a_refusal_keeps_the_filled_covars_weight():
    """
    A ``tied`` model refused on ``X`` of another width, then changed to
    ``full`` with parameters of that width, fits with the ``covars_weight``
    the refused call filled in, ``-(n_mix + n_features + 1)`` for four
    features: as a ``full`` model given that number.
    """
    rs = np.random.RandomState(0)
    model = _started("tied", rs.normal(size=(1, 2, 3)))
    X4 = rs.normal(size=(50, 4))
    with pytest.raises(ValueError):
        model.fit(X4)
    means = rs.normal(size=(1, 2, 4))
    model.covariance_type = "full"
    model.means_ = means
    model.covars_ = np.tile(np.eye(4), (1, 2, 1, 1))
    model.fit(X4)
    expected = _started(
        "full", means, covars_prior=0.0, covars_weight=-7.0
    ).fit(X4)
    assert _fit_bytes(model, X4) == _fit_bytes(expected, X4)


@pytest.mark.parametrize(
    "covariance_type, covars_weight", [("full", -3.0), ("tied", -4.0)]
)
def test_refit_at_another_width_keeps_the_filled_covars_weight(
    covariance_type, covars_weight
):
    """
    A model refused on one column, given ``n_features`` of its means and
    fitted on ``X`` of that width, fits with the ``covars_weight`` filled in
    for one feature: ``-(1 + 1 + 1)`` for ``full``, ``-(n_mix + 1 + 1)``
    for ``tied``.
    """
    rs = np.random.RandomState(0)
    means = rs.normal(size=(2, 2, 3))
    model = _started(covariance_type, means)
    X1 = rs.normal(size=(50, 1))
    X3 = rs.normal(size=(50, 3))
    with pytest.raises(ValueError):
        model.fit(X1)
    model.n_features = 3
    model.fit(X3)
    expected = _started(
        covariance_type, means, covars_prior=0.0, covars_weight=covars_weight
    ).fit(X3)
    assert _fit_bytes(model, X3) == _fit_bytes(expected, X3)


@pytest.mark.parametrize(
    "covariance_type, n_mix, X_width, priors, error, covars_prior, "
    "covars_weight",
    [
        ("full", 2, 4, {}, ValueError, 0.0, -6.0),
        ("tied", 2, 4, {}, ValueError, 0.0, -7.0),
        ("diag", 2, 4, {}, ValueError, -1.5, 0.0),
        ("spherical", 2, 4, {}, ValueError, -2.0, 0.0),
        ("diag", 2, 3, {"means_prior": np.zeros(5)}, ValueError, -1.5, 0.0),
        # the start counts the parameters of each known type
        ("bogus", 2, 3, {}, KeyError, None, None),
        # scikit-learn's KMeans refuses n_clusters=0
        ("diag", 0, 3, {}, ValueError, None, None),
    ],
    ids=[
        "full-width",
        "tied-width",
        "diag-width",
        "spherical-width",
        "diag-means_prior",
        "bogus-type",
        "diag-no-mixture-component",
    ],
)
def test_refused_fit_leaves_a_filled_covariance_prior_as_its_number(
    covariance_type, n_mix, X_width, priors, error, covars_prior, covars_weight
):
    """
    A ``fit`` refused after it filled in a ``None`` ``covars_prior`` or
    ``covars_weight`` leaves the number it filled in, a float without a
    shape: refused on the width of ``X``, or by another prior. A
    ``covariance_type`` that fills nothing leaves both ``None``, and so
    does ``n_mix=0``, whose priors are broadcast to an empty shape; the
    caller gets the error of the refusal.
    """
    rs = np.random.RandomState(0)
    means = rs.normal(size=(2, 2, 3))
    valid = "diag" if covariance_type == "bogus" else covariance_type
    model = _started(valid, means, **priors)
    model.covariance_type = covariance_type
    model.n_mix = n_mix
    with pytest.raises(error):
        model.fit(rs.normal(size=(50, X_width)))
    for name, number in [
        ("covars_prior", covars_prior),
        ("covars_weight", covars_weight),
    ]:
        value = getattr(model, name)
        if number is None:
            assert value is None
        else:
            assert type(value) is float
            assert value == number
