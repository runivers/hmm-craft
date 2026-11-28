import re

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from .. import hmm
from . import make_covar_matrix, normalized

# n_components != n_features, as in test_gmm_hmm.py, so that an array of
# length n_components cannot broadcast along the features.
NC, NF = 3, 2
COVARS_SHAPES = {
    "spherical": (NC,),
    "diag": (NC, NF),
    "full": (NC, NF, NF),
    "tied": (NF, NF),
}
# the shape covars_prior must broadcast to, per covariance type
COVARS_PRIOR_SHAPES = {
    "spherical": (NC, NF),
    "diag": (NC, NF),
    "full": (NC, NF, NF),
    "tied": (NF, NF),
}


def _start(covariance_type, n_features=NF):
    prng = np.random.RandomState(10)
    startprob = normalized(prng.rand(NC))
    transmat = normalized(prng.rand(NC, NC), axis=1)
    means = prng.randint(-20, 20, (NC, n_features)).astype(float)
    covars = make_covar_matrix(
        covariance_type, NC, n_features, random_state=prng
    )
    assert covars is not None
    if covariance_type in ("spherical", "diag"):
        covars = np.maximum(covars, 0.1)
    return prng, startprob, transmat, means, covars


def _model(covariance_type, priors, n_features=NF, params="stmc", n_iter=1):
    """A model at a fixed start and a seeded sample drawn from it."""
    prng, startprob, transmat, means, covars = _start(
        covariance_type, n_features
    )
    h = hmm.GaussianHMM(
        NC,
        covariance_type,
        init_params="",
        params=params,
        n_iter=n_iter,
        **priors,
    )
    h.startprob_, h.transmat_ = startprob, transmat
    h.means_, h.covars_ = means, covars
    X, _ = h.sample(200, random_state=prng)
    return h, X


def _fit(covariance_type, **priors):
    """One EM iteration from a fixed start on a seeded sample."""
    h, X = _model(covariance_type, priors)
    return h.fit(X)


def _fitted_state(h):
    """The fitted arrays and the history of a model, as bytes."""
    return [
        (np.shape(v), np.asarray(v).tobytes())
        for v in [
            h.startprob_,
            h.transmat_,
            h.means_,
            h._covars_,
            list(h.monitor_.history),
        ]
    ]


# (covariance_type, prior, shape of an array filled with one number)
FILLED = [
    *[
        (ct, "means_prior", shape)
        for ct in COVARS_SHAPES
        for shape in [(NC, NF), (NC, 1), (NF,), (1,), (1, NF)]
    ],
    *[
        (ct, name, shape)
        for ct in ["spherical", "diag"]
        for name in ["means_weight", "covars_prior"]
        for shape in [(NC, NF), (NC, 1), (NF,)]
    ],
    *[
        (ct, "means_weight", shape)
        for ct in ["full", "tied"]
        for shape in [(1,), (1, 1)]
    ],
    *[
        (ct, "covars_weight", shape)
        for ct in COVARS_SHAPES
        for shape in [(1,), (1, 1)]
    ],
    ("full", "covars_weight", (1, 1, 1)),
    ("tied", "covars_weight", (1, 1, 1)),
    *[
        ("full", "covars_prior", shape)
        for shape in [
            (NF, NF),
            (NC, NF, NF),
            (NF,),
            (1, NF),
            (1, 1, 1),
            (1, NF, NF),
            (NC, 1, 1),
        ]
    ],
    *[
        ("tied", "covars_prior", shape)
        for shape in [(NF, NF), (1, NF, NF), (1, 1, 1), (1, 1, NF)]
    ],
]
# the number each prior is filled with: means_weight as in
# test_fit_with_priors, covars_prior its default
NUMBERS = {"means_prior": 1.0, "means_weight": 2.0, "covars_prior": 1e-2}


def _number(covariance_type, prior):
    """
    The number a prior is filled with; covars_weight above the threshold
    of max(w - 1, 0) ("spherical", "diag") or max(w - n_features, 0)
    ("full", "tied"), so the fit reads it.
    """
    if prior == "covars_weight":
        return 3.0 if covariance_type in ("spherical", "diag") else NF + 2.0
    return NUMBERS[prior]


@pytest.mark.parametrize(
    "covariance_type, prior, shape",
    FILLED,
    ids=lambda v: str(v).replace(" ", ""),
)
def test_documented_prior_array_fits_like_its_number(
    covariance_type, prior, shape
):
    """
    Each documented array form of a prior fits, and an array filled with
    one number gives the fit of that number, so a number for "full" or
    "tied" covars_prior is added to every entry. An array with one element
    counts as a number. A "tied" prior of three dimensions gives the
    covariance of the number with a leading axis of length one, shape
    (1, n_features, n_features).
    """
    others = {"means_weight": 2.0} if prior == "means_prior" else {}
    number = _number(covariance_type, prior)
    ref = _fit(covariance_type, **others, **{prior: number})
    h = _fit(covariance_type, **others, **{prior: np.full(shape, number)})
    assert ref.means_.shape == (NC, NF)
    assert ref._covars_.shape == COVARS_SHAPES[covariance_type]
    assert_array_equal(h.means_, ref.means_)
    covars = h._covars_
    if covariance_type == "tied" and len(shape) == 3:
        assert covars.shape == (1, NF, NF)
        covars = covars[0]
    assert_array_equal(covars, ref._covars_)


@pytest.mark.parametrize(
    "covariance_type, prior",
    [
        *[
            (ct, name)
            for ct in ["spherical", "diag"]
            for name in ["means_prior", "means_weight", "covars_prior"]
        ],
        ("full", "means_prior"),
        ("tied", "means_prior"),
    ],
)
def test_prior_column_gives_one_value_per_state(covariance_type, prior):
    """
    A column of shape (n_components, 1) applies its row k to state k: after
    one iteration from the same start, state k's parameters are those of
    the fit with the number in row k.
    """
    others = {"means_weight": 2.0} if prior == "means_prior" else {}
    column = {
        "means_prior": [0.0, 1.0, 2.0],
        "means_weight": [0.0, 2.0, 4.0],
        "covars_prior": [1e-2, 1.0, 2.0],
    }[prior]
    h = _fit(covariance_type, **others, **{prior: np.array(column)[:, None]})
    for k, number in enumerate(column):
        ref = _fit(covariance_type, **others, **{prior: number})
        assert_array_equal(h.means_[k], ref.means_[k])
        if covariance_type != "tied":  # one covariance for all states
            assert_array_equal(h._covars_[k], ref._covars_[k])


def _broadcast(name, shape, value):
    return (
        f"{name} must broadcast to shape {shape}, "
        f"actual shape: {np.shape(value)}"
    )


def _number_message(name, value, covariance_type=None):
    where = f' with covariance_type="{covariance_type}"'
    return (
        f"{name} must be a number{where if covariance_type else ''}, "
        f"actual shape: {np.shape(value)}"
    )


def _symmetric(covariance_type):
    return (
        "covars_prior must be symmetric with "
        f'covariance_type="{covariance_type}"'
    )


def _refused(
    covariance_type, message, priors, n_features=NF, params="stmc", tag=""
):
    """
    A row of priors fit refuses with *message*; its id reads
    ``<type>-<prior>-shape-<dims>``.
    """
    name, value = next(iter(priors.items()))
    dims = "x".join(map(str, np.shape(value))) or "scalar"
    parts = [covariance_type, name, "shape", dims]
    if n_features != NF:
        parts.append(f"{n_features}-feature")
    if params != "stmc":
        parts.append(params)
    if tag:
        parts.append(tag)
    return pytest.param(
        covariance_type,
        priors,
        message,
        n_features,
        params,
        id="-".join(parts),
    )


def _assert_refused(covariance_type, priors, message, n_features, params):
    """fit raises *message* exactly and leaves the start as it was."""
    h, X = _model(covariance_type, priors, n_features, params)
    means, covars = np.array(h.means_), h._covars_.copy()
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        h.fit(X)
    assert_array_equal(h.means_, means)
    assert_array_equal(h._covars_, covars)


def _nonsymmetric(shape):
    """Distinct values, so the matrices are not symmetric."""
    return 0.1 * (1 + np.arange(np.prod(shape))).reshape(shape)


ROW = np.array([1.0, 2.0])  # one value per feature
_ones_3d = np.ones((NC, 1, 1))
_leading = np.full((1, NC, NF), 1e-2)
_weight_3d = np.full((1, 1, 1), 3.0)
_by_state = np.arange(1.0, NC + 1)

RAISES = [
    # an array of shape (n_components,) is matched to the features
    *[
        _refused(
            ct,
            _broadcast("means_prior", (NC, NF), np.zeros(NC)),
            priors={"means_prior": np.full(NC, 1.0)},
        )
        for ct in COVARS_SHAPES
    ],
    *[
        _refused(
            ct,
            _broadcast("means_weight", (NC, NF), np.zeros(NC)),
            priors={"means_weight": np.full(NC, 2.0)},
        )
        for ct in ["spherical", "diag"]
    ],
    *[
        _refused(
            ct,
            _broadcast("covars_prior", COVARS_PRIOR_SHAPES[ct], np.zeros(NC)),
            priors={"covars_prior": np.full(NC, 1e-2)},
        )
        for ct in COVARS_SHAPES
    ],
    # "full" and "tied" means_weight, and covars_weight, are numbers
    *[
        _refused(
            ct,
            _number_message("means_weight", value, ct),
            priors={"means_weight": value},
        )
        for ct in ["full", "tied"]
        for value in [
            np.full(NC, 2.0),
            np.full((NC, NF), 2.0),
            np.full((NC, 1), 2.0),
            ROW,
            np.full((1, 1, 1), 2.0),
        ]
    ],
    *[
        _refused(
            ct,
            _number_message("covars_weight", value),
            priors={"covars_weight": value},
        )
        for ct in COVARS_SHAPES
        for value in [np.full(NC, 2.0), np.full((NC, 1), 2.0)]
    ],
    *[
        _refused(
            ct,
            _number_message("covars_weight", _weight_3d),
            priors={"covars_weight": _weight_3d},
        )
        for ct in ["spherical", "diag"]
    ],
    # one covariance per state for "tied"
    _refused(
        "tied",
        _broadcast("covars_prior", (NF, NF), np.zeros((NC, NF, NF))),
        priors={"covars_prior": np.tile(np.eye(NF), (NC, 1, 1))},
    ),
    _refused(
        "tied",
        _broadcast("covars_prior", (NF, NF), _ones_3d),
        priors={"covars_prior": _ones_3d},
    ),
    # a "full" or "tied" covars_prior that is not symmetric
    *[
        _refused(ct, _symmetric(ct), priors={"covars_prior": ROW})
        for ct in ["full", "tied"]
    ],
    *[
        _refused(
            ct,
            _symmetric(ct),
            tag="nonsymmetric",
            priors={"covars_prior": _nonsymmetric(shape)},
        )
        for ct, shape in [
            ("full", (NF, NF)),
            ("full", (NC, NF, NF)),
            ("tied", (NF, NF)),
        ]
    ],
    # an added axis for "spherical" and "diag"
    *[
        _refused(
            ct,
            _broadcast("covars_prior", (NC, NF), value),
            priors={"covars_prior": value},
        )
        for ct in ["spherical", "diag"]
        for value in [_ones_3d, _leading, np.full((1, 1, 1), 1e-2)]
    ],
    # one feature: n_components values are not one per state
    _refused(
        "diag",
        _broadcast("means_prior", (NC, 1), np.zeros(NC)),
        n_features=1,
        priors={"means_prior": np.zeros(NC), "means_weight": np.full(NC, 2.0)},
    ),
    *[
        _refused(
            ct,
            _broadcast("covars_prior", (NC, 1), value),
            n_features=1,
            priors={"covars_prior": value},
        )
        for ct in ["spherical", "diag"]
        for value in [_by_state, _by_state[None], np.tile(_by_state, (NC, 1))]
    ],
    *[
        _refused(
            "spherical",
            _broadcast(name, (NC, 1), value),
            n_features=1,
            params="stc",
            priors={name: value},
        )
        for name in ["means_prior", "means_weight"]
        for value in [_by_state, _by_state[None]]
    ],
    # a "tied" means_weight of n_features x n_features entries
    *[
        _refused(
            "tied",
            _number_message("means_weight", value, "tied"),
            params="stc",
            priors={"means_weight": value},
        )
        for value in [
            np.array([[1.0, 2.0], [2.0, 3.0]]),
            np.array([[[1.0, 2.0], [2.0, 3.0]]]),
        ]
    ],
]


@pytest.mark.parametrize(
    "covariance_type, priors, message, n_features, params", RAISES
)
def test_undocumented_prior_shape_raises(
    covariance_type, priors, message, n_features, params
):
    """
    A prior the M-step reads under ``params`` that does not broadcast, as
    np.broadcast_to does, to the shape the constructor documents, or a
    "full" or "tied" covars_prior that is not symmetric, or an array of
    more than one element where a number is documented, makes fit raise a
    ValueError naming the prior, before the start is changed.
    """
    _assert_refused(covariance_type, priors, message, n_features, params)


def _constant(shape, number=2.0):
    return np.full(shape, number)


CONSTANT_ARRAY_WHERE_NUMBER = [
    *[
        _refused(
            ct,
            _number_message("means_weight", _constant(shape), ct),
            params=params,
            priors={"means_weight": _constant(shape)},
        )
        for ct in ["full", "tied"]
        for params, shape in [
            ("stmc", (NF,)),
            ("stmc", (1, NF)),
            ("stc", (NF,)),
            ("stc", (NF, 1)),
            ("stc", (NF, NF)),
            ("stc", (1, 1, NF)),
            ("stc", (1, NF, NF)),
        ]
    ],
    *[
        _refused(
            "spherical",
            _broadcast(name, (NC, 1), _constant(shape)),
            n_features=1,
            params="stc",
            priors={name: _constant(shape)},
        )
        for name in ["means_prior", "means_weight"]
        for shape in [(NC,), (1, NC), (NC, NC)]
    ],
]


@pytest.mark.parametrize(
    "covariance_type, priors, message, n_features, params",
    CONSTANT_ARRAY_WHERE_NUMBER,
)
def test_constant_array_of_an_undocumented_shape_raises(
    covariance_type, priors, message, n_features, params
):
    """
    The shape decides, not the values: an array whose entries are all
    equal is refused where its shape is, although its number is accepted.
    """
    _assert_refused(covariance_type, priors, message, n_features, params)


ONE_ELEMENT_WEIGHT_THREE_DIMS = [
    _refused(
        ct,
        _number_message("means_weight", _constant(shape), ct),
        n_features=n_features,
        params="stc",
        priors={"means_weight": _constant(shape)},
    )
    for ct in ["full", "tied"]
    for n_features in [NF, 1]
    for shape in [(1, 1, 1), (1, 1, 1, 1)]
]


@pytest.mark.parametrize(
    "covariance_type, priors, message, n_features, params",
    ONE_ELEMENT_WEIGHT_THREE_DIMS,
)
def test_one_element_weight_of_three_dimensions_raises(
    covariance_type, priors, message, n_features, params
):
    """
    A "full" or "tied" means_weight is a number of at most two dimensions,
    whether or not ``params`` updates the means.
    """
    _assert_refused(covariance_type, priors, message, n_features, params)


@pytest.mark.parametrize(
    "covariance_type, prior, value",
    [
        ("tied", "covars_prior", np.tile(np.eye(NF), (NC, 1, 1))),
        ("full", "covars_prior", ROW),
        ("full", "means_weight", ROW),
        ("spherical", "covars_prior", np.ones((NC, 1, 1))),
        ("diag", "covars_weight", np.full(NC, 2.0)),
    ],
    ids=lambda v: v if isinstance(v, str) else "x".join(map(str, v.shape)),
)
def test_refused_prior_raises_with_n_iter_zero(covariance_type, prior, value):
    """
    The priors are tested whatever ``n_iter`` is: a fit with n_iter=0,
    which runs no M-step, refuses them as a fit with one iteration does.
    """
    h, X = _model(covariance_type, {prior: value}, n_iter=0)
    with pytest.raises(ValueError, match=f"^{prior} must "):
        h.fit(X)


@pytest.mark.parametrize("covariance_type", ["full", "tied"])
def test_prior_close_to_symmetric_fits(covariance_type):
    """
    The symmetry test is np.allclose's: a covars_prior one ulp from
    symmetric fits, close to the fit of its symmetric copy, and so does a
    prior built by make_covar_matrix, which is symmetric to rounding.
    """
    prior = np.array([[1.0, 0.5], [0.5, 2.0]])
    moved = prior.copy()
    moved[0, 1] = np.nextafter(moved[0, 1], 1.0)
    assert not np.array_equal(moved, moved.T)
    ref = _fit(covariance_type, covars_prior=prior)
    h = _fit(covariance_type, covars_prior=moved)
    assert_allclose(h._covars_, ref._covars_, rtol=1e-12)

    built = make_covar_matrix(covariance_type, NC, NF, random_state=0)
    assert built is not None
    assert not np.array_equal(built, np.swapaxes(built, -1, -2))
    h = _fit(covariance_type, covars_prior=built)
    assert h._covars_.shape == COVARS_SHAPES[covariance_type]


@pytest.mark.parametrize(
    "covariance_type, params, priors, same_as",
    [
        ("spherical", "stm", {"covars_prior": np.ones((NC, 1, 1))}, {}),
        ("diag", "stm", {"covars_weight": np.full(NC, 2.0)}, {}),
        ("tied", "stm", {"covars_prior": np.ones((NC, NF, NF))}, {}),
        ("diag", "st", {"means_prior": np.zeros(NC)}, {}),
        ("full", "st", {"means_weight": np.full(NC, 2.0)}, {}),
        (
            "full",
            "stm",
            {"means_weight": np.full((NC, NF), 2.0)},
            {"means_weight": 2.0},
        ),
    ],
    ids=[
        "spherical-covars_prior-stm",
        "diag-covars_weight-stm",
        "tied-covars_prior-stm",
        "diag-means_prior-st",
        "full-means_weight-st",
        "full-means_weight-by-state-stm",
    ],
)
def test_prior_the_fit_does_not_read_is_not_tested(
    covariance_type, params, priors, same_as
):
    """
    A prior the M-step does not read under ``params`` is not tested: the
    fit has the bytes of the fit without it. A "full" means_weight with
    "m" and not "c" weighs the means only, where a value per state and
    feature is valid, and gives the fit of its number when all are equal.
    """
    ref, X = _model(covariance_type, same_as, params=params)
    ref.fit(X)
    h, X = _model(covariance_type, priors, params=params)
    h.fit(X)
    assert _fitted_state(h) == _fitted_state(ref)


PREDICTED_WITH = [
    ("tied", "covars_prior", np.tile(np.eye(NF), (NC, 1, 1))),
    ("full", "covars_prior", ROW),
    ("tied", "covars_prior", ROW),
    ("full", "means_weight", ROW),
    ("tied", "means_weight", ROW),
    ("spherical", "covars_prior", np.ones((NC, 1, 1))),
    ("diag", "covars_prior", np.ones((NC, 1, 1))),
    ("diag", "covars_weight", np.full(NC, 2.0)),
]


@pytest.mark.parametrize(
    "covariance_type, prior, value",
    PREDICTED_WITH,
    ids=[
        f"{ct}-{name}-shape-{'x'.join(map(str, np.shape(v)))}"
        for ct, name, v in PREDICTED_WITH
    ],
)
def test_prediction_ignores_a_refused_prior(covariance_type, prior, value):
    """
    No prediction reads the priors: a fitted model given a prior fit
    refuses scores, decodes, predicts and samples with the bytes of the
    same model without it.
    """
    h, X = _model(covariance_type, {})
    h.fit(X)

    def predictions():
        logprob, states = h.decode(X)
        map_logprob, map_states = h.decode(X, algorithm="map")
        sample, sample_states = h.sample(5, random_state=0)
        return [
            np.float64(h.score(X)).tobytes(),
            np.float64(logprob).tobytes(),
            states.tobytes(),
            np.float64(map_logprob).tobytes(),
            map_states.tobytes(),
            h.predict_proba(X).tobytes(),
            sample.tobytes(),
            sample_states.tobytes(),
        ]

    expected = predictions()
    setattr(h, prior, value)
    assert predictions() == expected


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
