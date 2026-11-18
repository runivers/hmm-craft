"""The E-step by blocks of sequences, ``_AbstractHMM._do_estep``.

The package's model classes compute the E-step for blocks of consecutive
sequences: one call of the kernel's block function per block, then the
statistics of each sequence in ``lengths`` order. A subclass that replaces
an E-step method, and every input the block path is not proven for, are
computed sequence by sequence, as hmmlearn 0.3.3 does. Both paths give the
same fitted arrays, history, warnings and log records, as bytes.
"""

import functools
import logging
import os
import pickle
import sys
import warnings

import numpy as np
import pytest
from scipy import linalg, special

from hmmcraft import _emissions, _hmmc, base, hmm, stats, vhmm
from hmmcraft.base import BaseHMM

from .test_base import (
    IMPOSSIBLE,
    UNDERFLOW_FIT_CASES,
    UNDERFLOW_FIT_IDS,
    UNDERFLOW_TEXT,
    _impossible,
)

# Hand-chosen lengths, one-row sequences among them: 105 rows.
# With 16 rows per block, the first two blocks end exactly on the bound and
# the sequences of 21 and 34 rows are blocks alone.
LENGTHS = [1, 2, 3, 5, 5, 8, 8, 13, 1, 1, 21, 34, 1, 2]
N_ROWS = sum(LENGTHS)
# The blocks of LENGTHS at 16 rows, for two states.
BLOCKS_AT_16 = [(0, 5), (5, 7), (7, 10), (10, 11), (11, 12), (12, 14)]

# numpy's default error state; this test package sets every report to "warn".
DEFAULT_STATE = {
    "divide": "warn",
    "over": "warn",
    "under": "ignore",
    "invalid": "warn",
}
REPORT_STATES = [{"all": "warn"}, {"all": "call"}, {"all": "raise"}]
REPORT_IDS = ["all=warn", "all=call", "all=raise"]
KERNEL = ["estep_log", "estep_scaling", "forward_log", "forward_scaling"]
COVARIANCE_TYPES = ["spherical", "diag", "full", "tied"]
N_ITER = 3

ESTEP_HOOKS = {
    "_do_estep",
    "_compute_estep_factors",
    "_by_implementation",
    "_initialize_sufficient_statistics",
    "_estep_begin",
    "_fit_log",
    "_fit_scaling",
    "_compute_likelihood",
    "_compute_log_likelihood",
    "_compute_subnorm_likelihood",
    "_compute_subnorm_log_likelihood",
    "_compute_log_weighted_gaussian_densities",
    "_mixture_log_likelihoods",
    "_hooks_are_own",
    "_compute_posteriors_log",
    "_compute_posteriors_scaling",
    "_accumulate_sufficient_statistics",
    "_accumulate_sufficient_statistics_log",
    "_accumulate_sufficient_statistics_scaling",
    "_needs_sufficient_statistics_for_mean",
    "_needs_sufficient_statistics_for_covars",
}


def _data(n_rows=N_ROWS):
    """Seeded continuous data, symbols, Poisson and multinomial counts."""
    rs = np.random.RandomState(10)
    continuous = rs.normal(size=(n_rows, 2)) + 3 * rs.randint(
        0, 2, (n_rows, 1)
    )
    symbols = rs.randint(0, 3, (n_rows, 1))
    # int64 on every platform: RandomState returns C long, which is 32 bits
    # on Windows, and the shorter Poisson and multinomial forms take int64.
    counts = rs.poisson(4, (n_rows, 3)).astype(np.int64)
    multinomial = rs.multinomial(
        5, [0.2, 0.3, 0.5], size=n_rows).astype(np.int64)
    return continuous, symbols, counts, multinomial


def _variants():
    """(id, factory(n_components, implementation), X): 16 model variants."""
    continuous, symbols, counts, multinomial = _data()
    common = {"n_iter": N_ITER, "random_state": 0}
    variants = [
        (
            f"GaussianHMM-{cov}",
            lambda n, impl, cov=cov: hmm.GaussianHMM(
                n, cov, implementation=impl, **common
            ),
            continuous,
        )
        for cov in COVARIANCE_TYPES
    ]
    variants += [
        (
            f"GMMHMM-{cov}",
            lambda n, impl, cov=cov: hmm.GMMHMM(
                n, n_mix=2, covariance_type=cov, implementation=impl, **common
            ),
            continuous,
        )
        for cov in COVARIANCE_TYPES
    ]
    variants += [
        (
            "CategoricalHMM",
            lambda n, impl: hmm.CategoricalHMM(
                n, implementation=impl, **common
            ),
            symbols,
        ),
        (
            "MultinomialHMM",
            lambda n, impl: hmm.MultinomialHMM(
                n, n_trials=5, implementation=impl, **common
            ),
            multinomial,
        ),
        (
            "PoissonHMM",
            lambda n, impl: hmm.PoissonHMM(n, implementation=impl, **common),
            counts,
        ),
        (
            "VariationalCategoricalHMM",
            lambda n, impl: vhmm.VariationalCategoricalHMM(
                n, implementation=impl, **common
            ),
            symbols,
        ),
    ]
    variants += [
        (
            f"VariationalGaussianHMM-{cov}",
            lambda n, impl, cov=cov: vhmm.VariationalGaussianHMM(
                n, covariance_type=cov, implementation=impl, **common
            ),
            continuous,
        )
        for cov in COVARIANCE_TYPES
    ]
    return variants


VARIANTS = _variants()
VARIANT_IDS = [variant[0] for variant in VARIANTS]


def _variant(name):
    return VARIANTS[VARIANT_IDS.index(name)]


def per_sequence(model):
    """
    ``model`` as an instance of a subclass whose ``_do_estep`` calls the
    package's: the guard refuses it, so the E-step runs sequence by sequence.
    """

    class PerSequence(type(model)):
        def _do_estep(self, X, lengths):
            return super()._do_estep(X, lengths)

    model.__class__ = PerSequence
    return model


def _fit_state(model):
    """
    Every entry of ``vars(model)`` ending in ``_`` but ``monitor_`` (dtype,
    shape, bytes), the history as bytes with the type of each value, and the
    number of iterations.
    """
    state = {}
    for name, value in vars(model).items():
        if name.endswith("_") and name != "monitor_":
            array = np.asarray(value)
            state[name] = (array.dtype.str, array.shape, array.tobytes())
    history = list(model.monitor_.history)
    state["monitor_.history"] = np.array(history, dtype=float).tobytes()
    state["monitor_.history types"] = [type(v).__name__ for v in history]
    state["monitor_.iter"] = model.monitor_.iter
    return state


class _Records(logging.Handler):
    """Keeps (level, message) of each record it is given."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.seen = []

    def emit(self, record):
        self.seen.append((record.levelname, record.getMessage()))


def _error(error):
    return (
        "error",
        type(error).__name__,
        str(error),
        error.__context__ is None,
    )


def _run(call, state):
    """
    What ``call()`` gives under the numpy error ``state``: its result or
    the error (type, text, whether it has no context); the warnings
    (category, text, file name) in order; the calls of an error handler;
    the records of ``hmmcraft.base``.
    """
    handled = []
    records = _Records()
    logger = logging.getLogger("hmmcraft.base")
    logger.addHandler(records)
    try:
        with (
            warnings.catch_warnings(record=True) as caught,
            np.errstate(call=lambda kind, flag: handled.append(kind), **state),
        ):
            warnings.simplefilter("always")
            try:
                result = call()
            except (
                FloatingPointError,
                ValueError,
                np.linalg.LinAlgError,
            ) as error:
                result = _error(error)
    finally:
        logger.removeHandler(records)
    reports = [
        (w.category.__name__, str(w.message), os.path.basename(w.filename))
        for w in caught
    ]
    return result, reports, handled, records.seen


def _fitted(model, X, lengths=LENGTHS, state=DEFAULT_STATE):
    """What ``model.fit(X, lengths)`` gives: its state, or its error."""

    def fit():
        model.fit(X, lengths)
        return _fit_state(model)

    return _run(fit, state)


def _estep(model, X, lengths, state):
    """What one ``_do_estep`` gives: the statistics and the log-likelihood
    as bytes, or its error."""

    def estep():
        stats, logprob = model._do_estep(X, lengths)
        got = {
            name: np.asarray(value).tobytes() for name, value in stats.items()
        }
        got["logprob"] = (
            type(logprob).__name__,
            np.float64(logprob).tobytes(),
        )
        return got

    return _run(estep, state)


@pytest.fixture
def kernel_calls(monkeypatch):
    """Record each call of the kernel's pass functions: (name, the block
    functions' ``with_xi_sum`` or None)."""
    calls = []
    for name in KERNEL:
        real = getattr(_hmmc, name)

        def counting(*args, _real=real, _name=name):
            with_xi_sum = args[4] if _name.startswith("estep") else None
            calls.append((_name, with_xi_sum))
            return _real(*args)

        monkeypatch.setattr(_hmmc, name, counting)
    return calls


@pytest.fixture
def rows_16(monkeypatch):
    """Blocks of at most 16 rows."""
    monkeypatch.setattr(base, "_ESTEP_BLOCK_ROWS", 16, raising=False)


def _count(calls, prefix):
    return sum(1 for name, _ in calls if name.startswith(prefix))


# The guard's hook list and the blocks ---------------------------------------


def test_e_step_hooks_are_the_methods_the_loop_calls():
    """The guard compares every method the per-sequence E-step calls on the
    model, 0.3.3's names included, and the one the E-step by blocks calls
    before its first block, ``_compute_estep_factors``; each once."""
    assert set(base._ESTEP_HOOKS) == ESTEP_HOOKS
    assert len(base._ESTEP_HOOKS) == len(ESTEP_HOOKS)


@pytest.mark.parametrize(
    "lengths, block_rows, n_components, expected",
    [
        ([1, 2, 3, 5, 5, 8, 8], 16, 1, [(0, 5), (5, 7)]),
        ([13, 1, 1, 21, 34, 1, 2], 16, 1, [(0, 3), (3, 4), (4, 5), (5, 7)]),
        ([1, 1, 1, 1, 1], 6, 3, [(0, 2), (2, 4), (4, 5)]),
        ([1, 1, 1], 2, 3, [(0, 1), (1, 2), (2, 3)]),
        ([7, 7], 1, 1, [(0, 1), (1, 2)]),
        ([5], 4096, 3, [(0, 1)]),
        (LENGTHS, 16, 2, BLOCKS_AT_16),
    ],
    ids=[
        "rows-bound-reached",
        "longer-sequences-alone",
        "sequence-bound",
        "bound-below-one-sequence",
        "sequence-longer-than-a-block",
        "one-sequence",
        "check-cycle",
    ],
)
def test_blocks_hold_whole_sequences_in_order(
    monkeypatch, lengths, block_rows, n_components, expected
):
    """A block takes the next sequence while its rows stay within the
    bound and it has fewer sequences than the bound divided by the number
    of states; it always takes at least one sequence."""
    monkeypatch.setattr(base, "_ESTEP_BLOCK_ROWS", block_rows, raising=False)
    assert base._estep_blocks(lengths, n_components) == expected


# Which path a fit takes ------------------------------------------------------


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
def test_a_model_class_of_the_package_fits_by_blocks(
    kernel_calls, rows_16, variant, implementation
):
    """Each model class of the package calls the kernel's block function
    once per block and iteration, without the transition sums, and never a
    per-sequence pass function."""
    _, factory, X = variant
    model = factory(2, implementation)
    result, *_ = _fitted(model, X)
    assert isinstance(result, dict)
    assert kernel_calls == [("estep_" + implementation, False)] * (
        len(BLOCKS_AT_16) * model.monitor_.iter
    )


def _replaced_by_subclass(model, name):
    """A subclass of the model's class whose method ``name`` calls the
    class's own."""

    def method(self, *args, **kwargs):
        return getattr(super(sub, self), name)(*args, **kwargs)

    sub = type(f"Replaced{name}", (type(model),), {name: method})
    model.__class__ = sub
    return model


def _replaced_on_instance(model, name):
    """The model with an instance attribute ``name`` that calls the class's
    method."""
    bound = getattr(model, name)

    @functools.wraps(bound)
    def method(*args, **kwargs):
        return bound(*args, **kwargs)

    setattr(model, name, method)
    return model


def _hook_owner(name):
    """A model variant whose class has the hook ``name``."""
    if name in (
        "_compute_log_weighted_gaussian_densities",
        "_mixture_log_likelihoods",
        "_hooks_are_own",
    ):
        return _variant("GMMHMM-diag")
    if name.startswith("_compute_subnorm"):
        return _variant("VariationalCategoricalHMM")
    if name == "_compute_estep_factors":
        return _variant("GaussianHMM-full")
    return _variant("GaussianHMM-diag")


def _hook_implementation(name):
    """The implementation under which the hook ``name`` is called."""
    if "scaling" in name or (
        name.endswith("_likelihood") and "log" not in name
    ):
        return "scaling"
    return "log"


HOOK_CASES = [
    (name, how)
    for name in sorted(ESTEP_HOOKS)
    for how in ("subclass", "instance")
]


@pytest.mark.parametrize(
    "name, how", HOOK_CASES, ids=[f"{n}-{h}" for n, h in HOOK_CASES]
)
def test_a_replaced_hook_keeps_the_per_sequence_loop(
    kernel_calls, rows_16, name, how
):
    """A model whose E-step method is replaced, by a subclass or on the
    instance, calls a pass function once per sequence and iteration and no
    block function, and fits as the package's class does."""
    _, factory, X = _hook_owner(name)
    implementation = _hook_implementation(name)
    plain = _fitted(factory(2, implementation), X)
    replace = {
        "subclass": _replaced_by_subclass,
        "instance": _replaced_on_instance,
    }[how]
    model = replace(factory(2, implementation), name)
    del kernel_calls[:]
    got = _fitted(model, X)
    assert _count(kernel_calls, "estep") == 0
    assert _count(kernel_calls, "forward") == (
        len(LENGTHS) * model.monitor_.iter
    )
    assert got == plain


class _OwnLikelihood(BaseHMM):
    """A model derived from BaseHMM with its own likelihood."""

    def _get_n_fit_scalars_per_param(self):
        return None

    def _compute_log_likelihood(self, X):
        return -0.5 * (X - np.arange(self.n_components)) ** 2


def test_a_model_without_an_e_step_owner_keeps_the_loop(kernel_calls):
    """A class of no model of the package is computed sequence by
    sequence."""
    X = _data()[0][:, :1]
    model = _OwnLikelihood(2, n_iter=N_ITER, random_state=0)
    result, *_ = _fitted(model, X)
    assert isinstance(result, dict)
    assert _count(kernel_calls, "estep") == 0
    assert _count(kernel_calls, "forward") == (
        len(LENGTHS) * model.monitor_.iter
    )


def test_more_states_than_numpy_buffer_keep_the_loop(kernel_calls):
    """A model with more states than numpy's buffer holds is computed
    sequence by sequence."""
    _, factory, X = _variant("CategoricalHMM")
    old = np.setbufsize(16)
    try:
        plain = _fitted(factory(17, "log"), X)
        assert _count(kernel_calls, "estep") == 0
        assert _count(kernel_calls, "forward") > 0
        loop = _fitted(per_sequence(factory(17, "log")), X)
    finally:
        np.setbufsize(old)
    assert plain == loop


def _zero_weight_gmm(implementation):
    """A two-state GMMHMM with a zero mixture weight in its second state."""
    _, factory, X = _variant("GMMHMM-diag")
    start = factory(2, "log").fit(X, LENGTHS)
    model = hmm.GMMHMM(
        2,
        n_mix=2,
        n_iter=1,
        implementation=implementation,
        init_params="",
        random_state=0,
    )
    model.startprob_, model.transmat_ = start.startprob_, start.transmat_
    model.means_, model.covars_ = start.means_, start.covars_
    model.weights_ = start.weights_.copy()
    model.weights_[1] = [1.0, 0.0]
    return model, X


def _negative_determinant_vgmm(implementation):
    """A VariationalGaussianHMM whose first scale matrix has a negative
    determinant: its likelihood warns, through ``warnings``, at each call."""
    _, factory, X = _variant("VariationalGaussianHMM-full")
    model = factory(2, "log").fit(X, LENGTHS)
    model.implementation, model.init_params, model.n_iter = (
        implementation,
        "",
        1,
    )
    model.scale_posterior_ = model.scale_posterior_.copy()
    model.scale_posterior_[0] = [[1.0, 2.0], [2.0, 1.0]]
    return model, X


def _singular_vgmm(implementation):
    """A VariationalGaussianHMM whose first scale matrix is singular: its
    likelihood raises ``LinAlgError`` at each call."""
    model, X = _negative_determinant_vgmm(implementation)
    model.scale_posterior_[0] = [[1.0, 1.0], [1.0, 1.0]]
    return model, X


PRE_CHECKS = {
    "GMMHMM-zero-weight": (
        _zero_weight_gmm,
        hmm.GMMHMM,
        "_compute_log_weighted_gaussian_densities",
    ),
    "VariationalGaussianHMM-negative-determinant": (
        _negative_determinant_vgmm,
        vhmm.VariationalGaussianHMM,
        "_compute_subnorm_log_likelihood",
    ),
    "VariationalGaussianHMM-singular": (
        _singular_vgmm,
        vhmm.VariationalGaussianHMM,
        "_compute_subnorm_log_likelihood",
    ),
}


@pytest.mark.parametrize(
    "state", [DEFAULT_STATE, {"all": "warn"}], ids=["default", "all=warn"]
)
@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("name", list(PRE_CHECKS))
def test_a_report_known_from_the_parameters_keeps_the_loop(
    monkeypatch, kernel_calls, rows_16, name, implementation, state
):
    """Parameters whose likelihood warns or raises at every call (a zero
    mixture weight, a singular scale matrix or one whose inverse has a
    negative determinant) are computed
    sequence by sequence, with the loop's emission calls; the test that
    refuses the blocks reports nothing, so the fit, its warnings in order,
    handler calls and log records are those of the loop."""
    build, cls, hook = PRE_CHECKS[name]
    model, X = build(implementation)
    loop = per_sequence(build(implementation)[0])
    calls = _counted_hook(monkeypatch, cls, hook)
    del kernel_calls[:]
    got = _fitted(model, X, state=state)
    assert _count(kernel_calls, "estep") == 0
    n_calls = len(calls)
    assert got == _fitted(loop, X, state=state)
    assert n_calls == len(calls) - n_calls


def test_a_subclass_hooks_are_called_sequence_by_sequence():
    """A subclass that replaces the likelihood and the statistics sees them
    called for one sequence after the other, as in hmmlearn 0.3.3."""
    _, factory, X = _variant("GaussianHMM-diag")
    seen = []

    class Recording(hmm.GaussianHMM):
        def _compute_log_likelihood(self, X):
            seen.append(("e", len(X)))
            return super()._compute_log_likelihood(X)

        def _accumulate_sufficient_statistics(self, stats, X, *args):
            seen.append(("a", len(X)))
            super()._accumulate_sufficient_statistics(stats, X, *args)

    model = factory(2, "log")
    model.__class__ = Recording
    model.n_iter = 1
    _fitted(model, X)
    assert seen == [(kind, n) for n in LENGTHS for kind in ("e", "a")]


# The two paths give the same bytes -------------------------------------------


@pytest.mark.parametrize(
    "n_components, block_rows",
    [(1, 1), (1, 16), (3, 1), (3, 16), (3, None)],
    ids=[
        "1-state-1-row",
        "1-state-16-rows",
        "3-states-1-row",
        "3-states-16-rows",
        "3-states-default",
    ],
)
@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
def test_block_and_per_sequence_fits_are_equal_as_bytes(
    monkeypatch,
    kernel_calls,
    variant,
    implementation,
    n_components,
    block_rows,
):
    """A fit by blocks equals the fit sequence by sequence: every fitted
    array, ``state_occupancy_``, the history and its types, the warnings
    and the log records."""
    if block_rows is not None:
        monkeypatch.setattr(
            base, "_ESTEP_BLOCK_ROWS", block_rows, raising=False
        )
    _, factory, X = variant
    plain = _fitted(factory(n_components, implementation), X)
    assert isinstance(plain[0], dict)
    assert _count(kernel_calls, "estep") > 0
    assert plain == _fitted(
        per_sequence(factory(n_components, implementation)), X
    )


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
def test_one_sequence_fits_are_equal_as_bytes(
    kernel_calls, variant, implementation
):
    """Without ``lengths``, the one sequence is one block."""
    _, factory, X = variant
    plain = _fitted(factory(2, implementation), X, None)
    assert _count(kernel_calls, "estep") > 0
    assert plain == _fitted(per_sequence(factory(2, implementation)), X, None)


@pytest.mark.parametrize("implementation", ["log", "scaling"])
def test_the_density_loop_and_numpy_meet_in_one_block(
    kernel_calls, rows_16, implementation
):
    """A variance of 1e-300 and a far sample: the C++ density loop gives
    no value for the far sample's rows, so its block is computed by the
    numpy expression, and the other sequences of the block by the loop when
    computed one at a time."""
    X = _data()[0].copy()
    X[8] = [1e5, 0.0]

    def factory():
        model = hmm.GaussianHMM(
            2,
            "diag",
            n_iter=1,
            implementation=implementation,
            init_params="",
        )
        model.startprob_ = [0.6, 0.4]
        model.transmat_ = [[0.7, 0.3], [0.4, 0.6]]
        model.means_ = [[0.0, 0.0], [3.0, 3.0]]
        model.covars_ = [[1e-300, 1.0], [1.0, 1.0]]
        return model

    plain = _fitted(factory(), X)
    assert _count(kernel_calls, "estep") == len(BLOCKS_AT_16)
    assert plain == _fitted(per_sequence(factory()), X)


# Which inputs are computed by one emission call per block -----------------


def _counted_hook(monkeypatch, cls, name):
    """Count the calls of the hook ``name`` through the class ``cls``."""
    calls = []
    real = getattr(cls, name)

    def counting(self, X, *args):
        calls.append(len(X))
        return real(self, X, *args)

    monkeypatch.setattr(cls, name, counting)
    return calls


def _fixed_gaussian_f_means():
    model = hmm.GaussianHMM(2, "diag", n_iter=1, init_params="")
    model.startprob_ = [0.6, 0.4]
    model.transmat_ = [[0.7, 0.3], [0.4, 0.6]]
    model.means_ = np.asfortranarray([[0.0, 0.0], [3.0, 3.0]])
    model.covars_ = [[1.0, 1.0], [1.0, 1.0]]
    assert not model.means_.flags.c_contiguous
    return model


def _row_cases():
    """(id, class, hook, factory, X, scipy.special state or None, one call
    per block)."""
    continuous, symbols, counts, multinomial = _data()
    strided = np.repeat(continuous, 2, axis=0)[::2]
    log = "_compute_log_likelihood"

    def gaussian(cov):
        return lambda: hmm.GaussianHMM(2, cov, n_iter=1, random_state=0)

    def poisson():
        return hmm.PoissonHMM(2, n_iter=1, random_state=0)

    def multi():
        return hmm.MultinomialHMM(2, n_trials=5, n_iter=1, random_state=0)

    return [
        (
            "GaussianHMM-diag",
            hmm.GaussianHMM,
            log,
            gaussian("diag"),
            continuous,
            None,
            True,
        ),
        (
            "GaussianHMM-spherical",
            hmm.GaussianHMM,
            log,
            gaussian("spherical"),
            continuous,
            None,
            True,
        ),
        (
            "CategoricalHMM",
            hmm.CategoricalHMM,
            log,
            lambda: hmm.CategoricalHMM(2, n_iter=1, random_state=0),
            symbols,
            None,
            True,
        ),
        (
            "VariationalCategoricalHMM",
            vhmm.VariationalCategoricalHMM,
            "_compute_subnorm_log_likelihood",
            lambda: vhmm.VariationalCategoricalHMM(
                2, n_iter=1, random_state=0
            ),
            symbols,
            None,
            True,
        ),
        ("PoissonHMM-int64", hmm.PoissonHMM, log, poisson, counts, None, True),
        (
            "PoissonHMM-float64",
            hmm.PoissonHMM,
            log,
            poisson,
            counts.astype(np.float64),
            None,
            True,
        ),
        (
            "MultinomialHMM-int64",
            hmm.MultinomialHMM,
            log,
            multi,
            multinomial,
            None,
            True,
        ),
        (
            "GaussianHMM-full",
            hmm.GaussianHMM,
            log,
            gaussian("full"),
            continuous,
            None,
            False,
        ),
        (
            "GaussianHMM-tied",
            hmm.GaussianHMM,
            log,
            gaussian("tied"),
            continuous,
            None,
            False,
        ),
        (
            "GaussianHMM-F-ordered-X",
            hmm.GaussianHMM,
            log,
            gaussian("diag"),
            np.asfortranarray(continuous),
            None,
            False,
        ),
        (
            "GaussianHMM-strided-X",
            hmm.GaussianHMM,
            log,
            gaussian("diag"),
            strided,
            None,
            False,
        ),
        (
            "GaussianHMM-integer-X",
            hmm.GaussianHMM,
            log,
            gaussian("diag"),
            np.rint(3 * continuous).astype(np.int64),
            None,
            False,
        ),
        (
            "GaussianHMM-float32-X",
            hmm.GaussianHMM,
            log,
            gaussian("diag"),
            continuous.astype(np.float32),
            None,
            False,
        ),
        (
            "GaussianHMM-F-ordered-means",
            hmm.GaussianHMM,
            log,
            _fixed_gaussian_f_means,
            continuous,
            None,
            False,
        ),
        (
            "PoissonHMM-int32",
            hmm.PoissonHMM,
            log,
            poisson,
            counts.astype(np.int32),
            None,
            False,
        ),
        (
            "MultinomialHMM-int32",
            hmm.MultinomialHMM,
            log,
            multi,
            multinomial.astype(np.int32),
            None,
            False,
        ),
        (
            "PoissonHMM-special-warn",
            hmm.PoissonHMM,
            log,
            poisson,
            counts,
            "warn",
            False,
        ),
        (
            "MultinomialHMM-special-warn",
            hmm.MultinomialHMM,
            log,
            multi,
            multinomial,
            "warn",
            False,
        ),
        (
            "GMMHMM",
            hmm.GMMHMM,
            log,
            lambda: hmm.GMMHMM(2, n_mix=2, n_iter=1, random_state=0),
            continuous,
            None,
            False,
        ),
        (
            "VariationalGaussianHMM",
            vhmm.VariationalGaussianHMM,
            "_compute_subnorm_log_likelihood",
            lambda: vhmm.VariationalGaussianHMM(2, n_iter=1, random_state=0),
            continuous,
            None,
            False,
        ),
    ]


ROW_CASES = _row_cases()


@pytest.mark.parametrize("case", ROW_CASES, ids=[c[0] for c in ROW_CASES])
def test_inputs_not_proven_row_by_row_fill_the_block_per_sequence(
    monkeypatch, rows_16, case
):
    """The emission hook is called once per block for the inputs whose rows
    do not depend on each other in one call (plain float64 arrays in C
    order, int64 counts, symbols), and once per sequence for every other
    input; both paths give the same fit."""
    _, cls, name, factory, X, special_state, by_block = case
    calls = _counted_hook(monkeypatch, cls, name)
    with special.errstate(all=special_state or "ignore"):
        plain = _fitted(factory(), X)
        expected = len(BLOCKS_AT_16) if by_block else len(LENGTHS)
        assert len(calls) == expected
        assert sum(calls) == N_ROWS
        assert plain == _fitted(per_sequence(factory()), X)


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize(
    "factory",
    [
        lambda impl: hmm.CategoricalHMM(
            2, n_features=3, n_iter=2, implementation=impl, random_state=0
        ),
        lambda impl: vhmm.VariationalCategoricalHMM(
            2, n_features=3, n_iter=2, implementation=impl, random_state=0
        ),
    ],
    ids=["CategoricalHMM", "VariationalCategoricalHMM"],
)
def test_the_deprecated_categorical_layout_keeps_the_loop(
    kernel_calls, rows_16, factory, implementation
):
    """Symbols in two columns, the deprecated layout, are computed sequence
    by sequence: no block call, the same deprecation warnings and the same
    outcome as the loop."""
    X = np.random.RandomState(10).randint(0, 3, (N_ROWS, 2))
    plain = _fitted(factory(implementation), X)
    assert _count(kernel_calls, "estep") == 0
    assert plain == _fitted(per_sequence(factory(implementation)), X)


# Factors of full and tied covariances --------------------------------------


@pytest.fixture
def stats_linalg_calls(monkeypatch):
    """Record each call of ``linalg.cholesky`` and ``linalg.solve_triangular``
    made from ``hmmcraft.stats``: ("cholesky", the input, the array returned
    or None when it raised) and ("solve_triangular", the factor, the number
    of columns of the right-hand side)."""
    calls = []

    def from_stats():
        return sys._getframe(2).f_globals.get("__name__") == stats.__name__

    real_cholesky = linalg.cholesky
    real_solve = linalg.solve_triangular

    def cholesky(a, *args, **kwargs):
        if not from_stats():
            return real_cholesky(a, *args, **kwargs)
        try:
            factor = real_cholesky(a, *args, **kwargs)
        except linalg.LinAlgError:
            calls.append(("cholesky", a, None))
            raise
        calls.append(("cholesky", a, factor))
        return factor

    def solve_triangular(a, b, *args, **kwargs):
        if from_stats():
            calls.append(("solve_triangular", a, np.shape(b)[1]))
        return real_solve(a, b, *args, **kwargs)

    monkeypatch.setattr(linalg, "cholesky", cholesky)
    monkeypatch.setattr(linalg, "solve_triangular", solve_triangular)
    return calls


def _factored(calls):
    """The number of ``cholesky`` calls that returned a factor."""
    return sum(1 for name, _, got in calls
               if name == "cholesky" and got is not None)


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("covariance_type", ["full", "tied"])
def test_full_and_tied_covariances_are_factored_once_per_e_step(
    stats_linalg_calls, rows_16, covariance_type, implementation
):
    """An E-step by blocks factors each ``full`` covariance once (``tied``:
    its one covariance once) and solves once per state and sequence with
    that sequence's rows; sequence by sequence, each sequence factors every
    state's covariance again. Both give the same fit, and no factor is left
    on the model."""
    _, factory, X = _variant(f"GaussianHMM-{covariance_type}")
    model = factory(3, implementation)
    plain = _fitted(model, X)
    assert isinstance(plain[0], dict)
    n_iter = model.monitor_.iter
    per_covariance = 3 if covariance_type == "full" else 1
    assert _factored(stats_linalg_calls) == per_covariance * n_iter
    columns = [n for name, _, n in stats_linalg_calls
               if name == "solve_triangular"]
    assert columns == [n for n in LENGTHS for _ in range(3)] * n_iter
    assert "_estep_factors" not in vars(model)

    del stats_linalg_calls[:]
    loop = per_sequence(factory(3, implementation))
    assert _fitted(loop, X) == plain
    assert _factored(stats_linalg_calls) == (
        3 * len(LENGTHS) * loop.monitor_.iter
    )


def _fixed_full_or_tied(covariance_type, implementation):
    """A two-state GaussianHMM with fixed parameters, checked."""
    model = hmm.GaussianHMM(
        2,
        covariance_type,
        n_iter=1,
        implementation=implementation,
        init_params="",
    )
    model.startprob_ = [0.6, 0.4]
    model.transmat_ = [[0.7, 0.3], [0.4, 0.6]]
    model.means_ = [[0.0, 0.0], [3.0, 3.0]]
    model.n_features = 2
    if covariance_type == "full":
        model.covars_ = [[[1.0, 0.5], [0.5, 1.0]], [[2.0, 0.0], [0.0, 1.0]]]
    else:
        model.covars_ = [[1.0, 0.5], [0.5, 1.0]]
    model._check()
    return model


def _set_covariance(value):
    """State 1's covariance (tied: the covariance), set past the checks."""

    def change(model, X):
        covars = model._covars_.copy()
        if model.covariance_type == "tied":
            covars[...] = value
        else:
            covars[1] = value
        model._covars_ = covars
        return X

    return change


def _set_row(value, mean=None, row=38):
    """A row, by default row 38 in the eighth sequence, set to ``value``;
    every mean set to ``mean`` if given."""

    def change(model, X):
        X = X.copy()
        X[row] = value
        if mean is not None:
            model.means_ = np.full_like(model.means_, mean)
        return X

    return change


def _reports_then_raises(model, X):
    """State 0's arithmetic overflows on the first sequence, its one row;
    state 1 cannot be factored."""
    X = _set_covariance([[1.0, 2.0], [2.0, 1.0]])(model, X)
    return _set_row(1e200, row=0)(model, X)


# name: (change of the model and data, factors held, covariance types,
#        implementations)
HELD_CASES = {
    "plain": (lambda model, X: X, True, ("full", "tied"), ("log", "scaling")),
    "singular covariance": (
        _set_covariance(1.0), True, ("full", "tied"), ("log", "scaling")),
    "not positive definite": (
        _set_covariance([[1.0, 2.0], [2.0, 1.0]]),
        False,
        ("full", "tied"),
        ("log", "scaling"),
    ),
    "NaN covariance": (
        _set_covariance(np.nan), False, ("full", "tied"), ("log", "scaling")),
    "1e308 against -1e308": (
        _set_row(1e308, -1e308), True, ("full", "tied"), ("log", "scaling")),
    "row of 1e200": (
        _set_row(1e200), True, ("full", "tied"), ("log", "scaling")),
    "state 0 reports, state 1 raises": (
        _reports_then_raises, False, ("full",), ("log", "scaling")),
    "row of 1e4, flagged": (
        _set_row(1e4), True, ("full", "tied"), ("scaling",)),
}
HELD_ROWS = [
    (name, cov, impl)
    for name, (_, _, types, impls) in HELD_CASES.items()
    for cov in types
    for impl in impls
]
HELD_STATES = [DEFAULT_STATE, *REPORT_STATES, {"all": "ignore"}]
HELD_STATE_IDS = ["default", *REPORT_IDS, "all=ignore"]


@pytest.mark.parametrize("state", HELD_STATES, ids=HELD_STATE_IDS)
@pytest.mark.parametrize(
    "name, covariance_type, implementation",
    HELD_ROWS,
    ids=[f"{n}-{c}-{i}" for n, c, i in HELD_ROWS],
)
def test_held_factors_give_the_loop_outcome(
    stats_linalg_calls, rows_16, name, covariance_type, implementation, state
):
    """An E-step by blocks with factors held gives the statistics or the
    error, the warnings in order, the handler calls and the log records of
    the loop; the sequences of a block that reports, and a flagged
    sequence, are computed with the held factors, so the E-step factors
    each covariance once. Nothing is left on the model."""
    change, held, _, _ = HELD_CASES[name]
    X = _data()[0]
    model = _fixed_full_or_tied(covariance_type, implementation)
    X = change(model, X)
    loop = per_sequence(_fixed_full_or_tied(covariance_type, implementation))
    change(loop, _data()[0])
    del stats_linalg_calls[:]
    got = _estep(model, X, LENGTHS, state)
    n_factored = _factored(stats_linalg_calls)
    assert "_estep_factors" not in vars(model)
    assert got == _estep(loop, X, LENGTHS, state)
    if held:
        assert n_factored == (2 if covariance_type == "full" else 1)


def _factors_read_by_the_kernel(monkeypatch, model):
    """Record ``model._estep_factors`` at each call of the kernel's block
    functions."""
    seen = []
    for name in ("estep_log", "estep_scaling"):
        real = getattr(_hmmc, name)

        def reading(*args, _real=real):
            seen.append(model._estep_factors)
            return _real(*args)

        monkeypatch.setattr(_hmmc, name, reading)
    return seen


def _densities_called(monkeypatch):
    """Record the ``factors`` argument of each call of the full and tied
    log-densities, through ``hmmcraft.stats`` and ``hmmcraft._emissions``."""
    seen = []
    for module in (stats, _emissions):
        for name in (
            "_log_multivariate_normal_density_full",
            "_log_multivariate_normal_density_tied",
        ):
            real = getattr(module, name)

            def recording(*args, _real=real, **kwargs):
                seen.append(kwargs.get("factors"))
                return _real(*args, **kwargs)

            monkeypatch.setattr(module, name, recording)
    return seen


@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
def test_only_full_and_tied_gaussian_models_hold_factors(
    monkeypatch, rows_16, variant
):
    """While the E-step computes by blocks, only a GaussianHMM with full or
    tied covariances holds factors; none is left after a fit, after an
    E-step that raises, or in a pickle, and the predictions factor at each
    call."""
    name, factory, X = variant
    holds = name in ("GaussianHMM-full", "GaussianHMM-tied")
    model = factory(2, "log")
    with monkeypatch.context() as patch:
        seen = _factors_read_by_the_kernel(patch, model)
        model.fit(X, LENGTHS)
    assert seen and all((held is not None) == holds for held in seen)
    if holds:
        _, factors = seen[-1]
        assert len(factors) == 2
    assert "_estep_factors" not in vars(model)
    assert "_estep_factors" not in vars(pickle.loads(pickle.dumps(model)))

    def raising(*args):
        raise ValueError("raised in the block")

    with monkeypatch.context() as patch:
        patch.setattr(_hmmc, "estep_log", raising)
        with pytest.raises(ValueError, match="raised in the block"):
            model._do_estep(X, LENGTHS)
    assert "_estep_factors" not in vars(model)

    if holds:
        with monkeypatch.context() as patch:
            called = _densities_called(patch)
            model.score(X, LENGTHS)
        assert len(called) >= len(LENGTHS)
        assert all(factors is None for factors in called)


# Reports and underflow -------------------------------------------------------


def _impossible_symbol():
    """A log CategoricalHMM, and symbols with an impossible one in the
    middle of the eighth sequence."""
    model = hmm.CategoricalHMM(
        2, n_features=3, init_params="", implementation="log"
    )
    model.startprob_ = np.array([0.6, 0.4])
    model.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]])
    model.emissionprob_ = np.array([[0.5, 0.5, 0.0], [0.4, 0.6, 0.0]])
    X = np.random.RandomState(10).randint(0, 2, (N_ROWS, 1))
    X[38] = 2
    return model, X


def _huge_count():
    """A scaling PoissonHMM, and counts with one of 1000, whose probability
    underflows in every state, in the middle of the eighth sequence."""
    model = hmm.PoissonHMM(2, init_params="", implementation="scaling")
    model.startprob_ = np.array([0.5, 0.5])
    model.transmat_ = np.array([[0.7, 0.3], [0.3, 0.7]])
    model.lambdas_ = np.array([[2.0, 5.0], [5.0, 2.0]])
    model.n_features = 2
    X = np.random.RandomState(10).poisson([2.0, 5.0], size=(N_ROWS, 2))
    X[38] = 1000
    return model, X


REPORT_CASES = {
    "posteriors": (_impossible_symbol, "estep_log"),
    "emissions": (_huge_count, None),
}


@pytest.mark.parametrize("state", REPORT_STATES, ids=REPORT_IDS)
@pytest.mark.parametrize("name", list(REPORT_CASES))
def test_a_report_in_a_block_gives_the_loop_outcome(
    kernel_calls, rows_16, name, state
):
    """A floating-point report the caller's error state asks for, in a
    block's posteriors or emissions, gives the statistics or the error,
    the warnings in order and the handler calls of the loop."""
    build, attempted = REPORT_CASES[name]
    model, X = build()
    model._check()
    loop = per_sequence(build()[0])
    del kernel_calls[:]
    got = _estep(model, X, LENGTHS, state)
    if attempted is not None:
        assert (attempted, False) in kernel_calls
    assert got[1] != [] or got[2] != [] or isinstance(got[0], tuple)
    assert got == _estep(loop, X, LENGTHS, state)


def _far_gaussian(implementation="scaling"):
    model = hmm.GaussianHMM(
        2, "diag", n_iter=2, implementation=implementation, init_params=""
    )
    model.startprob_ = [0.6, 0.4]
    model.transmat_ = [[0.7, 0.3], [0.4, 0.6]]
    model.means_ = [[0.0], [3.0]]
    model.covars_ = [[1.0], [1.0]]
    return model


FAR_ROWS = {
    "one-row-sequence": [0],
    "first-row": [6],
    "middle-row": [8],
    "last-row": [10],
    "two-in-one-block": [1, 12],
    "sequence-alone": [80],
}


@pytest.mark.parametrize("rows", list(FAR_ROWS.values()), ids=list(FAR_ROWS))
def test_underflow_in_a_block_falls_back_as_the_loop(
    kernel_calls, rows_16, rows
):
    """A sequence whose scaled forward pass underflows, at any place in its
    block, is computed with the log functions at its own place: the fit
    and the logged warning are those of the loop."""
    X = _data()[0][:, :1].copy()
    X[rows] = 1e4
    plain = _fitted(_far_gaussian(), X)
    assert _count(kernel_calls, "estep_scaling") > 0
    assert any("underflowed" in message for _, message in plain[3])
    assert plain == _fitted(per_sequence(_far_gaussian()), X)


def _cycle(n_rows):
    """LENGTHS repeated over ``n_rows`` rows, the last one cut."""
    lengths, total = [], 0
    while total < n_rows:
        length = min(LENGTHS[len(lengths) % len(LENGTHS)], n_rows - total)
        lengths.append(length)
        total += length
    return lengths


@pytest.mark.parametrize("case", UNDERFLOW_FIT_CASES, ids=UNDERFLOW_FIT_IDS)
def test_underflow_fits_by_blocks_as_the_loop(kernel_calls, rows_16, case):
    """The underflow fits of every model class, over many short sequences:
    the fit and the logged warning are those of the loop."""
    _, factory, X = case
    lengths = _cycle(len(X))
    plain = _fitted(factory("scaling"), X, lengths)
    assert _count(kernel_calls, "estep_scaling") > 0
    assert plain == _fitted(per_sequence(factory("scaling")), X, lengths)


@pytest.mark.parametrize(
    "path", [lambda m: m, per_sequence], ids=["blocks", "loop"]
)
def test_an_impossible_sequence_still_raises_the_underflow_error(
    kernel_calls, path
):
    """An impossible sequence, second of three, raises the kernel's
    underflow error on both paths."""
    X = np.concatenate([IMPOSSIBLE[:2], IMPOSSIBLE, [[0], [1], [1]]])
    model = path(_impossible())
    result, *_ = _fitted(model, X, [2, 5, 3])
    assert result == ("error", "ValueError", UNDERFLOW_TEXT, True)
    assert "_accumulate_as_log" not in vars(model)
    assert "_estep_values" not in vars(model)


@pytest.mark.parametrize(
    "name",
    [
        "GaussianHMM-diag",
        "GaussianHMM-full",
        "GaussianHMM-tied",
        "GMMHMM-diag",
        "PoissonHMM",
        "VariationalCategoricalHMM",
    ],
)
def test_a_flag_without_an_underflow_is_computed_per_sequence(
    monkeypatch, rows_16, name
):
    """A sequence flagged by the block function whose forward pass does not
    underflow is computed by the per-sequence code: the fit equals the fit
    without the flag."""
    _, factory, X = _variant(name)
    expected = _fitted(factory(2, "scaling"), X)
    real = _hmmc.estep_scaling

    def flagging(startprob, transmat, lattice, lengths, with_xi_sum,
                 n_threads):
        log_prob, fwd, bwd, xi, underflow = real(
            startprob, transmat, lattice, lengths, with_xi_sum, n_threads
        )
        if len(lengths) > 1:
            a, b = lengths[0], lengths[0] + lengths[1]
            underflow[1] = True
            log_prob[1] = fwd[a:b] = bwd[a:b] = np.nan
        return log_prob, fwd, bwd, xi, underflow

    monkeypatch.setattr(_hmmc, "estep_scaling", flagging)
    assert _fitted(factory(2, "scaling"), X) == expected
