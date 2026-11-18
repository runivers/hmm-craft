"""The E-step's sequences on several threads, ``n_threads``.

Every test that starts a thread is here. Each sequence is computed on one
thread with the arithmetic of one thread, and the statistics are added in
``lengths`` order, so a fit, a kernel call and the floating-point reports
are those of one thread, as bytes, at any thread count. The counts are 2, 4
and more threads than sequences, whatever the number of cores.
"""

import ctypes
import inspect
import os
import pickle
import re
import sys
import threading

import numpy as np
import pytest
from sklearn.base import clone

from hmmcraft import _hmmc, base, hmm, vhmm
from hmmcraft.base import BaseHMM, VariationalBaseHMM

from . import test_estep_blocks as blocks
from .test_base import UNDERFLOW_FIT_CASES, UNDERFLOW_FIT_IDS
from .test_estep_blocks import (
    DEFAULT_STATE,
    FAR_ROWS,
    LENGTHS,
    REPORT_CASES,
    VARIANT_IDS,
    VARIANTS,
    _count,
    _cycle,
    _data,
    _estep,
    _far_gaussian,
    _fitted,
    _variant,
    kernel_calls,  # noqa: F401 (a fixture)
    rows_16,  # noqa: F401 (a fixture)
)
from .test_hmmc import (
    BLOCK_FUNCTIONS,
    BLOCK_RESULTS,
    SHORT_AND_LONG,
    UNDERFLOW_LENGTHS,
    _assert_same_bytes,
    _block_inputs,
    _per_sequence,
)

try:
    import resource
except ImportError:  # Not on Windows.
    resource = None

LINUX = sys.platform.startswith("linux")
THREAD_COUNTS = [2, 4, len(LENGTHS) + 1]
CLASSES = [
    hmm.CategoricalHMM,
    hmm.GaussianHMM,
    hmm.GMMHMM,
    hmm.MultinomialHMM,
    hmm.PoissonHMM,
    vhmm.VariationalCategoricalHMM,
    vhmm.VariationalGaussianHMM,
    BaseHMM,
    VariationalBaseHMM,
]
ERROR_STATES = [
    DEFAULT_STATE,
    {"all": "warn"},
    {"all": "call"},
    {"all": "raise"},
    {"all": "ignore"},
]
ERROR_STATE_IDS = ["default", "all=warn", "all=call", "all=raise",
                   "all=ignore"]
# Every exception flag bit of the C library; it ignores the others.
ANY_FLAG = 0xFF


def _threaded_block(implementation, start, trans, lattice, lengths,
                    with_xi_sum, n_threads):
    """Call the block function of ``implementation`` on ``n_threads``
    threads; name its results."""
    results = getattr(_hmmc, "estep_" + implementation)(
        start, trans, lattice, lengths, with_xi_sum, n_threads=n_threads)
    names = BLOCK_RESULTS + (["underflow"] if implementation == "scaling"
                             else [])
    assert len(results) == len(names)
    return dict(zip(names, results))


@pytest.fixture
def block_calls(request):
    """``test_estep_blocks``'s ``kernel_calls``: (name, ``with_xi_sum``)
    of each kernel pass call."""
    return request.getfixturevalue("kernel_calls")


@pytest.fixture
def requested_threads(monkeypatch):
    """Record the thread count each call of a block function receives."""
    counts = []
    for name in BLOCK_FUNCTIONS:
        real = getattr(_hmmc, name)

        def recording(*args, _real=real, **kwargs):
            counts.append(args[5] if len(args) > 5
                          else kwargs.get("n_threads", 1))
            return _real(*args, **kwargs)

        monkeypatch.setattr(_hmmc, name, recording)
    return counts


# The kernel ------------------------------------------------------------------


@pytest.mark.parametrize("data", ["dense", "zeros"])
@pytest.mark.parametrize(
    "lengths",
    [[1], [7], [1, 1, 1], SHORT_AND_LONG, SHORT_AND_LONG + [300]],
    ids=["one-row", "one-sequence", "one-row-each", "short", "short-long"],
)
@pytest.mark.parametrize("n_components", [1, 2, 8, 11])
@pytest.mark.parametrize("implementation", ["log", "scaling"])
def test_block_functions_on_threads_equal_the_per_sequence_functions(
    implementation, n_components, lengths, data
):
    """On 2, 4 and more threads than sequences, the block functions return
    the bytes of the per-sequence functions, with the transition sums and
    without them."""
    start, trans, lattice = _block_inputs(
        implementation, n_components, lengths, data)
    lengths = np.array(lengths, dtype=np.intp)
    expected = _per_sequence(implementation, start, trans, lattice, lengths)
    without = dict(expected,
                   xi_sum=np.empty((0, n_components, n_components)))
    for n_threads in (2, 4, len(lengths) + 1):
        _assert_same_bytes(
            _threaded_block(implementation, start, trans, lattice, lengths,
                            True, n_threads),
            expected)
        _assert_same_bytes(
            _threaded_block(implementation, start, trans, lattice, lengths,
                            False, n_threads),
            without)


@pytest.mark.parametrize(
    "rows, value",
    [([3], 0.), ([7], 0.), ([1, 8], 0.), ([4], 1e-301)],
    ids=["zero-first-row", "zero-one-row-sequence", "two-sequences",
         "below-the-floor"],
)
def test_estep_scaling_on_threads_flags_the_sequences_of_one_thread(
    rows, value
):
    """On several threads, the sequences whose scaled forward pass
    underflows get their flag and NaN in every result, as on one."""
    start, trans, frame = _block_inputs(
        "scaling", 2, UNDERFLOW_LENGTHS, "dense")
    frame[rows] = value
    lengths = np.array(UNDERFLOW_LENGTHS)
    expected = _per_sequence("scaling", start, trans, frame, lengths)
    assert expected["underflow"].any()
    for n_threads in (2, 4, len(UNDERFLOW_LENGTHS) + 1):
        _assert_same_bytes(
            _threaded_block("scaling", start, trans, frame, lengths, True,
                            n_threads),
            expected)


@pytest.mark.parametrize("n_threads", [0, -1])
@pytest.mark.parametrize("name", BLOCK_FUNCTIONS)
def test_block_functions_refuse_fewer_than_one_thread(name, n_threads):
    with pytest.raises(ValueError, match="^n_threads must be at least 1$"):
        getattr(_hmmc, name)(
            np.full(2, 0.5), np.full((2, 2), 0.5), np.full((5, 2), 0.5),
            np.array([2, 3]), True, n_threads=n_threads)


@pytest.mark.skipif(
    not LINUX, reason="reads the C library's flags through ctypes")
def test_flags_raised_on_threads_are_raised_on_the_calling_thread():
    """The floating-point exception flags a call on 4 threads leaves on
    the calling thread are those of the call on one thread, whichever
    thread computed the sequence that raises one."""
    libc = ctypes.CDLL(None)
    lengths = [3] * 40
    start, trans, lattice = _block_inputs("log", 2, lengths, "dense")
    plain = lattice.copy()
    lattice[3 * 17 + 1, 0] = -1e4
    lengths = np.array(lengths, dtype=np.intp)

    def flags(lattice, n_threads):
        libc.feclearexcept(ANY_FLAG)
        _hmmc.estep_log(start, trans, lattice, lengths, True,
                        n_threads=n_threads)
        return libc.fetestexcept(ANY_FLAG)

    one = flags(lattice, 1)
    assert one != flags(plain, 1)
    assert [flags(lattice, 4) for _ in range(40)] == [one] * 40


@pytest.mark.skipif(
    not LINUX, reason="lowers the process limit through resource")
def test_a_thread_the_system_refuses_leaves_the_result_unchanged():
    """With the process limit at one, no thread starts (for an unprivileged
    user) and the call on 4 threads returns the bytes of one thread."""
    start, trans, lattice = _block_inputs(
        "scaling", 3, SHORT_AND_LONG, "zeros")
    lengths = np.array(SHORT_AND_LONG, dtype=np.intp)
    expected = _threaded_block(
        "scaling", start, trans, lattice, lengths, True, 1)
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    resource.setrlimit(resource.RLIMIT_NPROC, (1, hard))
    try:
        got = _threaded_block(
            "scaling", start, trans, lattice, lengths, True, 4)
        refused = False
        try:
            threading.Thread(target=lambda: None).start()
        except RuntimeError:
            refused = True
    finally:
        resource.setrlimit(resource.RLIMIT_NPROC, (soft, hard))
    # The limit does not apply to root.
    assert refused or os.geteuid() == 0
    _assert_same_bytes(got, expected)


# The blocks ------------------------------------------------------------------


@pytest.mark.parametrize(
    "lengths, block_rows, n_components, n_threads, expected",
    [
        ([20, 20, 20], 16, 1, 1, [(0, 1), (1, 2), (2, 3)]),
        ([20, 20, 20], 16, 1, 2, [(0, 2), (2, 3)]),
        ([20, 20, 20], 16, 1, 3, [(0, 3)]),
        ([20, 20, 20], 16, 1, 5, [(0, 3)]),
        (LENGTHS, 16, 2, 2, [(0, 5), (5, 7), (7, 10), (10, 12), (12, 14)]),
    ],
    ids=["longer-1", "longer-2", "longer-3", "longer-5", "check-cycle-2"],
)
def test_a_block_holds_at_least_n_threads_sequences(
    monkeypatch, lengths, block_rows, n_components, n_threads, expected
):
    """Above one thread, a block takes sequences until it holds
    ``n_threads`` of them, then goes on within the bounds of one thread."""
    monkeypatch.setattr(base, "_ESTEP_BLOCK_ROWS", block_rows, raising=False)
    assert base._estep_blocks(lengths, n_components, n_threads) == expected


def test_one_thread_cuts_the_blocks_of_the_default(monkeypatch):
    """At one thread, every case of the test of the default cut gives its
    blocks."""
    test = blocks.test_blocks_hold_whole_sequences_in_order
    (mark,) = [m for m in test.pytestmark if m.name == "parametrize"]
    for lengths, block_rows, n_components, expected in mark.args[1]:
        monkeypatch.setattr(
            base, "_ESTEP_BLOCK_ROWS", block_rows, raising=False)
        assert base._estep_blocks(lengths, n_components, 1) == expected


# The parameter ---------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSES, ids=lambda cls: cls.__name__)
def test_n_threads_is_the_last_parameter_of_every_model(cls):
    """Default 1; stored as given, read by ``get_params``, kept by
    ``clone``; a value is tested by ``fit`` only."""
    parameters = inspect.signature(cls.__init__).parameters
    assert list(parameters)[-1] == "n_threads"
    assert parameters["n_threads"].default == 1
    assert cls().n_threads == 1
    assert cls().get_params()["n_threads"] == 1
    for value in (3, np.int64(3), "not a count"):
        model = cls(n_threads=value)
        assert model.n_threads is value
        assert model.get_params()["n_threads"] is value
        copy = clone(model)
        assert type(copy.n_threads) is type(value)
        assert copy.n_threads == value


INVALID = [0, -1, True, False, np.True_, 2.0, None, "2"]
INVALID_IDS = ["zero", "negative", "true", "false", "numpy-true", "float",
               "none", "string"]


@pytest.mark.parametrize("value", INVALID, ids=INVALID_IDS)
@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
def test_fit_refuses_an_n_threads_that_is_not_a_count(
    block_calls, variant, value
):
    _, factory, X = variant
    model = factory(2, "log").set_params(n_threads=value)
    message = ("n_threads must be an integer of at least 1, "
               f"got {value!r}")
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        model.fit(X, LENGTHS)
    assert block_calls == []


@pytest.mark.parametrize(
    "value", [2, np.int64(2), 2**70],
    ids=["int", "numpy-int64", "beyond-a-c-integer"],
)
@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
def test_fit_accepts_any_integer_count(variant, value):
    _, factory, X = variant
    expected = _fitted(factory(2, "log"), X)
    model = factory(2, "log").set_params(n_threads=value)
    assert _fitted(model, X) == expected
    assert model.n_threads is value


@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
def test_a_prediction_does_not_read_n_threads(variant):
    _, factory, X = variant
    model = factory(2, "log").fit(X, LENGTHS)
    expected = model.score(X, LENGTHS)
    model.n_threads = 0
    assert model.score(X, LENGTHS) == expected


@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
def test_a_model_pickled_without_n_threads_loads_with_one(variant):
    """A model pickled before ``n_threads`` existed loads with 1 and fits
    as the model it was pickled from."""
    _, factory, X = variant
    fitted = factory(2, "log").fit(X, LENGTHS)
    state = pickle.loads(pickle.dumps(fitted)).__getstate__()
    del state["n_threads"]
    loaded = type(fitted).__new__(type(fitted))
    loaded.__setstate__(state)
    assert loaded.get_params()["n_threads"] == 1
    assert _fitted(loaded, X) == _fitted(fitted, X)


# Fits ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_components, block_rows",
    [(1, 16), (1, None), (3, 16), (3, None)],
    ids=["1-state-16-rows", "1-state-default", "3-states-16-rows",
         "3-states-default"],
)
@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
def test_fits_on_threads_equal_the_one_thread_fit(
    monkeypatch, requested_threads, variant, implementation, n_components,
    block_rows
):
    """Every fitted array, ``state_occupancy_``, the history and its types,
    the warnings and the log records of a fit on 2, 4 and more threads
    than sequences are those of one thread."""
    if block_rows is not None:
        monkeypatch.setattr(
            base, "_ESTEP_BLOCK_ROWS", block_rows, raising=False)
    _, factory, X = variant
    expected = _fitted(factory(n_components, implementation), X)
    assert isinstance(expected[0], dict)
    assert set(requested_threads) == {1}
    for n_threads in THREAD_COUNTS:
        del requested_threads[:]
        model = factory(n_components, implementation).set_params(
            n_threads=n_threads)
        assert _fitted(model, X) == expected, n_threads
        assert max(requested_threads) > 1


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
def test_one_sequence_on_threads_fits_as_on_one(
    requested_threads, variant, implementation
):
    """Without ``lengths``, the one sequence is computed on one thread."""
    _, factory, X = variant
    expected = _fitted(factory(2, implementation), X, None)
    del requested_threads[:]
    model = factory(2, implementation).set_params(n_threads=4)
    assert _fitted(model, X, None) == expected
    assert set(requested_threads) == {1}


def _long_X(name, n_rows):
    """The data of the variant ``name``, over ``n_rows`` rows."""
    continuous, symbols, counts, multinomial = _data(n_rows)
    if name.startswith("Multinomial"):
        return multinomial
    if name.startswith("Poisson"):
        return counts
    if "Categorical" in name:
        return symbols
    return continuous


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
def test_sequences_longer_than_a_block_share_one_on_threads(
    block_calls, variant, implementation
):
    """Two sequences longer than a block's rows are one block on two
    threads and two on one; the fit is the same."""
    name, factory, _ = variant
    rows = base._ESTEP_BLOCK_ROWS
    lengths = [rows + 1, rows + 1, 1]
    X = _long_X(name, sum(lengths))
    expected = _fitted(factory(2, implementation), X, lengths)
    assert isinstance(expected[0], dict)
    n_iter = expected[0]["monitor_.iter"]
    assert _count(block_calls, "estep") == 3 * n_iter
    del block_calls[:]
    model = factory(2, implementation).set_params(n_threads=2)
    assert _fitted(model, X, lengths) == expected
    assert _count(block_calls, "estep") == 2 * n_iter


@pytest.fixture
def xi_sum_calls(monkeypatch):
    """Record each call of the per-sequence transition-sum functions."""
    calls = []
    for name in ("compute_log_xi_sum", "compute_scaling_xi_sum"):
        real = getattr(_hmmc, name)

        def counting(*args, _real=real, _name=name):
            calls.append(_name)
            return _real(*args)

        monkeypatch.setattr(_hmmc, name, counting)
    return calls


@pytest.mark.usefixtures("rows_16")
@pytest.mark.parametrize("params", ["stmc", "smc"])
@pytest.mark.parametrize("implementation", ["log", "scaling"])
def test_threads_take_the_transition_sums_from_the_block_call(
    block_calls, xi_sum_calls, implementation, params
):
    """On two threads the block call computes the transition sums when
    ``"t"`` is in ``params``, and no per-sequence sum is computed; on one
    thread the per-sequence functions compute them."""
    _, factory, X = _variant("GaussianHMM-diag")
    one = factory(2, implementation).set_params(params=params)
    expected = _fitted(one, X)
    assert isinstance(expected[0], dict)
    assert {with_xi for _, with_xi in block_calls} == {False}
    assert bool(xi_sum_calls) == ("t" in params)
    del block_calls[:], xi_sum_calls[:]
    model = factory(2, implementation).set_params(
        params=params, n_threads=2)
    assert _fitted(model, X) == expected
    assert {with_xi for _, with_xi in block_calls} == {"t" in params}
    assert xi_sum_calls == []
    assert "_estep_xi_sum" not in vars(model)


@pytest.mark.usefixtures("rows_16")
@pytest.mark.parametrize("implementation", ["log", "scaling"])
def test_the_block_transition_sums_are_not_left_on_the_model(
    monkeypatch, implementation
):
    """An error in the statistics of a sequence leaves no transition sums
    on the model."""
    name = "_accumulate_sufficient_statistics_" + implementation
    seen = []

    def failing(self, *args, **kwargs):
        seen.append(self._estep_xi_sum is not None)
        raise RuntimeError("stop")

    monkeypatch.setattr(BaseHMM, name, failing)
    _, factory, X = _variant("GaussianHMM-diag")
    model = factory(2, implementation).set_params(n_threads=2)
    with pytest.raises(RuntimeError, match="^stop$"):
        model.fit(X, LENGTHS)
    assert seen == [True]
    assert "_estep_xi_sum" not in vars(model)


# Underflow and reports -------------------------------------------------------


@pytest.mark.parametrize("rows", list(FAR_ROWS.values()), ids=list(FAR_ROWS))
@pytest.mark.usefixtures("rows_16")
def test_underflow_on_threads_falls_back_as_on_one(rows):
    """A sequence whose scaled forward pass underflows is computed with the
    log functions on 2 and 4 threads as on one: same fit, same warning."""
    X = _data()[0][:, :1].copy()
    X[rows] = 1e4
    expected = _fitted(_far_gaussian(), X)
    assert any("underflowed" in message for _, message in expected[3])
    for n_threads in (2, 4):
        model = _far_gaussian().set_params(n_threads=n_threads)
        assert _fitted(model, X) == expected, n_threads


@pytest.mark.parametrize("case", UNDERFLOW_FIT_CASES, ids=UNDERFLOW_FIT_IDS)
@pytest.mark.usefixtures("rows_16")
def test_underflow_fits_on_threads_equal_one(case):
    """The underflow fits of every model class, over many short sequences,
    on 2 and 4 threads: the fit and the logged warning of one thread."""
    _, factory, X = case
    lengths = _cycle(len(X))
    expected = _fitted(factory("scaling"), X, lengths)
    assert any("underflowed" in message for _, message in expected[3])
    for n_threads in (2, 4):
        model = factory("scaling").set_params(n_threads=n_threads)
        assert _fitted(model, X, lengths) == expected, n_threads


@pytest.mark.parametrize("state", ERROR_STATES, ids=ERROR_STATE_IDS)
@pytest.mark.parametrize("name", list(REPORT_CASES))
@pytest.mark.usefixtures("rows_16")
def test_a_report_on_threads_gives_the_one_thread_outcome(name, state):
    """A floating-point report in a block's emissions or posteriors, on two
    threads, gives the statistics or the error, the warnings in order, the
    handler calls and the log records of one thread."""
    build, _ = REPORT_CASES[name]
    model, X = build()
    model._check()
    threaded = build()[0].set_params(n_threads=2)
    threaded._check()
    expected = _estep(model, X, LENGTHS, state)
    if state in blocks.REPORT_STATES:
        assert (expected[1] != [] or expected[2] != []
                or isinstance(expected[0], tuple))
    assert _estep(threaded, X, LENGTHS, state) == expected


@pytest.mark.parametrize("state", ERROR_STATES, ids=ERROR_STATE_IDS)
@pytest.mark.parametrize("rows", list(FAR_ROWS.values()), ids=list(FAR_ROWS))
@pytest.mark.usefixtures("rows_16")
def test_far_rows_on_threads_give_the_one_thread_outcome(rows, state):
    """Far rows under ``implementation="scaling"``, on two threads, give
    the E-step of one thread under each error state."""
    X = _data()[0][:, :1].copy()
    X[rows] = 1e4
    model = _far_gaussian()
    threaded = _far_gaussian().set_params(n_threads=2)
    for each in (model, threaded):
        # What fit does before its first E-step.
        each._check()
        each._store_checked()
    assert (_estep(threaded, X, LENGTHS, state)
            == _estep(model, X, LENGTHS, state))
