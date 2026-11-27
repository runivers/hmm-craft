"""Input checks and values of the C++ kernel, ``hmmcraft._hmmc``."""

import numpy as np
import pytest

from hmmcraft import _hmmc, base
from hmmcraft.hmm import CategoricalHMM

UNDERFLOW_TEXT = (
    "forward pass failed with underflow; consider using "
    "implementation='log' instead"
)


def _arguments(name, n_samples, n_components):
    """Return valid-looking arguments of kernel function ``name``.

    The matrices have ``n_samples`` rows and ``n_components`` columns.
    """
    start = np.full(n_components, 1 / max(n_components, 1))
    trans = np.full((n_components, n_components), 1 / max(n_components, 1))
    frame = np.full((n_samples, n_components), 0.5)
    lattice = np.zeros((n_samples, n_components))
    return {
        "forward_log": (start, trans, frame),
        "forward_scaling": (start, trans, frame),
        "backward_log": (start, trans, frame),
        "backward_scaling": (start, trans, frame, np.ones(n_samples)),
        "compute_log_xi_sum": (lattice, trans, lattice, frame),
        "compute_scaling_xi_sum": (lattice, trans, lattice, frame),
        "viterbi": (start, trans, frame),
        "estep_log": (start, trans, frame, np.array([n_samples]), True),
        "estep_scaling": (start, trans, frame, np.array([n_samples]), True),
    }[name]


KERNEL_FUNCTIONS = [
    "forward_log",
    "forward_scaling",
    "backward_log",
    "backward_scaling",
    "compute_log_xi_sum",
    "compute_scaling_xi_sum",
    "viterbi",
    "estep_log",
    "estep_scaling",
]


@pytest.mark.parametrize("name", KERNEL_FUNCTIONS)
def test_kernel_refuses_a_sequence_without_samples(name):
    """A matrix with no rows raises before any row is read or written."""
    with pytest.raises(ValueError, match="at least one sample"):
        getattr(_hmmc, name)(*_arguments(name, 0, 2))


@pytest.mark.parametrize("name", KERNEL_FUNCTIONS)
def test_kernel_refuses_a_model_without_states(name):
    """A model with no states raises before any column is read."""
    with pytest.raises(ValueError, match="at least one state"):
        getattr(_hmmc, name)(*_arguments(name, 3, 0))


@pytest.mark.parametrize("name", KERNEL_FUNCTIONS)
def test_kernel_accepts_a_single_sample(name):
    """One sample is the smallest sequence and runs."""
    getattr(_hmmc, name)(*_arguments(name, 1, 2))


class _NoRowsCategoricalHMM(CategoricalHMM):
    """An emission hook that returns no rows, whatever ``X`` holds."""

    def _compute_likelihood(self, X):
        return super()._compute_likelihood(X)[:0]

    def _compute_log_likelihood(self, X):
        return super()._compute_log_likelihood(X)[:0]


class _OneColumnCategoricalHMM(CategoricalHMM):
    """An emission hook that returns one state's column of two."""

    def _compute_likelihood(self, X):
        return super()._compute_likelihood(X)[:, :1]

    def _compute_log_likelihood(self, X):
        return super()._compute_log_likelihood(X)[:, :1]


def _hooked(cls, implementation):
    h = cls(2, n_features=2, init_params="", implementation=implementation)
    h.startprob_ = np.array([0.5, 0.5])
    h.transmat_ = np.array([[0.5, 0.5], [0.5, 0.5]])
    h.emissionprob_ = np.array([[0.5, 0.5], [0.5, 0.5]])
    return h


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("method", ["score", "decode", "predict_proba", "fit"])
def test_emission_hook_without_rows_raises(implementation, method):
    """A subclass whose emissions have no rows gets an error, not a value."""
    h = _hooked(_NoRowsCategoricalHMM, implementation)
    with pytest.raises(ValueError, match="at least one sample"):
        getattr(h, method)(np.array([[0], [1]]))


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("method", ["score", "decode", "predict_proba", "fit"])
def test_emission_hook_with_too_few_columns_raises(implementation, method):
    """
    A subclass whose emissions do not have one column per state gets the
    kernel's shape error under either implementation, not a fallback.
    """
    h = _hooked(_OneColumnCategoricalHMM, implementation)
    with pytest.raises(ValueError, match=r"^shape mismatch$"):
        getattr(h, method)(np.array([[0], [1]]))


@pytest.mark.parametrize("row", [0, 2], ids=["first-row", "later-row"])
def test_forward_scaling_underflow_text_starts_with_the_constant(row):
    """
    A sample whose probabilities sum to zero over the states raises
    ``ValueError`` with the kernel's full text, which the package recognises
    by its start.
    """
    start, trans, frame = _arguments("forward_scaling", 4, 2)
    frame[row] = 0
    with pytest.raises(ValueError) as info:
        _hmmc.forward_scaling(start, trans, frame)
    assert type(info.value) is ValueError
    assert str(info.value) == UNDERFLOW_TEXT
    assert str(info.value).startswith(base._SCALING_UNDERFLOW)
    assert base._is_scaling_underflow(info.value)


@pytest.mark.parametrize(
    "n_samples, n_components, frame_columns",
    [
        (0, 2, 2),
        (3, 0, 0),
        (3, 2, 3),
    ],
    ids=["no-sample", "no-state", "shape-mismatch"],
)
def test_other_kernel_errors_are_not_taken_for_underflow(
    n_samples, n_components, frame_columns
):
    """The kernel's input errors never match the underflow test."""
    start, trans, _ = _arguments("forward_scaling", n_samples, n_components)
    frame = np.full((n_samples, frame_columns), 0.5)
    with pytest.raises(ValueError) as info:
        _hmmc.forward_scaling(start, trans, frame)
    assert not base._is_scaling_underflow(info.value)


BLOCK_FUNCTIONS = ["estep_log", "estep_scaling"]
BLOCK_RESULTS = ["log_prob", "fwdlattice", "bwdlattice", "xi_sum"]
SHORT_AND_LONG = [1, 5, 1, 2, 13, 3, 1, 8]


def _block_inputs(implementation, n_components, lengths, data, seed=0):
    """
    Return seeded start probabilities, a transition matrix and the stacked
    emission lattice of sequences of ``lengths`` rows.

    ``data`` is ``"dense"`` (no zero) or ``"zeros"``: a zero start
    probability and a zero in each transition row when there are several
    states, and zero emission probabilities scattered over the lattice. The
    log lattice holds ``-inf`` where the probability is zero.
    """
    rng = np.random.RandomState(seed)
    n_rows = int(np.sum(lengths))
    start = rng.rand(n_components) + 0.1
    trans = rng.rand(n_components, n_components) + 0.1
    frame = rng.rand(n_rows, n_components) + 0.01
    if data == "zeros":
        if n_components > 1:
            start[rng.randint(n_components)] = 0
            columns = rng.randint(n_components, size=n_components)
            trans[np.arange(n_components), columns] = 0
        frame[rng.rand(n_rows, n_components) < 0.25] = 0
    start /= start.sum()
    trans /= trans.sum(axis=1, keepdims=True)
    if implementation == "scaling":
        return start, trans, frame
    log_frame = np.full_like(frame, -np.inf)
    positive = frame > 0
    log_frame[positive] = np.log(frame[positive])
    return start, trans, log_frame


def _per_sequence(implementation, start, trans, lattice, lengths):
    """
    Return what the per-sequence kernel functions give for each sequence,
    stacked as the block functions stack it.

    Under scaling, a sequence whose forward pass raises the underflow error
    gets the flag and NaN in every result.
    """
    log_probs, fwds, bwds, sums, flags = [], [], [], [], []
    n_components = lattice.shape[1]
    for rows in np.split(lattice, np.cumsum(lengths)[:-1]):
        flag = False
        if implementation == "log":
            log_prob, fwd = _hmmc.forward_log(start, trans, rows)
            bwd = _hmmc.backward_log(start, trans, rows)
            xi_sum = _hmmc.compute_log_xi_sum(fwd, trans, bwd, rows)
        else:
            try:
                log_prob, fwd, scaling = _hmmc.forward_scaling(
                    start, trans, rows
                )
            except ValueError as exc:
                assert str(exc) == UNDERFLOW_TEXT
                flag = True
                log_prob = np.nan
                fwd = np.full(rows.shape, np.nan)
                bwd = np.full(rows.shape, np.nan)
                xi_sum = np.full((n_components, n_components), np.nan)
            else:
                bwd = _hmmc.backward_scaling(start, trans, rows, scaling)
                xi_sum = _hmmc.compute_scaling_xi_sum(fwd, trans, bwd, rows)
        log_probs.append(log_prob)
        fwds.append(fwd)
        bwds.append(bwd)
        sums.append(xi_sum)
        flags.append(flag)
    expected = dict(
        zip(
            BLOCK_RESULTS,
            [
                np.array(log_probs),
                np.concatenate(fwds),
                np.concatenate(bwds),
                np.stack(sums),
            ],
            strict=False,
        )
    )
    if implementation == "scaling":
        expected["underflow"] = np.array(flags)
    return expected


def _block(implementation, start, trans, lattice, lengths, with_xi_sum=True):
    """Call the block function of ``implementation``; name its results."""
    results = getattr(_hmmc, "estep_" + implementation)(
        start, trans, lattice, lengths, with_xi_sum
    )
    names = BLOCK_RESULTS + (
        ["underflow"] if implementation == "scaling" else []
    )
    assert len(results) == len(names)
    return dict(zip(names, results, strict=False))


def _assert_same_bytes(got, expected):
    """Every result has the expected dtype, shape and bytes, NaN included."""
    assert got.keys() == expected.keys()
    for key in expected:
        assert got[key].dtype == expected[key].dtype, key
        assert got[key].shape == expected[key].shape, key
        assert got[key].tobytes() == expected[key].tobytes(), key


@pytest.mark.parametrize("data", ["dense", "zeros"])
@pytest.mark.parametrize(
    "lengths",
    [[1], [7], [1, 1, 1], SHORT_AND_LONG, [*SHORT_AND_LONG, 300]],
    ids=["one-row", "one-sequence", "one-row-each", "short", "short-long"],
)
@pytest.mark.parametrize("n_components", [1, 2, 8, 11])
@pytest.mark.parametrize("implementation", ["log", "scaling"])
def test_block_functions_equal_the_per_sequence_functions_as_bytes(
    implementation, n_components, lengths, data
):
    """
    For each sequence of a block, the block function returns the bytes of
    the per-sequence forward, backward and transition-sum functions, in new
    C-contiguous writeable arrays, and leaves its inputs unchanged.
    """
    start, trans, lattice = _block_inputs(
        implementation, n_components, lengths, data
    )
    lengths = np.array(lengths, dtype=np.intp)
    inputs = [start, trans, lattice, lengths]
    before = [array.tobytes() for array in inputs]
    got = _block(implementation, start, trans, lattice, lengths)
    assert [array.tobytes() for array in inputs] == before
    expected = _per_sequence(implementation, start, trans, lattice, lengths)
    _assert_same_bytes(got, expected)
    for array in got.values():
        assert array.flags.c_contiguous
        assert array.flags.writeable


def test_estep_log_of_a_hand_written_one_state_block():
    """
    One state, sequences of one and two rows: the values worked out by hand,
    all exact in binary.
    """
    got = _block(
        "log",
        np.array([1.0]),
        np.array([[1.0]]),
        np.array([[-1.0], [-2.0], [-0.5]]),
        np.array([1, 2]),
    )
    expected = {
        "log_prob": np.array([-1.0, -2.5]),
        "fwdlattice": np.array([[-1.0], [-2.0], [-2.5]]),
        "bwdlattice": np.array([[0.0], [-0.5], [0.0]]),
        "xi_sum": np.array([[[-np.inf]], [[0.0]]]),
    }
    _assert_same_bytes(got, expected)


UNDERFLOW_LENGTHS = [3, 4, 1, 2]


@pytest.mark.parametrize(
    "rows, value, flagged",
    [
        ([3], 0.0, [False, True, False, False]),
        ([4], 0.0, [False, True, False, False]),
        ([6], 0.0, [False, True, False, False]),
        ([7], 0.0, [False, False, True, False]),
        ([1, 8], 0.0, [True, False, False, True]),
        ([4], 1e-301, [False, True, False, False]),
        ([4], 1e-299, [False, False, False, False]),
    ],
    ids=[
        "zero-first-row",
        "zero-middle-row",
        "zero-last-row",
        "zero-one-row-sequence",
        "two-sequences",
        "below-the-floor",
        "above-the-floor",
    ],
)
def test_estep_scaling_flags_an_underflow_and_raises_nothing(
    rows, value, flagged
):
    """
    A sequence whose forward pass would raise the underflow error gets its
    flag and NaN in every result; the other sequences are computed as the
    per-sequence functions compute them.
    """
    start, trans, frame = _block_inputs(
        "scaling", 2, UNDERFLOW_LENGTHS, "dense"
    )
    frame[rows] = value
    lengths = np.array(UNDERFLOW_LENGTHS)
    got = _block("scaling", start, trans, frame, lengths)
    assert got["underflow"].tolist() == flagged
    ends = np.cumsum(UNDERFLOW_LENGTHS)
    for s, sequence in enumerate(np.split(frame, ends[:-1])):
        if not flagged[s]:
            continue
        assert np.isnan(got["log_prob"][s])
        own_rows = slice(ends[s] - len(sequence), ends[s])
        assert np.isnan(got["fwdlattice"][own_rows]).all()
        assert np.isnan(got["bwdlattice"][own_rows]).all()
        assert np.isnan(got["xi_sum"][s]).all()
        with pytest.raises(ValueError) as info:
            _hmmc.forward_scaling(start, trans, sequence)
        assert str(info.value) == UNDERFLOW_TEXT
    _assert_same_bytes(
        got, _per_sequence("scaling", start, trans, frame, lengths)
    )


def test_estep_log_of_a_zero_probability_sequence_raises_nothing():
    """
    A sequence with a row of zero probability gets a log-likelihood of
    ``-inf`` and NaN transition sums, the bytes of the per-sequence functions.
    """
    start, trans, log_frame = _block_inputs(
        "log", 3, UNDERFLOW_LENGTHS, "dense"
    )
    log_frame[4] = -np.inf
    lengths = np.array(UNDERFLOW_LENGTHS)
    got = _block("log", start, trans, log_frame, lengths)
    assert got["log_prob"][1] == -np.inf
    assert np.isfinite(got["log_prob"][[0, 2, 3]]).all()
    assert np.isnan(got["xi_sum"][1]).all()
    _assert_same_bytes(
        got, _per_sequence("log", start, trans, log_frame, lengths)
    )


@pytest.mark.parametrize("implementation", ["log", "scaling"])
def test_block_functions_without_transition_sums(implementation):
    """
    Without transition sums, the sums have no slice and every other result
    is the one computed with them.
    """
    start, trans, lattice = _block_inputs(
        implementation, 3, SHORT_AND_LONG, "zeros"
    )
    lengths = np.array(SHORT_AND_LONG)
    with_sums = _block(implementation, start, trans, lattice, lengths, True)
    without = _block(implementation, start, trans, lattice, lengths, False)
    assert without["xi_sum"].dtype == np.float64
    assert without["xi_sum"].shape == (0, 3, 3)
    del with_sums["xi_sum"], without["xi_sum"]
    _assert_same_bytes(without, with_sums)


@pytest.mark.parametrize(
    "layout",
    [
        "fortran",
        "every-second-row",
        "float32",
        "int32-lengths",
        "list-lengths",
    ],
)
@pytest.mark.parametrize("implementation", ["log", "scaling"])
def test_block_functions_read_other_layouts_and_dtypes(implementation, layout):
    """
    A lattice that is not a C-contiguous float64 array, and lengths of
    another integer type or a list, give the bytes of the per-sequence
    functions on the same inputs.
    """
    start, trans, lattice = _block_inputs(
        implementation, 3, SHORT_AND_LONG, "dense"
    )
    lengths = np.array(SHORT_AND_LONG, dtype=np.intp)
    if layout == "fortran":
        lattice = np.asfortranarray(lattice)
        assert not lattice.flags.c_contiguous
    elif layout == "every-second-row":
        taller = np.full((2 * len(lattice), lattice.shape[1]), 0.25)
        taller[::2] = lattice
        lattice = taller[::2]
        assert not lattice.flags.c_contiguous
    elif layout == "float32":
        lattice = lattice.astype(np.float32)
    elif layout == "int32-lengths":
        lengths = lengths.astype(np.int32)
    else:
        lengths = list(SHORT_AND_LONG)
    expected = _per_sequence(
        implementation, start, trans, lattice, SHORT_AND_LONG
    )
    _assert_same_bytes(
        _block(implementation, start, trans, lattice, lengths), expected
    )


SUM_TEXT = "^lengths do not sum to the lattice rows$"
SAMPLE_TEXT = "^a sequence needs at least one sample$"
SHAPE_TEXT = "^shape mismatch$"


@pytest.mark.parametrize(
    "change, text",
    [
        ({"lengths": np.array([2, 2])}, SUM_TEXT),
        ({"lengths": np.array([2, 4])}, SUM_TEXT),
        ({"lengths": np.array([2**62, 2**62])}, SUM_TEXT),
        ({"lengths": np.array([], dtype=np.intp)}, SUM_TEXT),
        ({"lengths": np.array([2, 0, 3])}, SAMPLE_TEXT),
        ({"lengths": np.array([-1, 6])}, SAMPLE_TEXT),
        ({"lengths": np.array([[2, 3]])}, SHAPE_TEXT),
        ({"lattice": np.full(5, 0.5)}, SHAPE_TEXT),
        ({"start": np.ones(1)}, SHAPE_TEXT),
        ({"trans": np.full((2, 3), 1 / 3)}, SHAPE_TEXT),
    ],
    ids=[
        "sum-below-rows",
        "sum-above-rows",
        "huge-lengths",
        "no-lengths",
        "zero-length",
        "negative-length",
        "2d-lengths",
        "1d-lattice",
        "short-start",
        "non-square-transitions",
    ],
)
@pytest.mark.parametrize("name", BLOCK_FUNCTIONS)
def test_block_functions_refuse_inconsistent_inputs(name, change, text):
    """Lengths and shapes that do not fit raise before any row is read."""
    arguments = {
        "start": np.full(2, 0.5),
        "trans": np.full((2, 2), 0.5),
        "lattice": np.full((5, 2), 0.5),
        "lengths": np.array([2, 3]),
    }
    arguments.update(change)
    with pytest.raises(ValueError, match=text):
        getattr(_hmmc, name)(
            arguments["start"],
            arguments["trans"],
            arguments["lattice"],
            arguments["lengths"],
            True,
        )
