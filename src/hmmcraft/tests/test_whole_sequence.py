import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from hmmcraft import hmm

# upstream's CategoricalHMM test model (test_categorical_hmm.py)
STARTPROB = np.array([0.6, 0.4])
TRANSMAT = np.array([[0.7, 0.3], [0.4, 0.6]])
EMISSIONPROB = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])


def _model(implementation, algorithm="viterbi"):
    h = hmm.CategoricalHMM(
        2, algorithm=algorithm, implementation=implementation
    )
    h.n_features = 3
    h.startprob_ = STARTPROB
    h.transmat_ = TRANSMAT
    h.emissionprob_ = EMISSIONPROB
    return h


def _symbols(n_samples, seed):
    return np.random.RandomState(seed).randint(3, size=(n_samples, 1))


def _forward_filter(X):
    """P(state at t | samples 0..t), one normalised step at a time."""
    filtered = []
    alpha = STARTPROB * EMISSIONPROB[:, X[0, 0]]
    filtered.append(alpha / alpha.sum())
    for x in X[1:, 0]:
        alpha = (filtered[-1] @ TRANSMAT) * EMISSIONPROB[:, x]
        filtered.append(alpha / alpha.sum())
    return np.array(filtered)


@pytest.mark.parametrize("implementation", ["log", "scaling"])
def test_last_posterior_of_each_prefix_is_the_filtered_probability(
    implementation,
):
    """
    The last row of ``predict_proba(X[:t + 1])`` is the probability of each
    state given the samples up to ``t`` only, for every prefix.
    """
    h = _model(implementation)
    X = _symbols(30, seed=0)
    expected = _forward_filter(X)
    actual = np.array([h.predict_proba(X[: t + 1])[-1] for t in range(len(X))])
    assert_allclose(actual, expected, rtol=1e-10, atol=0)


@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize(
    "method, algorithm",
    [
        ("predict_proba", "viterbi"),
        ("predict", "viterbi"),
        ("predict", "map"),
    ],
)
def test_each_sequence_uses_only_its_own_samples(
    implementation, method, algorithm
):
    """
    With ``lengths``, the result for one sequence equals the result for that
    sequence alone. The two sequences are chosen so that every result
    differs when they are decoded as one sequence.
    """
    h = _model(implementation, algorithm)
    first, second = np.array([[0], [1], [1]]), np.array([[0], [0]])
    together = getattr(h, method)(np.concatenate([first, second]), [3, 2])
    alone = np.concatenate(
        [getattr(h, method)(first), getattr(h, method)(second)]
    )
    assert_array_equal(together, alone)
