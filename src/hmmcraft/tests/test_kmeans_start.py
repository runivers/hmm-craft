"""
The seeded k-means start does not depend on the number of threads.

scikit-learn's k-means adds per-thread partial sums, so its result can change
with the number of threads. ``GaussianHMM``, ``GMMHMM`` and
``VariationalGaussianHMM`` run it on one thread.
"""

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from ..hmm import GMMHMM, GaussianHMM
from ..vhmm import VariationalGaussianHMM

COVARIANCE_TYPES = ["spherical", "diag", "full", "tied"]
MODELS = {
    "GaussianHMM": lambda covariance_type: GaussianHMM(
        3, covariance_type, random_state=0, n_iter=0),
    "GMMHMM": lambda covariance_type: GMMHMM(
        3, n_mix=2, covariance_type=covariance_type, random_state=0,
        n_iter=0),
    "VariationalGaussianHMM": lambda covariance_type: VariationalGaussianHMM(
        3, covariance_type=covariance_type, random_state=0, n_iter=0),
}


def _three_groups(n_samples=1200, seed=0):
    # Three groups in random order. scikit-learn's k-means shares blocks of
    # 256 rows among its threads and adds the threads' sums, so with two
    # threads or more the rows of every group are summed in parts.
    rng = np.random.RandomState(seed)
    centers = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    return (centers[rng.randint(3, size=n_samples)]
            + rng.normal(size=(n_samples, 2)))


def _start(model, covariance_type, threads):
    """Every attribute ``fit`` sets when it runs no iteration: the start."""
    h = MODELS[model](covariance_type)
    with threadpool_limits(limits=threads):
        h.fit(_three_groups())
    return {name: np.asarray(value) for name, value in vars(h).items()
            if name.endswith("_") and name != "monitor_"}


@pytest.mark.parametrize("threads", [2, 4])
@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPES)
@pytest.mark.parametrize("model", list(MODELS))
def test_seeded_kmeans_start_is_the_one_thread_start(
        model, covariance_type, threads):
    """
    With ``random_state`` fixed, the start at two or four threads has the
    bytes of the start at one thread, in every array it sets. Unless
    ``OMP_NUM_THREADS`` is set, scikit-learn uses no more OpenMP threads than
    the machine has cores, so on one core this test cannot fail.
    """
    one = _start(model, covariance_type, 1)
    many = _start(model, covariance_type, threads)
    assert many.keys() == one.keys()
    for name, value in one.items():
        assert many[name].dtype == value.dtype, name
        assert many[name].shape == value.shape, name
        assert many[name].tobytes() == value.tobytes(), name
