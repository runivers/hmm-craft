"""
The package must not use numpy API that numpy has deprecated.

Assigning to ``ndarray.shape`` warns from numpy 2.5 on; ``reshape`` replaces
it. The source is scanned for such assignments, and the one-feature
initialisation of the models that once gave a 0-d covariance a ``(1, 1)``
shape this way runs with a ``DeprecationWarning`` from the package turned
into an error.
"""

import ast
import warnings
from pathlib import Path

import numpy as np
import pytest

from .. import hmm


def _assignment_targets(node):
    if isinstance(node, ast.Assign):
        return node.targets
    if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        return [node.target]
    return []


def _shape_assignments():
    found = []
    for path in sorted(Path(hmm.__file__).parent.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            for target in _assignment_targets(node):
                if isinstance(target, ast.Attribute) and (
                        target.attr == "shape"):
                    found.append(f"{path.name}:{node.lineno}")
    return found


def test_no_module_assigns_to_an_array_shape():
    assert _shape_assignments() == []


@pytest.mark.parametrize("covariance_type",
                         ["spherical", "diag", "full", "tied"])
@pytest.mark.parametrize("model_class, kwargs",
                         [(hmm.GaussianHMM, {}),
                          (hmm.GMMHMM, {"n_mix": 2})],
                         ids=["GaussianHMM", "GMMHMM"])
def test_one_feature_init_assigns_no_shape(
        model_class, kwargs, covariance_type):
    X = np.random.RandomState(0).normal(size=(40, 1))
    model = model_class(2, covariance_type=covariance_type, n_iter=1,
                        random_state=0, **kwargs)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", category=DeprecationWarning, module=r"hmmcraft\..*")
        model.fit(X)
    assert np.isfinite(model.score(X))
