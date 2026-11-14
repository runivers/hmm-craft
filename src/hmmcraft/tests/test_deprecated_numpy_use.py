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
    """Expressions a statement or comprehension clause binds to."""
    if isinstance(node, ast.Assign):
        return node.targets
    if isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For,
                         ast.AsyncFor, ast.comprehension)):
        return [node.target]
    if isinstance(node, (ast.With, ast.AsyncWith)):
        return [item.optional_vars for item in node.items
                if item.optional_vars is not None]
    return []


def _leaves(target):
    """The names and attributes inside a tuple, list or starred target."""
    if isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _leaves(element)
    elif isinstance(target, ast.Starred):
        yield from _leaves(target.value)
    else:
        yield target


def _is_setattr_of_shape(node):
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "shape")


def _shape_assignments_in(source):
    """Line numbers of the assignments to a ``.shape`` attribute in source."""
    found = []
    for node in ast.walk(ast.parse(source)):
        for target in _assignment_targets(node):
            found += [leaf.lineno for leaf in _leaves(target)
                      if isinstance(leaf, ast.Attribute)
                      and leaf.attr == "shape"]
        if _is_setattr_of_shape(node):
            found.append(node.lineno)
    return sorted(found)


def _shape_assignments():
    found = []
    for path in sorted(Path(hmm.__file__).parent.glob("*.py")):
        found += [f"{path.name}:{line}"
                  for line in _shape_assignments_in(path.read_text())]
    return found


@pytest.mark.parametrize("source", [
    "a.shape = (2, 3)",
    "a.shape += (1,)",
    "a.shape: tuple = (2, 3)",
], ids=["assign", "augmented", "annotated"])
def test_scan_finds_a_plain_shape_assignment(source):
    assert _shape_assignments_in(source) == [1]


@pytest.mark.parametrize("source", [
    "a.shape, b = (2, 3), 1",
    "[a.shape, b] = (2, 3), 1",
    "(b, (a.shape, c)) = 1, ((2, 3), 2)",
    "b, *a.shape = 1, 2, 3",
    "for a.shape in [(2, 3)]: pass",
    "with open('f') as a.shape: pass",
    "setattr(a, 'shape', (2, 3))",
    "setattr(self.cv, \"shape\", (1, 1))",
], ids=["tuple", "list", "nested", "starred", "for", "with", "setattr",
        "setattr_attribute"])
def test_scan_finds_an_unpacking_or_setattr_shape_assignment(source):
    assert _shape_assignments_in(source) == [1]


@pytest.mark.parametrize("source", [
    "a.reshape(2, 3)",
    "a, b = x.shape",
    "setattr(a, 'size', 3)",
    "setattr(a, name, 3)",
    "shape = (2, 3)",
], ids=["reshape", "read", "other_attribute", "dynamic_name", "local_name"])
def test_scan_ignores_code_that_does_not_assign_a_shape(source):
    assert _shape_assignments_in(source) == []


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
