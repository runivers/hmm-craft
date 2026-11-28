"""
Run the examples in the package's docstrings.

Each docstring of a :mod:`hmmcraft` module that holds examples is one
case, run by the standard :mod:`doctest` finder and runner as a reader
would type it: in a namespace of its own, with the ``ELLIPSIS`` option
that pytest's doctest run also sets, under the suite's warning filters.
The modules of the test package are not searched.
"""

import doctest
import importlib
import pkgutil
import random

import numpy as np
import pytest

import hmmcraft


def _raise_import_error(name):
    # Without it, pkgutil.walk_packages skips a subpackage that fails to
    # import; it is called inside the handler, so the cause is chained.
    raise ImportError(f"cannot import {name}")


def _module_names():
    """Return the names of the package's modules, the tests excluded."""
    names = [hmmcraft.__name__]
    names.extend(
        info.name
        for info in pkgutil.walk_packages(
            hmmcraft.__path__, "hmmcraft.", onerror=_raise_import_error
        )
        if not (info.name + ".").startswith("hmmcraft.tests.")
    )
    return names


def _docstrings_with_examples():
    """Return a doctest for each docstring of the package with examples."""
    finder = doctest.DocTestFinder()
    tests = []
    for name in _module_names():
        module = importlib.import_module(name)
        tests.extend(
            test for test in finder.find(module, globs={}) if test.examples
        )
    return sorted(tests)


DOCTESTS = _docstrings_with_examples()


def test_docstring_examples_are_found():
    """The search finds examples, so the run below cannot pass on none."""
    assert DOCTESTS


@pytest.mark.parametrize(
    "test", [pytest.param(test, id=test.name) for test in DOCTESTS]
)
def test_docstring_examples_print_what_they_show(test):
    """
    Each example of the docstring prints the output written under it.

    The examples leave the global random state as they found it, so the
    tests after them draw the same numbers.
    """
    np_state = np.random.get_state()
    py_state = random.getstate()
    report = []
    runner = doctest.DocTestRunner(verbose=False, optionflags=doctest.ELLIPSIS)
    try:
        result = runner.run(test, out=report.append)
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
    if result.failed:
        pytest.fail("".join(report), pytrace=False)
