"""
Tests that the declared minimums are the ones a CI cell installs and runs.

A floor is never declared on a cell that has not run: the floor cell installs
exactly the pins of ``ci/floor_constraints.txt``, so each minimum in
``pyproject.toml`` must equal its pin there. The pinned build requirements of
``ci/build_requirements.txt`` must satisfy what ``[build-system]`` asks for.
The documentation lock names the docs extra and the runtime dependencies, each
pinned at a version the declaration allows.

Run with ``python -m pytest scripts/test_floor_constraints.py``.
"""

import pathlib
import re
import tomllib

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = pathlib.Path(__file__).resolve().parent.parent
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\;]+)")


def _pins(path):
    """Return ``{name: version}`` for each ``name==version`` line of a file."""
    pins = {}
    for line in path.read_text().splitlines():
        match = PIN.match(line)
        if match:
            pins[canonicalize_name(match.group(1))] = Version(match.group(2))
    return pins


def _pyproject():
    with open(ROOT / "pyproject.toml", "rb") as file:
        return tomllib.load(file)


def _declared_dependencies():
    texts = _pyproject()["project"]["dependencies"]
    return [Requirement(text) for text in texts]


def _only(requirement):
    """Return the lower bound of a requirement made of one ``>=`` clause."""
    clauses = list(requirement.specifier)
    assert [c.operator for c in clauses] == [">="], requirement
    return Version(clauses[0].version)


def test_floor_file_pins_exactly():
    text = (ROOT / "ci" / "floor_constraints.txt").read_text()
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert lines
    for line in lines:
        assert PIN.fullmatch(line), f"not an exact pin: {line!r}"


@pytest.mark.parametrize("requirement", _declared_dependencies(), ids=str)
def test_declared_minimum_is_the_floor_pin(requirement):
    pins = _pins(ROOT / "ci" / "floor_constraints.txt")
    name = canonicalize_name(requirement.name)
    assert name in pins, f"{name} is declared but not pinned by the floor cell"
    assert pins[name] == _only(requirement)


def test_every_runtime_dependency_is_declared_once():
    names = [canonicalize_name(r.name) for r in _declared_dependencies()]
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "requirement",
    [Requirement(text) for text in _pyproject()["build-system"]["requires"]],
    ids=str,
)
def test_pinned_build_requirement_satisfies_the_declared_one(requirement):
    pins = _pins(ROOT / "ci" / "build_requirements.txt")
    name = canonicalize_name(requirement.name)
    assert name in pins, f"{name} is required to build but not pinned"
    assert requirement.specifier.contains(pins[name]), (name, pins[name])


DOCS_IN = ROOT / "ci" / "docs_requirements.in"
DOCS_TXT = ROOT / "ci" / "docs_requirements.txt"


def _docs_requirements():
    """Return the docs extra and the runtime dependencies, as requirements."""
    project = _pyproject()["project"]
    texts = project["optional-dependencies"]["docs"] + project["dependencies"]
    return [Requirement(text) for text in texts]


def _docs_in():
    """Return ``{name: version}`` for the ``.in`` file's exact pins."""
    assert DOCS_IN.exists(), "ci/docs_requirements.in is missing"
    return _pins(DOCS_IN)


@pytest.mark.parametrize("requirement", _docs_requirements(), ids=str)
def test_docs_lock_pins_each_docs_and_runtime_requirement(requirement):
    name = canonicalize_name(requirement.name)
    direct = _docs_in()
    assert name in direct, f"{name} is declared but not in the docs lock"
    assert requirement.specifier.contains(direct[name]), (name, direct[name])
    assert DOCS_TXT.exists(), "ci/docs_requirements.txt is missing"
    assert _pins(DOCS_TXT).get(name) == direct[name]


def test_docs_lock_names_only_the_docs_extra_and_the_dependencies():
    expected = {canonicalize_name(r.name) for r in _docs_requirements()}
    assert set(_docs_in()) == expected
