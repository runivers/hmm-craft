"""
Tests of ``scripts/check_ruff_baseline.py``, the style ratchet.

ruff is never run: ``check`` gets a fake ``run_ruff`` that returns the JSON
list ruff prints, written by hand. File names are synthetic.

Run with ``python -m pytest scripts/test_check_ruff_baseline.py``.
"""

import json
import pathlib
import re

import check_ruff_baseline as ratchet
import pytest

UPDATE = "python3 -I scripts/check_ruff_baseline.py --update"


def _finding(root, name, code):
    """Return one entry of ruff's JSON output."""
    return {"code": code, "filename": str(root / name), "message": "m"}


def _fake(root, *findings):
    """Return a ``run_ruff`` that answers with ``findings``."""
    answer = [_finding(root, name, code) for name, code in findings]

    def run_ruff(_root):
        return answer

    return run_ruff


def _write(root, baseline):
    """Write ``ci/ruff_baseline.json`` under ``root``."""
    (root / "ci").mkdir(exist_ok=True)
    (root / "ci" / "ruff_baseline.json").write_text(
        json.dumps(baseline, indent=2) + "\n"
    )


def _check(root, capsys, *findings, update=False):
    """Run ``check``; return ``(status, output)``."""
    status = ratchet.check(root, _fake(root, *findings), update=update)
    return status, capsys.readouterr().out


def test_count_groups_by_relative_file_and_rule(tmp_path):
    """Findings are counted per (file, rule), paths relative to the root."""
    found = [
        _finding(tmp_path, "a/x.py", "F401"),
        _finding(tmp_path, "a/x.py", "F401"),
        _finding(tmp_path, "a/x.py", "I001"),
        _finding(tmp_path, "y.py", "F401"),
    ]
    assert ratchet.count(found, tmp_path) == {
        "a/x.py": {"F401": 2, "I001": 1},
        "y.py": {"F401": 1},
    }


def test_count_names_a_finding_without_a_code(tmp_path):
    """A syntax error has no code; it is counted under its own name."""
    found = [{"code": None, "filename": str(tmp_path / "x.py")}]
    assert ratchet.count(found, tmp_path) == {"x.py": {"syntax-error": 1}}


def test_equal_counts_pass(tmp_path, capsys):
    """The same counts as the baseline exit 0 and print nothing wrong."""
    _write(tmp_path, {"x.py": {"I001": 1}})
    status, out = _check(tmp_path, capsys, ("x.py", "I001"))
    assert status == 0, out


def test_no_findings_and_empty_baseline_pass(tmp_path, capsys):
    """A clean tree against an empty baseline passes."""
    _write(tmp_path, {})
    status, out = _check(tmp_path, capsys)
    assert status == 0, out


def test_a_count_that_rose_fails_and_is_named(tmp_path, capsys):
    """A rule counted more often in a file fails, naming file and rule."""
    _write(tmp_path, {"x.py": {"I001": 1}})
    status, out = _check(tmp_path, capsys, ("x.py", "I001"), ("x.py", "I001"))
    assert status == 1
    assert "x.py" in out and "I001" in out
    assert "1 to 2" in out


def test_a_new_pair_fails_and_is_named(tmp_path, capsys):
    """A rule new in a file, and a new file, fail, each named."""
    _write(tmp_path, {"x.py": {"I001": 1}})
    status, out = _check(
        tmp_path, capsys, ("x.py", "I001"), ("x.py", "F841"), ("z.py", "I001")
    )
    assert status == 1
    assert re.search(r"x\.py.*F841", out)
    assert re.search(r"z\.py.*I001", out)
    assert "x.py: I001" not in out


def test_a_count_that_fell_fails_with_the_remedy(tmp_path, capsys):
    """A lower count with the baseline not lowered fails, with the command."""
    _write(tmp_path, {"x.py": {"I001": 2}, "y.py": {"F841": 1}})
    status, out = _check(tmp_path, capsys, ("x.py", "I001"))
    assert status == 1
    assert "2 to 1" in out
    assert re.search(r"y\.py.*F841.*1 to 0", out)
    assert UPDATE in out


def test_rise_and_fall_are_both_reported(tmp_path, capsys):
    """One run names the rises and the falls together."""
    _write(tmp_path, {"x.py": {"I001": 2}})
    status, out = _check(tmp_path, capsys, ("x.py", "I001"), ("y.py", "F841"))
    assert status == 1
    assert "y.py" in out and "x.py" in out


def test_update_writes_the_counts_sorted(tmp_path, capsys):
    """--update rewrites the baseline from the findings, keys sorted."""
    _write(tmp_path, {"x.py": {"I001": 3}, "b.py": {"F841": 2}})
    status, out = _check(
        tmp_path, capsys, ("x.py", "I001"), ("b.py", "F841"), update=True
    )
    assert status == 0, out
    text = (tmp_path / "ci" / "ruff_baseline.json").read_text()
    assert json.loads(text) == {"b.py": {"F841": 1}, "x.py": {"I001": 1}}
    assert list(json.loads(text)) == ["b.py", "x.py"]
    assert text.endswith("}\n")
    status, out = _check(tmp_path, capsys, ("x.py", "I001"), ("b.py", "F841"))
    assert status == 0, out


def test_update_refuses_to_raise_a_count(tmp_path, capsys):
    """--update never records a rise: the floor only moves down."""
    _write(tmp_path, {"x.py": {"I001": 1}})
    before = (tmp_path / "ci" / "ruff_baseline.json").read_text()
    status, out = _check(
        tmp_path, capsys, ("x.py", "I001"), ("x.py", "I001"), update=True
    )
    assert status == 1
    assert "x.py" in out
    assert (tmp_path / "ci" / "ruff_baseline.json").read_text() == before


def test_missing_baseline_is_an_error(tmp_path, capsys):
    """Without a baseline file the status is 2, not a verdict."""
    status, out = _check(tmp_path, capsys)
    assert status == 2
    assert "ruff_baseline.json" in out


@pytest.mark.parametrize(
    "text", ["not json", "[1]", '{"x.py": 1}', '{"x.py": {"I001": "1"}}']
)
def test_unreadable_baseline_is_an_error(tmp_path, capsys, text):
    """A baseline that is not ``{file: {rule: int}}`` is status 2."""
    (tmp_path / "ci").mkdir()
    (tmp_path / "ci" / "ruff_baseline.json").write_text(text)
    status, out = _check(tmp_path, capsys)
    assert status == 2
    assert "ruff_baseline.json" in out


def test_ruff_failure_is_an_error(tmp_path, capsys):
    """A ruff that cannot run or answers badly is status 2."""
    _write(tmp_path, {})

    def broken(_root):
        raise ratchet.RuffFailure("no ruff here")

    status = ratchet.check(tmp_path, broken)
    out = capsys.readouterr().out
    assert status == 2
    assert "no ruff here" in out


def test_parse_ruff_output_rejects_junk():
    """Output that is not a JSON list of objects is a ``RuffFailure``."""
    with pytest.raises(ratchet.RuffFailure):
        ratchet.parse_output("not json")
    with pytest.raises(ratchet.RuffFailure):
        ratchet.parse_output('{"a": 1}')
    assert ratchet.parse_output("[]") == []


def test_rule_list_is_explicit_codes_without_duplicates():
    """The set is a list of rule codes, never a prefix or ``ALL``."""
    assert len(ratchet.RULES) > 100
    assert len(set(ratchet.RULES)) == len(ratchet.RULES)
    assert all(re.fullmatch(r"[A-Z]{1,6}[0-9]{3,4}", c) for c in ratchet.RULES)
    assert list(ratchet.RULES) == sorted(ratchet.RULES)


def test_the_committed_baseline_is_well_formed():
    """The repository's baseline reads as ``{file: {rule: positive int}}``.

    It may be empty.
    """
    root = pathlib.Path(ratchet.__file__).resolve().parent.parent
    baseline = ratchet.read_baseline(root)
    for name, rules in baseline.items():
        assert not name.startswith("/"), name
        assert all(n > 0 and c in ratchet.RULES for c, n in rules.items())
