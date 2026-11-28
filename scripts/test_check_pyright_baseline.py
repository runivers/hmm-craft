"""
Tests of ``scripts/check_pyright_baseline.py``, the type-check ratchet.

pyright is never run: ``check`` gets a fake ``run`` that returns the object
pyright prints with ``--outputjson``, written by hand, and a fake lookup of
installed versions. File names and package names are synthetic.

Run with ``python -m pytest scripts/test_check_pyright_baseline.py``.
"""

import json
import pathlib
import re
import subprocess
import sys

import check_pyright_baseline as ratchet
import pytest

UPDATE = "python -I scripts/check_pyright_baseline.py --update"
SCRIPT = pathlib.Path(ratchet.__file__).resolve()


def _diagnostic(root, name, rule, severity="error"):
    """Return one entry of pyright's ``generalDiagnostics``."""
    item = {"file": str(root / name), "severity": severity, "message": "m"}
    if rule is not None:
        item["rule"] = rule
    return item


def _report(root, *found, analysed=3):
    """Return the object pyright prints, with ``found`` as (file, rule)."""
    return {
        "generalDiagnostics": [_diagnostic(root, n, r) for n, r in found],
        "summary": {"filesAnalyzed": analysed},
    }


def _fake(report):
    """Return a ``run`` that answers with ``report``, and records calls."""
    calls = []

    def run(_root):
        calls.append(1)
        return report

    run.calls = calls
    return run


def _versions(**installed):
    """Return a ``version_of`` that knows ``installed`` (name to version)."""

    def version_of(name):
        try:
            return installed[name]
        except KeyError:
            raise LookupError(name) from None

    return version_of


LOCKED = _versions(**{"type-pkg": "4.0", "other-pkg": "1.2"})


def _write(root, baseline):
    """Write ``ci/pyright_baseline.json`` under ``root``."""
    (root / "ci").mkdir(exist_ok=True)
    (root / "ci" / "pyright_baseline.json").write_text(
        json.dumps(baseline, indent=2) + "\n"
    )


def _lock(root, text=None):
    """Write ``ci/type_requirements.txt`` under ``root``."""
    (root / "ci").mkdir(exist_ok=True)
    if text is None:
        text = (
            "# header\n"
            "type-pkg==4.0 \\\n    --hash=sha256:" + "0" * 64 + "\n"
            "other-pkg==1.2 \\\n    --hash=sha256:" + "0" * 64 + "\n"
            "win-pkg==9.9 ; sys_platform == 'win32' \\\n"
            "    --hash=sha256:" + "0" * 64 + "\n"
        )
    (root / "ci" / "type_requirements.txt").write_text(text)


def _check(root, capsys, *found, update=False, allow_rise=False, **extra):
    """Run ``check``; return ``(status, output)``."""
    _lock(root)
    status = ratchet.check(
        root,
        extra.get("run") or _fake(_report(root, *found)),
        version_of=extra.get("version_of", LOCKED),
        update=update,
        allow_rise=allow_rise,
    )
    return status, capsys.readouterr().out


def _baseline_text(root):
    return (root / "ci" / "pyright_baseline.json").read_text()


def test_count_groups_by_relative_file_and_rule(tmp_path):
    """Diagnostics are counted per (file, rule), paths relative to root."""
    report = _report(
        tmp_path,
        ("a/x.py", "reportArgumentType"),
        ("a/x.py", "reportArgumentType"),
        ("a/x.py", "reportCallIssue"),
        ("y.py", "reportArgumentType"),
    )
    assert ratchet.count(report, tmp_path) == {
        "a/x.py": {"reportArgumentType": 2, "reportCallIssue": 1},
        "y.py": {"reportArgumentType": 1},
    }


def test_count_takes_errors_and_warnings_not_information(tmp_path):
    """``information`` is ignored; ``warning`` counts with ``error``."""
    report = _report(tmp_path)
    report["generalDiagnostics"] = [
        _diagnostic(tmp_path, "x.py", "reportA", "error"),
        _diagnostic(tmp_path, "x.py", "reportA", "warning"),
        _diagnostic(tmp_path, "x.py", "reportA", "information"),
    ]
    assert ratchet.count(report, tmp_path) == {"x.py": {"reportA": 2}}


def test_count_names_a_diagnostic_without_a_rule(tmp_path):
    """A syntax error has no rule; it is counted as ``no-rule``."""
    report = _report(tmp_path, ("x.py", None))
    assert ratchet.count(report, tmp_path) == {"x.py": {"no-rule": 1}}


def test_count_refuses_a_file_outside_the_root(tmp_path):
    """A diagnostic for a file outside root is a failure, not a count."""
    root = tmp_path / "root"
    root.mkdir()
    report = _report(root, ("../elsewhere.py", "reportA"))
    with pytest.raises(ratchet.PyrightFailure):
        ratchet.count(report, root)


def test_equal_counts_pass(tmp_path, capsys):
    """The same counts as the baseline exit 0."""
    _write(tmp_path, {"x.py": {"reportA": 1}})
    status, out = _check(tmp_path, capsys, ("x.py", "reportA"))
    assert status == 0, out


def test_no_diagnostics_and_empty_baseline_pass(tmp_path, capsys):
    """A clean tree against an empty baseline passes."""
    _write(tmp_path, {})
    status, out = _check(tmp_path, capsys)
    assert status == 0, out


def test_a_count_that_rose_fails_and_is_named(tmp_path, capsys):
    """A rule counted more often in a file fails: file, rule, old, new."""
    _write(tmp_path, {"x.py": {"reportA": 1}})
    status, out = _check(
        tmp_path, capsys, ("x.py", "reportA"), ("x.py", "reportA")
    )
    assert status == 1
    assert "x.py" in out and "reportA" in out
    assert "1 to 2" in out


def test_a_new_pair_fails_and_is_named(tmp_path, capsys):
    """A rule new in a file, and a new file, fail, each named."""
    _write(tmp_path, {"x.py": {"reportA": 1}})
    status, out = _check(
        tmp_path,
        capsys,
        ("x.py", "reportA"),
        ("x.py", "reportB"),
        ("z.py", "reportA"),
    )
    assert status == 1
    assert re.search(r"x\.py.*reportB", out)
    assert re.search(r"z\.py.*reportA", out)
    assert "x.py: reportA" not in out


def test_a_count_that_fell_fails_with_the_remedy(tmp_path, capsys):
    """A lower count with the baseline not lowered fails, with the command."""
    _write(tmp_path, {"x.py": {"reportA": 2}, "y.py": {"reportB": 1}})
    status, out = _check(tmp_path, capsys, ("x.py", "reportA"))
    assert status == 1
    assert "2 to 1" in out
    assert re.search(r"y\.py.*reportB.*1 to 0", out)
    assert UPDATE in out


def test_rise_and_fall_are_both_reported(tmp_path, capsys):
    """One run names the rises and the falls together."""
    _write(tmp_path, {"x.py": {"reportA": 2}})
    status, out = _check(
        tmp_path, capsys, ("x.py", "reportA"), ("y.py", "reportB")
    )
    assert status == 1
    assert "y.py" in out and "x.py" in out


def test_update_writes_the_counts_sorted(tmp_path, capsys):
    """--update rewrites the baseline from the counts, keys sorted."""
    _write(tmp_path, {"x.py": {"reportA": 3}, "b.py": {"reportB": 2}})
    status, out = _check(
        tmp_path,
        capsys,
        ("x.py", "reportA"),
        ("b.py", "reportB"),
        update=True,
    )
    assert status == 0, out
    text = _baseline_text(tmp_path)
    assert json.loads(text) == {
        "b.py": {"reportB": 1},
        "x.py": {"reportA": 1},
    }
    assert list(json.loads(text)) == ["b.py", "x.py"]
    assert text.endswith("}\n")
    status, out = _check(
        tmp_path, capsys, ("x.py", "reportA"), ("b.py", "reportB")
    )
    assert status == 0, out


def test_update_refuses_to_raise_a_count(tmp_path, capsys):
    """--update alone never records a rise; the file is untouched."""
    _write(tmp_path, {"x.py": {"reportA": 1}})
    before = _baseline_text(tmp_path)
    status, out = _check(
        tmp_path,
        capsys,
        ("x.py", "reportA"),
        ("x.py", "reportA"),
        update=True,
    )
    assert status == 1
    assert "x.py" in out
    assert _baseline_text(tmp_path) == before


def test_update_with_allow_rise_writes_the_rise(tmp_path, capsys):
    """--update --allow-rise records rises, new pairs and falls alike."""
    _write(tmp_path, {"x.py": {"reportA": 1}, "y.py": {"reportB": 4}})
    status, out = _check(
        tmp_path,
        capsys,
        ("x.py", "reportA"),
        ("x.py", "reportA"),
        ("z.py", "reportC"),
        update=True,
        allow_rise=True,
    )
    assert status == 0, out
    assert json.loads(_baseline_text(tmp_path)) == {
        "x.py": {"reportA": 2},
        "z.py": {"reportC": 1},
    }


def test_update_with_allow_rise_writes_a_first_baseline(tmp_path, capsys):
    """With no baseline file yet, --update --allow-rise creates it."""
    status, out = _check(
        tmp_path, capsys, ("x.py", "reportA"), update=True, allow_rise=True
    )
    assert status == 0, out
    assert json.loads(_baseline_text(tmp_path)) == {"x.py": {"reportA": 1}}


def test_allow_rise_without_update_is_refused():
    """--allow-rise alone is a usage error, status 2, before any run."""
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--allow-rise"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--update" in result.stderr


def test_main_refuses_allow_rise_without_update(capsys):
    """``main`` exits 2 on --allow-rise without --update."""
    with pytest.raises(SystemExit) as raised:
        ratchet.main(["--allow-rise"])
    assert raised.value.code == 2
    assert "--update" in capsys.readouterr().err


def test_missing_baseline_is_an_error(tmp_path, capsys):
    """Without a baseline file the status is 2, not a verdict."""
    status, out = _check(tmp_path, capsys)
    assert status == 2
    assert "pyright_baseline.json" in out


@pytest.mark.parametrize(
    "text", ["not json", "[1]", '{"x.py": 1}', '{"x.py": {"reportA": "1"}}']
)
def test_unreadable_baseline_is_an_error(tmp_path, capsys, text):
    """A baseline that is not ``{file: {rule: int}}`` is status 2."""
    (tmp_path / "ci").mkdir()
    (tmp_path / "ci" / "pyright_baseline.json").write_text(text)
    status, out = _check(tmp_path, capsys)
    assert status == 2
    assert "pyright_baseline.json" in out


@pytest.mark.parametrize("update", [False, True])
def test_pyright_failure_is_an_error(tmp_path, capsys, update):
    """A pyright that cannot run or answers badly is status 2."""
    _write(tmp_path, {})

    def broken(_root):
        raise ratchet.PyrightFailure("no pyright here")

    status, out = _check(tmp_path, capsys, run=broken, update=update)
    assert status == 2
    assert "no pyright here" in out


def test_parse_output_rejects_junk():
    """Output that is not pyright's JSON object is a ``PyrightFailure``."""
    for text in ("not json", "[]", "{}", '{"summary": {}}'):
        with pytest.raises(ratchet.PyrightFailure):
            ratchet.parse_output(text)


def test_parse_output_rejects_a_run_that_analysed_no_file():
    """No file analysed would read as every error fixed: it is a failure."""
    text = json.dumps(
        {"generalDiagnostics": [], "summary": {"filesAnalyzed": 0}}
    )
    with pytest.raises(ratchet.PyrightFailure, match="no file"):
        ratchet.parse_output(text)


def test_parse_output_returns_the_object():
    """A well-formed answer comes back as the parsed object."""
    report = {"generalDiagnostics": [], "summary": {"filesAnalyzed": 2}}
    assert ratchet.parse_output(json.dumps(report)) == report


def test_zero_files_analysed_in_the_fake_is_not_a_clean_tree(tmp_path, capsys):
    """``check`` turns a report with no analysed file into status 2."""
    _write(tmp_path, {"x.py": {"reportA": 1}})
    report = _report(tmp_path, analysed=0)

    def run(_root):
        return ratchet.parse_output(json.dumps(report))

    status, _ = _check(tmp_path, capsys, run=run, update=True, allow_rise=True)
    assert status == 2
    assert json.loads(_baseline_text(tmp_path)) == {"x.py": {"reportA": 1}}


def test_read_lock_returns_pins_and_skips_marker_lines(tmp_path):
    """``name==version`` lines are read; a line with a marker is skipped."""
    _lock(tmp_path)
    assert ratchet.read_lock(tmp_path) == [
        ("type-pkg", "4.0"),
        ("other-pkg", "1.2"),
    ]


def test_read_lock_missing_file_is_a_failure(tmp_path):
    """A missing lock is an error, not an empty list."""
    with pytest.raises(ratchet.EnvironmentMismatch, match="type_requirements"):
        ratchet.read_lock(tmp_path)


@pytest.mark.parametrize(
    ("installed", "named"),
    [
        ({"type-pkg": "4.0"}, "other-pkg"),
        ({"type-pkg": "4.1", "other-pkg": "1.2"}, "type-pkg"),
    ],
    ids=["not-installed", "other-version"],
)
def test_a_pin_not_matched_is_status_2_before_pyright_runs(
    tmp_path, capsys, installed, named
):
    """A missing or different pin names itself and the install command."""
    _write(tmp_path, {})
    run = _fake(_report(tmp_path))
    status, out = _check(
        tmp_path, capsys, run=run, version_of=_versions(**installed)
    )
    assert status == 2
    assert named in out
    assert "pip install --require-hashes --only-binary :all:" in out
    assert run.calls == []


def test_a_marker_line_pin_is_not_looked_up(tmp_path, capsys):
    """``win-pkg`` is in the lock under a marker and is not installed."""
    _write(tmp_path, {})
    status, out = _check(tmp_path, capsys)
    assert status == 0, out


def test_child_environment_drops_every_pyright_python_variable():
    """PYRIGHT_PYTHON_* would send pyright to npm or another version."""
    environ = {
        "PATH": "/bin",
        "PYRIGHT_PYTHON_FORCE_VERSION": "latest",
        "PYRIGHT_PYTHON_USE_BUNDLED_PYRIGHT": "0",
        "PYRIGHT_PYTHON_X": "1",
        "PYRIGHT_OTHER": "kept",
    }
    assert ratchet.child_environment(environ) == {
        "PATH": "/bin",
        "PYRIGHT_OTHER": "kept",
    }
    assert "PYRIGHT_PYTHON_FORCE_VERSION" in environ


@pytest.mark.parametrize("returncode", [2, 3])
def test_run_pyright_rejects_an_exit_other_than_0_or_1(
    tmp_path, monkeypatch, returncode
):
    """pyright's exit 2 or 3 is a failure to run, not a count."""
    seen = {}

    def fake_run(command, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, returncode, "", "boom")

    monkeypatch.setenv("PYRIGHT_PYTHON_FORCE_VERSION", "latest")
    monkeypatch.setattr(ratchet.subprocess, "run", fake_run)
    with pytest.raises(ratchet.PyrightFailure, match=f"exited {returncode}"):
        ratchet.run_pyright(tmp_path)
    assert "PYRIGHT_PYTHON_FORCE_VERSION" not in seen["env"]


def test_the_committed_baseline_is_well_formed():
    """The repository's baseline reads as ``{file: {rule: positive int}}``."""
    baseline = ratchet.read_baseline(ratchet.ROOT)
    assert baseline
    for name, rules in baseline.items():
        assert not name.startswith("/") and ".." not in name.split("/")
        assert rules
        for rule, number in rules.items():
            assert re.fullmatch(r"report[A-Z][A-Za-z]+|no-rule", rule)
            assert number > 0
