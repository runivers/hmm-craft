"""
Tests of ``scripts/check_diff_coverage.py``, the coverage of added lines.

The pure parts get literal text. The end-to-end cases build a small
repository in a temporary directory, with git's global and system
configuration and hooks switched off, and write the two coverage reports by
hand: line numbers are synthetic, nothing is measured.

Run with ``python -m pytest scripts/test_check_diff_coverage.py``.
"""

import json
import os
import pathlib
import subprocess
import sys

import check_diff_coverage as gate
import pytest

SCRIPT = pathlib.Path(gate.__file__).resolve()
PY = "src/hmmcraft/a.py"
CPP = "ext/k.cpp"
ME = ("Ann Example", "ann@example.org")


def _env():
    """Return an environment that reaches no configuration or repository
    outside the test's own."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_AUTHOR_NAME=ME[0],
        GIT_AUTHOR_EMAIL=ME[1],
        GIT_COMMITTER_NAME=ME[0],
        GIT_COMMITTER_EMAIL=ME[1],
    )
    return env


def _git(repo, *args):
    result = subprocess.run(
        ["git", "-C", str(repo), "-c", "core.hooksPath=" + os.devnull, *args],
        env=_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(repo, files):
    """Write ``files`` (path -> text), commit them, return the commit."""
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _git(repo, "add", "--", *files)
    _git(repo, "commit", "-q", "-m", "c")
    return _git(repo, "rev-parse", "HEAD")


def _lines(n, first=1):
    """Return ``n`` one-line statements numbered from ``first``."""
    return "".join(f"x{i} = {i}\n" for i in range(first, first + n))


def _py_report(run, missed, taken=(), not_taken=(), path=PY):
    """Return a ``coverage json`` object for one file."""
    return {
        "files": {
            path: {
                "executed_lines": list(run),
                "missing_lines": list(missed),
                "executed_branches": [[n, n + 1] for n in taken],
                "missing_branches": [[n, n + 1] for n in not_taken],
            }
        }
    }


def _kernel_report(entries, path=CPP):
    """Return a ``gcovr --json`` object; ``entries`` is a list of
    ``(line, count, [branch counts], excluded)``."""
    lines = []
    for number, count, branches, excluded in entries:
        entry = {
            "line_number": number,
            "count": count,
            "branches": [{"count": c} for c in branches],
        }
        if excluded:
            entry["gcovr/excluded"] = True
        lines.append(entry)
    return {"files": [{"file": path, "lines": lines}]}


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    return tmp_path


def _run(repo, base, head, *args, reports=()):
    """Run the script in ``repo``; ``reports`` are (flag, object) pairs."""
    argv = [sys.executable, "-I", str(SCRIPT), base, head]
    for flag, data in reports:
        path = repo / (flag.lstrip("-") + ".json")
        path.write_text(json.dumps(data))
        argv += [flag, str(path)]
    return subprocess.run(
        [*argv, *args],
        cwd=repo,
        env=_env(),
        check=False,
        capture_output=True,
        text=True,
    )


def _pair(repo, added, name=PY, base_text="a = 0\n"):
    """Commit ``name`` with one line, then with ``added`` lines after it;
    return the two commits."""
    base = _commit(repo, {name: base_text})
    head = _commit(repo, {name: base_text + added})
    return base, head


@pytest.mark.parametrize(
    ("diff", "expected"),
    [
        pytest.param(
            "+++ b/p.py\n@@ -3,0 +4,3 @@\n", {"p.py": {4, 5, 6}}, id="range"
        ),
        pytest.param(
            "+++ b/p.py\n@@ -3 +7 @@ x\n", {"p.py": {7}}, id="one-line"
        ),
        pytest.param("+++ b/p.py\n@@ -3,2 +2,0 @@\n", {}, id="deletion-only"),
        pytest.param(
            "--- /dev/null\n+++ b/n.py\n@@ -0,0 +1,2 @@\n",
            {"n.py": {1, 2}},
            id="new-file",
        ),
        pytest.param(
            "--- a/o.py\n+++ /dev/null\n@@ -1,4 +0,0 @@\n",
            {},
            id="deleted-file",
        ),
        pytest.param(
            "+++ b/p.py\n@@ -1 +1,2 @@\n+++ b/q.py\n@@ -5,0 +6 @@\n",
            {"p.py": {1, 2}, "q.py": {6}},
            id="two-files",
        ),
        pytest.param(
            "+++ b/a b.py\t\n@@ -0,0 +1 @@\n",
            {"a b.py": {1}},
            id="space-in-path",
        ),
        pytest.param("", {}, id="empty"),
    ],
)
def test_added_lines_reads_the_new_side_of_each_hunk(diff, expected):
    """Every new-side line of a ``-U0`` hunk is added; none for a deletion."""
    assert gate.added_lines(diff) == expected


def test_added_lines_refuses_a_quoted_path():
    """A quoted path is an error, not a guess."""
    with pytest.raises(ValueError, match="quoted"):
        gate.added_lines('+++ "b/we ird.py"\n@@ -0,0 +1 @@\n')


def test_covered_added_lines_pass_and_the_figures_are_printed(repo):
    base, head = _pair(repo, _lines(10, 2))
    report = _py_report(range(1, 12), [], taken=range(2, 12))
    result = _run(repo, base, head, reports=[("--python", report)])
    assert result.returncode == 0, result.stdout
    assert "10 of 10 added lines run" in result.stdout
    assert "10 of 10 branches taken" in result.stdout
    assert "pass" in result.stdout


def test_lines_below_ninety_percent_fail_and_are_named(repo):
    base, head = _pair(repo, _lines(10, 2))
    report = _py_report(range(1, 10), [10, 11])
    result = _run(repo, base, head, reports=[("--python", report)])
    assert result.returncode == 1
    assert "8 of 10 added lines run" in result.stdout
    assert f"{PY}:10" in result.stdout
    assert f"{PY}:11" in result.stdout
    assert f"{PY}:9" not in result.stdout


@pytest.mark.parametrize(
    ("missed", "not_taken", "code"),
    [
        pytest.param([11], (), 0, id="lines-9-of-10"),
        pytest.param([10, 11], (), 1, id="lines-8-of-10"),
        pytest.param([], (2, 3, 4), 0, id="branches-17-of-20"),
        pytest.param([], (2, 3, 4, 5), 1, id="branches-16-of-20"),
    ],
)
def test_limits_are_compared_in_integers_at_the_boundary(
    repo, missed, not_taken, code
):
    """90% of ten lines and 85% of twenty branches pass; one less fails."""
    base, head = _pair(repo, _lines(10, 2))
    run = [n for n in range(1, 12) if n not in missed]
    taken = [n for n in range(2, 12) if n not in not_taken]
    # two branches per added line: the first always taken, the second where
    # the line is not in ``not_taken``
    report = _py_report(run, missed)
    report["files"][PY]["executed_branches"] = [
        [n, 0] for n in range(2, 12)
    ] + [[n, 1] for n in taken]
    report["files"][PY]["missing_branches"] = [[n, 1] for n in not_taken]
    result = _run(repo, base, head, reports=[("--python", report)])
    assert result.returncode == code, result.stdout


def test_blank_comment_and_deleted_lines_are_not_counted(repo):
    """Lines the report does not measure, and deleted ones, count nothing."""
    base = _commit(repo, {PY: "a = 0\nb = 1\nc = 2\n"})
    head = _commit(repo, {PY: "a = 0\n\n\n\n# note\nc = 2\n"})
    report = _py_report([1, 6], [])
    result = _run(repo, base, head, reports=[("--python", report)])
    assert result.returncode == 0, result.stdout
    assert "0 of 0 added lines run" in result.stdout


def test_an_empty_diff_passes(repo):
    base, _ = _pair(repo, _lines(2, 2))
    result = _run(
        repo, base, base, reports=[("--python", _py_report([1], [2, 3]))]
    )
    assert result.returncode == 0, result.stdout
    assert "0 of 0 added lines run" in result.stdout


def test_files_absent_from_the_reports_count_nothing(repo):
    base, head = _pair(repo, _lines(5, 2), name="docs/notes.py")
    result = _run(
        repo, base, head, reports=[("--python", _py_report([1], [2, 3]))]
    )
    assert result.returncode == 0, result.stdout
    assert "0 of 0 added lines run" in result.stdout


def test_python_paths_are_mapped_to_the_repository(repo):
    """A report keyed by an installed path matches only with ``--map``."""
    base, head = _pair(repo, _lines(4, 2))
    installed = "/x/site-packages/hmmcraft/a.py"
    report = _py_report([1], [2, 3, 4, 5], path=installed)
    mapped = _run(
        repo,
        base,
        head,
        "--map",
        "/x/site-packages/hmmcraft=src/hmmcraft",
        reports=[("--python", report)],
    )
    assert mapped.returncode == 1
    assert "0 of 4 added lines run" in mapped.stdout
    unmapped = _run(repo, base, head, reports=[("--python", report)])
    assert unmapped.returncode == 0
    assert "0 of 0 added lines run" in unmapped.stdout


def test_a_map_that_matches_no_report_path_is_an_error(repo):
    """A wrong map would leave every file unmatched and pass silently."""
    base, head = _pair(repo, _lines(4, 2))
    report = _py_report([1], [2, 3, 4, 5], path="/x/site-packages/h/a.py")
    result = _run(
        repo,
        base,
        head,
        "--map",
        "/elsewhere=src/hmmcraft",
        reports=[("--python", report)],
    )
    assert result.returncode == 2
    assert "/elsewhere" in result.stderr


def test_kernel_template_instances_count_once_each(repo):
    """A line with two entries, one run and one not, is 1 of 2."""
    base, head = _pair(repo, _lines(1, 2), name=CPP)
    report = _kernel_report([(2, 5, [], False), (2, 0, [], False)])
    result = _run(repo, base, head, reports=[("--kernel", report)])
    assert result.returncode == 1
    assert "1 of 2 added lines run" in result.stdout
    assert f"{CPP}:2" in result.stdout


def test_kernel_excluded_lines_and_branches_are_not_counted(repo):
    base, head = _pair(repo, _lines(2, 2), name=CPP)
    entries = [
        (2, 3, [4, 0], False),
        (3, 0, [0, 0], True),
    ]
    report = _kernel_report(entries)
    report["files"][0]["lines"][0]["branches"].append(
        {"count": 0, "gcovr/excluded": True}
    )
    result = _run(repo, base, head, reports=[("--kernel", report)])
    assert "1 of 1 added lines run" in result.stdout
    assert "1 of 2 branches taken" in result.stdout


def test_excluded_added_lines_are_printed_and_do_not_fail(repo):
    """An added line a report marks excluded is named, not counted."""
    base = _commit(repo, {PY: "a = 0\n", CPP: "a = 0\n"})
    head = _commit(
        repo, {PY: "a = 0\nb = 1\nc = 2\n", CPP: "a = 0\nb = 1\nc = 2\n"}
    )
    python = _py_report([1, 2], [])
    python["files"][PY]["excluded_lines"] = [3]
    kernel = _kernel_report([(2, 1, [], False), (3, 0, [], True)])
    reports = [("--python", python), ("--kernel", kernel)]
    result = _run(repo, base, head, reports=reports)
    assert result.returncode == 0, result.stdout
    assert f"{PY}:3 excluded" in result.stdout
    assert f"{CPP}:3 excluded" in result.stdout
    assert f"{PY}:2 excluded" not in result.stdout
    assert "1 of 1 added lines run" in result.stdout


def test_python_branches_belong_to_their_from_line(repo):
    """A missing arc from an added line counts; one from an old line not."""
    base, head = _pair(repo, _lines(1, 2))
    # 1 of 2 branches taken at added line 2: fails at 50%; the old line 1
    # has no taken arc and is not counted.
    report = _py_report([1, 2], [], taken=[2], not_taken=[1, 2])
    result = _run(repo, base, head, reports=[("--python", report)])
    assert result.returncode == 1
    assert "1 of 2 branches taken" in result.stdout
    assert f"{PY}:2" in result.stdout


def test_a_rename_without_a_change_counts_nothing(repo):
    base = _commit(repo, {PY: _lines(6)})
    _git(repo, "mv", PY, "src/hmmcraft/b.py")
    _git(repo, "commit", "-q", "-m", "mv")
    head = _git(repo, "rev-parse", "HEAD")
    report = _py_report([], range(1, 7), path="src/hmmcraft/b.py")
    result = _run(repo, base, head, reports=[("--python", report)])
    assert result.returncode == 0, result.stdout
    assert "0 of 0 added lines run" in result.stdout


def test_the_reports_are_gated_one_by_one(repo):
    """Python passes, the kernel fails: the run fails and names the kernel."""
    base = _commit(repo, {PY: "a = 0\n", CPP: "a = 0\n"})
    head = _commit(repo, {PY: "a = 0\nb = 1\n", CPP: "a = 0\nb = 1\n"})
    reports = [
        ("--python", _py_report([1, 2], [])),
        ("--kernel", _kernel_report([(2, 0, [], False)])),
    ]
    result = _run(repo, base, head, reports=reports)
    assert result.returncode == 1
    out = result.stdout
    assert "python:" in out
    assert "pass" in out.split("kernel:")[0]
    assert "fail" in out.split("kernel:")[1]


def test_usage_and_read_errors_exit_two(repo):
    base, head = _pair(repo, _lines(1, 2))
    report = _py_report([1, 2], [])
    unknown = _run(repo, "nosuchref", head, reports=[("--python", report)])
    assert unknown.returncode == 2
    no_report = _run(repo, base, head)
    assert no_report.returncode == 2
    (repo / "bad.json").write_text("{")
    bad = _run(repo, base, head, "--kernel", str(repo / "bad.json"))
    assert bad.returncode == 2
    missing = _run(repo, base, head, "--kernel", str(repo / "none.json"))
    assert missing.returncode == 2
