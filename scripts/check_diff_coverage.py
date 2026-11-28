"""
Check that the lines a pull request adds are run by the tests.

usage: python scripts/check_diff_coverage.py BASE HEAD
           [--python REPORT --map DIR=src/hmmcraft] [--kernel REPORT]

The lines added or changed between BASE and HEAD (``git diff -U0``, renames
followed) are looked up in the reports: ``--python`` is ``coverage json``,
``--kernel`` is ``gcovr --json``; at least one is needed. Each report is
gated on its own: at least 90% of the added lines it measures must run and at
least 85% of their branches must be taken. A line counts only when its report
measures it, so blank lines, comments, docstrings, excluded lines, deleted
lines and files outside the reports count nothing; a report with no measured
added line passes. Each added line a report marks excluded is printed.

``--map`` rewrites the report paths that start with DIR to the repository's
(``coverage json`` writes the installed package's absolute paths). It is an
error when no path of the Python report starts with DIR, so that a wrong map
cannot leave every file unmatched.

The kernel report has one entry per line and function, so a line of a
template counts once per instance, as in gcovr's own summary.

Exit status: 0 when every report passes, 1 when one fails (each line not run
is printed), 2 on a usage, git or report error. Standard library only.
"""

import argparse
import json
import re
import subprocess
import sys

LINE_LIMIT = 90
BRANCH_LIMIT = 85
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


class ReportError(Exception):
    """A report that cannot be read or has not the expected shape."""


def _git(*args):
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    ).stdout


def added_lines(diff_text):
    """Return ``{path: set of line numbers}`` added by a ``-U0`` diff.

    With no context every new-side line of a hunk is added or changed; a
    hunk of length 0 is a pure deletion, and a deleted file has no new side.
    """
    found = {}
    path = None
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            # git ends the line with a tab when the path holds a space
            name = line[4:].removesuffix("\t")
            if name.startswith('"'):
                raise ValueError(f"quoted path not supported: {name}")
            path = None if name == "/dev/null" else name.removeprefix("b/")
            continue
        match = HUNK.match(line)
        if match and path is not None:
            start = int(match.group(1))
            count = int(match.group(2) or "1")
            if count:
                found.setdefault(path, set()).update(
                    range(start, start + count)
                )
    return found


def python_report(data, mapping=None):
    """Return ``{path: (lines, branches, excluded)}`` from ``coverage json``
    data.

    ``lines`` maps a line number to a list with one flag, run or not;
    ``branches`` maps a line to the flags of the arcs leaving it;
    ``excluded`` is the set of lines the report marks excluded. A path
    starting with ``mapping[0] + "/"`` is rewritten to start with
    ``mapping[1]``.
    """
    report = {}
    for name, entry in data["files"].items():
        if mapping and name.startswith(mapping[0] + "/"):
            name = mapping[1] + name[len(mapping[0]) :]
        lines = {}
        for number in entry["executed_lines"]:
            lines[number] = [True]
        for number in entry["missing_lines"]:
            lines[number] = [False]
        branches = {}
        for key, taken in (
            ("executed_branches", True),
            ("missing_branches", False),
        ):
            for origin, _ in entry.get(key, []):
                branches.setdefault(origin, []).append(taken)
        excluded = set(entry.get("excluded_lines", []))
        report[name] = (lines, branches, excluded)
    return report


def kernel_report(data):
    """Return ``{path: (lines, branches, excluded)}`` from ``gcovr --json``
    data, one flag per non-excluded line entry and per non-excluded branch,
    and the set of excluded lines."""
    report = {}
    for entry in data["files"]:
        lines = {}
        branches = {}
        excluded = set()
        for line in entry["lines"]:
            if line.get("gcovr/excluded"):
                excluded.add(line["line_number"])
                continue
            number = line["line_number"]
            lines.setdefault(number, []).append(line["count"] > 0)
            for branch in line["branches"]:
                if not branch.get("gcovr/excluded"):
                    branches.setdefault(number, []).append(branch["count"] > 0)
        report[entry["file"]] = (lines, branches, excluded)
    return report


def measure(report, added):
    """Return ``(lines run, lines, branches taken, branches, not run,
    partial, excluded)`` for the added lines found in ``report``.

    ``not run`` and ``excluded`` list ``(path, line)``; ``partial`` lists
    ``(path, line, taken, total)`` for a line with an arc not taken.
    """
    run = total = taken = arcs = 0
    not_run = []
    partial = []
    excluded = []
    for path in sorted(added):
        if path not in report:
            continue
        lines, branches, skipped = report[path]
        for number in sorted(added[path]):
            if number in skipped:
                excluded.append((path, number))
            flags = lines.get(number, [])
            total += len(flags)
            run += sum(flags)
            if not all(flags):
                not_run.append((path, number))
            arc_flags = branches.get(number, [])
            arcs += len(arc_flags)
            taken += sum(arc_flags)
            if not all(arc_flags):
                partial.append((path, number, sum(arc_flags), len(arc_flags)))
    return run, total, taken, arcs, not_run, partial, excluded


def _percent(part, whole):
    return f"{100 * part / whole:.2f}%, " if whole else ""


def gate(label, report, added):
    """Print the figures of one report and return whether it passes; an
    excluded added line is printed, and counts nothing."""
    run, total, taken, arcs, not_run, partial, excluded = measure(
        report, added
    )
    passed = run * 100 >= LINE_LIMIT * total and taken * 100 >= (
        BRANCH_LIMIT * arcs
    )
    print(
        f"{label}: {run} of {total} added lines run "
        f"({_percent(run, total)}limit {LINE_LIMIT}%), "
        f"{taken} of {arcs} branches taken "
        f"({_percent(taken, arcs)}limit {BRANCH_LIMIT}%): "
        + ("pass" if passed else "fail")
    )
    for path, number in not_run:
        print(f"  {path}:{number} not run")
    for path, number, got, of in partial:
        print(f"  {path}:{number} {got} of {of} branches taken")
    for path, number in excluded:
        print(f"  {path}:{number} excluded")
    return passed


def _load(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as error:
        raise ReportError(f"cannot read report {path}: {error}") from error


def _gate_reports(args, added):
    """Gate each report given; return whether all pass."""
    passed = True
    try:
        if args.python:
            data = _load(args.python)
            mapping = tuple(args.map.split("=", 1)) if args.map else None
            if mapping and not any(
                name.startswith(mapping[0] + "/") for name in data["files"]
            ):
                raise ReportError(
                    f"no path of {args.python} starts with {mapping[0]}/"
                )
            passed = gate("python", python_report(data, mapping), added)
        if args.kernel:
            report = kernel_report(_load(args.kernel))
            passed = gate("kernel", report, added) and passed
    except (KeyError, TypeError, AttributeError) as error:
        raise ReportError(f"unexpected report layout: {error!r}") from error
    return passed


def main(argv):
    parser = argparse.ArgumentParser(
        usage=__doc__.strip().split("\n\n")[1].replace("usage: ", "")
    )
    parser.add_argument("base")
    parser.add_argument("head")
    parser.add_argument("--python")
    parser.add_argument("--map")
    parser.add_argument("--kernel")
    args = parser.parse_args(argv)
    if not (args.python or args.kernel):
        parser.error("give at least one of --python and --kernel")
    if args.map and ("=" not in args.map or not args.python):
        parser.error("--map is DIR=PATH and goes with --python")
    try:
        diff = _git(
            "diff",
            "--no-color",
            "--no-ext-diff",
            "--no-textconv",
            "--find-renames",
            "--src-prefix=a/",
            "--dst-prefix=b/",
            "-U0",
            args.base,
            args.head,
        )
        added = added_lines(diff)
    except subprocess.CalledProcessError as error:
        print(error.stderr.strip(), file=sys.stderr)
        return 2
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    try:
        passed = _gate_reports(args, added)
    except ReportError as error:
        print(error, file=sys.stderr)
        return 2
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
