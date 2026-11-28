"""
Check that no type error is added: a per-file, per-rule count ratchet.

usage: python -I scripts/check_pyright_baseline.py [--root DIR]
           [--update [--allow-rise]]

Runs pyright over DIR (default: the repository), with the settings in
``[tool.pyright]`` of ``pyproject.toml``, and compares the count of each
(file, rule) with ``ci/pyright_baseline.json``. It fails when a count rose or
a pair is new, and also when a count fell and the baseline was not lowered, so
the baseline stays as tight as the tree. ``--update`` rewrites the baseline
from the current counts; it refuses when any count rose, unless
``--allow-rise`` is given too, for the change that moves
``ci/type_requirements.txt`` (a newer pyright or numpy can report errors that
no change to the code removes). ``--allow-rise`` alone is refused.

pyright and the packages whose types it reads are the ones the Python that
runs this script has installed. Before pyright runs, each pin of
``ci/type_requirements.txt`` is compared with the installed version, so that a
different environment does not change the counts: install the lock with
``pip install --require-hashes --only-binary :all: -r
ci/type_requirements.txt``. The ``PYRIGHT_PYTHON_*`` variables are dropped
from pyright's environment, so that it uses the copy bundled in its wheel and
the node of ``nodejs-wheel-binaries``, and downloads nothing.

Exit status: 0 when the counts equal the baseline, 1 when they differ, 2 when
the environment is not the lock's, or pyright or the baseline cannot be read.
Standard library only.
"""

import argparse
import importlib.metadata
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = "ci/pyright_baseline.json"
LOCK = "ci/type_requirements.txt"
UPDATE = "python -I scripts/check_pyright_baseline.py --update"
INSTALL = f"pip install --require-hashes --only-binary :all: -r {LOCK}"
COUNTED = ("error", "warning")

_PIN = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)\s*\\?")


class PyrightFailure(Exception):
    """pyright could not run, or its answer could not be read."""


class BaselineError(Exception):
    """The baseline file is missing or is not ``{file: {rule: count}}``."""


class EnvironmentMismatch(Exception):
    """The running environment is not the one the lock pins."""


def installed_version(name):
    """Return the installed version of ``name``; ``LookupError`` if none."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        raise LookupError(name) from None


def child_environment(environ):
    """Return ``environ`` without any ``PYRIGHT_PYTHON_*`` variable."""
    return {
        key: value
        for key, value in environ.items()
        if not key.startswith("PYRIGHT_PYTHON_")
    }


def parse_output(text):
    """Return the object pyright prints with ``--outputjson``.

    Raises ``PyrightFailure`` for anything else, and for a run that analysed
    no file: that would read as every error fixed.
    """
    try:
        report = json.loads(text)
    except ValueError as error:
        raise PyrightFailure(
            f"pyright's output is not JSON: {error}"
        ) from error
    summary = report.get("summary") if isinstance(report, dict) else None
    found = report.get("generalDiagnostics") if summary is not None else None
    if not isinstance(summary, dict) or not isinstance(found, list):
        raise PyrightFailure("pyright's output has no summary or diagnostics")
    analysed = summary.get("filesAnalyzed")
    if not isinstance(analysed, int) or analysed <= 0:
        raise PyrightFailure(
            "pyright analysed no file; check [tool.pyright] in pyproject.toml"
        )
    return report


def run_pyright(root):
    """Return pyright's report for the project under ``root``."""
    command = [
        sys.executable,
        "-I",
        "-m",
        "pyright",
        "--outputjson",
        "--pythonpath",
        sys.executable,
        "--project",
        str(root),
    ]
    try:
        done = subprocess.run(
            command,
            cwd=root,
            env=child_environment(os.environ),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise PyrightFailure(f"cannot run pyright: {error}") from error
    # 0: no error, 1: errors found. Anything else is a failure to run.
    if done.returncode not in (0, 1):
        raise PyrightFailure(
            f"pyright exited {done.returncode}: {done.stderr or done.stdout}"
        )
    return parse_output(done.stdout)


def count(report, root):
    """Return ``{file: {rule: n}}`` with paths relative to ``root``."""
    counts = {}
    root = pathlib.Path(root).resolve()
    for item in report["generalDiagnostics"]:
        if item.get("severity") not in COUNTED:
            continue
        path = pathlib.Path(item["file"]).resolve()
        try:
            name = path.relative_to(root).as_posix()
        except ValueError:
            raise PyrightFailure(f"{path} is outside {root}") from None
        rule = item.get("rule") or "no-rule"
        per_file = counts.setdefault(name, {})
        per_file[rule] = per_file.get(rule, 0) + 1
    return counts


def read_baseline(root, missing_ok=False):
    """Return the baseline under ``root``, or raise ``BaselineError``.

    With ``missing_ok`` an absent file reads as empty, for a first baseline.
    """
    path = pathlib.Path(root) / BASELINE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        if missing_ok:
            return {}
        raise BaselineError(f"{BASELINE}: {error.strerror}") from error
    except OSError as error:
        raise BaselineError(f"{BASELINE}: {error.strerror}") from error
    except ValueError as error:
        raise BaselineError(f"{BASELINE}: not JSON: {error}") from error
    if not isinstance(data, dict) or not all(
        isinstance(rules, dict)
        and all(
            isinstance(n, int) and not isinstance(n, bool)
            for n in rules.values()
        )
        for rules in data.values()
    ):
        raise BaselineError(f"{BASELINE}: not {{file: {{rule: count}}}}")
    return data


def read_lock(root):
    """Return the ``(name, version)`` pins of the lock without a marker."""
    path = pathlib.Path(root) / LOCK
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise EnvironmentMismatch(f"{LOCK}: {error.strerror}") from error
    pins = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "--hash=")):
            continue
        if ";" in stripped:
            continue
        match = _PIN.fullmatch(stripped)
        if match is None:
            raise EnvironmentMismatch(f"{LOCK}: not a pin: {stripped!r}")
        pins.append((match.group(1), match.group(2)))
    if not pins:
        raise EnvironmentMismatch(f"{LOCK}: no pin found")
    return pins


def check_environment(root, version_of=installed_version):
    """Raise ``EnvironmentMismatch`` unless each pin is installed as pinned."""
    for name, wanted in read_lock(root):
        try:
            have = version_of(name)
        except LookupError:
            have = None
        if have != wanted:
            now = "not installed" if have is None else f"{have} installed"
            raise EnvironmentMismatch(
                f"{name}=={wanted} is pinned in {LOCK}, but {now}. "
                f"Install the lock in an environment of its own: {INSTALL}"
            )


def compare(baseline, current):
    """Return ``(rises, falls)``, each a sorted list of report lines."""
    rises = []
    falls = []
    for name in sorted(set(baseline) | set(current)):
        before = baseline.get(name, {})
        now = current.get(name, {})
        for rule in sorted(set(before) | set(now)):
            old = before.get(rule, 0)
            new = now.get(rule, 0)
            if new > old:
                kind = "new" if old == 0 else "rose"
                rises.append(f"{name}: {rule} {kind}, {old} to {new}")
            elif new < old:
                falls.append(f"{name}: {rule} fell, {old} to {new}")
    return rises, falls


def check(
    root,
    run=run_pyright,
    version_of=installed_version,
    update=False,
    allow_rise=False,
):
    """Compare the errors under ``root`` with the baseline; return status."""
    try:
        check_environment(root, version_of)
        baseline = read_baseline(root, missing_ok=update and allow_rise)
        current = count(run(root), root)
    except (BaselineError, PyrightFailure, EnvironmentMismatch) as error:
        print(f"error: {error}")
        return 2
    rises, falls = compare(baseline, current)
    for line in rises:
        print(f"rise: {line}")
    if rises and not (update and allow_rise):
        for line in falls:
            print(f"fall: {line}")
        print("A type error was added: fix it, do not raise the baseline.")
        return 1
    if update:
        text = json.dumps(current, indent=2, sort_keys=True) + "\n"
        (pathlib.Path(root) / BASELINE).write_text(text, encoding="utf-8")
        print(f"{BASELINE} rewritten")
        return 0
    for line in falls:
        print(f"fall: {line}")
    if falls:
        print(f"The baseline is above the errors; lower it with: {UPDATE}")
        return 1
    print("ok: no type error added")
    return 0


def main(argv=None):
    """Run the check; ``argv`` is the argument list."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--allow-rise", action="store_true")
    args = parser.parse_args(argv)
    if args.allow_rise and not args.update:
        parser.error("--allow-rise only goes with --update")
    return check(args.root, update=args.update, allow_rise=args.allow_rise)


if __name__ == "__main__":
    sys.exit(main())
