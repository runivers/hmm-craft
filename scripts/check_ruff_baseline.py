"""
Check that no ruff finding is added: a per-file, per-rule count ratchet.

usage: python3 -I scripts/check_ruff_baseline.py [--root DIR] [--update]

Runs ``ruff check`` over DIR (default: the repository) with the fixed rule
list ``RULES`` and compares the count of each (file, rule) with
``ci/ruff_baseline.json``. It fails when a count rose or a pair is new, and
also when a count fell and the baseline was not lowered, so the floor only
moves down. ``--update`` rewrites the baseline from the current findings; it
refuses when any count rose. ``ruff`` is taken from ``PATH``.

``RULES`` is the set ruff 0.17.5 selects by default, written out as codes so
that a ruff upgrade cannot change it. The line length and the classic set
that must stay at zero are in ``pyproject.toml``; this check ignores that
file (``--isolated``).

Exit status: 0 when the counts equal the baseline, 1 when they differ, 2
when ruff or the baseline cannot be read. Standard library only.
"""

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = "ci/ruff_baseline.json"
UPDATE = "python3 -I scripts/check_ruff_baseline.py --update"

# ruff 0.17.5's default rules, from `ruff check --show-settings`.
RULES = (
    "ASYNC100", "ASYNC105", "ASYNC115", "ASYNC116", "ASYNC210",
    "ASYNC220", "ASYNC221", "ASYNC222", "ASYNC230", "ASYNC251", "B002",
    "B003", "B004", "B005", "B006", "B008", "B009", "B010", "B012",
    "B013", "B014", "B015", "B016", "B017", "B018", "B019", "B020",
    "B021", "B022", "B023", "B025", "B026", "B029", "B030", "B031",
    "B032", "B033", "B035", "B039", "BLE001", "C400", "C401", "C402",
    "C403", "C404", "C405", "C406", "C408", "C409", "C410", "C411",
    "C413", "C414", "C415", "C417", "C418", "C419", "D419", "DTZ002",
    "DTZ003", "DTZ004", "E722", "E902", "EXE001", "EXE002", "EXE004",
    "EXE005", "F401", "F402", "F404", "F406", "F407", "F501", "F502",
    "F503", "F504", "F505", "F506", "F507", "F508", "F509", "F521",
    "F522", "F523", "F524", "F525", "F541", "F601", "F602", "F621",
    "F622", "F631", "F632", "F633", "F634", "F701", "F702", "F704",
    "F706", "F707", "F811", "F821", "F822", "F823", "F841", "F842",
    "F901", "FA100", "FA102", "FLY002", "FURB105", "FURB122", "FURB129",
    "FURB132", "FURB136", "FURB157", "FURB161", "FURB162", "FURB163",
    "FURB166", "FURB167", "FURB168", "FURB169", "FURB177", "FURB181",
    "FURB188", "FURB192", "G010", "G101", "G201", "G202", "I001",
    "INT001", "INT002", "INT003", "ISC004", "LOG001", "LOG002", "LOG009",
    "LOG014", "LOG015", "N999", "PERF101", "PERF102", "PERF402", "PGH005",
    "PIE790", "PIE794", "PIE796", "PIE800", "PIE804", "PIE807", "PIE808",
    "PIE810", "PLC0105", "PLC0131", "PLC0132", "PLC0205", "PLC0206",
    "PLC0208", "PLC0414", "PLC3002", "PLE0100", "PLE0101", "PLE0115",
    "PLE0116", "PLE0117", "PLE0118", "PLE0303", "PLE0304", "PLE0305",
    "PLE0307", "PLE0308", "PLE0309", "PLE0604", "PLE0605", "PLE0643",
    "PLE0704", "PLE1132", "PLE1142", "PLE1205", "PLE1206", "PLE1300",
    "PLE1307", "PLE1310", "PLE1507", "PLE1519", "PLE1520", "PLE1700",
    "PLE2502", "PLE2510", "PLE2512", "PLE2513", "PLE2514", "PLE2515",
    "PLR0124", "PLR0133", "PLR0206", "PLR0402", "PLR1704", "PLR1708",
    "PLR1711", "PLR1716", "PLR1722", "PLR1730", "PLR1733", "PLR1736",
    "PLR2044", "PLW0120", "PLW0127", "PLW0128", "PLW0129", "PLW0131",
    "PLW0133", "PLW0177", "PLW0211", "PLW0245", "PLW0406", "PLW0602",
    "PLW0604", "PLW0642", "PLW0711", "PLW1501", "PLW1507", "PLW1508",
    "PLW1509", "PLW1510", "PLW2101", "PT010", "PT014", "PT020", "PT025",
    "PT026", "PT031", "PTH124", "PTH210", "PYI001", "PYI002", "PYI003",
    "PYI004", "PYI005", "PYI006", "PYI007", "PYI008", "PYI009", "PYI010",
    "PYI012", "PYI013", "PYI015", "PYI016", "PYI017", "PYI018", "PYI019",
    "PYI020", "PYI025", "PYI026", "PYI029", "PYI030", "PYI032", "PYI033",
    "PYI034", "PYI035", "PYI036", "PYI041", "PYI042", "PYI043", "PYI044",
    "PYI045", "PYI046", "PYI047", "PYI048", "PYI049", "PYI050", "PYI052",
    "PYI055", "PYI057", "PYI058", "PYI059", "PYI061", "PYI062", "PYI063",
    "PYI064", "PYI066", "RET501", "RUF007", "RUF008", "RUF009", "RUF010",
    "RUF012", "RUF013", "RUF015", "RUF016", "RUF017", "RUF018", "RUF019",
    "RUF020", "RUF022", "RUF023", "RUF024", "RUF026", "RUF028", "RUF030",
    "RUF032", "RUF033", "RUF034", "RUF040", "RUF041", "RUF046", "RUF048",
    "RUF049", "RUF051", "RUF053", "RUF057", "RUF058", "RUF059", "RUF063",
    "RUF068", "RUF100", "RUF101", "RUF200", "S102", "S110", "S112",
    "SIM101", "SIM102", "SIM103", "SIM107", "SIM113", "SIM114", "SIM115",
    "SIM117", "SIM118", "SIM201", "SIM202", "SIM208", "SIM210", "SIM211",
    "SIM220", "SIM221", "SIM222", "SIM223", "SIM401", "SIM905", "SIM911",
    "T100", "TC004", "TC005", "TC007", "TC010", "TRY002", "TRY004",
    "TRY201", "TRY203", "TRY401", "UP001", "UP003", "UP004", "UP005",
    "UP006", "UP007", "UP008", "UP009", "UP010", "UP011", "UP012",
    "UP014", "UP017", "UP018", "UP019", "UP020", "UP021", "UP022",
    "UP023", "UP024", "UP025", "UP026", "UP028", "UP029", "UP030",
    "UP031", "UP032", "UP033", "UP034", "UP035", "UP036", "UP037",
    "UP039", "UP040", "UP041", "UP043", "UP044", "UP045", "UP046",
    "UP047", "UP049", "UP050", "W605", "YTT101", "YTT102", "YTT103",
    "YTT201", "YTT202", "YTT203", "YTT204", "YTT301", "YTT302", "YTT303",
)


class RuffFailure(Exception):
    """ruff could not run, or its answer could not be read."""


class BaselineError(Exception):
    """The baseline file is missing or is not ``{file: {rule: count}}``."""


def parse_output(text):
    """Return the findings of ruff's JSON output, a list of objects."""
    try:
        found = json.loads(text)
    except ValueError as error:
        raise RuffFailure(f"ruff's output is not JSON: {error}") from error
    if not isinstance(found, list) or not all(
        isinstance(item, dict) for item in found
    ):
        raise RuffFailure("ruff's output is not a list of findings")
    return found


def run_ruff(root):
    """Return the findings of ``ruff check`` under ``root``."""
    command = [
        "ruff",
        "check",
        "--isolated",
        "--no-cache",
        "--exit-zero",
        "--output-format=json",
        "--select=" + ",".join(RULES),
        ".",
    ]
    try:
        done = subprocess.run(
            command, cwd=root, capture_output=True, text=True, check=False
        )
    except OSError as error:
        raise RuffFailure(f"cannot run ruff: {error}") from error
    if done.returncode != 0:
        raise RuffFailure(f"ruff exited {done.returncode}: {done.stderr}")
    return parse_output(done.stdout)


def count(findings, root):
    """Return ``{file: {rule: n}}`` with paths relative to ``root``."""
    counts = {}
    root = pathlib.Path(root).resolve()
    for item in findings:
        path = pathlib.Path(item["filename"]).resolve()
        name = path.relative_to(root).as_posix()
        rule = item.get("code") or "syntax-error"
        per_file = counts.setdefault(name, {})
        per_file[rule] = per_file.get(rule, 0) + 1
    return counts


def read_baseline(root):
    """Return the baseline under ``root``, or raise ``BaselineError``."""
    path = pathlib.Path(root) / BASELINE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
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


def check(root, run=run_ruff, update=False):
    """Compare the findings under ``root`` with the baseline; return status."""
    try:
        baseline = read_baseline(root)
        current = count(run(root), root)
    except (BaselineError, RuffFailure) as error:
        print(f"error: {error}")
        return 2
    rises, falls = compare(baseline, current)
    for line in rises:
        print(f"rise: {line}")
    if rises:
        for line in falls:
            print(f"fall: {line}")
        print("A style finding was added: fix it, do not raise the baseline.")
        return 1
    if update:
        text = json.dumps(current, indent=2, sort_keys=True) + "\n"
        (pathlib.Path(root) / BASELINE).write_text(text, encoding="utf-8")
        print(f"{BASELINE} rewritten")
        return 0
    for line in falls:
        print(f"fall: {line}")
    if falls:
        print(f"The baseline is above the findings; lower it with: {UPDATE}")
        return 1
    print("ok: no style finding added")
    return 0


def main(argv=None):
    """Run the check; ``argv`` is the argument list."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args(argv)
    return check(args.root, update=args.update)


if __name__ == "__main__":
    sys.exit(main())
