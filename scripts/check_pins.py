"""
Check that each hashed pin under ``ci/`` is the newest release on PyPI.

usage: python3 -I scripts/check_pins.py [--root DIR]

Reads ``ci/build_requirements.txt`` and ``ci/tool_requirements.txt`` under DIR
(default: the repository) and compares each pin with the newest final,
unyanked release that PyPI lists. A pin that is older is named with the
command that updates it. ``ci/floor_constraints.txt`` pins the declared
minimums on purpose and is never read.

Exit status: 0 when every pin is the newest, 1 when a pin is stale or a hold
is out of date, 2 when a file cannot be read or a lookup fails (an error
outranks a stale verdict: the run is incomplete). Standard library only.
"""

import argparse
import http.client
import json
import pathlib
import re
import sys
import typing
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
# The floor file, ci/floor_constraints.txt, is not in this list on purpose,
# and no argument names a file, so it cannot be reached.
FILES = ("ci/build_requirements.txt", "ci/tool_requirements.txt")
PYPI = "https://pypi.org/pypi/{}/json"
TIMEOUT = 30
USER_AGENT = "hmm-craft check_pins.py"


class Hold(typing.NamedTuple):
    """A cap that another pin imposes on a pin."""

    below: str
    by: str
    by_version: str
    reason: str


# A pin capped by another pin cannot follow PyPI past the cap (uv keeps it
# below), so it is compared with the newest release below the cap. Each hold
# names the capping pin at the version whose metadata was read, and is
# reported as out of date once that pin moves, until someone re-reads the cap.
HOLDS = {
    "vcs-versioning": Hold(
        below="3",
        by="setuptools-scm",
        by_version="10.3.4",
        reason="setuptools-scm 10.3.4 requires vcs-versioning<3",
    ),
}


class Pin(typing.NamedTuple):
    """One ``name==version`` line, with its environment marker if any."""

    name: str
    version: str
    marker: str | None


class LookupFailure(Exception):
    """PyPI could not be asked, or its answer could not be read."""


class PinFileError(Exception):
    """A requirements file has a line, or lacks pins, the check cannot read."""


_PIN = re.compile(
    r"([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?==([^\s;\\]+)\s*"
    r"(?:;\s*(.*?))?\s*\\?\s*"
)
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# PEP 440's normalised final and post releases, with the spellings of the
# post segment it allows (``post``, ``rev``, ``r``, and the implicit ``-N``).
_FINAL = re.compile(
    r"\s*v?(?:(?P<epoch>\d+)!)?(?P<release>\d+(?:\.\d+)*)"
    r"(?P<post>[-_.]?(?:post|rev|r)[-_.]?(?P<number>\d+)?"
    r"|-(?P<implicit>\d+))?\s*",
    re.IGNORECASE,
)


def normalise(name):
    """Return ``name`` in PEP 503 form."""
    return re.sub(r"[-_.]+", "-", name).lower()


def final_key(version):
    """Return a sort key for a final or post release, else ``None``.

    The key is ``(epoch, release, post)`` with trailing zeros of the release
    stripped, so ``1.0`` equals ``1.0.0``, and ``post`` is -1 when absent.
    Pre-releases, dev releases, local versions and legacy strings give
    ``None``. No ``packaging`` import: the job installs nothing, and the
    check would depend on a package it checks.
    """
    match = _FINAL.fullmatch(version)
    if match is None:
        return None
    release = [int(part) for part in match.group("release").split(".")]
    while release and release[-1] == 0:
        release.pop()
    post = -1
    if match.group("post") is not None:
        number = match.group("number") or match.group("implicit")
        post = int(number) if number else 0
    return (int(match.group("epoch") or 0), tuple(release), post)


def parse_pins(text):
    """Return the ``Pin`` list of a hashed requirements file.

    Blank lines, comments and ``--hash=`` lines are skipped. Any other line
    that is not ``name==version`` (with extras and a marker allowed) raises
    ``PinFileError``, as does a file with no pin.
    """
    pins = []
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "--hash=")):
            continue
        match = _PIN.fullmatch(stripped)
        if match is None:
            raise PinFileError(f"line {number} is not a pin: {stripped!r}")
        pins.append(Pin(match.group(1), match.group(3), match.group(4)))
    if not pins:
        raise PinFileError("no pin found")
    return pins


def direct_names(text):
    """Return the normalised names of a ``.in`` file."""
    names = set()
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _NAME.match(stripped)
        if match is None:
            raise PinFileError(f"line {number} names no package: {stripped!r}")
        names.add(normalise(match.group(0)))
    return names


def regeneration_command(text):
    """Return the command in a file's leading comments, or ``None``.

    It is the first comment line that starts with ``uv``, joined with the
    comment lines that follow while the previous one ends in a backslash.
    """
    parts = []
    collecting = False
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        body = line[1:].strip()
        if not collecting:
            if not body.startswith("uv "):
                continue
            collecting = True
        parts.append(body.rstrip("\\").strip())
        if not body.endswith("\\"):
            break
    return " ".join(parts) if parts else None


def available(releases):
    """Return ``{key: version}`` of the unyanked final releases.

    A release with no file, or with every file yanked, is not available.
    """
    found = {}
    for version, files in releases.items():
        key = final_key(version)
        if key is not None and any(not f.get("yanked") for f in files):
            found[key] = version
    return found


def fetch_pypi(name, urlopen=urllib.request.urlopen):
    """Return the ``releases`` mapping PyPI lists for ``name``.

    Raises ``LookupFailure`` when PyPI cannot be asked or answers with
    something else than the expected object. No retry.
    """
    request = urllib.request.Request(
        PYPI.format(normalise(name)),
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=TIMEOUT) as response:
            body = json.load(response)
    except (OSError, http.client.HTTPException, ValueError) as error:
        raise LookupFailure(str(error)) from error
    releases = body.get("releases") if isinstance(body, dict) else None
    if not isinstance(releases, dict) or not all(
        isinstance(files, list) and all(isinstance(f, dict) for f in files)
        for files in releases.values()
    ):
        raise LookupFailure("the answer has no usable list of releases")
    return releases


def _read(root):
    """Return ``([(label, stem, pins, direct, command)], errors)``."""
    files = []
    errors = []
    for label in FILES:
        path = root / label
        in_label = str(pathlib.PurePosixPath(label).with_suffix(".in"))
        where = label
        try:
            text = path.read_text(encoding="utf-8")
            pins = parse_pins(text)
            where = in_label
            direct = direct_names(
                path.with_suffix(".in").read_text(encoding="utf-8")
            )
        except OSError as error:
            errors.append(f"{where}: {error.strerror or 'cannot be read'}")
        except (UnicodeDecodeError, PinFileError) as error:
            errors.append(f"{where}: {error}")
        else:
            files.append(
                (label, path.stem, pins, direct, regeneration_command(text))
            )
    return files, errors


def _hold_findings(files, holds):
    """Return the lines for each hold that no longer matches the files."""
    out = []
    redo = "; re-read the cap and update HOLDS in scripts/check_pins.py"
    for name, hold in holds.items():
        seen = False
        for label, _, pins, _, _ in files:
            versions = {}
            for pin in pins:
                versions.setdefault(normalise(pin.name), []).append(
                    pin.version
                )
            if name not in versions:
                continue
            seen = True
            cap = final_key(hold.below)
            for version in versions[name]:
                key = final_key(version)
                if key is not None and key >= cap:
                    out.append(
                        f"hold    {name}: pinned at {version} in {label}, "
                        f"not below {hold.below}{redo}"
                    )
            if hold.by_version not in versions.get(hold.by, []):
                now = ", ".join(versions.get(hold.by, [])) or "not pinned"
                out.append(
                    f"hold    {name}: written for {hold.by} "
                    f"{hold.by_version}, pinned at {now} in {label}{redo}"
                )
        if not seen:
            out.append(
                f"hold    {name}: written for {hold.by} {hold.by_version}, "
                f"but {name} is not pinned in either file{redo}"
            )
    return out


def _remedy(label, stem, name, newest, direct, command):
    """Print the lines that tell how to move a stale pin."""
    pad = " " * 8
    if command is None:
        step = (
            f"{label} carries no regeneration command in its first "
            f"lines; regenerate it with uv pip compile"
        )
        if normalise(name) in direct:
            print(f"{pad}set {name}=={newest} in ci/{stem}.in, then")
            print(f"{pad}{step}.")
        else:
            print(f"{pad}{step} and --upgrade-package {name}.")
        return
    if normalise(name) in direct:
        print(f"{pad}set {name}=={newest} in ci/{stem}.in, then from ci/ run:")
        print(f"{pad}{command}")
    else:
        print(f"{pad}from ci/ run:")
        print(f"{pad}{command} --upgrade-package {name}")
    print(
        f"{pad}and put back the comment lines at the top of {label}, "
        "which --no-header drops."
    )
    if normalise(name) not in direct:
        print(f"{pad}(the command keeps the versions already in the file)")


def _rank(pin):
    """Return a key that orders the pins of one name; non-finals lowest."""
    key = final_key(pin.version)
    return (0, ()) if key is None else (1, key)


def check(root, fetch=fetch_pypi, holds=HOLDS):
    """Print the verdict on every pin under ``root``; return the status.

    Every file is read before any lookup. The status is 2 on an error, else
    1 when a pin is stale or a hold is out of date, else 0.
    """
    files, errors = _read(pathlib.Path(root))
    for message in errors:
        print(f"error   {message}")
    if errors:
        return 2
    findings = _hold_findings(files, holds)
    for line in findings:
        print(line)
    cache = {}
    counts = {"checked": 0, "stale": 0, "held": 0, "failed": 0}

    def releases_of(name):
        if name not in cache:
            try:
                cache[name] = fetch(name)
            except LookupFailure as error:
                cache[name] = error
        return cache[name]

    for label, stem, pins, direct, command in files:
        top = {}
        for index, pin in enumerate(pins):
            name = normalise(pin.name)
            if name not in top or _rank(pin) > _rank(pins[top[name]]):
                top[name] = index
        for index, pin in enumerate(pins):
            name = normalise(pin.name)
            spec = f"{pin.name}=={pin.version}"
            if top[name] != index:
                print(
                    f"skip    {label}  {spec} ; {pin.marker}: a lower pin "
                    "under its marker, not compared"
                )
                continue
            counts["checked"] += 1
            found = releases_of(name)
            if isinstance(found, LookupFailure):
                counts["failed"] += 1
                print(f"error   {label}  {spec}: PyPI lookup failed: {found}")
                continue
            avail = available(found)
            hold = holds.get(name)
            cap = final_key(hold.below) if hold else None
            usable = [k for k in avail if cap is None or k < cap]
            if not usable:
                counts["failed"] += 1
                print(
                    f"error   {label}  {spec}: no final, unyanked release "
                    "on PyPI" + (f" below {hold.below}" if hold else "")
                )
                continue
            newest = avail[max(usable)]
            key = final_key(pin.version)
            reason = None
            if key is None:
                reason = f"is not a final release; newest is {newest}"
            elif key not in avail:
                reason = f"is yanked or not on PyPI; newest is {newest}"
            elif key < max(usable):
                reason = f"newest is {newest}"
            if reason:
                counts["stale"] += 1
                print(f"stale   {label}  {spec}: {reason}")
                _remedy(label, stem, pin.name, newest, direct, command)
            elif hold and key < cap <= max(avail):
                counts["held"] += 1
                print(
                    f"held    {label}  {spec}: newest is "
                    f"{avail[max(avail)]}; held below {hold.below}: "
                    f"{hold.reason}"
                )
            else:
                print(f"ok      {label}  {spec}")
    print(
        f"{counts['checked']} pin(s) checked: {counts['stale']} stale, "
        f"{counts['held']} held, {counts['failed']} lookup failure(s)."
    )
    if counts["failed"]:
        return 2
    return 1 if counts["stale"] or findings else 0


def main(argv=None):
    """Parse the arguments and run the check."""
    parser = argparse.ArgumentParser(
        prog="check_pins.py", description=__doc__.strip().splitlines()[0]
    )
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=ROOT,
        help="directory that holds ci/ (default: the repository)",
    )
    return check(parser.parse_args(argv).root)


if __name__ == "__main__":
    sys.exit(main())
