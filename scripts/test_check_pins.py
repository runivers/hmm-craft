"""
Tests of ``scripts/check_pins.py``, the check of the hashed pins under ``ci/``.

The PyPI lookup is injected, so no test reaches the network: ``check`` gets a
fake ``fetch`` built from hand-written releases, and ``fetch_pypi`` gets a fake
``urlopen``. Package names are synthetic unless the real name is the point.

Run with ``python -m pytest scripts/test_check_pins.py``.
"""

import http.client
import io
import json
import pathlib
import subprocess
import sys
import urllib.error

import check_pins
import pytest
from check_pins import Hold, LookupFailure, PinFileError

SCRIPT = pathlib.Path(check_pins.__file__).resolve()
BUILD_COMMAND = (
    "uv pip compile --universal --python-version 3.12 --generate-hashes"
    " --no-header --no-annotate build_requirements.in"
    " -o build_requirements.txt"
)
TOOL_COMMAND = BUILD_COMMAND.replace("build_requirements", "tool_requirements")
HASH = "    --hash=sha256:" + "0" * 64
BUILD_TXT = ("alpha-pkg==1.0", "packaging==26.3")
BUILD_IN = ("alpha-pkg==1.0",)
TOOL_TXT = ("build==1.6.1", "packaging==26.3")
TOOL_IN = ("build==1.6.1",)
CURRENT = {
    "alpha-pkg": ["1.0"],
    "build": ["1.6.1"],
    "packaging": ["26.3"],
}


def _header(stem):
    """Return the three comment lines a regenerated file starts with."""
    return (
        f"# Pinned with hashes. Regenerate, from this directory, after"
        f" editing {stem}.in:\n"
        "#   uv pip compile --universal --python-version 3.12"
        " --generate-hashes \\\n"
        f"#     --no-header --no-annotate {stem}.in -o {stem}.txt\n"
    )


def _txt(stem, pins, header=True):
    """Return the text of a hashed requirements file pinning ``pins``."""
    lines = [_header(stem)] if header else []
    for pin in pins:
        lines.append(f"{pin} \\\n{HASH} \\\n{HASH}\n")
    return "".join(lines)


def _tree(
    root,
    build=BUILD_TXT,
    build_in=BUILD_IN,
    tool=TOOL_TXT,
    tool_in=TOOL_IN,
    header=True,
):
    """Write ``ci/`` under ``root`` and return ``root``."""
    ci = root / "ci"
    ci.mkdir(exist_ok=True)
    (ci / "build_requirements.txt").write_text(
        _txt("build_requirements", build, header)
    )
    (ci / "tool_requirements.txt").write_text(
        _txt("tool_requirements", tool, header)
    )
    if build_in is not None:
        (ci / "build_requirements.in").write_text("\n".join(build_in) + "\n")
    if tool_in is not None:
        (ci / "tool_requirements.in").write_text("\n".join(tool_in) + "\n")
    return root


def _releases(*versions, yanked=(), empty=()):
    """Return a PyPI ``releases`` mapping of ``versions``."""
    return {
        v: [] if v in empty else [{"yanked": v in yanked}] for v in versions
    }


class FakeFetch:
    """A ``fetch`` that serves hand-written releases and records its asks."""

    def __init__(self, world):
        self.world = {}
        for name, value in world.items():
            if isinstance(value, list):
                value = _releases(*value)
            self.world[name] = value
        self.asked = []

    def __call__(self, name):
        self.asked.append(name)
        value = self.world[name]
        if isinstance(value, Exception):
            raise value
        return value


def _run(tmp_path, capsys, world=None, holds=None, **tree):
    """Check a tree; return ``(status, output, fake fetch)``."""
    _tree(tmp_path, **tree)
    fake = FakeFetch({**CURRENT, **(world or {})})
    status = check_pins.check(tmp_path, fetch=fake, holds=holds or {})
    return status, capsys.readouterr().out, fake


def _lines(output, word):
    """Return the lines of ``output`` that start with ``word``."""
    return [x for x in output.splitlines() if x.split(" ")[0] == word]


def test_current_pins_pass(tmp_path, capsys):
    """Pins equal to the newest final release pass, one line each."""
    status, out, _ = _run(tmp_path, capsys)
    assert status == 0, out
    assert len(_lines(out, "ok")) == 4
    assert not _lines(out, "stale")
    assert "4 pin(s) checked: 0 stale, 0 held, 0 lookup failure(s)." in out


def test_stale_pin_names_file_package_pin_and_newest(tmp_path, capsys):
    """A stale pin fails and its line names file, package, pin, newest."""
    status, out, _ = _run(
        tmp_path,
        capsys,
        world={"build": ["1.5.0", "1.6.1"]},
        tool=("build==1.5.0", "packaging==26.3"),
        tool_in=("build==1.5.0",),
    )
    assert status == 1
    (line,) = _lines(out, "stale")
    for text in ("ci/tool_requirements.txt", "build==1.5.0", "1.6.1"):
        assert text in line


def test_stale_direct_pin_remedy_edits_the_in_file(tmp_path, capsys):
    """A pin named in the .in file is fixed there, then regenerated."""
    status, out, _ = _run(
        tmp_path,
        capsys,
        world={"build": ["1.5.0", "1.6.1"]},
        tool=("build==1.5.0", "packaging==26.3"),
        tool_in=("build==1.5.0",),
    )
    assert status == 1
    assert "set build==1.6.1 in ci/tool_requirements.in" in out
    assert TOOL_COMMAND in out
    assert "--upgrade-package" not in out
    assert "--no-header drops" in out


def test_stale_transitive_pin_remedy_adds_upgrade_package(tmp_path, capsys):
    """A pin absent from the .in file needs --upgrade-package."""
    status, out, _ = _run(
        tmp_path, capsys, world={"packaging": ["26.3", "26.4"]}
    )
    assert status == 1
    assert TOOL_COMMAND + " --upgrade-package packaging" in out
    assert BUILD_COMMAND + " --upgrade-package packaging" in out
    assert "set packaging" not in out


def test_remedy_without_a_header_still_names_the_pin(tmp_path, capsys):
    """A file without its regeneration command still gets a remedy."""
    status, out, _ = _run(
        tmp_path, capsys, world={"packaging": ["26.3", "26.4"]}, header=False
    )
    assert status == 1
    assert "carries no regeneration command" in out
    assert "--upgrade-package packaging" in out


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a==1.0\n", [("a", "1.0", None)]),
        (
            "a==1.0 \\\n    --hash=sha256:ab \\\n    --hash=sha256:cd\n",
            [("a", "1.0", None)],
        ),
        (
            "colorama==0.4.6 ; os_name == 'nt' \\\n    --hash=sha256:ab\n",
            [("colorama", "0.4.6", "os_name == 'nt'")],
        ),
        ("a[extra]==1.0\n", [("a", "1.0", None)]),
        (
            "# a comment\n\n   \na==1.0\n# another\nB_c==2\n",
            [("a", "1.0", None), ("B_c", "2", None)],
        ),
    ],
    ids=["plain", "hash-lines", "marker", "extras", "comments-and-blanks"],
)
def test_parse_pins_reads_pin_lines(text, expected):
    """Names, versions and markers are read; comments and hashes skipped."""
    assert [tuple(p) for p in check_pins.parse_pins(text)] == expected


def test_environment_marker_pin_is_checked(tmp_path, capsys):
    """A stale pin carrying an environment marker is reported."""
    status, out, _ = _run(
        tmp_path,
        capsys,
        world={"colorama": ["0.4.6", "0.4.7"]},
        tool=("build==1.6.1", "colorama==0.4.6 ; os_name == 'nt'"),
    )
    assert status == 1
    (line,) = _lines(out, "stale")
    assert "colorama==0.4.6" in line
    assert "0.4.7" in line


@pytest.mark.parametrize(
    "bad", ["-e .", "pkg>=1.0", "--index-url x", "pkg==1.0 extra"]
)
def test_unknown_line_is_an_error(bad, tmp_path, capsys):
    """A line the parser does not know fails; it is never skipped."""
    with pytest.raises(PinFileError):
        check_pins.parse_pins("a==1.0\n" + bad + "\n")
    status, out, fake = _run(tmp_path, capsys, build=("alpha-pkg==1.0", bad))
    assert status == 2
    assert out.startswith("error")
    assert fake.asked == []


def test_file_without_pins_is_an_error(tmp_path, capsys):
    """A file with no pin at all is an error, not a pass."""
    with pytest.raises(PinFileError):
        check_pins.parse_pins("# only a comment\n\n")
    status, _out, fake = _run(tmp_path, capsys, tool=())
    assert status == 2
    assert fake.asked == []


def test_missing_in_file_is_an_error(tmp_path, capsys):
    """A missing .in file fails before any lookup."""
    status, out, fake = _run(tmp_path, capsys, tool_in=None)
    assert status == 2
    assert "tool_requirements.in" in out
    assert fake.asked == []


def test_bad_in_file_line_names_the_in_file(tmp_path, capsys):
    """A line the .in parser rejects is reported under the .in file."""
    status, out, fake = _run(tmp_path, capsys, tool_in=("-r other.in",))
    assert status == 2
    (line,) = _lines(out, "error")
    assert line.startswith("error   ci/tool_requirements.in:")
    assert fake.asked == []


@pytest.mark.parametrize(
    "name", ["Setuptools_SCM", "setuptools.scm", "setuptools--scm"]
)
def test_normalise(name):
    """Names are compared in PEP 503 form."""
    assert check_pins.normalise(name) == "setuptools-scm"


def test_in_name_matches_its_pin_after_normalisation(tmp_path, capsys):
    """``Alpha_Pkg`` in the .in file makes ``alpha-pkg`` a direct pin."""
    status, out, _ = _run(
        tmp_path,
        capsys,
        world={"alpha-pkg": ["1.0", "1.1"]},
        build_in=("Alpha_Pkg==1.0",),
    )
    assert status == 1
    assert "set alpha-pkg==1.1 in ci/build_requirements.in" in out


@pytest.mark.parametrize(
    ("greater", "lesser"),
    [
        ("1.10", "1.9"),
        ("1!0.1", "2.0"),
        ("26.3.post1", "26.3"),
        ("1.0-1", "1.0"),
    ],
    ids=["numeric-not-text", "epoch", "post", "implicit-post"],
)
def test_final_key_orders(greater, lesser):
    """Final and post releases order as PEP 440 says."""
    assert check_pins.final_key(greater) > check_pins.final_key(lesser)


@pytest.mark.parametrize(
    ("one", "other"),
    [("1.0", "1.0.0"), ("1.0-1", "1.0.post1"), ("V1.0", "1.0")],
    ids=["trailing-zeros", "post-spellings", "v-prefix"],
)
def test_final_key_equal_spellings(one, other):
    """Spellings of one version give one key."""
    assert check_pins.final_key(one) == check_pins.final_key(other)
    assert check_pins.final_key(one) is not None


@pytest.mark.parametrize(
    "version",
    [
        "27.0rc1",
        "27.0a1",
        "27.0b2",
        "27.0c1",
        "27.0.dev0",
        "27.0.post1.dev0",
        "27.0+local",
        "0.6-preview",
        "not-a-version",
        "",
    ],
)
def test_final_key_rejects(version):
    """Pre-releases, dev and local versions and junk are not final."""
    assert check_pins.final_key(version) is None


def test_prerelease_is_not_newest(tmp_path, capsys):
    """A pre-release or dev release is never the newest."""
    status, out, _ = _run(
        tmp_path, capsys, world={"packaging": ["26.3", "27.0rc1", "27.0.dev0"]}
    )
    assert status == 0, out


def test_yanked_release_is_not_newest(tmp_path, capsys):
    """A release whose files are all yanked is not the newest."""
    world = {"packaging": _releases("26.3", "27.0", yanked=("27.0",))}
    status, out, _ = _run(tmp_path, capsys, world=world)
    assert status == 0, out


def test_release_with_one_unyanked_file_counts(tmp_path, capsys):
    """One file not yanked is enough for a release to be available."""
    world = {
        "packaging": {
            "26.3": [{"yanked": False}],
            "27.0": [{"yanked": True}, {"yanked": False}],
        }
    }
    status, out, _ = _run(tmp_path, capsys, world=world)
    assert status == 1
    assert "newest is 27.0" in out


def test_release_without_files_is_not_newest(tmp_path, capsys):
    """A release with no file is not the newest."""
    world = {"packaging": _releases("26.3", "27.0", empty=("27.0",))}
    status, out, _ = _run(tmp_path, capsys, world=world)
    assert status == 0, out


def test_post_release_is_newer(tmp_path, capsys):
    """A post release is newer than the final it follows."""
    status, out, _ = _run(
        tmp_path, capsys, world={"packaging": ["26.3", "26.3.post1"]}
    )
    assert status == 1
    assert "newest is 26.3.post1" in out


def test_unparseable_release_is_ignored(tmp_path, capsys):
    """A release key that is not a version is skipped, not an error."""
    status, out, _ = _run(
        tmp_path, capsys, world={"packaging": ["26.3", "dev-branch"]}
    )
    assert status == 0, out


def test_yanked_pin_is_reported(tmp_path, capsys):
    """A pin whose release was yanked is stale."""
    world = {"build": _releases("1.6.0", "1.6.1", yanked=("1.6.1",))}
    status, out, _ = _run(tmp_path, capsys, world=world)
    assert status == 1
    (line,) = _lines(out, "stale")
    assert "yanked or not on PyPI" in line
    assert "newest is 1.6.0" in line


def test_prerelease_pin_is_reported(tmp_path, capsys):
    """A pin that is a pre-release is stale."""
    status, out, _ = _run(
        tmp_path,
        capsys,
        world={"build": ["1.6.1", "1.7rc1"]},
        tool=("build==1.7rc1", "packaging==26.3"),
        tool_in=("build",),
    )
    assert status == 1
    (line,) = _lines(out, "stale")
    assert "not a final release" in line
    assert "newest is 1.6.1" in line


def test_floor_file_is_not_read(tmp_path, capsys):
    """The floor file is never parsed, and its packages never asked."""
    _tree(tmp_path)
    (tmp_path / "ci" / "floor_constraints.txt").write_text(
        "numpy==1.26.0\nthis is not a pin\n"
    )
    fake = FakeFetch(CURRENT)
    status = check_pins.check(tmp_path, fetch=fake, holds={})
    capsys.readouterr()
    assert status == 0
    assert "numpy" not in fake.asked


def test_files_exclude_the_floor_file():
    """The fixed list of files has no floor file."""
    assert check_pins.FILES == (
        "ci/build_requirements.txt",
        "ci/tool_requirements.txt",
    )


def test_lookup_failure_has_its_own_message_and_status(tmp_path, capsys):
    """A failed lookup is status 2 on its own line, not a stale verdict."""
    world = {"alpha-pkg": LookupFailure("boom")}
    status, out, _ = _run(tmp_path, capsys, world=world)
    assert status == 2
    (line,) = _lines(out, "error")
    assert "PyPI lookup failed: boom" in line
    assert "alpha-pkg==1.0" in line
    assert not _lines(out, "stale")
    assert len(_lines(out, "ok")) == 3
    assert "1 lookup failure(s)" in out


def test_lookup_failure_does_not_hide_a_stale_pin(tmp_path, capsys):
    """With a failure and a stale pin, status is 2 and both are printed."""
    world = {"alpha-pkg": LookupFailure("boom"), "packaging": ["26.3", "26.4"]}
    status, out, _ = _run(tmp_path, capsys, world=world)
    assert status == 2
    assert _lines(out, "stale")
    assert _lines(out, "error")


def test_no_final_release_is_a_lookup_error(tmp_path, capsys):
    """A package with no final, unyanked release cannot be decided."""
    status, out, _ = _run(tmp_path, capsys, world={"alpha-pkg": ["1.0rc1"]})
    assert status == 2
    (line,) = _lines(out, "error")
    assert "no final, unyanked release" in line


def test_package_in_both_files_is_asked_once(tmp_path, capsys):
    """One lookup per package, though both files pin it."""
    _, _, fake = _run(tmp_path, capsys)
    assert fake.asked.count("packaging") == 1
    assert sorted(fake.asked) == ["alpha-pkg", "build", "packaging"]


HOLD = Hold(
    below="3",
    by="beta-pkg",
    by_version="10.3.4",
    reason="beta-pkg 10.3.4 requires gamma<3",
)
HOLD_TREE = {
    "build": ("gamma==2.6.0", "beta-pkg==10.3.4"),
    "build_in": ("beta-pkg==10.3.4",),
}


def _held(tmp_path, capsys, versions, holds=None):
    world = {"gamma": versions, "beta-pkg": ["10.3.4"]}
    return _run(
        tmp_path,
        capsys,
        world=world,
        holds={"gamma": HOLD} if holds is None else holds,
        **HOLD_TREE,
    )


def test_held_pin_is_not_stale_while_the_cap_binds(tmp_path, capsys):
    """A pin at the newest release below its cap passes as held."""
    status, out, _ = _held(tmp_path, capsys, ["2.6.0", "3.0.0"])
    assert status == 0, out
    (line,) = _lines(out, "held")
    assert "3.0.0" in line
    assert HOLD.reason in line
    assert "1 held" in out


def test_held_pin_is_stale_below_the_cap(tmp_path, capsys):
    """A newer release below the cap is named, with the upgrade remedy."""
    status, out, _ = _held(tmp_path, capsys, ["2.6.0", "2.7.0", "3.0.0"])
    assert status == 1
    (line,) = _lines(out, "stale")
    assert "newest is 2.7.0" in line
    assert "--upgrade-package gamma" in out


def test_hold_is_compared_normally_when_the_cap_does_not_bind(
    tmp_path, capsys
):
    """With no release at or above the cap, the hold changes nothing."""
    status, out, _ = _held(tmp_path, capsys, ["2.6.0", "2.7.0"])
    assert status == 1
    assert "newest is 2.7.0" in _lines(out, "stale")[0]
    assert not _lines(out, "held")


def test_held_pin_at_the_newest_release_passes(tmp_path, capsys):
    """With no release at or above the cap, a current pin is just ok."""
    status, out, _ = _held(tmp_path, capsys, ["2.5.0", "2.6.0"])
    assert status == 0, out
    assert not _lines(out, "held")


def test_hold_out_of_date_when_its_capping_pin_moves(tmp_path, capsys):
    """A hold written for another version of its capping pin is reported."""
    old = HOLD._replace(by_version="10.3.3")
    status, out, _ = _held(
        tmp_path, capsys, ["2.6.0", "3.0.0"], holds={"gamma": old}
    )
    assert status == 1
    (line,) = _lines(out, "hold")
    assert "10.3.3" in line
    assert "10.3.4" in line


def test_hold_for_a_name_not_pinned_is_reported(tmp_path, capsys):
    """A hold for a package no file pins is out of date."""
    status, out, _ = _run(tmp_path, capsys, holds={"delta": HOLD})
    assert status == 1
    (line,) = _lines(out, "hold")
    assert "delta" in line
    assert "not pinned" in line


def test_pin_at_or_above_its_cap_is_reported(tmp_path, capsys):
    """A held pin at or above its cap means the hold no longer holds."""
    tree = {**HOLD_TREE, "build": ("gamma==3.0.0", "beta-pkg==10.3.4")}
    status, out, _ = _run(
        tmp_path,
        capsys,
        world={"gamma": ["2.6.0", "3.0.0"], "beta-pkg": ["10.3.4"]},
        holds={"gamma": HOLD},
        **tree,
    )
    assert status == 1, out
    (line,) = _lines(out, "hold")
    assert "gamma" in line
    assert "3.0.0" in line
    assert "below 3" in line
    assert not _lines(out, "held")


def test_marker_split_pin_compares_the_highest(tmp_path, capsys):
    """Of one name pinned under several markers, the highest is compared."""
    status, out, _ = _run(
        tmp_path,
        capsys,
        world={"alpha-pkg": ["1.0", "2.0"]},
        build=(
            "alpha-pkg==1.0 ; python_version < '3.13'",
            "alpha-pkg==2.0 ; python_version >= '3.13'",
            "packaging==26.3",
        ),
    )
    assert status == 0, out
    (line,) = _lines(out, "skip")
    assert "alpha-pkg==1.0" in line


class _Response(io.BytesIO):
    """A response body that is a context manager, as urlopen returns."""


def _urlopen(body=None, error=None):
    """Return a fake ``urlopen`` and the list its calls are recorded in."""
    calls = []

    def urlopen(request, timeout=None):
        calls.append((request, timeout))
        if error is not None:
            raise error
        return _Response(body)

    return urlopen, calls


def test_fetch_asks_the_normalised_url():
    """The URL is the normalised name's, with a timeout and a User-Agent."""
    body = json.dumps({"releases": {"1.0": [{"yanked": False}]}}).encode()
    urlopen, calls = _urlopen(body)
    releases = check_pins.fetch_pypi("Setuptools_SCM", urlopen=urlopen)
    assert releases == {"1.0": [{"yanked": False}]}
    ((request, timeout),) = calls
    assert request.full_url == "https://pypi.org/pypi/setuptools-scm/json"
    assert timeout == check_pins.TIMEOUT
    assert request.get_header("User-agent") == check_pins.USER_AGENT


@pytest.mark.parametrize(
    ("body", "error"),
    [
        (None, urllib.error.URLError("no route")),
        (
            None,
            urllib.error.HTTPError(
                "https://pypi.org/x", 404, "Not Found", {}, io.BytesIO()
            ),
        ),
        (None, TimeoutError("timed out")),
        (None, http.client.IncompleteRead(b"")),
        (b"not json", None),
        (b"\xff\xfe", None),
        (b"[]", None),
        (b'{"info": {}}', None),
        (b'{"releases": []}', None),
        (b'{"releases": {"1.0": "x"}}', None),
    ],
    ids=[
        "urlerror",
        "http-404",
        "timeout",
        "incomplete-read",
        "bad-json",
        "bad-utf8",
        "not-an-object",
        "no-releases",
        "releases-a-list",
        "files-not-a-list",
    ],
)
def test_fetch_failure_is_a_lookup_failure(body, error):
    """Every failure of the lookup is a LookupFailure, nothing else."""
    urlopen, _ = _urlopen(body, error)
    with pytest.raises(LookupFailure):
        check_pins.fetch_pypi("alpha-pkg", urlopen=urlopen)


def test_every_hold_matches_the_pinned_files():
    """Each real hold's package and capping pin are pinned as it says."""
    for name, hold in check_pins.HOLDS.items():
        found = False
        for label in check_pins.FILES:
            pins = check_pins.parse_pins(
                (check_pins.ROOT / label).read_text(encoding="utf-8")
            )
            versions = {check_pins.normalise(p.name): p.version for p in pins}
            if name in versions:
                found = True
                assert versions[hold.by] == hold.by_version
        assert found
        assert "<" + hold.below in hold.reason


@pytest.mark.parametrize("label", check_pins.FILES)
def test_each_pinned_file_keeps_its_regeneration_command(label):
    """The first lines of each file hold the command that regenerates it."""
    path = check_pins.ROOT / label
    command = check_pins.regeneration_command(path.read_text(encoding="utf-8"))
    assert command is not None
    assert "--generate-hashes" in command
    assert command.endswith("-o " + path.name)


def test_regeneration_command_joins_continued_lines():
    """A command split by trailing backslashes comes back on one line."""
    assert (
        check_pins.regeneration_command(_header("tool_requirements"))
        == TOOL_COMMAND
    )
    assert check_pins.regeneration_command("a==1\n") is None


def test_script_fails_cleanly_on_a_missing_root(tmp_path):
    """A root without the files is status 2 and no traceback."""
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--root", str(tmp_path / "none")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr + result.stdout
    assert "error" in result.stdout
