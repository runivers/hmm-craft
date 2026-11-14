"""
Tests of ``scripts/check_signoff.py``, the sign-off check of a pull request.

Each test builds a small repository in a temporary directory, with git's
global and system configuration and hooks switched off, and runs the check on
a range of it.

Run with ``python -m pytest scripts/test_check_signoff.py``.
"""

import os
import pathlib
import subprocess
import sys

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parent / "check_signoff.py"
ME = ("Ann Example", "ann@example.org")
OTHER = ("Bob Example", "bob@example.org")
BOT = ("dependabot[bot]",
       "49699333+dependabot[bot]@users.noreply.github.com")


def _env():
    """Return an environment that reaches no configuration or repository
    outside the test's own."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return env


def _git(repo, *args, author=ME, committer=ME):
    env = _env()
    env.update(GIT_AUTHOR_NAME=author[0], GIT_AUTHOR_EMAIL=author[1],
               GIT_COMMITTER_NAME=committer[0],
               GIT_COMMITTER_EMAIL=committer[1])
    result = subprocess.run(
        ["git", "-C", str(repo), "-c", "core.hooksPath=" + os.devnull,
         *args],
        env=env, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _signoff(who):
    return f"Signed-off-by: {who[0]} <{who[1]}>"


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    _commit(tmp_path, "base", "chore: base\n\n" + _signoff(ME))
    return tmp_path


def _commit(repo, name, message, author=ME, committer=ME,
            cleanup="default"):
    (repo / name).write_text(name + "\n")
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "--cleanup=" + cleanup, "-m", message,
         author=author, committer=committer)
    return _git(repo, "rev-parse", "HEAD")


def _check(repo, base, head="HEAD"):
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT), base, head],
        cwd=repo, env=_env(), capture_output=True, text=True, check=False)


def _branch(repo):
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-q", "-c", "topic")
    return base


def test_signed_off_by_author_passes(repo):
    base = _branch(repo)
    _commit(repo, "a", "fix: a\n\n" + _signoff(ME))
    result = _check(repo, base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 commit" in result.stdout


def test_missing_signoff_fails_and_names_the_commit(repo):
    base = _branch(repo)
    sha = _commit(repo, "a", "fix: a")
    result = _check(repo, base)
    assert result.returncode == 1
    assert sha[:12] in result.stdout


def test_one_bad_commit_among_good_ones_fails(repo):
    base = _branch(repo)
    _commit(repo, "a", "fix: a\n\n" + _signoff(ME))
    bad = _commit(repo, "b", "fix: b")
    _commit(repo, "c", "fix: c\n\n" + _signoff(ME))
    result = _check(repo, base)
    assert result.returncode == 1
    assert bad[:12] in result.stdout
    assert "1 of 3" in result.stdout


def test_signoff_by_someone_else_fails(repo):
    base = _branch(repo)
    _commit(repo, "a", "fix: a\n\n" + _signoff(OTHER))
    assert _check(repo, base).returncode == 1


def test_address_compared_without_case(repo):
    base = _branch(repo)
    _commit(repo, "a", "fix: a\n\nSigned-off-by: Ann <ANN@Example.ORG>")
    assert _check(repo, base).returncode == 0


def test_crlf_message_passes(repo):
    base = _branch(repo)
    _commit(repo, "a", "fix: a\r\n\r\n" + _signoff(ME) + "\r\n",
            cleanup="verbatim")
    stored = subprocess.run(["git", "-C", str(repo), "cat-file", "-p",
                             "HEAD"], env=_env(), check=True,
                            capture_output=True).stdout
    assert b"\r\n" in stored
    assert _check(repo, base).returncode == 0


def test_empty_message_fails(repo):
    base = _branch(repo)
    (repo / "a").write_text("a\n")
    _git(repo, "add", "a")
    _git(repo, "commit", "-q", "--allow-empty-message", "-m", "")
    result = _check(repo, base)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("separator", ["\f", "\v", "\x1c", "\u2028"])
def test_signoff_after_a_non_newline_separator_fails(repo, separator):
    # git splits a message into lines at "\n" only.
    base = _branch(repo)
    _commit(repo, "a", "fix: a\n\nSee " + separator + _signoff(ME))
    assert _check(repo, base).returncode == 1


def test_control_characters_in_a_message_are_read(repo):
    base = _branch(repo)
    _commit(repo, "a", "fix: a \x1e\x1fb\n\n" + _signoff(ME))
    result = _check(repo, base)
    assert result.returncode == 0, result.stdout + result.stderr


def test_committer_signoff_accepted_for_a_commit_taken_over(repo):
    # A commit taken from upstream keeps its author, who did not sign off;
    # the committer who took it signs off.
    base = _branch(repo)
    _commit(repo, "a", "fix: a\n\n" + _signoff(ME), author=OTHER,
            committer=ME)
    assert _check(repo, base).returncode == 0


def test_signoff_line_must_be_a_whole_line(repo):
    base = _branch(repo)
    _commit(repo, "a",
            "fix: a\n\nSee " + _signoff(ME) + " in the template.")
    assert _check(repo, base).returncode == 1


def test_dependabot_commit_exempt(repo):
    base = _branch(repo)
    _commit(repo, "a", "chore: bump an action", author=BOT, committer=OTHER)
    result = _check(repo, base)
    assert result.returncode == 0, result.stdout
    assert "exempt" in result.stdout


def test_dependabot_name_with_another_address_not_exempt(repo):
    base = _branch(repo)
    _commit(repo, "a", "chore: a", author=(BOT[0], "x@example.org"))
    assert _check(repo, base).returncode == 1


def test_merge_commit_skipped(repo):
    base = _branch(repo)
    _commit(repo, "a", "fix: a\n\n" + _signoff(ME))
    _git(repo, "switch", "-q", "main")
    _commit(repo, "m", "fix: m\n\n" + _signoff(ME))
    _git(repo, "switch", "-q", "topic")
    _git(repo, "merge", "-q", "--no-ff", "--no-edit", "main")
    result = _check(repo, base)
    assert result.returncode == 0, result.stdout
    assert "2 commit" in result.stdout


def test_empty_range_passes(repo):
    head = _git(repo, "rev-parse", "HEAD")
    result = _check(repo, head, head)
    assert result.returncode == 0
    assert "0 commit" in result.stdout


def test_unknown_revision_is_an_error(repo):
    result = _check(repo, "0" * 40)
    assert result.returncode == 2


def test_wrong_arguments_are_an_error(repo):
    result = subprocess.run([sys.executable, "-I", str(SCRIPT), "HEAD"],
                            cwd=repo, env=_env(), capture_output=True,
                            text=True, check=False)
    assert result.returncode == 2
