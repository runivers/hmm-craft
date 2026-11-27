"""
Check that every commit of a pull request is signed off.

usage: python scripts/check_signoff.py BASE HEAD

Each commit reachable from HEAD and not from BASE, merges excepted, needs a
line ``Signed-off-by: Name <address>`` whose address equals, ignoring case,
the address of the commit's author or of its committer. The committer's
sign-off covers a commit taken over from hmmlearn, whose author did not sign
off. Commits made by Dependabot are exempt; the exemption needs its author
name and address and GitHub as committer, since anyone can set an author.
Lines are split at "\n" only, as git splits them.

Exit status: 0 when every commit passes, 1 when one does not (each one is
printed), 2 on a usage or git error. Standard library only.
"""

import re
import subprocess
import sys

# (author name, author address, committer name, committer address)
EXEMPT_IDENTITIES = {
    (
        "dependabot[bot]",
        "49699333+dependabot[bot]@users.noreply.github.com",
        "GitHub",
        "noreply@github.com",
    ),
}
SIGNOFF = re.compile(r"^Signed-off-by: .+ <([^<>\s]+)>\s*$")
# The fields of one commit, separated by NUL; the message comes last, so a
# separator inside it does not shift the other fields.
FORMAT = "%H%x00%an%x00%ae%x00%cn%x00%ce%x00%B"


def _git(*args):
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    ).stdout


def commits(base, head):
    """Return ``(sha, author name, author address, committer name,
    committer address, message)`` for each non-merge commit of
    ``base..head``."""
    shas = _git("rev-list", "--no-merges", f"{base}..{head}").split()
    return [
        tuple(_git("show", "-s", "--format=" + FORMAT, sha).split("\x00", 5))
        for sha in shas
    ]


def signed_off(message, addresses):
    """Return whether a sign-off line of ``message`` names an address of
    ``addresses`` (compared in lower case)."""
    for line in message.split("\n"):
        match = SIGNOFF.match(line)
        if match and match.group(1).lower() in addresses:
            return True
    return False


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    base, head = argv
    try:
        found = commits(base, head)
    except subprocess.CalledProcessError as error:
        print(error.stderr.strip(), file=sys.stderr)
        return 2
    failed = 0
    for sha, name, author, committer_name, committer, message in found:
        subject = message.split("\n", 1)[0]
        if (name, author, committer_name, committer) in EXEMPT_IDENTITIES:
            print(f"exempt   {sha[:12]} {subject} ({name})")
        elif signed_off(message, {author.lower(), committer.lower()}):
            print(f"ok       {sha[:12]} {subject}")
        else:
            failed += 1
            print(
                f"missing  {sha[:12]} {subject}: no Signed-off-by line"
                f" for {author} or {committer}"
            )
    print(
        f"{len(found)} commit(s) checked; {failed} of {len(found)} without"
        " a sign-off."
    )
    if failed:
        print(
            "Add one with: git commit --amend -s, or git rebase --signoff"
            " for several commits, then push again."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
