# How a release is built

A release is a tag. Nothing is built or uploaded from a maintainer's machine, and
no credential is stored in the repository or its settings: PyPI
accepts the upload because it trusts the `release.yml` workflow of this
repository, through [trusted publishing](https://docs.pypi.org/trusted-publishers/).

## What runs, and when

| Workflow | Runs on | Does |
|---|---|---|
| `ci.yml` | pull request, push to `main`, manual, weekly | The style rule (no Python line of 80 characters or more), then the suite and the bit-equality check against hmmlearn 0.3.3 in two cells: `floor` (the declared minimum versions) and `latest` (the newest wheels). On a pull request, `sign-off` checks that every commit except a merge carries a `Signed-off-by` line with its author's or its committer's address (`scripts/check_signoff.py`, the base branch's copy); Dependabot's commits are exempt. The weekly run shows a new release of NumPy, SciPy or scikit-learn that breaks the suite, or under which the fits differ from hmmlearn 0.3.3's; both packages are installed at the same versions, so a release that changes both fits alike is not seen. |
| `wheels.yml` | pull request, manual, weekly, and called by `release.yml` | cibuildwheel builds the wheels and tests each one from the installed wheel; one job builds the sdist, installs it and runs the suite. A pull request builds the Linux x86_64 wheels only; the other runs build all 20. |
| `release.yml` | a pushed tag `v<version>` | Calls `wheels.yml` for everything, then publishes after the maintainer approves the `pypi` environment. |

The wheels are CPython 3.12 to 3.15 (GIL builds) for Linux x86_64 and aarch64
(`manylinux_2_28`, built on native runners), macOS arm64 and x86_64, and Windows
AMD64: 20 wheels and the sdist. There are no musllinux, free-threaded or 32-bit
wheels. No cell skips its tests: a cell that cannot run them ships no wheel.

GitHub stops the weekly runs of a public repository after 60 days without
activity; the maintainer turns them back on.

The bit-equality check, and what it compares, is described in `DIFFERENCES.md`.
`scripts/ci_bit_equality.sh floor <directory>` (or `latest`) runs the same cell
on a machine with Python 3.12 and a network connection.

## Pinned inputs

- **Build requirements** are in `ci/build_requirements.txt`, pinned with hashes
  and installed before every build, which then runs without build isolation.
  `ci/tool_requirements.txt` pins `build`, which makes the sdist. Both are
  regenerated from their `.in` files with the command in their first lines.
- **The floor** is `ci/floor_constraints.txt`. The `floor` cell installs exactly
  those versions from binary wheels and checks that `numpy`, `scipy` and
  `scikit-learn` are the declared minimums. `scripts/test_floor_constraints.py`
  fails if a minimum in `pyproject.toml`, the file's pin or a build requirement
  drifts apart.
- **Actions** are pinned by commit, with the release in a comment. Dependabot
  opens one pull request a month with the newer releases
  (`.github/dependabot.yml`); the maintainer reads their changes before merging
  it. Dependabot does not touch the build requirements, which are updated by
  hand.

## Making a release

1. Move the entries under "Unreleased" in `CHANGES.rst` to a heading for the
   version, in a commit on `main`, and wait for its checks to pass.
2. Tag that commit `v<version>` (for example `v0.4.0`) and push the tag. The
   version of every file comes from the tag, and the publish job refuses files
   that carry another one, or a count other than 20 wheels and one sdist.
3. Approve the `pypi` environment when the build jobs have passed. The upload
   follows; a file uploaded to PyPI cannot be replaced, so a mistake is corrected
   by the next version.

The PyPI project needs a trusted publisher for this repository, the workflow
`release.yml` and the environment `pypi`; the environment needs the maintainer
as its required reviewer.

macOS x86_64 is the first platform dropped when its cell breaks or GitHub retires
the runner.
