# How a release is built

A release is a tag. Nothing is built or uploaded from a maintainer's machine, and
no credential is stored in the repository or its settings: PyPI
accepts the upload because it trusts the `release.yml` workflow of this
repository, through [trusted publishing](https://docs.pypi.org/trusted-publishers/).

## What runs, and when

| Workflow | Runs on | Does |
|---|---|---|
| `ci.yml` | pull request, push to `main`, manual, weekly | The style rules (no Python line of 80 characters or more, `ruff check`, no style finding added), `types (pyright)` (no type error added), `kernel warnings (<cell>)` (both modules of the C++ kernel compiled with warnings as errors by GCC, clang, Apple clang and MSVC, each the runner image's compiler, and a planted warning that must fail the build), `docs (sphinx)` (the documentation built with Sphinx, the gallery's examples run, any warning an error), then the suite and the bit-equality check against hmmlearn 0.3.3 in two cells: `floor` (the declared minimum versions) and `latest` (the newest wheels). On a pull request, `sign-off` checks that every commit except a merge carries a `Signed-off-by` line with its author's or its committer's address (`scripts/check_signoff.py`, the base branch's copy); Dependabot's commits are exempt. The weekly run shows a new release of NumPy, SciPy or scikit-learn that breaks the suite, or under which the fits differ from hmmlearn 0.3.3's; both packages are installed at the same versions, so a release that changes both fits alike is not seen. A manual run also starts `contraction-arm`, a measurement that gates nothing: on Linux aarch64 and macOS arm64, `scripts/measure_contraction.sh` builds hmmlearn 0.3.3 with and without `-ffp-contract=off` and this package, compares their results and uploads them. |
| `instrumented.yml` | pull request, push to `main`, manual, weekly | Three jobs, each with the newest NumPy, SciPy and scikit-learn wheels; nothing they build is published. `coverage` builds the C++ kernel with `--coverage` and without optimisation, runs the suite under coverage.py, and reports line and branch coverage of the Python package (tests left out) and of `ext/` (gcovr). It fails when the Python total or the kernel's line coverage is below its floor in `pyproject.toml` (`[tool.coverage.report]`, `[tool.gcovr]`); the kernel's branch coverage has no floor. On a pull request it also fails when fewer than 90% of the lines the pull request adds or changes run, or fewer than 85% of their branches are taken, the Python package and the kernel each on its own (`scripts/check_diff_coverage.py`, the base branch's copy). A floor is raised by hand and lowered only in a commit that says why. Reports: the log and the run's `coverage-reports` artifact. `sanitizers` builds the kernel with AddressSanitizer and UndefinedBehaviorSanitizer and runs the suite; an invalid memory access or undefined behaviour in the kernel that the sanitizers detect stops the suite and fails the job, with the report in the log. Floating-point checks are off: the kernel takes the logarithm of zero on purpose. `thread-sanitizer` builds the kernel with ThreadSanitizer, which cannot be combined with AddressSanitizer in one build, and runs the tests of the kernel's threads (`src/hmmcraft/tests/test_kernel_threads.py`); a data race between the threads that ThreadSanitizer detects stops the run and fails the job, with the report in the log. |
| `wheels.yml` | pull request, manual, weekly, and called by `release.yml` | cibuildwheel builds the wheels and tests each one from the installed wheel; one job builds the sdist, installs it and runs the suite. A pull request builds the Linux x86_64 wheels only; the other runs build all 20. |
| `release.yml` | a pushed tag `v<version>` | Calls `wheels.yml` for everything, then publishes after the maintainer approves the `pypi` environment. |
| `pins.yml` | manual, weekly | `scripts/check_pins.py` compares each pin of `ci/build_requirements.txt`, `ci/tool_requirements.txt`, `ci/coverage_requirements.txt`, `ci/lint_requirements.txt`, `ci/type_requirements.txt` and `ci/docs_requirements.txt` with the newest final, unyanked release on PyPI and fails naming each older pin, with the command that updates it; `ci/floor_constraints.txt` is not read. It does not run on a pull request, which a release elsewhere must not turn red; its unit tests run in `ci.yml`. |

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
  `ci/tool_requirements.txt` pins `build`, which makes the sdist.
  `ci/coverage_requirements.txt` pins coverage.py and gcovr for the `coverage`
  job; neither the release build nor the sdist job installs it.
  `ci/lint_requirements.txt` pins ruff for the `style` job.
  `ci/type_requirements.txt` pins pyright, the node it runs on, and the
  numeric stack and pytest whose types it reads, for the `types` job.
  `ci/docs_requirements.txt` pins Sphinx, its extensions, matplotlib and the
  package's dependencies for the `docs` job. It is compiled for Python 3.13,
  which the newest Sphinx requires, and the job runs on 3.13; a Sphinx that
  needs a newer Python moves the lock's `--python-version` and the job's
  Python together (the docs job's Python is a tool environment, not the
  package's floor). All six are regenerated from their `.in` files with the
  command in their first lines. The command's `--no-header` drops those
  lines, so they are put back by hand.
  It keeps the versions already in the file, so a pin that is not in the `.in`
  file moves only when `--upgrade-package <name>` is added. `pins.yml` fails
  each week while a pin of any of them is older than the newest release on
  PyPI, and prints the command that updates it. A pin that another pin caps
  is compared with the newest release below the cap, and reported as held
  while a newer release exists above it (`HOLDS` in `scripts/check_pins.py`,
  each with its reason). A hold fails the check once the capping pin moves
  or the held pin reaches the cap, until its cap is read again.
- **The documentation** is built in `ci.yml`'s `docs` job with `-W`. The job
  downloads scikit-learn's Sphinx inventory with retries to
  `doc/source/sklearn-objects.inv` (untracked) before the build; `conf.py`
  reads that copy first and the site second.
- **Style checks** run in `ci.yml`'s `style` job with the pinned ruff.
  `ruff check .` reads `[tool.ruff]` in `pyproject.toml` (pyflakes and syntax
  errors, which must stay at zero). `scripts/check_ruff_baseline.py` counts the
  findings of a wider, fixed list of rules (the ones ruff 0.17.5 selects by
  default, written out in the script) per file and rule, and compares them
  with `ci/ruff_baseline.json`: a count that rises, or a new file or rule,
  fails; a count that falls fails too until the baseline is lowered. After
  fixing findings, lower it with
  `python3 -I scripts/check_ruff_baseline.py --update`, which refuses while
  any count is above the baseline, and commit the file. The baseline is empty:
  no file has a finding, so any finding fails the check. The list does not
  follow ruff's defaults, so a ruff upgrade adds no rule; it can still find
  more under a listed rule. The baseline never rises: the commit that moves
  to a newer ruff fixes what it finds, or the pin stays.
- **Type check** runs in `ci.yml`'s `types` job, not in `style`. pyright's
  basic mode runs over `src/hmmcraft` (`[tool.pyright]` in `pyproject.toml`)
  from the hashed `ci/type_requirements.txt`, which holds pyright with its node
  (`nodejs-wheel-binaries`); nothing is downloaded when it runs.
  `scripts/check_pyright_baseline.py` first compares each pin of the lock with
  the installed version, then counts the errors per file and rule and compares
  them with `ci/pyright_baseline.json`: a count that rises, or a new file or
  rule, fails; a count that falls fails too until the baseline is lowered with
  `python -I scripts/check_pyright_baseline.py --update`, which refuses while
  any count is above the baseline. The numeric stack and pytest are pinned so
  that a release elsewhere does not move the count. A newer pyright or numpy
  stub can report errors that no change to the code removes, so
  `--update --allow-rise` may raise the baseline, in a commit that changes
  `ci/type_requirements.txt` and the baseline and nothing else; the diff of
  the baseline shows each rise.
- **The floor** is `ci/floor_constraints.txt`. The `floor` cell installs exactly
  those versions from binary wheels and checks that `numpy`, `scipy`,
  `scikit-learn` and `threadpoolctl` are the declared minimums.
  `scripts/test_floor_constraints.py` fails if a minimum in `pyproject.toml`,
  the file's pin or a build requirement drifts apart. The cell also turns
  numpy's AVX-512 loops off (`NPY_DISABLE_CPU_FEATURES`, see `DIFFERENCES.md`).
  Once the numpy floor is 2.0.2 or later, that setting is removed from
  `scripts/ci_bit_equality.sh`.
- **Actions** are pinned by commit, with the release in a comment. Dependabot
  opens one pull request a month with the newer releases
  (`.github/dependabot.yml`); the maintainer reads their changes before merging
  it. Dependabot does not touch the build requirements, which are updated by
  hand.

## Version numbers

Versions are `MAJOR.MINOR.PATCH`. The first release of hmm-craft is `0.4.0`:
hmmlearn's last release is `0.3.3`, and `0.3.4` would read as a patch of it.

- A **patch** release (`0.4.1`) changes no fit and adds no public name: it
  fixes a crash, a wrong error, the packaging or the documentation. It may
  also make a fit that depended on where an array was placed in memory come
  out the same on every run; that change is named in `CHANGES.rst` and
  `DIFFERENCES.md`.
- A **minor** release (`0.5.0`) may add an option, a parameter or a function,
  or change a fit. Each changed fit is named in `CHANGES.rst` and
  `DIFFERENCES.md`.

Whatever the version of hmm-craft, the fits are compared with hmmlearn 0.3.3's.

## Deprecations

A public name, a parameter or a default is removed or changed only after one
minor release in which using it raises a `DeprecationWarning` that says what
replaces it, with a line in `CHANGES.rst`. A model saved with pickle or joblib
by one minor release is not promised to load in another.

## Supported versions

A Python version is supported until its end of life and dropped in the next
minor release after it; Python 3.12 reaches its end of life in October 2028.
The minimum versions of NumPy, SciPy, scikit-learn and threadpoolctl rise only
then, or when a fix needs an API that the minimum does not have. A new CPython
version gets wheels once NumPy, SciPy and scikit-learn all publish wheels for
it.

## Making a release

1. Move the entries under "Unreleased" in `CHANGES.rst` to a heading for the
   version, in a commit on `main`, and wait for its checks to pass.
2. Run `wheels.yml` by hand on `main` while that commit is its head, and wait
   until every cell has passed. Only then is the commit tagged.
3. Tag that commit `v<version>` (for example `v0.4.0`) and push the tag. The
   version of every file comes from the tag, and the publish job refuses files
   that carry another one, or a count other than 20 wheels and one sdist.
4. Approve the `pypi` environment when the build jobs have passed. The upload
   follows; a file uploaded to PyPI cannot be replaced, so a mistake is corrected
   by the next version. A release found to be broken is yanked on PyPI, never
   deleted, so that an install pinned to it still finds it.

The PyPI project needs a trusted publisher for this repository, the workflow
`release.yml` and the environment `pypi`; the environment needs the maintainer
as its required reviewer.

## Where a file came from

The publish step uploads each file with an attestation (PEP 740), made through
trusted publishing, that names this repository and the workflow run that built
the file. PyPI shows it on the file's page. Tags are not signed: the attestation
is the record of where a published file came from.

macOS x86_64 is the first platform dropped when its cell breaks or GitHub retires
the runner.
