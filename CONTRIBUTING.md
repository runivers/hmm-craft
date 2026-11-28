# Contributing to hmm-craft

hmm-craft is a fork of [hmmlearn](https://github.com/hmmlearn/hmmlearn) with one
maintainer. It exists first for that maintainer's own use and is published so
that others can install the same fixes. Issues and pull requests are read as
time allows; there is no response time.

## Reporting a problem

Open an issue with a short example that runs as posted, in a new empty file.
Generate the data with a seeded random number generator inside the example, and
do not attach data files. State the versions of hmm-craft, Python, NumPy, SciPy
and scikit-learn, and whether the same example also fails on hmmlearn 0.3.3.
An issue without an example is closed.

A feature request without a pull request may be closed. Questions about using
hidden Markov models are not answered here.

Report a security problem as described in `SECURITY.md`, not in an issue.

## Sending a pull request

1. Describe the change and say whether it changes the result of a fit. Fits are
   checked to equal hmmlearn 0.3.3's to the bit, apart from the differences
   listed in `DIFFERENCES.md`. A change to a fit outside that list is accepted
   only behind an option whose default is hmmlearn 0.3.3's behaviour. Add a
   test that fails without the change.
2. Build and run the tests from a checkout:

       pip install ".[tests]"
       python -m pytest --pyargs hmmcraft.tests doc

   Each test run seeds the global random number generators of NumPy and of
   Python's `random` module and prints the seed before the tests
   (`set RNG seed to <n>`). The seed is random unless `HMMCRAFT_SEED=<n>` is
   set, which reruns the same tests with that seed; CI sets `HMMCRAFT_SEED=0`.

   CI also measures how much of the Python code and of the C++ kernel the
   tests run, and fails below the floors in `pyproject.toml`. To see the
   Python figure:

       pip install coverage
       python -m coverage run -m pytest --pyargs hmmcraft.tests doc
       python -m coverage report

   On a pull request CI also measures the lines the pull request adds or
   changes, and their branches, in the Python package and in the C++ kernel
   (`ext/`). It fails when fewer than 90% of the added lines that a report
   measures run, or fewer than 85% of their branches are taken, the Python
   package and the kernel each on its own; blank lines, comments, docstrings
   and deleted lines are not measured and do not count, and a pull request
   that adds no measured line passes. A line the pull request changes counts
   as added, also when only its style changed. To run the same
   check, with the suite already run under coverage.py and the kernel built
   with `--coverage` as `.github/workflows/instrumented.yml` does:

       python -m coverage json --fail-under=0 -o python.json
       python -m gcovr --root . --fail-under-line 0 --json kernel.json
       pkg=$(python -c "import hmmcraft, os; \
           print(os.path.dirname(hmmcraft.__file__))")
       python scripts/check_diff_coverage.py \
           "$(git merge-base main HEAD)" HEAD \
           --python python.json --map "$pkg=src/hmmcraft" \
           --kernel kernel.json

   Code that no test can reach may be excluded with the tools' own markers:
   `# pragma: no cover` (a line) and `# pragma: no branch` (a branch) for
   coverage.py, `GCOVR_EXCL_LINE`, `GCOVR_EXCL_START` / `GCOVR_EXCL_STOP` (a
   range) and `GCOVR_EXCL_BR_LINE` (a line's branches) for gcovr. They are for
   unreachable code only; code that is hard to test gets a test. Each use
   gives its reason on the marker's line or on the comment line above it. A
   `# pragma: no cover` on a line that opens a block excludes the whole
   block. An excluded line counts nowhere, the totals included; the check
   prints each excluded line a pull request adds.

3. Keep every Python line under 80 characters, as upstream does. CI also runs
   `ruff check .` (pyflakes, syntax errors, a docstring on every public
   module, class and function of the package, and ruff's bugbear, numpy,
   performance, return, simplification, comprehension, pie, ruff and
   pyupgrade rules), `ruff format --check .` (the formatter's style) and a
   check that no other style finding is added to a file. With the ruff
   version CI uses
   (`pip install --require-hashes -r ci/lint_requirements.txt`; another
   version can count or format differently):

       ruff check .
       ruff format --check .
       python3 -I scripts/check_ruff_baseline.py

   All three must pass. `ruff format .` fixes a format failure. The format
   check also covers Python code blocks in Markdown files. The last check
   fails on any finding of its rules, as `ci/ruff_baseline.json` is empty: fix
   the finding; the baseline does not rise. A finding kept on purpose is
   ignored for its file in `pyproject.toml`, with its reason in a comment
   above the entry: a `noqa` comment for a rule outside the last check's set
   is itself a finding of that check (`RUF100`), and a `noqa` also needs its
   code in `select`.

   The formatting commit is listed in `.git-blame-ignore-revs`; `git config
   blame.ignoreRevsFile .git-blame-ignore-revs` makes local `git blame` skip
   it.

   CI also runs pyright, a type checker, in its basic mode over
   `src/hmmcraft` (`[tool.pyright]` in `pyproject.toml`) and fails when a
   file has more errors of a rule than `ci/pyright_baseline.json` records, or
   fewer. The package has no type annotations, and the baseline holds the
   errors of the tree when the check was added. With the versions CI uses, in
   an environment of their own:

       pip install --require-hashes --only-binary :all: \
         -r ci/type_requirements.txt
       python -I scripts/check_pyright_baseline.py

   The script uses the Python that runs it, for pyright and for the packages
   it reads, and stops if that environment does not match the lock. A new
   error is fixed in the code; a `# type: ignore` comment is not used. After
   fixing errors, lower the baseline with
   `python -I scripts/check_pyright_baseline.py --update` and commit it.
   `--update --allow-rise` may raise it, only in a commit that changes
   `ci/type_requirements.txt` and the baseline and nothing else.

   CI also compiles both modules of the C++ kernel with warnings as errors,
   with GCC, clang, Apple clang (`-Wall -Wextra -Werror`) and MSVC (`-W4
   -WX`). Python's and pybind11's headers are passed as system headers, so a
   warning in them does not count. With GCC or clang, after `pip install
   --require-hashes -r ci/build_requirements.txt`:

       work=$(mktemp -d)
       pybind11=$(python -c 'import pybind11; print(pybind11.get_include())')
       include=$(python -c \
         'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY"))')
       CPPFLAGS="-isystem $pybind11 -isystem $include" \
         CXXFLAGS="-O2 -DNDEBUG -Wall -Wextra -Werror" \
         python setup.py build_ext \
         --build-temp "$work/temp" --build-lib "$work/lib"

   The objects go to a temporary directory, not `build/`, which a later `pip
   install .` would reuse. With MSVC, set
   `_CL_=-W4 -WX -external:anglebrackets -external:W0` instead, as the job
   does (cl takes `-` or `/` before an option). A warning is fixed in the
   code; where a fix would change what the kernel computes, it is silenced on
   its line with the reason in a comment, never by dropping a flag.

   CI also builds the documentation with Sphinx and fails on any warning; the
   examples in `examples/` run as part of the build. In an environment of its
   own, with Python 3.13, which the newest Sphinx requires:

       pip install --require-hashes --only-binary :all: \
         -r ci/build_requirements.txt
       pip install --no-build-isolation --no-deps .
       pip install --require-hashes --only-binary :all: \
         -r ci/docs_requirements.txt
       sphinx-build -W --keep-going -b html doc/source doc/build/html

   `-D plot_gallery=0` skips running the examples. The methods the models
   inherit from scikit-learn link to its glossary through its Sphinx
   inventory. CI downloads the inventory with retries to
   `doc/source/sklearn-objects.inv` before the build; a local build fetches it
   as before unless a copy is present there (the file is not tracked). A
   reference to a missing label, document or glossary term fails the build; a
   misspelt Python name in backquotes does not.
4. Sign off every commit with `git commit -s`. The line `Signed-off-by: Your
   Name <you@example.com>` certifies the
   [Developer Certificate of Origin](https://developercertificate.org/): you
   wrote the change or have the right to submit it under the project's licence.
   A check on every pull request fails if a commit other than a merge has no
   sign-off with its author's or its committer's address.
5. Whoever opens a pull request answers for every line of it: can explain it,
   has run the tests it names, and certifies the right to submit it by the
   sign-off. If an AI tool helped, say so as described in "Use of AI tools".

Your contribution is licensed under the project's BSD 3-Clause licence
(`LICENSE.txt`).

## Use of AI tools

Everyone says when an AI tool helped write a change, the maintainer
included.

A commit that an AI tool helped write ends with a trailer naming the tool,
placed before the sign-off:

    Assisted-by: Claude Code (Anthropic)

Name the tool, without a model name or version. Use this trailer instead of
the co-author line or the footer that some tools add. The pull request
template asks whether an AI tool helped with the pull request as a whole.

The maintainer's commits from ea4e856 to fff5707 were written with Claude
Code (Anthropic) before this rule existed, and carry no trailer. Every later
commit written with AI help carries it. Commits taken from pull requests to
hmmlearn keep their authors' messages as written.

## How a change is merged

Every change goes through a pull request. The maintainer merges it once the
required checks pass and the maintainer has read it. No second approval is
required. Nobody pushes to `main` directly.

## Conduct

Everyone taking part follows `CODE_OF_CONDUCT.md`.
