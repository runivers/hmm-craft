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
