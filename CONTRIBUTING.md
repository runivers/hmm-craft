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

3. Keep every Python line under 80 characters, as upstream does.
4. Sign off every commit with `git commit -s`. The line `Signed-off-by: Your
   Name <you@example.com>` certifies the
   [Developer Certificate of Origin](https://developercertificate.org/): you
   wrote the change or have the right to submit it under the project's licence.
5. Whoever opens a pull request answers for every line of it: can explain it,
   has run the tests it names, and certifies the right to submit it by the
   sign-off. How a change was written is not asked.

Your contribution is licensed under the project's BSD 3-Clause licence
(`LICENSE.txt`).

## How a change is merged

Every change goes through a pull request. The maintainer merges it once the
required checks pass and the maintainer has read it. No second approval is
required. Nobody pushes to `main` directly.

## Conduct

Everyone taking part follows `CODE_OF_CONDUCT.md`.
