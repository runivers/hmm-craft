hmm-craft
=========

hmm-craft is a fork of hmmlearn_, a set of algorithms for **unsupervised**
learning and inference of Hidden Markov Models. It is an independent project
and is not endorsed by the hmmlearn maintainers. The licence and the list of
authors are hmmlearn's, kept, with a line for this fork added.

.. _hmmlearn: https://github.com/hmmlearn/hmmlearn

For supervised learning of HMMs and similar models see seqlearn_.

.. _seqlearn: https://github.com/larsmans/seqlearn

The names
=========

The distribution is ``hmm-craft``. The package you import is ``hmmcraft``::

    pip install hmm-craft

and in Python::

    from hmmcraft import hmm

``pip install hmmcraft`` finds nothing: the hyphen belongs to the
distribution name only.

hmm-craft and hmmlearn share no file, so both can be installed in one
environment and imported in one process. A package that requires hmmlearn by
name keeps using hmmlearn.

To move code over from hmmlearn, change the import. ``from hmmlearn import
hmm`` becomes ``from hmmcraft import hmm``.

Stored models
=============

A file pickled or dumped with joblib by hmmlearn records module paths that
begin with ``hmmlearn.``. It does not load under hmm-craft: Python raises
``ModuleNotFoundError``. Loading it needs a mapping from ``hmmlearn.*`` to
``hmmcraft.*`` for the length of the load, and hmm-craft does not ship one.

A file saved with hmm-craft records ``hmmcraft.*`` and does not load in an
environment that has only hmmlearn.

What differs from hmmlearn 0.3.3
================================

* The package and distribution names above.
* The supported versions: Python 3.12 or newer, NumPy 1.26.0, SciPy 1.11.2,
  scikit-learn 1.3.1 and threadpoolctl 3.5.0, or newer. hmmlearn 0.3.3
  declares older minimums (NumPy 1.10, SciPy 0.19, scikit-learn 0.16) that
  cannot be installed on any Python it supports.
* The covariances of a spherical ``GaussianHMM``: ``covars_`` has one matrix
  per state, as documented, and ``sample()`` draws each state with its own
  variance.
* ``decode(algorithm="map")`` returns the sum of the log of the posterior
  probability of each chosen state, not the sum of the probabilities.
* ``state_occupancy_``: after ``fit``, the expected number of samples in each
  state, for every model.
* ``GaussianHMM(covars_floor=...)``: a floor on the covariances after every
  M-step. The default, ``None``, applies none and gives hmmlearn 0.3.3's fit.
* A ``GaussianHMM`` that updates covariances but not means (``params``
  containing ``"c"`` without ``"m"``) now estimates its covariances. What
  hmmlearn 0.3.3 returned there was not an estimate, or raised; the exact
  result, for the default priors, is in ``DIFFERENCES.md``. This and the next
  two items are the three changes to a fit made without an option.
* The ``full`` and ``tied`` Gaussian log-densities no longer depend on where
  memory is placed. With NumPy before 2.0.2 on a CPU with AVX-512, hmmlearn
  0.3.3's last bits there could depend on it; this package's can differ from
  them. ``DIFFERENCES.md`` gives the details, and which of them were observed.
* The k-means that starts a fit runs on one thread. It starts the means of
  ``GaussianHMM`` and ``GMMHMM`` and the posterior parameters of
  ``VariationalGaussianHMM``. With two or more threads, hmmlearn 0.3.3's
  start for a fixed ``random_state`` can depend on the number of threads and
  on the order in which they finish; this package gives the one-thread
  start, so a fit from it can differ from hmmlearn 0.3.3's. ``DIFFERENCES.md``
  names the thread pools this covers and the test that compares the starts,
  with its data, versions and thread counts.
* ``normalize(a, axis=0)`` gives zeros, not NaN, for a slice that sums to
  zero.
* ``lengths`` must be a one-dimensional array of positive integers; anything
  else raises ``ValueError``, including a two-dimensional array and a single
  integer, which hmmlearn 0.3.3 accepted. hmmlearn 0.3.3 also accepted zero
  and negative entries when the total matched; a zero entry, and some negative
  ones, gave its kernel a sequence without samples, and it then indexed outside
  its arrays.

Apart from these, the fits of this package are checked to equal hmmlearn
0.3.3's to the bit, on the synthetic data, with the versions, the seeds and the
single thread given in the result table of ``DIFFERENCES.md``; nothing is
claimed beyond that check. A change to a fit that hmmlearn 0.3.3 computes
correctly and repeatably is made only behind an option whose default is
hmmlearn 0.3.3's behaviour. The k-means start above is the exception: with
two or more threads it changes without an option, whether or not hmmlearn
0.3.3 computes it repeatably there. ``DIFFERENCES.md`` lists each difference
and describes the check that compares the two packages. Every change is
listed in ``CHANGES.rst``.

Installation
============

Requires Python 3.12 or newer and, to build from source, a C++ compiler.

To install from a checkout of this repository::

    pip install .

The tutorial, the API reference and the examples are in ``doc/`` and
``examples/``. There is no hosted documentation site.
``ARCHITECTURE.md`` describes how the package is built, from the Python API
to the C++ kernel.

Maintenance
===========

This fork has one maintainer and exists first for that maintainer's own use. It
is published so that others can install the same fixes. Issues and pull
requests are read as time allows; there is no response time. A bug with a short
example that runs as posted is looked at first. A feature request without a
pull request may be closed. Questions about using hidden Markov models are not
answered here.

If maintenance stops, the repository is archived on GitHub and the README
says so. The published releases stay on PyPI, and the ``hmm-craft`` name there
is not handed to anyone the maintainer does not know.

See ``CONTRIBUTING.md`` before opening an issue or a pull request,
``SECURITY.md`` to report a security problem, and ``CODE_OF_CONDUCT.md`` for
the rules of the project's spaces.

Licence
=======

BSD 3-Clause, see ``LICENSE.txt``. Copyright belongs to the hmmlearn authors
and contributors and, for the changes made here, to the hmm-craft authors and
contributors, both listed in ``AUTHORS.rst``.
