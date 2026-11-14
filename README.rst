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
* The supported versions: Python 3.12 or newer, NumPy 1.26.0, SciPy 1.11.2 and
  scikit-learn 1.3.1, or newer. hmmlearn 0.3.3 declares older minimums
  (NumPy 1.10, SciPy 0.19, scikit-learn 0.16) that cannot be installed on any
  Python it supports.
* Nothing else yet. A change that alters the result of a fit is made only
  behind an option whose default is hmmlearn 0.3.3's behaviour.

Every change is listed in ``CHANGES.rst``.

Installation
============

Requires Python 3.12 or newer and, to build from source, a C++ compiler.

To install from a checkout of this repository::

    pip install .

The tutorial, the API reference and the examples are in ``doc/`` and
``examples/``. There is no hosted documentation site.

Maintenance
===========

This fork has one maintainer and exists first for that maintainer's own use. It
is published so that others can install the same fixes. Issues and pull
requests are read as time allows; there is no response time. A bug with a short
example that runs as posted is looked at first. A feature request without a
pull request may be closed. Questions about using hidden Markov models are not
answered here.

See ``CONTRIBUTING.md`` before opening an issue or a pull request, and
``CODE_OF_CONDUCT.md`` for the rules of the project's spaces.

Licence
=======

BSD 3-Clause, see ``LICENSE.txt``. Copyright belongs to the hmmlearn authors
and contributors and, for the changes made here, to the hmm-craft authors and
contributors, both listed in ``AUTHORS.rst``.
