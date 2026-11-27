"""
Probe what hmmlearn 0.3.3 computes for a GaussianHMM that updates
covariances but not means (``params="c"``).

    python -I scripts/probe_covariance_only_fit.py [--package hmmlearn]

The state mass is not accumulated when ``"m"`` is not in ``params``, so the
M-step divides by the floor of 1e-5. Each clause below is checked on small
synthetic data (a seeded generator, two sequences) with a fixed start and
printed as ``ok`` or ``FAILED``. The exit status is 0 if every clause holds
as described, which is the case for hmmlearn 0.3.3, and 1 otherwise.

Clauses:

1. With the default ``covars_weight=1`` and ``means_weight=0``, for
   ``"spherical"`` and ``"diag"`` the covariances after an M-step equal, as
   bytes, (``covars_prior`` + the posterior-weighted sum of squares) / 1e-5;
   for ``"spherical"`` averaged over the features.
2. With ``covars_weight`` above 1 the divisor is ``covars_weight - 1``.
3. With ``means_weight`` not 0 the numerator adds
   ``means_weight * (means - means_prior) ** 2``.
4. With the defaults, ``"full"`` and ``"tied"`` have infinite covariances
   after the first M-step and raise ``ValueError`` from the second iteration.
5. They raise whenever ``covars_weight`` is at most ``n_features``, and fit
   with finite covariances when it is larger.

The probe is not part of the bit-equality check. It uses numpy and the
package under test only.
"""

import argparse
import importlib
import logging
import sys
import warnings

import numpy as np

N_FEATURES = 2
LENGTHS = (150, 150)
START_MEANS = np.array([[0.0, 0.0], [3.0, 3.0]])
COVARS_PRIOR = 1e-2  # the default of GaussianHMM


def make_data():
    rs = np.random.RandomState(10)
    first = rs.normal(0, 1, (LENGTHS[0], N_FEATURES))
    second = rs.normal(3, 0.5, (LENGTHS[1], N_FEATURES))
    return [first, second]


def make_model(hmm, covariance_type, n_iter, **kwargs):
    model = hmm.GaussianHMM(
        2,
        covariance_type,
        params="c",
        init_params="",
        n_iter=n_iter,
        tol=-np.inf,
        **kwargs,
    )
    model.startprob_ = np.array([0.5, 0.5])
    model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])
    model.means_ = START_MEANS.copy()
    model.covars_ = {
        "diag": np.ones((2, N_FEATURES)),
        "spherical": np.ones(2),
        "full": np.tile(np.eye(N_FEATURES), (2, 1, 1)),
        "tied": np.eye(N_FEATURES),
    }[covariance_type]
    return model


def first_m_step(hmm, covariance_type, sequences, **kwargs):
    """
    Fit one iteration. Return the covariances after the M-step, the
    posterior-weighted sum of squares computed here, outside the model (one
    sequence at a time, as the model accumulates it), and the state mass the
    M-step was given.
    """
    model = make_model(hmm, covariance_type, 1, **kwargs)
    weighted = 0
    for part in sequences:
        weighted = weighted + model.predict_proba(part).T @ part**2
    seen = {}
    do_mstep = model._do_mstep

    def spy(stats):
        seen["post"] = np.copy(stats["post"])
        do_mstep(stats)

    model._do_mstep = spy
    model.fit(np.concatenate(sequences), list(LENGTHS))
    return model._covars_, weighted, seen["post"]


def expected(covariance_type, weighted, divisor, extra=0.0):
    value = (COVARS_PRIOR + extra + weighted) / divisor
    if covariance_type == "spherical":
        return np.tile(value.mean(axis=1)[:, None], (1, N_FEATURES))
    return value


def fit_outcome(hmm, covariance_type, sequences, n_iter, **kwargs):
    """Return ``"raised"`` or the fitted covariances."""
    model = make_model(hmm, covariance_type, n_iter, **kwargs)
    try:
        model.fit(np.concatenate(sequences), list(LENGTHS))
    except ValueError:
        return "raised"
    return np.asarray(model._covars_)


def check(results, text, holds):
    results.append(bool(holds))
    print(f"{'ok' if holds else 'FAILED':6} {text}")


def probe(hmm):
    sequences = make_data()
    results = []
    spherical_diag = ("diag", "spherical")

    print("1. defaults: (covars_prior + weighted sum of squares) / 1e-5")
    for covariance_type in spherical_diag:
        got, weighted, mass = first_m_step(hmm, covariance_type, sequences)
        check(
            results,
            f"{covariance_type}: equal as bytes to the formula",
            got.tobytes()
            == expected(covariance_type, weighted, 1e-5).tobytes(),
        )
        check(
            results,
            f"{covariance_type}: the state mass given was zero",
            not mass.any(),
        )

    print("2. covars_weight above 1: the divisor is covars_weight - 1")
    for covariance_type in spherical_diag:
        for weight in (3.0, 10.0):
            got, weighted, _ = first_m_step(
                hmm, covariance_type, sequences, covars_weight=weight
            )
            check(
                results,
                f"{covariance_type}, covars_weight={weight}: formula with "
                "divisor covars_weight - 1; the 1e-5 formula does not hold",
                got.tobytes()
                == expected(covariance_type, weighted, weight - 1).tobytes()
                and got.tobytes()
                != expected(covariance_type, weighted, 1e-5).tobytes(),
            )

    print("3. means_weight not 0: the numerator adds a term")
    extra = 0.5 * (START_MEANS - 0.0) ** 2
    for covariance_type in spherical_diag:
        got, weighted, _ = first_m_step(
            hmm, covariance_type, sequences, means_weight=0.5
        )
        check(
            results,
            f"{covariance_type}, means_weight=0.5: formula plus "
            "means_weight * (means - means_prior)**2; without it it fails",
            got.tobytes()
            == expected(covariance_type, weighted, 1e-5, extra).tobytes()
            and got.tobytes()
            != expected(covariance_type, weighted, 1e-5).tobytes(),
        )

    print("4. full and tied, defaults: infinite after one M-step, raise after")
    for covariance_type in ("full", "tied"):
        one = fit_outcome(hmm, covariance_type, sequences, 1)
        check(
            results,
            f"{covariance_type}, n_iter=1: no error, covariances infinite",
            not isinstance(one, str)
            and np.isinf(one).any()
            and not np.isnan(one).any(),
        )
        for n_iter in (2, 5, 10):
            check(
                results,
                f"{covariance_type}, n_iter={n_iter}: ValueError",
                isinstance(
                    fit_outcome(hmm, covariance_type, sequences, n_iter), str
                ),
            )

    print("5. full and tied: raise when covars_weight <= n_features, else fit")
    for covariance_type in ("full", "tied"):
        for weight, raises in (
            (1.0, True),
            (2.0, True),
            (3.0, False),
            (10.0, False),
        ):
            out = fit_outcome(
                hmm, covariance_type, sequences, 5, covars_weight=weight
            )
            holds = (
                isinstance(out, str)
                if raises
                else not isinstance(out, str) and np.isfinite(out).all()
            )
            check(
                results,
                f"{covariance_type}, covars_weight={weight}: "
                f"{'ValueError' if raises else 'fits, finite'}",
                holds,
            )
        out = fit_outcome(hmm, covariance_type, sequences, 5, means_weight=0.5)
        check(
            results,
            f"{covariance_type}, means_weight=0.5: ValueError",
            isinstance(out, str),
        )
    return all(results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--package",
        default="hmmlearn",
        choices=("hmmlearn", "hmmcraft"),
        help="the package to probe (default hmmlearn)",
    )
    args = parser.parse_args(argv)
    warnings.simplefilter("ignore")
    logging.disable(logging.CRITICAL)
    package = importlib.import_module(args.package)
    version = getattr(package, "__version__", "unknown")
    print(f"{args.package} {version}, numpy {np.__version__}")
    holds = probe(importlib.import_module(f"{args.package}.hmm"))
    print("every clause holds" if holds else "a clause FAILED")
    return 0 if holds else 1


if __name__ == "__main__":
    sys.exit(main())
