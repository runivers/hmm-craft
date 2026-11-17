"""
Check that a fit of this package is bit-equal to hmmlearn 0.3.3's.

The two packages cannot be imported together, so the check has two steps, run
in two Python environments that have the same Python, numpy, scipy,
scikit-learn and threadpoolctl versions:

    python -I scripts/bit_equality.py dump --package hmmlearn --out REF
    python -I scripts/bit_equality.py dump --package hmmcraft --out FORK \
        [--load-pickles REF]
    python -I scripts/bit_equality.py compare REF FORK [--allow E1,E2]

``dump`` fits a fixed list of models on synthetic data (a seeded generator,
no file read) and writes every fitted array, the score, both decodes, the
posteriors and a sample, one ``.npz`` per model. It then runs the underflow
cases: models whose scaled forward pass underflows on one sample, in a
prediction and in a fit with ``implementation="scaling"``. It refuses to run
unless the thread-count variables are all ``1``: where a start is seeded, the
k-means partial sums are added in the order the threads finish. ``compare``
checks the two dumps byte for byte. A difference is accepted only where the
table ``EXCEPTIONS`` names it, only if listed by ``--allow``, and only if it
has exactly the shape that entry describes. Exit status of ``compare``: 0
equal, 1 a difference, 2 the two dumps cannot be compared.

Only numpy, threadpoolctl and the package under test are used, so the script
runs under ``python -I`` in either environment.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
import pickle
import platform
import sys
import warnings
from collections import Counter, namedtuple
from pathlib import Path

import numpy as np

N_COMPONENTS = 3
N_ITER = 10
SEQUENCE_LENGTHS = (300, 200)
N_TRIALS = 5
MODEL_SEED = 0
SAMPLE_SEED = 1
N_SAMPLES = 50
COVARIANCE_TYPES = ("spherical", "diag", "full", "tied")
IMPLEMENTATIONS = ("log", "scaling")
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
DISTRIBUTIONS = {"hmmlearn": "hmmlearn", "hmmcraft": "hmm-craft"}

# id -> (item, what it excuses). The rules are in ``_RULES`` below.
EXCEPTIONS = {
    "E1": ("hmmlearn pull request 589",
           "covars_ of a spherical GaussianHMM has one matrix per state"),
    "E2": ("hmmlearn pull request 589",
           ("sample() of a spherical GaussianHMM uses the variance of the "
           "state drawn")),
    "E3": ("hmmlearn issue 188",
           "the log-probability of decode(algorithm='map')"),
    "E4": ("hmmlearn pull request 573",
           "a GaussianHMM that updates covariances but not means fits"),
    "E5": ("hmmlearn pull request 584",
           "normalize(axis=0) of a zero-sum slice is 0, not NaN"),
    "E6": ("fork additions",
           "state_occupancy_ and the covars_floor parameter"),
    "E7": (
        "hmm-craft issue 40",
        (
            "a sequence that underflows with implementation='scaling' is "
            "computed with the log functions"
        ),
    ),
}
# The error hmmlearn 0.3.3 raises where E7 applies.
UNDERFLOW_ERROR = [
    "ValueError",
    (
        "forward pass failed with underflow; consider using "
        "implementation='log' instead"
    ),
]

FORK_ONLY_ARRAYS = ("state_occupancy_",)
FORK_ONLY_PARAMS = {"covars_floor": None}


# ---------------------------------------------------------------------------
# Synthetic data. Pure numpy, RandomState(10), no call into the package.
# ---------------------------------------------------------------------------

_STARTPROB = np.array([0.5, 0.3, 0.2])
_TRANSMAT = np.array([[0.80, 0.15, 0.05],
                      [0.10, 0.80, 0.10],
                      [0.05, 0.15, 0.80]])
_GAUSS3_MEANS = np.array([[0.0, 0.0, 0.0], [3.0, -2.0, 1.0],
                          [-3.0, 2.0, 4.0]])
_GAUSS3_FACTORS = np.array([
    np.diag([0.5, 0.7, 0.9]),
    [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.3, 0.2, 0.9]],
    np.diag([1.2, 0.6, 0.8]),
])
_GAUSS1_MEANS = np.array([[-2.0], [0.0], [3.0]])
_GAUSS1_SCALES = np.array([0.5, 1.0, 0.7])
_MIX_MEANS = np.array([[[-4.0, -4.0], [-1.0, -1.0]],
                       [[1.0, 5.0], [4.0, 2.0]],
                       [[6.0, -3.0], [9.0, -6.0]]])
_MIX_SCALE = 0.8
_CAT_PROBS = np.array([[0.7, 0.1, 0.1, 0.1],
                       [0.1, 0.7, 0.1, 0.1],
                       [0.1, 0.1, 0.4, 0.4]])
_MULTI_PROBS = np.array([[0.4, 0.3, 0.2, 0.1],
                         [0.1, 0.2, 0.3, 0.4],
                         [0.25, 0.25, 0.25, 0.25]])
_LAMBDAS = np.array([[2.0, 5.0, 1.0], [10.0, 2.0, 4.0], [1.0, 1.0, 12.0]])
# 120 consecutive rows replaced by one repeated row, far from the rest: a
# state that holds only these rows has no spread, and a covariance floor binds.
_COLLAPSE_ROWS = slice(40, 160)
_COLLAPSE_ROW = np.array([9.0, 9.0, 9.0])
# One feature with no spread at all: a pooled (tied) covariance has an
# eigenvalue near zero, where a per-state collapse leaves it far from zero.
_FLAT_VALUE = 5.0


def _chain(rs, lengths):
    states = []
    for n in lengths:
        s = np.empty(n, dtype=int)
        s[0] = (_STARTPROB.cumsum() > rs.rand()).argmax()
        for t in range(1, n):
            s[t] = (_TRANSMAT[s[t - 1]].cumsum() > rs.rand()).argmax()
        states.append(s)
    return np.concatenate(states)


def make_data():
    """
    Return ``{name: dict(X=, lengths=, truth=)}`` for every data family.

    ``truth`` holds the parameters the data was drawn from; the fixed starts
    are these moved by a fixed amount.
    """
    rs = np.random.RandomState(10)
    lengths = np.array(SEQUENCE_LENGTHS)
    n = int(lengths.sum())
    states = _chain(rs, lengths)
    nc = N_COMPONENTS
    common = {"startprob": _STARTPROB, "transmat": _TRANSMAT}

    z = rs.standard_normal((n, 3))
    gauss3 = (_GAUSS3_MEANS[states]
              + np.einsum("tij,tj->ti", _GAUSS3_FACTORS[states], z))
    covs3 = np.einsum("kij,klj->kil", _GAUSS3_FACTORS, _GAUSS3_FACTORS)
    collapse = gauss3.copy()
    collapse[_COLLAPSE_ROWS] = _COLLAPSE_ROW
    flat = gauss3.copy()
    flat[:, 2] = _FLAT_VALUE

    z = rs.standard_normal((n, 1))
    gauss1 = _GAUSS1_MEANS[states] + _GAUSS1_SCALES[states][:, None] * z
    covs1 = (_GAUSS1_SCALES ** 2)[:, None, None]

    comp = (rs.rand(n) > 0.5).astype(int)
    z = rs.standard_normal((n, 2))
    gmm = _MIX_MEANS[states, comp] + _MIX_SCALE * z
    mix_covs = np.tile(_MIX_SCALE ** 2 * np.eye(2), (nc, 1, 1))

    cdf = _CAT_PROBS.cumsum(axis=1)[states]
    cat = (cdf > rs.rand(n)[:, None]).argmax(axis=1)[:, None]

    multi = np.array([rs.multinomial(N_TRIALS, _MULTI_PROBS[s])
                      for s in states])
    pois = rs.poisson(_LAMBDAS[states])

    gauss3_truth = dict(common, means=_GAUSS3_MEANS, covs=covs3)
    # The fixed start of the collapse family puts state 2 on the repeated row
    # (the start is these means plus 0.5).
    collapse_means = _GAUSS3_MEANS.copy()
    collapse_means[2] = _COLLAPSE_ROW - 0.5
    collapse_truth = dict(gauss3_truth, means=collapse_means)
    gauss1_truth = dict(common, means=_GAUSS1_MEANS, covs=covs1)
    # GMMHMM: two components per state, around each state's mean.
    gauss1_truth["mix_means"] = (
        _GAUSS1_MEANS[:, None, :] + np.array([-0.5, 0.5])[None, :, None])
    gauss1_truth["mix_covs"] = covs1
    gmm_truth = dict(common, mix_means=_MIX_MEANS, mix_covs=mix_covs)
    return {
        "gauss3": {"X": gauss3, "lengths": lengths, "truth": gauss3_truth},
        "flat": {"X": flat, "lengths": lengths, "truth": gauss3_truth},
        "gauss1": {"X": gauss1, "lengths": lengths, "truth": gauss1_truth},
        "collapse": {"X": collapse, "lengths": lengths,
                     "truth": collapse_truth},
        "gmm": {"X": gmm, "lengths": lengths, "truth": gmm_truth},
        "cat": {"X": cat, "lengths": lengths,
                "truth": dict(common, emission=_CAT_PROBS)},
        "multi": {"X": multi, "lengths": lengths,
                  "truth": dict(common, emission=_MULTI_PROBS)},
        "pois": {"X": pois, "lengths": lengths,
                 "truth": dict(common, lambdas=_LAMBDAS)},
    }


# The underflow families: a copy of a family with one row far from every
# state. No generator is drawn from.
_FAR_ROW = 50
_FAR_VALUE = 1000
_FAR_SOURCES = {"far-gauss3": "gauss3", "far-gmm": "gmm", "far-pois": "pois"}


def make_underflow_data(data):
    """
    Return ``{name: dict(X=, lengths=, truth=)}`` for the underflow families,
    each a copy of its source family in ``data`` with row ``_FAR_ROW`` (in the
    first sequence) set to ``_FAR_VALUE``.
    """
    out = {}
    for name, source in _FAR_SOURCES.items():
        X = data[source]["X"].copy()
        X[_FAR_ROW] = _FAR_VALUE
        out[name] = dict(data[source], X=X)
    return out


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# The list of fits.
# ---------------------------------------------------------------------------

# class -> (module, name of the sufficient statistic holding the state mass,
#           data families)
_CLASSES = {
    "GaussianHMM": (
        "hmm", "post", ("gauss3", "gauss1", "collapse", "flat")),
    "GMMHMM": ("hmm", "post_sum", ("gmm", "gauss1")),
    "VariationalGaussianHMM": ("vhmm", "post", ("gauss3",)),
    "CategoricalHMM": ("hmm", None, ("cat",)),
    "MultinomialHMM": ("hmm", None, ("multi",)),
    "PoissonHMM": ("hmm", "post", ("pois",)),
    "VariationalCategoricalHMM": ("vhmm", None, ("cat",)),
}
_COVARIANCE_CLASSES = ("GaussianHMM", "GMMHMM", "VariationalGaussianHMM")
_FIXED_CLASSES = (
    "GaussianHMM", "GMMHMM", "CategoricalHMM", "MultinomialHMM", "PoissonHMM")
_GAUSSIAN_PARAMS = (
    ("stmc", "neginf"), ("stmc", "default"), ("stm", "default"),
    ("st", "default"), ("stc", "default"))


def _case_name(cls, data, cov, start, impl, params, tol):
    suffix = "" if tol == "default" else "-tol" + tol
    return "|".join(
        [cls, data, cov or "-", start, impl, (params or "default") + suffix])


def build_cases():
    """Return the list of fit cases; each is a dict, the model's recipe."""
    cases = []
    for cls, (_, stat_key, families) in _CLASSES.items():
        covs = COVARIANCE_TYPES if cls in _COVARIANCE_CLASSES else (None,)
        starts = ("fixed", "seeded") if cls in _FIXED_CLASSES else ("seeded",)
        variants = (_GAUSSIAN_PARAMS if cls == "GaussianHMM"
                    else ((None, "default"),))
        for data in families:
            for cov in covs:
                for start in starts:
                    for impl in IMPLEMENTATIONS:
                        for params, tol in variants:
                            stat_defined = stat_key is not None and (
                                cls != "GaussianHMM" or "m" in params)
                            cases.append({
                                "name": _case_name(
                                    cls, data, cov, start, impl, params, tol),
                                "kind": "fit", "cls": cls, "data": data,
                                "cov": cov, "start": start, "impl": impl,
                                "params": params,
                                "tol": tol, "stat_key": stat_key,
                                "stat_defined": stat_defined})
    return cases


# The underflow cases: (class, data family, start, family of the warm start).
# "fixed" is the fixed start of build_model; "tiny" and "zero" move the
# fixed start's emission column _TINY_COLUMN into the next and set it to
# _TINY or to zero; "warm" is the seeded scaling fit on the clean family,
# refitted without initialization.
_UNDERFLOW_CLASSES = (
    ("GaussianHMM", "far-gauss3", "fixed", None),
    ("GMMHMM", "far-gmm", "fixed", None),
    ("PoissonHMM", "far-pois", "fixed", None),
    ("CategoricalHMM", "cat", "tiny", None),
    ("MultinomialHMM", "multi", "tiny", None),
    ("VariationalGaussianHMM", "far-gauss3", "warm", "gauss3"),
    ("VariationalCategoricalHMM", "cat", "warm", "cat"),
    ("CategoricalHMM", "cat", "zero", None),
)
# Below the kernel's threshold of 1e-300, above zero. Column 0, not the last:
# some scipy versions replace the last probability of a categorical row.
_TINY = 1e-320
_TINY_COLUMN = 0
# The emission count of the warm VariationalCategoricalHMM's column 0.
_TINY_COUNT = 1e-5


def build_underflow_cases():
    """
    Return the underflow cases. ``expect`` is ``"fallback"`` where the log
    functions give a finite log-likelihood, ``"error"`` where none can.
    """
    cases = []
    for cls, data, start, clean in _UNDERFLOW_CLASSES:
        covs = COVARIANCE_TYPES if cls in _COVARIANCE_CLASSES else (None,)
        for cov in covs:
            for step in ("predict", "fit"):
                if step == "predict" and cls == "VariationalCategoricalHMM":
                    # Prediction reads emissionprob_, which the warm start
                    # leaves as the clean fit made it: nothing underflows.
                    continue
                cases.append(
                    {
                        "name": "|".join(
                            ["underflow", step, cls, data, cov or "-", start]
                        ),
                        "kind": "underflow",
                        "step": step,
                        "cls": cls,
                        "data": data,
                        "clean": clean,
                        "cov": cov,
                        "start": start,
                        "impl": "scaling",
                        "params": None,
                        "tol": "default",
                        "expect": "error" if start == "zero" else "fallback",
                    }
                )
    return cases


def _spherical_variances(covs):
    return np.trace(covs, axis1=-2, axis2=-1) / covs.shape[-1]


def _gaussian_covars(covs, covariance_type):
    covs = covs * 1.5
    if covariance_type == "full":
        return covs
    if covariance_type == "diag":
        return np.diagonal(covs, axis1=-2, axis2=-1).copy()
    if covariance_type == "tied":
        return covs.mean(axis=0)
    return _spherical_variances(covs)


def _mixture_covars(covs, covariance_type, n_mix):
    covs = covs * 1.5
    if covariance_type == "full":
        return np.repeat(covs[:, None], n_mix, axis=1)
    if covariance_type == "diag":
        diag = np.diagonal(covs, axis1=-2, axis2=-1)
        return np.repeat(diag[:, None], n_mix, axis=1)
    if covariance_type == "tied":
        return covs
    return np.repeat(_spherical_variances(covs)[:, None], n_mix, axis=1)


def build_model(modules, case, data):
    """Return the unfitted model of ``case``."""
    cls = getattr(modules[_CLASSES[case["cls"]][0]], case["cls"])
    name, fixed = case["cls"], case["start"] == "fixed"
    kw = {"n_components": N_COMPONENTS, "n_iter": N_ITER,
              "implementation": case["impl"], "random_state": MODEL_SEED}
    if fixed:
        kw["init_params"] = ""
    if case["cov"]:
        kw["covariance_type"] = case["cov"]
    if case["params"]:
        kw["params"] = case["params"]
    if case["tol"] == "neginf":
        kw["tol"] = -np.inf
    if name == "GMMHMM":
        kw["n_mix"] = 2
    if name == "MultinomialHMM":
        kw["n_trials"] = N_TRIALS
    model = cls(**kw)
    if fixed:
        _set_fixed_start(model, case, data["truth"])
    return model


def _set_fixed_start(model, case, truth):
    nc = N_COMPONENTS
    model.startprob_ = 0.8 * truth["startprob"] + 0.2 / nc
    model.transmat_ = 0.8 * truth["transmat"] + 0.2 / nc
    name = case["cls"]
    if name == "GaussianHMM":
        model.means_ = truth["means"] + 0.5
        model.covars_ = _gaussian_covars(truth["covs"], case["cov"])
    elif name == "GMMHMM":
        model.weights_ = np.full((nc, 2), 0.5)
        model.means_ = truth["mix_means"] + 0.5
        model.covars_ = _mixture_covars(truth["mix_covs"], case["cov"], 2)
    elif name in ("CategoricalHMM", "MultinomialHMM"):
        n_symbols = truth["emission"].shape[1]
        model.emissionprob_ = 0.8 * truth["emission"] + 0.2 / n_symbols
    elif name == "PoissonHMM":
        model.lambdas_ = truth["lambdas"] + 0.5


# ---------------------------------------------------------------------------
# What is recorded for one fitted model.
# ---------------------------------------------------------------------------

def _as_array(name, value):
    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError(f"{name}: cannot record an object array")
    return array


def _model_arrays(model):
    """Every fitted array of ``model``, after the fit or after its failure."""
    arrays = {}
    for name, value in vars(model).items():
        if name.endswith("_") and name != "monitor_":
            arrays[name] = _as_array(name, value)
    n_features = getattr(model, "n_features", None)
    if n_features is not None:
        arrays["n_features"] = np.asarray(n_features)
    monitor = model.monitor_
    arrays["monitor_history"] = np.asarray(
        list(monitor.history), dtype=np.float64)
    arrays["monitor_iter"] = np.asarray(monitor.iter)
    arrays["monitor_converged"] = np.asarray(monitor.converged)
    rs = vars(model).get("random_state")
    if isinstance(rs, np.random.RandomState):
        _, key, pos, has_gauss, cached = rs.get_state()
        arrays["random_state_key"] = key
        arrays["random_state_pos"] = np.asarray(pos)
        arrays["random_state_gauss"] = np.asarray([has_gauss, cached])
    _, key, pos, _, _ = np.random.get_state()
    arrays["global_random_key"] = key
    arrays["global_random_pos"] = np.asarray(pos)
    return arrays


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.random.RandomState):
        return "<RandomState>"
    text = repr(value)
    if " at 0x" in text:
        raise TypeError(f"parameter without a stable repr: {text}")
    return text


def _params_json(model):
    return json.loads(json.dumps(model.get_params(), default=_json_default))


def _error(exc):
    return [type(exc).__name__, str(exc)]


def _recorders(arrays, errors, prefix):
    """Return ``run`` and ``put``, which record into ``arrays``, ``errors``."""

    def run(step, func):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            errors[prefix + step] = _error(exc)
            return None

    def put(key, value):
        if value is not None:
            arrays[prefix + key] = _as_array(key, value)

    return run, put


def _probe_inference(model, X, lengths, arrays, errors, prefix=""):
    """Score, both decodes and the posteriors, recording each result."""
    run, put = _recorders(arrays, errors, prefix)
    put("score", run("score", lambda: model.score(X, lengths)))
    for algorithm in ("viterbi", "map"):
        out = run(f"decode_{algorithm}",
                  lambda a=algorithm: model.decode(X, lengths, a))
        if out is not None:
            put(f"decode_{algorithm}_logprob", out[0])
            put(f"decode_{algorithm}_states", out[1])
    put("predict_proba", run("predict_proba",
                             lambda: model.predict_proba(X, lengths)))


def _probe(model, X, lengths, arrays, errors, prefix=""):
    """Run every read-only call on a fitted model, recording each result."""
    _probe_inference(model, X, lengths, arrays, errors, prefix)
    run, put = _recorders(arrays, errors, prefix)
    out = run("sample",
              lambda: model.sample(N_SAMPLES, random_state=SAMPLE_SEED))
    if out is not None:
        put("sample_X", out[0])
        put("sample_states", out[1])
    try:
        put("covars_", model.covars_)
    except AttributeError:
        pass
    except Exception as exc:  # noqa: BLE001
        errors[prefix + "covars_"] = _error(exc)


def _recording_class(cls, stat_key):
    """
    Subclass of ``cls`` that keeps what its last M-step was given: the state
    mass the class itself accumulates (``stat_key``) and an independent sum of
    the posteriors of the E-step.
    """

    class Recorder(cls):
        def _accumulate_sufficient_statistics(
                self, stats, X, lattice, posteriors, fwdlattice, bwdlattice):
            if getattr(self, "_rec_sum", None) is None:
                self._rec_sum = np.zeros(self.n_components)
            self._rec_sum += posteriors.sum(axis=0)
            super()._accumulate_sufficient_statistics(
                stats, X, lattice, posteriors, fwdlattice, bwdlattice)

        def _do_mstep(self, stats):
            self._rec_post_sum = self._rec_sum
            self._rec_sum = None
            if stat_key is not None:
                self._rec_stat = stats[stat_key].copy()
            super()._do_mstep(stats)

    Recorder.__name__ = cls.__name__
    Recorder.__qualname__ = cls.__qualname__
    return Recorder


def _fit_recorded(modules, case, data, plain_arrays):
    """Refit with the recording subclass; check it changed nothing."""
    recorder_cls = _recording_class(
        getattr(modules[_CLASSES[case["cls"]][0]], case["cls"]),
        case["stat_key"])
    np.random.seed(0)
    model = build_model(modules, case, data)
    model.__class__ = recorder_cls
    model.fit(data["X"], data["lengths"])
    arrays = _model_arrays(model)
    for key in sorted(set(arrays) | set(plain_arrays)):
        if key not in arrays or key not in plain_arrays or (
                arrays[key].tobytes() != plain_arrays[key].tobytes()):
            raise AssertionError(
                f"{case['name']}: the recording subclass moved {key}")
    out = {"m01_post_sum_last": model._rec_post_sum}
    if case["stat_key"] is not None:
        out["m01_stat_last"] = model._rec_stat
    return out


def _dump_fit(modules, case, data, pickle_dir):
    arrays, errors = {}, {}
    X, lengths = data["X"], data["lengths"]
    np.random.seed(0)
    model = build_model(modules, case, data)
    params = None
    try:
        model.fit(X, lengths)
    except Exception as exc:  # noqa: BLE001
        errors["fit"] = _error(exc)
    arrays.update(_model_arrays(model))
    if "fit" not in errors:
        params = _params_json(model)
        _probe(model, X, lengths, arrays, errors)
        arrays.update(_fit_recorded(modules, case, data, _model_arrays(model)))
        if pickle_dir is not None:
            blob = pickle.dumps(model)
            (pickle_dir / f"{case['index']:04d}.pkl").write_bytes(blob)
    return {"meta": case, "arrays": arrays, "errors": errors, "params": params}


def _underflow_model(modules, case, data):
    """The model of an underflow case, before the call that underflows."""
    if case["start"] == "warm":
        clean = data[case["clean"]]
        model = build_model(modules, dict(case, start="seeded"), clean)
        model.fit(clean["X"], clean["lengths"])
        model.init_params = ""
        if case["cls"] == "VariationalCategoricalHMM":
            model.emissionprob_posterior_[:, _TINY_COLUMN] = _TINY_COUNT
        return model
    model = build_model(modules, dict(case, start="fixed"), data[case["data"]])
    if case["start"] in ("tiny", "zero"):
        emission = np.array(model.emissionprob_)
        emission[:, _TINY_COLUMN + 1] += emission[:, _TINY_COLUMN]
        emission[:, _TINY_COLUMN] = _TINY if case["start"] == "tiny" else 0
        model.emissionprob_ = emission
    return model


def _dump_underflow(modules, case, data):
    """
    A prediction case predicts on the first sequence alone, with
    ``implementation="log"`` (the plain keys) and then ``"scaling"`` (keys
    prefixed ``scaling:``); a fit case fits on every sequence and, when the
    fit returns, probes the fitted model.
    """
    arrays, errors = {}, {}
    family = data[case["data"]]
    X, lengths = family["X"], family["lengths"]
    np.random.seed(0)
    model = _underflow_model(modules, case, data)
    params = None
    if case["step"] == "predict":
        first = X[: lengths[0]]
        for impl, prefix in (("log", ""), ("scaling", "scaling:")):
            model.implementation = impl
            _probe_inference(model, first, None, arrays, errors, prefix)
    else:
        try:
            model.fit(X, lengths)
        except Exception as exc:  # noqa: BLE001
            errors["fit"] = _error(exc)
        arrays.update(_model_arrays(model))
        if "fit" not in errors:
            params = _params_json(model)
            _probe(model, X, lengths, arrays, errors)
    return {"meta": case, "arrays": arrays, "errors": errors, "params": params}


class _MappingUnpickler(pickle.Unpickler):
    """Load a pickle written under another package name."""

    def __init__(self, file, source, target):
        super().__init__(file)
        self._source, self._target = source, target

    def find_class(self, module, name):
        if module == self._source or module.startswith(self._source + "."):
            module = self._target + module[len(self._source):]
        return super().find_class(module, name)


def _dump_pickled(package, pickle_file, case, data):
    arrays, errors = {}, {}
    with open(pickle_file, "rb") as f:
        model = _MappingUnpickler(f, "hmmlearn", package).load()
    _probe(model, data["X"], data["lengths"], arrays, errors)
    return {"meta": dict(case, kind="pickled"), "arrays": arrays,
                "errors": errors, "params": _params_json(model)}


# ---------------------------------------------------------------------------
# Utilities: normalize and fill_covars on fixed arrays.
# ---------------------------------------------------------------------------

def _utility_inputs():
    rs = np.random.RandomState(10)
    cases = []
    a1 = rs.rand(7) + 0.1
    cases += [("normalize", {"axis": None}, a1),
              ("normalize", {"axis": 0}, a1)]
    a2 = rs.rand(4, 5) + 0.1
    a2[1, :] = 0
    a2[:, 3] = 0
    a2f = a2.astype(np.float32)
    for axis in (None, 0, 1, -1):
        cases.append(("normalize", {"axis": axis}, a2))
    cases.append(("normalize", {"axis": 0}, a2f))
    a3 = rs.rand(3, 4, 5) + 0.1
    a3[1, 2, :] = 0
    a3[:, 1, 3] = 0
    a3[2, :, 4] = 0
    for axis in (None, 0, 1, 2, -1):
        cases.append(("normalize", {"axis": axis}, a3))
    nc, nf = 3, 2
    cases += [
        ("fill_covars", {"covariance_type": "full", "n_components": nc,
                             "n_features": nf}, rs.rand(nc, nf, nf) + 1),
        ("fill_covars", {"covariance_type": "diag", "n_components": nc,
                             "n_features": nf}, rs.rand(nc, nf) + 1),
        ("fill_covars", {"covariance_type": "tied", "n_components": nc,
                             "n_features": nf}, rs.rand(nf, nf) + 1),
        ("fill_covars", {"covariance_type": "spherical", "n_components": nc,
                             "n_features": nf}, rs.rand(nc) + 1),
    ]
    # The layouts a spherical GaussianHMM holds: one variance repeated over
    # the features, whether it comes from the start or from a fit.
    for label, values, nf in (
            ("start-0.1", np.full(3, 0.1), 3),
            ("start-steps", np.array([0.1, 0.2, 0.3]), 3),
            ("start-one-feature", np.array([0.1, 0.7]), 1),
            ("fitted", rs.rand(3) + 0.5, 3),
            ("fitted-wide", rs.rand(2) + 0.5, 9)):
        tiled = np.tile(values[:, None], (1, nf))
        cases.append(("fill_covars", {
            "covariance_type": "spherical", "n_components": len(values),
            "n_features": nf, "layout": label}, tiled))
    return cases


def _utility_name(index, kind, options):
    text = ",".join(f"{k}={v}" for k, v in sorted(options.items()))
    return f"{kind}|{index:02d}|{text}"


def build_utility_cases():
    out = []
    for index, (kind, options, array) in enumerate(_utility_inputs()):
        out.append({
            "name": _utility_name(index, kind, options), "kind": kind,
            "options": options, "input": array, "dtype": str(array.dtype)})
    return out


def _dump_utility(package, case):
    utils = importlib.import_module(f"{package}.utils")
    options = {k: v for k, v in case["options"].items() if k != "layout"}
    source = case["input"]
    errors = {}
    result = None
    try:
        if case["kind"] == "normalize":
            result = source.copy()
            with np.errstate(all="ignore"):
                utils.normalize(result, **options)
        else:
            result = utils.fill_covars(source.copy(), **options)
    except Exception as exc:  # noqa: BLE001
        errors["call"] = _error(exc)
    arrays = {"input": source}
    if result is not None:
        arrays["result"] = _as_array("result", result)
    meta = {k: v for k, v in case.items() if k != "input"}
    return {"meta": meta, "arrays": arrays, "errors": errors, "params": None}


# ---------------------------------------------------------------------------
# Writing and reading a dump.
# ---------------------------------------------------------------------------

def write_dump(out, environment, data_sha256, cases):
    """
    Write ``cases`` (dicts ``meta``, ``arrays``, ``errors``, ``params``) and
    ``environment`` under the directory ``out``.
    """
    out = Path(out)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    entries = {}
    for number, case in enumerate(cases):
        file = f"{number:04d}.npz"
        np.savez(out / "cases" / file, **case["arrays"])
        entries[case["meta"]["name"]] = {
            "meta": case["meta"], "errors": case["errors"],
            "params": case["params"], "file": file}
    manifest = {"format": 1, "environment": environment,
                "data_sha256": data_sha256, "cases": entries}
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True, default=_json_default,
                   allow_nan=True))


def read_dump(path):
    """Return ``(manifest, {case name: {arrays, ...}})`` for a dump."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    cases = {}
    for name, entry in manifest["cases"].items():
        with np.load(path / "cases" / entry["file"]) as npz:
            arrays = {key: npz[key] for key in npz.files}
        cases[name] = dict(entry, arrays=arrays)
    return manifest, cases


# ---------------------------------------------------------------------------
# The dump command.
# ---------------------------------------------------------------------------

def _require_one_thread():
    wrong = {v: os.environ.get(v) for v in THREAD_VARIABLES
             if os.environ.get(v) != "1"}
    if wrong:
        raise SystemExit(
            "refusing to run: set " + ", ".join(THREAD_VARIABLES)
            + " to 1 before starting Python (now: " + repr(wrong) + ")")


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _environment(package, modules):
    import scipy
    import sklearn
    import threadpoolctl

    extension = importlib.import_module(f"{package}._hmmc")
    return {
        "package": package,
        "package_version": importlib.metadata.version(DISTRIBUTIONS[package]),
        "extension_sha256": _file_sha256(extension.__file__),
        "script_sha256": _file_sha256(__file__),
        "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "threadpoolctl": threadpoolctl.__version__,
        "machine": f"{platform.system()} {platform.machine()}",
        "thread_variables": {v: os.environ.get(v) for v in THREAD_VARIABLES},
        # threadpoolctl lists the libraries in the order they were found,
        # which changes between runs.
        "threadpool_info": sorted(
            threadpoolctl.threadpool_info(),
            key=lambda lib: (lib["filepath"], lib["internal_api"]))}


def dump(package, out, load_pickles=None):
    """Fit every case with ``package`` and write the dump to ``out``."""
    _require_one_thread()
    import threadpoolctl

    warnings.simplefilter("ignore")
    logging.disable(logging.CRITICAL)
    modules = {
        name: importlib.import_module(f"{package}.{name}")
        for name in ("hmm", "vhmm")}
    data = make_data()
    out = Path(out)
    pickle_dir = None
    if package == "hmmlearn":
        pickle_dir = out / "pickles"
        pickle_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with threadpoolctl.threadpool_limits(limits=1):
        environment = _environment(package, modules)
        for index, case in enumerate(build_cases()):
            family = data[case["data"]]
            case = dict(case, index=index,
                        lengths=family["lengths"].tolist(),
                        n_features=int(family["X"].shape[1]))
            results.append(_dump_fit(modules, case, family, pickle_dir))
        for case in build_utility_cases():
            results.append(_dump_utility(package, case))
        if load_pickles is not None:
            results += _pickle_pass(package, Path(load_pickles), data)
        # Last, so the cases above keep their file numbers.
        data.update(make_underflow_data(data))
        for case in build_underflow_cases():
            family = data[case["data"]]
            lengths = family["lengths"].tolist()
            if case["step"] == "predict":
                lengths = lengths[:1]
            case = dict(
                case, lengths=lengths, n_features=int(family["X"].shape[1])
            )
            results.append(_dump_underflow(modules, case, data))
    data_sha256 = {name: _digest(d["X"], d["lengths"])
                   for name, d in data.items()}
    for case in build_utility_cases():
        data_sha256[case["name"]] = _digest(case["input"])
    environment["pickles_loaded"] = load_pickles is not None
    write_dump(out, environment, data_sha256, results)
    n_fits = sum(r["meta"]["kind"] == "fit" for r in results)
    n_underflow = sum(r["meta"]["kind"] == "underflow" for r in results)
    print(
        f"wrote {len(results)} cases ({n_fits} fits, {n_underflow} "
        f"underflow cases) to {out}"
    )
    return 0


def _pickle_pass(package, ref_dir, data):
    manifest = json.loads((ref_dir / "manifest.json").read_text())
    results = []
    for name, entry in manifest["cases"].items():
        meta = entry["meta"]
        if meta["kind"] != "fit" or entry["errors"].get("fit"):
            continue
        file = ref_dir / "pickles" / f"{meta['index']:04d}.pkl"
        case = dict(meta, name="pickled:" + name)
        results.append(_dump_pickled(package, file, case, data[meta["data"]]))
    return results


# ---------------------------------------------------------------------------
# The compare command.
# ---------------------------------------------------------------------------

Difference = namedtuple("Difference", "case key text")

_COMPARED_ENVIRONMENT = (
    "python", "numpy", "scipy", "scikit_learn", "threadpoolctl",
    "script_sha256", "machine")
_PROBE_STEPS = (
    "score", "decode_viterbi", "decode_map", "predict_proba", "sample",
    "covars_")
_PICKLED = "pickled:"
# The prediction steps E7 applies to -> the keys each records.
_E7_STEPS = {
    "score": ("score",),
    "predict_proba": ("predict_proba",),
    "decode_map": ("decode_map_logprob", "decode_map_states"),
}


class Result:
    """Outcome of a comparison."""

    def __init__(self):
        self.refused = []
        self.differences = []
        self.fired = Counter()
        self.examples = {}
        self.n_cases = 0
        self.n_arrays = 0
        self.ref_environment = None
        self.fork_environment = None

    @property
    def code(self):
        if self.refused:
            return 2
        return 1 if self.differences else 0

    def fire(self, exception, case):
        self.fired[exception] += 1
        self.examples.setdefault(exception, [])
        if case not in self.examples[exception]:
            self.examples[exception].append(case)


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and (
        a.tobytes() == b.tobytes())


def _finite(value):
    return value.dtype.kind != "f" or bool(np.isfinite(value).all())


def _describe(ref, fork):
    if ref.dtype != fork.dtype:
        return f"dtype: reference {ref.dtype}, fork {fork.dtype}"
    if ref.shape != fork.shape:
        return f"shape: reference {ref.shape}, fork {fork.shape}"
    size = ref.dtype.itemsize
    rb = np.ascontiguousarray(ref).view(np.uint8).reshape(-1, size)
    fb = np.ascontiguousarray(fork).view(np.uint8).reshape(-1, size)
    first = int((rb != fb).any(axis=1).argmax())
    # Plain Python values: numpy 2 prints a scalar as np.float64(...).
    index = tuple(int(i) for i in np.unravel_index(first, ref.shape))
    a, b = ref.reshape(-1)[first].item(), fork.reshape(-1)[first].item()
    return (f"{ref.dtype} {ref.shape}: first difference at {index}: "
            f"reference {a!r}, fork {b!r}")


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=True)


def _normalised_error(error):
    if error is None:
        return None
    return [error[0], error[1].replace("hmmcraft", "hmmlearn")]


def _refusals(ref, fork):
    """Reasons the two dumps cannot be compared; empty if they can."""
    reasons = []
    ref_env, fork_env = ref[0]["environment"], fork[0]["environment"]
    for field in _COMPARED_ENVIRONMENT:
        if ref_env.get(field) != fork_env.get(field):
            reasons.append(
                f"environment differs in {field}: reference "
                f"{ref_env.get(field)!r}, fork {fork_env.get(field)!r}")
    for label, env in (("reference", ref_env), ("fork", fork_env)):
        wrong = {k: v for k, v in env["thread_variables"].items()
                 if v != "1"}
        if wrong:
            reasons.append(f"{label} was not run at one thread: {wrong}")
    if ref[0]["data_sha256"] != fork[0]["data_sha256"]:
        reasons.append("the two sides saw different input data")
    ref_cases = {n: c for n, c in ref[1].items()
                 if not n.startswith(_PICKLED)}
    fork_cases = {n: c for n, c in fork[1].items()
                  if not n.startswith(_PICKLED)}
    if set(ref_cases) != set(fork_cases):
        reasons.append(
            "the case lists differ: "
            f"{sorted(set(ref_cases) ^ set(fork_cases))[:5]}")
    else:
        for name in ref_cases:
            if ref_cases[name]["meta"] != fork_cases[name]["meta"]:
                reasons.append(f"the recipe of case {name} differs")
                break
    for name in fork[1]:
        if name.startswith(_PICKLED) and name[len(_PICKLED):] not in ref[1]:
            reasons.append(f"{name} has no reference case")
    return reasons


def _map_sum(posteriors, lengths, logged):
    """The sum ``decode`` forms for the MAP states, in its order."""
    total = 0
    for part in np.split(posteriors, np.cumsum(lengths))[:-1]:
        peak = np.max(part, axis=1)
        total += (np.log(peak) if logged else peak).sum()
    return np.asarray(total)


class _Case:
    """What is compared for one case, and the exceptions that may excuse it."""

    def __init__(self, name, ref, fork, allow, result):
        self.name, self.ref, self.fork = name, ref, fork
        self.allow, self.result = allow, result
        self.meta = fork["meta"]
        self.pickled = name.startswith(_PICKLED)
        self.pending = {}

    @property
    def is_fit(self):
        return self.meta["kind"] in ("fit", "pickled")

    def _is(self, cls, covariance_type=None):
        return (self.is_fit and self.meta["cls"] == cls and (
            covariance_type is None
            or self.meta["cov"] == covariance_type))

    def collect(self):
        """Strict comparison: every difference, before any exception."""
        ref, fork = self.ref, self.fork
        steps = set(ref["errors"]) | set(fork["errors"])
        if self.pickled:
            steps &= set(_PROBE_STEPS)
        for step in sorted(steps):
            a = _normalised_error(ref["errors"].get(step))
            b = _normalised_error(fork["errors"].get(step))
            if a != b:
                self.pending["errors:" + step] = (
                    f"reference {a}, fork {b}")
        ref_params, fork_params = ref["params"], fork["params"]
        if (ref_params is None) != (fork_params is None):
            self.pending["params"] = "one side has no parameters"
        elif ref_params is not None:
            for name in sorted(set(ref_params) | set(fork_params)):
                a = _canonical(ref_params.get(name, "<absent>"))
                b = _canonical(fork_params.get(name, "<absent>"))
                if a != b:
                    self.pending["params:" + name] = (
                        f"reference {a}, fork {b}")
        keys = set(fork["arrays"]) if self.pickled else (
            set(ref["arrays"]) | set(fork["arrays"]))
        for key in sorted(keys):
            if key not in fork["arrays"]:
                self.pending[key] = "missing in the fork"
            elif key not in ref["arrays"]:
                self.pending[key] = "only in the fork"
            else:
                self.result.n_arrays += 1
                a, b = ref["arrays"][key], fork["arrays"][key]
                if not _same(a, b):
                    self.pending[key] = _describe(a, b)

    def problem(self, key, text):
        self.result.differences.append(Difference(self.name, key, text))

    def excuse(self, exception, key):
        self.pending.pop(key, None)
        self.result.fire(exception, self.name)

    def apply_exceptions(self):
        allow = self.allow
        if "E7" in allow and self._e7_applies():
            self._rule_e7()
        e4_clean = False
        if self.pending and "E4" in allow and self._e4_applies():
            e4_clean = self._rule_e4()
        elif self.pending:
            for key in list(self.pending):
                for exception, rule in _KEY_RULES:
                    if exception in allow and rule(self, key):
                        break
        if "E6" in allow:
            self._rule_e6(skip_ref=e4_clean)
        for key, text in self.pending.items():
            self.problem(key, text)

    # E4 -----------------------------------------------------------------

    def _e4_applies(self):
        params = self.meta.get("params") or ""
        return (self.meta["kind"] == "fit" and self._is("GaussianHMM")
                and "c" in params and "m" not in params)

    def _rule_e4(self):
        fork = self.fork
        if fork["errors"]:
            self.problem("E4", "the fork's fit or probe raised: "
                         f"{fork['errors']}")
            return False
        bad = [k for k, v in fork["arrays"].items()
               if v.dtype.kind == "f" and not np.isfinite(v).all()]
        if bad:
            self.problem("E4", f"non-finite in the fork: {bad}")
            return False
        for key in list(self.pending):
            if key not in FORK_ONLY_ARRAYS and not key.startswith(
                    "params:covars_floor"):
                self.pending.pop(key)
        self.result.fire("E4", self.name)
        return True

    # E7 -----------------------------------------------------------------

    def _e7_applies(self):
        return (
            self.meta["kind"] == "underflow"
            and self.meta["expect"] == "fallback"
        )

    def _rule_e7(self):
        if self.meta["step"] == "fit":
            self._rule_e7_fit()
            return
        ref, fork = self.ref, self.fork
        for step, keys in _E7_STEPS.items():
            name = "scaling:" + step
            if ref["errors"].get(name) != UNDERFLOW_ERROR:
                self.problem(
                    "E7",
                    "the reference did not raise the underflow "
                    f"error in {step}: {ref['errors'].get(name)}",
                )
                continue
            if name in fork["errors"]:
                self.problem(
                    "E7", f"the fork raised in {step}: {fork['errors'][name]}"
                )
                continue
            twins = {key: ref["arrays"].get(key) for key in keys}
            if step == "decode_map":
                # 0.3.3's twin sums the posteriors (E3): the fork's value is
                # the sum of their logs.
                posteriors = ref["arrays"].get("predict_proba")
                twins["decode_map_logprob"] = (
                    None
                    if posteriors is None
                    else (
                        _map_sum(posteriors, self.meta["lengths"], logged=True)
                    )
                )
            ok = True
            for key, twin in twins.items():
                value = fork["arrays"].get("scaling:" + key)
                if value is None or twin is None:
                    self.problem(
                        "E7", f"scaling:{key} or its log twin is missing"
                    )
                    ok = False
                elif not _finite(value):
                    self.problem("E7", f"scaling:{key} is not finite")
                    ok = False
                elif not _same(twin, value):
                    self.problem(
                        "E7",
                        f"scaling:{key} differs from the reference's "
                        "log twin: " + _describe(twin, value),
                    )
                    ok = False
            if ok:
                for key in ("errors:" + name,) + tuple(
                    "scaling:" + k for k in keys
                ):
                    if key in self.pending:
                        self.excuse("E7", key)

    def _rule_e7_fit(self):
        if self.ref["errors"].get("fit") != UNDERFLOW_ERROR:
            self.problem(
                "E7",
                "the reference's fit did not raise the "
                f"underflow error: {self.ref['errors']}",
            )
            return
        if self.fork["errors"]:
            self.problem(
                "E7", f"the fork's fit or probe raised: {self.fork['errors']}"
            )
            return
        bad = [k for k, v in self.fork["arrays"].items() if not _finite(v)]
        if bad:
            self.problem("E7", f"non-finite in the fork: {bad}")
            return
        self.pending.clear()
        self.result.fire("E7", self.name)

    # E6 -----------------------------------------------------------------

    def _rule_e6(self, skip_ref):
        for key in list(self.pending):
            name = key[len("params:"):]
            if key.startswith("params:") and name in FORK_ONLY_PARAMS and (
                    name not in (self.ref["params"] or {})
                    and (self.fork["params"] or {}).get(name)
                    == FORK_ONLY_PARAMS[name]):
                self.excuse("E6", key)
        if self.meta["kind"] != "fit" or "fit" in self.fork["errors"]:
            return
        key = "state_occupancy_"
        occupancy = self.fork["arrays"].get(key)
        if occupancy is None:
            self.pending[key] = "missing in the fork"
            return
        self.result.n_arrays += 1
        recorded = []
        for side, arrays in (("reference", self.ref["arrays"]),
                             ("fork", self.fork["arrays"])):
            if side == "reference" and skip_ref:
                continue
            names = ["m01_post_sum_last"]
            if self.meta.get("stat_defined"):
                names.append("m01_stat_last")
            recorded += [(side, n, arrays[n]) for n in names if n in arrays]
        if not any(side == "fork" for side, _, _ in recorded):
            self.pending[key] = "no sum of the posteriors recorded to check"
            return
        for side, name, value in recorded:
            if not _same(value, occupancy):
                self.pending[key] = (
                    f"differs from the {side}'s {name}: "
                    + _describe(value, occupancy))
                return
        self.excuse("E6", key)


def _rule_e1(case, key):
    meta, ref, fork = case.meta, case.ref, case.fork
    if case._is("GaussianHMM", "spherical") and key in (
            "_covars_", "covars_"):
        n_features = meta["n_features"]
    elif (meta["kind"] == "fill_covars" and key == "result"
          and meta["options"]["covariance_type"] == "spherical"):
        n_features = meta["options"]["n_features"]
    else:
        return False
    a, b = ref["arrays"][key], fork["arrays"][key]
    if a.dtype != b.dtype or a.shape == b.shape:
        return False
    if a.ndim == 2 and b.ndim == 1 and a.shape[0] == b.shape[0]:
        ok = all(_same(np.ascontiguousarray(a[:, j]), b)
                 for j in range(a.shape[1]))
    elif a.ndim == 3 and b.ndim == 3:
        ok = _same(np.ascontiguousarray(a[::n_features]), b)
    else:
        ok = False
    if ok:
        case.excuse("E1", key)
    return ok


def _rule_e2(case, key):
    if not (case._is("GaussianHMM", "spherical") and key == "sample_X"
            and case.meta["n_features"] > 1):
        return False
    ref, fork = case.ref["arrays"], case.fork["arrays"]
    states = ref.get("sample_states")
    if states is None or not _same(states, fork.get("sample_states")):
        return False
    first = states == 0
    a, b = ref[key], fork[key]
    if a.dtype != b.dtype or a.shape != b.shape or not _same(
            np.ascontiguousarray(a[first]), np.ascontiguousarray(b[first])):
        return False
    case.excuse("E2", key)
    return True


def _rule_e3(case, key):
    if key != "decode_map_logprob" or "predict_proba" not in case.ref[
            "arrays"]:
        return False
    posteriors = case.ref["arrays"]["predict_proba"]
    lengths = case.meta["lengths"]
    a, b = case.ref["arrays"][key], case.fork["arrays"][key]
    if not (_same(a, _map_sum(posteriors, lengths, logged=False))
            and _same(b, _map_sum(posteriors, lengths, logged=True))):
        return False
    case.excuse("E3", key)
    return True


def _rule_e5(case, key):
    meta = case.meta
    if not (meta["kind"] == "normalize" and key == "result"
            and meta["options"].get("axis") == 0):
        return False
    source = case.ref["arrays"]["input"]
    zero = np.broadcast_to(source.sum(axis=0) == 0, source.shape)
    a, b = case.ref["arrays"][key], case.fork["arrays"][key]
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    ok = (np.isnan(a[zero]).all() and (b[zero] == 0).all()
          and _same(np.ascontiguousarray(a[~zero]),
                    np.ascontiguousarray(b[~zero])))
    if ok:
        case.excuse("E5", key)
    return ok


_KEY_RULES = (("E1", _rule_e1), ("E2", _rule_e2), ("E3", _rule_e3),
              ("E5", _rule_e5))


def compare_dumps(ref_dir, fork_dir, allow=()):
    """
    Compare two dumps and return a ``Result``.

    ``allow`` lists the exception ids a difference may be excused by.
    """
    result = Result()
    ref, fork = read_dump(ref_dir), read_dump(fork_dir)
    result.ref_environment = ref[0]["environment"]
    result.fork_environment = fork[0]["environment"]
    result.refused = _refusals(ref, fork)
    if result.refused:
        return result
    allow = set(allow)
    for name, fork_case in fork[1].items():
        ref_name = name.removeprefix(_PICKLED)
        case = _Case(name, ref[1][ref_name], fork_case, allow, result)
        case.collect()
        case.apply_exceptions()
        result.n_cases += 1
    if result.fork_environment.get("pickles_loaded"):
        eligible = {n for n, c in ref[1].items()
                    if c["meta"]["kind"] == "fit"
                    and "fit" not in c["errors"]}
        loaded = {n[len(_PICKLED):] for n in fork[1]
                  if n.startswith(_PICKLED)}
        for name in sorted(eligible - loaded):
            result.differences.append(Difference(
                _PICKLED + name, "case", "no pickled model was loaded"))
    return result


def format_result(result):
    """The text ``compare`` prints."""
    lines = []
    for label, env in (("reference", result.ref_environment),
                       ("fork", result.fork_environment)):
        lines.append(
            f"{label}: {env['package']} {env['package_version']}, "
            f"Python {env['python']}, numpy {env['numpy']}, scipy "
            f"{env['scipy']}, scikit-learn {env['scikit_learn']}, "
            f"threadpoolctl {env['threadpoolctl']}, {env['machine']}, "
            f"threads {sorted(set(env['thread_variables'].values()))}")
    for reason in result.refused:
        lines.append(f"REFUSED: {reason}")
    if not result.refused:
        lines.append(
            f"compared {result.n_cases} cases, {result.n_arrays} arrays "
            "byte for byte")
        for exception, count in sorted(result.fired.items()):
            item, what = EXCEPTIONS[exception]
            lines.append(
                f"exception {exception} ({item}: {what}) excused {count} "
                f"difference(s) in {len(result.examples[exception])} "
                f"case(s), e.g. {result.examples[exception][0]}")
    for diff in result.differences:
        lines.append(f"DIFFERENCE {diff.case} | {diff.key} | {diff.text}")
    equal = ("equal apart from the named exceptions" if result.fired
             else "bit-equal")
    verdict = {0: equal, 1: f"{len(result.differences)} difference(s)",
               2: "not comparable"}[result.code]
    lines.append(f"result: {verdict} (exit {result.code})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command line.
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("dump")
    p.add_argument("--package", choices=sorted(DISTRIBUTIONS), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--load-pickles", metavar="REFDIR")
    p = sub.add_parser("compare")
    p.add_argument("ref")
    p.add_argument("fork")
    p.add_argument("--allow", default="")
    args = parser.parse_args(argv)
    if args.command == "dump":
        return dump(args.package, args.out, args.load_pickles)
    allow = [a for a in args.allow.split(",") if a]
    unknown = [a for a in allow if a not in EXCEPTIONS]
    if unknown:
        parser.error(f"unknown exception id(s): {unknown}")
    result = compare_dumps(args.ref, args.fork, allow)
    print(format_result(result))
    return result.code


if __name__ == "__main__":
    sys.exit(main())
