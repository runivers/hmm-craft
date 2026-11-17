"""
Tests of the bit-equality check itself.

A check that cannot fail is not a check: every test here builds two small
dumps with the script's own writer, in memory-sized numpy arrays and with no
package installed, perturbs one side by the smallest possible amount and
expects ``compare_dumps`` to name exactly that difference. The exceptions are
tested the other way round: each excuses its own shape of difference and
nothing else.

Run with ``python -m pytest scripts/test_bit_equality.py``.
"""

import copy
import json
import sys
import types

import bit_equality as be
import numpy as np
import pytest

LENGTHS = [4, 2]
NC, NF = 3, 2
ENVIRONMENT = {
    "package": "hmmlearn", "package_version": "0.3.3",
    "extension_sha256": "a", "script_sha256": "s", "python": "3.12.3",
    "numpy": "1.26.0", "scipy": "1.11.2", "scikit_learn": "1.3.1",
    "threadpoolctl": "3.8.0", "machine": "Linux x86_64",
    "thread_variables": {v: "1" for v in be.THREAD_VARIABLES},
    "threadpool_info": [], "pickles_loaded": False}
DATA_SHA256 = {"d": "0" * 64}
MASS = np.array([1.0, 2.0, 3.0])
ALL_BUT_E6 = tuple(e for e in be.EXCEPTIONS if e != "E6")


def _posteriors():
    rs = np.random.RandomState(0)
    return rs.dirichlet(np.ones(NC), size=sum(LENGTHS))


def _map_logprob(posteriors, logged):
    total = 0
    for part in np.split(posteriors, np.cumsum(LENGTHS))[:-1]:
        peak = np.max(part, axis=1)
        total += (np.log(peak) if logged else peak).sum()
    return np.asarray(total)


def fit_case(name, cov="diag", params="stmc", cls="GaussianHMM", nf=NF):
    """One fitted case of a Gaussian model, with the keys ``dump`` writes."""
    rs = np.random.RandomState(1)
    post = _posteriors()
    if cov == "spherical":
        covars, full = rs.rand(NC) + 1, np.tile(np.eye(nf), (NC, 1, 1))
    else:
        covars, full = rs.rand(NC, nf) + 1, np.tile(np.eye(nf), (NC, 1, 1))
    arrays = {
        "means_": rs.rand(NC, nf), "_covars_": covars, "covars_": full,
        "transmat_": rs.rand(NC, NC), "startprob_": rs.rand(NC),
        "monitor_history": rs.rand(3), "score": np.asarray(rs.rand()),
        "decode_viterbi_logprob": np.asarray(rs.rand()),
        "decode_viterbi_states": rs.randint(0, NC, sum(LENGTHS)),
        "decode_map_logprob": _map_logprob(post, logged=False),
        "decode_map_states": post.argmax(axis=1),
        "predict_proba": post,
        "sample_X": rs.rand(5, nf), "sample_states": rs.randint(0, NC, 5),
        "m01_post_sum_last": MASS.copy(), "m01_stat_last": MASS.copy(),
    }
    meta = {"name": name, "kind": "fit", "cls": cls, "cov": cov,
            "params": params, "data": "d", "lengths": LENGTHS,
            "n_features": nf, "index": 0, "stat_key": "post",
            "stat_defined": "m" in params}
    return {"meta": meta, "arrays": arrays, "errors": {},
            "params": {"n_iter": 10}}


def make_dump(path, cases, environment=None, data_sha256=None):
    be.write_dump(path, environment or copy.deepcopy(ENVIRONMENT),
                  data_sha256 or DATA_SHA256, cases)
    return path


def run(tmp_path, ref_cases, fork_cases, allow=(), **kwargs):
    ref = make_dump(tmp_path / "ref", ref_cases, kwargs.get("ref_env"),
                    kwargs.get("ref_data"))
    fork = make_dump(tmp_path / "fork", fork_cases, kwargs.get("fork_env"),
                     kwargs.get("fork_data"))
    return be.compare_dumps(ref, fork, allow)


def pair(**kwargs):
    ref = fit_case("c", **kwargs)
    return ref, copy.deepcopy(ref)


def with_occupancy(fork):
    fork["arrays"]["state_occupancy_"] = MASS.copy()
    return fork


def test_identical_dumps_are_equal(tmp_path):
    ref, fork = pair()
    result = run(tmp_path, [ref], [fork])
    assert result.code == 0
    assert result.differences == []
    assert result.n_arrays > 10
    assert not result.fired


def test_one_ulp_in_one_array_fails(tmp_path):
    ref, fork = pair()
    value = fork["arrays"]["means_"]
    value[1, 0] = np.nextafter(value[1, 0], np.inf)
    result = run(tmp_path, [ref], [fork])
    assert result.code == 1
    assert [(d.case, d.key) for d in result.differences] == [("c", "means_")]
    text = result.differences[0].text
    assert "(1, 0)" in text
    assert "np." not in text


def test_one_flipped_state_fails(tmp_path):
    ref, fork = pair()
    states = fork["arrays"]["decode_viterbi_states"]
    states[2] = (states[2] + 1) % NC
    result = run(tmp_path, [ref], [fork])
    assert result.code == 1
    assert [d.key for d in result.differences] == ["decode_viterbi_states"]


def test_shape_change_outside_an_exception_fails(tmp_path):
    ref, fork = pair(cov="diag")
    fork["arrays"]["_covars_"] = fork["arrays"]["_covars_"].reshape(NF, NC)
    result = run(tmp_path, [ref], [fork], allow=("E1", "E2", "E3"))
    assert result.code == 1
    assert [d.key for d in result.differences] == ["_covars_"]


def test_dtype_change_fails(tmp_path):
    ref, fork = pair()
    fork["arrays"]["means_"] = fork["arrays"]["means_"].astype(np.float32)
    result = run(tmp_path, [ref], [fork])
    assert [d.key for d in result.differences] == ["means_"]


def test_nan_is_compared_by_its_bits(tmp_path):
    ref, fork = pair()
    ref["arrays"]["score"] = np.asarray(np.nan)
    fork["arrays"]["score"] = np.asarray(np.nan)
    assert run(tmp_path, [ref], [fork]).code == 0
    other = np.asarray(np.nan).view(np.uint64) | np.uint64(1)
    fork["arrays"]["score"] = other.view(np.float64)
    result = run(tmp_path / "again", [ref], [fork])
    assert [d.key for d in result.differences] == ["score"]


def test_missing_key_fails(tmp_path):
    ref, fork = pair()
    del fork["arrays"]["transmat_"]
    result = run(tmp_path, [ref], [fork], allow=ALL_BUT_E6)
    assert result.code == 1
    assert [d.key for d in result.differences] == ["transmat_"]


def test_extra_fork_key_fails_unless_named(tmp_path):
    ref, fork = pair()
    fork["arrays"]["posterior_mass_"] = np.ones(NC)
    result = run(tmp_path, [ref], [fork], allow=ALL_BUT_E6)
    assert [d.key for d in result.differences] == ["posterior_mass_"]


def test_extra_fork_parameter_fails_unless_named(tmp_path):
    ref, fork = pair()
    with_occupancy(fork)
    fork["params"]["covars_floor"] = None
    assert run(tmp_path, [ref], [fork]).code == 1
    assert run(tmp_path / "e6", [ref], [fork], allow=("E6",)).code == 0
    fork["params"]["covars_floor"] = 0.5
    assert run(tmp_path / "set", [ref], [fork], allow=("E6",)).code == 1


def test_state_occupancy_is_checked_against_the_recorded_sums(tmp_path):
    ref, fork = pair()
    with_occupancy(fork)
    assert run(tmp_path, [ref], [fork]).code == 1
    assert run(tmp_path / "e6", [ref], [fork], allow=("E6",)).code == 0
    value = fork["arrays"]["state_occupancy_"]
    value[2] = np.nextafter(value[2], np.inf)
    result = run(tmp_path / "ulp", [ref], [fork], allow=("E6",))
    assert result.code == 1
    assert {d.key for d in result.differences} == {"state_occupancy_"}
    del fork["arrays"]["state_occupancy_"]
    result = run(tmp_path / "gone", [ref], [fork], allow=("E6",))
    assert [d.key for d in result.differences] == ["state_occupancy_"]


def test_state_occupancy_skips_the_class_statistic_where_it_is_undefined(
        tmp_path):
    ref, fork = pair(params="stc")
    ref["arrays"]["m01_stat_last"] = np.zeros(NC)
    fork["arrays"]["m01_stat_last"] = np.zeros(NC)
    with_occupancy(fork)
    assert run(tmp_path, [ref], [fork], allow=("E6",)).code == 0


def test_exception_not_allowed_fails(tmp_path):
    ref, fork = _spherical_pair()
    result = run(tmp_path, [ref], [fork])
    assert result.code == 1
    assert {d.key for d in result.differences} == {"_covars_", "covars_"}
    assert run(tmp_path / "e1", [ref], [fork], allow=("E1",)).code == 0
    assert run(tmp_path / "e2", [ref], [fork], allow=("E2", "E3")).code == 1


def _spherical_pair(nf=3):
    """Reference with 0.3.3's spherical layouts, fork with the documented."""
    ref = fit_case("c", cov="spherical", nf=nf)
    fork = copy.deepcopy(ref)
    variances = np.array([0.5, 1.5, 2.5])
    ref["arrays"]["_covars_"] = np.tile(variances[:, None], (1, nf))
    ref["arrays"]["covars_"] = np.concatenate(
        [v * np.eye(nf)[None] for v in variances.repeat(nf)])
    fork["arrays"]["_covars_"] = variances
    fork["arrays"]["covars_"] = variances[:, None, None] * np.eye(nf)
    return ref, fork


def test_exception_applies_only_where_named(tmp_path):
    ref, fork = pair(cov="diag")
    fork["arrays"]["_covars_"] = fork["arrays"]["_covars_"][:, 0]
    result = run(tmp_path / "diag", [ref], [fork], allow=("E1",))
    assert [d.key for d in result.differences] == ["_covars_"]
    ref, fork = _spherical_pair()
    result = run(tmp_path / "spherical", [ref], [fork], allow=("E1",))
    assert result.code == 0
    assert result.fired["E1"] == 2
    column = fork["arrays"]["_covars_"]
    column[1] = np.nextafter(column[1], np.inf)
    result = run(tmp_path / "ulp", [ref], [fork], allow=("E1",))
    assert {d.key for d in result.differences} == {"_covars_"}


def test_e1_needs_every_column_to_hold_the_variance(tmp_path):
    ref, fork = _spherical_pair()
    value = ref["arrays"]["_covars_"]
    value[1, 2] = np.nextafter(value[1, 2], np.inf)
    result = run(tmp_path, [ref], [fork], allow=("E1",))
    assert "_covars_" in {d.key for d in result.differences}


def test_e1_on_covars_takes_the_first_matrix_of_each_state(tmp_path):
    ref, fork = _spherical_pair()
    ref["arrays"]["covars_"][3] = 9 * np.eye(3)
    result = run(tmp_path, [ref], [fork], allow=("E1",))
    assert {d.key for d in result.differences} == {"covars_"}


def test_e2_excuses_only_rows_of_the_first_state(tmp_path):
    ref, fork = _spherical_pair()
    states = np.array([0, 1, 0, 2, 1])
    for side in (ref, fork):
        side["arrays"]["sample_states"] = states
    ref["arrays"]["sample_X"] = np.arange(15.0).reshape(5, 3)
    fork["arrays"]["sample_X"] = ref["arrays"]["sample_X"].copy()
    fork["arrays"]["sample_X"][[1, 3, 4]] += 1
    result = run(tmp_path, [ref], [fork], allow=("E1", "E2"))
    assert result.code == 0
    assert result.fired["E2"] == 1
    fork["arrays"]["sample_X"][2, 0] += 1
    result = run(tmp_path / "state0", [ref], [fork], allow=("E1", "E2"))
    assert [d.key for d in result.differences] == ["sample_X"]
    fork["arrays"]["sample_X"][2, 0] = ref["arrays"]["sample_X"][2, 0]
    fork["arrays"]["sample_states"] = states[::-1].copy()
    result = run(tmp_path / "states", [ref], [fork], allow=("E1", "E2"))
    assert "sample_states" in {d.key for d in result.differences}


def test_map_rule_is_exact(tmp_path):
    ref, fork = pair()
    post = ref["arrays"]["predict_proba"]
    fork["arrays"]["decode_map_logprob"] = _map_logprob(post, logged=True)
    result = run(tmp_path / "ok", [ref], [fork], allow=("E3",))
    assert result.code == 0
    assert result.fired["E3"] == 1
    value = fork["arrays"]["decode_map_logprob"]
    fork["arrays"]["decode_map_logprob"] = np.nextafter(value, np.inf)
    result = run(tmp_path / "fork", [ref], [fork], allow=("E3",))
    assert [d.key for d in result.differences] == ["decode_map_logprob"]
    fork["arrays"]["decode_map_logprob"] = _map_logprob(post, logged=True)
    value = ref["arrays"]["decode_map_logprob"]
    ref["arrays"]["decode_map_logprob"] = np.nextafter(value, np.inf)
    result = run(tmp_path / "ref", [ref], [fork], allow=("E3",))
    assert [d.key for d in result.differences] == ["decode_map_logprob"]


def test_map_rule_does_not_excuse_a_changed_state_sequence(tmp_path):
    ref, fork = pair()
    fork["arrays"]["decode_map_logprob"] = _map_logprob(
        ref["arrays"]["predict_proba"], logged=True)
    states = fork["arrays"]["decode_map_states"]
    states[0] = (states[0] + 1) % NC
    result = run(tmp_path, [ref], [fork], allow=("E3",))
    assert [d.key for d in result.differences] == ["decode_map_states"]


def test_e4_needs_a_fit_that_does_not_raise_and_is_finite(tmp_path):
    ref, fork = pair(params="stc")
    ref["errors"] = {"fit": ["ValueError", "array must not contain infs"]}
    ref["arrays"] = {"means_": ref["arrays"]["means_"]}
    ref["params"] = None
    assert run(tmp_path / "off", [ref], [fork]).code == 1
    result = run(tmp_path / "on", [ref], [fork], allow=("E4",))
    assert result.code == 0
    assert result.fired["E4"] == 1
    fork["arrays"]["_covars_"][0, 0] = np.inf
    result = run(tmp_path / "inf", [ref], [fork], allow=("E4",))
    assert result.code == 1
    fork["arrays"]["_covars_"][0, 0] = 1.0
    fork["errors"] = {"fit": ["ValueError", "array must not contain infs"]}
    result = run(tmp_path / "raised", [ref], [fork], allow=("E4",))
    assert result.code == 1


def test_e4_applies_only_where_covariances_are_fitted_without_means(tmp_path):
    ref, fork = pair(params="stmc")
    fork["arrays"]["means_"] = fork["arrays"]["means_"] + 1
    result = run(tmp_path / "stmc", [ref], [fork], allow=("E4",))
    assert [d.key for d in result.differences] == ["means_"]
    ref, fork = pair(params="st")
    fork["arrays"]["means_"] = fork["arrays"]["means_"] + 1
    assert run(tmp_path / "st", [ref], [fork], allow=("E4",)).code == 1


def test_e4_stays_silent_when_the_two_fits_are_equal(tmp_path):
    ref, fork = pair(params="stc")
    result = run(tmp_path, [ref], [fork], allow=("E4",))
    assert result.code == 0
    assert not result.fired


def test_same_exception_type_required_when_the_reference_raised(tmp_path):
    ref, fork = pair()
    for side in (ref, fork):
        side["errors"] = {"fit": ["ValueError", "boom"]}
        side["params"] = None
    assert run(tmp_path / "same", [ref], [fork]).code == 0
    fork["errors"] = {"fit": ["RuntimeError", "boom"]}
    result = run(tmp_path / "type", [ref], [fork])
    assert result.code == 1
    assert result.differences[0].key == "errors:fit"
    fork["errors"] = {}
    result = run(tmp_path / "none", [ref], [fork])
    assert result.code == 1


def test_a_probe_that_raises_on_one_side_only_fails(tmp_path):
    ref, fork = pair()
    fork["errors"] = {"decode_map": ["ValueError", "x"]}
    del fork["arrays"]["decode_map_states"]
    del fork["arrays"]["decode_map_logprob"]
    result = run(tmp_path, [ref], [fork])
    assert result.code == 1
    assert "errors:decode_map" in {d.key for d in result.differences}


def _normalize_pair(axis, zero):
    source = np.array([[1.0, 0.0, 2.0], [3.0, 0.0, 2.0]])
    expected = np.array([[0.25, 0.0, 0.5], [0.75, 0.0, 0.5]])
    ref_result = expected.copy()
    ref_result[:, 1] = np.nan if zero == "nan" else 0.0
    kind = {"name": "n", "kind": "normalize", "options": {"axis": axis},
                "dtype": "float64"}
    ref = {"meta": kind, "arrays": {"input": source, "result": ref_result},
               "errors": {}, "params": None}
    fork = {"meta": kind, "arrays": {"input": source, "result": expected},
                "errors": {}, "params": None}
    return ref, fork


def test_e5_excuses_a_zero_slice_on_axis_zero_only(tmp_path):
    ref, fork = _normalize_pair(0, "nan")
    assert run(tmp_path / "off", [ref], [fork]).code == 1
    result = run(tmp_path / "on", [ref], [fork], allow=("E5",))
    assert result.code == 0
    assert result.fired["E5"] == 1
    ref, fork = _normalize_pair(1, "nan")
    assert run(tmp_path / "axis1", [ref], [fork], allow=("E5",)).code == 1
    ref, fork = _normalize_pair(0, "nan")
    fork["arrays"]["result"][0, 0] = 0.5
    assert run(tmp_path / "other", [ref], [fork], allow=("E5",)).code == 1
    ref, fork = _normalize_pair(0, "zero")
    assert run(tmp_path / "equal", [ref], [fork], allow=("E5",)).code == 0


def test_e1_applies_to_fill_covars_results(tmp_path):
    variances = np.array([0.1, 0.2])
    source = np.tile(variances[:, None], (1, 3))
    meta = {"name": "f", "kind": "fill_covars", "dtype": "float64",
                "options": {"covariance_type": "spherical", "n_components": 2,
                             "n_features": 3}}
    ref_result = np.concatenate(
        [v * np.eye(3)[None] for v in variances.repeat(3)])
    fork_result = variances[:, None, None] * np.eye(3)
    ref = {"meta": meta, "arrays": {"input": source, "result": ref_result},
               "errors": {}, "params": None}
    fork = {"meta": meta, "arrays": {"input": source, "result": fork_result},
                "errors": {}, "params": None}
    assert run(tmp_path / "off", [ref], [fork]).code == 1
    assert run(tmp_path / "on", [ref], [fork], allow=("E1",)).code == 0
    fork_result[1] = np.nextafter(fork_result[1], np.inf)
    assert run(tmp_path / "ulp", [ref], [fork], allow=("E1",)).code == 1


def test_pickled_case_is_compared_with_its_reference_case(tmp_path):
    ref, fork = pair()
    fork_pickled = copy.deepcopy(fork)
    fork_pickled["meta"] = dict(fork["meta"], name="pickled:c", kind="pickled")
    fork_pickled["arrays"] = {
        k: v for k, v in fork["arrays"].items()
        if k in ("score", "predict_proba", "covars_")}
    result = run(tmp_path / "ok", [ref], [fork, fork_pickled])
    assert result.code == 0
    value = fork_pickled["arrays"]["score"]
    fork_pickled["arrays"]["score"] = np.nextafter(value, np.inf)
    result = run(tmp_path / "ulp", [ref], [fork, fork_pickled])
    assert [(d.case, d.key) for d in result.differences] == [
        ("pickled:c", "score")]


def test_environment_mismatch_refuses(tmp_path):
    ref, fork = pair()
    env = copy.deepcopy(ENVIRONMENT)
    env["numpy"] = "2.5.4"
    result = run(tmp_path, [ref], [fork], fork_env=env)
    assert result.code == 2
    assert any("numpy" in text for text in result.refused)


@pytest.mark.parametrize("field", ["python", "scipy", "scikit_learn",
                                   "threadpoolctl", "script_sha256"])
def test_every_version_and_the_script_must_match(tmp_path, field):
    ref, fork = pair()
    env = copy.deepcopy(ENVIRONMENT)
    env[field] = "other"
    assert run(tmp_path, [ref], [fork], fork_env=env).code == 2


def test_more_than_one_thread_refuses(tmp_path):
    ref, fork = pair()
    env = copy.deepcopy(ENVIRONMENT)
    env["thread_variables"]["OMP_NUM_THREADS"] = "4"
    assert run(tmp_path, [ref], [fork], fork_env=env).code == 2


def test_different_inputs_refuse(tmp_path):
    ref, fork = pair()
    result = run(tmp_path, [ref], [fork], fork_data={"d": "1" * 64})
    assert result.code == 2


def test_different_case_lists_refuse(tmp_path):
    ref, fork = pair()
    other = fit_case("other")
    assert run(tmp_path, [ref, other], [fork]).code == 2


def test_dump_refuses_to_run_with_threads_unpinned(tmp_path, monkeypatch):
    for name in be.THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    with pytest.raises(SystemExit, match="refusing to run"):
        be.dump("hmmcraft", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_cli_exit_status_follows_the_result(tmp_path, capsys):
    ref, fork = pair()
    ref_dir = make_dump(tmp_path / "ref", [ref])
    fork["arrays"]["means_"][0, 0] += 1
    fork_dir = make_dump(tmp_path / "fork", [fork])
    assert be.main(["compare", str(ref_dir), str(fork_dir)]) == 1
    out = capsys.readouterr().out
    assert "means_" in out
    with pytest.raises(SystemExit):
        be.main(["compare", str(ref_dir), str(fork_dir), "--allow", "E9"])


def test_data_are_reproducible_and_pure_numpy():
    first, second = be.make_data(), be.make_data()
    assert set(first) == {
        "gauss3", "gauss1", "collapse", "flat", "gmm", "cat", "multi", "pois"}
    for name in first:
        assert first[name]["X"].tobytes() == second[name]["X"].tobytes()
        assert first[name]["lengths"].sum() == len(first[name]["X"])
    rows = first["collapse"]["X"][be._COLLAPSE_ROWS]
    assert (rows == rows[0]).all()
    assert not (first["gauss3"]["X"][be._COLLAPSE_ROWS] == rows[0]).all()


def test_cases_have_unique_names_and_cover_the_table():
    names = [c["name"] for c in be.build_cases()]
    assert len(names) == len(set(names))
    for cov in be.COVARIANCE_TYPES:
        assert f"GaussianHMM|gauss3|{cov}|fixed|log|stc" in names
    assert any(n.startswith("VariationalCategoricalHMM") for n in names)
    assert json.dumps(be.build_cases())


# A computation that takes the rows of many sequences in blocks of up to this
# many rows takes several on the short-sequence family.
BLOCK_ROWS = 4096
N_TWO_SEQUENCE_CASES = 374


def _short_cases():
    return [(index, c) for index, c in enumerate(be.build_cases())
            if c["data"].startswith("short-")]


def test_short_sequence_family_covers_every_class():
    cases = be.build_cases()
    short = _short_cases()
    assert short
    assert {c["cls"] for _, c in short} == set(be._CLASSES)
    for cls in be._COVARIANCE_CLASSES:
        assert {c["cov"] for _, c in short if c["cls"] == cls} == set(
            be.COVARIANCE_TYPES)
    assert {c["impl"] for _, c in short} == set(be.IMPLEMENTATIONS)
    # Appended after every two-sequence case, so those keep their numbers.
    assert [i for i, _ in short] == list(
        range(N_TWO_SEQUENCE_CASES, len(cases)))
    assert not any(c["data"].startswith("short-")
                   for c in cases[:N_TWO_SEQUENCE_CASES])
    names = [c["name"] for c in cases]
    assert len(names) == len(set(names))
    state = np.random.get_state()
    first, second = be.make_short_data(), be.make_short_data()
    after = np.random.get_state()
    assert all(_same_state(a, b) for a, b in zip(state, after))
    assert {c["data"] for _, c in short} == set(first)
    for name, family in first.items():
        assert name.startswith("short-")
        lengths = family["lengths"]
        assert 1 in lengths
        assert len(lengths) == 220
        assert lengths.sum() > BLOCK_ROWS
        assert lengths.sum() == len(family["X"])
        assert family["X"].tobytes() == second[name]["X"].tobytes()
        assert lengths.tobytes() == second[name]["lengths"].tobytes()


def test_one_ulp_in_a_short_sequence_case_fails(tmp_path):
    gaussian = [(i, c) for i, c in _short_cases()
                if c["cls"] == "GaussianHMM"]
    assert gaussian
    index, recipe = gaussian[0]
    lengths = np.asarray(be.SHORT_LENGTHS)
    ref = fit_case(recipe["name"], cov=recipe["cov"],
                   params=recipe["params"], nf=3)
    ref["meta"] = dict(ref["meta"], **recipe, index=index,
                       lengths=lengths.tolist(), n_features=3)
    ref["arrays"]["predict_proba"] = np.random.RandomState(0).dirichlet(
        np.ones(NC), size=int(lengths.sum()))
    fork = copy.deepcopy(ref)
    old = fit_case("old")
    starts = np.cumsum(lengths) - lengths
    row = int(starts[(lengths == 1) & (starts >= BLOCK_ROWS)][0])
    value = fork["arrays"]["predict_proba"]
    value[row, 0] = np.nextafter(value[row, 0], np.inf)
    for allow in ((), ALL_BUT_E6):
        result = run(tmp_path / str(len(allow)), [old, ref],
                     [copy.deepcopy(old), fork], allow=allow)
        assert result.code == 1
        assert [(d.case, d.key) for d in result.differences] == [
            (recipe["name"], "predict_proba")]
        assert f"({row}, 0)" in result.differences[0].text


def test_threadpool_info_is_recorded_in_a_fixed_order(monkeypatch, tmp_path):
    # threadpoolctl lists the loaded libraries in the order they were found,
    # which changes between runs; two dumps of the same code must not.
    import threadpoolctl

    libraries = [
        {"filepath": path, "internal_api": api, "num_threads": 1}
        for path, api in [("/lib/b.so", "openblas"), ("/lib/a.so", "openmp"),
                          ("/lib/c.so", "openblas")]]
    extension = tmp_path / "_hmmc.so"
    extension.write_bytes(b"x")
    package = types.ModuleType("fakepackage")
    package._hmmc = types.ModuleType("fakepackage._hmmc")
    package._hmmc.__file__ = str(extension)
    monkeypatch.setitem(sys.modules, "fakepackage", package)
    monkeypatch.setitem(sys.modules, "fakepackage._hmmc", package._hmmc)
    monkeypatch.setitem(be.DISTRIBUTIONS, "fakepackage", "numpy")
    recorded = []
    for order in (libraries, libraries[::-1], libraries[1:] + libraries[:1]):
        monkeypatch.setattr(threadpoolctl, "threadpool_info",
                            lambda order=order: list(order))
        recorded.append(be._environment("fakepackage", {})["threadpool_info"])
    assert recorded[0] == recorded[1] == recorded[2]
    assert sorted(lib["filepath"] for lib in libraries) == [
        lib["filepath"] for lib in recorded[0]]


def test_verdict_says_bit_equal_only_when_no_exception_fired(tmp_path):
    ref, fork = pair()
    result = run(tmp_path / "none", [ref], [fork], allow=("E3",))
    assert result.code == 0 and not result.fired
    assert be.format_result(result).endswith("result: bit-equal (exit 0)")
    fork["arrays"]["decode_map_logprob"] = _map_logprob(
        ref["arrays"]["predict_proba"], logged=True)
    result = run(tmp_path / "fired", [ref], [fork], allow=("E3",))
    assert result.code == 0 and result.fired["E3"] == 1
    last = be.format_result(result).splitlines()[-1]
    assert last == "result: equal apart from the named exceptions (exit 0)"


# E7: a sequence that underflows with implementation="scaling" ---------------

UNDERFLOW = [
    "ValueError",
    (
        "forward pass failed with underflow; consider "
        "using implementation='log' instead"
    ),
]
ONE_SEQUENCE = [6]
E7_PREDICT_ERRORS = ("score", "decode_map", "predict_proba")


def _underflow_meta(step, expect="fallback", kind="underflow"):
    return {
        "name": "u",
        "kind": kind,
        "step": step,
        "cls": "GaussianHMM",
        "data": "d",
        "clean": None,
        "cov": "diag",
        "start": "fixed",
        "impl": "scaling",
        "params": None,
        "tol": "default",
        "expect": expect,
        "lengths": ONE_SEQUENCE,
        "n_features": NF,
    }


def underflow_predict_pair():
    """
    A prediction case on one sequence: the log twin under the plain keys,
    the scaling run under ``scaling:``. The reference raises the underflow
    error in each scaled step but Viterbi; the fork returns the log twin.
    """
    rs = np.random.RandomState(2)
    post = rs.dirichlet(np.ones(NC), size=ONE_SEQUENCE[0])
    viterbi = {
        "decode_viterbi_logprob": np.asarray(rs.rand()),
        "decode_viterbi_states": rs.randint(0, NC, ONE_SEQUENCE[0]),
    }
    twin = dict(
        viterbi,
        score=np.asarray(rs.rand()),
        decode_map_states=post.argmax(axis=1),
        predict_proba=post,
    )
    ref_arrays = dict(
        twin, decode_map_logprob=be._map_sum(post, ONE_SEQUENCE, logged=False)
    )
    fork_arrays = dict(
        twin, decode_map_logprob=be._map_sum(post, ONE_SEQUENCE, logged=True)
    )
    for key, value in viterbi.items():
        ref_arrays["scaling:" + key] = value.copy()
    fork_arrays.update(
        {
            "scaling:" + key: np.copy(value)
            for key, value in list(fork_arrays.items())
        }
    )
    ref = {
        "meta": _underflow_meta("predict"),
        "arrays": ref_arrays,
        "errors": {"scaling:" + s: list(UNDERFLOW) for s in E7_PREDICT_ERRORS},
        "params": None,
    }
    fork = {
        "meta": _underflow_meta("predict"),
        "arrays": copy.deepcopy(fork_arrays),
        "errors": {},
        "params": None,
    }
    return ref, fork


def underflow_fit_pair():
    """A fit case: the reference raised the underflow error, the fork fit."""
    fitted = fit_case("u")
    ref = {
        "meta": _underflow_meta("fit"),
        "arrays": {
            "means_": fitted["arrays"]["means_"].copy(),
            "monitor_history": np.zeros(0),
        },
        "errors": {"fit": list(UNDERFLOW)},
        "params": None,
    }
    fork = with_occupancy(
        dict(copy.deepcopy(fitted), meta=_underflow_meta("fit"))
    )
    return ref, fork


def test_e7_excuses_a_prediction_equal_to_the_log_twin(tmp_path):
    ref, fork = underflow_predict_pair()
    result = run(tmp_path / "off", [ref], [fork], allow=("E3",))
    assert result.code == 1
    result = run(tmp_path / "on", [ref], [fork], allow=("E3", "E7"))
    assert result.code == 0, result.differences
    assert result.fired["E7"] == 7
    assert result.fired["E3"] == 1


@pytest.mark.parametrize(
    "key",
    [
        "scaling:score",
        "scaling:predict_proba",
        "scaling:decode_map_logprob",
        "scaling:decode_map_states",
    ],
)
def test_e7_fails_on_one_ulp_or_one_state(tmp_path, key):
    ref, fork = underflow_predict_pair()
    value = fork["arrays"][key]
    if value.dtype.kind == "f":
        value.reshape(-1)[0] = np.nextafter(value.reshape(-1)[0], np.inf)
    else:
        value[0] = (value[0] + 1) % NC
    result = run(tmp_path, [ref], [fork], allow=("E3", "E7"))
    assert result.code == 1
    keys = {d.key for d in result.differences}
    assert "E7" in keys and key in keys
    assert any(key in d.text for d in result.differences if d.key == "E7")


@pytest.mark.parametrize("reference", ["value", "text", "type"])
def test_e7_needs_the_reference_to_raise_the_underflow_error(
    tmp_path, reference
):
    ref, fork = underflow_predict_pair()
    if reference == "value":
        del ref["errors"]["scaling:score"]
        ref["arrays"]["scaling:score"] = fork["arrays"]["scaling:score"]
    elif reference == "text":
        ref["errors"]["scaling:score"] = ["ValueError", "shape mismatch"]
    else:
        ref["errors"]["scaling:score"] = ["RuntimeError", UNDERFLOW[1]]
    result = run(tmp_path, [ref], [fork], allow=("E3", "E7"))
    assert result.code == 1
    assert "E7" in {d.key for d in result.differences}


def test_e7_fails_when_the_fork_raises_as_before(tmp_path):
    ref, fork = underflow_predict_pair()
    fork["errors"] = copy.deepcopy(ref["errors"])
    for step in (
        "score",
        "predict_proba",
        "decode_map_logprob",
        "decode_map_states",
    ):
        del fork["arrays"]["scaling:" + step]
    result = run(tmp_path, [ref], [fork], allow=("E3", "E7"))
    assert result.code == 1
    assert "E7" in {d.key for d in result.differences}
    assert not result.fired["E7"]


def test_e7_fails_on_a_non_finite_value_equal_to_its_twin(tmp_path):
    ref, fork = underflow_predict_pair()
    for arrays in (ref["arrays"], fork["arrays"]):
        arrays["score"] = np.asarray(-np.inf)
    fork["arrays"]["scaling:score"] = np.asarray(-np.inf)
    result = run(tmp_path, [ref], [fork], allow=("E3", "E7"))
    assert result.code == 1
    assert {d.key for d in result.differences} == {
        "E7",
        "scaling:score",
        "errors:scaling:score",
    }


def test_e7_excuses_a_fit_that_returns_finite(tmp_path):
    ref, fork = underflow_fit_pair()
    assert run(tmp_path / "off", [ref], [fork], allow=("E6",)).code == 1
    result = run(tmp_path / "on", [ref], [fork], allow=("E6", "E7"))
    assert result.code == 0, result.differences
    assert result.fired["E7"] == 1


@pytest.mark.parametrize("change", ["fork-raised", "fork-inf", "other-text"])
def test_e7_fit_fails_outside_its_shape(tmp_path, change):
    ref, fork = underflow_fit_pair()
    if change == "fork-raised":
        fork["errors"] = {"fit": list(UNDERFLOW)}
    elif change == "fork-inf":
        fork["arrays"]["means_"][0, 0] = np.inf
    else:
        ref["errors"] = {"fit": ["ValueError", "shape mismatch"]}
    result = run(tmp_path, [ref], [fork], allow=("E6", "E7"))
    assert result.code == 1
    assert "E7" in {d.key for d in result.differences}
    assert not result.fired["E7"]


def test_e7_does_not_apply_to_an_ordinary_fit(tmp_path):
    ref, fork = underflow_fit_pair()
    for side in (ref, fork):
        side["meta"] = dict(
            side["meta"],
            kind="fit",
            stat_key="post",
            stat_defined=True,
            index=0,
        )
    result = run(tmp_path, [ref], [fork], allow=tuple(be.EXCEPTIONS))
    assert result.code == 1
    assert not result.fired["E7"]


def test_e7_does_not_apply_where_the_error_is_expected(tmp_path):
    ref, fork = underflow_predict_pair()
    for side in (ref, fork):
        side["meta"] = dict(side["meta"], expect="error")
    fork["errors"] = copy.deepcopy(ref["errors"])
    fork["arrays"] = copy.deepcopy(ref["arrays"])
    for allow in ((), tuple(be.EXCEPTIONS)):
        assert (
            run(
                tmp_path / f"same{len(allow)}", [ref], [fork], allow=allow
            ).code
            == 0
        )
    ref, fork = underflow_predict_pair()
    for side in (ref, fork):
        side["meta"] = dict(side["meta"], expect="error")
    for allow in ((), tuple(be.EXCEPTIONS)):
        result = run(
            tmp_path / f"fork{len(allow)}", [ref], [fork], allow=allow
        )
        assert result.code == 1
        assert not result.fired["E7"]


def test_underflow_cases_have_unique_names_and_cover_the_classes():
    cases = be.build_underflow_cases()
    names = [c["name"] for c in cases]
    assert len(cases) == 33
    assert len(names) == len(set(names))
    assert not set(names) & {c["name"] for c in be.build_cases()}
    assert len(be.build_cases()) == 428
    assert len(be.build_cases()) - len(_short_cases()) == 374
    assert {c["cls"] for c in cases} == set(be._CLASSES)
    for cls in be._COVARIANCE_CLASSES:
        assert {c["cov"] for c in cases if c["cls"] == cls} == set(
            be.COVARIANCE_TYPES
        )
    assert {c["step"] for c in cases} == {"predict", "fit"}
    assert sum(c["expect"] == "error" for c in cases) == 2
    assert all(
        c["kind"] == "underflow" and c["impl"] == "scaling" for c in cases
    )
    assert json.dumps(cases)


def test_underflow_data_differ_from_their_source_in_one_row():
    data = be.make_data()
    before = {name: d["X"].tobytes() for name, d in data.items()}
    state = np.random.get_state()
    far = be.make_underflow_data(data)
    after = np.random.get_state()
    assert all(_same_state(a, b) for a, b in zip(state, after))
    assert {name: d["X"].tobytes() for name, d in data.items()} == before
    assert set(far) == {"far-gauss3", "far-gmm", "far-pois"}
    for name, family in far.items():
        source = data[name[len("far-") :]]
        changed = (family["X"] != source["X"]).any(axis=1)
        assert changed.nonzero()[0].tolist() == [be._FAR_ROW]
        assert family["lengths"].tobytes() == source["lengths"].tobytes()
        assert family["truth"] is source["truth"]


def _same_state(a, b):
    return np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b


def test_ci_script_allows_every_exception():
    script = (be.Path(be.__file__).parent / "ci_bit_equality.sh").read_text()
    (line,) = [x for x in script.splitlines() if x.startswith("allow=")]
    assert line.split("=", 1)[1].split(",") == list(be.EXCEPTIONS)
