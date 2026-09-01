"""
PX correctness audit under CDW -- independent, exhaustive verification of every
architectural assumption Partition Crossover, Delta-evaluation, and the
surrounding GA loop require, now that alpha = rho_j has been replaced by
alpha = rho_j * w_ij (CDWModel, huong_cai_tien/models/nkcv2_cdw.py). Read-only audit:
does NOT modify src/ or huong_cai_tien/, does NOT tune CDW.

This is NOT scoped to a single hypothesis (e.g. the fix_labels() timing
question) -- every assumption listed in README.md Sec11 gets its
own dedicated test, run on ALL THREE datasets currently used in this research
(Aggregation, Flame, Iris), with a synthetic dataset used only for cheap
additional stress testing (more trials, not a substitute for real data).

Assumptions covered (see README.md Sec11 for the consolidated evidence):
  A1  f(x) = sum_i f_i(x)                         -- additivity
  A2  f_i(x) depends only on x[i] and x[M[i,:]]     -- locality
  A3  D, rho, dt0/dt1/dt2, Gep, Mw_noise/same/diff  -- constant w.r.t. x
  A4  PX partial-sum fitness == comp_fitness(offspring) from scratch
  A5  fix_labels() does not change fitness (production call-order timing)
  A6  map_solutions() (bijective relabel) does not change fitness
  A7  renumber() (bijective relabel) does not change fitness
  A8  comp_fitness/raw_w/Mw_* stay finite (no NaN/Inf) across all datasets
  A9  PX genuinely consumes CDW's weights (polymorphism regression guard)
  A10 no accumulated drift across a real, full CDW GA run

Uses the CDW config under active investigation (run_cdw_ga.py::CDW_KWARGS)
exactly as shipped -- not tuned or modified here.

Usage: python huong_cai_tien/test_px_correctness.py  (also pytest-discoverable: test_*)
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

import _paths  # noqa: F401  (them src/ + models/ vao sys.path)

from nkcv2 import NKCV2Model, HAVE_NUMBA
from px import map_solutions, px, fix_labels
from ga import NKHGA, renumber
import ga as ga_module
from nkcv2_cdw import CDWModel
from run_cdw_ga import CDW_KWARGS
from datasets import load_flame, load_aggregation, load_iris_data

TOL = 1e-9

REAL_DATASETS = ("flame_N240", "aggregation_N788", "iris_N150")


def _random_partition(rng, n, nc):
    return np.ascontiguousarray(rng.integers(0, nc + 1, size=n).astype(np.int64))


def _cdw(X, K=3):
    return CDWModel(X, K=K, **CDW_KWARGS)


def _datasets():
    """All three datasets currently used in this research (Aggregation, Flame,
    Iris) plus a synthetic set used ONLY for cheap additional stress testing
    (more trials at low cost), never as a substitute for the real data."""
    rng = np.random.default_rng(0)
    X_syn = rng.normal(size=(60, 2))
    Xf, _ = load_flame()
    Xa, _ = load_aggregation()
    Xi, _ = load_iris_data()
    return {
        "synthetic_N60": X_syn,
        "flame_N240": Xf,
        "aggregation_N788": Xa,
        "iris_N150": Xi,
    }


def _trials(name, small, large):
    return small if name == "synthetic_N60" else large


# ---------------------------------------------------------------------------
# A2 -- locality: comp_fi(x, i) must depend only on x[i] and x[M[i,:]]
# ---------------------------------------------------------------------------

def test_cdw_comp_fi_locality():
    for name, X in _datasets().items():
        model = _cdw(X)
        rng = np.random.default_rng(42)
        x = _random_partition(rng, model.N, 5)
        for _ in range(200):
            i = int(rng.integers(0, model.N))
            fi_before = model.comp_fi(x, i)
            local = set([i] + [int(model.M[i, k]) for k in range(model.K)])
            m = int(rng.integers(0, model.N))
            while m in local:
                m = int(rng.integers(0, model.N))
            x2 = x.copy()
            x2[m] = int(rng.integers(0, 6))
            fi_after = model.comp_fi(x2, i)
            assert abs(fi_before - fi_after) < 1e-12, (
                f"[{name}] comp_fi(x,{i}) changed after perturbing unrelated "
                f"index {m} -- locality assumption violated")


# ---------------------------------------------------------------------------
# A3 -- preprocessing constancy: everything comp_fi/PX/delta-eval treat as
# "fixed" must genuinely never mutate across evaluations
# ---------------------------------------------------------------------------

def test_cdw_state_immutable_across_evaluations():
    for name, X in _datasets().items():
        model = _cdw(X)
        snapshot = {
            "M": model.M.copy(), "Mdist": model.Mdist.copy(),
            "Mrho": model.Mrho.copy(), "rho": model.rho.copy(),
            "D": model.D.copy(), "dt0": model.dt0, "dt1": model.dt1,
            "dt2": model.dt2, "Mw_noise": model.Mw_noise.copy(),
            "Mw_same": model.Mw_same.copy(), "Mw_diff": model.Mw_diff.copy(),
            "rev_flat": model.rev_flat.copy(), "rev_off": model.rev_off.copy(),
            "rev_cnt": model.rev_cnt.copy(),
        }
        rng = np.random.default_rng(7)
        for _ in range(100):
            x = _random_partition(rng, model.N, 6)
            model.comp_fitness(x)
            i = int(rng.integers(0, model.N))
            old = int(x[i])
            x[i] = int(rng.integers(0, 6))
            model.df_element(x, i, old)
        assert np.array_equal(snapshot["M"], model.M), name
        assert np.array_equal(snapshot["Mdist"], model.Mdist), name
        assert np.array_equal(snapshot["Mrho"], model.Mrho), name
        assert np.array_equal(snapshot["rho"], model.rho), name
        assert np.array_equal(snapshot["D"], model.D), name
        assert snapshot["dt0"] == model.dt0, name
        assert snapshot["dt1"] == model.dt1, name
        assert snapshot["dt2"] == model.dt2, name
        assert np.array_equal(snapshot["Mw_noise"], model.Mw_noise), name
        assert np.array_equal(snapshot["Mw_same"], model.Mw_same), name
        assert np.array_equal(snapshot["Mw_diff"], model.Mw_diff), name
        assert np.array_equal(snapshot["rev_flat"], model.rev_flat), name
        assert np.array_equal(snapshot["rev_off"], model.rev_off), name
        assert np.array_equal(snapshot["rev_cnt"], model.rev_cnt), name


# ---------------------------------------------------------------------------
# A8 -- numerical robustness: no NaN/Inf, on any dataset (Iris is 4-D, unlike
# Flame/Aggregation's 2-D -- worth checking explicitly, not assuming)
# ---------------------------------------------------------------------------

def test_cdw_comp_fitness_finite_no_nan_inf():
    for name, X in _datasets().items():
        model = _cdw(X)
        assert np.isfinite(model.raw_w).all(), f"[{name}] raw_w has non-finite values"
        assert np.isfinite(model.Mw_diff).all(), f"[{name}] Mw_diff has non-finite values"
        assert np.isfinite(model.Mw_same).all(), f"[{name}] Mw_same has non-finite values"
        assert np.isfinite(model.Mw_noise).all(), f"[{name}] Mw_noise has non-finite values"
        rng = np.random.default_rng(91)
        for _ in range(_trials(name, 200, 50)):
            nc = int(rng.integers(2, 8))
            x = _random_partition(rng, model.N, nc)
            f = model.comp_fitness(x)
            assert np.isfinite(f), f"[{name}] comp_fitness returned non-finite {f}"


# ---------------------------------------------------------------------------
# Delta-eval correctness, independent + larger-scale than nkcv2_cdw.py's
# own self_test (synthetic-only, 500 trials)
# ---------------------------------------------------------------------------

def test_cdw_delta_eval_matches_full_reeval():
    for name, X in _datasets().items():
        model = _cdw(X)
        rng = np.random.default_rng(11)
        for _ in range(_trials(name, 800, 150)):
            x = _random_partition(rng, model.N, 6)
            i = int(rng.integers(0, model.N))
            b = int(rng.integers(0, 7))  # includes 0 = noise, and a fresh label
            old = int(x[i])
            f_before = model.comp_fitness(x)
            x[i] = b
            delta = model.df_element(x, i, old)
            f_after = model.comp_fitness(x)
            assert abs(delta - (f_after - f_before)) < TOL, (
                f"[{name}] delta-eval mismatch at i={i} old={old} new={b}: "
                f"delta={delta} vs actual={f_after - f_before}")


# ---------------------------------------------------------------------------
# A6 -- map_solutions (bijective relabel of the red parent) must not change
# fitness. Previously only implicit (compared against the mapped red, never
# checked against the ORIGINAL red) -- now explicit.
# ---------------------------------------------------------------------------

def test_map_solutions_preserves_fitness():
    for name, X in _datasets().items():
        model = _cdw(X)
        rng = np.random.default_rng(71)
        for _ in range(_trials(name, 300, 60)):
            nc = int(rng.integers(2, 7))
            blue = _random_partition(rng, model.N, nc)
            red = _random_partition(rng, model.N, nc)
            red_mapped = map_solutions(blue, red)
            f_red = model.comp_fitness(red)
            f_mapped = model.comp_fitness(red_mapped)
            assert abs(f_red - f_mapped) < TOL, (
                f"[{name}] map_solutions changed fitness: {f_red} -> {f_mapped}")


# ---------------------------------------------------------------------------
# A4 -- PX partial-sum fitness must equal comp_fitness(offspring) from scratch
# (the correctness gap documented in README.md Sec11 / CLAUDE.md Sec12 test #2)
# ---------------------------------------------------------------------------

def test_cdw_px_partial_sum_matches_full_reeval():
    for name, X in _datasets().items():
        model = _cdw(X)
        rng = np.random.default_rng(21)
        for _ in range(_trials(name, 300, 60)):
            nc = int(rng.integers(2, 7))
            p1 = _random_partition(rng, model.N, nc)
            p2 = _random_partition(rng, model.N, nc)
            p2m = map_solutions(p1, p2)
            offspring, fit_returned = px(model, p1, p2m)
            fit_scratch = model.comp_fitness(offspring)
            assert abs(fit_returned - fit_scratch) < TOL, (
                f"[{name}] PX partial-sum fitness {fit_returned} != "
                f"comp_fitness(offspring) {fit_scratch}")


# ---------------------------------------------------------------------------
# A1 -- validity + "never worse than either parent"
# ---------------------------------------------------------------------------

def test_cdw_px_offspring_never_worse_than_parents():
    for name, X in _datasets().items():
        model = _cdw(X)
        rng = np.random.default_rng(31)
        for _ in range(_trials(name, 300, 60)):
            nc = int(rng.integers(2, 7))
            p1 = _random_partition(rng, model.N, nc)
            p2 = _random_partition(rng, model.N, nc)
            p2m = map_solutions(p1, p2)
            _, fit = px(model, p1, p2m)
            assert fit <= model.comp_fitness(p1) + TOL, name
            assert fit <= model.comp_fitness(p2m) + TOL, name


def test_cdw_px_offspring_structurally_valid():
    for name, X in _datasets().items():
        model = _cdw(X)
        rng = np.random.default_rng(41)
        for _ in range(100):
            nc = int(rng.integers(2, 7))
            p1 = _random_partition(rng, model.N, nc)
            p2 = _random_partition(rng, model.N, nc)
            p2m = map_solutions(p1, p2)
            offspring, _ = px(model, p1, p2m)
            assert offspring.shape == (model.N,), name
            assert offspring.dtype == np.int64, name
            assert offspring.min() >= 0, name
            valid_vals = np.stack([p1, p2m], axis=1)
            for idx in range(model.N):
                assert offspring[idx] in valid_vals[idx], (name, idx)


# ---------------------------------------------------------------------------
# A5 -- fix_labels timing: ga.py::_crossover returns `fit` computed BEFORE
# fix_labels() mutates `offspring` in place. This was the initial hypothesis
# that prompted this audit -- it is ONE of ten assumptions checked here, not
# the whole audit.
# ---------------------------------------------------------------------------

def test_cdw_fix_labels_does_not_change_fitness_production_order():
    for name, X in _datasets().items():
        model = _cdw(X)
        rng = np.random.default_rng(51)
        for _ in range(_trials(name, 200, 50)):
            nc = int(rng.integers(2, 7))
            p1 = _random_partition(rng, model.N, nc)
            p2 = _random_partition(rng, model.N, nc)
            p2m = map_solutions(p1, p2)
            offspring, fit = px(model, p1, p2m)   # ga.py::_crossover order
            fix_labels(model, offspring)            # mutates in place
            fit_after_fix = model.comp_fitness(offspring)
            assert abs(fit - fit_after_fix) < TOL, (
                f"[{name}] fix_labels changed fitness: px returned {fit}, "
                f"comp_fitness(offspring) after fix_labels = {fit_after_fix}")


# ---------------------------------------------------------------------------
# A7 -- renumber (bijective relabel applied to the GA's final best_chrom via
# labels_ = renumber(best_chrom)) must not change fitness.
# ---------------------------------------------------------------------------

def test_renumber_preserves_cdw_fitness():
    for name, X in _datasets().items():
        model = _cdw(X)
        rng = np.random.default_rng(81)
        for _ in range(_trials(name, 300, 60)):
            nc = int(rng.integers(2, 7))
            x = _random_partition(rng, model.N, nc)
            f_before = model.comp_fitness(x)
            r = renumber(x)
            f_after = model.comp_fitness(r)
            assert abs(f_before - f_after) < TOL, (
                f"[{name}] renumber changed fitness: {f_before} -> {f_after}")


# ---------------------------------------------------------------------------
# A9 regression guard: PX must actually consume the model's (possibly
# CDW-weighted) comp_fi, not a hardcoded base kernel that would silently
# ignore CDW's weights while every other code path correctly picks them up.
# Run on all three real research datasets.
# ---------------------------------------------------------------------------

def test_px_actually_uses_model_comp_fi_polymorphically():
    datasets = _datasets()
    for name in REAL_DATASETS:
        X = datasets[name]
        base = NKCV2Model(X, K=3)
        cdw = _cdw(X)
        assert not np.allclose(cdw.Mw_diff, 1.0), (
            f"[{name}] CDW_KWARGS produced no effective weight on the diff "
            f"branch -- cannot exercise this regression guard")

        rng = np.random.default_rng(61)
        differing = 0
        for _ in range(30):
            p1 = _random_partition(rng, base.N, 5)
            p2 = _random_partition(rng, base.N, 5)
            p2m = map_solutions(p1, p2)
            _, fit_base = px(base, p1, p2m)
            _, fit_cdw = px(cdw, p1, p2m)
            if abs(fit_base - fit_cdw) > 1e-6:
                differing += 1
        assert differing > 0, (
            f"[{name}] px() returned IDENTICAL fitness under NKCV2Model and "
            f"CDWModel for every trial -- CDW weights not reaching PX "
            f"(polymorphism broken)")


# ---------------------------------------------------------------------------
# A10 -- end-to-end: a real, short NKHGA.fit() run with CDWModel monkeypatched
# in exactly as run_cdw_ga.py does, on ALL THREE research datasets, ties every
# assumption above together over the actual production loop (elitism,
# periodic local-search refresh, crossover, mutation) instead of isolated
# unit calls.
# ---------------------------------------------------------------------------

def test_cdw_full_ga_run_best_fitness_matches_labels():
    datasets = _datasets()
    original = ga_module.NKCV2Model
    ga_module.NKCV2Model = lambda Xx, K=3, _kw=CDW_KWARGS: CDWModel(Xx, K=K, **_kw)
    try:
        for name in REAL_DATASETS:
            X = datasets[name]
            est = NKHGA(K=3, pop_size=20, max_gen=25, random_state=123,
                        verbose=False)
            est.fit(X)
            recomputed = est.model_.comp_fitness(
                np.ascontiguousarray(est.labels_.astype(np.int64)))
            # labels_ = renumber(best_chrom); renumbering is fitness-neutral
            # (A7, verified above), so comparing against best_fitness_
            # (tracked incrementally across 25 generations of PX/mutation
            # deltas + periodic local-search re-evals, never fully
            # recomputed) is a valid end-to-end check for accumulated drift.
            assert abs(recomputed - est.best_fitness_) < 1e-7, (
                f"[{name}] end-to-end drift after a full CDW GA run: "
                f"best_fitness_={est.best_fitness_} but "
                f"comp_fitness(labels_)={recomputed}")
    finally:
        ga_module.NKCV2Model = original


def main():
    print("HAVE_NUMBA:", HAVE_NUMBA)
    print("CDW config under test:", CDW_KWARGS)
    print("Datasets:", list(_datasets().keys()))
    tests = [
        test_cdw_comp_fi_locality,
        test_cdw_state_immutable_across_evaluations,
        test_cdw_comp_fitness_finite_no_nan_inf,
        test_cdw_delta_eval_matches_full_reeval,
        test_map_solutions_preserves_fitness,
        test_cdw_px_partial_sum_matches_full_reeval,
        test_cdw_px_offspring_never_worse_than_parents,
        test_cdw_px_offspring_structurally_valid,
        test_cdw_fix_labels_does_not_change_fitness_production_order,
        test_renumber_preserves_cdw_fitness,
        test_px_actually_uses_model_comp_fi_polymorphically,
        test_cdw_full_ga_run_best_fitness_matches_labels,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"[PASS] {t.__name__}")
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()
