"""
Ablation study: which component(s) of CDW's w_ij = c_ij * d_ij * b_ij drive the
observed improvement in NK Hybrid GA?

Frozen research scope. Does NOT modify NK Hybrid GA, CDW, or any existing
file; does NOT invent a new weighting scheme (every variant is the EXISTING
CDW formula with 0-3 of its own multiplicative terms switched off, via
ablation_model.py::AblationCDWModel -- a faithful decomposition self-tested
against CDWModel and NKCV2Model bit-for-bit); does NOT tune GA parameters.

TWO-PHASE PROTOCOL (screening first, validation second)
  Phase 1 -- screening: one dataset (Flame), 2 seeds, 3 variants
      (baseline / boundary-only / full CDW). Cheap; establishes whether full
      CDW is actually ahead before spending time on the full grid.
          python run_ablation.py --datasets flame --seeds 100 101 \
              --variants 1_baseline 4_boundary_only 8_full_cdw --tag screening
  Phase 2 -- validation (only if screening shows full CDW ahead): the full
      8 variants x 3 datasets x 3 seeds grid.
          python run_ablation.py --tag full

Reducing the experiment's SIZE is the only thing the flags do. GA parameters,
seeds, stopping criteria, preprocessing, and CDW's weighting parameters are
identical in both phases -- a screening row and a validation row for the same
(dataset, variant, seed) are the same experiment.

SPEED (scheduling only -- recorded numbers unchanged): one model instance per
(dataset, variant), reused across seeds. Safe because the model is provably
never mutated by evaluation -- see huong_cai_tien/test_px_correctness.py::
test_cdw_state_immutable_across_evaluations (assumption A3, 12/12 PASS).
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time

import numpy as np

from _paths import RESULTS

from sklearn.metrics import adjusted_rand_score

import ga as ga_module
from ga import NKHGA
from nkcv2 import NKCV2Model
from ablation_model import AblationCDWModel, self_test
from datasets import DATASETS

# Identical across ALL experiments and both phases (frozen).
GA_KWARGS = dict(K=3, pop_size=40, max_gen=60)
ALL_SEEDS = (100, 101, 102)
# Everything about CDW's weighting EXCEPT which components are active --
# identical to run_cdw_ga.py::CDW_KWARGS (the config under investigation).
CDW_FIXED = dict(branches=("diff",), w_min=0.01, w_max=5.0, lam=1.0)

VARIANTS = {
    # name:                    (use_compactness, use_density, use_boundary)
    "1_baseline":             (False, False, False),
    "2_compactness_only":     (True,  False, False),
    "3_density_only":         (False, True,  False),
    "4_boundary_only":        (False, False, True),
    "5_compactness_density":  (True,  True,  False),
    "6_compactness_boundary": (True,  False, True),
    "7_density_boundary":     (False, True,  True),
    "8_full_cdw":             (True,  True,  True),
}
ALL_DATASETS = ("aggregation", "flame", "iris")


def load_dataset(name):
    return DATASETS[name]()


def build_model(X, use_c, use_d, use_b):
    K = GA_KWARGS["K"]
    if not (use_c or use_d or use_b):
        return NKCV2Model(X, K=K)
    return AblationCDWModel(X, K=K, use_compactness=use_c, use_density=use_d,
                            use_boundary=use_b, **CDW_FIXED)


def ga_run(X, model, seed):
    """Run NKHGA against an already-built model (reused across seeds)."""
    original = ga_module.NKCV2Model
    ga_module.NKCV2Model = lambda Xx, K=3, _m=model: _m
    try:
        est = NKHGA(random_state=seed, verbose=False, **GA_KWARGS)
        t0 = time.time()
        labels = est.fit_predict(X)
        return labels, float(est.best_fitness_), int(est.n_clusters_), time.time() - t0
    finally:
        ga_module.NKCV2Model = original


def sweep(datasets, variants, seeds):
    rows = []
    for dname in datasets:
        X, y_true = load_dataset(dname)
        print(f"########## {dname.upper()} (N={X.shape[0]}) ##########")

        # baseline-GA reference solution, reused as the diff% reference for
        # every variant (protocol of run_cdw_screen.py / README.md Sec7)
        ref_labels, _, _, _ = ga_run(X, build_model(X, False, False, False),
                                     seeds[0])
        ref_x = np.ascontiguousarray(np.asarray(ref_labels).astype(np.int64))
        y_true_x = np.ascontiguousarray(y_true.astype(np.int64))

        for vname in variants:
            use_c, use_d, use_b = VARIANTS[vname]
            model = build_model(X, use_c, use_d, use_b)
            f_true, f_ref = model.comp_fitness(y_true_x), model.comp_fitness(ref_x)
            dp = 100.0 * (f_true - f_ref) / f_ref if f_ref != 0 else float("nan")

            for seed in seeds:
                labels, fitness, n_clusters, elapsed = ga_run(X, model, seed)
                ari = float(adjusted_rand_score(y_true, labels))
                rows.append(dict(
                    dataset=dname, variant=vname, seed=seed,
                    use_compactness=use_c, use_density=use_d, use_boundary=use_b,
                    ari=ari, fitness=fitness, diff_pct=float(dp),
                    n_clusters=n_clusters, runtime_s=elapsed))
                print(f"  {vname:<24} seed={seed} ARI={ari:.4f} "
                      f"fit={fitness:.6f} diff%={dp:+.1f} "
                      f"Nc={n_clusters} {elapsed:.0f}s")
    return rows


def _agg(rows, dataset, variant, field):
    vals = [r[field] for r in rows
            if r["dataset"] == dataset and r["variant"] == variant]
    if not vals:
        return float("nan"), 0.0
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def write_summary(rows, datasets, variants, seeds, path, phase):
    def gain(dname, vname):
        base, _ = _agg(rows, dname, "1_baseline", "ari")
        cur, _ = _agg(rows, dname, vname, "ari")
        return cur - base

    L = [f"# CDW Component Ablation -- {phase}\n"]
    L.append(
        "Frozen research scope: no change to NK Hybrid GA, CDW, GA parameters, "
        "seeds, or stopping criteria. Each variant is the existing "
        "`w_ij = c_ij * d_ij * b_ij` with 0-3 of its own terms switched to the "
        "neutral element 1.0. Every number is read directly off the CSV.\n")
    L.append(f"\n**Scope of this run:** datasets `{list(datasets)}`, variants "
             f"`{list(variants)}`, seeds `{list(seeds)}`.")
    L.append(f"\n**Fixed:** `{GA_KWARGS}`, CDW weighting parameters "
             f"`{CDW_FIXED}`.\n")

    L.append("\n## Results (mean over seeds, +/- stdev)\n")
    for dname in datasets:
        L.append(f"\n### {dname}\n")
        L.append("| Variant | ARI | fitness | diff% | clusters | runtime (s) |")
        L.append("|---|---|---|---|---|---|")
        for vname in variants:
            ari_m, ari_s = _agg(rows, dname, vname, "ari")
            fit_m, fit_s = _agg(rows, dname, vname, "fitness")
            dp_m, _ = _agg(rows, dname, vname, "diff_pct")
            nc_m, _ = _agg(rows, dname, vname, "n_clusters")
            rt_m, _ = _agg(rows, dname, vname, "runtime_s")
            L.append(f"| `{vname}` | {ari_m:.4f} +/- {ari_s:.4f} | "
                     f"{fit_m:.6f} +/- {fit_s:.6f} | {dp_m:+.1f}% | "
                     f"{nc_m:.1f} | {rt_m:.0f} |")

    L.append("\n## Gain over baseline\n")
    L.append("| Dataset | " + " | ".join(f"`{v}`" for v in variants
                                          if v != "1_baseline") + " |")
    L.append("|---" * (1 + sum(1 for v in variants if v != "1_baseline")) + "|")
    for dname in datasets:
        cells = [f"{gain(dname, v):+.4f}" for v in variants if v != "1_baseline"]
        L.append(f"| {dname} | " + " | ".join(cells) + " |")

    if "8_full_cdw" in variants:
        L.append("\n## Is full CDW ahead of every other variant tested here?\n")
        L.append("| Dataset | best non-full variant | its ARI | full_cdw ARI | full ahead? |")
        L.append("|---|---|---|---|---|")
        for dname in datasets:
            others = [v for v in variants if v != "8_full_cdw"]
            best_name, best_ari = None, -2.0
            for v in others:
                m, _ = _agg(rows, dname, v, "ari")
                if m > best_ari:
                    best_ari, best_name = m, v
            full_m, _ = _agg(rows, dname, "8_full_cdw", "ari")
            L.append(f"| {dname} | `{best_name}` | {best_ari:.4f} | {full_m:.4f} | "
                     f"{'YES' if full_m >= best_ari - 1e-12 else 'NO'} |")

    L.append("\nNo claim here goes beyond what the table it is attached to "
             "shows. A screening run answers only 'is the trend worth "
             "validating?', not 'which component wins overall'.\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(ALL_DATASETS),
                    choices=list(ALL_DATASETS))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS),
                    choices=list(VARIANTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(ALL_SEEDS))
    ap.add_argument("--tag", default="full",
                    help="output file suffix, e.g. 'screening'")
    ap.add_argument("--skip-self-test", action="store_true")
    args = ap.parse_args()

    # keep canonical ordering regardless of the order given on the CLI
    datasets = [d for d in ALL_DATASETS if d in args.datasets]
    variants = [v for v in VARIANTS if v in args.variants]
    seeds = tuple(args.seeds)

    if not args.skip_self_test:
        print("Integrity check before the sweep:")
        self_test()
    print(f"\nphase={args.tag}  datasets={datasets}  variants={variants}  "
          f"seeds={seeds}")
    print(f"GA_KWARGS={GA_KWARGS}  CDW_FIXED={CDW_FIXED}")
    print(f"total GA runs = {len(datasets)} x ({len(variants)} x {len(seeds)} "
          f"+ 1 reference) = {len(datasets)*(len(variants)*len(seeds)+1)}\n")

    t0 = time.time()
    rows = sweep(datasets, variants, seeds)

    os.makedirs(RESULTS, exist_ok=True)
    csv_path = os.path.join(RESULTS, f"ablation_results_{args.tag}.csv")
    md_path = os.path.join(RESULTS, f"ablation_summary_{args.tag}.md")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    write_summary(rows, datasets, variants, seeds, md_path, args.tag)

    print(f"\nWrote {csv_path} ({len(rows)} rows)")
    print(f"Wrote {md_path}")
    print(f"Total wall clock: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
