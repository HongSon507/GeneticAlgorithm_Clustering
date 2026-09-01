"""
A9 -- chay GA THAT (khong chi sang loc) voi CDWModel, dung cau hinh da dat A1
tren Flame o run_cdw_screen.py (branches=diff, w_min=0.01, w_max=5, lam=1.0).

So sanh:
  * Flame       -- bo dang that bai voi ban goc (ARI ~0.01) -- muc tieu: thang ro
  * Aggregation -- bo bat goc rat manh (ARI 0.9876) -- tieu chi: KHONG tut

Khong sua src/: monkeypatch `ga.NKCV2Model` bang `CDWModel` trong try/finally,
dung khuon mau cua run_persist.py::run_once.

Chay:
    python huong_cai_tien/run_cdw_ga.py
    python huong_cai_tien/run_cdw_ga.py --runs 3
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import _paths  # noqa: F401  (them src/ + models/ vao sys.path)

from sklearn.metrics import adjusted_rand_score

import ga
from ga import NKHGA
from nkcv2_cdw import CDWModel, self_test, HAVE_NUMBA
from datasets import load_flame, n_clusters, n_noise
from datasets import load_aggregation

DATASETS = {"flame": load_flame, "aggregation": load_aggregation}

CDW_KWARGS = dict(lam=1.0, branches=("diff",), w_min=0.01, w_max=5.0)


def run_once(X, K, pop, gen, seed, use_cdw, verbose):
    original = ga.NKCV2Model
    if use_cdw:
        ga.NKCV2Model = (lambda Xx, K=3, _kw=CDW_KWARGS:
                         CDWModel(Xx, K=K, **_kw))
    try:
        est = NKHGA(K=K, pop_size=pop, max_gen=gen, random_state=seed,
                    verbose=verbose)
        t0 = time.time()
        labels = est.fit_predict(X)
        elapsed = time.time() - t0
    finally:
        ga.NKCV2Model = original
    return labels, elapsed


def evaluate(name, X, y_true, args, verbose):
    print(f"\n########## {name.upper()} (N={X.shape[0]}, "
         f"cụm thật={len(np.unique(y_true))}) ##########")
    rows = []
    for run_idx in range(args.runs):
        seed = args.seed + run_idx
        print(f"\n--- NK-HGA gốc (seed={seed}) ---")
        lab, dt = run_once(X, args.K, args.pop, args.gen, seed, False, verbose)
        rows.append(("gốc", seed, adjusted_rand_score(y_true, lab),
                     n_clusters(lab), n_noise(lab), dt))

        print(f"\n--- NK-HGA + CDW (diff-only, w_min=0.01, lam=1) (seed={seed}) ---")
        lab, dt = run_once(X, args.K, args.pop, args.gen, seed, True, verbose)
        rows.append(("CDW", seed, adjusted_rand_score(y_true, lab),
                     n_clusters(lab), n_noise(lab), dt))

    print(f"\n=== {name}: ARI vs nhãn thật ===")
    print(f"{'phương pháp':<10}{'seed':>6}{'ARI':>9}{'cụm':>6}{'nhiễu':>7}{'time(s)':>9}")
    for r in rows:
        print(f"{r[0]:<10}{r[1]:>6d}{r[2]:>9.4f}{r[3]:>6d}{r[4]:>7d}{r[5]:>9.1f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=80)
    ap.add_argument("--gen", type=int, default=100)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--data", nargs="+", default=["flame", "aggregation"],
                    choices=list(DATASETS))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    print(f"HAVE_NUMBA={HAVE_NUMBA}")
    print(f"CDW config: {CDW_KWARGS}\n")
    self_test()

    for name in args.data:
        X, y_true = DATASETS[name]()
        evaluate(name, X, y_true, args, not args.quiet)


if __name__ == "__main__":
    main()
