"""
A1 -- sang loc 30 giay (README.md Sec7) cho CDWModel.

So f(nhan that) vs f(loi giai GA hien tai cua BAN GOC) DUOI HAM MUC TIEU MOI, tren
Aggregation / Flame / Iris, quet vai gia tri lambda. Neu f(nhan that) van > f(GA)
tren mot bo thi lambda do bi loai NGAY -- khong can chay GA duoi ham moi.

Khong sua src/: chi import NKCV2Model/NKHGA nguyen ban de lay "loi giai GA hien
tai", roi danh gia lai duoi CDWModel (tao rieng, khong monkeypatch gi ca vi day
la buoc sang loc, chua phai chay GA duoi ham moi).

Chay:
    python huong_cai_tien/experiments/run_cdw_screen.py
    python huong_cai_tien/experiments/run_cdw_screen.py --lam 0 0.1 0.3 0.6 1.0
    python huong_cai_tien/experiments/run_cdw_screen.py --data flame iris
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import _paths  # noqa: F401  (them src/ + models/ vao sys.path)

from ga import NKHGA
from nkcv2_cdw import CDWModel, self_test, HAVE_NUMBA
from datasets import DATASETS as ALL_DATASETS

DATASETS = {k: ALL_DATASETS[k] for k in ("aggregation", "flame", "iris")}


def screen_one(name, X, y_true, lambdas, k_c, b_min, pop, gen, seed,
              branches=("noise", "same", "diff"), w_min=0.1, w_max=3.0):
    print(f"\n########## {name.upper()} (N={X.shape[0]}) ##########")

    t0 = time.time()
    est = NKHGA(K=3, pop_size=pop, max_gen=gen, random_state=seed, verbose=False)
    labels_ga = est.fit_predict(X)
    print(f"  loi giai GA goc: {len(np.unique(labels_ga[labels_ga > 0]))} cum, "
         f"{(labels_ga == 0).sum()} nhieu, {time.time()-t0:.1f}s")

    y_true_x = np.ascontiguousarray(y_true.astype(np.int64))
    labels_ga_x = np.ascontiguousarray(labels_ga.astype(np.int64))

    rows = []
    for lam in lambdas:
        model = CDWModel(X, K=3, lam=lam, k_c=k_c, b_min=b_min,
                         branches=branches, w_min=w_min, w_max=w_max)
        f_true = model.comp_fitness(y_true_x)
        f_ga = model.comp_fitness(labels_ga_x)
        diff_pct = 100.0 * (f_true - f_ga) / f_ga if f_ga != 0 else float("nan")

        cross = y_true[:, None] != y_true[model.M]
        w_cross_mean = model.raw_w[cross].mean() if cross.any() else float("nan")
        w_all_mean = model.raw_w.mean()

        verdict = "PASS (f_true < f_GA)" if f_true < f_ga else "FAIL (f_true >= f_GA)"
        rows.append((lam, f_true, f_ga, diff_pct, w_cross_mean, w_all_mean, verdict))

    print(f"  {'lambda':>7}{'f(true)':>12}{'f(GA)':>12}{'diff%':>9}"
         f"{'raw_w|cross':>13}{'raw_w|all':>11}  verdict")
    for lam, f_true, f_ga, diff_pct, wc, wa, verdict in rows:
        print(f"  {lam:>7.2f}{f_true:>12.6f}{f_ga:>12.6f}{diff_pct:>8.1f}%"
             f"{wc:>13.3f}{wa:>11.3f}  {verdict}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.6, 1.0])
    ap.add_argument("--k-c", type=int, default=None)
    ap.add_argument("--b-min", type=float, default=0.5)
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gen", type=int, default=80)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--data", nargs="+", default=["aggregation", "flame", "iris"],
                    choices=list(DATASETS))
    ap.add_argument("--branches", nargs="+", default=["noise", "same", "diff"],
                    choices=["noise", "same", "diff"])
    ap.add_argument("--w-min", type=float, default=0.1)
    ap.add_argument("--w-max", type=float, default=3.0)
    ap.add_argument("--skip-self-test", action="store_true")
    args = ap.parse_args()

    print(f"HAVE_NUMBA={HAVE_NUMBA}\n")
    if not args.skip_self_test:
        self_test()

    for name in args.data:
        X, y_true = DATASETS[name]()
        screen_one(name, X, y_true, args.lam, args.k_c, args.b_min,
                  args.pop, args.gen, args.seed,
                  branches=tuple(args.branches), w_min=args.w_min, w_max=args.w_max)


if __name__ == "__main__":
    main()
