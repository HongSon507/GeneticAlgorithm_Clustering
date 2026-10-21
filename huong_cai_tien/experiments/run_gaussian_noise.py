"""
Thi nghiem tren "mo hinh Gauss co nhieu" (models/gaussian_model.py), giao thuc
theo bai bao NK-HGA (muc V-A1, Bang V-VI): moi to hop (Nc, l) gom 3 muc can bang
x 3 bo ngau nhien = 9 bo du lieu, N = 800, 1% nhieu.

Phuong phap:
  goc            -- src/ga.py::NKHGA nguyen ban, StandardScaler (nhu src/run_comparison.py);
  cai_tien       -- ImprovedNKHGA (CDW + gieo + PX nhanh + gop manh vun), datasets.preprocess;
  cai_tien_truoc_gop -- cung lan chay cai_tien, nhan truoc buoc gop manh vun;
  hdbscan, gmm_bic   -- moc tham chieu khong can biet truoc k, chay mot lan moi bo.

ARI tinh tren toan bo diem, nhieu that va nhieu du doan deu la nhan 0.
Moc cua bai bao cho Nc=2, l=2 co nhieu: NK-HGA ARI trung binh 0.96 +- 0.03 (Bang V).

Chay:
    python huong_cai_tien/experiments/run_gaussian_noise.py --nc 2 --l 2
    python huong_cai_tien/experiments/run_gaussian_noise.py --nc 2 --l 2 --plot
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from _paths import RESULTS

import fast_px
from ga import NKHGA
from gaussian_model import generate
from improved import ImprovedNKHGA
from datasets import n_clusters, n_noise, preprocess
from run_final_table import gmm_bic

from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

FIELDS = ["nc", "l", "balance", "data_seed", "method", "seed", "ari",
          "n_clusters", "n_noise", "runtime_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", type=int, default=2)
    ap.add_argument("--l", type=int, default=2)
    ap.add_argument("--data-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102, 103, 104])
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gen", type=int, default=80)
    ap.add_argument("--plot", action="store_true",
                    help="ve 9 bo du lieu va nhan cai_tien (chi khi l = 2)")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"gaussian_noise_nc{args.nc}_l{args.l}.csv")
    shown = {}
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()

        def emit(bal, ds, method, seed, y, lab, dt):
            r = dict(nc=args.nc, l=args.l, balance=bal, data_seed=ds,
                     method=method, seed=seed, ari=adjusted_rand_score(y, lab),
                     n_clusters=n_clusters(lab), n_noise=n_noise(lab), runtime_s=dt)
            w.writerow(r)
            f.flush()
            print(f"cb={bal} bo={ds} {method:<20}{seed:>5} ARI={r['ari']:.4f} "
                  f"cụm={r['n_clusters']:>3} nhiễu={r['n_noise']:>3}", flush=True)

        for bal in (1, 2, 3):
            for ds in args.data_seeds:
                X_raw, y = generate(args.nc, args.l, balance=bal, seed=ds)
                X_std = StandardScaler().fit_transform(X_raw)
                X_new = preprocess(X_raw)
                pool = None
                for seed in args.seeds:
                    fast_px.install()
                    try:
                        t0 = time.time()
                        lab = NKHGA(K=3, pop_size=args.pop, max_gen=args.gen,
                                    random_state=seed).fit_predict(X_std)
                        emit(bal, ds, "goc", seed, y, lab, time.time() - t0)
                    finally:
                        fast_px.uninstall()
                    t0 = time.time()
                    est = ImprovedNKHGA(pool=pool, pop_size=args.pop,
                                        max_gen=args.gen, random_state=seed).fit(X_new)
                    pool = est.pool
                    dt = time.time() - t0
                    emit(bal, ds, "cai_tien", seed, y, est.labels_, dt)
                    emit(bal, ds, "cai_tien_truoc_gop", seed, y,
                         est.labels_before_merge_, dt)
                    if seed == args.seeds[0]:
                        shown[(bal, ds)] = (X_raw, y, est.labels_)
                t0 = time.time()
                emit(bal, ds, "hdbscan", -1, y,
                     HDBSCAN(min_cluster_size=15, copy=True).fit_predict(X_new) + 1,
                     time.time() - t0)
                t0 = time.time()
                emit(bal, ds, "gmm_bic", -1, y, gmm_bic(X_new), time.time() - t0)

    if args.plot and args.l == 2:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 3, figsize=(11, 10))
        for (bal, ds), (X, y, lab) in shown.items():
            ax = axes[bal - 1][args.data_seeds.index(ds)]
            noise = lab == 0
            ax.scatter(X[~noise, 0], X[~noise, 1], c=lab[~noise], s=6, cmap="tab10")
            ax.scatter(X[noise, 0], X[noise, 1], c="black", marker="x", s=18,
                       label="nhieu (du doan)")
            truth_noise = y == 0
            ax.scatter(X[truth_noise, 0], X[truth_noise, 1], facecolors="none",
                       edgecolors="red", s=40, label="nhieu (that)")
            ax.set_title(f"can bang {bal}, bo {ds}: ARI={adjusted_rand_score(y, lab):.3f}",
                         fontsize=9)
        axes[0][0].legend(fontsize=7, loc="best")
        fig.suptitle(f"NK-HGA cai tien, mo hinh Gauss co nhieu, Nc={args.nc}, l={args.l} (seed {args.seeds[0]})")
        fig.tight_layout()
        path = os.path.join(RESULTS, f"gaussian_noise_nc{args.nc}_l{args.l}.png")
        fig.savefig(path, dpi=110)
        print("Hinh:", path)

    import csv as _csv
    with open(out) as fh:
        rows = list(_csv.DictReader(fh))
    print(f"\n=== Nc={args.nc}, l={args.l}: {len(args.data_seeds) * 3} bo x {len(args.seeds)} seed ===")
    for m in ["goc", "cai_tien_truoc_gop", "cai_tien", "hdbscan", "gmm_bic"]:
        a = np.array([float(r["ari"]) for r in rows if r["method"] == m])
        c = np.array([float(r["n_clusters"]) for r in rows if r["method"] == m])
        hit = np.mean(c == args.nc) * 100
        print(f"{m:<20} ARI={a.mean():.3f} ± {a.std():.3f}   so cum TB={c.mean():.2f}"
              f"   dung so cum={hit:.0f}%")


if __name__ == "__main__":
    main()
