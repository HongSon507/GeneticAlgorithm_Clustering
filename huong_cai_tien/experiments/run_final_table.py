"""
Bang cuoi cho luan van: NK-HGA ban goc so voi ban cai tien, tren 10 bo du lieu.

  goc      -- src/ga.py::NKHGA nguyen ban (NKCV2, khoi tao ngau nhien),
              StandardScaler tren moi cot nhu src/run_comparison.py.
              `fast_px` duoc bat chi de tang toc: dau ra trung tung phan tu voi
              src/px.py, nen ket qua giong het ban Python.
  cai_tien -- models/improved.py::ImprovedNKHGA (NKCV2 goc + gieo quan the +
              PX nhanh + gop manh vun) tren datasets.preprocess (bo cot gan hang
              so + MinMax);
  cai_tien_truoc_gop -- cung lan chay cai_tien, nhan truoc buoc gop manh vun.
  hdbscan, gmm_bic -- moc tham chieu khong can biet truoc k, cung tien xu ly voi
              cai_tien; chay mot lan moi bo (khong phu thuoc seed GA).

S1-S4 chay ban day du 5000 diem (dung --subsample de chay nhanh).
CSV ghi tung dong vao results/final_table_<cac bo>.csv.

Chay:
    python huong_cai_tien/experiments/run_final_table.py --datasets flame jain
    python huong_cai_tien/experiments/run_final_table.py --datasets s1 --seeds 100 101
    python huong_cai_tien/experiments/run_final_table.py --summary   # in bang tu CSV
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import time

import numpy as np

from _paths import RESULTS

import fast_px
from ga import NKHGA
from improved import ImprovedNKHGA
from seeded_ga import build_seed_pool
from datasets import DATASETS, load_raw, n_clusters, n_noise, preprocess, subsample

from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ORDER = ["flame", "aggregation", "jain", "iris", "ecoli", "glass",
         "s1", "s2", "s3", "s4"]
METHODS = ["goc", "cai_tien_truoc_gop", "cai_tien", "hdbscan", "gmm_bic"]
FIELDS = ["dataset", "n", "method", "seed", "ari", "n_clusters", "n_noise",
          "runtime_s"]


def gmm_bic(X, k_max=20):
    best, best_bic = None, np.inf
    for k in range(1, k_max + 1):
        g = GaussianMixture(k, covariance_type="full", reg_covar=1e-4, n_init=2,
                            random_state=0).fit(X)
        if g.bic(X) < best_bic:
            best, best_bic = g, g.bic(X)
    return best.predict(X) + 1


def run_dataset(name, args, writer, fh):
    X_raw, y = subsample(*load_raw(name), args.subsample)
    X_std = StandardScaler().fit_transform(X_raw)  # nhu src/run_comparison.py
    X_new = preprocess(X_raw)
    pool = build_seed_pool(X_new)

    def emit(method, seed, labels, dt):
        r = dict(dataset=name, n=len(y), method=method, seed=seed,
                 ari=adjusted_rand_score(y, labels), n_clusters=n_clusters(labels),
                 n_noise=n_noise(labels), runtime_s=dt)
        writer.writerow(r)
        fh.flush()
        print(f"{name:<12}{method:<10}{seed:>5} ARI={r['ari']:.4f} "
              f"cụm={r['n_clusters']:>3} nhiễu={r['n_noise']:>4} {dt:7.1f}s",
              flush=True)

    for seed in args.seeds:
        fast_px.install()
        try:
            t0 = time.time()
            lab = NKHGA(K=3, pop_size=args.pop, max_gen=args.gen,
                        random_state=seed).fit_predict(X_std)
            emit("goc", seed, lab, time.time() - t0)
        finally:
            fast_px.uninstall()
        t0 = time.time()
        est = ImprovedNKHGA(pool=pool, pop_size=args.pop, max_gen=args.gen,
                            random_state=seed).fit(X_new)
        dt = time.time() - t0
        emit("cai_tien_truoc_gop", seed, est.labels_before_merge_, dt)
        emit("cai_tien", seed, est.labels_, dt)

    if not args.no_refs:
        t0 = time.time()
        lab = HDBSCAN(min_cluster_size=15, copy=True).fit_predict(X_new) + 1
        emit("hdbscan", -1, lab, time.time() - t0)  # nhieu: -1 -> 0
        t0 = time.time()
        emit("gmm_bic", -1, gmm_bic(X_new), time.time() - t0)


def summary():
    rows = []
    for path in glob.glob(os.path.join(RESULTS, "final_table_*.csv")):
        with open(path) as f:
            rows += list(csv.DictReader(f))
    if not rows:
        print("Chua co ket qua trong", RESULTS)
        return
    print(f"{'bo':<12}{'N':>6} | " + " | ".join(f"{m:^22}" for m in METHODS))
    for name in ORDER:
        sel = [r for r in rows if r["dataset"] == name]
        if not sel:
            continue
        cells = []
        for m in METHODS:
            s = [r for r in sel if r["method"] == m]
            if not s:
                cells.append(f"{'-':^22}")
                continue
            a = np.array([float(r["ari"]) for r in s])
            c = np.mean([float(r["n_clusters"]) for r in s])
            sd = f"±{a.std():.2f}" if len(a) > 1 else "     "
            cells.append(f"{a.mean():.3f}{sd} ({c:4.1f}) n={len(a):<2}")
        print(f"{name:<12}{sel[0]['n']:>6} | " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ORDER)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(100, 110)))
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gen", type=int, default=80)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--no-refs", action="store_true",
                    help="bo qua HDBSCAN va GMM-BIC")
    ap.add_argument("--summary", action="store_true",
                    help="chi in bang tong hop tu cac CSV da co")
    args = ap.parse_args()
    if args.summary:
        summary()
        return

    os.makedirs(RESULTS, exist_ok=True)
    tag = "_".join(args.datasets) + (f"_n{args.subsample}" if args.subsample else "")
    with open(os.path.join(RESULTS, f"final_table_{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for name in args.datasets:
            if name not in DATASETS:
                raise SystemExit(f"Khong co bo du lieu '{name}'")
            run_dataset(name, args, w, f)


if __name__ == "__main__":
    main()
