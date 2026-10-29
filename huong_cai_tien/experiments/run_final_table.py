"""
Bang cuoi va Ablation Study cho NK-HGA:
  goc              -- NKHGA nguyen ban tren StandardScaler (src/);
  goc_prep         -- NKHGA nguyen ban tren datasets.preprocess (MinMax + bo cot gan hang);
  cai_tien_cu      -- NKCV-M cung + gieo GMM/KM + gop 2% N (phien ban cu muc 7h);
  nkcv_m_mem       -- Kiem tra rieng fitness mem: NKCV-M soft, khoi tao ngau nhien, khong gop;
  ga_noi_tai       -- Kiem tra rieng toan tu GA: NKCV2 goc, khoi tao do thi Gep + dot bien Gep;
  day_du_ga        -- Ban de xuat thuan GA: fitness mem + khoi tao Gep + dot bien Gep, KHONG hau xu ly;
  day_du_evidence  -- Ban de xuat + hau xu ly kiem chung bang chung lien ket va Delta f <= 0;
  tomato, hdbscan, gmm_bic -- Cac phuong phap tham chieu khong can biet truoc k.

Chay:
    python huong_cai_tien/experiments/run_final_table.py --datasets flame jain --seeds 100 101
    python huong_cai_tien/experiments/run_final_table.py --summary
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

ALL_METHODS = [
    "goc",
    "goc_prep",
    "cai_tien_cu",
    "nkcv_m_mem",
    "ga_noi_tai",
    "day_du_ga",
    "day_du_evidence",
    "tomato",
    "hdbscan",
    "gmm_bic"
]

DEFAULT_METHODS = [
    "goc",
    "cai_tien_cu",
    "ga_noi_tai",
    "day_du_ga",
    "day_du_evidence"
]

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
    X_std = StandardScaler().fit_transform(X_raw)
    X_new = preprocess(X_raw)
    pool = None
    active = set(args.methods)

    def emit(method, seed, labels, dt):
        r = dict(dataset=name, n=len(y), method=method, seed=seed,
                 ari=adjusted_rand_score(y, labels), n_clusters=n_clusters(labels),
                 n_noise=n_noise(labels), runtime_s=dt)
        writer.writerow(r)
        fh.flush()
        print(f"{name:<12}{method:<16}{seed:>5} ARI={r['ari']:.4f} "
              f"cụm={r['n_clusters']:>3} nhiễu={r['n_noise']:>4} {dt:7.1f}s",
              flush=True)

    for seed in args.seeds:
        # 1. goc (StandardScaler)
        if "goc" in active:
            fast_px.install()
            try:
                t0 = time.time()
                lab = NKHGA(K=3, pop_size=args.pop, max_gen=args.gen,
                            random_state=seed).fit_predict(X_std)
                emit("goc", seed, lab, time.time() - t0)
            finally:
                fast_px.uninstall()

        # 2. goc_prep (NKHGA goc tren X_new)
        if "goc_prep" in active:
            fast_px.install()
            try:
                t0 = time.time()
                lab = NKHGA(K=3, pop_size=args.pop, max_gen=args.gen,
                            random_state=seed).fit_predict(X_new)
                emit("goc_prep", seed, lab, time.time() - t0)
            finally:
                fast_px.uninstall()

        # 3. cai_tien_cu (phien ban muc 7h: mode_hard, gieo pool, merge legacy)
        if "cai_tien_cu" in active:
            if pool is None:
                pool = build_seed_pool(X_new)
            t0 = time.time()
            lab = ImprovedNKHGA(pool=pool, init_source="pool", pop_size=args.pop,
                                max_gen=args.gen, fitness="mode_hard", merge="legacy",
                                use_graph_mutations=False, random_state=seed).fit_predict(X_new)
            emit("cai_tien_cu", seed, lab, time.time() - t0)

        # 4. nkcv_m_mem (fitness mem mot minh, khoi tao ngau nhien)
        if "nkcv_m_mem" in active:
            t0 = time.time()
            lab = ImprovedNKHGA(init_source="random", pop_size=args.pop,
                                max_gen=args.gen, fitness="mode", merge=False,
                                use_graph_mutations=False, random_state=seed).fit_predict(X_new)
            emit("nkcv_m_mem", seed, lab, time.time() - t0)

        # 5. ga_noi_tai (NKCV2 goc + toan tu va khoi tao Gep)
        if "ga_noi_tai" in active:
            t0 = time.time()
            lab = ImprovedNKHGA(init_source="internal", pop_size=args.pop,
                                max_gen=args.gen, fitness="nkcv2", merge=False,
                                use_graph_mutations=True, random_state=seed).fit_predict(X_new)
            emit("ga_noi_tai", seed, lab, time.time() - t0)

        # 6. day_du_ga (Thuan GA: fitness mem + toan tu va khoi tao Gep, khong hau xu ly)
        if "day_du_ga" in active or "day_du_evidence" in active:
            t0 = time.time()
            est = ImprovedNKHGA(init_source="internal", pop_size=args.pop,
                                max_gen=args.gen, fitness="mode", merge="evidence",
                                use_graph_mutations=True, random_state=seed).fit(X_new)
            dt = time.time() - t0
            if "day_du_ga" in active:
                emit("day_du_ga", seed, est.labels_before_merge_, dt)
            if "day_du_evidence" in active:
                emit("day_du_evidence", seed, est.labels_, dt)
            basin = est.model_.basin_ + 1

    if not args.no_refs:
        if "tomato" in active:
            if 'basin' not in locals():
                from nkcv2_mode import ModeNKCV2Model
                m_temp = ModeNKCV2Model(X_new, K=3)
                basin = m_temp.basin_ + 1
            emit("tomato", -1, basin, 0.0)
        if "hdbscan" in active:
            t0 = time.time()
            lab = HDBSCAN(min_cluster_size=15, copy=True).fit_predict(X_new) + 1
            emit("hdbscan", -1, lab, time.time() - t0)
        if "gmm_bic" in active:
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

    # Lay cac method thuc te co trong du lieu
    present_methods = [m for m in ALL_METHODS if any(r["method"] == m for r in rows)]
    print(f"{'bo':<12}{'N':>6} | " + " | ".join(f"{m:^20}" for m in present_methods))
    for name in ORDER:
        sel = [r for r in rows if r["dataset"] == name]
        if not sel:
            continue
        cells = []
        for m in present_methods:
            s = [r for r in sel if r["method"] == m]
            if not s:
                cells.append(f"{'-':^20}")
                continue
            a = np.array([float(r["ari"]) for r in s])
            c = np.mean([float(r["n_clusters"]) for r in s])
            sd = f"±{a.std():.2f}" if len(a) > 1 else "     "
            cells.append(f"{a.mean():.3f}{sd} ({c:4.1f}) n={len(a):<2}")
        print(f"{name:<12}{sel[0]['n']:>6} | " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ORDER)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(100, 105)))
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gen", type=int, default=80)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
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
