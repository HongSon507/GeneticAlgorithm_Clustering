"""
Chay NK-HGA goc va ban cai tien hien hanh, luu NHAN cua tung seed de ve khong
gian phan cum (plot_before_after.py doc cac file nay, khong chay lai GA).

Cau hinh (pop = 60, gen = 80, seed 100-104, S1-S4 lay mau 1000 diem):
  goc              NKHGA nguyen ban tren StandardScaler -- cot "truoc cai tien";
  goc_prep         NKHGA nguyen ban tren preprocess -- tach anh huong tien xu ly;
  goc_ls26         nhu goc_prep, leng_local = 26 nhu ban cai tien -- doi chung:
                   voi gen = 80 va leng_local = 100 thi pha lam moi KHONG chay;
  day_du_ga        ImprovedNKHGA mac dinh, truoc hau xu ly;
  day_du_evidence  ImprovedNKHGA mac dinh, sau hau xu ly -- cot "sau cai tien".

Toa do ve: du lieu goc neu 2 chieu, PCA 2 thanh phan cua preprocess(X) neu nhieu
chieu. Dau ra: results/khong_gian_cum/labels_<bo>.npz va runs_<bo>.csv.

Chay:
    python huong_cai_tien/experiments/run_before_after.py --datasets flame jain
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from _paths import RESULTS

import fast_px
from datasets import load_raw, n_clusters, n_noise, preprocess, subsample
from ga import NKHGA
from improved import ImprovedNKHGA

from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

OUT_DIR = os.path.join(RESULTS, "khong_gian_cum")
ORDER = ["flame", "aggregation", "jain", "iris", "ecoli", "glass",
         "s1", "s2", "s3", "s4"]
METHODS = ["goc", "goc_prep", "goc_ls26", "day_du_ga", "day_du_evidence"]
SEEDS = [100, 101, 102, 103, 104]
SUBSAMPLE = 1000
POP, GEN = 60, 80
LS_PERIOD = max(15, GEN // 3)  # leng_local cua SeededNKHGA voi gen = 80
FIELDS = ["dataset", "n", "method", "seed", "ari", "n_clusters", "n_noise",
          "runtime_s"]


def run_original(X: np.ndarray, seed: int, leng_local: int) -> np.ndarray:
    fast_px.install()
    try:
        return NKHGA(K=3, pop_size=POP, max_gen=GEN, leng_local=leng_local,
                     random_state=seed).fit_predict(X)
    finally:
        fast_px.uninstall()


def plot_coords(X_raw: np.ndarray, X_new: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Toa do 2 chieu de ve va ti le phuong sai giai thich (rong neu khong PCA)."""
    if X_new.shape[1] == 2:
        return X_raw, []
    pca = PCA(n_components=2, random_state=0).fit(X_new)
    return pca.transform(X_new), [float(v) for v in pca.explained_variance_ratio_]


def run_dataset(name: str, seeds: list[int]) -> None:
    X_raw, y = subsample(*load_raw(name), SUBSAMPLE)
    X_std = StandardScaler().fit_transform(X_raw)
    X_new = preprocess(X_raw)
    labels = {m: np.zeros((len(seeds), len(y)), dtype=np.int64) for m in METHODS}
    rows = []

    def record(method: str, s: int, lab: np.ndarray, dt: float) -> None:
        labels[method][s] = lab
        rows.append(dict(dataset=name, n=len(y), method=method, seed=seeds[s],
                         ari=adjusted_rand_score(y, lab), n_clusters=n_clusters(lab),
                         n_noise=n_noise(lab), runtime_s=dt))
        r = rows[-1]
        print(f"{name:<12}{method:<16}{r['seed']:>5} ARI={r['ari']:.4f} "
              f"cum={r['n_clusters']:>3} nhieu={r['n_noise']:>4} {dt:6.1f}s", flush=True)

    for s, seed in enumerate(seeds):
        t0 = time.time()
        record("goc", s, run_original(X_std, seed, 100), time.time() - t0)
        t0 = time.time()
        record("goc_prep", s, run_original(X_new, seed, 100), time.time() - t0)
        t0 = time.time()
        record("goc_ls26", s, run_original(X_new, seed, LS_PERIOD), time.time() - t0)
        t0 = time.time()
        est = ImprovedNKHGA(pop_size=POP, max_gen=GEN, random_state=seed).fit(X_new)
        dt = time.time() - t0
        record("day_du_ga", s, est.labels_before_merge_, dt)
        record("day_du_evidence", s, est.labels_, dt)

    coords, evr = plot_coords(X_raw, X_new)
    np.savez_compressed(os.path.join(OUT_DIR, f"labels_{name}.npz"), coords=coords,
                        y=y, seeds=np.array(seeds), evr=np.array(evr),
                        **{f"lab_{m}": v for m, v in labels.items()})
    with open(os.path.join(OUT_DIR, f"runs_{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def run_paper_protocol(name: str, seeds: list[int]) -> None:
    """Doi chung ngan sach: NKHGA goc dung giao thuc bai bao (pop 100, N/2 giay).

    Kiem tra cot "truoc cai tien" khong xau di vi chi co gen = 80. Dung thoi gian
    thuc nen so lieu phu thuoc toc do may; fast_px lam goc chay nhanh hon, co loi
    cho goc. N la co sau lay mau; ga.py khong tinh thoi gian khoi tao va chi kiem
    tra dung o dau the he. Dau ra: runs_baibao_<bo>.csv.
    """
    X_raw, y = subsample(*load_raw(name), SUBSAMPLE)
    X_std = StandardScaler().fit_transform(X_raw)
    rows = []
    for seed in seeds:
        t0 = time.time()
        fast_px.install()
        try:
            lab = NKHGA(K=3, pop_size=100, time_budget=True,
                        random_state=seed).fit_predict(X_std)
        finally:
            fast_px.uninstall()
        rows.append(dict(dataset=name, n=len(y), method="goc_baibao", seed=seed,
                         ari=adjusted_rand_score(y, lab), n_clusters=n_clusters(lab),
                         n_noise=n_noise(lab), runtime_s=time.time() - t0))
        print(rows[-1], flush=True)
    with open(os.path.join(OUT_DIR, f"runs_baibao_{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ORDER, choices=ORDER)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--paper-protocol", action="store_true",
                    help="chi chay doi chung NKHGA goc pop 100, N/2 giay")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in args.datasets:
        if args.paper_protocol:
            run_paper_protocol(name, args.seeds)
        else:
            run_dataset(name, args.seeds)


if __name__ == "__main__":
    main()
