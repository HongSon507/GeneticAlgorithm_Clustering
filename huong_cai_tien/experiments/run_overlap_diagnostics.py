"""
Chan doan cho Ecoli/Glass va S1-S4 (muc 9 cua DE_XUAT_FITNESS_CHONG_LAN.md).

  prep      -- Ecoli/Glass: tien xu ly (Standard hien tai, MinMax, bo cot gan hang
               so / log1p) x (khoi tao ngau nhien, co gieo), NKCV2 goc.
  a1        -- S1-S4: f(GMM 15 cum, ~toi uu Bayes) so voi f(loi giai GA co gieo).
               f(GMM) > f(GA) nghia la NKCV2 xep loi giai gop cum tot hon.
  baselines -- S1-S4: HDBSCAN, tran DBSCAN (eps chon bang nhan that -- chi la can
               tren lac quan), GMM full covariance chon k theo BIC.

S1-S4 dung mau con 1000 diem (seed 0) nhu run_seeding_grid.py --subsample 1000.

Chay:
    python huong_cai_tien/experiments/run_overlap_diagnostics.py prep a1 baselines
"""

from __future__ import annotations

import sys

import numpy as np

import _paths  # noqa: F401  (them src/ + models/ vao sys.path)

import fast_px
from ga import NKHGA, renumber
from nkcv2 import NKCV2Model
from seeded_ga import SeededNKHGA, build_seed_pool
from run_seeding_grid import subsample
from datasets import DATA_DIR, DATASETS

from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SEEDS = [100, 101, 102]
GA_KW = dict(K=3, pop_size=60, max_gen=80)


def prep():
    for name, d, drop in [("ecoli", 7, [2, 3]), ("glass", 9, None)]:
        Z = np.loadtxt(f"{DATA_DIR}/{name}.txt")
        X0, y = Z[:, :d], Z[:, -1].astype(int)
        variants = {"Standard (hien tai)": StandardScaler().fit_transform(X0),
                    "MinMax": MinMaxScaler().fit_transform(X0)}
        if drop:
            variants["Standard, bo lip+chg"] = StandardScaler().fit_transform(
                np.delete(X0, drop, axis=1))
        else:
            variants["Standard sau log1p"] = StandardScaler().fit_transform(
                np.log1p(X0 - X0.min(0)))
        for vn, X in variants.items():
            pool = build_seed_pool(X)
            for seeded in (False, True):
                a, c = [], []
                for s in SEEDS:
                    est = (SeededNKHGA(pool=pool, seed_frac=0.3, random_state=s, **GA_KW)
                           if seeded else NKHGA(random_state=s, **GA_KW)).fit(X)
                    a.append(ari(y, est.labels_))
                    c.append(est.n_clusters_)
                print(f"{name:6}{vn:24}{'gieo' if seeded else 'ngau nhien':11}"
                      f" ARI={np.mean(a):.3f} cum={np.mean(c):.1f}", flush=True)


def a1():
    for d in ["s1", "s2", "s3", "s4"]:
        X, y = subsample(*DATASETS[d](), 1000)
        m = NKCV2Model(X, 3)
        g = GaussianMixture(15, covariance_type="full", n_init=5, random_state=0).fit(X)
        gl = renumber(g.predict(X) + 1)
        est = SeededNKHGA(pool=build_seed_pool(X), seed_frac=0.3, random_state=100,
                          **GA_KW).fit(X)
        fg = m.comp_fitness(gl.astype(np.int64))
        print(f"{d}: GMM-15 ARI={ari(y, gl):.3f} f={fg:.5f} | GA f={est.best_fitness_:.5f}"
              f" ({est.n_clusters_} cum, ARI={ari(y, est.labels_):.3f}) -> "
              f"{'A1 PASS' if fg < est.best_fitness_ else 'A1 FAIL'}", flush=True)


def baselines():
    for d in ["s1", "s2", "s3", "s4"]:
        X, y = subsample(*DATASETS[d](), 1000)
        h = HDBSCAN(min_cluster_size=15, copy=True).fit_predict(X)
        kd = np.sort(NearestNeighbors(n_neighbors=5).fit(X).kneighbors(X)[0][:, -1])
        db = max(ari(y, DBSCAN(eps=e, min_samples=5).fit_predict(X))
                 for e in np.quantile(kd, np.linspace(.05, .95, 30)))
        gms = [GaussianMixture(k, covariance_type="full", n_init=2, random_state=0).fit(X)
               for k in range(2, 21)]
        gb = min(gms, key=lambda g: g.bic(X))
        print(f"{d}: HDBSCAN ARI={ari(y, h):.3f} | tran DBSCAN={db:.3f} | "
              f"GMM-BIC k={gb.n_components} ARI={ari(y, gb.predict(X)):.3f}", flush=True)


if __name__ == "__main__":
    fast_px.install()
    for task in sys.argv[1:] or ["prep", "a1", "baselines"]:
        {"prep": prep, "a1": a1, "baselines": baselines}[task]()
