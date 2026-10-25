"""
Ke hoach buoc 1-2: co dinh fitness (NKCV-A, A-CA, beta=0.5) va doi trong tam
sang tim kiem (gieo quan the tu GMM/k-means/DPC).

Bon cau hinh, tach tac dung cua fitness va cua gieo:
  base         -- NKCV2 goc, khoi tao ngau nhien (NK-HGA nguyen ban)
  base+seed    -- NKCV2 goc, co gieo
  nkcva        -- NKCV-A, khoi tao ngau nhien
  nkcva+seed   -- NKCV-A, co gieo (cau hinh de xuat)

Khong sua src/: monkeypatch `ga.NKCV2Model` trong try/finally.
A_ij va kho loi giai gieo tinh mot lan moi bo, dung chung cho moi seed.
CSV ghi tung dong.

Chay:
    python huong_cai_tien/experiments/run_seeding_grid.py --datasets flame
    python huong_cai_tien/experiments/run_seeding_grid.py --datasets ecoli \
        --seeds 100 101 102 --pop 60 --gen 80
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

import _paths  # noqa: F401  (them src/ + models/ vao sys.path)

from sklearn.metrics import adjusted_rand_score

import ga
import fast_px
from ga import NKHGA
from nkcv2_affinity import AffinityNKCV2Model, build_affinity
from seeded_ga import SeededNKHGA, build_seed_pool
from datasets_std import DATASETS, n_clusters, n_noise

CONFIGS = ["base", "base+seed", "nkcva", "nkcva+seed"]
BETA = 0.5


def subsample(X, y, n, seed=0):
    if n is None or n >= len(y):
        return X, y
    idx = np.random.default_rng(seed).choice(len(y), n, replace=False)
    return X[idx], y[idx]


def run_once(cfg, X, A, pool, args, seed):
    beta = BETA if cfg.startswith("nkcva") else 0.0
    original = ga.NKCV2Model
    ga.NKCV2Model = lambda Xx, K=3: AffinityNKCV2Model(Xx, K=K, beta=beta, A=A)
    try:
        kw = dict(K=args.K, pop_size=args.pop, max_gen=args.gen, random_state=seed)
        est = (SeededNKHGA(pool=pool, seed_frac=args.seed_frac, **kw)
               if cfg.endswith("+seed") else NKHGA(**kw))
        t0 = time.time()
        est.fit(X)
        return est, time.time() - t0
    finally:
        ga.NKCV2Model = original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(100, 110)))
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gen", type=int, default=80)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--seed-frac", type=float, default=0.3)
    ap.add_argument("--subsample", type=int, default=None,
                    help="lay ngau nhien n diem (seed 0) -- dung cho S1-S4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    fast_px.install()  # map_solutions/fix_labels nhanh, dau ra trung ban goc

    tag = "_".join(args.datasets) + (f"_n{args.subsample}" if args.subsample else "")
    out = args.out or os.path.join(os.path.dirname(__file__), "..", "results",
                                   f"seeding_grid_{tag}.csv")
    fields = ["dataset", "n", "config", "seed", "ari", "n_clusters", "n_noise",
              "fitness", "runtime_s"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name in args.datasets:
            X, y = subsample(*DATASETS[name](), args.subsample)
            A = build_affinity(X, "ca_gmm", K=args.K)
            pool = build_seed_pool(X)
            for cfg in args.configs:
                for seed in args.seeds:
                    est, dt = run_once(cfg, X, A, pool, args, seed)
                    r = dict(dataset=name, n=len(y), config=cfg, seed=seed,
                             ari=adjusted_rand_score(y, est.labels_),
                             n_clusters=n_clusters(est.labels_),
                             n_noise=n_noise(est.labels_),
                             fitness=est.best_fitness_, runtime_s=dt)
                    w.writerow(r)
                    f.flush()
                    print(f"{name:<12}{cfg:<12}{seed:>5} ARI={r['ari']:.4f} "
                          f"cụm={r['n_clusters']:>3} nhiễu={r['n_noise']:>3} "
                          f"{dt:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
