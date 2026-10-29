"""
Hieu chuan z0 cua NKCV-M mem tren du lieu RONG (chi co 1 cum) -- khong dung nhan
that, khong dung bo benchmark nao de chon. So lieu: DE_XUAT_FITNESS_CHONG_LAN.md
muc 7i.5.

  --part dinh : (A) ti le co >= 2 dinh (dinh gia) tren 1 khoi Gauss / phan bo
                deu, N = 240 va 1000, l = 2 va 4, 30 lan lap moi o;
                (B) so dinh theo z0 tren 10 bo benchmark (S1-S4 mau 1000) va 9 bo
                Gauss co nhieu Nc = 2 (chi doi chieu, khong dung de chon).
  --part ga   : ImprovedNKHGA mac dinh voi z0 = 1.5 va 2.0, moi thu khac giu
                nguyen, 3 seed, tren flame / s1 / s2 / khoi deu 2D N = 1000.

Chay:
    python huong_cai_tien/experiments/run_z0_calibration.py --part dinh
    python huong_cai_tien/experiments/run_z0_calibration.py --part ga --datasets flame deu2d
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import _paths  # noqa: F401  (them models/ va src/ vao sys.path)

from datasets import load_raw, n_clusters, preprocess, subsample
from gaussian_model import generate
from improved import ImprovedNKHGA
from nkcv2 import _cutoff_distance, _squared_distance_matrix
from nkcv2_mode import significant_modes

from sklearn.metrics import adjusted_rand_score

Z0S = [1.0, 1.5, 1.75, 2.0, 2.5, 3.0]
BENCH = [("flame", 2), ("aggregation", 7), ("jain", 2), ("iris", 3), ("ecoli", 8),
         ("glass", 6), ("s1", 15), ("s2", 15), ("s3", 15), ("s4", 15)]


def n_modes(X_raw):
    X = np.ascontiguousarray(preprocess(X_raw), dtype=np.float64)
    D = _squared_distance_matrix(X)
    dc = _cutoff_distance(D, D.shape[0])
    return [int(significant_modes(D, dc, z0=z)[0].max()) + 1 for z in Z0S]


def part_modes(reps):
    print("== (A) DU LIEU RONG: % lan co dinh gia / so dinh TB")
    print(f"{'phan bo':<10}{'N':>6}{'l':>3} | " + " | ".join(f"z0={z:<4}" for z in Z0S))
    for dist in ("gauss", "deu"):
        for N in (240, 1000):
            for l in (2, 4):
                res = []
                for r in range(reps):
                    rng = np.random.default_rng(1000 * N + 10 * l + r)
                    X = rng.normal(size=(N, l)) if dist == "gauss" else rng.uniform(size=(N, l))
                    res.append(n_modes(X))
                res = np.array(res)
                cells = [f"{(res[:, j] > 1).mean() * 100:4.0f}% {res[:, j].mean():4.2f}"
                         for j in range(len(Z0S))]
                print(f"{dist:<10}{N:>6}{l:>3} | " + " | ".join(cells), flush=True)

    print("\n== (B) SO DINH TREN BENCHMARK (k that) -- chi doi chieu")
    print(f"{'bo':<14}{'k':>3} | " + " | ".join(f"z0={z:<4}" for z in Z0S))
    for name, k in BENCH:
        X, _ = subsample(*load_raw(name), 1000)
        print(f"{name:<14}{k:>3} | " + " | ".join(f"{m:^7}" for m in n_modes(X)), flush=True)
    hits = np.zeros(len(Z0S))
    for bal in (1, 2, 3):
        for ds in (0, 1, 2):
            m = n_modes(generate(2, 2, balance=bal, seed=ds)[0])
            print(f"{f'gauss b{bal} s{ds}':<14}{2:>3} | " + " | ".join(f"{v:^7}" for v in m),
                  flush=True)
            hits += np.array(m) == 2
    print(f"{'gauss dung 2':<17} | " + " | ".join(f"{h:.0f}/9".center(7) for h in hits))


def part_ga(datasets, seeds):
    for name in datasets:
        if name == "deu2d":
            X_raw = np.random.default_rng(7).uniform(size=(1000, 2))
            y = np.ones(1000, dtype=int)
        else:
            X_raw, y = subsample(*load_raw(name), 1000)
        X = preprocess(X_raw)
        for z0 in (1.5, 2.0):
            for seed in seeds:
                t0 = time.time()
                est = ImprovedNKHGA(z0=z0, random_state=seed).fit(X)
                m = est.model_
                pre, post = est.labels_before_merge_, est.labels_
                print(f"{name:<8} z0={z0} seed={seed} dinh={m.n_modes_:>2} "
                      f"canh_w>0={int((m.Mcross > 0).sum()):>4} w_max_thuc={m.Mcross.max():.3f} | "
                      f"truoc_gop ARI={adjusted_rand_score(y, pre):.3f} cum={n_clusters(pre):>2} | "
                      f"sau_gop ARI={adjusted_rand_score(y, post):.3f} cum={n_clusters(post):>2} | "
                      f"{time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["dinh", "ga"], default="dinh")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--datasets", nargs="+", default=["flame", "s1", "s2", "deu2d"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    args = ap.parse_args()
    if args.part == "dinh":
        part_modes(args.reps)
    else:
        part_ga(args.datasets, args.seeds)


if __name__ == "__main__":
    main()
