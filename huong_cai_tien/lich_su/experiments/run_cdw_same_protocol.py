"""
CDW chay theo CUNG giao thuc voi bang cuoi (experiments/run_final_table.py):
10 bo, pop 60, gen 80; --seeds va --subsample nhu run_final_table.py
(vd. 5 seed va S1-S4 mau con 1000 diem de chay nhanh).

  cdw          -- CDWModel (cau hinh A9: diff-only, lam=1, w_min=0.01, w_max=5),
                  khoi tao ngau nhien, StandardScaler -- dung nhu luc CDW duoc de xuat.
  cdw_seed     -- CDWModel + gieo quan the (seeded_ga) + tien xu ly moi
                  (datasets.preprocess), de xem CDW co cong huong voi cai tien tim kiem.

`fast_px` duoc bat de tang toc (dau ra trung src/px.py). Model CDW duoc dung mot
lan moi bo va dung chung cho moi seed (model khong doi trang thai trong GA).

Chay:
    python huong_cai_tien/lich_su/experiments/run_cdw_same_protocol.py --datasets flame jain
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import _paths  # noqa: F401
from _paths import RESULTS

from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import fast_px
import ga
from ga import NKHGA
from nkcv2_cdw import CDWModel
from run_cdw_ga import CDW_KWARGS
from seeded_ga import SeededNKHGA, build_seed_pool
from datasets import load_raw, n_clusters, n_noise, preprocess, subsample

FIELDS = ["dataset", "n", "method", "seed", "ari", "n_clusters", "n_noise",
          "runtime_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(100, 110)))
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gen", type=int, default=80)
    ap.add_argument("--subsample", type=int, default=None,
                    help="lay ngau nhien n diem (seed 0), nhu run_final_table.py")
    args = ap.parse_args()

    fast_px.install()
    tag = "_".join(args.datasets) + (f"_n{args.subsample}" if args.subsample else "")
    out = os.path.join(RESULTS, f"cdw_same_protocol_{tag}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for name in args.datasets:
            X_raw, y = subsample(*load_raw(name), args.subsample)
            variants = {
                "cdw": (StandardScaler().fit_transform(X_raw), False),
                "cdw_seed": (preprocess(X_raw), True),
            }
            for method, (X, seeded) in variants.items():
                model = CDWModel(X, K=3, **CDW_KWARGS)
                pool = build_seed_pool(X) if seeded else None
                original = ga.NKCV2Model
                ga.NKCV2Model = lambda Xx, K=3, _m=model: _m
                try:
                    for seed in args.seeds:
                        kw = dict(K=3, pop_size=args.pop, max_gen=args.gen,
                                  random_state=seed)
                        est = (SeededNKHGA(pool=pool, seed_frac=0.3, **kw)
                               if seeded else NKHGA(**kw))
                        t0 = time.time()
                        lab = est.fit_predict(X)
                        r = dict(dataset=name, n=len(y), method=method, seed=seed,
                                 ari=adjusted_rand_score(y, lab),
                                 n_clusters=n_clusters(lab), n_noise=n_noise(lab),
                                 runtime_s=time.time() - t0)
                        w.writerow(r)
                        f.flush()
                        print(f"{name:<12}{method:<10}{seed:>5} ARI={r['ari']:.4f} "
                              f"cụm={r['n_clusters']:>4}", flush=True)
                finally:
                    ga.NKCV2Model = original


if __name__ == "__main__":
    main()
