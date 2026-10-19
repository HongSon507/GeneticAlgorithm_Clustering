"""
Thi nghiem de xuat Sec1 (NKCV-A) + Sec4 (fitness hai tang) cua
DE_XUAT_FITNESS_CHONG_LAN.md.

Moi lan chay GA cho ra hai ket qua:
  * `inner`  -- nhan NK-HGA (NKCV2 hoac NKCV-A)
  * `merged` -- nhan sau khi gop tham lam theo BIC (two_level_merge.bic_merge)
nen tang ngoai khong ton them lan chay GA nao.

Bien the (A_ij tinh mot lan moi bo, dung chung cho cac seed):
  base        -- NKCV2 goc (beta = 0)
  <src>@<b>   -- NKCV-A voi nguon A_ij = src, beta = b

Khong sua src/: monkeypatch `ga.NKCV2Model` trong try/finally.
CSV ghi tung dong (flush) de khong mat du lieu neu bi ngat.

Chay:
    python huong_cai_tien/experiments/run_nkcva_two_level.py --datasets iris
    python huong_cai_tien/experiments/run_nkcva_two_level.py \
        --variants base ca_gmm@0.5 snn@0.5 --seeds 100 101 102
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
from ga import NKHGA, renumber
from nkcv2 import NKCV2Model
from nkcv2_affinity import AffinityNKCV2Model, build_affinity
from two_level_merge import bic_merge
from datasets import DATASETS, n_clusters, n_noise

DEFAULT_VARIANTS = ["base", "ca_gmm@0.25", "ca_gmm@0.5", "ca_gmm@0.75",
                    "gmm@0.5", "snn@0.5"]


def parse_variant(v):
    if v == "base":
        return None, 0.0
    src, beta = v.split("@")
    return src, float(beta)


def run_once(X, A, beta, K, pop, gen, seed):
    original = ga.NKCV2Model
    if A is not None:
        ga.NKCV2Model = lambda Xx, K=3: AffinityNKCV2Model(Xx, K=K, beta=beta, A=A)
    try:
        est = NKHGA(K=K, pop_size=pop, max_gen=gen, random_state=seed)
        t0 = time.time()
        est.fit(X)
        return est.labels_, est.model_, time.time() - t0
    finally:
        ga.NKCV2Model = original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["flame", "aggregation", "iris", "jain"])
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--gen", type=int, default=40)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(
        os.path.dirname(__file__), "..", "results",
        f"nkcva_two_level_{'_'.join(args.datasets)}.csv")
    fields = ["dataset", "variant", "source", "beta", "seed", "stage", "ari",
              "n_clusters", "n_noise", "runtime_s"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name in args.datasets:
            X, y = DATASETS[name]()
            cache = {}
            for v in args.variants:
                src, beta = parse_variant(v)
                if src is not None and src not in cache:
                    cache[src] = build_affinity(X, src, K=args.K)
                A = cache.get(src)
                for seed in args.seeds:
                    lab, model, dt = run_once(X, A, beta, args.K, args.pop,
                                              args.gen, seed)
                    merged = renumber(bic_merge(X, lab, M=model.M))
                    for stage, L in (("inner", lab), ("merged", merged)):
                        r = dict(dataset=name, variant=v, source=src or "-",
                                 beta=beta, seed=seed, stage=stage,
                                 ari=adjusted_rand_score(y, L),
                                 n_clusters=n_clusters(L), n_noise=n_noise(L),
                                 runtime_s=dt)
                        w.writerow(r)
                        f.flush()
                        print(f"{name:<12}{v:<13}{seed:>5} {stage:<7}"
                              f"ARI={r['ari']:.4f} cụm={r['n_clusters']:>3} "
                              f"nhiễu={r['n_noise']:>3} {dt:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
