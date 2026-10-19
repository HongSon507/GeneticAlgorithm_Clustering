"""
Sang loc "thay do do, giu nguyen cong thuc": chay NK-HGA that voi
MetricNKCV2Model (models/nkcv2_metric.py) tren bon bo du lieu.

Khong sua src/: monkeypatch `ga.NKCV2Model` trong try/finally nhu run_cdw_ga.py.
Moi seed duoc dung chung cho cac do do; D' tinh mot lan moi bo (cache).

Chay:
    python huong_cai_tien/experiments/run_metric_screen.py
    python huong_cai_tien/experiments/run_metric_screen.py --datasets iris jain \
        --metrics euclid ca_gmm --seeds 100 101 102 --pop 40 --gen 40
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
from ga import NKHGA
from nkcv2_metric import MetricNKCV2Model, build_dissimilarity
from datasets import DATASETS, n_clusters, n_noise


def run_once(X, D, metric, K, pop, gen, seed):
    original = ga.NKCV2Model
    ga.NKCV2Model = lambda Xx, K=3: MetricNKCV2Model(Xx, K=K, metric=metric, D=D)
    try:
        est = NKHGA(K=K, pop_size=pop, max_gen=gen, random_state=seed)
        t0 = time.time()
        labels = est.fit_predict(X)
        return labels, time.time() - t0
    finally:
        ga.NKCV2Model = original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["flame", "aggregation", "iris", "jain"])
    ap.add_argument("--metrics", nargs="+",
                    default=["euclid", "iso", "ca_gmm", "ca_km"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--gen", type=int, default=40)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "results", "metric_screen.csv"))
    args = ap.parse_args()

    rows = []
    for name in args.datasets:
        X, y = DATASETS[name]()
        for metric in args.metrics:
            D = build_dissimilarity(X, metric)
            for seed in args.seeds:
                lab, dt = run_once(X, D, metric, args.K, args.pop, args.gen, seed)
                r = dict(dataset=name, metric=metric, seed=seed,
                         ari=adjusted_rand_score(y, lab),
                         n_clusters=n_clusters(lab), n_noise=n_noise(lab),
                         runtime_s=dt)
                rows.append(r)
                print(f"{name:<12}{metric:<8}{seed:>5}  ARI={r['ari']:.4f}  "
                      f"cụm={r['n_clusters']:>3}  nhiễu={r['n_noise']:>3}  "
                      f"{dt:6.1f}s", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n=== Trung bình theo bộ × độ đo ===")
    print(f"{'bộ':<12}{'độ đo':<8}{'ARI TB':>8}{'SD':>8}{'cụm TB':>8}")
    for name in args.datasets:
        for metric in args.metrics:
            sel = [r for r in rows if r["dataset"] == name and r["metric"] == metric]
            a = np.array([r["ari"] for r in sel])
            c = np.mean([r["n_clusters"] for r in sel])
            print(f"{name:<12}{metric:<8}{a.mean():>8.4f}{a.std():>8.4f}{c:>8.1f}")


if __name__ == "__main__":
    main()
