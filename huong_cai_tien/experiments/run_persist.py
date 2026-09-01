"""
Chạy & so sánh: NK-HGA gốc  vs  NK-HGA + cổng persistence (Đề xuất 5).

Test trên CẢ HAI chế độ:
  * Flame       -> hai cụm dính nhau qua vùng nối dày (bản gốc hỏng: ARI ~0.01)
  * Aggregation -> cụm tách rời "bình thường"          (bản gốc rất mạnh: ARI 0.99)

Tiêu chí chấp nhận: THẮNG rõ trên Flame mà KHÔNG tụt trên Aggregation.

Không sửa src/: monkeypatch `ga.NKCV2Model` bằng `PersistGateModel`.

Chạy:
    python huong_cai_tien/run_persist.py --self-test
    python huong_cai_tien/run_persist.py --quiet
    python huong_cai_tien/run_persist.py --data flame --w 0 0.1
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import _paths  # noqa: F401  (them src/ + models/ vao sys.path)

from sklearn.metrics import adjusted_rand_score

import ga
from ga import NKHGA
from nkcv2 import NKCV2Model
from nkcv2_persist import PersistGateModel, HAVE_NUMBA
from datasets import load_flame, n_clusters, n_noise
from datasets import load_aggregation

DATASETS = {"flame": load_flame, "aggregation": load_aggregation}


def self_test() -> None:
    print("Self-test PersistGateModel...")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))

    base = NKCV2Model(X, K=3)
    g1 = PersistGateModel(X, K=3, w_cross=1.0)
    x = np.ascontiguousarray(rng.integers(0, 5, size=60).astype(np.int64))
    assert abs(base.comp_fitness(x) - g1.comp_fitness(x)) < 1e-12, \
        "w_cross=1 phải trùng bản gốc"
    print("  [PASS] w_cross=1 trùng khớp bản gốc chính xác")

    gm = PersistGateModel(X, K=3, w_cross=0.1, knn=8)
    assert gm.Mw.shape == (gm.N, gm.K)
    assert np.all((gm.Mw == 1.0) | (gm.Mw == 0.1)), "Mw chỉ nhận 2 giá trị"
    print(f"  [PASS] cổng hợp lệ (tau={gm.tau:.4f}, {gm.n_modes_} mode, "
          f"{(gm.Mw < 1).mean():.1%} cạnh được mở cổng)")

    for _ in range(500):
        x = np.ascontiguousarray(rng.integers(1, 6, size=gm.N).astype(np.int64))
        i = int(rng.integers(0, gm.N))
        b = int(rng.integers(0, 6))
        old = int(x[i])
        f0 = gm.comp_fitness(x)
        x[i] = b
        delta = gm.df_element(x, i, old)
        f1 = gm.comp_fitness(x)
        assert abs(delta - (f1 - f0)) < 1e-9, "delta != full re-eval"
    print("  [PASS] 500/500 delta khớp full re-eval (w_cross=0.1)")


def run_once(X, K, pop, gen, seed, w_cross, knn, verbose):
    """w_cross=None -> bản gốc; ngược lại monkeypatch PersistGateModel."""
    original = ga.NKCV2Model
    if w_cross is not None:
        ga.NKCV2Model = (lambda Xx, K=3, _w=w_cross, _k=knn:
                         PersistGateModel(Xx, K=K, w_cross=_w, knn=_k))
    try:
        est = NKHGA(K=K, pop_size=pop, max_gen=gen, random_state=seed,
                    verbose=verbose)
        t0 = time.time()
        labels = est.fit_predict(X)
        elapsed = time.time() - t0
    finally:
        ga.NKCV2Model = original
    return labels, est, elapsed


def evaluate(name, X, y_true, args, verbose):
    print(f"\n########## {name.upper()} "
          f"(N={X.shape[0]}, cụm thật={len(np.unique(y_true))}) ##########")
    rows = []

    print("\n--- NK-HGA gốc ---")
    lab, _, dt = run_once(X, args.K, args.pop, args.gen, args.seed,
                          None, args.knn, verbose)
    rows.append(("gốc", adjusted_rand_score(y_true, lab), n_clusters(lab),
                 n_noise(lab), dt))

    for w in args.w:
        print(f"\n--- NK-HGA + cổng persistence (w_cross={w}) ---")
        lab, _, dt = run_once(X, args.K, args.pop, args.gen, args.seed,
                              w, args.knn, verbose)
        rows.append((f"cổng (w={w})", adjusted_rand_score(y_true, lab),
                     n_clusters(lab), n_noise(lab), dt))

    print(f"\n=== {name}: ARI vs nhãn thật ===")
    print(f"{'phương pháp':<22}{'ARI':>8}{'cụm':>6}{'nhiễu':>7}{'time(s)':>9}")
    for r in rows:
        print(f"{r[0]:<22}{r[1]:>8.4f}{r[2]:>6d}{r[3]:>7d}{r[4]:>9.1f}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=80)
    ap.add_argument("--gen", type=int, default=100)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--w", type=float, nargs="+", default=[0.0, 0.1, 0.3],
                    help="hệ số phạt còn lại cho cạnh vượt mode (1 = bản gốc)")
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--data", nargs="+", default=["flame", "aggregation"],
                    choices=list(DATASETS))
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    print(f"HAVE_NUMBA={HAVE_NUMBA}\n")
    self_test()
    if args.self_test:
        return

    summary = {}
    for name in args.data:
        X, y_true = DATASETS[name]()
        summary[name] = evaluate(name, X, y_true, args, not args.quiet)

    print("\n\n############## TỔNG KẾT (ARI / số cụm) ##############")
    print(f"{'phương pháp':<22}" + "".join(f"{n[:12]:>15}" for n in summary))
    methods = [r[0] for r in next(iter(summary.values()))]
    for idx, m in enumerate(methods):
        cells = "".join(f"{summary[n][idx][1]:>10.4f}/{summary[n][idx][2]:<4d}"
                        for n in summary)
        print(f"{m:<22}{cells}")


if __name__ == "__main__":
    main()
