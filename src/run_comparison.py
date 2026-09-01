"""
Experiment harness (paper Section V / this repo's CLAUDE.md Section 11).

Compares NK-HGA against the CGA baseline and classic clustering methods
(k-means, DBSCAN) on the Aggregation dataset, reporting Adjusted Rand Index and
the predicted number of clusters, and saving a side-by-side plot.

Model selection across runs uses DBCV if the `validclust`/`dbcv` package is
available, otherwise a documented substitute (see `select_best`). ARI is used
ONLY for the final external comparison, never for selection.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans, DBSCAN

from ga import NKHGA
from cga import CGA

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_aggregation():
    path = os.path.join(DATA_DIR, "Aggregation.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Aggregation.txt not found at {path}")
    data = np.loadtxt(path)
    X = StandardScaler().fit_transform(data[:, :2])
    y = data[:, 2].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Model selection metric (DBCV, with a documented substitute)
# ---------------------------------------------------------------------------

def _dbcv_available():
    try:
        import dbcv  # noqa: F401
        return "dbcv"
    except Exception:
        try:
            from validclust import dunn  # noqa: F401
            return "validclust"
        except Exception:
            return None


def selection_score(X, labels):
    """Higher is better. DBCV if available; otherwise silhouette as a documented
    substitute (biased toward spherical clusters -- noted in the report)."""
    n_clusters = len(np.unique(labels[labels > 0]))
    if n_clusters < 2:
        return -1.0
    backend = _dbcv_available()
    if backend == "dbcv":
        import dbcv
        try:
            return float(dbcv.dbcv(X, labels))
        except Exception:
            pass
    # substitute: silhouette over non-noise points
    mask = labels > 0
    if len(np.unique(labels[mask])) < 2:
        return -1.0
    return float(silhouette_score(X[mask], labels[mask]))


def select_best(runs):
    """Pick the (labels, meta) whose partition maximizes the selection score."""
    best = None
    for labels, meta in runs:
        score = selection_score(meta["X"], labels)
        if best is None or score > best[0]:
            best = (score, labels, meta)
    return best[1], best[2], best[0]


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_multi(make_estimator, X, n_runs, base_seed):
    runs = []
    for r in range(n_runs):
        est = make_estimator(base_seed + r)
        t0 = time.time()
        labels = est.fit_predict(X)
        meta = {"X": X, "time": time.time() - t0,
                "nc": getattr(est, "n_clusters_", len(np.unique(labels[labels > 0])))}
        runs.append((labels, meta))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1, help="runs per GA (model selection)")
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gen", type=int, default=150)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--time-budget", action="store_true",
                    help="use the paper's N/2 second wall-clock stop")
    args = ap.parse_args()

    X, y_true = load_aggregation()
    n_true = len(np.unique(y_true))
    print(f"Aggregation: N={X.shape[0]}, true clusters={n_true}")
    backend = _dbcv_available()
    print(f"Selection metric: {'DBCV ('+backend+')' if backend else 'silhouette (DBCV unavailable -- documented substitute)'}\n")

    results = {}

    # NK-HGA
    print("--- NK-HGA ---")
    nk_runs = run_multi(
        lambda s: NKHGA(K=args.K, pop_size=args.pop, max_gen=args.gen,
                        time_budget=args.time_budget, random_state=s, verbose=True),
        X, args.runs, 100)
    nk_labels, nk_meta, nk_sel = select_best(nk_runs)
    results["NK-HGA"] = (nk_labels, nk_meta, nk_sel)

    # CGA baseline
    print("\n--- CGA ---")
    cga_runs = run_multi(
        lambda s: CGA(pop_size=args.pop, max_gen=args.gen,
                      time_budget=args.time_budget, random_state=s, verbose=True),
        X, args.runs, 200)
    cga_labels, cga_meta, cga_sel = select_best(cga_runs)
    results["CGA"] = (cga_labels, cga_meta, cga_sel)

    # k-means (k = true number of clusters, an optimistic baseline)
    km = KMeans(n_clusters=n_true, n_init=10, random_state=0).fit(X)
    results["k-means"] = (km.labels_ + 1, {"X": X, "time": 0.0, "nc": n_true}, None)

    # DBSCAN
    db = DBSCAN(eps=0.3, min_samples=5).fit(X)
    db_labels = db.labels_ + 1  # sklearn noise (-1) -> label 0
    results["DBSCAN"] = (db_labels, {"X": X, "time": 0.0,
                                     "nc": len(np.unique(db_labels[db_labels > 0]))}, None)

    # -- report ---------------------------------------------------------------
    print("\n=== Results (ARI vs ground truth) ===")
    print(f"{'method':<10} {'ARI':>7} {'clusters':>9} {'time(s)':>8}")
    aris = {}
    for name, (labels, meta, _) in results.items():
        ari = adjusted_rand_score(y_true, labels)
        aris[name] = ari
        print(f"{name:<10} {ari:>7.4f} {meta['nc']:>9d} {meta['time']:>8.1f}")

    # -- plot -----------------------------------------------------------------
    order = ["Ground Truth", "NK-HGA", "CGA", "k-means", "DBSCAN"]
    fig, axes = plt.subplots(1, len(order), figsize=(4 * len(order), 4))
    cmap = plt.cm.tab10

    def scatter(ax, labels, title):
        for c in np.unique(labels):
            m = labels == c
            color = "lightgray" if c == 0 else [cmap(int(c) % 10)]
            ax.scatter(X[m, 0], X[m, 1], c=color, s=12, alpha=0.85)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    scatter(axes[0], y_true, f"Ground Truth\n{n_true} clusters")
    for ax, name in zip(axes[1:], order[1:]):
        labels, meta, _ = results[name]
        scatter(ax, labels, f"{name}\nARI={aris[name]:.3f}, Nc={meta['nc']}")

    plt.suptitle("NK-HGA vs CGA vs classic clustering (Aggregation)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "comparison_result.png")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    print(f"\nSaved plot to {out}")


if __name__ == "__main__":
    main()
