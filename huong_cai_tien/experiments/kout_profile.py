"""
Đo & phân tích OUTDEGREE (Kout) của đồ thị tương tác Gep.

Vì sao đo
---------
Bài báo NK-HGA tự nêu ở future work: *"the outdegree of Gep was not restricted...
investigation of ways of restricting the outdegree is an important future work"*.
Indegree của mọi v_i bị ép đúng K, nhưng OUTDEGREE không có trần: một đỉnh mật độ
cao có thể bị rất nhiều điểm khác trỏ density-link vào. Khi đó:

    df_element              tốn O(Kout * K)
    mutationMerge / Split   tốn O(Kout * K * |C|)

nên chi phí phình bất định tuỳ dataset.

Bất biến cần nhớ khi đọc bảng
-----------------------------
    sum_i |R_i| = N * K   (mỗi f_i đọc đúng K biến)

=> mean(|R_i|) = K với MỌI dữ liệu — đó là hằng đẳng thức, không phải kết quả thực
nghiệm. Chỉ có ĐUÔI và MAX của phân bố mới phình. Vì vậy chặn Kout là chuyện của
worst-case và của độ méo trọng số, không phải của chi phí trung bình.

Hai chỉ số đáng chú ý
---------------------
* `imbalance = kout_max / K` — hệ số bất định, đúng thứ bài báo lo ngại.
* `w_top1%` — vì rho_j được cộng đúng |R_j| lần vào f, các đỉnh hub chiếm phần
  trọng số lớn bất thường trong hàm mục tiêu. Đây là một thiên lệch KHÔNG chủ ý.
* `n_isolated` — số biến có |R_i| = 0: đổi nhãn chúng chỉ ảnh hưởng đúng f_i của
  chính nó, nên chúng gần như "vô hình" với phần còn lại của bài toán.

Chạy:
    python huong_cai_tien/experiments/kout_profile.py                    # bảng chính
    python huong_cai_tien/experiments/kout_profile.py --data iris
    python huong_cai_tien/experiments/kout_profile.py --dim-sweep --plot
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from _paths import RESULTS as RESULTS_DIR

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from nkcv2 import NKCV2Model
from datasets import DATASETS as ALL_DATASETS

HUB_FACTOR = 2          # đỉnh "hub" := |R_i| > HUB_FACTOR * K
TOP_FRACTION = 0.01     # tỉ lệ đỉnh nặng nhất dùng cho w_top1%
DIM_SWEEP = (2, 4, 8, 16, 32, 64)
DIM_SWEEP_N = 500
DIM_SWEEP_CENTERS = 5

# chi can toa do X, khong can nhan
DATASETS = {k: (lambda _k=k: ALL_DATASETS[_k]()[0])
            for k in ("iris", "aggregation", "flame")}


# ---------------------------------------------------------------------------
# Đo
# ---------------------------------------------------------------------------

def profile(model: NKCV2Model) -> dict:
    """Các chỉ số phân bố outdegree, tính từ `model.rev_cnt` (|R_i|)."""
    counts = np.asarray(model.rev_cnt, dtype=np.float64)
    N, K = model.N, model.K

    total = float(counts.sum())
    assert int(total) == N * K, "vi phạm bất biến sum|R_i| = N*K"
    assert abs(counts.mean() - K) < 1e-9, "mean(|R_i|) phải bằng K"

    n_top = max(1, int(round(TOP_FRACTION * N)))
    top_share = float(np.sort(counts)[::-1][:n_top].sum() / total)

    return {
        "N": N,
        "K": K,
        "kout_max": int(counts.max()),
        "mean": float(counts.mean()),
        "p50": float(np.percentile(counts, 50)),
        "p95": float(np.percentile(counts, 95)),
        "p99": float(np.percentile(counts, 99)),
        "n_hub": int((counts > HUB_FACTOR * K).sum()),
        "n_isolated": int((counts == 0).sum()),
        "imbalance": float(counts.max() / K),
        "w_top1": top_share,
        "counts": counts,
    }


HEADER = (f"{'bộ dữ liệu':<14}{'N':>5}{'K':>3}{'Kout_max':>9}{'mean':>7}"
          f"{'p95':>6}{'p99':>6}{'hub':>5}{'cô lập':>8}{'mất cân':>9}"
          f"{'w_top1%':>9}")


def format_row(name: str, r: dict) -> str:
    return (f"{name:<14}{r['N']:>5d}{r['K']:>3d}{r['kout_max']:>9d}"
            f"{r['mean']:>7.1f}{r['p95']:>6.0f}{r['p99']:>6.0f}"
            f"{r['n_hub']:>5d}{r['n_isolated']:>8d}"
            f"{r['imbalance']:>8.1f}x{r['w_top1'] * 100:>8.1f}%")


def main_table(data_names, k_values) -> dict:
    print("=== Phân bố outdegree |R_i| của Gep ===")
    print("(mean = K là HẰNG ĐẲNG THỨC vì sum|R_i| = N*K — chỉ đuôi/max mới phình)\n")
    print(HEADER)
    results = {}
    for name in data_names:
        X = DATASETS[name]()
        for K in k_values:
            r = profile(NKCV2Model(X, K=K))
            results[(name, K)] = r
            print(format_row(name, r))
        print()
    return results


def dim_sweep(K: int) -> list:
    """Vì sao dữ liệu 2D không chạm trần mà chiều cao thì phình: hiện tượng hubness."""
    print(f"=== Xu hướng theo số chiều (make_blobs, N={DIM_SWEEP_N}, "
          f"{DIM_SWEEP_CENTERS} cụm, K={K}) ===\n")
    print(f"{'d':>4}{'Kout_max':>10}{'p99':>7}{'hub':>6}{'mất cân':>10}"
          f"{'w_top1%':>10}")
    rows = []
    for d in DIM_SWEEP:
        X, _ = make_blobs(n_samples=DIM_SWEEP_N, centers=DIM_SWEEP_CENTERS,
                          n_features=d, random_state=0)
        X = StandardScaler().fit_transform(X)
        model = NKCV2Model(X, K=K)
        if not model.dc > 0.0:
            print(f"{d:>4}   (bỏ qua: dc suy biến ở số chiều này)")
            continue
        r = profile(model)
        rows.append((d, r))
        print(f"{d:>4}{r['kout_max']:>10d}{r['p99']:>7.0f}{r['n_hub']:>6d}"
              f"{r['imbalance']:>9.1f}x{r['w_top1'] * 100:>9.1f}%")
    print()
    return rows


# ---------------------------------------------------------------------------
# Biểu đồ
# ---------------------------------------------------------------------------

def make_plot(results: dict, sweep_rows: list, k_plot: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[CẢNH BÁO] không vẽ được biểu đồ (thiếu matplotlib?): {exc}")
        return

    panels = [(n, r) for (n, k), r in results.items() if k == k_plot]
    n_axes = len(panels) + (1 if sweep_rows else 0)
    fig, axes = plt.subplots(1, n_axes, figsize=(4.2 * n_axes, 3.6))
    axes = np.atleast_1d(axes)

    for ax, (name, r) in zip(axes, panels):
        ax.hist(r["counts"], bins=np.arange(r["counts"].max() + 2) - 0.5,
                color="steelblue", edgecolor="white")
        ax.axvline(r["K"], color="crimson", ls="--", lw=1.2,
                   label=f"mean = K = {r['K']}")
        ax.set_title(f"{name} (Kout_max={r['kout_max']})")
        ax.set_xlabel("|R_i|")
        ax.set_ylabel("số đỉnh")
        ax.legend(fontsize=8)

    if sweep_rows:
        ax = axes[-1]
        ax.plot([d for d, _ in sweep_rows],
                [r["kout_max"] for _, r in sweep_rows], "o-", color="darkorange")
        ax.axhline(k_plot, color="crimson", ls="--", lw=1.2, label=f"K = {k_plot}")
        ax.set_xscale("log", base=2)
        ax.set_title("Kout_max theo số chiều (hubness)")
        ax.set_xlabel("số chiều d")
        ax.set_ylabel("Kout_max")
        ax.legend(fontsize=8)

    fig.suptitle(f"Phân bố outdegree của Gep (K={k_plot})")
    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "kout_profile.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Đã lưu biểu đồ: {path}\n")


# ---------------------------------------------------------------------------
# Kết luận
# ---------------------------------------------------------------------------

def conclusions(results: dict, sweep_rows: list, k_ref: int) -> None:
    print("=== Kết luận (số liệu dùng thẳng cho báo cáo) ===")
    ref = {n: r for (n, k), r in results.items() if k == k_ref}
    if not ref:
        return

    print(f"1. mean(|R_i|) = K = {k_ref} trên MỌI bộ dữ liệu — hệ quả của "
          f"sum|R_i| = N*K, nên chặn Kout không hề giảm chi phí TRUNG BÌNH; "
          f"giá trị của nó nằm ở CẬN worst-case.")

    worst = max(ref.items(), key=lambda kv: kv[1]["imbalance"])
    print(f"2. Trên các bộ đang dùng, mất cân lớn nhất chỉ {worst[1]['imbalance']:.1f}x "
          f"({worst[0]}: Kout_max={worst[1]['kout_max']}), tức trần Kout gần như "
          f"KHÔNG bị chạm -> chặn Kout ở đây là no-op, an toàn nhưng vô ích.")

    if sweep_rows:
        d0, r0 = sweep_rows[0]
        dN, rN = sweep_rows[-1]
        print(f"3. Vấn đề thật nằm ở SỐ CHIỀU: Kout_max đi từ {r0['kout_max']} (d={d0}) "
              f"lên {rN['kout_max']} (d={dN}), mất cân {rN['imbalance']:.1f}x — "
              f"đúng hiện tượng hubness của kNN ở chiều cao.")

    hub_ds = max(ref.items(), key=lambda kv: kv[1]["w_top1"])
    print(f"4. Thiên lệch trọng số: rho_j được cộng đúng |R_j| lần vào f, nên "
          f"{TOP_FRACTION * 100:.0f}% đỉnh nặng nhất chiếm tới "
          f"{hub_ds[1]['w_top1'] * 100:.1f}% tổng ảnh hưởng ({hub_ds[0]}) — "
          f"một thiên lệch KHÔNG chủ ý của NKCV2.")

    iso = {n: r["n_isolated"] for n, r in ref.items() if r["n_isolated"] > 0}
    if iso:
        txt = ", ".join(f"{n}: {v}" for n, v in iso.items())
        print(f"5. Có đỉnh CÔ LẬP (|R_i| = 0) — đổi nhãn chúng chỉ ảnh hưởng f_i của "
              f"chính nó, gần như vô hình với phần còn lại: {txt}.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=list(DATASETS),
                    choices=list(DATASETS))
    ap.add_argument("--K", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--dim-sweep", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    k_ref = 3 if 3 in args.K else args.K[0]
    results = main_table(args.data, args.K)
    sweep_rows = dim_sweep(k_ref) if args.dim_sweep else []
    if args.plot:
        make_plot(results, sweep_rows, k_ref)
    conclusions(results, sweep_rows, k_ref)


if __name__ == "__main__":
    main()
