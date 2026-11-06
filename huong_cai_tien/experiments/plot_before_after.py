"""
Ve khong gian phan cum truoc va sau cai tien tu nhan da luu boi run_before_after.py.
Khong chay lai GA (tru mo hinh NKCV-M tren Flame cho hinh co che, xac dinh).

Quy tac chon lan chay: moi phuong phap hien lan chay co ARI TRUNG VI cua cac seed,
khong chon tay, de hinh dai dien cho phan phoi chu khong phai lan dep nhat; seed
trung vi cua hai cot co the khac nhau. ARI chi dung de chon lan hien thi va ghi
tren hinh, khong dung de chinh thuat toan.

Ma hoa: cum du doan khop 1-1 (Hungarian) voi cum that VA co Jaccard >= 0.5 thi
mang mau cum that; cum con lai (manh nho hoac cum gop nhieu cum that) ve dau x;
nhieu (nhan 0) mau xam. Mau chi doi cach to, khong dong toi nhan.

Toa do: du lieu goc neu 2 chieu; PCA cua preprocess(X) neu nhieu chieu. NK-HGA goc
chay tren StandardScaler nen khong gian ve khong phai khong gian no thay.

Dau ra (results/khong_gian_cum/, PNG 300 dpi; mac dinh chi ban tieng Anh, hau to _en;
--lang vi ve them ban tieng Viet, khong hau to):
  fig1_co_che            Flame: dinh mat do, canh vuot yen ngua, can bang gop/cat
  fig2_ari               ARI goc, hai doi chung va cai tien tren 10 bo
  fig3_khong_gian_chinh  Flame, Jain, S2, S3
  fig4_khong_gian_phu    Aggregation, S1, S4
  fig5_nhieu_chieu       Iris, Ecoli, Glass (chieu PCA)
  fig6_hau_xu_ly         S1, S2 truoc va sau hau xu ly (cung mot lan chay)

Chay:
    python huong_cai_tien/experiments/plot_before_after.py
    python huong_cai_tien/experiments/plot_before_after.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from string import ascii_lowercase

import numpy as np

import _paths  # noqa: F401  (them models/ va src/ vao sys.path)
from _paths import RESULTS

from datasets import load_raw, n_clusters, n_noise, preprocess
from nkcv2 import NKCV2Model
from nkcv2_mode import ModeNKCV2Model, raw_density

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

OUT_DIR = os.path.join(RESULTS, "khong_gian_cum")
BEFORE, AFTER = "goc", "day_du_evidence"
PRE_MERGE, CONTROL, PAPER = "day_du_ga", "goc_ls26", "goc_baibao"
MATCH_JACCARD = 0.5
PAPER_COLOR = "#86b6ef"
BAR_H, BAR_GAP = 0.19, 0.02  # chieu cao cot va khe giua cac cot trong fig2
DISPLAY = {"flame": "Flame", "aggregation": "Aggregation", "jain": "Jain", "iris": "Iris",
           "ecoli": "Ecoli", "glass": "Glass", "s1": "S1", "s2": "S2", "s3": "S3", "s4": "S4"}
ALL_SETS = list(DISPLAY)
SUBSAMPLED = {"s1", "s2", "s3", "s4"}

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
NOISE, HAIRLINE = "#898781", "#e1e0d9"
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])

TEXT = {
    "vi": {"truth": "Nhãn thật", "before": "NK-HGA gốc", "after": "NK-HGA cải tiến",
           "pre": "Thuần GA (chưa hậu xử lý)", "post": "Sau hậu xử lý",
           "cluster": "cụm", "clusters": "cụm", "noise": "nhiễu",
           "matched": "Cụm khớp nhãn thật (Jaccard ≥ 0.5)", "extra": "Cụm không khớp",
           "noise_lg": "Nhiễu (nhãn 0)", "mean": "TB", "seed": "seed",
           "ari": "ARI (cao hơn là tốt hơn)", "control": "Gốc + tiền xử lý + chu kỳ LS 26",
           "paper": "Gốc, pop 100, N/2 giây (3 seed)",
           "a": "(a) Mật độ và hai đỉnh", "b": "(b) Đồ thị $G_{ep}$ và cạnh vượt yên ngựa",
           "c": "(c) Fitness toàn cục (thấp là tốt)", "peak": "Đỉnh mật độ",
           "edge": "Cạnh thường", "cross": "Cạnh vượt yên ngựa", "pairs": "cặp",
           "cost": "f × N × K",
           "merge": "Gộp thành 1 cụm", "cut": "Cắt tại yên ngựa", "rho": "ρ thô"},
    "en": {"truth": "Ground truth", "before": "Original NK-HGA", "after": "Improved NK-HGA",
           "pre": "GA only (before post-processing)", "post": "After post-processing",
           "cluster": "cluster", "clusters": "clusters", "noise": "noise",
           "matched": "Matched to truth (Jaccard ≥ 0.5)", "extra": "Unmatched cluster",
           "noise_lg": "Noise (label 0)", "mean": "mean", "seed": "seeds",
           "ari": "ARI (higher is better)", "control": "Original + preprocessing + LS period 26",
           "paper": "Original, pop 100, N/2 s (3 seeds)",
           "a": "(a) Density and the two peaks", "b": "(b) $G_{ep}$ and saddle-crossing edges",
           "c": "(c) Fitness (lower is better)", "peak": "Density peak",
           "edge": "Ordinary edge", "cross": "Saddle-crossing edge", "pairs": "pairs",
           "cost": "f × N × K",
           "merge": "Merge into 1 cluster", "cut": "Cut at the saddle", "rho": "raw ρ"},
}


@dataclass(frozen=True)
class Run:
    """Nhan da luu cua mot bo du lieu."""

    name: str
    coords: np.ndarray
    y: np.ndarray
    evr: np.ndarray
    labels: dict[str, np.ndarray]
    ari: dict[str, np.ndarray]


def load_run(name: str) -> Run:
    """Doc npz, tinh lai ARI tu nhan va doi chieu voi CSV de bat file lech nhau."""
    data = np.load(os.path.join(OUT_DIR, f"labels_{name}.npz"))
    labels = {k[4:]: data[k] for k in data.files if k.startswith("lab_")}
    y = data["y"]
    ari = {m: np.array([adjusted_rand_score(y, row) for row in v]) for m, v in labels.items()}
    with open(os.path.join(OUT_DIR, f"runs_{name}.csv"), newline="") as f:
        for r in csv.DictReader(f):
            s = int(np.flatnonzero(data["seeds"] == int(r["seed"]))[0])
            if abs(float(r["ari"]) - ari[r["method"]][s]) > 1e-12:
                raise ValueError(f"{name}/{r['method']}/{r['seed']}: CSV lech npz")
    return Run(name, data["coords"], y, data["evr"], labels, ari)


def paper_ari(name: str) -> np.ndarray:
    """ARI doi chung pop 100, N/2 giay; rong neu chua chay."""
    path = os.path.join(OUT_DIR, f"runs_baibao_{name}.csv")
    if not os.path.exists(path):
        return np.array([])
    with open(path, newline="") as f:
        return np.array([float(r["ari"]) for r in csv.DictReader(f)])


def median_index(values: np.ndarray) -> int:
    """Chi so cua phan tu trung vi (so seed chan: lay phan tu thap hon)."""
    return int(np.argsort(values, kind="stable")[(len(values) - 1) // 2])


def match_clusters(y: np.ndarray, pred: np.ndarray) -> dict[int, int]:
    """Ghep 1-1 cuc dai diem chung (bo nhieu), chi giu cap co Jaccard >= MATCH_JACCARD."""
    preds = np.unique(pred[pred > 0])
    truths = np.unique(y)
    overlap = np.array([[np.sum((pred == p) & (y == t)) for t in truths] for p in preds])
    if overlap.size == 0:
        return {}
    rows, cols = linear_sum_assignment(-overlap)
    kept = {}
    for r, c in zip(rows, cols):
        union = np.sum(pred == preds[r]) + np.sum(y == truths[c]) - overlap[r, c]
        if overlap[r, c] / union >= MATCH_JACCARD:
            kept[int(preds[r])] = int(truths[c])
    return kept


def pick_color(center: np.ndarray, taken: list[tuple[np.ndarray, int]]) -> int:
    """Mau cua bang 8 mau xa nhat voi cac cum cung mau; hoa -> so thu tu nho nhat."""
    best, best_d = 0, -1.0
    for c in range(len(PALETTE)):
        d = min((float(np.linalg.norm(center - p)) for p, i in taken if i == c),
                default=np.inf)
        if d > best_d:
            best, best_d = c, d
    return best


def unit_box(coords: np.ndarray) -> np.ndarray:
    span = np.ptp(coords, axis=0)
    return (coords - coords.min(axis=0)) / np.where(span > 0, span, 1.0)


def truth_colors(coords: np.ndarray, y: np.ndarray) -> dict[int, int]:
    """Gan 8 mau cho cum that sao cho cum gan nhau khac mau (S1-S4 co 15 cum)."""
    box = unit_box(coords)
    taken: list[tuple[np.ndarray, int]] = []
    colors = {}
    for t in np.unique(y):
        center = box[y == t].mean(axis=0)
        colors[int(t)] = pick_color(center, taken)
        taken.append((center, colors[int(t)]))
    return colors


def draw_partition(ax: plt.Axes, run: Run, pred: np.ndarray, palette: dict[int, int],
                   size: float) -> None:
    """To mot phan hoach: cum khop mang mau cum that, cum khac dau x, nhieu mau xam."""
    box = unit_box(run.coords)
    match = match_clusters(run.y, pred)
    taken = [(box[run.y == t].mean(axis=0), palette[t]) for t in palette]
    noise = pred == 0
    ax.scatter(*run.coords[noise].T, s=size * 0.8, c=NOISE, marker="s", linewidths=0, zorder=4)
    for p in np.unique(pred[pred > 0]):
        mask = pred == p
        if int(p) in match:
            color = PALETTE[palette[match[int(p)]]]
            ax.scatter(*run.coords[mask].T, s=size, c=color, linewidths=0, zorder=2)
        else:
            center = box[mask].mean(axis=0)
            slot = pick_color(center, taken)
            taken.append((center, slot))
            ax.scatter(*run.coords[mask].T, s=size * 1.6, c=PALETTE[slot], marker="x",
                       linewidths=0.6, zorder=3)


def style_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ax.spines.values():
        spine.set_color(HAIRLINE)


def count_text(lab: np.ndarray, t: dict[str, str]) -> str:
    k = n_clusters(lab)
    text = f"{k} {t['cluster'] if k == 1 else t['clusters']}"
    return f"{text}, {n_noise(lab)} {t['noise']}" if n_noise(lab) else text


def caption(run: Run, method: str, idx: int, t: dict[str, str]) -> str:
    a = run.ari[method]
    low, high, shown = (round(float(v), 3) + 0.0 for v in (a.min(), a.max(), a[idx]))
    spread = f"{t['mean']} {a.mean():.3f}"  # + 0.0 o tren tranh in "-0.000"
    if high > low:
        spread += f", [{low:.3f}; {high:.3f}]"
    return (f"ARI {shown:.3f}, {count_text(run.labels[method][idx], t)}"
            f"\n{len(a)} {t['seed']}: {spread}")


def row_label(run: Run) -> str:
    name = DISPLAY[run.name]
    if run.evr.size:
        return f"{name} (PCA {run.evr.sum() * 100:.0f}%)"
    if run.name in SUBSAMPLED:
        return f"{name} (n = {len(run.y)})"
    return name


def partition_legend(fig: plt.Figure, t: dict[str, str]) -> None:
    handles = [
        Line2D([], [], ls="", marker="o", ms=4, c=INK, label=t["matched"]),
        Line2D([], [], ls="", marker="x", ms=4, c=INK, label=t["extra"]),
        Line2D([], [], ls="", marker="s", ms=3, c=NOISE, label=t["noise_lg"]),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=8)


def panel_letter(ax: plt.Axes, index: int) -> None:
    ax.text(0.015, 0.975, f"({ascii_lowercase[index]})", transform=ax.transAxes,
            fontsize=8, color=INK, va="top")


def fig_partitions(names: list[str], t: dict[str, str]) -> plt.Figure:
    """Moi hang mot bo: nhan that | NK-HGA goc | NK-HGA cai tien (lan chay trung vi)."""
    fig, axes = plt.subplots(len(names), 3, figsize=(7.16, 2.05 * len(names) + 0.45),
                             layout="constrained", squeeze=False)
    for row, name in enumerate(names):
        run = load_run(name)
        palette = truth_colors(run.coords, run.y)
        size = 7.0 if len(run.y) <= 500 else 3.5
        draw_partition(axes[row, 0], run, run.y, palette, size)
        axes[row, 0].set_title(count_text(run.y, t), fontsize=8, color=INK_2)
        axes[row, 0].set_ylabel(row_label(run), fontsize=8, color=INK)
        for col, method in ((1, BEFORE), (2, AFTER)):
            idx = median_index(run.ari[method])
            draw_partition(axes[row, col], run, run.labels[method][idx], palette, size)
            axes[row, col].set_title(caption(run, method, idx, t), fontsize=8, color=INK_2)
        for col, ax in enumerate(axes[row]):
            style_axis(ax)
            panel_letter(ax, 3 * row + col)
    for col, key in enumerate(("truth", "before", "after")):
        axes[0, col].annotate(t[key], (0.5, 1.0), xytext=(0, 28), textcoords="offset points",
                              xycoords="axes fraction", ha="center", fontsize=9,
                              fontweight="bold", color=INK)
    partition_legend(fig, t)
    return fig


def fig_post_processing(names: list[str], t: dict[str, str]) -> plt.Figure:
    """Cung mot lan chay (trung vi theo ARI sau hau xu ly), truoc va sau hau xu ly."""
    fig, axes = plt.subplots(len(names), 2, figsize=(5.8, 3.0 * len(names)),
                             layout="constrained", squeeze=False)
    for row, name in enumerate(names):
        run = load_run(name)
        palette = truth_colors(run.coords, run.y)
        idx = median_index(run.ari[AFTER])
        for col, (method, key) in enumerate(((PRE_MERGE, "pre"), (AFTER, "post"))):
            lab = run.labels[method][idx]
            ax = axes[row, col]
            draw_partition(ax, run, lab, palette, 4.0)
            ax.set_title(f"{t[key]}\nARI {run.ari[method][idx]:.3f}, {count_text(lab, t)}",
                         fontsize=8, color=INK_2)
            style_axis(ax)
            panel_letter(ax, 2 * row + col)
        axes[row, 0].set_ylabel(row_label(run), fontsize=8, color=INK)
    partition_legend(fig, t)
    return fig


def panel_density(ax: plt.Axes, X: np.ndarray, rho: np.ndarray, peaks: np.ndarray,
                  t: dict[str, str]) -> None:
    im = ax.scatter(*X.T, c=rho, s=10, cmap=SEQ_CMAP, linewidths=0)
    ax.scatter(*X[peaks].T, marker="*", s=70, c=PALETTE[1], edgecolors="white",
               linewidths=0.5, zorder=3, label=t["peak"])
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.ax.tick_params(labelsize=6.5, length=2)
    cbar.set_label(t["rho"], fontsize=7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=7, frameon=False)
    ax.set_title(t["a"], fontsize=8, color=INK)


def panel_graph(ax: plt.Axes, X: np.ndarray, m: ModeNKCV2Model, t: dict[str, str]) -> None:
    for i, k in zip(*np.nonzero(m.Mcross == 0)):
        ax.plot(*X[[i, m.M[i, k]]].T, c=HAIRLINE, lw=0.45, zorder=1)
    for i, k in zip(*np.nonzero(m.Mcross > 0)):
        ax.plot(*X[[i, m.M[i, k]]].T, c=INK, lw=1.0, zorder=3)
    for b in range(m.n_modes_):
        ax.scatter(*X[m.basin_ == b].T, s=6, c=PALETTE[b], linewidths=0, zorder=2)
    pairs = {frozenset((int(i), int(m.M[i, k]))) for i, k in zip(*np.nonzero(m.Mcross > 0))}
    handles = [Line2D([], [], c=HAIRLINE, lw=1, label=t["edge"]),
               Line2D([], [], c=INK, lw=1.2, label=f"{t['cross']} "
                      f"({len(pairs)} {t['pairs']}, w = {m.Mcross.max():.2f})")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=7,
              frameon=False, ncol=1)
    ax.set_title(t["b"], fontsize=8, color=INK)


def panel_balance(ax: plt.Axes, costs: dict[str, tuple[float, float]],
                  t: dict[str, str]) -> None:
    """Fitness cua hai phan hoach (ca N * K), cung tap canh, duoi hai ham muc tieu."""
    for g, (merge, cut) in enumerate(costs.values()):
        bars = ax.bar([g - 0.19, g + 0.19], [merge, cut], width=0.36,
                      color=[PALETTE[0], PALETTE[1]], hatch=["", "///"],
                      edgecolor="white", linewidth=0.8)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, color=INK, padding=1.5)
    ax.set_xticks(range(len(costs)), list(costs), fontsize=8, color=INK)
    ax.set_ylabel(t["cost"], fontsize=7.5, color=INK_2)
    ax.tick_params(axis="y", labelsize=7, colors=INK_2, length=2)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 9.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(NOISE)
    handles = [Patch(fc=PALETTE[0], label=t["merge"]),
               Patch(fc=PALETTE[1], hatch="///", ec="white", label=t["cut"])]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=7,
              frameon=False)
    ax.set_title(t["c"], fontsize=8, color=INK)


def flame_balance(X: np.ndarray, m: ModeNKCV2Model) -> dict[str, tuple[float, float]]:
    """So sanh 'mot cum' voi 'cat theo luu vuc' duoi NKCV2 goc va NKCV-M."""
    base = NKCV2Model(X, K=m.K)
    one = np.ones(m.N, dtype=np.int64)
    cut = (m.basin_ + 1).astype(np.int64)
    scale = m.N * m.K
    return {"NKCV2": (base.comp_fitness(one) * scale, base.comp_fitness(cut) * scale),
            "NKCV-M": (m.comp_fitness(one) * scale, m.comp_fitness(cut) * scale)}


def fig_mechanism(t: dict[str, str]) -> plt.Figure:
    X = preprocess(load_raw("flame")[0])
    m = ModeNKCV2Model(X, K=3)
    rho = raw_density(m.D, m.dc)
    peaks = np.array([np.flatnonzero(m.basin_ == b)[np.argmax(rho[m.basin_ == b])]
                      for b in range(m.n_modes_)])
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.9), layout="constrained",
                             width_ratios=[1.1, 1, 0.92])
    panel_density(axes[0], X, rho, peaks, t)
    panel_graph(axes[1], X, m, t)
    panel_balance(axes[2], flame_balance(X, m), t)
    for ax in axes[:2]:
        style_axis(ax)
    return fig


def ari_series(name: str) -> list[tuple[np.ndarray, str]]:
    """Bon day ARI tu tren xuong duoi (goc, bai bao, doi chung, cai tien); bo day rong."""
    run = load_run(name)
    series = [(run.ari[BEFORE], PALETTE[0]), (paper_ari(name), PAPER_COLOR),
              (run.ari[CONTROL], NOISE), (run.ari[AFTER], PALETTE[1])]
    return [(v, c) for v, c in series if v.size]


def ari_bars(ax: plt.Axes, names: list[str]) -> None:
    """Cot ngang nhom theo bo; vach la khoang min-max cua cac seed, so la trung binh."""
    for row, name in enumerate(names):
        series = ari_series(name)
        for k, (vals, color) in enumerate(series):
            y = row + (k - (len(series) - 1) / 2) * (BAR_H + BAR_GAP)
            ax.barh(y, vals.mean(), height=BAR_H, color=color, linewidth=0, zorder=2)
            ax.plot([vals.min(), vals.max()], [y, y], c=INK_2, lw=0.6, zorder=3,
                    solid_capstyle="butt")
            ax.text(max(vals.mean(), vals.max()) + 0.012, y, f"{vals.mean():.3f}",
                    fontsize=6.5, color=INK, va="center", ha="left")
    ax.set_yticks(range(len(names)), [DISPLAY[n] for n in names], fontsize=8)
    ax.set(xlim=(0, 1.15), ylim=(len(names) - 0.5, -0.5))  # truc y dao: hang dau o tren
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="x", labelsize=7, colors=INK_2, length=2)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.grid(axis="x", c=HAIRLINE, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(NOISE)


def fig_ari(t: dict[str, str]) -> plt.Figure:
    """Hai panel cot ngang chung thang ARI; moi bo bon cot, khong the chong nhau."""
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 4.6), layout="constrained", sharex=True)
    ari_bars(axes[0], ALL_SETS[:5])
    ari_bars(axes[1], ALL_SETS[5:])
    for ax in axes:
        ax.set_xlabel(t["ari"], fontsize=8, color=INK_2)
    handles = [Patch(fc=PALETTE[0], label=t["before"]), Patch(fc=PAPER_COLOR, label=t["paper"]),
               Patch(fc=NOISE, label=t["control"]), Patch(fc=PALETTE[1], label=t["after"])]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False, fontsize=7.5)
    return fig


def save(fig: plt.Figure, stem: str, lang: str) -> None:
    path = os.path.join(OUT_DIR, f"{stem}{'' if lang == 'vi' else '_en'}.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print("Hinh:", path)


def render(lang: str) -> None:
    t = TEXT[lang]
    save(fig_mechanism(t), "fig1_co_che", lang)
    save(fig_ari(t), "fig2_ari", lang)
    save(fig_partitions(["flame", "jain", "s2", "s3"], t), "fig3_khong_gian_chinh", lang)
    save(fig_partitions(["aggregation", "s1", "s4"], t), "fig4_khong_gian_phu", lang)
    save(fig_partitions(["iris", "ecoli", "glass"], t), "fig5_nhieu_chieu", lang)
    save(fig_post_processing(["s1", "s2"], t), "fig6_hau_xu_ly", lang)


def self_test() -> None:
    """Ghep cum dung voi hoan vi nhan, loai manh nho; trung vi dung quy tac."""
    y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    assert match_clusters(y, np.array([7, 7, 7, 4, 4, 4, 9, 9, 9])) == {7: 1, 4: 2, 9: 3}
    split = match_clusters(y, np.array([1, 1, 5, 2, 2, 2, 3, 3, 3]))
    assert split == {1: 1, 2: 2, 3: 3}, "cum 5 phai khong khop"
    assert match_clusters(y, np.zeros(9, dtype=int)) == {}, "toan nhieu"
    fragment = match_clusters(y, np.array([1, 1, 1, 1, 1, 2, 3, 3, 3]))
    assert 2 not in fragment, "manh 1 diem khong duoc tinh la khop"
    big = np.r_[np.ones(10, dtype=int), np.full(10, 2), np.full(10, 3)]
    merged = match_clusters(big, np.r_[np.ones(20, dtype=int), np.full(10, 2)])
    assert merged == {1: 1, 2: 3} or merged == {1: 2, 2: 3}, "Jaccard 0.5 van khop"
    merged3 = match_clusters(big, np.ones(30, dtype=int))
    assert merged3 == {}, "gop ca ba cum: Jaccard 1/3, khong khop"
    assert median_index(np.array([0.9, 0.1, 0.5, 0.7, 0.3])) == 2
    assert median_index(np.array([0.2, 0.2, 0.2, 0.9, 0.1])) == 1
    near = [(np.array([0.0, 0.0]), 0)]
    assert pick_color(np.array([0.1, 0.1]), near) == 1, "khong trung mau cum gan"
    print("self-test PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="+", default=["en"], choices=list(TEXT))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    plt.rcParams.update({"font.size": 8, "hatch.linewidth": 0.6})
    for lang in args.lang:
        render(lang)


if __name__ == "__main__":
    main()
