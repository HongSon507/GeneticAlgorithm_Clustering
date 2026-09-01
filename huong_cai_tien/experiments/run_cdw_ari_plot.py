"""
Hinh tong ket: ARI cua NK-HGA goc vs NK-HGA + CDW tren CA 4 bo du lieu co
trong repo (Aggregation / Flame / Iris / Jain), moi bo 3 seed.

Vi sao can chay lai chu khong ghep so co san: truoc file nay, IRIS chua tung
duoc chay GA voi CDW -- README.md ghi A1 truot (+58.9%) nen theo quy
trinh sang loc da bo qua. Mot hinh "4 bo du lieu" ma thieu mot cot thi phai chay
that, khong duoc dien so suy doan.

Khong sua src/, khong sua CDW: monkeypatch `ga.NKCV2Model` trong try/finally
va import nguyen ban `CDW_KWARGS` tu run_cdw_ga.py -- dung khuon mau cua
run_cdw_jain.py.

San pham duy nhat: results/cdw_ari_4datasets.png. Bang so van duoc in ra
stdout de chep vao bao cao, nhung khong ghi file CSV.

Chay:
    python huong_cai_tien/experiments/run_cdw_ari_plot.py
"""

from __future__ import annotations

import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _paths import RESULTS as RESULTS_DIR  # (cung them src/ + models/ vao sys.path)

from sklearn.metrics import adjusted_rand_score

import ga as ga_module
from ga import NKHGA
from nkcv2_cdw import CDWModel
from run_cdw_ga import CDW_KWARGS
from datasets import DATASETS, n_clusters, n_noise

# Thu tu cot: xep theo ket qua cua CDW (thang -> hoa -> thua) de hinh doc duoc
# tu trai sang phai nhu mot cau chuyen.
ORDER = ("flame", "aggregation", "iris", "jain")
POP, GEN, K = 60, 80, 3       # dong bo voi run_cdw_jain.py
SEEDS = (100, 101, 102)       # dung seed da dung cho A9 Flame/Aggregation

PNG_NAME = "cdw_ari_4datasets.png"

# Bang mau categorical da qua validator (skill dataviz):
# adjacent-pair Delta-E 24.7 protan / 33.6 normal -- PASS ca 5 kiem tra.
COLOR = {"goc": "#2a78d6", "CDW": "#eb6834"}
LABEL = {"goc": "NK-HGA gốc", "CDW": "NK-HGA + CDW"}
INK = "#0b0b0b"
INK_SOFT = "#52514e"


def run_once(X, seed, use_cdw):
    original = ga_module.NKCV2Model
    if use_cdw:
        ga_module.NKCV2Model = (lambda Xx, K=3, _kw=CDW_KWARGS:
                                CDWModel(Xx, K=K, **_kw))
    try:
        est = NKHGA(K=K, pop_size=POP, max_gen=GEN, random_state=seed,
                    verbose=False)
        t0 = time.time()
        labels = est.fit_predict(X)
        elapsed = time.time() - t0
    finally:
        ga_module.NKCV2Model = original
    return labels, elapsed


def measure() -> list[dict]:
    """24 lan chay GA: 4 bo x 3 seed x {goc, CDW}."""
    rows = []
    for name in ORDER:
        X, y_true = DATASETS[name]()
        n_true = len(np.unique(y_true))
        print(f"\n########## {name.upper()} (N={X.shape[0]}, "
              f"cụm thật={n_true}) ##########")
        for tag, use_cdw in (("goc", False), ("CDW", True)):
            for seed in SEEDS:
                labels, elapsed = run_once(X, seed, use_cdw)
                row = dict(dataset=name, n=X.shape[0], n_true=n_true,
                           method=tag, seed=seed,
                           ari=adjusted_rand_score(y_true, labels),
                           nc=n_clusters(labels), noise=n_noise(labels),
                           seconds=round(elapsed, 1))
                rows.append(row)
                print(f"  {tag:<4} seed={seed}  ARI={row['ari']:.4f}  "
                      f"cụm={row['nc']:<3d} nhiễu={row['noise']:<3d} "
                      f"{row['seconds']:.1f}s")
    return rows


def summarize(rows):
    """{dataset: {method: [ari,...]}} + so cum trung binh."""
    out = {}
    for r in rows:
        out.setdefault(r["dataset"], {}).setdefault(r["method"], []).append(r)
    return out


def plot(rows, path):
    data = summarize(rows)
    names = [n for n in ORDER if n in data]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    offset = {"goc": -0.16, "CDW": 0.16}
    for xi, name in enumerate(names):
        for tag in ("goc", "CDW"):
            recs = data[name].get(tag, [])
            if not recs:
                continue
            aris = [r["ari"] for r in recs]
            xs = xi + offset[tag]
            # tung seed mot diem (jitter nhe de khong chong nhau hoan toan)
            jit = np.linspace(-0.045, 0.045, len(aris))
            ax.scatter(xs + jit, aris, s=42, facecolors="none",
                       edgecolors=COLOR[tag], linewidths=1.6, zorder=3)
            # trung binh: gach ngang day + nhan truc tiep
            m = float(np.mean(aris))
            ax.plot([xs - 0.105, xs + 0.105], [m, m], color=COLOR[tag],
                    linewidth=2.6, solid_capstyle="round", zorder=4)
            # Nhan tach theo phuong DOC: goc xuong duoi, CDW len tren.
            # Tach ngang khong du -- nhan cua CDW nhom nay se cham nhan cua goc
            # nhom ke tiep (vi du Flame-CDW 0.933 vs Aggregation-goc 0.933).
            # Neo vao diem NGOAI CUNG cua nhom (khong phai vao gach trung binh)
            # de nhan khong bao gio de len mot seed nao -- vi du Jain: trung
            # binh 0.327 nhung co seed 0.270 nam ngay duoi.
            anchor, dy, va = ((min(aris), -13, "top") if tag == "goc"
                              else (max(aris), 13, "bottom"))
            ax.annotate(f"{m:.3f}", (xs, anchor), textcoords="offset points",
                        xytext=(0, dy), ha="center", va=va, fontsize=9.5,
                        color=INK, fontweight="bold", zorder=5)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(
        [f"{n}\nN={data[n]['goc'][0]['n']}, {data[n]['goc'][0]['n_true']} cụm"
         for n in names], fontsize=10.5, color=INK)
    ax.set_xlim(-0.55, len(names) - 0.45)
    ax.set_ylim(-0.12, 1.10)   # cho du cho nhan nam duoi gach ARI~0.01
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("ARI so với nhãn thật  (càng cao càng tốt)",
                  fontsize=10.5, color=INK_SOFT)
    ax.tick_params(colors=INK_SOFT, length=0)
    ax.grid(axis="y", color="#d9d8d4", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d9d8d4")

    handles = [plt.Line2D([], [], color=COLOR[t], marker="o", linestyle="-",
                          markerfacecolor="none", markeredgewidth=1.6,
                          markersize=7, linewidth=2.6, label=LABEL[t])
               for t in ("goc", "CDW")]
    # legend dat NGOAI vung ve (phia tren) -- de trong vung ve thi no de len
    # nhom Flame o goc duoi trai.
    ax.legend(handles=handles, frameon=False, loc="lower left",
              bbox_to_anchor=(0.0, 1.005), fontsize=10.5, labelcolor=INK,
              ncol=2, handletextpad=0.6, columnspacing=2.0)

    ax.set_title(
        "ARI của CDW trên 4 bộ dữ liệu — mỗi vòng tròn là 1 seed, gạch ngang là trung bình",
        fontsize=12.5, color=INK, fontweight="bold", pad=34, loc="left")
    fig.text(0.5, -0.02,
             f"GA: K={K}, pop={POP}, gen={GEN}, seed={SEEDS} · "
             f"CDW: {CDW_KWARGS}",
             ha="center", fontsize=8.5, color=INK_SOFT)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=150,
                facecolor=fig.get_facecolor())
    print(f"\nĐã lưu hình: {path}")


def print_table(rows):
    data = summarize(rows)
    print(f"\n{'bộ':<13}{'gốc TB':>9}{'CDW TB':>9}{'chênh':>9}"
          f"{'cụm gốc':>9}{'cụm CDW':>9}{'cụm thật':>10}")
    for name in ORDER:
        if name not in data:
            continue
        b = [r["ari"] for r in data[name]["goc"]]
        c = [r["ari"] for r in data[name]["CDW"]]
        nb = np.mean([r["nc"] for r in data[name]["goc"]])
        nc_ = np.mean([r["nc"] for r in data[name]["CDW"]])
        print(f"{name:<13}{np.mean(b):>9.4f}{np.mean(c):>9.4f}"
              f"{np.mean(c) - np.mean(b):>+9.4f}{nb:>9.1f}{nc_:>9.1f}"
              f"{data[name]['goc'][0]['n_true']:>10d}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    png_path = os.path.join(RESULTS_DIR, PNG_NAME)

    print(f"CDW_KWARGS (nguyên bản): {CDW_KWARGS}")
    print(f"GA: K={K}, pop={POP}, gen={GEN}, seeds={SEEDS}")
    print(f"Tổng số lần chạy GA = {len(ORDER)} bộ x {len(SEEDS)} seed x 2 = "
          f"{len(ORDER) * len(SEEDS) * 2}")
    t0 = time.time()
    rows = measure()
    print(f"\nXong {len(rows)} lần chạy trong {time.time() - t0:.0f}s")

    print_table(rows)
    plot(rows, png_path)


if __name__ == "__main__":
    main()
