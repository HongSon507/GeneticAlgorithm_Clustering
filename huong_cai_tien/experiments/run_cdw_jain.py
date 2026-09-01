"""
Task 3 -- Generalization (Jain). Muc tieu: kiem tra CDW co khop qua muc voi
hinh hoc rieng cua Flame khong, bang cach chay DUNG pipeline da dung cho
Flame/Aggregation (A1 sang loc + A9 GA that, cung CDW_KWARGS) tren mot bo du
lieu THU TU chua tung dung de thiet ke hay chinh CDW: Jain (373 diem, 2 cum
dinh nhau qua vung mat do cao -- cung ho voi Flame nhung hinh hoc khac han:
2 vong cung dai thay vi 2 khoi tron dinh qua eo hep). Day la phep kiem tra
tong quat hoa bo sung cho ket qua Flame duoc tong hop trong README.md.

Khong sua src/, khong sua CDW, khong doi CDW_KWARGS/GA_KWARGS -- import
nguyen ban tu nkcv2_cdw.py va run_cdw_ga.py::CDW_KWARGS. File nay doc lap,
khong ghi de bat ky file nao da co.

File nay chi BAO SO, khong ve hinh. Hinh ARI cua CDW tren ca 4 bo du lieu do
`run_cdw_ari_plot.py` sinh (results/cdw_ari_4datasets.png).

Chay:
    python huong_cai_tien/experiments/run_cdw_jain.py
"""

from __future__ import annotations

import time

import numpy as np

import _paths  # noqa: F401  (them src/ + models/ vao sys.path)

from sklearn.metrics import adjusted_rand_score

import ga as ga_module
from ga import NKHGA
from nkcv2_cdw import CDWModel
from run_cdw_ga import CDW_KWARGS
from datasets import load_jain, n_clusters, n_noise

POP, GEN = 60, 80          # identical to run_cdw_ga.py's A9 defaults
SEEDS = (100, 101, 102)    # identical seeds used for Flame/Aggregation A9


def run_once(X, K, pop, gen, seed, use_cdw, verbose=False):
    original = ga_module.NKCV2Model
    if use_cdw:
        ga_module.NKCV2Model = (lambda Xx, K=3, _kw=CDW_KWARGS:
                                CDWModel(Xx, K=K, **_kw))
    try:
        est = NKHGA(K=K, pop_size=pop, max_gen=gen, random_state=seed,
                    verbose=verbose)
        t0 = time.time()
        labels = est.fit_predict(X)
        elapsed = time.time() - t0
        model = est.model_
    finally:
        ga_module.NKCV2Model = original
    return labels, elapsed, model


def a1_screen(X, y_true, K=3):
    """README.md Sec7 protocol: f(true) vs f(baseline GA solution), under
    CDWModel, BEFORE running any GA under the candidate model."""
    labels_base, _, _ = run_once(X, K, POP, GEN, SEEDS[0], use_cdw=False)
    model = CDWModel(X, K=K, **CDW_KWARGS)
    y_true_x = np.ascontiguousarray(y_true.astype(np.int64))
    base_x = np.ascontiguousarray(labels_base.astype(np.int64))
    f_true = model.comp_fitness(y_true_x)
    f_base = model.comp_fitness(base_x)
    diff_pct = 100.0 * (f_true - f_base) / f_base if f_base != 0 else float("nan")
    verdict = "PASS (f_true < f_base)" if f_true < f_base else "FAIL (f_true >= f_base)"
    return labels_base, f_true, f_base, diff_pct, verdict


def main():
    X, y_true = load_jain()
    n_true = len(np.unique(y_true))
    print(f"########## JAIN (N={X.shape[0]}, cum that={n_true}) — "
         f"kiem tra tong quat hoa CDW ngoai Flame ##########")
    print(f"CDW_KWARGS (nguyen ban, khong doi): {CDW_KWARGS}")
    print(f"GA: K=3, pop={POP}, gen={GEN}, seeds={SEEDS}\n")

    print("--- A1: sang loc 30 giay (README.md Sec7) ---")
    labels_base_screen, f_true, f_base, diff_pct, verdict = a1_screen(X, y_true)
    print(f"  f(nhan that)={f_true:.6f}  f(GA goc)={f_base:.6f}  "
         f"diff%={diff_pct:+.1f}%  -> {verdict}")
    if f_true >= f_base:
        print("  Theo dung quy trinh README.md Sec7: A1 TRUOT -> ket qua GA "
             "day du duoi day van duoc chay va bao cao day du (khong bo qua, "
             "vi day la thu nghiem tong quat hoa can biet ket qua thuc te), "
             "nhung KHONG nen ky vong CDW thang tren Jain neu A1 truot.\n")
    else:
        print("  A1 DAT -> dang gia chay GA that.\n")

    print("--- A9: GA that, baseline vs CDW, 3 seed ---")
    rows = []
    for use_cdw in (False, True):
        tag = "CDW" if use_cdw else "goc"
        for seed in SEEDS:
            labels, elapsed, model = run_once(X, 3, POP, GEN, seed, use_cdw)
            ari = adjusted_rand_score(y_true, labels)
            nc, nz = n_clusters(labels), n_noise(labels)
            rows.append((tag, seed, ari, nc, nz, elapsed))
            print(f"  {tag:<5} seed={seed:<4d} ARI={ari:.4f} cum={nc:<3d} "
                 f"nhieu={nz:<3d} time={elapsed:.1f}s")

    print(f"\n{'phuong phap':<8}{'seed':>6}{'ARI':>9}{'cum':>6}{'nhieu':>7}{'time(s)':>9}")
    for tag, seed, ari, nc, nz, elapsed in rows:
        print(f"{tag:<8}{seed:>6d}{ari:>9.4f}{nc:>6d}{nz:>7d}{elapsed:>9.1f}")

    ari_base = [r[2] for r in rows if r[0] == "goc"]
    ari_cdw = [r[2] for r in rows if r[0] == "CDW"]
    print(f"\nARI trung binh (3 seed): goc={np.mean(ari_base):.4f}  "
         f"CDW={np.mean(ari_cdw):.4f}  chenh={np.mean(ari_cdw)-np.mean(ari_base):+.4f}")


if __name__ == "__main__":
    main()
