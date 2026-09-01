"""Kiem dinh gia thuyet: CDW thang tren Flame chu yeu vi ha cuong do
nhanh khac-nhan xuong khoang 0.25, khong phai vi cau truc theo tung canh.

Day la phep thu chan doan doc lap. Khong sua ``src/``, NK-HGA hay CDW.
Moi model chi thay he so alpha=rho_j thanh alpha'=rho_j*w tren nhanh
``x_i != x_j``; tham so GA, seed va tieu chi dung duoc giu dung nhu hai
giao thuc da bao cao:

* ablation: pop=40, gen=60, seed 100/101;
* A9:       pop=60, gen=80, seed 100/101/102.

Khong ghi CSV. Ket qua duoc in ra stdout; ket luan da tong hop trong
``README.md`` Sec10.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

import _paths  # noqa: F401  (them src/ + models/ vao sys.path)

from sklearn.metrics import adjusted_rand_score

import ga as ga_module
from ga import NKHGA
from nkcv2 import NKCV2Model
from weighted_model import WeightedNKCV2Model, check_delta_eval
from nkcv2_cdw import CDWModel
from ablation_model import AblationCDWModel
from datasets import load_flame, n_clusters, n_noise


CDW_FIXED = dict(lam=1.0, branches=("diff",), w_min=0.01, w_max=5.0)
PROTOCOLS = {
    "ablation": dict(K=3, pop_size=40, max_gen=60, seeds=(100, 101)),
    "A9": dict(K=3, pop_size=60, max_gen=80, seeds=(100, 101, 102)),
}


class ConstantDiffWeightModel(WeightedNKCV2Model):
    """NKCV2 voi mot hang so ``w`` tren duy nhat nhanh khac-nhan."""

    def __init__(self, X, K=3, weight=1.0):
        super().__init__(X, K=K)
        self.weight = float(weight)
        if not np.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("weight phai huu han va > 0")
        raw_w = np.full((self.N, self.K), self.weight, dtype=np.float64)
        self._set_weights(raw_w, lam=1.0, branches=("diff",))


class EdgeArrayDiffWeightModel(WeightedNKCV2Model):
    """NKCV2 voi mot mang trong so tinh san tren nhanh khac-nhan.

    Dung rieng cho doi chung hoan vi: giu nguyen phan phoi trong so CDW nhung
    pha lien he giua tung gia tri va canh Gep ma no duoc gan ban dau.
    """

    def __init__(self, X, K=3, raw_w=None):
        super().__init__(X, K=K)
        weights = np.asarray(raw_w, dtype=np.float64)
        if weights.shape != (self.N, self.K):
            raise ValueError(f"raw_w phai co shape {(self.N, self.K)}")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("raw_w phai huu han va > 0")
        self._set_weights(np.ascontiguousarray(weights.copy()), lam=1.0,
                          branches=("diff",))


def self_test():
    """Khoa hai bat bien: w=1 trung ban goc; delta khop full re-eval."""
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(60, 2))
    x = np.ascontiguousarray(rng.integers(0, 6, size=60).astype(np.int64))

    base = NKCV2Model(X, K=3)
    neutral = ConstantDiffWeightModel(X, K=3, weight=1.0)
    assert abs(base.comp_fitness(x) - neutral.comp_fitness(x)) < 1e-12
    assert np.all(neutral.Mw_diff == 1.0)
    print("[PASS] w=1 trung NKCV2 goc den 1e-12")

    model = ConstantDiffWeightModel(X, K=3, weight=0.25)
    n = check_delta_eval(model, n_trials=500, seed=2026)
    print(f"[PASS] {n}/{n} delta-eval khop full re-eval tai w=0.25")


def structure_self_test(X, full, shuffled):
    """Hoan vi phai chi pha vi tri, khong doi phan phoi hay tinh cuc bo."""
    assert np.array_equal(full.M, shuffled.M)
    assert np.array_equal(np.sort(full.raw_w.ravel()),
                          np.sort(shuffled.raw_w.ravel()))
    assert abs(float(full.raw_w.mean()) -
               float(shuffled.raw_w.mean())) < 1e-15
    n = check_delta_eval(shuffled, n_trials=300, seed=314159)
    print("[PASS] shuffled giu nguyen chinh xac phan phoi raw_w cua full CDW")
    print(f"[PASS] {n}/{n} delta-eval cho doi chung shuffled")


def ga_run(X, y_true, model, ga_kwargs, seed):
    """Chay NKHGA voi model da tao san; monkeypatch luon duoc hoan tac."""
    original = ga_module.NKCV2Model
    ga_module.NKCV2Model = lambda Xx, K=3, _model=model: _model
    try:
        est = NKHGA(
            K=ga_kwargs["K"], pop_size=ga_kwargs["pop_size"],
            max_gen=ga_kwargs["max_gen"], random_state=seed, verbose=False)
        t0 = time.time()
        labels = est.fit_predict(X)
        elapsed = time.time() - t0
        return {
            "ari": float(adjusted_rand_score(y_true, labels)),
            "fitness": float(est.best_fitness_),
            "clusters": int(n_clusters(labels)),
            "noise": int(n_noise(labels)),
            "runtime_s": elapsed,
        }
    finally:
        ga_module.NKCV2Model = original


def build_models(X):
    full = CDWModel(X, K=3, **CDW_FIXED)
    cb = AblationCDWModel(
        X, K=3, use_compactness=True, use_density=False,
        use_boundary=True, **CDW_FIXED)
    mean_full = float(full.raw_w.mean())
    mean_cb = float(cb.raw_w.mean())
    models = {
        "baseline": NKCV2Model(X, K=3),
        "const_0.50": ConstantDiffWeightModel(X, K=3, weight=0.50),
        "const_0.25": ConstantDiffWeightModel(X, K=3, weight=0.25),
        "const_mean_full": ConstantDiffWeightModel(X, K=3, weight=mean_full),
        "const_mean_cb": ConstantDiffWeightModel(X, K=3, weight=mean_cb),
        "cxb": cb,
        "full_cdw": full,
    }
    return models, mean_full, mean_cb


def diff_pct(model, truth, reference):
    f_true = float(model.comp_fitness(truth))
    f_ref = float(model.comp_fitness(reference))
    return 100.0 * (f_true - f_ref) / f_ref


def run_protocol(name, X, y_true, models, config):
    seeds = config["seeds"]
    ga_kwargs = {k: config[k] for k in ("K", "pop_size", "max_gen")}

    # Nghiem tham chieu A1 dung dung baseline/seed dau cua giao thuc ablation.
    ref_est = NKHGA(
        K=ga_kwargs["K"], pop_size=ga_kwargs["pop_size"],
        max_gen=ga_kwargs["max_gen"], random_state=seeds[0], verbose=False)
    original = ga_module.NKCV2Model
    ga_module.NKCV2Model = lambda Xx, K=3, _m=models["baseline"]: _m
    try:
        reference_labels = ref_est.fit_predict(X)
    finally:
        ga_module.NKCV2Model = original

    truth = np.ascontiguousarray(y_true.astype(np.int64))
    reference = np.ascontiguousarray(reference_labels.astype(np.int64))
    rows = []
    print(f"\n{'=' * 78}\n{name}: {ga_kwargs}, seeds={seeds}\n{'=' * 78}")
    for variant, model in models.items():
        dp = diff_pct(model, truth, reference)
        for seed in seeds:
            result = ga_run(X, y_true, model, ga_kwargs, seed)
            row = {"variant": variant, "seed": seed, "diff_pct": dp, **result}
            rows.append(row)
            print(
                f"{variant:<17} seed={seed} ARI={result['ari']:.4f} "
                f"fit={result['fitness']:.6f} diff%={dp:+.1f} "
                f"Nc={result['clusters']:>2} noise={result['noise']:>2} "
                f"time={result['runtime_s']:.1f}s")

    print("\nTRUNG BINH")
    print(f"{'variant':<17}{'ARI':>9}{'fitness':>12}{'diff%':>9}{'clusters':>10}")
    for variant in models:
        subset = [r for r in rows if r["variant"] == variant]
        print(
            f"{variant:<17}"
            f"{statistics.mean(r['ari'] for r in subset):>9.4f}"
            f"{statistics.mean(r['fitness'] for r in subset):>12.6f}"
            f"{subset[0]['diff_pct']:>+9.1f}"
            f"{statistics.mean(r['clusters'] for r in subset):>10.1f}")
    return rows


def summarize(all_rows):
    print("\n" + "#" * 78)
    print("DOI CHIEU GIA THUYET CUONG DO")
    print("#" * 78)
    for protocol, rows in all_rows.items():
        means = {}
        for variant in {r["variant"] for r in rows}:
            means[variant] = statistics.mean(
                r["ari"] for r in rows if r["variant"] == variant)
        print(
            f"{protocol:<10}: const_0.25={means['const_0.25']:.4f}, "
            f"const_mean_full={means['const_mean_full']:.4f}, "
            f"cxb={means['cxb']:.4f}, full_cdw={means['full_cdw']:.4f}, "
            f"baseline={means['baseline']:.4f}")


def build_structure_models(X):
    full = CDWModel(X, K=3, **CDW_FIXED)
    mean_full = float(full.raw_w.mean())
    flat = full.raw_w.ravel().copy()
    np.random.default_rng(314159).shuffle(flat)
    shuffled = EdgeArrayDiffWeightModel(
        X, K=3, raw_w=flat.reshape(full.raw_w.shape))
    models = {
        "baseline": NKCV2Model(X, K=3),
        "const_mean": ConstantDiffWeightModel(X, K=3, weight=mean_full),
        "shuffled_cdw": shuffled,
        "full_cdw": full,
    }
    structure_self_test(X, full, shuffled)
    return models, mean_full


def _paired_values(rows, variant):
    return np.asarray([
        r["ari"] for r in sorted(rows, key=lambda row: row["seed"])
        if r["variant"] == variant
    ], dtype=np.float64)


def bootstrap_delta_ci(rows, comparator, n_boot=20000):
    """CI percentile cho delta cap: ARI(full) - ARI(comparator)."""
    full = _paired_values(rows, "full_cdw")
    comp = _paired_values(rows, comparator)
    delta = full - comp
    rng = np.random.default_rng(20260811)
    indices = rng.integers(0, delta.size, size=(n_boot, delta.size))
    boot = delta[indices].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(delta.mean()), float(lo), float(hi), delta


def run_structure_validation(X, y_true):
    """10-seed A9: hang so vs hoan vi vs full CDW."""
    models, mean_full = build_structure_models(X)
    seeds = tuple(range(100, 110))
    ga_kwargs = dict(K=3, pop_size=60, max_gen=80)
    rows = []
    print("\n" + "=" * 78)
    print("STRUCTURE VALIDATION: A9, Flame, seeds=100..109")
    print(f"mean(raw_w full CDW)={mean_full:.6f}; practical margin=0.02 ARI")
    print("=" * 78)

    for variant, model in models.items():
        for seed in seeds:
            result = ga_run(X, y_true, model, ga_kwargs, seed)
            rows.append({"variant": variant, "seed": seed, **result})
            print(
                f"{variant:<14} seed={seed} ARI={result['ari']:.4f} "
                f"fit={result['fitness']:.6f} Nc={result['clusters']:>2} "
                f"time={result['runtime_s']:.1f}s", flush=True)

    means = {
        variant: float(_paired_values(rows, variant).mean())
        for variant in models
    }
    base_gain = means["full_cdw"] - means["baseline"]
    print("\nTRUNG BINH 10 SEED")
    for variant in models:
        vals = _paired_values(rows, variant)
        print(f"{variant:<14} ARI={vals.mean():.4f} sd={vals.std(ddof=1):.4f}")

    margin = 0.02
    decisions = []
    for comparator in ("const_mean", "shuffled_cdw"):
        mean_d, lo, hi, delta = bootstrap_delta_ci(rows, comparator)
        retention = ((means[comparator] - means["baseline"]) / base_gain
                     if base_gain != 0.0 else float("nan"))
        equivalent = lo >= -margin and hi <= margin
        decisions.append(equivalent and retention >= 0.95)
        print(
            f"\nfull - {comparator}: mean={mean_d:+.4f}, "
            f"bootstrap95%=[{lo:+.4f}, {hi:+.4f}], "
            f"retain={100.0 * retention:.1f}%, "
            f"paired={np.array2string(delta, precision=4)}")
        print(f"equivalent within +/-{margin:.2f}: {equivalent}")

    removable = all(decisions)
    print("\nDECISION")
    if removable:
        print("PASS: cau truc theo canh khong cho loi ich thuc dung dang ke tren Flame; "
              "co the giu huong giam w va bo c*d*b khoi objective don gian hoa.")
    else:
        print("INCONCLUSIVE/FAIL: chua du bang chung de bo cau truc theo canh.")
    return rows, removable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure-only", action="store_true",
                    help="chi chay validation 10 seed cho gia tri cau truc theo canh")
    ap.add_argument("--original-only", action="store_true",
                    help="chi chay hai giao thuc doi chung cuong do ban dau")
    args = ap.parse_args()

    self_test()
    X, y_true = load_flame()
    print("Khong ghi file ket qua; chi in stdout.\n")

    if not args.structure_only:
        models, mean_full, mean_cb = build_models(X)
        print(f"Flame: mean(raw_w full CDW)={mean_full:.6f}")
        print(f"Flame: mean(raw_w cxb)={mean_cb:.6f}")
        all_rows = {}
        for name, config in PROTOCOLS.items():
            all_rows[name] = run_protocol(name, X, y_true, models, config)
        summarize(all_rows)

    if not args.original_only:
        run_structure_validation(X, y_true)


if __name__ == "__main__":
    main()
