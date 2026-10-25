"""
Ablation wrapper quanh CDW -- bat/tat tung thanh phan trong BA thanh phan nhan
san co cua w_ij = c_ij * d_ij * b_ij (nkcv2_cdw.py::CDWModel), de do dong gop
rieng cua tung thanh phan trong dieu kien moi thu khac giu nguyen.

KHONG phai mot so do trong so moi: `compactness_term`, `density_term`,
`boundary_term` deu import y nguyen tu nkcv2_cdw.py. "Tat mot thanh phan"
nghia la thay no bang phan tu don vi cua phep nhan (mang 1.0) -- cach chuan de
ablate mot tich cac thua so doc lap.

Bat bien (self_test): bat ca ba == CDWModel chinh xac; tat ca ba == NKCV2Model
chinh xac. Ket qua do duoc tong hop trong README.md Sec9-10.
"""

from __future__ import annotations

import numpy as np

from weighted_model import (
    WeightedNKCV2Model, NKCV2Model, BRANCHES, HAVE_NUMBA, check_delta_eval)
from nkcv2_cdw import CDWModel, compactness_term, density_term, boundary_term


class AblationCDWModel(WeightedNKCV2Model):
    """CDW voi ba thanh phan bat/tat doc lap. Tham so trung khop CDWModel,
    them ba co boolean."""

    def __init__(self, X, K=3, lam=1.0, k_c=None, b_min=0.5,
                 w_min=0.1, w_max=3.0, branches=BRANCHES,
                 use_compactness=True, use_density=True, use_boundary=True):
        super().__init__(X, K=K)
        self.lam = float(lam)
        self.k_c = int(k_c) if k_c is not None else max(2 * self.K, 10)
        self.b_min = float(b_min)
        self.w_min, self.w_max = float(w_min), float(w_max)
        self.branches = tuple(branches)
        self.use_compactness = bool(use_compactness)
        self.use_density = bool(use_density)
        self.use_boundary = bool(use_boundary)

        if self.lam <= 0.0 or not (use_compactness or use_density or use_boundary):
            self._no_weights()
            return

        raw_w = np.ones((self.N, self.K), dtype=np.float64)
        if self.use_compactness:
            c_ij, self.c_ = compactness_term(self, self.k_c)
            raw_w = raw_w * c_ij
        if self.use_density:
            raw_w = raw_w * density_term(self)
        if self.use_boundary:
            raw_w = raw_w * boundary_term(self, self.b_min)

        self._set_weights(np.clip(raw_w, self.w_min, self.w_max),
                          self.lam, self.branches)


def self_test() -> None:
    """Kiem tra wrapper la mot phan ra TRUNG THUC cua cong thuc CDW san co,
    khong phai mot cong thuc moi."""
    print("Self-test AblationCDWModel...")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    x = np.ascontiguousarray(rng.integers(0, 5, size=60).astype(np.int64))
    kw = dict(lam=1.0, branches=("diff",), w_min=0.01, w_max=5.0)

    full = AblationCDWModel(X, K=3, use_compactness=True, use_density=True,
                            use_boundary=True, **kw)
    ref = CDWModel(X, K=3, **kw)
    assert abs(full.comp_fitness(x) - ref.comp_fitness(x)) < 1e-12
    assert np.allclose(full.raw_w, ref.raw_w, atol=1e-12)
    print("  [PASS] bat ca ba thanh phan == CDWModel chinh xac")

    off = AblationCDWModel(X, K=3, use_compactness=False, use_density=False,
                           use_boundary=False, **kw)
    assert abs(off.comp_fitness(x) - NKCV2Model(X, K=3).comp_fitness(x)) < 1e-12
    print("  [PASS] tat ca ba thanh phan == NKCV2Model chinh xac")

    ok = 0
    for t in range(8):  # 8 to hop bat/tat
        model = AblationCDWModel(
            X, K=3, use_compactness=bool(t & 1), use_density=bool(t & 2),
            use_boundary=bool(t & 4), **kw)
        ok += check_delta_eval(model, n_trials=40, seed=t)
    print(f"  [PASS] {ok} phep kiem delta-eval tren ca 8 to hop bat/tat")


__all__ = ["AblationCDWModel", "self_test", "HAVE_NUMBA"]

if __name__ == "__main__":
    print(f"HAVE_NUMBA={HAVE_NUMBA}\n")
    self_test()
