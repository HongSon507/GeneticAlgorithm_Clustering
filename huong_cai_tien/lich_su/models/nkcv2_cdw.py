"""
CDW -- de xuat cua nguoi dung: alpha = rho_j  ->  alpha = rho_j * w_ij.

w_ij = c_ij * d_ij * b_ij, ca 3 thanh phan tinh MOT LAN khi tien xu ly, KHONG
doc nhan x hien tai (bat buoc theo README.md Sec5: comp_fi phai hang so theo x
de Delta-eval O(Kout*K) va Partition Crossover con dung). Cong thuc, ket qua
Flame va kiem tra tong quat hoa Jain duoc tong hop trong README.md Sec6-10.

  c_ij = sqrt(c_i * c_j)   "compactness": c_i = 1/(1+scale_i/median(scale)),
      scale_i = trung binh khoang cach binh phuong toi k_c lang gieng gan nhat.
  d_ij = rho_mid(i,j) / max(rho_i, rho_j)   "local density": mat do tai TRUNG
      DIEM canh so voi hai dau mut; nho <=> co thung lung mat do giua i va j.
  b_ij = b_min + (1-b_min)*clip((D_ij-dt0)/(dt1-dt0), 0, 1)   "boundary
      distance": dung lai dt0/dt1 co san, khong them nguong moi.

  raw_w = clip(c_ij*d_ij*b_ij, w_min, w_max)
  w_ij(lambda) = (1-lambda) + lambda*raw_w    -- lambda=0 => trung ban goc 1e-12

Kernel, cach tron lambda va ba method danh gia nam o weighted_model.py (dung
chung voi MahalanobisModel va AblationCDWModel). README.md Sec9-10 ghi day du
ablation va doi chung cuong do/cau truc theo canh.
"""

from __future__ import annotations

import numpy as np

from weighted_model import (
    WeightedNKCV2Model, NKCV2Model, BRANCHES, HAVE_NUMBA, check_delta_eval)


# ---------------------------------------------------------------------------
# Ba thanh phan (deu chi phu thuoc du lieu tho, khong phu thuoc nhan)
# ---------------------------------------------------------------------------

def _local_scale(D: np.ndarray, N: int, k_c: int) -> np.ndarray:
    """scale_i = trung binh khoang cach binh phuong toi k_c lang gieng gan nhat."""
    order = np.argsort(D, axis=1, kind="stable")  # cot 0 la chinh no (d=0)
    nn = order[:, 1:k_c + 1]
    return np.take_along_axis(D, nn, axis=1).mean(axis=1)


def _compactness(scale: np.ndarray) -> np.ndarray:
    med = np.median(scale)
    return 1.0 / (1.0 + scale / (med if med > 0.0 else 1.0))


def _raw_density(D: np.ndarray, dc: float, N: int) -> np.ndarray:
    """Mat do Gaussian-kernel CHUA chuan hoa -- can lam moc thang do cho
    rho_mid, vi NKCV2Model.rho da chia cho max nen khong suy nguoc duoc."""
    raw = np.zeros(N, dtype=np.float64)
    iu = np.triu_indices(N, k=1)
    contrib = np.exp(-((D[iu] / dc) ** 2))
    np.add.at(raw, iu[0], contrib)
    np.add.at(raw, iu[1], contrib)
    return raw


def _valley_density_ratio(X, dc, raw_rho, rho_norm, M, N, K) -> np.ndarray:
    """d_ij = rho_mid(i,j) / max(rho_i, rho_j), cung thang chuan hoa voi rho."""
    raw_max = raw_rho.max()
    d_ratio = np.empty((N, K), dtype=np.float64)
    for i in range(N):
        for k in range(K):
            j = int(M[i, k])
            m = 0.5 * (X[i] + X[j])
            diff = X - m[None, :]
            rho_mid = np.sum(np.exp(-((np.sum(diff * diff, axis=1) / dc) ** 2)))
            denom = max(rho_norm[i], rho_norm[j], 1e-12)
            d_ratio[i, k] = (rho_mid / raw_max) / denom
    return d_ratio


# -- ba thanh phan o dang dung lai duoc (ablation_model.py dung y nguyen) ----

def compactness_term(model, k_c):
    """c_ij (N, K) + vector c_i."""
    c = _compactness(_local_scale(model.D, model.N, k_c))
    return np.sqrt(c[:, None] * c[model.M]), c


def density_term(model):
    """d_ij (N, K)."""
    raw_rho = _raw_density(model.D, model.dc, model.N)
    return _valley_density_ratio(model.X, model.dc, raw_rho, model.rho,
                                 model.M, model.N, model.K)


def boundary_term(model, b_min):
    """b_ij (N, K)."""
    return b_min + (1.0 - b_min) * np.clip(
        (model.Mdist - model.dt0) / (model.dt1 - model.dt0), 0.0, 1.0)


# ---------------------------------------------------------------------------

class CDWModel(WeightedNKCV2Model):
    """NKCV2 + w_ij = compactness * local-density * boundary-distance.

        lam        he so tron 0..1; 0 = tai tao ban goc chinh xac.
        k_c        so lang gieng tinh compactness (mac dinh 2K, toi thieu 10).
        b_min      san duoi cua thanh phan boundary (0, 1].
        w_min/max  chan raw_w truoc khi tron bang lambda.
        branches   tap con {"noise","same","diff"} duoc ap w_ij; nhanh ngoai
                   tap nay giu w=1 (hanh vi ban goc).
    """

    def __init__(self, X, K=3, lam=0.3, k_c=None, b_min=0.5,
                 w_min=0.1, w_max=3.0, branches=BRANCHES):
        super().__init__(X, K=K)
        self.lam = float(lam)
        self.k_c = int(k_c) if k_c is not None else max(2 * self.K, 10)
        self.b_min = float(b_min)
        self.w_min, self.w_max = float(w_min), float(w_max)
        self.branches = tuple(branches)

        if self.lam <= 0.0:
            self._no_weights()
            return

        c_ij, self.c_ = compactness_term(self, self.k_c)
        raw_w = np.clip(
            c_ij * density_term(self) * boundary_term(self, self.b_min),
            self.w_min, self.w_max)
        self._set_weights(raw_w, self.lam, self.branches)


def self_test() -> None:
    print("Self-test CDWModel...")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    x = np.ascontiguousarray(rng.integers(0, 5, size=60).astype(np.int64))

    assert abs(NKCV2Model(X, K=3).comp_fitness(x)
               - CDWModel(X, K=3, lam=0.0).comp_fitness(x)) < 1e-12, \
        "lambda=0 phai trung ban goc"
    print("  [PASS] lambda=0 trung khop ban goc chinh xac")

    gm = CDWModel(X, K=3, lam=0.5, k_c=8)
    assert gm.Mw_diff.shape == (gm.N, gm.K)
    assert np.all(gm.raw_w >= gm.w_min - 1e-12) and np.all(gm.raw_w <= gm.w_max + 1e-12)
    print(f"  [PASS] w_ij hop le (raw_w in [{gm.w_min},{gm.w_max}], "
          f"trung binh={gm.raw_w.mean():.3f})")

    n = check_delta_eval(gm, n_trials=500, seed=1)
    print(f"  [PASS] {n}/{n} delta khop full re-eval (lambda=0.5)")


__all__ = ["CDWModel", "self_test", "HAVE_NUMBA", "compactness_term",
           "density_term", "boundary_term"]

if __name__ == "__main__":
    print(f"HAVE_NUMBA={HAVE_NUMBA}\n")
    self_test()
