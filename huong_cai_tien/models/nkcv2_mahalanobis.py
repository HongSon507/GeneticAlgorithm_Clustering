"""
Huong thu 8 (da dong): alpha = rho_j -> rho_j * w_ij voi w_ij la ty le
Mahalanobis cuc bo -- tin hieu HINH DANG/HUONG phan bo, khac chat voi CDW
(toan tin hieu mat do/khoang cach). Ket qua va ly do dong duoc tong hop trong
README.md Sec12.1.

Tien xu ly (MOT LAN, khong doc nhan x -- rang buoc README.md Sec5):
  Sigma_i    = hiep phuong sai cuc bo tu k_cov lang gieng gan nhat + ridge
  Sigma~_ij  = 0.5 * (Sigma_i + Sigma_j)
  Dtilde_ij  = (y_i - y_j)^T Sigma~_ij^-1 (y_i - y_j)
  r_ij       = clip(D_ij / Dtilde_ij, w_min, w_max)
  w_ij(lam)  = (1 - lam) + lam * r_ij

File nay gop ca dinh nghia model, self_test, va sang loc A1 (README.md Sec7).

    python huong_cai_tien/models/nkcv2_mahalanobis.py
    python huong_cai_tien/models/nkcv2_mahalanobis.py --branches diff --invert
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from weighted_model import (  # keo theo _bootstrap: sys.path + UTF-8
    WeightedNKCV2Model, NKCV2Model, BRANCHES, HAVE_NUMBA, check_delta_eval)
from datasets import DATASETS

REAL_DATASETS = ("aggregation", "flame", "iris")


# ---------------------------------------------------------------------------
# Tien xu ly (numpy thuan, chay mot lan trong __init__)
# ---------------------------------------------------------------------------

def _local_covariance(X, D, N, l, k_cov, eps_scale):
    """Sigma_i tu k_cov lang gieng gan nhat cua i (khong ke i), + ridge nho."""
    nn = np.argsort(D, axis=1, kind="stable")[:, 1:k_cov + 1]
    centered = X[nn] - X[nn].mean(axis=1)[:, None, :]
    Sigma = np.einsum("nkl,nkm->nlm", centered, centered) / k_cov
    eps = eps_scale * np.trace(Sigma, axis1=1, axis2=2) / l
    eps = np.where(eps > 0.0, eps, 1e-9)
    return Sigma + eps[:, None, None] * np.eye(l)[None, :, :]


def _mahalanobis_ratio(X, M, Mdist, Sigma, normalize_scale=True):
    """r_ij = D_ij (dang huong) / Dtilde_ij (Mahalanobis cuc bo).

    SUA SO VOI CONG THUC NGUYEN VAN (do thuc, xem README.md Sec12.1): dung
    Sigma_i tho thi o vung DANG HUONG
    (Sigma = sigma^2 I) ty le nay bang sigma^2 -- tuc chi con la MAT DO cuc bo,
    khong phai tin hieu hinh dang. Do tren Flame: median ~0.037, 99.7% duoi
    0.2 => moi canh bi kep ve san w_min, lap lai dung huong da chet #4.

    Chuan hoa Sigma_i ve dinh thuc 1 truoc khi nghich dao moi tach duoc tin
    hieu HINH DANG khoi tin hieu MAT DO. Sau chuan hoa: vung dang huong cho
    r_ij == 1 chinh xac; tren Flame median 1.006, p10-p90 = 0.81-1.33.
    `normalize_scale=False` giu lai lam doi chung.
    """
    if normalize_scale:
        l = Sigma.shape[-1]
        det = np.maximum(np.linalg.det(Sigma), 1e-300)
        Sigma = Sigma / np.power(det, 1.0 / l)[:, None, None]
    Sigma_inv = np.linalg.inv(0.5 * (Sigma[:, None, :, :] + Sigma[M]))
    diff = X[:, None, :] - X[M]
    Dtilde = np.maximum(np.einsum("nkl,nklm,nkm->nk", diff, Sigma_inv, diff), 1e-12)
    return Mdist / Dtilde


# ---------------------------------------------------------------------------

class MahalanobisModel(WeightedNKCV2Model):
    """NKCV2 + w_ij = ty le Mahalanobis cuc bo (PCA lan can moi diem).

        lam              he so tron 0..1; 0 = tai tao ban goc chinh xac.
        k_cov            so lang gieng uoc luong Sigma_i (mac dinh max(2K,10)).
        w_min/max        chan r_ij truoc khi tron.
        eps_scale        he so ridge cho Sigma_i.
        normalize_scale  chuan hoa Sigma_i ve dinh thuc 1 (mac dinh True, BAT
                         BUOC de tach hinh dang khoi mat do -- xem tren).
        invert           dung 1/r_ij thay vi r_ij. Ly do: tren Flame cac canh
                         bac qua bien THAT nam DOC huong co gian cuc bo
                         (r_ij > 1 o do), nen nhan w>1 vao nhanh khac-nhan lai
                         KHUECH DAI chi phi cua nhan that -- sai chieu.
        branches         tap con {"noise","same","diff"} duoc ap w_ij.
    """

    def __init__(self, X, K=3, lam=1.0, k_cov=None, w_min=0.2, w_max=5.0,
                 eps_scale=1e-6, normalize_scale=True, invert=False,
                 branches=BRANCHES):
        super().__init__(X, K=K)
        self.lam = float(lam)
        self.k_cov = int(k_cov) if k_cov is not None else max(2 * self.K, 10)
        self.w_min, self.w_max = float(w_min), float(w_max)
        self.eps_scale = float(eps_scale)
        self.normalize_scale = bool(normalize_scale)
        self.invert = bool(invert)
        self.branches = tuple(branches)

        if self.k_cov >= self.N:
            raise ValueError("k_cov phai nho hon N")
        if self.lam <= 0.0:
            self._no_weights()
            return

        self.Sigma_ = _local_covariance(self.X, self.D, self.N,
                                        self.X.shape[1], self.k_cov,
                                        self.eps_scale)
        ratio = _mahalanobis_ratio(self.X, self.M, self.Mdist, self.Sigma_,
                                   self.normalize_scale)
        if self.invert:
            ratio = 1.0 / np.maximum(ratio, 1e-12)
        self._set_weights(np.clip(ratio, self.w_min, self.w_max),
                          self.lam, self.branches)


def self_test() -> None:
    print("Self-test MahalanobisModel...")
    rng = np.random.default_rng(0)

    for l, N in ((2, 60), (4, 50)):   # l=2 giong Flame/Aggregation, l=4 giong Iris
        X = rng.normal(size=(N, l))
        x = np.ascontiguousarray(rng.integers(0, 5, size=N).astype(np.int64))
        assert abs(NKCV2Model(X, K=3).comp_fitness(x)
                   - MahalanobisModel(X, K=3, lam=0.0).comp_fitness(x)) < 1e-12
        gm = MahalanobisModel(X, K=3, lam=1.0, k_cov=10)
        assert np.all(gm.raw_w >= gm.w_min - 1e-9) and np.all(gm.raw_w <= gm.w_max + 1e-9)
        n = check_delta_eval(gm, n_trials=200 if l == 2 else 60, seed=l)
        print(f"  [PASS] l={l}: lam=0 trung ban goc, {n} phep kiem delta-eval")

    # bat bien quan trong nhat: normalize_scale phai tach hinh dang khoi mat do
    Xiso = rng.normal(scale=0.05, size=(300, 2))   # dac, dang huong
    g_norm = MahalanobisModel(Xiso, K=3, lam=1.0, k_cov=10, normalize_scale=True)
    g_raw = MahalanobisModel(Xiso, K=3, lam=1.0, k_cov=10, normalize_scale=False)
    assert abs(g_norm.raw_w.mean() - 1.0) < 0.3, g_norm.raw_w.mean()
    assert g_raw.raw_w.mean() < 0.3, g_raw.raw_w.mean()
    print(f"  [PASS] normalize_scale=True cho raw_w~1 tren vung dang huong "
          f"({g_norm.raw_w.mean():.3f}); =False sup ve {g_raw.raw_w.mean():.3f}")


# ---------------------------------------------------------------------------
# A1 -- sang loc 30 giay (README.md Sec7)
# ---------------------------------------------------------------------------

def screen_one(name, X, y_true, lambdas, k_cov, w_min, w_max, pop, gen, seed,
               branches=BRANCHES, invert=False):
    from ga import NKHGA

    print(f"\n########## {name.upper()} (N={X.shape[0]}) ##########")
    t0 = time.time()
    labels_ga = NKHGA(K=3, pop_size=pop, max_gen=gen,
                      random_state=seed, verbose=False).fit_predict(X)
    print(f"  loi giai GA goc: {len(np.unique(labels_ga[labels_ga > 0]))} cum, "
          f"{(labels_ga == 0).sum()} nhieu, {time.time()-t0:.1f}s")

    y_x = np.ascontiguousarray(y_true.astype(np.int64))
    ga_x = np.ascontiguousarray(labels_ga.astype(np.int64))
    print(f"  {'lambda':>7}{'f(true)':>12}{'f(GA)':>12}{'diff%':>9}"
          f"{'raw_w':>9}  verdict")
    for lam in lambdas:
        m = MahalanobisModel(X, K=3, lam=lam, k_cov=k_cov, w_min=w_min,
                             w_max=w_max, branches=branches, invert=invert)
        f_true, f_ga = m.comp_fitness(y_x), m.comp_fitness(ga_x)
        dp = 100.0 * (f_true - f_ga) / f_ga if f_ga else float("nan")
        verdict = "PASS" if f_true < f_ga else "FAIL (f_true >= f_GA)"
        print(f"  {lam:>7.2f}{f_true:>12.6f}{f_ga:>12.6f}{dp:>8.1f}%"
              f"{m.raw_w.mean():>9.3f}  {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, nargs="+", default=[0.0, 0.3, 0.6, 1.0])
    ap.add_argument("--k-cov", type=int, default=None)
    ap.add_argument("--w-min", type=float, default=0.2)
    ap.add_argument("--w-max", type=float, default=5.0)
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gen", type=int, default=80)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--branches", nargs="+", default=list(BRANCHES),
                    choices=list(BRANCHES))
    ap.add_argument("--data", nargs="+", default=list(REAL_DATASETS),
                    choices=list(REAL_DATASETS))
    ap.add_argument("--invert", action="store_true",
                    help="dung 1/r_ij thay vi r_ij (dao chieu tin hieu)")
    ap.add_argument("--skip-self-test", action="store_true")
    args = ap.parse_args()

    print(f"HAVE_NUMBA={HAVE_NUMBA}\n")
    if not args.skip_self_test:
        self_test()
    for name in args.data:
        X, y_true = DATASETS[name]()
        screen_one(name, X, y_true, args.lam, args.k_cov, args.w_min,
                   args.w_max, args.pop, args.gen, args.seed,
                   branches=tuple(args.branches), invert=args.invert)


__all__ = ["MahalanobisModel", "self_test", "HAVE_NUMBA"]

if __name__ == "__main__":
    main()
