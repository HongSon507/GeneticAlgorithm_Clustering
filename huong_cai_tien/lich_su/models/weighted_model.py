"""
Nen chung cho moi bien the NKCV2 co trong so canh: alpha = rho_j -> rho_j * w_ij.

Ba model trong nhanh cai tien (`nkcv2_cdw.CDWModel`,
`nkcv2_mahalanobis.MahalanobisModel`, `ablation_model.AblationCDWModel`) chi
khac nhau o CACH TINH `raw_w`; phan con lai -- kernel numba, cach tron lambda,
cach bat/tat tung nhanh, ba method danh gia -- truoc refactor bi lap y nguyen
ba lan (va kernel thi nam nho trong nkcv2_cdw.py, hai file kia import cheo).
File nay giu dung mot ban.

Bat bien an toan cho MOI lop con: `lam = 0` (hoac khong bat thanh phan nao)
=> `w_ij == 1` moi canh => tai tao `NKCV2Model` goc chinh xac den 1e-12.

Kernel giu nguyen cong thuc cua `src/nkcv2.py::_comp_fi`, chi nhan them mot he
so tra cuu theo (i, k) o tung nhanh -- ba mang trong so tach rieng de co the
bat/tat tung nhanh ma khong doi kernel.
"""

from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401  (them src/ vao sys.path)

from nkcv2 import njit, HAVE_NUMBA, NKCV2Model

BRANCHES = ("noise", "same", "diff")


# ---------------------------------------------------------------------------
# Kernels (numba khi co, thuan Python khi khong -- theo src/nkcv2.py)
# ---------------------------------------------------------------------------

@njit(cache=True)
def comp_fi_weighted(x, i, M, Mdist, Mrho, Mw_noise, Mw_same, Mw_diff, rho,
                     dt0, dt1, dt2, d_rho, N, K):
    fi = 0.0
    xi = x[i]
    for k in range(K):
        j = M[i, k]
        d = Mdist[i, k]
        fmax = Mrho[i, k]
        if xi == 0:
            if rho[i] <= d_rho and d > dt2:
                pass
            else:
                fi += fmax * Mw_noise[i, k]
        elif xi == x[j]:
            w = Mw_same[i, k]
            if d > dt0 and d <= dt1:
                fi += fmax * w * (d - dt0) / (dt1 - dt0)
            elif d > dt1:
                fi += fmax * w
        else:
            w = Mw_diff[i, k]
            if d <= dt0:
                fi += fmax * w
            elif d <= dt1:
                fi += (fmax - fmax * (d - dt0) / (dt1 - dt0)) * w
    return fi / (N * K)


@njit(cache=True)
def comp_fitness_weighted(x, M, Mdist, Mrho, Mw_noise, Mw_same, Mw_diff, rho,
                          dt0, dt1, dt2, d_rho, N, K):
    f = 0.0
    for i in range(N):
        f += comp_fi_weighted(x, i, M, Mdist, Mrho, Mw_noise, Mw_same,
                              Mw_diff, rho, dt0, dt1, dt2, d_rho, N, K)
    return f


@njit(cache=True)
def df_element_weighted(x, i, xi_old, M, Mdist, Mrho, Mw_noise, Mw_same,
                        Mw_diff, rho, rev_flat, rev_off, rev_cnt,
                        dt0, dt1, dt2, d_rho, N, K):
    xi_new = x[i]
    df = comp_fi_weighted(x, i, M, Mdist, Mrho, Mw_noise, Mw_same, Mw_diff,
                          rho, dt0, dt1, dt2, d_rho, N, K)
    x[i] = xi_old
    df -= comp_fi_weighted(x, i, M, Mdist, Mrho, Mw_noise, Mw_same, Mw_diff,
                           rho, dt0, dt1, dt2, d_rho, N, K)
    x[i] = xi_new
    start = rev_off[i]
    for r in range(rev_cnt[i]):
        p = rev_flat[start + r]
        df += comp_fi_weighted(x, p, M, Mdist, Mrho, Mw_noise, Mw_same,
                               Mw_diff, rho, dt0, dt1, dt2, d_rho, N, K)
        x[i] = xi_old
        df -= comp_fi_weighted(x, p, M, Mdist, Mrho, Mw_noise, Mw_same,
                               Mw_diff, rho, dt0, dt1, dt2, d_rho, N, K)
        x[i] = xi_new
    return df


# ---------------------------------------------------------------------------
# Lop nen
# ---------------------------------------------------------------------------

class WeightedNKCV2Model(NKCV2Model):
    """NKCV2 + mot mang trong so canh tinh san. Lop con chi can tinh `raw_w`
    roi goi `self._set_weights(raw_w, lam, branches)` -- hoac `self._no_weights()`
    de tra ve dung hanh vi ban goc."""

    def _no_weights(self):
        ones = np.ones((self.N, self.K), dtype=np.float64)
        self.raw_w = self.Mw_noise = self.Mw_same = self.Mw_diff = ones

    def _set_weights(self, raw_w, lam, branches):
        """raw_w: mang (N, K) da clip. lam=0 => trung ban goc chinh xac."""
        self.raw_w = raw_w
        ones = np.ones((self.N, self.K), dtype=np.float64)
        w = np.ascontiguousarray((1.0 - lam) + lam * raw_w)
        self.Mw_noise = w if "noise" in branches else ones
        self.Mw_same = w if "same" in branches else ones
        self.Mw_diff = w if "diff" in branches else ones

    # -- API danh gia (dung chung cho moi lop con) -------------------------

    def _args(self):
        return (self.M, self.Mdist, self.Mrho, self.Mw_noise, self.Mw_same,
                self.Mw_diff, self.rho)

    def comp_fi(self, x, i):
        return comp_fi_weighted(x, i, *self._args(), self.dt0, self.dt1,
                                self.dt2, self.d_rho, self.N, self.K)

    def comp_fitness(self, x):
        return comp_fitness_weighted(x, *self._args(), self.dt0, self.dt1,
                                     self.dt2, self.d_rho, self.N, self.K)

    def df_element(self, x, i, xi_old):
        return df_element_weighted(x, i, xi_old, *self._args(), self.rev_flat,
                                   self.rev_off, self.rev_cnt, self.dt0,
                                   self.dt1, self.dt2, self.d_rho,
                                   self.N, self.K)


def check_delta_eval(model, n_trials=300, seed=0, tol=1e-9):
    """Bat bien bat buoc (CLAUDE.md Sec12 test #1): df_element phai khop voi
    hieu hai lan comp_fitness. Dung chung cho self_test cua moi lop con."""
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        x = np.ascontiguousarray(rng.integers(1, 6, size=model.N).astype(np.int64))
        i = int(rng.integers(0, model.N))
        old = int(x[i])
        f0 = model.comp_fitness(x)
        x[i] = int(rng.integers(0, 6))
        delta = model.df_element(x, i, old)
        assert abs(delta - (model.comp_fitness(x) - f0)) < tol, \
            "delta-eval khong khop voi tinh lai tu dau"
    return n_trials


__all__ = ["WeightedNKCV2Model", "NKCV2Model", "BRANCHES", "HAVE_NUMBA",
           "comp_fi_weighted", "comp_fitness_weighted", "df_element_weighted",
           "check_delta_eval"]
