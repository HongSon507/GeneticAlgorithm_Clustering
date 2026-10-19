"""
NKCV-A -- de xuat Sec1 trong DE_XUAT_FITNESS_CHONG_LAN.md.

Them mot so hang ai luc cap TINH A_ij in [0, 1] (xac suat i, j cung cum) vao
nhanh cung nhan va khac nhan, tron voi NKCV2 goc bang beta:

    cung nhan : rho_j * [(1-beta) * ramp_same(D) + beta * (1 - A_ij)]
    khac nhan : rho_j * [(1-beta) * ramp_diff(D) + beta * A_ij      ]
    nhieu     : giu nguyen NKCV2

Khac CDW (chi nhan trong so): so hang `beta * (1 - A_ij)` phat ca viec GOP SAI
hai diem gan nhau -- dieu NKCV2 khong bieu dien duoc vi canh ngan cung nhan
luon dong gop 0.

A_ij chi doc du lieu, khong doc x => Delta Evaluation va PX van dung.
beta = 0 => tai tao NKCV2Model goc den 1e-12.

Nguon A_ij:
  ca_gmm  -- dong xuat hien tu ensemble GMM full covariance (k = 2..k_max)
  ca_km   -- nhu tren voi k-means (doi chung: khong co tin hieu hinh dang)
  gmm     -- sum_c gamma_ic * gamma_jc cua mot GMM chon k theo BIC
  snn     -- |kNN(i) n kNN(j)| / k (shared nearest neighbours)
"""

from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401  (them src/ vao sys.path)

from nkcv2 import njit, NKCV2Model, _squared_distance_matrix
from nkcv2_metric import coassociation_dissimilarity


# ---------------------------------------------------------------------------
# Ma tran ai luc A (N x N), tinh mot lan
# ---------------------------------------------------------------------------

def gmm_posterior_affinity(X, k_max=10, seed=0):
    from sklearn.mixture import GaussianMixture
    best, best_bic = None, np.inf
    for k in range(1, k_max + 1):
        g = GaussianMixture(k, covariance_type="full", reg_covar=1e-4,
                            n_init=3, random_state=seed).fit(X)
        bic = g.bic(X)
        if bic < best_bic:
            best, best_bic = g, bic
    P = best.predict_proba(X)
    return P @ P.T


def snn_affinity(X, k=None, K=3):
    k = k or max(2 * K, 10)
    D = _squared_distance_matrix(np.ascontiguousarray(X, dtype=np.float64))
    nn = np.argsort(D, axis=1)[:, 1:k + 1]
    B = np.zeros(D.shape)
    np.put_along_axis(B, nn, 1.0, axis=1)
    return (B @ B.T) / k


def build_affinity(X, source, K=3, **kw):
    if source == "ca_gmm":
        return 1.0 - coassociation_dissimilarity(X, kind="gmm", **kw)
    if source == "ca_km":
        return 1.0 - coassociation_dissimilarity(X, kind="km", **kw)
    if source == "gmm":
        return gmm_posterior_affinity(X, **kw)
    if source == "snn":
        return snn_affinity(X, K=K, **kw)
    raise ValueError(source)


# ---------------------------------------------------------------------------
# Kernels -- giu cong thuc src/nkcv2.py::_comp_fi, them so hang beta*A
# ---------------------------------------------------------------------------

@njit(cache=True)
def comp_fi_aff(x, i, M, Mdist, Mrho, MA, rho, beta, dt0, dt1, dt2, d_rho, N, K):
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
                fi += fmax
        elif xi == x[j]:
            ramp = 0.0
            if d > dt0 and d <= dt1:
                ramp = (d - dt0) / (dt1 - dt0)
            elif d > dt1:
                ramp = 1.0
            fi += fmax * ((1.0 - beta) * ramp + beta * (1.0 - MA[i, k]))
        else:
            ramp = 0.0
            if d <= dt0:
                ramp = 1.0
            elif d <= dt1:
                ramp = 1.0 - (d - dt0) / (dt1 - dt0)
            fi += fmax * ((1.0 - beta) * ramp + beta * MA[i, k])
    return fi / (N * K)


@njit(cache=True)
def comp_fitness_aff(x, M, Mdist, Mrho, MA, rho, beta, dt0, dt1, dt2, d_rho, N, K):
    f = 0.0
    for i in range(N):
        f += comp_fi_aff(x, i, M, Mdist, Mrho, MA, rho, beta,
                         dt0, dt1, dt2, d_rho, N, K)
    return f


@njit(cache=True)
def df_element_aff(x, i, xi_old, M, Mdist, Mrho, MA, rho, beta, rev_flat,
                   rev_off, rev_cnt, dt0, dt1, dt2, d_rho, N, K):
    xi_new = x[i]
    df = comp_fi_aff(x, i, M, Mdist, Mrho, MA, rho, beta,
                     dt0, dt1, dt2, d_rho, N, K)
    x[i] = xi_old
    df -= comp_fi_aff(x, i, M, Mdist, Mrho, MA, rho, beta,
                      dt0, dt1, dt2, d_rho, N, K)
    x[i] = xi_new
    start = rev_off[i]
    for r in range(rev_cnt[i]):
        p = rev_flat[start + r]
        df += comp_fi_aff(x, p, M, Mdist, Mrho, MA, rho, beta,
                          dt0, dt1, dt2, d_rho, N, K)
        x[i] = xi_old
        df -= comp_fi_aff(x, p, M, Mdist, Mrho, MA, rho, beta,
                          dt0, dt1, dt2, d_rho, N, K)
        x[i] = xi_new
    return df


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AffinityNKCV2Model(NKCV2Model):
    """NKCV2 + so hang ai luc tinh. `A` (N x N) co the truyen san de dung chung
    giua cac seed; neu khong se tinh tu `source`."""

    def __init__(self, X, K=3, beta=0.5, source="ca_gmm", A=None, **kw):
        super().__init__(X, K=K)
        self.beta = float(beta)
        self.source = source
        if A is None:
            A = build_affinity(X, source, K=K, **kw)
        self.MA = np.ascontiguousarray(
            np.take_along_axis(np.asarray(A, dtype=np.float64), self.M, axis=1))

    def comp_fi(self, x, i):
        return comp_fi_aff(x, i, self.M, self.Mdist, self.Mrho, self.MA,
                           self.rho, self.beta, self.dt0, self.dt1, self.dt2,
                           self.d_rho, self.N, self.K)

    def comp_fitness(self, x):
        return comp_fitness_aff(x, self.M, self.Mdist, self.Mrho, self.MA,
                                self.rho, self.beta, self.dt0, self.dt1,
                                self.dt2, self.d_rho, self.N, self.K)

    def df_element(self, x, i, xi_old):
        return df_element_aff(x, i, xi_old, self.M, self.Mdist, self.Mrho,
                              self.MA, self.rho, self.beta, self.rev_flat,
                              self.rev_off, self.rev_cnt, self.dt0, self.dt1,
                              self.dt2, self.d_rho, self.N, self.K)


def self_test(X, source="ca_gmm", trials=300, seed=0):
    """beta=0 tai tao goc; Delta Evaluation khop danh gia lai tu dau."""
    rng = np.random.default_rng(seed)
    A = build_affinity(X, source)
    base = NKCV2Model(X, K=3)
    zero = AffinityNKCV2Model(X, K=3, beta=0.0, A=A)
    for _ in range(20):
        x = rng.integers(0, 6, size=X.shape[0])
        assert abs(base.comp_fitness(x) - zero.comp_fitness(x)) < 1e-12
    m = AffinityNKCV2Model(X, K=3, beta=0.5, A=A)
    x = rng.integers(0, 6, size=X.shape[0])
    for _ in range(trials):
        i = int(rng.integers(X.shape[0]))
        old = int(x[i])
        f0 = m.comp_fitness(x)
        x[i] = int(rng.integers(0, 6))
        d = m.df_element(x, i, old)
        assert abs((m.comp_fitness(x) - f0) - d) < 1e-12
    return True


if __name__ == "__main__":
    from datasets import load_iris_data
    X, _ = load_iris_data()
    for src in ("ca_gmm", "ca_km", "gmm", "snn"):
        print(src, "self-test:", self_test(X, src))
