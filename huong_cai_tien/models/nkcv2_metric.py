"""
NKCV2 voi do do thay the -- KHONG them so hang nao vao cong thuc.

Y tuong: `NKCV2Model` dung moi thu (dc, rho, d_rho, Gep, dt0/dt1/dt2) tu DUY
NHAT ma tran khoang cach D. Model nay chi thay D bang mot do bat tuong dong
tinh khac; cong thuc `comp_fi` va toan bo kernel trong `src/nkcv2.py` giu
nguyen tung ky tu. Vi D' van la hang so theo x, Delta Evaluation va PX dung.

Cac do do:
  euclid   -- D goc (binh phuong Euclid); phai tai tao NKCV2Model den 1e-12
  iso      -- Isolation dissimilarity kieu aNNE (Qin, Ting, Zhu, Lee, AAAI 2019):
              t lan lay psi diem mau, chia khong gian thanh o Voronoi;
              sim_ij = ti le lan i, j cung o; D' = 1 - sim
  ca_gmm   -- khoang cach dong xuat hien tu ensemble GMM full covariance
              (k = 2..k_max, nhieu seed): D' = 1 - CA
  ca_km    -- nhu tren nhung voi k-means (doi chung: khong co tin hieu hinh dang)

D' co nhieu gia tri bang nhau (vd. CA = 1 trong long cum) lam quy tac 2% cho
dc = 0. Them tie-breaker rat nho `eps * D/max(D)` de giu thu tu lan can theo
Euclid khi D' hoa nhau. eps=0 de tat.
"""

from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401  (them src/ vao sys.path)

from nkcv2 import (NKCV2Model, _squared_distance_matrix, _cutoff_distance,
                   _density, _build_graph, _three_sigma, _build_reverse)


# ---------------------------------------------------------------------------
# Do bat tuong dong
# ---------------------------------------------------------------------------

def isolation_dissimilarity(X, psi=16, t=200, seed=0):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    same = np.zeros((N, N))
    for _ in range(t):
        idx = rng.choice(N, size=min(psi, N), replace=False)
        cell = np.argmin(((X[:, None, :] - X[None, idx, :]) ** 2).sum(-1), axis=1)
        same += cell[:, None] == cell[None, :]
    return 1.0 - same / t


def coassociation_dissimilarity(X, kind="gmm", k_max=10, n_seeds=5, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    N = X.shape[0]
    same = np.zeros((N, N))
    runs = 0
    for k in range(2, k_max + 1):
        for s in range(n_seeds):
            rs = seed * 1000 + k * 37 + s
            if kind == "gmm":
                lab = GaussianMixture(k, covariance_type="full", reg_covar=1e-4,
                                      random_state=rs).fit_predict(X)
            else:
                lab = KMeans(k, n_init=1, random_state=rs).fit_predict(X)
            same += lab[:, None] == lab[None, :]
            runs += 1
    return 1.0 - same / runs


def build_dissimilarity(X, metric, eps=1e-3, **kw):
    D = _squared_distance_matrix(np.ascontiguousarray(X, dtype=np.float64))
    if metric == "euclid":
        return D
    if metric == "iso":
        Dp = isolation_dissimilarity(X, **kw)
    elif metric == "ca_gmm":
        Dp = coassociation_dissimilarity(X, kind="gmm", **kw)
    elif metric == "ca_km":
        Dp = coassociation_dissimilarity(X, kind="km", **kw)
    else:
        raise ValueError(metric)
    Dp = Dp + eps * D / D.max()
    np.fill_diagonal(Dp, 0.0)
    return Dp


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MetricNKCV2Model(NKCV2Model):
    """NKCV2Model dung D' thay cho D; moi buoc khac giong het src/nkcv2.py."""

    def __init__(self, X, K=3, metric="euclid", D=None, **kw):
        X = np.ascontiguousarray(X, dtype=np.float64)
        self.X = X
        self.N = X.shape[0]
        self.K = int(K)
        N, K = self.N, self.K
        if K >= N:
            raise ValueError("K must be smaller than the number of objects N")

        self.metric = metric
        self.D = np.ascontiguousarray(
            D if D is not None else build_dissimilarity(X, metric, **kw))
        self.dc = _cutoff_distance(self.D, N)
        self.rho = _density(self.D, self.dc, N)

        m_rho, s_rho = self.rho.mean(), self.rho.std()
        d_rho = m_rho - s_rho
        if d_rho <= 0.0:
            d_rho = m_rho / 2.0
        self.d_rho = float(d_rho)

        self.M = np.ascontiguousarray(_build_graph(self.D, self.rho, K, N))
        self.Mdist = np.ascontiguousarray(
            np.take_along_axis(self.D, self.M, axis=1))
        self.Mrho = np.ascontiguousarray(self.rho[self.M])
        self.dt0, self.dt1, self.dt2 = _three_sigma(self.Mdist)

        self.rev_flat, self.rev_off, self.rev_cnt = _build_reverse(self.M, N, K)
        self.Kout = int(self.rev_cnt.max())


def self_test(X, metric="iso", trials=300, seed=0):
    """euclid tai tao goc; Delta Evaluation khop danh gia lai tu dau."""
    base = NKCV2Model(X, K=3)
    same = MetricNKCV2Model(X, K=3, metric="euclid")
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 6, size=X.shape[0])
    assert abs(base.comp_fitness(x) - same.comp_fitness(x)) < 1e-12
    m = MetricNKCV2Model(X, K=3, metric=metric)
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
    for met in ("iso", "ca_gmm", "ca_km"):
        print(met, "self-test:", self_test(X, met))
