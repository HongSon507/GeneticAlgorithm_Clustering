"""
NK-HGA co gieo quan the -- buoc 2 cua ke hoach trong DE_XUAT_FITNESS_CHONG_LAN.md.

Fitness KHONG doi. Chi doi nguon cua mot phan ca the:
  * quan the ban dau: `seed_frac * pop_size` ca the dau lay lan luot tu kho
    loi giai gieo (moi ca the van qua `ls_fi` nhu immigrant thuong);
  * immigrant o moi lan lam moi dinh ky: voi xac suat `seed_frac` lay ngau
    nhien mot loi giai trong kho thay vi phan hoach ngau nhien.

Kho loi giai gieo duoc dung KHONG dung nhan that: GMM full covariance,
k-means va Density Peaks (DPC, Rodriguez & Laio) voi k = 2..k_max, trong do
k_max = min(20, max(6, sqrt(N)/2)) chi phu thuoc N.

Vi gieo chi thay diem xuat phat, PX va Delta Evaluation khong bi anh huong.
"""

from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401  (them src/ vao sys.path)

from ga import NKHGA
from search import ls_fi
from nkcv2 import _squared_distance_matrix, _cutoff_distance, _density


def default_k_max(N):
    return int(min(20, max(6, np.sqrt(N) / 2)))


def dpc_labels(X, k, D=None, rho=None):
    """Density Peaks: k tam co gamma = rho * delta lon nhat; moi diem nhan nhan
    cua lang gieng gan nhat co mat do cao hon (duyet theo mat do giam dan)."""
    if D is None:
        D = _squared_distance_matrix(np.ascontiguousarray(X, dtype=np.float64))
    N = D.shape[0]
    if rho is None:
        rho = _density(D, _cutoff_distance(D, N), N)
    order = np.argsort(-rho, kind="stable")
    delta = np.empty(N)
    parent = np.full(N, -1)
    delta[order[0]] = D[order[0]].max()
    for r in range(1, N):
        i = order[r]
        higher = order[:r]
        j = higher[np.argmin(D[i, higher])]
        parent[i], delta[i] = j, D[i, j]
    centers = np.argsort(-(rho * delta))[:k]
    labels = np.zeros(N, dtype=np.int64)
    labels[centers] = np.arange(1, k + 1)
    if labels[order[0]] == 0:  # diem dac nhat luon la tam
        labels[centers[-1]] = 0
        labels[order[0]] = k
    for i in order:
        if labels[i] == 0:
            labels[i] = labels[parent[i]]
    return labels


def build_seed_pool(X, k_max=None, sources=("gmm", "km", "dpc"), seed=0):
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    N = X.shape[0]
    k_max = k_max or default_k_max(N)
    pool = []
    D = rho = None
    if "dpc" in sources:
        D = _squared_distance_matrix(np.ascontiguousarray(X, dtype=np.float64))
        rho = _density(D, _cutoff_distance(D, N), N)
    for k in range(2, k_max + 1):
        if "gmm" in sources:
            pool.append(GaussianMixture(k, covariance_type="full", reg_covar=1e-4,
                                        random_state=seed + k).fit_predict(X) + 1)
        if "km" in sources:
            pool.append(KMeans(k, n_init=3, random_state=seed + k).fit_predict(X) + 1)
        if "dpc" in sources:
            pool.append(dpc_labels(X, k, D=D, rho=rho))
    return [np.ascontiguousarray(p, dtype=np.int64) for p in pool]


class SeededNKHGA(NKHGA):
    """NKHGA voi kho loi giai gieo `pool` (list nhan 1-based)."""

    def __init__(self, pool=None, seed_frac=0.3, **kw):
        super().__init__(**kw)
        self.pool = pool or []
        self.seed_frac = float(seed_frac)

    def fit(self, X):
        n_init = min(len(self.pool), int(round(self.seed_frac * self.pop_size)))
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(len(self.pool))[:n_init] if n_init else []
        self._init_queue = [self.pool[i] for i in idx]
        return super().fit(X)

    def _seeded(self, model, x, rng):
        x = x.copy()
        f = model.comp_fitness(x)
        f = ls_fi(model, x, f, rng)
        return x, f

    def _immigrant(self, model, rng):
        if self._init_queue:
            return self._seeded(model, self._init_queue.pop(0), rng)
        if self.pool and rng.random() < self.seed_frac:
            return self._seeded(model, self.pool[int(rng.integers(len(self.pool)))], rng)
        return super()._immigrant(model, rng)
