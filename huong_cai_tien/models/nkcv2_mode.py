"""
NKCV-M -- NKCV2 voi dinh nghia cum theo DINH MAT DO (thay vi "thanh phan lien
thong mat do").

NKCV2 chi cho dat bien o noi cac lang gieng cach xa nhau (khe mat do), nen hai
cum chong lan khong co khe luon bi gop. NKCV-M doi dinh nghia: moi cum la luu
vuc cua mot dinh mat do CO Y NGHIA; bien duoc dat o yen ngua giua hai dinh, ke
ca khi yen ngua van dac.

1. Dinh co y nghia (tinh MOT lan, khong dung nhan that) -- ToMATo (Chazal et
   al.) tren do thi kNN doi xung, nhung do sau yen ngua do theo DON VI NHIEU:

       z = (rho_tho(dinh) - rho_tho(yen ngua)) / sqrt(rho_tho(dinh))

   rho_tho = sum_j exp(-(D_ij/dc)^2) xap xi so lang gieng dem duoc nen co nhieu
   kieu Poisson voi do lech chuan ~ sqrt(rho_tho). Dinh yeu hon chi duoc giu khi
   z > z0 (mac dinh 2: "hai sigma"). Persistence tuyet doi danh gia thap cum thua
   that; persistence tuong doi tao dinh gia o vung thua -- z sua ca hai.

2. Ham muc tieu -- canh Gep (j -> f_i) co cross_ij = [luu vuc(i) != luu vuc(j)]:
     * cross_ij = 0: dong gop NKCV2 nguyen ban;
     * cross_ij = 1 (canh vuot yen ngua giua hai dinh co y nghia):
           cung nhan  -> rho_j   (gop hai dinh that bi phat)
           khac nhan  -> 0       (cat tai yen ngua khong bi phat)
     * nhanh nhieu (x_i = 0) giu nguyen.
   Khong them so hang nao: chi THAY nhanh cung/khac nhan tren canh vuot yen ngua.
   Neu chi co mot dinh moi thanh phan lien thong, NKCV-M trung NKCV2.

cross_ij chi phu thuoc du lieu => Delta Evaluation va PX van dung (self_test).
"""

from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401  (them src/ vao sys.path)

from nkcv2 import njit, NKCV2Model

Z0 = 2.0
KNN = 10


# ---------------------------------------------------------------------------
# Dinh mat do co y nghia
# ---------------------------------------------------------------------------

def _knn_adjacency(D, knn):
    N = D.shape[0]
    nbrs = np.argsort(D, axis=1, kind="stable")[:, 1:knn + 1]
    adj = [set() for _ in range(N)]
    for i in range(N):
        for j in nbrs[i]:
            adj[i].add(int(j))
            adj[int(j)].add(i)
    return [sorted(s) for s in adj]


def raw_density(D, dc):
    """rho chua chuan hoa (cung kernel voi src/nkcv2.py, bo chinh no)."""
    return np.exp(-(D / dc) ** 2).sum(axis=1) - 1.0


def significant_modes(D, dc, knn=KNN, z0=Z0):
    """Tra ve (nhan luu vuc 0..m-1, danh sach z cua cac lan hai dinh gap nhau)."""
    rho = raw_density(D, dc)
    adj = _knn_adjacency(D, knn)
    N = len(rho)
    parent = np.arange(N)
    zs = []

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in np.argsort(-rho, kind="stable"):
        i = int(i)
        higher = [j for j in adj[i] if rho[j] > rho[i]]
        if not higher:
            continue  # dinh moi: parent[i] = i
        roots = {find(j) for j in higher}
        peak = max(roots, key=lambda r: rho[r])
        parent[i] = peak
        for r in roots:
            if r == peak:
                continue
            z = (rho[r] - rho[i]) / np.sqrt(max(rho[r], 1e-12))
            zs.append(float(z))
            if z <= z0:
                parent[r] = peak
    roots = np.array([find(i) for i in range(N)])
    _, basin = np.unique(roots, return_inverse=True)
    return basin.astype(np.int64), zs


# ---------------------------------------------------------------------------
# Kernels -- giu cong thuc src/nkcv2.py::_comp_fi, thay nhanh tren canh vuot yen ngua
# ---------------------------------------------------------------------------

@njit(cache=True)
def comp_fi_mode(x, i, M, Mdist, Mrho, Mcross, rho, dt0, dt1, dt2, d_rho, N, K):
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
        elif Mcross[i, k]:
            if xi == x[j]:
                fi += fmax          # gop hai dinh co y nghia
            # khac nhan: cat tai yen ngua, khong phat
        elif xi == x[j]:
            if d > dt0 and d <= dt1:
                fi += fmax * (d - dt0) / (dt1 - dt0)
            elif d > dt1:
                fi += fmax
        else:
            if d <= dt0:
                fi += fmax
            elif d <= dt1:
                fi += fmax - fmax * (d - dt0) / (dt1 - dt0)
    return fi / (N * K)


@njit(cache=True)
def comp_fitness_mode(x, M, Mdist, Mrho, Mcross, rho, dt0, dt1, dt2, d_rho, N, K):
    f = 0.0
    for i in range(N):
        f += comp_fi_mode(x, i, M, Mdist, Mrho, Mcross, rho, dt0, dt1, dt2,
                          d_rho, N, K)
    return f


@njit(cache=True)
def df_element_mode(x, i, xi_old, M, Mdist, Mrho, Mcross, rho, rev_flat,
                    rev_off, rev_cnt, dt0, dt1, dt2, d_rho, N, K):
    xi_new = x[i]
    df = comp_fi_mode(x, i, M, Mdist, Mrho, Mcross, rho, dt0, dt1, dt2, d_rho, N, K)
    x[i] = xi_old
    df -= comp_fi_mode(x, i, M, Mdist, Mrho, Mcross, rho, dt0, dt1, dt2, d_rho, N, K)
    x[i] = xi_new
    start = rev_off[i]
    for r in range(rev_cnt[i]):
        p = rev_flat[start + r]
        df += comp_fi_mode(x, p, M, Mdist, Mrho, Mcross, rho, dt0, dt1, dt2,
                           d_rho, N, K)
        x[i] = xi_old
        df -= comp_fi_mode(x, p, M, Mdist, Mrho, Mcross, rho, dt0, dt1, dt2,
                           d_rho, N, K)
        x[i] = xi_new
    return df


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ModeNKCV2Model(NKCV2Model):
    """NKCV-M. `z0=None` tat dinh nghia moi (moi canh cross=0) => trung NKCV2."""

    def __init__(self, X, K=3, z0=Z0, knn=KNN):
        super().__init__(X, K=K)
        self.z0 = z0
        if z0 is None:
            self.basin_ = np.zeros(self.N, dtype=np.int64)
            self.saddle_z_ = []
        else:
            self.basin_, self.saddle_z_ = significant_modes(self.D, self.dc, knn, z0)
        self.n_modes_ = int(self.basin_.max()) + 1
        self.Mcross = np.ascontiguousarray(
            (self.basin_[self.M] != self.basin_[:, None]).astype(np.bool_))

    def _args(self):
        return (self.M, self.Mdist, self.Mrho, self.Mcross, self.rho)

    def comp_fi(self, x, i):
        return comp_fi_mode(x, i, *self._args(), self.dt0, self.dt1, self.dt2,
                            self.d_rho, self.N, self.K)

    def comp_fitness(self, x):
        return comp_fitness_mode(x, *self._args(), self.dt0, self.dt1, self.dt2,
                                 self.d_rho, self.N, self.K)

    def df_element(self, x, i, xi_old):
        return df_element_mode(x, i, xi_old, *self._args(), self.rev_flat,
                               self.rev_off, self.rev_cnt, self.dt0, self.dt1,
                               self.dt2, self.d_rho, self.N, self.K)


def self_test(X, trials=300, seed=0):
    """(1) z0=None trung NKCV2; (2) Delta Evaluation khop danh gia lai;
    (3) PX tra fitness dung va con khong te hon cha me; fix_labels giu fitness."""
    from px import map_solutions, px, fix_labels
    from search import ls_fi
    rng = np.random.default_rng(seed)
    base = NKCV2Model(X, K=3)
    off = ModeNKCV2Model(X, K=3, z0=None)
    for _ in range(20):
        x = rng.integers(0, 6, size=X.shape[0])
        assert abs(base.comp_fitness(x) - off.comp_fitness(x)) < 1e-12
    m = ModeNKCV2Model(X, K=3)
    x = rng.integers(0, 6, size=X.shape[0])
    for _ in range(trials):
        i = int(rng.integers(X.shape[0]))
        old = int(x[i])
        f0 = m.comp_fitness(x)
        x[i] = int(rng.integers(0, 6))
        assert abs((m.comp_fitness(x) - f0) - m.df_element(x, i, old)) < 1e-12
    for _ in range(10):
        a = rng.integers(1, 6, size=X.shape[0]).astype(np.int64)
        b = rng.integers(1, 6, size=X.shape[0]).astype(np.int64)
        fa = ls_fi(m, a, m.comp_fitness(a), rng)
        fb = ls_fi(m, b, m.comp_fitness(b), rng)
        child, f = px(m, a, map_solutions(a, b))
        assert abs(f - m.comp_fitness(child)) < 1e-12 and f <= min(fa, fb) + 1e-12
        fc = m.comp_fitness(child)
        fix_labels(m, child)
        assert abs(fc - m.comp_fitness(child)) < 1e-12
    return m.n_modes_


if __name__ == "__main__":
    from datasets import DATASETS, subsample
    for name, n in [("flame", None), ("aggregation", None), ("s2", 1000)]:
        X, _ = subsample(*DATASETS[name](), n)
        print(f"{name}: self-test PASS, so dinh = {self_test(X)}")
