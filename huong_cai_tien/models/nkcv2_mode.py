"""
NKCV-M -- NKCV2 voi dinh nghia cum theo DINH MAT DO (ToMATo) duoi dang TIN HIEU PHU (soft auxiliary signal).

1. Dinh mat do va do sau yen ngua (ToMATo tren do thi kNN doi xung):
       z = (rho_tho(dinh) - rho_tho(yen ngua)) / sqrt(rho_tho(dinh))
   Dinh duoc giu khi z > z0 (mac dinh z0 = 1.5; hieu chuan tren du lieu rong:
   experiments/run_z0_calibration.py, DE_XUAT_FITNESS_CHONG_LAN.md muc 7i.5).

2. Tín hiệu phụ mềm (Soft Mode Penalty, w_max <= 0.65):
   Thay vi phat cung (cung nhan -> fmax, khac nhan -> 0), tin hieu yen ngua
   chi dong vai tro DIEU CHINH:
       dong_gop = (1 - w_ij) * NKCV2_ij + w_ij * Mode_ij
   trong do:
       w_ij = w_max * clip(0.5 + 0.5 * (z_AB - z0) / (z_full - z0), 0.5, 1.0)
   (mac dinh w_max = 0.6, z_full = 4.0: canh vuot yen ngua nhan w tu 0.3 tai
   z = z0 len 0.6 tu z = z_full).
   - Cung nhan: phat (1 - w_ij) * ramp_same(d) + w_ij * fmax
   - Khac nhan: phat (1 - w_ij) * ramp_diff(d)
   Khi w_max ~ 0.6, chi phi tach va gop o yen ngua canh tranh nhau; hai dinh van
   co the chung mot cum neu GA thay gop tot hon cho fitness tong the.
   z0 = None hoac w_max = 0 -> trung NKCV2 goc (sai so < 1e-12).

3. Chuan hoa thang khoang cach cuc bo (Tuy chon local_scale):
   Giam thiet hai do nguong dt0, dt1 toan cuc gay ra cho cac cum thua (nhu Jain).
   Khoang cach hieu dung d_eff giup cac canh lien tiep trong cum thua khong bi
   vuot nguong dt1 mot cach gia tao.

Moi he so la tinh theo du lieu -> Delta Evaluation va PX chinh xac 100%.
"""

from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401  (them src/ vao sys.path)

from nkcv2 import njit, NKCV2Model

Z0 = 1.5
Z_FULL = 4.0
W_MAX = 0.6
KNN = 10


# ---------------------------------------------------------------------------
# Dinh mat do va yen ngua ToMATo
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


def significant_modes(D, dc, knn=KNN, z0=Z0, return_pairs=False):
    """Tra ve (nhan luu vuc 0..m-1, danh sach z cua cac lan hai dinh gap nhau).
    return_pairs=True: tra ve them dict {(a, b): z_min} giua cac luu vuc a < b."""
    rho = raw_density(D, dc)
    adj = _knn_adjacency(D, knn)
    N = len(rho)
    parent = np.arange(N)
    zs = []
    met = []

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in np.argsort(-rho, kind="stable"):
        i = int(i)
        higher = [j for j in adj[i] if rho[j] > rho[i]]
        if not higher:
            continue
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
            else:
                met.append((r, peak, float(z)))

    roots = np.array([find(i) for i in range(N)])
    uniq, basin = np.unique(roots, return_inverse=True)
    basin = basin.astype(np.int64)

    if not return_pairs:
        return basin, zs

    pos = {int(r): b for b, r in enumerate(uniq)}
    pairs = {}
    for r, p, z in met:
        a, b = pos[int(find(r))], pos[int(find(p))]
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        pairs[key] = min(z, pairs.get(key, np.inf))
    return basin, zs, pairs


def saddle_strength(z, z0=Z0, z_full=Z_FULL):
    if z <= z0:
        return 0.0
    return float(np.clip(0.5 + 0.5 * (z - z0) / max(z_full - z0, 1e-12), 0.5, 1.0))


# ---------------------------------------------------------------------------
# Kernels -- Delta Evaluation va Fitness numba
# ---------------------------------------------------------------------------

@njit(cache=True)
def comp_fi_mode(x, i, M, Mdist, Mrho, Mcross, rho, dt0, dt1, dt2, d_rho, N, K):
    """Mcross: float [N, K] = w_ij trong [0, 1] (0 = canh thuong)."""
    fi = 0.0
    xi = x[i]
    for k in range(K):
        j = M[i, k]
        d = Mdist[i, k]
        fmax = Mrho[i, k]
        w = Mcross[i, k]

        if xi == 0:
            if rho[i] <= d_rho and d > dt2:
                pass
            else:
                fi += fmax
        elif w > 0.0:
            # Tron NKCV2 voi tin hieu yen ngua mem
            c = 0.0
            if xi == x[j]:
                if d > dt0 and d <= dt1:
                    c = fmax * (d - dt0) / (dt1 - dt0)
                elif d > dt1:
                    c = fmax
                fi += (1.0 - w) * c + w * fmax
            else:
                if d <= dt0:
                    c = fmax
                elif d <= dt1:
                    c = fmax - fmax * (d - dt0) / (dt1 - dt0)
                fi += (1.0 - w) * c
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
    """NKCV-M ban mem voi tin hieu yen ngua ToMATo lam phu tro.

    w_max = 0 hoac z0 = None -> trung khit NKCV2 goc (sai so < 1e-12).
    soft = True: w_ij = w_max * s(z yen ngua).
    soft = False: w_ij = w_max tren moi canh vuot yen ngua (ban cu).
    """

    def __init__(self, X, K=3, z0=Z0, knn=KNN, soft=True, w_max=W_MAX,
                 z_full=Z_FULL, local_scale=False):
        super().__init__(X, K=K)
        self.z0, self.soft, self.w_max, self.z_full = z0, soft, w_max, z_full
        self.local_scale = local_scale
        self.pair_z_ = {}

        if z0 is None or w_max <= 0:
            self.basin_ = np.zeros(self.N, dtype=np.int64)
            self.saddle_z_ = []
        else:
            self.basin_, self.saddle_z_, self.pair_z_ = significant_modes(
                self.D, self.dc, knn, z0, return_pairs=True)

        self.n_modes_ = int(self.basin_.max()) + 1
        cross = self.basin_[self.M] != self.basin_[:, None]
        W = np.zeros((self.N, self.K), dtype=np.float64)

        if z0 is not None and w_max > 0:
            if soft:
                raw = raw_density(self.D, self.dc)
                peak = np.zeros(self.n_modes_)
                np.maximum.at(peak, self.basin_, raw)
                for i, k in zip(*np.nonzero(cross)):
                    j = self.M[i, k]
                    a, b = int(self.basin_[i]), int(self.basin_[j])
                    z = self.pair_z_.get((min(a, b), max(a, b)))
                    if z is None:
                        p = min(peak[a], peak[b])
                        z = (p - min(raw[i], raw[j])) / np.sqrt(max(p, 1e-12))
                    W[i, k] = w_max * saddle_strength(z, z0, z_full)
            else:
                W[cross] = w_max

        self.Mcross = np.ascontiguousarray(W)

        # Chuan hoa thang khoang cach cuc bo (tuy chon)
        if local_scale:
            # Scale cuc bo dua tren trung vi khoang cach lang gieng
            scale_i = np.median(self.Mdist, axis=1)
            med = np.median(scale_i)
            if med > 0:
                scale_norm = np.clip(scale_i / med, 0.5, 2.0)
                # d_eff = d / sqrt(scale_i * scale_j)
                denom = np.sqrt(scale_norm[:, None] * scale_norm[self.M])
                self.Mdist = np.ascontiguousarray(self.Mdist / denom)

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
    """(1) z0=None hoac w_max=0 trung NKCV2; (2) Delta Evaluation khop danh gia lai;
    (3) PX tra fitness dung va con khong te hon cha me; fix_labels giu fitness."""
    from px import map_solutions, px, fix_labels
    from search import ls_fi
    rng = np.random.default_rng(seed)

    base = NKCV2Model(X, K=3)
    off = ModeNKCV2Model(X, K=3, z0=None)
    zero = ModeNKCV2Model(X, K=3, w_max=0.0)

    for _ in range(20):
        x = rng.integers(0, 6, size=X.shape[0])
        assert abs(base.comp_fitness(x) - off.comp_fitness(x)) < 1e-12
        assert abs(base.comp_fitness(x) - zero.comp_fitness(x)) < 1e-12

    m = ModeNKCV2Model(X, K=3, w_max=0.6, soft=True)
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
