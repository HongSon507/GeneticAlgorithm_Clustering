"""
NKCV2 model — faithful port of the authors' reference (rtinos/NKGAclust, nk.h).

Provides preprocessing (squared-Euclidean distances, cutoff dc via the 2% rule,
Gaussian density rho), the directed interaction graph Gep (density-link +
proximity-fill, indegree exactly K, NO self-loops), the NKCV2 fitness (MINIMIZED,
scaled by 1/(N*K)) and its delta-evaluation primitive.

All thresholds and rho are computed on SQUARED Euclidean distances, exactly as in
the reference (`sqrEucDist`). Labels are 1-based; label 0 means noise.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    # Fallback: run the kernels as plain Python. Correct but slower; keeps the
    # code runnable on interpreters without a numba wheel (e.g. Python 3.14).
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        # Support both @njit and @njit(cache=True) call styles.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator


# ---------------------------------------------------------------------------
# Core NKCV2 kernels (operate on flat arrays so numba can accelerate them)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _comp_fi(x, i, M, Mdist, Mrho, rho, dt0, dt1, dt2, d_rho, N, K):
    """Subfunction f_i(x), scaled by 1/(N*K). fmin=0, fmax=rho[j]."""
    fi = 0.0
    xi = x[i]
    for k in range(K):
        j = M[i, k]
        d = Mdist[i, k]
        fmax = Mrho[i, k]  # = rho[j]
        if xi == 0:
            # noise branch (Eq. 9): cheap only if i is truly noise
            if rho[i] <= d_rho and d > dt2:
                pass  # + fmin (= 0)
            else:
                fi += fmax
        elif xi == x[j]:
            # same label: penalize being far apart, ramp up dt0 -> dt1
            if d > dt0 and d <= dt1:
                fi += fmax * (d - dt0) / (dt1 - dt0)
            elif d > dt1:
                fi += fmax
            # else (d <= dt0): + 0
        else:
            # different label: penalize being close, ramp down dt0 -> dt1
            if d <= dt0:
                fi += fmax
            elif d <= dt1:
                fi += fmax - fmax * (d - dt0) / (dt1 - dt0)
            # else (d > dt1): + 0
    return fi / (N * K)


@njit(cache=True)
def _comp_fitness(x, M, Mdist, Mrho, rho, dt0, dt1, dt2, d_rho, N, K):
    f = 0.0
    for i in range(N):
        f += _comp_fi(x, i, M, Mdist, Mrho, rho, dt0, dt1, dt2, d_rho, N, K)
    return f


@njit(cache=True)
def _df_element(x, i, xi_old, M, Mdist, Mrho, rho, rev_flat, rev_off, rev_cnt,
                dt0, dt1, dt2, d_rho, N, K):
    """Change in f caused by x[i] going from xi_old to its current value x[i].

    Only f_i and the subfunctions f_p of objects p that i INFLUENCES (the reverse
    neighbours R_i = {p : i in M_p}) can change. x is mutated in place and restored.
    """
    xi_new = x[i]
    # own subfunction
    df = _comp_fi(x, i, M, Mdist, Mrho, rho, dt0, dt1, dt2, d_rho, N, K)
    x[i] = xi_old
    df -= _comp_fi(x, i, M, Mdist, Mrho, rho, dt0, dt1, dt2, d_rho, N, K)
    x[i] = xi_new
    # subfunctions of objects influenced by i
    start = rev_off[i]
    cnt = rev_cnt[i]
    for r in range(cnt):
        p = rev_flat[start + r]
        df += _comp_fi(x, p, M, Mdist, Mrho, rho, dt0, dt1, dt2, d_rho, N, K)
        x[i] = xi_old
        df -= _comp_fi(x, p, M, Mdist, Mrho, rho, dt0, dt1, dt2, d_rho, N, K)
        x[i] = xi_new
    return df


# ---------------------------------------------------------------------------
# Preprocessing helpers (run once per dataset)
# ---------------------------------------------------------------------------

def _squared_distance_matrix(X):
    """Full N x N squared Euclidean distance matrix (reference `sqrEucDist`)."""
    sq = np.sum(X * X, axis=1)
    D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(D, 0.0, out=D)  # clip tiny negatives from round-off
    np.fill_diagonal(D, 0.0)
    return D


def _cutoff_distance(D, N, percent=2):
    """dc via the 2% rule: sort all N(N-1)/2 pairwise distances, take the 2%-th."""
    iu = np.triu_indices(N, k=1)
    vD = np.sort(D[iu])
    position = (len(vD) * percent) // 100
    return vD[position]


def _density(D, dc, N):
    """Gaussian-kernel density over ALL pairs, scaled to [0, 1]."""
    rho = np.zeros(N, dtype=np.float64)
    iu = np.triu_indices(N, k=1)
    contrib = np.exp(-((D[iu] / dc) ** 2))
    np.add.at(rho, iu[0], contrib)
    np.add.at(rho, iu[1], contrib)
    rho /= rho.max()
    return rho


def _build_graph(D, rho, K, N):
    """Directed interaction graph Gep. Returns M[N, K]: the K objects that
    influence f_i (i.e. M_i, the group). Indegree is exactly K; no self-loops.
    """
    M = np.empty((N, K), dtype=np.int64)
    dlink = np.empty(N, dtype=np.int64)

    # Step 1 -- density link: nearest object with strictly higher density.
    for i in range(N):
        higher = rho > rho[i]
        if higher.any():
            di = np.where(higher, D[i], np.inf)
            dlink[i] = int(np.argmin(di))
        else:
            # globally densest object: link to its nearest object (reference choice)
            di = D[i].copy()
            di[i] = np.inf
            dlink[i] = int(np.argmin(di))

    # Step 2 -- proximity fill: add nearest objects until indegree == K.
    for i in range(N):
        order = np.argsort(D[i], kind="stable")  # ascending; self (d=0) first
        chosen = [int(dlink[i])]
        for j in order:
            j = int(j)
            if j == i or j == dlink[i]:
                continue
            chosen.append(j)
            if len(chosen) == K:
                break
        M[i] = np.array(chosen[:K], dtype=np.int64)
    return M


def _three_sigma(Mdist):
    """dt0 = mean, dt1 = mean + 2*std, dt2 = mean + std over the N*K group dists."""
    vals = Mdist.reshape(-1)
    m = vals.mean()
    s = vals.std()  # population std (ddof=0), as in the reference
    dt0 = m
    dt1 = m + 2.0 * s
    dt2 = m + s
    return dt0, dt1, dt2


def _build_reverse(M, N, K):
    """CSR reverse adjacency R_i = {p : i in M_p} (objects influenced by i)."""
    counts = np.zeros(N, dtype=np.int64)
    for i in range(N):
        for k in range(K):
            counts[M[i, k]] += 1
    off = np.zeros(N, dtype=np.int64)
    off[1:] = np.cumsum(counts)[:-1]
    total = int(counts.sum())
    flat = np.empty(total, dtype=np.int64)
    cursor = off.copy()
    for i in range(N):
        for k in range(K):
            p = M[i, k]
            flat[cursor[p]] = i
            cursor[p] += 1
    return flat, off, counts


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NKCV2Model:
    """Holds all fixed-per-dataset structures and exposes NKCV2 evaluation."""

    def __init__(self, X, K=3):
        X = np.ascontiguousarray(X, dtype=np.float64)
        self.X = X
        self.N = X.shape[0]
        self.K = int(K)
        N, K = self.N, self.K
        if K >= N:
            raise ValueError("K must be smaller than the number of objects N")

        self.D = _squared_distance_matrix(X)
        self.dc = _cutoff_distance(self.D, N)
        self.rho = _density(self.D, self.dc, N)

        # noise density cutoff d_rho (crho)
        m_rho, s_rho = self.rho.mean(), self.rho.std()
        d_rho = m_rho - s_rho
        if d_rho <= 0.0:
            d_rho = m_rho / 2.0
        self.d_rho = float(d_rho)

        self.M = np.ascontiguousarray(_build_graph(self.D, self.rho, K, N))
        self.Mdist = np.ascontiguousarray(
            np.take_along_axis(self.D, self.M, axis=1)
        )
        self.Mrho = np.ascontiguousarray(self.rho[self.M])
        self.dt0, self.dt1, self.dt2 = _three_sigma(self.Mdist)

        self.rev_flat, self.rev_off, self.rev_cnt = _build_reverse(self.M, N, K)
        self.Kout = int(self.rev_cnt.max())

    # -- public evaluation API -------------------------------------------------

    def comp_fi(self, x, i):
        return _comp_fi(x, i, self.M, self.Mdist, self.Mrho, self.rho,
                        self.dt0, self.dt1, self.dt2, self.d_rho, self.N, self.K)

    def comp_fitness(self, x):
        return _comp_fitness(x, self.M, self.Mdist, self.Mrho, self.rho,
                             self.dt0, self.dt1, self.dt2, self.d_rho,
                             self.N, self.K)

    def df_element(self, x, i, xi_old):
        return _df_element(x, i, xi_old, self.M, self.Mdist, self.Mrho, self.rho,
                           self.rev_flat, self.rev_off, self.rev_cnt,
                           self.dt0, self.dt1, self.dt2, self.d_rho,
                           self.N, self.K)
