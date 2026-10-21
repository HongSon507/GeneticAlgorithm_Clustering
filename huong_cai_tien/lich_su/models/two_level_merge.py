"""
Fitness hai tang -- de xuat Sec4 trong DE_XUAT_FITNESS_CHONG_LAN.md.

Tang trong: NK-HGA (NKCV2 hoac NKCV-A) cho mot phan hoach cung, thuong phan
manh co kiem soat. Tang ngoai (ham nay): gop tham lam cac cum con theo mot tieu
chi PHI PHAN RA ma NK khong chua duoc -- o day la BIC cua mo hinh Gauss full
covariance gan cung (classification likelihood), moi cum con la mot thanh phan.

Vi tang ngoai chay SAU GA, PX va Delta Evaluation khong bi dong toi.

Rang buoc lien thong: chi xet gop hai cum co it nhat mot canh Gep noi nhau
(cung de khong gop hai cum o hai dau du lieu chi vi BIC thich it thanh phan).
Nhan 0 (nhieu) giu nguyen.
"""

from __future__ import annotations

import numpy as np


def _cluster_loglik(Xc, n_total, reg=1e-4):
    """Log-likelihood gan cung cua mot cum: sum log N(x|mu,Sigma) + n log(n/N)."""
    n, d = Xc.shape
    mu = Xc.mean(axis=0)
    S = np.cov(Xc, rowvar=False, bias=True) if n > 1 else np.zeros((d, d))
    S = np.atleast_2d(S) + reg * np.eye(d)
    sign, logdet = np.linalg.slogdet(S)
    diff = Xc - mu
    maha = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(S), diff)
    ll = -0.5 * (maha.sum() + n * (d * np.log(2 * np.pi) + logdet))
    return ll + n * np.log(n / n_total)


def _bic_terms(X, labels, n_total):
    d = X.shape[1]
    p = d + d * (d + 1) / 2 + 1  # mu, Sigma, ty le tron
    return {c: (_cluster_loglik(X[labels == c], n_total), p)
            for c in np.unique(labels[labels > 0])}


def bic_merge(X, labels, M=None, connected_only=True, max_steps=None):
    """Gop tham lam cac cap cum lam BIC giam nhieu nhat, den khi khong giam.

    labels: nhan 1-based, 0 = nhieu. M: nhom Gep (N, K) cua NKCV2Model.
    Tra ve nhan moi (chua renumber).
    """
    labels = np.asarray(labels).copy()
    n_total = int(np.sum(labels > 0))
    if n_total == 0:
        return labels
    logn = np.log(n_total)
    terms = _bic_terms(X, labels, n_total)
    steps = 0
    while len(terms) > 1 and (max_steps is None or steps < max_steps):
        clusters = sorted(terms)
        if connected_only and M is not None:
            a = labels[:, None]
            b = labels[M]
            mask = (a > 0) & (b > 0) & (a != b)
            pairs = {(min(u, v), max(u, v))
                     for u, v in zip(np.broadcast_to(a, b.shape)[mask], b[mask])}
        else:
            pairs = {(u, v) for i, u in enumerate(clusters) for v in clusters[i + 1:]}
        best, best_gain = None, 0.0
        for u, v in pairs:
            ll_u, p = terms[u]
            ll_v, _ = terms[v]
            ll_uv = _cluster_loglik(X[(labels == u) | (labels == v)], n_total)
            # BIC = -2 LL + p log n ; gain > 0 nghia la BIC giam khi gop
            gain = (-2 * (ll_u + ll_v) + 2 * p * logn) - (-2 * ll_uv + p * logn)
            if gain > best_gain:
                best, best_gain = (u, v, ll_uv), gain
        if best is None:
            break
        u, v, ll_uv = best
        labels[labels == v] = u
        terms[u] = (ll_uv, terms[u][1])
        del terms[v]
        steps += 1
    return labels
