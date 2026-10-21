"""
Bo sinh du lieu "mo hinh Gauss co nhieu" theo bai bao NK-HGA (muc V-A1) va
Milligan (1981), "A Monte Carlo study of thirty internal criterion measures for
cluster analysis", Psychometrika 46(2).

Kho ma cua tac gia (rtinos/NKGAclust) KHONG kem bo sinh, nen day la BAN TAI TAO
tu mo ta; cac chi tiet bai bao khong neu duoc ghi ro ben duoi.

Theo bai bao:
  * N = 800 doi tuong; Nc cum tu Nc phan phoi chuan bi cat cut, moi cum co tam
    va do dai bien rieng (ngau nhien); do lech chuan moi chieu = 1/3 do dai bien;
  * cac cum khong chong lan o chieu thu nhat (Milligan); cac chieu khac duoc
    phep chong lan;
  * 1% doi tuong la nhieu, phan phoi deu (nhan 0);
  * 3 muc can bang: (1) cac cum xap xi bang nhau; (2) cum dau co 0.1N doi
    tuong; (3) cum dau co 0.6N; phan con lai chia deu.
Theo Milligan (bai bao khong neu):
  * do dai bien moi chieu ~ U(10, 40); phan phoi bi cat tai bien, tuc +-1.5 SD;
  * khoang cach giua bien hai cum lien tiep o chieu thu nhat =
    U(0.25, 0.75) * (SD_cum_truoc + SD_cum_sau).
Lua chon rieng cua ban tai tao:
  * tam o cac chieu con lai ~ U(0, 40), cung thang voi do dai bien de cac cum
    thuong chong lan o cac chieu do;
  * nhieu phan phoi deu trong hop bao cua cac diem thuoc cum.
"""

from __future__ import annotations

import numpy as np


def _cluster_sizes(n_obj, Nc, balance, N):
    if balance == 1:
        first = n_obj // Nc
    elif balance == 2:
        first = int(round(0.1 * N))
    elif balance == 3:
        first = int(round(0.6 * N))
    else:
        raise ValueError("balance phai la 1, 2 hoac 3")
    rest = n_obj - first
    sizes = [first] + [rest // (Nc - 1)] * (Nc - 1) if Nc > 1 else [n_obj]
    sizes[-1] += n_obj - sum(sizes)
    return sizes


def _truncated_normal(rng, mean, sd, lo, hi, n):
    out = np.empty(0)
    while out.size < n:
        z = rng.normal(mean, sd, size=2 * n)
        out = np.concatenate([out, z[(z >= lo) & (z <= hi)]])
    return out[:n]


def generate(Nc, l, balance=1, seed=0, N=800, noise_frac=0.01):
    """Tra ve (X, y): X (N, l); y 1-based, 0 = nhieu. Thu tu diem duoc xao tron."""
    rng = np.random.default_rng(seed)
    n_noise = int(round(noise_frac * N))
    sizes = _cluster_sizes(N - n_noise, Nc, balance, N)
    length = rng.uniform(10, 40, size=(Nc, l))
    sd = length / 3.0

    centers = np.empty((Nc, l))
    lower = 0.0
    for c in range(Nc):  # chieu 0: xep lien tiep, khong chong lan
        centers[c, 0] = lower + length[c, 0] / 2
        if c + 1 < Nc:
            gap = rng.uniform(0.25, 0.75) * (sd[c, 0] + sd[c + 1, 0])
            lower += length[c, 0] + gap
    centers[:, 1:] = rng.uniform(0, 40, size=(Nc, l - 1))

    X, y = [], []
    for c in range(Nc):
        pts = np.column_stack([
            _truncated_normal(rng, centers[c, d], sd[c, d],
                              centers[c, d] - length[c, d] / 2,
                              centers[c, d] + length[c, d] / 2, sizes[c])
            for d in range(l)])
        X.append(pts)
        y.append(np.full(sizes[c], c + 1))
    X = np.vstack(X)
    lo, hi = X.min(axis=0), X.max(axis=0)
    X = np.vstack([X, rng.uniform(lo, hi, size=(n_noise, l))])
    y = np.concatenate(y + [np.zeros(n_noise, dtype=int)])
    perm = rng.permutation(len(y))
    return X[perm], y[perm].astype(int)
