"""
Gop manh vun sau GA -- buoc hau xu ly cua phien ban hien hanh.

CDW ha phat nhanh khac nhan nen cho phep dat bien trong vung dac (giup du lieu
chong lan), nhung de lai nhieu manh vun: tren S2 (mau 1000 diem) CDW + gieo cho
34 cum, trong do 13 cum that co >= 47 diem va 21 manh co <= 14 diem (da so 1-3).

Quy tac (khong dung nhan that):
  * manh vun = cum co it hon tau = max(K+1, ceil(2% N)) diem -- cung tinh than
    "quy tac 2%" bai bao dung de chon dc;
  * xu ly lan luot tu manh nho nhat: gan ca manh cho nhan chiem da so trong cac
    lang gieng Gep (ca hai chieu: M_i va R_i) cua no, uu tien cum KHONG phai
    manh vun; neu khong co lang gieng nao thuoc cum (moi lang gieng la nhieu),
    manh bi gan thanh nhieu (nhan 0).

Chay sau GA nen khong anh huong PX hay Delta Evaluation. Nhan tra ve chua
renumber.
"""

from __future__ import annotations

import math

import numpy as np


def fragment_threshold(N, K=3, frac=0.02):
    return max(K + 1, math.ceil(frac * N))


def merge_fragments(labels, model, frac=0.02):
    """labels: nhan 1-based (0 = nhieu). model: NKCV2Model (dung M, rev_*)."""
    x = np.asarray(labels, dtype=np.int64).copy()
    N, K = model.N, model.K
    tau = fragment_threshold(N, K, frac)
    M, rev_flat, rev_off, rev_cnt = model.M, model.rev_flat, model.rev_off, model.rev_cnt

    def neighbours(i):
        out = list(M[i])
        s = rev_off[i]
        out.extend(rev_flat[s:s + rev_cnt[i]])
        return out

    while True:
        labs, sizes = np.unique(x[x > 0], return_counts=True)
        size_of = dict(zip(labs.tolist(), sizes.tolist()))
        frags = [l for l in labs[np.argsort(sizes, kind="stable")] if size_of[l] < tau]
        if not frags or len(labs) == 1:
            return x
        c = int(frags[0])
        members = np.flatnonzero(x == c)
        votes_big, votes_any = {}, {}
        for i in members:
            for j in neighbours(i):
                lj = int(x[j])
                if lj == 0 or lj == c:
                    continue
                votes_any[lj] = votes_any.get(lj, 0) + 1
                if size_of[lj] >= tau:
                    votes_big[lj] = votes_big.get(lj, 0) + 1
        votes = votes_big or votes_any
        if votes:
            # da so; hoa thi lay nhan nho nhat de tat dinh
            target = max(sorted(votes), key=lambda l: votes[l])
            x[members] = target
        else:
            x[members] = 0
