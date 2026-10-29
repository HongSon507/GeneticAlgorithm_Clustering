"""
Xu ly manh vun dua tren bang chung lien ket bien va mat do:
  merge_fragments_evidence:
    - Chi xem xet cac manh sieu nho (kich thuoc <= K hoac <= 3 diem) hoac cac manh
      co lien ket bien ap dao (>= 75% canh noi vao duy nhat mot cum lang gieng L).
    - Neu manh roi rac va mat do thap (median(rho) <= d_rho) -> gan nhan nhieu 0.
    - Neu manh co lien ket ap dao sang L va khong lam pha vo canh quan fitness
      (Delta f <= 1e-4) -> gop vao L.
    - Cac cum nho that (nhu tren Ecoli, Glass co tu 5 diem tro len, hoac co lien ket
      noi bo rieng) DUOC BAO VE TUYET DOI, khong bi xoa so nhu quy tac cu 2% N.
"""

from __future__ import annotations

import math
import numpy as np


def fragment_threshold(N, K=3, frac=0.02):
    return max(K + 1, math.ceil(frac * N))


def merge_fragments_evidence(labels, model, frac=0.02, max_delta_f=1e-4):
    """Gop manh vun co dieu kien dua tren bang chung lien ket Gep va bao ve cum nho that."""
    x = np.asarray(labels, dtype=np.int64).copy()
    N, K = model.N, model.K
    M, rev_flat, rev_off, rev_cnt = model.M, model.rev_flat, model.rev_off, model.rev_cnt
    rho = model.rho
    d_rho = model.d_rho
    dt1 = model.dt1

    def neighbours(i):
        out = list(M[i])
        s = rev_off[i]
        out.extend(rev_flat[s:s + rev_cnt[i]])
        return out

    f_current = model.comp_fitness(x)

    while True:
        pos_mask = x > 0
        if not np.any(pos_mask):
            return x

        labs, sizes = np.unique(x[pos_mask], return_counts=True)
        size_of = dict(zip(labs.tolist(), sizes.tolist()))

        # Chi coi la manh ung vien neu kich thuoc rat nho (<= max(K, 3)) hoac <= 2% N
        tau = fragment_threshold(N, K, frac)
        frags = [l for l in labs[np.argsort(sizes, kind="stable")] if size_of[l] < tau]
        if not frags or len(labs) <= 1:
            return x

        merged_any = False
        for c in frags:
            c = int(c)
            members = np.flatnonzero(x == c)
            sz = len(members)
            if sz == 0:
                continue

            med_rho = float(np.median(rho[members]))

            # Dem phieu lang gieng
            votes_all = {}
            votes_short = {}
            for i in members:
                for j in neighbours(i):
                    lj = int(x[j])
                    if lj == 0 or lj == c:
                        continue
                    votes_all[lj] = votes_all.get(lj, 0) + 1
                    # Canh ngan Gep
                    if model.D[i, j] <= dt1:
                        votes_short[lj] = votes_short.get(lj, 0) + 1

            total_votes = sum(votes_all.values())

            # 1. Neu hoan toan khong co lang gieng hoac mat do thap va tach biet:
            if total_votes == 0 or not votes_short:
                if med_rho <= d_rho:
                    x[members] = 0
                    merged_any = True
                continue

            # 2. Chon cum lang gieng chiem da so
            target = max(sorted(votes_all), key=lambda l: votes_all[l])
            target_ratio = votes_all[target] / float(total_votes)
            short_target = votes_short.get(target, 0)

            # DIEU KIEN GOP:
            # - Kich thuoc sieu nho (<= K diem) va lien ket ap dao (>= 70%) sang target
            # - HOAC kich thuoc < tau nhung tat ca canh ngan deu tro sang target
            should_merge = False
            if sz <= K and target_ratio >= 0.7:
                should_merge = True
            elif sz < tau and short_target >= 0.8 * sum(votes_short.values()) and target_ratio >= 0.8:
                should_merge = True

            if should_merge:
                x_trial = x.copy()
                x_trial[members] = target
                f_trial = model.comp_fitness(x_trial)
                # Cho phep dung sai nho do tinh roi rac cua GA
                if f_trial <= f_current + max_delta_f:
                    x = x_trial
                    f_current = f_trial
                    merged_any = True

        if not merged_any:
            break

    return x


def merge_fragments(labels, model, frac=0.02):
    """Quy tac gop manh cu (de so sanh doi chieu ablation)."""
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
            target = max(sorted(votes), key=lambda l: votes[l])
            x[members] = target
        else:
            x[members] = 0
