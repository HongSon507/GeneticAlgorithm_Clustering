"""
Toan tu dot bien va tinh chinh cuc bo cho NK-HGA dua tren do thi Gep:
  1. mutation_merge_graph: gop cum dua tren mat do lien ket bien Gep va chenh lech fitness;
  2. mutation_split_graph: tach cum voi hai tam duoc tach xa nhau theo rho * D;
  3. ls_fi_with_noise: tinh chinh cuc bo cho phep gan nhan nhieu 0 cho cac diem thoa rho <= d_rho.

Toan bo toan tu deu doc truc tiep tu NKCV2Model (M, Mdist, D, rho, dt0, dt1),
tuan thu tuyet doi tinh bat bien Delta Evaluation O(Kout*K).
"""

from __future__ import annotations

import numpy as np


def _cluster_prototypes(model, parent, label_max):
    """Chi so diem co mat do cao nhat cua moi cum; -1 neu rong."""
    proto = np.full(label_max + 1, -1, dtype=np.int64)
    rho = model.rho
    for i in range(model.N):
        lab = int(parent[i])
        if lab > 0:
            if proto[lab] == -1 or rho[i] > rho[proto[lab]]:
                proto[lab] = i
    return proto


def mutation_merge_graph(model, parent, rng):
    """Gop hai cum co lien ket bien Gep manh nhat hoac hai tam gan nhau.
    Uu tien xu ly cac manh vien kẹp sat cum lon."""
    offspring = parent.copy()
    label_max = int(parent.max())
    if label_max <= 1:
        return offspring

    labs, sizes = np.unique(offspring[offspring > 0], return_counts=True)
    if len(labs) <= 1:
        return offspring

    size_map = dict(zip(labs.tolist(), sizes.tolist()))
    proto = _cluster_prototypes(model, parent, label_max)

    # 1. Tinh lien ket bien Gep giua cac cum
    # Dem so canh Gep (i -> j hoac j in M_i) co d <= dt1 noi giua hai cum khac nhau
    M = model.M
    Mdist = model.Mdist
    dt1 = model.dt1
    N, K = model.N, model.K

    links = {}  # (min(la, lb), max(la, lb)) -> count
    for i in range(N):
        li = int(offspring[i])
        if li == 0:
            continue
        for k in range(K):
            lj = int(offspring[M[i, k]])
            if lj == 0 or lj == li:
                continue
            if Mdist[i, k] <= dt1:
                key = (min(li, lj), max(li, lj))
                links[key] = links.get(key, 0) + 1

    # 2. Chon cap cum de gop:
    # 70% xac suat: chon cap co do gan ket bien manh nhat link / min(|A|, |B|)
    # 30% xac suat: fallback theo 1 / D(proto1, proto2) nhu goc
    pair_chosen = None
    if links and rng.random() < 0.7:
        best_score = -1.0
        for (a, b), cnt in links.items():
            sa, sb = size_map[a], size_map[b]
            # Uu tien gop neu mot ben la manh nho
            score = cnt / float(min(sa, sb))
            if score > best_score:
                best_score = score
                pair_chosen = (a, b)

    if pair_chosen is None:
        # Fallback theo khoang cach prototype cua ban goc
        cluster1 = int(labs[rng.integers(len(labs))])
        D = model.D
        p = np.zeros(label_max + 1, dtype=np.float64)
        total = 0.0
        for l in labs:
            if l != cluster1 and proto[l] >= 0 and proto[cluster1] >= 0:
                d = D[proto[l], proto[cluster1]]
                if d > 0:
                    p[l] = 1.0 / d
                    total += p[l]
        if total > 0:
            p /= total
            r = rng.random()
            cum = 0.0
            for l in labs:
                cum += p[l]
                if cum >= r:
                    pair_chosen = (cluster1, int(l))
                    break

    if pair_chosen is not None:
        c1, c2 = pair_chosen
        # Gop cum nho hon vao cum lon hon de on dinh nhan
        if size_map.get(c1, 0) < size_map.get(c2, 0):
            c1, c2 = c2, c1
        offspring[parent == c2] = c1

    return offspring


def mutation_split_graph(model, parent, rng):
    """Tach mot cum voi hai tam chon theo mat do va khoang cach xa nhau (rho * D)."""
    offspring = parent.copy()
    label_max = int(parent.max())
    if label_max < 1:
        return offspring

    size_clust = np.zeros(label_max + 1, dtype=np.int64)
    for i in range(model.N):
        lab = int(parent[i])
        if lab > 0:
            size_clust[lab] += 1
    size_total = int(size_clust.sum())
    if size_total < 1:
        return offspring

    # Chon cum theo kich thuoc
    ri = int(rng.integers(1, size_total + 1))
    partial = 0
    chosen = 0
    for l in range(1, label_max + 1):
        partial += size_clust[l]
        if ri <= partial:
            chosen = l
            break
    if chosen == 0 or size_clust[chosen] <= 3:
        return offspring

    mask = parent == chosen
    idx = np.where(mask)[0]
    rho = model.rho[idx]

    # Tam 1: chon theo xac suat ty le voi mat do rho
    rsum = rho.sum()
    if rsum <= 0:
        return offspring
    p1 = rho / rsum
    proto1 = idx[int(rng.choice(len(idx), p=p1))]

    # Tam 2: chon theo rho * D(i, proto1) de dam bao hai tam o hai vung khac nhau
    D1 = model.D[idx, proto1]
    weights2 = rho * D1
    w2sum = weights2.sum()
    if w2sum <= 0:
        return offspring
    p2 = weights2 / w2sum
    proto2 = idx[int(rng.choice(len(idx), p=p2))]

    if proto1 == proto2:
        return offspring

    # Nhan moi: nhan trong nho nhat hoac label_max + 1
    new_label = 0
    for l in range(1, label_max + 1):
        if size_clust[l] == 0:
            new_label = l
            break
    if new_label == 0:
        new_label = label_max + 1

    D = model.D
    move = idx[D[idx, proto2] < D[idx, proto1]]
    if len(move) > 0 and len(move) < len(idx):
        offspring[move] = new_label

    return offspring


def ls_fi_with_noise(model, x, f, rng, test_noise=True):
    """First-improvement local search ho tro thu nghiem nhan nhieu 0 cho cac diem thoa rho <= d_rho."""
    N, K, M = model.N, model.K, model.M
    d_rho = model.d_rho
    rho = model.rho
    count_it = 0
    count_plato = 0
    cont = True

    while cont and count_it < N // 2:
        v_out = rng.permutation(N)
        improving = False

        for j in range(N):
            i = int(v_out[j])
            xi_old = int(x[i])

            # 1. Thu cac nhan tu lang gieng Gep
            for k in range(K):
                cand = int(x[M[i, k]])
                if cand == xi_old:
                    continue
                x[i] = cand
                df = model.df_element(x, i, xi_old)
                if df < 0.0:
                    improving = True
                    break
                elif df > 0.0:
                    x[i] = xi_old
                else:
                    xi_old = cand

            if improving:
                break

            # 2. Neu diem co mat do thap va chua duoc cai thien, thu gan ve nhan nhieu 0
            if test_noise and rho[i] <= d_rho and xi_old != 0:
                x[i] = 0
                df = model.df_element(x, i, xi_old)
                if df < 0.0:
                    improving = True
                    break
                else:
                    x[i] = xi_old

        if improving:
            count_plato = 0
        else:
            count_plato += 1
            if count_plato > 3:
                cont = False
        count_it += 1

    return model.comp_fitness(x)

