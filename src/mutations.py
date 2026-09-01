"""
Mutation operators -- faithful port of the reference nk.h.

  mutation_reclassify (Mutation 1): best delta relabel of one variable to a
      neighbour's label OR to noise (0); always applies the best move.
  mutation_merge      (Mutation 2): merge a random cluster with another chosen
      with probability inversely proportional to prototype distance.
  mutation_split      (Mutation 3): split a size-weighted cluster into two using
      two density-weighted prototypes.

Labels are 1-based; 0 denotes noise.
"""

from __future__ import annotations

import numpy as np


def mutation_reclassify(model, parent, fit_parent, rng):
    """Return (offspring, fitness). Reference `mutationReclassify`."""
    offspring = parent.copy()
    N, K, M = model.N, model.K, model.M
    ri = int(rng.integers(0, N))
    old = int(offspring[ri])

    df_min = 0.0
    best = old
    started = False
    for k in range(K):
        cand = int(offspring[M[ri, k]])
        offspring[ri] = cand
        df = model.df_element(offspring, ri, old)
        if not started or df < df_min:
            df_min = df
            best = cand
            started = True

    # also test relabelling to noise (0)
    offspring[ri] = 0
    df = model.df_element(offspring, ri, old)
    if not started or df < df_min:
        df_min = df
        best = 0

    offspring[ri] = best
    return offspring, fit_parent + df_min


def _cluster_prototypes(model, parent, label_max):
    """Prototype (highest-density object) index per label; -1 if empty. 1-based."""
    proto = np.full(label_max + 1, -1, dtype=np.int64)
    rho = model.rho
    for i in range(model.N):
        lab = int(parent[i])
        if lab > 0:
            if proto[lab] == -1 or rho[i] > rho[proto[lab]]:
                proto[lab] = i
    return proto


def mutation_merge(model, parent, rng):
    """Return offspring. Reference `mutationMerge` (fitness recomputed by caller)."""
    offspring = parent.copy()
    label_max = int(parent.max())
    if label_max <= 1:
        return offspring

    proto = _cluster_prototypes(model, parent, label_max)
    D = model.D

    # cluster1: uniform among existing clusters (via the reference's max-random trick)
    cluster1 = -1
    aux = -1.0
    for l in range(1, label_max + 1):
        if proto[l] >= 0:
            r = rng.random()
            if r > aux:
                aux = r
                cluster1 = l
    if cluster1 < 0:
        return offspring

    # cluster2: probability proportional to 1 / distance between prototypes
    p = np.zeros(label_max + 1, dtype=np.float64)
    total = 0.0
    for l in range(1, label_max + 1):
        if proto[l] >= 0 and l != cluster1 and D[proto[l], proto[cluster1]] > 0.0:
            p[l] = 1.0 / D[proto[l], proto[cluster1]]
            total += p[l]
    if total <= 0.0:
        return offspring
    p /= total

    r = rng.random()
    cum = 0.0
    l = 1
    while l <= label_max and cum <= r:
        cum += p[l]
        l += 1
    cluster2 = l - 1 if l <= label_max else label_max

    if cluster1 > 0 and cluster2 > 0 and cluster2 != cluster1:
        offspring[parent == cluster2] = cluster1
    return offspring


def mutation_split(model, parent, rng):
    """Return offspring. Reference `mutationSplit` (fitness recomputed by caller)."""
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

    # choose a cluster with probability proportional to its size
    ri = int(rng.integers(1, size_total + 1))
    partial = 0
    chosen = 0
    for l in range(1, label_max + 1):
        partial += size_clust[l]
        if ri <= partial:
            chosen = l
            break
    if chosen == 0 or size_clust[chosen] <= 1:
        return offspring

    # two prototypes sampled with probability proportional to density
    rho = model.rho
    mask = parent == chosen
    weights = np.where(mask, rho, 0.0)
    wsum = weights.sum()
    if wsum <= 0.0:
        return offspring
    prob = weights / wsum

    r1, r2 = rng.random(), rng.random()
    if r1 > r2:
        r1, r2 = r2, r1
    cum = 0.0
    proto1 = proto2 = 0
    got1 = got2 = False
    for i in range(model.N):
        cum += prob[i]
        if not got1 and cum >= r1:
            proto1 = i
            got1 = True
        if not got2 and cum >= r2:
            proto2 = i
            got2 = True

    # new label = smallest empty label, else label_max + 1
    new_label = 0
    for l in range(1, label_max + 1):
        if size_clust[l] == 0:
            new_label = l
            break
    if new_label == 0:
        new_label = label_max + 1

    D = model.D
    idx = np.where(mask)[0]
    move = idx[D[idx, proto2] < D[idx, proto1]]
    offspring[move] = new_label
    return offspring
