"""
Partition crossover for clustering -- faithful port of the reference nk.h.

  map_solutions: relabel the red parent's clusters to best match the blue parent
      (greedy maximum-overlap assignment). MUST run before px, otherwise the
      redundant integer encoding makes identical partitions look fully different.
  px:            build the symmetric union graph over the Gep edges of the
      differing vertices, find connected components, and for each recombining
      component pick the parent side with the lower partial cost. Partial costs
      are obtained by bucketing comp_fi over ALL N vertices by component -- this
      correctly accounts for boundary subfunctions.
  fix_labels:    give distinct labels to spatially disconnected clusters that
      happen to share the same integer label.

Labels are 1-based; 0 denotes noise.
"""

from __future__ import annotations

import numpy as np


class _DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def map_solutions(solution_blue, solution_red):
    """Return a relabelled copy of solution_red aligned to solution_blue."""
    lcrom = len(solution_blue)
    nc_blue = int(solution_blue.max())
    nc_red = int(solution_red.max())
    red_new = solution_red.copy()
    if nc_blue == 0 or nc_red == 0:
        return red_new
    nc_min = min(nc_blue, nc_red)

    overlap = np.zeros((nc_blue + 1, nc_red + 1), dtype=np.int64)
    for i in range(lcrom):
        b, r = int(solution_blue[i]), int(solution_red[i])
        if b > 0 and r > 0:
            overlap[b][r] += 1

    lin_free = np.ones(nc_blue + 1, dtype=bool)
    col_free = np.ones(nc_red + 1, dtype=bool)
    map_br = np.zeros(nc_red + 1, dtype=np.int64)

    for _ in range(nc_min):
        best, lin, col = -1, -1, -1
        for i in range(1, nc_blue + 1):
            if not lin_free[i]:
                continue
            for j in range(1, nc_red + 1):
                if col_free[j] and overlap[i][j] > best:
                    best, lin, col = overlap[i][j], i, j
        if col < 0:
            break
        map_br[col] = lin
        lin_free[lin] = False
        col_free[col] = False

    # red labels with no mapping get assigned to still-unused labels.
    # `missing` is sized over max(nc_blue, nc_red): a mapped red carries a BLUE
    # label that can exceed nc_red, and unmapped reds get fresh labels above
    # nc_blue. (Reference `missing_red`, allocated Nc_max+1.)
    nc_max = max(nc_blue, nc_red)
    missing = np.zeros(nc_max + 1, dtype=np.int64)
    missing[1:nc_red + 1] = 1
    for r in range(1, nc_red + 1):
        if map_br[r] > 0:
            missing[map_br[r]] = 0
    i = 1
    for r in range(1, nc_red + 1):
        if map_br[r] == 0:
            while missing[i] == 0:
                i += 1
            map_br[r] = i
            missing[i] = 0

    for k in range(lcrom):
        r = int(solution_red[k])
        red_new[k] = 0 if r == 0 else map_br[r]
    return red_new


def px(model, solution_blue, solution_red):
    """Return (offspring, fitness). solution_red is assumed already mapped."""
    N = model.N
    rev_flat, rev_off, rev_cnt = model.rev_flat, model.rev_off, model.rev_cnt

    differ = solution_blue != solution_red

    # symmetric union graph over the Gep edges of the differing vertices
    dsu = _DSU(N)
    for i in range(N):
        if differ[i]:
            start, cnt = rev_off[i], rev_cnt[i]
            for r in range(cnt):
                dsu.union(i, int(rev_flat[start + r]))

    comp_id = np.array([dsu.find(i) for i in range(N)], dtype=np.int64)
    _, comp_id = np.unique(comp_id, return_inverse=True)
    n_comp = int(comp_id.max()) + 1

    sizes = np.bincount(comp_id, minlength=n_comp)
    # a component recombines if it has >1 vertex or contains a differing vertex
    is_true = sizes > 1
    for i in range(N):
        if differ[i]:
            is_true[comp_id[i]] = True

    cost_blue = np.zeros(n_comp, dtype=np.float64)
    cost_red = np.zeros(n_comp, dtype=np.float64)
    common = 0.0
    for i in range(N):
        c = comp_id[i]
        fb = model.comp_fi(solution_blue, i)
        if is_true[c]:
            cost_blue[c] += fb
        else:
            common += fb
    for i in range(N):
        c = comp_id[i]
        if is_true[c]:
            cost_red[c] += model.comp_fi(solution_red, i)

    select_red = np.zeros(n_comp, dtype=bool)
    fit = common
    for c in range(n_comp):
        if is_true[c]:
            if cost_red[c] < cost_blue[c]:
                select_red[c] = True
                fit += cost_red[c]
            else:
                fit += cost_blue[c]

    offspring = np.where(select_red[comp_id], solution_red, solution_blue)
    offspring = np.ascontiguousarray(offspring, dtype=np.int64)
    return offspring, fit


def fix_labels(model, offspring):
    """Split disconnected same-label clusters by giving them fresh labels.

    Modifies offspring in place. Reference `nk::fixLabels`.
    """
    N = model.N
    rev_flat, rev_off, rev_cnt = model.rev_flat, model.rev_off, model.rev_cnt
    label_max = int(offspring.max())
    if label_max <= 1:
        return offspring

    # union graph keeping only same-label Gep edges
    dsu = _DSU(N)
    for i in range(N):
        start, cnt = rev_off[i], rev_cnt[i]
        for r in range(cnt):
            k = int(rev_flat[start + r])
            if offspring[i] == offspring[k]:
                dsu.union(i, k)

    roots = np.array([dsu.find(i) for i in range(N)], dtype=np.int64)
    uniq_roots, comp_id = np.unique(roots, return_inverse=True)
    n_comp = len(uniq_roots)

    comp_label = np.zeros(n_comp, dtype=np.int64)
    for i in range(N):
        comp_label[comp_id[i]] = offspring[i]

    # count components per label; relabel repeats after the first occurrence
    seen = set()
    new_label = label_max + 1
    remap = comp_label.copy()
    for c in range(n_comp):
        lab = int(comp_label[c])
        if lab <= 0:
            continue
        if lab not in seen:
            seen.add(lab)
        else:
            remap[c] = new_label
            new_label += 1

    for i in range(N):
        offspring[i] = remap[comp_id[i]]
    return offspring
