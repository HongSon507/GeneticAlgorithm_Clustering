"""
CGA baseline (Hruschka & Ebecken clustering genetic algorithm), paper Section II-A.

A genuine competing GA -- NOT NK-HGA with local search disabled:
  * integer encoding (labels 1..Nc, same shape as NK-HGA, no noise label);
  * fitness = centroid-based silhouette width criterion, MAXIMIZED (O(N*Nc));
  * mutations (equal probability when no crossover): centroid `merge` and
    centroid `split`;
  * grouping-style crossover (rate pc): inherit whole clusters, then fill the
    remaining objects by nearest centroid;
  * pop_size 100, tournament size 3, elitism.

This is a from-scratch estimator; it does not use the NKCV2 model.
"""

from __future__ import annotations

import time

import numpy as np


def _renumber(labels):
    """Relabel to 1..Nc by first appearance (CGA has no noise label)."""
    out = np.zeros_like(labels)
    mapping = {}
    nxt = 1
    for i in range(len(labels)):
        lab = int(labels[i])
        if lab not in mapping:
            mapping[lab] = nxt
            nxt += 1
        out[i] = mapping[lab]
    return out


def _centroids(X, labels, n_clusters):
    """Centroid per label 1..n_clusters and each cluster's size (1-based)."""
    d = X.shape[1]
    cent = np.zeros((n_clusters + 1, d), dtype=np.float64)
    size = np.zeros(n_clusters + 1, dtype=np.int64)
    for i in range(X.shape[0]):
        lab = int(labels[i])
        cent[lab] += X[i]
        size[lab] += 1
    for c in range(1, n_clusters + 1):
        if size[c] > 0:
            cent[c] /= size[c]
    return cent, size


def _silhouette(X, labels):
    """Centroid-based silhouette width criterion (mean over objects), MAXIMIZE.

    For object j: a = distance to own centroid, b = distance to the nearest
    other centroid, s = (b - a) / max(a, b). Singleton clusters contribute 0.
    """
    n_clusters = int(labels.max())
    if n_clusters < 2:
        return -1.0
    cent, size = _centroids(X, labels, n_clusters)
    # distance from every object to every centroid (Euclidean)
    diff = X[:, None, :] - cent[None, 1:, :]  # (N, Nc, d)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (N, Nc), col c-1 == label c+1
    s_sum = 0.0
    for i in range(X.shape[0]):
        lab = int(labels[i])
        if size[lab] <= 1:  # singleton -> s = 0
            continue
        a = dist[i, lab - 1]
        row = dist[i].copy()
        row[lab - 1] = np.inf
        b = row.min()
        m = max(a, b)
        if m > 0.0:
            s_sum += (b - a) / m
    return s_sum / X.shape[0]


class CGA:
    """Clustering Genetic Algorithm baseline (silhouette-maximizing)."""

    def __init__(self, pop_size=100, p_cross=0.5, tournament_size=3,
                 nc_max=None, max_gen=200, time_budget=False,
                 random_state=0, verbose=False):
        self.pop_size = pop_size
        self.p_cross = p_cross
        self.tournament_size = tournament_size
        self.nc_max = nc_max
        self.max_gen = max_gen
        self.time_budget = time_budget
        self.random_state = random_state
        self.verbose = verbose

    # -- operators -------------------------------------------------------------

    def _random_partition(self, rng, N, nc_max):
        k = int(rng.integers(2, nc_max + 1))
        return _renumber(rng.integers(1, k + 1, size=N).astype(np.int64))

    def _selection(self, fitness, rng):
        # silhouette is MAXIMIZED -> pick the highest-fitness of the tournament
        pool = rng.integers(0, len(fitness), size=self.tournament_size)
        return int(pool[np.argmax(fitness[pool])])

    def _mutation_merge(self, X, parent, rng):
        offspring = parent.copy()
        nc = int(parent.max())
        if nc <= 2:
            return offspring
        cent, size = _centroids(X, parent, nc)
        c1 = int(rng.integers(1, nc + 1))
        # nearest other cluster by centroid distance
        best, c2 = np.inf, -1
        for c in range(1, nc + 1):
            if c == c1 or size[c] == 0:
                continue
            dd = np.sum((cent[c] - cent[c1]) ** 2)
            if dd < best:
                best, c2 = dd, c
        if c2 > 0:
            offspring[parent == c2] = c1
        return _renumber(offspring)

    def _mutation_split(self, X, parent, rng):
        offspring = parent.copy()
        nc = int(parent.max())
        cent, size = _centroids(X, parent, nc)
        candidates = [c for c in range(1, nc + 1) if size[c] > 1]
        if not candidates:
            return offspring
        c1 = int(rng.choice(candidates))
        idx = np.where(parent == c1)[0]
        # farthest object from the original centroid
        d_c = np.sum((X[idx] - cent[c1]) ** 2, axis=1)
        far = idx[int(np.argmax(d_c))]
        # objects closer to `far` than to the centroid form the new cluster
        d_far = np.sum((X[idx] - X[far]) ** 2, axis=1)
        move = idx[d_far < d_c]
        if 0 < len(move) < len(idx):
            offspring[move] = nc + 1
        return _renumber(offspring)

    def _mutation(self, X, parent, rng):
        if rng.random() < 0.5:
            return self._mutation_merge(X, parent, rng)
        return self._mutation_split(X, parent, rng)

    def _crossover(self, X, p1, p2, rng):
        """Grouping-style crossover: inherit whole clusters, fill by nearest
        centroid (paper Section II-A)."""
        N = len(p1)
        offspring = np.zeros(N, dtype=np.int64)
        nc1 = int(p1.max())
        labels1 = np.arange(1, nc1 + 1)
        rng.shuffle(labels1)
        k1 = int(rng.integers(1, nc1 + 1))
        nxt = 1
        for lab in labels1[:k1]:
            mask = p1 == lab
            if mask.any():
                offspring[mask] = nxt
                nxt += 1
        # copy non-overlapping clusters from parent 2
        for lab in range(1, int(p2.max()) + 1):
            mask = p2 == lab
            if mask.any() and np.all(offspring[mask] == 0):
                offspring[mask] = nxt
                nxt += 1
        # assign the rest to the nearest existing centroid
        unassigned = offspring == 0
        if unassigned.any():
            if nxt == 1:  # nothing inherited -> fall back to parent 1
                return _renumber(p1.copy())
            cent, _ = _centroids(X, offspring, nxt - 1)
            diff = X[unassigned][:, None, :] - cent[None, 1:nxt, :]
            dist = np.sum(diff * diff, axis=2)
            offspring[unassigned] = np.argmin(dist, axis=1) + 1
        return _renumber(offspring)

    # -- main loop -------------------------------------------------------------

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        N = X.shape[0]
        nc_max = self.nc_max or max(2, int(np.sqrt(N)))

        chroms = [self._random_partition(rng, N, nc_max)
                  for _ in range(self.pop_size)]
        fits = np.array([_silhouette(X, c) for c in chroms], dtype=np.float64)

        best_idx = int(np.argmax(fits))
        best_chrom = chroms[best_idx].copy()
        best_fit = float(fits[best_idx])

        t_start = time.time()
        gen = 0

        def stop():
            if self.time_budget:
                return (time.time() - t_start) >= (N / 2.0)
            return gen >= self.max_gen

        while not stop():
            gen += 1
            cur_best = int(np.argmax(fits))
            new_chroms = [None] * self.pop_size
            new_fits = np.empty(self.pop_size, dtype=np.float64)
            for j in range(self.pop_size - 1):
                if rng.random() < self.p_cross:
                    p1 = chroms[self._selection(fits, rng)]
                    p2 = chroms[self._selection(fits, rng)]
                    child = self._crossover(X, p1, p2, rng)
                else:
                    p = chroms[self._selection(fits, rng)]
                    child = self._mutation(X, p, rng)
                new_chroms[j] = child
                new_fits[j] = _silhouette(X, child)
            # elitism
            new_chroms[self.pop_size - 1] = chroms[cur_best].copy()
            new_fits[self.pop_size - 1] = fits[cur_best]
            chroms, fits = new_chroms, new_fits

            gb = int(np.argmax(fits))
            if fits[gb] > best_fit:
                best_fit = float(fits[gb])
                best_chrom = chroms[gb].copy()

            if self.verbose and gen % 20 == 0:
                nc = len(np.unique(best_chrom))
                print(f"  [CGA] gen {gen}: silhouette={best_fit:.4f}, clusters={nc}")

        self.labels_ = _renumber(best_chrom)
        self.best_fitness_ = best_fit
        self.n_clusters_ = len(np.unique(self.labels_))
        self.n_gen_ = gen
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
