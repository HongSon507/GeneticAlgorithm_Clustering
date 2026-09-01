"""
NK Hybrid Genetic Algorithm for Clustering (NK-HGA).

Faithful port of the reference `ga_clustering.cpp` main loop:
  * memetic structure -- local search is applied at initialization (immigrants)
    and PERIODICALLY (every `leng_local` generations to a fraction of the
    population), NOT to every child;
  * each generation produces popsize-1 offspring, each either a mutation of one
    selected parent (prob 1-pc) or a partition crossover of two parents (prob pc),
    plus one elite copy;
  * stopping criterion is wall-clock N/2 seconds, with a portable max_gen fallback.

Labels are 1-based; 0 denotes noise.
"""

from __future__ import annotations

import time

import numpy as np

from nkcv2 import NKCV2Model
from search import ls_fi
from mutations import mutation_reclassify, mutation_merge, mutation_split
from px import map_solutions, px, fix_labels


def renumber(x):
    """Relabel positive clusters to 1..Nc by first appearance; keep 0 as noise."""
    out = np.zeros_like(x)
    mapping = {}
    nxt = 1
    for i in range(len(x)):
        lab = int(x[i])
        if lab == 0:
            continue
        if lab not in mapping:
            mapping[lab] = nxt
            nxt += 1
        out[i] = mapping[lab]
    return out


class NKHGA:
    """sklearn-style NK-HGA clustering estimator."""

    def __init__(self, K=3, pop_size=100, p_cross=0.6, tournament_size=3,
                 leng_local=100, imig_ls_ratio=0.7, max_gen=200,
                 time_budget=False, random_state=0, verbose=False):
        self.K = K
        self.pop_size = pop_size
        self.p_cross = p_cross
        self.tournament_size = tournament_size
        self.leng_local = leng_local
        self.imig_ls_ratio = imig_ls_ratio
        self.max_gen = max_gen
        self.time_budget = time_budget  # True -> stop at N/2 seconds (paper protocol)
        self.random_state = random_state
        self.verbose = verbose

    # -- building blocks -------------------------------------------------------

    def _immigrant(self, model, rng):
        k = int(rng.integers(2, max(3, int(np.sqrt(model.N))) + 1))
        x = rng.integers(1, k + 1, size=model.N).astype(np.int64)
        x = np.ascontiguousarray(x)
        f = model.comp_fitness(x)
        f = ls_fi(model, x, f, rng)
        return x, f

    def _selection(self, fitness, rng):
        pool = rng.integers(0, len(fitness), size=self.tournament_size)
        return int(pool[np.argmin(fitness[pool])])

    def _mutation(self, model, parent, fit_parent, rng):
        r = rng.random()
        if r < 0.6:
            return mutation_reclassify(model, parent, fit_parent, rng)
        elif r < 0.8:
            child = mutation_merge(model, parent, rng)
            return child, model.comp_fitness(child)
        else:
            child = mutation_split(model, parent, rng)
            return child, model.comp_fitness(child)

    def _crossover(self, model, p1, p2):
        p2_mapped = map_solutions(p1, p2)
        offspring, fit = px(model, p1, p2_mapped)
        fix_labels(model, offspring)
        return offspring, fit

    # -- main loop -------------------------------------------------------------

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        model = NKCV2Model(X, K=self.K)
        self.model_ = model
        N = model.N

        if self.verbose:
            print(f"NK-HGA: N={N}, K={self.K}, pop={self.pop_size}, "
                  f"Kout={model.Kout}")

        # initialization: immigrants (random + local search)
        chroms = [None] * self.pop_size
        fits = np.empty(self.pop_size, dtype=np.float64)
        for i in range(self.pop_size):
            chroms[i], fits[i] = self._immigrant(model, rng)

        best_idx = int(np.argmin(fits))
        best_chrom = chroms[best_idx].copy()
        best_fit = float(fits[best_idx])

        t_start = time.time()
        gen = 0
        gen_init = 0

        def stop():
            if self.time_budget:
                return (time.time() - t_start) >= (N / 2.0)
            return gen >= self.max_gen

        while not stop():
            gen += 1

            # periodic local-search / immigrant refresh
            if (gen - gen_init) > self.leng_local:
                gen_init = gen
                chroms[0] = best_chrom.copy()
                fits[0] = best_fit
                n_ls = int(self.imig_ls_ratio * self.pop_size)
                for i in range(n_ls):
                    fits[i] = ls_fi(model, chroms[i], fits[i], rng)
                for i in range(n_ls, self.pop_size):
                    chroms[i], fits[i] = self._immigrant(model, rng)

            # produce the new population
            cur_best = int(np.argmin(fits))
            new_chroms = [None] * self.pop_size
            new_fits = np.empty(self.pop_size, dtype=np.float64)
            for j in range(self.pop_size - 1):
                if rng.random() > self.p_cross:
                    p = self._selection(fits, rng)
                    child, cf = self._mutation(model, chroms[p], fits[p], rng)
                else:
                    p1 = self._selection(fits, rng)
                    p2 = self._selection(fits, rng)
                    child, cf = self._crossover(model, chroms[p1], chroms[p2])
                new_chroms[j] = child
                new_fits[j] = cf
            # elitism: last slot holds the current best
            new_chroms[self.pop_size - 1] = chroms[cur_best].copy()
            new_fits[self.pop_size - 1] = fits[cur_best]

            chroms, fits = new_chroms, new_fits

            gb = int(np.argmin(fits))
            if fits[gb] < best_fit:
                best_fit = float(fits[gb])
                best_chrom = chroms[gb].copy()

            if self.verbose and gen % 20 == 0:
                nc = len(np.unique(best_chrom[best_chrom > 0]))
                print(f"  gen {gen}: fitness={best_fit:.6f}, clusters={nc}")

        self.labels_ = renumber(best_chrom)
        self.best_fitness_ = best_fit
        self.n_clusters_ = len(np.unique(self.labels_[self.labels_ > 0]))
        self.n_gen_ = gen
        if self.verbose:
            print(f"Done. gen={gen}, fitness={best_fit:.6f}, "
                  f"clusters={self.n_clusters_}, time={time.time()-t_start:.1f}s")
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
