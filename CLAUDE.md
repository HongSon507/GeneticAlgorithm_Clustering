# CLAUDE.md — NK Hybrid Genetic Algorithm for Clustering (NK-HGA)

This file is the **canonical implementation specification agents**.
It has been **verified
line-by-line against the authors' reference source** `nk.h` / `ga_clustering.cpp`
(https://github.com/rtinos/NKGAclust) — the earlier "reconstruction / [VERIFY]" caveats are
resolved and folded into the correct behavior below.

**Source paper:** R. Tinós, L. Zhao, F. Chicano, D. Whitley, *"NK Hybrid Genetic Algorithm for
Clustering"*, IEEE Transactions on Evolutionary Computation.
**Reference implementation (authors):** https://github.com/rtinos/NKGAclust
**Local translation of the paper:** `240203813v(truoc)1.md`

> The implementation lives in the modular files below and is validated: the required unit tests
> pass, and on **Aggregation** NK-HGA reaches **ARI ≈ 0.99 with 7 clusters**, beating CGA (≈0.77),
> k-means (≈0.71) and DBSCAN (≈0.73). See §14.

> Repository layout (see `README.md`): algorithm modules in `src/`, unit tests in `tests/`, the
> dataset in `data/`, the paper + translation in `docs/`, the **authors' original C++ source** in
> `reference/`, and generated figures in `results/`. The old monolithic prototype `nk_hga.py` has
> been removed — the modular `src/` files are authoritative.

---

## 0. How to use this guide

Implement in this order — each layer depends only on earlier ones:

```
1. Preprocessing (§2)  ─┐
2. Interaction graph Gep (§3)  ─┼─► nkcv2.py
3. NKCV2 evaluation (§4)  ─┤
4. Delta-evaluation (§5)  ─┘
5. Mutations (§6) ─► mutations.py     6. Local search (§7) ─► search.py
7. Partition crossover (§8) ─► px.py  8. Main GA loop (§9) ─► ga.py
9. CGA baseline (§10) ─► cga.py        10. Harness (§11) ─► run_comparison.py
```

Conventions used throughout:
- `N` = number of objects; `l` = dimensionality; `Y = {y_1..y_N}` the dataset (rows of `X`).
- `x ∈ ℕ^N` = a partition (solution). `x_j = i` means object `y_j` belongs to cluster `C_i`.
- `K` = group size: each subfunction `f_i` reads **exactly `K`** other objects. Default `K = 3`.
- **All distances are SQUARED Euclidean** (`sqrEucDist`): `D_ij = ‖y_i − y_j‖²`. Every threshold
  and the density `ρ` are computed on these squared distances — do not take square roots.
- **NKCV2 is MINIMIZED** and scaled by `1/(N·K)`.
- Labels are **1-based**; **`x_j = 0` means noise**.

### 0.1 Gap table — corrected behavior vs the old `nk_hga.py`

| Topic | Correct (this guide = reference) | old `nk_hga.py` did |
|---|---|---|
| Distances | **squared** Euclidean everywhere (§2) | plain Euclidean |
| Interaction graph | density-link + proximity-fill, **indegree exactly K, NO self-loops** (§3) | symmetric KNN with self-loops |
| Density ρ | Gaussian kernel over **all** pairs, `dc` via 2% rule, scaled by max (§2) | kernel over KNN, `dc=0.05·max` |
| Thresholds | `dt0=mean`, `dt1=mean+2σ`, `dt2=mean+σ` over the `N·K` group distances (§4) | `dt0/dt1/dt2` ad-hoc |
| Ramps | **both** same-label and different-label ramp over `dt0→dt1`; `dt2` used **only** in the noise branch (§4) | wrong thresholds per case |
| Noise | label `0`: cheap iff `ρ_i ≤ d_rho` **and** `D > dt2` (§4) | not supported |
| Mutations | `mutationReclassify` + `mutationMerge` + `mutationSplit` (§6) | one neighbor-copy |
| Local search | first-improvement + neutral drift + plateau streak, Δ-eval (§7) | approximate |
| Partition crossover | `mapSolutions` first, then bucket `comp_fi` over **all N**, then `fixLabels` (§8) | partial sum over `f_i` only ❌ |
| Main loop | memetic: LS at init + **periodic** LS refresh, not per-child (§9) | LS per child / off |
| Stopping | wall-clock `N/2` s, with a portable `max_gen` fallback (§9) | fixed generations |
| CGA baseline | real silhouette-maximizing CGA (§10) | NK-HGA with LS off |

---

## 1. Problem statement & encoding

Hard partitional clustering: given `Y`, find the partition `C = {C_1..C_{Nc}}` that minimizes the
internal validation criterion NKCV2. The number of clusters `Nc` is **not** fixed — it emerges
from optimization.

**Integer encoding** (Section II of the paper):
- Solution vector `x ∈ ℕ^N`. `x_j = i > 0` ⇒ `y_j ∈ C_i`.
- **`x_j = 0` ⇒ `y_j` is noise.**
- The encoding is **redundant**: `{1,1,1,3,3,2,2}` and `{2,2,2,1,1,3,3}` denote the same partition,
  so **renumber** (canonicalize labels, 1..Nc by first appearance, keep 0) after recombination.
  See `renumber()` in `ga.py`.

---

## 2. Preprocessing (compute ONCE per dataset) → `nkcv2.py`

### 2.1 Pairwise squared distances
`D_ij = ‖y_i − y_j‖²` (reference `sqrEucDist`). Store the full `N×N` matrix `D` (symmetric, zero
diagonal). Everything downstream uses these squared values.

### 2.2 Cutoff distance `dc` — the 2% rule (Rodriguez & Laio, Section II-B)
Collect all `N(N-1)/2` upper-triangle distances into `vD`, sort ascending, and take
```
position = ⌊ (N(N-1)/2) · 2 / 100 ⌋
dc = vD[position]
```
i.e. the value below which ~2% of pairs fall (≈ average of 2% neighbors per point).

### 2.3 Local density ρ (Gaussian kernel over all pairs)
```
ρ_i = Σ_{j ≠ i} exp( −(D_ij / dc)² )
```
Then **scale to [0,1]** by dividing by `max(ρ)`. Compute `m_ρ, s_ρ` (population mean/std of ρ).

### 2.4 Noise density cutoff `d_rho`
```
d_rho = m_ρ − s_ρ ;   if d_rho ≤ 0 :  d_rho = m_ρ / 2
```

---

## 3. Interaction graph `Gep` (Section III-A) → `nkcv2.py`

`Gep` is a **directed** graph on vertices `v_1..v_N`. Edge `(v_j, v_i)` means "`x_j` influences
subfunction `f_i`". The **group** `M_i = { j : (v_j, v_i) ∈ Gep }` is the set of **exactly `K`**
object indices that feed `f_i`. **There are NO self-loops** (`i ∉ M_i`) and **indegree(`v_i`) = K**.

Build in two steps:

1. **Density link.** For each `v_i`, find `a_i` = the **nearest object with strictly higher
   density** (`ρ_{a_i} > ρ_i`, minimal `D_{i,a_i}`) and add edge `(v_{a_i}, v_i)`.
   - The single globally-densest object (no higher-density neighbor) links to its **nearest object**
     (`argmin_{j≠i} D_{ij}`), *not* to the farthest one. (Reference deviates from Density-Peaks here.)

2. **Proximity fill.** For each `v_i`, add incoming edges from its **nearest remaining objects**
   (ascending `D_ij`, excluding self and duplicates) until **indegree(`v_i`) = K**.

**Outputs to store** (all fixed per dataset):
- `M[N,K]` — the group `M_i` (the K objects influencing `f_i`).
- `Mdist[N,K] = D[i, M_i]` and `Mrho[N,K] = ρ[M_i]` (neighbor densities → `fmax`).
- **Reverse adjacency** `R_i = { p : i ∈ M_p }` in flat CSR arrays (`rev_flat`, `rev_off`,
  `rev_cnt`) — the subfunctions that *read* `x_i`; needed for Δ-evaluation (§5).
- **`Kout` = max_i |R_i|** — drives mutation/Δ cost; report it.

Complexity: `O(N² log N)` once.

---

## 4. NKCV2 evaluation (Section III-B) → `nkcv2.py`

```
f(x) = ( Σ_{i=1}^N f_i(x) ) / (N·K)          ── MINIMIZE
```
After preprocessing, `f(x)` costs `O(N·K)`.

### 4.1 Subfunction `f_i` (Eq. 8)
For each `j ∈ M_i` let `D = D_ij`, `fmax = ρ_j` (density of the neighbor, `fmax0 = 1`),
`fmin = 0` (`fmin0 = 0`). Sum the contribution of each neighbor per the three cases below,
then divide the whole `f_i` by `N·K`.

### 4.2 The α cases (reference `comp_fi`, exact)

**Noise** — `x_i = 0`:
```
contribution = fmin (= 0)   if ρ_i ≤ d_rho AND D > dt2      (object legitimately noise: cheap)
contribution = fmax (= ρ_j) otherwise                        (forcing noise elsewhere is costly)
```

**Same label** — `x_i = x_j` (both non-noise). Penalize being far apart; ramp **dt0 → dt1**:
```
contribution = 0                                  if D ≤ dt0
contribution = ρ_j · (D − dt0)/(dt1 − dt0)        if dt0 < D ≤ dt1
contribution = ρ_j                                if D > dt1
```

**Different label** — `x_i ≠ x_j` (both non-noise). Penalize being close; ramp **dt0 → dt1**:
```
contribution = ρ_j                                if D ≤ dt0
contribution = ρ_j · (1 − (D − dt0)/(dt1 − dt0))  if dt0 < D ≤ dt1
contribution = 0                                  if D > dt1
```

Note: **both** ramps span `dt0→dt1`. `dt2` appears **only** in the noise test.

### 4.3 Thresholds (reference `threeSigmaRule`, exact)
Let `xg` = the `N·K` group distances `{ D_{i,j} : j ∈ M_i }`, with population mean `m` and std `s`:
```
dt0 = m
dt1 = m + 2·s
dt2 = m + s
```
(The commented-out `dt0=m+s` variant in the source is **not** used.) These, plus `d_rho` (§2.4),
are the only tunables and should be exposed as parameters.

---

## 5. Delta-evaluation → `nkcv2.py`

Changing a single variable `x_i` from label `a` to label `b` affects **only**:
- **(a)** its own subfunction `f_i`, and
- **(b)** every `f_p` such that `i ∈ M_p` — the reverse neighbors `R_i` from §3.

```
Δ = [f_i(b) − f_i(a)] + Σ_{p ∈ R_i} [f_p(b) − f_p(a)]
```
Recompute only those subfunctions. Cost `O(Kout · K)`. Correctness of the whole system hinges on
Δ matching a from-scratch re-evaluation — this is unit-tested (§12).

---

## 6. Mutation operators → `mutations.py`

When a child is produced by mutation, one operator is chosen with the split used in `ga.py`
(reclassify 60% / merge 20% / split 20%). Each returns a mutated copy `x'`.

### 6.1 `mutation_reclassify(x)` — reference `mutationReclassify`
Best-improvement relabel of one random variable to a group member's label or to noise:
```
i = random index
best = argmin over v ∈ { x_j : j ∈ M_i } ∪ {0}  of  Δ_{i,v}(x)      # via §5
x'_i = best                                                          # always applied
return (x', fit_parent + Δ_min)
```
Returns the child *and* its fitness (from the incremental Δ), so no full re-eval is needed.

### 6.2 `mutation_merge(x)` — reference `mutationMerge`
```
1. prototype(l) = highest-density object of cluster l   (medoid by ρ)
2. cluster1 = a cluster chosen uniformly among existing labels (max-random trick)
3. cluster2 chosen with probability ∝ 1 / D(prototype(l), prototype(cluster1))   (closer → likelier)
4. reassign every object of cluster2 to cluster1
```
Fitness is recomputed by the caller (`comp_fitness`).

### 6.3 `mutation_split(x)` — reference `mutationSplit`
```
1. cluster1 chosen with probability ∝ |C_l|            (larger clusters likelier to split)
2. two prototypes m1, m2 sampled from cluster1 with probability ∝ ρ (density-weighted)
3. new_label = smallest empty label, else max_label + 1
4. each object j ∈ C_cluster1 with D(j, m2) < D(j, m1) → new_label
```
Fitness recomputed by the caller.

---

## 7. Local search — reference `LsFi` → `search.py`

First-improvement hill climber with neutral drift:
```
count_it = 0 ; plateau = 0
while plateau ≤ 3  and  count_it < N/2:
    for i in random permutation of 1..N:
        for cand in { x_j : j ∈ M_i } (skip cand == x_i):
            Δ = Δ_{i,cand}(x')                       # §5
            if Δ < 0:  accept, mark improving, break # first improvement
            elif Δ > 0: revert
            else:       accept neutral move and keep scanning (drift)
        if improving: break
    plateau = 0 if improving else plateau + 1
    count_it += 1
return comp_fitness(x')                              # exact fitness after drift
```
Neutral moves (`Δ = 0`) are intentionally accepted so the search drifts across plateaus.

---

## 8. Partition Crossover (PX) — reference `mapSolutions` + `px` + `fixLabels` → `px.py`

PX deterministically returns the **best of `2^q` offspring** at `O(N·K)`, where `q` is the number
of recombining components. **Run the three steps in order:**

### 8.1 `map_solutions(blue, red)` — MUST run first
The integer encoding is redundant, so before recombining, relabel the **red** parent to best match
the **blue** parent by greedy maximum-overlap assignment (reference `mapSolutions`):
```
overlap[b][r] = #{ i : blue_i = b, red_i = r }
repeat min(Nc_blue, Nc_red) times: match the (blue,red) pair with the largest free overlap
unmatched red labels get fresh labels not used by the matched blue labels
  (the `missing` array is sized over max(Nc_blue, Nc_red) — a matched red carries a BLUE label
   that can exceed Nc_red, and extra red clusters get labels above Nc_blue)
```

### 8.2 `px(blue, red_mapped)`
```
1. build the SYMMETRIC union graph over the Gep edges of the differing vertices
       for each i with blue_i ≠ red_i: union i with every j in i's Gep adjacency
2. connected components → comp_id[N]
3. a component is "recombining" (test=1) if it has size > 1 OR contains a differing vertex
4. bucket comp_fi over ALL N vertices:
       for recombining comps: accumulate cost_blue / cost_red per component
       for non-recombining:   accumulate into common_cost (identical for both parents)
5. per recombining component: pick the cheaper side (blue if cost_blue < cost_red, else red)
6. offspring = chosen side per component; return (offspring, common_cost + Σ chosen costs)
```
❗ **The partial cost must bucket `comp_fi` over every vertex**, not just `f_i` for `i` in the
component — boundary subfunctions that *read* a component variable also differ between parents. The
returned fitness equals `comp_fitness(offspring)` exactly (unit-tested, §12).

### 8.3 `fix_labels(offspring)`
Give distinct labels to spatially **disconnected** clusters that share the same integer label:
union only same-label Gep edges, find components, and relabel every repeated-label component after
its first occurrence (reference `fixLabels`).

---

## 9. Main GA loop — reference `ga_clustering.cpp` → `ga.py`

A **memetic** generational GA. Local search is applied at initialization and **periodically**
(not to every child):
```
model = NKCV2Model(X, K)
# initialization: immigrants = random partition (Nc random) + local search
for each of pop_size individuals: x = random; f = comp_fitness(x); f = LsFi(x, f)
track best-ever (minimize)

while not stop():                                   # §9.1
    # periodic refresh every `leng_local` generations:
    if (gen - gen_init) > leng_local:
        keep the best; apply LsFi to a fraction (imig_ls_ratio) of the population;
        replace the rest with fresh immigrants
    # produce the next generation (pop_size-1 offspring + 1 elite):
    for j in 1..pop_size-1:
        if rand() < pc:  child = PX(map+px+fixLabels) of two tournament parents   # §8
        else:            child = mutation of one tournament parent                 # §6 (60/20/20)
    elite slot = current best
return renumber(best-ever)
```
`tournament_select` picks the best of 3 random individuals. `renumber` canonicalizes labels
(1..Nc by first appearance; 0 stays noise).

### 9.1 Defaults (Section V-A3)
| Param | Value |
|---|---|
| `pop_size` | 100 |
| `pc` (crossover rate) | 0.6 |
| `K` | 3 |
| tournament size | 3 |
| elitism | on (1 individual) |
| `leng_local` (periodic LS interval) | 100 |
| `imig_ls_ratio` | 0.7 |
| stopping | wall-clock `N/2` seconds |

`time_budget=True` uses the paper's `N/2`-second protocol; `max_gen` is a **portable fallback**
so runs are reproducible across machine speeds.

---

## 10. CGA baseline (Section II-A) → `cga.py`

A genuine competing GA (Hruschka & Ebecken) — **not** in the reference repo; implemented from the
paper. **Not** NK-HGA with LS off.
- **Encoding:** integer, 1-based, no noise label.
- **Fitness:** centroid-based **silhouette width criterion — MAXIMIZE**.
  For object `j`: `a` = distance to its own centroid, `b` = distance to the nearest other centroid,
  `s = (b − a)/max(a, b)`; singleton clusters contribute `0`. Fitness = mean `s`. `O(N·Nc)`.
- **Mutations (equal prob when no crossover):**
  - *merge:* merge a random cluster with the **nearest** cluster (centroid distance).
  - *split:* pick a random cluster, find its object farthest from the centroid, move objects
    closer to that far object than to the centroid into a new cluster.
- **Crossover (grouping-GA style, `pc = 0.5`):** inherit `k1` whole clusters from parent 1, copy
  non-overlapping clusters from parent 2, then assign remaining objects to the nearest centroid.
- `pop_size = 100`, tournament size 3, elitism.

---

## 11. Experiment harness → `run_comparison.py`

- **Dataset:** Aggregation from local `Aggregation.txt` (`X = cols 0:2`, `y = col 2`; 788 objects,
  7 clusters). Standardize with `StandardScaler`.
- **External metric:** Adjusted Rand Index (`adjusted_rand_score`) + predicted number of clusters.
- **Model selection across runs:** DBCV if a `dbcv`/`validclust` package is installed; otherwise a
  **documented substitute** — silhouette over non-noise points (biased toward spherical clusters,
  flagged in the report). ARI is used **only** for the final external comparison, never to select;
  NK-HGA is **not** selected on NKCV2 (unfair). See `selection_score` / `select_best`.
- Compares NK-HGA vs CGA vs k-means vs DBSCAN and saves `comparison_result.png`.
- CLI: `--runs`, `--pop`, `--gen`, `--K`, `--time-budget`.

---

## 12. Module layout & complexity

```
src/nkcv2.py     density (§2), Gep (§3), f_i / f (§4), df_element (§5); numba-accelerated
                 kernels with a pure-Python fallback (runs without numba, e.g. Python 3.14)
src/mutations.py mutation_reclassify (§6.1), mutation_merge (§6.2), mutation_split (§6.3)
src/search.py    ls_fi (§7)
src/px.py        map_solutions, px, fix_labels (§8)
src/ga.py        NKHGA main loop (§9), renumber, tournament, elitism
src/cga.py       CGA baseline (§10)
src/run_comparison.py  datasets, ARI, DBCV/substitute selection, plots (§11)
tests/           run_tests.py (standalone unit-test runner, no pytest needed)
data/            Aggregation.txt        docs/       paper PDF + Markdown translation
reference/       authors' C++ source     results/    generated figures
```

| Operation | Cost |
|---|---|
| ρ + `dc` + `Gep` build (once) | `O(N² log N)` |
| Full NKCV2 eval `f(x)` | `O(N·K)` |
| Δ-eval (one variable) | `O(Kout·K)` |
| Partition crossover | `O(N·K)` |
| mutationMerge / Split | `O(Kout·K·|C|)` |

**Required unit tests (all passing):**
1. `df_element(x, i, b)` equals `comp_fitness(x_with_i=b) − comp_fitness(x)` for random `x, i, b`.
2. PX offspring's true `comp_fitness` equals the returned partial-sum fitness.
3. `renumber` is idempotent and preserves the partition.
Plus: PX offspring never worse than either parent; `fix_labels` splits disconnected same-label
clusters.

---

## 13. Reference cross-checks — RESOLVED

All items previously flagged `[VERIFY]` were confirmed against `nk.h` / `ga_clustering.cpp`:
- **Distances** are squared Euclidean (`sqrEucDist`); `dc`, ρ and all thresholds use them.
- **Thresholds** `dt0=m`, `dt1=m+2σ`, `dt2=m+σ` over the `N·K` group distances (`threeSigmaRule`).
- **α noise branch:** cheap iff `ρ_i ≤ d_rho` **and** `D > dt2`, else `ρ_j` (`comp_fi`).
- **Graph:** indegree exactly `K`, no self-loops; densest object links to its nearest object.
- **PX:** `mapSolutions` → symmetric union graph → bucket `comp_fi` over all `N` → `fixLabels`.
- **Main loop:** memetic generational GA with LS at init + periodic refresh, tournament, elitism.
- **`mutationReclassify` / `mutationMerge` / `mutationSplit`** match the reference operators.

---

## 14. Acceptance check — PASSED

Running `run_comparison.py` on **Aggregation** (pop 80, 100 gens, pure-Python fallback) produced:

| Method | ARI | Clusters |
|---|---|---|
| **NK-HGA** | **0.9876** | **7** |
| CGA | 0.7735 | 4 |
| k-means | 0.7113 | 7 |
| DBSCAN | 0.7338 | 4 |

NK-HGA reaches the paper's target (**ARI ≈ 0.99, ~7 clusters**, Table VII) and beats the CGA
baseline (paper CGA ≈ 0.70, Table V) as well as k-means and DBSCAN — confirming the density-based
NKCV2 + PX + local-search machinery is faithful to the paper. Installing `numba` (optional) makes
the kernels run much faster; the pure-Python fallback keeps the code runnable everywhere.
