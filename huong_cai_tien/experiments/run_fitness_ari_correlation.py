"""
Do tuong quan fitness--ARI cua NKCV2 goc va CDW tren cung pipeline GA.

Muc tieu la kiem tra: khi fitness (can toi thieu) giam, ARI (can tang) co thuc
su tang hay khong. Tuong quan am la dung chieu; tuong quan duong cho thay ham
muc tieu co the dang uu tien loi giai sai.

Khong sua src/: chi IMPORT nguyen ban NKCV2Model + cac building block cua GA
(ls_fi, mutation_*, px/map_solutions/fix_labels) tu src/, roi tu viet lai vong
lap chinh cua NKHGA.fit() (ga.py, CLAUDE.md Sec9) VOI THEM MOC GHI (fitness, ARI)
tai moi the he -- vi ga.py:NKHGA.fit() khong expose quan the noi bo nen khong co
cach nao "moc" vao no ma khong sua file goc. Logic toi uu (thu tu goi rng, ti le
mutation/crossover, elitism, local-search dinh ky) giu nguyen 1-1 so voi ga.py.

Hai phep do:
  (A) "Trajectory" -- (best_fitness_so_far, ARI(best_chrom_so_far)) tai moi the
      he duoc log. best_fitness don dieu KHONG TANG (elitism). Neu ARI khong
      tang cung -- Spearman rho gan 0 hoac duong -- day la bang chung TRUC TIEP
      rang "toi uu fitness tot hon" KHONG dan den "ARI tot hon tren du lieu nay",
      tuc loi nam o ham muc tieu, khong phai o optimizer (dung luan diem Sec1).
  (B) "Population" -- gop (fitness, ARI) cua TOAN BO quan the tai moi moc log --
      pham vi tin hieu rong hon (gom ca ca the ngau nhien lan ca the da toi uu).

Chay ban goc de tai lap bang cu:
    python huong_cai_tien/experiments/run_fitness_ari_correlation.py

So sanh ban goc va CDW tren bon bo du lieu:
    python huong_cai_tien/experiments/run_fitness_ari_correlation.py --models original cdw --data aggregation flame iris jain --gen 80 --runs 3

Vi du rut gon:
    python huong_cai_tien/experiments/run_fitness_ari_correlation.py --data flame iris --runs 3
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import _paths  # noqa: F401  (them src/ + models/ vao sys.path)

from sklearn.metrics import adjusted_rand_score
from scipy.stats import pearsonr, spearmanr

from nkcv2 import NKCV2Model
from nkcv2_cdw import CDWModel
from search import ls_fi
from mutations import mutation_reclassify, mutation_merge, mutation_split
from px import map_solutions, px, fix_labels
from datasets import DATASETS as ALL_DATASETS

DATASETS = dict(ALL_DATASETS)
CDW_KWARGS = dict(lam=1.0, branches=("diff",), w_min=0.01, w_max=5.0)
MODEL_NAMES = ("original", "cdw")


def _make_model(X, K, model_name):
    """Khoi tao dung objective; khong monkeypatch va khong sua src/."""
    if model_name == "original":
        return NKCV2Model(X, K=K)
    if model_name == "cdw":
        return CDWModel(X, K=K, **CDW_KWARGS)
    raise ValueError(f"model khong hop le: {model_name}")


# ---------------------------------------------------------------------------
# Ban sao cac building block cua NKHGA.fit() (ga.py) -- khong sua, chi import
# ---------------------------------------------------------------------------

def _immigrant(model, rng):
    k = int(rng.integers(2, max(3, int(np.sqrt(model.N))) + 1))
    x = rng.integers(1, k + 1, size=model.N).astype(np.int64)
    x = np.ascontiguousarray(x)
    f = model.comp_fitness(x)
    f = ls_fi(model, x, f, rng)
    return x, f


def _selection(fitness, rng, tournament_size):
    pool = rng.integers(0, len(fitness), size=tournament_size)
    return int(pool[np.argmin(fitness[pool])])


def _mutation(model, parent, fit_parent, rng):
    r = rng.random()
    if r < 0.6:
        return mutation_reclassify(model, parent, fit_parent, rng)
    elif r < 0.8:
        child = mutation_merge(model, parent, rng)
        return child, model.comp_fitness(child)
    else:
        child = mutation_split(model, parent, rng)
        return child, model.comp_fitness(child)


def _crossover(model, p1, p2):
    p2_mapped = map_solutions(p1, p2)
    offspring, fit = px(model, p1, p2_mapped)
    fix_labels(model, offspring)
    return offspring, fit


def run_and_log(X, y_true, K, pop_size, p_cross, tournament_size, leng_local,
                imig_ls_ratio, max_gen, seed, log_every,
                model_name="original"):
    """Chay 1 lan GA (logic giong het NKHGA.fit()), ghi (fitness, ARI) moi
    `log_every` the he. Tra ve (traj, pop_log, best_fit, best_ari)."""
    rng = np.random.default_rng(seed)
    model = _make_model(X, K, model_name)

    chroms = [None] * pop_size
    fits = np.empty(pop_size, dtype=np.float64)
    for i in range(pop_size):
        chroms[i], fits[i] = _immigrant(model, rng)

    best_idx = int(np.argmin(fits))
    best_chrom = chroms[best_idx].copy()
    best_fit = float(fits[best_idx])

    traj = []      # (gen, best_fit, best_ari)
    pop_log = []   # (fit, ari) cho toan bo quan the tai cac moc log

    def log_population():
        for i in range(pop_size):
            ari = adjusted_rand_score(y_true, chroms[i])
            pop_log.append((float(fits[i]), float(ari)))

    def log_traj(gen):
        ari_best = adjusted_rand_score(y_true, best_chrom)
        traj.append((gen, best_fit, float(ari_best)))

    log_traj(0)
    log_population()

    gen = 0
    gen_init = 0
    while gen < max_gen:
        gen += 1

        if (gen - gen_init) > leng_local:
            gen_init = gen
            chroms[0] = best_chrom.copy()
            fits[0] = best_fit
            n_ls = int(imig_ls_ratio * pop_size)
            for i in range(n_ls):
                fits[i] = ls_fi(model, chroms[i], fits[i], rng)
            for i in range(n_ls, pop_size):
                chroms[i], fits[i] = _immigrant(model, rng)

        cur_best = int(np.argmin(fits))
        new_chroms = [None] * pop_size
        new_fits = np.empty(pop_size, dtype=np.float64)
        for j in range(pop_size - 1):
            if rng.random() > p_cross:
                p = _selection(fits, rng, tournament_size)
                child, cf = _mutation(model, chroms[p], fits[p], rng)
            else:
                p1 = _selection(fits, rng, tournament_size)
                p2 = _selection(fits, rng, tournament_size)
                child, cf = _crossover(model, chroms[p1], chroms[p2])
            new_chroms[j] = child
            new_fits[j] = cf
        new_chroms[pop_size - 1] = chroms[cur_best].copy()
        new_fits[pop_size - 1] = fits[cur_best]

        chroms, fits = new_chroms, new_fits

        gb = int(np.argmin(fits))
        if fits[gb] < best_fit:
            best_fit = float(fits[gb])
            best_chrom = chroms[gb].copy()

        if gen % log_every == 0 or gen == max_gen:
            log_traj(gen)
            log_population()

    return traj, pop_log, best_fit, adjusted_rand_score(y_true, best_chrom)


# ---------------------------------------------------------------------------
# Tuong quan + bao cao
# ---------------------------------------------------------------------------

def _corr(pairs, x_idx, y_idx):
    xs = np.array([p[x_idx] for p in pairs], dtype=np.float64)
    ys = np.array([p[y_idx] for p in pairs], dtype=np.float64)
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    pr, pp = pearsonr(xs, ys)
    sr, sp = spearmanr(xs, ys)
    return float(pr), float(pp), float(sr), float(sp)


def evaluate(name, X, y_true, model_name, args):
    print(f"\n########## {name.upper()} / {model_name.upper()} "
         f"(N={X.shape[0]}, cum that={len(np.unique(y_true))}) ##########")

    all_traj = []
    all_pop = []
    for run_idx in range(args.runs):
        seed = args.seed + run_idx
        t0 = time.time()
        traj, pop_log, bf, bari = run_and_log(
            X, y_true, args.K, args.pop, args.p_cross, args.tournament,
            args.leng_local, args.imig_ls_ratio, args.gen, seed, args.log_every,
            model_name=model_name)
        print(f"  seed={seed}: best_fitness={bf:.6f}, ARI(best)={bari:.4f}, "
             f"{time.time()-t0:.1f}s")
        all_traj.extend(traj)
        all_pop.extend(pop_log)

    traj_pairs = [(t[1], t[2]) for t in all_traj]  # (best_fit, best_ari)
    pr_t, pp_t, sr_t, sp_t = _corr(traj_pairs, 0, 1)
    pr_p, pp_p, sr_p, sp_p = _corr(all_pop, 0, 1)

    print(f"\n  (A) Trajectory (best-so-far moi the he, n={len(traj_pairs)}):")
    print(f"      Pearson  r={pr_t:+.3f} (p={pp_t:.2e})")
    print(f"      Spearman rho={sr_t:+.3f} (p={sp_t:.2e})")
    print(f"  (B) Population (toan bo ca the moi moc log, n={len(all_pop)}):")
    print(f"      Pearson  r={pr_p:+.3f} (p={pp_p:.2e})")
    print(f"      Spearman rho={sr_p:+.3f} (p={sp_p:.2e})")

    return dict(name=name, model=model_name,
               pearson_traj=pr_t, spearman_traj=sr_t,
               pearson_pop=pr_p, spearman_pop=sr_p,
               n_traj=len(traj_pairs), n_pop=len(all_pop))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gen", type=int, default=150)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--p-cross", type=float, default=0.6)
    ap.add_argument("--tournament", type=int, default=3)
    ap.add_argument("--leng-local", type=int, default=100)
    ap.add_argument("--imig-ls-ratio", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--data", nargs="+", default=["aggregation", "flame", "iris"],
                    choices=list(DATASETS))
    ap.add_argument("--models", nargs="+", default=["original"],
                    choices=MODEL_NAMES,
                    help="objective can do: original, cdw")
    args = ap.parse_args()

    results = []
    for name in args.data:
        X, y_true = DATASETS[name]()
        for model_name in args.models:
            results.append(evaluate(name, X, y_true, model_name, args))

    print("\n\n=== TOM TAT: tuong quan fitness (cang NHO cang tot) "
         "vs ARI (cang LON cang tot) ===")
    print("Ky vong neu ham muc tieu tot: tuong quan AM MANH "
         "(fitness giam <=> ARI tang).")
    print(f"{'bo du lieu':<14}{'model':<10}{'Pearson(A)':>12}{'Spearman(A)':>13}"
         f"{'Pearson(B)':>12}{'Spearman(B)':>13}")
    for r in results:
        print(f"{r['name']:<14}{r['model']:<10}"
             f"{r['pearson_traj']:>+12.3f}{r['spearman_traj']:>+13.3f}"
             f"{r['pearson_pop']:>+12.3f}{r['spearman_pop']:>+13.3f}")


if __name__ == "__main__":
    main()
