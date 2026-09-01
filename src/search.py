"""
Local search -- faithful port of the reference `nk::LsFi` (first improvement).

Scans variables in random order; for each, tries relabelling to the label of one
of the objects that influence f_i (its group M_i). Accepts improving moves and
drifts across neutral moves (delta == 0). Stops on a plateau streak > 3 or after
N/2 outer iterations. Uses delta-evaluation, so each trial costs O(Kout*K).
"""

from __future__ import annotations


def ls_fi(model, x, f, rng):
    """In-place first-improvement local search on partition x with fitness f.

    Returns the (possibly improved) fitness. x is modified in place.
    """
    N, K, M = model.N, model.K, model.M
    count_it = 0
    count_plato = 0
    cont = True

    while cont and count_it < N // 2:
        v_out = rng.permutation(N)
        improving = False

        for j in range(N):
            i = int(v_out[j])
            xi_old = int(x[i])
            for k in range(K):
                cand = int(x[M[i, k]])
                if cand == int(x[i]):
                    continue
                x[i] = cand
                df = model.df_element(x, i, xi_old)
                if df < 0.0:
                    improving = True
                    break  # keep improving move (first improvement)
                elif df > 0.0:
                    x[i] = xi_old  # revert worsening move
                else:
                    # neutral move: keep it and drift
                    xi_old = cand
            if improving:
                break

        if improving:
            count_plato = 0
        else:
            count_plato += 1
            if count_plato > 3:
                cont = False
        count_it += 1

    # Recompute once so the returned fitness exactly matches x.
    return model.comp_fitness(x)
