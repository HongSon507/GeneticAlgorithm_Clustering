"""
Ban nhanh cua `map_solutions` va `fix_labels` (src/px.py) -- nut that thuc su
cua NK-HGA khi N lon. Dau ra TRUNG TUNG PHAN TU voi ban Python (xem self_test).

Profile tren S1 (N=1000, pop 20, 3 the he): 89.6 / 90.7 giay nam trong
`map_solutions`; sau khi sua, `fix_labels` (union-find bang Python) chiem ~1/2.

map_solutions -- ban goc: lap nc_min lan "tim o (i, j) con trong co overlap lon
  nhat, pha hoa theo thu tu hang", O(nc_min * nc_blue * nc_red) bang Python.
  Ban nay: dem overlap thua tren cac cap nhan xuat hien that (nnz <= N), sap xep
  mot lan theo (-overlap, i, j) roi duyet, bo qua o co hang/cot da dung. Day
  dung la day lua chon cua ban goc vi o con trong co overlap lon nhat (pha hoa
  (i, j) nho nhat) luon la o dau tien con trong trong danh sach da sap. Khi het
  o duong, ban goc ghep cac o overlap 0 theo thu tu hang trong nho nhat voi cot
  trong nho nhat -- lam y het. Do phuc tap O(N log N).

fix_labels -- giu nguyen thu tu union va luat rank cua _DSU nen goc moi thanh
  phan, thu tu `np.unique` cac goc va thu tu cap nhan moi khong doi; chi chuyen
  vong lap sang numba.

Dung: `install()` thay `ga.map_solutions` va `ga.fix_labels`; `uninstall()`
hoan nguyen.
"""

from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401  (them src/ vao sys.path)

from nkcv2 import njit
import px as _px


# ---------------------------------------------------------------------------
# map_solutions
# ---------------------------------------------------------------------------

@njit(cache=True)
def _greedy_match(rows, cols, nc_blue, nc_red, nc_min):
    """rows/cols: cap (blue, red) da sap theo (-overlap, blue, red)."""
    lin_free = np.ones(nc_blue + 1, dtype=np.bool_)
    col_free = np.ones(nc_red + 1, dtype=np.bool_)
    map_br = np.zeros(nc_red + 1, dtype=np.int64)
    matched = 0
    for t in range(rows.shape[0]):
        if matched == nc_min:
            break
        i, j = rows[t], cols[t]
        if lin_free[i] and col_free[j]:
            map_br[j] = i
            lin_free[i] = False
            col_free[j] = False
            matched += 1
    # het o duong: ghep hang trong nho nhat voi cot trong nho nhat
    i = 1
    j = 1
    while matched < nc_min:
        while not lin_free[i]:
            i += 1
        while not col_free[j]:
            j += 1
        map_br[j] = i
        lin_free[i] = False
        col_free[j] = False
        matched += 1
    return map_br


@njit(cache=True)
def _assign_missing_and_relabel(solution_red, map_br, nc_blue, nc_red):
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
    red_new = solution_red.copy()
    for k in range(solution_red.shape[0]):
        r = solution_red[k]
        red_new[k] = 0 if r == 0 else map_br[r]
    return red_new


def map_solutions(solution_blue, solution_red):
    blue = np.ascontiguousarray(solution_blue, dtype=np.int64)
    red = np.ascontiguousarray(solution_red, dtype=np.int64)
    nc_blue, nc_red = int(blue.max()), int(red.max())
    if nc_blue == 0 or nc_red == 0:
        return solution_red.copy()
    nc_min = min(nc_blue, nc_red)
    mask = (blue > 0) & (red > 0)
    key, cnt = np.unique(blue[mask] * (nc_red + 1) + red[mask], return_counts=True)
    rows, cols = key // (nc_red + 1), key % (nc_red + 1)
    order = np.lexsort((cols, rows, -cnt))  # khoa chinh: -cnt, roi hang, roi cot
    map_br = _greedy_match(np.ascontiguousarray(rows[order]),
                           np.ascontiguousarray(cols[order]),
                           nc_blue, nc_red, nc_min)
    out = _assign_missing_and_relabel(red, map_br, nc_blue, nc_red)
    return out.astype(solution_red.dtype, copy=False)


# ---------------------------------------------------------------------------
# fix_labels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _find(parent, x):
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        nxt = parent[x]
        parent[x] = root
        x = nxt
    return root


@njit(cache=True)
def _fix_labels_nb(offspring, rev_flat, rev_off, rev_cnt, N):
    label_max = 0
    for i in range(N):
        if offspring[i] > label_max:
            label_max = offspring[i]
    if label_max <= 1:
        return
    parent = np.arange(N)
    rank = np.zeros(N, dtype=np.int64)
    for i in range(N):
        start = rev_off[i]
        for r in range(rev_cnt[i]):
            k = rev_flat[start + r]
            if offspring[i] == offspring[k]:
                ra = _find(parent, i)
                rb = _find(parent, k)
                if ra == rb:
                    continue
                if rank[ra] < rank[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                if rank[ra] == rank[rb]:
                    rank[ra] += 1
    # thanh phan duoc danh so theo thu tu tang dan cua goc (nhu np.unique)
    comp_of_root = np.full(N, -1, dtype=np.int64)
    roots = np.empty(N, dtype=np.int64)
    for i in range(N):
        roots[i] = _find(parent, i)
    n_comp = 0
    for v in range(N):
        if roots[v] == v:
            comp_of_root[v] = n_comp
            n_comp += 1
    comp_label = np.zeros(n_comp, dtype=np.int64)
    for i in range(N):
        comp_label[comp_of_root[roots[i]]] = offspring[i]
    seen = np.zeros(label_max + 1, dtype=np.bool_)
    new_label = label_max + 1
    remap = comp_label.copy()
    for c in range(n_comp):
        lab = comp_label[c]
        if lab <= 0:
            continue
        if not seen[lab]:
            seen[lab] = True
        else:
            remap[c] = new_label
            new_label += 1
    for i in range(N):
        offspring[i] = remap[comp_of_root[roots[i]]]


def fix_labels(model, offspring):
    _fix_labels_nb(offspring, model.rev_flat, model.rev_off, model.rev_cnt,
                   model.N)
    return offspring


# ---------------------------------------------------------------------------

_python_map_solutions = _px.map_solutions
_python_fix_labels = _px.fix_labels


def install():
    import ga
    ga.map_solutions = map_solutions
    ga.fix_labels = fix_labels


def uninstall():
    import ga
    ga.map_solutions = _python_map_solutions
    ga.fix_labels = _python_fix_labels


def self_test(trials=400, seed=0):
    """Dau ra trung tung phan tu voi src/px.py, ca truong hop co nhieu va hoa."""
    from nkcv2 import NKCV2Model
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        n = int(rng.integers(5, 400))
        kb, kr = int(rng.integers(1, 60)), int(rng.integers(1, 60))
        b = rng.integers(0, kb + 1, size=n).astype(np.int64)
        r = rng.integers(0, kr + 1, size=n).astype(np.int64)
        assert np.array_equal(map_solutions(b, r), _python_map_solutions(b, r))
    X = rng.normal(size=(300, 2))
    model = NKCV2Model(X, K=3)
    for _ in range(trials // 4):
        k = int(rng.integers(1, 40))
        x = rng.integers(0, k + 1, size=300).astype(np.int64)
        a, b = x.copy(), x.copy()
        _python_fix_labels(model, a)
        fix_labels(model, b)
        assert np.array_equal(a, b)
    return True


if __name__ == "__main__":
    print("map_solutions + fix_labels nhanh == Python:", self_test())
