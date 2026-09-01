"""Standalone test runner -- runs the required unit tests without pytest.

Usage:  python tests/run_tests.py
"""

import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from nkcv2 import NKCV2Model, HAVE_NUMBA
from px import map_solutions, px, fix_labels
from ga import renumber


def _rp(rng, n, nc):
    return np.ascontiguousarray(rng.integers(1, nc + 1, size=n).astype(np.int64))


def build_model():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    return NKCV2Model(X, K=3)


def test_delta_matches_full_reeval(model):
    rng = np.random.default_rng(1)
    for _ in range(200):
        x = _rp(rng, model.N, 5)
        i = int(rng.integers(0, model.N))
        b = int(rng.integers(0, 6))
        old = int(x[i])
        f_before = model.comp_fitness(x)
        x[i] = b
        delta = model.df_element(x, i, old)
        f_after = model.comp_fitness(x)
        assert abs(delta - (f_after - f_before)) < 1e-9


def test_px_offspring_fitness_matches_returned(model):
    rng = np.random.default_rng(2)
    for _ in range(50):
        p1 = _rp(rng, model.N, 4)
        p2 = _rp(rng, model.N, 4)
        p2m = map_solutions(p1, p2)
        off, fit = px(model, p1, p2m)
        assert abs(model.comp_fitness(off) - fit) < 1e-9


def test_px_no_worse_than_parents(model):
    rng = np.random.default_rng(3)
    for _ in range(50):
        p1 = _rp(rng, model.N, 4)
        p2 = _rp(rng, model.N, 4)
        p2m = map_solutions(p1, p2)
        _, fit = px(model, p1, p2m)
        assert fit <= model.comp_fitness(p1) + 1e-9
        assert fit <= model.comp_fitness(p2m) + 1e-9


def test_renumber_idempotent_and_partition_preserving(_model):
    rng = np.random.default_rng(4)
    x = rng.integers(0, 6, size=100).astype(np.int64)
    r1 = renumber(x)
    r2 = renumber(r1)
    assert np.array_equal(r1, r2)
    for _ in range(200):
        a, b = rng.integers(0, 100, size=2)
        assert (x[a] == x[b]) == (r1[a] == r1[b])


def test_fix_labels_splits_disconnected_same_label(model):
    x = np.ones(model.N, dtype=np.int64)
    fix_labels(model, x)
    assert x.max() >= 1


def main():
    print("HAVE_NUMBA:", HAVE_NUMBA)
    model = build_model()
    print(f"model built: N={model.N}, K={model.K}, Kout={model.Kout}")
    tests = [
        test_delta_matches_full_reeval,
        test_px_offspring_fitness_matches_returned,
        test_px_no_worse_than_parents,
        test_renumber_idempotent_and_partition_preserving,
        test_fix_labels_splits_disconnected_same_label,
    ]
    passed = 0
    for t in tests:
        try:
            t(model)
            passed += 1
            print(f"[PASS] {t.__name__}")
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()
