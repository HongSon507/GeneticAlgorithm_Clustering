"""
Cac bo du lieu dung trong nhanh cai tien -- MOT cho duy nhat.

Truoc refactor: `flame_data.py` va `jain_data.py` la hai file gan nhu y het
nhau (chi khac ten file), con loader Iris bi chep 4 lan trong
`run_cdw_screen.py`, `nkcv2_mahalanobis.py`, `test_px_correctness.py`,
`run_ablation.py`. Bon file .txt deu co cung quy uoc: cac cot dau la dac trung,
cot cuoi la nhan that. Iris duoc luu tai data/ thay vi phu thuoc vao ban du lieu
dong goi trong scikit-learn, nen du an co the sao chep va chay doc lap.

Tien xu ly giu nguyen quy uoc cua repo (`src/run_comparison.py`): 2 cot dau la
toa do, chuan hoa bang StandardScaler; cot cuoi la nhan that (1-based).

    from datasets import load_flame, load_iris_data, DATASETS
    X, y = DATASETS["flame"]()
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.preprocessing import StandardScaler

from _bootstrap import ROOT  # goc repo, tinh tu vi tri cua models/_bootstrap.py

DATA_DIR = os.path.join(ROOT, "data")

# (ten file, so dac trung) -- cot cuoi cung luon la nhan that
_DATA_SPECS = {
    "aggregation": ("Aggregation.txt", 2),  # 788 diem, 7 cum tach roi
    "flame": ("flame.txt", 2),              # 240 diem, 2 cum dinh nhau qua eo hep
    "iris": ("iris.txt", 4),                # 150 diem, 3 lop, du lieu tu sklearn
    "jain": ("jain.txt", 2),                # 373 diem, 2 vong cung dai dinh nhau
}


def _load_table(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Doc data/: cac cot dau la X, cot cuoi la nhan 1-based."""
    filename, n_features = _DATA_SPECS[name]
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Khong thay {filename} tai {path}.")
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != n_features + 1:
        raise ValueError(
            f"{filename} phai co {n_features} cot dac trung + 1 cot nhan")
    labels = data[:, -1]
    if not np.allclose(labels, np.rint(labels)):
        raise ValueError(f"Nhan trong {filename} phai la so nguyen")
    return StandardScaler().fit_transform(data[:, :-1]), labels.astype(int)


def load_aggregation():
    return _load_table("aggregation")


def load_flame():
    return _load_table("flame")


def load_jain():
    return _load_table("jain")


def load_iris_data():
    """Iris 4 chieu tu data/iris.txt, nhan da la 1-based."""
    return _load_table("iris")


DATASETS = {
    "aggregation": load_aggregation,
    "flame": load_flame,
    "iris": load_iris_data,
    "jain": load_jain,
}


def n_clusters(labels: np.ndarray) -> int:
    """So cum khong ke nhieu (nhan 0)."""
    return len(np.unique(labels[labels > 0]))


def n_noise(labels: np.ndarray) -> int:
    return int(np.sum(labels == 0))


__all__ = ["DATASETS", "DATA_DIR", "load_aggregation", "load_flame",
           "load_jain", "load_iris_data", "n_clusters", "n_noise"]
