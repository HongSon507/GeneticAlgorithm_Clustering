"""
Cac bo du lieu cua nhanh cai tien -- MOT cho duy nhat.

Moi file trong data/ co cung quy uoc: cac cot dau la dac trung, cot cuoi la
nhan that 1-based.

Tien xu ly cua phien ban hien hanh (`preprocess`): bo cac cot gan nhu hang so
roi MinMax. Ly do va so lieu: DE_XUAT_FITNESS_CHONG_LAN.md muc 7d.3 va 7e.
`src/run_comparison.py` (ban tai hien bai bao) van dung StandardScaler va khong
bi dong toi; cac script trong lich_su/ dung `lich_su/models/datasets_std.py`
de giu dung StandardScaler nhu luc tao ra so lieu cua chung.

    from datasets import DATASETS, load_raw, preprocess
    X, y = DATASETS["flame"]()          # da tien xu ly
    X_raw, y = load_raw("ecoli")        # du lieu goc
"""

from __future__ import annotations

import os

import numpy as np

from _bootstrap import ROOT  # goc repo, tinh tu vi tri cua models/_bootstrap.py

DATA_DIR = os.path.join(ROOT, "data")

# (ten file, so dac trung) -- cot cuoi cung luon la nhan that
_DATA_SPECS = {
    "aggregation": ("Aggregation.txt", 2),  # 788 diem, 7 cum tach roi
    "flame": ("flame.txt", 2),              # 240 diem, 2 cum dinh nhau qua eo hep
    "iris": ("iris.txt", 4),                # 150 diem, 3 lop, du lieu tu sklearn
    "jain": ("jain.txt", 2),                # 373 diem, 2 vong cung dai dinh nhau
    # Nguon: bo benchmark cua Gagolewski (github.com/gagolews/clustering-data-v1),
    # ban sao cua UCI va SIPU (cs.uef.fi/sipu/datasets); nhan lay tu labels0.
    "ecoli": ("ecoli.txt", 7),              # 336 diem, 8 lop rat lech (143..2)
    "glass": ("glass.txt", 9),              # 214 diem, 6 lop, chong lan manh
    "s1": ("s1.txt", 2),                    # 5000 diem, 15 cum Gauss, chong lan it nhat
    "s2": ("s2.txt", 2),
    "s3": ("s3.txt", 2),
    "s4": ("s4.txt", 2),                    # chong lan nhieu nhat
}

NEAR_CONSTANT = 0.95  # cot co gia tri pho bien nhat chiem >= 95% so dong


def load_raw(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Doc data/<file>: tra ve (X chua chuan hoa, nhan 1-based)."""
    filename, n_features = _DATA_SPECS[name]
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Khong thay {filename} tai {path}.")
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != n_features + 1:
        raise ValueError(
            f"{filename} phai co {n_features} cot dac trung + 1 cot nhan")
    labels = data[:, -1]
    if not np.allclose(labels, np.rint(labels)):
        raise ValueError(f"Nhan trong {filename} phai la so nguyen")
    return data[:, :-1], labels.astype(int)


def near_constant_columns(X, threshold=NEAR_CONSTANT):
    """Chi so cac cot gan nhu hang so (vd. `lip`, `chg` cua Ecoli)."""
    cols = []
    for j in range(X.shape[1]):
        _, counts = np.unique(X[:, j], return_counts=True)
        if counts.max() / X.shape[0] >= threshold:
            cols.append(j)
    return cols


def preprocess(X, method="minmax", drop_near_constant=True):
    """Chuan hoa dac trung truoc NK-HGA.

    Cot gan hang so bi loai truoc: sau StandardScaler mot diem hiem gap nam
    cach trung binh ~sqrt(N) do lech chuan va chi phoi khoang cach binh
    phuong (Ecoli: ARI 0.038 -> ~0.45). method: "minmax" hoac "standard".
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    X = np.asarray(X, dtype=np.float64)
    if drop_near_constant:
        drop = near_constant_columns(X)
        if drop and len(drop) < X.shape[1]:
            X = np.delete(X, drop, axis=1)
    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    return scaler.fit_transform(X)


def subsample(X, y, n, seed=0):
    """Lay ngau nhien n diem (dung de chay nhanh S1-S4); n=None giu nguyen."""
    if n is None or n >= len(y):
        return X, y
    idx = np.random.default_rng(seed).choice(len(y), n, replace=False)
    return X[idx], y[idx]


def load(name: str):
    X, y = load_raw(name)
    return preprocess(X), y


DATASETS = {name: (lambda n=name: load(n)) for name in _DATA_SPECS}


def n_clusters(labels: np.ndarray) -> int:
    """So cum khong ke nhieu (nhan 0)."""
    return len(np.unique(labels[labels > 0]))


def n_noise(labels: np.ndarray) -> int:
    return int(np.sum(labels == 0))


__all__ = ["DATASETS", "DATA_DIR", "load", "load_raw", "near_constant_columns",
           "preprocess", "subsample", "n_clusters", "n_noise"]
