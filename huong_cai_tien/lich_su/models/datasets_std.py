"""Bo nap du lieu voi tien xu ly CU (StandardScaler) cho cac script lich su.

Moi so lieu trong lich_su/ (README.md muc 4-13, DE_XUAT muc 7b-7d) duoc tao ra
voi StandardScaler tren toan bo cot. Phien ban hien hanh doi sang
`datasets.preprocess` (bo cot gan hang so roi MinMax); file nay giu cach cu de
cac so lieu do van tai lap duoc.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np
from sklearn.preprocessing import StandardScaler

from datasets import (DATA_DIR, _DATA_SPECS, load_raw, n_clusters, n_noise,
                      subsample)


def _load_std(name):
    X, y = load_raw(name)
    return StandardScaler().fit_transform(X), y


def load_aggregation():
    return _load_std("aggregation")


def load_flame():
    return _load_std("flame")


def load_jain():
    return _load_std("jain")


def load_iris_data():
    return _load_std("iris")


DATASETS = {name: (lambda n=name: _load_std(n)) for name in _DATA_SPECS}

__all__ = ["DATASETS", "DATA_DIR", "load_aggregation", "load_flame", "load_jain",
           "load_iris_data", "n_clusters", "n_noise", "subsample"]
