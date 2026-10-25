"""Chuan bi moi truong cho module/script trong `huong_cai_tien/lich_su/`.

`lich_su/` chua cac huong da thu nhung khong chon (CDW, Mahalanobis,
Persistence, NKCV-A, gop BIC, cac phep chan doan). Chung duoc giu de tai lap
so lieu trong README.md va DE_XUAT_FITNESS_CHONG_LAN.md, khong phai phien ban
hien hanh (xem `huong_cai_tien/models/improved.py`).

Chen vao sys.path, uu tien tu tren xuong: `lich_su/models/`,
`lich_su/experiments/`, `huong_cai_tien/models/` (seeded_ga, fast_px, ...) va
`src/`. Ket qua ghi vao `lich_su/results/`.

Script lich su doc du lieu qua `datasets_std` (StandardScaler nhu luc tao ra
so lieu), khong qua `datasets` cua phien ban hien hanh (MinMax).
"""

from __future__ import annotations

import os
import sys

MODELS = os.path.dirname(os.path.abspath(__file__))       # lich_su/models
HISTORY = os.path.dirname(MODELS)                         # lich_su/
EXPERIMENTS = os.path.join(HISTORY, "experiments")
RESULTS = os.path.join(HISTORY, "results")
BRANCH = os.path.dirname(HISTORY)                         # huong_cai_tien/
CURRENT_MODELS = os.path.join(BRANCH, "models")
ROOT = os.path.dirname(BRANCH)                            # goc repo
SRC = os.path.join(ROOT, "src")

for _p in (SRC, CURRENT_MODELS, EXPERIMENTS, MODELS):     # MODELS cuoi = uu tien nhat
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass
