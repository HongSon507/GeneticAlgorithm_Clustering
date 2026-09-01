"""Chuan bi moi truong chay cho moi module/script trong `huong_cai_tien/`.

Truoc day 10 file lap lai cung mot khoi 13 dong: ep stdout/stderr ve UTF-8,
roi chen `src/` vao `sys.path`. Gop lai mot cho -- sua duong dan mot lan la
xong, thay vi 10 lan.

Sau khi tach thu muc, file nay nam trong `models/` va chen ba duong dan:
`src/` (thuat toan goc), `models/` (cac model cai tien) va `experiments/`
(vai script import cheo nhau, vi du `CDW_KWARGS` cua `run_cdw_ga.py`).

Dung:
    import _bootstrap                 # chi can hieu ung phu (sys.path)
    from _bootstrap import RESULTS    # khi can duong dan ghi ket qua

Module trong `models/` import truc tiep duoc (cung thu muc). Script trong
`experiments/` phai chen `models/` vao `sys.path` truoc -- xem khoi 3 dong o
dau moi file trong do.
"""

from __future__ import annotations

import os
import sys

MODELS = os.path.dirname(os.path.abspath(__file__))
BRANCH = os.path.dirname(MODELS)            # huong_cai_tien/
EXPERIMENTS = os.path.join(BRANCH, "experiments")
ROOT = os.path.dirname(BRANCH)              # goc repo
SRC = os.path.join(ROOT, "src")
RESULTS = os.path.join(BRANCH, "results")

for _p in (SRC, MODELS, EXPERIMENTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass
