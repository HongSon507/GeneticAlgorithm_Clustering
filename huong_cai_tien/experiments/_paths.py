"""Cau noi phia `experiments/` sang `models/_bootstrap.py`.

Python chi tu dat thu muc CUA SCRIPT vao `sys.path[0]`, nen mot script trong
`experiments/` khong thay `models/`. File nay chen `models/` vao truoc roi kich
hoat `_bootstrap` that (no lo not `src/`, `models/`, `experiments/` va UTF-8).

Khong dat ten `_bootstrap.py` o day duoc: trung ten module voi ban trong
`models/`, ban nao vao `sys.modules` truoc se che ban kia.

Dung:
    import _paths                 # chi can hieu ung phu (sys.path)
    from _paths import RESULTS    # khi can duong dan ghi ket qua
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

from _bootstrap import (  # noqa: E402  (bat buoc sau khi da chen sys.path)
    BRANCH, EXPERIMENTS, MODELS, RESULTS, ROOT, SRC)

__all__ = ["BRANCH", "EXPERIMENTS", "MODELS", "RESULTS", "ROOT", "SRC"]
