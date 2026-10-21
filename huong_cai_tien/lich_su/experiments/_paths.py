"""Cau noi phia `lich_su/experiments/` sang `lich_su/models/_bootstrap.py`.

Python chi tu dat thu muc CUA SCRIPT vao `sys.path[0]`; file nay chen
`lich_su/models/` roi kich hoat `_bootstrap` cua lich su (no lo `src/`,
`huong_cai_tien/models/`, `lich_su/experiments/` va UTF-8).

Dung:
    import _paths                 # chi can hieu ung phu (sys.path)
    from _paths import RESULTS    # lich_su/results
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

from _bootstrap import (  # noqa: E402  (bat buoc sau khi da chen sys.path)
    BRANCH, EXPERIMENTS, MODELS, RESULTS, ROOT, SRC)

__all__ = ["BRANCH", "EXPERIMENTS", "MODELS", "RESULTS", "ROOT", "SRC"]
