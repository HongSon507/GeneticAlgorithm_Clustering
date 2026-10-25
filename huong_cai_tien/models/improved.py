"""
NK-HGA cai tien -- phien ban hien hanh cua nhanh huong_cai_tien.

So voi ban tai hien bai bao trong src/ (KHONG bi sua):
  * fitness: NKCV-M (nkcv2_mode.py) -- NKCV2 voi dinh nghia cum theo dinh mat
    do co y nghia (z > 2); `fitness="nkcv2"` de dung NKCV2 goc;
  * tim kiem: gieo 30% quan the ban dau va immigrant tu GMM / k-means / Density
    Peaks voi k = 2..k_max(N), khong dung nhan that (seeded_ga.py);
  * hau xu ly: gop manh vun (fragment_merge.py) de keo so cum ve dung;
  * toc do: map_solutions O(N log N) va fix_labels numba, dau ra trung tung
    phan tu voi src/px.py (fast_px.py);
  * tien xu ly: `datasets.preprocess` -- bo cot gan hang so roi MinMax.

Dung:
    from improved import ImprovedNKHGA, preprocess
    est = ImprovedNKHGA(random_state=0).fit(preprocess(X_raw))
    est.labels_                  # sau khi gop manh vun
    est.labels_before_merge_     # nhan GA truoc khi gop

Ly do chon tung thanh phan, va vi sao CDW khong nam trong ban chinh (no thang
du lieu chong lan nhung xe cum thua tren mo hinh Gauss co nhieu cua bai bao):
../DE_XUAT_FITNESS_CHONG_LAN.md muc 7d-7h.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (them src/ vao sys.path)

import numpy as np

import fast_px
import ga
from datasets import preprocess  # noqa: F401  (tai xuat cho nguoi dung)
from fragment_merge import merge_fragments
from ga import renumber
from nkcv2 import NKCV2Model
from nkcv2_mode import ModeNKCV2Model
from seeded_ga import SeededNKHGA, build_seed_pool


class ImprovedNKHGA(SeededNKHGA):
    """NKHGA (src/ga.py) + fitness NKCV-M + gieo quan the + PX nhanh + gop manh vun.

    Kho loi giai gieo duoc dung trong `fit` neu khong truyen `pool`.
    Mac dinh pop_size=60, max_gen=80: giao thuc dung trong muc 7b-7h.
    """

    def __init__(self, pool=None, seed_frac=0.3, pop_size=60, max_gen=80,
                 fitness="mode", merge=True, **kw):
        super().__init__(pool=pool, seed_frac=seed_frac, pop_size=pop_size,
                         max_gen=max_gen, **kw)
        self.fitness = fitness
        self.merge = merge

    def _make_model(self, X):
        if self.fitness == "mode":
            return ModeNKCV2Model(X, K=self.K)
        return NKCV2Model(X, K=self.K)

    def fit(self, X):
        model = self._make_model(np.ascontiguousarray(X, dtype=np.float64))
        self.model_ = model
        original = ga.NKCV2Model
        ga.NKCV2Model = lambda Xx, K=3: model  # dung chung model voi buoc gop
        fast_px.install()
        try:
            if not self.pool:
                self.pool = build_seed_pool(X)
            super().fit(X)
        finally:
            fast_px.uninstall()
            ga.NKCV2Model = original
        self.labels_before_merge_ = self.labels_
        if self.merge:
            self.labels_ = renumber(merge_fragments(self.labels_, model))
            self.n_clusters_ = len(np.unique(self.labels_[self.labels_ > 0]))
        return self
