"""
NK-HGA cai tien -- phien ban hien hanh cua nhanh huong_cai_tien.

So voi ban tai hien bai bao trong src/ (KHONG bi sua):
  * fitness: giu NKCV2 goc (src/nkcv2.py), khong them so hang nao;
  * tim kiem: gieo 30% quan the ban dau va immigrant tu GMM / k-means / Density
    Peaks voi k = 2..k_max(N), khong dung nhan that (seeded_ga.py);
  * toc do: map_solutions O(N log N) va fix_labels numba, dau ra trung tung
    phan tu voi src/px.py (fast_px.py);
  * tien xu ly: `datasets.preprocess` -- bo cot gan hang so roi MinMax.

Dung:
    from improved import ImprovedNKHGA, preprocess
    labels = ImprovedNKHGA(random_state=0).fit_predict(preprocess(X_raw))

Lich su cac huong da thu va vi sao khong chon nam o ../lich_su/ va
../DE_XUAT_FITNESS_CHONG_LAN.md muc 7b-7d.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (them src/ vao sys.path)

import fast_px
from datasets import preprocess  # noqa: F401  (tai xuat cho nguoi dung)
from seeded_ga import SeededNKHGA, build_seed_pool


class ImprovedNKHGA(SeededNKHGA):
    """NKHGA (src/ga.py) + gieo quan the + PX nhanh. Fitness la NKCV2 goc.

    Kho loi giai gieo duoc dung trong `fit` neu khong truyen `pool`.
    Mac dinh pop_size=60, max_gen=80: giao thuc dung trong toan bo muc 7b-7d.
    """

    def __init__(self, pool=None, seed_frac=0.3, pop_size=60, max_gen=80, **kw):
        super().__init__(pool=pool, seed_frac=seed_frac, pop_size=pop_size,
                         max_gen=max_gen, **kw)

    def fit(self, X):
        fast_px.install()
        try:
            if not self.pool:
                self.pool = build_seed_pool(X)
            return super().fit(X)
        finally:
            fast_px.uninstall()
