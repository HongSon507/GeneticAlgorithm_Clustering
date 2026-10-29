"""
NK-HGA cai tien -- phien ban noi tai (thuan GA) va cac cau hinh ablation.

Cai tien cot loi:
  * fitness: NKCV-M mem (nkcv2_mode.py) -- dinh mat do la tin hieu phu tro (w_max ~ 0.6),
    khong ep buoc moi dinh thanh mot cum;
  * khoi tao: noi tai 100% tren do thi Gep (cay dlink + Voronoi ma tran D, khong goi K-means/GMM);
  * toan tu GA: Graph-aware Merge/Split theo lien ket bien Gep va ho tro nhan nhieu 0;
  * chu ky Memetic: leng_local thich nghi theo max_gen;
  * toc do: fast_px (O(N log N) map_solutions va fix_labels numba, trung bit voi src/px.py);
  * hau xu ly (tuy chon): merge="evidence" chi gop khi co bang chung Delta f <= 0,
    bao ve cum nho that va giu nhan nhieu 0. Mac dinh co the tat (merge=False) de dung thuan GA.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (them src/ vao sys.path)

import numpy as np

import fast_px
import ga
from datasets import preprocess  # noqa: F401
from fragment_merge import merge_fragments, merge_fragments_evidence
from ga import renumber
from nkcv2 import NKCV2Model
from nkcv2_mode import ModeNKCV2Model
from seeded_ga import SeededNKHGA, build_seed_pool


class ImprovedNKHGA(SeededNKHGA):
    """NKHGA cai tien voi day du cac tru cot noi tai.

    Tham so:
      fitness: "mode" (mem, mac dinh), "mode_hard" (ban cu), "nkcv2" (goc)
      init_source: "internal" (thuan GA tren do thi Gep, mac dinh), "pool", "random"
      use_graph_mutations: True (toan tu merge/split theo Gep, mac dinh)
      merge: False (thuan GA khong hau xu ly), "evidence" (kiem chung Delta f <= 0),
             "legacy" (quy tac cu 2% N), True (trung "evidence")
      local_scale: True / False (chuan hoa thang khoang cach cuc bo)
    """

    def __init__(self, pool=None, seed_frac=0.3, pop_size=60, max_gen=80,
                 fitness="mode", merge="evidence", init_source="internal",
                 use_graph_mutations=True, test_noise=True, local_scale=False,
                 w_max=0.6, z0=1.5, z_full=4.0, **kw):
        super().__init__(pool=pool, seed_frac=seed_frac, pop_size=pop_size,
                         max_gen=max_gen, init_source=init_source,
                         use_graph_mutations=use_graph_mutations,
                         test_noise=test_noise, **kw)
        self.fitness = fitness
        self.merge = merge
        self.local_scale = local_scale
        self.w_max = w_max
        self.z0 = z0
        self.z_full = z_full

    def _make_model(self, X):
        if self.fitness == "mode":
            return ModeNKCV2Model(X, K=self.K, z0=self.z0, z_full=self.z_full,
                                  soft=True, w_max=self.w_max,
                                  local_scale=self.local_scale)
        elif self.fitness == "mode_hard":
            return ModeNKCV2Model(X, K=self.K, z0=self.z0, z_full=self.z_full,
                                  soft=False, w_max=1.0,
                                  local_scale=self.local_scale)
        return NKCV2Model(X, K=self.K)

    def fit(self, X):
        model = self._make_model(np.ascontiguousarray(X, dtype=np.float64))
        self.model_ = model
        original = ga.NKCV2Model
        ga.NKCV2Model = lambda Xx, K=3: model
        fast_px.install()
        try:
            if self.init_source == "pool" and not self.pool:
                self.pool = build_seed_pool(X)
            super().fit(X)
        finally:
            fast_px.uninstall()
            ga.NKCV2Model = original

        self.labels_before_merge_ = self.labels_

        if self.merge is True or self.merge == "evidence":
            self.labels_ = renumber(merge_fragments_evidence(self.labels_, model))
        elif self.merge == "legacy":
            self.labels_ = renumber(merge_fragments(self.labels_, model))

        self.n_clusters_ = len(np.unique(self.labels_[self.labels_ > 0]))
        return self
