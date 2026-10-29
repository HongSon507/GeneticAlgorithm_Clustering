"""
Khoi tao da cau truc va toan tu tien hoa noi tai cho NK-HGA.

1. Khởi tạo nội tại (Internal Multi-Structure Immigrants, 100% thuần GA):
   - init_density_tree: cắt ngẫu nhiên trên rừng cây liên kết mật độ `dlink = M[:, 0]`;
   - init_voronoi_internal: phân vùng Voronoi ngẫu nhiên trên ma trận khoảng cách D;
   - immigrant ngẫu nhiên gốc: gán nhãn ngẫu nhiên + ls_fi.
   Cả ba cơ chế đều chỉ dùng cấu trúc nội tại của NKCV2Model (D, rho, M), không gọi
   bất kỳ thư viện phân cụm ngoại lai nào (K-means, GMM), không cố định k mục tiêu.

2. Khởi tạo ngoại lai (Tùy chọn tương thích ngược):
   - build_seed_pool: sinh nghiệm từ GMM / k-means / DPC để làm mốc đối chiếu ablation.

3. Tích hợp toán tử đồ thị Gep và chu kỳ Memetic thích nghi (leng_local).
"""

from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401  (them src/ vao sys.path)

from ga import NKHGA
from search import ls_fi
from nkcv2 import _squared_distance_matrix, _cutoff_distance, _density
from ga_operators import mutation_merge_graph, mutation_split_graph, ls_fi_with_noise


def default_k_max(N):
    return int(min(20, max(6, np.sqrt(N) / 2)))


# ---------------------------------------------------------------------------
# 1. Khoi tao noi tai tren do thi Gep va ma tran D (Thuan GA)
# ---------------------------------------------------------------------------

def init_density_tree(model, rng, m=None):
    """Khoi tao cat rung cay lien ket mat do dlink = M[:, 0]."""
    N = model.N
    if m is None:
        m = int(rng.integers(2, max(3, int(np.sqrt(N))) + 1))

    dlink = model.M[:, 0]
    rho = model.rho

    # Tinh trong so chon goc: uu tien diem dac va khoang cach toi dlink
    D_dlink = np.array([model.D[i, dlink[i]] for i in range(N)])
    w = rho * (1.0 + D_dlink / (D_dlink.mean() + 1e-12))
    w_sum = w.sum()
    if w_sum > 0:
        p = w / w_sum
    else:
        p = np.full(N, 1.0 / N)

    # Chon m hat giong lam goc
    roots = rng.choice(N, size=min(m, N), replace=False, p=p)
    labels = np.zeros(N, dtype=np.int64)
    for idx, r in enumerate(roots):
        labels[r] = idx + 1

    # Lan truyen nhan theo chieu giam dan cua mat do
    order = np.argsort(-rho, kind="stable")
    for i in order:
        if labels[i] == 0:
            parent = dlink[i]
            if labels[parent] > 0:
                labels[i] = labels[parent]
            else:
                labels[i] = int(rng.integers(1, m + 1))

    return np.ascontiguousarray(labels)


def init_voronoi_internal(model, rng, m=None):
    """Khoi tao phan vung Voronoi ngau nhien truc tiep tu ma tran khoang cach D."""
    N = model.N
    if m is None:
        m = int(rng.integers(2, max(3, int(np.sqrt(N))) + 1))

    # Chon hat giong dau tien ngau nhien
    seeds = [int(rng.integers(N))]
    D = model.D

    # Chon cac hat giong tiep theo theo phuong phap D^2 sampling
    min_dist = D[seeds[0]].copy()
    for _ in range(1, m):
        p = min_dist / max(min_dist.sum(), 1e-12)
        nxt = int(rng.choice(N, p=p))
        seeds.append(nxt)
        min_dist = np.minimum(min_dist, D[nxt])

    # Gan moi diem ve hat giong gan nhat
    labels = np.argmin(D[:, seeds], axis=1) + 1
    return np.ascontiguousarray(labels, dtype=np.int64)


def internal_immigrant(model, rng, test_noise=True):
    """Tao mot ca the moi hoan toan bang phuong phap noi tai cua GA."""
    r = rng.random()
    if r < 0.45:
        x = init_density_tree(model, rng)
    elif r < 0.90:
        x = init_voronoi_internal(model, rng)
    else:
        # Ngau nhien trang nhu ban goc
        k = int(rng.integers(2, max(3, int(np.sqrt(model.N))) + 1))
        x = rng.integers(1, k + 1, size=model.N).astype(np.int64)

    f = model.comp_fitness(x)
    if test_noise:
        f = ls_fi_with_noise(model, x, f, rng)
    else:
        f = ls_fi(model, x, f, rng)
    return x, f


# ---------------------------------------------------------------------------
# 2. Khoi tao ngoai lai (K-means, GMM, DPC) lam moc doi chieu
# ---------------------------------------------------------------------------

def dpc_labels(X, k, D=None, rho=None):
    """Density Peaks cho kho loi giai gieo doi chieu."""
    if D is None:
        D = _squared_distance_matrix(np.ascontiguousarray(X, dtype=np.float64))
    N = D.shape[0]
    if rho is None:
        rho = _density(D, _cutoff_distance(D, N), N)
    order = np.argsort(-rho, kind="stable")
    delta = np.empty(N)
    parent = np.full(N, -1)
    delta[order[0]] = D[order[0]].max()
    for r in range(1, N):
        i = order[r]
        higher = order[:r]
        j = higher[np.argmin(D[i, higher])]
        parent[i], delta[i] = j, D[i, j]
    centers = np.argsort(-(rho * delta))[:k]
    labels = np.zeros(N, dtype=np.int64)
    labels[centers] = np.arange(1, k + 1)
    if labels[order[0]] == 0:
        labels[centers[-1]] = 0
        labels[order[0]] = k
    for i in order:
        if labels[i] == 0:
            labels[i] = labels[parent[i]]
    return labels


def build_seed_pool(X, k_max=None, sources=("gmm", "km", "dpc"), seed=0):
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    N = X.shape[0]
    k_max = k_max or default_k_max(N)
    pool = []
    D = rho = None
    if "dpc" in sources:
        D = _squared_distance_matrix(np.ascontiguousarray(X, dtype=np.float64))
        rho = _density(D, _cutoff_distance(D, N), N)
    for k in range(2, k_max + 1):
        if "gmm" in sources:
            pool.append(GaussianMixture(k, covariance_type="full", reg_covar=1e-4,
                                        random_state=seed + k).fit_predict(X) + 1)
        if "km" in sources:
            pool.append(KMeans(k, n_init=3, random_state=seed + k).fit_predict(X) + 1)
        if "dpc" in sources:
            pool.append(dpc_labels(X, k, D=D, rho=rho))
    return [np.ascontiguousarray(p, dtype=np.int64) for p in pool]


# ---------------------------------------------------------------------------
# 3. Estimator NK-HGA nang cao
# ---------------------------------------------------------------------------

class SeededNKHGA(NKHGA):
    """NKHGA ho tro khoi tao noi tai Gep (mac dinh) hoac kho nghiem gieo pool,
    dong thoi tich hop toan tu dot bien Gep va chu ky Memetic thich nghi."""

    def __init__(self, pool=None, seed_frac=0.3, init_source="internal",
                 use_graph_mutations=True, test_noise=True, **kw):
        super().__init__(**kw)
        self.pool = pool or []
        self.seed_frac = float(seed_frac)
        self.init_source = init_source  # "internal", "pool", hoac "random"
        self.use_graph_mutations = use_graph_mutations
        self.test_noise = test_noise

        # Chu ky Memetic thich nghi theo max_gen
        # Dam bao pha lam moi va local search dinh ky duoc kich hoat
        if self.max_gen and self.max_gen > 20:
            self.leng_local = min(self.leng_local, max(15, self.max_gen // 3))

    def fit(self, X):
        n_init = min(len(self.pool), int(round(self.seed_frac * self.pop_size)))
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(len(self.pool))[:n_init] if n_init else []
        self._init_queue = [self.pool[i] for i in idx]
        return super().fit(X)

    def _seeded(self, model, x, rng):
        x = x.copy()
        f = model.comp_fitness(x)
        if self.test_noise:
            f = ls_fi_with_noise(model, x, f, rng)
        else:
            f = ls_fi(model, x, f, rng)
        return x, f

    def _immigrant(self, model, rng):
        if self.pool and self._init_queue:
            return self._seeded(model, self._init_queue.pop(0), rng)
        if self.pool and self.init_source == "pool" and rng.random() < self.seed_frac:
            return self._seeded(model, self.pool[int(rng.integers(len(self.pool)))], rng)
        if self.init_source == "internal":
            return internal_immigrant(model, rng, test_noise=self.test_noise)
        return super()._immigrant(model, rng)

    def _mutation(self, model, parent, fit_parent, rng):
        if not self.use_graph_mutations:
            return super()._mutation(model, parent, fit_parent, rng)

        r = rng.random()
        if r < 0.6:
            # Reclassify
            from mutations import mutation_reclassify
            return mutation_reclassify(model, parent, fit_parent, rng)
        elif r < 0.8:
            # Graph-aware Merge
            child = mutation_merge_graph(model, parent, rng)
            return child, model.comp_fitness(child)
        else:
            # Graph-aware Split
            child = mutation_split_graph(model, parent, rng)
            return child, model.comp_fitness(child)
