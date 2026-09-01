"""
Cải tiến (Đề xuất 5): CỔNG PERSISTENCE cho nhánh phạt "khác nhãn mà ở gần".

Đường đi tới ý tưởng (các hướng đã bị bác bỏ bằng số liệu, mã đã xoá)
--------------------------------------------------------------------
1. K_i thích nghi theo mật độ: Aggregation tụt còn 0.81-0.88 (gốc 0.9876).
2. Ngưỡng cục bộ dt0_i: Aggregation 0.9097, Flame 0.0128 — dt0_i nhỏ ở vùng đặc
   làm nới phạt quá tay nên hai cụm bị gộp.
3. Đổi Gep theo "thung lũng dọc đoạn nối": vô ích — Gep gốc trên Flame đã sạch sẵn
   (chỉ 13/720 cạnh vượt biên thật), tức sửa nhầm chỗ.
4. Cổng theo thung lũng CỤC BỘ: chết ở cạnh NGẮN. Đo thực tế: 10 cạnh biên ngắn của
   Flame đều có valley = 0. Hai điểm kề nhau thì giữa chúng không có chỗ lõm nào để
   đo — tính hợp lệ của biên là thuộc tính TOÀN CỤC.
5. Mutual reachability (HDBSCAN/DBCV): không đảo được thứ hạng, kể cả bản bất đối
   xứng (compactness dùng D, separation dùng d_mr). Lý do: hai cụm Flame chạm nhau
   trong vùng ĐẶC nên core_k ở biên NHỎ, d_mr ~ D.
6. Minimax path trên MST của đồ thị mutual reachability (mượn thẳng DBCV, ref [11]):
   trượt điều kiện cần trên CẢ BA bộ, ở cả bản đối xứng lẫn bản bất đối xứng đúng
   cấu trúc DBCV. Cơ chế có chạy (diff-close% về 0.0%) nhưng vô ích: 46% cạnh vượt
   biên thật của Flame và 43-50% của Iris VẪN bị coi là "gần" dưới minimax — không
   có nút cổ chai ở biên thật để MST bắt được, nên đổi ngưỡng cũng vô nghĩa. Ngoài
   ra trần lợi ích quá thấp: Bảng III cho thấy chính DBCV cũng chỉ đạt 0.53 trên
   Flame (với 4 cụm) và 0.23 trên Jain (17 cụm), còn Iris thì DBCV = NKCV2 = 0.57.

Chẩn đoán gốc rễ
----------------
GA tìm được nghiệm fitness THẤP HƠN nhãn thật trên Flame (0.007220 < 0.008818):
tiêu chí sai, không phải bộ tối ưu sai. Bóc tách cho thấy nhánh "khác nhãn mà gần"
tính phạt CỐ ĐỊNH rho_j cho mỗi cạnh vượt biên, nên tiêu chí luôn thích cắt ở chỗ
đồ thị không có cạnh, và trừng phạt biên thật của dữ liệu dính nhau.

Sửa
---
Hỏi ở ĐÚNG THANG ĐO: hai điểm này có thuộc hai ĐỈNH MẬT ĐỘ khác nhau, ngăn cách bởi
một yên ngựa ĐỦ SÂU không? Đó là persistence topo (ToMATo, Chazal et al.):

    * dựng đồ thị kNN, duyệt điểm theo mật độ giảm dần, hợp nhất bằng union-find;
    * khi hai thành phần gặp nhau tại điểm i, thành phần yếu chết với
      persistence = rho(đỉnh của nó) - rho(i)   <- đúng bằng độ sâu yên ngựa;
    * giữ lại các đỉnh có persistence > tau  =>  nhãn "mode" của từng điểm.

Cạnh (i, j) vượt ranh giới hai mode  =>  tách chúng là HỢP LỆ  =>  mở cổng:

    w_ik = w_cross   nếu mode(i) != mode(M[i,k])
    w_ik = 1         nếu cùng mode
    đóng góp nhánh khác-nhãn  *=  w_ik

tau chọn TỰ ĐỘNG từ biểu đồ persistence (khe hở lớn nhất), KHÔNG dùng nhãn thật.

Vì sao không phá dữ liệu bình thường
------------------------------------
Cạnh nằm trong lòng một mode giữ nguyên w = 1, nên GA vẫn bị cấm xẻ đôi cụm đặc.
Cổng chỉ gỡ khoản phạt VÔ LÝ ở những cạnh vắt qua yên ngựa sâu. Cổng KHÔNG áp đặt
phân hoạch: số cụm, nhãn nhiễu, nhánh same-label và nhánh nhiễu vẫn do GA tối ưu
NKCV2 quyết định — mode chỉ là tri thức tiên nghiệm về chỗ nào được phép cắt.
`w_cross = 1` => TRÙNG KHỚP CHÍNH XÁC bản gốc.
"""

from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401  (them src/ vao sys.path)

from nkcv2 import njit, HAVE_NUMBA, NKCV2Model

# Số mode nhiều nhất được cân nhắc khi tự chọn tau từ biểu đồ persistence.
MAX_MODES_CONSIDERED = 25


# ---------------------------------------------------------------------------
# Kernel: giống bản gốc, chỉ nhân trọng số cổng Mw vào nhánh khác-nhãn
# ---------------------------------------------------------------------------

@njit(cache=True)
def _comp_fi_g(x, i, M, Mdist, Mrho, Mw, rho, dt0, dt1, dt2, d_rho, N, K):
    fi = 0.0
    xi = x[i]
    for k in range(K):
        j = M[i, k]
        d = Mdist[i, k]
        fmax = Mrho[i, k]
        if xi == 0:
            if rho[i] <= d_rho and d > dt2:
                pass
            else:
                fi += fmax
        elif xi == x[j]:
            if d > dt0 and d <= dt1:
                fi += fmax * (d - dt0) / (dt1 - dt0)
            elif d > dt1:
                fi += fmax
        else:
            w = Mw[i, k]
            if d <= dt0:
                fi += fmax * w
            elif d <= dt1:
                fi += (fmax - fmax * (d - dt0) / (dt1 - dt0)) * w
    return fi / (N * K)


@njit(cache=True)
def _comp_fitness_g(x, M, Mdist, Mrho, Mw, rho, dt0, dt1, dt2, d_rho, N, K):
    f = 0.0
    for i in range(N):
        f += _comp_fi_g(x, i, M, Mdist, Mrho, Mw, rho, dt0, dt1, dt2, d_rho, N, K)
    return f


@njit(cache=True)
def _df_element_g(x, i, xi_old, M, Mdist, Mrho, Mw, rho, rev_flat, rev_off,
                  rev_cnt, dt0, dt1, dt2, d_rho, N, K):
    xi_new = x[i]
    df = _comp_fi_g(x, i, M, Mdist, Mrho, Mw, rho, dt0, dt1, dt2, d_rho, N, K)
    x[i] = xi_old
    df -= _comp_fi_g(x, i, M, Mdist, Mrho, Mw, rho, dt0, dt1, dt2, d_rho, N, K)
    x[i] = xi_new
    start = rev_off[i]
    cnt = rev_cnt[i]
    for r in range(cnt):
        p = rev_flat[start + r]
        df += _comp_fi_g(x, p, M, Mdist, Mrho, Mw, rho, dt0, dt1, dt2,
                         d_rho, N, K)
        x[i] = xi_old
        df -= _comp_fi_g(x, p, M, Mdist, Mrho, Mw, rho, dt0, dt1, dt2,
                         d_rho, N, K)
        x[i] = xi_new
    return df


# ---------------------------------------------------------------------------
# ToMATo: hợp nhất mode theo persistence
# ---------------------------------------------------------------------------

def _knn_adjacency(D: np.ndarray, knn: int) -> list[list[int]]:
    """Đồ thị kNN ĐỐI XỨNG (bỏ chính nó)."""
    N = D.shape[0]
    nbrs = np.argsort(D, axis=1, kind="stable")[:, 1:knn + 1]
    adj: list[set] = [set() for _ in range(N)]
    for i in range(N):
        for j in nbrs[i]:
            j = int(j)
            adj[i].add(j)
            adj[j].add(i)
    return [sorted(s) for s in adj]


def _tomato(rho: np.ndarray, adj, tau: float):
    """Trả về (nhãn mode theo gốc union-find, danh sách persistence đã chết)."""
    N = len(rho)
    parent = np.arange(N)
    deaths: list[float] = []

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in np.argsort(-rho):
        i = int(i)
        higher = [j for j in adj[i] if rho[j] > rho[i]]
        if not higher:
            parent[i] = i          # đỉnh mới
            continue
        roots = {find(j) for j in higher}
        peak = max(roots, key=lambda r: rho[r])
        parent[i] = peak
        for r in roots:
            if r == peak:
                continue
            persistence = rho[r] - rho[i]      # độ sâu yên ngựa
            deaths.append(float(persistence))
            if persistence < tau:
                parent[r] = peak
    labels = np.array([find(i) for i in range(N)], dtype=np.int64)
    return labels, deaths


def auto_tau(rho: np.ndarray, adj) -> float:
    """Chọn tau tại KHE HỞ lớn nhất của biểu đồ persistence (không dùng nhãn)."""
    _, deaths = _tomato(rho, adj, np.inf)      # gộp hết -> lấy đủ biểu đồ
    if not deaths:
        return 0.0
    vals = np.sort(np.array(deaths))[::-1][:MAX_MODES_CONSIDERED]
    if vals.size < 2:
        return float(vals[0]) / 2.0
    gaps = vals[:-1] - vals[1:]
    g = int(np.argmax(gaps))
    return float((vals[g] + vals[g + 1]) / 2.0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PersistGateModel(NKCV2Model):
    """NKCV2 + cổng persistence. `w_cross=1` == bản gốc chính xác.

    Tham số:
        w_cross    hệ số nhân phạt cho cạnh trong vùng tranh chấp (0 = miễn hẳn).
        knn        bậc của đồ thị kNN dùng cho ToMATo.
        tau        ngưỡng persistence; None => tự chọn theo khe hở lớn nhất.
        seam_hops  số bước lan vùng giáp ranh; -1 = tắt (chỉ mở cổng đúng cạnh
                   vượt mode).
    """

    def __init__(self, X, K=3, w_cross=0.1, knn=10, tau=None, seam_hops=0):
        super().__init__(X, K=K)
        self.w_cross = float(w_cross)
        self.knn = int(knn)
        self.seam_hops = int(seam_hops)

        if self.w_cross >= 1.0:
            self.tau = 0.0
            self.mode_ = None
            self.Mw = np.ones((self.N, self.K), dtype=np.float64)
            return

        adj = _knn_adjacency(self.D, self.knn)
        self.tau = float(auto_tau(self.rho, adj)) if tau is None else float(tau)
        self.mode_, _ = _tomato(self.rho, adj, self.tau)
        self.n_modes_ = int(len(np.unique(self.mode_)))

        cross = self.mode_[self.M] != self.mode_[:, None]
        if self.seam_hops >= 0:
            seam = self._seam_points(adj)
            self.seam_ = seam
            cross |= seam[:, None] | seam[self.M]
        else:
            self.seam_ = None
        self.Mw = np.ascontiguousarray(
            np.where(cross, self.w_cross, 1.0).astype(np.float64))

    def _seam_points(self, adj) -> np.ndarray:
        """Điểm nằm trong VÙNG GIÁP RANH giữa hai mode.

        Ranh giới do ToMATo vẽ không trùng khít biên thật, nên các cạnh biên thật
        thường rơi vào trong CÙNG một mode và cổng theo mode không mở cho chúng.
        Ta nới ra: điểm có láng giềng kNN khác mode (rồi lan thêm `seam_hops` bước)
        đều được coi là vùng tranh chấp, để NKCV2 tự quyết cắt ở đâu trong đó.
        """
        seam = np.array([any(self.mode_[j] != self.mode_[i] for j in adj[i])
                         for i in range(self.N)])
        for _ in range(self.seam_hops):
            seam = np.array([seam[i] or any(seam[j] for j in adj[i])
                             for i in range(self.N)])
        return seam

    # -- ghi đè API đánh giá (dùng lại kernel có trọng số của nkcv2_gate) --------

    def comp_fi(self, x, i):
        return _comp_fi_g(x, i, self.M, self.Mdist, self.Mrho, self.Mw, self.rho,
                          self.dt0, self.dt1, self.dt2, self.d_rho,
                          self.N, self.K)

    def comp_fitness(self, x):
        return _comp_fitness_g(x, self.M, self.Mdist, self.Mrho, self.Mw,
                               self.rho, self.dt0, self.dt1, self.dt2,
                               self.d_rho, self.N, self.K)

    def df_element(self, x, i, xi_old):
        return _df_element_g(x, i, xi_old, self.M, self.Mdist, self.Mrho,
                             self.Mw, self.rho, self.rev_flat, self.rev_off,
                             self.rev_cnt, self.dt0, self.dt1, self.dt2,
                             self.d_rho, self.N, self.K)


__all__ = ["PersistGateModel", "auto_tau", "HAVE_NUMBA"]
