# Đề xuất cải tiến hàm fitness NKCV2 cho dữ liệu chồng lấn

> Bổ sung cho [`README.md`](README.md). Tài liệu này **chỉ đề xuất**; chưa có thí nghiệm nào trong đây
> được chạy. Mọi đề xuất đều tuân thủ ràng buộc ở README §5: hệ số trong `comp_fi` phải **tĩnh theo
> lời giải `x`**, để Delta Evaluation và Partition Crossover vẫn đúng.

## 0. Chẩn đoán: vì sao NKCV2 thua trên dữ liệu chồng lấn

Các bằng chứng đã có trong README (§4, §8, §12) quy về một điểm:

1. **Tín hiệu duy nhất của NKCV2 là khoảng cách `D_ij` cộng mật độ `ρ_j`.** Trong vùng chồng lấn,
   hai điểm khác lớp nằm sát nhau (`D ≤ dt0`). Khi đó nhánh khác nhãn phạt tối đa (`ρ_j`), còn nhánh
   cùng nhãn phạt `0`, nên NKCV2 luôn ưu tiên **gộp** hai lớp. Không có thung lũng mật độ để phân
   biệt. Vì vậy mọi biến thể chỉ xoay quanh `D` và `ρ` (CDW, ToMATo, MST, mutual reachability) đều chạm
   trần khoảng 0.57 ARI trên Iris.
2. **Trọng số nhân (`α' = ρ_j·w_ij`) không đủ.** Với cạnh ngắn cùng nhãn (`D ≤ dt0`), đóng góp bằng
   `0` bất kể `w`. Vì thế CDW chỉ làm việc *tách* rẻ đi mà không làm việc *gộp sai* đắt lên, và GA
   phân mảnh (Iris 12 cụm, Jain 26 cụm).
3. **Tín hiệu hình dạng (covariance) cần cho chồng lấn lại phụ thuộc phân hoạch**, nên bị cấm trong
   NK. Mahalanobis cục bộ tĩnh thì đúng chiều nhưng quá yếu (README §12.1).
4. **Compactness `c_ij` của CDW phụ thuộc mật độ**, nên trên Jain (hai cụm mật độ rất khác nhau) cụm
   loãng bị xé.

Hướng đi rút ra: cần **một tín hiệu quan hệ cặp `(i, j)` mới**. Tín hiệu đó phải (a) không suy ra được
từ `D` và `ρ`, (b) mang thông tin hình dạng hoặc ranh giới, (c) tính **một lần** trước GA, và (d) tác
động lên **cả hai** nhánh cùng nhãn và khác nhãn.

## 1. Đề xuất chính — NKCV-A: thêm số hạng ái lực cặp tĩnh `A_ij`

### 1.1 Công thức

Gọi `A_ij ∈ [0,1]` là xác suất tĩnh để `i` và `j` thuộc cùng cụm (cách tính ở §1.2). Với `β ∈ [0,1]`:

```text
cùng nhãn  : contribution = ρ_j · [ (1-β)·ramp_same(D_ij) + β·(1 - A_ij) ]
khác nhãn  : contribution = ρ_j · [ (1-β)·ramp_diff(D_ij) + β·A_ij       ]
nhiễu      : giữ nguyên NKCV2
```

- `β = 0` tái tạo NKCV2 gốc đến 1e-12, đúng quy tắc §16.2 của README.
- `A_ij` là mảng `(N, K)` tĩnh, nên `f_i` vẫn chỉ đọc `x_i` và `x_j` với `j ∈ M_i`. Delta Evaluation
  và PX giữ nguyên tính đúng, và có thể kiểm bằng `test_px_correctness.py`.
- Điểm khác CDW: số hạng `β(1 − A_ij)` **phạt cả việc gộp sai hai điểm gần nhau**. Đây chính là phần
  NKCV2 và CDW không biểu diễn được (điểm 2 ở §0).

Cài đặt: thêm `Mbeta_same`/`Mbeta_diff` theo dạng *cộng* vào `comp_fi_weighted` trong
`models/weighted_model.py` (hiện kernel chỉ có trọng số *nhân*). Mô hình con chỉ cần tính `A[N, K]`.

### 1.2 Ba cách tính `A_ij`, xếp từ rẻ đến mạnh

| Biến thể | `A_ij` | Tín hiệu đưa vào | Nguồn |
|---|---|---|---|
| **A-SNN** | `|kNN(i) ∩ kNN(j)| / k`, hoặc bản có trọng số mờ | cấu trúc lân cận, ít phụ thuộc thang mật độ | DPC-FWSN (Pattern Recognition 2023); họ SNN-DPC |
| **A-CA** (khuyên dùng) | ma trận đồng xuất hiện (co-association) từ một tập phân cụm cơ sở: GMM full covariance với nhiều `k` và seed, k-means với nhiều `k` | **hình dạng/covariance**, nhưng đã đông cứng thành hằng số theo `x` | họ ensemble clustering: LWEA/LWGP (IEEE TCYB), EC-CMS (Jia et al. 2022/2023, arXiv) |
| **A-GMM** | `Σ_c γ_ic γ_jc` (tích vô hướng posterior của một GMM được chọn theo BIC) | xác suất mềm cùng thành phần Gauss | GMM full covariance đạt 0.90 ARI trên Iris (README §4.3) |

Vì sao A-CA là ứng viên tốt nhất cho chồng lấn:

- Nó **lách được ràng buộc kiến trúc**. Covariance theo cụm hiện tại bị cấm vì phụ thuộc `x`. Covariance
  của các mô hình cơ sở thì được ước lượng *một lần, ngoài GA*, nên `A_ij` là hằng số.
- Ensemble nhiều `k` đóng vai trò làm mềm. NK-HGA vẫn quyết định số cụm qua merge/split và PX, còn
  `A_ij` chỉ nói hai điểm có hay đi cùng nhau hay không.
- `β` điều khiển mức tin vào mô hình hình dạng so với mật độ. Với Flame/Aggregation (ít chồng lấn) có
  thể chọn `β` nhỏ, với S3/S4/Iris thì `β` lớn. Chọn `β` theo DBCV hoặc silhouette, **không** theo ARI
  (quy tắc §16.6).

Rủi ro cần nói thẳng trong luận văn: A-CA và A-GMM thừa hưởng thiên lệch của mô hình cơ sở. GMM thích
cụm elip, nên trên Jain/Flame (phi lồi) `β` lớn có thể làm hại kết quả. Đây chính là lý do giữ phần
`(1-β)·NKCV2` thay vì thay hẳn.

### 1.3 Kiểm tra nhanh trước khi chạy GA (A1)

Với mỗi bộ dữ liệu và `β ∈ {0, .25, .5, .75}`: tính `diff% = [f(true) − f(GA baseline)]/f(GA baseline)`.
NKCV-A đáng chạy GA khi **Iris chuyển sang PASS mà Aggregation và Flame không chuyển sang FAIL**.

## 2. Đề xuất phụ — thay compactness của CDW bằng chỉ số biên không phụ thuộc mật độ

**Nguồn:** CDC — Peng et al., *Nature Communications* 13, 2022 (Q1). Chỉ số *local direction centrality*
(DCM) của mỗi điểm đo độ đều về hướng của `k` láng giềng. Điểm trong cụm có láng giềng bao quanh mọi
hướng, điểm biên thì lệch về một phía. Quan trọng: theo thiết kế, chỉ số này **độc lập với mật độ**,
nhắm thẳng vào dữ liệu "mật độ không đồng nhất và liên kết yếu". Bài LoDD (Peng et al., arXiv 2023,
chưa xác nhận nơi đăng) mở rộng ý này bằng trị riêng covariance cục bộ.

Áp dụng: `b_i = DCM_i` chuẩn hóa về `[0,1]` (1 = biên), tính tĩnh. Sau đó:

```text
w_ij(diff) = 1 - γ · sqrt(b_i b_j)        # tách giữa hai điểm biên thì rẻ
w_ij(same) = 1 + γ · (1 - sqrt(b_i b_j))  # (dùng kèm số hạng cộng của §1 nếu muốn)
```

Lý do: nó trực tiếp sửa nguyên nhân thất bại trên Jain (README §6.7: compactness `c_ij` phụ thuộc mật
độ nên xé cụm loãng). Đề xuất kiểm định: chạy lại ablation §9 với `c → b_DCM`, trên đủ bốn bộ.

## 3. Đề xuất phụ — số hạng "torque" cho các lần gộp bất thường

**Nguồn:** Torque Clustering — Yang & Lin, *IEEE TPAMI* 2025 (Q1). Ý chính: một phép gộp hai cụm là bất
thường khi *cả* khối lượng lẫn khoảng cách đều lớn (`τ = m_1·m_2·d²`). Thuật toán không cần tham số, tự
tìm số cụm và nhiễu, và các tác giả báo cáo AMI trung bình 97.7% trên 1000 bộ dữ liệu.

Phiên bản tĩnh cho NK: dùng cây hút mật độ (cạnh "density link" đầu tiên của `Gep`, README §3.2). Khối
lượng `m_i` là số điểm trong cây con có gốc tại `i`, tính một lần. Đặt `τ_ij = m_i·m_j·D_ij` trên cạnh
`Gep` và thay `D_ij` bằng `τ_ij` (chuẩn hóa) trong ramp cùng nhãn. Hai khối lớn nối bằng một cạnh dài
sẽ bị phạt mạnh khi gộp. Đây là thông tin cấu trúc phân cấp mà `ρ` cục bộ không thấy. Mức ưu tiên thấp
hơn §1 vì trên Iris không có cạnh nối dài giữa versicolor và virginica.

## 4. Đề xuất kiến trúc — fitness hai tầng (giữ NK bên trong, đưa phần phi phân rã ra ngoài)

Khi tín hiệu cần thiết **buộc phải** phụ thuộc phân hoạch (covariance theo cụm, BIC), đừng nhét nó vào
`f_i`. Thay vào đó:

1. **Tầng trong (NK-HGA):** tối ưu NKCV2/NKCV-A với `β` nhỏ, ưu tiên **phân mảnh có kiểm soát**. Các cụm
   con dựa trên mật độ thường *thuần* (purity cao) dù số lượng nhiều; README cho thấy Iris bị phân
   mảnh chứ không bị trộn ngẫu nhiên.
2. **Tầng ngoài (gộp cụm con):** ghép cụm con theo tiêu chí phi phân rã, ví dụ gộp nếu BIC của GMM
   giảm, hoặc nếu DBCV tăng. Chi phí `O(Nc²)` mỗi lần, chạy trên lời giải cuối hoặc mỗi `leng_local`
   thế hệ.

Đây là mô thức "phân mảnh rồi gộp" phổ biến trong họ DPC cải tiến (gán hai bước, chọn tâm ứng viên rồi
gộp; xem R-MDPC, Pattern Recognition 2024; FNaN-DPC, Neurocomputing 2025). Nó không phá PX hay
Delta Evaluation vì tầng ngoài nằm ngoài vòng GA. Đây là phương án an toàn nhất nếu §1 thất bại.

## 5. Đề xuất đa mục tiêu (dài hạn)

Dòng MOCK → Δ-MOCK (Garza-Fabre, Handl, Knowles, IEEE TEVC 2018) cho thấy dùng **hai tiêu chí bổ trợ**
(compactness kiểu tâm và connectivity kiểu lân cận) bền hơn một CVI duy nhất trên dữ liệu có chồng lấn
khác nhau. Phân tích tính khả chấp (admissibility) của các hàm mục tiêu này (arXiv 2022) cho thấy điều
kiện cần: phân hoạch đúng phải nằm trên mặt Pareto. Điều kiện này cũng chính là phép thử A1 ở README.

Với NK: nếu cả hai mục tiêu đều phân rã trên cùng `Gep` (NKCV2 và một số hạng A-CA), thì PX theo tổng
có trọng số vẫn đúng với mỗi `λ`. Có thể quét `λ` để xấp xỉ mặt Pareto rồi chọn knee hoặc chọn bằng
DBCV. Cách này tốn hơn §1 nhiều, nên chỉ làm khi §1 cho thấy không có `β` cố định nào tốt trên mọi bộ.

## 6. Bộ dữ liệu để chứng minh tính tổng quát

Flame/Aggregation/Jain gần như không chồng lấn. Cần thêm dữ liệu có **mức chồng lấn được kiểm soát**:

| Bộ | Vì sao |
|---|---|
| **S1–S4** (Fränti & Sieranoja, *Applied Intelligence* 2018) | 5000 điểm, 15 cụm Gauss, chồng lấn tăng dần từ S1 đến S4: vẽ được đường ARI theo mức chồng lấn |
| G2-sets (cùng nguồn) | chồng lấn theo số chiều và độ lệch chuẩn |
| Iris, Wine, Seeds (UCI) | chồng lấn thực, nhiều chiều |
| Giữ Flame, Aggregation, Jain | kiểm tra cải tiến không làm hỏng dữ liệu phi lồi |

Kết quả nên trình bày theo dạng: *ARI theo mức chồng lấn (S1 đến S4)*, so sánh NKCV2, CDW, NKCV-A và
GMM/DBSCAN/HDBSCAN. Nếu NKCV-A giữ được Flame/Aggregation và bám theo GMM trên S3/S4, đó là luận điểm
tổng quát hóa mạnh.

## 7. Thứ tự thực hiện đề xuất

1. Mở rộng `weighted_model.py` với số hạng cộng `β·A_ij` (§1.1), kèm self-test `β=0` và Delta/PX.
2. Cài A-SNN (rẻ), A-GMM và A-CA, rồi chạy A1 trên Iris, Flame, Aggregation, Jain và S1–S4.
3. Chạy GA đầy đủ (≥10 seed, bootstrap CI như README §10.2) cho biến thể qua A1.
4. Nếu Jain vẫn hỏng: thay `c_ij` bằng `b_DCM` (§2).
5. Nếu Iris/S4 vẫn hỏng: thêm tầng gộp ngoài (§4).

## 8. Nguồn tham khảo (đã kiểm tra tồn tại; xếp hạng Q theo SJR nên tra lại tại scimagojr.com)

| # | Tài liệu | Nơi đăng | Hạng | Dùng ở |
|---|---|---|---|---|
| 1 | Peng D., Gui Z., et al. *Clustering by measuring local direction centrality for data with heterogeneous density and weak connectivity* — [PMC9481560](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9481560/) | Nature Communications 13 (2022) | Q1 | §2 |
| 2 | Yang J., Lin C.-T. *Autonomous Clustering by Fast Find of Mass and Distance Peaks* (Torque Clustering), DOI 10.1109/TPAMI.2025.3535743 — [UTS](https://www.uts.edu.au/news/2025/02/truly-autonomous-ai-is-on-the-horizon) | IEEE TPAMI (2025) | Q1 | §3 |
| 3 | *Density peaks clustering algorithm based on fuzzy and weighted shared neighbor for uneven density datasets* (DPC-FWSN) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320323001073) | Pattern Recognition (2023) | Q1 | §1.2 A-SNN |
| 4 | *Fast main density peak clustering within relevant regions via a robust decision graph* (R-MDPC) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320324002097) | Pattern Recognition 152 (2024) | Q1 | §4 |
| 5 | *Density peaks clustering algorithm via fusing natural neighbor and fuzzy information* (FNaN-DPC) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231225018247) | Neurocomputing 653 (2025) | Q1 | §4, điểm biên |
| 6 | *A novel cluster validity index based on augmented non-shared nearest neighbors* (ANCV) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417423002853) | Expert Systems with Applications (2023) | Q1 | CVI dựa SNN, có thể dùng làm tiêu chí chọn `β` |
| 7 | Ren C. et al. *DPC based on local fair density and fuzzy kNN membership allocation* (LF-DPC) — [SAGE](https://journals.sagepub.com/doi/abs/10.3233/JIFS-202449) | J. Intelligent & Fuzzy Systems (2022) | Q2–Q3, ưu tiên thấp | §1.2 |
| 8 | Fränti P., Sieranoja S. *K-means properties on six clustering benchmark datasets*, DOI 10.1007/s10489-018-1238-7 — [UEF datasets](https://cs.uef.fi/sipu/datasets) | Applied Intelligence 48 (2018) | Q2 | §6 (dữ liệu) |
| 9 | Garza-Fabre M., Handl J., Knowles J. *An improved and more scalable evolutionary approach to multiobjective clustering* (Δ-MOCK) — [Birmingham](https://research.birmingham.ac.uk/en/publications/an-improved-and-more-scalable-evolutionary-approach-to-multiobjec/) | IEEE TEVC 22(4) (2018) | Q1, nền tảng (trước 2021) | §5 |
| 10 | *An Analysis of the Admissibility of the Objective Functions Applied in Evolutionary Multi-objective Clustering* — [arXiv 2206.09483](https://arxiv.org/pdf/2206.09483) | preprint (2022) | chưa xếp hạng | §5 |
| 11 | Jia Y. et al. *Ensemble Clustering via Co-association Matrix Self-enhancement* (EC-CMS) — [arXiv 2205.05937](https://arxiv.org/pdf/2205.05937) | preprint (2022/2023); chưa xác nhận nơi đăng | — | §1.2 A-CA |
| 12 | Peng D. et al. *A Robust and Efficient Boundary Point Detection Method by Measuring Local Direction Dispersion* — [arXiv 2312.04065](https://arxiv.org/pdf/2312.04065) | preprint (2023/2025) | — | §2 |

Ghi chú: Huang et al. *Locally weighted ensemble clustering* (IEEE TCYB 2018, [arXiv 1605.05011](https://arxiv.org/pdf/1605.05011))
là nguồn kinh điển cho ma trận co-association có trọng số. Bài trước 2021, chỉ dùng làm nền. Chưa tìm
thấy công trình tiếp nối NK-HGA cho phân cụm của nhóm Tinós sau 2018 (bản arXiv 2402.03813 là bản đăng
lại của bài TEVC 2018).
