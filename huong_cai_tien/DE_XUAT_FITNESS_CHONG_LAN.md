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
`lich_su/models/weighted_model.py` (hiện kernel chỉ có trọng số *nhân*). Mô hình con chỉ cần tính `A[N, K]`.

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

## 7b. Kết quả thực nghiệm §1 (NKCV-A) và §4 (hai tầng)

Mã: `lich_su/models/nkcv2_affinity.py` (NKCV-A), `lich_su/models/two_level_merge.py` (gộp
theo BIC), `lich_su/experiments/run_nkcva_two_level.py`. Dữ liệu thô:
`lich_su/results/nkcva_two_level_<bộ>.csv`.

Tên trong code: `ca_gmm` = **A-CA** (đồng xuất hiện từ ensemble GMM, k = 2..10, 5 seed mỗi k),
`gmm` = **A-GMM** (posterior của GMM chọn k theo BIC), `snn` = **A-SNN** (`k = 10`).

Kiểm tra đúng đắn, tất cả PASS: `β=0` trùng NKCV2 đến 1e-12; Delta Evaluation khớp đánh giá lại
qua 300 nước đi cho cả bốn nguồn; PX trả fitness đúng đến 1e-16, con không tệ hơn cha mẹ, và
`fix_labels` giữ nguyên fitness.

Giao thức như README §8: seed 100–102, `pop=60`, `gen=80`, `K=3`. `inner` là nhãn NK-HGA, `merged` là
nhãn sau bước gộp BIC (cùng một lần chạy GA). Số trong ngoặc là số cụm trung bình.

| Bộ | base (NKCV2) | A-CA β=.25 | A-CA β=.5 | A-CA β=.75 | A-GMM β=.5 | A-SNN β=.5 |
|---|---|---|---|---|---|---|
| Aggregation, inner | 0.928 (6.3) | 0.910 (6.0) | 0.883 (5.7) | 0.885 (5.7) | 0.836 (5.0) | **0.965** (6.7) |
| Flame, inner | 0.013 (2.0) | 0.013 (1.0) | 0.013 (1.0) | 0.013 (1.0) | 0.332 ± 0.55 (1.3) | 0.284 ± 0.47 (1.7) |
| Flame, merged | 0.000 (1.0) | 0.013 | 0.013 | 0.013 | 0.013 | 0.011 |
| Iris, inner | 0.528 (6.3) | 0.544 (3.0) | 0.541 (3.3) | 0.551 (2.3) | 0.548 (2.0) | 0.532 (4.7) |
| Iris, merged | 0.561 (2.0) | 0.548 (2.0) | 0.548 (2.0) | 0.552 (2.0) | 0.548 (2.0) | 0.548 (2.0) |
| Jain, inner | 0.327 (7.7) | 0.240 (5.7) | **0.539** ± 0.20 (4.0) | 0.247 (2.7) | 0.440 (4.7) | 0.170 (10.7) |
| Jain, merged | 0.348 (4.3) | 0.116 | 0.539 | 0.168 | 0.440 | 0.259 |

Trên Aggregation bước gộp không đổi nhãn nào, nên chỉ ghi dòng `inner`.

### Đọc kết quả

1. **NKCV-A không giải được chồng lấn trên Iris.** Phép thử A1 (nhãn thật so với lời giải GA) trượt
   với mọi `β` và mọi `K ∈ {3, 6, 10}`. Lý do đo được: trên các cạnh Gep cắt ngang versicolor/virginica,
   ái lực vẫn cao (A-CA 0.58, A-GMM 0.90, A-SNN 0.44), so với cạnh cùng lớp (0.86 / 1.00 / 0.67). Ở
   thang 3 láng giềng gần nhất, hai lớp thực sự trộn lẫn.
2. **Iris 3 lớp không xác định được nếu không cho trước số cụm.** GMM full covariance trên Iris đã
   chuẩn hóa có BIC(k=2) = 794.7 < BIC(k=3) = 801.5, nên **chính GMM cũng chọn 2 cụm**. Nó chỉ đạt
   ARI 0.904 khi được cho trước `k=3`. Lời giải "setosa / phần còn lại" có ARI 0.568, và mọi biến thể
   ở trên đều nằm sát trần này. Các phương pháp báo Iris ≈ 0.77–0.90 (như DP-GA-PFCM, Kuo et al.
   2023) đều **cho trước `k`**. Do đó Iris không phải phép thử công bằng cho NK-HGA (tự tìm số cụm);
   cần dùng bộ có chồng lấn được kiểm soát (S1–S4, §6).
3. **Flame: NKCV-A sửa được fitness nhưng GA không tìm tới.** Với `K=3`, `β=0.5`, A1 **đạt**:
   f(nhãn thật) = 0.02818 < f(GA) = 0.02880. Thế nhưng GA vẫn trả về 1–2 cụm. A-GMM và A-SNN thành
   công ở 1/3 seed (độ lệch chuẩn ~0.5). Đây là giới hạn của bộ tìm kiếm, không còn là của fitness.
4. **Có cải thiện cục bộ:** A-SNN nâng Aggregation 0.928 → 0.965 (2/3 seed đạt 0.99 với đúng 7 cụm;
   trung bình vẫn thấp hơn CDW 0.981 ở README §8). Trên Flame, A-GMM seed 102 đạt 0.971 và A-SNN seed
   101 đạt 0.828; các seed còn lại sập. A-CA β=0.5 nâng Jain 0.327 → 0.539, nhưng không ổn định
   (β=0.25 và 0.75 đều tệ hơn base). Chưa có cấu hình nào thắng đồng thời cả bốn bộ.
5. **§4 (gộp theo BIC) giảm phân mảnh nhưng thừa hưởng thiên lệch Gauss.** Nó đưa Iris từ 6.3 về
   2 cụm (ARI 0.528 → 0.561) và Jain base từ 7.7 về 4.3 cụm. Nhưng trên Flame nó gộp hai cụm phi
   Gauss thành một, và xóa các seed A-GMM/A-SNN đã tách đúng. BIC kiểu Gauss không phải tiêu chí gộp
   an toàn cho dữ liệu phi lồi.

### Hệ quả cho bước tiếp theo

- Với Flame, đầu tư vào **tìm kiếm** chứ không phải công thức. Fitness NKCV-A đã đúng, cần khởi tạo
  hoặc mutation giúp GA tách được hai cụm dính (ví dụ khởi tạo một phần quần thể từ nhãn DPC hoặc từ
  ensemble dùng để tính A-CA).
- Với tầng ngoài §4, thay BIC bằng tiêu chí không giả định Gauss (DBCV hoặc một chỉ số dựa trên
  liên thông), và chỉ gộp khi tiêu chí cải thiện rõ.
- Đánh giá tổng quát trên **S1–S4** thay cho việc lấy Iris làm thước đo chính về chồng lấn.

## 7c. Tách lỗi tìm kiếm khỏi lỗi fitness: thử gieo quần thể ban đầu

Script tạm ban đầu đã được thay bằng `models/seeded_ga.py` và
`lich_su/experiments/run_seeding_grid.py` (mục 7d.2 lặp lại kết luận này với 5 seed). Gieo 10 cá thể ban đầu từ GMM và k-means với `k = 2..6`
(không dùng nhãn thật), mỗi cá thể qua `ls_fi` như immigrant thường. Fitness giữ nguyên.
Seed 100–102, `pop=60`, `gen=80`. A-CA được dùng với `β=0.5`.

| Bộ | Fitness | Khởi tạo ngẫu nhiên | Khởi tạo có gieo | ARI tốt nhất trong các lời giải gieo |
|---|---|---|---|---|
| Flame | NKCV2 gốc | 0.013 (3/3 seed sập) | 0.614 (0.923, 0.906, 0.013) | 0.597 |
| Flame | NKCV-A (A-CA, β=.5) | 0.013 (3/3 seed sập) | **0.933 ± 0.008**, đúng 2 cụm ở cả 3 seed | 0.597 |
| Jain | NKCV2 gốc | 0.323 | 0.272 | 0.553 |
| Jain | NKCV-A (A-CA, β=.5) | 0.539 | 0.409 | 0.553 |

Đọc:

- **Flame là lỗi tìm kiếm.** Cùng một fitness NKCV-A, chỉ đổi khởi tạo mà ARI tăng từ 0.013 lên 0.933,
  ổn định ở cả 3 seed. GA còn vượt xa lời giải gieo tốt nhất (0.597), nên đây là GA + PX + LS thực
  sự làm việc, không phải chép lại lời giải gieo. Ngay cả NKCV2 gốc cũng đạt 0.92 ở 2/3 seed khi
  được gieo.
- **Jain là lỗi fitness.** Gieo không giúp (còn tệ hơn), số cụm vẫn 4.7–9. Mật độ thay đổi dọc theo
  hai vòng cung khiến NKCV2 thích cắt vòng cung thành nhiều đoạn.
- **Iris là lỗi xác định được** (§7b): không tiêu chí không giám sát nào đã thử, kể cả GMM-BIC, chọn
  3 cụm.

## 7d. Đánh giá trên 10 bộ: tốc độ, gieo quần thể, Ecoli/Glass, S1–S4

### 7d.1 Tăng tốc (không đổi thuật toán)

Profile cho thấy nút thắt là `map_solutions` và `fix_labels` trong `src/px.py`, không phải local
search. `models/fast_px.py` viết lại hai hàm này: ghép nhãn bằng danh sách overlap đã sắp xếp
(`O(Nc³)` Python → `O(N log N)`) và union-find bằng numba. Đầu ra **trùng từng phần tử** với bản gốc
(`self_test`), và một lần chạy GA đầy đủ trên Flame cho nhãn cuối trùng khớp.

| Cấu hình | Trước | Sau |
|---|---|---|
| S1, N=1000, pop 60, gen 80 | ~30 phút (91 s cho 3 thế hệ) | 6.1 s |
| S1 đầy đủ, N=5000, pop 60, gen 80 | không khả thi | 36.5 s |

### 7d.2 Lưới 4 cấu hình × 10 bộ

`lich_su/experiments/run_seeding_grid.py`, seed 100–104, `pop=60`, `gen=80`, `K=3`. S1–S4 dùng mẫu con 1000
điểm (seed 0). Gieo: 30% quần thể ban đầu và immigrant từ GMM/k-means/DPC với `k = 2..k_max(N)`,
không dùng nhãn thật (`models/seeded_ga.py`). Dữ liệu Ecoli, Glass, S1–S4 lấy từ bộ benchmark của
Gagolewski (bản sao UCI/SIPU). ARI trung bình ± SD (số cụm):

| Bộ | k thật | base | base+seed | nkcva | nkcva+seed |
|---|---|---|---|---|---|
| Flame | 2 | 0.013 (1.8) | 0.741 ± 0.41 (2.4) | 0.013 (1.0) | **0.939** ± 0.00 (2.0) |
| Aggregation | 7 | 0.871 (5.6) | **0.988** ± 0.00 (7.0) | 0.895 (5.8) | 0.912 (6.0) |
| Jain | 2 | 0.401 (7.4) | 0.279 (8.0) | **0.414** (3.6) | 0.313 (5.0) |
| Iris | 3 | 0.526 (5.8) | 0.524 (4.8) | 0.543 (3.0) | 0.547 (2.4) |
| Ecoli | 8 | 0.037 (2.6) | 0.038 (3.0) | 0.038 (2.0) | 0.038 (2.0) |
| Glass | 6 | 0.131 (6.2) | 0.164 (6.6) | 0.139 (7.4) | **0.200** (6.4) |
| S1 | 15 | **0.905** (14.0) | **0.905** (14.0) | 0.806 (13.2) | 0.780 (13.0) |
| S2 | 15 | **0.449** (8.8) | 0.339 (8.6) | 0.405 (7.4) | 0.442 (7.8) |
| S3 | 15 | 0.131 (5.2) | **0.313** (7.8) | 0.071 (2.4) | 0.170 (5.6) |
| S4 | 15 | 0.022 (4.2) | **0.079** (5.6) | 0.025 (3.0) | 0.074 (3.0) |

Đọc:

- **Gieo quần thể là cải tiến đáng tin nhất.** Nó giúp hoặc trung tính trên 8/10 bộ: Aggregation
  0.871 → 0.988 (đúng 7 cụm ở cả 5 seed), Flame (với NKCV-A) 0.013 → 0.939, S3 0.131 → 0.313, Glass
  +0.03–0.06. Nó làm hại Jain và S2 với fitness gốc.
- **NKCV-A không nên cố định làm fitness chung.** Nó thắng Flame và Glass nhưng thua S1 (0.905 →
  0.806) và Aggregation khi có gieo (0.988 → 0.912). Thay `k_max` của A-CA theo `N` (15 thay vì 10)
  và tăng `β` lên 0.8 cũng không giúp S2/S3 (vẫn ~7 cụm). Lựa chọn ít rủi ro nhất là **giữ NKCV2 gốc
  + gieo**.

### 7d.3 Ecoli/Glass: lỗi tiền xử lý, không phải fitness hay tìm kiếm

Ecoli có hai cột gần như hằng số: `lip` (2 giá trị, 97% cùng giá trị) và `chg` (335/336 điểm cùng
giá trị). Sau `StandardScaler`, điểm duy nhất khác ở `chg` cách trung bình ~√335 ≈ 18 SD và chi phối
khoảng cách bình phương. Glass có `Ba` (82% bằng 0) và `Fe` (67% bằng 0).

`lich_su/experiments/run_overlap_diagnostics.py prep`, NKCV2 gốc, 3 seed:

| Bộ | Tiền xử lý | Ngẫu nhiên | Gieo |
|---|---|---|---|
| Ecoli | Standard (hiện tại) | 0.038 (2.7 cụm) | 0.038 (3.0) |
| Ecoli | MinMax | 0.443 (5.7) | 0.443 (4.7) |
| Ecoli | Standard, bỏ `lip`+`chg` | 0.448 (8.3) | **0.454** (8.0) |
| Glass | Standard (hiện tại) | 0.106 (6.0) | 0.161 (6.7) |
| Glass | MinMax | 0.180 (6.3) | **0.211** (7.0) |
| Glass | Standard sau `log1p` | 0.178 (5.0) | 0.189 (5.7) |

Để đối chiếu, DP-GA-PFCM (Kuo et al. 2023, **cho trước k**) báo Ecoli 0.35–0.44 và Glass 0.16–0.19
ARI. NK-HGA chỉ cần sửa tiền xử lý đã ngang hoặc hơn, mà không cần biết trước k.

### 7d.4 S1–S4: đây mới là điểm yếu chồng lấn thật, và nó nằm ở fitness

Khác Iris, trên S1/S2 các cụm **xác định được**: GMM-BIC tự chọn k = 15/16 và đạt ARI 0.97/0.93.
Phép thử A1 so với lời giải GMM 15 cụm (gần tối ưu Bayes):

| Bộ | GMM-15 ARI | f(GMM-15) | f(GA) | Cụm GA | ARI GA | A1 |
|---|---|---|---|---|---|---|
| S1 | 0.974 | 0.00119 | 0.00094 | 14 | 0.905 | FAIL |
| S2 | 0.928 | 0.00320 | 0.00205 | 9 | 0.324 | FAIL |
| S3 | 0.708 | 0.00858 | 0.00363 | 10 | 0.446 | FAIL |
| S4 | 0.675 | 0.01039 | 0.00285 | 5 | 0.039 | FAIL |

NKCV2 xếp lời giải gộp cụm tốt hơn lời giải gần đúng, và khoảng cách tăng theo mức chồng lấn. Cơ chế:
NKCV2 chỉ cho phép đặt biên ở nơi các láng giềng cách xa nhau (thung lũng mật độ). Các cụm Gauss
chồng lấn không có thung lũng, nên mọi biên đi qua vùng đặc đều bị phạt và GA gộp cụm.

Đây là giới hạn của cả họ phương pháp mật độ, nhưng NK-HGA kém hơn hai đại diện của họ này:

| Bộ | NK-HGA tốt nhất (§7d.2) | HDBSCAN | DBSCAN (trần, eps chọn bằng nhãn thật) | GMM-BIC |
|---|---|---|---|---|
| S1 | 0.905 | 0.872 | 0.916 | 0.974 |
| S2 | 0.449 | 0.576 | 0.680 | 0.926 |
| S3 | 0.313 | 0.220 | 0.450 | 0.598 |
| S4 | 0.079 | 0.176 | 0.311 | 0.220 |

Lưu ý: mẫu con 1000 điểm (~67 điểm/cụm) làm ước lượng mật độ thô hơn bản 5000 điểm; với tốc độ hiện
tại, bản đầy đủ chạy được (~37 s/lần) và nên dùng cho bảng cuối của luận văn.

### 7d.5 Kế hoạch đã điều chỉnh theo bằng chứng

1. **Fitness: giữ NKCV2 gốc**, không cố định NKCV-A. NKCV-A chỉ thắng cục bộ (Flame, Glass).
2. **Tìm kiếm: gieo quần thể** là mặc định (lợi trên 8/10 bộ). Mutation dời đoạn biên chưa làm.
3. **Tiền xử lý:** dùng MinMax, hoặc loại cột gần hằng số trước khi chuẩn hóa. Đây là sửa đổi rẻ
   nhất, lợi lớn nhất cho dữ liệu thực nhiều chiều (Ecoli ×12).
4. **Đánh giá chồng lấn bằng S1–S4** (bản 5000 điểm). Iris chỉ làm phụ lục, kèm ghi chú GMM-BIC chọn
   2 cụm.
5. **Chồng lấn kiểu S2–S4 là lỗi fitness có cấu trúc**: NKCV2 định nghĩa cụm là thành phần liên thông
   mật độ. Nếu muốn giải, phải đổi định nghĩa (ví dụ một số hạng dựa trên mô hình, chỉ áp ở vùng không
   có thung lũng), không phải chỉnh trọng số. Đây là quyết định phạm vi của luận văn, không phải một
   tinh chỉnh.

## 7e. Phiên bản hiện hành, bảng cuối và so sánh với CDW

### 7e.1 Chọn tiền xử lý (10 bộ, 3 seed, bản gieo, S1–S4 mẫu 1000 điểm)

| Bộ | Standard | Standard + bỏ cột gần hằng | **MinMax + bỏ cột gần hằng** |
|---|---|---|---|
| Flame | 0.619 ± 0.43 | 0.619 ± 0.43 | **0.923 ± 0.00** |
| Aggregation | 0.988 | 0.988 | 0.988 |
| Jain | 0.273 | 0.273 | **0.323** |
| Iris | 0.523 | 0.523 | **0.544** |
| Ecoli | 0.038 | **0.454** | 0.437 |
| Glass | 0.161 | 0.161 | **0.211** |
| S1 | 0.905 | 0.905 | 0.905 |
| S2 | 0.324 | 0.324 | **0.392** |
| S3 | **0.434** | **0.434** | 0.419 |
| S4 | **0.102** | **0.102** | 0.044 |

MinMax + bỏ cột gần hằng số bằng hoặc tốt hơn trên 8/10 bộ, nên được chọn làm mặc định
(`models/datasets.py::preprocess`).

### 7e.2 Bảng cuối: bản gốc so với bản hiện hành (10 seed, S1–S4 đủ 5000 điểm)

`experiments/run_final_table.py`. ARI trung bình ± SD (số cụm). HDBSCAN và GMM-BIC chạy một lần,
cùng tiền xử lý với bản hiện hành.

| Bộ | Gốc | Hiện hành | HDBSCAN | GMM-BIC |
|---|---|---|---|---|
| Flame | 0.104 ± 0.27 (1.7) | **0.920 ± 0.01** (2.6) | 0.409 (2) | 0.600 (3) |
| Aggregation | 0.902 ± 0.05 (6.8) | **0.988 ± 0.00** (7.0) | 0.734 (4) | 0.568 (11) |
| Jain | 0.377 ± 0.16 (7.8) | 0.316 ± 0.03 (8.9) | **0.931** (2) | 0.293 (5) |
| Iris | 0.529 ± 0.01 (5.1) | 0.546 ± 0.00 (4.8) | 0.568 (2) | 0.568 (2) |
| Ecoli | 0.038 ± 0.00 (2.9) | 0.444 ± 0.02 (7.8) | 0.367 (2) | **0.641** (6) |
| Glass | 0.127 ± 0.04 (5.4) | 0.212 ± 0.00 (6.9) | **0.252** (2) | 0.152 (3) |
| S1 | 0.429 ± 0.25 (239) | 0.756 ± 0.00 (12) | 0.925 (15) | **0.986** (16) |
| S2 | 0.207 ± 0.13 (156) | 0.331 ± 0.02 (5.9) | 0.648 (15) | **0.942** (15) |
| S3 | 0.126 ± 0.14 (269) | 0.033 ± 0.04 (2.0) | 0.166 (17) | **0.731** (15) |
| S4 | 0.069 ± 0.09 (254) | 0.000 ± 0.00 (1.6) | 0.162 (19) | **0.673** (15) |

Trên S1–S4 bản 5000 điểm, bản gốc chưa hội tụ sau 80 thế hệ (150–270 cụm), còn bản hiện hành gộp
quá tay (S3/S4 chỉ còn 1–2 cụm). Cả hai đều kém hơn bản mẫu 1000 điểm.

### 7e.3 CDW cũ so với base + gieo, cùng giao thức

`lich_su/experiments/run_cdw_same_protocol.py`. Seed 100–104, S1–S4 mẫu 1000 điểm. CDW chạy như
lúc đề xuất (Standard, khởi tạo ngẫu nhiên); `CDW+gieo` dùng thêm gieo quần thể và tiền xử lý mới.

| Bộ | Gốc | CDW cũ | Base + gieo (hiện hành) | CDW + gieo |
|---|---|---|---|---|
| Flame | 0.013 (1.8) | **0.929** ± 0.02 (3.0) | 0.923 ± 0.00 (2.8) | 0.562 ± 0.13 (3.2) |
| Aggregation | 0.903 (6.0) | 0.984 (7.0) | **0.988** (7.0) | **0.988** (7.0) |
| Jain | **0.404** ± 0.23 (7.6) | 0.142 (26.0) | 0.306 (9.0) | 0.176 (23.2) |
| Iris | 0.526 (5.8) | 0.535 (11.8) | 0.545 (4.8) | **0.597** (8.6) |
| Ecoli | 0.038 (2.8) | 0.038 (5.0) | 0.440 (7.6) | **0.676** (23.0) |
| Glass | 0.129 (5.8) | **0.214** (12.2) | 0.211 (6.8) | 0.169 (12.0) |
| S1 | 0.905 (14.0) | 0.953 (32.0) | 0.905 (14.0) | **0.967** (22.0) |
| S2 | 0.407 (8.6) | 0.761 (31.0) | 0.416 (8.0) | **0.783** (31.0) |
| S3 | 0.030 (4.4) | 0.554 (39.8) | 0.398 (9.0) | **0.574** (33.2) |
| S4 | 0.049 (6.0) | 0.381 (22.0) | 0.047 (5.8) | **0.416** (18.4) |
| **ARI TB 10 bộ** | 0.340 | 0.549 | 0.518 | **0.591** |
| **SD TB** | 0.047 | 0.026 | **0.018** | 0.028 |

Đọc:

- **CDW hiệu quả hơn về ARI**, chủ yếu nhờ S2–S4: S2 0.76 so với 0.42, S4 0.38 so với 0.05. Điều
  này sửa lại nhận định ở 7d.4–7d.5: đổi fitness *có* giúp được dữ liệu chồng lấn. CDW hạ phạt nhánh
  khác nhãn, nên cho phép đặt biên trong vùng đặc.
- **Cái giá là phân mảnh.** CDW cho 22–40 cụm trên S-sets (thật là 15), 26 cụm trên Jain (thật là
  2), 12 trên Glass (thật là 6). ARI vẫn cao vì phần lớn mảnh thừa rất nhỏ, nhưng **số cụm sai**. Đây
  là điều quan trọng với NK-HGA, vì tự tìm số cụm là điểm mạnh được quảng bá.
- **Base + gieo ổn định nhất** (SD 0.018) và cho **số cụm gần đúng** trên Flame, Aggregation, Ecoli,
  Glass. Nhưng nó không giải được chồng lấn (S2–S4).
- Kết hợp CDW + gieo cho ARI trung bình cao nhất (0.591) nhưng làm Flame tụt (0.56) và vẫn phân mảnh.

## 7f. CDW + gieo + gộp mảnh vụn, và thử trên mô hình Gauss có nhiễu

### 7f.1 Gộp mảnh vụn (`models/fragment_merge.py`)

Mảnh vụn là cụm có ít hơn `τ = max(K+1, ⌈2% N⌉)` điểm (cùng tinh thần quy tắc 2% chọn `dc`). Từ
mảnh nhỏ nhất, cả mảnh được gán cho nhãn chiếm đa số trong các láng giềng Gep của nó (ưu tiên cụm
không phải mảnh vụn); mảnh cô lập thành nhiễu. Chạy sau GA nên không ảnh hưởng PX hay Delta
Evaluation. Thử nhanh CDW + gieo, 1 seed (trước → sau khi gộp):

| Bộ | ARI | Số cụm (thật) |
|---|---|---|
| S1 | 0.964 → 0.976 | 24 → **15** (15) |
| S2 | 0.788 → 0.824 | 34 → 13 (15) |
| S3 | 0.586 → 0.603 | 34 → 12 (15) |
| Iris | 0.621 → 0.669 | 9 → 4 (3) |
| Ecoli | 0.700 → 0.747 | 25 → 5 (8) |
| Glass | 0.163 → 0.151 | 11 → **6** (6) |
| Jain | 0.157 → 0.167 | 22 → 10 (2) |

### 7f.2 Mô hình Gauss có nhiễu (`models/gaussian_model.py`, `experiments/run_gaussian_noise.py`)

Bản tái tạo bộ sinh của bài báo (mục V-A1) theo Milligan (1981); kho mã của tác giả không kèm bộ
sinh. Nc = 2, l = 2, N = 800, 1% nhiễu đều, 3 mức cân bằng × 3 bộ = 9 bộ, 5 seed.

| Phương pháp | ARI | Số cụm TB | Đúng 2 cụm |
|---|---|---|---|
| Gốc | **0.977 ± 0.010** | 4.78 | 18% |
| CDW + gieo + gộp (trước gộp) | 0.875 ± 0.145 | 12.3 | 2% |
| CDW + gieo + gộp | 0.820 ± 0.217 | 2.84 | 33% |
| HDBSCAN | 0.981 ± 0.005 | 2.00 | 100% |
| GMM-BIC | 0.901 ± 0.217 | 2.11 | 89% |

Bản gốc đạt 0.977, khớp mốc của bài báo (0.96 ± 0.03, Bảng V): bộ sinh tái tạo đúng chế độ dữ liệu.
Hình: `results/gaussian_noise_nc2_l2.png`.

Tách thành phần (9 bộ × 2 seed):

| Cấu hình | ARI | Số cụm TB | Đúng 2 cụm |
|---|---|---|---|
| NKCV2 gốc, Standard | 0.976 ± 0.012 | 4.83 | 44% |
| **NKCV2 + gieo + gộp**, Standard | **0.975 ± 0.010** | **2.33** | **72%** |
| NKCV2 + gieo + gộp, MinMax | 0.975 ± 0.013 | 2.39 | 61% |
| CDW, không gieo, không gộp | 0.888 ± 0.094 | 13.6 | 0% |
| CDW + gộp, không gieo | 0.815 ± 0.214 | 2.61 | 39% |

**CDW là nguyên nhân giảm ARI** trên dữ liệu Gauss tách rời có nhiễu, đúng loại dữ liệu bài báo nói
NK-HGA mạnh nhất. CDW xé cụm thưa thành các mảnh lớn (cùng kiểu lỗi với Jain), và bước gộp không sửa
được vì các mảnh không nhỏ. NKCV2 + gieo + gộp giữ nguyên ARI của bài báo và tăng tỉ lệ đúng số cụm
từ 44% lên 72%.

Hệ quả: **chưa có cấu hình nào thắng trên mọi loại dữ liệu.** CDW thắng chồng lấn (S2–S4), NKCV2 thắng
Gauss tách rời có nhiễu, Jain và giữ đúng số cụm.

## 7g. Chốt phiên bản hiện hành (hướng A)

Quyết định: **NKCV2 gốc + gieo quần thể + gộp mảnh vụn + PX nhanh + tiền xử lý (bỏ cột gần hằng số,
MinMax)**. CDW quay về `lich_su/` vì nó làm giảm ARI trên mô hình Gauss có nhiễu của bài báo (7f).
Các bảng cũ ở 7e được chuyển sang `lich_su/results/ban_cu_final_table_*.csv`.

Đóng góp giữ lại cho bài báo:

| Đóng góp | Bằng chứng | Vai trò |
|---|---|---|
| Gieo quần thể + gộp mảnh vụn | Flame 0.013 → 0.932; Aggregation 0.903 → 0.988; Gauss của bài báo giữ ARI 0.975, đúng số cụm 18% → 60% | đóng góp chính: sửa giới hạn bài báo tự nêu, không làm hỏng điểm mạnh |
| PX `O(N log N)` | trùng từng bit với `src/px.py`; S1 N=5000 chạy 37 s | đóng góp phụ: khả năng mở rộng |
| Chẩn đoán NKCV2 | A1 so với nhãn thật và lời giải GMM (7c, 7d.4) | phần phân tích |
| Tiền xử lý | Ecoli 0.038 → 0.442 | chi tiết thực nghiệm |

### Bảng cuối với bản đã chốt (5 seed; S1–S4 mẫu 1000 điểm)

`experiments/run_final_table.py`. ARI trung bình ± SD (số cụm). `trước gộp` là cùng lần chạy, nhãn GA
trước bước gộp mảnh vụn.

| Bộ (k thật) | Gốc | Cải tiến, trước gộp | **Cải tiến** | HDBSCAN | GMM-BIC |
|---|---|---|---|---|---|
| Flame (2) | 0.013 (1.8) | 0.923 (2.8) | **0.932** (2.0) | 0.409 (2) | 0.600 (3) |
| Aggregation (7) | 0.903 ± 0.05 (6.0) | 0.988 (7.0) | **0.988** (7.0) | 0.734 (4) | 0.568 (11) |
| Jain (2) | 0.404 ± 0.20 (7.6) | 0.306 (9.0) | 0.308 (7.6) | **0.931** (2) | 0.293 (5) |
| Iris (3) | 0.526 (5.8) | 0.545 (4.8) | 0.559 (2.8) | **0.568** (2) | **0.568** (2) |
| Ecoli (8) | 0.038 (2.8) | 0.440 (7.6) | 0.442 (2.4) | 0.367 (2) | **0.641** (6) |
| Glass (6) | 0.129 (5.8) | **0.211** (6.8) | 0.205 (3.0) | **0.252** (2) | 0.152 (3) |
| S1 (15) | 0.905 (14.0) | 0.905 (14.0) | 0.905 (14.0) | 0.875 (15) | **0.976** (15) |
| S2 (15) | 0.407 (8.6) | 0.416 (8.0) | 0.416 (8.0) | 0.563 (15) | **0.880** (14) |
| S3 (15) | 0.030 (4.4) | 0.398 (9.0) | 0.399 (7.2) | 0.219 (15) | **0.604** (11) |
| S4 (15) | 0.049 (6.0) | 0.047 (5.8) | 0.047 (3.6) | 0.183 (15) | **0.550** (11) |

Mô hình Gauss có nhiễu, Nc = 2, l = 2 (9 bộ × 5 seed, `experiments/run_gaussian_noise.py`):

| Phương pháp | ARI | Số cụm TB | Đúng 2 cụm |
|---|---|---|---|
| Gốc | 0.977 ± 0.010 | 4.78 | 18% |
| Cải tiến, trước gộp | 0.976 ± 0.013 | 4.82 | 31% |
| **Cải tiến** | 0.975 ± 0.012 | **2.40** | **60%** |
| HDBSCAN | 0.981 ± 0.005 | 2.00 | 100% |
| GMM-BIC | 0.901 ± 0.217 | 2.11 | 89% |

Đọc:

- Bản cải tiến **không thua bản gốc trên bộ nào trừ Jain** (0.404 → 0.308; bản gốc dao động mạnh,
  SD 0.20). Thắng rõ trên Flame, Aggregation, Ecoli, Glass, S3; giữ nguyên ARI trên dữ liệu Gauss của
  bài báo.
- **Gộp mảnh vụn** kéo số cụm về gần đúng trên Flame, Iris, dữ liệu Gauss. Nhưng nó **gộp quá tay trên
  dữ liệu thực có lớp rất nhỏ**: Ecoli 7.6 → 2.4 cụm (thật 8, trong đó ba lớp chỉ có 5, 2, 2 điểm) và
  Glass 6.8 → 3.0 (thật 6). ARI gần như không đổi, nhưng số cụm sai. Cần nêu là giới hạn, hoặc xét lại
  ngưỡng 2% cho N nhỏ.
- HDBSCAN thắng Jain và tìm đúng số cụm trên dữ liệu Gauss; GMM-BIC thắng S1–S4. Hai mốc này phải có
  trong bài báo.
- Đây là kết quả 5 seed, chọn theo trung bình. Bài báo cần theo giao thức gốc: 25 lần chạy, chọn lời
  giải bằng DBCV, kiểm định Wilcoxon/Friedman, đủ 144 bộ Gauss.

## 7h. NKCV-M: định nghĩa cụm theo đỉnh mật độ (hướng chốt)

Hướng chốt: **gieo quần thể + tiền xử lý + fitness NKCV-M** (`models/nkcv2_mode.py`). PX nhanh và gộp
mảnh vụn giữ làm thành phần phụ; bảng ghi cả trước và sau bước gộp.

### 7h.1 Đỉnh có ý nghĩa theo đơn vị nhiễu

ToMATo trên đồ thị kNN đối xứng (k = 10), độ sâu yên ngựa đo bằng
`z = (ρ_thô(đỉnh) − ρ_thô(yên ngựa)) / √ρ_thô(đỉnh)`; giữ đỉnh khi `z > 2`. Lý do: persistence tuyệt
đối đánh giá thấp cụm thưa thật; persistence tương đối tạo đỉnh giả ở vùng thưa (Gauss cân bằng 2:
1.0; 0.92; 0.84…). Chia cho √ρ đưa độ sâu về đơn vị nhiễu Poisson của mật độ đếm.

| Bộ | Số đỉnh (z > 2) | Thật |
|---|---|---|
| S1 / S2 / S3 / S4 (mẫu 1000) | 15 / 15 / 14 / 13 | 15 |
| Aggregation / Jain / Iris | 7 / 2 / 2 | 7 / 2 / 3 |
| Flame | 1 (z = 1.8) | 2 |
| Gauss có nhiễu Nc = 2 | 2 ở 8/9 bộ | 2 |

### 7h.2 Hàm mục tiêu

Trên cạnh Gep vượt yên ngựa giữa hai đỉnh có ý nghĩa: cùng nhãn → `ρ_j`, khác nhãn → `0`. Mọi cạnh
khác giữ NKCV2. Không thêm số hạng; `z0=None` trùng NKCV2. `self_test`: trùng NKCV2 khi tắt, Delta
Evaluation khớp, PX trả fitness đúng và con không tệ hơn cha mẹ, `fix_labels` giữ fitness.

### 7h.3 Kết quả (5 seed; S1–S4 mẫu 1000 điểm)

`experiments/run_final_table.py`. `NKCV2+gieo` là bản chốt ở 7g; `ToMATo` là chính các lưu vực đỉnh mà
NKCV-M dùng.

| Bộ (k thật) | Gốc | NKCV2 + gieo | **NKCV-M + gieo** | ToMATo (z>2) | HDBSCAN | GMM-BIC |
|---|---|---|---|---|---|---|
| Flame (2) | 0.013 (1.8) | **0.932** (2.0) | **0.932** (2.0) | 0.000 (1) | 0.409 (2) | 0.600 (3) |
| Aggregation (7) | 0.903 (6.0) | 0.988 (7.0) | **0.997** (7.0) | 0.991 (7) | 0.734 (4) | 0.568 (11) |
| Jain (2) | 0.404 (7.6) | 0.308 (7.6) | 0.322 (8.8) | 0.621 (2) | **0.931** (2) | 0.293 (5) |
| Iris (3) | 0.526 (5.8) | 0.559 (2.8) | 0.559 (2.8) | **0.568** (2) | **0.568** (2) | **0.568** (2) |
| Ecoli (8) | 0.038 (2.8) | 0.442 (2.4) | 0.473 (2.4) | 0.317 (2) | 0.367 (2) | **0.641** (6) |
| Glass (6) | 0.129 (5.8) | 0.205 (3.0) | 0.205 (3.0) | 0.000 (1) | **0.252** (2) | 0.152 (3) |
| S1 (15) | 0.905 (14.0) | 0.905 (14.0) | **0.975** (15.0) | 0.958 (15) | 0.875 (15) | **0.976** (15) |
| S2 (15) | 0.407 (8.6) | 0.416 (8.0) | **0.888** (15.0) | 0.862 (15) | 0.563 (15) | 0.880 (14) |
| S3 (15) | 0.030 (4.4) | 0.399 (7.2) | 0.599 (13.0) | 0.568 (14) | 0.219 (15) | **0.604** (11) |
| S4 (15) | 0.049 (6.0) | 0.047 (3.6) | **0.546** (14.4) | 0.483 (13) | 0.183 (15) | **0.550** (11) |

Mô hình Gauss có nhiễu, Nc = 2, l = 2 (9 bộ × 5 seed):

| Phương pháp | ARI | Số cụm TB | Đúng 2 cụm |
|---|---|---|---|
| Gốc | 0.977 ± 0.010 | 4.78 | 18% |
| NKCV2 + gieo | 0.975 ± 0.012 | 2.40 | 60% |
| **NKCV-M + gieo** | 0.976 ± 0.011 | 2.36 | **64%** |
| ToMATo (z > 2) | 0.869 ± 0.307 | 1.89 | 89% |
| HDBSCAN | 0.981 ± 0.005 | 2.00 | 100% |
| GMM-BIC | 0.901 ± 0.217 | 2.11 | 89% |

Đọc:

- **Chồng lấn được giải mà không làm hỏng dữ liệu Gauss của bài báo.** S2 0.407 → 0.888, S3 0.030 →
  0.599, S4 0.049 → 0.546, ngang GMM-BIC (mô hình đúng giả định của dữ liệu). Số cụm 13–15 so với 15
  thật. Dữ liệu Gauss có nhiễu giữ ARI 0.976, đúng số cụm 18% → 64%. CDW không làm được điều này (7f).
- **NK-HGA vượt chính các lưu vực đỉnh** trên S1–S4, Aggregation, Ecoli và Gauss (ToMATo 0.869 ±
  0.307): phần tối ưu hóa có đóng góp thật, không chỉ chép lại lưu vực.
- **Còn yếu:** Jain (0.322; ToMATo 0.621 và HDBSCAN 0.931 đều hơn — GA làm tệ đi so với đầu vào lưu
  vực), Glass (số cụm 3 so với 6), và Flame chỉ đúng nhờ gieo (z = 1.8 nên không có đỉnh thứ hai).
- Bước gộp mảnh vụn chủ yếu sửa số cụm (S2 20.2 → 15.0; S4 28.2 → 14.4), ARI gần như không đổi.
- Kết quả 5 seed, chưa theo giao thức đầy đủ của bài báo (25 lần chạy, chọn bằng DBCV, 144 bộ Gauss,
  kiểm định thống kê).

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
