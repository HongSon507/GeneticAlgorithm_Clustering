# Hướng cải tiến NK-HGA — tài liệu tổng hợp

Tài liệu này là nguồn duy nhất mô tả toàn bộ nhánh nghiên cứu cải tiến hàm mục tiêu NKCV2:
cơ sở toán học, CDW, các hướng thất bại, kiểm thử tính đúng đắn, kết quả thực nghiệm, giới hạn,
cách chạy lại và gợi ý viết báo cáo. Mã gốc trong `src/` không bị sửa; mọi model thử nghiệm nằm
trong `models/` và được đưa vào NK-HGA bằng monkeypatch có hoàn nguyên trong `try/finally`.

> Bản Markdown chuyển đổi từ bài báo gốc được giữ riêng tại
> [`../docs/240203813v(truoc)1.md`](<../docs/240203813v(truoc)1.md>).

## 1. Kết luận nghiên cứu

1. Bản tái hiện NK-HGA là đúng và đạt ARI `0.9876`, 7 cụm trên Aggregation.
2. Lỗi chính trên Flame và Iris nằm ở **NKCV2**, không phải bộ tối ưu: nhãn thật có fitness xấu
   hơn lời giải GA; trên Flame, Spearman giữa fitness tốt nhất và ARI là `+0.999`.
3. CDW thay `α = ρ_j` bằng `α' = ρ_j w_ij` trên nhánh khác nhãn, với
   `w_ij = c_ij d_ij b_ij` được tính tĩnh trước khi chạy GA.
4. CDW thắng lớn trên Flame (`0.0128 → 0.9333`) và ổn định Aggregation, nhưng không giải đúng Iris
   và thất bại trên Jain (`0.3265 → 0.1462`). Đây là cải tiến **đúng nhưng hẹp**, không phải cải tiến
   tổng quát cho mọi dữ liệu.
5. Hạ cường độ nhánh khác nhãn xuống khoảng `0.25–0.28` tạo phần lớn bước nhảy trên Flame, nhưng
   cấu trúc theo cạnh vẫn có giá trị: full CDW hơn trọng số hằng `0.0424 ARI` và giảm SD khoảng 61%.
   Vì vậy chưa có căn cứ thay CDW bằng một hằng số.
6. CDW không phá Delta Evaluation hay Partition Crossover: 12/12 kiểm thử, bao phủ 10 giả định
   kiến trúc, đều PASS.

## 2. Ký hiệu

| Ký hiệu | Nghĩa |
|---|---|
| `N` | số đối tượng |
| `l` | số chiều dữ liệu |
| `y_i` | vector đặc trưng của đối tượng `i` |
| `x_i` | nhãn cụm của `i`; `x_i=0` nghĩa là nhiễu |
| `K` | số láng giềng trong mỗi hàm con NKCV2, mặc định `3` |
| `D_ij` | `||y_i-y_j||²`, khoảng cách Euclid **bình phương** |
| `ρ_i` | mật độ cục bộ chuẩn hóa của `i` |
| `M_i` | nhóm `K` đối tượng ảnh hưởng hàm con `f_i` |
| `R_i={p:i∈M_p}` | các hàm con đọc nhãn `x_i`, dùng cho Delta Evaluation |
| `Gep` | đồ thị tương tác có hướng; cạnh `(j,i)` nghĩa là `x_j` ảnh hưởng `f_i` |
| `dt0,dt1,dt2` | các ngưỡng khoảng cách tĩnh của NKCV2 |
| `d_rho` | ngưỡng mật độ để nhận diện nhiễu |
| `c_ij,d_ij,b_ij` | compactness, valley-density và boundary-distance của CDW |
| `w_ij` | trọng số tĩnh của cạnh `i→j` |

NKCV2 được **cực tiểu hóa**. Tất cả khoảng cách trong code và công thức là khoảng cách bình phương,
không khai căn.

## 3. NKCV2 gốc

### 3.1 Tiền xử lý

Khoảng cách:

```text
D_ij = ||y_i - y_j||²
```

Khoảng cách cắt theo quy tắc 2%. Gọi `vD` là các phần tử tam giác trên của ma trận khoảng cách đã
sắp tăng:

```text
position = floor(N(N-1)/2 × 2/100)
dc       = vD[position]
```

Mật độ Gauss:

```text
rho_raw_i = sum_{j != i} exp(-(D_ij/dc)²)
rho_i     = rho_raw_i / max_k(rho_raw_k)
```

Ngưỡng nhiễu:

```text
d_rho = mean(rho) - std(rho)
```

Nếu giá trị này không dương, code dùng `mean(rho)/2`.

Gọi `xg` là `N·K` khoảng cách trên các cạnh nhóm, có trung bình `m` và độ lệch chuẩn tổng thể `s`:

```text
dt0 = m
dt1 = m + 2s
dt2 = m + s
```

### 3.2 Đồ thị tương tác `Gep`

Mỗi đỉnh có đúng `K` cạnh vào và không có self-loop. Cạnh đầu tiên đến từ đối tượng gần nhất có
mật độ cao hơn; nếu không có thì dùng đối tượng gần nhất. Các vị trí còn lại được lấp bằng các
láng giềng gần nhất chưa được chọn. `Gep` là đồ thị có hướng và được tính một lần.

### 3.3 Hàm mục tiêu

```text
f(x) = sum_i f_i(x) / (N K)                     [cực tiểu hóa]
f_i(x) = sum_{j in M_i} contribution(i,j)
alpha = rho_j
```

Với `D=D_ij`, đóng góp của mỗi cạnh gồm ba nhánh:

```text
1. Nhiễu, x_i = 0
   contribution = 0      nếu rho_i <= d_rho và D > dt2
   contribution = alpha  ngược lại

2. Cùng nhãn, x_i = x_j != 0
   contribution = 0                                  nếu D <= dt0
   contribution = alpha (D-dt0)/(dt1-dt0)            nếu dt0 < D <= dt1
   contribution = alpha                              nếu D > dt1

3. Khác nhãn, x_i != x_j và cả hai khác 0
   contribution = alpha                              nếu D <= dt0
   contribution = alpha [1-(D-dt0)/(dt1-dt0)]        nếu dt0 < D <= dt1
   contribution = 0                                  nếu D > dt1
```

Nhánh cùng nhãn phạt hai điểm cùng cụm nhưng quá xa. Nhánh khác nhãn phạt hai điểm khác cụm nhưng
quá gần. `dt2` chỉ xuất hiện trong nhánh nhiễu.

### 3.4 Delta Evaluation

Đổi `x_i` từ nhãn `a` sang `b` chỉ ảnh hưởng `f_i` và các `f_p` với `p∈R_i`:

```text
Delta = [f_i(b)-f_i(a)] + sum_{p in R_i}[f_p(b)-f_p(a)]
```

Chi phí là `O(Kout·K)` thay vì `O(N·K)`, với `Kout=max_i |R_i|`.

## 4. Hạn chế gốc của NKCV2

### 4.1 Fitness không đại diện đúng ARI

| Bộ dữ liệu | `f(nhãn thật)` | `f(GA)` | Nhận xét |
|---|---:|---:|---|
| Aggregation | khoảng 0.008 | thấp hơn khoảng 13% | GA vẫn khớp nhãn thật |
| Flame | 0.008818 | 0.007220 | lời giải sai được NKCV2 ưu tiên khoảng 22% |
| Iris | 0.018328 | 0.007545 | lời giải sai được ưu tiên khoảng 143% |

Tương quan đo trên toàn quỹ đạo:

| Bộ | Pearson, best-so-far | Spearman, best-so-far | Pearson, toàn quần thể | Spearman, toàn quần thể |
|---|---:|---:|---:|---:|
| Aggregation | -0.614 | **-0.987** | -0.529 | -0.908 |
| Flame | +0.869 | **+0.999** | +0.036 | -0.041 |
| Iris | -0.944 | **+0.623** | -0.700 | +0.332 |

Nếu fitness tốt, fitness giảm phải đi với ARI tăng, tức tương quan âm. Flame có Spearman `+0.999`:
GA càng tối ưu NKCV2 thì ARI càng giảm gần như đơn điệu. Iris có các hệ số mâu thuẫn dấu, cho thấy
quan hệ không đơn điệu. Vì vậy không thể quy lỗi cho mutation, crossover hay khả năng tìm kiếm.

### 4.2 Bốn giả định ngầm

| Giả định | Khi bị vi phạm |
|---|---|
| Số mode mật độ xấp xỉ số cụm thật | Iris có 2 mode nhưng 3 lớp, gây phân mảnh/ghép sai |
| Rất ít cạnh `Gep` vượt biên thật | Flame cho phép dịch biên đến vị trí rẻ hơn |
| Mật độ trong mỗi cụm tương đối đều | cụm loãng bị nuốt hoặc bị xé |
| Số chiều đủ thấp để tránh hubness | `Kout` có đuôi lớn, một số điểm chi phối nhiều hàm con |

Aggregation thỏa khá tốt các giả định này; chỉ `13/2364 = 0.55%` cạnh vượt biên thật. ARI cao trên
Aggregation chứng minh dữ liệu phù hợp giả định, không chứng minh NKCV2 mạnh trên mọi hình học.

### 4.3 Hai kiểu thất bại khác nhau

| Kiểu | Dữ liệu | Triệu chứng | Tín hiệu cần có |
|---|---|---|---|
| Dời biên | Flame | số cụm gần đúng nhưng ranh giới sai | định vị biên |
| Phân mảnh/chồng lấn | Iris | quá nhiều cụm và nhiễu | hình dạng/hướng |

Các phương pháp dựa thuần mật độ có trần thực nghiệm khoảng `0.57` trên Iris: NK-HGA `0.5231`,
DBSCAN `0.5536`, ToMATo `0.5681`, trong khi GMM full covariance đạt `0.9039`.

## 5. Ràng buộc kiến trúc đối với mọi cải tiến

Delta Evaluation và Partition Crossover chỉ đúng nếu `D`, `rho`, các ngưỡng, `Gep` và mọi hệ số
trong `comp_fi` là **hằng số theo lời giải x**.

Nếu một trọng số đọc phân hoạch hiện tại:

- đổi một nhãn có thể làm trọng số của nhiều cạnh đổi, khiến Delta Evaluation bỏ sót chi phí;
- phần chi phí được PX xem là chung giữa hai cha mẹ không còn chung;
- fitness con tính bằng tổng cục bộ có thể khác fitness tính lại từ đầu.

Do đó hiệp phương sai theo cụm hiện tại, compactness theo cụm hiện tại và ngưỡng theo cụm hiện tại
đều bị loại. Khe hở hợp lệ là các đại lượng **cục bộ theo điểm**, được tính đúng một lần trước GA.

## 6. CDW: Context-Dependent Weighting

### 6.1 Thay đổi duy nhất

NKCV2 gốc dùng:

```text
alpha_ij = rho_j
```

CDW dùng:

```text
alpha'_ij = rho_j w_ij
```

Cấu hình chính chỉ áp dụng `w_ij` vào nhánh **khác nhãn**. Nhánh nhiễu và cùng nhãn giữ nguyên.

### 6.2 Compactness `c_ij`

Đặt `k_c=max(2K,10)`. Với mỗi điểm `i`, lấy `k_c` láng giềng gần nhất theo `D`:

```text
scale_i = (1/k_c) sum_{p in kNN_kc(i)} D_ip
scale_med = median_i(scale_i)
c_i = 1 / (1 + scale_i/scale_med)
c_ij = sqrt(c_i c_j)
```

- `scale_i` nhỏ: vùng quanh `i` chặt.
- `c_i` gần 1: điểm nằm trong vùng chặt; `c_i` gần 0: vùng loãng.
- Trung bình nhân `sqrt(c_i c_j)` đối xứng, dương và giảm nếu một đầu cạnh loãng.
- Đây là compactness **tĩnh của lân cận điểm**, không phải compactness của cụm hiện tại.

### 6.3 Valley-density `d_ij`

Trung điểm cạnh:

```text
m_ij = (y_i+y_j)/2
```

Mật độ Gauss chưa chuẩn hóa tại trung điểm:

```text
rho_mid_raw_ij = sum_p exp(-( ||y_p-m_ij||² / dc )²)
rho_mid_ij = rho_mid_raw_ij / max_k(rho_raw_k)
d_ij = rho_mid_ij / max(rho_i,rho_j,epsilon)
```

- `d_ij` nhỏ: mật độ giữa hai đầu thấp hơn mật độ đầu cạnh, có dấu hiệu thung lũng.
- `d_ij` gần hoặc lớn hơn 1: đoạn giữa không phải vùng mật độ thấp.
- `epsilon=1e-12` tránh chia cho 0.
- Thành phần này có chi phí tiền xử lý lớn nhất và ablation cho thấy đóng góp riêng yếu.

### 6.4 Boundary ramp `b_ij`

```text
t_ij = clip((D_ij-dt0)/(dt1-dt0),0,1)
b_ij = b_min + (1-b_min)t_ij
```

Với `b_min=0.5`:

- `D_ij <= dt0` cho `b_ij=0.5`;
- `dt0 < D_ij < dt1` tăng tuyến tính;
- `D_ij >= dt1` cho `b_ij=1`.

Nó dùng lại hai ngưỡng của NKCV2 nên không tạo thêm ngưỡng khoảng cách.

### 6.5 Ghép ba thành phần

```text
raw_w_ij = clip(c_ij d_ij b_ij, w_min, w_max)
w_ij(lambda) = (1-lambda) + lambda raw_w_ij
```

Cấu hình thắng:

```text
lambda = 1.0
w_min = 0.01
w_max = 5.0
branches = ('diff',)
```

Ý nghĩa từng biến:

- `raw_w_ij`: trọng số dữ liệu trước khi trộn với baseline.
- `w_min`: tránh làm cạnh mất hoàn toàn ảnh hưởng.
- `w_max`: chặn cạnh ngoại lai quá lớn.
- `lambda`: mức can thiệp; `0` là bản gốc, `1` là CDW đầy đủ.
- `branches`: xác định nhánh nào được nhân trọng số.

Bất biến quan trọng:

```text
lambda=0 => w_ij=1 => CDW trùng NKCV2 đến 1e-12
```

### 6.6 Hàm đóng góp sau CDW

Nếu chỉ bật nhánh khác nhãn:

```text
x_i != x_j, D <= dt0:
    contribution = rho_j w_ij

x_i != x_j, dt0 < D <= dt1:
    contribution = rho_j w_ij [1-(D-dt0)/(dt1-dt0)]

x_i != x_j, D > dt1:
    contribution = 0
```

Hai nhánh còn lại dùng đúng công thức NKCV2 gốc. CDW không thay mutation, crossover, mã hóa,
quần thể hay điều kiện dừng.

### 6.7 Tác dụng thực tế

Trên Flame, NKCV2 phạt quá mạnh nhiều cạnh khác nhãn ngắn nên ưu tiên dời biên. CDW hạ giá phần
lớn các cạnh này. Tuy nhiên cơ chế đo được không đơn giản là “phát hiện chính xác mọi biên”:

- hạ mức chung của nhánh khác nhãn tạo phần lớn bước nhảy;
- phân phối và vị trí trọng số theo cạnh bổ sung chất lượng và độ ổn định;
- CDW không thêm tín hiệu hình dạng, nên không giải được chồng lấn kiểu Iris;
- compactness làm CDW dễ phân mảnh cụm có mật độ biến đổi như Jain.

## 7. Sàng lọc A1

Trước khi chạy GA, tính fitness của nhãn thật và lời giải GA baseline dưới hàm mục tiêu ứng viên:

```text
diff% = 100 [f(true)-f(GA)] / f(GA)
```

`diff%<0` là PASS: hàm mục tiêu ít nhất đã xếp nhãn thật tốt hơn lời giải lỗi. Đây chỉ là điều kiện
cần. Số đo trên 8 variant:

| A1 | Số trường hợp | GA thất bại | GA thành công |
|---|---:|---:|---:|
| FAIL | 3 | 3 | 0 |
| PASS | 6 | 4 | 2 |

A1 là bộ lọc loại tốt, không phải bộ dự báo thắng và không dùng để xếp hạng variant.

Kết quả CDW chính:

| Bộ | `f(true)` | `f(GA baseline)` | diff% |
|---|---:|---:|---:|
| Flame | 0.005461 | 0.007220 | **-24.4% PASS** |
| Aggregation | 0.003106 | 0.003024 | +2.7% FAIL |
| Iris | 0.012825 | 0.008071 | +58.9% FAIL |
| Jain | 0.008602 | 0.004259 | +102.0% FAIL |

## 8. Kết quả GA và khả năng tổng quát

Ba seed `100,101,102`, `pop=60`, `gen=80`:

| Bộ | ARI gốc TB | ARI CDW TB | Chênh | cụm gốc TB | cụm CDW TB | cụm thật |
|---|---:|---:|---:|---:|---:|---:|
| Flame | 0.0128 | **0.9333** | +0.9206 | 2.0 | 3.0 | 2 |
| Aggregation | 0.9333 | **0.9809** | +0.0476 | 6.3 | 7.0 | 7 |
| Iris | 0.5278 | 0.5648 | +0.0370 | 6.3 | **12.0** | 3 |
| Jain | **0.3265** | 0.1462 | -0.1803 | 7.7 | **26.3** | 2 |

Đọc đúng:

- Flame: thắng lớn nhưng thường tạo 3 cụm thay vì 2.
- Aggregation: chủ yếu giảm phương sai; hai seed baseline đã tốt hơi giảm, seed baseline sập được cứu.
- Iris: ARI tăng nhẹ nhưng số cụm tăng gần gấp đôi; đây không phải cải thiện cấu trúc.
- Jain: thua cả 3 seed và phân mảnh 26–27 cụm; CDW không tổng quát hóa.

Hình tổng kết: [`results/cdw_ari_4datasets.png`](results/cdw_ari_4datasets.png).

## 9. Ablation tám biến thể

Mọi thành phần không dùng được thay bằng phần tử trung hòa `1`. Thí nghiệm Flame, seed 100–101,
`pop=40`, `gen=60`:

| Variant | `c` | `d` | `b` | ARI TB | diff% |
|---|:-:|:-:|:-:|---:|---:|
| baseline | - | - | - | 0.0128 | +21.9% |
| compactness only | ✓ | - | - | 0.0095 | -15.2% |
| density only | - | ✓ | - | 0.0128 | +30.3% |
| boundary only | - | - | ✓ | 0.0128 | -6.5% |
| compactness × density | ✓ | ✓ | - | 0.0128 | -12.0% |
| compactness × boundary | ✓ | - | ✓ | **0.8518** | -26.1% |
| density × boundary | - | ✓ | ✓ | 0.0128 | -2.2% |
| full CDW | ✓ | ✓ | ✓ | **0.9307** | -24.5% |

Không thành phần đơn lẻ nào tạo ra cải thiện. Trong họ tám biến thể, chỉ hai biến thể chứa đồng thời
`c×b` thắng; bỏ `d` vẫn giữ khoảng 91% kết quả. Tuy nhiên bảng này đồng thời thay đổi cấu trúc và
trung bình trọng số, nên chưa chứng minh hiệu ứng nhân quả là tương tác hình học.

## 10. Đối chứng cường độ và cấu trúc theo cạnh

### 10.1 Screening ban đầu

| Giao thức | Baseline | `w=.50` | `w=.25` | `w=.28407` | `c×b` | Full |
|---|---:|---:|---:|---:|---:|---:|
| 2 seed, pop40/gen60 | .0128 | .0128 | .8367 | .9144 | .8518 | .9307 |
| 3 seed, pop60/gen80 | .0128 | .0128 | .8645 | .9225 | .9017 | .9333 |

`mean(raw_w)` của full CDW trên Flame là `0.284070`. Kết quả nhỏ cho thấy trọng số hằng giữ hơn
98% bước nhảy, nhưng số seed chưa đủ để quyết định xóa cấu trúc.

### 10.2 Kiểm định cấu trúc 10 seed

`shuffled_cdw` giữ nguyên chính xác phân phối, min, max và mean của `raw_w`, nhưng hoán vị trọng số
giữa các cạnh. Tiêu chí tương đương được đặt trước là CI 95% của chênh lệch phải nằm hoàn toàn trong
`±0.02 ARI`.

| Variant | ARI TB | SD | `full - variant` | bootstrap 95% | tương đương? |
|---|---:|---:|---:|---:|---|
| Baseline | 0.1037 | 0.2877 | - | - | - |
| Hằng `w=.284070` | 0.8820 | 0.0686 | **+0.0424** | **[+0.0162,+0.0721]** | không |
| Shuffled CDW | 0.9054 | 0.0477 | +0.0189 | [-0.0095,+0.0499] | chưa chứng minh |
| **Full CDW** | **0.9243** | **0.0266** | - | - | mốc |

Kết luận giới hạn trên Flame:

- giảm cường độ khoảng 3.5–4 lần tạo phần lớn bước nhảy;
- phân phối/vị trí theo cạnh thêm `0.0424 ARI` so với hằng mean;
- SD của full thấp hơn trọng số hằng khoảng 61%;
- cả đối chứng hằng và hoán vị đều không đạt tiêu chí tương đương.

Do đó giữ full `c_ij d_ij b_ij`; không thay bằng hằng số. Đối chứng này chưa chạy trên ba bộ còn lại.

## 11. Kiểm định Partition Crossover và Delta Evaluation

Mười giả định được kiểm:

1. hàm mục tiêu vẫn là tổng các `f_i`;
2. `f_i` chỉ đọc `x_i` và `x_j` với `j∈M_i`;
3. các mảng CDW không thay đổi giữa các lần đánh giá;
4. tổng cục bộ PX bằng fitness tính lại từ đầu;
5. `fix_labels` bảo toàn phân hoạch và fitness;
6. `map_solutions` bảo toàn fitness;
7. `renumber` bảo toàn fitness;
8. không tạo NaN/Inf;
9. PX thực sự gọi `comp_fi` đa hình của CDW;
10. fitness lưu trong GA khớp fitness tính lại từ nhãn cuối.

Các kiểm thử bổ sung xác nhận Delta Evaluation khớp full reevaluation. Kết quả: **12/12 PASS** trên
Aggregation, Flame, Iris và dữ liệu tổng hợp. Vì `w_ij` là mảng `(N,K)` tĩnh, locality của `f_i`
không đổi; PX và Delta Evaluation vẫn đúng về toán học.

## 12. Các hướng đã thử nhưng không trở thành cải tiến chính

- **K thích nghi theo điểm:** chỉ đổi độ mịn của đồ thị, không sửa thứ tự fitness giữa nhãn thật và
  lời giải lỗi.
- **Ngưỡng khoảng cách cục bộ:** làm các `f_i` mất tính so sánh và vẫn không làm biên thật rẻ hơn
  biên giả.
- **Dựng `Gep` theo thung lũng:** sửa nhầm chỗ vì `Gep` Flame vốn chỉ có 13/720 cạnh vượt biên.
- **Cổng thung lũng cục bộ:** 10/10 cạnh biên ngắn cần sửa có tín hiệu valley bằng 0.
- **Mutual reachability:** nâng nền khoảng cách nhưng không tạo nút cổ chai đúng vị trí biên.
- **MST minimax:** loại được cạnh gần khác cụm nhưng làm `same-far` tăng mạnh và vẫn sai nhiều cạnh.
- **Persistence ToMATo:** thắng nhẹ Aggregation (`0.9876→0.9942`), nhưng Flame đạt khoảng 0.65 bằng
  cách vỡ thành 15 cụm; tham số không cực đoan thì gần như vô hiệu.
- **Mahalanobis cục bộ:** có tín hiệu hình dạng đúng chiều nhưng quá yếu; đảo dấu giúp Flame nhưng
  đánh đổi Aggregation/Iris và vẫn không qua A1.

### 12.1 Lỗi công thức Mahalanobis đã phát hiện

Thiết kế thô dùng covariance `Sigma_i`. Trong vùng đẳng hướng `Sigma_i=sigma²I`, tỷ lệ Euclid trên
Mahalanobis rút gọn thành `sigma²`, nên nó đo mật độ/thể tích thay vì hình dạng. Bản kiểm thử đã sửa:

```text
Sigma_hat_i = Sigma_i / det(Sigma_i)^(1/l)
Sigma_bar_ij = (Sigma_hat_i+Sigma_hat_j)/2
D_maha_ij = (y_i-y_j)^T Sigma_bar_ij^-1 (y_i-y_j)
r_ij = clip(D_ij/D_maha_ij,w_min,w_max)
```

Sau chuẩn hóa, vùng đẳng hướng nhân tạo cho `r=1`; Flame có median `1.006`, p10–p90 `0.81–1.33`.
Tín hiệu đã đúng nghĩa hình dạng nhưng chưa đủ mạnh để sửa fitness.

### 12.2 Persistence ToMATo

Điểm được duyệt theo mật độ giảm dần trên đồ thị kNN. Khi hai thành phần gặp nhau, mode yếu chết tại
điểm yên ngựa, với:

```text
persistence = rho(peak) - rho(merge_point)
```

Ngưỡng `tau` lấy tự động từ khe hở lớn nhất của persistence diagram, không dùng nhãn thật. Cạnh giữa
hai mode được nhân `w_cross`; `w_cross=1` tái tạo baseline. Hướng này có ý nghĩa tô-pô nhưng không
giải được ranh giới Flame nằm trong cùng một mode mật độ.

## 13. `Kout` và các hạn chế kỹ thuật nhỏ

| Bộ | N | d | `Kout max`, K=3 | `Kout` trung bình |
|---|---:|---:|---:|---:|
| Flame | 240 | 2 | 6 | 3.0 |
| Aggregation | 788 | 2 | 6 | 3.0 |
| Iris | 150 | 4 | 7 | 3.0 |
| blobs | 500 | 64 | 29 | 3.0 |

Luôn có `sum_i |R_i|=NK`, nên trung bình `|R_i|=K`. Chặn `Kout` chủ yếu xử lý worst-case/hubness,
không sửa lỗi ARI trên Flame hoặc Iris. Giao thức dừng `N/2` giây của bài báo cũng không tái lập hoàn
toàn giữa phần cứng; thí nghiệm nghiên cứu dùng số thế hệ cố định.

## 14. Cấu trúc mã nguồn

```text
huong_cai_tien/
├── README.md                         tài liệu tổng hợp này
├── models/                           model và hạ tầng hàm mục tiêu
│   ├── _bootstrap.py
│   ├── datasets.py
│   ├── weighted_model.py
│   ├── nkcv2_cdw.py
│   ├── nkcv2_persist.py
│   ├── nkcv2_mahalanobis.py
│   └── ablation_model.py
├── experiments/                      screening, GA, ablation và kiểm thử
│   ├── _paths.py
│   ├── run_cdw_screen.py
│   ├── run_cdw_ga.py
│   ├── run_cdw_jain.py
│   ├── run_cdw_ari_plot.py
│   ├── run_ablation.py
│   ├── test_constant_weight_hypothesis.py
│   ├── run_persist.py
│   ├── run_fitness_ari_correlation.py
│   ├── kout_profile.py
│   └── test_px_correctness.py
└── results/
    ├── ablation_results.csv
    ├── cdw_ari_4datasets.png
    └── kout_profile.png
```

`WeightedNKCV2Model` là lớp chung duy nhất cài ba kernel có trọng số. Các model con chỉ tính
`raw_w`. `datasets.py` là nguồn nạp bốn bộ dữ liệu đã chuẩn hóa.

## 15. Cách tái lập

Từ thư mục gốc, dùng môi trường Python đã cài dependencies:

```text
python tests/run_tests.py
python huong_cai_tien/experiments/test_px_correctness.py
python huong_cai_tien/models/nkcv2_cdw.py
python huong_cai_tien/models/ablation_model.py
python huong_cai_tien/models/nkcv2_mahalanobis.py

python huong_cai_tien/experiments/run_cdw_screen.py
python huong_cai_tien/experiments/run_cdw_ga.py --runs 3
python huong_cai_tien/experiments/run_cdw_jain.py
python huong_cai_tien/experiments/run_cdw_ari_plot.py
python huong_cai_tien/experiments/run_ablation.py --datasets flame --seeds 100 101 --tag screening
python huong_cai_tien/experiments/test_constant_weight_hypothesis.py --structure-only
```

`run_cdw_ari_plot.py` chỉ sinh PNG và phải chạy lại GA để vẽ. `nkcv2_persist.py` không có khối
`__main__`; chạy kiểm thử của nó qua `experiments/run_persist.py`.

## 16. Quy tắc nghiên cứu

1. Không sửa `src/` khi thử objective mới.
2. Model mới phải có tham số trung tính tái tạo baseline trong `1e-12`.
3. `df_element` phải khớp full reevaluation qua hàng trăm nước đi ngẫu nhiên.
4. Trọng số phải tĩnh theo `x` để bảo toàn Delta Evaluation và PX.
5. A1 phải chạy trước GA, nhưng chỉ dùng làm bộ lọc loại.
6. ARI chỉ dùng đánh giá ngoài, không dùng chọn model trong quá trình tối ưu.
7. Phải ghi dataset, seed, `K`, population, generation và stopping criterion cho mọi bảng.

## 17. Khung báo cáo khóa luận đề xuất

### Chương 1 — Mở đầu

- Bài toán phân cụm và hạn chế của tiêu chí tâm cụm.
- NK-HGA, NKCV2 và mục tiêu tái hiện.
- Đóng góp riêng: chẩn đoán objective, quy trình A1, thiết kế và kiểm định CDW.

### Chương 2 — Cơ sở lý thuyết

- Mã hóa phân hoạch, tính dư thừa nhãn và `renumber`.
- Mô hình NK và hàm mục tiêu phân rã.
- Density Peaks, `dc`, `rho`, `Gep` và NKCV2.
- Local search, mutation, Partition Crossover và Delta Evaluation.

### Chương 3 — Tái hiện và kiểm chứng

- Kiến trúc `src/` và đối chiếu mã C++ tham chiếu.
- Ba bất biến: Delta Evaluation, fitness PX, `renumber`.
- Kết quả Aggregation: NK-HGA 0.9876, CGA 0.7735, k-means 0.7113, DBSCAN 0.7338.

### Chương 4 — Phân tích giới hạn NKCV2

- So sánh fitness nhãn thật và lời giải GA.
- Tương quan fitness–ARI.
- Bốn giả định ngầm và hai kiểu thất bại.
- Trần mật độ trên Iris và ràng buộc kiến trúc.

### Chương 5 — Cải tiến CDW

- Động cơ thay `rho_j` bằng `rho_j w_ij`.
- Trình bày từng biến `c`, `d`, `b`, clipping, `lambda`, branches.
- Chứng minh trọng số tĩnh và không phá PX/Delta Evaluation.
- A1, kết quả GA và ablation.

### Chương 6 — Tổng quát hóa và bàn luận

- Aggregation, Flame, Iris và Jain.
- Đối chứng hằng/hoán vị để tách cường độ khỏi cấu trúc.
- Các hướng thất bại và bài học.
- Phạm vi áp dụng thực tế của CDW.

### Chương 7 — Kết luận

- NK-HGA tối ưu đúng objective nhưng objective có giả định hẹp.
- CDW giải được một kiểu dời biên trên Flame nhưng không tổng quát.
- Hướng tiếp theo phải thêm tín hiệu phân biệt biên thật với thay đổi mật độ dọc cụm mà vẫn tĩnh.

### Checklist trước khi nộp

- Không viết “CDW cải thiện NK-HGA nói chung”; phải nêu kết quả Jain.
- Không gọi ARI tăng nhẹ trên Iris là thành công khi số cụm tăng 6.3 lên 12.
- Không gọi A1 là bộ dự báo thắng.
- Ghi rõ khoảng cách bình phương và cấu hình diff-only.
- Phân biệt ablation 2 seed/pop40/gen60 với A9 3 seed/pop60/gen80.
- Nêu kiểm định cấu trúc 10 seed và bootstrap CI.
- Trích dẫn bài báo và repository gốc; bổ sung giấy phép phù hợp cho mã tham chiếu.

## 18. Hình và dữ liệu kết quả

| Tệp | Dùng cho báo cáo |
|---|---|
| [`results/cdw_ari_4datasets.png`](results/cdw_ari_4datasets.png) | ARI gốc và CDW trên bốn bộ |
| [`results/kout_profile.png`](results/kout_profile.png) | phân bố outdegree |
| [`results/ablation_results.csv`](results/ablation_results.csv) | 8 variant × 2 seed trên Flame |
| `../results/comparison_result.png` | NK-HGA, CGA, k-means và DBSCAN trên Aggregation |

## 19. Kết luận cuối

CDW hợp lệ về toán học và kiến trúc, có một kết quả mạnh trên Flame, nhưng bằng chứng Jain phủ định
tính tổng quát. Phân tích nhân quả cho thấy hạ trọng số nhánh khác nhãn là cơ chế chính, còn cấu trúc
theo cạnh giúp tăng thêm chất lượng và ổn định nên chưa thể xóa. Giá trị lớn nhất của nghiên cứu không
chỉ là một công thức mới, mà là chuỗi bằng chứng chỉ ra **khi nào NKCV2 sai, vì sao sai, cách loại sớm
một cải tiến không khả thi và cách kiểm tra objective mới mà không phá kiến trúc NK-HGA**.
