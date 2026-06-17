# ANFIS Load Forecasting

## 1. Giới thiệu dự án

ANFIS Load Forecasting là đề tài nghiên cứu ứng dụng mô hình **Adaptive Neuro-Fuzzy Inference System (ANFIS)** trong bài toán dự báo phụ tải điện ngắn hạn cho hộ gia đình.

Do hạn chế về nguồn dữ liệu thực tế, nghiên cứu sử dụng bộ dữ liệu được xây dựng từ dữ liệu thời tiết lịch sử tại Hà Nội kết hợp với mô hình mô phỏng hành vi tiêu thụ điện của hộ gia đình. Trên cơ sở đó, nhóm triển khai quy trình tiền xử lý dữ liệu, xây dựng tập đặc trưng, huấn luyện mô hình ANFIS và so sánh với các mô hình Machine Learning phổ biến.

Các mốc dự báo được nghiên cứu gồm:

* Dự báo phụ tải điện 1 giờ tiếp theo (1-hour ahead forecasting)
* Dự báo phụ tải điện 24 giờ tiếp theo (24-hour ahead forecasting)

Các mô hình được triển khai trong dự án:

* Linear Regression
* Decision Tree Regression
* Random Forest Regression
* XGBoost Regression
* ANFIS (Adaptive Neuro-Fuzzy Inference System)

---

## 2. Mục tiêu nghiên cứu

### Mục tiêu tổng quát

Xây dựng hệ thống dự báo phụ tải điện ngắn hạn và đánh giá khả năng ứng dụng của mô hình ANFIS trong bài toán dự báo chuỗi thời gian.

### Mục tiêu cụ thể

* Thu thập dữ liệu thời tiết lịch sử tại Hà Nội.
* Mô phỏng dữ liệu tiêu thụ điện hộ gia đình.
* Xây dựng pipeline tiền xử lý dữ liệu.
* Sinh đặc trưng phục vụ dự báo phụ tải điện.
* Xây dựng Core Dataset và Extended Dataset.
* Huấn luyện các mô hình Baseline.
* Huấn luyện mô hình ANFIS.
* Đánh giá và so sánh kết quả thực nghiệm.

---

## 3. Workflow nghiên cứu

```text
Thu thập dữ liệu thời tiết
            ↓
Mô phỏng dữ liệu phụ tải điện
            ↓
Tiền xử lý dữ liệu
            ↓
Sinh đặc trưng
            ↓
Tạo Core Dataset
            ↓
Tạo Extended Dataset
            ↓
Huấn luyện Baseline Models
            ↓
Huấn luyện ANFIS
            ↓
Đánh giá mô hình
            ↓
So sánh kết quả
            ↓
Phân tích kết quả thực nghiệm
```

---

## 4. Cấu trúc dự án

```text
ANFIS/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── intermediate/
│
├── latex/
│   ├── figures/
│   ├── sections/
│   └── main.tex
│
├── results/
│
├── scripts/
│
├── src/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── evaluation/
│
├── test/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 5. Bộ dữ liệu

### Dữ liệu thời tiết

* Khu vực: Hà Nội, Việt Nam
* Giai đoạn: 2021 – 2025
* Nguồn dữ liệu: Open-Meteo API
* Tần suất: Theo giờ

Các thuộc tính thời tiết chính:

* Temperature
* Apparent Temperature
* Humidity
* Rainfall
* Wind Speed
* Cloud Cover

### Dữ liệu phụ tải điện

Dữ liệu phụ tải điện được mô phỏng dựa trên:

* Hồ sơ sinh hoạt hộ gia đình
* Mức độ hiện diện trong nhà
* Thời gian sử dụng thiết bị điện
* Hiệu ứng thời tiết
* Đặc trưng ngày trong tuần và mùa vụ

---

## 6. Bộ đặc trưng

### Core Features

Được sử dụng cho mô hình ANFIS nhằm hạn chế số lượng luật mờ phát sinh.

* apparent_temperature
* humidity
* hour_sin
* hour_cos
* occupancy_level
* load_lag_24

### Extended Features

Được sử dụng cho các mô hình Machine Learning baseline.

Bao gồm:

* Đặc trưng thời gian
* Đặc trưng hành vi sử dụng điện
* Đặc trưng thời tiết
* Lag Features
* Rolling Features

---

## 7. Các mô hình sử dụng

### Baseline Models

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* XGBoost Regressor

### Mô hình chính

* ANFIS (Adaptive Neuro-Fuzzy Inference System)

ANFIS được xây dựng dựa trên hệ suy luận mờ Sugeno kết hợp với khả năng học của mạng nơ-ron nhân tạo.

---

## 8. Cài đặt dự án

Clone repository:

```bash
git clone https://github.com/ngoclogg/anfis-load-forecasting.git
```

Di chuyển vào thư mục dự án:

```bash
cd anfis-load-forecasting
```

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

---

## 9. Hướng dẫn chạy dự án

### Thu thập dữ liệu thời tiết

```bash
python -m src.data.pipeline.get_hanoi_weather
```

### Mô phỏng dữ liệu phụ tải điện

```bash
python -m src.data.pipeline.build_hanoi_load_dataset
```

### Tiền xử lý dữ liệu

```bash
python -m src.data.pipeline.preprocess_hanoi_load_dataset
```

### Huấn luyện Baseline Models

```bash
python -m src.model.train_baselines
```

### Trực quan hóa kết quả Baseline

```bash
python -m src.model.visualize_baselines
```

### Huấn luyện ANFIS

```bash
python -m src.model.train_anfis_hourly
```

### Huấn luyện ANFIS với số Membership Functions khác

```bash
python -m src.model.train_anfis_hourly --n-mfs 3
```

---

## 10. Đánh giá mô hình

Các mô hình được đánh giá bằng các chỉ số:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* MAPE (Mean Absolute Percentage Error)
* R² Score

Kết quả chi tiết được lưu trong thư mục:

```text
results/
```

và các file báo cáo sinh tự động trong quá trình thực nghiệm.

---

## 11. Báo cáo nghiên cứu

Mã nguồn báo cáo được lưu trong thư mục:

```text
latex/
```

Nội dung báo cáo bao gồm:

* Chương 1: Tổng quan đề tài
* Chương 2: Cơ sở lý thuyết
* Chương 3: Dữ liệu và tiền xử lý
* Chương 4: Thiết kế mô hình
* Chương 5: Thực nghiệm
* Chương 6: Đánh giá kết quả
* Tổng kết và định hướng phát triển

---

## 12. Hạn chế của nghiên cứu

* Dữ liệu phụ tải điện được xây dựng bằng phương pháp mô phỏng.
* Chưa sử dụng dữ liệu công tơ điện thực tế.
* Chưa đánh giá trên nhiều khu vực địa lý khác nhau.
* ANFIS hiện chỉ sử dụng tập Core Features để kiểm soát độ phức tạp của hệ luật mờ.
* Kết quả thực nghiệm hiện tại mang tính nghiên cứu và đánh giá phương pháp.

---

## 13. Hướng phát triển

* Thu thập dữ liệu phụ tải điện thực tế.
* Tối ưu lựa chọn đặc trưng đầu vào cho ANFIS.
* Nghiên cứu các phương pháp sinh luật mờ hiệu quả hơn.
* Cá nhân hóa mô hình theo từng hộ gia đình.
* Xây dựng hệ thống dự báo phụ tải điện thời gian thực.

---

## 14. Công nghệ sử dụng

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* XGBoost
* ANFIS
* PyTest
* Git
* GitHub

---

## 15. Thành viên nhóm

Đề tài được thực hiện trong khuôn khổ môn học **Tính toán mềm (Soft Computing)**.

### Phân công công việc

* Xây dựng dữ liệu và tiền xử lý.
* Huấn luyện và đánh giá Baseline Models.
* Xây dựng và huấn luyện ANFIS.
* Phân tích và so sánh kết quả thực nghiệm.
* Xây dựng báo cáo nghiên cứu.

# 16. Kết quả thực nghiệm cuối cùng

Nghiên cứu được thực hiện trên bài toán dự báo phụ tải điện hộ gia đình theo hai horizon:

* Dự báo 1 giờ tới (1h Ahead Forecasting)
* Dự báo 24 giờ tới (24h Ahead Forecasting)

Các mô hình được đánh giá bằng:

* MAE
* RMSE
* MAPE
* R²

Kết quả thực nghiệm cho thấy:

* ANFIS có khả năng mô hình hóa mối quan hệ phi tuyến giữa thời tiết, thời gian và hành vi sử dụng điện.
* ANFIS đạt kết quả cạnh tranh trên cả hai horizon dự báo.
* Random Forest và XGBoost đạt độ chính xác cao hơn khi sử dụng Extended Features.
* ANFIS có ưu thế về khả năng diễn giải thông qua hệ luật mờ IF–THEN và cấu trúc Neuro-Fuzzy.

Cấu hình ANFIS cuối cùng:

* Input Features: 7
* Membership Functions / Feature: 2 Gaussian MFs
* Rule Generation: Grid Partition
* Total Fuzzy Rules: 2⁷ = 128
* Inference Type: First-order Sugeno

Core Features sử dụng cho ANFIS:

* apparent_temperature
* humidity
* hour_sin
* hour_cos
* occupancy_level
* load_lag_1
* load_lag_24

Kết quả chi tiết được lưu trong:

```text
results/1h/
results/24h/
```

bao gồm:

* metrics
* predictions
* plots
* trained models
* experiment artifacts

# 17. Các cải tiến so với bản nộp lần 1

Trong quá trình hoàn thiện dự án, nhóm đã thực hiện nhiều cải tiến quan trọng so với phiên bản ban đầu.

## Cải tiến dữ liệu

- Bổ sung đặc trưng load_lag_1 bên cạnh load_lag_24.
- Hoàn thiện pipeline sinh dữ liệu phụ tải điện.
- Chuẩn hóa quy trình tạo target_1h và target_24h.
- Kiểm tra và loại bỏ dữ liệu thiếu, dữ liệu trùng lặp.

## Cải tiến Feature Engineering

- Hoàn thiện nhóm Core Features cho ANFIS.
- Hoàn thiện nhóm Extended Features cho các baseline models.
- Bổ sung các đặc trưng thời tiết và hành vi sử dụng điện.
- Tách riêng dữ liệu Core và Extended để phục vụ các mục tiêu thực nghiệm khác nhau.

## Cải tiến mô hình ANFIS

- Chuẩn hóa kiến trúc ANFIS cuối cùng:
  - 7 input features
  - 2 Gaussian membership functions mỗi biến
  - 128 fuzzy rules

- Hoàn thiện cơ chế huấn luyện và lưu artifact.
- Tách riêng thực nghiệm cho horizon 1h và 24h.

## Cải tiến đánh giá thực nghiệm

- Bổ sung các mô hình baseline:
  - Linear Regression
  - Decision Tree
  - Random Forest
  - XGBoost

- Bổ sung đánh giá bằng:
  - MAE
  - RMSE
  - MAPE
  - R²

- Bổ sung các biểu đồ:
  - Actual vs Predicted
  - RMSE Comparison
  - R² Comparison
  - Residual Distribution

## Cải tiến báo cáo

- Viết lại Chương 3 theo cấu trúc nghiên cứu hoàn chỉnh:
  - Nguồn dữ liệu
  - Khám phá dữ liệu
  - Thiết kế đặc trưng
  - Tiền xử lý dữ liệu

- Hoàn thiện Chương 4 về kiến trúc ANFIS.
- Hoàn thiện Chương 5 về thực nghiệm và đánh giá.
- Bổ sung phần tổng kết và hướng phát triển.
- Đồng bộ toàn bộ thuật ngữ, ký hiệu và workflow trong báo cáo.

Những cải tiến trên giúp dự án chuyển từ một mô hình ANFIS thử nghiệm ban đầu thành một quy trình nghiên cứu hoàn chỉnh bao gồm xây dựng dữ liệu, tiền xử lý, huấn luyện, đánh giá và phân tích kết quả thực nghiệm.
````

