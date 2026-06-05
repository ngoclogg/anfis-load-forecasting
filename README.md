# ANFIS Load Forecasting

## 1. Giới thiệu dự án

ANFIS Load Forecasting là dự án xây dựng hệ thống dự báo phụ tải điện ngắn hạn cho hộ gia đình tại Hà Nội bằng mô hình ANFIS (Adaptive Neuro-Fuzzy Inference System) và các mô hình Machine Learning truyền thống.

Dự án mô phỏng dữ liệu tiêu thụ điện dựa trên:

* Dữ liệu thời tiết thực tế tại Hà Nội.
* Đặc trưng thời gian và mùa vụ.
* Hành vi sử dụng điện của hộ gia đình.
* Mức độ hiện diện của người dùng trong nhà.
* Các thiết bị điện và phụ tải sinh hoạt.

Ngoài ANFIS, dự án còn triển khai và đánh giá các mô hình:

* Linear Regression
* Decision Tree Regression
* Random Forest Regression
* XGBoost Regression

Mục tiêu là so sánh hiệu quả dự báo giữa ANFIS và các phương pháp học máy phổ biến trên cùng bộ dữ liệu.

---

## 2. Mục tiêu dự án

### Mục tiêu tổng quát

Xây dựng hệ thống dự báo phụ tải điện ngắn hạn cho hộ gia đình và đánh giá hiệu quả của mô hình ANFIS.

### Mục tiêu cụ thể

* Thu thập dữ liệu thời tiết Hà Nội.
* Mô phỏng dữ liệu tiêu thụ điện hộ gia đình.
* Xây dựng pipeline tiền xử lý dữ liệu.
* Tạo bộ đặc trưng Core và Extended.
* Huấn luyện các mô hình Baseline.
* Huấn luyện mô hình ANFIS.
* Đánh giá và so sánh kết quả dự báo.

---

## 3. Cấu trúc dự án

```text
ANFIS/
│
├── data/
│   ├── raw/
│   │   ├── hanoi_weather_2021_2025.csv
│   │   └── hanoi_load_dataset.csv
│   │
│   └── processed/
│       ├── raw/
│       │   ├── core/
│       │   └── extended/
│       │
│       ├── scaled/
│       │   ├── core/
│       │   └── extended/
│       │
│       └── stats/
│
├── reports/
│
├── results/
│   ├── 1h/
│   └── 24h/
│
├── scripts/
│
├── src/
│   ├── config/
│   ├── data/
│   │   ├── analysis/
│   │   ├── pipeline/
│   │   └── utils/
│   │
│   └── model/
│       ├── anfis.py
│       ├── trainer.py
│       ├── evaluator.py
│       ├── visualizer.py
│       ├── train_anfis_hourly.py
│       ├── train_baselines.py
│       └── visualize_baselines.py
│
├── test/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 4. Workflow dự án

```text
Thu thập dữ liệu thời tiết
            ↓
Mô phỏng dữ liệu phụ tải điện
            ↓
Tiền xử lý dữ liệu
            ↓
Tạo bộ dữ liệu Core và Extended
            ↓
Huấn luyện Baseline Models
            ↓
Huấn luyện ANFIS
            ↓
Đánh giá mô hình
            ↓
So sánh kết quả
```

---

## 5. Bộ dữ liệu

### Dữ liệu thời tiết

* Địa điểm: Hà Nội, Việt Nam
* Thời gian: 2021 – 2025
* Nguồn: Open-Meteo API

### Bộ đặc trưng Core

Bao gồm các đặc trưng quan trọng nhất được sử dụng cho ANFIS:

* apparent_temperature
* humidity
* hour_sin
* hour_cos
* occupancy_level
* load_lag_24

### Bộ đặc trưng Extended

Ngoài Core Features còn bao gồm:

* Đặc trưng thời gian
* Đặc trưng hành vi sử dụng điện
* Đặc trưng thời tiết
* Lag Features
* Rolling Features

---

## 6. Các mô hình được sử dụng

### Baseline Models

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* XGBoost Regressor

### Mô hình chính

* ANFIS (Adaptive Neuro-Fuzzy Inference System)

---

## 7. Cài đặt dự án

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

## 8. Hướng dẫn chạy dự án

### Bước 1. Thu thập dữ liệu thời tiết

```bash
python -m src.data.pipeline.get_hanoi_weather
```

### Bước 2. Mô phỏng dữ liệu phụ tải điện

```bash
python -m src.data.pipeline.build_hanoi_load_dataset
```

### Bước 3. Tiền xử lý dữ liệu

```bash
python -m src.data.pipeline.preprocess_hanoi_load_dataset
```

### Bước 4. Huấn luyện Baseline Models

```bash
python -m src.model.train_baselines
```

### Bước 5. Trực quan hóa kết quả Baseline

```bash
python -m src.model.visualize_baselines
```

### Bước 6. Huấn luyện ANFIS

```bash
python -m src.model.train_anfis_hourly
```

### Huấn luyện ANFIS với số Membership Functions khác

```bash
python -m src.model.train_anfis_hourly --n-mfs 3
```

---

## 9. Kết quả thực nghiệm

### Baseline Models (Core Dataset)

| Model             | Horizon |   RMSE |     R² |
| ----------------- | ------- | -----: | -----: |
| Linear Regression | 1h      | 0.3440 | 0.6561 |
| Decision Tree     | 1h      | 0.1796 | 0.9063 |
| Random Forest     | 1h      | 0.1552 | 0.9300 |
| XGBoost           | 1h      | 0.1568 | 0.9286 |
| Linear Regression | 24h     | 0.2928 | 0.7509 |
| Decision Tree     | 24h     | 0.2423 | 0.8294 |
| Random Forest     | 24h     | 0.2240 | 0.8542 |
| XGBoost           | 24h     | 0.2239 | 0.8543 |

### ANFIS (3 Membership Functions)

| Horizon |   RMSE |    MAE |   MAPE |     R² |
| ------- | -----: | -----: | -----: | -----: |
| 1h      | 0.1905 | 0.1135 | 18.42% | 0.8945 |
| 24h     | 0.2347 | 0.1477 | 23.83% | 0.8399 |

---

## 10. Công nghệ sử dụng

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

## 11. Thành viên nhóm

Dự án được thực hiện trong khuôn khổ môn học Tính toán mềm (Soft Computing).

### Vai trò

* Xây dựng dữ liệu và tiền xử lý.
* Huấn luyện và đánh giá Baseline Models.
* Xây dựng và huấn luyện ANFIS.
* Phân tích và so sánh kết quả thực nghiệm.
