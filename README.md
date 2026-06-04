# ANFIS Load Forecasting

## 1. Giới thiệu dự án

Dự án này tập trung vào bài toán dự báo phụ tải điện ngắn hạn cho hộ gia đình bằng mô hình ANFIS (Adaptive Neuro-Fuzzy Inference System).

Dữ liệu trong dự án được mô phỏng dựa trên:

* Dữ liệu thời tiết thực tế tại Hà Nội
* Hành vi sử dụng điện của hộ gia đình
* Thói quen sinh hoạt theo thời gian
* Các thiết bị điện và mức độ sử dụng

Dự án bao gồm:

* Thu thập dữ liệu thời tiết
* Mô phỏng dữ liệu phụ tải điện
* Sinh đặc trưng (feature engineering)
* Tiền xử lý dữ liệu
* Khám phá dữ liệu (EDA)
* Huấn luyện và đánh giá mô hình ANFIS

---

## 2. Mục tiêu dự án

Các mục tiêu chính của dự án:

* Mô phỏng dữ liệu tiêu thụ điện thực tế
* Xây dựng pipeline tiền xử lý dữ liệu
* Tạo tập đặc trưng Core và Extended
* Ứng dụng ANFIS để dự báo phụ tải điện
* Đánh giá hiệu suất mô hình

---

## 3. Cấu trúc dự án

```text id="n2vexj"
ANFIS/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── figures/
│
├── reports/
├── results/
│
├── src/
│   ├── config/
│   ├── data/
│   └── model/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 4. Chức năng chính

### Thu thập dữ liệu

* Tải dữ liệu thời tiết Hà Nội từ Open-Meteo API

### Mô phỏng phụ tải điện

* Mô phỏng nhiều kiểu hộ gia đình
* Mô phỏng hành vi sử dụng điện thực tế

### Sinh đặc trưng

* Đặc trưng thời gian
* Đặc trưng thời tiết
* Đặc trưng hành vi
* Lag features và rolling features

### Tiền xử lý dữ liệu

* Chia tập train/test
* Chuẩn hóa Min-Max
* Tạo tập Core và Extended

### Trực quan hóa dữ liệu

* Phân phối phụ tải
* Phụ tải trung bình theo giờ
* Heatmap tương quan
* So sánh phụ tải theo profile

---

## 5. Cài đặt dự án

Clone repository:

```bash id="h3n7p1"
git clone https://github.com/ngoclogg/anfis-load-forecasting.git
```

Di chuyển vào thư mục project:

```bash id="k2sjcw"
cd anfis-load-forecasting
```

Cài đặt thư viện:

```bash id="p7ehsq"
pip install -r requirements.txt
```

---

## 6. Hướng dẫn chạy dự án

### Bước 1: Tải dữ liệu thời tiết

```bash id="52u5rk"
python -m src.data.pipeline.get_hanoi_weather
```

### Bước 2: Mô phỏng dữ liệu phụ tải điện

```bash id="ysd4vb"
python -m src.data.pipeline.build_hanoi_load_dataset
```

### Bước 3: Tiền xử lý dữ liệu

```bash id="kic1cs"
python -m src.data.pipeline.preprocess_hanoi_load_dataset
```

---

## 7. Hướng dẫn sử dụng Git

### Lấy code mới nhất từ GitHub

```bash id="7gq6tz"
git pull
```

### Thêm file thay đổi

```bash id="7f34ii"
git add .
```

### Tạo commit

```bash id="s97wgs"
git commit -m "Update project"
```

### Đẩy code lên GitHub

```bash id="iz4klv"
git push
```

---

## 8. Công nghệ sử dụng

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* ANFIS
* Git & GitHub

---

## 9. Thông tin dữ liệu

### Dữ liệu thời tiết

* Địa điểm: Hà Nội, Việt Nam
* Thời gian: 2021 - 2025
* Nguồn: Open-Meteo API

### Các nhóm đặc trưng

* Đặc trưng thời gian
* Đặc trưng thời tiết
* Đặc trưng hành vi sử dụng điện
* Lag features

---

## 10. Tác giả

Dự án được phát triển phục vụ môn học Tính toán mềm và nghiên cứu dự báo phụ tải điện bằng ANFIS.
