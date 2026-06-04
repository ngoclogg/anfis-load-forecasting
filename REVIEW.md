# Chi tiết dự án ANFIS Load Forecasting

Tài liệu này cung cấp cái nhìn chi tiết về từng tệp tin và thư mục trong dự án (loại trừ các tệp tin bị `.gitignore` ẩn như dữ liệu đã xử lý quy mô lớn hoặc kết quả chạy cục bộ).

## 1. Thư mục Gốc (Root)
- **`README.md`**: Tài liệu hướng dẫn chung về dự án, cách cài đặt và mục tiêu hệ thống.
- **`requirements.txt`**: Danh sách các thư viện Python cần thiết (numpy, pandas, scikit-learn, matplotlib, pytest).
- **`REVIEW.md`**: (File này) Bản đồ kỹ thuật tóm tắt cấu trúc và lệnh vận hành dự án.

## 2. Thư mục Mã nguồn (`src/`)

### 📂 `src/config/` (Cấu hình hệ thống)
- **`paths.py`**: Định nghĩa tập trung các đường dẫn thư mục trong dự án (Data, Results, Figures, Reports) giúp mã nguồn không bị phụ thuộc vào máy cá nhân.

### 📂 `src/data/` (Kỹ thuật dữ liệu)
- **`get_hanoi_weather.py`**: Script kết nối API (hoặc giả lập) để lấy dữ liệu thời tiết lịch sử của Hà Nội.
- **`build_hanoi_load_dataset.py`**: Script chính để tạo ra bộ dữ liệu phụ tải (`hanoi_load_dataset.csv`). Nó tích hợp dữ liệu thời tiết, tạo các đặc trưng thời gian (sin/cos hour), và tính toán các biến trễ như `load_lag_24`.
- **`preprocess_hanoi_load_dataset.py`**: Thực hiện làm sạch dữ liệu, chuẩn hóa Min-Max (tính toán dựa trên tập Train) và phân tách dữ liệu thành các tập Train/Test theo mốc thời gian 2025.
- **`eda_utils.py`**: Các hàm tiện ích để khám phá dữ liệu (kiểm tra thiếu hụt, thống kê mô tả).
- **`eda_visualization.py`**: Script tạo ra các biểu đồ phân tích dữ liệu khám phá (biểu đồ phân phối, tương quan, tính chu kỳ).

### 📂 `src/model/` (Lõi mô hình & Huấn luyện)
- **`anfis.py`**: Triển khai lớp `ANFIS`. Bao gồm:
  - Khởi tạo hàm thành viên (Membership Functions) dựa trên phân phối dữ liệu đầu vào.
  - Forward pass để tính toán trọng số kích hoạt của các luật mờ.
  - Tối ưu hóa hệ số hệ quả Sugeno (consequent coefficients) bằng phương pháp Ridge Regression (Least Squares với điều chuẩn).
  - Hỗ trợ lưu và tải mô hình (`.npz`).
- **`data_loader.py`**: Lớp trung gian để đọc dữ liệu từ `data/processed/`. Nó đảm bảo schema đầu vào đúng cho mô hình ANFIS (chọn bộ đặc trưng Core hoặc Extended) và quản lý việc nghịch đảo chuẩn hóa (Inverse Scaling) để trả về đơn vị kWh.
- **`train_anfis_hourly.py`**: Pipeline CLI chính. Nhiệm vụ:
  - Tiếp nhận các tham số huấn luyện (n_mfs, ridge_alpha).
  - Điều phối quá trình huấn luyện mô hình.
  - Đánh giá trên tập Test và so sánh với Baseline Lag-24.
  - Xuất toàn bộ kết quả (metrics, config, predictions, plots, rule summary).

## 3. Thư mục Dữ liệu (`data/`)
- **`raw/hanoi_load_dataset.csv`**: Dữ liệu phụ tải gốc sau khi tổng hợp.
- **`raw/hanoi_weather_2021_2025.csv`**: Dữ liệu thời tiết gốc.
- **`processed/feature_config.json`**: Lưu trữ cấu hình về các tập đặc trưng (Core/Extended) và mốc thời gian phân tách dữ liệu.
- **`processed/feature_scaler_stats.csv` & `target_scaler_stats.csv`**: Lưu trữ giá trị Min/Max của từng cột trong tập Train để dùng cho việc chuẩn hóa tập Test và dự báo tương lai.
- **`preprocessing_summary.csv`**: Tóm tắt kết quả quá trình tiền xử lý (số lượng mẫu, các cột đã xử lý).

## 4. Thư mục Kiểm thử (`tests/`)
- **`conftest.py`**: Chứa các fixture và hàm tạo dữ liệu giả lập (synthetic data) phục vụ cho việc kiểm thử nhanh.
- **`test_anfis.py`**: Kiểm tra các logic toán học của mô hình ANFIS (forward pass, firing strength, save/load).
- **`test_data_loader.py`**: Xác minh việc load dữ liệu đúng mốc thời gian và schema.
- **`test_metrics.py`**: Kiểm tra tính chính xác của các hàm tính toán MAE, RMSE, MAPE.
- **`test_cli.py`**: Kiểm tra khả năng chạy thông suốt của toàn bộ pipeline từ đầu đến cuối.

## 5. Thư mục Hình ảnh & Báo cáo (`figures/`, `latex/`, `reports/`)
- **`figures/`**: Chứa các biểu đồ phân tích dữ liệu đầu vào (tương quan, phân phối phụ tải).
- **`latex/main.tex` & `latex/sections/*.tex`**: Mã nguồn LaTeX cho báo cáo kỹ thuật/luận văn của dự án.
- **`reports/main.pdf`**: Bản PDF cuối cùng của báo cáo dự án.
- **`reports/workflow.docx`**: Tài liệu mô tả quy trình làm việc.

## 6. Thư mục Scripts (`scripts/`)
- **`verify_anfis_consequent_fit.py`**: Script độc lập để kiểm tra độ khớp của các hệ số hệ quả ANFIS, dùng để debug sâu vào quá trình tối ưu hóa.

## 7. Hướng dẫn lệnh thực thi (Execution Commands)

Dưới đây là các lệnh chính để vận hành dự án từ đầu đến cuối:

### Bước 1: Tải dữ liệu thời tiết

```bash id="52u5rk"
python -m src.data.get_hanoi_weather
```

### Bước 2: Mô phỏng dữ liệu phụ tải điện

```bash id="ysd4vb"
python -m src.data.build_hanoi_load_dataset
```

### Bước 3: Tiền xử lý dữ liệu

```bash id="kic1cs"
python -m src.data.preprocess_hanoi_load_dataset
```

### 🚀 Bước 2: Huấn luyện và Đánh giá mô hình ANFIS
Sử dụng script chính để chạy huấn luyện và kiểm tra kết quả trên năm 2025:
```bash
# Chạy run Core Global (dùng cho báo cáo cuối cùng)
python -m src.model.train_anfis_hourly --feature-set core --run-name core_global_final --n-mfs 2 --ridge-alpha 1e-4
```

### 🧪 Bước 3: Chạy Kiểm thử tự động
Xác minh tính đúng đắn của toàn bộ code:
```powershell
# Trên Windows (PowerShell) để đảm bảo hỗ trợ UTF-8
$env:PYTHONUTF8='1'; pytest --basetemp=tests/tmp
```

### 📊 Xem kết quả
Kết quả của mỗi lần chạy sẽ nằm trong:
`results/anfis_hourly/core/<run_name>_<timestamp>/`
Bao gồm file `metrics.json` (sai số) và các hình ảnh `actual_vs_predicted.png`, `residuals.png`.
