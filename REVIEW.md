# Chi tiết dự án ANFIS Load Forecasting (Refactored)

Tài liệu này cung cấp cái nhìn chi tiết về cấu trúc thư mục, chức năng của từng script và quy trình vận hành hệ thống sau khi đã được tái cấu trúc và mô-đun hóa.

## 1. Thư mục Gốc (Root)
- **`README.md`**: Hướng dẫn chung về dự án và cài đặt.
- **`requirements.txt`**: Danh sách thư viện cần thiết.
- **`REVIEW.md`**: (File này) Bản đồ kỹ thuật tóm tắt cấu trúc và quy trình.

## 2. Thư mục Mã nguồn (`src/`)

### 📂 `src/config/` (Cấu hình)
- **`paths.py`**: Định nghĩa tập trung các hằng số đường dẫn. Hỗ trợ cấu trúc đa horizon (1h, 24h) và tự động tạo thư mục cần thiết.

### 📂 `src/data/` (Xử lý dữ liệu)
- **`pipeline/`**: Luồng xử lý chính.
    - **`get_hanoi_weather.py`**: Thu thập dữ liệu thời tiết.
    - **`build_hanoi_load_dataset.py`**: Tạo bộ dữ liệu phụ tải (`hanoi_load_dataset.csv`).
    - **`preprocess_hanoi_load_dataset.py`**: Làm sạch, chuẩn hóa Min-Max và phân tách dữ liệu theo cấu trúc thư mục mới.
- **`analysis/`**:
    - **`eda_visualization.py`**: Các biểu đồ phân tích khám phá dữ liệu.
- **`utils/`**:
    - **`eda_utils.py`**: Hàm tiện ích hỗ trợ khám phá dữ liệu.

### 📂 `src/model/` (Lõi mô hình & Huấn luyện)
- **`anfis.py`**: Triển khai lớp `ANFIS` (Logic lõi: MF, Forward, Sugeno Fit).
- **`trainer.py`**: Module điều phối huấn luyện (`ANFISTrainer`), fit tham số và lưu artifact `.npz`.
- **`evaluator.py`**: Tính toán các chỉ số metrics (MAE, RMSE, MAPE, R2) trên đơn vị kWh.
- **`visualizer.py`**: Vẽ biểu đồ kết quả (Actual vs Predicted, Residuals, Membership Functions).
- **`data_loader.py`**: Đọc dữ liệu từ `data/processed/`, quản lý `CoreDataBundle` và nghịch đảo chuẩn hóa.
- **`train_anfis_hourly.py`**: **Orchestrator chính**. Phối hợp trainer, evaluator và visualizer để thực hiện pipeline end-to-end cho cả 1h và 24h.

## 3. Cấu trúc Dữ liệu và Kết quả

### 📂 `data/processed/` (Dữ liệu đã xử lý)
- **`raw/core/` & `raw/extended/`**: Dữ liệu chưa chuẩn hóa.
- **`scaled/core/` & `scaled/extended/`**: Dữ liệu đã chuẩn hóa Min-Max.
- **`stats/`**: Chứa `feature_config.json` và các file thống kê scaler (`feature_scaler_stats_1h.csv`, ...).

### 📂 `results/` (Kết quả chạy mô hình)
Mỗi horizon (`1h`, `24h`) có thư mục riêng với cấu trúc:
- **`models/`**: Lưu trữ artifact mô hình `.npz`.
- **`metrics/`**: Kết quả sai số (JSON, CSV) và log huấn luyện.
- **`plots/`**: Biểu đồ kết quả và luật mờ.
- **`predictions/`**: Kết quả dự báo chi tiết trên tập test (CSV).

## 4. Quy trình chạy Code (Workflow)

Thực hiện theo trình tự sau để chạy toàn bộ hệ thống:

### Bước 1: Thu thập và Xây dựng dữ liệu gốc
```bash
python -m src.data.pipeline.get_hanoi_weather
python -m src.data.pipeline.build_hanoi_load_dataset
```

### Bước 2: Tiền xử lý dữ liệu (Cấu trúc lại thư mục)
Lệnh này sẽ tạo ra dữ liệu chuẩn hóa và phân chia Train/Test vào đúng các thư mục con:
```bash
python -m src.data.pipeline.preprocess_hanoi_load_dataset
```

### Bước 3: Huấn luyện, Đánh giá và Xuất kết quả
Chạy orchestrator chính để thực hiện huấn luyện cho cả 1h và 24h (mặc định):
```bash
python -m src.model.train_anfis_hourly --run-name final_refactor_test --n-mfs 2 --ridge-alpha 1e-4
```

### Nếu muốn test nhanh xem mô hình có chạy không thì chạy lệnh này
Lệnh này sẽ tạo ra 1 bộ dữ liệu giả cực nhỏ để đưa vào cho mô hình train test. Mục đích của lệnh này là đảm bảo rằng đã thu được output đầy đủ từ mô hình. Toàn bộ code của bước này nằm trong `tests/` nếu không cần bước này có thể loại bỏ hẳn `tests/` ra khỏi dự án
```powershell
$env:PYTHONUTF8='1'; pytest --basetemp=tests/tmp
```

## 5. Các Artifacts đầu ra chính
Sau khi chạy Bước 3, kiểm tra các thư mục:
- `results/1h/metrics/final_refactor_test_1h_<timestamp>_metrics.json`
- `results/1h/plots/final_refactor_test_1h_<timestamp>_actual_vs_predicted.png`
- `results/24h/models/final_refactor_test_24h_<timestamp>_anfis_model.npz`
