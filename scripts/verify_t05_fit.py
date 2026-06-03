"""Verification script for Task T05: Fitting Sugeno consequents and validation evaluation."""

import sys
from pathlib import Path

# Add src to path if needed
sys.path.append(str(Path.cwd()))

import numpy as np
import pandas as pd
from src.model.data_loader import load_core_data, split_train_val_test
from src.model.anfis import ANFIS

def main():
    
    # 1. Load data
    print("Đang tải dữ liệu Core...")
    bundle = load_core_data("data/processed")
    train_fit, val, test = split_train_val_test(bundle, val_start="2024-01-01")
    
    print(f"Số mẫu Train-fit: {len(train_fit.features)}")
    print(f"Số mẫu Validation: {len(val.features)}")
    
    # 2. Khởi tạo mô hình
    print("Đang khởi tạo mô hình ANFIS...")
    model = ANFIS(
        n_mfs=2,
        feature_order=bundle.config["core_features"],
        ridge_alpha=1e-4,
        min_sigma=0.10
    )
    
    # 3. Khởi tạo tham số tiền đề (Premise Parameters)
    print("Đang khởi tạo tham số tiền đề (mu, sigma) từ train-fit...")
    model.initialize_memberships(train_fit.features.to_numpy())
    
    # 4. Huấn luyện hệ quả (Consequent Parameters)
    print("Đang huấn luyện hệ quả bằng Ridge Least Squares...")
    model.fit_consequents(
        train_fit.features.to_numpy(),
        train_fit.target_scaled.to_numpy()
    )
    
    # 5. Đánh giá nội bộ
    print("Đang thực hiện đánh giá nội bộ...")
    train_pred = model.predict(train_fit.features.to_numpy())
    val_pred = model.predict(val.features.to_numpy())
    
    train_rmse = np.sqrt(np.mean((train_pred - train_fit.target_scaled.to_numpy())**2))
    val_rmse = np.sqrt(np.mean((val_pred - val.target_scaled.to_numpy())**2))
    
    print(f"RMSE Train-fit (scaled): {train_rmse:.6f}")
    print(f"RMSE Validation (scaled): {val_rmse:.6f}")
    
    # Kiểm tra NaN/Inf
    if not np.isfinite(val_pred).all():
        print("LỖI: Dự báo validation chứa NaN hoặc Inf!")
        sys.exit(1)
    
    # 6. Lưu model artifact
    model_path = Path("results/test_t05/model.npz")
    model.save_model(model_path)
    print(f"Đã lưu mô hình tại: {model_path}")
    

if __name__ == "__main__":
    main()
