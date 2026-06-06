"""Verify ANFIS Sugeno consequent fitting on the Core validation split."""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

from src.config.paths import RESULT_HORIZONS
from src.model.anfis import ANFIS
from src.model.data_loader import load_core_data, split_train_val_test


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit ANFIS Sugeno consequents and verify validation RMSE."
    )
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results/anfis_consequent_fit")
    parser.add_argument("--horizon", choices=list(RESULT_HORIZONS), default="1h")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--ridge-alpha", type=float, default=1e-4)
    parser.add_argument("--min-sigma", type=float, default=0.10)
    parser.add_argument("--n-mfs", type=int, default=2)
    return parser.parse_args()


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def _write_training_log(path: Path, row: dict[str, object]) -> None:
    fieldnames = [
        "epoch",
        "horizon",
        "target_column",
        "train_fit_rows",
        "validation_rows",
        "test_rows",
        "ridge_alpha",
        "min_sigma",
        "n_mfs",
        "n_rules",
        "train_rmse_scaled",
        "validation_rmse_scaled",
        "predictions_finite",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir) / args.horizon
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Đang tải dữ liệu Core cho horizon {args.horizon}...")
    bundle = load_core_data(args.processed_dir, horizon=args.horizon)
    train_fit, val, test = split_train_val_test(
        bundle,
        val_start=args.validation_start,
    )

    print(f"Số mẫu Train-fit: {len(train_fit.features)}")
    print(f"Số mẫu Validation: {len(val.features)}")

    print("Đang khởi tạo mô hình ANFIS...")
    model = ANFIS(
        n_mfs=args.n_mfs,
        feature_order=bundle.config["core_features"],
        ridge_alpha=args.ridge_alpha,
        min_sigma=args.min_sigma,
        metadata={
            "horizon": args.horizon,
            "target_column": bundle.config["target_column"],
            "validation_start": args.validation_start,
            "verification_script": "scripts/verify_anfis_consequent_fit.py",
        },
    )

    print("Đang khởi tạo tham số tiền đề (mu, sigma) từ train-fit...")
    model.initialize_memberships(train_fit.features.to_numpy())

    print("Đang huấn luyện hệ quả bằng Ridge Least Squares...")
    model.fit_consequents(
        train_fit.features.to_numpy(),
        train_fit.target_scaled.to_numpy(),
        ridge_alpha=args.ridge_alpha,
    )

    print("Đang thực hiện đánh giá nội bộ...")
    train_pred = model.predict(train_fit.features.to_numpy())
    val_pred = model.predict(val.features.to_numpy())

    predictions_finite = bool(np.isfinite(train_pred).all() and np.isfinite(val_pred).all())
    if not predictions_finite:
        print("LỖI: Dự báo train-fit hoặc validation chứa NaN/Inf!")
        sys.exit(1)

    train_rmse = _rmse(train_pred, train_fit.target_scaled.to_numpy())
    val_rmse = _rmse(val_pred, val.target_scaled.to_numpy())

    print(f"RMSE Train-fit (scaled): {train_rmse:.6f}")
    print(f"RMSE Validation (scaled): {val_rmse:.6f}")

    model_path = output_dir / "model.npz"
    model.save_model(model_path)
    print(f"Đã lưu mô hình tại: {model_path}")

    log_path = output_dir / "training_log.csv"
    _write_training_log(
        log_path,
        {
            "epoch": 0,
            "horizon": args.horizon,
            "target_column": bundle.config["target_column"],
            "train_fit_rows": len(train_fit.features),
            "validation_rows": len(val.features),
            "test_rows": len(test.features),
            "ridge_alpha": args.ridge_alpha,
            "min_sigma": args.min_sigma,
            "n_mfs": args.n_mfs,
            "n_rules": model.n_rules,
            "train_rmse_scaled": f"{train_rmse:.10f}",
            "validation_rmse_scaled": f"{val_rmse:.10f}",
            "predictions_finite": predictions_finite,
        },
    )
    print(f"Đã ghi log huấn luyện tại: {log_path}")


if __name__ == "__main__":
    main()
