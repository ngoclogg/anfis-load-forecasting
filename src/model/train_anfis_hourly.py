"""Train the main ANFIS hourly model and persist run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.anfis import ANFIS
from src.model.data_loader import (
    TARGET_COLUMN,
    CoreDataBundle,
    CoreDataset,
    inverse_transform_target,
    load_core_data,
    split_train_val_test,
)


DEFAULT_FEATURE_SET = "core"
DEFAULT_RESULTS_DIR = Path("results/anfis_hourly")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_VALIDATION_START = "2024-01-01"
DEFAULT_RIDGE_ALPHA = 1e-4
DEFAULT_N_MFS = 2
DEFAULT_PLOT_DAYS = 14
DEFAULT_SEED = 20260527
BASELINE_LAG24_FEATURE = "load_lag_24"
BASELINE_LAG24_COLUMN = "baseline_lag24_kwh"
METRIC_EPSILON = 1e-8


@dataclass(frozen=True)
class TrainingMetrics:
    """Scaled metrics from the one-step Ridge consequent fit."""

    train_rmse_scaled: float
    validation_rmse_scaled: float
    predictions_finite: bool
    train_fit_rows: int
    validation_rows: int
    test_rows: int


@dataclass(frozen=True)
class TestEvaluation:
    """Prediction frame, metrics, and provenance for the final test split."""

    predictions: pd.DataFrame
    metrics: dict[str, Any]
    baseline_source: str
    baseline_missing_count: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the hourly ANFIS trainer."""
    parser = argparse.ArgumentParser(
        description="Train the main hourly ANFIS model for the Core feature set."
    )
    parser.add_argument(
        "--feature-set",
        default=DEFAULT_FEATURE_SET,
        choices=[DEFAULT_FEATURE_SET],
        help="Feature set to train. T06 supports only 'core'.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run label. The timestamp is always appended to the run_id.",
    )
    parser.add_argument(
        "--n-mfs",
        type=int,
        default=DEFAULT_N_MFS,
        help="Number of membership functions per input.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=DEFAULT_RIDGE_ALPHA,
        help="Ridge regression alpha for Sugeno consequents.",
    )
    parser.add_argument(
        "--validation-start",
        default=DEFAULT_VALIDATION_START,
        help="Validation start timestamp within the train split.",
    )
    parser.add_argument(
        "--plot-profile",
        default=None,
        help="Profile code reserved for validation plots in a later task.",
    )
    parser.add_argument(
        "--plot-days",
        type=int,
        default=DEFAULT_PLOT_DAYS,
        help="Number of days reserved for validation plots in a later task.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed recorded in the run config and model metadata.",
    )
    parser.add_argument(
        "--processed-dir",
        default=str(DEFAULT_PROCESSED_DIR),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    """Run the ANFIS training pipeline."""
    args = parse_args()
    if args.plot_days <= 0:
        raise ValueError("--plot-days must be a positive integer.")

    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = build_run_id(args.run_name, run_timestamp)
    run_dir = Path(args.results_dir) / args.feature_set / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    print("Loading Core processed data...")
    bundle = load_core_data(args.processed_dir)
    train_fit, validation, test = split_train_val_test(
        bundle,
        val_start=args.validation_start,
    )
    feature_order = list(bundle.config["core_features"])

    print(
        "Split rows: "
        f"train_fit={len(train_fit.features)}, "
        f"validation={len(validation.features)}, "
        f"test={len(test.features)}"
    )

    np.random.seed(args.seed)
    model = ANFIS(
        n_mfs=args.n_mfs,
        feature_order=feature_order,
        ridge_alpha=args.ridge_alpha,
        random_state=args.seed,
        metadata={
            "feature_set": args.feature_set,
            "run_id": run_id,
            "run_name": args.run_name,
            "target_column": TARGET_COLUMN,
            "validation_start": args.validation_start,
            "timestamp": timestamp,
        },
    )

    print("Initializing membership parameters from train-fit data...")
    model.initialize_memberships(train_fit.features.to_numpy())

    print("Fitting Sugeno consequents with Ridge least squares...")
    model.fit_consequents(
        train_fit.features.to_numpy(),
        train_fit.target_scaled.to_numpy(),
        ridge_alpha=args.ridge_alpha,
    )

    print("Computing train-fit and validation RMSE on the scaled target...")
    train_predictions = model.predict(train_fit.features.to_numpy())
    validation_predictions = model.predict(validation.features.to_numpy())
    predictions_finite = bool(
        np.isfinite(train_predictions).all()
        and np.isfinite(validation_predictions).all()
    )
    if not predictions_finite:
        raise FloatingPointError("Train-fit or validation predictions contain NaN/Inf.")

    metrics = TrainingMetrics(
        train_rmse_scaled=rmse(train_predictions, train_fit.target_scaled.to_numpy()),
        validation_rmse_scaled=rmse(
            validation_predictions,
            validation.target_scaled.to_numpy(),
        ),
        predictions_finite=predictions_finite,
        train_fit_rows=len(train_fit.features),
        validation_rows=len(validation.features),
        test_rows=len(test.features),
    )

    print("Evaluating ANFIS and Lag-24 baseline on the test split...")
    evaluation = evaluate_test_split(
        model=model,
        bundle=bundle,
        test=test,
        run_id=run_id,
        timestamp=timestamp,
        feature_set=args.feature_set,
    )

    config = build_run_config(
        args=args,
        run_id=run_id,
        timestamp=timestamp,
        run_dir=run_dir,
        feature_order=feature_order,
        n_rules=model.n_rules,
        bundle_paths=bundle.paths,
        metrics=metrics,
        evaluation=evaluation,
    )

    config_path = run_dir / "config.json"
    model_path = run_dir / "model.npz"
    training_log_path = run_dir / "training_log.csv"
    predictions_path = run_dir / "predictions_test.csv"
    metrics_path = run_dir / "metrics.json"

    write_json(config_path, config)
    model.save_model(model_path)
    write_training_log(
        training_log_path,
        args=args,
        metrics=metrics,
        n_rules=model.n_rules,
    )
    write_predictions(predictions_path, evaluation.predictions)
    write_json(metrics_path, evaluation.metrics)

    print(f"Run ID: {run_id}")
    print(f"Config: {config_path}")
    print(f"Model: {model_path}")
    print(f"Training log: {training_log_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Baseline Lag-24 source: {evaluation.baseline_source}")
    print(
        "Scaled RMSE: "
        f"train_fit={metrics.train_rmse_scaled:.10f}, "
        f"validation={metrics.validation_rmse_scaled:.10f}"
    )


def build_run_id(run_name: str | None, timestamp: str) -> str:
    """Build a filesystem-safe run ID from an optional label and timestamp."""
    safe_name = sanitize_run_name(run_name)
    return f"{safe_name}_{timestamp}" if safe_name else timestamp


def sanitize_run_name(run_name: str | None) -> str:
    """Return a compact ASCII-ish run label safe for directory names."""
    if run_name is None:
        return ""
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("._-")
    return normalized.lower()


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def evaluate_test_split(
    *,
    model: ANFIS,
    bundle: CoreDataBundle,
    test: CoreDataset,
    run_id: str,
    timestamp: str,
    feature_set: str,
) -> TestEvaluation:
    """Build final-test predictions and metrics for ANFIS and Lag-24."""
    predicted_scaled = model.predict(test.features.to_numpy())
    if not np.isfinite(predicted_scaled).all():
        raise FloatingPointError("Test predictions contain NaN or Inf.")

    predicted_kwh = np.asarray(
        inverse_transform_target(bundle, predicted_scaled),
        dtype=float,
    )
    actual_kwh = test.target_kwh.to_numpy(dtype=float)
    baseline_kwh, baseline_source = resolve_baseline_lag24_kwh(bundle, test)
    baseline_missing_count = int((~np.isfinite(baseline_kwh)).sum())
    if baseline_missing_count:
        raise ValueError(
            f"{BASELINE_LAG24_COLUMN} contains {baseline_missing_count} missing or "
            "non-finite values on the test split."
        )

    prediction_error = actual_kwh - predicted_kwh
    prediction_frame = pd.DataFrame(
        {
            "datetime": test.metadata["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "profile_code": test.metadata["profile_code"].to_numpy(),
            "profile_name": test.metadata["profile_name"].to_numpy(),
            "actual_kwh": actual_kwh,
            "predicted_kwh": predicted_kwh,
            BASELINE_LAG24_COLUMN: baseline_kwh,
            "error_kwh": prediction_error,
            "abs_error_kwh": np.abs(prediction_error),
            "ape": absolute_percentage_error(actual_kwh, predicted_kwh),
        }
    )

    anfis_metrics = regression_metrics(actual_kwh, predicted_kwh)
    lag24_metrics = regression_metrics(actual_kwh, baseline_kwh)
    metrics_payload = build_metrics_payload(
        run_id=run_id,
        timestamp=timestamp,
        feature_set=feature_set,
        test=test,
        actual_kwh=actual_kwh,
        anfis_metrics=anfis_metrics,
        lag24_metrics=lag24_metrics,
        baseline_source=baseline_source,
        baseline_missing_count=baseline_missing_count,
    )

    return TestEvaluation(
        predictions=prediction_frame,
        metrics=metrics_payload,
        baseline_source=baseline_source,
        baseline_missing_count=baseline_missing_count,
    )


def resolve_baseline_lag24_kwh(
    bundle: CoreDataBundle,
    test: CoreDataset,
) -> tuple[np.ndarray, str]:
    """Resolve Lag-24 baseline values independently from model predictions."""
    if BASELINE_LAG24_FEATURE in test.raw_frame.columns:
        values = pd.to_numeric(
            test.raw_frame[BASELINE_LAG24_FEATURE],
            errors="coerce",
        ).to_numpy(dtype=float)
        source = f"raw_frame.{BASELINE_LAG24_FEATURE}"
    elif BASELINE_LAG24_FEATURE in test.features.columns:
        scaled_values = test.features[BASELINE_LAG24_FEATURE].to_numpy(dtype=float)
        values = inverse_transform_feature(
            bundle,
            BASELINE_LAG24_FEATURE,
            scaled_values,
        )
        source = f"scaled_feature.{BASELINE_LAG24_FEATURE}_inverse_feature_scaler"
    elif BASELINE_LAG24_FEATURE in test.scaled_frame.columns:
        scaled_values = pd.to_numeric(
            test.scaled_frame[BASELINE_LAG24_FEATURE],
            errors="coerce",
        ).to_numpy(dtype=float)
        values = inverse_transform_feature(
            bundle,
            BASELINE_LAG24_FEATURE,
            scaled_values,
        )
        source = f"scaled_frame.{BASELINE_LAG24_FEATURE}_inverse_feature_scaler"
    else:
        raise ValueError(
            "Cannot build Lag-24 baseline because load_lag_24 is missing from "
            "both raw and scaled test features."
        )

    if values.shape[0] != len(test.features):
        raise ValueError(
            f"{BASELINE_LAG24_COLUMN} row count mismatch: "
            f"{values.shape[0]} != {len(test.features)}."
        )
    return values, source


def inverse_transform_feature(
    bundle: CoreDataBundle,
    feature_name: str,
    scaled_values: np.ndarray,
) -> np.ndarray:
    """Convert a scaled feature back to its raw units using feature scaler stats."""
    stats = bundle.feature_scaler_stats.set_index("column")
    if feature_name not in stats.index:
        raise ValueError(f"Missing feature scaler stats for {feature_name!r}.")

    feature_stats = stats.loc[feature_name]
    return (
        np.asarray(scaled_values, dtype=float) * float(feature_stats["range"])
        + float(feature_stats["min"])
    )


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    """Compute regression metrics on kWh values."""
    error = actual - prediction
    mae = float(np.mean(np.abs(error)))
    rmse_value = float(np.sqrt(np.mean(np.square(error))))
    mape = float(np.mean(absolute_percentage_error(actual, prediction)))
    ss_res = float(np.sum(np.square(error)))
    ss_tot = float(np.sum(np.square(actual - float(np.mean(actual)))))
    r2 = None if ss_tot <= METRIC_EPSILON else float(1.0 - ss_res / ss_tot)

    return {
        "mae": mae,
        "rmse": rmse_value,
        "mape": mape,
        "r2": r2,
    }


def absolute_percentage_error(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Compute APE in percent with epsilon protection for near-zero actual values."""
    denominator = np.maximum(np.abs(actual), METRIC_EPSILON)
    return np.abs(actual - prediction) / denominator * 100.0


def build_metrics_payload(
    *,
    run_id: str,
    timestamp: str,
    feature_set: str,
    test: CoreDataset,
    actual_kwh: np.ndarray,
    anfis_metrics: dict[str, float | None],
    lag24_metrics: dict[str, float | None],
    baseline_source: str,
    baseline_missing_count: int,
) -> dict[str, Any]:
    """Create metrics.json for final-test evaluation."""
    test_datetimes = test.metadata["datetime"]
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "feature_set": feature_set,
        "target_column": TARGET_COLUMN,
        "unit": "kWh",
        "test": {
            "rows": int(len(test.features)),
            "start": test_datetimes.min().isoformat(),
            "end": test_datetimes.max().isoformat(),
            "profile_count": int(test.metadata["profile_code"].nunique()),
            "target_stats_kwh": {
                "min": float(np.min(actual_kwh)),
                "max": float(np.max(actual_kwh)),
                "mean": float(np.mean(actual_kwh)),
                "std": float(np.std(actual_kwh)),
            },
        },
        "anfis": anfis_metrics,
        "baselines": {
            "lag24": {
                **lag24_metrics,
                "column": BASELINE_LAG24_COLUMN,
                "source": baseline_source,
                "missing_count": baseline_missing_count,
                "independent_from_model_output": True,
            }
        },
        "comparison": metric_deltas(anfis_metrics, lag24_metrics),
    }


def metric_deltas(
    anfis_metrics: dict[str, float | None],
    lag24_metrics: dict[str, float | None],
) -> dict[str, float | None]:
    """Compare ANFIS metrics against the Lag-24 baseline."""
    return {
        "mae_delta_anfis_minus_lag24": subtract_metric(anfis_metrics["mae"], lag24_metrics["mae"]),
        "rmse_delta_anfis_minus_lag24": subtract_metric(anfis_metrics["rmse"], lag24_metrics["rmse"]),
        "mape_delta_anfis_minus_lag24": subtract_metric(anfis_metrics["mape"], lag24_metrics["mape"]),
        "r2_delta_anfis_minus_lag24": subtract_metric(anfis_metrics["r2"], lag24_metrics["r2"]),
    }


def subtract_metric(left: float | None, right: float | None) -> float | None:
    """Subtract two metrics while preserving JSON null for undefined values."""
    if left is None or right is None:
        return None
    return float(left - right)


def build_run_config(
    *,
    args: argparse.Namespace,
    run_id: str,
    timestamp: str,
    run_dir: Path,
    feature_order: list[str],
    n_rules: int,
    bundle_paths: dict[str, Path],
    metrics: TrainingMetrics,
    evaluation: TestEvaluation,
) -> dict[str, Any]:
    """Create the JSON-serializable run configuration."""
    data_paths = {
        name: relative_path(path)
        for name, path in sorted(bundle_paths.items(), key=lambda item: item[0])
    }
    return {
        "feature_set": args.feature_set,
        "run_name": args.run_name,
        "run_id": run_id,
        "n_mfs": args.n_mfs,
        "n_rules": n_rules,
        "ridge_alpha": args.ridge_alpha,
        "validation_start": args.validation_start,
        "seed": args.seed,
        "feature_order": feature_order,
        "target_column": TARGET_COLUMN,
        "model_scope": "global",
        "timestamp": timestamp,
        "plot_profile": args.plot_profile,
        "plot_days": args.plot_days,
        "plot_artifacts_status": "placeholder_for_t10",
        "data_paths": data_paths,
        "processed_dir": relative_path(Path(args.processed_dir)),
        "output_paths": {
            "run_dir": relative_path(run_dir),
            "config": relative_path(run_dir / "config.json"),
            "model": relative_path(run_dir / "model.npz"),
            "training_log": relative_path(run_dir / "training_log.csv"),
            "predictions_test": relative_path(run_dir / "predictions_test.csv"),
            "metrics": relative_path(run_dir / "metrics.json"),
        },
        "split_rows": {
            "train_fit": metrics.train_fit_rows,
            "validation": metrics.validation_rows,
            "test": metrics.test_rows,
        },
        "scaled_metrics_epoch_0": {
            "train_rmse_scaled": metrics.train_rmse_scaled,
            "validation_rmse_scaled": metrics.validation_rmse_scaled,
            "predictions_finite": metrics.predictions_finite,
        },
        "evaluation": {
            "target_scale": "kWh",
            "prediction_source": "ANFIS model predictions inverse-transformed from scaled target",
            "baseline_lag24": {
                "column": BASELINE_LAG24_COLUMN,
                "source": evaluation.baseline_source,
                "missing_count": evaluation.baseline_missing_count,
                "test_rows": metrics.test_rows,
                "independent_from_model_output": True,
            },
        },
    }


def relative_path(path: Path) -> str:
    """Return a POSIX-style path relative to the repository root when possible."""
    path = Path(path)
    absolute = path if path.is_absolute() else REPO_ROOT / path
    try:
        return absolute.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with deterministic formatting and UTF-8 encoding."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_training_log(
    path: Path,
    *,
    args: argparse.Namespace,
    metrics: TrainingMetrics,
    n_rules: int,
) -> None:
    """Write the one-row training log required by T06."""
    fieldnames = [
        "epoch",
        "train_fit_rows",
        "validation_rows",
        "test_rows",
        "ridge_alpha",
        "n_mfs",
        "n_rules",
        "train_rmse_scaled",
        "validation_rmse_scaled",
        "predictions_finite",
    ]
    row = {
        "epoch": 0,
        "train_fit_rows": metrics.train_fit_rows,
        "validation_rows": metrics.validation_rows,
        "test_rows": metrics.test_rows,
        "ridge_alpha": args.ridge_alpha,
        "n_mfs": args.n_mfs,
        "n_rules": n_rules,
        "train_rmse_scaled": f"{metrics.train_rmse_scaled:.10f}",
        "validation_rmse_scaled": f"{metrics.validation_rmse_scaled:.10f}",
        "predictions_finite": metrics.predictions_finite,
    }
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def write_predictions(path: Path, frame: pd.DataFrame) -> None:
    """Write final-test predictions with stable numeric formatting."""
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.10f")


if __name__ == "__main__":
    main()
