"""Orchestrate ANFIS training and evaluation for hourly forecast horizons."""

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

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import paths as project_paths
from src.model.anfis import DEFAULT_SEED
from src.model.data_loader import (
    CoreDataBundle,
    CoreDataset,
    inverse_transform_target,
    load_core_data,
    split_train_val_test,
)
from src.model.evaluator import (
    absolute_percentage_error,
    calculate_metrics,
    save_metrics,
)
from src.model.trainer import ANFISTrainer, TrainingResult
from src.model.visualizer import (
    plot_actual_vs_predicted,
    plot_membership_functions,
    plot_residuals,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DEFAULT_FEATURE_SET = "core"
DEFAULT_VALIDATION_START = "2024-01-01"
DEFAULT_RIDGE_ALPHA = 1e-4
DEFAULT_N_MFS = 2
DEFAULT_PLOT_DAYS = 14
BASELINE_LAG24_FEATURE = "load_lag_24"
BASELINE_LAG24_COLUMN = "baseline_lag24_kwh"


@dataclass(frozen=True)
class TrainingMetrics:
    """Scaled diagnostics from the train-fit and validation splits."""

    train_rmse_scaled: float
    validation_rmse_scaled: float
    predictions_finite: bool
    train_fit_rows: int
    validation_rows: int
    test_rows: int


@dataclass(frozen=True)
class TestEvaluation:
    """Prediction frame, metric payload, and baseline provenance."""

    predictions: pd.DataFrame
    metrics: dict[str, Any]
    baseline_source: str
    baseline_missing_count: int


@dataclass(frozen=True)
class HorizonRunResult:
    """Artifacts and summary values produced for one forecast horizon."""

    horizon: str
    run_id: str
    training_result: TrainingResult
    training_metrics: TrainingMetrics
    evaluation: TestEvaluation
    artifacts: dict[str, Path]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the ANFIS hourly orchestrator."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate ANFIS Core models for hourly load horizons."
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=list(project_paths.RESULT_HORIZONS),
        choices=list(project_paths.RESULT_HORIZONS),
        help="Forecast horizons to run.",
    )
    parser.add_argument(
        "--feature-set",
        default=DEFAULT_FEATURE_SET,
        choices=[DEFAULT_FEATURE_SET],
        help="Feature set to train. The ANFIS entry point currently supports only Core.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run label. The horizon and timestamp are appended to the run_id.",
    )
    parser.add_argument(
        "--n-mfs",
        type=int,
        default=DEFAULT_N_MFS,
        help="Number of Gaussian membership functions per input.",
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
        help="Optional profile_code to use for the actual-vs-predicted plot window.",
    )
    parser.add_argument(
        "--plot-days",
        type=int,
        default=DEFAULT_PLOT_DAYS,
        help="Number of days to show in the actual-vs-predicted plot window.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed recorded in model metadata.",
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Optional processed data root. Defaults to src.config.paths.PROCESSED_DATA_DIR.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Optional results root. Defaults to src.config.paths.RESULTS_DIR.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run ANFIS training, evaluation, and plotting for all requested horizons."""
    args = parse_args(argv)
    validate_args(args)
    configure_result_root(args.results_dir)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now().replace(microsecond=0).isoformat()

    results: list[HorizonRunResult] = []
    for horizon in args.horizons:
        results.append(
            run_horizon(
                args=args,
                horizon=horizon,
                run_timestamp=run_timestamp,
                timestamp=timestamp,
            )
        )

    print("\nCompleted ANFIS horizons:")
    for result in results:
        print_horizon_summary(result)


def validate_args(args: argparse.Namespace) -> None:
    if args.plot_days <= 0:
        raise ValueError("--plot-days must be a positive integer.")
    if args.n_mfs <= 0:
        raise ValueError("--n-mfs must be a positive integer.")
    if args.ridge_alpha < 0:
        raise ValueError("--ridge-alpha must be non-negative.")


def configure_result_root(results_dir: str | None) -> None:
    """Apply an optional results root override while keeping path helpers in use."""
    if results_dir:
        project_paths.RESULTS_DIR = Path(results_dir)
    project_paths.create_all_paths()


def run_horizon(
    *,
    args: argparse.Namespace,
    horizon: str,
    run_timestamp: str,
    timestamp: str,
) -> HorizonRunResult:
    """Train, evaluate, plot, and persist one horizon run."""
    run_id = build_run_id(args.run_name, horizon, run_timestamp)
    processed_dir = Path(args.processed_dir) if args.processed_dir else None

    print(f"\n[{horizon}] Loading Core processed data...")
    bundle = load_core_data(processed_dir, horizon=horizon)
    train_fit, validation, test = split_train_val_test(
        bundle,
        val_start=args.validation_start,
    )
    feature_order = list(bundle.config["core_features"])
    target_column = str(bundle.config["target_column"])

    print(
        f"[{horizon}] Split rows: "
        f"train_fit={len(train_fit.features)}, "
        f"validation={len(validation.features)}, "
        f"test={len(test.features)}"
    )

    trainer = ANFISTrainer(
        n_mfs=args.n_mfs,
        ridge_alpha=args.ridge_alpha,
        feature_order=feature_order,
        horizon=horizon,
        random_state=args.seed,
        metadata={
            "feature_set": args.feature_set,
            "run_id": run_id,
            "run_name": args.run_name,
            "target_column": target_column,
            "timestamp": timestamp,
            "validation_start": args.validation_start,
        },
    )

    print(f"[{horizon}] Training ANFIS...")
    training_result = trainer.train(
        train_fit.features.to_numpy(dtype=float),
        train_fit.target_scaled.to_numpy(dtype=float),
        save_artifact=True,
        model_filename=f"{run_id}_anfis_model.npz",
        metadata={
            "train_fit_rows": len(train_fit.features),
            "validation_rows": len(validation.features),
            "test_rows": len(test.features),
        },
    )

    training_metrics = evaluate_training_splits(
        training_result=training_result,
        train_fit=train_fit,
        validation=validation,
        test=test,
    )

    print(f"[{horizon}] Evaluating test split...")
    evaluation = evaluate_test_split(
        model=training_result.model,
        bundle=bundle,
        test=test,
        run_id=run_id,
        timestamp=timestamp,
        feature_set=args.feature_set,
    )

    artifacts = write_run_artifacts(
        args=args,
        horizon=horizon,
        run_id=run_id,
        timestamp=timestamp,
        bundle=bundle,
        feature_order=feature_order,
        training_result=training_result,
        training_metrics=training_metrics,
        evaluation=evaluation,
    )

    result = HorizonRunResult(
        horizon=horizon,
        run_id=run_id,
        training_result=training_result,
        training_metrics=training_metrics,
        evaluation=evaluation,
        artifacts=artifacts,
    )
    print_horizon_summary(result)
    return result


def evaluate_training_splits(
    *,
    training_result: TrainingResult,
    train_fit: CoreDataset,
    validation: CoreDataset,
    test: CoreDataset,
) -> TrainingMetrics:
    """Compute scaled train-fit and validation RMSE diagnostics."""
    validation_predictions = training_result.model.predict(
        validation.features.to_numpy(dtype=float)
    )
    predictions_finite = bool(
        np.isfinite(training_result.train_predictions).all()
        and np.isfinite(validation_predictions).all()
    )
    if not predictions_finite:
        raise FloatingPointError("Train-fit or validation predictions contain NaN/Inf.")

    return TrainingMetrics(
        train_rmse_scaled=float(training_result.train_rmse),
        validation_rmse_scaled=rmse(
            validation_predictions,
            validation.target_scaled.to_numpy(dtype=float),
        ),
        predictions_finite=predictions_finite,
        train_fit_rows=len(train_fit.features),
        validation_rows=len(validation.features),
        test_rows=len(test.features),
    )


def write_run_artifacts(
    *,
    args: argparse.Namespace,
    horizon: str,
    run_id: str,
    timestamp: str,
    bundle: CoreDataBundle,
    feature_order: list[str],
    training_result: TrainingResult,
    training_metrics: TrainingMetrics,
    evaluation: TestEvaluation,
) -> dict[str, Path]:
    """Persist predictions, metrics, plots, and run metadata for one horizon."""
    predictions_dir = project_paths.get_results_subdir(horizon, "predictions")
    metrics_dir = project_paths.get_results_subdir(horizon, "metrics")

    predictions_path = predictions_dir / f"{run_id}_test_predictions.csv"
    write_predictions(predictions_path, evaluation.predictions)

    plot_frame = select_plot_frame(
        evaluation.predictions,
        plot_profile=args.plot_profile,
        plot_days=args.plot_days,
    )
    actual_vs_predicted_path = plot_actual_vs_predicted(
        plot_frame,
        horizon,
        f"{run_id}_actual_vs_predicted.png",
        title=f"ANFIS Actual vs Predicted - {horizon}",
    )
    residuals_path = plot_residuals(
        evaluation.predictions["actual_kwh"].to_numpy(dtype=float),
        evaluation.predictions["predicted_kwh"].to_numpy(dtype=float),
        horizon,
        f"{run_id}_residuals.png",
        title=f"ANFIS Residual Distribution - {horizon}",
    )
    membership_path = plot_membership_functions(
        training_result.model.mu,
        training_result.model.sigma,
        feature_order,
        horizon,
        f"{run_id}_membership_functions.png",
    )

    training_log_path = metrics_dir / f"{run_id}_training_log.csv"
    write_training_log(
        training_log_path,
        horizon=horizon,
        run_id=run_id,
        args=args,
        training_metrics=training_metrics,
        evaluation=evaluation,
        n_rules=training_result.model.n_rules,
    )

    artifacts: dict[str, Path] = {
        "model": require_path(training_result.model_path, "model_path"),
        "predictions": predictions_path,
        "actual_vs_predicted_plot": actual_vs_predicted_path,
        "residuals_plot": residuals_path,
        "membership_functions_plot": membership_path,
        "training_log": training_log_path,
    }
    metrics_payload = build_run_metrics_payload(
        args=args,
        horizon=horizon,
        run_id=run_id,
        timestamp=timestamp,
        bundle=bundle,
        feature_order=feature_order,
        training_result=training_result,
        training_metrics=training_metrics,
        evaluation=evaluation,
        artifacts=artifacts,
    )
    metrics_json_path = save_metrics(
        metrics_payload,
        horizon,
        f"{run_id}_metrics",
        format="json",
    )
    metrics_csv_path = save_metrics(
        flatten_metrics(metrics_payload),
        horizon,
        f"{run_id}_metrics",
        format="csv",
    )
    artifacts["metrics_json"] = metrics_json_path
    artifacts["metrics_csv"] = metrics_csv_path

    config_path = metrics_dir / f"{run_id}_config.json"
    write_json(
        config_path,
        build_run_config(
            args=args,
            horizon=horizon,
            run_id=run_id,
            timestamp=timestamp,
            bundle=bundle,
            feature_order=feature_order,
            training_result=training_result,
            training_metrics=training_metrics,
            artifacts=artifacts,
        ),
    )
    artifacts["config"] = config_path
    return artifacts


def evaluate_test_split(
    *,
    model: Any,
    bundle: CoreDataBundle,
    test: CoreDataset,
    run_id: str,
    timestamp: str,
    feature_set: str,
) -> TestEvaluation:
    """Build final-test predictions and metrics in kWh for ANFIS and Lag-24."""
    predicted_scaled = model.predict(test.features.to_numpy(dtype=float))
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
            f"{BASELINE_LAG24_COLUMN} contains {baseline_missing_count} missing "
            "or non-finite values on the test split."
        )

    error = actual_kwh - predicted_kwh
    prediction_frame = pd.DataFrame(
        {
            "datetime": test.metadata["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "profile_code": test.metadata["profile_code"].to_numpy(),
            "profile_name": test.metadata["profile_name"].to_numpy(),
            "actual_kwh": actual_kwh,
            "predicted_kwh": predicted_kwh,
            BASELINE_LAG24_COLUMN: baseline_kwh,
            "error_kwh": error,
            "abs_error_kwh": np.abs(error),
            "ape": absolute_percentage_error(actual_kwh, predicted_kwh),
        }
    )

    anfis_metrics = calculate_metrics(actual_kwh, predicted_kwh)
    lag24_metrics = calculate_metrics(actual_kwh, baseline_kwh)
    metrics_payload = build_metrics_payload(
        run_id=run_id,
        timestamp=timestamp,
        feature_set=feature_set,
        bundle=bundle,
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


def build_metrics_payload(
    *,
    run_id: str,
    timestamp: str,
    feature_set: str,
    bundle: CoreDataBundle,
    test: CoreDataset,
    actual_kwh: np.ndarray,
    anfis_metrics: dict[str, float],
    lag24_metrics: dict[str, float],
    baseline_source: str,
    baseline_missing_count: int,
) -> dict[str, Any]:
    """Create the final-test metrics payload before run-level fields are added."""
    test_datetimes = test.metadata["datetime"]
    target_column = str(bundle.config["target_column"])
    payload: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": timestamp,
        "horizon": bundle.config.get("horizon"),
        "feature_set": feature_set,
        "target_column": target_column,
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
    return payload


def build_run_metrics_payload(
    *,
    args: argparse.Namespace,
    horizon: str,
    run_id: str,
    timestamp: str,
    bundle: CoreDataBundle,
    feature_order: list[str],
    training_result: TrainingResult,
    training_metrics: TrainingMetrics,
    evaluation: TestEvaluation,
    artifacts: dict[str, Path],
) -> dict[str, Any]:
    """Merge test metrics with training diagnostics and artifact provenance."""
    return {
        **evaluation.metrics,
        "horizon": horizon,
        "timestamp": timestamp,
        "parameters": {
            "n_mfs": args.n_mfs,
            "ridge_alpha": args.ridge_alpha,
            "validation_start": args.validation_start,
            "seed": args.seed,
            "n_rules": training_result.model.n_rules,
        },
        "feature_order": feature_order,
        "split_rows": {
            "train_fit": training_metrics.train_fit_rows,
            "validation": training_metrics.validation_rows,
            "test": training_metrics.test_rows,
        },
        "scaled_metrics": {
            "train_rmse": training_metrics.train_rmse_scaled,
            "validation_rmse": training_metrics.validation_rmse_scaled,
            "predictions_finite": training_metrics.predictions_finite,
        },
        "data_paths": {
            name: relative_path(path)
            for name, path in sorted(bundle.paths.items(), key=lambda item: item[0])
        },
        "artifact_paths": {
            name: relative_path(path)
            for name, path in sorted(artifacts.items(), key=lambda item: item[0])
        },
    }


def build_run_config(
    *,
    args: argparse.Namespace,
    horizon: str,
    run_id: str,
    timestamp: str,
    bundle: CoreDataBundle,
    feature_order: list[str],
    training_result: TrainingResult,
    training_metrics: TrainingMetrics,
    artifacts: dict[str, Path],
) -> dict[str, Any]:
    """Create JSON run configuration for one horizon."""
    return {
        "run_id": run_id,
        "run_name": args.run_name,
        "timestamp": timestamp,
        "horizon": horizon,
        "feature_set": args.feature_set,
        "target_column": bundle.config["target_column"],
        "scaled_target_column": bundle.config["scaled_target_column"],
        "feature_order": feature_order,
        "parameters": {
            "n_mfs": args.n_mfs,
            "ridge_alpha": args.ridge_alpha,
            "validation_start": args.validation_start,
            "plot_profile": args.plot_profile,
            "plot_days": args.plot_days,
            "seed": args.seed,
            "n_rules": training_result.model.n_rules,
        },
        "split_rows": {
            "train_fit": training_metrics.train_fit_rows,
            "validation": training_metrics.validation_rows,
            "test": training_metrics.test_rows,
        },
        "data_paths": {
            name: relative_path(path)
            for name, path in sorted(bundle.paths.items(), key=lambda item: item[0])
        },
        "artifact_paths": {
            name: relative_path(path)
            for name, path in sorted(artifacts.items(), key=lambda item: item[0])
        },
    }


def select_plot_frame(
    predictions: pd.DataFrame,
    *,
    plot_profile: str | None,
    plot_days: int,
) -> pd.DataFrame:
    """Select a compact profile window for the actual-vs-predicted plot."""
    frame = predictions.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    if frame["datetime"].isna().any():
        raise ValueError("Cannot plot predictions with invalid datetime values.")

    selected_profile = (
        str(frame["profile_code"].iloc[0])
        if plot_profile is None
        else str(plot_profile)
    )
    frame = frame[frame["profile_code"].astype(str) == selected_profile].sort_values(
        "datetime"
    )
    if frame.empty:
        raise ValueError(f"No test rows found for plot profile {selected_profile!r}.")

    return frame.head(int(plot_days * 24)).copy()


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
    """Convert a scaled feature back to raw units using feature scaler stats."""
    stats = bundle.feature_scaler_stats.set_index("column")
    if feature_name not in stats.index:
        raise ValueError(f"Missing feature scaler stats for {feature_name!r}.")

    feature_stats = stats.loc[feature_name]
    return (
        np.asarray(scaled_values, dtype=float) * float(feature_stats["range"])
        + float(feature_stats["min"])
    )


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Backward-compatible wrapper around ``src.model.evaluator.calculate_metrics``."""
    return calculate_metrics(actual, prediction)


def metric_deltas(
    anfis_metrics: dict[str, float | None],
    lag24_metrics: dict[str, float | None],
) -> dict[str, float | None]:
    """Compare ANFIS metrics against the Lag-24 baseline."""
    return {
        "mae_delta_anfis_minus_lag24": subtract_metric(
            anfis_metrics["mae"],
            lag24_metrics["mae"],
        ),
        "rmse_delta_anfis_minus_lag24": subtract_metric(
            anfis_metrics["rmse"],
            lag24_metrics["rmse"],
        ),
        "mape_delta_anfis_minus_lag24": subtract_metric(
            anfis_metrics["mape"],
            lag24_metrics["mape"],
        ),
        "r2_delta_anfis_minus_lag24": subtract_metric(
            anfis_metrics["r2"],
            lag24_metrics["r2"],
        ),
    }


def subtract_metric(left: float | None, right: float | None) -> float | None:
    """Subtract two metric values while preserving JSON null semantics."""
    if left is None or right is None:
        return None
    return float(left - right)


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def build_run_id(run_name: str | None, horizon: str, timestamp: str) -> str:
    """Build a filesystem-safe run ID from an optional label, horizon, and timestamp."""
    safe_name = sanitize_run_name(run_name)
    prefix = f"{safe_name}_" if safe_name else ""
    return f"{prefix}{horizon}_{timestamp}"


def sanitize_run_name(run_name: str | None) -> str:
    """Return a compact ASCII-ish run label safe for filenames."""
    if run_name is None:
        return ""
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("._-")
    return normalized.lower()


def write_predictions(path: Path, frame: pd.DataFrame) -> None:
    """Write final-test predictions with stable numeric formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.10f")


def write_training_log(
    path: Path,
    *,
    horizon: str,
    run_id: str,
    args: argparse.Namespace,
    training_metrics: TrainingMetrics,
    evaluation: TestEvaluation,
    n_rules: int,
) -> None:
    """Write a one-row training and evaluation summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "horizon",
        "train_fit_rows",
        "validation_rows",
        "test_rows",
        "ridge_alpha",
        "n_mfs",
        "n_rules",
        "train_rmse_scaled",
        "validation_rmse_scaled",
        "test_mae_kwh",
        "test_rmse_kwh",
        "test_mape",
        "test_r2",
        "predictions_finite",
    ]
    row = {
        "run_id": run_id,
        "horizon": horizon,
        "train_fit_rows": training_metrics.train_fit_rows,
        "validation_rows": training_metrics.validation_rows,
        "test_rows": training_metrics.test_rows,
        "ridge_alpha": args.ridge_alpha,
        "n_mfs": args.n_mfs,
        "n_rules": n_rules,
        "train_rmse_scaled": f"{training_metrics.train_rmse_scaled:.10f}",
        "validation_rmse_scaled": f"{training_metrics.validation_rmse_scaled:.10f}",
        "test_mae_kwh": f"{evaluation.metrics['anfis']['mae']:.10f}",
        "test_rmse_kwh": f"{evaluation.metrics['anfis']['rmse']:.10f}",
        "test_mape": f"{evaluation.metrics['anfis']['mape']:.10f}",
        "test_r2": f"{evaluation.metrics['anfis']['r2']:.10f}",
        "predictions_finite": training_metrics.predictions_finite,
    }
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with deterministic formatting and UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def flatten_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested metrics payload for one-row CSV output."""
    flattened: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, nested_value in value.items():
                nested_key = f"{prefix}.{key}" if prefix else str(key)
                visit(nested_key, nested_value)
        elif isinstance(value, list):
            flattened[prefix] = json.dumps(value, ensure_ascii=False)
        else:
            flattened[prefix] = value

    visit("", payload)
    return flattened


def relative_path(path: Path) -> str:
    """Return a POSIX-style path relative to the repository root when possible."""
    absolute = Path(path)
    if not absolute.is_absolute():
        absolute = project_paths.ROOT_DIR / absolute
    try:
        return absolute.resolve().relative_to(project_paths.ROOT_DIR.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def require_path(path: Path | None, name: str) -> Path:
    """Return a path after validating it exists as a value."""
    if path is None:
        raise ValueError(f"{name} was not produced.")
    return path


def print_horizon_summary(result: HorizonRunResult) -> None:
    """Print the concise per-horizon result summary requested by the CLI."""
    metrics = result.evaluation.metrics["anfis"]
    print(
        f"[{result.horizon}] run_id={result.run_id} | "
        f"RMSE={metrics['rmse']:.6f} kWh | "
        f"MAE={metrics['mae']:.6f} kWh | "
        f"MAPE={metrics['mape']:.3f}% | "
        f"R2={metrics['r2']:.6f} | "
        f"val_RMSE_scaled={result.training_metrics.validation_rmse_scaled:.6f}"
    )


if __name__ == "__main__":
    main()
