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

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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
    rule_summary = build_rule_summary(model, test.features)

    config_path = run_dir / "config.json"
    model_path = run_dir / "model.npz"
    training_log_path = run_dir / "training_log.csv"
    predictions_path = run_dir / "predictions_test.csv"
    metrics_path = run_dir / "metrics.json"
    rule_summary_path = run_dir / "rule_summary.csv"
    actual_vs_predicted_path = run_dir / "actual_vs_predicted.png"
    residuals_path = run_dir / "residuals.png"

    print("Writing test visualizations...")
    plot_artifacts = write_test_visualizations(
        evaluation.predictions,
        actual_vs_predicted_path=actual_vs_predicted_path,
        residuals_path=residuals_path,
        plot_profile=args.plot_profile,
        plot_days=args.plot_days,
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
        plot_artifacts=plot_artifacts,
    )

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
    write_rule_summary(rule_summary_path, rule_summary)

    print(f"Run ID: {run_id}")
    print(f"Config: {config_path}")
    print(f"Model: {model_path}")
    print(f"Training log: {training_log_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Rule summary: {rule_summary_path}")
    print(f"Actual vs predicted plot: {actual_vs_predicted_path}")
    print(f"Residuals plot: {residuals_path}")
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
    plot_artifacts: dict[str, Any],
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
        "plot_artifacts_status": plot_artifacts["status"],
        "data_paths": data_paths,
        "processed_dir": relative_path(Path(args.processed_dir)),
        "output_paths": {
            "run_dir": relative_path(run_dir),
            "config": relative_path(run_dir / "config.json"),
            "model": relative_path(run_dir / "model.npz"),
            "training_log": relative_path(run_dir / "training_log.csv"),
            "predictions_test": relative_path(run_dir / "predictions_test.csv"),
            "metrics": relative_path(run_dir / "metrics.json"),
            "rule_summary": relative_path(run_dir / "rule_summary.csv"),
            "actual_vs_predicted": relative_path(run_dir / "actual_vs_predicted.png"),
            "residuals": relative_path(run_dir / "residuals.png"),
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
            "visualizations": plot_artifacts,
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


def build_rule_summary(model: ANFIS, test_features: pd.DataFrame) -> pd.DataFrame:
    """Create a ranked fuzzy-rule summary from final-test activations."""
    feature_order = list(model.feature_order)
    if list(test_features.columns) != feature_order:
        raise ValueError(
            "test feature columns must match model.feature_order when writing "
            "rule_summary.csv."
        )

    normalized = model.normalized_firing_strengths(test_features.to_numpy())
    if normalized.shape != (len(test_features), model.n_rules):
        raise ValueError(
            "normalized firing strengths must have shape "
            f"({len(test_features)}, {model.n_rules}); got {normalized.shape}."
        )
    if not np.isfinite(normalized).all():
        raise FloatingPointError(
            "Rule summary activations contain NaN or Inf values."
        )

    coefficients = np.asarray(model.consequent_coefficients, dtype=float)
    expected_coeff_shape = (model.n_rules, len(feature_order) + 1)
    if coefficients.shape != expected_coeff_shape:
        raise ValueError(
            "consequent_coefficients must have shape "
            f"{expected_coeff_shape}; got {coefficients.shape}."
        )

    activation_mean = normalized.mean(axis=0)
    rows: list[dict[str, Any]] = []
    for rule_id, mf_indices in enumerate(model.rule_indices):
        coefficient_values = [float(value) for value in coefficients[rule_id]]
        intercept = coefficient_values[0]
        row: dict[str, Any] = {"rule_id": rule_id}
        for feature_name, mf_index in zip(feature_order, mf_indices):
            row[f"{feature_name}_mf_label"] = f"MF{int(mf_index)}"

        row["activation_mean"] = float(activation_mean[rule_id])
        row["consequent_coefficients"] = json.dumps(
            coefficient_values,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        row["consequent_intercept"] = intercept
        for feature_name, slope in zip(feature_order, coefficient_values[1:]):
            row[f"consequent_slope_{feature_name}"] = slope
        row["contribution_score"] = row["activation_mean"] * abs(intercept)
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["contribution_score", "activation_mean", "rule_id"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def write_rule_summary(path: Path, frame: pd.DataFrame) -> None:
    """Write the ranked fuzzy-rule summary required by T09."""
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.10f")


def write_test_visualizations(
    predictions: pd.DataFrame,
    *,
    actual_vs_predicted_path: Path,
    residuals_path: Path,
    plot_profile: str | None,
    plot_days: int,
) -> dict[str, Any]:
    """Write Matplotlib test-result plots and return config metadata."""
    frame = predictions.copy()
    if frame.empty:
        raise ValueError("Cannot plot an empty predictions frame.")

    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    if frame["datetime"].isna().any():
        raise ValueError("Cannot plot predictions with invalid datetime values.")

    selected_profile = (
        str(frame["profile_code"].iloc[0])
        if plot_profile is None
        else str(plot_profile)
    )
    profile_frame = frame[
        frame["profile_code"].astype(str) == selected_profile
    ].sort_values("datetime")
    if profile_frame.empty:
        raise ValueError(f"No test rows found for plot profile {selected_profile!r}.")

    window_rows = int(plot_days * 24)
    window_frame = profile_frame.head(window_rows).copy()
    if window_frame.empty:
        raise ValueError("Cannot plot an empty actual-vs-predicted window.")

    numeric_columns = ["actual_kwh", "predicted_kwh", BASELINE_LAG24_COLUMN]
    for column in numeric_columns:
        window_frame[column] = pd.to_numeric(window_frame[column], errors="coerce")
        if not np.isfinite(window_frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Cannot plot non-finite values in {column!r}.")

    residuals = pd.to_numeric(
        frame["error_kwh"],
        errors="coerce",
    ).to_numpy(dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        raise ValueError("Cannot plot residuals because no finite errors were found.")

    profile_name = str(window_frame["profile_name"].iloc[0])
    profile_label = selected_profile
    if profile_name and profile_name != selected_profile:
        profile_label = f"{selected_profile} - {profile_name}"

    write_actual_vs_predicted_plot(
        window_frame,
        actual_vs_predicted_path,
        profile_label=profile_label,
    )
    write_residuals_plot(residuals, residuals_path)

    return {
        "status": "generated",
        "actual_vs_predicted": {
            "path": relative_path(actual_vs_predicted_path),
            "plot_type": "line",
            "profile_code": selected_profile,
            "profile_name": profile_name,
            "requested_days": int(plot_days),
            "expected_hourly_samples": window_rows,
            "window_rows": int(len(window_frame)),
            "start": window_frame["datetime"].iloc[0].isoformat(),
            "end": window_frame["datetime"].iloc[-1].isoformat(),
            "series": [
                "actual_kwh",
                "predicted_kwh",
                BASELINE_LAG24_COLUMN,
            ],
        },
        "residuals": {
            "path": relative_path(residuals_path),
            "plot_type": "histogram",
            "residual_definition": "actual_kwh - predicted_kwh",
            "rows": int(residuals.size),
        },
    }


def write_actual_vs_predicted_plot(
    frame: pd.DataFrame,
    path: Path,
    *,
    profile_label: str,
) -> None:
    """Plot the representative test window in kWh units."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(
        frame["datetime"],
        frame["actual_kwh"],
        label="Actual kWh",
        linewidth=1.8,
    )
    ax.plot(
        frame["datetime"],
        frame["predicted_kwh"],
        label="Predicted kWh",
        linewidth=1.5,
    )
    ax.plot(
        frame["datetime"],
        frame[BASELINE_LAG24_COLUMN],
        label="Baseline Lag-24 kWh",
        linewidth=1.3,
        linestyle="--",
    )
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_xlabel("Time")
    ax.set_ylabel("kWh")
    ax.set_title(f"Actual vs Predicted Test Load - Profile {profile_label}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_residuals_plot(residuals: np.ndarray, path: Path) -> None:
    """Plot final-test residual distribution in kWh units."""
    bins = min(60, max(10, int(np.sqrt(residuals.size))))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(residuals, bins=bins, color="#4C78A8", edgecolor="white", alpha=0.9)
    ax.axvline(
        0.0,
        color="#E45756",
        linestyle="--",
        linewidth=1.4,
        label="Zero error",
    )
    ax.set_xlabel("Residual (Actual - Predicted) [kWh]")
    ax.set_ylabel("Frequency")
    ax.set_title("Test Residual Distribution")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_predictions(path: Path, frame: pd.DataFrame) -> None:
    """Write final-test predictions with stable numeric formatting."""
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.10f")


if __name__ == "__main__":
    main()
