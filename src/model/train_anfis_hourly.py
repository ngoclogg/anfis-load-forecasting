"""Train the main ANFIS hourly model and persist run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.model.anfis import ANFIS
from src.model.data_loader import TARGET_COLUMN, load_core_data, split_train_val_test


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_SET = "core"
DEFAULT_RESULTS_DIR = Path("results/anfis_hourly")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_VALIDATION_START = "2024-01-01"
DEFAULT_RIDGE_ALPHA = 1e-4
DEFAULT_N_MFS = 2
DEFAULT_PLOT_DAYS = 14
DEFAULT_SEED = 20260527


@dataclass(frozen=True)
class TrainingMetrics:
    """Scaled metrics from the one-step Ridge consequent fit."""

    train_rmse_scaled: float
    validation_rmse_scaled: float
    predictions_finite: bool
    train_fit_rows: int
    validation_rows: int
    test_rows: int


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

    config = build_run_config(
        args=args,
        run_id=run_id,
        timestamp=timestamp,
        run_dir=run_dir,
        feature_order=feature_order,
        n_rules=model.n_rules,
        bundle_paths=bundle.paths,
        metrics=metrics,
    )

    config_path = run_dir / "config.json"
    model_path = run_dir / "model.npz"
    training_log_path = run_dir / "training_log.csv"

    write_json(config_path, config)
    model.save_model(model_path)
    write_training_log(
        training_log_path,
        args=args,
        metrics=metrics,
        n_rules=model.n_rules,
    )

    print(f"Run ID: {run_id}")
    print(f"Config: {config_path}")
    print(f"Model: {model_path}")
    print(f"Training log: {training_log_path}")
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


if __name__ == "__main__":
    main()
