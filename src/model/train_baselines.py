"""Train baseline regression models for Core and Extended 1h/24h load forecasting."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor

from src.config.paths import (
    PROCESSED_SCALED_CORE_DIR,
    PROCESSED_SCALED_EXTENDED_DIR,
    RESULTS_DIR,
)
from src.model.evaluator import calculate_metrics


BASELINE_RESULTS_DIR = RESULTS_DIR / "baselines"
BASELINE_METRICS_DIR = BASELINE_RESULTS_DIR / "metrics"
BASELINE_PREDICTIONS_DIR = BASELINE_RESULTS_DIR / "predictions"
BASELINE_MODELS_DIR = BASELINE_RESULTS_DIR / "models"

BASELINE_METRICS_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_models() -> dict[str, object]:
    """Create baseline models."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=10,
            random_state=42,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            objective="reg:squarederror",
        ),
    }


def load_train_test(
    dataset_dir: Path,
    train_file: str,
    test_file: str,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load train/test data and split features/target."""
    train_df = pd.read_csv(dataset_dir / train_file)
    test_df = pd.read_csv(dataset_dir / test_file)

    drop_columns = [
        "datetime",
        "profile_code",
        "profile_name",
        target_column,
        f"{target_column}_scaled",
    ]

    X_train = train_df.drop(columns=drop_columns)
    y_train = train_df[target_column]

    X_test = test_df.drop(columns=drop_columns)
    y_test = test_df[target_column]

    return X_train, y_train, X_test, y_test


def train_for_horizon(
    dataset_name: str,
    dataset_dir: Path,
    horizon: str,
    train_file: str,
    test_file: str,
    target_column: str,
) -> pd.DataFrame:
    """Train all baseline models for one dataset and one forecasting horizon."""
    X_train, y_train, X_test, y_test = load_train_test(
        dataset_dir=dataset_dir,
        train_file=train_file,
        test_file=test_file,
        target_column=target_column,
    )

    print(f"\n[{dataset_name.upper()} - {horizon}]")
    print(f"Actual min: {y_test.min()}")
    print(f"Actual mean: {y_test.mean()}")
    print(f"Actual values < 1: {(y_test < 1).sum()}")

    models = get_models()

    metrics_rows = []
    prediction_df = pd.DataFrame({
        "actual": y_test.to_numpy(),
    })

    for model_name, model in models.items():
        print(f"Training {model_name} for {dataset_name} {horizon} forecast...")

        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        safe_model_name = model_name.replace(" ", "_").lower()
        model_filename = f"{dataset_name}_{safe_model_name}_{horizon}.pkl"
        model_path = BASELINE_MODELS_DIR / model_filename
        joblib.dump(model, model_path)

        metrics = calculate_metrics(
            actual=y_test.to_numpy(),
            prediction=prediction,
        )

        metrics_rows.append({
            "horizon": horizon,
            "dataset": dataset_name,
            "model": model_name,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mape": metrics["mape"],
            "r2": metrics["r2"],
        })

        prediction_df[model_name] = prediction

        print(f"Saved model: {model_path}")

    metrics_df = pd.DataFrame(metrics_rows)

    metrics_path = BASELINE_METRICS_DIR / f"baseline_metrics_{dataset_name}_{horizon}.csv"
    predictions_path = (
        BASELINE_PREDICTIONS_DIR
        / f"baseline_predictions_{dataset_name}_{horizon}.csv"
    )

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")
    prediction_df.to_csv(predictions_path, index=False, encoding="utf-8")

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")

    return metrics_df


def main() -> None:
    """Run baseline training for Core first, then Extended."""
    results = []

    datasets = [
        ("core", PROCESSED_SCALED_CORE_DIR),
        ("extended", PROCESSED_SCALED_EXTENDED_DIR),
    ]

    for dataset_name, dataset_dir in datasets:
        print(
            "\n"
            "============================================================\n"
            f"Training baseline models on {dataset_name.upper()} dataset\n"
            "============================================================"
        )

        results.append(
            train_for_horizon(
                dataset_name=dataset_name,
                dataset_dir=dataset_dir,
                horizon="1h",
                train_file="train_1h.csv",
                test_file="test_1h.csv",
                target_column="target_1h",
            )
        )

        results.append(
            train_for_horizon(
                dataset_name=dataset_name,
                dataset_dir=dataset_dir,
                horizon="24h",
                train_file="train_24h.csv",
                test_file="test_24h.csv",
                target_column="target_24h",
            )
        )

    final_metrics = pd.concat(results, ignore_index=True)
    final_path = BASELINE_METRICS_DIR / "baseline_metrics_all.csv"
    final_metrics.to_csv(final_path, index=False, encoding="utf-8")

    print("\nFinal baseline metrics:")
    print(final_metrics)
    print(f"\nSaved final metrics: {final_path}")


if __name__ == "__main__":
    main()