"""Visualize baseline model results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.config.paths import RESULTS_DIR


BASELINE_RESULTS_DIR = RESULTS_DIR / "baselines"
BASELINE_METRICS_PATH = BASELINE_RESULTS_DIR / "metrics" / "baseline_metrics_all.csv"
BASELINE_PREDICTIONS_DIR = BASELINE_RESULTS_DIR / "predictions"
BASELINE_FIGURES_DIR = BASELINE_RESULTS_DIR / "figures"

BASELINE_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_metric_comparison(metrics_df: pd.DataFrame, metric: str) -> None:
    """Plot model comparison for one metric."""
    for horizon in ["1h", "24h"]:
        horizon_df = metrics_df[metrics_df["horizon"] == horizon]

        plt.figure(figsize=(9, 5))
        plt.bar(horizon_df["model"], horizon_df[metric])
        plt.title(f"{metric.upper()} Comparison - {horizon} Forecast")
        plt.xlabel("Model")
        plt.ylabel(metric.upper())
        plt.xticks(rotation=20)
        plt.tight_layout()

        output_path = BASELINE_FIGURES_DIR / f"{metric}_comparison_{horizon}.png"
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Saved figure: {output_path}")


def plot_actual_vs_prediction(
    prediction_file: str,
    model_name: str,
    horizon: str,
    sample_size: int = 200,
) -> None:
    """Plot actual values against predicted values."""
    prediction_path = BASELINE_PREDICTIONS_DIR / prediction_file
    prediction_df = pd.read_csv(prediction_path)

    sample_df = prediction_df.head(sample_size)

    plt.figure(figsize=(12, 5))
    plt.plot(sample_df["actual"], label="Actual")
    plt.plot(sample_df[model_name], label=model_name)
    plt.title(f"Actual vs Prediction - {model_name} - {horizon} Forecast")
    plt.xlabel("Sample")
    plt.ylabel("Load")
    plt.legend()
    plt.tight_layout()

    safe_model_name = model_name.replace(" ", "_").lower()
    output_path = BASELINE_FIGURES_DIR / f"actual_vs_prediction_{safe_model_name}_{horizon}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure: {output_path}")


def main() -> None:
    """Run baseline result visualization."""
    metrics_df = pd.read_csv(BASELINE_METRICS_PATH)

    for metric in ["mae", "rmse", "mape", "r2"]:
        plot_metric_comparison(metrics_df, metric)

    plot_actual_vs_prediction(
        prediction_file="baseline_predictions_1h.csv",
        model_name="Random Forest",
        horizon="1h",
    )

    plot_actual_vs_prediction(
        prediction_file="baseline_predictions_24h.csv",
        model_name="XGBoost",
        horizon="24h",
    )


if __name__ == "__main__":
    main()