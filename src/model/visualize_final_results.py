"""Visualize final experiment results: Baselines + ANFIS."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config.paths import RESULTS_DIR


BASELINE_METRICS_PATH = RESULTS_DIR / "baselines" / "metrics" / "baseline_metrics_all.csv"
FINAL_FIGURES_DIR = RESULTS_DIR / "final_figures"
FINAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


ANFIS_RESULTS = [
    {
        "horizon": "1h",
        "dataset": "core",
        "model": "ANFIS",
        "mae": 0.110881,
        "rmse": 0.187050,
        "mape": 18.140,
        "r2": 0.898323,
    },
    {
        "horizon": "24h",
        "dataset": "core",
        "model": "ANFIS",
        "mae": 0.145392,
        "rmse": 0.231865,
        "mape": 23.355,
        "r2": 0.843810,
    },
]


def load_final_metrics() -> pd.DataFrame:
    baseline_df = pd.read_csv(BASELINE_METRICS_PATH)

    # Dùng extended baseline để so sánh chính với ANFIS trong báo cáo
    baseline_df = baseline_df[baseline_df["dataset"] == "extended"].copy()

    anfis_df = pd.DataFrame(ANFIS_RESULTS)

    final_df = pd.concat([baseline_df, anfis_df], ignore_index=True)
    return final_df


def plot_metric(final_df: pd.DataFrame, metric: str, horizon: str) -> None:
    plot_df = final_df[final_df["horizon"] == horizon].copy()

    model_order = [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost",
        "ANFIS",
    ]

    plot_df["model"] = pd.Categorical(
        plot_df["model"],
        categories=model_order,
        ordered=True,
    )
    plot_df = plot_df.sort_values("model")

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["model"].astype(str), plot_df[metric])
    plt.title(f"{metric.upper()} Comparison - {horizon} Forecast")
    plt.xlabel("Model")
    plt.ylabel(metric.upper())
    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = FINAL_FIGURES_DIR / f"{metric}_comparison_{horizon}_final.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_horizon_line_chart(final_df: pd.DataFrame, metric: str) -> None:
    """Plot metric changes across forecast horizons for all models."""
    model_order = [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost",
        "ANFIS",
    ]

    pivot_df = final_df.pivot_table(
        index="horizon",
        columns="model",
        values=metric,
        aggfunc="first",
    )

    pivot_df = pivot_df.loc[["1h", "24h"], model_order]

    plt.figure(figsize=(9, 5))

    for model_name in model_order:
        plt.plot(
            pivot_df.index,
            pivot_df[model_name],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    plt.title(f"{metric.upper()} across Forecast Horizons")
    plt.xlabel("Forecast Horizon")
    plt.ylabel(metric.upper())
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = FINAL_FIGURES_DIR / f"{metric}_across_horizons_final.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main() -> None:
    final_df = load_final_metrics()

    output_csv = FINAL_FIGURES_DIR / "final_metrics_for_report.csv"
    final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_csv}")

    for horizon in ["1h", "24h"]:
        for metric in ["rmse", "r2"]:
            plot_metric(final_df, metric, horizon)

    # for metric in ["mae", "rmse", "mape", "r2"]:
        # plot_horizon_line_chart(final_df, metric)


if __name__ == "__main__":
    main()