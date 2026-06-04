"""Module for visualizing ANFIS results, including predictions, residuals, and membership functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src.config.paths import get_results_subdir


def plot_actual_vs_predicted(
    frame: pd.DataFrame,
    horizon: str,
    filename: str,
    title: str = "Actual vs Predicted Load",
    ylabel: str = "kWh",
) -> Path:
    """
    Plot actual vs predicted load values.
    
    Args:
        frame: DataFrame containing 'datetime', 'actual_kwh', and 'predicted_kwh'.
               Can also contain 'baseline_lag24_kwh'.
        horizon: Forecast horizon for saving the plot.
        filename: Filename for the plot.
        title: Plot title.
        ylabel: Label for the Y axis.
        
    Returns:
        Path to the saved plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if "datetime" in frame.columns:
        x = pd.to_datetime(frame["datetime"])
        ax.plot(x, frame["actual_kwh"], label="Actual", linewidth=1.5)
        ax.plot(x, frame["predicted_kwh"], label="Predicted", linewidth=1.5, alpha=0.8)
        
        if "baseline_lag24_kwh" in frame.columns:
            ax.plot(x, frame["baseline_lag24_kwh"], label="Baseline (Lag-24)", 
                    linestyle="--", alpha=0.6)
            
        locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set_xlabel("Time")
    else:
        ax.plot(frame["actual_kwh"].values, label="Actual")
        ax.plot(frame["predicted_kwh"].values, label="Predicted")
        ax.set_xlabel("Sample index")

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    save_path = save_plot(fig, horizon, filename)
    plt.close(fig)
    return save_path


def plot_residuals(
    actual: np.ndarray,
    prediction: np.ndarray,
    horizon: str,
    filename: str,
    title: str = "Residual Distribution",
) -> Path:
    """
    Plot the distribution of residuals (actual - prediction).
    """
    residuals = np.asarray(actual) - np.asarray(prediction)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    save_path = save_plot(fig, horizon, filename)
    plt.close(fig)
    return save_path


def plot_membership_functions(
    mu: np.ndarray,
    sigma: np.ndarray,
    feature_names: Sequence[str],
    horizon: str,
    filename: str,
) -> Path:
    """
    Plot Gaussian membership functions for all inputs.
    
    Args:
        mu: Gaussian centers, shape (n_inputs, n_mfs).
        sigma: Gaussian widths, shape (n_inputs, n_mfs).
        feature_names: Names of the features.
        horizon: Forecast horizon.
        filename: Filename for the plot.
    """
    n_inputs = mu.shape[0]
    n_mfs = mu.shape[1]
    
    # Calculate grid size for subplots
    cols = 2
    rows = (n_inputs + 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten()
    
    x = np.linspace(0, 1, 500)
    
    for i in range(n_inputs):
        ax = axes[i]
        for j in range(n_mfs):
            # Gaussian MF: exp(-0.5 * ((x - mu) / sigma)^2)
            y = np.exp(-0.5 * ((x - mu[i, j]) / sigma[i, j])**2)
            ax.plot(x, y, label=f"MF{j+1}")
        
        ax.set_title(f"Input: {feature_names[i]}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
            
    # Hide unused subplots
    for i in range(n_inputs, len(axes)):
        axes[i].axis("off")
        
    fig.tight_layout()
    save_path = save_plot(fig, horizon, filename)
    plt.close(fig)
    return save_path


def save_plot(fig: plt.Figure, horizon: str, filename: str) -> Path:
    """
    Save a figure to the appropriate results directory.
    """
    plots_dir = get_results_subdir(horizon, "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    if not (filename.endswith(".png") or filename.endswith(".pdf") or filename.endswith(".svg")):
        filename += ".png"
        
    path = plots_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
