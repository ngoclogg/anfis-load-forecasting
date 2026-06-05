from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.paths import get_results_subdir

METRIC_EPSILON = 1e-8


def absolute_percentage_error(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """
    Compute Absolute Percentage Error (APE) in percent.
    
    Includes epsilon protection for near-zero actual values to prevent division by zero.
    """
    denominator = np.maximum(np.abs(actual), METRIC_EPSILON)
    return np.abs(actual - prediction) / denominator * 100.0


def calculate_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """
    Compute standard regression metrics (MAE, RMSE, MAPE, R2).
    
    Args:
        actual: Ground truth values (typically in raw units like kWh).
        prediction: Model predictions (typically in raw units like kWh).
        
    Returns:
        Dictionary containing MAE, RMSE, MAPE, and R2.
    """
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    
    error = actual - prediction
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mape = float(np.mean(absolute_percentage_error(actual, prediction)))
    
    ss_res = float(np.sum(np.square(error)))
    ss_tot = float(np.sum(np.square(actual - float(np.mean(actual)))))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > METRIC_EPSILON else 0.0
    
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
    }


def save_metrics(
    metrics: dict[str, Any],
    horizon: str,
    filename: str,
    format: str = "json"
) -> Path:
    """
    Save metrics to the appropriate results directory.
    
    Args:
        metrics: Dictionary of metrics to save.
        horizon: Forecast horizon (e.g., '1h', '24h').
        filename: Name of the file (without extension if format is provided).
        format: 'json' or 'csv'.
        
    Returns:
        Path to the saved file.
    """
    metrics_dir = get_results_subdir(horizon, "metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    if not filename.endswith(f".{format}"):
        path = metrics_dir / f"{filename}.{format}"
    else:
        path = metrics_dir / filename
        
    if format == "json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
    elif format == "csv":
        # Handle nested dictionaries by flattening if necessary, 
        # but usually metrics is a flat dict.
        df = pd.DataFrame([metrics])
        df.to_csv(path, index=False, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported format: {format}")
        
    return path