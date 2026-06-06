from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.data_loader import SCALED_TARGET_COLUMN, TARGET_COLUMN


CORE_FEATURES = [
    "apparent_temperature",
    "humidity",
    "hour_sin",
    "hour_cos",
    "occupancy_level",
    "load_lag_24",
]
TARGET_MIN = 100.0
TARGET_RANGE = 100.0


@pytest.fixture
def synthetic_processed_dir(tmp_path: Path) -> Path:
    return write_core_processed_dir(tmp_path / "processed")


def write_core_processed_dir(
    processed_dir: Path,
    *,
    include_feature_config: bool = True,
    test_start: str = "2025-01-01",
) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_raw, train_scaled = _make_split_frame("2023-12-30", periods=96)
    test_raw, test_scaled = _make_split_frame(test_start, periods=48)

    if include_feature_config:
        (processed_dir / "feature_config.json").write_text(
            json.dumps(
                {
                    "train_end_exclusive": "2025-01-01",
                    "target_column": TARGET_COLUMN,
                    "core_features": CORE_FEATURES,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    train_raw.to_csv(processed_dir / "train_core_raw.csv", index=False, encoding="utf-8")
    train_scaled.to_csv(
        processed_dir / "train_core_scaled.csv",
        index=False,
        encoding="utf-8",
    )
    test_raw.to_csv(processed_dir / "test_core_raw.csv", index=False, encoding="utf-8")
    test_scaled.to_csv(
        processed_dir / "test_core_scaled.csv",
        index=False,
        encoding="utf-8",
    )
    _feature_scaler_stats().to_csv(
        processed_dir / "feature_scaler_stats.csv",
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "column": TARGET_COLUMN,
                "min": TARGET_MIN,
                "max": TARGET_MIN + TARGET_RANGE,
                "range": TARGET_RANGE,
            }
        ]
    ).to_csv(processed_dir / "target_scaler_stats.csv", index=False, encoding="utf-8")

    return processed_dir


def _make_split_frame(start: str, *, periods: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    datetimes = pd.date_range(start=start, periods=periods, freq="h")
    row_index = np.arange(periods, dtype=float)
    hour = datetimes.hour.to_numpy(dtype=float)
    angle = 2.0 * np.pi * hour / 24.0

    load_kwh = (
        130.0
        + 12.0 * np.sin(angle)
        + 4.0 * np.cos(angle)
        + 0.05 * row_index
    )
    lag24_kwh = load_kwh - 3.0

    scaled = pd.DataFrame(
        {
            "datetime": datetimes.strftime("%Y-%m-%d %H:%M:%S"),
            "profile_code": "pytest_profile",
            "profile_name": "Pytest profile",
            "apparent_temperature": 0.50 + 0.25 * np.sin(angle),
            "humidity": 0.55 + 0.20 * np.cos(angle),
            "hour_sin": (np.sin(angle) + 1.0) / 2.0,
            "hour_cos": (np.cos(angle) + 1.0) / 2.0,
            "occupancy_level": np.where((hour >= 7.0) & (hour <= 22.0), 0.75, 0.25),
            "load_lag_24": (lag24_kwh - TARGET_MIN) / TARGET_RANGE,
            TARGET_COLUMN: load_kwh,
            SCALED_TARGET_COLUMN: (load_kwh - TARGET_MIN) / TARGET_RANGE,
        }
    )
    raw = scaled.drop(columns=[SCALED_TARGET_COLUMN]).copy()
    raw.loc[:, "load_lag_24"] = lag24_kwh
    return raw, scaled


def _feature_scaler_stats() -> pd.DataFrame:
    rows = [
        {"column": "apparent_temperature", "min": 0.0, "max": 1.0, "range": 1.0},
        {"column": "humidity", "min": 0.0, "max": 1.0, "range": 1.0},
        {"column": "hour_sin", "min": 0.0, "max": 1.0, "range": 1.0},
        {"column": "hour_cos", "min": 0.0, "max": 1.0, "range": 1.0},
        {"column": "occupancy_level", "min": 0.0, "max": 1.0, "range": 1.0},
        {
            "column": "load_lag_24",
            "min": TARGET_MIN,
            "max": TARGET_MIN + TARGET_RANGE,
            "range": TARGET_RANGE,
        },
    ]
    return pd.DataFrame(rows)