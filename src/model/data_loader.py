"""Core dataset loader and validation for the ANFIS hourly pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import paths as project_paths


TARGET_COLUMN = "load_kwh"
SCALED_TARGET_COLUMN = "load_kwh_scaled"
METADATA_COLUMNS = ("datetime", "profile_code", "profile_name")


class CoreDataError(ValueError):
    """Raised when processed Core artifacts are missing or invalid."""


@dataclass(frozen=True)
class CoreDataset:
    """Validated Core split prepared for ANFIS training or evaluation."""

    metadata: pd.DataFrame
    features: pd.DataFrame
    target_scaled: pd.Series
    target_kwh: pd.Series
    scaled_frame: pd.DataFrame
    raw_frame: pd.DataFrame


@dataclass(frozen=True)
class CoreDataBundle:
    """All processed Core data and scaler metadata needed by later tasks."""

    config: dict[str, Any]
    train: CoreDataset
    test: CoreDataset
    feature_scaler_stats: pd.DataFrame
    target_scaler_stats: pd.DataFrame
    paths: dict[str, Path]


def load_core_data(
    processed_dir: str | Path | None = None,
    *,
    horizon: str | None = None,
) -> CoreDataBundle:
    """
    Load and validate Core train/test artifacts.

    When ``horizon`` is provided, the loader reads the T02 layout:
    ``data/processed/{raw,scaled}/core/{train,test}_{horizon}.csv`` and
    horizon-specific scaler stats from ``data/processed/stats``. Without a
    horizon it keeps the legacy flat layout for small test fixtures.
    """

    processed_path = (
        Path(processed_dir)
        if processed_dir is not None
        else project_paths.PROCESSED_DATA_DIR
    )
    horizon = None if horizon is None else str(horizon)
    paths = _resolve_core_paths(processed_path, horizon)
    _validate_artifacts_exist(paths)

    config = _load_feature_config(paths["feature_config"])
    target_column = _resolve_target_column(config, horizon)
    scaled_target_column = f"{target_column}_scaled"
    config = {
        **config,
        "target_column": target_column,
        "scaled_target_column": scaled_target_column,
    }
    if horizon is not None:
        config["horizon"] = horizon
    core_features = _validate_feature_config(config)

    train = _load_split(
        split_name="train",
        scaled_path=paths["train_core_scaled"],
        raw_path=paths["train_core_raw"],
        core_features=core_features,
        target_column=target_column,
        scaled_target_column=scaled_target_column,
    )
    test = _load_split(
        split_name="test",
        scaled_path=paths["test_core_scaled"],
        raw_path=paths["test_core_raw"],
        core_features=core_features,
        target_column=target_column,
        scaled_target_column=scaled_target_column,
    )

    feature_scaler_stats = _load_scaler_stats(
        path=paths["feature_scaler_stats"],
        expected_columns=core_features,
        stats_name="feature_scaler_stats",
    )
    target_scaler_stats = _load_scaler_stats(
        path=paths["target_scaler_stats"],
        expected_columns=[target_column],
        stats_name="target_scaler_stats",
    )

    return CoreDataBundle(
        config=config,
        train=train,
        test=test,
        feature_scaler_stats=feature_scaler_stats,
        target_scaler_stats=target_scaler_stats,
        paths=paths,
    )


def split_train_val_test(
    bundle: CoreDataBundle,
    val_start: str | pd.Timestamp = "2024-01-01",
) -> tuple[CoreDataset, CoreDataset, CoreDataset]:
    """
    Split bundle into train-fit, validation, and test datasets.

    Train-fit contains data from ``bundle.train`` before ``val_start``.
    Validation contains data from ``bundle.train`` starting from ``val_start``.
    Test is simply ``bundle.test``.
    """
    val_start_ts = pd.to_datetime(val_start)
    train_metadata = bundle.train.metadata

    train_fit_mask = train_metadata["datetime"] < val_start_ts
    val_mask = train_metadata["datetime"] >= val_start_ts

    if not train_fit_mask.any():
        raise CoreDataError(f"No training data found before {val_start}.")
    if not val_mask.any():
        raise CoreDataError(f"No validation data found starting from {val_start}.")

    train_fit = _subset_dataset(bundle.train, train_fit_mask)
    val = _subset_dataset(bundle.train, val_mask)
    test = bundle.test

    return train_fit, val, test


def inverse_transform_target(
    bundle: CoreDataBundle,
    scaled_values: np.ndarray | pd.Series | float,
) -> np.ndarray | float:
    """Convert scaled target values back to kWh using bundle scaler stats."""
    target_column = str(bundle.config.get("target_column", TARGET_COLUMN))
    stats = bundle.target_scaler_stats.set_index("column").loc[target_column]
    min_val = float(stats["min"])
    range_val = float(stats["range"])

    if isinstance(scaled_values, (np.ndarray, pd.Series)):
        return scaled_values * range_val + min_val

    return float(scaled_values) * range_val + min_val


def _subset_dataset(dataset: CoreDataset, mask: pd.Series | np.ndarray) -> CoreDataset:
    """Create a new CoreDataset from a subset of an existing one."""
    return CoreDataset(
        metadata=dataset.metadata[mask].reset_index(drop=True),
        features=dataset.features[mask].reset_index(drop=True),
        target_scaled=dataset.target_scaled[mask].reset_index(drop=True),
        target_kwh=dataset.target_kwh[mask].reset_index(drop=True),
        scaled_frame=dataset.scaled_frame[mask].reset_index(drop=True),
        raw_frame=dataset.raw_frame[mask].reset_index(drop=True),
    )


def _resolve_core_paths(processed_path: Path, horizon: str | None) -> dict[str, Path]:
    """Return artifact paths for the requested Core processed layout."""
    if horizon is not None:
        horizon_paths = _horizon_core_paths(processed_path, horizon)
        if _all_paths_exist(horizon_paths):
            return horizon_paths

        legacy_paths = _legacy_core_paths(processed_path)
        if _all_paths_exist(legacy_paths):
            return legacy_paths

        return horizon_paths

    return _legacy_core_paths(processed_path)


def _horizon_core_paths(processed_path: Path, horizon: str) -> dict[str, Path]:
    stats_dir = processed_path / "stats"
    feature_config = stats_dir / "feature_config.json"
    if not feature_config.is_file():
        feature_config = processed_path / "feature_config.json"

    return {
        "feature_config": feature_config,
        "train_core_scaled": processed_path / "scaled" / "core" / f"train_{horizon}.csv",
        "test_core_scaled": processed_path / "scaled" / "core" / f"test_{horizon}.csv",
        "train_core_raw": processed_path / "raw" / "core" / f"train_{horizon}.csv",
        "test_core_raw": processed_path / "raw" / "core" / f"test_{horizon}.csv",
        "feature_scaler_stats": stats_dir / f"feature_scaler_stats_{horizon}.csv",
        "target_scaler_stats": stats_dir / f"target_scaler_stats_{horizon}.csv",
    }


def _legacy_core_paths(processed_path: Path) -> dict[str, Path]:
    return {
        "feature_config": processed_path / "feature_config.json",
        "train_core_scaled": processed_path / "train_core_scaled.csv",
        "test_core_scaled": processed_path / "test_core_scaled.csv",
        "train_core_raw": processed_path / "train_core_raw.csv",
        "test_core_raw": processed_path / "test_core_raw.csv",
        "feature_scaler_stats": processed_path / "feature_scaler_stats.csv",
        "target_scaler_stats": processed_path / "target_scaler_stats.csv",
    }


def _all_paths_exist(paths: dict[str, Path]) -> bool:
    return all(path.is_file() for path in paths.values())


def _validate_artifacts_exist(paths: dict[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required processed artifacts: " + ", ".join(missing)
        )


def _load_feature_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as exc:
        raise CoreDataError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise CoreDataError(f"{path} must contain a JSON object.")
    return config


def _resolve_target_column(config: dict[str, Any], horizon: str | None) -> str:
    if horizon is not None:
        target_columns = config.get("target_columns")
        if isinstance(target_columns, dict) and horizon in target_columns:
            target_column = target_columns[horizon]
            if isinstance(target_column, str) and target_column:
                return target_column

    target_column = config.get("target_column")
    if isinstance(target_column, str) and target_column:
        return target_column

    if horizon is None:
        raise CoreDataError(
            "feature_config.json must define target_column for the legacy layout."
        )

    raise CoreDataError(
        "feature_config.json must define target_columns with an entry for "
        f"horizon {horizon!r}."
    )


def _validate_feature_config(config: dict[str, Any]) -> list[str]:
    core_features = config.get("core_features")
    if not isinstance(core_features, list) or not core_features:
        raise CoreDataError("feature_config.json must define a non-empty core_features list.")

    invalid_features = [
        feature for feature in core_features if not isinstance(feature, str) or not feature
    ]
    if invalid_features:
        raise CoreDataError(
            "feature_config.json core_features contains invalid entries: "
            f"{invalid_features!r}."
        )

    return list(core_features)


def _load_split(
    *,
    split_name: str,
    scaled_path: Path,
    raw_path: Path,
    core_features: list[str],
    target_column: str,
    scaled_target_column: str,
) -> CoreDataset:
    scaled_frame = _read_csv(scaled_path)
    raw_frame = _read_csv(raw_path)

    _validate_columns(
        frame=scaled_frame,
        required_columns=[*METADATA_COLUMNS, *core_features, target_column, scaled_target_column],
        frame_name=f"{split_name} scaled data",
    )
    _validate_columns(
        frame=raw_frame,
        required_columns=[*METADATA_COLUMNS, *core_features, target_column],
        frame_name=f"{split_name} raw data",
    )

    if len(scaled_frame) != len(raw_frame):
        raise CoreDataError(
            f"{split_name} scaled/raw row count mismatch: "
            f"{len(scaled_frame)} != {len(raw_frame)}."
        )

    metadata = _prepare_metadata(scaled_frame, f"{split_name} scaled data")
    raw_metadata = _prepare_metadata(raw_frame, f"{split_name} raw data")
    if not metadata.reset_index(drop=True).equals(raw_metadata.reset_index(drop=True)):
        raise CoreDataError(f"{split_name} scaled/raw metadata rows do not match.")

    features = _numeric_frame(
        scaled_frame,
        core_features,
        f"{split_name} Core features",
    )
    raw_features = _numeric_frame(
        raw_frame,
        core_features,
        f"{split_name} raw Core features",
    )
    target_scaled = _numeric_series(
        scaled_frame,
        scaled_target_column,
        f"{split_name} target scaled",
    )
    target_kwh_from_scaled = _numeric_series(
        scaled_frame,
        target_column,
        f"{split_name} target kWh in scaled data",
    )
    target_kwh = _numeric_series(
        raw_frame,
        target_column,
        f"{split_name} target kWh in raw data",
    )

    _validate_nonnegative_target(target_kwh, f"{split_name} raw target")
    _validate_nonnegative_target(target_kwh_from_scaled, f"{split_name} scaled target")
    if not np.allclose(
        target_kwh.to_numpy(dtype=float),
        target_kwh_from_scaled.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-8,
    ):
        raise CoreDataError(f"{split_name} raw/scaled load_kwh values do not match.")

    scaled_frame = scaled_frame.copy()
    raw_frame = raw_frame.copy()
    scaled_frame.loc[:, core_features] = features
    scaled_frame.loc[:, target_column] = target_kwh_from_scaled
    scaled_frame.loc[:, scaled_target_column] = target_scaled
    raw_frame.loc[:, core_features] = raw_features
    raw_frame.loc[:, target_column] = target_kwh

    return CoreDataset(
        metadata=metadata,
        features=features,
        target_scaled=target_scaled,
        target_kwh=target_kwh,
        scaled_frame=scaled_frame,
        raw_frame=raw_frame,
    )


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    return _drop_empty_unnamed_columns(frame)


def _drop_empty_unnamed_columns(frame: pd.DataFrame) -> pd.DataFrame:
    empty_unnamed_columns = [
        column
        for column in frame.columns
        if str(column).startswith("Unnamed:") and frame[column].isna().all()
    ]
    if not empty_unnamed_columns:
        return frame
    return frame.drop(columns=empty_unnamed_columns)


def _validate_columns(
    *,
    frame: pd.DataFrame,
    required_columns: list[str],
    frame_name: str,
) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise CoreDataError(f"{frame_name} is missing required columns: {missing}.")


def _prepare_metadata(frame: pd.DataFrame, frame_name: str) -> pd.DataFrame:
    parsed_datetime = pd.to_datetime(frame["datetime"], errors="coerce")
    missing_datetime = int(parsed_datetime.isna().sum())
    if missing_datetime:
        raise CoreDataError(
            f"{frame_name} has {missing_datetime} rows with invalid datetime values."
        )
    return pd.DataFrame(
        {
            "datetime": parsed_datetime,
            "profile_code": frame["profile_code"].to_numpy(),
            "profile_name": frame["profile_name"].to_numpy(),
        },
        index=frame.index,
    )


def _numeric_frame(
    frame: pd.DataFrame,
    columns: list[str],
    value_name: str,
) -> pd.DataFrame:
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    _validate_finite_numeric(numeric, value_name)
    return numeric


def _numeric_series(
    frame: pd.DataFrame,
    column: str,
    value_name: str,
) -> pd.Series:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    _validate_finite_numeric(numeric.to_frame(name=column), value_name)
    numeric.name = column
    return numeric


def _validate_finite_numeric(values: pd.DataFrame, value_name: str) -> None:
    nan_by_column = values.isna().sum()
    nan_by_column = nan_by_column[nan_by_column > 0]
    if not nan_by_column.empty:
        details = ", ".join(
            f"{column}={count}" for column, count in nan_by_column.astype(int).items()
        )
        raise CoreDataError(f"{value_name} contains NaN or non-numeric values: {details}.")

    data = values.to_numpy(dtype=float)
    if not np.isfinite(data).all():
        invalid_total = int((~np.isfinite(data)).sum())
        raise CoreDataError(f"{value_name} contains {invalid_total} Inf values.")


def _validate_nonnegative_target(target: pd.Series, value_name: str) -> None:
    negative_count = int((target < 0).sum())
    if negative_count:
        raise CoreDataError(f"{value_name} contains {negative_count} negative load_kwh values.")


def _load_scaler_stats(
    *,
    path: Path,
    expected_columns: list[str],
    stats_name: str,
) -> pd.DataFrame:
    stats = _read_csv(path)
    _validate_columns(
        frame=stats,
        required_columns=["column", "min", "max", "range"],
        frame_name=stats_name,
    )

    missing = [
        column for column in expected_columns if column not in set(stats["column"].astype(str))
    ]
    if missing:
        raise CoreDataError(f"{stats_name} is missing scaler rows for: {missing}.")

    numeric_stats = _numeric_frame(stats, ["min", "max", "range"], stats_name)
    if (numeric_stats["range"] <= 0).any():
        invalid = stats.loc[numeric_stats["range"] <= 0, "column"].tolist()
        raise CoreDataError(f"{stats_name} has non-positive range for: {invalid}.")

    stats = stats.copy()
    stats.loc[:, ["min", "max", "range"]] = numeric_stats
    return stats
