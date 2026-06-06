from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.model.data_loader import (
    SCALED_TARGET_COLUMN,
    TARGET_COLUMN,
    load_core_data,
    split_train_val_test,
)

from .conftest import CORE_FEATURES, write_core_processed_dir


def test_load_core_data_schema_and_split_boundaries(synthetic_processed_dir: Path) -> None:
    bundle = load_core_data(synthetic_processed_dir)

    assert bundle.config["target_column"] == TARGET_COLUMN
    assert list(bundle.train.features.columns) == CORE_FEATURES
    assert list(bundle.test.features.columns) == CORE_FEATURES
    assert SCALED_TARGET_COLUMN in bundle.train.scaled_frame.columns
    assert bundle.train.target_kwh.equals(bundle.train.raw_frame[TARGET_COLUMN])
    assert bundle.test.target_kwh.equals(bundle.test.raw_frame[TARGET_COLUMN])

    split_boundary = pd.Timestamp("2025-01-01")
    assert bundle.train.metadata["datetime"].max() < split_boundary
    assert bundle.test.metadata["datetime"].min() >= split_boundary


def test_split_train_val_test_keeps_validation_and_2025_test(
    synthetic_processed_dir: Path,
) -> None:
    bundle = load_core_data(synthetic_processed_dir)
    train_fit, validation, test = split_train_val_test(
        bundle,
        val_start="2024-01-01",
    )

    validation_start = pd.Timestamp("2024-01-01")
    assert train_fit.metadata["datetime"].max() < validation_start
    assert validation.metadata["datetime"].min() >= validation_start
    assert test.metadata["datetime"].min() == pd.Timestamp("2025-01-01")
    assert len(train_fit.features) == 48
    assert len(validation.features) == 48
    assert len(test.features) == 48


def test_load_core_data_fails_when_schema_missing(tmp_path: Path) -> None:
    processed_dir = write_core_processed_dir(
        tmp_path / "processed",
        include_feature_config=False,
    )

    with pytest.raises(FileNotFoundError, match="feature_config.json"):
        load_core_data(processed_dir)