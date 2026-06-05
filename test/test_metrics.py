from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.model.data_loader import inverse_transform_target, load_core_data
from src.model.train_anfis_hourly import (
    BASELINE_LAG24_FEATURE,
    absolute_percentage_error,
    evaluate_test_split,
    regression_metrics,
    resolve_baseline_lag24_kwh,
)


def test_regression_metrics_expected_values() -> None:
    actual = np.array([100.0, 200.0, 300.0])
    prediction = np.array([110.0, 190.0, 330.0])

    metrics = regression_metrics(actual, prediction)

    assert metrics["mae"] == pytest.approx((10.0 + 10.0 + 30.0) / 3.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt((100.0 + 100.0 + 900.0) / 3.0))
    assert metrics["mape"] == pytest.approx((10.0 + 5.0 + 10.0) / 3.0)
    assert metrics["r2"] == pytest.approx(1.0 - 1100.0 / 20000.0)
    assert np.allclose(absolute_percentage_error(actual, prediction), [10.0, 5.0, 10.0])


def test_inverse_transform_target_uses_kwh_scaler(synthetic_processed_dir: Path) -> None:
    bundle = load_core_data(synthetic_processed_dir)

    restored = inverse_transform_target(bundle, np.array([0.0, 0.5, 1.0]))

    assert np.allclose(restored, [100.0, 150.0, 200.0])


def test_evaluate_test_split_metrics_are_computed_on_kwh(
    synthetic_processed_dir: Path,
) -> None:
    bundle = load_core_data(synthetic_processed_dir)
    model = _ConstantScaledModel(value=0.25)

    evaluation = evaluate_test_split(
        model=model,
        bundle=bundle,
        test=bundle.test,
        run_id="pytest_run",
        timestamp="2026-06-04T00:00:00",
        feature_set="core",
    )

    actual_kwh = bundle.test.target_kwh.to_numpy(dtype=float)
    predicted_kwh = np.full_like(actual_kwh, 125.0, dtype=float)
    expected_mae_kwh = float(np.mean(np.abs(actual_kwh - predicted_kwh)))
    scaled_mae = float(
        np.mean(np.abs(bundle.test.target_scaled.to_numpy(dtype=float) - 0.25))
    )

    assert evaluation.metrics["unit"] == "kWh"
    assert evaluation.metrics["target_column"] == "load_kwh"
    assert evaluation.metrics["anfis"]["mae"] == pytest.approx(expected_mae_kwh)
    assert evaluation.metrics["anfis"]["mae"] != pytest.approx(scaled_mae)
    assert np.allclose(evaluation.predictions["predicted_kwh"], predicted_kwh)
    assert "lag24" in evaluation.metrics["baselines"]


def test_missing_lag24_baseline_fails_clearly(synthetic_processed_dir: Path) -> None:
    bundle = load_core_data(synthetic_processed_dir)
    test_without_baseline = replace(
        bundle.test,
        features=bundle.test.features.drop(columns=[BASELINE_LAG24_FEATURE]),
        raw_frame=bundle.test.raw_frame.drop(columns=[BASELINE_LAG24_FEATURE]),
        scaled_frame=bundle.test.scaled_frame.drop(columns=[BASELINE_LAG24_FEATURE]),
    )

    with pytest.raises(ValueError, match="Cannot build Lag-24 baseline"):
        resolve_baseline_lag24_kwh(bundle, test_without_baseline)


class _ConstantScaledModel:
    def __init__(self, *, value: float) -> None:
        self.value = float(value)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.value, dtype=float)