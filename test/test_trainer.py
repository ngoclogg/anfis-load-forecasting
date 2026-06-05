from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import paths as project_paths
from src.model.anfis import ANFIS
from src.model.trainer import ANFISTrainer

from .conftest import CORE_FEATURES


def test_trainer_fits_and_saves_model_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    results_dir = tmp_path / "results"
    monkeypatch.setattr(project_paths, "RESULTS_DIR", results_dir)

    x = np.array(
        [
            [0.10, 0.80, 0.50, 1.00, 0.20, 0.15],
            [0.20, 0.75, 0.63, 0.98, 0.25, 0.18],
            [0.30, 0.65, 0.75, 0.93, 0.40, 0.22],
            [0.45, 0.55, 0.85, 0.75, 0.70, 0.30],
            [0.60, 0.45, 0.95, 0.55, 0.80, 0.35],
            [0.70, 0.40, 0.50, 0.05, 0.75, 0.45],
            [0.55, 0.50, 0.15, 0.25, 0.50, 0.40],
            [0.35, 0.70, 0.25, 0.85, 0.30, 0.28],
        ],
        dtype=float,
    )
    y = 0.2 + x @ np.array([0.10, -0.05, 0.04, 0.03, 0.08, 0.20])
    trainer = ANFISTrainer(
        n_mfs=2,
        mf_type="gaussian",
        ridge_alpha=1e-3,
        feature_order=CORE_FEATURES,
        horizon="1h",
        random_state=123,
        metadata={"test_case": "trainer"},
    )

    result = trainer.train(
        x,
        y,
        save_artifact=True,
        model_filename="pytest_trainer",
        metadata={"run_id": "unit"},
    )

    expected_path = results_dir / "1h" / "models" / "pytest_trainer.npz"
    assert result.model_path == expected_path
    assert expected_path.is_file()
    assert result.train_predictions.shape == (x.shape[0],)
    assert np.isfinite(result.train_predictions).all()
    assert result.train_rmse >= 0.0
    assert result.metadata["mf_type"] == "gaussian"
    assert result.metadata["horizon"] == "1h"
    assert result.metadata["test_case"] == "trainer"
    assert result.metadata["run_id"] == "unit"

    loaded = ANFIS.load_model(expected_path)
    assert np.allclose(loaded.predict(x), result.model.predict(x))