from __future__ import annotations

from pathlib import Path

import numpy as np

from src.model.anfis import ANFIS

from .conftest import CORE_FEATURES


def test_anfis_initializes_memberships_and_forward_is_finite() -> None:
    x = _sample_inputs()
    y = 0.2 + x @ np.array([0.10, -0.05, 0.04, 0.03, 0.08, 0.20])
    model = ANFIS(n_mfs=2, feature_order=CORE_FEATURES, ridge_alpha=1e-3)

    model.initialize_memberships(x)
    model.fit_consequents(x, y)
    predictions = model.forward(x)

    assert model.mu.shape == (len(CORE_FEATURES), 2)
    assert model.sigma.shape == (len(CORE_FEATURES), 2)
    assert np.isfinite(model.mu).all()
    assert np.isfinite(model.sigma).all()
    assert (model.sigma > 0).all()
    assert predictions.shape == (x.shape[0],)
    assert np.isfinite(predictions).all()


def test_anfis_save_load_preserves_parameters_and_predictions(tmp_path: Path) -> None:
    x = _sample_inputs()
    y = 0.3 + x @ np.array([0.02, 0.03, -0.01, 0.04, 0.05, 0.06])
    model = ANFIS(
        n_mfs=2,
        feature_order=CORE_FEATURES,
        ridge_alpha=1e-3,
        metadata={"test_case": "save_load"},
    )
    model.initialize_memberships(x).fit_consequents(x, y)
    path = tmp_path / "model.npz"

    model.save_model(path)
    loaded = ANFIS.load_model(path)

    assert np.allclose(loaded.mu, model.mu)
    assert np.allclose(loaded.sigma, model.sigma)
    assert np.array_equal(loaded.rule_indices, model.rule_indices)
    assert np.allclose(loaded.consequent_coefficients, model.consequent_coefficients)
    assert loaded.feature_order == model.feature_order
    assert loaded.metadata["test_case"] == "save_load"
    assert np.allclose(loaded.predict(x), model.predict(x))

    blank = ANFIS(n_mfs=1, n_inputs=len(CORE_FEATURES))
    blank.load_model(path)
    assert np.allclose(blank.predict(x), model.predict(x))


def _sample_inputs() -> np.ndarray:
    return np.array(
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