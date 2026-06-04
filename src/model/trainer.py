"""Reusable ANFIS training orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.config import paths as project_paths
from src.model.anfis import ANFIS, DEFAULT_SEED


SUPPORTED_MF_TYPES = ("gaussian",)


@dataclass(frozen=True)
class TrainingResult:
    """Artifacts and diagnostics returned by an ANFIS training run."""

    model: ANFIS
    train_predictions: np.ndarray
    train_rmse: float
    model_path: Path | None
    metadata: dict[str, Any]


@dataclass
class ANFISTrainer:
    """
    Coordinate ANFIS model initialization, forward pass, consequent fitting, and saving.

    The core ANFIS math stays in :class:`src.model.anfis.ANFIS`. This class only
    wires the model into the project training pipeline and result directory layout.
    """

    n_mfs: int = 2
    mf_type: str = "gaussian"
    ridge_alpha: float = 1e-4
    feature_order: Sequence[str] | None = None
    horizon: str = "1h"
    model_filename: str = "anfis_model.npz"
    random_state: int = DEFAULT_SEED
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate trainer settings that are not delegated to ANFIS."""
        self.mf_type = _validate_mf_type(self.mf_type)

    def train(
        self,
        x_train: np.ndarray | Sequence[Sequence[float]],
        y_train: np.ndarray | Sequence[float],
        *,
        save_artifact: bool = False,
        model_filename: str | None = None,
        model_path: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrainingResult:
        """
        Fit an ANFIS model on scaled features and target values.

        Parameters
        ----------
        x_train:
            Scaled training inputs with shape ``(n_samples, n_features)``.
        y_train:
            Scaled training target values with shape ``(n_samples,)``.
        save_artifact:
            When true, persist the model as ``.npz`` after fitting.
        model_filename:
            Optional filename placed under ``results/{horizon}/models/``.
        model_path:
            Optional explicit output path. When provided, it takes precedence over
            ``model_filename`` and the horizon model directory.
        metadata:
            Per-run metadata merged into the model metadata.
        """
        x_array = _as_2d_float_array(x_train, "x_train")
        y_array = _as_1d_float_array(y_train, "y_train")
        if y_array.shape[0] != x_array.shape[0]:
            raise ValueError(
                "y_train must have the same number of rows as x_train; "
                f"got {y_array.shape[0]} and {x_array.shape[0]}."
            )

        model = self.build_model(metadata=metadata)
        model.initialize_memberships(x_array)
        model.fit_consequents(x_array, y_array, ridge_alpha=self.ridge_alpha)

        train_predictions = model.forward(x_array)
        if not np.isfinite(train_predictions).all():
            raise FloatingPointError("ANFIS training predictions contain NaN or Inf.")

        saved_path = (
            self.save_model(
                model,
                model_filename=model_filename,
                model_path=model_path,
            )
            if save_artifact
            else None
        )

        return TrainingResult(
            model=model,
            train_predictions=train_predictions,
            train_rmse=_rmse(train_predictions, y_array),
            model_path=saved_path,
            metadata=dict(model.metadata),
        )

    def fit(
        self,
        x_train: np.ndarray | Sequence[Sequence[float]],
        y_train: np.ndarray | Sequence[float],
        **kwargs: Any,
    ) -> TrainingResult:
        """Alias for :meth:`train` for callers that use estimator-style naming."""
        return self.train(x_train, y_train, **kwargs)

    def build_model(self, *, metadata: Mapping[str, Any] | None = None) -> ANFIS:
        """Create an ANFIS instance configured for this training run."""
        model_metadata = self._merged_metadata(metadata)
        return ANFIS(
            n_mfs=self.n_mfs,
            feature_order=self.feature_order,
            ridge_alpha=self.ridge_alpha,
            random_state=self.random_state,
            metadata=model_metadata,
        )

    def save_model(
        self,
        model: ANFIS,
        *,
        model_filename: str | None = None,
        model_path: str | Path | None = None,
    ) -> Path:
        """Save a trained ANFIS model artifact and return its path."""
        output_path = (
            Path(model_path)
            if model_path is not None
            else self.get_model_path(model_filename=model_filename)
        )
        model.save_model(output_path)
        return output_path

    def get_model_path(self, *, model_filename: str | None = None) -> Path:
        """Return the default ``results/{horizon}/models/`` artifact path."""
        models_dir = project_paths.get_results_subdir(self.horizon, "models")
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir / _normalize_npz_filename(model_filename or self.model_filename)

    def _merged_metadata(self, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        merged: dict[str, Any] = {
            "horizon": self.horizon,
            "mf_type": self.mf_type,
            "ridge_alpha": self.ridge_alpha,
        }
        if self.metadata:
            merged.update(dict(self.metadata))
        if metadata:
            merged.update(dict(metadata))
        return merged


def _validate_mf_type(mf_type: str) -> str:
    normalized = str(mf_type).strip().lower()
    if normalized not in SUPPORTED_MF_TYPES:
        allowed = ", ".join(SUPPORTED_MF_TYPES)
        raise ValueError(f"Unsupported mf_type {mf_type!r}. Expected one of: {allowed}.")
    return normalized


def _normalize_npz_filename(filename: str) -> str:
    path = Path(str(filename))
    if path.name != str(filename) or path.name in {"", ".", ".."}:
        raise ValueError("model_filename must be a filename, not a path.")
    if path.suffix and path.suffix.lower() != ".npz":
        raise ValueError("model_filename must use the .npz extension.")
    if not path.suffix:
        path = path.with_suffix(".npz")
    return path.name


def _as_2d_float_array(
    values: np.ndarray | Sequence[Sequence[float]],
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf values.")
    return array


def _as_1d_float_array(values: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array; got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf values.")
    return array


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - target))))
