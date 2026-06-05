"""NumPy ANFIS core model for the hourly load forecasting pipeline."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_CORE_FEATURES = (
    "apparent_temperature",
    "humidity",
    "hour_sin",
    "hour_cos",
    "occupancy_level",
    "load_lag_24",
)
DEFAULT_SEED = 20260527
MODEL_VERSION = "1.0"


class ANFIS:
    """
    Five-layer ANFIS with Gaussian MFs and first-order Sugeno consequents.

    The default configuration is the Core Global MVP model: six scaled inputs,
    two membership functions per input, and a 2^6 rule grid.
    """

    def __init__(
        self,
        *,
        n_inputs: int | None = None,
        n_mfs: int = 2,
        feature_order: Sequence[str] | None = None,
        mu: np.ndarray | Sequence[Sequence[float]] | None = None,
        sigma: np.ndarray | Sequence[Sequence[float]] | None = None,
        rule_indices: np.ndarray | Sequence[Sequence[int]] | None = None,
        consequent_coefficients: np.ndarray | Sequence[Sequence[float]] | None = None,
        ridge_alpha: float = 1e-4,
        eps: float = 1e-12,
        min_sigma: float = 0.10,
        random_state: int = DEFAULT_SEED,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if n_inputs is None and feature_order is None and mu is not None:
            n_inputs = int(np.asarray(mu).shape[0])

        self.n_mfs = _validate_positive_int(n_mfs, "n_mfs")
        self.feature_order = _resolve_feature_order(feature_order, n_inputs)
        self.n_inputs = len(self.feature_order)
        self.ridge_alpha = _validate_positive_float(ridge_alpha, "ridge_alpha", allow_zero=True)
        self.eps = _validate_positive_float(eps, "eps")
        self.min_sigma = _validate_positive_float(min_sigma, "min_sigma")
        self.random_state = int(random_state)

        self.mu = self._coerce_parameter_matrix(mu, "mu") if mu is not None else self._default_mu()
        self.sigma = (
            self._coerce_parameter_matrix(sigma, "sigma")
            if sigma is not None
            else self._default_sigma()
        )
        if (self.sigma <= 0).any():
            raise ValueError("sigma values must be positive.")

        self.rule_indices = (
            self._coerce_rule_indices(rule_indices)
            if rule_indices is not None
            else self._build_rule_indices()
        )
        self.consequent_coefficients = (
            self._coerce_consequent_coefficients(consequent_coefficients)
            if consequent_coefficients is not None
            else np.zeros((self.n_rules, self.n_inputs + 1), dtype=float)
        )
        self.metadata = self._build_metadata(metadata)

    @property
    def n_rules(self) -> int:
        """Number of Sugeno rules in the rule grid."""
        return int(self.rule_indices.shape[0])

    def initialize_memberships(self, x: np.ndarray | Sequence[Sequence[float]]) -> "ANFIS":
        """
        Initialize Gaussian centers and sigmas from scaled train-fit inputs.

        For the default two MFs, centers are q25/q75 per feature and sigma is
        max((q75 - q25) / 2, min_sigma). Near-constant features fall back to
        default scaled-domain centers.
        """
        x_array = self._as_input_array(x)
        quantile_levels = (
            np.array([0.25, 0.75], dtype=float)
            if self.n_mfs == 2
            else np.linspace(0.25, 0.75, self.n_mfs, dtype=float)
        )
        centers = np.quantile(x_array, quantile_levels, axis=0).T.astype(float)
        default_centers = self._default_mu()

        spans = centers[:, -1] - centers[:, 0] if self.n_mfs > 1 else np.zeros(self.n_inputs)
        near_constant = spans <= self.eps
        if near_constant.any():
            centers[near_constant, :] = default_centers[near_constant, :]
            spans = centers[:, -1] - centers[:, 0] if self.n_mfs > 1 else spans

        if self.n_mfs == 1:
            sigma_values = np.full(self.n_inputs, self.min_sigma, dtype=float)
        else:
            sigma_values = np.maximum(spans / 2.0, self.min_sigma)

        self.mu = centers
        self.sigma = np.repeat(sigma_values[:, np.newaxis], self.n_mfs, axis=1)
        return self

    def membership_values(self, x: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """
        Layer 1: Gaussian membership values.

        Returns an array with shape (n_samples, n_inputs, n_mfs).
        """
        x_array = self._as_input_array(x)
        return np.exp(self._log_membership_values(x_array))

    def firing_strengths(self, x: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """
        Layer 2: product rule firing strengths.

        Returns an array with shape (n_samples, n_rules).
        """
        x_array = self._as_input_array(x)
        log_strengths = self._rule_log_strengths(x_array)
        strengths = np.exp(np.clip(log_strengths, -745.0, 709.0))
        return np.nan_to_num(strengths, nan=0.0, posinf=np.finfo(float).max, neginf=0.0)

    def normalized_firing_strengths(
        self,
        x: np.ndarray | Sequence[Sequence[float]],
    ) -> np.ndarray:
        """
        Layer 3: normalized firing strengths.

        A log-softmax style normalization keeps rows finite and summing to 1,
        even when raw firing strengths are very small.
        """
        x_array = self._as_input_array(x)
        log_strengths = self._rule_log_strengths(x_array)
        max_log = np.max(log_strengths, axis=1, keepdims=True)
        shifted = np.exp(log_strengths - max_log)
        denominator = shifted.sum(axis=1, keepdims=True)
        normalized = shifted / np.maximum(denominator, self.eps)

        bad_rows = (~np.isfinite(normalized).all(axis=1)) | (
            normalized.sum(axis=1) <= self.eps
        )
        if bad_rows.any():
            normalized[bad_rows, :] = 1.0 / self.n_rules

        row_sums = normalized.sum(axis=1, keepdims=True)
        normalized = normalized / np.maximum(row_sums, self.eps)
        return normalized

    def consequent_outputs(self, x: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """
        Layer 4: first-order Sugeno output per rule.

        Each rule output is a_r,0 + sum(a_r,i * x_i).
        """
        x_array = self._as_input_array(x)
        basis = np.column_stack([np.ones(x_array.shape[0], dtype=float), x_array])
        return basis @ self.consequent_coefficients.T

    def forward(self, x: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """
        Layers 1-5 forward pass.

        Parameters
        ----------
        x:
            Batch of scaled Core inputs with shape (n_samples, 6) by default.

        Returns
        -------
        np.ndarray
            Scaled predictions with shape (n_samples,).
        """
        x_array = self._as_input_array(x)
        normalized = self.normalized_firing_strengths(x_array)
        rule_outputs = self.consequent_outputs(x_array)
        predictions = np.sum(normalized * rule_outputs, axis=1)

        if not np.isfinite(predictions).all():
            raise FloatingPointError("ANFIS forward produced NaN or Inf predictions.")
        return predictions

    def predict(self, x: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """Run forward inference and return a NumPy array."""
        return np.asarray(self.forward(x), dtype=float)

    def design_matrix(self, x: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """
        Build the normalized Sugeno design matrix for Ridge least squares.

        The block for each rule is w_bar_r * [1, x_1, ..., x_n].
        """
        x_array = self._as_input_array(x)
        normalized = self.normalized_firing_strengths(x_array)
        basis = np.column_stack([np.ones(x_array.shape[0], dtype=float), x_array])
        return (normalized[:, :, np.newaxis] * basis[:, np.newaxis, :]).reshape(
            x_array.shape[0],
            self.n_rules * (self.n_inputs + 1),
        )

    def fit_consequents(
        self,
        x: np.ndarray | Sequence[Sequence[float]],
        y: np.ndarray | Sequence[float],
        *,
        ridge_alpha: float | None = None,
    ) -> "ANFIS":
        """Fit first-order Sugeno consequent coefficients by Ridge least squares."""
        x_array = self._as_input_array(x)
        y_array = np.asarray(y, dtype=float)
        if y_array.ndim != 1 or y_array.shape[0] != x_array.shape[0]:
            raise ValueError(
                "y must be a one-dimensional array with the same number of rows as x."
            )
        if not np.isfinite(y_array).all():
            raise ValueError("y contains NaN or Inf values.")

        alpha = self.ridge_alpha if ridge_alpha is None else _validate_positive_float(
            ridge_alpha,
            "ridge_alpha",
            allow_zero=True,
        )
        design = self.design_matrix(x_array)
        identity = np.eye(design.shape[1], dtype=float)
        lhs = design.T @ design + alpha * identity
        rhs = design.T @ y_array
        try:
            flat_coefficients = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            flat_coefficients = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

        self.consequent_coefficients = flat_coefficients.reshape(
            self.n_rules,
            self.n_inputs + 1,
        )
        self.ridge_alpha = alpha
        return self

    def save_model(self, path: str | Path) -> None:
        """Persist ANFIS parameters and metadata to a compressed .npz file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_path,
            mu=self.mu,
            sigma=self.sigma,
            rule_indices=self.rule_indices,
            consequent_coefficients=self.consequent_coefficients,
            feature_order=np.asarray(self.feature_order, dtype=str),
            n_inputs=np.asarray(self.n_inputs, dtype=np.int64),
            n_mfs=np.asarray(self.n_mfs, dtype=np.int64),
            ridge_alpha=np.asarray(self.ridge_alpha, dtype=float),
            eps=np.asarray(self.eps, dtype=float),
            min_sigma=np.asarray(self.min_sigma, dtype=float),
            random_state=np.asarray(self.random_state, dtype=np.int64),
            metadata_json=np.asarray(json.dumps(self.metadata, ensure_ascii=False)),
        )

    def load_model(self_or_path, path: str | Path | None = None) -> "ANFIS":
        """
        Load an ANFIS model from .npz.

        Supports both ``ANFIS.load_model(path)`` and ``model.load_model(path)``.
        The instance form mutates and returns the existing object.
        """
        if isinstance(self_or_path, ANFIS):
            if path is None:
                raise TypeError("path is required when load_model is called on an instance.")
            loaded = type(self_or_path)._read_model(path)
            self_or_path._copy_state_from(loaded)
            return self_or_path

        if path is not None:
            raise TypeError("ANFIS.load_model accepts exactly one path argument.")
        return ANFIS._read_model(self_or_path)

    @classmethod
    def _read_model(cls, path: str | Path) -> "ANFIS":
        input_path = Path(path)
        with np.load(input_path, allow_pickle=False) as data:
            required = {
                "mu",
                "sigma",
                "rule_indices",
                "consequent_coefficients",
                "feature_order",
                "metadata_json",
            }
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"Model artifact is missing required arrays: {missing}.")

            metadata_json = data["metadata_json"]
            metadata = json.loads(str(metadata_json.item()))
            feature_order = tuple(str(value) for value in data["feature_order"].tolist())

            return cls(
                n_inputs=int(data["n_inputs"].item()) if "n_inputs" in data else None,
                n_mfs=int(data["n_mfs"].item()) if "n_mfs" in data else int(data["mu"].shape[1]),
                feature_order=feature_order,
                mu=data["mu"],
                sigma=data["sigma"],
                rule_indices=data["rule_indices"],
                consequent_coefficients=data["consequent_coefficients"],
                ridge_alpha=(
                    float(data["ridge_alpha"].item()) if "ridge_alpha" in data else 1e-4
                ),
                eps=float(data["eps"].item()) if "eps" in data else 1e-12,
                min_sigma=(
                    float(data["min_sigma"].item()) if "min_sigma" in data else 0.10
                ),
                random_state=(
                    int(data["random_state"].item())
                    if "random_state" in data
                    else DEFAULT_SEED
                ),
                metadata=metadata,
            )

    def _copy_state_from(self, other: "ANFIS") -> None:
        self.n_mfs = other.n_mfs
        self.feature_order = other.feature_order
        self.n_inputs = other.n_inputs
        self.ridge_alpha = other.ridge_alpha
        self.eps = other.eps
        self.min_sigma = other.min_sigma
        self.random_state = other.random_state
        self.mu = other.mu.copy()
        self.sigma = other.sigma.copy()
        self.rule_indices = other.rule_indices.copy()
        self.consequent_coefficients = other.consequent_coefficients.copy()
        self.metadata = dict(other.metadata)

    def _default_mu(self) -> np.ndarray:
        if self.n_mfs == 1:
            centers = np.array([0.5], dtype=float)
        else:
            centers = np.linspace(0.33, 0.67, self.n_mfs, dtype=float)
        return np.tile(centers, (self.n_inputs, 1))

    def _default_sigma(self) -> np.ndarray:
        centers = self._default_mu()[0]
        if self.n_mfs == 1:
            sigma_value = self.min_sigma
        else:
            sigma_value = max(
                (float(centers[-1]) - float(centers[0])) / 2.0,
                self.min_sigma,
            )
        return np.full((self.n_inputs, self.n_mfs), sigma_value, dtype=float)

    def _build_rule_indices(self) -> np.ndarray:
        return np.asarray(
            list(itertools.product(range(self.n_mfs), repeat=self.n_inputs)),
            dtype=np.int64,
        )

    def _coerce_parameter_matrix(
        self,
        values: np.ndarray | Sequence[Sequence[float]],
        name: str,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        expected_shape = (self.n_inputs, self.n_mfs)
        if array.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}; got {array.shape}.")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf values.")
        return array.copy()

    def _coerce_rule_indices(
        self,
        values: np.ndarray | Sequence[Sequence[int]],
    ) -> np.ndarray:
        array = np.asarray(values, dtype=np.int64)
        if array.ndim != 2 or array.shape[1] != self.n_inputs:
            raise ValueError(
                "rule_indices must have shape (n_rules, n_inputs); "
                f"got {array.shape}."
            )
        if array.shape[0] == 0:
            raise ValueError("rule_indices must define at least one rule.")
        if (array < 0).any() or (array >= self.n_mfs).any():
            raise ValueError("rule_indices contains MF indices outside [0, n_mfs).")
        return array.copy()

    def _coerce_consequent_coefficients(
        self,
        values: np.ndarray | Sequence[Sequence[float]],
    ) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        expected_shape = (self.n_rules, self.n_inputs + 1)
        if array.shape != expected_shape:
            raise ValueError(
                "consequent_coefficients must have shape "
                f"{expected_shape}; got {array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError("consequent_coefficients contains NaN or Inf values.")
        return array.copy()

    def _build_metadata(self, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        base = {
            "model_version": MODEL_VERSION,
            "model_scope": "global",
            "architecture": "anfis-5-layer-gaussian-first-order-sugeno",
            "n_inputs": self.n_inputs,
            "n_mfs": self.n_mfs,
            "n_rules": self.n_rules,
            "seed": self.random_state,
        }
        if metadata:
            base.update(dict(metadata))
        return base

    def _as_input_array(self, x: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        array = np.asarray(x, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError(f"x must be a 2D array; got shape {array.shape}.")
        if array.shape[1] != self.n_inputs:
            raise ValueError(
                f"x must have {self.n_inputs} columns; got {array.shape[1]}."
            )
        if not np.isfinite(array).all():
            raise ValueError("x contains NaN or Inf values.")
        return array

    def _log_membership_values(self, x_array: np.ndarray) -> np.ndarray:
        z = (x_array[:, :, np.newaxis] - self.mu[np.newaxis, :, :]) / self.sigma[
            np.newaxis,
            :,
            :,
        ]
        z = np.clip(z, -1e6, 1e6)
        return -0.5 * z * z

    def _rule_log_strengths(self, x_array: np.ndarray) -> np.ndarray:
        log_memberships = self._log_membership_values(x_array)
        selected = np.empty((x_array.shape[0], self.n_rules, self.n_inputs), dtype=float)
        for input_index in range(self.n_inputs):
            selected[:, :, input_index] = log_memberships[
                :,
                input_index,
                self.rule_indices[:, input_index],
            ]
        return selected.sum(axis=2)


def _resolve_feature_order(
    feature_order: Sequence[str] | None,
    n_inputs: int | None,
) -> tuple[str, ...]:
    if feature_order is None:
        if n_inputs is None or int(n_inputs) == len(DEFAULT_CORE_FEATURES):
            return DEFAULT_CORE_FEATURES
        return tuple(f"x_{index + 1}" for index in range(_validate_positive_int(n_inputs, "n_inputs")))

    features = tuple(str(feature) for feature in feature_order)
    if not features or any(not feature for feature in features):
        raise ValueError("feature_order must contain non-empty feature names.")
    if len(set(features)) != len(features):
        raise ValueError("feature_order contains duplicate feature names.")

    if n_inputs is not None and int(n_inputs) != len(features):
        raise ValueError(
            f"n_inputs={n_inputs} does not match feature_order length {len(features)}."
        )
    return features


def _validate_positive_int(value: int, name: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _validate_positive_float(
    value: float,
    name: str,
    *,
    allow_zero: bool = False,
) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    if allow_zero:
        if number < 0:
            raise ValueError(f"{name} must be non-negative.")
    elif number <= 0:
        raise ValueError(f"{name} must be positive.")
    return number