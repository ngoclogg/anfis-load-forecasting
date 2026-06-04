from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .conftest import write_core_processed_dir


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_train_cli_small_data_writes_expected_artifacts(tmp_path: Path) -> None:
    processed_dir = write_core_processed_dir(tmp_path / "processed")
    results_dir = tmp_path / "results"
    command = [
        sys.executable,
        "src/model/train_anfis_hourly.py",
        "--processed-dir",
        str(processed_dir),
        "--results-dir",
        str(results_dir),
        "--run-name",
        "pytest",
        "--horizons",
        "1h",
        "--validation-start",
        "2024-01-01",
        "--plot-days",
        "1",
        "--n-mfs",
        "1",
        "--seed",
        "123",
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["MPLBACKEND"] = "Agg"

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    model_paths = sorted((results_dir / "1h" / "models").glob("pytest_1h_*_anfis_model.npz"))
    assert len(model_paths) == 1
    run_id = model_paths[0].stem.removesuffix("_anfis_model")

    expected_artifacts = {
        "model": model_paths[0],
        "config": results_dir / "1h" / "metrics" / f"{run_id}_config.json",
        "metrics_json": results_dir / "1h" / "metrics" / f"{run_id}_metrics.json",
        "metrics_csv": results_dir / "1h" / "metrics" / f"{run_id}_metrics.csv",
        "training_log": results_dir / "1h" / "metrics" / f"{run_id}_training_log.csv",
        "predictions": results_dir / "1h" / "predictions" / f"{run_id}_test_predictions.csv",
        "actual_vs_predicted": results_dir / "1h" / "plots" / f"{run_id}_actual_vs_predicted.png",
        "residuals": results_dir / "1h" / "plots" / f"{run_id}_residuals.png",
        "membership_functions": results_dir / "1h" / "plots" / f"{run_id}_membership_functions.png",
    }
    for artifact_name, artifact_path in expected_artifacts.items():
        assert artifact_path.is_file(), f"Missing CLI artifact: {artifact_name}"
        assert artifact_path.stat().st_size > 0, f"Empty CLI artifact: {artifact_name}"

    assert expected_artifacts["actual_vs_predicted"].read_bytes().startswith(b"\x89PNG")
    assert expected_artifacts["residuals"].read_bytes().startswith(b"\x89PNG")
    assert expected_artifacts["membership_functions"].read_bytes().startswith(b"\x89PNG")

    metrics = json.loads(expected_artifacts["metrics_json"].read_text(encoding="utf-8"))
    assert metrics["unit"] == "kWh"
    assert metrics["horizon"] == "1h"
    assert metrics["baselines"]["lag24"]["missing_count"] == 0
    assert metrics["baselines"]["lag24"]["independent_from_model_output"] is True
    assert metrics["split_rows"]["test"] == 48

    config = json.loads(expected_artifacts["config"].read_text(encoding="utf-8"))
    assert config["horizon"] == "1h"
    assert config["artifact_paths"]["metrics_json"].endswith(f"{run_id}_metrics.json")

    predictions = pd.read_csv(expected_artifacts["predictions"])
    assert {"actual_kwh", "predicted_kwh", "baseline_lag24_kwh"}.issubset(
        predictions.columns
    )
    assert len(predictions) == 48
