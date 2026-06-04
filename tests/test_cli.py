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
    run_dirs = sorted((results_dir / "core").glob("pytest_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    expected_artifacts = [
        "config.json",
        "model.npz",
        "training_log.csv",
        "predictions_test.csv",
        "metrics.json",
        "rule_summary.csv",
        "actual_vs_predicted.png",
        "residuals.png",
    ]
    for artifact_name in expected_artifacts:
        artifact_path = run_dir / artifact_name
        assert artifact_path.is_file(), f"Missing CLI artifact: {artifact_name}"
        assert artifact_path.stat().st_size > 0, f"Empty CLI artifact: {artifact_name}"

    assert (run_dir / "actual_vs_predicted.png").read_bytes().startswith(b"\x89PNG")
    assert (run_dir / "residuals.png").read_bytes().startswith(b"\x89PNG")

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["unit"] == "kWh"
    assert metrics["baselines"]["lag24"]["missing_count"] == 0
    assert metrics["baselines"]["lag24"]["independent_from_model_output"] is True

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["plot_artifacts_status"] == "generated"
    assert config["evaluation"]["visualizations"]["actual_vs_predicted"]["window_rows"] == 24

    predictions = pd.read_csv(run_dir / "predictions_test.csv")
    assert {"actual_kwh", "predicted_kwh", "baseline_lag24_kwh"}.issubset(
        predictions.columns
    )
    assert len(predictions) == 48

    rule_summary = pd.read_csv(run_dir / "rule_summary.csv")
    assert not rule_summary.empty
