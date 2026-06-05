from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.config.paths import (
    PROCESSED_RAW_CORE_DIR,
    PROCESSED_RAW_EXTENDED_DIR,
    PROCESSED_SCALED_CORE_DIR,
    PROCESSED_SCALED_EXTENDED_DIR,
    PROCESSED_STATS_DIR,
    RAW_DATA_DIR,
    ROOT_DIR,
)
from src.data.utils.eda_utils import explore_dataframe

sys.stdout.reconfigure(encoding="utf-8")


INPUT_FILE = RAW_DATA_DIR / "hanoi_load_dataset.csv"
OUTPUT_DIRS = (
    PROCESSED_RAW_CORE_DIR,
    PROCESSED_RAW_EXTENDED_DIR,
    PROCESSED_SCALED_CORE_DIR,
    PROCESSED_SCALED_EXTENDED_DIR,
    PROCESSED_STATS_DIR,
)

TRAIN_END = "2025-01-01"

TARGET_COLUMNS = {
    "1h": "target_1h",
    "24h": "target_24h",
}

CORE_FEATURES = [
    "apparent_temperature",
    "humidity",
    "hour_sin",
    "hour_cos",
    "occupancy_level",
    "load_lag_24",
]

EXTENDED_FEATURES = [
    "household_size",
    "temperature",
    "humidity",
    "apparent_temperature",
    "rain",
    "wind_speed",
    "cloud_cover",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "holiday_effect",
    "usage_period",
    "working_hour",
    "sleep_hour",
    "morning_peak",
    "evening_peak",
    "meal_activity",
    "school_vacation",
    "cooling_degree",
    "heating_degree",
    "humid_hot",
    "rainy_hour",
    "heavy_rain",
    "dark_or_cloudy",
    "occupancy_share",
    "occupancy_level",
    "load_lag_1",
    "load_lag_24",
    "load_rolling_3",
    "load_rolling_24",
]

META_COLUMNS = [
    "datetime",
    "profile_code",
    "profile_name",
]

DIAGNOSTIC_COLUMNS = [
    "laundry_activity",
    "cooking_spike",
    "appliance_spike",
    "ac_startup_spike",
    "outage_anomaly",
]


def read_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_FILE}. Run build_hanoi_load_dataset.py first."
        )

    df = pd.read_csv(INPUT_FILE, parse_dates=["datetime"])
    df = df.sort_values(["profile_code", "datetime"]).reset_index(drop=True)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = set(
        META_COLUMNS
        + CORE_FEATURES
        + EXTENDED_FEATURES
        + DIAGNOSTIC_COLUMNS
        + list(TARGET_COLUMNS.values())
    )

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["datetime"] < TRAIN_END].copy()
    test_df = df[df["datetime"] >= TRAIN_END].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split is empty. Check TRAIN_END.")

    return train_df, test_df


def calculate_minmax_stats(train_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    stats = pd.DataFrame(
        {
            "column": columns,
            "min": train_df[columns].min().to_numpy(dtype=np.float64),
            "max": train_df[columns].max().to_numpy(dtype=np.float64),
        }
    )

    stats["range"] = (stats["max"] - stats["min"]).replace(0, 1.0)
    return stats


def minmax_scale(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    result = df.copy()

    min_values = stats.set_index("column").loc[columns, "min"]
    range_values = stats.set_index("column").loc[columns, "range"]

    result[columns] = (
        result[columns] - min_values.to_numpy()
    ) / range_values.to_numpy()

    result[columns] = result[columns].clip(-0.25, 1.25)
    return result


def remove_invalid_rows(
    df: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    columns_to_check = sorted(
        set(CORE_FEATURES + EXTENDED_FEATURES + [target_column])
    )

    before = len(df)

    result = df.dropna(subset=columns_to_check).copy()
    result = result[result[target_column] >= 0].copy()

    removed = before - len(result)

    if removed:
        print(f"Removed invalid rows for {target_column}: {removed}")

    return result.reset_index(drop=True)


def build_raw_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    columns = META_COLUMNS + feature_columns + [target_column]
    return df[columns].copy()


def build_scaled_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
    feature_stats: pd.DataFrame,
    target_stats: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    columns = META_COLUMNS + feature_columns + [target_column]

    scaled = minmax_scale(
        df[columns].copy(),
        feature_stats,
        feature_columns,
    )

    target_min = float(target_stats.loc[0, "min"])
    target_range = float(target_stats.loc[0, "range"])

    scaled[f"{target_column}_scaled"] = (
        (scaled[target_column] - target_min) / target_range
    ).clip(0.0, 1.25)

    return scaled


def format_project_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def write_feature_config() -> Path:
    config = {
        "input_file": format_project_path(INPUT_FILE),
        "train_end_exclusive": TRAIN_END,
        "target_columns": TARGET_COLUMNS,
        "scaling": "minmax fitted on train split only",
        "output_dirs": {
            "raw_core": format_project_path(PROCESSED_RAW_CORE_DIR),
            "raw_extended": format_project_path(PROCESSED_RAW_EXTENDED_DIR),
            "scaled_core": format_project_path(PROCESSED_SCALED_CORE_DIR),
            "scaled_extended": format_project_path(PROCESSED_SCALED_EXTENDED_DIR),
            "stats": format_project_path(PROCESSED_STATS_DIR),
        },
        "core_features": CORE_FEATURES,
        "extended_features": EXTENDED_FEATURES,
        "diagnostic_not_default_features": DIAGNOSTIC_COLUMNS,
        "notes": [
            "Core features are intended for ANFIS.",
            "Extended features are intended for baseline machine learning models.",
            "target_1h is used for 1-hour-ahead forecasting.",
            "target_24h is used for 24-hour-ahead forecasting.",
        ],
    }

    output_path = PROCESSED_STATS_DIR / "feature_config.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    return output_path


def write_summary(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    horizon_name: str,
) -> Path:
    summary = pd.DataFrame(
        [
            {
                "horizon": horizon_name,
                "target_column": target_column,
                "split": "all",
                "rows": len(df),
                "start_datetime": df["datetime"].min(),
                "end_datetime": df["datetime"].max(),
                "target_mean": df[target_column].mean(),
                "target_std": df[target_column].std(),
                "target_min": df[target_column].min(),
                "target_max": df[target_column].max(),
            },
            {
                "horizon": horizon_name,
                "target_column": target_column,
                "split": "train",
                "rows": len(train_df),
                "start_datetime": train_df["datetime"].min(),
                "end_datetime": train_df["datetime"].max(),
                "target_mean": train_df[target_column].mean(),
                "target_std": train_df[target_column].std(),
                "target_min": train_df[target_column].min(),
                "target_max": train_df[target_column].max(),
            },
            {
                "horizon": horizon_name,
                "target_column": target_column,
                "split": "test",
                "rows": len(test_df),
                "start_datetime": test_df["datetime"].min(),
                "end_datetime": test_df["datetime"].max(),
                "target_mean": test_df[target_column].mean(),
                "target_std": test_df[target_column].std(),
                "target_min": test_df[target_column].min(),
                "target_max": test_df[target_column].max(),
            },
        ]
    )

    output_path = PROCESSED_STATS_DIR / f"preprocessing_summary_{horizon_name}.csv"
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def process_one_horizon(
    df: pd.DataFrame,
    horizon_name: str,
    target_column: str,
) -> list[Path]:
    print(f"\nProcessing horizon: {horizon_name} | Target: {target_column}")

    horizon_df = remove_invalid_rows(df, target_column)
    train_df, test_df = split_by_time(horizon_df)

    explore_dataframe(
        horizon_df,
        f"DỮ LIỆU SAU TIỀN XỬ LÝ - {horizon_name}",
        target_column=target_column,
    )

    explore_dataframe(
        train_df,
        f"TẬP TRAIN - {horizon_name}",
        target_column=target_column,
    )

    explore_dataframe(
        test_df,
        f"TẬP TEST - {horizon_name}",
        target_column=target_column,
    )

    all_feature_columns = sorted(set(CORE_FEATURES + EXTENDED_FEATURES))

    feature_stats = calculate_minmax_stats(train_df, all_feature_columns)
    target_stats = calculate_minmax_stats(train_df, [target_column])

    feature_stats_path = PROCESSED_STATS_DIR / f"feature_scaler_stats_{horizon_name}.csv"
    feature_stats.to_csv(feature_stats_path, index=False, encoding="utf-8-sig")

    target_stats_path = PROCESSED_STATS_DIR / f"target_scaler_stats_{horizon_name}.csv"
    target_stats.to_csv(target_stats_path, index=False, encoding="utf-8-sig")

    outputs = {
        PROCESSED_RAW_CORE_DIR / f"train_{horizon_name}.csv": build_raw_frame(
            train_df,
            CORE_FEATURES,
            target_column,
        ),
        PROCESSED_RAW_CORE_DIR / f"test_{horizon_name}.csv": build_raw_frame(
            test_df,
            CORE_FEATURES,
            target_column,
        ),
        PROCESSED_SCALED_CORE_DIR / f"train_{horizon_name}.csv": build_scaled_frame(
            train_df,
            CORE_FEATURES,
            feature_stats,
            target_stats,
            target_column,
        ),
        PROCESSED_SCALED_CORE_DIR / f"test_{horizon_name}.csv": build_scaled_frame(
            test_df,
            CORE_FEATURES,
            feature_stats,
            target_stats,
            target_column,
        ),
        PROCESSED_RAW_EXTENDED_DIR / f"train_{horizon_name}.csv": build_raw_frame(
            train_df,
            EXTENDED_FEATURES,
            target_column,
        ),
        PROCESSED_RAW_EXTENDED_DIR / f"test_{horizon_name}.csv": build_raw_frame(
            test_df,
            EXTENDED_FEATURES,
            target_column,
        ),
        PROCESSED_SCALED_EXTENDED_DIR / f"train_{horizon_name}.csv": build_scaled_frame(
            train_df,
            EXTENDED_FEATURES,
            feature_stats,
            target_stats,
            target_column,
        ),
        PROCESSED_SCALED_EXTENDED_DIR / f"test_{horizon_name}.csv": build_scaled_frame(
            test_df,
            EXTENDED_FEATURES,
            feature_stats,
            target_stats,
            target_column,
        ),
    }

    for output_path, frame in outputs.items():
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary_path = write_summary(
        horizon_df,
        train_df,
        test_df,
        target_column,
        horizon_name,
    )

    print(f"Rows {horizon_name}: all={len(horizon_df):,}, train={len(train_df):,}, test={len(test_df):,}")

    return [
        *outputs.keys(),
        feature_stats_path,
        target_stats_path,
        summary_path,
    ]


def main() -> None:
    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)

    df = read_dataset()
    validate_columns(df)

    all_outputs = []

    for horizon_name, target_column in TARGET_COLUMNS.items():
        outputs = process_one_horizon(
            df,
            horizon_name,
            target_column,
        )

        all_outputs.extend(outputs)

    all_outputs.append(write_feature_config())

    print("\nProcessed directories:")
    for output_dir in OUTPUT_DIRS:
        print(f"- {format_project_path(output_dir)}")

    print(f"\nCore features ({len(CORE_FEATURES)}):")
    print(CORE_FEATURES)

    print(f"\nExtended features ({len(EXTENDED_FEATURES)}):")
    print(EXTENDED_FEATURES)

    print("\nGenerated files:")
    for output_path in sorted(all_outputs):
        print(f"- {format_project_path(output_path)}")


if __name__ == "__main__":
    main()