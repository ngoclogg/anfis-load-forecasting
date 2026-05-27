from __future__ import annotations

from src.config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data.eda_utils import explore_dataframe

import json
import numpy as np
import pandas as pd
import sys
sys.stdout.reconfigure(encoding="utf-8")


INPUT_FILE = RAW_DATA_DIR / "hanoi_load_dataset.csv"
PROCESSED_DIR = PROCESSED_DATA_DIR

TRAIN_END = "2025-01-01"
TARGET_COLUMN = "load_kwh"

# Tập hợp tính năng nhỏ cho thí nghiệm ANFIS đầu tiên.
# 2 hàm thành viên/đầu vào cho ra 2^6 = 64 quy tắc mờ.
CORE_FEATURES = [
    "apparent_temperature",
    "humidity",
    "hour_sin",
    "hour_cos",
    "occupancy_level",
    "load_lag_24",
]

# Tập hợp tính năng lớn hơn cho các thử nghiệm hoặc lựa chọn tính năng mở rộng nếu có.
# Tránh các cờ sự kiện không xác định trong tương lai như cooking_spike/outage_anomaly.
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
        + [TARGET_COLUMN]
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


def minmax_scale(df: pd.DataFrame, stats: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    min_values = stats.set_index("column").loc[columns, "min"]
    range_values = stats.set_index("column").loc[columns, "range"]
    result[columns] = (result[columns] - min_values.to_numpy()) / range_values.to_numpy()
    result[columns] = result[columns].clip(-0.25, 1.25)
    return result


def build_raw_frame(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    columns = META_COLUMNS + feature_columns + [TARGET_COLUMN]
    return df[columns].copy()


def build_scaled_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
    feature_stats: pd.DataFrame,
    target_stats: pd.DataFrame,
) -> pd.DataFrame:
    columns = META_COLUMNS + feature_columns + [TARGET_COLUMN]
    scaled = minmax_scale(df[columns].copy(), feature_stats, feature_columns)

    target_min = float(target_stats.loc[0, "min"])
    target_range = float(target_stats.loc[0, "range"])
    scaled[f"{TARGET_COLUMN}_scaled"] = (
        (scaled[TARGET_COLUMN] - target_min) / target_range
    ).clip(0.0, 1.25)
    return scaled


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_check = sorted(set(CORE_FEATURES + EXTENDED_FEATURES + [TARGET_COLUMN]))
    before = len(df)
    result = df.dropna(subset=columns_to_check).copy()
    result = result[result[TARGET_COLUMN] >= 0].copy()
    removed = before - len(result)
    if removed:
        print(f"Removed invalid rows: {removed}")
    return result.reset_index(drop=True)


def write_feature_config() -> None:
    config = {
        "input_file": str(INPUT_FILE),
        "train_end_exclusive": TRAIN_END,
        "target_column": TARGET_COLUMN,
        "scaling": "minmax fitted on train split only",
        "core_features": CORE_FEATURES,
        "extended_features": EXTENDED_FEATURES,
        "diagnostic_not_default_features": DIAGNOSTIC_COLUMNS,
        "notes": [
            "Core features are intended for the first compact ANFIS model.",
            "Extended features are prepared for later feature selection.",
            "Spike/anomaly flags are kept for analysis but excluded from default model inputs.",
        ],
    }
    with (PROCESSED_DIR / "feature_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def write_summary(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    summary = pd.DataFrame(
        [
            {
                "split": "all",
                "rows": len(df),
                "start_datetime": df["datetime"].min(),
                "end_datetime": df["datetime"].max(),
                "target_mean": df[TARGET_COLUMN].mean(),
                "target_std": df[TARGET_COLUMN].std(),
                "target_min": df[TARGET_COLUMN].min(),
                "target_max": df[TARGET_COLUMN].max(),
            },
            {
                "split": "train",
                "rows": len(train_df),
                "start_datetime": train_df["datetime"].min(),
                "end_datetime": train_df["datetime"].max(),
                "target_mean": train_df[TARGET_COLUMN].mean(),
                "target_std": train_df[TARGET_COLUMN].std(),
                "target_min": train_df[TARGET_COLUMN].min(),
                "target_max": train_df[TARGET_COLUMN].max(),
            },
            {
                "split": "test",
                "rows": len(test_df),
                "start_datetime": test_df["datetime"].min(),
                "end_datetime": test_df["datetime"].max(),
                "target_mean": test_df[TARGET_COLUMN].mean(),
                "target_std": test_df[TARGET_COLUMN].std(),
                "target_min": test_df[TARGET_COLUMN].min(),
                "target_max": test_df[TARGET_COLUMN].max(),
            },
        ]
    )
    summary.to_csv(PROCESSED_DIR / "preprocessing_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = read_dataset()
    validate_columns(df)
    df = remove_invalid_rows(df)
    train_df, test_df = split_by_time(df)

    explore_dataframe(df, "DỮ LIỆU SAU TIỀN XỬ LÝ CƠ BẢN", target_column="load_kwh")
    explore_dataframe(train_df, "TẬP TRAIN", target_column="load_kwh")
    explore_dataframe(test_df, "TẬP TEST", target_column="load_kwh")

    all_feature_columns = sorted(set(CORE_FEATURES + EXTENDED_FEATURES))
    feature_stats = calculate_minmax_stats(train_df, all_feature_columns)
    target_stats = calculate_minmax_stats(train_df, [TARGET_COLUMN])
    feature_stats.to_csv(PROCESSED_DIR / "feature_scaler_stats.csv", index=False, encoding="utf-8-sig")
    target_stats.to_csv(PROCESSED_DIR / "target_scaler_stats.csv", index=False, encoding="utf-8-sig")

    outputs = {
        "train_core_raw.csv": build_raw_frame(train_df, CORE_FEATURES),
        "test_core_raw.csv": build_raw_frame(test_df, CORE_FEATURES),
        "train_core_scaled.csv": build_scaled_frame(
            train_df,
            CORE_FEATURES,
            feature_stats,
            target_stats,
        ),
        "test_core_scaled.csv": build_scaled_frame(
            test_df,
            CORE_FEATURES,
            feature_stats,
            target_stats,
        ),
        "train_extended_raw.csv": build_raw_frame(train_df, EXTENDED_FEATURES),
        "test_extended_raw.csv": build_raw_frame(test_df, EXTENDED_FEATURES),
        "train_extended_scaled.csv": build_scaled_frame(
            train_df,
            EXTENDED_FEATURES,
            feature_stats,
            target_stats,
        ),
        "test_extended_scaled.csv": build_scaled_frame(
            test_df,
            EXTENDED_FEATURES,
            feature_stats,
            target_stats,
        ),
    }

    for filename, frame in outputs.items():
        frame.to_csv(PROCESSED_DIR / filename, index=False, encoding="utf-8-sig")

    write_feature_config()
    write_summary(df, train_df, test_df)

    print(f"Processed directory: {PROCESSED_DIR}")
    print(f"Rows: all={len(df):,}, train={len(train_df):,}, test={len(test_df):,}")
    
    print(f"\nCore features ({len(CORE_FEATURES)}):")
    print(CORE_FEATURES)

    print(f"\nExtended features ({len(EXTENDED_FEATURES)}):")
    print(EXTENDED_FEATURES)

    print("Generated files:")
    for filename in sorted(outputs):
        print(f"- {filename}")
    print("- feature_config.json")
    print("- feature_scaler_stats.csv")
    print("- target_scaler_stats.csv")
    print("- preprocessing_summary.csv")


if __name__ == "__main__":
    main()
