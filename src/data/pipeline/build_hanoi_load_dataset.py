from src.config.paths import RAW_DATA_DIR
from src.data.utils.eda_utils import explore_dataframe
from src.data.analysis.eda_visualization import create_load_dataset_figures

import numpy as np
import pandas as pd
import sys
sys.stdout.reconfigure(encoding="utf-8")


WEATHER_FILE = RAW_DATA_DIR / "hanoi_weather_2021_2025.csv"
OUTPUT_FILE = RAW_DATA_DIR / "hanoi_load_dataset.csv"
RANDOM_SEED = 20260527


# Mỗi hồ sơ thể hiện một mô hình sử dụng điện thực tế.
# Mục tiêu: mô phỏng hành vi sử dụng điện, với thời tiết thực tế tại Hà Nội.
PROFILES = [
    {
        "profile_code": 0,
        "profile_name": "single_worker",
        "household_size": 1,
        "base_load_kw": 0.11,
        "appliance_scale": 0.75,
        "ac_sensitivity": 0.75,
        "home_workday": 0.10,
        "home_weekend": 0.68,
        "evening_activity": 0.85,
        "cooking_scale": 0.55,
        "spike_scale": 0.75,
    },
    {
        "profile_code": 1,
        "profile_name": "student",
        "household_size": 1,
        "base_load_kw": 0.10,
        "appliance_scale": 0.65,
        "ac_sensitivity": 0.60,
        "home_workday": 0.42,
        "home_weekend": 0.72,
        "evening_activity": 1.05,
        "cooking_scale": 0.40,
        "spike_scale": 0.65,
    },
    {
        "profile_code": 2,
        "profile_name": "family_3",
        "household_size": 3,
        "base_load_kw": 0.18,
        "appliance_scale": 1.05,
        "ac_sensitivity": 0.95,
        "home_workday": 0.22,
        "home_weekend": 0.78,
        "evening_activity": 0.95,
        "cooking_scale": 1.00,
        "spike_scale": 1.05,
    },
    {
        "profile_code": 3,
        "profile_name": "family_4_wfh",
        "household_size": 4,
        "base_load_kw": 0.24,
        "appliance_scale": 1.25,
        "ac_sensitivity": 1.10,
        "home_workday": 0.58,
        "home_weekend": 0.82,
        "evening_activity": 1.00,
        "cooking_scale": 1.15,
        "spike_scale": 1.20,
    },
]


HOLIDAY_RANGES = [
    ("2021-01-01", "2021-01-01"),
    ("2021-02-10", "2021-02-16"),
    ("2021-04-30", "2021-05-03"),
    ("2021-09-02", "2021-09-03"),
    ("2022-01-01", "2022-01-03"),
    ("2022-01-31", "2022-02-04"),
    ("2022-04-30", "2022-05-03"),
    ("2022-09-01", "2022-09-04"),
    ("2023-01-01", "2023-01-02"),
    ("2023-01-20", "2023-01-26"),
    ("2023-04-29", "2023-05-03"),
    ("2023-09-01", "2023-09-04"),
    ("2024-01-01", "2024-01-01"),
    ("2024-02-08", "2024-02-14"),
    ("2024-04-30", "2024-05-01"),
    ("2024-09-02", "2024-09-03"),
    ("2025-01-01", "2025-01-01"),
    ("2025-01-25", "2025-02-02"),
    ("2025-04-30", "2025-05-01"),
    ("2025-09-02", "2025-09-03"),
]


def read_weather() -> pd.DataFrame:
    if not WEATHER_FILE.exists():
        raise FileNotFoundError(
            f"Missing {WEATHER_FILE}. Run get_hanoi_weather.py first."
        )

    df = pd.read_csv(WEATHER_FILE, parse_dates=["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

    required_columns = {
        "datetime",
        "temperature",
        "humidity",
        "apparent_temperature",
        "rain",
        "wind_speed",
        "cloud_cover",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Weather file is missing columns: {sorted(missing_columns)}")

    weather_columns = [
        "temperature",
        "humidity",
        "apparent_temperature",
        "rain",
        "wind_speed",
        "cloud_cover",
    ]
    df[weather_columns] = df[weather_columns].interpolate(limit_direction="both")
    df["rain"] = df["rain"].clip(lower=0)
    df["humidity"] = df["humidity"].clip(lower=0, upper=100)
    df["cloud_cover"] = df["cloud_cover"].clip(lower=0, upper=100)
    return df


def mark_holidays(datetime_series: pd.Series) -> pd.Series:
    dates = datetime_series.dt.normalize()
    is_holiday = pd.Series(False, index=datetime_series.index)

    for start, end in HOLIDAY_RANGES:
        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end)
        is_holiday |= dates.between(start_date, end_date)

    return is_holiday.astype(int)


def add_time_and_habit_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    dt = result["datetime"]

    result["hour"] = dt.dt.hour
    result["day_of_week"] = dt.dt.dayofweek
    result["day_of_year"] = dt.dt.dayofyear
    result["month"] = dt.dt.month
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    result["holiday_effect"] = mark_holidays(dt)
    result["is_workday"] = (
        (result["is_weekend"] == 0) & (result["holiday_effect"] == 0)
    ).astype(int)

    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["day_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["day_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)

    # 0=cool/dry, 1=spring, 2=hot/humid, 3=autumn.
    result["season_code"] = np.select(
        [
            result["month"].isin([12, 1, 2]),
            result["month"].isin([3, 4]),
            result["month"].isin([5, 6, 7, 8, 9]),
        ],
        [0, 1, 2],
        default=3,
    )

    result["sleep_hour"] = ((result["hour"] <= 5) | (result["hour"] >= 23)).astype(int)
    result["morning_peak"] = result["hour"].between(6, 8).astype(int)
    result["evening_peak"] = result["hour"].between(18, 22).astype(int)
    result["working_hour"] = (
        result["hour"].between(8, 17) & (result["is_workday"] == 1)
    ).astype(int)
    result["breakfast_hour"] = result["hour"].between(6, 7).astype(int)
    result["lunch_hour"] = result["hour"].between(11, 12).astype(int)
    result["dinner_hour"] = result["hour"].between(18, 20).astype(int)
    result["meal_activity"] = (
        0.8 * result["breakfast_hour"]
        + 0.5 * result["lunch_hour"]
        + 1.2 * result["dinner_hour"]
        + 0.25 * result["holiday_effect"]
    )
    result["school_vacation"] = result["month"].isin([6, 7]).astype(int)

    # Usage period is not an electricity tariff. It is a daily behavior class:
    # 0=low-use night, 1=normal, 2=high-use morning/evening.
    result["usage_period"] = np.select(
        [
            result["hour"].between(0, 5),
            result["hour"].between(6, 8) | result["hour"].between(18, 22),
        ],
        [0, 2],
        default=1,
    )

    result["cooling_degree"] = result["apparent_temperature"].sub(27).clip(lower=0)
    result["heating_degree"] = (18 - result["apparent_temperature"]).clip(lower=0)
    result["humid_hot"] = (
        (result["temperature"] >= 30) & (result["humidity"] >= 75)
    ).astype(int)
    result["rainy_hour"] = (result["rain"] > 0).astype(int)
    result["heavy_rain"] = (result["rain"] >= 10).astype(int)
    result["dark_or_cloudy"] = (
        (result["hour"] >= 18)
        | (result["hour"] <= 5)
        | (result["cloud_cover"] >= 85)
    ).astype(int)
    result["temp_humidity_interaction"] = result["temperature"] * result["humidity"]
    return result


def estimate_occupancy(df: pd.DataFrame, profile: dict) -> np.ndarray:
    hour = df["hour"]
    weekend = df["is_weekend"].to_numpy(dtype=bool)
    holiday = df["holiday_effect"].to_numpy(dtype=bool)
    sleep = df["sleep_hour"].to_numpy(dtype=bool)
    morning = df["morning_peak"].to_numpy(dtype=bool)
    evening = df["evening_peak"].to_numpy(dtype=bool)
    lunch = df["lunch_hour"].to_numpy(dtype=bool)
    working = df["working_hour"].to_numpy(dtype=bool)
    rainy = df["rainy_hour"].to_numpy(dtype=bool)
    hot = (df["cooling_degree"] >= 6).to_numpy(dtype=bool)

    occupancy = np.full(len(df), 0.34)
    occupancy[working] = profile["home_workday"]
    occupancy[(weekend | holiday) & hour.between(8, 17).to_numpy()] = profile[
        "home_weekend"
    ]
    occupancy[lunch & ~weekend & ~holiday] = np.maximum(
        occupancy[lunch & ~weekend & ~holiday],
        0.26,
    )
    occupancy[morning] = np.maximum(occupancy[morning], 0.72)
    occupancy[evening] = np.maximum(occupancy[evening], 0.90)
    occupancy[sleep] = np.maximum(occupancy[sleep], 0.95)
    occupancy[holiday] = np.minimum(1.0, occupancy[holiday] + 0.10)
    occupancy[rainy | hot] = np.minimum(1.0, occupancy[rainy | hot] + 0.05)
    return occupancy


def create_profile_dataset(base_df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + profile["profile_code"])
    df = base_df.copy()

    household_size = profile["household_size"]
    occupancy_share = estimate_occupancy(df, profile)
    occupancy_level = occupancy_share * household_size
    n_rows = len(df)

    night_lighting_factor = np.where(
        (df["hour"] >= 18) | (df["hour"] <= 5),
        1.0,
        np.where(df["cloud_cover"] >= 85, 0.35, 0.0),
    )
    dinner_boost = np.where(df["dinner_hour"] == 1, 0.16, 0.0)
    weekend_laundry_window = (df["is_weekend"] == 1) & df["hour"].between(8, 16)
    holiday_laundry_window = (df["holiday_effect"] == 1) & df["hour"].between(9, 16)
    workday_laundry_window = (df["is_workday"] == 1) & df["hour"].between(19, 21)
    laundry_activity = (
        weekend_laundry_window.to_numpy() & (rng.random(n_rows) < 0.055)
    ) | (
        holiday_laundry_window.to_numpy() & (rng.random(n_rows) < 0.045)
    ) | (
        workday_laundry_window.to_numpy() & (rng.random(n_rows) < 0.010)
    )

    meal_window = (
        (df["breakfast_hour"] == 1)
        | (df["lunch_hour"] == 1)
        | (df["dinner_hour"] == 1)
    ).to_numpy()
    cooking_spike_probability = (
        0.010
        + 0.010 * df["is_weekend"].to_numpy()
        + 0.014 * df["holiday_effect"].to_numpy()
        + 0.004 * (household_size - 1)
    )
    cooking_spike = meal_window & (rng.random(n_rows) < cooking_spike_probability)
    cooking_spike_load = (
        cooking_spike.astype(float)
        * rng.uniform(0.35, 1.45, n_rows)
        * profile["spike_scale"]
    )

    appliance_window = (
        (df["hour"].between(7, 22))
        & ((df["is_weekend"] == 1) | (df["holiday_effect"] == 1) | (df["evening_peak"] == 1))
    ).to_numpy()
    appliance_spike_probability = 0.004 + 0.004 * household_size
    appliance_spike = appliance_window & (
        rng.random(n_rows) < appliance_spike_probability
    )
    appliance_spike_load = (
        appliance_spike.astype(float)
        * rng.uniform(0.45, 1.80, n_rows)
        * profile["spike_scale"]
    )

    ac_startup_spike = (
        (df["cooling_degree"].to_numpy() >= 5)
        & (df["evening_peak"].to_numpy() == 1)
        & (rng.random(n_rows) < 0.010)
    )
    ac_startup_spike_load = (
        ac_startup_spike.astype(float)
        * rng.uniform(0.35, 1.10, n_rows)
        * profile["ac_sensitivity"]
    )

    outage_anomaly = rng.random(n_rows) < 0.001

    standby = profile["base_load_kw"] * (1 + 0.06 * (household_size - 1))
    fridge = 0.045 + 0.010 * df["temperature"].sub(24).clip(lower=0)
    lighting = (
        0.12
        * night_lighting_factor
        * np.sqrt(household_size)
        * occupancy_share
    )
    cooking = (
        0.26
        * df["meal_activity"]
        * profile["cooking_scale"]
        * (0.65 + 0.18 * household_size)
    )
    entertainment = (
        0.15
        * profile["evening_activity"]
        * df["evening_peak"]
        * occupancy_share
        * np.sqrt(household_size)
    )
    work_study = (
        0.07
        * (df["working_hour"] + 0.7 * df["evening_peak"])
        * occupancy_share
        * household_size
    )
    cooling = (
        0.085
        * df["cooling_degree"]
        * profile["ac_sensitivity"]
        * occupancy_share
        * (0.85 + 0.17 * household_size)
    )
    heating = (
        0.060
        * df["heating_degree"]
        * occupancy_share
        * (0.80 + 0.15 * household_size)
    )
    fan_dehumidifier = (
        0.045
        * df["humid_hot"]
        * occupancy_share
        * (0.8 + 0.2 * household_size)
    )
    laundry = 0.55 * laundry_activity.astype(float) * profile["appliance_scale"]
    rain_adjustment = -0.035 * df["heavy_rain"] + 0.020 * df["rainy_hour"]
    holiday_adjustment = (
        0.055
        * df["holiday_effect"]
        * occupancy_share
        * (0.8 + 0.2 * household_size)
    )
    usage_period_adjustment = np.select(
        [df["usage_period"] == 0, df["usage_period"] == 2],
        [-0.025, 0.045],
        default=0.0,
    )
    personal_devices = (
        0.030
        * household_size
        * (0.5 * df["morning_peak"] + df["evening_peak"] + 0.25 * df["sleep_hour"])
    )

    load = (
        standby
        + fridge
        + lighting
        + cooking
        + entertainment
        + work_study
        + cooling
        + heating
        + fan_dehumidifier
        + laundry
        + rain_adjustment
        + holiday_adjustment
        + usage_period_adjustment
        + personal_devices
        + dinner_boost
        + cooking_spike_load
        + appliance_spike_load
        + ac_startup_spike_load
    )
    noise = rng.normal(0, 0.040 + 0.022 * np.sqrt(np.maximum(load, 0)), n_rows)
    load = np.clip(load + noise, 0.05, None)
    load[outage_anomaly] *= rng.uniform(0.05, 0.30, outage_anomaly.sum())
    load = np.clip(load, 0.02, None)

    df["profile_code"] = profile["profile_code"]
    df["profile_name"] = profile["profile_name"]
    df["household_size"] = household_size
    df["occupancy_level"] = np.round(occupancy_level, 3)
    df["occupancy_share"] = np.round(occupancy_share, 3)
    df["laundry_activity"] = laundry_activity.astype(int)
    df["cooking_spike"] = cooking_spike.astype(int)
    df["appliance_spike"] = appliance_spike.astype(int)
    df["ac_startup_spike"] = ac_startup_spike.astype(int)
    df["outage_anomaly"] = outage_anomaly.astype(int)
    df["load_kwh"] = np.round(load, 4)
    return df


def add_lag_features(dataset: pd.DataFrame) -> pd.DataFrame:
    result = dataset.sort_values(["profile_code", "datetime"]).copy()
    grouped_load = result.groupby("profile_code", sort=False)["load_kwh"]

    result["load_lag_1"] = grouped_load.shift(1)
    result["load_lag_24"] = grouped_load.shift(24)
    # Target cho dự báo
    result["target_1h"] = grouped_load.shift(-1)
    result["target_24h"] = grouped_load.shift(-24)
    result["load_rolling_3"] = grouped_load.transform(
        lambda load: load.shift(1).rolling(3).mean()
    )
    result["load_rolling_24"] = grouped_load.transform(
        lambda load: load.shift(1).rolling(24).mean()
    )

    lag_columns = [
        "load_lag_1",
        "load_lag_24",
        "load_rolling_3",
        "load_rolling_24",
        "target_1h",
        "target_24h",
    ]
    result = result.dropna(subset=lag_columns).reset_index(drop=True)
    result[lag_columns] = result[lag_columns].round(4)
    return result


def build_dataset() -> pd.DataFrame:
    weather_df = read_weather()
    base_df = add_time_and_habit_features(weather_df)
    dataset = pd.concat(
        [create_profile_dataset(base_df, profile) for profile in PROFILES],
        ignore_index=True,
    )
    dataset = add_lag_features(dataset)

    ordered_columns = [
        "datetime",
        "profile_code",
        "profile_name",
        "household_size",
        "temperature",
        "humidity",
        "apparent_temperature",
        "rain",
        "wind_speed",
        "cloud_cover",
        "hour",
        "day_of_week",
        "day_of_year",
        "month",
        "season_code",
        "is_weekend",
        "is_workday",
        "holiday_effect",
        "usage_period",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "month_sin",
        "month_cos",
        "sleep_hour",
        "morning_peak",
        "evening_peak",
        "working_hour",
        "breakfast_hour",
        "lunch_hour",
        "dinner_hour",
        "meal_activity",
        "school_vacation",
        "cooling_degree",
        "heating_degree",
        "humid_hot",
        "rainy_hour",
        "heavy_rain",
        "dark_or_cloudy",
        "temp_humidity_interaction",
        "occupancy_share",
        "occupancy_level",
        "laundry_activity",
        "cooking_spike",
        "appliance_spike",
        "ac_startup_spike",
        "outage_anomaly",
        "load_lag_1",
        "load_lag_24",
        "load_rolling_3",
        "load_rolling_24",
        "load_kwh",
        "target_1h",
        "target_24h",
    ]
    return dataset[ordered_columns]


def main() -> None:
    dataset = build_dataset()

    explore_dataframe(
        dataset,
        "DỮ LIỆU PHỤ TẢI ĐIỆN SAU MÔ PHỎNG",
        target_column="load_kwh",
    )
    create_load_dataset_figures(dataset)

    dataset.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"Saved file: {OUTPUT_FILE}")
    print(f"Rows: {len(dataset):,}")
    print(f"Profiles: {dataset['profile_name'].nunique()}")
    print(dataset[["profile_name", "load_kwh"]].groupby("profile_name").describe())
    print(
        dataset[
            [
                "cooking_spike",
                "appliance_spike",
                "ac_startup_spike",
                "outage_anomaly",
                "holiday_effect",
            ]
        ].mean()
    )
    print(dataset.head())


if __name__ == "__main__":
    main()
