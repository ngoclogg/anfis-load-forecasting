from src.config.paths import RAW_DATA_DIR
from src.data.utils.eda_utils import explore_dataframe

import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import requests


OUTPUT_FILE = RAW_DATA_DIR / "hanoi_weather_2021_2025.csv"

LATITUDE = 21.0285
LONGITUDE = 105.8542
START_DATE = "2021-01-01"
END_DATE = "2025-12-31"


def main() -> None:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "rain",
            "wind_speed_10m",
            "cloud_cover",
        ],
        "timezone": "Asia/Bangkok",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "hourly" not in data:
        raise RuntimeError(f"API response has no hourly data: {data}")

    hourly = data["hourly"]
    df = pd.DataFrame(
        {
            "datetime": hourly["time"],
            "temperature": hourly["temperature_2m"],
            "humidity": hourly["relative_humidity_2m"],
            "apparent_temperature": hourly["apparent_temperature"],
            "rain": hourly["rain"],
            "wind_speed": hourly["wind_speed_10m"],
            "cloud_cover": hourly["cloud_cover"],
        }
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    explore_dataframe(df, "DỮ LIỆU THỜI TIẾT HÀ NỘI")
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"Saved file: {OUTPUT_FILE}")
    print(f"Rows: {len(df):,}")
    print(df.head())
    print(df.tail())
    print(df.isnull().sum())


if __name__ == "__main__":
    main()
