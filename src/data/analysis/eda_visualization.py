from src.config.paths import FIGURES_DIR

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_DIR = FIGURES_DIR

# LOAD DISTRIBUTION HISTOGRAM
def plot_load_distribution(df, target_column="load_kwh"):
    if target_column not in df.columns:
        return

    data = df[target_column].dropna()

    plt.figure(figsize=(8, 5))

    counts, bins, patches = plt.hist(data, bins=50)

    colors = plt.cm.viridis(np.linspace(0, 1, len(patches)))

    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)

    plt.title("Load Distribution")
    plt.xlabel(target_column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "load_distribution.png")
    plt.close()

# LOAD OVER TIME
def plot_load_over_time(df, target_column="load_kwh"):
    if "datetime" not in df.columns or target_column not in df.columns:
        return

    sample_df = df.sort_values("datetime").head(24 * 14)

    plt.figure(figsize=(12, 5))
    plt.plot(sample_df["datetime"], sample_df[target_column])
    plt.title("Load Over Time")
    plt.xlabel("Datetime")
    plt.ylabel(target_column)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "load_over_time_sample.png")
    plt.close()

# AVERAGE LOAD BY HOUR
def plot_load_by_hour(df, target_column="load_kwh"):
    if "hour" not in df.columns or target_column not in df.columns:
        return

    hourly_mean = df.groupby("hour")[target_column].mean()

    plt.figure(figsize=(8, 5))
    plt.plot(hourly_mean.index, hourly_mean.values, marker="o")
    plt.title("Average Load by Hour")
    plt.xlabel("Hour")
    plt.ylabel(f"Average {target_column}")
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "average_load_by_hour.png")
    plt.close()

# AVERAGE LOAD BY PROFILE
def plot_load_by_profile(df, target_column="load_kwh"):
    if "profile_name" not in df.columns or target_column not in df.columns:
        return

    profile_mean = df.groupby("profile_name")[target_column].mean()

    plt.figure(figsize=(8, 5))
    plt.bar(profile_mean.index, profile_mean.values)
    plt.title("Average Load by Profile")
    plt.xlabel("Profile")
    plt.ylabel(f"Average {target_column}")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "average_load_by_profile.png")
    plt.close()

# FEATURE CORRELATION HEATMAP
def plot_correlation_heatmap(df, columns):
    if not all(col in df.columns for col in columns):
        return

    corr = df[columns].corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_heatmap.png")
    plt.close()

# GENERATE ALL FIGURES
def create_load_dataset_figures(df):
    plot_load_distribution(df)
    plot_load_over_time(df)
    plot_load_by_hour(df)
    plot_load_by_profile(df)

    corr_columns = [
        "temperature",
        "humidity",
        "apparent_temperature",
        "rain",
        "wind_speed",
        "cloud_cover",
        "occupancy_level",
        "load_lag_1",
        "load_lag_24",
        "load_kwh",
    ]

    existing_columns = [col for col in corr_columns if col in df.columns]
    plot_correlation_heatmap(df, existing_columns)

    print(f"\nFigures saved to: {FIGURE_DIR}")