from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_RAW_DIR = PROCESSED_DATA_DIR / "raw"
PROCESSED_SCALED_DIR = PROCESSED_DATA_DIR / "scaled"
PROCESSED_STATS_DIR = PROCESSED_DATA_DIR / "stats"
PROCESSED_RAW_CORE_DIR = PROCESSED_RAW_DIR / "core"
PROCESSED_RAW_EXTENDED_DIR = PROCESSED_RAW_DIR / "extended"
PROCESSED_SCALED_CORE_DIR = PROCESSED_SCALED_DIR / "core"
PROCESSED_SCALED_EXTENDED_DIR = PROCESSED_SCALED_DIR / "extended"

FIGURES_DIR = ROOT_DIR / "figures"
REPORTS_DIR = ROOT_DIR / "reports"
RESULTS_DIR = ROOT_DIR / "results"

RESULT_HORIZONS = ("1h", "24h")
RESULT_SUBDIRS = ("plots", "metrics", "models", "predictions")


def get_results_dir(horizon: str) -> Path:
    """Return the base results directory for a supported horizon."""
    horizon = str(horizon)
    if horizon not in RESULT_HORIZONS:
        allowed = ", ".join(RESULT_HORIZONS)
        raise ValueError(f"Unsupported horizon '{horizon}'. Expected one of: {allowed}.")
    return RESULTS_DIR / horizon


def get_results_subdir(horizon: str, subdir: str) -> Path:
    """Return a purpose-specific results directory within a horizon folder."""
    subdir = str(subdir)
    if subdir not in RESULT_SUBDIRS:
        allowed = ", ".join(RESULT_SUBDIRS)
        raise ValueError(f"Unsupported result subdir '{subdir}'. Expected one of: {allowed}.")
    return get_results_dir(horizon) / subdir


def get_results_subdirs(horizon: str) -> dict[str, Path]:
    """Return all purpose-specific results directories for a horizon."""
    return {subdir: get_results_subdir(horizon, subdir) for subdir in RESULT_SUBDIRS}


def create_all_paths() -> None:
    """Create all configured project directories if they do not already exist."""
    base_paths = (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        PROCESSED_RAW_DIR,
        PROCESSED_RAW_CORE_DIR,
        PROCESSED_RAW_EXTENDED_DIR,
        PROCESSED_SCALED_DIR,
        PROCESSED_SCALED_CORE_DIR,
        PROCESSED_SCALED_EXTENDED_DIR,
        PROCESSED_STATS_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
        RESULTS_DIR,
    )
    for path in base_paths:
        path.mkdir(parents=True, exist_ok=True)

    for horizon in RESULT_HORIZONS:
        get_results_dir(horizon).mkdir(parents=True, exist_ok=True)
        for path in get_results_subdirs(horizon).values():
            path.mkdir(parents=True, exist_ok=True)


create_all_paths()
