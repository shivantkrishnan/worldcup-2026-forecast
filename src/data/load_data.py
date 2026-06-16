"""Utilities for loading local historical match result datasets."""

from pathlib import Path

import pandas as pd

from src.utils.config import RAW_RESULTS_PATH


REQUIRED_RESULTS_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
}


def load_raw_results(path: str | Path = RAW_RESULTS_PATH) -> pd.DataFrame:
    """Load the manually downloaded historical international results CSV."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            "Historical results file not found. Manually download the "
            "international football results CSV and place it at "
            f"{file_path}. Raw data should not be committed."
        )

    if file_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file for historical results: {file_path}")

    results = pd.read_csv(file_path)
    missing = REQUIRED_RESULTS_COLUMNS.difference(results.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Missing required results columns: {missing_columns}")

    results["date"] = pd.to_datetime(results["date"], errors="raise")
    return results


def load_results(path: str | Path) -> pd.DataFrame:
    """Backward-compatible wrapper for loading historical raw results."""
    return load_raw_results(path)
