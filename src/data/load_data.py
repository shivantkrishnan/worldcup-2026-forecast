"""Utilities for loading local match result datasets."""

from pathlib import Path

import pandas as pd


def load_results(path: str | Path) -> pd.DataFrame:
    """Load match results from a local CSV or Parquet file."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")

    if file_path.suffix == ".csv":
        return pd.read_csv(file_path)

    if file_path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)

    raise ValueError(f"Unsupported file type: {file_path.suffix}")
