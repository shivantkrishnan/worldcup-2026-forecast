"""Small in-memory data pipeline helpers."""

from pathlib import Path

import pandas as pd

from src.data.clean_results import clean_results, filter_baseline_training_matches
from src.data.duplicate_resolution import (
    DuplicateResolutionResult,
    resolve_duplicate_matches,
)
from src.data.load_data import load_raw_results
from src.utils.config import DEFAULT_TRAINING_CUTOFF_DATE, RAW_RESULTS_PATH


def load_and_clean_results(
    raw_path: str | Path = RAW_RESULTS_PATH,
    training_cutoff_date: str = DEFAULT_TRAINING_CUTOFF_DATE,
    resolve_duplicates: bool = True,
) -> pd.DataFrame:
    """Load raw historical results and return canonical cleaned matches."""
    raw_results = load_raw_results(raw_path)
    cleaned = clean_results(raw_results, training_cutoff_date=training_cutoff_date)
    if not resolve_duplicates:
        return cleaned

    return resolve_duplicate_matches(cleaned).resolved_matches


def load_and_clean_results_with_quarantine(
    raw_path: str | Path = RAW_RESULTS_PATH,
    training_cutoff_date: str = DEFAULT_TRAINING_CUTOFF_DATE,
) -> DuplicateResolutionResult:
    """Load, clean, resolve duplicates, and return quarantine details."""
    raw_results = load_raw_results(raw_path)
    cleaned = clean_results(raw_results, training_cutoff_date=training_cutoff_date)
    return resolve_duplicate_matches(cleaned)


def load_baseline_training_matches(
    raw_path: str | Path = RAW_RESULTS_PATH,
    training_cutoff_date: str = DEFAULT_TRAINING_CUTOFF_DATE,
) -> pd.DataFrame:
    """Load canonical matches and return rows eligible for baseline training."""
    cleaned = load_and_clean_results(
        raw_path=raw_path,
        training_cutoff_date=training_cutoff_date,
        resolve_duplicates=True,
    )
    return filter_baseline_training_matches(
        cleaned,
        training_cutoff_date=training_cutoff_date,
    )
