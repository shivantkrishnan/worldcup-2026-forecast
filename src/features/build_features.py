"""Feature-building utilities.

Rolling features must only use matches before the prediction date. This module
provides the main feature-building entry point for future modeling.
"""

import pandas as pd

from src.features.team_form import build_match_level_features


def build_match_features(
    results: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """Return leakage-safe match-level team-form features."""
    return build_match_level_features(results, windows=windows)
