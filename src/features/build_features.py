"""Feature-building utilities.

Rolling features must only use matches before the prediction date. This module
provides the main feature-building entry point for future modeling.
"""

import pandas as pd

from src.features.elo import ELO_FEATURE_COLUMNS, build_elo_features
from src.features.team_form import build_match_level_features


def build_match_features(
    results: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10),
    include_elo: bool = False,
    initial_elo_rating: float = 1500.0,
    elo_k_factor: float = 20.0,
) -> pd.DataFrame:
    """Return leakage-safe match-level features for modeling."""
    return build_modeling_features(
        results,
        windows=windows,
        include_elo=include_elo,
        initial_elo_rating=initial_elo_rating,
        elo_k_factor=elo_k_factor,
    )


def build_modeling_features(
    results: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10),
    include_elo: bool = False,
    initial_elo_rating: float = 1500.0,
    elo_k_factor: float = 20.0,
) -> pd.DataFrame:
    """Return leakage-safe model features, optionally including Elo features."""
    features = build_match_level_features(results, windows=windows)
    if not include_elo:
        return features

    elo_features = build_elo_features(
        results,
        initial_rating=initial_elo_rating,
        k_factor=elo_k_factor,
    )
    return features.merge(
        elo_features[["match_id", *ELO_FEATURE_COLUMNS]],
        on="match_id",
        how="left",
        validate="one_to_one",
    )
