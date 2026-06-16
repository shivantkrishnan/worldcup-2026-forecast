"""Leakage-safe team-form feature engineering."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.data.clean_results import (
    OUTCOME_DRAW,
    OUTCOME_TEAM_A_WIN,
    OUTCOME_TEAM_B_WIN,
)

TEAM_FORM_FEATURE_COLUMNS = [
    "matches_played_before",
    "days_since_last_match",
    "expanding_points_per_match",
    "expanding_goal_diff_avg",
    "expanding_win_rate",
]

ROLLING_BASE_COLUMNS = [
    "points",
    "goals_for",
    "goals_against",
    "goal_diff",
    "win",
    "draw",
    "loss",
]

ROLLING_FEATURE_NAME_MAP = {
    "points": "rolling_points_per_match",
    "goals_for": "rolling_goals_for_avg",
    "goals_against": "rolling_goals_against_avg",
    "goal_diff": "rolling_goal_diff_avg",
    "win": "rolling_win_rate",
    "draw": "rolling_draw_rate",
    "loss": "rolling_loss_rate",
}


def _result_from_team_a_perspective(result: str) -> str:
    if result == OUTCOME_TEAM_A_WIN:
        return "win"
    if result == OUTCOME_TEAM_B_WIN:
        return "loss"
    if result == OUTCOME_DRAW:
        return "draw"
    raise ValueError(f"Unexpected result label: {result}")


def _result_from_team_b_perspective(result: str) -> str:
    if result == OUTCOME_TEAM_A_WIN:
        return "loss"
    if result == OUTCOME_TEAM_B_WIN:
        return "win"
    if result == OUTCOME_DRAW:
        return "draw"
    raise ValueError(f"Unexpected result label: {result}")


def _points(result_from_team_perspective: str) -> int:
    if result_from_team_perspective == "win":
        return 3
    if result_from_team_perspective == "draw":
        return 1
    if result_from_team_perspective == "loss":
        return 0
    raise ValueError(f"Unexpected team result: {result_from_team_perspective}")


def _normalize_windows(windows: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(dict.fromkeys(int(window) for window in windows))
    if not normalized:
        raise ValueError("At least one rolling window is required.")
    if any(window <= 0 for window in normalized):
        raise ValueError("Rolling windows must be positive integers.")
    return normalized


def _rolling_feature_columns(windows: Sequence[int]) -> list[str]:
    columns: list[str] = []
    for window in _normalize_windows(windows):
        for feature_prefix in ROLLING_FEATURE_NAME_MAP.values():
            columns.append(f"{feature_prefix}_{window}")
    return columns


def get_team_form_feature_columns(windows: Sequence[int] = (5, 10)) -> list[str]:
    """Return team-form feature column names for the requested windows."""
    return [*TEAM_FORM_FEATURE_COLUMNS, *_rolling_feature_columns(windows)]


def build_team_match_panel(canonical_matches: pd.DataFrame) -> pd.DataFrame:
    """Return a long panel with one row per team per completed match."""
    matches = canonical_matches.copy(deep=True)
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="raise")

    team_a = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "match_date": matches["match_date"],
            "team": matches["team_a"],
            "opponent": matches["team_b"],
            "goals_for": matches["team_a_goals"],
            "goals_against": matches["team_b_goals"],
            "is_team_a": True,
            "is_home": ~matches["is_neutral"].astype(bool),
            "is_neutral": matches["is_neutral"].astype(bool),
            "tournament": matches["tournament"],
            "result_from_team_perspective": matches["result"].map(
                _result_from_team_a_perspective
            ),
        }
    )

    team_b = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "match_date": matches["match_date"],
            "team": matches["team_b"],
            "opponent": matches["team_a"],
            "goals_for": matches["team_b_goals"],
            "goals_against": matches["team_a_goals"],
            "is_team_a": False,
            "is_home": False,
            "is_neutral": matches["is_neutral"].astype(bool),
            "tournament": matches["tournament"],
            "result_from_team_perspective": matches["result"].map(
                _result_from_team_b_perspective
            ),
        }
    )

    panel = pd.concat([team_a, team_b], ignore_index=True)
    panel["goal_diff"] = panel["goals_for"] - panel["goals_against"]
    panel["points"] = panel["result_from_team_perspective"].map(_points).astype(int)
    panel["win"] = (panel["result_from_team_perspective"] == "win").astype(int)
    panel["draw"] = (panel["result_from_team_perspective"] == "draw").astype(int)
    panel["loss"] = (panel["result_from_team_perspective"] == "loss").astype(int)

    return panel[
        [
            "match_id",
            "match_date",
            "team",
            "opponent",
            "goals_for",
            "goals_against",
            "goal_diff",
            "points",
            "win",
            "draw",
            "loss",
            "is_team_a",
            "is_home",
            "is_neutral",
            "tournament",
            "result_from_team_perspective",
        ]
    ]


def add_team_rolling_features(
    team_panel: pd.DataFrame,
    windows: Sequence[int] = (5, 10),
) -> pd.DataFrame:
    """Add leakage-safe rolling and expanding team-form features."""
    normalized_windows = _normalize_windows(windows)
    panel = team_panel.copy(deep=True)
    panel["match_date"] = pd.to_datetime(panel["match_date"], errors="raise")
    panel = panel.sort_values(
        ["team", "match_date", "match_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    def add_group_features(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values(["match_date", "match_id"], kind="mergesort").copy()
        group["matches_played_before"] = range(len(group))
        group["days_since_last_match"] = group["match_date"].diff().dt.days

        shifted = {column: group[column].shift(1) for column in ROLLING_BASE_COLUMNS}

        for window in normalized_windows:
            for base_column, feature_prefix in ROLLING_FEATURE_NAME_MAP.items():
                group[f"{feature_prefix}_{window}"] = shifted[base_column].rolling(
                    window=window,
                    min_periods=window,
                ).mean()

        group["expanding_points_per_match"] = shifted["points"].expanding(
            min_periods=1
        ).mean()
        group["expanding_goal_diff_avg"] = shifted["goal_diff"].expanding(
            min_periods=1
        ).mean()
        group["expanding_win_rate"] = shifted["win"].expanding(min_periods=1).mean()
        return group

    return (
        panel.groupby("team", group_keys=False)
        .apply(add_group_features)
        .sort_values(["team", "match_date", "match_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def build_match_level_features(
    canonical_matches: pd.DataFrame,
    windows: Sequence[int] = (5, 10),
) -> pd.DataFrame:
    """Return one row per match with team-form features and differences."""
    matches = canonical_matches.copy(deep=True)
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="raise")

    panel = build_team_match_panel(matches)
    panel_with_features = add_team_rolling_features(panel, windows=windows)
    feature_columns = get_team_form_feature_columns(windows)

    team_a_features = (
        panel_with_features.loc[panel_with_features["is_team_a"], ["match_id", *feature_columns]]
        .copy()
        .add_prefix("team_a_")
        .rename(columns={"team_a_match_id": "match_id"})
    )
    team_b_features = (
        panel_with_features.loc[
            ~panel_with_features["is_team_a"], ["match_id", *feature_columns]
        ]
        .copy()
        .add_prefix("team_b_")
        .rename(columns={"team_b_match_id": "match_id"})
    )

    match_features = matches[
        ["match_id", "match_date", "team_a", "team_b", "result", "tournament", "is_neutral"]
    ].copy()
    match_features = match_features.merge(team_a_features, on="match_id", how="left")
    match_features = match_features.merge(team_b_features, on="match_id", how="left")

    for feature_column in feature_columns:
        match_features[f"{feature_column}_diff"] = (
            match_features[f"team_a_{feature_column}"]
            - match_features[f"team_b_{feature_column}"]
        )

    return match_features
