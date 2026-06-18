"""Leakage-safe Elo-style team strength features."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from src.data.clean_results import (
    OUTCOME_DRAW,
    OUTCOME_TEAM_A_WIN,
    OUTCOME_TEAM_B_WIN,
)

ELO_FEATURE_COLUMNS = [
    "elo_team_a_pre",
    "elo_team_b_pre",
    "elo_diff_team_a_minus_team_b",
    "elo_effective_diff_team_a_minus_team_b",
    "elo_expected_score_team_a",
    "elo_home_advantage_applied",
    "elo_matches_before_team_a",
    "elo_matches_before_team_b",
]


def expected_score(rating_a: float, rating_b: float) -> float:
    """Return standard Elo expected score for team A."""
    return float(1 / (1 + 10 ** ((rating_b - rating_a) / 400)))


def result_to_score(result: str) -> float:
    """Map a team_a-perspective result label to an Elo score for team_a."""
    if result == OUTCOME_TEAM_A_WIN:
        return 1.0
    if result == OUTCOME_DRAW:
        return 0.5
    if result == OUTCOME_TEAM_B_WIN:
        return 0.0
    raise ValueError(f"Unexpected result label: {result}")


def update_elo_pair(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k_factor: float = 20.0,
    expected_score_a: float | None = None,
) -> tuple[float, float]:
    """Return updated Elo ratings for a completed match."""
    expected_a = (
        expected_score_a
        if expected_score_a is not None
        else expected_score(rating_a, rating_b)
    )
    expected_b = 1 - expected_a
    score_b = 1 - score_a

    updated_a = rating_a + k_factor * (score_a - expected_a)
    updated_b = rating_b + k_factor * (score_b - expected_b)
    return (float(updated_a), float(updated_b))


def _coerce_bool(value: object) -> bool:
    """Coerce common boolean-like values without treating missing as neutral."""
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


def _is_neutral(row: object) -> bool:
    """Return whether a match row is neutral-site using canonical columns."""
    if hasattr(row, "is_neutral"):
        return _coerce_bool(getattr(row, "is_neutral"))
    if hasattr(row, "neutral"):
        return _coerce_bool(getattr(row, "neutral"))
    return False


def _home_advantage_applied(row: object, home_advantage: float) -> float:
    """Return the numeric home bonus applied to team_a for expected score."""
    if _is_neutral(row):
        return 0.0
    return float(home_advantage)


def build_elo_features(
    canonical_matches: pd.DataFrame,
    initial_rating: float = 1500.0,
    k_factor: float = 20.0,
    home_advantage: float = 0.0,
    date_col: str = "match_date",
) -> pd.DataFrame:
    """Return pre-match Elo features, updating ratings after each date block."""
    matches = canonical_matches.copy(deep=True)
    matches[date_col] = pd.to_datetime(matches[date_col], errors="raise")
    matches = matches.sort_values([date_col, "match_id"], kind="mergesort").reset_index(
        drop=True
    )

    ratings: dict[str, float] = {}
    match_counts: dict[str, int] = {}
    feature_rows: list[dict[str, object]] = []

    for _, date_matches in matches.groupby(date_col, sort=True):
        start_ratings = ratings.copy()
        start_counts = match_counts.copy()

        for row in date_matches.itertuples(index=False):
            team_a = str(getattr(row, "team_a"))
            team_b = str(getattr(row, "team_b"))
            rating_a = start_ratings.get(team_a, initial_rating)
            rating_b = start_ratings.get(team_b, initial_rating)
            applied_home_advantage = _home_advantage_applied(row, home_advantage)
            effective_rating_a = rating_a + applied_home_advantage
            expected_a = expected_score(effective_rating_a, rating_b)

            feature_rows.append(
                {
                    "match_id": getattr(row, "match_id"),
                    "match_date": getattr(row, date_col),
                    "team_a": team_a,
                    "team_b": team_b,
                    "elo_team_a_pre": float(rating_a),
                    "elo_team_b_pre": float(rating_b),
                    "elo_diff_team_a_minus_team_b": float(rating_a - rating_b),
                    "elo_effective_diff_team_a_minus_team_b": float(
                        effective_rating_a - rating_b
                    ),
                    "elo_expected_score_team_a": expected_a,
                    "elo_home_advantage_applied": applied_home_advantage,
                    "elo_matches_before_team_a": int(start_counts.get(team_a, 0)),
                    "elo_matches_before_team_b": int(start_counts.get(team_b, 0)),
                }
            )

        rating_deltas: defaultdict[str, float] = defaultdict(float)
        count_deltas: defaultdict[str, int] = defaultdict(int)
        for row in date_matches.itertuples(index=False):
            team_a = str(getattr(row, "team_a"))
            team_b = str(getattr(row, "team_b"))
            rating_a = start_ratings.get(team_a, initial_rating)
            rating_b = start_ratings.get(team_b, initial_rating)
            score_a = result_to_score(str(getattr(row, "result")))
            applied_home_advantage = _home_advantage_applied(row, home_advantage)
            expected_a = expected_score(rating_a + applied_home_advantage, rating_b)

            updated_a, updated_b = update_elo_pair(
                rating_a,
                rating_b,
                score_a,
                k_factor=k_factor,
                expected_score_a=expected_a,
            )
            rating_deltas[team_a] += updated_a - rating_a
            rating_deltas[team_b] += updated_b - rating_b
            count_deltas[team_a] += 1
            count_deltas[team_b] += 1

        for team, delta in rating_deltas.items():
            ratings[team] = start_ratings.get(team, initial_rating) + delta
        for team, count_delta in count_deltas.items():
            match_counts[team] = start_counts.get(team, 0) + count_delta

    return pd.DataFrame(
        feature_rows,
        columns=[
            "match_id",
            "match_date",
            "team_a",
            "team_b",
            *ELO_FEATURE_COLUMNS,
        ],
    )


def add_elo_features_to_matches(
    canonical_matches: pd.DataFrame,
    initial_rating: float = 1500.0,
    k_factor: float = 20.0,
    home_advantage: float = 0.0,
) -> pd.DataFrame:
    """Return canonical matches with leakage-safe pre-match Elo features added."""
    matches = canonical_matches.copy(deep=True)
    elo_features = build_elo_features(
        matches,
        initial_rating=initial_rating,
        k_factor=k_factor,
        home_advantage=home_advantage,
    )
    return matches.merge(
        elo_features[["match_id", *ELO_FEATURE_COLUMNS]],
        on="match_id",
        how="left",
        validate="one_to_one",
    )
