"""Leakage-safe feature rows for scheduled fixtures."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.clean_results import OUTCOME_DRAW
from src.features.build_features import build_modeling_features

REQUIRED_FIXTURE_COLUMNS = {"match_id", "match_date", "team_a", "team_b"}


def _coerce_bool(value: Any) -> bool | None:
    """Coerce common boolean-like values, preserving missing as unknown."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    return bool(value)


def _is_2026_world_cup_fixture(row: pd.Series) -> bool:
    """Return whether a fixture appears to be a 2026 World Cup match."""
    match_year = pd.Timestamp(row["match_date"]).year
    tournament = str(row.get("tournament", "")).strip().casefold()
    return match_year == 2026 and "world cup" in tournament


def _fixture_neutral_flag(row: pd.Series) -> bool:
    """Return the neutral-site flag used for fixture feature generation."""
    is_neutral = _coerce_bool(row.get("is_neutral"))
    if is_neutral is not None:
        return is_neutral

    neutral = _coerce_bool(row.get("neutral"))
    if neutral is not None:
        return neutral

    if _is_2026_world_cup_fixture(row):
        # Generic historical home advantage is not a host-country model. USA,
        # Canada, and Mexico effects should be explicit tournament-state features later.
        return True

    return False


def _validate_fixture_columns(fixtures: pd.DataFrame) -> None:
    """Raise a clear error when required fixture columns are missing."""
    missing = REQUIRED_FIXTURE_COLUMNS.difference(fixtures.columns)
    if missing:
        raise ValueError(
            "Missing required fixture columns: " + ", ".join(sorted(missing))
        )


def _placeholder_match_from_fixture(fixture: pd.Series) -> dict[str, object]:
    """Return a canonical placeholder row whose outcome is never used as a feature."""
    tournament = str(fixture.get("tournament", "Scheduled Fixture")).strip()
    if not tournament or tournament == "nan":
        tournament = "Scheduled Fixture"

    return {
        "match_id": fixture["match_id"],
        "match_date": fixture["match_date"],
        "team_a": str(fixture["team_a"]).strip(),
        "team_b": str(fixture["team_b"]).strip(),
        "team_a_goals": 0,
        "team_b_goals": 0,
        "result": OUTCOME_DRAW,
        "tournament": tournament,
        "is_neutral": _fixture_neutral_flag(fixture),
    }


def _completed_history_for_fixture(
    completed_matches: pd.DataFrame,
    fixture_date: pd.Timestamp,
    feature_cutoff_date: str | None,
) -> pd.DataFrame:
    """Return completed rows available before one fixture date."""
    history = completed_matches.copy(deep=True)
    history["match_date"] = pd.to_datetime(history["match_date"], errors="raise")

    mask = history["match_date"] < fixture_date
    if feature_cutoff_date is not None:
        cutoff = pd.Timestamp(feature_cutoff_date)
        mask &= history["match_date"] <= cutoff
    return history.loc[mask].copy()


def build_fixture_feature_rows(
    completed_matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    include_elo: bool = True,
    elo_k_factor: float = 10.0,
    elo_home_advantage: float = 50.0,
    feature_cutoff_date: str | None = None,
) -> pd.DataFrame:
    """Build selected-baseline feature rows for scheduled fixtures.

    Fixtures do not need scores or result labels. For each fixture, completed
    match history is filtered to rows strictly before the fixture date. With
    date-only timestamps, same-date completed matches are excluded because their
    kickoff order is unknown.
    """
    _validate_fixture_columns(fixtures)

    completed = completed_matches.copy(deep=True)
    fixture_rows = fixtures.copy(deep=True)
    fixture_rows["match_date"] = pd.to_datetime(
        fixture_rows["match_date"],
        errors="raise",
    )

    output_rows: list[pd.DataFrame] = []
    for _, fixture in fixture_rows.sort_values(
        ["match_date", "match_id"],
        kind="mergesort",
    ).iterrows():
        fixture_date = pd.Timestamp(fixture["match_date"])
        history = _completed_history_for_fixture(
            completed,
            fixture_date=fixture_date,
            feature_cutoff_date=feature_cutoff_date,
        )
        placeholder = pd.DataFrame([_placeholder_match_from_fixture(fixture)])
        modeling_input = pd.concat([history, placeholder], ignore_index=True)

        features = build_modeling_features(
            modeling_input,
            include_elo=include_elo,
            elo_k_factor=elo_k_factor,
            elo_home_advantage=elo_home_advantage,
        )
        fixture_features = features.loc[
            features["match_id"].eq(fixture["match_id"])
        ].copy()
        fixture_features = fixture_features.drop(columns=["result"], errors="ignore")
        output_rows.append(fixture_features)

    if not output_rows:
        return pd.DataFrame()

    return pd.concat(output_rows, ignore_index=True)
