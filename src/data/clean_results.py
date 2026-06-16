"""Cleaning helpers for historical international football results."""

from __future__ import annotations

import pandas as pd

OUTCOME_TEAM_A_WIN = "team_a_win"
OUTCOME_DRAW = "draw"
OUTCOME_TEAM_B_WIN = "team_b_win"

REQUIRED_RESULT_COLUMNS = {
    "date",
    "team_a",
    "team_b",
    "team_a_score",
    "team_b_score",
}


def label_outcome(team_a_score: int | float, team_b_score: int | float) -> str:
    """Return the 3-class outcome label from Team A's perspective."""
    if team_a_score > team_b_score:
        return OUTCOME_TEAM_A_WIN
    if team_a_score < team_b_score:
        return OUTCOME_TEAM_B_WIN
    return OUTCOME_DRAW


def validate_required_columns(results: pd.DataFrame) -> None:
    """Raise a clear error when expected match result columns are missing."""
    missing = REQUIRED_RESULT_COLUMNS.difference(results.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_columns}")


def clean_results(results: pd.DataFrame) -> pd.DataFrame:
    """Clean raw match results and add the target outcome label."""
    validate_required_columns(results)

    cleaned = results.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["team_a"] = cleaned["team_a"].astype(str).str.strip()
    cleaned["team_b"] = cleaned["team_b"].astype(str).str.strip()
    cleaned["team_a_score"] = pd.to_numeric(cleaned["team_a_score"], errors="coerce")
    cleaned["team_b_score"] = pd.to_numeric(cleaned["team_b_score"], errors="coerce")

    cleaned = cleaned.dropna(
        subset=["date", "team_a", "team_b", "team_a_score", "team_b_score"]
    )

    cleaned["team_a_score"] = cleaned["team_a_score"].astype(int)
    cleaned["team_b_score"] = cleaned["team_b_score"].astype(int)
    cleaned["outcome"] = [
        label_outcome(team_a_score, team_b_score)
        for team_a_score, team_b_score in zip(
            cleaned["team_a_score"], cleaned["team_b_score"], strict=True
        )
    ]

    return cleaned.sort_values("date").reset_index(drop=True)
