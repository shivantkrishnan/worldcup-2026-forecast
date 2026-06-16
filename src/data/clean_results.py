"""Cleaning helpers for historical international football results."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from numbers import Number
from typing import Any

import pandas as pd

from src.utils.config import DEFAULT_TRAINING_CUTOFF_DATE

OUTCOME_TEAM_A_WIN = "team_a_win"
OUTCOME_DRAW = "draw"
OUTCOME_TEAM_B_WIN = "team_b_win"

REQUIRED_RAW_RESULT_COLUMNS = {
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

CANONICAL_MATCH_COLUMNS = [
    "match_id",
    "match_date",
    "team_a",
    "team_b",
    "team_a_goals",
    "team_b_goals",
    "result",
    "tournament",
    "city",
    "country",
    "neutral",
    "is_neutral",
    "goal_diff_team_a",
    "total_goals",
    "training_cutoff_date",
    "is_baseline_train_eligible",
]

TRUE_VALUES = {"1", "true", "t", "yes", "y"}
FALSE_VALUES = {"0", "false", "f", "no", "n"}


def label_outcome(team_a_score: int | float, team_b_score: int | float) -> str:
    """Return the 3-class outcome label from Team A's perspective."""
    if team_a_score > team_b_score:
        return OUTCOME_TEAM_A_WIN
    if team_a_score < team_b_score:
        return OUTCOME_TEAM_B_WIN
    return OUTCOME_DRAW


def validate_required_columns(results: pd.DataFrame) -> None:
    """Raise a clear error when expected match result columns are missing."""
    missing = REQUIRED_RAW_RESULT_COLUMNS.difference(results.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Missing required raw result columns: {missing_columns}")


def _slugify(value: Any) -> str:
    """Return a filesystem- and CSV-friendly token for deterministic IDs."""
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_value.lower()).strip("_")
    return slug or "unknown"


def _coerce_bool(value: Any) -> bool:
    """Coerce common raw CSV boolean values without treating strings as truthy."""
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        raise ValueError("neutral contains missing values")

    if isinstance(value, Number) and value in {0, 1}:
        return bool(value)

    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False

    raise ValueError(f"Unable to parse neutral value as boolean: {value!r}")


def create_match_id(row: pd.Series) -> str:
    """Create a deterministic match ID from date, teams, and tournament."""
    match_date = pd.Timestamp(row["match_date"]).date().isoformat()
    team_a = str(row["team_a"]).strip()
    team_b = str(row["team_b"]).strip()
    tournament = str(row["tournament"]).strip()

    key = "|".join([match_date, team_a.lower(), team_b.lower(), tournament.lower()])
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    readable = "_".join(
        [_slugify(match_date), _slugify(team_a), _slugify(team_b), _slugify(tournament)]
    )
    return f"{readable}_{digest}"


def clean_results(
    results: pd.DataFrame,
    training_cutoff_date: str = DEFAULT_TRAINING_CUTOFF_DATE,
) -> pd.DataFrame:
    """Convert raw historical results into the canonical match schema."""
    validate_required_columns(results)

    cleaned = results.copy(deep=True)
    cutoff = pd.Timestamp(training_cutoff_date)

    cleaned["match_date"] = pd.to_datetime(cleaned["date"], errors="raise")
    cleaned["team_a"] = cleaned["home_team"].astype(str).str.strip()
    cleaned["team_b"] = cleaned["away_team"].astype(str).str.strip()
    cleaned["tournament"] = cleaned["tournament"].astype(str).str.strip()
    cleaned["city"] = cleaned["city"].astype(str).str.strip()
    cleaned["country"] = cleaned["country"].astype(str).str.strip()
    cleaned["team_a_goals"] = pd.to_numeric(cleaned["home_score"], errors="raise")
    cleaned["team_b_goals"] = pd.to_numeric(cleaned["away_score"], errors="raise")
    cleaned = cleaned.dropna(
        subset=[
            "match_date",
            "team_a",
            "team_b",
            "team_a_goals",
            "team_b_goals",
            "tournament",
        ]
    )
    cleaned["team_a_goals"] = cleaned["team_a_goals"].astype(int)
    cleaned["team_b_goals"] = cleaned["team_b_goals"].astype(int)
    cleaned["neutral"] = cleaned["neutral"].map(_coerce_bool).astype(bool)
    cleaned["is_neutral"] = cleaned["neutral"]
    cleaned["goal_diff_team_a"] = cleaned["team_a_goals"] - cleaned["team_b_goals"]
    cleaned["total_goals"] = cleaned["team_a_goals"] + cleaned["team_b_goals"]
    cleaned["result"] = [
        label_outcome(team_a_goals, team_b_goals)
        for team_a_goals, team_b_goals in zip(
            cleaned["team_a_goals"], cleaned["team_b_goals"], strict=True
        )
    ]
    cleaned["match_id"] = cleaned.apply(create_match_id, axis=1)
    cleaned["training_cutoff_date"] = training_cutoff_date
    cleaned["is_baseline_train_eligible"] = cleaned["match_date"] <= cutoff

    return cleaned.loc[:, CANONICAL_MATCH_COLUMNS]


def filter_baseline_training_matches(
    results: pd.DataFrame,
    training_cutoff_date: str = DEFAULT_TRAINING_CUTOFF_DATE,
) -> pd.DataFrame:
    """Return matches on or before the baseline training cutoff date."""
    filtered = results.copy(deep=True)
    filtered["match_date"] = pd.to_datetime(filtered["match_date"], errors="raise")
    cutoff = pd.Timestamp(training_cutoff_date)
    return filtered.loc[filtered["match_date"] <= cutoff].reset_index(drop=True)
