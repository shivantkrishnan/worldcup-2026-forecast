"""Load, normalize, and validate manually maintained tournament fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import FIXTURES_2026_PATH

REQUIRED_TOURNAMENT_FIXTURE_COLUMNS = {
    "match_id",
    "match_date",
    "team_a",
    "team_b",
    "group",
    "stage",
}
OPTIONAL_TOURNAMENT_FIXTURE_COLUMNS = {
    "kickoff_time",
    "venue",
    "city",
    "country",
    "neutral",
    "is_neutral",
    "source",
    "last_updated",
}
VALID_STAGE_VALUES = {
    "group",
    "round_of_32",
    "round_of_16",
    "quarterfinal",
    "semifinal",
    "third_place",
    "final",
}


def _coerce_bool_or_missing(value: Any) -> bool | None:
    """Coerce common boolean values while preserving missing values."""
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
    raise ValueError(f"Unable to parse neutral value as boolean: {value!r}")


def _fixture_neutral_value(row: pd.Series) -> bool:
    """Return normalized neutral flag for a 2026 tournament fixture row."""
    is_neutral = _coerce_bool_or_missing(row.get("is_neutral"))
    if is_neutral is not None:
        return is_neutral

    neutral = _coerce_bool_or_missing(row.get("neutral"))
    if neutral is not None:
        return neutral

    if pd.Timestamp(row["match_date"]).year == 2026:
        return True
    return False


def normalize_tournament_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Return normalized tournament fixtures without mutating input."""
    normalized = fixtures.copy(deep=True)
    missing = REQUIRED_TOURNAMENT_FIXTURE_COLUMNS.difference(normalized.columns)
    if missing:
        raise ValueError(
            "Missing required tournament fixture columns: "
            + ", ".join(sorted(missing))
        )

    normalized["match_id"] = normalized["match_id"].astype(str).str.strip()
    normalized["match_date"] = pd.to_datetime(
        normalized["match_date"],
        errors="raise",
    )
    for column in ["team_a", "team_b", "group", "stage"]:
        normalized[column] = normalized[column].astype("string").str.strip()

    normalized["stage"] = normalized["stage"].str.casefold()
    normalized["is_neutral"] = normalized.apply(_fixture_neutral_value, axis=1)
    normalized["neutral"] = normalized["is_neutral"]
    return normalized


def validate_tournament_fixtures(fixtures: pd.DataFrame) -> None:
    """Raise ValueError if manually maintained tournament fixtures are invalid."""
    normalized = normalize_tournament_fixtures(fixtures)

    if normalized["match_id"].isna().any() or normalized["match_id"].eq("").any():
        raise ValueError("match_id must be non-null for every fixture.")
    if normalized["match_id"].duplicated().any():
        raise ValueError("match_id must be unique in tournament fixtures.")

    for column in ["team_a", "team_b", "stage"]:
        if normalized[column].isna().any() or normalized[column].eq("").any():
            raise ValueError(f"{column} must be non-null for every fixture.")

    same_team = normalized["team_a"].str.casefold().eq(normalized["team_b"].str.casefold())
    if same_team.any():
        raise ValueError("team_a and team_b must be different for every fixture.")

    invalid_stage = ~normalized["stage"].isin(VALID_STAGE_VALUES)
    if invalid_stage.any():
        invalid_values = ", ".join(sorted(normalized.loc[invalid_stage, "stage"].unique()))
        raise ValueError(f"Invalid tournament stage value(s): {invalid_values}")

    group_stage = normalized["stage"].eq("group")
    missing_group = normalized["group"].isna() | normalized["group"].eq("")
    if (group_stage & missing_group).any():
        raise ValueError("group must be present for group-stage fixtures.")


def load_tournament_fixtures(
    path: str | Path = FIXTURES_2026_PATH,
) -> pd.DataFrame:
    """Load, normalize, and validate manually maintained tournament fixtures."""
    fixture_path = Path(path)
    if not fixture_path.exists():
        raise FileNotFoundError(
            f"Missing tournament fixture file: {fixture_path}. "
            "Create data/tournament/fixtures_2026.csv using the schema in "
            "docs/fixtures_2026_template.md."
        )

    fixtures = pd.read_csv(fixture_path)
    normalized = normalize_tournament_fixtures(fixtures)
    validate_tournament_fixtures(normalized)
    return normalized
