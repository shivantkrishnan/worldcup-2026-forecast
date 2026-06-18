"""Load, normalize, and validate manually maintained 2026 tournament results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import RESULTS_2026_PATH

REQUIRED_TOURNAMENT_RESULT_COLUMNS = {
    "match_id",
    "match_date",
    "team_a",
    "team_b",
    "team_a_goals",
    "team_b_goals",
    "result",
    "status",
}
OPTIONAL_TOURNAMENT_RESULT_COLUMNS = {
    "went_to_extra_time",
    "went_to_penalties",
    "team_a_penalties",
    "team_b_penalties",
    "source",
    "last_updated",
}
VALID_RESULT_LABELS = {"team_a_win", "draw", "team_b_win"}
VALID_RESULT_STATUS_VALUES = {"completed"}
SCORE_COLUMNS = ["team_a_goals", "team_b_goals"]


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
    raise ValueError(f"Unable to parse tournament result boolean: {value!r}")


def _expected_result(team_a_goals: int, team_b_goals: int) -> str:
    """Return the canonical result label implied by a scoreline."""
    if team_a_goals > team_b_goals:
        return "team_a_win"
    if team_a_goals < team_b_goals:
        return "team_b_win"
    return "draw"


def _validate_required_columns(results: pd.DataFrame) -> None:
    """Raise a clear error when required result columns are missing."""
    missing = REQUIRED_TOURNAMENT_RESULT_COLUMNS.difference(results.columns)
    if missing:
        raise ValueError(
            "Missing required tournament result columns: "
            + ", ".join(sorted(missing))
        )


def normalize_tournament_results(results: pd.DataFrame) -> pd.DataFrame:
    """Return normalized tournament results without mutating input."""
    _validate_required_columns(results)
    normalized = results.copy(deep=True)

    normalized["match_id"] = normalized["match_id"].astype(str).str.strip()
    normalized["match_date"] = pd.to_datetime(
        normalized["match_date"],
        errors="raise",
    )
    for column in ["team_a", "team_b", "result", "status"]:
        normalized[column] = normalized[column].astype("string").str.strip()

    normalized["result"] = normalized["result"].str.casefold()
    normalized["status"] = normalized["status"].str.casefold()

    for column in SCORE_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    for column in ["team_a_penalties", "team_b_penalties"]:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    for column in ["went_to_extra_time", "went_to_penalties"]:
        if column in normalized.columns:
            normalized[column] = normalized[column].map(_coerce_bool_or_missing)

    return normalized


def _validate_score_values(results: pd.DataFrame) -> None:
    """Validate score completeness and integer-ness for completed results."""
    completed = results["status"].eq("completed")
    for column in SCORE_COLUMNS:
        missing_scores = completed & results[column].isna()
        if missing_scores.any():
            raise ValueError(f"{column} must be non-null for completed results.")

        score_values = results.loc[completed, column]
        non_integer = ~np.isclose(score_values, np.floor(score_values))
        if non_integer.any():
            raise ValueError(f"{column} must contain integer scores.")

        if (score_values < 0).any():
            raise ValueError(f"{column} must be nonnegative.")


def _validate_result_consistency(results: pd.DataFrame) -> None:
    """Validate that result labels agree with the listed scoreline."""
    completed = results.loc[results["status"].eq("completed")]
    for row in completed.itertuples(index=False):
        expected = _expected_result(
            int(getattr(row, "team_a_goals")),
            int(getattr(row, "team_b_goals")),
        )
        actual = str(getattr(row, "result"))
        if actual != expected:
            raise ValueError(
                "result must be consistent with team_a_goals and team_b_goals "
                f"for match_id {getattr(row, 'match_id')}."
            )


def _validate_join_to_predictions(
    results: pd.DataFrame,
    fixtures_or_predictions: pd.DataFrame,
) -> None:
    """Validate result match IDs and team orientation against fixture rows."""
    required = {"match_id", "team_a", "team_b"}
    missing = required.difference(fixtures_or_predictions.columns)
    if missing:
        raise ValueError(
            "Fixture/prediction table is missing required join columns: "
            + ", ".join(sorted(missing))
        )

    reference = fixtures_or_predictions.copy(deep=True)
    reference["match_id"] = reference["match_id"].astype(str).str.strip()
    reference["team_a"] = reference["team_a"].astype(str).str.strip()
    reference["team_b"] = reference["team_b"].astype(str).str.strip()
    reference = reference.set_index("match_id", drop=False)

    missing_match_ids = sorted(set(results["match_id"]).difference(reference.index))
    if missing_match_ids:
        raise ValueError(
            "Tournament results contain match_id values not found in fixtures or "
            "predictions: "
            + ", ".join(missing_match_ids)
        )

    for row in results.itertuples(index=False):
        match_id = str(getattr(row, "match_id"))
        reference_row = reference.loc[match_id]
        if (
            str(getattr(row, "team_a")) != str(reference_row["team_a"])
            or str(getattr(row, "team_b")) != str(reference_row["team_b"])
        ):
            raise ValueError(
                "Tournament result team orientation does not match fixture or "
                f"prediction row for match_id {match_id}."
            )


def validate_tournament_results(
    results: pd.DataFrame,
    fixtures_or_predictions: pd.DataFrame | None = None,
) -> None:
    """Raise ValueError if manually maintained tournament results are invalid."""
    normalized = normalize_tournament_results(results)

    if normalized["match_id"].isna().any() or normalized["match_id"].eq("").any():
        raise ValueError("match_id must be non-null for every tournament result.")
    if normalized["match_id"].duplicated().any():
        raise ValueError("match_id must be unique in tournament results.")

    for column in ["team_a", "team_b", "result", "status"]:
        if normalized[column].isna().any() or normalized[column].eq("").any():
            raise ValueError(f"{column} must be non-null for every result.")

    same_team = normalized["team_a"].str.casefold().eq(
        normalized["team_b"].str.casefold()
    )
    if same_team.any():
        raise ValueError("team_a and team_b must be different for every result.")

    invalid_status = ~normalized["status"].isin(VALID_RESULT_STATUS_VALUES)
    if invalid_status.any():
        invalid_values = ", ".join(
            sorted(normalized.loc[invalid_status, "status"].astype(str).unique())
        )
        raise ValueError(f"Invalid tournament result status value(s): {invalid_values}")

    invalid_result = ~normalized["result"].isin(VALID_RESULT_LABELS)
    if invalid_result.any():
        invalid_values = ", ".join(
            sorted(normalized.loc[invalid_result, "result"].astype(str).unique())
        )
        raise ValueError(f"Invalid tournament result label(s): {invalid_values}")

    _validate_score_values(normalized)
    _validate_result_consistency(normalized)

    if fixtures_or_predictions is not None:
        _validate_join_to_predictions(normalized, fixtures_or_predictions)


def load_tournament_results(
    path: str | Path = RESULTS_2026_PATH,
    fixtures_or_predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load, normalize, and validate manually maintained tournament results."""
    result_path = Path(path)
    if not result_path.exists():
        raise FileNotFoundError(
            f"Missing tournament result file: {result_path}. "
            "Create data/tournament/results_2026.csv using the schema in "
            "docs/results_2026_template.md."
        )

    results = pd.read_csv(result_path)
    normalized = normalize_tournament_results(results)
    validate_tournament_results(
        normalized,
        fixtures_or_predictions=fixtures_or_predictions,
    )
    return normalized


def merge_completed_results_with_fixtures_or_predictions(
    predictions: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Return predictions annotated with completed-result conditioning columns."""
    validate_tournament_results(results, fixtures_or_predictions=predictions)
    normalized_results = normalize_tournament_results(results)
    completed = normalized_results.loc[
        normalized_results["status"].eq("completed"),
        [
            "match_id",
            "match_date",
            "team_a",
            "team_b",
            "team_a_goals",
            "team_b_goals",
            "result",
            "status",
        ],
    ].copy()
    completed = completed.rename(
        columns={
            "match_date": "result_match_date",
            "team_a": "result_team_a",
            "team_b": "result_team_b",
            "result": "actual_result",
            "status": "result_status",
        }
    )

    merged = predictions.copy(deep=True).merge(
        completed,
        on="match_id",
        how="left",
        validate="one_to_one",
    )
    merged["is_completed"] = merged["actual_result"].notna()
    return merged
