"""In-memory quality checks for canonical match data."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.data.clean_results import (
    CANONICAL_MATCH_COLUMNS,
    OUTCOME_DRAW,
    OUTCOME_TEAM_A_WIN,
    OUTCOME_TEAM_B_WIN,
)
from src.utils.config import DEFAULT_TRAINING_CUTOFF_DATE

VALID_RESULT_LABELS = {
    OUTCOME_TEAM_A_WIN,
    OUTCOME_DRAW,
    OUTCOME_TEAM_B_WIN,
}


@dataclass(frozen=True)
class DataQualityReport:
    """Structured report for canonical match quality checks."""

    duplicate_match_id_count: int
    negative_score_count: int
    null_required_value_count: int
    invalid_result_label_count: int
    invalid_neutral_value_count: int
    cutoff_inconsistency_count: int
    matches_after_cutoff_count: int
    passed: bool
    messages: list[str]

    def to_dict(self) -> dict[str, object]:
        """Return the report as a plain dictionary."""
        return asdict(self)


def _count_invalid_boolean_values(series: pd.Series) -> int:
    """Count values that are not real boolean values."""
    if pd.api.types.is_bool_dtype(series):
        return 0
    return int((~series.isin([True, False])).sum())


def validate_canonical_matches(
    df: pd.DataFrame,
    training_cutoff_date: str = DEFAULT_TRAINING_CUTOFF_DATE,
) -> DataQualityReport:
    """Validate canonical completed-match data without mutating input."""
    messages: list[str] = []
    missing_columns = [
        column for column in CANONICAL_MATCH_COLUMNS if column not in df.columns
    ]

    if missing_columns:
        messages.append(
            "Missing required canonical columns: " + ", ".join(missing_columns)
        )

    present_required_columns = [
        column for column in CANONICAL_MATCH_COLUMNS if column in df.columns
    ]
    null_required_value_count = int(df[present_required_columns].isna().sum().sum())
    if null_required_value_count:
        messages.append(
            f"Found {null_required_value_count} null values in required columns."
        )

    duplicate_match_id_count = 0
    if "match_id" in df.columns:
        duplicate_match_id_count = int(df["match_id"].duplicated(keep=False).sum())
        if duplicate_match_id_count:
            messages.append(
                f"Found {duplicate_match_id_count} rows with duplicate match_id values."
            )

    negative_score_count = 0
    if {"team_a_goals", "team_b_goals"}.issubset(df.columns):
        team_a_goals = pd.to_numeric(df["team_a_goals"], errors="coerce")
        team_b_goals = pd.to_numeric(df["team_b_goals"], errors="coerce")
        negative_score_count = int(((team_a_goals < 0) | (team_b_goals < 0)).sum())
        if negative_score_count:
            messages.append(f"Found {negative_score_count} rows with negative scores.")

    invalid_result_label_count = 0
    if "result" in df.columns:
        invalid_result_label_count = int((~df["result"].isin(VALID_RESULT_LABELS)).sum())
        if invalid_result_label_count:
            messages.append(
                f"Found {invalid_result_label_count} rows with invalid result labels."
            )

    invalid_neutral_value_count = 0
    for column in ("neutral", "is_neutral"):
        if column in df.columns:
            invalid_neutral_value_count += _count_invalid_boolean_values(df[column])
    if invalid_neutral_value_count:
        messages.append(
            f"Found {invalid_neutral_value_count} invalid neutral boolean values."
        )

    cutoff_inconsistency_count = 0
    matches_after_cutoff_count = 0
    if {"match_date", "is_baseline_train_eligible"}.issubset(df.columns):
        match_dates = pd.to_datetime(df["match_date"], errors="coerce")
        cutoff = pd.Timestamp(training_cutoff_date)
        expected_eligibility = match_dates <= cutoff
        actual_eligibility = df["is_baseline_train_eligible"]
        cutoff_inconsistency_count = int((actual_eligibility != expected_eligibility).sum())
        matches_after_cutoff_count = int((match_dates > cutoff).sum())

        if cutoff_inconsistency_count:
            messages.append(
                "Found "
                f"{cutoff_inconsistency_count} rows with baseline cutoff "
                "eligibility inconsistencies."
            )

    failure_count = (
        len(missing_columns)
        + duplicate_match_id_count
        + negative_score_count
        + null_required_value_count
        + invalid_result_label_count
        + invalid_neutral_value_count
        + cutoff_inconsistency_count
    )
    passed = failure_count == 0

    if matches_after_cutoff_count:
        messages.append(
            f"{matches_after_cutoff_count} matches occur after {training_cutoff_date} "
            "and are excluded from baseline training."
        )

    if passed:
        messages.insert(0, "All required canonical data quality checks passed.")

    return DataQualityReport(
        duplicate_match_id_count=duplicate_match_id_count,
        negative_score_count=negative_score_count,
        null_required_value_count=null_required_value_count,
        invalid_result_label_count=invalid_result_label_count,
        invalid_neutral_value_count=invalid_neutral_value_count,
        cutoff_inconsistency_count=cutoff_inconsistency_count,
        matches_after_cutoff_count=matches_after_cutoff_count,
        passed=passed,
        messages=messages,
    )
