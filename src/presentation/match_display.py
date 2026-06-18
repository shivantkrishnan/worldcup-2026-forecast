"""Build display-safe match tables from fixtures, predictions, and results."""

from __future__ import annotations

from typing import Any

import pandas as pd

DISPLAY_STATUS_COMPLETED = "completed"
DISPLAY_STATUS_SCHEDULED = "scheduled"
DISPLAY_STATUS_PREDICTION_MISSING = "prediction_missing"

FIXTURE_COLUMNS = ["match_id", "match_date", "group", "team_a", "team_b"]
PREDICTION_COLUMNS = [
    "p_team_a_win",
    "p_draw",
    "p_team_b_win",
    "predicted_class",
    "favorite_display",
    "confidence_label",
    "forecast_mode",
    "is_backfilled",
]
RESULT_COLUMNS = ["team_a_goals", "team_b_goals", "result"]
DISPLAY_COLUMNS = [
    "match_id",
    "match_date",
    "group",
    "team_a",
    "team_b",
    "display_status",
    "team_a_goals",
    "team_b_goals",
    "actual_result",
    "p_team_a_win",
    "p_draw",
    "p_team_b_win",
    "predicted_class",
    "favorite_display",
    "confidence_label",
    "forecast_mode",
    "is_backfilled",
    "prediction_display_label",
    "audit_available",
]


def _require_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    """Raise a clear error when a source table is missing required columns."""
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: " + ", ".join(missing)
        )


def _normalize_match_ids(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with match IDs normalized for one-to-one joins."""
    normalized = dataframe.copy(deep=True)
    normalized["match_id"] = normalized["match_id"].astype(str).str.strip()
    return normalized


def _validate_unique_match_id(dataframe: pd.DataFrame, table_name: str) -> None:
    """Ensure a display input has at most one row per match."""
    duplicated = dataframe["match_id"].duplicated(keep=False)
    if duplicated.any():
        duplicate_ids = sorted(
            dataframe.loc[duplicated, "match_id"].astype(str).unique()
        )
        raise ValueError(
            f"{table_name} must have unique match_id values. Duplicates: "
            + ", ".join(duplicate_ids)
        )


def _coerce_display_bool(value: Any) -> bool:
    """Coerce common boolean values for display flags."""
    if value is None or pd.isna(value):
        return False
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
    raise ValueError(f"Unable to parse display boolean value: {value!r}")


def _validate_optional_orientation(
    fixtures: pd.DataFrame,
    table: pd.DataFrame,
    table_name: str,
) -> None:
    """Validate team orientation when an overlay table carries team columns."""
    if not {"team_a", "team_b"}.issubset(table.columns) or table.empty:
        return

    reference = fixtures[["match_id", "team_a", "team_b"]].copy(deep=True)
    overlay = table[["match_id", "team_a", "team_b"]].copy(deep=True)
    merged = overlay.merge(
        reference,
        on="match_id",
        how="left",
        suffixes=("_overlay", "_fixture"),
        validate="one_to_one",
    )

    missing_fixture = merged["team_a_fixture"].isna()
    if missing_fixture.any():
        missing_ids = sorted(merged.loc[missing_fixture, "match_id"].astype(str).unique())
        raise ValueError(
            f"{table_name} contains match_id values not found in fixtures: "
            + ", ".join(missing_ids)
        )

    mismatch = (
        merged["team_a_overlay"].astype(str).ne(merged["team_a_fixture"].astype(str))
        | merged["team_b_overlay"].astype(str).ne(merged["team_b_fixture"].astype(str))
    )
    if mismatch.any():
        mismatch_ids = sorted(merged.loc[mismatch, "match_id"].astype(str).unique())
        raise ValueError(
            f"{table_name} team orientation does not match fixtures for match_id: "
            + ", ".join(mismatch_ids)
        )


def _prepare_predictions(
    fixtures: pd.DataFrame,
    predictions: pd.DataFrame | None,
) -> pd.DataFrame:
    """Return prediction columns ready for a left join to fixtures."""
    if predictions is None:
        return pd.DataFrame(
            columns=["match_id", *PREDICTION_COLUMNS, "has_prediction"]
        )

    prepared = _normalize_match_ids(predictions)
    _require_columns(prepared, ["match_id"], "predictions")
    _validate_unique_match_id(prepared, "predictions")
    _validate_optional_orientation(fixtures, prepared, "predictions")

    selected = prepared[["match_id"]].copy(deep=True)
    for column in PREDICTION_COLUMNS:
        selected[column] = prepared[column] if column in prepared.columns else pd.NA
    selected["has_prediction"] = True
    return selected


def _prepare_results(fixtures: pd.DataFrame, results: pd.DataFrame | None) -> pd.DataFrame:
    """Return completed-result columns ready for a left join to fixtures."""
    if results is None:
        return pd.DataFrame(
            columns=[
                "match_id",
                "team_a_goals",
                "team_b_goals",
                "actual_result",
                "has_result",
            ]
        )

    prepared = _normalize_match_ids(results)
    _require_columns(prepared, ["match_id", *RESULT_COLUMNS], "results")
    _validate_unique_match_id(prepared, "results")
    _validate_optional_orientation(fixtures, prepared, "results")

    if "status" in prepared.columns:
        status = prepared["status"].astype("string").str.strip().str.casefold()
        prepared = prepared.loc[status.eq("completed")].copy(deep=True)

    selected = prepared[
        ["match_id", "team_a_goals", "team_b_goals", "result"]
    ].copy(deep=True)
    selected = selected.rename(columns={"result": "actual_result"})
    selected["has_result"] = True
    return selected


def _prediction_display_label(row: pd.Series) -> str:
    """Return a label that separates current forecasts from audit probabilities."""
    has_prediction = bool(row["has_prediction"])
    if not has_prediction:
        return "Prediction unavailable"

    is_backfilled = bool(row["is_backfilled"])
    is_completed = row["display_status"] == DISPLAY_STATUS_COMPLETED

    if is_completed and is_backfilled:
        return "Backfilled ex-ante model probability"
    if is_completed:
        return "Model audit probability"
    if is_backfilled:
        return "Backfilled ex-ante prediction"
    return "Prediction"


def build_match_display_table(
    fixtures: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one display-safe row per fixture.

    Completed matches are labeled as completed and show actual scores from
    results. Prediction probabilities can remain present for audit, but their
    labels do not describe the row as a current scheduled-match prediction.
    """
    fixture_rows = _normalize_match_ids(fixtures)
    _require_columns(fixture_rows, FIXTURE_COLUMNS, "fixtures")
    _validate_unique_match_id(fixture_rows, "fixtures")

    display = fixture_rows[FIXTURE_COLUMNS].copy(deep=True)
    prediction_rows = _prepare_predictions(fixture_rows, predictions)
    result_rows = _prepare_results(fixture_rows, results)

    display = display.merge(
        prediction_rows,
        on="match_id",
        how="left",
        validate="one_to_one",
    )
    display = display.merge(
        result_rows,
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    display["has_prediction"] = display["has_prediction"].fillna(False).astype(bool)
    display["has_result"] = display["has_result"].fillna(False).astype(bool)
    display["is_backfilled"] = display["is_backfilled"].map(_coerce_display_bool)

    display["display_status"] = DISPLAY_STATUS_PREDICTION_MISSING
    display.loc[display["has_prediction"], "display_status"] = DISPLAY_STATUS_SCHEDULED
    display.loc[display["has_result"], "display_status"] = DISPLAY_STATUS_COMPLETED

    display["audit_available"] = (
        display["display_status"].eq(DISPLAY_STATUS_COMPLETED)
        & display["has_prediction"]
    )
    display["prediction_display_label"] = display.apply(
        _prediction_display_label,
        axis=1,
    )

    return display[DISPLAY_COLUMNS].copy(deep=True)


def build_prediction_audit_table(display_table: pd.DataFrame) -> pd.DataFrame:
    """Return completed matches with model probabilities labeled for audit."""
    _require_columns(display_table, DISPLAY_COLUMNS, "display_table")
    audit = display_table.loc[
        display_table["audit_available"].astype(bool)
    ].copy(deep=True)

    result_to_probability_column = {
        "team_a_win": "p_team_a_win",
        "draw": "p_draw",
        "team_b_win": "p_team_b_win",
    }
    audit["actual_outcome_probability"] = audit.apply(
        lambda row: row[result_to_probability_column[str(row["actual_result"])]],
        axis=1,
    )

    return audit[
        [
            "match_id",
            "match_date",
            "group",
            "team_a",
            "team_b",
            "team_a_goals",
            "team_b_goals",
            "actual_result",
            "actual_outcome_probability",
            "predicted_class",
            "favorite_display",
            "forecast_mode",
            "is_backfilled",
            "prediction_display_label",
        ]
    ].copy(deep=True)
