"""In-memory feature-readiness auditing before model training."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.splits import chronological_train_test_split, summarize_split

NON_FEATURE_COLUMNS = {
    "match_id",
    "match_date",
    "team_a",
    "team_b",
    "result",
    "target",
    "actual_result",
    "predicted_class",
    "tournament",
    "is_neutral",
    "training_cutoff_date",
    "is_baseline_train_eligible",
    "model_version",
    "prediction_timestamp",
    "feature_cutoff_timestamp",
    "notes",
}


def _format_range(start: Any, end: Any) -> tuple[str | None, str | None]:
    """Return a date range tuple for report serialization."""
    if start is None or pd.isna(start) or end is None or pd.isna(end):
        return (None, None)
    return (
        pd.Timestamp(start).date().isoformat(),
        pd.Timestamp(end).date().isoformat(),
    )


def _candidate_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that are not explicitly excluded metadata/target fields."""
    return [column for column in df.columns if column not in NON_FEATURE_COLUMNS]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns suitable for baseline modeling."""
    feature_columns: list[str] = []
    for column in _candidate_columns(df):
        series = df[column]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
            series
        ):
            feature_columns.append(column)
    return feature_columns


def _non_numeric_feature_candidates(df: pd.DataFrame) -> list[str]:
    """Return non-numeric columns that were not excluded as metadata."""
    non_numeric_columns: list[str] = []
    for column in _candidate_columns(df):
        series = df[column]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(
            series
        ):
            non_numeric_columns.append(column)
    return non_numeric_columns


def _missingness(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    """Return missing-value share by feature column."""
    if not feature_columns:
        return {}
    return {
        column: float(df[column].isna().mean()) if len(df) else 0.0
        for column in feature_columns
    }


def _target_distribution(df: pd.DataFrame, target_col: str) -> dict[str, int]:
    """Return target distribution as plain Python ints."""
    if target_col not in df.columns:
        return {}
    return {
        str(label): int(count)
        for label, count in df[target_col].value_counts().to_dict().items()
    }


def audit_feature_readiness(
    features_df: pd.DataFrame,
    test_start_date: str = "2022-01-01",
    date_col: str = "match_date",
    target_col: str = "result",
    high_missingness_threshold: float = 0.50,
) -> dict[str, object]:
    """Audit a match-level feature table for baseline modeling readiness."""
    features = features_df.copy(deep=True)
    features[date_col] = pd.to_datetime(features[date_col], errors="raise")
    feature_columns = get_feature_columns(features)
    non_numeric_feature_candidates = _non_numeric_feature_candidates(features)
    train_df, test_df = chronological_train_test_split(
        features,
        test_start_date=test_start_date,
        date_col=date_col,
    )
    split_summary = summarize_split(train_df, test_df)

    missingness_by_feature_overall = _missingness(features, feature_columns)
    missingness_by_feature_train = _missingness(train_df, feature_columns)
    missingness_by_feature_test = _missingness(test_df, feature_columns)

    fully_missing_features = [
        column
        for column, missing_share in missingness_by_feature_overall.items()
        if missing_share == 1.0
    ]
    high_missingness_features = [
        column
        for column, missing_share in missingness_by_feature_overall.items()
        if missing_share > high_missingness_threshold
    ]

    messages: list[str] = []
    if not feature_columns:
        messages.append("No numeric model candidate feature columns found.")
    if len(train_df) == 0:
        messages.append("Train split has zero rows.")
    if len(test_df) == 0:
        messages.append("Test split has zero rows.")
    if target_col not in features.columns:
        messages.append(f"Target column is missing: {target_col}")
    if fully_missing_features:
        messages.append(
            f"{len(fully_missing_features)} feature columns are fully missing."
        )
    if high_missingness_features:
        messages.append(
            f"{len(high_missingness_features)} feature columns exceed "
            f"{high_missingness_threshold:.0%} missingness."
        )
    if non_numeric_feature_candidates:
        messages.append(
            f"{len(non_numeric_feature_candidates)} non-numeric candidate columns "
            "were excluded from model features."
        )

    passed = not messages
    if passed:
        messages.append("Feature table passed readiness checks.")

    return {
        "row_count": int(len(features)),
        "column_count": int(features.shape[1]),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "train_row_count": int(len(train_df)),
        "test_row_count": int(len(test_df)),
        "train_date_range": (
            split_summary["train_start_date"],
            split_summary["train_end_date"],
        ),
        "test_date_range": (
            split_summary["test_start_date"],
            split_summary["test_end_date"],
        ),
        "target_distribution_overall": _target_distribution(features, target_col),
        "target_distribution_train": _target_distribution(train_df, target_col),
        "target_distribution_test": _target_distribution(test_df, target_col),
        "missingness_by_feature_overall": missingness_by_feature_overall,
        "missingness_by_feature_train": missingness_by_feature_train,
        "missingness_by_feature_test": missingness_by_feature_test,
        "high_missingness_features": high_missingness_features,
        "fully_missing_features": fully_missing_features,
        "non_numeric_feature_candidates": non_numeric_feature_candidates,
        "passed": passed,
        "messages": messages,
    }


def summarize_feature_audit(report: dict[str, object]) -> str:
    """Convert a feature audit report into readable console text."""
    lines = [
        "Feature Readiness Audit",
        "=======================",
        f"passed: {report['passed']}",
        f"rows: {report['row_count']}",
        f"columns: {report['column_count']}",
        f"numeric feature columns: {report['feature_count']}",
        f"train rows: {report['train_row_count']}",
        f"test rows: {report['test_row_count']}",
        f"train date range: {report['train_date_range']}",
        f"test date range: {report['test_date_range']}",
        f"fully missing features: {len(report['fully_missing_features'])}",
        f"high missingness features: {len(report['high_missingness_features'])}",
        "messages:",
    ]
    for message in report["messages"]:
        lines.append(f"- {message}")
    return "\n".join(lines)
