"""Time-aware train/test split utilities."""

from __future__ import annotations

import pandas as pd


def _sorted_by_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Return a copy sorted by date with parsed datetime values."""
    sorted_df = df.copy(deep=True)
    sorted_df[date_col] = pd.to_datetime(sorted_df[date_col], errors="raise")
    return sorted_df.sort_values(date_col, kind="mergesort").reset_index(drop=True)


def _format_date_or_none(value: pd.Timestamp | None) -> str | None:
    """Format a timestamp as YYYY-MM-DD, preserving empty ranges as None."""
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def chronological_train_test_split(
    df: pd.DataFrame,
    test_start_date: str = "2022-01-01",
    date_col: str = "match_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into train before test_start_date and test on/after it."""
    sorted_df = _sorted_by_date(df, date_col)
    test_start = pd.Timestamp(test_start_date)

    train_df = sorted_df.loc[sorted_df[date_col] < test_start].reset_index(drop=True)
    test_df = sorted_df.loc[sorted_df[date_col] >= test_start].reset_index(drop=True)

    return train_df, test_df


def rolling_origin_splits(
    df: pd.DataFrame,
    initial_train_end_date: str,
    test_window_months: int = 12,
    step_months: int = 12,
    final_test_end_date: str | None = None,
    date_col: str = "match_date",
) -> list[dict[str, object]]:
    """Create expanding-window, time-aware rolling-origin splits."""
    if test_window_months <= 0:
        raise ValueError("test_window_months must be positive.")
    if step_months <= 0:
        raise ValueError("step_months must be positive.")

    sorted_df = _sorted_by_date(df, date_col)
    if sorted_df.empty:
        return []

    current_train_end = pd.Timestamp(initial_train_end_date)
    final_test_end = (
        pd.Timestamp(final_test_end_date)
        if final_test_end_date is not None
        else sorted_df[date_col].max()
    )

    splits: list[dict[str, object]] = []
    while True:
        test_start = current_train_end + pd.Timedelta(days=1)
        if test_start > final_test_end:
            break

        test_end = test_start + pd.DateOffset(months=test_window_months) - pd.Timedelta(
            days=1
        )
        test_end = min(test_end, final_test_end)

        train_df = sorted_df.loc[sorted_df[date_col] <= current_train_end].reset_index(
            drop=True
        )
        test_df = sorted_df.loc[
            (sorted_df[date_col] >= test_start) & (sorted_df[date_col] <= test_end)
        ].reset_index(drop=True)

        if not train_df.empty and not test_df.empty:
            splits.append(
                {
                    "train_start_date": _format_date_or_none(train_df[date_col].min()),
                    "train_end_date": current_train_end.date().isoformat(),
                    "test_start_date": test_start.date().isoformat(),
                    "test_end_date": test_end.date().isoformat(),
                    "train_df": train_df,
                    "test_df": test_df,
                }
            )

        current_train_end = current_train_end + pd.DateOffset(months=step_months)

    return splits


def summarize_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, object]:
    """Return counts, date ranges, and result distributions for a split."""
    train_dates = pd.to_datetime(train_df["match_date"], errors="raise")
    test_dates = pd.to_datetime(test_df["match_date"], errors="raise")

    return {
        "train_count": int(len(train_df)),
        "test_count": int(len(test_df)),
        "train_start_date": _format_date_or_none(
            train_dates.min() if not train_df.empty else None
        ),
        "train_end_date": _format_date_or_none(
            train_dates.max() if not train_df.empty else None
        ),
        "test_start_date": _format_date_or_none(
            test_dates.min() if not test_df.empty else None
        ),
        "test_end_date": _format_date_or_none(
            test_dates.max() if not test_df.empty else None
        ),
        "train_result_distribution": train_df["result"].value_counts().to_dict()
        if "result" in train_df.columns
        else {},
        "test_result_distribution": test_df["result"].value_counts().to_dict()
        if "result" in test_df.columns
        else {},
    }
