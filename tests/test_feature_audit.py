import pandas as pd

from src.features.feature_audit import (
    audit_feature_readiness,
    get_feature_columns,
    summarize_feature_audit,
)


def make_feature_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3", "m4"],
            "match_date": ["2021-01-01", "2021-06-01", "2022-01-01", "2022-06-01"],
            "team_a": ["A", "B", "C", "D"],
            "team_b": ["E", "F", "G", "H"],
            "result": ["team_a_win", "draw", "team_b_win", "team_a_win"],
            "tournament": ["Friendly", "Friendly", "Friendly", "Friendly"],
            "is_neutral": [False, True, False, True],
            "team_a_matches_played_before": [5, 6, 7, 8],
            "team_b_matches_played_before": [4, 5, 6, 7],
            "matches_played_before_diff": [1, 1, 1, 1],
            "team_a_bad_feature": ["x", "y", "z", "w"],
        }
    )


def test_get_feature_columns_excludes_metadata_and_non_features() -> None:
    columns = get_feature_columns(make_feature_df())

    assert "match_id" not in columns
    assert "match_date" not in columns
    assert "team_a" not in columns
    assert "team_b" not in columns
    assert "result" not in columns
    assert "tournament" not in columns
    assert "is_neutral" not in columns
    assert "team_a_bad_feature" not in columns


def test_get_feature_columns_includes_numeric_feature_columns() -> None:
    columns = get_feature_columns(make_feature_df())

    assert columns == [
        "team_a_matches_played_before",
        "team_b_matches_played_before",
        "matches_played_before_diff",
    ]


def test_audit_feature_readiness_reports_row_and_feature_counts() -> None:
    report = audit_feature_readiness(make_feature_df())

    assert report["row_count"] == 4
    assert report["column_count"] == 11
    assert report["feature_count"] == 3


def test_audit_feature_readiness_reports_train_and_test_counts() -> None:
    report = audit_feature_readiness(make_feature_df())

    assert report["train_row_count"] == 2
    assert report["test_row_count"] == 2
    assert report["train_date_range"] == ("2021-01-01", "2021-06-01")
    assert report["test_date_range"] == ("2022-01-01", "2022-06-01")


def test_audit_feature_readiness_reports_missingness() -> None:
    df = make_feature_df()
    df.loc[0, "team_a_matches_played_before"] = None

    report = audit_feature_readiness(df)

    assert report["missingness_by_feature_overall"][
        "team_a_matches_played_before"
    ] == 0.25
    assert report["missingness_by_feature_train"][
        "team_a_matches_played_before"
    ] == 0.50
    assert report["missingness_by_feature_test"][
        "team_a_matches_played_before"
    ] == 0.00


def test_fully_missing_features_are_flagged() -> None:
    df = make_feature_df()
    df["fully_missing_feature"] = pd.Series([float("nan")] * len(df), dtype="float64")

    report = audit_feature_readiness(df)

    assert "fully_missing_feature" in report["fully_missing_features"]
    assert report["passed"] is False


def test_high_missingness_features_are_flagged() -> None:
    df = make_feature_df()
    df["mostly_missing_feature"] = pd.Series(
        [1.0, float("nan"), float("nan"), float("nan")],
        dtype="float64",
    )

    report = audit_feature_readiness(df, high_missingness_threshold=0.50)

    assert "mostly_missing_feature" in report["high_missingness_features"]
    assert report["passed"] is False


def test_non_numeric_feature_candidates_do_not_enter_model_features() -> None:
    report = audit_feature_readiness(make_feature_df())

    assert "team_a_bad_feature" not in report["feature_columns"]
    assert "team_a_bad_feature" in report["non_numeric_feature_candidates"]
    assert report["passed"] is False


def test_summarize_feature_audit_returns_readable_text() -> None:
    report = audit_feature_readiness(make_feature_df())

    text = summarize_feature_audit(report)

    assert "Feature Readiness Audit" in text
    assert "numeric feature columns: 3" in text
    assert "messages:" in text
