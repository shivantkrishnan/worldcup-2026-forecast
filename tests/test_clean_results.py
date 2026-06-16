import pandas as pd
import pytest

from src.data.clean_results import (
    CANONICAL_MATCH_COLUMNS,
    OUTCOME_DRAW,
    OUTCOME_TEAM_A_WIN,
    OUTCOME_TEAM_B_WIN,
    clean_results,
    create_match_id,
    filter_baseline_training_matches,
    label_outcome,
)


def make_raw_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-06-09", "2026-06-10", "2026-06-11"],
            "home_team": [" Team A ", "Team C", "Team E"],
            "away_team": ["Team B", " Team D ", "Team F"],
            "home_score": [2, 1, 0],
            "away_score": [1, 1, 3],
            "tournament": [" Friendly ", "FIFA World Cup", "FIFA World Cup"],
            "city": [" City One ", "City Two", "City Three"],
            "country": [" Country One ", "Country Two", "Country Three"],
            "neutral": ["FALSE", "TRUE", False],
        }
    )


def test_label_outcome_team_a_win() -> None:
    assert label_outcome(2, 1) == OUTCOME_TEAM_A_WIN


def test_label_outcome_draw() -> None:
    assert label_outcome(1, 1) == OUTCOME_DRAW


def test_label_outcome_team_b_win() -> None:
    assert label_outcome(0, 3) == OUTCOME_TEAM_B_WIN


def test_clean_results_creates_result_labels() -> None:
    cleaned = clean_results(make_raw_results())

    assert cleaned["result"].tolist() == [
        OUTCOME_TEAM_A_WIN,
        OUTCOME_DRAW,
        OUTCOME_TEAM_B_WIN,
    ]


def test_clean_results_returns_canonical_columns_in_expected_order() -> None:
    cleaned = clean_results(make_raw_results())

    assert cleaned.columns.tolist() == CANONICAL_MATCH_COLUMNS


def test_clean_results_parses_match_date() -> None:
    cleaned = clean_results(make_raw_results())

    assert pd.api.types.is_datetime64_any_dtype(cleaned["match_date"])


def test_clean_results_coerces_neutral_columns_to_boolean() -> None:
    cleaned = clean_results(make_raw_results())

    assert pd.api.types.is_bool_dtype(cleaned["neutral"])
    assert pd.api.types.is_bool_dtype(cleaned["is_neutral"])
    assert cleaned["neutral"].tolist() == [False, True, False]
    assert cleaned["is_neutral"].tolist() == [False, True, False]


def test_clean_results_calculates_goal_features() -> None:
    cleaned = clean_results(make_raw_results())

    assert cleaned["goal_diff_team_a"].tolist() == [1, 0, -3]
    assert cleaned["total_goals"].tolist() == [3, 2, 3]


def test_clean_results_sets_training_cutoff_eligibility() -> None:
    cleaned = clean_results(make_raw_results())

    assert cleaned["training_cutoff_date"].tolist() == [
        "2026-06-10",
        "2026-06-10",
        "2026-06-10",
    ]
    assert cleaned["is_baseline_train_eligible"].tolist() == [True, True, False]


def test_filter_baseline_training_matches_excludes_matches_after_cutoff() -> None:
    cleaned = clean_results(make_raw_results())

    filtered = filter_baseline_training_matches(cleaned)

    assert filtered["match_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-06-09",
        "2026-06-10",
    ]


def test_clean_results_does_not_mutate_input_dataframe() -> None:
    raw = make_raw_results()
    original = raw.copy(deep=True)

    clean_results(raw)

    pd.testing.assert_frame_equal(raw, original)


def test_clean_results_standardizes_string_columns() -> None:
    cleaned = clean_results(make_raw_results())

    assert cleaned.loc[0, "team_a"] == "Team A"
    assert cleaned.loc[1, "team_b"] == "Team D"
    assert cleaned.loc[0, "tournament"] == "Friendly"
    assert cleaned.loc[0, "city"] == "City One"
    assert cleaned.loc[0, "country"] == "Country One"


def test_create_match_id_is_deterministic() -> None:
    cleaned = clean_results(make_raw_results())

    assert create_match_id(cleaned.iloc[0]) == cleaned.loc[0, "match_id"]


def test_clean_results_requires_expected_columns() -> None:
    raw = pd.DataFrame({"date": ["2024-01-01"]})

    with pytest.raises(ValueError, match="Missing required raw result columns"):
        clean_results(raw)
