import pandas as pd

from src.data.clean_results import clean_results
from src.data.quality_checks import validate_canonical_matches


def make_valid_canonical_matches() -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "date": ["2026-06-09", "2026-06-11"],
            "home_team": ["Team A", "Team C"],
            "away_team": ["Team B", "Team D"],
            "home_score": [2, 0],
            "away_score": [1, 3],
            "tournament": ["Friendly", "FIFA World Cup"],
            "city": ["City One", "City Two"],
            "country": ["Country One", "Country Two"],
            "neutral": [False, True],
        }
    )
    return clean_results(raw)


def test_valid_dataframe_passes() -> None:
    report = validate_canonical_matches(make_valid_canonical_matches())

    assert report.passed is True
    assert report.duplicate_match_id_count == 0
    assert report.matches_after_cutoff_count == 1


def test_duplicate_match_id_fails() -> None:
    matches = make_valid_canonical_matches()
    matches.loc[1, "match_id"] = matches.loc[0, "match_id"]

    report = validate_canonical_matches(matches)

    assert report.passed is False
    assert report.duplicate_match_id_count == 2


def test_negative_scores_fail() -> None:
    matches = make_valid_canonical_matches()
    matches.loc[0, "team_a_goals"] = -1

    report = validate_canonical_matches(matches)

    assert report.passed is False
    assert report.negative_score_count == 1


def test_invalid_result_label_fails() -> None:
    matches = make_valid_canonical_matches()
    matches.loc[0, "result"] = "home_win"

    report = validate_canonical_matches(matches)

    assert report.passed is False
    assert report.invalid_result_label_count == 1


def test_cutoff_inconsistency_fails() -> None:
    matches = make_valid_canonical_matches()
    matches.loc[1, "is_baseline_train_eligible"] = True

    report = validate_canonical_matches(matches)

    assert report.passed is False
    assert report.cutoff_inconsistency_count == 1


def test_missing_required_column_fails() -> None:
    matches = make_valid_canonical_matches().drop(columns=["total_goals"])

    report = validate_canonical_matches(matches)

    assert report.passed is False
    assert any("total_goals" in message for message in report.messages)


def test_invalid_neutral_type_fails() -> None:
    matches = make_valid_canonical_matches()
    matches["neutral"] = matches["neutral"].astype(object)
    matches["is_neutral"] = matches["is_neutral"].astype(object)
    matches.loc[0, "neutral"] = "FALSE"
    matches.loc[1, "is_neutral"] = "TRUE"

    report = validate_canonical_matches(matches)

    assert report.passed is False
    assert report.invalid_neutral_value_count == 2
