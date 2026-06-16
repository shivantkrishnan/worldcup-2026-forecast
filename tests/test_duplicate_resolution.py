import pandas as pd

from src.data.clean_results import OUTCOME_TEAM_B_WIN, clean_results
from src.data.duplicate_resolution import (
    CONFLICTING_DUPLICATE,
    METADATA_DUPLICATE,
    resolve_duplicate_matches,
)


def make_canonical_matches() -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "date": ["2026-06-01", "2026-06-02"],
            "home_team": ["Team A", "Team C"],
            "away_team": ["Team B", "Team D"],
            "home_score": [2, 1],
            "away_score": [1, 1],
            "tournament": ["Friendly", "Friendly"],
            "city": ["City One", "City Two"],
            "country": ["Country One", "Country Two"],
            "neutral": [False, True],
        }
    )
    return clean_results(raw)


def test_no_duplicates_returns_unchanged_data_and_empty_quarantine() -> None:
    matches = make_canonical_matches()

    result = resolve_duplicate_matches(matches)

    pd.testing.assert_frame_equal(result.resolved_matches, matches)
    assert result.quarantined_matches.empty
    assert result.duplicate_report.duplicate_group_count == 0
    assert result.duplicate_report.duplicate_row_count == 0
    assert result.duplicate_report.quarantined_row_count == 0
    assert result.duplicate_report.resolved_match_count == len(matches)
    assert result.duplicate_report.passed is True


def test_metadata_duplicates_keep_one_row_and_quarantine_extras() -> None:
    matches = make_canonical_matches()
    duplicate = matches.iloc[[0]].copy()
    duplicate.loc[duplicate.index[0], "city"] = "Alternate City"
    with_duplicate = pd.concat([matches, duplicate], ignore_index=True)

    result = resolve_duplicate_matches(with_duplicate)

    assert len(result.resolved_matches) == len(matches)
    assert len(result.quarantined_matches) == 1
    assert result.quarantined_matches.loc[0, "quarantine_reason"] == METADATA_DUPLICATE
    assert result.duplicate_report.metadata_duplicate_group_count == 1
    assert result.duplicate_report.conflicting_duplicate_group_count == 0
    assert result.duplicate_report.quarantined_row_count == 1


def test_conflicting_duplicates_quarantine_full_group() -> None:
    matches = make_canonical_matches()
    duplicate = matches.iloc[[0]].copy()
    duplicate.loc[duplicate.index[0], "team_a_goals"] = 0
    duplicate.loc[duplicate.index[0], "team_b_goals"] = 3
    duplicate.loc[duplicate.index[0], "result"] = OUTCOME_TEAM_B_WIN
    with_duplicate = pd.concat([matches, duplicate], ignore_index=True)

    result = resolve_duplicate_matches(with_duplicate)

    assert len(result.resolved_matches) == 1
    assert result.resolved_matches["match_id"].tolist() == [matches.loc[1, "match_id"]]
    assert len(result.quarantined_matches) == 2
    assert result.quarantined_matches["quarantine_reason"].tolist() == [
        CONFLICTING_DUPLICATE,
        CONFLICTING_DUPLICATE,
    ]
    assert result.duplicate_report.conflicting_duplicate_group_count == 1
    assert result.duplicate_report.quarantined_row_count == 2


def test_resolved_output_has_unique_match_ids() -> None:
    matches = make_canonical_matches()
    metadata_duplicate = matches.iloc[[0]].copy()
    conflicting_duplicate = matches.iloc[[1]].copy()
    conflicting_duplicate.loc[conflicting_duplicate.index[0], "team_a_goals"] = 4
    with_duplicates = pd.concat(
        [matches, metadata_duplicate, conflicting_duplicate],
        ignore_index=True,
    )

    result = resolve_duplicate_matches(with_duplicates)

    assert result.resolved_matches["match_id"].is_unique
    assert result.duplicate_report.passed is True


def test_duplicate_report_counts_are_correct() -> None:
    matches = make_canonical_matches()

    metadata_duplicate = matches.iloc[[0]].copy()
    metadata_duplicate.loc[metadata_duplicate.index[0], "city"] = "Alternate City"

    conflicting_duplicate = matches.iloc[[1]].copy()
    conflicting_duplicate.loc[conflicting_duplicate.index[0], "team_a_goals"] = 0
    conflicting_duplicate.loc[conflicting_duplicate.index[0], "team_b_goals"] = 3
    conflicting_duplicate.loc[conflicting_duplicate.index[0], "result"] = (
        OUTCOME_TEAM_B_WIN
    )

    with_duplicates = pd.concat(
        [matches, metadata_duplicate, conflicting_duplicate],
        ignore_index=True,
    )

    report = resolve_duplicate_matches(with_duplicates).duplicate_report

    assert report.duplicate_group_count == 2
    assert report.duplicate_row_count == 4
    assert report.metadata_duplicate_group_count == 1
    assert report.conflicting_duplicate_group_count == 1
    assert report.quarantined_row_count == 3
    assert report.resolved_match_count == 1
