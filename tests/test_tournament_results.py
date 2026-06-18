from pathlib import Path

import pandas as pd
import pytest

from src.data.tournament_results import (
    load_tournament_results,
    merge_completed_results_with_fixtures_or_predictions,
    normalize_tournament_results,
    validate_tournament_results,
)


def make_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "match_date": "2026-06-12",
                "group": "A",
                "team_a": "Alpha",
                "team_b": "Beta",
                "p_team_a_win": 0.20,
                "p_draw": 0.10,
                "p_team_b_win": 0.70,
            },
            {
                "match_id": "m2",
                "match_date": "2026-06-13",
                "group": "A",
                "team_a": "Gamma",
                "team_b": "Delta",
                "p_team_a_win": 1.00,
                "p_draw": 0.00,
                "p_team_b_win": 0.00,
            },
        ]
    )


def make_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "match_date": "2026-06-12",
                "team_a": "Alpha",
                "team_b": "Beta",
                "team_a_goals": 1,
                "team_b_goals": 2,
                "result": "team_b_win",
                "status": "completed",
            }
        ]
    )


def test_normalize_tournament_results_parses_dates_and_scores() -> None:
    normalized = normalize_tournament_results(make_results())

    assert pd.api.types.is_datetime64_any_dtype(normalized["match_date"])
    assert normalized.loc[0, "team_a_goals"] == 1
    assert normalized.loc[0, "status"] == "completed"


def test_results_validation_catches_duplicate_match_id() -> None:
    results = pd.concat([make_results(), make_results()], ignore_index=True)

    with pytest.raises(ValueError, match="unique"):
        validate_tournament_results(results)


def test_results_validation_catches_missing_required_fields() -> None:
    results = make_results().drop(columns=["team_a_goals"])

    with pytest.raises(ValueError, match="Missing required"):
        validate_tournament_results(results)


def test_results_validation_catches_result_score_inconsistency() -> None:
    results = make_results()
    results.loc[0, "result"] = "team_a_win"

    with pytest.raises(ValueError, match="consistent"):
        validate_tournament_results(results)


def test_results_join_to_fixture_predictions_by_match_id() -> None:
    merged = merge_completed_results_with_fixtures_or_predictions(
        make_predictions(),
        make_results(),
    )

    first = merged.loc[merged["match_id"].eq("m1")].iloc[0]
    second = merged.loc[merged["match_id"].eq("m2")].iloc[0]
    assert bool(first["is_completed"]) is True
    assert first["actual_result"] == "team_b_win"
    assert bool(second["is_completed"]) is False


def test_results_mismatched_team_orientation_raises_error() -> None:
    results = make_results()
    results.loc[0, "team_a"] = "Beta"
    results.loc[0, "team_b"] = "Alpha"

    with pytest.raises(ValueError, match="orientation"):
        merge_completed_results_with_fixtures_or_predictions(
            make_predictions(),
            results,
        )


def test_load_tournament_results_validates_against_predictions(tmp_path: Path) -> None:
    result_path = tmp_path / "results_2026.csv"
    make_results().to_csv(result_path, index=False)

    loaded = load_tournament_results(result_path, fixtures_or_predictions=make_predictions())

    assert len(loaded) == 1
