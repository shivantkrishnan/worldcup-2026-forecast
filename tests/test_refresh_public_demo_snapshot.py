import pandas as pd
import pytest

from scripts import ingest_official_results_2026 as official_results
from scripts.refresh_public_demo_snapshot import (
    _verified_result_generated_at,
    fetch_and_write_official_results,
    preserve_existing_result_metadata,
    validate_public_demo_snapshot,
)


def make_fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "match_date": "2026-06-11",
                "group": "A",
                "stage": "group",
                "team_a": "Alpha",
                "team_b": "Beta",
            },
            {
                "match_id": "m2",
                "match_date": "2026-06-15",
                "group": "A",
                "stage": "group",
                "team_a": "Gamma",
                "team_b": "Delta",
            },
        ]
    )


def make_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "match_date": "2026-06-11",
                "team_a": "Alpha",
                "team_b": "Beta",
                "team_a_goals": 2,
                "team_b_goals": 1,
                "result": "team_a_win",
                "status": "completed",
            }
        ]
    )


def make_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m2",
                "match_date": "2026-06-15",
                "group": "A",
                "stage": "group",
                "team_a": "Gamma",
                "team_b": "Delta",
                "p_team_a_win": 0.4,
                "p_draw": 0.3,
                "p_team_b_win": 0.3,
                "forecast_mode": "live",
                "feature_cutoff_date": "2026-06-11",
                "is_backfilled": False,
            }
        ]
    )


def test_validate_public_demo_snapshot_passes_for_valid_state() -> None:
    report = validate_public_demo_snapshot(
        make_fixtures(),
        make_results(),
        make_predictions(),
    )

    assert report["total_fixtures"] == 2
    assert report["completed_results"] == 1
    assert report["live_prediction_rows"] == 1
    assert report["feature_cutoff_date"] == "2026-06-11"


def test_validate_public_demo_snapshot_rejects_completed_prediction_overlap() -> None:
    predictions = make_predictions()
    predictions.loc[0, "match_id"] = "m1"

    with pytest.raises(ValueError, match="must omit completed fixture"):
        validate_public_demo_snapshot(make_fixtures(), make_results(), predictions)


def test_validate_public_demo_snapshot_rejects_stale_feature_cutoff() -> None:
    predictions = make_predictions()
    predictions.loc[0, "feature_cutoff_date"] = "2026-06-10"

    with pytest.raises(ValueError, match="feature_cutoff_date"):
        validate_public_demo_snapshot(make_fixtures(), make_results(), predictions)


def test_validate_public_demo_snapshot_rejects_count_mismatch() -> None:
    predictions = make_predictions().iloc[0:0].copy()

    with pytest.raises(ValueError, match="must equal total fixtures"):
        validate_public_demo_snapshot(make_fixtures(), make_results(), predictions)


def test_preserve_existing_result_metadata_keeps_unchanged_source_fields() -> None:
    previous = make_results()
    previous["source"] = "old official source"
    previous["last_updated"] = "2026-06-12"
    refreshed = make_results()
    refreshed["source"] = "new official source"
    refreshed["last_updated"] = "2026-06-18"

    output = preserve_existing_result_metadata(previous, refreshed)

    assert output.loc[0, "source"] == "old official source"
    assert output.loc[0, "last_updated"] == "2026-06-12"


def test_preserve_existing_result_metadata_allows_score_corrections() -> None:
    previous = make_results()
    previous["source"] = "old official source"
    previous["last_updated"] = "2026-06-12"
    refreshed = make_results()
    refreshed.loc[0, "team_a_goals"] = 3
    refreshed["source"] = "new official source"
    refreshed["last_updated"] = "2026-06-18"

    output = preserve_existing_result_metadata(previous, refreshed)

    assert output.loc[0, "source"] == "new official source"
    assert output.loc[0, "last_updated"] == "2026-06-18"


def test_fetch_and_write_official_results_retains_previous_missing_rows(
    tmp_path,
    monkeypatch,
) -> None:
    results_path = tmp_path / "results_2026.csv"
    previous = make_results()
    for column in official_results.RESULT_COLUMNS:
        if column not in previous.columns:
            previous[column] = ""
    previous["went_to_extra_time"] = "false"
    previous["went_to_penalties"] = "false"
    previous["source"] = "previous official source"
    previous["last_updated"] = "2026-06-12"
    previous[official_results.RESULT_COLUMNS].to_csv(results_path, index=False)
    monkeypatch.setattr(
        official_results,
        "fetch_fifa_payload",
        lambda url: {"Results": []},
    )

    refreshed, previous_count, _, _ = fetch_and_write_official_results(
        make_fixtures(),
        results_path,
        from_date="2026-06-11",
        to_date="2026-06-12",
    )

    assert previous_count == 1
    assert len(refreshed) == 1
    assert refreshed.loc[0, "match_id"] == "m1"
    assert pd.read_csv(results_path).loc[0, "source"] == "previous official source"


def test_verified_result_generated_at_uses_latest_completed_result_date() -> None:
    results = pd.concat(
        [
            make_results(),
            make_results().assign(match_id="m2", match_date="2026-06-13"),
        ],
        ignore_index=True,
    )

    assert _verified_result_generated_at(results) == "2026-06-13T23:59:59+00:00"
