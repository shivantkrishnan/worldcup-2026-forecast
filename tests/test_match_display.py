import pandas as pd

from src.presentation.match_display import (
    DISPLAY_STATUS_COMPLETED,
    DISPLAY_STATUS_PREDICTION_MISSING,
    DISPLAY_STATUS_SCHEDULED,
    build_match_display_table,
    build_prediction_audit_table,
)


def make_fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "match_date": "2026-06-12",
                "group": "A",
                "team_a": "Alpha",
                "team_b": "Beta",
            },
            {
                "match_id": "m2",
                "match_date": "2026-06-13",
                "group": "A",
                "team_a": "Gamma",
                "team_b": "Delta",
            },
            {
                "match_id": "m3",
                "match_date": "2026-06-14",
                "group": "B",
                "team_a": "Epsilon",
                "team_b": "Zeta",
            },
        ]
    )


def make_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "team_a": "Alpha",
                "team_b": "Beta",
                "p_team_a_win": 0.25,
                "p_draw": 0.25,
                "p_team_b_win": 0.50,
                "predicted_class": "team_b_win",
                "favorite_display": "Beta",
                "confidence_label": "Medium",
                "forecast_mode": "backfilled_ex_ante",
                "is_backfilled": "true",
            },
            {
                "match_id": "m2",
                "team_a": "Gamma",
                "team_b": "Delta",
                "p_team_a_win": 0.60,
                "p_draw": 0.20,
                "p_team_b_win": 0.20,
                "predicted_class": "team_a_win",
                "favorite_display": "Gamma",
                "confidence_label": "High",
                "forecast_mode": "pre_tournament",
                "is_backfilled": False,
            },
        ]
    )


def make_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "team_a": "Alpha",
                "team_b": "Beta",
                "team_a_goals": 2,
                "team_b_goals": 1,
                "result": "team_a_win",
                "status": "completed",
            }
        ]
    )


def test_completed_match_with_result_and_prediction_is_labeled_completed() -> None:
    display = build_match_display_table(
        make_fixtures(),
        predictions=make_predictions(),
        results=make_results(),
    )

    row = display.loc[display["match_id"].eq("m1")].iloc[0]

    assert row["display_status"] == DISPLAY_STATUS_COMPLETED
    assert row["team_a_goals"] == 2
    assert row["team_b_goals"] == 1
    assert row["actual_result"] == "team_a_win"


def test_completed_match_is_not_labeled_as_current_prediction() -> None:
    display = build_match_display_table(
        make_fixtures(),
        predictions=make_predictions(),
        results=make_results(),
    )

    row = display.loc[display["match_id"].eq("m1")].iloc[0]

    assert row["prediction_display_label"] != "Prediction"
    assert "Backfilled ex-ante" in row["prediction_display_label"]


def test_scheduled_match_with_prediction_is_labeled_scheduled() -> None:
    display = build_match_display_table(
        make_fixtures(),
        predictions=make_predictions(),
        results=make_results(),
    )

    row = display.loc[display["match_id"].eq("m2")].iloc[0]

    assert row["display_status"] == DISPLAY_STATUS_SCHEDULED
    assert row["prediction_display_label"] == "Prediction"
    assert row["favorite_display"] == "Gamma"


def test_missing_prediction_row_gets_prediction_missing_status() -> None:
    display = build_match_display_table(
        make_fixtures(),
        predictions=make_predictions(),
        results=make_results(),
    )

    row = display.loc[display["match_id"].eq("m3")].iloc[0]

    assert row["display_status"] == DISPLAY_STATUS_PREDICTION_MISSING
    assert row["prediction_display_label"] == "Prediction unavailable"


def test_display_table_handles_remaining_only_live_predictions() -> None:
    remaining_only_predictions = make_predictions().loc[
        make_predictions()["match_id"].eq("m2")
    ]

    display = build_match_display_table(
        make_fixtures(),
        predictions=remaining_only_predictions,
        results=make_results(),
    )

    completed = display.loc[display["match_id"].eq("m1")].iloc[0]
    scheduled = display.loc[display["match_id"].eq("m2")].iloc[0]
    missing = display.loc[display["match_id"].eq("m3")].iloc[0]
    assert completed["display_status"] == DISPLAY_STATUS_COMPLETED
    assert completed["actual_result"] == "team_a_win"
    assert completed["prediction_display_label"] == "Prediction unavailable"
    assert scheduled["display_status"] == DISPLAY_STATUS_SCHEDULED
    assert scheduled["prediction_display_label"] == "Prediction"
    assert missing["display_status"] == DISPLAY_STATUS_PREDICTION_MISSING


def test_actual_scores_from_results_do_not_overwrite_audit_probabilities() -> None:
    display = build_match_display_table(
        make_fixtures(),
        predictions=make_predictions(),
        results=make_results(),
    )

    row = display.loc[display["match_id"].eq("m1")].iloc[0]

    assert row["actual_result"] == "team_a_win"
    assert row["p_team_a_win"] == 0.25
    assert row["p_draw"] == 0.25
    assert row["p_team_b_win"] == 0.50
    assert bool(row["audit_available"]) is True


def test_prediction_audit_table_labels_completed_prediction_as_audit() -> None:
    display = build_match_display_table(
        make_fixtures(),
        predictions=make_predictions(),
        results=make_results(),
    )

    audit = build_prediction_audit_table(display)

    assert len(audit) == 1
    assert audit.loc[0, "match_id"] == "m1"
    assert audit.loc[0, "actual_outcome_probability"] == 0.25
    assert "Backfilled ex-ante" in audit.loc[0, "prediction_display_label"]


def test_inputs_are_not_mutated() -> None:
    fixtures = make_fixtures()
    predictions = make_predictions()
    results = make_results()
    original_fixtures = fixtures.copy(deep=True)
    original_predictions = predictions.copy(deep=True)
    original_results = results.copy(deep=True)

    build_match_display_table(fixtures, predictions=predictions, results=results)

    pd.testing.assert_frame_equal(fixtures, original_fixtures)
    pd.testing.assert_frame_equal(predictions, original_predictions)
    pd.testing.assert_frame_equal(results, original_results)
