import pandas as pd

from src.presentation.dashboard import (
    build_current_group_table,
    format_percent,
    get_teams_from_fixtures,
    prepare_full_tournament_summary,
    prepare_match_table,
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
                "team_b": "Alpha",
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
                "team_a_goals": 2,
                "team_b_goals": 1,
                "result": "team_a_win",
                "status": "completed",
            }
        ]
    )


def make_display_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "match_date": "2026-06-12",
                "group": "A",
                "team_a": "Alpha",
                "team_b": "Beta",
                "display_status": "completed",
                "team_a_goals": 2,
                "team_b_goals": 1,
                "actual_result": "team_a_win",
                "p_team_a_win": 0.55,
                "p_draw": 0.25,
                "p_team_b_win": 0.20,
                "predicted_class": "team_a_win",
                "favorite_display": "Alpha",
                "confidence_label": "Medium",
                "forecast_mode": "backfilled_ex_ante",
                "is_backfilled": True,
                "prediction_display_label": "Backfilled ex-ante model probability",
                "audit_available": True,
            },
            {
                "match_id": "m2",
                "match_date": "2026-06-13",
                "group": "A",
                "team_a": "Gamma",
                "team_b": "Alpha",
                "display_status": "scheduled",
                "team_a_goals": pd.NA,
                "team_b_goals": pd.NA,
                "actual_result": pd.NA,
                "p_team_a_win": 0.40,
                "p_draw": 0.30,
                "p_team_b_win": 0.30,
                "predicted_class": "team_a_win",
                "favorite_display": "Gamma",
                "confidence_label": "Low",
                "forecast_mode": "live",
                "is_backfilled": False,
                "prediction_display_label": "Prediction",
                "audit_available": False,
            },
        ]
    )


def test_build_current_group_table_uses_completed_results_only() -> None:
    table = build_current_group_table(make_fixtures(), make_results())

    alpha = table.loc[table["team"].eq("Alpha")].iloc[0]
    beta = table.loc[table["team"].eq("Beta")].iloc[0]
    gamma = table.loc[table["team"].eq("Gamma")].iloc[0]

    assert alpha["points"] == 3
    assert alpha["goals_for"] == 2
    assert beta["points"] == 0
    assert gamma["played"] == 0
    assert list(table.loc[table["group"].eq("A"), "rank"]) == [1, 2, 3]


def test_prepare_match_table_hides_completed_probabilities_by_default() -> None:
    table = prepare_match_table(make_display_table(), show_audit_probabilities=False)

    completed = table.loc[table["match"].eq("Alpha vs Beta")].iloc[0]
    scheduled = table.loc[table["match"].eq("Gamma vs Alpha")].iloc[0]

    assert completed["status"] == "Completed"
    assert completed["score"] == "2-1"
    assert completed["Team A win"] == "-"
    assert scheduled["status"] == "Scheduled prediction"
    assert scheduled["Team A win"] == "40.0%"


def test_prepare_match_table_can_show_audit_probabilities() -> None:
    table = prepare_match_table(make_display_table(), show_audit_probabilities=True)

    completed = table.loc[table["match"].eq("Alpha vs Beta")].iloc[0]

    assert completed["Team A win"] == "55.0%"
    assert completed["model context"] == "Backfilled ex-ante model probability"


def test_format_percent_handles_missing_values() -> None:
    assert format_percent(0.1234) == "12.3%"
    assert format_percent(None) == "-"


def test_get_teams_from_fixtures_returns_sorted_unique_teams() -> None:
    assert get_teams_from_fixtures(make_fixtures()) == ["Alpha", "Beta", "Gamma"]


def test_prepare_full_tournament_summary_formats_path_probabilities() -> None:
    summary = pd.DataFrame(
        [
            {
                "team": "Alpha",
                "group": "A",
                "advance_from_group_prob": 0.8,
                "reach_round_of_16_prob": 0.55,
                "reach_quarterfinal_prob": 0.3,
                "reach_semifinal_prob": 0.2,
                "reach_final_prob": 0.1,
                "champion_prob": 0.04,
            }
        ]
    )

    table = prepare_full_tournament_summary(summary)

    assert table.loc[0, "Champion"] == "4.0%"
    assert table.loc[0, "Final"] == "10.0%"
    assert table.loc[0, "Advance"] == "80.0%"
