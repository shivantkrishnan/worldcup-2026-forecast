from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.simulate_group_stage import main as simulate_group_stage_main
from src.simulation.tournament import (
    VALID_OUTCOMES,
    rank_group_table,
    rank_third_place_teams,
    sample_match_outcome,
    simulate_fixture_results_once,
    simulate_group_stage,
    simulate_group_stage_once,
    summarize_advancement_probabilities,
    validate_fixture_probability_table,
)


def one_match_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "group": "A",
                "team_a": "Alpha",
                "team_b": "Beta",
                "p_team_a_win": 1.0,
                "p_draw": 0.0,
                "p_team_b_win": 0.0,
            }
        ]
    )


def deterministic_two_group_fixtures() -> pd.DataFrame:
    rows = [
        ("a1", "A", "A1", "A2", "team_a_win"),
        ("a2", "A", "A1", "A3", "team_a_win"),
        ("a3", "A", "A1", "A4", "team_a_win"),
        ("a4", "A", "A2", "A3", "team_a_win"),
        ("a5", "A", "A2", "A4", "team_a_win"),
        ("a6", "A", "A3", "A4", "team_a_win"),
        ("b1", "B", "B1", "B2", "team_a_win"),
        ("b2", "B", "B1", "B3", "team_a_win"),
        ("b3", "B", "B1", "B4", "team_a_win"),
        ("b4", "B", "B2", "B3", "team_a_win"),
        ("b5", "B", "B2", "B4", "team_a_win"),
        ("b6", "B", "B3", "B4", "draw"),
    ]
    probability_map = {
        "team_a_win": (1.0, 0.0, 0.0),
        "draw": (0.0, 1.0, 0.0),
        "team_b_win": (0.0, 0.0, 1.0),
    }
    return pd.DataFrame(
        [
            {
                "match_id": match_id,
                "group": group,
                "team_a": team_a,
                "team_b": team_b,
                "p_team_a_win": probability_map[outcome][0],
                "p_draw": probability_map[outcome][1],
                "p_team_b_win": probability_map[outcome][2],
            }
            for match_id, group, team_a, team_b, outcome in rows
        ]
    )


def test_probabilities_must_sum_to_one() -> None:
    fixtures = one_match_fixture()
    fixtures.loc[0, "p_draw"] = 0.5

    with pytest.raises(ValueError, match="sum to 1"):
        validate_fixture_probability_table(fixtures)


def test_negative_probabilities_raise_error() -> None:
    fixtures = one_match_fixture()
    fixtures.loc[0, "p_draw"] = -0.1
    fixtures.loc[0, "p_team_a_win"] = 1.1

    with pytest.raises(ValueError, match="nonnegative"):
        validate_fixture_probability_table(fixtures)


def test_duplicate_match_id_raises_error() -> None:
    fixtures = pd.concat([one_match_fixture(), one_match_fixture()], ignore_index=True)

    with pytest.raises(ValueError, match="unique"):
        validate_fixture_probability_table(fixtures)


def test_one_match_simulation_samples_valid_outcome() -> None:
    rng = np.random.default_rng(42)
    outcome = sample_match_outcome(one_match_fixture().iloc[0], rng)

    assert outcome in VALID_OUTCOMES


def test_group_stage_simulation_awards_points_correctly() -> None:
    rng = np.random.default_rng(42)

    result = simulate_group_stage_once(one_match_fixture(), rng)

    alpha = result.loc[result["team"].eq("Alpha")].iloc[0]
    beta = result.loc[result["team"].eq("Beta")].iloc[0]
    assert alpha["points"] == 3
    assert alpha["wins"] == 1
    assert beta["points"] == 0
    assert beta["losses"] == 1


def test_completed_match_is_fixed_and_not_sampled() -> None:
    fixtures = one_match_fixture()
    fixtures["is_completed"] = True
    fixtures["actual_result"] = "team_b_win"
    fixtures["team_a_goals"] = 1
    fixtures["team_b_goals"] = 2
    rng = np.random.default_rng(42)

    result = simulate_group_stage_once(fixtures, rng)

    alpha = result.loc[result["team"].eq("Alpha")].iloc[0]
    beta = result.loc[result["team"].eq("Beta")].iloc[0]
    assert alpha["points"] == 0
    assert alpha["losses"] == 1
    assert alpha["goals_for"] == 1
    assert alpha["goals_against"] == 2
    assert beta["points"] == 3
    assert beta["wins"] == 1
    assert beta["goals_for"] == 2
    assert beta["goals_against"] == 1


def test_fixture_result_rows_include_fixed_and_sampled_scores() -> None:
    fixtures = pd.concat(
        [
            one_match_fixture().assign(
                is_completed=True,
                actual_result="team_b_win",
                team_a_goals=1,
                team_b_goals=2,
            ),
            pd.DataFrame(
                [
                    {
                        "match_id": "m2",
                        "group": "A",
                        "team_a": "Gamma",
                        "team_b": "Delta",
                        "p_team_a_win": 0.0,
                        "p_draw": 1.0,
                        "p_team_b_win": 0.0,
                        "is_completed": False,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    match_results = simulate_fixture_results_once(fixtures, np.random.default_rng(1))

    fixed = match_results.loc[match_results["match_id"].eq("m1")].iloc[0]
    sampled = match_results.loc[match_results["match_id"].eq("m2")].iloc[0]
    assert bool(fixed["is_fixed_result"]) is True
    assert fixed["sampled_result"] == "team_b_win"
    assert fixed["team_a_goals"] == 1
    assert fixed["team_b_goals"] == 2
    assert bool(sampled["is_fixed_result"]) is False
    assert sampled["sampled_result"] == "draw"
    assert sampled["team_a_goals"] == sampled["team_b_goals"]


def test_uncompleted_matches_are_still_sampled() -> None:
    fixtures = pd.concat(
        [
            one_match_fixture().assign(
                is_completed=True,
                actual_result="team_b_win",
                team_a_goals=0,
                team_b_goals=1,
            ),
            pd.DataFrame(
                [
                    {
                        "match_id": "m2",
                        "group": "A",
                        "team_a": "Gamma",
                        "team_b": "Delta",
                        "p_team_a_win": 1.0,
                        "p_draw": 0.0,
                        "p_team_b_win": 0.0,
                        "is_completed": False,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    rng = np.random.default_rng(42)

    result = simulate_group_stage_once(fixtures, rng)

    beta = result.loc[result["team"].eq("Beta")].iloc[0]
    gamma = result.loc[result["team"].eq("Gamma")].iloc[0]
    assert beta["points"] == 3
    assert gamma["points"] == 3


def test_seeded_simulations_are_reproducible() -> None:
    fixtures = deterministic_two_group_fixtures()

    first = simulate_group_stage(fixtures, n_simulations=25, random_seed=7)
    second = simulate_group_stage(fixtures, n_simulations=25, random_seed=7)

    pd.testing.assert_frame_equal(first, second)


def test_top_n_per_group_advancement_works() -> None:
    result = simulate_group_stage(
        deterministic_two_group_fixtures(),
        n_simulations=1,
        random_seed=42,
        top_n_per_group=1,
        include_best_third_place=False,
    )

    advanced = result.loc[result["advanced"]]

    assert set(advanced["team"]) == {"A1", "B1"}


def test_include_best_third_place_option_works() -> None:
    result = simulate_group_stage(
        deterministic_two_group_fixtures(),
        n_simulations=1,
        random_seed=42,
        top_n_per_group=2,
        include_best_third_place=True,
        n_best_third_place=1,
    )

    advanced = result.loc[result["advanced"]]

    assert len(advanced) == 5
    assert "A3" in set(advanced["team"])


def test_output_summary_has_required_columns() -> None:
    results = simulate_group_stage(
        deterministic_two_group_fixtures(),
        n_simulations=10,
        random_seed=42,
    )

    summary = summarize_advancement_probabilities(results)

    assert list(summary.columns) == [
        "team",
        "group",
        "simulations",
        "group_winner_prob",
        "top_2_prob",
        "third_place_prob",
        "best_third_place_advance_prob",
        "advance_prob",
        "avg_points",
        "avg_goals_for",
        "avg_goals_against",
        "avg_goal_difference",
        "avg_group_rank",
    ]


def test_ranking_uses_points_first() -> None:
    table = pd.DataFrame(
        [
            {"group": "A", "team": "Alpha", "points": 6, "goals_for": 2, "goal_difference": 1},
            {"group": "A", "team": "Beta", "points": 3, "goals_for": 4, "goal_difference": 3},
        ]
    )

    ranked = rank_group_table(table, rng=np.random.default_rng(1))

    assert list(ranked["team"]) == ["Alpha", "Beta"]


def test_ranking_uses_goal_difference_tie_break() -> None:
    table = pd.DataFrame(
        [
            {"group": "A", "team": "Alpha", "points": 4, "goals_for": 2, "goal_difference": 2},
            {"group": "A", "team": "Beta", "points": 4, "goals_for": 5, "goal_difference": 1},
        ]
    )

    ranked = rank_group_table(table, rng=np.random.default_rng(1))

    assert list(ranked["team"]) == ["Alpha", "Beta"]


def test_ranking_uses_goals_scored_tie_break() -> None:
    table = pd.DataFrame(
        [
            {"group": "A", "team": "Alpha", "points": 4, "goals_for": 5, "goal_difference": 1},
            {"group": "A", "team": "Beta", "points": 4, "goals_for": 3, "goal_difference": 1},
        ]
    )

    ranked = rank_group_table(table, rng=np.random.default_rng(1))

    assert list(ranked["team"]) == ["Alpha", "Beta"]


def test_ranking_uses_simple_head_to_head_tie_break() -> None:
    table = pd.DataFrame(
        [
            {"group": "A", "team": "Alpha", "points": 4, "goals_for": 3, "goal_difference": 1},
            {"group": "A", "team": "Beta", "points": 4, "goals_for": 3, "goal_difference": 1},
        ]
    )
    group_matches = pd.DataFrame(
        [
            {
                "group": "A",
                "team_a": "Alpha",
                "team_b": "Beta",
                "team_a_goals": 1,
                "team_b_goals": 0,
            }
        ]
    )

    ranked = rank_group_table(
        table,
        rng=np.random.default_rng(1),
        group_matches=group_matches,
    )

    assert list(ranked["team"]) == ["Alpha", "Beta"]


def test_third_place_ranking_uses_points_goal_difference_goals_for() -> None:
    third_place_teams = pd.DataFrame(
        [
            {"group": "A", "team": "Alpha", "points": 4, "goal_difference": 1, "goals_for": 2},
            {"group": "B", "team": "Beta", "points": 4, "goal_difference": 2, "goals_for": 1},
            {"group": "C", "team": "Gamma", "points": 4, "goal_difference": 2, "goals_for": 3},
        ]
    )

    ranked = rank_third_place_teams(third_place_teams, rng=np.random.default_rng(1))

    assert list(ranked["team"]) == ["Gamma", "Beta", "Alpha"]


def test_eight_best_third_place_teams_advance_in_twelve_group_scenario() -> None:
    rows = []
    for group_index in range(12):
        group = chr(ord("A") + group_index)
        teams = [f"{group}{team_index}" for team_index in range(1, 5)]
        outcomes = [
            (teams[0], teams[1], "team_a_win"),
            (teams[0], teams[2], "team_a_win"),
            (teams[0], teams[3], "team_a_win"),
            (teams[1], teams[2], "team_a_win"),
            (teams[1], teams[3], "team_a_win"),
            (teams[2], teams[3], "team_a_win"),
        ]
        for match_index, (team_a, team_b, outcome) in enumerate(outcomes):
            probabilities = {
                "team_a_win": (1.0, 0.0, 0.0),
                "draw": (0.0, 1.0, 0.0),
                "team_b_win": (0.0, 0.0, 1.0),
            }[outcome]
            rows.append(
                {
                    "match_id": f"{group}_{match_index}",
                    "group": group,
                    "team_a": team_a,
                    "team_b": team_b,
                    "p_team_a_win": probabilities[0],
                    "p_draw": probabilities[1],
                    "p_team_b_win": probabilities[2],
                }
            )

    result = simulate_group_stage(
        pd.DataFrame(rows),
        n_simulations=1,
        random_seed=42,
        top_n_per_group=2,
        include_best_third_place=True,
        n_best_third_place=8,
    )

    assert int(result["best_third_place_advanced"].sum()) == 8
    assert int(result["advanced"].sum()) == 32


def test_simulation_script_writes_no_files_by_default(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = simulate_group_stage_main([])
    output = capsys.readouterr().out

    assert result == 0
    assert "No simulation files were written." in output
    assert list(tmp_path.iterdir()) == []


def test_simulation_script_reports_backfilled_prediction_metadata(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    prediction_path = tmp_path / "data" / "tournament" / "fixture_predictions_2026.csv"
    prediction_path.parent.mkdir(parents=True)
    predictions = one_match_fixture()
    predictions["forecast_mode"] = "backfilled_ex_ante"
    predictions["is_backfilled"] = True
    predictions.to_csv(prediction_path, index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scripts.simulate_group_stage.DEFAULT_SIMULATION_COUNT", 5)

    result = simulate_group_stage_main([])
    output = capsys.readouterr().out

    assert result == 0
    assert "No results_2026.csv found" in output
    assert "forecast_mode values: backfilled_ex_ante=1" in output
    assert "backfilled rows: 1" in output
    assert "fixed completed result rows: 0" in output
    assert "completed matches sampled as predictions: yes" in output
    assert "This simulation includes backfilled ex-ante predictions" in output


def test_simulation_script_conditions_on_completed_results(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    prediction_path = tmp_path / "data" / "tournament" / "fixture_predictions_2026.csv"
    result_path = tmp_path / "data" / "tournament" / "results_2026.csv"
    prediction_path.parent.mkdir(parents=True)
    predictions = one_match_fixture()
    predictions["forecast_mode"] = "backfilled_ex_ante"
    predictions["is_backfilled"] = True
    predictions.to_csv(prediction_path, index=False)
    pd.DataFrame(
        [
            {
                "match_id": "m1",
                "match_date": "2026-06-12",
                "team_a": "Alpha",
                "team_b": "Beta",
                "team_a_goals": 0,
                "team_b_goals": 1,
                "result": "team_b_win",
                "status": "completed",
            }
        ]
    ).to_csv(result_path, index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scripts.simulate_group_stage.DEFAULT_SIMULATION_COUNT", 5)

    result = simulate_group_stage_main([])
    output = capsys.readouterr().out

    assert result == 0
    assert "Loaded completed results" in output
    assert "fixed completed result rows: 1" in output
    assert "backfilled rows still sampled: 0" in output
    assert "completed matches sampled as predictions: no" in output
