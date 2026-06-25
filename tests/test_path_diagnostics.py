import pandas as pd
import pytest

from src.simulation.path_diagnostics import (
    compare_top_contenders,
    head_to_head_probability_table,
    matchup_source_label,
    most_likely_opponents,
    path_difficulty_summary,
    summarize_team_path,
)


def make_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": "Argentina",
                "group": "J",
                "champion_prob": 0.50,
                "reach_final_prob": 1.00,
                "reach_semifinal_prob": 1.00,
                "reach_quarterfinal_prob": 1.00,
                "group_winner_prob": 1.00,
            },
            {
                "team": "France",
                "group": "I",
                "champion_prob": 0.25,
                "reach_final_prob": 1.00,
                "reach_semifinal_prob": 1.00,
                "reach_quarterfinal_prob": 1.00,
                "group_winner_prob": 0.50,
            },
            {
                "team": "Spain",
                "group": "H",
                "champion_prob": 0.10,
                "reach_final_prob": 0.00,
                "reach_semifinal_prob": 0.50,
                "reach_quarterfinal_prob": 1.00,
                "group_winner_prob": 0.50,
            },
            {
                "team": "Brazil",
                "group": "C",
                "champion_prob": 0.08,
                "reach_final_prob": 0.00,
                "reach_semifinal_prob": 0.00,
                "reach_quarterfinal_prob": 1.00,
                "group_winner_prob": 0.50,
            },
            {
                "team": "England",
                "group": "L",
                "champion_prob": 0.07,
                "reach_final_prob": 0.00,
                "reach_semifinal_prob": 0.00,
                "reach_quarterfinal_prob": 0.50,
                "group_winner_prob": 0.50,
            },
            {
                "team": "Mexico",
                "group": "A",
                "champion_prob": 0.00,
                "reach_final_prob": 0.00,
                "reach_semifinal_prob": 0.00,
                "reach_quarterfinal_prob": 0.00,
                "group_winner_prob": 0.00,
            },
        ]
    )


def make_trace_row(
    simulation_id: int,
    team: str,
    champion: bool,
    r16: bool = True,
    qf: bool = True,
    sf: bool = True,
    final: bool = True,
    r32_opponent: str = "Mexico",
) -> dict[str, object]:
    row: dict[str, object] = {
        "simulation_id": simulation_id,
        "team": team,
        "group": "J" if team == "Argentina" else "I",
        "final_group_position": 1,
        "group_winner": team == "Argentina",
        "top_two": True,
        "best_third_place_advanced": False,
        "advance_from_group": True,
        "reached_round_of_32": True,
        "reached_round_of_16": r16,
        "reached_quarterfinal": qf,
        "reached_semifinal": sf,
        "reached_final": final,
        "won_tournament": champion,
    }
    rounds = {
        "round_of_32": (r32_opponent, 0.70, r16),
        "round_of_16": ("Netherlands", 0.62, qf),
        "quarterfinal": ("Brazil", 0.58, sf),
        "semifinal": ("Spain", 0.55, final),
        "final": ("France" if team == "Argentina" else "Argentina", 0.52, champion),
    }
    for prefix, (opponent, advance_prob, advanced) in rounds.items():
        row[f"{prefix}_opponent"] = opponent
        row[f"{prefix}_team_advance_prob"] = advance_prob
        row[f"{prefix}_opponent_advance_prob"] = 1.0 - advance_prob
        row[f"{prefix}_advanced"] = advanced
    return row


def make_traces() -> pd.DataFrame:
    rows = [
        make_trace_row(1, "Argentina", champion=True),
        make_trace_row(2, "Argentina", champion=False, r32_opponent="England"),
        make_trace_row(1, "France", champion=False, r32_opponent="Brazil"),
        make_trace_row(2, "France", champion=True, r32_opponent="Spain"),
        make_trace_row(1, "Spain", champion=False, final=False, r32_opponent="Mexico"),
        make_trace_row(2, "Spain", champion=False, sf=False, final=False),
        make_trace_row(1, "Brazil", champion=False, sf=False, final=False),
        make_trace_row(2, "Brazil", champion=False, sf=False, final=False),
        make_trace_row(1, "England", champion=False, qf=False, sf=False, final=False),
        make_trace_row(2, "England", champion=False, qf=False, sf=False, final=False),
    ]
    return pd.DataFrame(rows)


def test_selected_team_decomposition_and_transitions_are_valid() -> None:
    team_path = summarize_team_path(make_traces(), "Argentina")

    assert team_path["available"] is True
    assert team_path["group_advancement_probability"] == pytest.approx(1.0)
    assert team_path["champion_probability"] == pytest.approx(0.5)

    reach_probabilities = [
        team_path["reach_round_of_32_probability"],
        team_path["reach_round_of_16_probability"],
        team_path["reach_quarterfinal_probability"],
        team_path["reach_semifinal_probability"],
        team_path["reach_final_probability"],
        team_path["champion_probability"],
    ]
    assert reach_probabilities == sorted(reach_probabilities, reverse=True)
    for key, value in team_path.items():
        if key.startswith("p_") and not pd.isna(value):
            assert 0.0 <= float(value) <= 1.0


def test_opponent_frequencies_sum_conditional_on_reaching_round() -> None:
    opponents = most_likely_opponents(make_traces(), "Argentina")

    frequency_sums = opponents.groupby("round")["opponent_frequency"].sum()

    assert all(value == pytest.approx(1.0) for value in frequency_sums)
    assert {"Mexico", "England"}.issubset(set(opponents["opponent"]))


def test_path_difficulty_summary_uses_opponent_strength_proxy() -> None:
    difficulty = path_difficulty_summary(make_traces(), make_summary(), "Argentina")

    assert difficulty["available"] is True
    assert difficulty["average_model_implied_advancement_probability"] == pytest.approx(
        0.594
    )
    assert difficulty["expected_elite_opponents_faced"] > 0


def test_compare_top_contenders_includes_requested_teams_when_present() -> None:
    comparison = compare_top_contenders(make_traces(), make_summary())

    assert {"Argentina", "France", "Spain", "Brazil", "England"} == set(
        comparison["team"]
    )
    assert "largest_likely_path_bottleneck" in comparison.columns


def test_missing_traces_are_handled_gracefully() -> None:
    empty = pd.DataFrame()

    assert summarize_team_path(empty, "Argentina")["available"] is False
    assert most_likely_opponents(empty, "Argentina").empty
    assert path_difficulty_summary(empty, make_summary(), "Argentina")[
        "available"
    ] is False
    assert compare_top_contenders(empty, pd.DataFrame()).empty


def test_head_to_head_table_includes_deploy_fallback_label() -> None:
    opponents = pd.DataFrame({"opponent": ["Mexico"]})
    probabilities = pd.DataFrame(
        [
            {
                "team_a": "Argentina",
                "team_b": "Mexico",
                "p_team_a_win": 0.60,
                "p_draw": 0.20,
                "p_team_b_win": 0.20,
            }
        ]
    )

    h2h = head_to_head_probability_table(
        "Argentina",
        opponents,
        probabilities,
        source_label="snapshot_strength_fallback",
    )

    assert h2h.loc[0, "p_selected_team_advances"] == pytest.approx(0.70)
    assert h2h.loc[0, "probability_source"] == "Approximate deploy-safe matchup estimate"
    assert matchup_source_label("selected_model_neutral_knockout") == (
        "Selected model neutral matchup estimate"
    )
