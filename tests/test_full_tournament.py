from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from scripts.simulate_tournament import main as simulate_tournament_main
from src.simulation.full_tournament import (
    assign_round_of_32_bracket,
    build_prediction_strength_knockout_probability_table,
    knockout_advancement_probabilities,
    sample_knockout_winner,
    simulate_full_tournament,
)
from src.simulation.knockout_bracket import (
    assign_third_place_groups_to_slots,
    get_round_of_32_matches,
)


GROUPS = list("ABCDEFGHIJKL")


def synthetic_group_standings() -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        for rank in range(1, 5):
            rows.append(
                {
                    "team": f"{group}{rank}",
                    "group": group,
                    "group_rank": rank,
                    "advanced": rank <= 2 or (rank == 3 and group in set("ABCDEFGH")),
                    "best_third_place_advanced": rank == 3 and group in set("ABCDEFGH"),
                }
            )
    return pd.DataFrame(rows)


def synthetic_group_fixtures() -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        teams = [f"{group}{rank}" for rank in range(1, 5)]
        pairings = [
            (teams[0], teams[1]),
            (teams[0], teams[2]),
            (teams[0], teams[3]),
            (teams[1], teams[2]),
            (teams[1], teams[3]),
            (teams[2], teams[3]),
        ]
        for index, (team_a, team_b) in enumerate(pairings, start=1):
            rows.append(
                {
                    "match_id": f"{group.lower()}{index}",
                    "match_date": "2026-06-11",
                    "group": group,
                    "team_a": team_a,
                    "team_b": team_b,
                    "p_team_a_win": 1.0,
                    "p_draw": 0.0,
                    "p_team_b_win": 0.0,
                }
            )
    return pd.DataFrame(rows)


def knockout_probabilities_for_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    teams = sorted(pd.concat([fixtures["team_a"], fixtures["team_b"]]).unique())
    return pd.DataFrame(
        [
            {
                "team_a": team_a,
                "team_b": team_b,
                "p_team_a_win": 0.45,
                "p_draw": 0.10,
                "p_team_b_win": 0.45,
            }
            for team_a, team_b in permutations(teams, 2)
        ]
    )


def test_round_of_32_config_has_16_matches() -> None:
    assert len(get_round_of_32_matches()) == 16


def test_assign_round_of_32_bracket_has_32_unique_teams() -> None:
    bracket = assign_round_of_32_bracket(synthetic_group_standings())
    teams = pd.concat([bracket["team_a"], bracket["team_b"]], ignore_index=True)

    assert len(bracket) == 16
    assert len(teams) == 32
    assert teams.nunique() == 32


def test_assign_round_of_32_preserves_top_two_and_best_third_routes() -> None:
    bracket = assign_round_of_32_bracket(synthetic_group_standings())
    routes = pd.concat([bracket["team_a_route"], bracket["team_b_route"]])

    assert (routes == "group_winner").sum() == 12
    assert (routes == "runner_up").sum() == 12
    assert (routes == "best_third").sum() == 8


def test_unsupported_third_place_group_fails_clearly() -> None:
    with pytest.raises(ValueError, match="Unsupported third-place group"):
        assign_third_place_groups_to_slots(["A", "B", "C", "D", "E", "F", "G", "Z"])


def test_draw_mass_is_split_evenly_for_advancement() -> None:
    p_a, p_b = knockout_advancement_probabilities(
        p_team_a_win=0.30,
        p_draw=0.40,
        p_team_b_win=0.30,
    )

    assert p_a == pytest.approx(0.5)
    assert p_b == pytest.approx(0.5)


def test_knockout_match_always_advances_one_team() -> None:
    winner = sample_knockout_winner(
        "Alpha",
        "Beta",
        pd.Series({"p_team_a_win": 0.0, "p_draw": 1.0, "p_team_b_win": 0.0}),
        np.random.default_rng(42),
    )

    assert winner in {"Alpha", "Beta"}


def test_fallback_knockout_probability_table_sums_to_one() -> None:
    fixtures = synthetic_group_fixtures()
    probabilities = build_prediction_strength_knockout_probability_table(
        fixtures,
        fixtures,
    ).probabilities
    probability_sums = probabilities[
        ["p_team_a_win", "p_draw", "p_team_b_win"]
    ].sum(axis=1)

    assert np.isclose(probability_sums, 1.0, atol=1e-6).all()


def test_full_tournament_champion_probabilities_sum_to_one() -> None:
    fixtures = synthetic_group_fixtures()
    summary = simulate_full_tournament(
        fixtures,
        knockout_probabilities_for_fixtures(fixtures),
        n_simulations=10,
        random_seed=7,
    )

    assert summary["champion_prob"].sum() == pytest.approx(1.0)


def test_full_tournament_round_probabilities_are_monotonic() -> None:
    fixtures = synthetic_group_fixtures()
    summary = simulate_full_tournament(
        fixtures,
        knockout_probabilities_for_fixtures(fixtures),
        n_simulations=10,
        random_seed=8,
    )

    assert (
        summary["champion_prob"]
        <= summary["reach_final_prob"]
    ).all()
    assert (
        summary["reach_final_prob"]
        <= summary["reach_semifinal_prob"]
    ).all()
    assert (
        summary["reach_semifinal_prob"]
        <= summary["reach_quarterfinal_prob"]
    ).all()
    assert (
        summary["reach_quarterfinal_prob"]
        <= summary["reach_round_of_16_prob"]
    ).all()
    assert (
        summary["reach_round_of_16_prob"]
        <= summary["reach_round_of_32_prob"]
    ).all()


def test_simulate_tournament_cli_smoke_fallback(capsys) -> None:
    exit_code = simulate_tournament_main(
        ["--simulations", "2", "--knockout-probability-source", "fallback"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Full Tournament Monte Carlo Simulation" in captured.out
    assert "Top Champion Probabilities" in captured.out
