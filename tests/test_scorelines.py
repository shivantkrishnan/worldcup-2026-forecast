import numpy as np
import pandas as pd

from src.simulation.scorelines import (
    build_empirical_scoreline_distributions,
    sample_result_class,
    sample_scoreline_from_probabilities,
    sample_scoreline_given_result,
)


def test_sampled_class_respects_probability_support() -> None:
    row = pd.Series({"p_team_a_win": 0.0, "p_draw": 1.0, "p_team_b_win": 0.0})
    rng = np.random.default_rng(42)

    assert {sample_result_class(row, rng) for _ in range(20)} == {"draw"}


def test_sampled_scoreline_is_consistent_with_class() -> None:
    rng = np.random.default_rng(42)

    team_a_goals, team_b_goals = sample_scoreline_given_result("team_a_win", rng)
    assert team_a_goals > team_b_goals

    team_a_goals, team_b_goals = sample_scoreline_given_result("draw", rng)
    assert team_a_goals == team_b_goals

    team_a_goals, team_b_goals = sample_scoreline_given_result("team_b_win", rng)
    assert team_a_goals < team_b_goals


def test_fallback_scoreline_distribution_works() -> None:
    row = pd.Series({"p_team_a_win": 1.0, "p_draw": 0.0, "p_team_b_win": 0.0})
    rng = np.random.default_rng(7)

    result_class, team_a_goals, team_b_goals = sample_scoreline_from_probabilities(
        row,
        rng,
    )

    assert result_class == "team_a_win"
    assert team_a_goals > team_b_goals


def test_empirical_distribution_uses_synthetic_historical_scores() -> None:
    completed = pd.DataFrame(
        [
            {"team_a_goals": 2, "team_b_goals": 1, "result": "team_a_win"},
            {"team_a_goals": 2, "team_b_goals": 1, "result": "team_a_win"},
            {"team_a_goals": 0, "team_b_goals": 0, "result": "draw"},
            {"team_a_goals": 1, "team_b_goals": 3, "result": "team_b_win"},
        ]
    )

    distributions = build_empirical_scoreline_distributions(completed)

    assert set(distributions) == {"team_a_win", "draw", "team_b_win"}
    team_a_win = distributions["team_a_win"]
    assert team_a_win.loc[0, "team_a_goals"] == 2
    assert team_a_win.loc[0, "team_b_goals"] == 1
    assert team_a_win.loc[0, "weight"] == 2


def test_seeded_scoreline_sampling_is_reproducible() -> None:
    row = pd.Series({"p_team_a_win": 0.4, "p_draw": 0.3, "p_team_b_win": 0.3})

    first_rng = np.random.default_rng(123)
    second_rng = np.random.default_rng(123)

    first = [sample_scoreline_from_probabilities(row, first_rng) for _ in range(10)]
    second = [sample_scoreline_from_probabilities(row, second_rng) for _ in range(10)]

    assert first == second
