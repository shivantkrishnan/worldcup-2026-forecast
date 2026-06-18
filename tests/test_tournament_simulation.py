from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.simulate_group_stage import main as simulate_group_stage_main
from src.simulation.tournament import (
    VALID_OUTCOMES,
    sample_match_outcome,
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
        "advance_prob",
        "avg_points",
        "avg_group_rank",
    ]


def test_simulation_script_writes_no_files_by_default(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = simulate_group_stage_main()
    output = capsys.readouterr().out

    assert result == 0
    assert "No simulation files were written." in output
    assert list(tmp_path.iterdir()) == []
