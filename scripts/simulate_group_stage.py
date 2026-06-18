"""Run group-stage Monte Carlo simulation without writing artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.tournament import (  # noqa: E402
    simulate_group_stage,
    summarize_advancement_probabilities,
)

FIXTURE_PREDICTIONS_PATH = Path("data/tournament/fixture_predictions_2026.csv")
DEFAULT_SIMULATION_COUNT = 1000


def _synthetic_fixture_predictions() -> pd.DataFrame:
    """Return a small two-group probability table for script smoke runs."""
    return pd.DataFrame(
        [
            {
                "match_id": "a1",
                "group": "A",
                "team_a": "Atlas",
                "team_b": "Boreal",
                "p_team_a_win": 0.55,
                "p_draw": 0.25,
                "p_team_b_win": 0.20,
            },
            {
                "match_id": "a2",
                "group": "A",
                "team_a": "Atlas",
                "team_b": "Cygnus",
                "p_team_a_win": 0.50,
                "p_draw": 0.28,
                "p_team_b_win": 0.22,
            },
            {
                "match_id": "a3",
                "group": "A",
                "team_a": "Boreal",
                "team_b": "Cygnus",
                "p_team_a_win": 0.35,
                "p_draw": 0.30,
                "p_team_b_win": 0.35,
            },
            {
                "match_id": "b1",
                "group": "B",
                "team_a": "Dynamo",
                "team_b": "Equinox",
                "p_team_a_win": 0.45,
                "p_draw": 0.30,
                "p_team_b_win": 0.25,
            },
            {
                "match_id": "b2",
                "group": "B",
                "team_a": "Dynamo",
                "team_b": "Fjord",
                "p_team_a_win": 0.40,
                "p_draw": 0.30,
                "p_team_b_win": 0.30,
            },
            {
                "match_id": "b3",
                "group": "B",
                "team_a": "Equinox",
                "team_b": "Fjord",
                "p_team_a_win": 0.36,
                "p_draw": 0.31,
                "p_team_b_win": 0.33,
            },
        ]
    )


def _load_fixture_predictions() -> tuple[pd.DataFrame, str]:
    """Load local fixture predictions or return a synthetic fallback."""
    if FIXTURE_PREDICTIONS_PATH.exists():
        return (
            pd.read_csv(FIXTURE_PREDICTIONS_PATH),
            f"Loaded fixture predictions from {FIXTURE_PREDICTIONS_PATH}.",
        )
    return (
        _synthetic_fixture_predictions(),
        "No local fixture_predictions_2026.csv found; using a synthetic example.",
    )


def main() -> int:
    """Run and print a group-stage simulation report."""
    fixtures, source_message = _load_fixture_predictions()
    simulation_results = simulate_group_stage(
        fixtures,
        n_simulations=DEFAULT_SIMULATION_COUNT,
        random_seed=42,
        top_n_per_group=2,
    )
    summary = summarize_advancement_probabilities(simulation_results)

    print("Group-Stage Monte Carlo Simulation")
    print("==================================")
    print(source_message)
    print(f"simulation count: {DEFAULT_SIMULATION_COUNT:,}")

    print("\nTop Advancement Probabilities")
    print("=============================")
    print(
        summary.sort_values(
            ["advance_prob", "group_winner_prob", "team"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .head(12)
        .to_string(index=False)
    )

    print("\nGroup Winner Probabilities")
    print("==========================")
    print(
        summary[["team", "group", "group_winner_prob", "avg_points"]]
        .sort_values(
            ["group", "group_winner_prob", "team"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .to_string(index=False)
    )

    print("\nNote")
    print("====")
    print(
        "Scoreline, goal-difference, and official tie-break rules are not "
        "implemented yet; ties currently use points, wins, then a seeded random "
        "tie-break placeholder."
    )
    print("\nNo simulation files were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
