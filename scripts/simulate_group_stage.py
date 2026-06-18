"""Run group-stage Monte Carlo simulation without writing artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tournament_results import (  # noqa: E402
    load_tournament_results,
    merge_completed_results_with_fixtures_or_predictions,
)
from src.simulation.tournament import (  # noqa: E402
    simulate_group_stage,
    summarize_advancement_probabilities,
    validate_fixture_probability_table,
)
from src.utils.config import (  # noqa: E402
    FIXTURE_PREDICTIONS_2026_PATH,
    RESULTS_2026_PATH,
)

FIXTURE_PREDICTIONS_PATH = Path(FIXTURE_PREDICTIONS_2026_PATH)
RESULTS_PATH = Path(RESULTS_2026_PATH)
DEFAULT_SIMULATION_COUNT = 1000


def parse_args(argv: list[str] | None = None) -> object:
    """Parse command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run group-stage Monte Carlo simulation without writing files."
    )
    parser.add_argument(
        "--predictions",
        default=FIXTURE_PREDICTIONS_2026_PATH,
        help="Path to fixture_predictions_2026.csv.",
    )
    parser.add_argument(
        "--results",
        default=RESULTS_2026_PATH,
        help="Path to manually maintained results_2026.csv.",
    )
    parser.add_argument(
        "--ignore-results",
        action="store_true",
        help="Sample every fixture from probabilities, even if results_2026.csv exists.",
    )
    return parser.parse_args(argv)


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


def _load_fixture_predictions(
    predictions_path: str | Path = FIXTURE_PREDICTIONS_PATH,
) -> tuple[pd.DataFrame, str]:
    """Load local fixture predictions or return a synthetic fallback."""
    path = Path(predictions_path)
    if path.exists():
        return (
            validate_fixture_probability_table(pd.read_csv(path)),
            f"Loaded fixture predictions from {path}.",
        )
    return (
        _synthetic_fixture_predictions(),
        "No local fixture_predictions_2026.csv found; using a synthetic example.",
    )


def _coerce_bool(value: object) -> bool:
    """Coerce common CSV boolean values for prediction metadata summaries."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def summarize_prediction_metadata(fixtures: pd.DataFrame) -> dict[str, object]:
    """Return forecast-mode metadata for a fixture prediction table."""
    if "forecast_mode" in fixtures.columns:
        forecast_mode_counts = {
            str(mode): int(count)
            for mode, count in fixtures["forecast_mode"]
            .fillna("missing")
            .value_counts(sort=False)
            .items()
        }
    else:
        forecast_mode_counts = {"missing": int(len(fixtures))}

    if "is_backfilled" in fixtures.columns:
        backfilled = fixtures["is_backfilled"].map(_coerce_bool)
        backfilled_count = int(backfilled.sum())
    else:
        backfilled = pd.Series([False] * len(fixtures), index=fixtures.index)
        backfilled_count = 0

    if "is_completed" in fixtures.columns:
        completed = fixtures["is_completed"].map(_coerce_bool)
        fixed_completed_count = int(completed.sum())
    else:
        completed = pd.Series([False] * len(fixtures), index=fixtures.index)
        fixed_completed_count = 0

    sampled_backfilled = backfilled & ~completed

    return {
        "forecast_mode_counts": forecast_mode_counts,
        "backfilled_count": backfilled_count,
        "fixed_completed_count": fixed_completed_count,
        "sampled_backfilled_count": int(sampled_backfilled.sum()),
        "completed_rows_sampled_as_predictions": bool(sampled_backfilled.any()),
    }


def _format_counts(counts: dict[str, int]) -> str:
    """Format count dictionaries for readable console output."""
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def _condition_on_completed_results(
    fixtures: pd.DataFrame,
    results_path: str | Path = RESULTS_PATH,
    ignore_results: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Return fixtures with completed results fixed when a result file exists."""
    if ignore_results:
        return fixtures, "Completed results ignored by --ignore-results."

    path = Path(results_path)
    if not path.exists():
        return (
            fixtures,
            f"No results_2026.csv found at {path}; sampling all prediction rows.",
        )

    results = load_tournament_results(path, fixtures_or_predictions=fixtures)
    conditioned = merge_completed_results_with_fixtures_or_predictions(
        fixtures,
        results,
    )
    conditioned = validate_fixture_probability_table(conditioned)
    fixed_count = int(conditioned["is_completed"].map(_coerce_bool).sum())
    return (
        conditioned,
        f"Loaded completed results from {path}; fixed completed matches: {fixed_count}.",
    )


def main(argv: list[str] | None = None) -> int:
    """Run and print a group-stage simulation report."""
    args = parse_args(argv)
    try:
        fixtures, source_message = _load_fixture_predictions(args.predictions)
        fixtures, results_message = _condition_on_completed_results(
            fixtures,
            results_path=args.results,
            ignore_results=args.ignore_results,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to run group-stage simulation: {error}")
        return 1

    prediction_metadata = summarize_prediction_metadata(fixtures)
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
    print(results_message)
    print(f"simulation count: {DEFAULT_SIMULATION_COUNT:,}")
    print(
        "forecast_mode values: "
        + _format_counts(prediction_metadata["forecast_mode_counts"])
    )
    print(f"backfilled rows: {prediction_metadata['backfilled_count']}")
    print(f"fixed completed result rows: {prediction_metadata['fixed_completed_count']}")
    print(
        "backfilled rows still sampled: "
        f"{prediction_metadata['sampled_backfilled_count']}"
    )
    print(
        "completed matches sampled as predictions: "
        + (
            "yes"
            if prediction_metadata["completed_rows_sampled_as_predictions"]
            else "no"
        )
    )
    if prediction_metadata["completed_rows_sampled_as_predictions"]:
        print(
            "This simulation includes backfilled ex-ante predictions. It is "
            "not a true live simulation unless completed results are fixed "
            "from results_2026.csv."
        )

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
