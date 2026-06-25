"""Run full World Cup tournament simulation without writing artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.simulate_group_stage import (  # noqa: E402
    _coerce_bool,
    _format_counts,
    _load_scoreline_distributions,
    prepare_simulation_fixture_table,
    summarize_prediction_metadata,
)
from src.data.pipeline import load_baseline_training_matches  # noqa: E402
from src.data.tournament_fixtures import load_tournament_fixtures  # noqa: E402
from src.data.tournament_results import load_tournament_results  # noqa: E402
from src.simulation.full_tournament import (  # noqa: E402
    build_model_based_knockout_probability_table,
    build_prediction_strength_knockout_probability_table,
    simulate_full_tournament,
)
from src.simulation.knockout_bracket import BRACKET_SOURCE_NOTE  # noqa: E402
from src.simulation.path_diagnostics import (  # noqa: E402
    compare_top_contenders,
    most_likely_opponents,
    path_difficulty_summary,
    summarize_team_path,
)
from src.simulation.tournament import validate_fixture_probability_table  # noqa: E402
from src.utils.config import (  # noqa: E402
    FIXTURE_PREDICTIONS_2026_PATH,
    FIXTURES_2026_PATH,
    RESULTS_2026_PATH,
)

LIVE_PREDICTIONS_PATH = "data/tournament/fixture_predictions_2026_live.csv"
DEFAULT_SIMULATION_COUNT = 1000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run full-tournament Monte Carlo simulation without writing files."
    )
    parser.add_argument(
        "--fixtures",
        default=FIXTURES_2026_PATH,
        help="Path to fixtures_2026.csv.",
    )
    parser.add_argument(
        "--results",
        default=RESULTS_2026_PATH,
        help="Path to results_2026.csv.",
    )
    parser.add_argument(
        "--predictions",
        default=LIVE_PREDICTIONS_PATH,
        help="Path to live fixture prediction CSV.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=DEFAULT_SIMULATION_COUNT,
        help="Number of Monte Carlo simulations.",
    )
    parser.add_argument(
        "--knockout-probability-source",
        choices=["auto", "model", "fallback"],
        default="auto",
        help=(
            "Use selected model for arbitrary knockout matchups, use committed "
            "snapshot-strength fallback, or try model and fall back if raw data "
            "is unavailable."
        ),
    )
    return parser.parse_args(argv)


def _feature_cutoff_date(predictions: pd.DataFrame, results: pd.DataFrame) -> str | None:
    """Return the feature cutoff date used for knockout feature generation."""
    if "feature_cutoff_date" in predictions.columns and not predictions.empty:
        values = pd.to_datetime(
            predictions["feature_cutoff_date"],
            errors="coerce",
        ).dropna()
        if not values.empty:
            return str(values.max().date())
    if results is not None and not results.empty:
        return str(pd.to_datetime(results["match_date"], errors="raise").max().date())
    return None


def _build_knockout_probability_source(
    fixtures: pd.DataFrame,
    predictions: pd.DataFrame,
    results: pd.DataFrame,
    source_mode: str,
):
    """Build knockout probabilities from the selected source mode."""
    if source_mode in {"auto", "model"}:
        try:
            training_matches = load_baseline_training_matches()
            return build_model_based_knockout_probability_table(
                training_matches,
                fixtures,
                completed_results=results,
                feature_cutoff_date=_feature_cutoff_date(predictions, results),
            )
        except FileNotFoundError:
            if source_mode == "model":
                raise

    return build_prediction_strength_knockout_probability_table(
        fixtures,
        predictions,
        results=results,
    )


def _print_top_table(
    title: str,
    summary: pd.DataFrame,
    probability_column: str,
    n: int = 12,
) -> None:
    """Print a compact sorted probability table."""
    print(f"\n{title}")
    print("=" * len(title))
    output = summary.sort_values(
        [probability_column, "team"],
        ascending=[False, True],
        kind="mergesort",
    ).head(n)[["team", "group", probability_column]]
    print(output.to_string(index=False))


def _format_probability(value: object) -> str:
    """Format a probability for console output."""
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1%}"


def _print_contender_comparison(comparison: pd.DataFrame) -> None:
    """Print compact contender path diagnostics."""
    if comparison.empty:
        return
    print("\nTop Contender Path Diagnostics")
    print("==============================")
    display = comparison.copy(deep=True)
    probability_columns = [
        "champion_probability",
        "final_probability",
        "semifinal_probability",
        "quarterfinal_probability",
        "group_winner_probability",
        "average_r32_opponent_difficulty",
        "average_r16_opponent_difficulty",
        "average_qf_opponent_difficulty",
        "average_sf_opponent_difficulty",
        "average_final_opponent_difficulty",
    ]
    for column in probability_columns:
        if column in display:
            display[column] = display[column].map(_format_probability)
    print(display.to_string(index=False))


def _print_selected_team_diagnostic(
    team: str,
    summary: pd.DataFrame,
    traces: pd.DataFrame,
) -> None:
    """Print one selected team's path funnel and likely opponents."""
    team_path = summarize_team_path(traces, team)
    if not team_path.get("available"):
        return
    difficulty = path_difficulty_summary(traces, summary, team)
    opponents = most_likely_opponents(traces, team).head(10)

    print(f"\n{team} Path Decomposition")
    print("=" * (len(team) + 19))
    print(f"group winner probability: {_format_probability(team_path['group_winner_probability'])}")
    print(f"reach Round of 16: {_format_probability(team_path['reach_round_of_16_probability'])}")
    print(f"reach quarterfinal: {_format_probability(team_path['reach_quarterfinal_probability'])}")
    print(f"reach semifinal: {_format_probability(team_path['reach_semifinal_probability'])}")
    print(f"reach final: {_format_probability(team_path['reach_final_probability'])}")
    print(f"champion: {_format_probability(team_path['champion_probability'])}")
    if difficulty.get("available"):
        print(
            "average knockout advancement probability faced: "
            + _format_probability(
                difficulty["average_model_implied_advancement_probability"]
            )
        )
        print(
            "expected elite opponents faced: "
            + f"{difficulty['expected_elite_opponents_faced']:.2f}"
        )
    if not opponents.empty:
        opponent_view = opponents[
            [
                "round",
                "opponent",
                "opponent_frequency",
                "avg_team_advance_prob",
                "simulated_team_advance_rate",
            ]
        ].copy()
        for column in [
            "opponent_frequency",
            "avg_team_advance_prob",
            "simulated_team_advance_rate",
        ]:
            opponent_view[column] = opponent_view[column].map(_format_probability)
        print("\nMost likely opponents")
        print(opponent_view.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    """Run and print a full-tournament simulation report."""
    args = parse_args(argv)
    try:
        fixtures = load_tournament_fixtures(args.fixtures)
        results = load_tournament_results(args.results, fixtures_or_predictions=fixtures)
        predictions = validate_fixture_probability_table(pd.read_csv(args.predictions))
        simulation_fixtures = prepare_simulation_fixture_table(
            fixtures,
            predictions,
            results,
        )
        knockout_source = _build_knockout_probability_source(
            fixtures,
            predictions,
            results,
            args.knockout_probability_source,
        )
        scoreline_distributions, scoreline_message = _load_scoreline_distributions()
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to run full-tournament simulation: {error}")
        return 1

    prediction_metadata = summarize_prediction_metadata(simulation_fixtures)
    completed = (
        simulation_fixtures["is_completed"].map(_coerce_bool)
        if "is_completed" in simulation_fixtures.columns
        else pd.Series([False] * len(simulation_fixtures), index=simulation_fixtures.index)
    )
    fixed_completed_count = int(completed.sum())
    sampled_remaining_count = int((~completed).sum())

    simulation_output = simulate_full_tournament(
        simulation_fixtures,
        knockout_source.probabilities,
        n_simulations=args.simulations,
        random_seed=42,
        scoreline_distributions=scoreline_distributions,
        collect_traces=True,
    )
    summary = simulation_output.summary
    traces = simulation_output.traces

    print("Full Tournament Monte Carlo Simulation")
    print("======================================")
    print(f"fixture file path: {args.fixtures}")
    print(f"results file path: {args.results}")
    print(f"prediction file path: {args.predictions}")
    print(f"simulation count: {args.simulations:,}")
    print(f"fixed completed result rows: {fixed_completed_count}")
    print(f"sampled remaining group fixtures: {sampled_remaining_count}")
    print(scoreline_message)
    print(f"knockout probability source: {knockout_source.source_label}")
    print(f"knockout caveat: {knockout_source.caveat}")
    print(f"bracket mapping: {BRACKET_SOURCE_NOTE}")
    print("knockout draw treatment: regular-time draw mass split 50/50")
    print(
        "forecast_mode values: "
        + _format_counts(prediction_metadata["forecast_mode_counts"])
    )
    print(
        "feature_cutoff_date values: "
        + _format_counts(prediction_metadata["feature_cutoff_date_counts"])
    )
    print("player/market/live-lineup data: not used")

    _print_top_table("Top Champion Probabilities", summary, "champion_prob")
    _print_top_table("Top Finalist Probabilities", summary, "reach_final_prob")
    _print_top_table("Top Semifinal Probabilities", summary, "reach_semifinal_prob")
    _print_top_table("Top Quarterfinal Probabilities", summary, "reach_quarterfinal_prob")
    _print_top_table(
        "Top Group-Stage Advancement Probabilities",
        summary,
        "advance_from_group_prob",
    )
    _print_contender_comparison(compare_top_contenders(traces, summary))
    _print_selected_team_diagnostic("Argentina", summary, traces)
    _print_selected_team_diagnostic("France", summary, traces)

    print("\nNo simulation files were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
