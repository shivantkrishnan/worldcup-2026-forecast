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
from src.data.tournament_fixtures import load_tournament_fixtures  # noqa: E402
from src.data.pipeline import load_baseline_training_matches  # noqa: E402
from src.simulation.scorelines import build_empirical_scoreline_distributions  # noqa: E402
from src.simulation.tournament import (  # noqa: E402
    simulate_group_stage,
    summarize_advancement_probabilities,
    validate_fixture_probability_table,
)
from src.utils.config import (  # noqa: E402
    FIXTURE_PREDICTIONS_2026_PATH,
    FIXTURES_2026_PATH,
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
        "--fixtures",
        default=FIXTURES_2026_PATH,
        help="Path to fixtures_2026.csv, used when predictions contain remaining fixtures only.",
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


def _validate_prediction_orientation(
    fixtures: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Validate prediction rows against fixture orientation when possible."""
    if not {"team_a", "team_b"}.issubset(predictions.columns):
        return

    reference = fixtures[["match_id", "team_a", "team_b"]].copy(deep=True)
    predicted = predictions[["match_id", "team_a", "team_b"]].copy(deep=True)
    merged = predicted.merge(
        reference,
        on="match_id",
        how="left",
        suffixes=("_prediction", "_fixture"),
        validate="one_to_one",
    )
    missing_fixture = merged["team_a_fixture"].isna()
    if missing_fixture.any():
        missing_ids = sorted(merged.loc[missing_fixture, "match_id"].astype(str).unique())
        raise ValueError(
            "Prediction rows contain match_id values not found in fixtures: "
            + ", ".join(missing_ids)
        )

    mismatch = (
        merged["team_a_prediction"].astype(str).ne(merged["team_a_fixture"].astype(str))
        | merged["team_b_prediction"].astype(str).ne(merged["team_b_fixture"].astype(str))
    )
    if mismatch.any():
        mismatch_ids = sorted(merged.loc[mismatch, "match_id"].astype(str).unique())
        raise ValueError(
            "Prediction team orientation does not match fixtures for match_id: "
            + ", ".join(mismatch_ids)
        )


def _fill_completed_probability_placeholders(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Fill missing completed-row probabilities with one-hot actual outcomes."""
    output = fixtures.copy(deep=True)
    completed = output.get("is_completed", False)
    if not isinstance(completed, pd.Series):
        completed = pd.Series([False] * len(output), index=output.index)
    completed = completed.map(_coerce_bool)

    for column in ["p_team_a_win", "p_draw", "p_team_b_win"]:
        if column not in output.columns:
            output[column] = pd.NA

    missing_probabilities = output[
        ["p_team_a_win", "p_draw", "p_team_b_win"]
    ].isna().any(axis=1)
    fill_rows = completed & missing_probabilities
    if not fill_rows.any():
        return output

    output.loc[fill_rows, ["p_team_a_win", "p_draw", "p_team_b_win"]] = 0.0
    outcome_to_probability = {
        "team_a_win": "p_team_a_win",
        "draw": "p_draw",
        "team_b_win": "p_team_b_win",
    }
    for index, actual_result in output.loc[fill_rows, "actual_result"].items():
        output.loc[index, outcome_to_probability[str(actual_result)]] = 1.0
    return output


def prepare_simulation_fixture_table(
    fixtures: pd.DataFrame,
    predictions: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Return all fixtures with completed results fixed and predictions overlaid.

    Prediction files may contain only remaining unplayed fixtures, as long as
    every fixture missing a prediction is covered by a completed result.
    """
    fixture_rows = fixtures.copy(deep=True)
    prediction_rows = predictions.copy(deep=True)
    fixture_rows["match_id"] = fixture_rows["match_id"].astype(str).str.strip()
    prediction_rows["match_id"] = prediction_rows["match_id"].astype(str).str.strip()
    _validate_prediction_orientation(fixture_rows, prediction_rows)

    overlay_columns = [
        column
        for column in prediction_rows.columns
        if column
        not in {"match_date", "group", "stage", "team_a", "team_b"}
    ]
    merged = fixture_rows.merge(
        prediction_rows[overlay_columns],
        on="match_id",
        how="left",
        validate="one_to_one",
    )
    conditioned = merge_completed_results_with_fixtures_or_predictions(
        merged,
        results,
    )
    conditioned = _fill_completed_probability_placeholders(conditioned)

    completed = conditioned["is_completed"].map(_coerce_bool)
    missing_unplayed = ~completed & conditioned[
        ["p_team_a_win", "p_draw", "p_team_b_win"]
    ].isna().any(axis=1)
    if missing_unplayed.any():
        missing_ids = sorted(
            conditioned.loc[missing_unplayed, "match_id"].astype(str).unique()
        )
        raise ValueError(
            "Unplayed fixtures require prediction probabilities. Missing match_id: "
            + ", ".join(missing_ids)
        )

    return validate_fixture_probability_table(conditioned)


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

    if "feature_cutoff_date" in fixtures.columns:
        feature_cutoff_date_counts = {
            str(value): int(count)
            for value, count in fixtures["feature_cutoff_date"]
            .fillna("missing")
            .value_counts(sort=False)
            .items()
        }
    else:
        feature_cutoff_date_counts = {"missing": int(len(fixtures))}

    return {
        "forecast_mode_counts": forecast_mode_counts,
        "feature_cutoff_date_counts": feature_cutoff_date_counts,
        "backfilled_count": backfilled_count,
        "fixed_completed_count": fixed_completed_count,
        "sampled_backfilled_count": int(sampled_backfilled.sum()),
        "completed_rows_sampled_as_predictions": bool(sampled_backfilled.any()),
        "live_predictions_used": bool(forecast_mode_counts.get("live", 0)),
    }


def _format_counts(counts: dict[str, int]) -> str:
    """Format count dictionaries for readable console output."""
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def _load_scoreline_distributions() -> tuple[dict, str]:
    """Load empirical scoreline distributions or fall back to defaults."""
    try:
        completed_matches = load_baseline_training_matches()
    except FileNotFoundError:
        return {}, "Using fallback conditional scoreline distributions."

    distributions = build_empirical_scoreline_distributions(completed_matches)
    if not distributions:
        return {}, "Using fallback conditional scoreline distributions."
    return distributions, "Using empirical historical scoreline distributions."


def _condition_on_completed_results(
    fixtures: pd.DataFrame,
    results_path: str | Path = RESULTS_PATH,
    fixture_path: str | Path = FIXTURES_2026_PATH,
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

    fixture_reference_path = Path(fixture_path)
    if fixture_reference_path.exists():
        fixture_reference = load_tournament_fixtures(fixture_reference_path)
        results = load_tournament_results(
            path,
            fixtures_or_predictions=fixture_reference,
        )
        conditioned = prepare_simulation_fixture_table(
            fixture_reference,
            fixtures,
            results,
        )
    else:
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
            fixture_path=args.fixtures,
            ignore_results=args.ignore_results,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to run group-stage simulation: {error}")
        return 1

    prediction_metadata = summarize_prediction_metadata(fixtures)
    scoreline_distributions, scoreline_message = _load_scoreline_distributions()
    fixed_completed_count = int(prediction_metadata["fixed_completed_count"])
    sampled_remaining_count = int(len(fixtures) - fixed_completed_count)
    simulation_results = simulate_group_stage(
        fixtures,
        n_simulations=DEFAULT_SIMULATION_COUNT,
        random_seed=42,
        top_n_per_group=2,
        include_best_third_place=True,
        n_best_third_place=8,
        scoreline_distributions=scoreline_distributions,
    )
    summary = summarize_advancement_probabilities(simulation_results)

    print("Group-Stage Monte Carlo Simulation")
    print("==================================")
    print(f"prediction file path: {args.predictions}")
    print(f"results file path: {args.results}")
    print(source_message)
    print(results_message)
    print(f"simulation count: {DEFAULT_SIMULATION_COUNT:,}")
    print(scoreline_message)
    print("scoreline simulation used: yes")
    print(
        "forecast_mode values: "
        + _format_counts(prediction_metadata["forecast_mode_counts"])
    )
    print(
        "feature_cutoff_date values: "
        + _format_counts(prediction_metadata["feature_cutoff_date_counts"])
    )
    print(f"backfilled rows: {prediction_metadata['backfilled_count']}")
    print(f"fixed completed result rows: {prediction_metadata['fixed_completed_count']}")
    print(f"sampled remaining rows: {sampled_remaining_count}")
    print(
        "backfilled rows still sampled: "
        f"{prediction_metadata['sampled_backfilled_count']}"
    )
    print(f"sampled remaining matches per simulation: {sampled_remaining_count}")
    print(
        "completed matches sampled as predictions: "
        + (
            "yes"
            if prediction_metadata["completed_rows_sampled_as_predictions"]
            else "no"
        )
    )
    print(
        "live predictions used: "
        + ("yes" if prediction_metadata["live_predictions_used"] else "no")
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

    print("\nTop Top-Two Probabilities")
    print("=========================")
    print(
        summary.sort_values(
            ["top_2_prob", "group_winner_prob", "team"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .head(12)
        .to_string(index=False)
    )

    print("\nTop Best-Third Advancement Probabilities")
    print("========================================")
    third_place_summary = summary.loc[summary["best_third_place_advance_prob"] > 0]
    if third_place_summary.empty:
        print("No best-third-place advancement in this simulation configuration.")
    else:
        print(
            third_place_summary.sort_values(
                ["best_third_place_advance_prob", "advance_prob", "team"],
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
        "Goal difference, goals scored, and official-style tie-breaks are now "
        "approximated with a conditional scoreline layer. This is not yet a "
        "separately calibrated goals model, and knockout simulation is not "
        "implemented."
    )
    print("\nNo simulation files were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
