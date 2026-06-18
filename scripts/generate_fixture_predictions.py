"""Generate fixture predictions from manually maintained tournament fixtures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import load_baseline_training_matches  # noqa: E402
from src.data.tournament_fixtures import load_tournament_fixtures  # noqa: E402
from src.data.tournament_results import load_tournament_results  # noqa: E402
from src.features.fixture_features import (  # noqa: E402
    build_fixture_feature_rows,
    build_live_feature_history,
)
from src.models.forecast import (  # noqa: E402
    format_prediction_output,
    predict_fixture_probabilities,
    train_selected_baseline,
)
from src.utils.config import (  # noqa: E402
    DEFAULT_TRAINING_CUTOFF_DATE,
    FIXTURE_PREDICTIONS_2026_PATH,
    FIXTURES_2026_PATH,
    RESULTS_2026_PATH,
)

SELECTED_BASELINE_LABEL = "calibrated_logistic_team_form_elo_k10_home50"
MODEL_NAME = "sigmoid_calibrated_logistic_regression"
FORECAST_MODE_PRE_TOURNAMENT = "pre_tournament"
FORECAST_MODE_BACKFILLED_EX_ANTE = "backfilled_ex_ante"
FORECAST_MODE_LIVE = "live"
FORECAST_MODES = (
    FORECAST_MODE_PRE_TOURNAMENT,
    FORECAST_MODE_BACKFILLED_EX_ANTE,
    FORECAST_MODE_LIVE,
)


def _timestamp(generated_at: str | None = None) -> str:
    """Return the prediction generation timestamp used for metadata."""
    return generated_at or pd.Timestamp.utcnow().isoformat()


def resolve_forecast_mode(
    fixtures: pd.DataFrame,
    forecast_mode: str | None = None,
    generated_at: str | None = None,
) -> str:
    """Resolve the forecast mode from explicit input or fixture dates."""
    if forecast_mode is not None:
        if forecast_mode not in FORECAST_MODES:
            raise ValueError(
                "forecast_mode must be one of: " + ", ".join(FORECAST_MODES)
            )
        return forecast_mode

    generated_date = pd.Timestamp(_timestamp(generated_at)).date()
    fixture_dates = pd.to_datetime(fixtures["match_date"], errors="raise").dt.date
    if fixture_dates.lt(generated_date).any():
        return FORECAST_MODE_BACKFILLED_EX_ANTE
    return FORECAST_MODE_PRE_TOURNAMENT


def resolve_feature_cutoff_date(
    forecast_mode: str,
    feature_cutoff_date: str | None = None,
    generated_at: str | None = None,
    completed_results: pd.DataFrame | None = None,
) -> str:
    """Return the non-empty feature cutoff date implied by the forecast mode."""
    if feature_cutoff_date is not None:
        return str(pd.Timestamp(feature_cutoff_date).date())
    if forecast_mode in {
        FORECAST_MODE_PRE_TOURNAMENT,
        FORECAST_MODE_BACKFILLED_EX_ANTE,
    }:
        return DEFAULT_TRAINING_CUTOFF_DATE
    if forecast_mode == FORECAST_MODE_LIVE:
        if completed_results is None or completed_results.empty:
            raise ValueError(
                "Live forecast mode requires completed results to resolve the "
                "default feature cutoff date."
            )
        result_dates = pd.to_datetime(
            completed_results["match_date"],
            errors="raise",
        )
        return str(result_dates.max().date())
    raise ValueError("Unknown forecast mode: " + forecast_mode)


def _completed_result_match_ids(
    completed_results: pd.DataFrame,
    feature_cutoff_date: str,
) -> set[str]:
    """Return completed result IDs available through the feature cutoff."""
    results = completed_results.copy(deep=True)
    results["match_date"] = pd.to_datetime(results["match_date"], errors="raise")
    if "status" in results.columns:
        status = results["status"].astype("string").str.strip().str.casefold()
        results = results.loc[status.eq("completed")].copy(deep=True)

    cutoff = pd.Timestamp(feature_cutoff_date)
    results = results.loc[results["match_date"] <= cutoff]
    return set(results["match_id"].astype(str).str.strip())


def _validate_no_backfilled_live_rows(
    fixtures_to_predict: pd.DataFrame,
    generated_date: object,
) -> None:
    """Prevent live mode from silently predicting past uncompleted fixtures."""
    fixture_dates = pd.to_datetime(
        fixtures_to_predict["match_date"],
        errors="raise",
    ).dt.date
    past_uncompleted = fixture_dates < generated_date
    if past_uncompleted.any():
        missing_ids = sorted(
            fixtures_to_predict.loc[past_uncompleted, "match_id"]
            .astype(str)
            .unique()
        )
        raise ValueError(
            "Live forecast mode found fixture dates before the generation date "
            "without completed results. Update results_2026.csv or use an "
            "explicit audit mode. Affected match_id: "
            + ", ".join(missing_ids)
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate 2026 fixture predictions without writing by default."
    )
    parser.add_argument(
        "--fixtures",
        default=FIXTURES_2026_PATH,
        help="Path to manually maintained fixtures_2026.csv.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output CSV path. Use "
            f"{FIXTURE_PREDICTIONS_2026_PATH} for the default simulation input."
        ),
    )
    parser.add_argument(
        "--feature-cutoff-date",
        default=None,
        help="Optional latest completed-match date allowed in fixture features.",
    )
    parser.add_argument(
        "--forecast-mode",
        choices=FORECAST_MODES,
        default=None,
        help=(
            "Forecast semantics to use. Defaults to backfilled_ex_ante when "
            "generated after any fixture date, otherwise pre_tournament."
        ),
    )
    parser.add_argument(
        "--results",
        default=RESULTS_2026_PATH,
        help="Path to manually maintained results_2026.csv for live mode.",
    )
    parser.add_argument(
        "--include-completed-for-audit",
        action="store_true",
        help=(
            "In live mode, also generate rows for completed fixtures. By "
            "default live output includes remaining fixtures only."
        ),
    )
    return parser.parse_args(argv)


def generate_fixture_predictions(
    training_matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    completed_results: pd.DataFrame | None = None,
    feature_cutoff_date: str | None = None,
    forecast_mode: str | None = None,
    generated_at: str | None = None,
    include_completed_for_audit: bool = False,
) -> pd.DataFrame:
    """Train selected baseline in memory and return fixture predictions."""
    normalized_fixtures = fixtures.copy(deep=True)
    timestamp = _timestamp(generated_at)
    resolved_forecast_mode = resolve_forecast_mode(
        normalized_fixtures,
        forecast_mode=forecast_mode,
        generated_at=timestamp,
    )
    resolved_feature_cutoff_date = resolve_feature_cutoff_date(
        resolved_forecast_mode,
        feature_cutoff_date=feature_cutoff_date,
        generated_at=timestamp,
        completed_results=completed_results,
    )
    generated_date = pd.Timestamp(timestamp).date()

    model_bundle = train_selected_baseline(training_matches)
    feature_history = training_matches
    fixtures_to_predict = normalized_fixtures.copy(deep=True)

    if resolved_forecast_mode == FORECAST_MODE_LIVE:
        if completed_results is None:
            raise ValueError("Live forecast mode requires completed_results.")
        feature_history = build_live_feature_history(
            training_matches,
            completed_results,
            feature_cutoff_date=resolved_feature_cutoff_date,
        )
        if not include_completed_for_audit:
            completed_match_ids = _completed_result_match_ids(
                completed_results,
                resolved_feature_cutoff_date,
            )
            fixture_ids = normalized_fixtures["match_id"].astype(str).str.strip()
            fixtures_to_predict = normalized_fixtures.loc[
                ~fixture_ids.isin(completed_match_ids)
            ].copy(deep=True)
            _validate_no_backfilled_live_rows(fixtures_to_predict, generated_date)

    fixture_features = build_fixture_feature_rows(
        feature_history,
        fixtures_to_predict,
        include_elo=model_bundle.include_elo,
        elo_k_factor=model_bundle.elo_k_factor,
        elo_home_advantage=model_bundle.elo_home_advantage,
        feature_cutoff_date=resolved_feature_cutoff_date,
    )
    probabilities = predict_fixture_probabilities(model_bundle, fixture_features)
    output = format_prediction_output(probabilities, fixtures_to_predict)

    insert_after = output.columns.get_loc("match_date") + 1
    output.insert(insert_after, "group", fixtures_to_predict["group"].reset_index(drop=True))
    output.insert(
        insert_after + 1,
        "stage",
        fixtures_to_predict["stage"].reset_index(drop=True),
    )

    output["prediction_generated_at"] = timestamp
    output["training_cutoff_date"] = DEFAULT_TRAINING_CUTOFF_DATE
    output["feature_cutoff_date"] = resolved_feature_cutoff_date
    output["forecast_mode"] = resolved_forecast_mode
    output["model_name"] = MODEL_NAME
    output["selected_baseline_label"] = SELECTED_BASELINE_LABEL
    if resolved_forecast_mode == FORECAST_MODE_LIVE and not include_completed_for_audit:
        output["is_backfilled"] = False
    else:
        output["is_backfilled"] = (
            pd.to_datetime(fixtures_to_predict["match_date"]).dt.date < generated_date
        ).reset_index(drop=True)
    return output


def write_fixture_predictions(predictions: pd.DataFrame, output_path: str | Path) -> None:
    """Write fixture predictions only when the caller explicitly requests it."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)


def _print_prediction_table(predictions: pd.DataFrame) -> None:
    """Print a compact fixture-prediction table."""
    display_columns = [
        "match_id",
        "match_date",
        "group",
        "team_a",
        "team_b",
        "p_team_a_win",
        "p_draw",
        "p_team_b_win",
        "predicted_class",
        "favorite_display",
        "confidence_label",
    ]
    print(predictions[display_columns].to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    """Generate fixture predictions from local historical and fixture data."""
    args = parse_args(argv)
    generated_at = _timestamp()
    try:
        training_matches = load_baseline_training_matches()
        fixtures = load_tournament_fixtures(args.fixtures)
        forecast_mode = resolve_forecast_mode(
            fixtures,
            forecast_mode=args.forecast_mode,
            generated_at=generated_at,
        )
        completed_results = None
        if forecast_mode == FORECAST_MODE_LIVE:
            results_path = Path(args.results)
            if not results_path.exists():
                print(
                    "Live forecast mode requires manually maintained completed "
                    f"results at {results_path}."
                )
                print(
                    "No live predictions were generated; use "
                    "backfilled_ex_ante or pre_tournament mode until "
                    "results_2026.csv is available."
                )
                return 1
            completed_results = load_tournament_results(
                results_path,
                fixtures_or_predictions=fixtures,
            )
        feature_cutoff_date = resolve_feature_cutoff_date(
            forecast_mode,
            feature_cutoff_date=args.feature_cutoff_date,
            generated_at=generated_at,
            completed_results=completed_results,
        )
    except FileNotFoundError as error:
        print(f"Missing required data: {error}")
        print("See docs/fixtures_2026_template.md for the fixture CSV schema.")
        return 1
    except ValueError as error:
        print(f"Invalid fixture data: {error}")
        return 1

    predictions = generate_fixture_predictions(
        training_matches,
        fixtures,
        completed_results=completed_results,
        feature_cutoff_date=feature_cutoff_date,
        forecast_mode=forecast_mode,
        generated_at=generated_at,
        include_completed_for_audit=args.include_completed_for_audit,
    )

    print("Fixture Predictions")
    print("===================")
    print(f"forecast_mode: {forecast_mode}")
    print(f"feature_cutoff_date: {feature_cutoff_date}")
    if forecast_mode == FORECAST_MODE_LIVE:
        completed_count = len(fixtures) - len(predictions)
        print(f"completed fixtures omitted from live output: {completed_count}")
        print(
            "recommended live output path: "
            "data/tournament/fixture_predictions_2026_live.csv"
        )
    _print_prediction_table(predictions)

    if args.output:
        write_fixture_predictions(predictions, args.output)
        print(f"\nWrote fixture predictions to {args.output}")
    else:
        print("\nNo prediction file was written. Use --output to write a CSV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
