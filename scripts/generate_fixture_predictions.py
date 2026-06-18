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
from src.features.fixture_features import build_fixture_feature_rows  # noqa: E402
from src.models.forecast import (  # noqa: E402
    format_prediction_output,
    predict_fixture_probabilities,
    train_selected_baseline,
)
from src.utils.config import (  # noqa: E402
    DEFAULT_TRAINING_CUTOFF_DATE,
    FIXTURE_PREDICTIONS_2026_PATH,
    FIXTURES_2026_PATH,
)

SELECTED_BASELINE_LABEL = "calibrated_logistic_team_form_elo_k10_home50"
MODEL_NAME = "sigmoid_calibrated_logistic_regression"


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
    return parser.parse_args(argv)


def generate_fixture_predictions(
    training_matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    feature_cutoff_date: str | None = None,
    generated_at: str | None = None,
) -> pd.DataFrame:
    """Train selected baseline in memory and return fixture predictions."""
    normalized_fixtures = fixtures.copy(deep=True)
    model_bundle = train_selected_baseline(training_matches)
    fixture_features = build_fixture_feature_rows(
        training_matches,
        normalized_fixtures,
        include_elo=model_bundle.include_elo,
        elo_k_factor=model_bundle.elo_k_factor,
        elo_home_advantage=model_bundle.elo_home_advantage,
        feature_cutoff_date=feature_cutoff_date,
    )
    probabilities = predict_fixture_probabilities(model_bundle, fixture_features)
    output = format_prediction_output(probabilities, normalized_fixtures)

    insert_after = output.columns.get_loc("match_date") + 1
    output.insert(insert_after, "group", normalized_fixtures["group"].reset_index(drop=True))
    output.insert(
        insert_after + 1,
        "stage",
        normalized_fixtures["stage"].reset_index(drop=True),
    )

    timestamp = generated_at or pd.Timestamp.utcnow().isoformat()
    generated_date = pd.Timestamp(timestamp).date()
    output["prediction_generated_at"] = timestamp
    output["training_cutoff_date"] = DEFAULT_TRAINING_CUTOFF_DATE
    output["feature_cutoff_date"] = feature_cutoff_date
    output["model_name"] = MODEL_NAME
    output["selected_baseline_label"] = SELECTED_BASELINE_LABEL
    output["is_backfilled"] = (
        pd.to_datetime(normalized_fixtures["match_date"]).dt.date < generated_date
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
    try:
        training_matches = load_baseline_training_matches()
        fixtures = load_tournament_fixtures(args.fixtures)
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
        feature_cutoff_date=args.feature_cutoff_date,
    )

    print("Fixture Predictions")
    print("===================")
    _print_prediction_table(predictions)

    if args.output:
        write_fixture_predictions(predictions, args.output)
        print(f"\nWrote fixture predictions to {args.output}")
    else:
        print("\nNo prediction file was written. Use --output to write a CSV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
