"""Print sample fixture predictions without writing artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import load_baseline_training_matches  # noqa: E402
from src.features.fixture_features import build_fixture_feature_rows  # noqa: E402
from src.models.forecast import (  # noqa: E402
    format_prediction_output,
    predict_fixture_probabilities,
    train_selected_baseline,
)
from src.utils.config import FIXTURES_2026_PATH  # noqa: E402

SAMPLE_SIZE = 5


def _load_fixture_sample(path: str | Path = FIXTURES_2026_PATH) -> pd.DataFrame | None:
    """Load a small future-fixture sample if a local fixture file exists."""
    fixture_path = Path(path)
    if not fixture_path.exists():
        return None

    fixtures = pd.read_csv(fixture_path)
    if fixtures.empty:
        return None

    fixtures["match_date"] = pd.to_datetime(fixtures["match_date"], errors="raise")
    if "status" in fixtures.columns:
        status = fixtures["status"].astype(str).str.strip().str.casefold()
        fixtures = fixtures.loc[
            ~status.isin({"completed", "played", "final", "finished"})
        ].copy()

    today = pd.Timestamp.today().normalize()
    future = fixtures.loc[fixtures["match_date"] >= today].copy()
    if future.empty:
        future = fixtures.copy()

    return future.sort_values(["match_date", "match_id"], kind="mergesort").head(
        SAMPLE_SIZE
    )


def _synthetic_fixture_sample(training_matches: pd.DataFrame) -> pd.DataFrame:
    """Build a tiny fixture sample from teams present in historical data."""
    recent_matches = training_matches.sort_values(
        ["match_date", "match_id"],
        kind="mergesort",
    ).tail(50)
    teams = pd.unique(pd.concat([recent_matches["team_a"], recent_matches["team_b"]]))
    if len(teams) < 2:
        raise ValueError("Need at least two historical teams to build a sample fixture.")

    return pd.DataFrame(
        [
            {
                "match_id": "synthetic_fixture_1",
                "match_date": "2026-06-20",
                "team_a": str(teams[0]),
                "team_b": str(teams[1]),
                "tournament": "FIFA World Cup",
            }
        ]
    )


def main() -> int:
    """Train the selected baseline and print sample fixture predictions."""
    try:
        training_matches = load_baseline_training_matches()
    except FileNotFoundError as error:
        print(f"Missing historical results data: {error}")
        return 1

    fixtures = _load_fixture_sample()
    if fixtures is None:
        print("No local 2026 fixture file found; using a synthetic sample fixture.")
        fixtures = _synthetic_fixture_sample(training_matches)
    else:
        print(f"Loaded sample fixtures from {FIXTURES_2026_PATH}.")

    model_bundle = train_selected_baseline(training_matches)
    fixture_features = build_fixture_feature_rows(
        training_matches,
        fixtures,
        include_elo=model_bundle.include_elo,
        elo_k_factor=model_bundle.elo_k_factor,
        elo_home_advantage=model_bundle.elo_home_advantage,
    )
    probabilities = predict_fixture_probabilities(model_bundle, fixture_features)
    output = format_prediction_output(probabilities, fixtures)

    print("\nSample Fixture Predictions")
    print("==========================")
    display_columns = [
        "team_a",
        "team_b",
        "p_team_a_win",
        "p_draw",
        "p_team_b_win",
        "predicted_class",
        "favorite_display",
        "confidence_label",
    ]
    print(output[display_columns].to_string(index=False))
    print("\nNo model artifacts or processed feature files were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
