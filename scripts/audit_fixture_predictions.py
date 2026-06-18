"""Audit generated 2026 fixture predictions without writing artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import load_baseline_training_matches  # noqa: E402
from src.utils.config import FIXTURE_PREDICTIONS_2026_PATH  # noqa: E402

PROBABILITY_COLUMNS = ["p_team_a_win", "p_draw", "p_team_b_win"]
DISPLAY_COLUMNS = [
    "match_id",
    "group",
    "team_a",
    "team_b",
    "favorite_display",
    "favorite_probability",
    "confidence_label",
    "forecast_mode",
    "is_backfilled",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Audit generated fixture predictions without writing files."
    )
    parser.add_argument(
        "--predictions",
        default=FIXTURE_PREDICTIONS_2026_PATH,
        help="Path to fixture_predictions_2026.csv.",
    )
    parser.add_argument(
        "--skip-historical-support",
        action="store_true",
        help="Skip loading historical data for support-count heuristics.",
    )
    return parser.parse_args(argv)


def _coerce_bool(value: object) -> bool:
    """Coerce common CSV boolean values for audit summaries."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _load_historical_support_counts() -> dict[str, int]:
    """Return historical match counts per team when local raw data is present."""
    try:
        training_matches = load_baseline_training_matches()
    except FileNotFoundError:
        return {}

    teams = pd.concat(
        [training_matches["team_a"], training_matches["team_b"]],
        ignore_index=True,
    )
    return {str(team): int(count) for team, count in teams.value_counts().items()}


def _favorite_team(row: pd.Series) -> str | None:
    """Return the favorite team name, excluding draw favorites."""
    predicted_class = str(row.get("predicted_class", ""))
    if predicted_class == "team_a_win":
        return str(row["team_a"])
    if predicted_class == "team_b_win":
        return str(row["team_b"])
    return None


def _opponent_team(row: pd.Series, favorite: str | None) -> str | None:
    """Return the opponent of the favorite team."""
    if favorite is None:
        return None
    if favorite == str(row["team_a"]):
        return str(row["team_b"])
    return str(row["team_a"])


def _prediction_rows_with_favorite_probability(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return predictions with a max-probability helper column."""
    rows = predictions.copy(deep=True)
    rows["favorite_probability"] = rows[PROBABILITY_COLUMNS].max(axis=1)
    return rows


def _surprising_favorites(
    predictions: pd.DataFrame,
    historical_support_counts: dict[str, int],
) -> pd.DataFrame:
    """Return a simple local-data-only list of surprising favorites."""
    rows: list[dict[str, Any]] = []
    support_values = list(historical_support_counts.values())
    high_support_threshold = (
        float(pd.Series(support_values).quantile(0.75)) if support_values else None
    )

    for _, row in predictions.iterrows():
        favorite = _favorite_team(row)
        opponent = _opponent_team(row, favorite)
        if favorite is None or opponent is None:
            continue

        favorite_support = historical_support_counts.get(favorite)
        opponent_support = historical_support_counts.get(opponent)
        favorite_probability = float(row["favorite_probability"])
        reasons: list[str] = []

        if favorite_support is not None and favorite_support < 300:
            reasons.append("favorite has low historical support")
        if (
            favorite_support is not None
            and opponent_support is not None
            and favorite_support < opponent_support
            and favorite_probability >= 0.55
        ):
            reasons.append("favorite has less historical support than opponent")
        if (
            high_support_threshold is not None
            and opponent_support is not None
            and opponent_support >= high_support_threshold
            and favorite_probability >= 0.60
        ):
            reasons.append("high probability against high-support opponent")

        if not reasons:
            continue

        rows.append(
            {
                "match_id": row["match_id"],
                "group": row.get("group", ""),
                "favorite": favorite,
                "opponent": opponent,
                "favorite_probability": favorite_probability,
                "favorite_support": favorite_support,
                "opponent_support": opponent_support,
                "reason": "; ".join(reasons),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "match_id",
                "group",
                "favorite",
                "opponent",
                "favorite_probability",
                "favorite_support",
                "opponent_support",
                "reason",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["favorite_probability", "match_id"],
        ascending=[False, True],
        kind="mergesort",
    )


def audit_fixture_predictions(
    predictions_path: str | Path = FIXTURE_PREDICTIONS_2026_PATH,
    historical_support_counts: dict[str, int] | None = None,
) -> dict[str, object]:
    """Return a structured audit report for generated fixture predictions."""
    path = Path(predictions_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing fixture prediction file: {path}")

    predictions = pd.read_csv(path)
    missing_probability_columns = [
        column for column in PROBABILITY_COLUMNS if column not in predictions.columns
    ]
    if missing_probability_columns:
        raise ValueError(
            "Missing probability columns: " + ", ".join(missing_probability_columns)
        )

    predictions = _prediction_rows_with_favorite_probability(predictions)
    probability_sums = predictions[PROBABILITY_COLUMNS].sum(axis=1)
    forecast_mode_counts = (
        predictions.get("forecast_mode", pd.Series(["missing"] * len(predictions)))
        .fillna("missing")
        .value_counts(sort=False)
        .to_dict()
    )
    backfilled_count = (
        int(predictions["is_backfilled"].map(_coerce_bool).sum())
        if "is_backfilled" in predictions.columns
        else 0
    )
    support_counts = historical_support_counts or {}

    return {
        "prediction_count": int(len(predictions)),
        "probability_sum_max_deviation": float((probability_sums - 1.0).abs().max()),
        "forecast_mode_counts": {
            str(mode): int(count) for mode, count in forecast_mode_counts.items()
        },
        "backfilled_count": backfilled_count,
        "highest_confidence": predictions.sort_values(
            ["favorite_probability", "match_id"],
            ascending=[False, True],
            kind="mergesort",
        ).head(10),
        "surprising_favorites": _surprising_favorites(predictions, support_counts).head(10),
        "group_favorites": predictions.sort_values(
            ["group", "favorite_probability", "match_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("group", sort=True)
        .head(1),
        "historical_support_available": bool(support_counts),
    }


def _format_counts(counts: dict[str, int]) -> str:
    """Format count dictionaries for console output."""
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def summarize_fixture_prediction_audit(report: dict[str, object]) -> str:
    """Return a readable text summary for the fixture prediction audit."""
    highest_confidence = report["highest_confidence"]
    surprising_favorites = report["surprising_favorites"]
    group_favorites = report["group_favorites"]

    lines = [
        "Fixture Prediction Audit",
        "========================",
        f"predictions: {report['prediction_count']}",
        (
            "probability sum max deviation: "
            f"{report['probability_sum_max_deviation']:.12f}"
        ),
        "forecast_mode counts: "
        + _format_counts(report["forecast_mode_counts"]),
        f"backfilled rows: {report['backfilled_count']}",
        (
            "historical support counts: "
            + ("available" if report["historical_support_available"] else "unavailable")
        ),
        "",
        "Top 10 Highest-Confidence Predictions",
        "=====================================",
        highest_confidence[DISPLAY_COLUMNS].to_string(index=False),
        "",
        "Top 10 Heuristic Surprise Flags",
        "===============================",
    ]
    if len(surprising_favorites) == 0:
        lines.append("No heuristic surprise flags were found.")
    else:
        lines.append(surprising_favorites.to_string(index=False))

    lines.extend(
        [
            "",
            "Per-Group Favorite Summary",
            "==========================",
            group_favorites[
                [
                    "group",
                    "favorite_display",
                    "favorite_probability",
                    "team_a",
                    "team_b",
                    "forecast_mode",
                ]
            ].to_string(index=False),
            "",
            "No files were written.",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the fixture prediction audit from the command line."""
    args = parse_args(argv)
    support_counts = (
        {} if args.skip_historical_support else _load_historical_support_counts()
    )
    try:
        report = audit_fixture_predictions(
            predictions_path=args.predictions,
            historical_support_counts=support_counts,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Unable to audit fixture predictions: {error}")
        return 1

    print(summarize_fixture_prediction_audit(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
