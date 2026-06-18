"""Compare backfilled ex-ante and live fixture prediction files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import FIXTURE_PREDICTIONS_2026_PATH  # noqa: E402

DEFAULT_LIVE_PREDICTIONS_PATH = "data/tournament/fixture_predictions_2026_live.csv"
PROBABILITY_COLUMNS = ["p_team_a_win", "p_draw", "p_team_b_win"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare backfilled ex-ante and live prediction files."
    )
    parser.add_argument(
        "--backfilled",
        default=FIXTURE_PREDICTIONS_2026_PATH,
        help="Path to the backfilled_ex_ante prediction CSV.",
    )
    parser.add_argument(
        "--live",
        default=DEFAULT_LIVE_PREDICTIONS_PATH,
        help="Path to the live prediction CSV.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Number of largest probability shifts to print.",
    )
    return parser.parse_args(argv)


def compare_prediction_modes(
    backfilled_predictions: pd.DataFrame,
    live_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return row-level and team-level live-vs-backfilled differences."""
    missing_backfilled = set(["match_id", *PROBABILITY_COLUMNS]).difference(
        backfilled_predictions.columns
    )
    missing_live = set(["match_id", *PROBABILITY_COLUMNS]).difference(
        live_predictions.columns
    )
    if missing_backfilled:
        raise ValueError(
            "Backfilled predictions missing columns: "
            + ", ".join(sorted(missing_backfilled))
        )
    if missing_live:
        raise ValueError(
            "Live predictions missing columns: " + ", ".join(sorted(missing_live))
        )

    backfilled = backfilled_predictions.copy(deep=True)
    live = live_predictions.copy(deep=True)
    backfilled["match_id"] = backfilled["match_id"].astype(str).str.strip()
    live["match_id"] = live["match_id"].astype(str).str.strip()

    comparison = live.merge(
        backfilled,
        on="match_id",
        how="inner",
        suffixes=("_live", "_backfilled"),
        validate="one_to_one",
    )
    for column in PROBABILITY_COLUMNS:
        comparison[f"{column}_shift"] = (
            comparison[f"{column}_live"] - comparison[f"{column}_backfilled"]
        )
    shift_columns = [f"{column}_shift" for column in PROBABILITY_COLUMNS]
    comparison["max_abs_probability_shift"] = comparison[shift_columns].abs().max(axis=1)
    comparison["total_abs_probability_shift"] = comparison[shift_columns].abs().sum(axis=1)

    if {"predicted_class_live", "predicted_class_backfilled"}.issubset(
        comparison.columns
    ):
        comparison["favorite_changed"] = comparison["predicted_class_live"].ne(
            comparison["predicted_class_backfilled"]
        )
    else:
        comparison["favorite_changed"] = False

    if {"confidence_label_live", "confidence_label_backfilled"}.issubset(
        comparison.columns
    ):
        comparison["confidence_changed"] = comparison["confidence_label_live"].ne(
            comparison["confidence_label_backfilled"]
        )
    else:
        comparison["confidence_changed"] = False

    team_shift_rows: list[dict[str, object]] = []
    for row in comparison.itertuples(index=False):
        for suffix in ["a", "b"]:
            team_column = f"team_{suffix}_live"
            if hasattr(row, team_column):
                team_shift_rows.append(
                    {
                        "team": getattr(row, team_column),
                        "matches_compared": 1,
                        "total_abs_probability_shift": getattr(
                            row,
                            "total_abs_probability_shift",
                        ),
                    }
                )
    if team_shift_rows:
        team_summary = (
            pd.DataFrame(team_shift_rows)
            .groupby("team", as_index=False)
            .agg(
                matches_compared=("matches_compared", "sum"),
                total_abs_probability_shift=("total_abs_probability_shift", "sum"),
            )
            .sort_values(
                ["total_abs_probability_shift", "team"],
                ascending=[False, True],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
    else:
        team_summary = pd.DataFrame(
            columns=["team", "matches_compared", "total_abs_probability_shift"]
        )

    return comparison, team_summary


def main(argv: list[str] | None = None) -> int:
    """Print a read-only comparison of backfilled and live prediction modes."""
    args = parse_args(argv)
    backfilled_path = Path(args.backfilled)
    live_path = Path(args.live)
    if not backfilled_path.exists():
        print(f"Missing backfilled prediction file: {backfilled_path}")
        return 1
    if not live_path.exists():
        print(f"Missing live prediction file: {live_path}")
        return 1

    try:
        comparison, team_summary = compare_prediction_modes(
            pd.read_csv(backfilled_path),
            pd.read_csv(live_path),
        )
    except ValueError as error:
        print(f"Unable to compare prediction modes: {error}")
        return 1

    print("Prediction Mode Comparison")
    print("==========================")
    print(f"backfilled file: {backfilled_path}")
    print(f"live file: {live_path}")
    print(f"overlapping future fixtures: {len(comparison)}")
    if comparison.empty:
        print("No overlapping match_id values to compare.")
        return 0

    print(f"favorite changes: {int(comparison['favorite_changed'].sum())}")
    print(f"confidence label changes: {int(comparison['confidence_changed'].sum())}")

    display_columns = [
        "match_id",
        "team_a_live",
        "team_b_live",
        "favorite_display_backfilled",
        "favorite_display_live",
        "max_abs_probability_shift",
        "total_abs_probability_shift",
        "favorite_changed",
        "confidence_changed",
    ]
    available_display_columns = [
        column for column in display_columns if column in comparison.columns
    ]

    print("\nLargest Probability Shifts")
    print("==========================")
    print(
        comparison.sort_values(
            ["max_abs_probability_shift", "match_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(args.top_n)[available_display_columns]
        .to_string(index=False)
    )

    print("\nTeams Most Affected")
    print("===================")
    if team_summary.empty:
        print("No team-level shift summary available.")
    else:
        print(team_summary.head(args.top_n).to_string(index=False))

    print("\nNo comparison files were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
