"""Refresh public-demo result and live-prediction snapshots.

This script is intentionally file-based so GitHub Actions can commit changed
CSV snapshots and let Streamlit Community Cloud redeploy from GitHub. It does
not commit files itself.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import generate_fixture_predictions as prediction_generator  # noqa: E402
from scripts import ingest_official_results_2026 as official_results  # noqa: E402
from scripts.simulate_group_stage import prepare_simulation_fixture_table  # noqa: E402
from src.data.pipeline import load_baseline_training_matches  # noqa: E402
from src.data.tournament_fixtures import load_tournament_fixtures  # noqa: E402
from src.data.tournament_results import (  # noqa: E402
    load_tournament_results,
    validate_tournament_results,
)
from src.simulation.tournament import validate_fixture_probability_table  # noqa: E402
from src.utils.config import (  # noqa: E402
    FIXTURES_2026_PATH,
    RESULTS_2026_PATH,
)

LIVE_PREDICTIONS_2026_PATH = "data/tournament/fixture_predictions_2026_live.csv"
RESULT_CORE_COLUMNS = [
    "match_date",
    "team_a",
    "team_b",
    "team_a_goals",
    "team_b_goals",
    "result",
    "status",
]


@dataclass(frozen=True)
class RefreshSummary:
    """Summary values printed by the refresh script."""

    previous_completed_results: int
    new_completed_results: int
    results_file_changed: bool
    prediction_file_changed: bool
    live_prediction_rows: int
    feature_cutoff_date: str
    completed_matches_omitted: int
    scheduled_matches_included: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Refresh public Streamlit demo result and prediction snapshots."
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
        default=LIVE_PREDICTIONS_2026_PATH,
        help="Path to fixture_predictions_2026_live.csv.",
    )
    parser.add_argument(
        "--from-date",
        default="2026-06-11",
        help="First date requested from FIFA, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--to-date",
        default=date.today().isoformat(),
        help="Last date requested from FIFA, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--force-predictions",
        action="store_true",
        help="Regenerate live predictions even when results did not change.",
    )
    return parser.parse_args(argv)


def _file_hash(path: str | Path) -> str | None:
    """Return a SHA256 hash for a file, or None when it does not exist."""
    file_path = Path(path)
    if not file_path.exists():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_fifa_source_label() -> str:
    """Return a stable official-source label independent of request dates."""
    return (
        f"{official_results.FIFA_CALENDAR_API_URL}?language=en&count=100"
        f"&idCompetition={official_results.FIFA_COMPETITION_ID}"
        f"&idSeason={official_results.FIFA_2026_SEASON_ID}"
    )


def _load_previous_results_raw(results_path: str | Path) -> pd.DataFrame:
    """Load the previous result CSV without normalizing display metadata."""
    path = Path(results_path)
    if not path.exists():
        return pd.DataFrame(columns=official_results.RESULT_COLUMNS)
    return pd.read_csv(path)


def _result_core_key(row: pd.Series) -> tuple[Any, ...]:
    """Return result fields used to detect substantive score corrections."""
    values: list[Any] = []
    for column in RESULT_CORE_COLUMNS:
        value = row[column]
        if column == "match_date":
            value = pd.Timestamp(value).date().isoformat()
        elif column in {"team_a_goals", "team_b_goals"}:
            value = int(value)
        else:
            value = str(value)
        values.append(value)
    return tuple(values)


def preserve_existing_result_metadata(
    previous_results: pd.DataFrame,
    refreshed_results: pd.DataFrame,
) -> pd.DataFrame:
    """Preserve source/last_updated for unchanged existing result rows."""
    if previous_results.empty or refreshed_results.empty:
        return refreshed_results.copy(deep=True)

    previous_by_id = {
        str(row["match_id"]): row
        for _, row in previous_results.iterrows()
        if not pd.isna(row.get("match_id"))
    }
    output = refreshed_results.copy(deep=True)

    for index, row in output.iterrows():
        match_id = str(row["match_id"])
        previous = previous_by_id.get(match_id)
        if previous is None:
            continue
        if _result_core_key(previous) != _result_core_key(row):
            continue
        for column in ["source", "last_updated"]:
            if column in previous.index and not pd.isna(previous[column]):
                output.loc[index, column] = previous[column]
    return output


def fetch_and_write_official_results(
    fixtures: pd.DataFrame,
    results_path: str | Path,
    from_date: str,
    to_date: str,
) -> tuple[pd.DataFrame, int, list[str], list[str]]:
    """Fetch official results, protect against row loss, and write CSV."""
    previous_raw = _load_previous_results_raw(results_path)
    previous_count = len(previous_raw)

    fetch_url = official_results.build_fifa_results_url(from_date, to_date)
    payload = official_results.fetch_fifa_payload(fetch_url)
    refreshed, omitted, orientation_notes = official_results.build_results_from_fifa_payload(
        fixtures,
        payload,
        source_url=_stable_fifa_source_label(),
        last_updated=date.today().isoformat(),
    )
    refreshed = preserve_existing_result_metadata(previous_raw, refreshed)

    previous_ids = set(previous_raw.get("match_id", pd.Series(dtype=str)).astype(str))
    refreshed_ids = set(refreshed["match_id"].astype(str))
    lost_ids = sorted(previous_ids.difference(refreshed_ids))
    if lost_ids:
        retained = previous_raw.loc[
            previous_raw["match_id"].astype(str).isin(lost_ids),
            official_results.RESULT_COLUMNS,
        ]
        print(
            "Latest official response omitted previously validated result rows; "
            "retaining: "
            + ", ".join(lost_ids)
        )
        refreshed = pd.concat([refreshed, retained], ignore_index=True)
        refreshed = refreshed.sort_values(
            ["match_date", "match_id"],
            kind="mergesort",
        ).reset_index(drop=True)

    validate_tournament_results(refreshed, fixtures_or_predictions=fixtures)
    if len(refreshed) < previous_count:
        raise ValueError(
            "Official refresh would reduce completed result count from "
            f"{previous_count} to {len(refreshed)}."
        )

    official_results.write_results_csv(refreshed, results_path)
    loaded = load_tournament_results(results_path, fixtures_or_predictions=fixtures)
    return loaded, previous_count, omitted, orientation_notes


def regenerate_live_predictions(
    fixtures: pd.DataFrame,
    results: pd.DataFrame,
    predictions_path: str | Path,
) -> pd.DataFrame:
    """Regenerate and write remaining-fixture live predictions."""
    try:
        training_matches = load_baseline_training_matches()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "Unable to regenerate live predictions because data/raw/results.csv "
            "is missing. In GitHub Actions, configure RAW_RESULTS_CSV_URL to a "
            "private copy of the manually downloaded historical results CSV."
        ) from error

    predictions = prediction_generator.generate_fixture_predictions(
        training_matches,
        fixtures,
        completed_results=results,
        forecast_mode=prediction_generator.FORECAST_MODE_LIVE,
        generated_at=_verified_result_generated_at(results),
    )
    predictions = predictions.sort_values(
        ["match_date", "match_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    prediction_generator.write_fixture_predictions(predictions, predictions_path)
    return validate_fixture_probability_table(pd.read_csv(predictions_path))


def _verified_result_generated_at(results: pd.DataFrame) -> str | None:
    """Return a generation timestamp anchored to verified completed results.

    Scheduled GitHub refreshes can run after UTC midnight while official result
    feeds have not yet published every prior-day result. Anchoring live output
    metadata to the latest verified result date prevents unverified fixtures
    from being treated as silently backfilled predictions.
    """
    if results.empty:
        return None
    max_result_date = pd.to_datetime(results["match_date"], errors="raise").max()
    return f"{max_result_date.date().isoformat()}T23:59:59+00:00"


def validate_public_demo_snapshot(
    fixtures: pd.DataFrame,
    results: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Validate the public demo snapshot before GitHub Actions can commit it."""
    fixture_rows = fixtures.copy(deep=True)
    result_rows = results.copy(deep=True)
    prediction_rows = validate_fixture_probability_table(predictions)

    fixture_ids = set(fixture_rows["match_id"].astype(str))
    result_ids = set(result_rows["match_id"].astype(str))
    prediction_ids = set(prediction_rows["match_id"].astype(str))

    unknown_result_ids = sorted(result_ids.difference(fixture_ids))
    if unknown_result_ids:
        raise ValueError(
            "results_2026.csv contains match_id values not in fixtures_2026.csv: "
            + ", ".join(unknown_result_ids)
        )

    unknown_prediction_ids = sorted(prediction_ids.difference(fixture_ids))
    if unknown_prediction_ids:
        raise ValueError(
            "fixture_predictions_2026_live.csv contains match_id values not in "
            "fixtures_2026.csv: "
            + ", ".join(unknown_prediction_ids)
        )

    completed_prediction_ids = sorted(result_ids.intersection(prediction_ids))
    if completed_prediction_ids:
        raise ValueError(
            "Live predictions must omit completed fixture match_id values: "
            + ", ".join(completed_prediction_ids)
        )

    expected_total = len(fixture_rows)
    observed_total = len(result_rows) + len(prediction_rows)
    if observed_total != expected_total:
        raise ValueError(
            "Completed result count plus live prediction count must equal total "
            f"fixtures: {len(result_rows)} + {len(prediction_rows)} != {expected_total}."
        )

    if not prediction_rows.empty:
        forecast_modes = set(prediction_rows.get("forecast_mode", pd.Series()).astype(str))
        if forecast_modes != {prediction_generator.FORECAST_MODE_LIVE}:
            raise ValueError(
                "Live prediction snapshot must contain only forecast_mode=live rows."
            )
        if "is_backfilled" in prediction_rows.columns:
            backfilled = prediction_rows["is_backfilled"].map(_coerce_bool)
            if backfilled.any():
                raise ValueError("Live prediction snapshot must not contain backfilled rows.")

    if not result_rows.empty and not prediction_rows.empty:
        max_result_date = pd.to_datetime(result_rows["match_date"], errors="raise").max()
        feature_cutoffs = pd.to_datetime(
            prediction_rows["feature_cutoff_date"],
            errors="raise",
        )
        if (feature_cutoffs < max_result_date).any():
            raise ValueError(
                "Live prediction feature_cutoff_date must be at least the latest "
                "completed result date."
            )

    # Exercise the simulation conditioning path before allowing a commit.
    prepare_simulation_fixture_table(fixture_rows, prediction_rows, result_rows)

    probability_sums = prediction_rows[
        ["p_team_a_win", "p_draw", "p_team_b_win"]
    ].sum(axis=1)
    if not np.isclose(probability_sums, 1.0, atol=1e-6).all():
        raise ValueError("Live prediction probabilities must sum to 1.")

    feature_cutoff_date = (
        str(prediction_rows["feature_cutoff_date"].iloc[0])
        if not prediction_rows.empty and "feature_cutoff_date" in prediction_rows
        else "-"
    )
    return {
        "total_fixtures": int(expected_total),
        "completed_results": int(len(result_rows)),
        "live_prediction_rows": int(len(prediction_rows)),
        "feature_cutoff_date": feature_cutoff_date,
    }


def _coerce_bool(value: object) -> bool:
    """Coerce common CSV-backed boolean values."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def refresh_public_demo_snapshot(
    fixtures_path: str | Path = FIXTURES_2026_PATH,
    results_path: str | Path = RESULTS_2026_PATH,
    predictions_path: str | Path = LIVE_PREDICTIONS_2026_PATH,
    from_date: str = "2026-06-11",
    to_date: str = date.today().isoformat(),
    force_predictions: bool = False,
) -> RefreshSummary:
    """Refresh results and live predictions, then validate the snapshot."""
    fixtures = load_tournament_fixtures(fixtures_path)
    before_results_hash = _file_hash(results_path)
    before_predictions_hash = _file_hash(predictions_path)

    results, previous_count, omitted, orientation_notes = fetch_and_write_official_results(
        fixtures,
        results_path=results_path,
        from_date=from_date,
        to_date=to_date,
    )
    after_results_hash = _file_hash(results_path)
    results_changed = before_results_hash != after_results_hash

    if orientation_notes:
        print("Orientation corrections:")
        for note in orientation_notes:
            print(f"- {note}")

    if omitted:
        print("Omitted official completed rows:")
        for note in omitted:
            print(f"- {note}")

    should_regenerate_predictions = (
        force_predictions or results_changed or not Path(predictions_path).exists()
    )
    if should_regenerate_predictions:
        predictions = regenerate_live_predictions(fixtures, results, predictions_path)
    else:
        predictions = validate_fixture_probability_table(pd.read_csv(predictions_path))

    snapshot_report = validate_public_demo_snapshot(fixtures, results, predictions)
    after_predictions_hash = _file_hash(predictions_path)

    return RefreshSummary(
        previous_completed_results=previous_count,
        new_completed_results=int(snapshot_report["completed_results"]),
        results_file_changed=results_changed,
        prediction_file_changed=before_predictions_hash != after_predictions_hash,
        live_prediction_rows=int(snapshot_report["live_prediction_rows"]),
        feature_cutoff_date=str(snapshot_report["feature_cutoff_date"]),
        completed_matches_omitted=int(snapshot_report["completed_results"]),
        scheduled_matches_included=int(snapshot_report["live_prediction_rows"]),
    )


def print_summary(summary: RefreshSummary) -> None:
    """Print a concise CI-friendly refresh summary."""
    print("Public Demo Snapshot Refresh")
    print("============================")
    print(f"previous completed results: {summary.previous_completed_results}")
    print(f"new completed results: {summary.new_completed_results}")
    print(f"results file changed: {summary.results_file_changed}")
    print(f"prediction file changed: {summary.prediction_file_changed}")
    print(f"live prediction rows: {summary.live_prediction_rows}")
    print(f"feature cutoff date: {summary.feature_cutoff_date}")
    print(f"completed matches omitted from live predictions: {summary.completed_matches_omitted}")
    print(f"scheduled matches included in live predictions: {summary.scheduled_matches_included}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    try:
        summary = refresh_public_demo_snapshot(
            fixtures_path=args.fixtures,
            results_path=args.results,
            predictions_path=args.predictions,
            from_date=args.from_date,
            to_date=args.to_date,
            force_predictions=args.force_predictions,
        )
    except Exception as error:  # noqa: BLE001 - CI should receive a clear failure.
        print(f"Unable to refresh public demo snapshot: {error}")
        return 1

    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
