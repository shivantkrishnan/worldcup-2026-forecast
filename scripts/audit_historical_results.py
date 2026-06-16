"""Print descriptive diagnostics for the local historical results dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.clean_results import filter_baseline_training_matches  # noqa: E402
from src.data.load_data import load_raw_results  # noqa: E402
from src.data.pipeline import (  # noqa: E402
    load_and_clean_results_with_quarantine,
)
from src.data.quality_checks import validate_canonical_matches  # noqa: E402
from src.utils.config import DEFAULT_TRAINING_CUTOFF_DATE, RAW_RESULTS_PATH  # noqa: E402


def _format_date(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).date().isoformat()


def main() -> int:
    """Run the historical-results audit and print diagnostics to stdout."""
    try:
        raw_results = load_raw_results(RAW_RESULTS_PATH)
        duplicate_resolution = load_and_clean_results_with_quarantine(RAW_RESULTS_PATH)
        canonical = duplicate_resolution.resolved_matches
        quarantined = duplicate_resolution.quarantined_matches
        duplicate_report = duplicate_resolution.duplicate_report
        baseline = filter_baseline_training_matches(canonical)
    except FileNotFoundError as error:
        print(f"Missing historical results data: {error}")
        return 1

    teams = pd.concat([canonical["team_a"], canonical["team_b"]], ignore_index=True)
    quality_report = validate_canonical_matches(canonical)
    excluded_after_cutoff = int(
        (canonical["match_date"] > pd.Timestamp(DEFAULT_TRAINING_CUTOFF_DATE)).sum()
    )

    print("Historical Results Audit")
    print("========================")
    print(f"Raw matches: {len(raw_results):,}")
    print(f"Resolved canonical matches: {len(canonical):,}")
    print(
        "Raw rows excluded from resolved canonical matches: "
        f"{len(raw_results) - len(canonical):,}"
    )
    print(f"Baseline-train-eligible matches: {len(baseline):,}")
    print(
        "Date range: "
        f"{_format_date(canonical['match_date'].min())} to "
        f"{_format_date(canonical['match_date'].max())}"
    )
    print(f"Latest match date: {_format_date(canonical['match_date'].max())}")
    print(f"Unique teams: {teams.nunique():,}")
    print(f"Unique tournaments: {canonical['tournament'].nunique():,}")
    print(
        "Matches after "
        f"{DEFAULT_TRAINING_CUTOFF_DATE} excluded from baseline training: "
        f"{excluded_after_cutoff:,}"
    )

    print("\nTop 15 tournaments by match count:")
    print(canonical["tournament"].value_counts().head(15).to_string())

    print("\nResult distribution:")
    print(canonical["result"].value_counts().to_string())

    print("\nNeutral vs non-neutral counts:")
    neutral_counts = canonical["is_neutral"].map(
        {True: "neutral", False: "non_neutral"}
    )
    print(neutral_counts.value_counts().to_string())

    print("\nMissing values by canonical column:")
    print(canonical.isna().sum().to_string())

    print("\nDuplicate Resolution:")
    print(f"passed: {duplicate_report.passed}")
    print(f"duplicate_group_count: {duplicate_report.duplicate_group_count:,}")
    print(f"duplicate_row_count: {duplicate_report.duplicate_row_count:,}")
    print(
        "metadata_duplicate_group_count: "
        f"{duplicate_report.metadata_duplicate_group_count:,}"
    )
    print(
        "conflicting_duplicate_group_count: "
        f"{duplicate_report.conflicting_duplicate_group_count:,}"
    )
    print(f"quarantined_row_count: {duplicate_report.quarantined_row_count:,}")
    print(f"resolved_match_count: {duplicate_report.resolved_match_count:,}")
    print(f"quarantined_matches: {len(quarantined):,}")
    print("messages:")
    for message in duplicate_report.messages:
        print(f"- {message}")

    print("\nData Quality Checks:")
    print(f"passed: {quality_report.passed}")
    print(f"duplicate_match_id_count: {quality_report.duplicate_match_id_count:,}")
    print(f"negative_score_count: {quality_report.negative_score_count:,}")
    print(f"null_required_value_count: {quality_report.null_required_value_count:,}")
    print(f"invalid_result_label_count: {quality_report.invalid_result_label_count:,}")
    print(f"invalid_neutral_value_count: {quality_report.invalid_neutral_value_count:,}")
    print(f"cutoff_inconsistency_count: {quality_report.cutoff_inconsistency_count:,}")
    print(f"matches_after_cutoff_count: {quality_report.matches_after_cutoff_count:,}")
    print("messages:")
    for message in quality_report.messages:
        print(f"- {message}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
