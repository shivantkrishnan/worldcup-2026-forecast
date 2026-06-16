"""Print an in-memory readiness audit for match-level features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import load_baseline_training_matches  # noqa: E402
from src.features.feature_audit import (  # noqa: E402
    audit_feature_readiness,
    summarize_feature_audit,
)
from src.features.build_features import build_modeling_features  # noqa: E402
from src.features.elo import ELO_FEATURE_COLUMNS  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Audit match-level modeling features without writing files."
    )
    parser.add_argument(
        "--include-elo",
        action="store_true",
        help="Include leakage-safe pre-match Elo features in the audit.",
    )
    return parser.parse_args(argv)


def build_features_for_audit(
    baseline_matches,
    include_elo: bool = False,
):
    """Build the feature table used by this audit script."""
    return build_modeling_features(baseline_matches, include_elo=include_elo)


def _print_distribution(title: str, distribution: dict[str, int]) -> None:
    print(title)
    for label, count in distribution.items():
        print(f"  {label}: {count:,}")


def main(argv: list[str] | None = None) -> int:
    """Build features in memory and print readiness diagnostics."""
    args = parse_args(argv)
    try:
        baseline_matches = load_baseline_training_matches()
    except FileNotFoundError as error:
        print(f"Missing historical results data: {error}")
        return 1

    features = build_features_for_audit(
        baseline_matches,
        include_elo=args.include_elo,
    )
    report = audit_feature_readiness(features, test_start_date="2022-01-01")

    print(summarize_feature_audit(report))
    print(f"\ninclude_elo: {args.include_elo}")
    print(f"\nfeature table shape: {features.shape}")

    _print_distribution("\ntrain target distribution:", report["target_distribution_train"])
    _print_distribution("\ntest target distribution:", report["target_distribution_test"])

    overall_missingness = report["missingness_by_feature_overall"]
    top_missing = sorted(
        overall_missingness.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:20]

    print("\ntop 20 features by overall missingness:")
    for feature, missing_share in top_missing:
        print(f"  {feature}: {missing_share:.2%}")

    print("\nfully missing features:")
    if report["fully_missing_features"]:
        for feature in report["fully_missing_features"]:
            print(f"  {feature}")
    else:
        print("  none")

    print("\nhigh missingness features over 50%:")
    if report["high_missingness_features"]:
        for feature in report["high_missingness_features"]:
            print(f"  {feature}")
    else:
        print("  none")

    if args.include_elo:
        print("\nElo feature columns:")
        for column in ELO_FEATURE_COLUMNS:
            missing_share = report["missingness_by_feature_overall"].get(column)
            if missing_share is None:
                print(f"  {column}: missing from feature columns")
            else:
                print(f"  {column}: {missing_share:.2%} missing")

    print(
        "\nNote: missingness is expected for early-team-history rows and rolling "
        "windows. The later model pipeline must handle it explicitly."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
