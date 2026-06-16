"""Train and evaluate the first baseline model without writing artifacts."""

from __future__ import annotations

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
from src.features.team_form import build_match_level_features  # noqa: E402
from src.models.baseline import train_baseline_model  # noqa: E402


def _print_distribution(title: str, distribution: dict[str, int]) -> None:
    print(title)
    for label, count in distribution.items():
        print(f"  {label}: {count:,}")


def _print_metrics(title: str, metrics: dict[str, object]) -> None:
    print(title)
    print(f"  log_loss: {metrics['log_loss']:.6f}")
    print(f"  multiclass_brier_score: {metrics['multiclass_brier_score']:.6f}")
    print(f"  accuracy: {metrics['accuracy']:.6f}")
    print(f"  prediction_count: {metrics['prediction_count']:,}")


def main() -> int:
    """Run the first baseline training milestone."""
    try:
        baseline_matches = load_baseline_training_matches()
    except FileNotFoundError as error:
        print(f"Missing historical results data: {error}")
        return 1

    features = build_match_level_features(baseline_matches)
    readiness_report = audit_feature_readiness(features)
    result = train_baseline_model(features)

    print(summarize_feature_audit(readiness_report))
    print("\nBaseline Model Training")
    print("=======================")
    print(f"feature_count: {len(result['feature_columns'])}")
    print(f"train_row_count: {result['train_row_count']:,}")
    print(f"test_row_count: {result['test_row_count']:,}")
    print(f"train_date_range: {result['train_date_range']}")
    print(f"test_date_range: {result['test_date_range']}")

    _print_distribution("\ntrain target distribution:", result["target_distribution_train"])
    _print_distribution("\ntest target distribution:", result["target_distribution_test"])

    _print_metrics("\nclass-prior baseline metrics:", result["class_prior_metrics"])
    _print_metrics(
        "\nlogistic-regression baseline metrics:",
        result["logistic_regression_metrics"],
    )

    print("\nlogistic calibration bins:")
    print(result["calibration_table_logistic"].head(10).to_string(index=False))

    print("\nNo model artifacts or processed feature files were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
