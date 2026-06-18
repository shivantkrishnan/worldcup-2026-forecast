"""Run rolling-origin backtests for baseline models without writing artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import load_baseline_training_matches  # noqa: E402
from src.features.build_features import build_modeling_features  # noqa: E402
from src.models.backtest import (  # noqa: E402
    aggregate_backtest_results,
    format_metric_mean_std,
    run_rolling_origin_backtest,
    summarize_backtest_results,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Run rolling-origin baseline backtests without writing artifacts."
    )
    parser.add_argument(
        "--include-elo",
        action="store_true",
        help="Include leakage-safe pre-match Elo features.",
    )
    parser.add_argument(
        "--elo-k-factor",
        type=float,
        default=20.0,
        help="Elo K-factor when --include-elo is used.",
    )
    parser.add_argument(
        "--elo-home-advantage",
        type=float,
        default=0.0,
        help="Team A home-rating bonus for non-neutral matches when --include-elo is used.",
    )
    return parser.parse_args(argv)


def build_features_for_backtest(
    baseline_matches,
    include_elo: bool = False,
    elo_k_factor: float = 20.0,
    elo_home_advantage: float = 0.0,
):
    """Build the feature table used by this backtest script."""
    return build_modeling_features(
        baseline_matches,
        include_elo=include_elo,
        elo_k_factor=elo_k_factor,
        elo_home_advantage=elo_home_advantage,
    )


def _print_aggregate_metrics(aggregate: dict[str, object]) -> None:
    print("\nAggregate Metrics")
    print("=================")
    model_metrics = aggregate["model_metrics"]
    assert isinstance(model_metrics, dict)

    for model_name, metrics in model_metrics.items():
        assert isinstance(metrics, dict)
        print(f"\n{model_name}:")
        for metric_name in [
            "log_loss",
            "multiclass_brier_score",
            "accuracy",
            "ece",
        ]:
            print(
                "  "
                f"{metric_name}: "
                f"{format_metric_mean_std(metrics[f'{metric_name}_mean'], metrics[f'{metric_name}_std'])}"
            )


def _format_share(count: int, share: float | None, total: int) -> str:
    if share is None:
        return f"{count}/{total} (n/a)"
    return f"{count}/{total} ({share:.1%})"


def main(argv: list[str] | None = None) -> int:
    """Run rolling-origin baseline model backtests."""
    args = parse_args(argv)
    try:
        baseline_matches = load_baseline_training_matches()
    except FileNotFoundError as error:
        print(f"Missing historical results data: {error}")
        return 1

    features = build_features_for_backtest(
        baseline_matches,
        include_elo=args.include_elo,
        elo_k_factor=args.elo_k_factor,
        elo_home_advantage=args.elo_home_advantage,
    )
    results = run_rolling_origin_backtest(features)
    summary = summarize_backtest_results(results)
    aggregate = aggregate_backtest_results(summary)

    print("Rolling-Origin Baseline Backtest")
    print("================================")
    print(f"include_elo: {args.include_elo}")
    if args.include_elo:
        print(f"elo_k_factor: {args.elo_k_factor:g}")
        print(f"elo_home_advantage: {args.elo_home_advantage:g}")
    print(f"splits: {aggregate['split_count']}")
    print(f"evaluated splits: {aggregate['evaluated_split_count']}")

    display_columns = [
        "split_id",
        "train_start_date",
        "train_end_date",
        "test_start_date",
        "test_end_date",
        "test_row_count",
        "class_prior_log_loss",
        "logistic_regression_log_loss",
        "calibrated_logistic_regression_log_loss",
        "logistic_regression_ece",
        "calibrated_logistic_regression_ece",
    ]
    print("\nSplit Metrics")
    print("=============")
    if summary.empty:
        print("No rolling-origin splits were available.")
    else:
        print(summary[display_columns].to_string(index=False))

    _print_aggregate_metrics(aggregate)

    evaluated_total = int(aggregate["evaluated_split_count"])
    print("\nModel Comparisons")
    print("=================")
    print(
        "logistic beats class-prior on log loss: "
        + _format_share(
            int(aggregate["logistic_beats_class_prior_log_loss_count"]),
            aggregate["logistic_beats_class_prior_log_loss_share"],
            evaluated_total,
        )
    )
    print(
        "calibrated logistic beats logistic on log loss: "
        + _format_share(
            int(aggregate["calibrated_beats_logistic_log_loss_count"]),
            aggregate["calibrated_beats_logistic_log_loss_share"],
            evaluated_total,
        )
    )
    print(
        "calibrated logistic beats logistic on ECE: "
        + _format_share(
            int(aggregate["calibrated_beats_logistic_ece_count"]),
            aggregate["calibrated_beats_logistic_ece_share"],
            evaluated_total,
        )
    )
    print(
        "best model by mean log loss: "
        f"{aggregate['best_overall_model_by_mean_log_loss']}"
    )

    print("\nMessages")
    print("========")
    for message in aggregate["messages"]:
        print(f"- {message}")

    print("\nNo model artifacts or processed feature files were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
