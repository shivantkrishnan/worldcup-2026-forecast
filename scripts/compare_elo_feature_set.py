"""Compare rolling-form-only features against rolling-form-plus-Elo features."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import load_baseline_training_matches  # noqa: E402
from src.features.build_features import build_modeling_features  # noqa: E402
from src.features.feature_audit import audit_feature_readiness  # noqa: E402
from src.models.backtest import (  # noqa: E402
    aggregate_backtest_results,
    format_metric_mean_std,
    run_rolling_origin_backtest,
    summarize_backtest_results,
)
from src.models.baseline import train_baseline_model  # noqa: E402

SELECTED_SINGLE_METRICS_KEY = "calibrated_logistic_metrics"
SELECTED_BACKTEST_PREFIX = "calibrated_logistic_regression"
METRIC_LABELS = {
    "log_loss": "log loss",
    "multiclass_brier_score": "Brier",
    "accuracy": "accuracy",
    "ece": "ECE",
}


def _comparison_count(
    no_elo_summary: pd.DataFrame,
    elo_summary: pd.DataFrame,
    metric_column: str,
) -> tuple[int, float | None]:
    """Return count/share of rolling windows where Elo improves the metric."""
    merged = no_elo_summary[
        ["split_id", "test_start_date", "test_end_date", metric_column]
    ].merge(
        elo_summary[["split_id", "test_start_date", "test_end_date", metric_column]],
        on=["split_id", "test_start_date", "test_end_date"],
        suffixes=("_no_elo", "_elo"),
    )
    if merged.empty:
        return (0, None)

    valid = merged[[f"{metric_column}_no_elo", f"{metric_column}_elo"]].notna().all(
        axis=1
    )
    valid_merged = merged.loc[valid]
    if valid_merged.empty:
        return (0, None)

    improved = (
        valid_merged[f"{metric_column}_elo"]
        < valid_merged[f"{metric_column}_no_elo"]
    )
    count = int(improved.sum())
    return (count, float(count / len(valid_merged)))


def _metric_pair(aggregate: dict[str, object], metric_name: str) -> tuple[Any, Any]:
    """Return selected-model mean/std values from an aggregate report."""
    model_metrics = aggregate["model_metrics"]
    assert isinstance(model_metrics, dict)
    selected_metrics = model_metrics[SELECTED_BACKTEST_PREFIX]
    assert isinstance(selected_metrics, dict)
    return (
        selected_metrics[f"{metric_name}_mean"],
        selected_metrics[f"{metric_name}_std"],
    )


def _single_metric(result: dict[str, object], metric_name: str) -> float:
    """Return selected-model single-holdout metric."""
    metrics = result[SELECTED_SINGLE_METRICS_KEY]
    assert isinstance(metrics, dict)
    return float(metrics[metric_name])


def _single_ece(result: dict[str, object]) -> float:
    """Return selected-model single-holdout ECE."""
    calibration = result["calibrated_logistic_calibration_summary"]
    assert isinstance(calibration, dict)
    return float(calibration["expected_calibration_error"])


def evaluate_feature_set(
    baseline_matches: pd.DataFrame,
    include_elo: bool,
    windows: tuple[int, ...] = (5, 10),
    test_start_date: str = "2022-01-01",
    initial_train_end_date: str = "2014-12-31",
    test_window_months: int = 24,
    step_months: int = 24,
    final_test_end_date: str = "2026-06-10",
) -> dict[str, object]:
    """Build, audit, and evaluate one feature set without writing files."""
    matches = baseline_matches.copy(deep=True)
    features = build_modeling_features(
        matches,
        windows=windows,
        include_elo=include_elo,
    )
    audit = audit_feature_readiness(features, test_start_date=test_start_date)
    single_holdout = train_baseline_model(
        features,
        test_start_date=test_start_date,
    )
    backtest_results = run_rolling_origin_backtest(
        features,
        initial_train_end_date=initial_train_end_date,
        test_window_months=test_window_months,
        step_months=step_months,
        final_test_end_date=final_test_end_date,
    )
    backtest_summary = summarize_backtest_results(backtest_results)
    backtest_aggregate = aggregate_backtest_results(backtest_summary)

    return {
        "include_elo": include_elo,
        "features": features,
        "audit": audit,
        "single_holdout": single_holdout,
        "backtest_results": backtest_results,
        "backtest_summary": backtest_summary,
        "backtest_aggregate": backtest_aggregate,
    }


def compare_feature_sets(
    baseline_matches: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10),
    test_start_date: str = "2022-01-01",
    initial_train_end_date: str = "2014-12-31",
    test_window_months: int = 24,
    step_months: int = 24,
    final_test_end_date: str = "2026-06-10",
) -> dict[str, object]:
    """Evaluate no-Elo and Elo feature sets and return comparison results."""
    no_elo = evaluate_feature_set(
        baseline_matches,
        include_elo=False,
        windows=windows,
        test_start_date=test_start_date,
        initial_train_end_date=initial_train_end_date,
        test_window_months=test_window_months,
        step_months=step_months,
        final_test_end_date=final_test_end_date,
    )
    with_elo = evaluate_feature_set(
        baseline_matches,
        include_elo=True,
        windows=windows,
        test_start_date=test_start_date,
        initial_train_end_date=initial_train_end_date,
        test_window_months=test_window_months,
        step_months=step_months,
        final_test_end_date=final_test_end_date,
    )

    no_elo_summary = no_elo["backtest_summary"]
    with_elo_summary = with_elo["backtest_summary"]
    assert isinstance(no_elo_summary, pd.DataFrame)
    assert isinstance(with_elo_summary, pd.DataFrame)

    comparison_columns = {
        "log_loss": f"{SELECTED_BACKTEST_PREFIX}_log_loss",
        "multiclass_brier_score": f"{SELECTED_BACKTEST_PREFIX}_multiclass_brier_score",
        "ece": f"{SELECTED_BACKTEST_PREFIX}_ece",
    }
    improvement_counts = {
        metric_name: _comparison_count(no_elo_summary, with_elo_summary, column)
        for metric_name, column in comparison_columns.items()
    }

    no_elo_log_loss_mean, _ = _metric_pair(
        no_elo["backtest_aggregate"],
        "log_loss",
    )
    with_elo_log_loss_mean, _ = _metric_pair(
        with_elo["backtest_aggregate"],
        "log_loss",
    )
    elo_improves_mean_log_loss = (
        with_elo_log_loss_mean is not None
        and no_elo_log_loss_mean is not None
        and with_elo_log_loss_mean < no_elo_log_loss_mean
    )

    return {
        "no_elo": no_elo,
        "with_elo": with_elo,
        "improvement_counts": improvement_counts,
        "elo_improves_mean_rolling_log_loss": elo_improves_mean_log_loss,
    }


def _format_share(count: int, share: float | None) -> str:
    if share is None:
        return f"{count} (n/a)"
    return f"{count} ({share:.1%})"


def _print_single_holdout_table(comparison: dict[str, object]) -> None:
    print("\nSingle-Holdout Selected-Model Metrics")
    print("=====================================")
    rows = []
    for label, key in [("no_elo", "no_elo"), ("with_elo", "with_elo")]:
        feature_set = comparison[key]
        assert isinstance(feature_set, dict)
        result = feature_set["single_holdout"]
        assert isinstance(result, dict)
        rows.append(
            {
                "feature_set": label,
                "log_loss": _single_metric(result, "log_loss"),
                "brier": _single_metric(result, "multiclass_brier_score"),
                "accuracy": _single_metric(result, "accuracy"),
                "ece": _single_ece(result),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


def _print_rolling_table(comparison: dict[str, object]) -> None:
    print("\nRolling-Origin Selected-Model Aggregate Metrics")
    print("===============================================")
    rows = []
    for label, key in [("no_elo", "no_elo"), ("with_elo", "with_elo")]:
        feature_set = comparison[key]
        assert isinstance(feature_set, dict)
        aggregate = feature_set["backtest_aggregate"]
        assert isinstance(aggregate, dict)
        row = {"feature_set": label}
        for metric_name in ["log_loss", "multiclass_brier_score", "accuracy", "ece"]:
            mean_value, std_value = _metric_pair(aggregate, metric_name)
            row[METRIC_LABELS[metric_name]] = format_metric_mean_std(
                mean_value,
                std_value,
            )
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


def print_comparison_report(comparison: dict[str, object]) -> None:
    """Print a compact no-Elo vs Elo comparison report."""
    no_elo = comparison["no_elo"]
    with_elo = comparison["with_elo"]
    assert isinstance(no_elo, dict)
    assert isinstance(with_elo, dict)

    print("Elo Feature Set Comparison")
    print("==========================")
    print("selected model family: calibrated_logistic_regression")
    print(f"no-Elo feature count: {no_elo['audit']['feature_count']}")
    print(f"Elo feature count: {with_elo['audit']['feature_count']}")

    _print_single_holdout_table(comparison)
    _print_rolling_table(comparison)

    print("\nRolling-Origin Elo Improvement Counts")
    print("=====================================")
    improvement_counts = comparison["improvement_counts"]
    assert isinstance(improvement_counts, dict)
    for metric_name in ["log_loss", "multiclass_brier_score", "ece"]:
        count, share = improvement_counts[metric_name]
        print(f"{METRIC_LABELS[metric_name]}: {_format_share(count, share)}")

    print("\nSelection Signal")
    print("================")
    print(
        "Elo improves selected baseline by mean rolling-origin log loss: "
        f"{comparison['elo_improves_mean_rolling_log_loss']}"
    )
    print("\nNo model artifacts or processed feature files were written.")


def main() -> int:
    """Run the Elo feature-set comparison."""
    try:
        baseline_matches = load_baseline_training_matches()
    except FileNotFoundError as error:
        print(f"Missing historical results data: {error}")
        return 1

    comparison = compare_feature_sets(baseline_matches)
    print_comparison_report(comparison)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
