"""Compare simple Elo K-factor and home-advantage variants."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_elo_feature_set import (  # noqa: E402
    SELECTED_BACKTEST_PREFIX,
    evaluate_feature_set,
)
from src.data.pipeline import load_baseline_training_matches  # noqa: E402
from src.models.backtest import format_metric_mean_std  # noqa: E402

DEFAULT_K_FACTORS = (10.0, 20.0, 30.0)
DEFAULT_HOME_ADVANTAGES = (0.0, 50.0, 75.0, 100.0)
CURRENT_SIMPLE_ELO = {"elo_k_factor": 20.0, "elo_home_advantage": 0.0}


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


def _metric_column(metric_name: str) -> str:
    """Return the selected-model summary column for a metric."""
    return f"{SELECTED_BACKTEST_PREFIX}_{metric_name}"


def _windows_beating_baseline(
    variant_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    metric_name: str = "log_loss",
) -> tuple[int, float | None]:
    """Count rolling windows where variant metric beats simple-Elo baseline."""
    column = _metric_column(metric_name)
    merged = baseline_summary[
        ["split_id", "test_start_date", "test_end_date", column]
    ].merge(
        variant_summary[["split_id", "test_start_date", "test_end_date", column]],
        on=["split_id", "test_start_date", "test_end_date"],
        suffixes=("_baseline", "_variant"),
    )
    if merged.empty:
        return (0, None)

    valid = merged[[f"{column}_baseline", f"{column}_variant"]].notna().all(axis=1)
    valid_merged = merged.loc[valid]
    if valid_merged.empty:
        return (0, None)

    wins = valid_merged[f"{column}_variant"] < valid_merged[f"{column}_baseline"]
    count = int(wins.sum())
    return (count, float(count / len(valid_merged)))


def _variant_result_row(
    result: dict[str, object],
    baseline_summary: pd.DataFrame,
) -> dict[str, object]:
    """Flatten one evaluated variant into a comparison row."""
    aggregate = result["backtest_aggregate"]
    summary = result["backtest_summary"]
    single = result["single_holdout"]
    audit = result["audit"]
    assert isinstance(aggregate, dict)
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(single, dict)
    assert isinstance(audit, dict)

    log_loss_mean, log_loss_std = _metric_pair(aggregate, "log_loss")
    brier_mean, brier_std = _metric_pair(aggregate, "multiclass_brier_score")
    accuracy_mean, accuracy_std = _metric_pair(aggregate, "accuracy")
    ece_mean, ece_std = _metric_pair(aggregate, "ece")
    window_wins, window_win_share = _windows_beating_baseline(
        summary,
        baseline_summary,
        metric_name="log_loss",
    )

    single_metrics = single["calibrated_logistic_metrics"]
    single_calibration = single["calibrated_logistic_calibration_summary"]
    assert isinstance(single_metrics, dict)
    assert isinstance(single_calibration, dict)

    return {
        "elo_k_factor": float(result["elo_k_factor"]),
        "elo_home_advantage": float(result["elo_home_advantage"]),
        "feature_count": int(audit["feature_count"]),
        "single_log_loss": float(single_metrics["log_loss"]),
        "single_brier": float(single_metrics["multiclass_brier_score"]),
        "single_accuracy": float(single_metrics["accuracy"]),
        "single_ece": float(single_calibration["expected_calibration_error"]),
        "rolling_log_loss_mean": log_loss_mean,
        "rolling_log_loss_std": log_loss_std,
        "rolling_brier_mean": brier_mean,
        "rolling_brier_std": brier_std,
        "rolling_accuracy_mean": accuracy_mean,
        "rolling_accuracy_std": accuracy_std,
        "rolling_ece_mean": ece_mean,
        "rolling_ece_std": ece_std,
        "windows_beating_simple_elo_log_loss": window_wins,
        "share_beating_simple_elo_log_loss": window_win_share,
    }


def evaluate_elo_variant(
    baseline_matches: pd.DataFrame,
    elo_k_factor: float,
    elo_home_advantage: float,
    windows: tuple[int, ...] = (5, 10),
    test_start_date: str = "2022-01-01",
    initial_train_end_date: str = "2014-12-31",
    test_window_months: int = 24,
    step_months: int = 24,
    final_test_end_date: str = "2026-06-10",
) -> dict[str, object]:
    """Evaluate one Elo variant with the selected calibrated-logistic pipeline."""
    result = evaluate_feature_set(
        baseline_matches,
        include_elo=True,
        elo_k_factor=elo_k_factor,
        elo_home_advantage=elo_home_advantage,
        windows=windows,
        test_start_date=test_start_date,
        initial_train_end_date=initial_train_end_date,
        test_window_months=test_window_months,
        step_months=step_months,
        final_test_end_date=final_test_end_date,
    )
    result["elo_k_factor"] = float(elo_k_factor)
    result["elo_home_advantage"] = float(elo_home_advantage)
    return result


def compare_elo_variants(
    baseline_matches: pd.DataFrame,
    k_factors: tuple[float, ...] = DEFAULT_K_FACTORS,
    home_advantages: tuple[float, ...] = DEFAULT_HOME_ADVANTAGES,
    windows: tuple[int, ...] = (5, 10),
    test_start_date: str = "2022-01-01",
    initial_train_end_date: str = "2014-12-31",
    test_window_months: int = 24,
    step_months: int = 24,
    final_test_end_date: str = "2026-06-10",
) -> dict[str, object]:
    """Evaluate a compact Elo variant grid without writing files."""
    results: list[dict[str, object]] = []
    baseline_result: dict[str, object] | None = None

    for k_factor in k_factors:
        for home_advantage in home_advantages:
            result = evaluate_elo_variant(
                baseline_matches,
                elo_k_factor=k_factor,
                elo_home_advantage=home_advantage,
                windows=windows,
                test_start_date=test_start_date,
                initial_train_end_date=initial_train_end_date,
                test_window_months=test_window_months,
                step_months=step_months,
                final_test_end_date=final_test_end_date,
            )
            results.append(result)
            if (
                float(k_factor) == CURRENT_SIMPLE_ELO["elo_k_factor"]
                and float(home_advantage) == CURRENT_SIMPLE_ELO["elo_home_advantage"]
            ):
                baseline_result = result

    if baseline_result is None:
        baseline_result = evaluate_elo_variant(
            baseline_matches,
            elo_k_factor=CURRENT_SIMPLE_ELO["elo_k_factor"],
            elo_home_advantage=CURRENT_SIMPLE_ELO["elo_home_advantage"],
            windows=windows,
            test_start_date=test_start_date,
            initial_train_end_date=initial_train_end_date,
            test_window_months=test_window_months,
            step_months=step_months,
            final_test_end_date=final_test_end_date,
        )
        results.append(baseline_result)

    baseline_summary = baseline_result["backtest_summary"]
    assert isinstance(baseline_summary, pd.DataFrame)
    summary_rows = [
        _variant_result_row(result, baseline_summary)
        for result in results
    ]
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["rolling_log_loss_mean", "rolling_brier_mean", "rolling_ece_mean"],
        kind="mergesort",
    ).reset_index(drop=True)

    best_row = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    baseline_row = summary_df.loc[
        (summary_df["elo_k_factor"] == CURRENT_SIMPLE_ELO["elo_k_factor"])
        & (
            summary_df["elo_home_advantage"]
            == CURRENT_SIMPLE_ELO["elo_home_advantage"]
        )
    ].iloc[0].to_dict()

    best_beats_baseline = (
        bool(best_row)
        and best_row["rolling_log_loss_mean"] < baseline_row["rolling_log_loss_mean"]
    )
    ece_direction = "unchanged"
    if bool(best_row):
        if best_row["rolling_ece_mean"] < baseline_row["rolling_ece_mean"]:
            ece_direction = "improved"
        elif best_row["rolling_ece_mean"] > baseline_row["rolling_ece_mean"]:
            ece_direction = "worsened"

    return {
        "variant_results": results,
        "summary": summary_df,
        "baseline_row": baseline_row,
        "best_variant": best_row,
        "best_beats_simple_elo": best_beats_baseline,
        "best_ece_direction_vs_simple_elo": ece_direction,
    }


def _format_share(count: int, share: float | None) -> str:
    """Format a count/share pair for console output."""
    if share is None:
        return f"{count} (n/a)"
    return f"{count} ({share:.1%})"


def print_variant_report(comparison: dict[str, object]) -> None:
    """Print a compact Elo variant comparison report."""
    summary = comparison["summary"]
    baseline = comparison["baseline_row"]
    best = comparison["best_variant"]
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(baseline, dict)
    assert isinstance(best, dict)

    display = summary[
        [
            "elo_k_factor",
            "elo_home_advantage",
            "feature_count",
            "single_log_loss",
            "rolling_log_loss_mean",
            "rolling_log_loss_std",
            "rolling_brier_mean",
            "rolling_accuracy_mean",
            "rolling_ece_mean",
            "windows_beating_simple_elo_log_loss",
            "share_beating_simple_elo_log_loss",
        ]
    ].copy()
    display["share_beating_simple_elo_log_loss"] = display[
        "share_beating_simple_elo_log_loss"
    ].map(lambda value: "n/a" if pd.isna(value) else f"{value:.1%}")

    print("Elo Variant Comparison")
    print("======================")
    print("grid: K = 10, 20, 30; home_advantage = 0, 50, 75, 100")
    print("selected model family: calibrated_logistic_regression")
    print("\nVariant Metrics")
    print("===============")
    print(display.to_string(index=False))

    print("\nCurrent Simple Elo Baseline")
    print("===========================")
    print(
        "K={k:g}, home_advantage={home:g}, rolling log loss={loss}, ECE={ece}".format(
            k=baseline["elo_k_factor"],
            home=baseline["elo_home_advantage"],
            loss=format_metric_mean_std(
                baseline["rolling_log_loss_mean"],
                baseline["rolling_log_loss_std"],
            ),
            ece=format_metric_mean_std(
                baseline["rolling_ece_mean"],
                baseline["rolling_ece_std"],
            ),
        )
    )

    print("\nBest Variant By Rolling-Origin Mean Log Loss")
    print("============================================")
    print(
        "K={k:g}, home_advantage={home:g}, rolling log loss={loss}, Brier={brier}, "
        "accuracy={accuracy}, ECE={ece}".format(
            k=best["elo_k_factor"],
            home=best["elo_home_advantage"],
            loss=format_metric_mean_std(
                best["rolling_log_loss_mean"],
                best["rolling_log_loss_std"],
            ),
            brier=format_metric_mean_std(
                best["rolling_brier_mean"],
                best["rolling_brier_std"],
            ),
            accuracy=format_metric_mean_std(
                best["rolling_accuracy_mean"],
                best["rolling_accuracy_std"],
            ),
            ece=format_metric_mean_std(
                best["rolling_ece_mean"],
                best["rolling_ece_std"],
            ),
        )
    )
    print(
        "windows beating simple Elo on log loss: "
        + _format_share(
            int(best["windows_beating_simple_elo_log_loss"]),
            best["share_beating_simple_elo_log_loss"],
        )
    )
    print(
        "beats current simple Elo by mean rolling-origin log loss: "
        f"{comparison['best_beats_simple_elo']}"
    )
    print(
        "ECE vs current simple Elo: "
        f"{comparison['best_ece_direction_vs_simple_elo']}"
    )
    print("\nNo model artifacts or processed feature files were written.")


def main() -> int:
    """Run the compact Elo variant grid comparison."""
    try:
        baseline_matches = load_baseline_training_matches()
    except FileNotFoundError as error:
        print(f"Missing historical results data: {error}")
        return 1

    comparison = compare_elo_variants(baseline_matches)
    print_variant_report(comparison)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
