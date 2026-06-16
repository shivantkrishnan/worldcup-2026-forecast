import numpy as np
import pandas as pd

from src.models.backtest import (
    aggregate_backtest_results,
    run_rolling_origin_backtest,
    summarize_backtest_results,
    train_and_evaluate_on_split,
)


CLASS_LABELS = ["team_a_win", "draw", "team_b_win"]


def make_backtest_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2013-01-01", "2019-12-01", freq="MS")

    for index, match_date in enumerate(dates):
        class_index = index % len(CLASS_LABELS)
        rows.append(
            {
                "match_id": f"m{index}",
                "match_date": match_date.strftime("%Y-%m-%d"),
                "team_a": f"Team {index % 5}",
                "team_b": f"Opponent {index % 7}",
                "tournament": "Friendly",
                "is_neutral": False,
                "result": CLASS_LABELS[class_index],
                "team_a_feature": float(class_index) + index * 0.01,
                "team_b_feature": float(2 - class_index) - index * 0.01,
                "feature_diff": float(class_index - 1),
            }
        )

    return pd.DataFrame(rows)


def make_train_test_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = make_backtest_df()
    dates = pd.to_datetime(df["match_date"])
    train_df = df.loc[dates <= "2014-12-31"].reset_index(drop=True)
    test_df = df.loc[
        (dates >= "2015-01-01") & (dates <= "2015-12-31")
    ].reset_index(drop=True)
    return train_df, test_df


def test_train_and_evaluate_on_split_returns_expected_model_metrics() -> None:
    train_df, test_df = make_train_test_split()

    result = train_and_evaluate_on_split(train_df, test_df)

    assert result["train_row_count"] == 24
    assert result["test_row_count"] == 12
    assert result["feature_count"] == 3
    assert result["class_prior_metrics"]["prediction_count"] == 12
    assert result["logistic_regression_metrics"]["prediction_count"] == 12
    assert result["calibrated_logistic_regression_metrics"]["prediction_count"] == 12
    assert "expected_calibration_error" in result["logistic_calibration_summary"]
    assert "expected_calibration_error" in result[
        "calibrated_logistic_calibration_summary"
    ]


def test_run_rolling_origin_backtest_returns_multiple_split_results() -> None:
    results = run_rolling_origin_backtest(
        make_backtest_df(),
        initial_train_end_date="2014-12-31",
        test_window_months=12,
        step_months=12,
        final_test_end_date="2018-12-31",
    )

    assert len(results) == 4
    assert all(not result["skipped"] for result in results)


def test_summarize_backtest_results_returns_one_row_per_split() -> None:
    results = run_rolling_origin_backtest(
        make_backtest_df(),
        initial_train_end_date="2014-12-31",
        test_window_months=12,
        step_months=12,
        final_test_end_date="2018-12-31",
    )

    summary = summarize_backtest_results(results)

    assert len(summary) == len(results)
    assert "logistic_regression_log_loss" in summary.columns
    assert "calibrated_logistic_regression_ece" in summary.columns


def test_aggregate_backtest_results_returns_metrics_and_comparison_counts() -> None:
    results = run_rolling_origin_backtest(
        make_backtest_df(),
        initial_train_end_date="2014-12-31",
        test_window_months=12,
        step_months=12,
        final_test_end_date="2018-12-31",
    )
    summary = summarize_backtest_results(results)

    aggregate = aggregate_backtest_results(summary)

    assert aggregate["split_count"] == 4
    assert aggregate["evaluated_split_count"] == 4
    assert aggregate["model_metrics"]["class_prior"]["log_loss_mean"] is not None
    assert aggregate["model_metrics"]["logistic_regression"]["accuracy_std"] is not None
    assert "logistic_beats_class_prior_log_loss_count" in aggregate
    assert "calibrated_beats_logistic_log_loss_count" in aggregate
    assert "best_overall_model_by_mean_log_loss" in aggregate


def test_rolling_backtest_splits_are_time_ordered_and_future_only() -> None:
    results = run_rolling_origin_backtest(
        make_backtest_df(),
        initial_train_end_date="2014-12-31",
        test_window_months=12,
        step_months=12,
        final_test_end_date="2018-12-31",
    )

    for result in results:
        train_end = pd.Timestamp(result["train_date_range"][1])
        test_start = pd.Timestamp(result["test_date_range"][0])
        assert train_end < test_start


def test_backtest_functions_do_not_mutate_input_dataframes() -> None:
    df = make_backtest_df()
    original = df.copy(deep=True)
    train_df, test_df = make_train_test_split()
    original_train = train_df.copy(deep=True)
    original_test = test_df.copy(deep=True)

    train_and_evaluate_on_split(train_df, test_df)
    run_rolling_origin_backtest(
        df,
        initial_train_end_date="2014-12-31",
        test_window_months=12,
        step_months=12,
        final_test_end_date="2018-12-31",
    )

    pd.testing.assert_frame_equal(train_df, original_train)
    pd.testing.assert_frame_equal(test_df, original_test)
    pd.testing.assert_frame_equal(df, original)


def test_include_calibrated_false_still_works() -> None:
    train_df, test_df = make_train_test_split()

    result = train_and_evaluate_on_split(
        train_df,
        test_df,
        include_calibrated=False,
    )

    assert result["calibrated_logistic_regression_metrics"] is None
    assert result["calibrated_logistic_calibration_summary"] is None

    results = run_rolling_origin_backtest(
        make_backtest_df(),
        initial_train_end_date="2014-12-31",
        test_window_months=12,
        step_months=12,
        final_test_end_date="2016-12-31",
        include_calibrated=False,
    )
    summary = summarize_backtest_results(results)

    assert np.isnan(summary["calibrated_logistic_regression_log_loss"]).all()


def test_invalid_split_is_skipped_with_clear_message() -> None:
    df = make_backtest_df()
    df["result"] = "team_a_win"

    results = run_rolling_origin_backtest(
        df,
        initial_train_end_date="2014-12-31",
        test_window_months=12,
        step_months=12,
        final_test_end_date="2015-12-31",
    )

    assert len(results) == 1
    assert results[0]["skipped"]
    assert "missing target classes" in results[0]["messages"][0]
