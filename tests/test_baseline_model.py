import numpy as np
import pandas as pd

from src.models.baseline import (
    DEFAULT_CLASS_LABELS,
    build_class_prior_baseline,
    build_logistic_regression_pipeline,
    predict_class_prior,
    train_baseline_model,
)


def make_modeling_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [f"m{index}" for index in range(9)],
            "match_date": [
                "2021-01-01",
                "2021-02-01",
                "2021-03-01",
                "2021-04-01",
                "2021-05-01",
                "2021-06-01",
                "2022-01-01",
                "2022-02-01",
                "2022-03-01",
            ],
            "team_a": ["A"] * 9,
            "team_b": ["B"] * 9,
            "tournament": ["Friendly"] * 9,
            "is_neutral": [False] * 9,
            "result": [
                "team_a_win",
                "draw",
                "team_b_win",
                "team_a_win",
                "draw",
                "team_b_win",
                "team_a_win",
                "draw",
                "team_b_win",
            ],
            "team_a_feature": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 1_000.0, 1_000.0, 1_000.0],
            "team_b_feature": [10.0, 8.0, 6.0, 4.0, 2.0, 0.0, -1_000.0, -1_000.0, -1_000.0],
            "feature_diff": [-10.0, -6.0, -2.0, 2.0, 6.0, 10.0, 2_000.0, 2_000.0, 2_000.0],
        }
    )


def test_class_prior_probabilities_sum_to_one() -> None:
    priors = build_class_prior_baseline(
        pd.Series(["team_a_win", "draw", "team_a_win"]),
        DEFAULT_CLASS_LABELS,
    )

    assert np.isclose(priors.sum(), 1.0)


def test_class_prior_predictions_have_expected_shape() -> None:
    priors = np.array([0.5, 0.25, 0.25])
    predictions = predict_class_prior(4, priors)

    assert predictions.shape == (4, 3)
    assert np.allclose(predictions[0], priors)
    assert np.allclose(predictions[3], priors)


def test_logistic_pipeline_can_fit_and_predict_small_dataset() -> None:
    model = build_logistic_regression_pipeline()
    x_train = pd.DataFrame(
        {
            "feature_one": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_two": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        }
    )
    y_train = pd.Series(
        ["team_a_win", "draw", "team_b_win", "team_a_win", "draw", "team_b_win"]
    )

    model.fit(x_train, y_train)
    predictions = model.predict_proba(x_train)

    assert predictions.shape == (6, 3)
    assert np.allclose(predictions.sum(axis=1), 1.0)


def test_train_baseline_model_returns_class_prior_and_logistic_metrics() -> None:
    result = train_baseline_model(make_modeling_df(), test_start_date="2022-01-01")

    assert "class_prior_metrics" in result
    assert "logistic_regression_metrics" in result
    assert result["class_prior_metrics"]["prediction_count"] == 3
    assert result["logistic_regression_metrics"]["prediction_count"] == 3


def test_train_test_split_is_time_aware() -> None:
    result = train_baseline_model(make_modeling_df(), test_start_date="2022-01-01")

    assert result["train_row_count"] == 6
    assert result["test_row_count"] == 3
    assert result["train_date_range"] == ("2021-01-01", "2021-06-01")
    assert result["test_date_range"] == ("2022-01-01", "2022-03-01")


def test_feature_columns_exclude_target_date_team_and_tournament_metadata() -> None:
    result = train_baseline_model(make_modeling_df(), test_start_date="2022-01-01")

    assert result["feature_columns"] == [
        "team_a_feature",
        "team_b_feature",
        "feature_diff",
    ]


def test_imputer_is_fitted_on_train_rows_only() -> None:
    df = make_modeling_df()
    df.loc[2, "team_a_feature"] = np.nan

    result = train_baseline_model(df, test_start_date="2022-01-01")
    imputer = result["fitted_pipeline"].named_steps["imputer"]

    assert imputer.statistics_[0] == 6.0
