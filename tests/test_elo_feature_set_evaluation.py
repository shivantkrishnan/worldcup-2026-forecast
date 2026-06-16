import pandas as pd

from scripts.audit_features import (
    build_features_for_audit,
    parse_args as parse_audit_args,
)
from scripts.backtest_baseline_models import (
    build_features_for_backtest,
    parse_args as parse_backtest_args,
)
from scripts.compare_elo_feature_set import (
    compare_feature_sets,
    print_comparison_report,
)
from scripts.train_baseline_model import (
    build_features_for_training,
    parse_args as parse_training_args,
)
from src.features.elo import ELO_FEATURE_COLUMNS
from src.features.feature_audit import audit_feature_readiness, get_feature_columns
from src.models.backtest import run_rolling_origin_backtest
from src.models.baseline import train_baseline_model

CLASS_LABELS = ["team_a_win", "draw", "team_b_win"]


def make_canonical_matches() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2018-01-01", "2023-12-01", freq="MS")

    for index, match_date in enumerate(dates):
        result = CLASS_LABELS[index % len(CLASS_LABELS)]
        if result == "team_a_win":
            team_a_goals, team_b_goals = 2, 0
        elif result == "draw":
            team_a_goals, team_b_goals = 1, 1
        else:
            team_a_goals, team_b_goals = 0, 2

        rows.append(
            {
                "match_id": f"m{index}",
                "match_date": match_date.strftime("%Y-%m-%d"),
                "team_a": f"Team {index % 6}",
                "team_b": f"Team {(index + 2) % 6}",
                "team_a_goals": team_a_goals,
                "team_b_goals": team_b_goals,
                "result": result,
                "tournament": "Friendly",
                "is_neutral": False,
            }
        )

    return pd.DataFrame(rows)


def test_cli_parsers_default_to_no_elo_and_accept_include_elo() -> None:
    assert parse_audit_args([]).include_elo is False
    assert parse_training_args([]).include_elo is False
    assert parse_backtest_args([]).include_elo is False

    assert parse_audit_args(["--include-elo"]).include_elo is True
    assert parse_training_args(["--include-elo"]).include_elo is True
    assert parse_backtest_args(["--include-elo"]).include_elo is True


def test_audit_features_can_include_elo_features() -> None:
    features = build_features_for_audit(make_canonical_matches(), include_elo=True)
    report = audit_feature_readiness(features, test_start_date="2021-01-01")

    for column in ELO_FEATURE_COLUMNS:
        assert column in report["feature_columns"]


def test_default_feature_builders_remain_no_elo() -> None:
    features = build_features_for_training(make_canonical_matches())

    for column in ELO_FEATURE_COLUMNS:
        assert column not in features.columns


def test_train_baseline_model_can_run_with_elo_features() -> None:
    features = build_features_for_training(make_canonical_matches(), include_elo=True)

    result = train_baseline_model(features, test_start_date="2021-01-01")

    assert result["calibrated_logistic_metrics"]["prediction_count"] > 0
    for column in ELO_FEATURE_COLUMNS:
        assert column in result["feature_columns"]


def test_backtest_can_run_with_elo_features() -> None:
    features = build_features_for_backtest(make_canonical_matches(), include_elo=True)

    results = run_rolling_origin_backtest(
        features,
        initial_train_end_date="2019-12-31",
        test_window_months=12,
        step_months=12,
        final_test_end_date="2022-12-31",
    )

    assert len(results) == 3
    assert any(not result["skipped"] for result in results)


def test_compare_elo_feature_set_produces_comparison_output(capsys) -> None:
    comparison = compare_feature_sets(
        make_canonical_matches(),
        windows=(2,),
        test_start_date="2021-01-01",
        initial_train_end_date="2019-12-31",
        test_window_months=12,
        step_months=12,
        final_test_end_date="2022-12-31",
    )

    print_comparison_report(comparison)
    output = capsys.readouterr().out

    assert "Elo Feature Set Comparison" in output
    assert "no-Elo feature count" in output
    assert "Elo improves selected baseline by mean rolling-origin log loss" in output


def test_elo_feature_columns_are_numeric_candidate_features() -> None:
    features = build_features_for_audit(make_canonical_matches(), include_elo=True)
    feature_columns = get_feature_columns(features)

    for column in ELO_FEATURE_COLUMNS:
        assert column in feature_columns


def test_include_elo_feature_building_does_not_mutate_input_dataframe() -> None:
    canonical = make_canonical_matches()
    original = canonical.copy(deep=True)

    build_features_for_training(canonical, include_elo=True)

    pd.testing.assert_frame_equal(canonical, original)
