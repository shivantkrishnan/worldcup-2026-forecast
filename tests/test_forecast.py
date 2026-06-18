from pathlib import Path

import pandas as pd

from src.features.fixture_features import build_fixture_feature_rows
from src.models.forecast import (
    format_prediction_output,
    predict_fixture_probabilities,
    train_selected_baseline,
)

CLASS_LABELS = ["team_a_win", "draw", "team_b_win"]


def make_training_matches() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2020-01-01", "2023-12-01", freq="MS")
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
                "team_a": f"Team {index % 8}",
                "team_b": f"Team {(index + 3) % 8}",
                "team_a_goals": team_a_goals,
                "team_b_goals": team_b_goals,
                "result": result,
                "tournament": "Friendly",
                "is_neutral": False,
            }
        )
    return pd.DataFrame(rows)


def make_fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "fixture_1",
                "match_date": "2024-06-01",
                "team_a": "Team 0",
                "team_b": "Team 3",
                "tournament": "FIFA World Cup",
            }
        ]
    )


def test_favorite_display_maps_all_outcomes() -> None:
    predictions = pd.DataFrame(
        [
            {"p_team_a_win": 0.60, "p_draw": 0.25, "p_team_b_win": 0.15},
            {"p_team_a_win": 0.20, "p_draw": 0.50, "p_team_b_win": 0.30},
            {"p_team_a_win": 0.10, "p_draw": 0.25, "p_team_b_win": 0.65},
        ]
    )
    fixtures = pd.DataFrame(
        [
            {
                "match_id": "f1",
                "match_date": "2026-06-20",
                "team_a": "Alpha",
                "team_b": "Beta",
            },
            {
                "match_id": "f2",
                "match_date": "2026-06-21",
                "team_a": "Gamma",
                "team_b": "Delta",
            },
            {
                "match_id": "f3",
                "match_date": "2026-06-22",
                "team_a": "Epsilon",
                "team_b": "Zeta",
            },
        ]
    )

    output = format_prediction_output(predictions, fixtures)

    assert output["predicted_class"].tolist() == [
        "team_a_win",
        "draw",
        "team_b_win",
    ]
    assert output["favorite_display"].tolist() == ["Alpha", "Draw", "Zeta"]


def test_selected_forecast_defaults_use_selected_elo_variant() -> None:
    bundle = train_selected_baseline(make_training_matches())

    assert bundle.include_elo is True
    assert bundle.elo_k_factor == 10.0
    assert bundle.elo_home_advantage == 50.0


def test_predicted_probabilities_sum_to_one() -> None:
    training_matches = make_training_matches()
    fixtures = make_fixtures()
    bundle = train_selected_baseline(training_matches)
    fixture_features = build_fixture_feature_rows(
        training_matches,
        fixtures,
        include_elo=bundle.include_elo,
        elo_k_factor=bundle.elo_k_factor,
        elo_home_advantage=bundle.elo_home_advantage,
    )

    predictions = predict_fixture_probabilities(bundle, fixture_features)

    assert predictions.shape == (1, 3)
    assert predictions.sum(axis=1).iloc[0] == 1.0


def test_forecast_workflow_does_not_write_files_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    training_matches = make_training_matches()
    fixtures = make_fixtures()

    bundle = train_selected_baseline(training_matches)
    fixture_features = build_fixture_feature_rows(
        training_matches,
        fixtures,
        include_elo=bundle.include_elo,
        elo_k_factor=bundle.elo_k_factor,
        elo_home_advantage=bundle.elo_home_advantage,
    )
    predictions = predict_fixture_probabilities(bundle, fixture_features)
    format_prediction_output(predictions, fixtures)

    assert list(tmp_path.iterdir()) == []
