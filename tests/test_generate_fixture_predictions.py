from pathlib import Path

import numpy as np
import pandas as pd

from scripts import generate_fixture_predictions as generator
from src.data.tournament_fixtures import normalize_tournament_fixtures
from src.simulation.tournament import validate_fixture_probability_table

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
    return normalize_tournament_fixtures(
        pd.DataFrame(
            [
                {
                    "match_id": "wc2026_a1",
                    "match_date": "2026-06-12",
                    "team_a": "Team 0",
                    "team_b": "Team 3",
                    "group": "A",
                    "stage": "group",
                }
            ]
        )
    )


def make_live_fixtures() -> pd.DataFrame:
    return normalize_tournament_fixtures(
        pd.DataFrame(
            [
                {
                    "match_id": "wc2026_a1",
                    "match_date": "2026-06-12",
                    "team_a": "Team 0",
                    "team_b": "Team 3",
                    "group": "A",
                    "stage": "group",
                },
                {
                    "match_id": "wc2026_a2",
                    "match_date": "2026-06-16",
                    "team_a": "Team 1",
                    "team_b": "Team 4",
                    "group": "A",
                    "stage": "group",
                },
            ]
        )
    )


def make_live_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "wc2026_a1",
                "match_date": "2026-06-12",
                "team_a": "Team 0",
                "team_b": "Team 3",
                "team_a_goals": 1,
                "team_b_goals": 0,
                "result": "team_a_win",
                "status": "completed",
            }
        ]
    )


def test_generate_fixture_predictions_runs_on_synthetic_fixture_data() -> None:
    predictions = generator.generate_fixture_predictions(
        make_training_matches(),
        make_fixtures(),
        generated_at="2026-06-01T00:00:00+00:00",
    )

    assert len(predictions) == 1
    assert predictions.loc[0, "match_id"] == "wc2026_a1"
    assert predictions.loc[0, "selected_baseline_label"] == (
        generator.SELECTED_BASELINE_LABEL
    )
    assert predictions.loc[0, "forecast_mode"] == "pre_tournament"
    assert predictions.loc[0, "feature_cutoff_date"] == "2026-06-10"


def test_generated_probabilities_sum_to_one() -> None:
    predictions = generator.generate_fixture_predictions(
        make_training_matches(),
        make_fixtures(),
        generated_at="2026-06-01T00:00:00+00:00",
    )

    probability_sum = predictions[
        ["p_team_a_win", "p_draw", "p_team_b_win"]
    ].sum(axis=1)

    assert np.isclose(probability_sum.iloc[0], 1.0)


def test_past_fixture_dates_are_marked_backfilled() -> None:
    predictions = generator.generate_fixture_predictions(
        make_training_matches(),
        make_fixtures(),
        generated_at="2026-06-13T00:00:00+00:00",
    )

    assert bool(predictions.loc[0, "is_backfilled"]) is True
    assert predictions.loc[0, "forecast_mode"] == "backfilled_ex_ante"
    assert predictions.loc[0, "feature_cutoff_date"] == "2026-06-10"


def test_explicit_backfilled_ex_ante_uses_training_cutoff_by_default() -> None:
    predictions = generator.generate_fixture_predictions(
        make_training_matches(),
        make_fixtures(),
        forecast_mode="backfilled_ex_ante",
        generated_at="2026-06-01T00:00:00+00:00",
    )

    assert predictions.loc[0, "forecast_mode"] == "backfilled_ex_ante"
    assert predictions.loc[0, "feature_cutoff_date"] == "2026-06-10"


def test_optional_output_writing_only_happens_when_output_is_provided(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    fixture_path = tmp_path / "fixtures_2026.csv"
    make_fixtures().to_csv(fixture_path, index=False)
    monkeypatch.setattr(
        generator,
        "load_baseline_training_matches",
        lambda: make_training_matches(),
    )

    result = generator.main(["--fixtures", str(fixture_path)])
    capsys.readouterr()

    assert result == 0
    assert not (tmp_path / "fixture_predictions_2026.csv").exists()


def test_optional_output_writes_when_output_is_provided(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    fixture_path = tmp_path / "fixtures_2026.csv"
    output_path = tmp_path / "fixture_predictions_2026.csv"
    make_fixtures().to_csv(fixture_path, index=False)
    monkeypatch.setattr(
        generator,
        "load_baseline_training_matches",
        lambda: make_training_matches(),
    )

    result = generator.main(
        ["--fixtures", str(fixture_path), "--output", str(output_path)]
    )
    capsys.readouterr()

    assert result == 0
    assert output_path.exists()


def test_live_mode_fails_gracefully_without_results_file(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    fixture_path = tmp_path / "fixtures_2026.csv"
    missing_results_path = tmp_path / "results_2026.csv"
    make_fixtures().to_csv(fixture_path, index=False)
    monkeypatch.setattr(
        generator,
        "load_baseline_training_matches",
        lambda: make_training_matches(),
    )
    monkeypatch.setattr(generator, "RESULTS_2026_PATH", str(missing_results_path))

    result = generator.main(
        ["--fixtures", str(fixture_path), "--forecast-mode", "live"]
    )
    output = capsys.readouterr().out

    assert result == 1
    assert "Live forecast mode requires" in output
    assert "No live predictions were generated" in output


def test_live_mode_uses_max_completed_result_date_as_default_cutoff() -> None:
    predictions = generator.generate_fixture_predictions(
        make_training_matches(),
        make_live_fixtures(),
        completed_results=make_live_results(),
        forecast_mode="live",
        generated_at="2026-06-13T00:00:00+00:00",
    )

    assert set(predictions["match_id"]) == {"wc2026_a2"}
    assert predictions["forecast_mode"].eq("live").all()
    assert predictions["feature_cutoff_date"].eq("2026-06-12").all()
    assert predictions["is_backfilled"].eq(False).all()


def test_live_mode_trains_only_on_baseline_training_matches(monkeypatch) -> None:
    captured: dict[str, set[str]] = {}
    original_train = generator.train_selected_baseline

    def capture_training_matches(training_matches: pd.DataFrame):
        captured["training_match_ids"] = set(training_matches["match_id"].astype(str))
        return original_train(training_matches)

    monkeypatch.setattr(
        generator,
        "train_selected_baseline",
        capture_training_matches,
    )

    generator.generate_fixture_predictions(
        make_training_matches(),
        make_live_fixtures(),
        completed_results=make_live_results(),
        forecast_mode="live",
        generated_at="2026-06-13T00:00:00+00:00",
    )

    assert "wc2026_a1" not in captured["training_match_ids"]


def test_live_completed_results_are_used_for_feature_history(monkeypatch) -> None:
    captured: dict[str, set[str]] = {}
    original_build = generator.build_fixture_feature_rows

    def capture_feature_history(completed_matches: pd.DataFrame, *args, **kwargs):
        captured["feature_history_ids"] = set(completed_matches["match_id"].astype(str))
        return original_build(completed_matches, *args, **kwargs)

    monkeypatch.setattr(generator, "build_fixture_feature_rows", capture_feature_history)

    generator.generate_fixture_predictions(
        make_training_matches(),
        make_live_fixtures(),
        completed_results=make_live_results(),
        forecast_mode="live",
        generated_at="2026-06-13T00:00:00+00:00",
    )

    assert "wc2026_a1" in captured["feature_history_ids"]


def test_live_mode_can_include_completed_rows_for_audit_when_requested() -> None:
    predictions = generator.generate_fixture_predictions(
        make_training_matches(),
        make_live_fixtures(),
        completed_results=make_live_results(),
        forecast_mode="live",
        generated_at="2026-06-13T00:00:00+00:00",
        include_completed_for_audit=True,
    )

    assert set(predictions["match_id"]) == {"wc2026_a1", "wc2026_a2"}
    completed_row = predictions.loc[predictions["match_id"].eq("wc2026_a1")].iloc[0]
    assert bool(completed_row["is_backfilled"]) is True


def test_simulate_group_stage_can_consume_fixture_predictions_style_dataframe() -> None:
    predictions = generator.generate_fixture_predictions(
        make_training_matches(),
        make_fixtures(),
        generated_at="2026-06-01T00:00:00+00:00",
    )

    validated = validate_fixture_probability_table(predictions)

    assert len(validated) == 1
