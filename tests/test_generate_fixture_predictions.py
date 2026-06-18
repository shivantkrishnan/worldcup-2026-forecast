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


def test_simulate_group_stage_can_consume_fixture_predictions_style_dataframe() -> None:
    predictions = generator.generate_fixture_predictions(
        make_training_matches(),
        make_fixtures(),
        generated_at="2026-06-01T00:00:00+00:00",
    )

    validated = validate_fixture_probability_table(predictions)

    assert len(validated) == 1
