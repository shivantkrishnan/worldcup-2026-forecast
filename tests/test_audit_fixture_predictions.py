from pathlib import Path

import pandas as pd

from scripts import audit_fixture_predictions as audit


def make_prediction_file(path: Path) -> None:
    predictions = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "match_date": "2026-06-12",
                "group": "A",
                "team_a": "Alpha",
                "team_b": "Beta",
                "p_team_a_win": 0.70,
                "p_draw": 0.20,
                "p_team_b_win": 0.10,
                "predicted_class": "team_a_win",
                "favorite_display": "Alpha",
                "confidence_label": "High",
                "prediction_generated_at": "2026-06-18T00:00:00+00:00",
                "training_cutoff_date": "2026-06-10",
                "feature_cutoff_date": "2026-06-10",
                "forecast_mode": "backfilled_ex_ante",
                "model_name": "test_model",
                "selected_baseline_label": "test_baseline",
                "is_backfilled": True,
            },
            {
                "match_id": "m2",
                "match_date": "2026-06-18",
                "group": "A",
                "team_a": "Gamma",
                "team_b": "Delta",
                "p_team_a_win": 0.30,
                "p_draw": 0.25,
                "p_team_b_win": 0.45,
                "predicted_class": "team_b_win",
                "favorite_display": "Delta",
                "confidence_label": "Low",
                "prediction_generated_at": "2026-06-18T00:00:00+00:00",
                "training_cutoff_date": "2026-06-10",
                "feature_cutoff_date": "2026-06-10",
                "forecast_mode": "backfilled_ex_ante",
                "model_name": "test_model",
                "selected_baseline_label": "test_baseline",
                "is_backfilled": False,
            },
        ]
    )
    predictions.to_csv(path, index=False)


def test_audit_fixture_predictions_reports_core_counts(tmp_path: Path) -> None:
    prediction_path = tmp_path / "fixture_predictions_2026.csv"
    make_prediction_file(prediction_path)

    report = audit.audit_fixture_predictions(
        prediction_path,
        historical_support_counts={"Alpha": 20, "Beta": 200, "Delta": 50},
    )

    assert report["prediction_count"] == 2
    assert report["probability_sum_max_deviation"] < 1e-12
    assert report["forecast_mode_counts"] == {"backfilled_ex_ante": 2}
    assert report["backfilled_count"] == 1
    assert len(report["highest_confidence"]) == 2
    assert len(report["surprising_favorites"]) >= 1


def test_audit_fixture_predictions_script_runs_on_synthetic_file(
    tmp_path: Path,
    capsys,
) -> None:
    prediction_path = tmp_path / "fixture_predictions_2026.csv"
    make_prediction_file(prediction_path)

    result = audit.main(
        ["--predictions", str(prediction_path), "--skip-historical-support"]
    )
    output = capsys.readouterr().out

    assert result == 0
    assert "Fixture Prediction Audit" in output
    assert "predictions: 2" in output
    assert "forecast_mode counts: backfilled_ex_ante=2" in output
    assert "No files were written." in output
