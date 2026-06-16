from src.utils.config import DEFAULT_TRAINING_CUTOFF_DATE, PROJECT_NAME


def test_default_training_cutoff_date() -> None:
    assert DEFAULT_TRAINING_CUTOFF_DATE == "2026-06-10"


def test_project_name() -> None:
    assert PROJECT_NAME == "World Cup 2026 Forecasting Dashboard"
