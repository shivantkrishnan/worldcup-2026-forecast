from src.utils.config import (
    DEFAULT_TRAINING_CUTOFF_DATE,
    FIXTURES_2026_PATH,
    PROJECT_NAME,
    RAW_RESULTS_PATH,
    RESULTS_2026_PATH,
)


def test_default_training_cutoff_date() -> None:
    assert DEFAULT_TRAINING_CUTOFF_DATE == "2026-06-10"


def test_project_name() -> None:
    assert PROJECT_NAME == "World Cup 2026 Forecasting Dashboard"


def test_data_paths() -> None:
    assert RAW_RESULTS_PATH == "data/raw/results.csv"
    assert FIXTURES_2026_PATH == "data/tournament/fixtures_2026.csv"
    assert RESULTS_2026_PATH == "data/tournament/results_2026.csv"
