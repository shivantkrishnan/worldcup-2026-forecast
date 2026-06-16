from pathlib import Path

import pandas as pd
import pytest

from src.data.load_data import load_raw_results


def write_csv(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_load_raw_results_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    write_csv(
        csv_path,
        "\n".join(
            [
                "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral",
                "2024-01-01,Team A,Team B,2,1,Friendly,City,Country,TRUE",
            ]
        ),
    )

    results = load_raw_results(csv_path)

    assert len(results) == 1
    assert results.loc[0, "home_team"] == "Team A"


def test_load_raw_results_missing_file_message(tmp_path: Path) -> None:
    missing_path = tmp_path / "results.csv"

    with pytest.raises(FileNotFoundError) as exc_info:
        load_raw_results(missing_path)

    message = str(exc_info.value)
    assert "Manually download the international football results CSV" in message
    assert str(missing_path) in message


def test_load_raw_results_missing_columns_lists_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    write_csv(csv_path, "date,home_team\n2024-01-01,Team A\n")

    with pytest.raises(ValueError) as exc_info:
        load_raw_results(csv_path)

    message = str(exc_info.value)
    assert "Missing required results columns" in message
    assert "away_team" in message
    assert "home_score" in message


def test_load_raw_results_parses_date_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    write_csv(
        csv_path,
        "\n".join(
            [
                "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral",
                "2024-01-01,Team A,Team B,2,1,Friendly,City,Country,FALSE",
            ]
        ),
    )

    results = load_raw_results(csv_path)

    assert pd.api.types.is_datetime64_any_dtype(results["date"])
