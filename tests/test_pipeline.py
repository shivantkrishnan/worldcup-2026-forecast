from pathlib import Path

import pandas as pd
import pytest

from src.data.clean_results import CANONICAL_MATCH_COLUMNS
from src.data.pipeline import (
    load_and_clean_results,
    load_and_clean_results_with_quarantine,
    load_baseline_training_matches,
)


def write_raw_results_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral",
                "2026-06-09,Team A,Team B,2,1,Friendly,City One,Country One,FALSE",
                "2026-06-11,Team C,Team D,0,3,FIFA World Cup,City Two,Country Two,TRUE",
            ]
        ),
        encoding="utf-8",
    )


def test_load_and_clean_results_returns_canonical_columns(tmp_path: Path) -> None:
    raw_path = tmp_path / "results.csv"
    write_raw_results_csv(raw_path)

    cleaned = load_and_clean_results(raw_path)

    assert cleaned.columns.tolist() == CANONICAL_MATCH_COLUMNS


def test_load_baseline_training_matches_excludes_matches_after_cutoff(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "results.csv"
    write_raw_results_csv(raw_path)

    baseline = load_baseline_training_matches(raw_path)

    assert baseline["match_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-06-09"]


def test_pipeline_does_not_mutate_raw_csv(tmp_path: Path) -> None:
    raw_path = tmp_path / "results.csv"
    write_raw_results_csv(raw_path)
    original_text = raw_path.read_text(encoding="utf-8")

    load_and_clean_results(raw_path)

    assert raw_path.read_text(encoding="utf-8") == original_text
    raw_after_pipeline = pd.read_csv(raw_path)
    assert "home_team" in raw_after_pipeline.columns
    assert "team_a" not in raw_after_pipeline.columns


def test_missing_raw_file_raises_clear_file_not_found_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "results.csv"

    with pytest.raises(FileNotFoundError) as exc_info:
        load_and_clean_results(missing_path)

    assert "Manually download the international football results CSV" in str(
        exc_info.value
    )


def test_load_and_clean_results_resolves_metadata_duplicates_by_default(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "results.csv"
    raw_path.write_text(
        "\n".join(
            [
                "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral",
                "2026-06-09,Team A,Team B,2,1,Friendly,City One,Country One,FALSE",
                "2026-06-09,Team A,Team B,2,1,Friendly,Alternate City,Country One,FALSE",
            ]
        ),
        encoding="utf-8",
    )

    cleaned = load_and_clean_results(raw_path)

    assert len(cleaned) == 1
    assert cleaned["match_id"].is_unique


def test_load_and_clean_results_with_quarantine_returns_duplicate_details(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "results.csv"
    raw_path.write_text(
        "\n".join(
            [
                "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral",
                "2026-06-09,Team A,Team B,2,1,Friendly,City One,Country One,FALSE",
                "2026-06-09,Team A,Team B,2,1,Friendly,Alternate City,Country One,FALSE",
            ]
        ),
        encoding="utf-8",
    )

    result = load_and_clean_results_with_quarantine(raw_path)

    assert len(result.resolved_matches) == 1
    assert len(result.quarantined_matches) == 1
    assert result.duplicate_report.metadata_duplicate_group_count == 1
