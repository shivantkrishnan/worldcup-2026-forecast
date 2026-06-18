from pathlib import Path

import pandas as pd

from scripts.backtest_tournament_baseline import (
    MODEL_SETUPS,
    add_tournament_year,
    extract_tournament_years,
    make_tournament_holdout_split,
    print_tournament_backtest_report,
    run_tournament_backtest,
)
from src.features.build_features import build_modeling_features

CLASS_LABELS = ["team_a_win", "draw", "team_b_win"]


def make_tournament_matches() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    match_index = 0

    for date in pd.date_range("1998-01-01", "2001-12-01", freq="MS"):
        class_index = match_index % len(CLASS_LABELS)
        rows.append(_match_row(match_index, date, "Friendly", class_index))
        match_index += 1

    for year in [2002, 2006]:
        for offset in range(6):
            date = pd.Timestamp(f"{year}-06-01") + pd.Timedelta(days=offset)
            class_index = match_index % len(CLASS_LABELS)
            rows.append(_match_row(match_index, date, "FIFA World Cup", class_index))
            match_index += 1

        for offset in range(18):
            date = pd.Timestamp(f"{year + 1}-01-01") + pd.DateOffset(months=offset)
            class_index = match_index % len(CLASS_LABELS)
            rows.append(_match_row(match_index, date, "Friendly", class_index))
            match_index += 1

    for offset in range(3):
        date = pd.Timestamp("2026-06-01") + pd.Timedelta(days=offset)
        rows.append(_match_row(match_index, date, "FIFA World Cup", match_index % 3))
        match_index += 1

    return pd.DataFrame(rows)


def _match_row(
    match_index: int,
    date: pd.Timestamp,
    tournament: str,
    class_index: int,
) -> dict[str, object]:
    result = CLASS_LABELS[class_index]
    if result == "team_a_win":
        team_a_goals, team_b_goals = 2, 0
    elif result == "draw":
        team_a_goals, team_b_goals = 1, 1
    else:
        team_a_goals, team_b_goals = 0, 2

    return {
        "match_id": f"m{match_index}",
        "match_date": date.strftime("%Y-%m-%d"),
        "team_a": f"Team {match_index % 8}",
        "team_b": f"Team {(match_index + 3) % 8}",
        "team_a_goals": team_a_goals,
        "team_b_goals": team_b_goals,
        "result": result,
        "tournament": tournament,
        "is_neutral": tournament == "FIFA World Cup",
    }


def test_tournament_year_extraction_uses_dates_and_labels() -> None:
    matches = pd.DataFrame(
        [
            {
                "match_id": "world_cup",
                "match_date": "2002-06-01",
                "tournament": "FIFA World Cup",
            },
            {
                "match_id": "friendly",
                "match_date": "2002-07-01",
                "tournament": "Friendly",
            },
        ]
    )

    with_year = add_tournament_year(matches)

    assert with_year.loc[0, "tournament_year"] == 2002
    assert pd.isna(with_year.loc[1, "tournament_year"])


def test_tournament_holdout_training_rows_are_before_tournament_start() -> None:
    features = build_modeling_features(make_tournament_matches(), windows=(2,))

    train_df, test_df, tournament_start = make_tournament_holdout_split(features, 2002)

    assert not train_df.empty
    assert not test_df.empty
    assert pd.to_datetime(train_df["match_date"]).max() < tournament_start


def test_tournament_holdout_test_rows_are_only_target_tournament_year() -> None:
    features = build_modeling_features(make_tournament_matches(), windows=(2,))

    _, test_df, _ = make_tournament_holdout_split(features, 2006)

    assert set(test_df["tournament"]) == {"FIFA World Cup"}
    assert set(pd.to_datetime(test_df["match_date"]).dt.year) == {2006}


def test_extract_tournament_years_excludes_2026_by_default() -> None:
    years = extract_tournament_years(make_tournament_matches())

    assert 2002 in years
    assert 2006 in years
    assert 2026 not in years


def test_tournament_backtest_runs_on_synthetic_data() -> None:
    report = run_tournament_backtest(
        make_tournament_matches(),
        tournament_years=(2002, 2006, 2026),
        min_test_matches=3,
        windows=(2,),
    )
    summary = report["summary"]
    aggregate = report["aggregate"]

    assert set(summary["model_setup"]) == {setup.name for setup in MODEL_SETUPS}
    assert 2026 not in set(summary["tournament_year"])
    assert aggregate["evaluated_tournament_count"] == 2


def test_tournament_backtest_output_includes_all_three_model_setups(capsys) -> None:
    report = run_tournament_backtest(
        make_tournament_matches(),
        tournament_years=(2002,),
        min_test_matches=3,
        windows=(2,),
    )

    print_tournament_backtest_report(report)
    output = capsys.readouterr().out

    assert "no_elo_calibrated_logistic" in output
    assert "simple_elo_calibrated_logistic" in output
    assert "selected_elo_calibrated_logistic" in output


def test_tournament_backtest_does_not_write_files_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    run_tournament_backtest(
        make_tournament_matches(),
        tournament_years=(2002,),
        min_test_matches=3,
        windows=(2,),
    )

    assert list(tmp_path.iterdir()) == []
