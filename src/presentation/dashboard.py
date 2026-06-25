"""Presentation helpers for the Streamlit dashboard."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

PROBABILITY_COLUMNS = ["p_team_a_win", "p_draw", "p_team_b_win"]
STANDING_COLUMNS = [
    "group",
    "rank",
    "team",
    "played",
    "points",
    "wins",
    "draws",
    "losses",
    "goals_for",
    "goals_against",
    "goal_difference",
]


def format_percent(value: object) -> str:
    """Format a probability as a compact percentage for display."""
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1%}"


def format_number(value: object, decimals: int = 2) -> str:
    """Format a numeric value while preserving readable missing states."""
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{decimals}f}"


def _score_result(team_a_goals: int, team_b_goals: int) -> tuple[str, int, int, int, int]:
    """Return result label plus points/win/draw/loss for Team A."""
    if team_a_goals > team_b_goals:
        return "team_a_win", 3, 1, 0, 0
    if team_a_goals < team_b_goals:
        return "team_b_win", 0, 0, 0, 1
    return "draw", 1, 0, 1, 0


def _empty_standing_record(group: object, team: object) -> dict[str, object]:
    """Return an empty standings row for one team."""
    return {
        "group": str(group),
        "team": str(team),
        "played": 0,
        "points": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_difference": 0,
    }


def build_current_group_table(
    fixtures: pd.DataFrame,
    results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build current group standings from completed result rows only."""
    required_fixture_columns = {"match_id", "group", "team_a", "team_b"}
    missing = required_fixture_columns.difference(fixtures.columns)
    if missing:
        raise ValueError(
            "fixtures missing required columns: " + ", ".join(sorted(missing))
        )

    fixture_rows = fixtures.copy(deep=True)
    fixture_rows["match_id"] = fixture_rows["match_id"].astype(str).str.strip()
    records: dict[tuple[str, str], dict[str, object]] = {}
    for row in fixture_rows.itertuples(index=False):
        group = str(getattr(row, "group"))
        for team in [getattr(row, "team_a"), getattr(row, "team_b")]:
            key = (group, str(team))
            records.setdefault(key, _empty_standing_record(group, team))

    if results is not None and not results.empty:
        result_rows = results.copy(deep=True)
        result_rows["match_id"] = result_rows["match_id"].astype(str).str.strip()
        if "status" in result_rows.columns:
            status = result_rows["status"].astype("string").str.strip().str.casefold()
            result_rows = result_rows.loc[status.eq("completed")].copy(deep=True)

        completed = fixture_rows[
            ["match_id", "group", "team_a", "team_b"]
        ].merge(
            result_rows[["match_id", "team_a_goals", "team_b_goals"]],
            on="match_id",
            how="inner",
            validate="one_to_one",
        )
        for row in completed.itertuples(index=False):
            group = str(getattr(row, "group"))
            team_a = str(getattr(row, "team_a"))
            team_b = str(getattr(row, "team_b"))
            goals_a = int(getattr(row, "team_a_goals"))
            goals_b = int(getattr(row, "team_b_goals"))
            _, points_a, win_a, draw_a, loss_a = _score_result(goals_a, goals_b)
            _, points_b, win_b, draw_b, loss_b = _score_result(goals_b, goals_a)

            for team, points, win, draw, loss, gf, ga in [
                (team_a, points_a, win_a, draw_a, loss_a, goals_a, goals_b),
                (team_b, points_b, win_b, draw_b, loss_b, goals_b, goals_a),
            ]:
                record = records[(group, team)]
                record["played"] += 1
                record["points"] += points
                record["wins"] += win
                record["draws"] += draw
                record["losses"] += loss
                record["goals_for"] += gf
                record["goals_against"] += ga
                record["goal_difference"] += gf - ga

    table = pd.DataFrame(records.values())
    table = table.sort_values(
        ["group", "points", "goal_difference", "goals_for", "team"],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    table["rank"] = table.groupby("group", sort=False).cumcount() + 1
    return table[STANDING_COLUMNS].copy(deep=True)


def prepare_match_table(
    display_table: pd.DataFrame,
    show_audit_probabilities: bool = False,
) -> pd.DataFrame:
    """Return a formatted match table that respects display-status semantics."""
    rows = display_table.copy(deep=True)
    rows["match_date"] = pd.to_datetime(rows["match_date"], errors="coerce")
    rows["date"] = rows["match_date"].dt.strftime("%b %d").str.replace(
        " 0",
        " ",
        regex=False,
    )
    rows["match"] = rows["team_a"].astype(str) + " vs " + rows["team_b"].astype(str)
    rows["status"] = rows["display_status"].map(
        {
            "completed": "Completed",
            "scheduled": "Scheduled prediction",
            "prediction_missing": "Prediction missing",
        }
    )
    rows["score"] = rows.apply(
        lambda row: (
            f"{int(row['team_a_goals'])}-{int(row['team_b_goals'])}"
            if row["display_status"] == "completed"
            and not pd.isna(row["team_a_goals"])
            and not pd.isna(row["team_b_goals"])
            else "-"
        ),
        axis=1,
    )
    rows["favorite"] = rows.apply(
        lambda row: row["favorite_display"]
        if row["display_status"] == "scheduled"
        else "-",
        axis=1,
    )

    show_probabilities = rows["display_status"].eq("scheduled") | (
        show_audit_probabilities & rows["audit_available"].astype(bool)
    )
    probability_labels = {
        "p_team_a_win": "Team A win",
        "p_draw": "Draw",
        "p_team_b_win": "Team B win",
    }
    for source_column, display_column in probability_labels.items():
        rows[display_column] = [
            format_percent(value) if show else "-"
            for value, show in zip(rows[source_column], show_probabilities)
        ]

    rows["confidence"] = rows["confidence_label"].fillna("-")
    rows["forecast mode"] = rows["forecast_mode"].fillna("-")
    rows["model context"] = rows["prediction_display_label"].fillna("-")

    return rows[
        [
            "date",
            "group",
            "match",
            "status",
            "score",
            "favorite",
            "Team A win",
            "Draw",
            "Team B win",
            "confidence",
            "forecast mode",
            "model context",
        ]
    ].copy(deep=True)


def prepare_probability_summary(
    summary: pd.DataFrame,
    probability_columns: Iterable[str] = (
        "group_winner_prob",
        "top_2_prob",
        "third_place_prob",
        "best_third_place_advance_prob",
        "advance_prob",
    ),
) -> pd.DataFrame:
    """Return a formatted simulation summary for user-facing tables."""
    output = summary.copy(deep=True)
    label_map = {
        "group_winner_prob": "Group winner",
        "top_2_prob": "Top two",
        "third_place_prob": "Third place",
        "best_third_place_advance_prob": "Best third advance",
        "advance_prob": "Advance",
        "avg_points": "Avg points",
        "avg_goal_difference": "Avg GD",
    }
    for column in probability_columns:
        if column in output.columns:
            output[label_map[column]] = output[column].map(format_percent)
    if "avg_points" in output.columns:
        output["Avg points"] = output["avg_points"].map(
            lambda value: format_number(value, decimals=2)
        )
    if "avg_goal_difference" in output.columns:
        output["Avg GD"] = output["avg_goal_difference"].map(
            lambda value: format_number(value, decimals=2)
        )

    columns = [
        "team",
        "group",
        "Group winner",
        "Top two",
        "Third place",
        "Best third advance",
        "Advance",
        "Avg points",
        "Avg GD",
    ]
    return output[[column for column in columns if column in output.columns]].copy(
        deep=True
    )


def prepare_full_tournament_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Return a formatted full-tournament simulation table."""
    output = summary.copy(deep=True)
    label_map = {
        "reach_round_of_32_prob": "Round of 32",
        "reach_round_of_16_prob": "Round of 16",
        "reach_quarterfinal_prob": "Quarterfinal",
        "reach_semifinal_prob": "Semifinal",
        "reach_final_prob": "Final",
        "champion_prob": "Champion",
        "advance_from_group_prob": "Advance",
    }
    for source_column, display_column in label_map.items():
        if source_column in output.columns:
            output[display_column] = output[source_column].map(format_percent)

    columns = [
        "team",
        "group",
        "Advance",
        "Round of 16",
        "Quarterfinal",
        "Semifinal",
        "Final",
        "Champion",
    ]
    return output[[column for column in columns if column in output.columns]].copy(
        deep=True
    )


def get_teams_from_fixtures(fixtures: pd.DataFrame) -> list[str]:
    """Return sorted unique teams from the fixture table."""
    teams = pd.concat([fixtures["team_a"], fixtures["team_b"]], ignore_index=True)
    return sorted(teams.dropna().astype(str).unique())
