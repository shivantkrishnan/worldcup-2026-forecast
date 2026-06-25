"""Diagnostics for full-tournament knockout paths and champion probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.simulation.full_tournament import knockout_advancement_probabilities

DEFAULT_CONTENDER_TEAMS = ("Argentina", "France", "Spain", "Brazil", "England")


@dataclass(frozen=True)
class KnockoutRoundInfo:
    """Column mapping for one knockout round in the trace table."""

    prefix: str
    label: str
    reached_column: str
    advanced_to_column: str
    display_short: str


KNOCKOUT_ROUNDS = (
    KnockoutRoundInfo(
        "round_of_32",
        "Round of 32",
        "reached_round_of_32",
        "reached_round_of_16",
        "R32",
    ),
    KnockoutRoundInfo(
        "round_of_16",
        "Round of 16",
        "reached_round_of_16",
        "reached_quarterfinal",
        "R16",
    ),
    KnockoutRoundInfo(
        "quarterfinal",
        "Quarterfinal",
        "reached_quarterfinal",
        "reached_semifinal",
        "QF",
    ),
    KnockoutRoundInfo(
        "semifinal",
        "Semifinal",
        "reached_semifinal",
        "reached_final",
        "SF",
    ),
    KnockoutRoundInfo(
        "final",
        "Final",
        "reached_final",
        "won_tournament",
        "Final",
    ),
)


def _team_trace_rows(traces: pd.DataFrame, team: str) -> pd.DataFrame:
    """Return trace rows for one team, or an empty dataframe."""
    if traces is None or traces.empty or "team" not in traces.columns:
        return pd.DataFrame()
    return traces.loc[traces["team"].astype(str).eq(str(team))].copy(deep=True)


def _probability(rows: pd.DataFrame, column: str) -> float:
    """Return boolean mean for a trace column."""
    if rows.empty or column not in rows.columns:
        return float("nan")
    return float(rows[column].astype(bool).mean())


def _conditional_probability(
    rows: pd.DataFrame,
    numerator_column: str,
    denominator_column: str,
) -> float:
    """Return P(numerator | denominator) from boolean trace columns."""
    if (
        rows.empty
        or numerator_column not in rows.columns
        or denominator_column not in rows.columns
    ):
        return float("nan")
    denominator = rows[denominator_column].astype(bool)
    if not denominator.any():
        return float("nan")
    return float(rows.loc[denominator, numerator_column].astype(bool).mean())


def _dominant_group(rows: pd.DataFrame) -> str:
    """Return the most common group label for display."""
    if rows.empty or "group" not in rows.columns:
        return "-"
    values = rows["group"].dropna().astype(str)
    return str(values.mode().iloc[0]) if not values.empty else "-"


def summarize_team_path(traces: pd.DataFrame, team: str) -> dict[str, object]:
    """Summarize reach and conditional transition probabilities for one team."""
    rows = _team_trace_rows(traces, team)
    if rows.empty:
        return {
            "team": team,
            "group": "-",
            "simulation_count": 0,
            "available": False,
            "messages": ["No full-tournament trace rows are available for this team."],
        }

    summary: dict[str, object] = {
        "team": team,
        "group": _dominant_group(rows),
        "simulation_count": int(rows["simulation_id"].nunique())
        if "simulation_id" in rows.columns
        else int(len(rows)),
        "available": True,
        "messages": [],
        "group_advancement_probability": _probability(rows, "advance_from_group"),
        "group_winner_probability": _probability(rows, "group_winner"),
        "top_two_probability": _probability(rows, "top_two"),
        "best_third_place_advancement_probability": _probability(
            rows,
            "best_third_place_advanced",
        ),
        "reach_round_of_32_probability": _probability(rows, "reached_round_of_32"),
        "reach_round_of_16_probability": _probability(rows, "reached_round_of_16"),
        "reach_quarterfinal_probability": _probability(rows, "reached_quarterfinal"),
        "reach_semifinal_probability": _probability(rows, "reached_semifinal"),
        "reach_final_probability": _probability(rows, "reached_final"),
        "champion_probability": _probability(rows, "won_tournament"),
    }
    summary.update(
        {
            "p_reach_round_of_16_given_round_of_32": _conditional_probability(
                rows,
                "reached_round_of_16",
                "reached_round_of_32",
            ),
            "p_reach_quarterfinal_given_round_of_16": _conditional_probability(
                rows,
                "reached_quarterfinal",
                "reached_round_of_16",
            ),
            "p_reach_semifinal_given_quarterfinal": _conditional_probability(
                rows,
                "reached_semifinal",
                "reached_quarterfinal",
            ),
            "p_reach_final_given_semifinal": _conditional_probability(
                rows,
                "reached_final",
                "reached_semifinal",
            ),
            "p_champion_given_final": _conditional_probability(
                rows,
                "won_tournament",
                "reached_final",
            ),
        }
    )
    return summary


def most_likely_opponents(
    traces: pd.DataFrame,
    team: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return most common simulated knockout opponents for one team."""
    rows = _team_trace_rows(traces, team)
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "round",
                "opponent",
                "opponent_frequency",
                "avg_team_advance_prob",
                "avg_opponent_advance_prob",
                "simulated_team_advance_rate",
                "notes",
            ]
        )

    output_rows: list[dict[str, object]] = []
    for round_order, round_info in enumerate(KNOCKOUT_ROUNDS, start=1):
        reached = rows[round_info.reached_column].astype(bool)
        denominator = int(reached.sum())
        if denominator == 0:
            continue
        opponent_column = f"{round_info.prefix}_opponent"
        probability_column = f"{round_info.prefix}_team_advance_prob"
        opponent_probability_column = f"{round_info.prefix}_opponent_advance_prob"
        advanced_column = f"{round_info.prefix}_advanced"
        round_rows = rows.loc[reached & rows[opponent_column].notna()].copy()
        if round_rows.empty:
            continue

        grouped = round_rows.groupby(opponent_column, sort=True)
        for opponent, opponent_rows in grouped:
            output_rows.append(
                {
                    "round": round_info.label,
                    "round_order": round_order,
                    "round_key": round_info.prefix,
                    "opponent": str(opponent),
                    "opponent_frequency": float(len(opponent_rows) / denominator),
                    "avg_team_advance_prob": float(
                        opponent_rows[probability_column].mean()
                    ),
                    "avg_opponent_advance_prob": float(
                        opponent_rows[opponent_probability_column].mean()
                    ),
                    "simulated_team_advance_rate": float(
                        opponent_rows[advanced_column].astype(bool).mean()
                    ),
                    "notes": (
                        "Frequency reflects simulated group finish plus the "
                        "official bracket path."
                    ),
                }
            )

    table = pd.DataFrame(output_rows)
    if table.empty:
        return table
    return (
        table.sort_values(
            ["round_order", "opponent_frequency", "opponent"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("round_key", sort=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def _opponent_occurrences(traces: pd.DataFrame, team: str) -> pd.DataFrame:
    """Return one row per simulated knockout opponent occurrence."""
    rows = _team_trace_rows(traces, team)
    occurrence_rows: list[dict[str, object]] = []
    if rows.empty:
        return pd.DataFrame()

    for round_info in KNOCKOUT_ROUNDS:
        opponent_column = f"{round_info.prefix}_opponent"
        probability_column = f"{round_info.prefix}_team_advance_prob"
        round_rows = rows.loc[rows[opponent_column].notna()].copy()
        for row in round_rows.itertuples(index=False):
            occurrence_rows.append(
                {
                    "simulation_id": getattr(row, "simulation_id", np.nan),
                    "round": round_info.label,
                    "round_key": round_info.prefix,
                    "opponent": str(getattr(row, opponent_column)),
                    "team_advance_prob": float(getattr(row, probability_column)),
                }
            )
    return pd.DataFrame(occurrence_rows)


def _summary_probability_lookup(summary: pd.DataFrame, column: str) -> dict[str, float]:
    """Return team -> probability lookup from a simulation summary."""
    if summary is None or summary.empty or column not in summary.columns:
        return {}
    return {
        str(row["team"]): float(row[column])
        for _, row in summary[["team", column]].dropna().iterrows()
    }


def path_difficulty_summary(
    traces: pd.DataFrame,
    summary: pd.DataFrame,
    team: str,
    elite_n: int = 8,
) -> dict[str, object]:
    """Return transparent path-difficulty diagnostics for one team."""
    occurrences = _opponent_occurrences(traces, team)
    if occurrences.empty:
        return {
            "team": team,
            "available": False,
            "messages": ["No knockout opponent traces are available for this team."],
        }

    champion_lookup = _summary_probability_lookup(summary, "champion_prob")
    final_lookup = _summary_probability_lookup(summary, "reach_final_prob")
    elite_teams = set(
        summary.sort_values("champion_prob", ascending=False, kind="mergesort")
        .head(elite_n)["team"]
        .astype(str)
    ) if summary is not None and not summary.empty and "champion_prob" in summary else set()

    occurrences["opponent_champion_probability"] = occurrences["opponent"].map(
        champion_lookup
    )
    occurrences["opponent_final_probability"] = occurrences["opponent"].map(
        final_lookup
    )
    occurrences["opponent_is_elite"] = occurrences["opponent"].isin(elite_teams)
    elite_counts = occurrences.groupby("simulation_id")["opponent_is_elite"].sum()

    return {
        "team": team,
        "available": True,
        "average_model_implied_advancement_probability": float(
            occurrences["team_advance_prob"].mean()
        ),
        "average_opponent_champion_probability": float(
            occurrences["opponent_champion_probability"].mean()
        ),
        "average_opponent_final_probability": float(
            occurrences["opponent_final_probability"].mean()
        ),
        "expected_elite_opponents_faced": float(elite_counts.mean())
        if not elite_counts.empty
        else 0.0,
        "elite_definition": f"Top {elite_n} teams by simulated champion probability",
        "messages": [],
    }


def _average_round_opponent_difficulty(
    traces: pd.DataFrame,
    summary: pd.DataFrame,
    team: str,
    round_prefix: str,
) -> float:
    """Return mean opponent champion probability for one round."""
    rows = _team_trace_rows(traces, team)
    champion_lookup = _summary_probability_lookup(summary, "champion_prob")
    opponent_column = f"{round_prefix}_opponent"
    if rows.empty or opponent_column not in rows.columns or not champion_lookup:
        return float("nan")
    values = rows.loc[rows[opponent_column].notna(), opponent_column].map(
        champion_lookup
    )
    return float(values.mean()) if not values.dropna().empty else float("nan")


def _largest_path_bottleneck(team_path: dict[str, object]) -> str:
    """Return the lowest conditional transition label for one team."""
    candidates = {
        "R32 to R16": team_path.get("p_reach_round_of_16_given_round_of_32"),
        "R16 to QF": team_path.get("p_reach_quarterfinal_given_round_of_16"),
        "QF to SF": team_path.get("p_reach_semifinal_given_quarterfinal"),
        "SF to Final": team_path.get("p_reach_final_given_semifinal"),
        "Final to champion": team_path.get("p_champion_given_final"),
    }
    valid = {
        label: float(value)
        for label, value in candidates.items()
        if value is not None and not pd.isna(value)
    }
    if not valid:
        return "-"
    return min(valid, key=valid.get)


def compare_top_contenders(
    traces: pd.DataFrame,
    summary: pd.DataFrame,
    teams: Iterable[str] = DEFAULT_CONTENDER_TEAMS,
) -> pd.DataFrame:
    """Compare title-path diagnostics for selected top contenders."""
    if summary is None or summary.empty:
        return pd.DataFrame()
    summary_index = summary.set_index("team", drop=False)
    rows: list[dict[str, object]] = []
    round_columns = {
        "round_of_32": "average_r32_opponent_difficulty",
        "round_of_16": "average_r16_opponent_difficulty",
        "quarterfinal": "average_qf_opponent_difficulty",
        "semifinal": "average_sf_opponent_difficulty",
        "final": "average_final_opponent_difficulty",
    }
    for team in teams:
        if team not in summary_index.index:
            continue
        summary_row = summary_index.loc[team]
        team_path = summarize_team_path(traces, team)
        row = {
            "team": team,
            "champion_probability": float(summary_row.get("champion_prob", np.nan)),
            "final_probability": float(summary_row.get("reach_final_prob", np.nan)),
            "semifinal_probability": float(
                summary_row.get("reach_semifinal_prob", np.nan)
            ),
            "quarterfinal_probability": float(
                summary_row.get("reach_quarterfinal_prob", np.nan)
            ),
            "group_winner_probability": float(
                summary_row.get("group_winner_prob", np.nan)
            ),
            "largest_likely_path_bottleneck": _largest_path_bottleneck(team_path),
        }
        for round_prefix, column in round_columns.items():
            row[column] = _average_round_opponent_difficulty(
                traces,
                summary,
                team,
                round_prefix,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def matchup_source_label(source_label: str | None) -> str:
    """Return a consumer-facing label for knockout matchup probabilities."""
    normalized = (source_label or "").strip().casefold()
    if "fallback" in normalized or "snapshot" in normalized:
        return "Approximate deploy-safe matchup estimate"
    if normalized:
        return "Selected model neutral matchup estimate"
    return "Knockout matchup estimate"


def head_to_head_probability_table(
    selected_team: str,
    opponents: pd.DataFrame,
    knockout_probabilities: pd.DataFrame,
    source_label: str | None = None,
) -> pd.DataFrame:
    """Return model-implied H2H probabilities for likely knockout opponents."""
    if (
        opponents is None
        or opponents.empty
        or knockout_probabilities is None
        or knockout_probabilities.empty
    ):
        return pd.DataFrame()

    probability_lookup = {
        (str(row["team_a"]), str(row["team_b"])): row
        for _, row in knockout_probabilities.iterrows()
    }
    output_rows: list[dict[str, object]] = []
    for opponent in opponents["opponent"].dropna().astype(str).unique():
        direct = probability_lookup.get((selected_team, opponent))
        reverse = probability_lookup.get((opponent, selected_team))
        if direct is not None:
            p_win = float(direct["p_team_a_win"])
            p_draw = float(direct["p_draw"])
            p_loss = float(direct["p_team_b_win"])
        elif reverse is not None:
            p_win = float(reverse["p_team_b_win"])
            p_draw = float(reverse["p_draw"])
            p_loss = float(reverse["p_team_a_win"])
        else:
            continue

        p_selected_advance, p_opponent_advance = knockout_advancement_probabilities(
            p_win,
            p_draw,
            p_loss,
        )
        output_rows.append(
            {
                "opponent": opponent,
                "p_selected_team_advances": p_selected_advance,
                "p_opponent_advances": p_opponent_advance,
                "p_selected_team_regular_time_win": p_win,
                "p_regular_time_draw": p_draw,
                "p_selected_team_regular_time_loss": p_loss,
                "probability_source": matchup_source_label(source_label),
            }
        )
    return pd.DataFrame(output_rows)
