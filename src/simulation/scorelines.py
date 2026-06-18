"""Conditional scoreline simulation for group-stage table mechanics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

OUTCOME_TEAM_A_WIN = "team_a_win"
OUTCOME_DRAW = "draw"
OUTCOME_TEAM_B_WIN = "team_b_win"
VALID_OUTCOMES = [OUTCOME_TEAM_A_WIN, OUTCOME_DRAW, OUTCOME_TEAM_B_WIN]
PROBABILITY_COLUMNS = ["p_team_a_win", "p_draw", "p_team_b_win"]
SCORE_CAP = 6

FALLBACK_SCORELINES: dict[str, list[tuple[int, int]]] = {
    OUTCOME_TEAM_A_WIN: [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0)],
    OUTCOME_DRAW: [(0, 0), (1, 1), (2, 2), (3, 3)],
    OUTCOME_TEAM_B_WIN: [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3), (0, 4)],
}


def _scoreline_result(team_a_goals: int, team_b_goals: int) -> str:
    """Return the result class implied by a scoreline."""
    if team_a_goals > team_b_goals:
        return OUTCOME_TEAM_A_WIN
    if team_a_goals < team_b_goals:
        return OUTCOME_TEAM_B_WIN
    return OUTCOME_DRAW


def _cap_scoreline(team_a_goals: int, team_b_goals: int) -> tuple[int, int]:
    """Cap extreme historical scores while preserving result-class consistency."""
    capped_a = min(int(team_a_goals), SCORE_CAP)
    capped_b = min(int(team_b_goals), SCORE_CAP)
    result = _scoreline_result(int(team_a_goals), int(team_b_goals))

    if result == OUTCOME_TEAM_A_WIN and capped_a <= capped_b:
        capped_b = min(capped_b, SCORE_CAP - 1)
        capped_a = capped_b + 1
    elif result == OUTCOME_TEAM_B_WIN and capped_b <= capped_a:
        capped_a = min(capped_a, SCORE_CAP - 1)
        capped_b = capped_a + 1
    elif result == OUTCOME_DRAW:
        capped_score = min(capped_a, capped_b)
        capped_a = capped_score
        capped_b = capped_score
    return capped_a, capped_b


def _fallback_distribution(result_class: str) -> pd.DataFrame:
    """Return a fallback scoreline distribution for one result class."""
    rows = [
        {"team_a_goals": team_a, "team_b_goals": team_b, "weight": 1.0}
        for team_a, team_b in FALLBACK_SCORELINES[result_class]
    ]
    return pd.DataFrame(rows)


def build_empirical_scoreline_distributions(
    completed_matches: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build conditional scoreline frequency tables from completed matches.

    Extreme scores are capped at six goals per team. This keeps rare outliers
    from dominating table mechanics while preserving whether Team A won, drew,
    or lost.
    """
    required = {"team_a_goals", "team_b_goals", "result"}
    if completed_matches.empty or not required.issubset(completed_matches.columns):
        return {}

    rows: list[dict[str, Any]] = []
    for row in completed_matches.itertuples(index=False):
        result_class = str(getattr(row, "result"))
        if result_class not in VALID_OUTCOMES:
            continue
        team_a_goals, team_b_goals = _cap_scoreline(
            int(getattr(row, "team_a_goals")),
            int(getattr(row, "team_b_goals")),
        )
        if _scoreline_result(team_a_goals, team_b_goals) != result_class:
            continue
        rows.append(
            {
                "result": result_class,
                "team_a_goals": team_a_goals,
                "team_b_goals": team_b_goals,
            }
        )

    if not rows:
        return {}

    counts = (
        pd.DataFrame(rows)
        .groupby(["result", "team_a_goals", "team_b_goals"], sort=True)
        .size()
        .rename("weight")
        .reset_index()
    )
    return {
        result_class: group[["team_a_goals", "team_b_goals", "weight"]].reset_index(
            drop=True
        )
        for result_class, group in counts.groupby("result", sort=False)
    }


def sample_result_class(row: pd.Series, rng: np.random.Generator) -> str:
    """Sample a W/D/L result class from a fixture probability row."""
    probabilities = row[PROBABILITY_COLUMNS].astype(float).to_numpy()
    return str(rng.choice(VALID_OUTCOMES, p=probabilities))


def sample_scoreline_given_result(
    result_class: str,
    rng: np.random.Generator,
    scoreline_distributions: dict[str, pd.DataFrame] | None = None,
) -> tuple[int, int]:
    """Sample a scoreline conditional on a W/D/L result class."""
    if result_class not in VALID_OUTCOMES:
        raise ValueError("result_class must be one of: " + ", ".join(VALID_OUTCOMES))

    distribution = (
        scoreline_distributions.get(result_class)
        if scoreline_distributions is not None
        else None
    )
    if distribution is None or distribution.empty:
        distribution = _fallback_distribution(result_class)

    weights = distribution["weight"].astype(float).to_numpy()
    probabilities = weights / weights.sum()
    selected_index = int(rng.choice(np.arange(len(distribution)), p=probabilities))
    selected = distribution.iloc[selected_index]
    scoreline = int(selected["team_a_goals"]), int(selected["team_b_goals"])
    if _scoreline_result(*scoreline) != result_class:
        raise ValueError("Sampled scoreline is inconsistent with result_class.")
    return scoreline


def sample_scoreline_from_probabilities(
    row: pd.Series,
    rng: np.random.Generator,
    scoreline_distributions: dict[str, pd.DataFrame] | None = None,
) -> tuple[str, int, int]:
    """Sample a result class and scoreline from fixture probabilities."""
    result_class = sample_result_class(row, rng)
    team_a_goals, team_b_goals = sample_scoreline_given_result(
        result_class,
        rng,
        scoreline_distributions=scoreline_distributions,
    )
    return result_class, team_a_goals, team_b_goals
