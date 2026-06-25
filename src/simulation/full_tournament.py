"""Full-tournament Monte Carlo simulation with knockout rounds."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any

import numpy as np
import pandas as pd

from src.features.fixture_features import (
    build_fixture_feature_rows,
    build_live_feature_history,
)
from src.models.forecast import (
    ForecastModelBundle,
    predict_fixture_probabilities,
    train_selected_baseline,
)
from src.simulation.knockout_bracket import (
    FINAL,
    QUARTERFINAL,
    ROUND_OF_16,
    ROUND_OF_32,
    SEMIFINAL,
    SOURCE_GROUP_RUNNER_UP,
    SOURCE_GROUP_WINNER,
    SOURCE_MATCH_WINNER,
    SOURCE_THIRD_PLACE,
    BracketMatch,
    BracketSlot,
    assign_third_place_groups_to_slots,
    get_knockout_matches,
    get_round_of_32_matches,
)
from src.simulation.tournament import (
    PROBABILITY_COLUMNS,
    simulate_group_stage_once,
    validate_fixture_probability_table,
)

KNOCKOUT_PROBABILITY_COLUMNS = [
    "team_a",
    "team_b",
    "p_team_a_win",
    "p_draw",
    "p_team_b_win",
]
FULL_TOURNAMENT_SUMMARY_COLUMNS = [
    "team",
    "group",
    "simulations",
    "reach_round_of_32_prob",
    "reach_round_of_16_prob",
    "reach_quarterfinal_prob",
    "reach_semifinal_prob",
    "reach_final_prob",
    "champion_prob",
    "group_winner_prob",
    "top_two_prob",
    "best_third_place_advance_prob",
    "advance_from_group_prob",
]
TRACE_ROUND_PREFIXES = {
    ROUND_OF_32: "round_of_32",
    ROUND_OF_16: "round_of_16",
    QUARTERFINAL: "quarterfinal",
    SEMIFINAL: "semifinal",
    FINAL: "final",
}
REACH_COLUMNS_BY_ROUND = {
    ROUND_OF_32: "reached_round_of_16",
    ROUND_OF_16: "reached_quarterfinal",
    QUARTERFINAL: "reached_semifinal",
    SEMIFINAL: "reached_final",
    FINAL: "won_tournament",
}


@dataclass(frozen=True)
class KnockoutProbabilitySource:
    """In-memory knockout probability table plus display metadata."""

    probabilities: pd.DataFrame
    source_label: str
    caveat: str


@dataclass(frozen=True)
class FullTournamentSimulationOutput:
    """Full-tournament summary plus optional path-level simulation traces."""

    summary: pd.DataFrame
    traces: pd.DataFrame


def knockout_advancement_probabilities(
    p_team_a_win: float,
    p_draw: float,
    p_team_b_win: float,
) -> tuple[float, float]:
    """Return knockout advancement probabilities from regular-time W/D/L.

    First-pass rule: split regular-time draw mass evenly between both teams.
    """
    p_a_advance = float(p_team_a_win) + 0.5 * float(p_draw)
    p_b_advance = float(p_team_b_win) + 0.5 * float(p_draw)
    total = p_a_advance + p_b_advance
    if total <= 0:
        raise ValueError("Knockout advancement probabilities must have positive mass.")
    return p_a_advance / total, p_b_advance / total


def _all_tournament_teams(fixtures: pd.DataFrame) -> list[str]:
    """Return sorted unique team names from fixture rows."""
    teams = pd.concat([fixtures["team_a"], fixtures["team_b"]], ignore_index=True)
    return sorted(teams.dropna().astype(str).str.strip().unique())


def _safe_wdl_from_advancement_probability(
    p_team_a_advance: float,
    draw_probability: float = 0.24,
) -> tuple[float, float, float]:
    """Convert an advancement probability into regular-time W/D/L probabilities."""
    p_draw = float(np.clip(draw_probability, 0.05, 0.5))
    min_advancement = 0.5 * p_draw
    max_advancement = 1.0 - 0.5 * p_draw
    p_advance = float(np.clip(p_team_a_advance, min_advancement, max_advancement))
    p_team_a_win = p_advance - 0.5 * p_draw
    p_team_b_win = 1.0 - p_draw - p_team_a_win
    return float(p_team_a_win), float(p_draw), float(p_team_b_win)


def _completed_result_strength_scores(results: pd.DataFrame) -> dict[str, list[float]]:
    """Return per-team strength observations from completed result rows."""
    scores: dict[str, list[float]] = {}
    if results is None or results.empty:
        return scores

    result_rows = results.copy(deep=True)
    if "status" in result_rows.columns:
        status = result_rows["status"].astype("string").str.strip().str.casefold()
        result_rows = result_rows.loc[status.eq("completed")].copy(deep=True)

    for row in result_rows.itertuples(index=False):
        team_a = str(getattr(row, "team_a")).strip()
        team_b = str(getattr(row, "team_b")).strip()
        result = str(getattr(row, "result"))
        if result == "team_a_win":
            score_a, score_b = 1.0, 0.0
        elif result == "team_b_win":
            score_a, score_b = 0.0, 1.0
        else:
            score_a, score_b = 0.5, 0.5
        scores.setdefault(team_a, []).append(score_a)
        scores.setdefault(team_b, []).append(score_b)
    return scores


def _prediction_strength_scores(predictions: pd.DataFrame) -> dict[str, list[float]]:
    """Return per-team strength observations from fixture probabilities."""
    scores: dict[str, list[float]] = {}
    if predictions is None or predictions.empty:
        return scores

    for row in predictions.itertuples(index=False):
        team_a = str(getattr(row, "team_a")).strip()
        team_b = str(getattr(row, "team_b")).strip()
        p_team_a_win = float(getattr(row, "p_team_a_win"))
        p_draw = float(getattr(row, "p_draw"))
        p_team_b_win = float(getattr(row, "p_team_b_win"))
        scores.setdefault(team_a, []).append(p_team_a_win + 0.5 * p_draw)
        scores.setdefault(team_b, []).append(p_team_b_win + 0.5 * p_draw)
    return scores


def build_prediction_strength_knockout_probability_table(
    fixtures: pd.DataFrame,
    predictions: pd.DataFrame,
    results: pd.DataFrame | None = None,
) -> KnockoutProbabilitySource:
    """Build deploy-safe approximate knockout probabilities from snapshot data.

    This fallback does not claim to be the selected match model. It converts
    committed fixture probabilities and completed results into a team-strength
    proxy, then creates neutral W/D/L rows for arbitrary pairings.
    """
    teams = _all_tournament_teams(fixtures)
    score_observations = _completed_result_strength_scores(
        results if results is not None else pd.DataFrame()
    )
    for team, values in _prediction_strength_scores(predictions).items():
        score_observations.setdefault(team, []).extend(values)

    strength = {
        team: float(np.mean(score_observations.get(team, [0.5])))
        for team in teams
    }
    rows: list[dict[str, object]] = []
    for team_a, team_b in permutations(teams, 2):
        diff = strength[team_a] - strength[team_b]
        p_team_a_advance = 1.0 / (1.0 + np.exp(-3.2 * diff))
        p_team_a_win, p_draw, p_team_b_win = _safe_wdl_from_advancement_probability(
            p_team_a_advance,
            draw_probability=0.24 - min(abs(diff), 0.5) * 0.08,
        )
        rows.append(
            {
                "team_a": team_a,
                "team_b": team_b,
                "p_team_a_win": p_team_a_win,
                "p_draw": p_draw,
                "p_team_b_win": p_team_b_win,
            }
        )
    return KnockoutProbabilitySource(
        probabilities=validate_knockout_probability_table(pd.DataFrame(rows)),
        source_label="snapshot_strength_fallback",
        caveat=(
            "Knockout matchup probabilities are approximated from committed "
            "group-stage prediction strength and completed results because raw "
            "training data is not available in the deployed app."
        ),
    )


def build_model_based_knockout_probability_table(
    training_matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    completed_results: pd.DataFrame | None = None,
    feature_cutoff_date: str | None = None,
    model_bundle: ForecastModelBundle | None = None,
) -> KnockoutProbabilitySource:
    """Build arbitrary neutral knockout matchup probabilities with the model."""
    teams = _all_tournament_teams(fixtures)
    model = model_bundle or train_selected_baseline(training_matches)

    feature_history = training_matches.copy(deep=True)
    if completed_results is not None and not completed_results.empty and feature_cutoff_date:
        feature_history = build_live_feature_history(
            training_matches,
            completed_results,
            feature_cutoff_date=feature_cutoff_date,
        )

    fixture_date = "2026-06-28"
    rows = [
        {
            "match_id": f"knockout_neutral_{index:04d}",
            "match_date": fixture_date,
            "team_a": team_a,
            "team_b": team_b,
            "stage": "knockout",
            "group": "knockout",
            "neutral": True,
            "is_neutral": True,
            "tournament": "FIFA World Cup 2026",
        }
        for index, (team_a, team_b) in enumerate(permutations(teams, 2), start=1)
    ]
    neutral_fixtures = pd.DataFrame(rows)
    fixture_features = build_fixture_feature_rows(
        feature_history,
        neutral_fixtures,
        include_elo=model.include_elo,
        elo_k_factor=model.elo_k_factor,
        elo_home_advantage=model.elo_home_advantage,
        feature_cutoff_date=feature_cutoff_date,
    )
    probabilities = predict_fixture_probabilities(model, fixture_features)
    output = neutral_fixtures[["team_a", "team_b"]].reset_index(drop=True)
    output = pd.concat([output, probabilities[PROBABILITY_COLUMNS].reset_index(drop=True)], axis=1)
    return KnockoutProbabilitySource(
        probabilities=validate_knockout_probability_table(output),
        source_label="selected_model_neutral_knockout",
        caveat=(
            "Neutral knockout probabilities use the selected calibrated logistic "
            "model and live feature history through the feature cutoff. Simulated "
            "future group results do not update model features inside each path."
        ),
    )


def validate_knockout_probability_table(probabilities: pd.DataFrame) -> pd.DataFrame:
    """Validate arbitrary knockout matchup probability rows."""
    missing = set(KNOCKOUT_PROBABILITY_COLUMNS).difference(probabilities.columns)
    if missing:
        raise ValueError(
            "Missing knockout probability columns: " + ", ".join(sorted(missing))
        )
    output = probabilities.copy(deep=True)
    output["team_a"] = output["team_a"].astype(str).str.strip()
    output["team_b"] = output["team_b"].astype(str).str.strip()
    if output[["team_a", "team_b"]].isna().any().any():
        raise ValueError("Knockout probability teams must be non-null.")
    if output["team_a"].eq(output["team_b"]).any():
        raise ValueError("Knockout probability rows must compare two different teams.")
    if output[["team_a", "team_b"]].duplicated().any():
        raise ValueError("Knockout probability rows must be unique by team_a/team_b.")

    for column in PROBABILITY_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="raise")
    if (output[PROBABILITY_COLUMNS] < 0).any().any():
        raise ValueError("Knockout probabilities must be nonnegative.")
    probability_sums = output[PROBABILITY_COLUMNS].sum(axis=1)
    if not np.isclose(probability_sums, 1.0, atol=1e-6).all():
        raise ValueError("Knockout probabilities must sum to 1 for every row.")
    return output[KNOCKOUT_PROBABILITY_COLUMNS].copy(deep=True)


def _probability_lookup(probabilities: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    """Return ordered matchup probability lookup."""
    validated = validate_knockout_probability_table(probabilities)
    return {
        (str(row["team_a"]), str(row["team_b"])): row
        for _, row in validated.iterrows()
    }


def _get_matchup_probabilities(
    lookup: dict[tuple[str, str], pd.Series],
    team_a: str,
    team_b: str,
) -> pd.Series:
    """Return W/D/L probabilities for an ordered knockout matchup."""
    key = (team_a, team_b)
    if key in lookup:
        return lookup[key]

    reverse_key = (team_b, team_a)
    if reverse_key not in lookup:
        raise ValueError(f"Missing knockout probabilities for {team_a} vs {team_b}.")
    reverse = lookup[reverse_key]
    return pd.Series(
        {
            "team_a": team_a,
            "team_b": team_b,
            "p_team_a_win": float(reverse["p_team_b_win"]),
            "p_draw": float(reverse["p_draw"]),
            "p_team_b_win": float(reverse["p_team_a_win"]),
        }
    )


def sample_knockout_winner(
    team_a: str,
    team_b: str,
    probabilities: pd.Series,
    rng: np.random.Generator,
) -> str:
    """Sample one knockout advancing team from W/D/L probabilities."""
    p_a_advance, p_b_advance = knockout_advancement_probabilities(
        float(probabilities["p_team_a_win"]),
        float(probabilities["p_draw"]),
        float(probabilities["p_team_b_win"]),
    )
    return str(rng.choice([team_a, team_b], p=[p_a_advance, p_b_advance]))


def _participant_record(group_table: pd.DataFrame, group: str, rank: int) -> dict[str, Any]:
    """Return one participant record from simulated group standings."""
    rows = group_table.loc[
        group_table["group"].astype(str).str.upper().eq(group)
        & group_table["group_rank"].eq(rank)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one rank {rank} team in Group {group}.")
    row = rows.iloc[0]
    if rank == 1:
        route = "group_winner"
    elif rank == 2:
        route = "runner_up"
    else:
        route = "best_third"
    return {
        "team": str(row["team"]),
        "group": group,
        "group_rank": int(row["group_rank"]),
        "qualification_route": route,
    }


def _participant_for_slot(
    slot: BracketSlot,
    group_table: pd.DataFrame,
    third_place_assignment: dict[int, str],
    match_number: int,
) -> dict[str, Any]:
    """Resolve a Round-of-32 bracket slot to a concrete participant."""
    if slot.source_type == SOURCE_GROUP_WINNER:
        return _participant_record(group_table, str(slot.group), 1)
    if slot.source_type == SOURCE_GROUP_RUNNER_UP:
        return _participant_record(group_table, str(slot.group), 2)
    if slot.source_type == SOURCE_THIRD_PLACE:
        group = third_place_assignment.get(match_number)
        if group is None:
            raise ValueError(f"No third-place group assigned to match {match_number}.")
        return _participant_record(group_table, group, 3)
    raise ValueError(f"Unsupported Round-of-32 slot source: {slot.source_type}")


def assign_round_of_32_bracket(group_table: pd.DataFrame) -> pd.DataFrame:
    """Assign one simulated group-stage outcome to Round-of-32 matches."""
    required = {"team", "group", "group_rank", "advanced", "best_third_place_advanced"}
    missing = required.difference(group_table.columns)
    if missing:
        raise ValueError(
            "Missing required group standing columns: " + ", ".join(sorted(missing))
        )

    standings = group_table.copy(deep=True)
    standings["group"] = standings["group"].astype(str).str.upper()
    third_rows = standings.loc[
        standings["group_rank"].eq(3)
        & standings["best_third_place_advanced"].astype(bool)
    ].copy()
    third_groups = third_rows["group"].astype(str).tolist()
    third_assignment = assign_third_place_groups_to_slots(third_groups)

    rows: list[dict[str, object]] = []
    for match in get_round_of_32_matches():
        participant_a = _participant_for_slot(
            match.slot_a,
            standings,
            third_assignment,
            match.match_number,
        )
        participant_b = _participant_for_slot(
            match.slot_b,
            standings,
            third_assignment,
            match.match_number,
        )
        rows.append(
            {
                "match_number": match.match_number,
                "round": match.round_name,
                "slot_a_source": match.slot_a.label,
                "slot_b_source": match.slot_b.label,
                "team_a": participant_a["team"],
                "team_b": participant_b["team"],
                "team_a_group": participant_a["group"],
                "team_b_group": participant_b["group"],
                "team_a_route": participant_a["qualification_route"],
                "team_b_route": participant_b["qualification_route"],
            }
        )

    bracket = pd.DataFrame(rows)
    teams = pd.concat([bracket["team_a"], bracket["team_b"]], ignore_index=True)
    if len(teams) != 32 or teams.duplicated().any():
        duplicated = sorted(teams.loc[teams.duplicated()].astype(str).unique())
        raise ValueError(
            "Round-of-32 bracket must contain 32 unique teams. Duplicates: "
            + ", ".join(duplicated)
        )
    return bracket


def _resolve_match_winner_slot(slot: BracketSlot, winners_by_match: dict[int, str]) -> str:
    """Resolve a downstream winner-of-match slot."""
    if slot.source_type != SOURCE_MATCH_WINNER or slot.match_number is None:
        raise ValueError("Downstream knockout slots must reference match winners.")
    try:
        return winners_by_match[int(slot.match_number)]
    except KeyError as error:
        raise ValueError(f"Winner of match {slot.match_number} is not available.") from error


def _next_round_reached(round_name: str) -> str:
    """Return the reach flag awarded to a winner of the given round."""
    next_round = {
        ROUND_OF_32: "reach_round_of_16",
        ROUND_OF_16: "reach_quarterfinal",
        QUARTERFINAL: "reach_semifinal",
        SEMIFINAL: "reach_final",
        FINAL: "champion",
    }
    return next_round[round_name]


def _simulate_knockout_match(
    match: BracketMatch,
    team_a: str,
    team_b: str,
    probability_lookup: dict[tuple[str, str], pd.Series],
    rng: np.random.Generator,
) -> dict[str, object]:
    """Simulate one knockout match and return match-level metadata."""
    probabilities = _get_matchup_probabilities(probability_lookup, team_a, team_b)
    p_a_advance, p_b_advance = knockout_advancement_probabilities(
        float(probabilities["p_team_a_win"]),
        float(probabilities["p_draw"]),
        float(probabilities["p_team_b_win"]),
    )
    winner = sample_knockout_winner(team_a, team_b, probabilities, rng)
    return {
        "match_number": match.match_number,
        "round": match.round_name,
        "team_a": team_a,
        "team_b": team_b,
        "p_team_a_advance": p_a_advance,
        "p_team_b_advance": p_b_advance,
        "winner": winner,
        "loser": team_b if winner == team_a else team_a,
    }


def _empty_knockout_trace_fields() -> dict[str, object]:
    """Return empty per-round trace fields for one team/simulation."""
    fields: dict[str, object] = {}
    for prefix in TRACE_ROUND_PREFIXES.values():
        fields[f"{prefix}_opponent"] = pd.NA
        fields[f"{prefix}_team_advance_prob"] = np.nan
        fields[f"{prefix}_opponent_advance_prob"] = np.nan
        fields[f"{prefix}_advanced"] = False
    return fields


def _initial_trace_records(
    group_results: pd.DataFrame,
    simulation_id: int,
) -> dict[str, dict[str, object]]:
    """Return one trace record per team after group-stage resolution."""
    records: dict[str, dict[str, object]] = {}
    for row in group_results.itertuples(index=False):
        team = str(getattr(row, "team"))
        group_rank = int(getattr(row, "group_rank"))
        advanced = bool(getattr(row, "advanced"))
        best_third = bool(getattr(row, "best_third_place_advanced"))
        records[team] = {
            "simulation_id": simulation_id,
            "team": team,
            "group": str(getattr(row, "group")),
            "final_group_position": group_rank,
            "group_winner": group_rank == 1,
            "top_two": group_rank <= 2,
            "best_third_place_advanced": best_third,
            "advance_from_group": advanced,
            "reached_round_of_32": advanced,
            "reached_round_of_16": False,
            "reached_quarterfinal": False,
            "reached_semifinal": False,
            "reached_final": False,
            "won_tournament": False,
            **_empty_knockout_trace_fields(),
        }
    return records


def _record_knockout_trace(
    trace_records: dict[str, dict[str, object]] | None,
    match: BracketMatch,
    match_result: dict[str, object],
) -> None:
    """Record opponent/probability trace fields for both match participants."""
    if trace_records is None:
        return

    prefix = TRACE_ROUND_PREFIXES[match.round_name]
    team_a = str(match_result["team_a"])
    team_b = str(match_result["team_b"])
    winner = str(match_result["winner"])
    p_team_a_advance = float(match_result["p_team_a_advance"])
    p_team_b_advance = float(match_result["p_team_b_advance"])

    trace_records[team_a][f"{prefix}_opponent"] = team_b
    trace_records[team_a][f"{prefix}_team_advance_prob"] = p_team_a_advance
    trace_records[team_a][f"{prefix}_opponent_advance_prob"] = p_team_b_advance
    trace_records[team_a][f"{prefix}_advanced"] = winner == team_a

    trace_records[team_b][f"{prefix}_opponent"] = team_a
    trace_records[team_b][f"{prefix}_team_advance_prob"] = p_team_b_advance
    trace_records[team_b][f"{prefix}_opponent_advance_prob"] = p_team_a_advance
    trace_records[team_b][f"{prefix}_advanced"] = winner == team_b


def simulate_full_tournament_once(
    fixtures: pd.DataFrame,
    knockout_probabilities: pd.DataFrame,
    rng: np.random.Generator,
    scoreline_distributions: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Simulate one full tournament and return team-level reach flags."""
    team_results, _ = _simulate_full_tournament_once_with_optional_traces(
        fixtures,
        knockout_probabilities,
        rng,
        simulation_id=1,
        scoreline_distributions=scoreline_distributions,
        collect_traces=False,
    )
    return team_results


def _simulate_full_tournament_once_with_optional_traces(
    fixtures: pd.DataFrame,
    knockout_probabilities: pd.DataFrame,
    rng: np.random.Generator,
    simulation_id: int,
    scoreline_distributions: dict[str, pd.DataFrame] | None = None,
    collect_traces: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Simulate one full tournament and optionally return path traces."""
    group_results = simulate_group_stage_once(
        fixtures,
        rng=rng,
        top_n_per_group=2,
        include_best_third_place=True,
        n_best_third_place=8,
        scoreline_distributions=scoreline_distributions,
    )
    lookup = _probability_lookup(knockout_probabilities)
    team_records = {}
    trace_records = (
        _initial_trace_records(group_results, simulation_id)
        if collect_traces
        else None
    )
    for row in group_results.itertuples(index=False):
        team = str(getattr(row, "team"))
        group_rank = int(getattr(row, "group_rank"))
        advanced = bool(getattr(row, "advanced"))
        team_records[team] = {
            "team": team,
            "group": str(getattr(row, "group")),
            "group_winner": group_rank == 1,
            "top_two": group_rank <= 2,
            "best_third_place_advanced": bool(
                getattr(row, "best_third_place_advanced")
            ),
            "advance_from_group": advanced,
            "reach_round_of_32": advanced,
            "reach_round_of_16": False,
            "reach_quarterfinal": False,
            "reach_semifinal": False,
            "reach_final": False,
            "champion": False,
        }

    round_of_32 = assign_round_of_32_bracket(group_results)
    winners_by_match: dict[int, str] = {}

    for row in round_of_32.itertuples(index=False):
        match_config = next(
            match for match in get_round_of_32_matches() if match.match_number == row.match_number
        )
        match_result = _simulate_knockout_match(
            match_config,
            str(row.team_a),
            str(row.team_b),
            lookup,
            rng,
        )
        winner = str(match_result["winner"])
        _record_knockout_trace(trace_records, match_config, match_result)
        winners_by_match[int(row.match_number)] = winner
        reach_column = _next_round_reached(ROUND_OF_32)
        team_records[winner][reach_column] = True
        if trace_records is not None:
            trace_records[winner][REACH_COLUMNS_BY_ROUND[ROUND_OF_32]] = True

    downstream_matches = [
        match for match in get_knockout_matches() if match.round_name != ROUND_OF_32
    ]
    for match in downstream_matches:
        team_a = _resolve_match_winner_slot(match.slot_a, winners_by_match)
        team_b = _resolve_match_winner_slot(match.slot_b, winners_by_match)
        match_result = _simulate_knockout_match(match, team_a, team_b, lookup, rng)
        winner = str(match_result["winner"])
        _record_knockout_trace(trace_records, match, match_result)
        winners_by_match[match.match_number] = winner
        reach_column = _next_round_reached(match.round_name)
        team_records[winner][reach_column] = True
        if trace_records is not None:
            trace_records[winner][REACH_COLUMNS_BY_ROUND[match.round_name]] = True

    trace_frame = pd.DataFrame(trace_records.values()) if trace_records is not None else None
    return pd.DataFrame(team_records.values()), trace_frame


def simulate_full_tournament(
    fixtures: pd.DataFrame,
    knockout_probabilities: pd.DataFrame,
    n_simulations: int = 1000,
    random_seed: int = 42,
    scoreline_distributions: dict[str, pd.DataFrame] | None = None,
    collect_traces: bool = False,
) -> pd.DataFrame | FullTournamentSimulationOutput:
    """Run full-tournament simulations and return probabilities or traces."""
    if n_simulations <= 0:
        raise ValueError("n_simulations must be a positive integer.")
    validated_fixtures = validate_fixture_probability_table(fixtures)
    validated_knockout_probabilities = validate_knockout_probability_table(
        knockout_probabilities
    )
    rng = np.random.default_rng(random_seed)
    simulations: list[pd.DataFrame] = []
    traces: list[pd.DataFrame] = []
    for simulation_id in range(1, n_simulations + 1):
        result, trace = _simulate_full_tournament_once_with_optional_traces(
            validated_fixtures,
            validated_knockout_probabilities,
            rng=rng,
            simulation_id=simulation_id,
            scoreline_distributions=scoreline_distributions,
            collect_traces=collect_traces,
        )
        result["simulation_id"] = simulation_id
        simulations.append(result)
        if trace is not None:
            traces.append(trace)

    combined = pd.concat(simulations, ignore_index=True)
    grouped = combined.groupby(["team", "group"], sort=True)
    summary = grouped.agg(
        simulations=("simulation_id", "nunique"),
        reach_round_of_32_prob=(
            "reach_round_of_32",
            lambda values: float(values.astype(bool).mean()),
        ),
        reach_round_of_16_prob=(
            "reach_round_of_16",
            lambda values: float(values.astype(bool).mean()),
        ),
        reach_quarterfinal_prob=(
            "reach_quarterfinal",
            lambda values: float(values.astype(bool).mean()),
        ),
        reach_semifinal_prob=(
            "reach_semifinal",
            lambda values: float(values.astype(bool).mean()),
        ),
        reach_final_prob=(
            "reach_final",
            lambda values: float(values.astype(bool).mean()),
        ),
        champion_prob=("champion", lambda values: float(values.astype(bool).mean())),
        group_winner_prob=(
            "group_winner",
            lambda values: float(values.astype(bool).mean()),
        ),
        top_two_prob=("top_two", lambda values: float(values.astype(bool).mean())),
        best_third_place_advance_prob=(
            "best_third_place_advanced",
            lambda values: float(values.astype(bool).mean()),
        ),
        advance_from_group_prob=(
            "advance_from_group",
            lambda values: float(values.astype(bool).mean()),
        ),
    ).reset_index()

    summary = summary[FULL_TOURNAMENT_SUMMARY_COLUMNS].sort_values(
        ["champion_prob", "reach_final_prob", "team"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    if collect_traces:
        trace_frame = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
        return FullTournamentSimulationOutput(summary=summary, traces=trace_frame)
    return summary
