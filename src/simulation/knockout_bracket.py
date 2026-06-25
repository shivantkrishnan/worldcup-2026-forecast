"""2026 World Cup knockout bracket configuration and slot assignment."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

GROUPS = tuple("ABCDEFGHIJKL")
ROUND_OF_32 = "round_of_32"
ROUND_OF_16 = "round_of_16"
QUARTERFINAL = "quarterfinal"
SEMIFINAL = "semifinal"
FINAL = "final"

SOURCE_GROUP_WINNER = "group_winner"
SOURCE_GROUP_RUNNER_UP = "group_runner_up"
SOURCE_THIRD_PLACE = "third_place"
SOURCE_MATCH_WINNER = "match_winner"

BRACKET_SOURCE_NOTE = (
    "Round-of-32 slot pools and third-place assignments follow the FIFA World "
    "Cup 26 Regulations, Annex C, which lists the 495 possible combinations of "
    "the eight best third-placed teams."
)
FIFA_2026_REGULATIONS_URL = (
    "https://digitalhub.fifa.com/m/636f5c9c6f29771f/original/"
    "FWC2026_regulations_EN.pdf"
)
# Source: FIFA World Cup 26 Regulations, Annex C. The CSV is the official
# 495-row third-place assignment table extracted from that public PDF.
OFFICIAL_THIRD_PLACE_ASSIGNMENT_PATH = Path(__file__).with_name(
    "third_place_assignment_2026.csv"
)
THIRD_PLACE_WINNER_SLOTS = ("1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L")
THIRD_PLACE_SLOT_TO_MATCH = {
    "1A": 79,
    "1B": 85,
    "1D": 81,
    "1E": 74,
    "1G": 82,
    "1I": 77,
    "1K": 87,
    "1L": 80,
}
THIRD_PLACE_MATCH_TO_WINNER_SLOT = {
    match_number: slot for slot, match_number in THIRD_PLACE_SLOT_TO_MATCH.items()
}
THIRD_PLACE_SLOT_ALLOWED_GROUPS = {
    "1A": tuple("CEFHI"),
    "1B": tuple("EFGIJ"),
    "1D": tuple("BEFIJ"),
    "1E": tuple("ABCDF"),
    "1G": tuple("AEHIJ"),
    "1I": tuple("CDFGH"),
    "1K": tuple("DEIJL"),
    "1L": tuple("EHIJK"),
}
OFFICIAL_THIRD_PLACE_COMBINATION_COUNT = 495


@dataclass(frozen=True)
class BracketSlot:
    """One source slot in a knockout match."""

    source_type: str
    label: str
    group: str | None = None
    match_number: int | None = None
    allowed_third_groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class BracketMatch:
    """One configured knockout match."""

    match_number: int
    round_name: str
    slot_a: BracketSlot
    slot_b: BracketSlot
    notes: str = BRACKET_SOURCE_NOTE


def _winner(group: str) -> BracketSlot:
    return BracketSlot(SOURCE_GROUP_WINNER, f"Winner Group {group}", group=group)


def _runner_up(group: str) -> BracketSlot:
    return BracketSlot(SOURCE_GROUP_RUNNER_UP, f"Runner-up Group {group}", group=group)


def _third_place(groups: str) -> BracketSlot:
    allowed = tuple(groups)
    label = "Best 3rd place Group " + "/".join(allowed)
    return BracketSlot(SOURCE_THIRD_PLACE, label, allowed_third_groups=allowed)


def _match_winner(match_number: int) -> BracketSlot:
    return BracketSlot(
        SOURCE_MATCH_WINNER,
        f"Winner Match {match_number}",
        match_number=match_number,
    )


ROUND_OF_32_MATCHES: tuple[BracketMatch, ...] = (
    BracketMatch(73, ROUND_OF_32, _runner_up("A"), _runner_up("B")),
    BracketMatch(74, ROUND_OF_32, _winner("E"), _third_place("ABCDF")),
    BracketMatch(75, ROUND_OF_32, _winner("F"), _runner_up("C")),
    BracketMatch(76, ROUND_OF_32, _winner("C"), _runner_up("F")),
    BracketMatch(77, ROUND_OF_32, _winner("I"), _third_place("CDFGH")),
    BracketMatch(78, ROUND_OF_32, _runner_up("E"), _runner_up("I")),
    BracketMatch(79, ROUND_OF_32, _winner("A"), _third_place("CEFHI")),
    BracketMatch(80, ROUND_OF_32, _winner("L"), _third_place("EHIJK")),
    BracketMatch(81, ROUND_OF_32, _winner("D"), _third_place("BEFIJ")),
    BracketMatch(82, ROUND_OF_32, _winner("G"), _third_place("AEHIJ")),
    BracketMatch(83, ROUND_OF_32, _runner_up("K"), _runner_up("L")),
    BracketMatch(84, ROUND_OF_32, _winner("H"), _runner_up("J")),
    BracketMatch(85, ROUND_OF_32, _winner("B"), _third_place("EFGIJ")),
    BracketMatch(86, ROUND_OF_32, _winner("J"), _runner_up("H")),
    BracketMatch(87, ROUND_OF_32, _winner("K"), _third_place("DEIJL")),
    BracketMatch(88, ROUND_OF_32, _runner_up("D"), _runner_up("G")),
)

DOWNSTREAM_KNOCKOUT_MATCHES: tuple[BracketMatch, ...] = (
    BracketMatch(89, ROUND_OF_16, _match_winner(73), _match_winner(75)),
    BracketMatch(90, ROUND_OF_16, _match_winner(74), _match_winner(77)),
    BracketMatch(91, ROUND_OF_16, _match_winner(76), _match_winner(78)),
    BracketMatch(92, ROUND_OF_16, _match_winner(79), _match_winner(80)),
    BracketMatch(93, ROUND_OF_16, _match_winner(83), _match_winner(84)),
    BracketMatch(94, ROUND_OF_16, _match_winner(81), _match_winner(82)),
    BracketMatch(95, ROUND_OF_16, _match_winner(86), _match_winner(88)),
    BracketMatch(96, ROUND_OF_16, _match_winner(85), _match_winner(87)),
    BracketMatch(97, QUARTERFINAL, _match_winner(89), _match_winner(90)),
    BracketMatch(98, QUARTERFINAL, _match_winner(93), _match_winner(94)),
    BracketMatch(99, QUARTERFINAL, _match_winner(91), _match_winner(92)),
    BracketMatch(100, QUARTERFINAL, _match_winner(95), _match_winner(96)),
    BracketMatch(101, SEMIFINAL, _match_winner(97), _match_winner(98)),
    BracketMatch(102, SEMIFINAL, _match_winner(99), _match_winner(100)),
    BracketMatch(104, FINAL, _match_winner(101), _match_winner(102)),
)

KNOCKOUT_MATCHES: tuple[BracketMatch, ...] = (
    *ROUND_OF_32_MATCHES,
    *DOWNSTREAM_KNOCKOUT_MATCHES,
)


def get_round_of_32_matches() -> tuple[BracketMatch, ...]:
    """Return the configured Round-of-32 matches."""
    return ROUND_OF_32_MATCHES


def get_knockout_matches() -> tuple[BracketMatch, ...]:
    """Return all configured knockout matches except the third-place match."""
    return KNOCKOUT_MATCHES


def third_place_slots(
    matches: tuple[BracketMatch, ...] = ROUND_OF_32_MATCHES,
) -> list[tuple[int, BracketSlot]]:
    """Return third-place slots as ``(match_number, slot)`` pairs."""
    slots: list[tuple[int, BracketSlot]] = []
    for match in matches:
        for slot in [match.slot_a, match.slot_b]:
            if slot.source_type == SOURCE_THIRD_PLACE:
                slots.append((match.match_number, slot))
    return slots


def _normalize_third_place_group_key(
    qualified_third_groups: list[str] | tuple[str, ...],
) -> str:
    """Return the official Annex C lookup key for third-place group labels."""
    normalized_groups = tuple(
        sorted({str(group).strip().upper() for group in qualified_third_groups})
    )
    if len(normalized_groups) != 8:
        raise ValueError("Exactly 8 third-place groups must be supplied.")

    invalid_groups = sorted(set(normalized_groups).difference(GROUPS))
    if invalid_groups:
        raise ValueError(
            "Unsupported third-place group labels: " + ", ".join(invalid_groups)
        )
    return "".join(normalized_groups)


def _validate_official_third_place_slot_config(
    matches: tuple[BracketMatch, ...],
) -> None:
    """Verify that the bracket still matches the official Annex C slot layout."""
    slots_by_match = {
        match_number: slot for match_number, slot in third_place_slots(matches)
    }
    expected_matches = set(THIRD_PLACE_MATCH_TO_WINNER_SLOT)
    if set(slots_by_match) != expected_matches:
        raise ValueError(
            "Official Annex C third-place mapping is only supported for the "
            "configured FIFA 2026 Round-of-32 third-place slots."
        )

    for match_number, slot in slots_by_match.items():
        winner_slot = THIRD_PLACE_MATCH_TO_WINNER_SLOT[match_number]
        expected_groups = THIRD_PLACE_SLOT_ALLOWED_GROUPS[winner_slot]
        if tuple(slot.allowed_third_groups) != expected_groups:
            raise ValueError(
                "Round-of-32 third-place slot pools no longer match the "
                f"official Annex C layout for slot {winner_slot}."
            )


@lru_cache(maxsize=1)
def _official_third_place_assignment_table_cached() -> pd.DataFrame:
    """Load and validate the official FIFA Annex C assignment table."""
    table = pd.read_csv(OFFICIAL_THIRD_PLACE_ASSIGNMENT_PATH, dtype=str)
    required_columns = ["option", "qualified_third_groups", *THIRD_PLACE_WINNER_SLOTS]
    missing = set(required_columns).difference(table.columns)
    if missing:
        raise ValueError(
            "Missing official third-place assignment columns: "
            + ", ".join(sorted(missing))
        )

    table = table[required_columns].copy(deep=True)
    table["option"] = pd.to_numeric(table["option"], errors="raise").astype(int)
    table["qualified_third_groups"] = (
        table["qualified_third_groups"].astype(str).str.strip().str.upper()
    )
    for slot in THIRD_PLACE_WINNER_SLOTS:
        table[slot] = table[slot].astype(str).str.strip().str.upper()

    if len(table) != OFFICIAL_THIRD_PLACE_COMBINATION_COUNT:
        raise ValueError(
            "Official Annex C table must contain "
            f"{OFFICIAL_THIRD_PLACE_COMBINATION_COUNT} combinations."
        )
    if table["qualified_third_groups"].duplicated().any():
        raise ValueError(
            "Official Annex C table contains duplicate group combinations."
        )

    for row in table.to_dict("records"):
        key = str(row["qualified_third_groups"])
        assignments = [str(row[slot]) for slot in THIRD_PLACE_WINNER_SLOTS]
        if len(key) != 8 or set(key).difference(GROUPS):
            raise ValueError(f"Invalid Annex C third-place combination: {key}")
        if set(assignments) != set(key):
            raise ValueError(
                f"Annex C option {row['option']} does not assign exactly its "
                "qualified third-place groups."
            )
        for slot, group in zip(THIRD_PLACE_WINNER_SLOTS, assignments):
            if group not in THIRD_PLACE_SLOT_ALLOWED_GROUPS[slot]:
                raise ValueError(
                    f"Annex C option {row['option']} assigns Group {group} to "
                    f"unsupported slot {slot}."
                )
    return table.sort_values("option", kind="mergesort").reset_index(drop=True)


def official_third_place_assignment_table() -> pd.DataFrame:
    """Return the official FIFA Annex C third-place assignment table."""
    return _official_third_place_assignment_table_cached().copy(deep=True)


@lru_cache(maxsize=1)
def _official_third_place_assignment_lookup() -> dict[str, dict[int, str]]:
    """Return lookup from third-place group combination to match assignment."""
    table = _official_third_place_assignment_table_cached()
    lookup: dict[str, dict[int, str]] = {}
    for row in table.to_dict("records"):
        key = str(row["qualified_third_groups"])
        lookup[key] = {
            THIRD_PLACE_SLOT_TO_MATCH[slot]: str(row[slot])
            for slot in THIRD_PLACE_WINNER_SLOTS
        }
    return lookup


def assign_third_place_groups_to_slots(
    qualified_third_groups: list[str] | tuple[str, ...],
    matches: tuple[BracketMatch, ...] = ROUND_OF_32_MATCHES,
) -> dict[int, str]:
    """Assign qualified third-place groups using FIFA World Cup 26 Annex C."""
    _validate_official_third_place_slot_config(matches)
    key = _normalize_third_place_group_key(qualified_third_groups)
    lookup = _official_third_place_assignment_lookup()
    if key not in lookup:
        raise ValueError(
            "No official FIFA Annex C third-place assignment is configured for "
            f"group combination {key}."
        )
    return dict(sorted(lookup[key].items()))


def bracket_config_table() -> pd.DataFrame:
    """Return the bracket configuration as a readable dataframe."""
    rows: list[dict[str, object]] = []
    for match in KNOCKOUT_MATCHES:
        rows.append(
            {
                "match_number": match.match_number,
                "round": match.round_name,
                "slot_a": match.slot_a.label,
                "slot_b": match.slot_b.label,
                "slot_a_type": match.slot_a.source_type,
                "slot_b_type": match.slot_b.source_type,
                "notes": match.notes,
            }
        )
    return pd.DataFrame(rows)
