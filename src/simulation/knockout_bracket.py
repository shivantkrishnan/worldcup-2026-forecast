"""2026 World Cup knockout bracket configuration and slot assignment."""

from __future__ import annotations

from dataclasses import dataclass

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
    "Round-of-32 slot pools follow the published FIFA World Cup 26 schedule "
    "structure. The full Annex C third-place combination table is not encoded "
    "yet; selected third-place groups are assigned by deterministic constrained "
    "matching within the allowed slot pools."
)


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


def assign_third_place_groups_to_slots(
    qualified_third_groups: list[str] | tuple[str, ...],
    matches: tuple[BracketMatch, ...] = ROUND_OF_32_MATCHES,
) -> dict[int, str]:
    """Assign qualified third-place groups to allowed Round-of-32 slots.

    This is a deterministic constrained-matching approximation. It validates
    the published slot pools and fails clearly if a selected group combination
    cannot be assigned without duplicates.
    """
    normalized_groups = tuple(sorted({str(group).strip().upper() for group in qualified_third_groups}))
    if len(normalized_groups) != 8:
        raise ValueError("Exactly 8 third-place groups must be supplied.")

    invalid_groups = sorted(set(normalized_groups).difference(GROUPS))
    if invalid_groups:
        raise ValueError(
            "Unsupported third-place group labels: " + ", ".join(invalid_groups)
        )

    slots = third_place_slots(matches)
    if len(slots) != 8:
        raise ValueError("Round-of-32 bracket must contain exactly 8 third-place slots.")

    def backtrack(
        remaining_slots: list[tuple[int, BracketSlot]],
        remaining_groups: tuple[str, ...],
        assignment: dict[int, str],
    ) -> dict[int, str] | None:
        if not remaining_slots:
            return assignment if not remaining_groups else None

        ordered_slots = sorted(
            remaining_slots,
            key=lambda item: (
                len(set(item[1].allowed_third_groups).intersection(remaining_groups)),
                item[0],
            ),
        )
        match_number, slot = ordered_slots[0]
        rest_slots = [item for item in remaining_slots if item[0] != match_number]
        candidates = [
            group for group in remaining_groups if group in slot.allowed_third_groups
        ]
        for group in candidates:
            next_groups = tuple(item for item in remaining_groups if item != group)
            next_assignment = {**assignment, match_number: group}
            result = backtrack(rest_slots, next_groups, next_assignment)
            if result is not None:
                return result
        return None

    assignment = backtrack(slots, normalized_groups, {})
    if assignment is None:
        raise ValueError(
            "Could not assign qualified third-place groups to Round-of-32 slots "
            "under the configured slot pools: "
            + ", ".join(normalized_groups)
        )
    return dict(sorted(assignment.items()))


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
