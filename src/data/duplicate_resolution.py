"""In-memory duplicate resolution for canonical match data."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.data.clean_results import CANONICAL_MATCH_COLUMNS

OUTCOME_FIELDS = [
    "team_a",
    "team_b",
    "match_date",
    "tournament",
    "team_a_goals",
    "team_b_goals",
    "result",
]

QUARANTINE_REASON_COLUMN = "quarantine_reason"
METADATA_DUPLICATE = "metadata_duplicate"
CONFLICTING_DUPLICATE = "conflicting_duplicate"


@dataclass(frozen=True)
class DuplicateReport:
    """Structured duplicate-resolution summary."""

    duplicate_group_count: int
    duplicate_row_count: int
    metadata_duplicate_group_count: int
    conflicting_duplicate_group_count: int
    quarantined_row_count: int
    resolved_match_count: int
    passed: bool
    messages: list[str]

    def to_dict(self) -> dict[str, object]:
        """Return the report as a plain dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class DuplicateResolutionResult:
    """Container for resolved matches, quarantine rows, and report metadata."""

    resolved_matches: pd.DataFrame
    quarantined_matches: pd.DataFrame
    duplicate_report: DuplicateReport


def _empty_quarantine_frame() -> pd.DataFrame:
    """Return an empty quarantine frame with stable columns."""
    return pd.DataFrame(columns=[*CANONICAL_MATCH_COLUMNS, QUARANTINE_REASON_COLUMN])


def _build_report(
    duplicate_group_count: int,
    duplicate_row_count: int,
    metadata_duplicate_group_count: int,
    conflicting_duplicate_group_count: int,
    quarantined_row_count: int,
    resolved_match_count: int,
    resolved_match_ids_are_unique: bool,
) -> DuplicateReport:
    messages: list[str] = []

    if duplicate_group_count == 0:
        messages.append("No duplicate match_id groups found.")
    else:
        messages.append(
            f"Found {duplicate_group_count} duplicate match_id groups "
            f"covering {duplicate_row_count} rows."
        )

    if metadata_duplicate_group_count:
        messages.append(
            f"Resolved {metadata_duplicate_group_count} metadata duplicate groups by "
            "keeping the first row and quarantining extras."
        )

    if conflicting_duplicate_group_count:
        messages.append(
            f"Quarantined {conflicting_duplicate_group_count} conflicting duplicate "
            "groups in full."
        )

    if quarantined_row_count:
        messages.append(f"Quarantined {quarantined_row_count} duplicate rows.")

    if resolved_match_ids_are_unique:
        messages.append("Resolved matches have unique match_id values.")
    else:
        messages.append("Resolved matches still contain duplicate match_id values.")

    return DuplicateReport(
        duplicate_group_count=duplicate_group_count,
        duplicate_row_count=duplicate_row_count,
        metadata_duplicate_group_count=metadata_duplicate_group_count,
        conflicting_duplicate_group_count=conflicting_duplicate_group_count,
        quarantined_row_count=quarantined_row_count,
        resolved_match_count=resolved_match_count,
        passed=resolved_match_ids_are_unique,
        messages=messages,
    )


def resolve_duplicate_matches(df: pd.DataFrame) -> DuplicateResolutionResult:
    """Resolve duplicate canonical matches according to the quarantine policy."""
    if df.empty:
        report = _build_report(
            duplicate_group_count=0,
            duplicate_row_count=0,
            metadata_duplicate_group_count=0,
            conflicting_duplicate_group_count=0,
            quarantined_row_count=0,
            resolved_match_count=0,
            resolved_match_ids_are_unique=True,
        )
        return DuplicateResolutionResult(
            resolved_matches=df.copy(deep=True),
            quarantined_matches=_empty_quarantine_frame(),
            duplicate_report=report,
        )

    working = df.copy(deep=True)
    working["_original_order"] = range(len(working))
    stable = working.sort_values(
        ["match_id", "_original_order"],
        kind="mergesort",
    )

    duplicate_mask = stable["match_id"].duplicated(keep=False)
    unique_rows = stable.loc[~duplicate_mask].copy()
    duplicate_rows = stable.loc[duplicate_mask].copy()

    kept_rows: list[pd.DataFrame] = [unique_rows]
    quarantined_groups: list[pd.DataFrame] = []
    duplicate_group_count = 0
    metadata_duplicate_group_count = 0
    conflicting_duplicate_group_count = 0

    for _, group in duplicate_rows.groupby("match_id", sort=False):
        duplicate_group_count += 1
        outcome_versions = group[OUTCOME_FIELDS].drop_duplicates()

        if len(outcome_versions) == 1:
            metadata_duplicate_group_count += 1
            kept_rows.append(group.head(1))
            quarantine = group.iloc[1:].copy()
            quarantine[QUARANTINE_REASON_COLUMN] = METADATA_DUPLICATE
            quarantined_groups.append(quarantine)
            continue

        conflicting_duplicate_group_count += 1
        quarantine = group.copy()
        quarantine[QUARANTINE_REASON_COLUMN] = CONFLICTING_DUPLICATE
        quarantined_groups.append(quarantine)

    resolved = pd.concat(kept_rows, ignore_index=True)
    resolved = resolved.sort_values("_original_order", kind="mergesort")
    resolved = resolved.drop(columns=["_original_order"]).reset_index(drop=True)
    resolved = resolved.loc[:, CANONICAL_MATCH_COLUMNS]

    if quarantined_groups:
        quarantined = pd.concat(quarantined_groups, ignore_index=True)
        quarantined = quarantined.sort_values("_original_order", kind="mergesort")
        quarantined = quarantined.drop(columns=["_original_order"]).reset_index(
            drop=True
        )
        quarantined = quarantined.loc[
            :, [*CANONICAL_MATCH_COLUMNS, QUARANTINE_REASON_COLUMN]
        ]
    else:
        quarantined = _empty_quarantine_frame()

    duplicate_row_count = int(len(duplicate_rows))
    quarantined_row_count = int(len(quarantined))
    resolved_match_ids_are_unique = bool(resolved["match_id"].is_unique)
    report = _build_report(
        duplicate_group_count=duplicate_group_count,
        duplicate_row_count=duplicate_row_count,
        metadata_duplicate_group_count=metadata_duplicate_group_count,
        conflicting_duplicate_group_count=conflicting_duplicate_group_count,
        quarantined_row_count=quarantined_row_count,
        resolved_match_count=int(len(resolved)),
        resolved_match_ids_are_unique=resolved_match_ids_are_unique,
    )

    return DuplicateResolutionResult(
        resolved_matches=resolved,
        quarantined_matches=quarantined,
        duplicate_report=report,
    )
