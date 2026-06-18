# `fixtures_2026.csv` Template

Create this file manually at:

```text
data/tournament/fixtures_2026.csv
```

Do not fabricate official fixtures. Use this template only as a schema guide, then fill rows from an official fixture source.

## Header

```csv
match_id,match_date,team_a,team_b,group,stage,kickoff_time,venue,city,country,neutral,is_neutral,source,last_updated
```

## Columns

Required:

- `match_id`
- `match_date`
- `team_a`
- `team_b`
- `group`
- `stage`

Optional:

- `kickoff_time`
- `venue`
- `city`
- `country`
- `neutral`
- `is_neutral`
- `source`
- `last_updated`

## Notes

- `match_id` must be unique.
- `match_date` must parse as a date.
- `team_a` and `team_b` must be non-empty and different.
- `group` is required for group-stage fixtures.
- `stage` should use values such as `group`, `round_of_32`, `round_of_16`, `quarterfinal`, `semifinal`, `third_place`, or `final`.
- If `neutral` and `is_neutral` are missing for 2026 fixtures, the ingestion layer defaults to neutral.
- Host-country effects for USA, Canada, and Mexico are not modeled through generic home advantage yet.

## Non-Official Example Shape

The rows below are illustrative only and should not be copied as real tournament data.

```csv
match_id,match_date,team_a,team_b,group,stage,kickoff_time,venue,city,country,neutral,is_neutral,source,last_updated
example_match_001,2026-06-12,Example Team A,Example Team B,A,group,,,,,,true,manual_example,2026-06-01
```
