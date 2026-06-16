# Player Data Schemas

These proposed schemas define the player data layer for future work. This milestone does not implement scraping, API calls, downloads, feature calculations, modeling, or simulation.

## `data/player/player_registry.csv`

Central identity table used to join players across data sources.

| Column | Type | Description |
| --- | --- | --- |
| `player_id` | string | Stable internal player identifier. |
| `player_name` | string | Canonical display name. |
| `normalized_name` | string | Normalized name for matching and deduplication. |
| `date_of_birth` | date | Date of birth in `YYYY-MM-DD` format where available. |
| `nationality` | string | Player nationality. |
| `national_team` | string | National team represented or eligible context used by the project. |
| `current_club` | string | Current club as of `last_updated`. |
| `primary_position` | string | Main playing position. |
| `secondary_positions` | string | Delimited secondary positions. |
| `transfermarkt_id` | string | Transfermarkt-style source identifier, if available. |
| `fbref_id` | string | FBref-style source identifier, if available. |
| `statsbomb_id` | string | StatsBomb source identifier, if available. |
| `wyscout_id` | string | Wyscout source identifier, if available. |
| `api_football_id` | string | API-Football source identifier, if available. |
| `last_updated` | timestamp | Timestamp when the registry row was last reviewed or updated. |

## `data/player/squad_rosters_2026.csv`

Tournament squad roster table.

| Column | Type | Description |
| --- | --- | --- |
| `tournament` | string | Tournament name, such as `FIFA World Cup 2026`. |
| `team` | string | National team. |
| `player_id` | string | Internal player identifier from `player_registry.csv`. |
| `player_name` | string | Player display name. |
| `shirt_number` | integer | Tournament shirt number. |
| `position` | string | Tournament roster position. |
| `current_club` | string | Club listed for the tournament roster. |
| `is_goalkeeper` | boolean | Whether the player is a goalkeeper. |
| `roster_status` | string | Status such as `named`, `withdrawn`, `replacement`, `doubtful`, or `unavailable`. |
| `source` | string | Source used for the roster row. |
| `last_updated` | timestamp | Timestamp when the row was last updated. |

## `data/player/player_match_appearances.csv`

Player appearance-level table for club and international matches.

| Column | Type | Description |
| --- | --- | --- |
| `match_id` | string | Source or internal match identifier. |
| `match_date` | date | Match date in `YYYY-MM-DD` format. |
| `player_id` | string | Internal player identifier from `player_registry.csv`. |
| `player_name` | string | Player display name at time of source collection. |
| `team` | string | Player's team in the match. |
| `opponent` | string | Opponent in the match. |
| `club_or_country` | string | `club` or `country`. |
| `competition` | string | Competition name. |
| `is_international` | boolean | Whether the match is an international match. |
| `is_friendly` | boolean | Whether the match is a friendly. |
| `minutes` | integer | Minutes played. |
| `started` | boolean | Whether the player started. |
| `goals` | integer | Goals scored. |
| `assists` | integer | Assists credited. |
| `yellow_cards` | integer | Yellow cards. |
| `red_cards` | integer | Red cards. |
| `source` | string | Source used for the appearance row. |

## `data/player/player_recent_form.csv`

Player-level form features as of a specific date.

| Column | Type | Description |
| --- | --- | --- |
| `player_id` | string | Internal player identifier from `player_registry.csv`. |
| `as_of_date` | date | Feature date boundary. Only earlier data should be included. |
| `lookback_days` | integer | Number of days included in the form window. |
| `appearances` | integer | Recency-window appearances. |
| `starts` | integer | Recency-window starts. |
| `minutes` | integer | Recency-window minutes. |
| `goals` | integer | Recency-window goals. |
| `assists` | integer | Recency-window assists. |
| `cards` | integer | Recency-window cards. |
| `recent_minutes_score` | float | Recency-weighted minutes and match fitness score. |
| `attacking_form_score` | float | Position-normalized attacking form score. |
| `creative_form_score` | float | Position-normalized creative form score. |
| `defensive_form_score` | float | Position-normalized defensive form score. |
| `goalkeeper_form_score` | float | Goalkeeper-specific form score. |
| `discipline_penalty` | float | Penalty for cards or suspension risk. |
| `international_integration_score` | float | National-team involvement and integration score. |
| `club_strength_adjustment` | float | Adjustment for club or competition context. |
| `player_live_form_index` | float | Final composite player form index. |

## `data/player/team_player_features.csv`

Team-level aggregates derived from player features.

| Column | Type | Description |
| --- | --- | --- |
| `team` | string | National team. |
| `as_of_date` | date | Feature date boundary. Only earlier data should be included. |
| `expected_xi_strength` | float | Estimated strength of the likely starting XI. |
| `top_14_strength` | float | Estimated strength of likely starters plus primary substitutes. |
| `squad_depth_score` | float | Depth beyond the first-choice lineup. |
| `attacking_unit_form` | float | Aggregated attacking-unit player form. |
| `midfield_progression_form` | float | Aggregated midfield creation and progression form. |
| `defensive_unit_form` | float | Aggregated defensive-unit form. |
| `goalkeeper_form` | float | Likely goalkeeper or goalkeeper-pool form. |
| `injury_or_availability_penalty` | float | Team-level penalty for unavailable or limited players. |
| `national_team_cohesion` | float | Recent national-team integration and shared involvement. |
| `source_version` | string | Version of the player-feature source or build logic. |

## Schema Principles

- Join source tables through `player_id` whenever possible.
- Keep `as_of_date` explicit on all feature tables.
- Treat recent friendlies as selection and availability signals more than quality signals.
- Preserve source fields so future audits can trace feature values back to their inputs.
- Do not commit raw player datasets without an explicit data policy update.
