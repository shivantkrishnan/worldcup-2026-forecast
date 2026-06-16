# Data Schemas

These are proposed schemas for local CSV files used by the tournament-state and prediction-audit layers. They define structure only; this milestone does not implement ingestion, scraping, APIs, modeling, or simulation.

## `data/tournament/fixtures_2026.csv`

One row per scheduled 2026 World Cup match.

| Column | Type | Description |
| --- | --- | --- |
| `match_id` | string | Stable unique match identifier. |
| `match_date` | date | Match date in `YYYY-MM-DD` format. |
| `kickoff_time_local` | string | Local kickoff time, preferably ISO-like with timezone noted when available. |
| `kickoff_time_utc` | timestamp | UTC kickoff timestamp for ordering and pre-match logging checks. |
| `group` | string | Group label, if applicable. Blank for knockout matches if not applicable. |
| `stage` | string | Tournament stage, such as `group`, `round_of_32`, `round_of_16`, `quarterfinal`, `semifinal`, `third_place`, or `final`. |
| `team_a` | string | Team listed as Team A for modeling and display. |
| `team_b` | string | Team listed as Team B for modeling and display. |
| `venue` | string | Stadium or venue name. |
| `city` | string | Host city. |
| `country` | string | Host country. |
| `status` | string | Fixture state, such as `scheduled`, `in_progress`, `completed`, `postponed`, or `cancelled`. |

## `data/tournament/results_2026.csv`

One row per completed or partially known 2026 World Cup result.

| Column | Type | Description |
| --- | --- | --- |
| `match_id` | string | Stable match identifier matching `fixtures_2026.csv`. |
| `match_date` | date | Match date in `YYYY-MM-DD` format. |
| `team_a` | string | Team A, matching the fixture orientation. |
| `team_b` | string | Team B, matching the fixture orientation. |
| `team_a_goals` | integer | Team A goals after regulation plus extra time where applicable. |
| `team_b_goals` | integer | Team B goals after regulation plus extra time where applicable. |
| `result` | string | 3-class regulation/model outcome from Team A perspective: `team_a_win`, `draw`, or `team_b_win`. |
| `status` | string | Result state, usually `completed`; may include `in_progress` if live state is later added. |
| `went_to_extra_time` | boolean | Whether the match went to extra time. |
| `went_to_penalties` | boolean | Whether the match went to penalties. |
| `team_a_penalties` | integer | Team A penalty shootout goals, blank if no shootout. |
| `team_b_penalties` | integer | Team B penalty shootout goals, blank if no shootout. |

## `data/predictions/prediction_log.csv`

One row per match prediction for a specific model version and forecast mode.

| Column | Type | Description |
| --- | --- | --- |
| `match_id` | string | Stable match identifier matching tournament files. |
| `match_date` | date | Match date in `YYYY-MM-DD` format. |
| `team_a` | string | Team A, matching prediction orientation. |
| `team_b` | string | Team B, matching prediction orientation. |
| `model_version` | string | Model or ruleset version used to generate the prediction. |
| `prediction_timestamp` | timestamp | Timestamp when the prediction row was generated. Must be before kickoff for true logged predictions. |
| `feature_cutoff_timestamp` | timestamp | Latest timestamp included in the feature and tournament-state inputs. |
| `training_cutoff_date` | date | Training data cutoff for the model, defaulting to `2026-06-10` for the first baseline. |
| `p_team_a_win` | float | Predicted probability that Team A wins. |
| `p_draw` | float | Predicted probability of a draw. |
| `p_team_b_win` | float | Predicted probability that Team B wins. |
| `predicted_class` | string | Highest-probability class. |
| `actual_result` | string | Actual 3-class result once known: `team_a_win`, `draw`, or `team_b_win`. |
| `log_loss` | float | Per-match log loss once the actual result is known. |
| `brier_score` | float | Per-match multiclass Brier score once the actual result is known. |
| `is_backfilled` | boolean | `true` if the prediction was generated after the match was already completed or before logging began. |
| `notes` | string | Free-text audit notes, caveats, or data-quality flags. |

## Versioning Notes

- Keep schema changes documented here before implementing readers or validators.
- Preserve `match_id` as the join key across fixtures, results, and predictions.
- Keep `training_cutoff_date` explicit in prediction logs to protect the baseline leakage boundary.
- Do not store raw external datasets in this repository.
