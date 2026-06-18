# Tournament Data Ingestion

The first tournament-state inputs are manually maintained local CSV files:

```text
data/tournament/fixtures_2026.csv
data/tournament/results_2026.csv
```

This stage does not require paid data, scraping, external APIs, or automatic downloads.

## Why Tournament Data Is Separate

Historical match results in `data/raw/results.csv` are used to train the selected baseline model. Current tournament files are different:

- `fixtures_2026.csv` describes scheduled 2026 matches that need predictions.
- `results_2026.csv` describes completed 2026 matches that should be fixed in live tournament state.
- `fixture_predictions_2026.csv` is generated output containing model probabilities.

The selected baseline still trains only on baseline-eligible completed historical matches. The fixture file is used to build future prediction rows and simulation inputs, not to retrain on 2026 World Cup results.

Completed 2026 results can update standings, prediction audit, and live simulation state. They cannot train the first baseline model.

## Fixture Schema

Required columns:

- `match_id`
- `match_date`
- `team_a`
- `team_b`
- `group`
- `stage`

Optional columns:

- `kickoff_time`
- `venue`
- `city`
- `country`
- `neutral`
- `is_neutral`
- `source`
- `last_updated`

See `docs/fixtures_2026_template.md` for the CSV header and an explicitly non-official example shape.

## Result Schema

`data/tournament/results_2026.csv` is optional until matches are completed. When present, it stores manually maintained completed results with:

- `match_id`
- `match_date`
- `team_a`
- `team_b`
- `team_a_goals`
- `team_b_goals`
- `result`
- `status`

Optional fields include extra-time, penalty, source, and maintenance metadata. See `docs/results_2026_template.md`.

Results must join to fixtures or fixture predictions by `match_id`, and `team_a`/`team_b` orientation must match the fixture row exactly.

## Validation Rules

The ingestion layer validates that:

- `match_id` values are unique.
- `match_date` parses as a date.
- `team_a` and `team_b` are non-empty and different.
- `group` is present for group-stage fixtures.
- `stage` uses known stage labels such as `group`, `round_of_32`, `round_of_16`, `quarterfinal`, `semifinal`, `third_place`, or `final`.

If `neutral` and `is_neutral` are missing for 2026 fixtures, the normalized fixture table defaults to neutral. Generic historical home advantage should not be assumed for World Cup fixtures. USA, Canada, and Mexico host effects remain a later explicit tournament-state or venue feature.

## Generating Fixture Predictions

Run:

```bash
python scripts/generate_fixture_predictions.py
```

The script:

1. Loads historical completed matches through the existing cleaning pipeline.
2. Loads and validates `fixtures_2026.csv`.
3. Trains the selected baseline in memory.
4. Builds leakage-safe fixture feature rows.
5. Prints full 3-class probabilities and favorite display labels.

By default, no prediction file is written.

To write predictions explicitly:

```bash
python scripts/generate_fixture_predictions.py --forecast-mode backfilled_ex_ante --output data/tournament/fixture_predictions_2026.csv
```

The written file includes prediction metadata such as generation timestamp, training cutoff date, feature cutoff date, model name, selected baseline label, and `is_backfilled`.

`forecast_mode` is required metadata:

- `pre_tournament` uses only information available through the default training cutoff, `2026-06-10`, unless a cutoff is explicitly overridden for diagnostics.
- `backfilled_ex_ante` also uses `2026-06-10` as the default feature cutoff, but it may be generated after some fixture dates. Those rows are marked backfilled and are useful for methodology/demo work, not as true timestamped pre-match predictions.
- `live` is reserved for forecasts that condition on completed 2026 results through an explicit feature cutoff or current date. The first baseline still cannot train on those completed 2026 World Cup results.

The `feature_cutoff_date` field should never be blank. It records the latest completed-match date allowed into fixture feature construction.

## Simulation Input

The group-stage simulator consumes fixture prediction probabilities, not favorites alone. It expects rows with:

- `match_id`
- `group`
- `team_a`
- `team_b`
- `p_team_a_win`
- `p_draw`
- `p_team_b_win`

Once `fixture_predictions_2026.csv` exists, `scripts/simulate_group_stage.py` loads it automatically. If it is missing, the script keeps using a synthetic example for smoke testing.

If `data/tournament/results_2026.csv` exists, the simulation script validates it, fixes completed matches, and samples only remaining unplayed matches from prediction probabilities. To force a diagnostic probability-only run:

```bash
python scripts/simulate_group_stage.py --ignore-results
```

The prediction file is generated output. Keep it uncommitted unless the project later defines a deliberate snapshot policy for versioned prediction logs.
