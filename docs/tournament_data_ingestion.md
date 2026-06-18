# Tournament Data Ingestion

The first tournament-state source is a manually maintained local fixture file:

```text
data/tournament/fixtures_2026.csv
```

This stage does not require paid data, scraping, external APIs, or automatic downloads.

## Why Tournament Data Is Separate

Historical match results in `data/raw/results.csv` are used to train the selected baseline model. Current tournament fixture data is different: it describes scheduled 2026 matches that need predictions.

The selected baseline still trains only on baseline-eligible completed historical matches. The fixture file is used to build future prediction rows and simulation inputs, not to retrain on 2026 World Cup results.

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
python scripts/generate_fixture_predictions.py --output data/tournament/fixture_predictions_2026.csv
```

The written file includes prediction metadata such as generation timestamp, training cutoff date, feature cutoff date, model name, selected baseline label, and `is_backfilled`.

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
