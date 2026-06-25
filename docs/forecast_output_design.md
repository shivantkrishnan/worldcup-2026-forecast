# Forecast Output Design

The first forecast output layer turns scheduled fixtures into model-ready feature rows, trains the selected baseline in memory, and returns full 3-class match probabilities.

Fixture rows come from the manually maintained local tournament file `data/tournament/fixtures_2026.csv`. That file is separate from historical training data and is validated before predictions are generated.

## Probability Output

The model-facing output is always the full probability vector:

- `p_team_a_win`
- `p_draw`
- `p_team_b_win`

The dashboard needs the complete distribution because tournament simulation, prediction audit, calibration diagnostics, and user-facing uncertainty all depend on probabilities, not only the most likely class.

## User-Facing Favorite

The user-facing output displays a favorite plus the full probabilities:

- If `predicted_class = team_a_win`, `favorite_display = team_a`.
- If `predicted_class = team_b_win`, `favorite_display = team_b`.
- If `predicted_class = draw`, `favorite_display = Draw`.

The `confidence_label` is a deterministic display aid based on the largest probability and its margin over the second-largest probability. It is not a statistical confidence interval.

## Future Fixtures vs Completed Matches

Completed training rows include scores and result labels because they are historical outcomes. Scheduled fixtures do not have scores or result labels, and the fixture feature builder does not require them.

To reuse the same feature pipeline safely, each fixture is temporarily represented as a one-row placeholder after filtering completed history. The placeholder's dummy outcome is not used as a feature because rolling team-form values are shifted and Elo features are emitted before the current row is updated.

## Leakage Prevention

For each fixture, features are computed from completed matches strictly before the fixture date. With date-only timestamps, same-date completed matches are excluded because kickoff ordering is unknown.

If `feature_cutoff_date` is supplied, completed history is further restricted to:

```text
match_date <= feature_cutoff_date
```

and still strictly before the fixture date.

Every generated prediction row must carry a non-empty `feature_cutoff_date`.
This makes the information set auditable after the fact: reviewers should be
able to tell whether a row used only pre-tournament history, a later live-state
cutoff, or some intentionally overridden diagnostic cutoff.

## Forecast Modes

Fixture predictions now use explicit forecast-mode metadata:

- `pre_tournament`: uses the selected baseline trained through
  `2026-06-10` and defaults fixture features to the same cutoff.
- `backfilled_ex_ante`: reconstructs what the pre-tournament model would have
  predicted, even if the file is generated later. Rows for fixture dates before
  the generation date are marked `is_backfilled = true`.
- `live`: reserved for forecasts that use completed 2026 results to update
  tournament state and fixture features while still training the baseline only
  through `2026-06-10`. Live prediction generation appends completed
  `results_2026.csv` rows through an explicit `feature_cutoff_date` for feature
  construction only. These completed 2026 results are not used for model
  fitting, calibration fitting, imputation fitting, or scaling fitting.
  Backfilled rows are not silently treated as true live predictions.

The default mode is `backfilled_ex_ante` if prediction generation happens after
any fixture date in the file; otherwise it is `pre_tournament`.

## Display Status Guardrail

The dashboard display layer must join fixtures, predictions, and completed
results before rendering match rows. It should assign each fixture one explicit
display status:

- `completed`: a verified result exists in `results_2026.csv`.
- `scheduled`: no result exists and prediction probabilities are available.
- `prediction_missing`: neither a completed result nor a prediction row is
  available.

Completed matches should show actual scores and actual result first. Model
probabilities for completed matches may remain available for prediction audit,
but they should be labeled as audit probabilities, not current forecasts.
Backfilled ex-ante rows should clearly state that they were reconstructed after
the fact and are not true live predictions.

Only scheduled matches without completed results should be displayed as current
predictions.

The Streamlit dashboard consumes the display-safe match table directly. Its
matches, groups, and team views therefore inherit the same guardrail:
completed rows show score/status first, scheduled rows show W/D/L
probabilities, and completed-match probabilities appear only when the user
opens an audit-oriented context.

## Live Fixture-Feature Updating

Live prediction generation uses two separate match histories:

- Training history: baseline-eligible historical rows through `2026-06-10`.
- Feature history: the same baseline history plus verified completed 2026 World
  Cup results through the live `feature_cutoff_date`.

The selected baseline is trained on the training history only. The augmented
feature history is used only to compute rolling team form and pre-match Elo
state for remaining fixtures.

By default, live mode emits predictions only for fixtures that do not already
have completed results. A deliberate audit option may include completed
fixtures, but those rows must be interpreted as audit/reconstruction rows, not
current forecasts.

Use a separate live output path such as:

```text
data/tournament/fixture_predictions_2026_live.csv
```

This avoids confusing current live probabilities with the reproducible
backfilled ex-ante file. Generated live prediction files should remain
uncommitted unless the project defines a snapshot/versioning policy.

## 2026 World Cup Neutral Defaults

For future 2026 World Cup fixtures, missing `neutral` or `is_neutral` values default to neutral. This prevents the selected historical home-adjustment parameter from being blindly applied to World Cup matches.

The 50-point Elo home adjustment is mainly for learning from historical non-neutral matches. Host effects for USA, Canada, and Mexico should be modeled later as explicit tournament-state or venue features, not as generic home advantage.

## Simulation Readiness

This layer prepares the project for Monte Carlo tournament simulation by producing one row per fixture with:

- model-ready numeric features,
- full 3-class probabilities,
- predicted class,
- favorite display text,
- a simple display confidence label.

The simulator can later consume these probabilities directly for group-stage and knockout-path draws without retraining the model or rebuilding feature logic.

The first simulator now consumes these probability rows for group-stage simulations. It samples the full 3-class distribution, not just the favorite, which lets draws affect group standings naturally.

When `data/tournament/results_2026.csv` is available, the simulator fixes those
completed outcomes and samples only remaining unplayed matches. This separates
live tournament-state simulation from a backfilled ex-ante simulation that
samples every row from reconstructed probabilities.

`scripts/generate_fixture_predictions.py` can print predictions or, only when `--output` is supplied, write `data/tournament/fixture_predictions_2026.csv` for the simulation script. No model artifacts or processed feature files are written by default.

`fixture_predictions_2026.csv` is a reproducible generated output. It should
remain uncommitted unless the project defines a snapshot/versioning policy for
prediction files.

## Public Demo Refresh

The public app at `https://wc2026-forecast.streamlit.app/` uses the committed
live snapshot `data/tournament/fixture_predictions_2026_live.csv`. During the
tournament, GitHub Actions refreshes this file on a schedule after fetching
official completed results and regenerating remaining-fixture live predictions.

The refresh is intentionally outside the Streamlit runtime. The app reads
committed CSV snapshots; it does not scrape, call APIs, or train the model when
a user opens the page. Streamlit Community Cloud updates after the workflow
commits changed dashboard CSVs to GitHub.

The refresh workflow validates that:

- completed matches are present in `results_2026.csv`,
- completed matches are omitted from the live prediction snapshot,
- remaining live prediction probabilities sum to 1,
- completed result count plus live prediction count equals the 72 group-stage
  fixtures,
- feature cutoff metadata is at least as recent as the latest completed result
  date.

This process is not truly real-time and can lag final whistle. Official
source/API downtime or result corrections may require manual review.
