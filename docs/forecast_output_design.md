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

`scripts/generate_fixture_predictions.py` can print predictions or, only when `--output` is supplied, write `data/tournament/fixture_predictions_2026.csv` for the simulation script. No model artifacts or processed feature files are written by default.
