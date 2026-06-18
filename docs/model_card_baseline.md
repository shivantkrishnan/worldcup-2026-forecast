# Baseline Model Card

## Model Name

`calibrated_logistic_regression`

Current selected baseline: sigmoid-calibrated multinomial logistic regression over leakage-safe rolling team-form and pre-match Elo features using K-factor `10` and a `50`-point non-neutral home adjustment.

The K-factor is an empirically selected rating-smoothing parameter for this international football forecasting task. It should not be interpreted as a universal Elo constant or a chess/FIDE convention.

## Intended Use

This model is intended to provide baseline 3-class international football match probabilities:

- `team_a_win`
- `draw`
- `team_b_win`

It is the current probability-quality baseline for future pre-tournament and live forecasting work.

## Not Intended Use

This model is not intended for:

- Betting advice or financial decision-making.
- Final World Cup simulation quality claims.
- Player availability forecasting.
- Squad-strength modeling.
- Market-implied probability modeling.
- Claims that every probability bin is perfectly calibrated.

## Training Data

The v1 historical source is a manually downloaded international football results CSV placed locally at:

```text
data/raw/results.csv
```

The first baseline trains only on canonical completed matches with:

```text
match_date <= 2026-06-10
```

Completed 2026 World Cup matches after that cutoff may be used for standings, tournament state, live simulation state, and evaluation/audit, but not for first-baseline training.

## Target Definition

The target is the completed-match result from `team_a`'s perspective:

- `team_a_win`
- `draw`
- `team_b_win`

For the raw historical dataset, `team_a` maps to `home_team` and `team_b` maps to `away_team`. Neutral-site matches still use the listed first team as `team_a`.

## Features Used

The baseline uses leakage-safe team-form features built from a long team-match panel plus pre-match Elo-style team strength features.

Feature groups include:

- Prior matches played.
- Days since last match.
- Rolling points, goals, goal difference, win/draw/loss rates.
- Expanding points, goal difference, and win rate.
- Match-level differential features comparing `team_a` to `team_b`.
- Pre-match Elo ratings for both teams.
- Elo rating difference from `team_a`'s perspective.
- Elo effective rating difference after any non-neutral home adjustment.
- Elo expected score for `team_a`.
- Numeric home advantage applied for the match.
- Prior Elo match counts for both teams.

Current-match outcomes are excluded from feature calculations through shift/lag logic before rolling or expanding calculations.

Elo features are emitted before rating updates. Because only match dates are available, ratings are updated after the full date block so same-date results cannot affect same-date feature rows. Home advantage affects expected score for non-neutral matches only; it does not permanently inflate underlying ratings.

## Validation Design

Validation is time-aware.

The single holdout trains before `2022-01-01` and tests on baseline-eligible matches from `2022-01-01` through `2026-06-10`.

Rolling-origin validation uses expanding training windows and future test windows through `2026-06-10`. Each split refits preprocessing, imputation, scaling, logistic regression, and sigmoid calibration using training rows only.

## Metrics

Primary metrics:

- Log loss.
- Multiclass Brier score.
- Calibration diagnostics.

Secondary metric:

- Accuracy.

Single-holdout selected-model results:

| Feature Set / Model | Log Loss | Brier | Accuracy | ECE |
| --- | ---: | ---: | ---: | ---: |
| No-Elo calibrated logistic | 1.203647 | 0.548502 | 0.579812 | 0.032929 |
| Simple Elo calibrated logistic, K=20/home=0 | 1.202069 | 0.521042 | 0.594243 | 0.039671 |
| Selected Elo calibrated logistic, K=10/home=50 | 1.192015 | 0.522831 | 0.594480 | 0.041903 |

Rolling-origin selected-model results:

| Feature Set | Mean Log Loss | Std Log Loss | Mean Brier | Mean Accuracy | Mean ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| No-Elo | 1.201547 | 0.011609 | 0.546613 | 0.574590 | 0.035931 |
| Simple Elo, K=20/home=0 | 1.197724 | 0.013563 | 0.521042 | 0.594243 | 0.039671 |
| Selected Elo, K=10/home=50 | 1.186855 | 0.012516 | 0.522831 | 0.594480 | 0.041903 |

## Calibration Notes

The selected baseline feature set improves rolling-origin mean log loss over the no-Elo feature set and over the simple Elo setup. The K=10/home=50 variant beats simple Elo on log loss in 6 of 6 rolling-origin windows.

However, it does not consistently improve expected calibration error. The selected K=10/home=50 variant worsens mean ECE relative to simple Elo, so probability-bin calibration remains a visible caveat.

The model should therefore be described as the current probability-quality baseline, not as a fully calibrated or final model.

## Known Limitations

Known limitations:

- Only rolling team-form and simple Elo features are included.
- Elo has a fixed home advantage but no margin-of-victory, tournament weighting, host-country effect, or uncertainty adjustment yet.
- No player, squad, market, or tactical inputs.
- No tournament-specific backtesting yet.
- Calibration is limited to sigmoid calibration.
- Missing rolling-history values are handled by median imputation plus missingness indicators.
- Data source is manually downloaded and locally maintained.

## Ethical/Product Caveats

International football forecasts are uncertain and can be affected by injuries, squad rotation, tactical changes, travel, weather, motivation, and tournament context.

The dashboard should present probabilities with uncertainty-aware language. It should avoid implying certainty, betting suitability, or causal explanations beyond what the model actually measures.

## Next Planned Improvements

Next planned improvements:

- Evaluate margin-of-victory and tournament-weighted Elo variants.
- Test whether the K=10/home=50 variant remains selected under tournament-specific backtests.
- Add tournament-specific validation over prior World Cups and major tournaments.
- Revisit calibration methods after stronger feature sets are available.
