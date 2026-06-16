# Baseline Model Card

## Model Name

`calibrated_logistic_regression`

Current selected baseline: sigmoid-calibrated multinomial logistic regression over leakage-safe team-form features.

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

The baseline uses leakage-safe team-form features built from a long team-match panel.

Feature groups include:

- Prior matches played.
- Days since last match.
- Rolling points, goals, goal difference, win/draw/loss rates.
- Expanding points, goal difference, and win rate.
- Match-level differential features comparing `team_a` to `team_b`.

Current-match outcomes are excluded from feature calculations through shift/lag logic before rolling or expanding calculations.

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

Single-holdout results:

| Model | Log Loss | Brier | Accuracy | ECE |
| --- | ---: | ---: | ---: | ---: |
| Class-prior | 1.242354 | 0.633661 | 0.477775 | n/a |
| Logistic regression | 1.250315 | 0.544684 | 0.580688 | 0.026670 |
| Sigmoid-calibrated logistic regression | 1.203647 | 0.548502 | 0.579812 | 0.032929 |

Rolling-origin log-loss results:

| Model | Mean Log Loss | Std Log Loss |
| --- | ---: | ---: |
| Class-prior | 1.242557 | 0.009398 |
| Logistic regression | 1.253098 | 0.015040 |
| Sigmoid-calibrated logistic regression | 1.201547 | 0.011609 |

## Calibration Notes

The selected baseline improves log loss consistently across rolling-origin windows, beating uncalibrated logistic regression on log loss in 6 of 6 splits.

However, it does not consistently improve expected calibration error. It beats uncalibrated logistic regression on ECE in only 1 of 6 rolling-origin splits.

The model should therefore be described as the current probability-quality baseline, not as a fully calibrated or final model.

## Known Limitations

Known limitations:

- Only team-form features are included.
- No Elo or opponent-adjusted strength features yet.
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

- Add Elo-style team strength features.
- Rerun feature readiness, single-holdout evaluation, and rolling-origin backtests.
- Compare whether Elo improves log loss, Brier score, and calibration diagnostics.
- Add tournament-specific validation over prior World Cups and major tournaments.
- Revisit calibration methods after stronger feature sets are available.
