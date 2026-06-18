# Elo Variant Evaluation

This document defines how simple Elo variants are compared for the World Cup 2026 Forecasting Dashboard.

## What Is Being Compared

The compact variant grid changes two Elo parameters:

- K-factor: `10`, `20`, `30`.
- Home advantage: `0`, `50`, `75`, `100` rating points.

All variants keep the same model family, feature pipeline, training cutoff, single-holdout validation, and rolling-origin backtest design.

## Why K-Factor Matters

The K-factor controls how quickly ratings react to new results. A low K-factor produces more stable ratings but may adapt slowly when teams improve or decline. A high K-factor reacts faster but can overfit short-term noise, especially because international teams play irregular schedules.

The selected K-factor should be judged by out-of-sample performance, not by intuition alone.

## Why K=10 Is Defensible Here

K=10 is an empirically selected forecasting parameter for this international football dataset, not a borrowed chess convention.

International fixtures are sparse, irregular, and heterogeneous across friendlies, qualifiers, and tournaments. A lower K keeps Elo as a slower-moving opponent-adjusted strength signal while the existing rolling team-form features capture shorter-run momentum.

In the compact grid, K=10 with a 50-point non-neutral home adjustment had the best rolling-origin mean log loss and beat the simple K=20/home=0 setup in all 6 rolling windows. The choice remains provisional because ECE worsened and tournament-specific backtests have not yet been run.

## Why Home Advantage Matters

Historical non-neutral international matches often include travel, crowd, familiarity, and officiating/context effects that can favor the home side. Since `team_a` maps to the home team in the raw historical dataset for non-neutral matches, a home adjustment can improve the expected-score calculation.

If ignored, Elo can over-credit the home team's underlying strength for wins that partly reflect match location. The fixed home adjustment is therefore used to learn from historical non-neutral matches more cleanly, not to assert that location should dominate future tournament forecasts.

Home advantage should not apply to neutral-site matches. Many World Cup and tournament matches are marked neutral, and applying a home bonus there would add a false edge unless a separate host-country or crowd model is explicitly introduced later.

## Effective Rating, Not Permanent Rating Inflation

Home advantage affects the expected score for a match, not the team's underlying rating.

For a non-neutral match, `home_advantage` is added to `team_a`'s effective rating only while computing:

- `elo_expected_score_team_a`
- rating updates from the completed result
- `elo_effective_diff_team_a_minus_team_b`

The underlying pre-match ratings remain unchanged. This prevents a team from permanently carrying home advantage into future away or neutral matches.

## Neutral-Site Handling

When a match is marked `is_neutral=True` or `neutral=True`, the home advantage applied is `0`, even if the variant's home-advantage parameter is positive.

World Cup and tournament matches marked neutral should not receive a home bonus by default. Host-country effects can be modeled later as a separate feature once the tournament-state schema is mature.

The 2026 World Cup is especially important here because the tournament has three host nations and rotating venues. USA, Canada, and Mexico effects should not be approximated by blindly applying generic historical home advantage; they belong in an explicit host/venue feature family.

## Selection Metrics

Variant selection is based first on rolling-origin mean log loss for the selected sigmoid-calibrated logistic regression model.

Supporting diagnostics:

- Multiclass Brier score.
- Accuracy.
- Expected calibration error.
- Number of rolling windows beating the current simple Elo baseline.

ECE remains a supporting diagnostic rather than the sole selection criterion because calibration bins can be sensitive to sample size and binning. A variant that improves log loss but worsens ECE should be selected only provisionally and documented with calibration caveats.

## Provisional Selection

The selected Elo variant remains provisional. Future work may add:

- Margin-of-victory adjustment.
- Tournament weighting.
- Confederation adjustments.
- Host-country effects.
- Glicko-style rating uncertainty.

Each extension should rerun the feature audit, single-holdout evaluation, rolling-origin backtest, and model selection report.

## Evaluation Results

The compact grid was evaluated on the local baseline-eligible historical dataset:

- K-factor: `10`, `20`, `30`.
- Home advantage: `0`, `50`, `75`, `100`.

Best variant by rolling-origin mean log loss:

| K-Factor | Home Advantage | Mean Log Loss | Std Log Loss | Mean Brier | Mean Accuracy | Mean ECE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 50 | 1.186855 | 0.012516 | 0.522831 | 0.594480 | 0.041903 |

Current simple Elo baseline:

| K-Factor | Home Advantage | Mean Log Loss | Std Log Loss | Mean Brier | Mean Accuracy | Mean ECE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 0 | 1.197724 | 0.013563 | 0.521042 | 0.594243 | 0.039671 |

Selection outcome:

- K=10, home advantage=50 improves mean rolling-origin log loss.
- It beats simple Elo on log loss in 6 of 6 rolling windows.
- Mean ECE worsens, so calibration caveats remain.

The selected baseline changes provisionally to the K=10, home-advantage=50 Elo variant.
