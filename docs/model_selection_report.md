# Model Selection Report

## Objective

Select the current probability-quality baseline for the World Cup 2026 Forecasting Dashboard using the completed single-holdout and rolling-origin validation results.

The selected model should prioritize probabilistic forecast quality, especially log loss, while preserving the project's leakage-safe validation rules.

## Candidate Models Compared

Three baseline candidates were compared:

- Class-prior baseline: predicts the training-set outcome distribution for every test match.
- Logistic regression: median imputation, missingness indicators, scaling, and multinomial logistic regression.
- Sigmoid-calibrated logistic regression: the same logistic pipeline wrapped in train-only internal cross-validation calibration.

The first comparison used the leakage-safe rolling team-form feature table. Elo-style pre-match team strength was then evaluated as an added feature family using the same model family and validation design.

## Single-Holdout Results

The single holdout trains before `2022-01-01` and tests on baseline-eligible matches from `2022-01-01` through `2026-06-10`.

| Model | Log Loss | Brier | Accuracy | ECE |
| --- | ---: | ---: | ---: | ---: |
| Class-prior | 1.242354 | 0.633661 | 0.477775 | n/a |
| Logistic regression | 1.250315 | 0.544684 | 0.580688 | 0.026670 |
| Sigmoid-calibrated logistic regression | 1.203647 | 0.548502 | 0.579812 | 0.032929 |

On this holdout, calibrated logistic regression has the best log loss. Uncalibrated logistic regression has slightly better Brier score, accuracy, and ECE.

## Elo Feature Comparison

The Elo comparison evaluates the selected model family, sigmoid-calibrated logistic regression, with two feature sets:

- No-Elo: rolling team-form features only.
- With Elo: rolling team-form features plus pre-match Elo team strength features.

Feature counts:

| Feature Set | Numeric Feature Count |
| --- | ---: |
| No-Elo | 57 |
| With simple Elo | 65 |

Single-holdout selected-model results:

| Feature Set | Log Loss | Brier | Accuracy | ECE |
| --- | ---: | ---: | ---: | ---: |
| No-Elo | 1.203647 | 0.548502 | 0.579812 | 0.032929 |
| With simple Elo, K=20, home=0 | 1.202069 | 0.521042 | 0.594243 | 0.039671 |

Rolling-origin selected-model aggregate results:

| Feature Set | Mean Log Loss | Std Log Loss | Mean Brier | Mean Accuracy | Mean ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| No-Elo | 1.201547 | 0.011609 | 0.546613 | 0.574590 | 0.035931 |
| With simple Elo, K=20, home=0 | 1.197724 | 0.013563 | 0.521042 | 0.594243 | 0.039671 |

Rolling-origin Elo improvement counts:

- Log loss: 5 of 6 windows.
- Brier score: 6 of 6 windows.
- ECE: 2 of 6 windows.

Elo improves the selected model's rolling-origin mean log loss and Brier score, but it worsens mean ECE.

## Elo Variant Comparison

The next Elo refinement compared:

- K-factor: `10`, `20`, `30`.
- Home advantage: `0`, `50`, `75`, `100`.

Home advantage is applied only to non-neutral matches and only for expected-score calculation. It does not permanently inflate a team's underlying rating.

Best variant by rolling-origin mean log loss:

| K-Factor | Home Advantage | Mean Log Loss | Std Log Loss | Mean Brier | Mean Accuracy | Mean ECE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 50 | 1.186855 | 0.012516 | 0.522831 | 0.594480 | 0.041903 |

Compared with the current simple Elo setup, K=20 and home advantage=0:

- Mean rolling log loss improves from `1.197724` to `1.186855`.
- The best variant beats simple Elo on log loss in 6 of 6 rolling windows.
- Mean ECE worsens from `0.039671` to `0.041903`.

K=10 is methodologically defensible because it is selected by out-of-sample rolling-origin log loss for international football, not imported from chess-rating convention. International matches are sparse, irregular, and noisy; a lower K lets Elo act as a slower-moving opponent-adjusted strength signal while rolling team-form features capture short-run momentum.

The 50-point home adjustment is included primarily to learn from historical non-neutral matches where `team_a` is the home team. It affects expected score and rating updates for that match only, so it does not permanently inflate a team's rating. This should not be read as a generic home bonus for 2026 World Cup fixtures; most tournament matches should remain neutral unless later host/venue features explicitly model USA, Canada, or Mexico effects.

## Tournament-Specific Backtesting

The next validation layer evaluates the selected model family on held-out FIFA World Cups. For each target tournament year, training rows must occur strictly before that tournament starts, and test rows must come only from that World Cup.

This differs from broad rolling-origin validation because it focuses on the World Cup match environment. The first tournament-specific script compares:

- No-Elo calibrated logistic regression.
- Simple Elo calibrated logistic regression, K=20/home=0.
- Selected Elo calibrated logistic regression, K=10/home=50.

Results are documented in `docs/tournament_backtesting.md`. Model selection should be revisited if the selected K/home variant materially underperforms on World Cup-specific log loss.

The first tournament-specific run supports the selected setup by log loss:

- Selected K=10/home=50 mean tournament log loss: `1.137800`.
- Simple K=20/home=0 mean tournament log loss: `1.172695`.
- No-Elo mean tournament log loss: `1.168182`.
- Selected K=10/home=50 beats simple Elo in 6 of 6 World Cup holdouts.
- Selected K=10/home=50 beats no-Elo in 5 of 6 World Cup holdouts.

The ECE caveat remains: selected Elo mean tournament ECE is `0.117387`, worse than no-Elo at `0.067903`.

## Rolling-Origin Backtest Results

The rolling-origin backtest uses expanding training windows, 24-month test windows, and a final test end date of `2026-06-10`.

| Model | Mean Log Loss | Std Log Loss |
| --- | ---: | ---: |
| Class-prior | 1.242557 | 0.009398 |
| Logistic regression | 1.253098 | 0.015040 |
| Sigmoid-calibrated logistic regression | 1.201547 | 0.011609 |

Per-window comparisons:

- Logistic regression beat class-prior on log loss: 0 of 6 splits.
- Sigmoid-calibrated logistic regression beat uncalibrated logistic regression on log loss: 6 of 6 splits.
- Sigmoid-calibrated logistic regression beat uncalibrated logistic regression on ECE: 1 of 6 splits.
- Best model by mean log loss: `calibrated_logistic_regression`.

## Selected Current Baseline

The selected current baseline is:

```text
sigmoid-calibrated logistic regression with rolling team-form and pre-match Elo features (K=10, home advantage=50)
```

This selection is provisional and tied to the current feature set, validation design, and manually maintained historical data source.

The first forecast output layer now trains this selected baseline in memory and applies it to scheduled fixtures. It returns full `team_a_win`/`draw`/`team_b_win` probabilities plus a favorite display label, while preserving the same leakage rule that fixture features may use only completed matches before the fixture date.

## Why Calibrated Logistic Is Selected

The project prioritizes probability quality. Log loss directly rewards assigning higher probability to the true outcome and strongly penalizes confident wrong forecasts.

Sigmoid-calibrated logistic regression is selected because:

- It has the best single-holdout log loss.
- It has the best rolling-origin mean log loss.
- It beats uncalibrated logistic regression on log loss in all 6 rolling-origin windows.
- It keeps the model simple, interpretable, and compatible with the current feature set.
- Calibration is fitted only inside the training split, preserving the test set as a true holdout.
- The selected Elo variant improves rolling-origin mean log loss and beats the simple Elo setup in all 6 rolling windows on log loss.

## Why ECE Caveat Matters

The calibrated model and Elo-augmented feature set are not uniformly better on calibration diagnostics. They improve log loss, but ECE does not consistently improve.

This matters because ECE measures confidence-bin alignment between predicted confidence and empirical accuracy. The calibrated Elo-variant model's better log loss suggests better probability assignment to true classes overall, but the ECE result warns against claiming that all displayed confidence levels are better calibrated.

The dashboard should describe calibrated logistic regression as the current probability-quality baseline, not as a fully solved calibration model.

## Why Class-Prior and Uncalibrated Logistic Remain Useful Benchmarks

The class-prior baseline remains useful because it sets the minimum standard for probabilistic forecasts. A model that cannot beat class priors on log loss is not yet a strong probability model.

Uncalibrated logistic regression remains useful because it shows how much signal comes from the team-form features and linear model before post-hoc calibration. It also performs slightly better on Brier score, accuracy, and ECE in the current single holdout.

Both benchmarks should remain in evaluation reports until stronger models clearly dominate them across multiple metrics.

## Limitations

Current limitations:

- Features are limited to leakage-safe rolling team-form and simple Elo-style team strength with a fixed home adjustment.
- No squad, player, market, or tournament-stage features are included yet.
- The selected baseline is chosen by log loss, not by uniform superiority across all metrics.
- ECE does not consistently improve with sigmoid calibration or Elo features.
- Rolling-origin windows are broad 24-month windows; tournament-specific backtests are now a separate, smaller-sample stress test.
- Raw data is manually downloaded and locally maintained.
- No model artifact is written by default.

## Next Modeling Steps

The next recommended modeling step is to improve and stress-test the Elo feature family.

Candidate next steps:

- Test margin-of-victory or tournament-weighted Elo variants.
- Test whether the K=10, home=50 selection holds under tournament-specific backtests.
- Add tournament-specific backtests over prior World Cups and major tournaments.
- Feed fixture-level probabilities into the first Monte Carlo tournament simulator.
- Revisit calibration after stronger rating features are available.
