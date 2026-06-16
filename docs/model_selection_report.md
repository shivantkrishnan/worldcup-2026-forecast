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
| With Elo | 63 |

Single-holdout selected-model results:

| Feature Set | Log Loss | Brier | Accuracy | ECE |
| --- | ---: | ---: | ---: | ---: |
| No-Elo | 1.203647 | 0.548502 | 0.579812 | 0.032929 |
| With Elo | 1.202032 | 0.521968 | 0.596891 | 0.032373 |

Rolling-origin selected-model aggregate results:

| Feature Set | Mean Log Loss | Std Log Loss | Mean Brier | Mean Accuracy | Mean ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| No-Elo | 1.201547 | 0.011609 | 0.546613 | 0.574590 | 0.035931 |
| With Elo | 1.197716 | 0.013559 | 0.521043 | 0.594158 | 0.039863 |

Rolling-origin Elo improvement counts:

- Log loss: 5 of 6 windows.
- Brier score: 6 of 6 windows.
- ECE: 2 of 6 windows.

Elo improves the selected model's rolling-origin mean log loss and Brier score, but it worsens mean ECE.

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
sigmoid-calibrated logistic regression with rolling team-form and pre-match Elo features
```

This selection is provisional and tied to the current feature set, validation design, and manually maintained historical data source.

## Why Calibrated Logistic Is Selected

The project prioritizes probability quality. Log loss directly rewards assigning higher probability to the true outcome and strongly penalizes confident wrong forecasts.

Sigmoid-calibrated logistic regression is selected because:

- It has the best single-holdout log loss.
- It has the best rolling-origin mean log loss.
- It beats uncalibrated logistic regression on log loss in all 6 rolling-origin windows.
- It keeps the model simple, interpretable, and compatible with the current feature set.
- Calibration is fitted only inside the training split, preserving the test set as a true holdout.
- The Elo-augmented feature set improves selected-model rolling-origin mean log loss and wins 5 of 6 rolling windows on log loss.

## Why ECE Caveat Matters

The calibrated model and Elo-augmented feature set are not uniformly better on calibration diagnostics. They improve log loss, but ECE does not consistently improve.

This matters because ECE measures confidence-bin alignment between predicted confidence and empirical accuracy. The calibrated Elo model's better log loss suggests better probability assignment to true classes overall, but the ECE result warns against claiming that all displayed confidence levels are better calibrated.

The dashboard should describe calibrated logistic regression as the current probability-quality baseline, not as a fully solved calibration model.

## Why Class-Prior and Uncalibrated Logistic Remain Useful Benchmarks

The class-prior baseline remains useful because it sets the minimum standard for probabilistic forecasts. A model that cannot beat class priors on log loss is not yet a strong probability model.

Uncalibrated logistic regression remains useful because it shows how much signal comes from the team-form features and linear model before post-hoc calibration. It also performs slightly better on Brier score, accuracy, and ECE in the current single holdout.

Both benchmarks should remain in evaluation reports until stronger models clearly dominate them across multiple metrics.

## Limitations

Current limitations:

- Features are limited to leakage-safe rolling team-form and simple Elo-style team strength.
- No squad, player, market, or tournament-stage features are included yet.
- The selected baseline is chosen by log loss, not by uniform superiority across all metrics.
- ECE does not consistently improve with sigmoid calibration or Elo features.
- Rolling-origin windows are broad 24-month windows, not tournament-specific backtests.
- Raw data is manually downloaded and locally maintained.
- No model artifact is written by default.

## Next Modeling Steps

The next recommended modeling step is to improve and stress-test the Elo feature family.

Candidate next steps:

- Add home advantage and neutral-site Elo handling.
- Compare K-factor values.
- Test margin-of-victory or tournament-weighted Elo variants.
- Add tournament-specific backtests over prior World Cups and major tournaments.
- Revisit calibration after stronger rating features are available.
