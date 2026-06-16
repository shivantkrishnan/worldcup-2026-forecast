# Model Selection Report

## Objective

Select the current probability-quality baseline for the World Cup 2026 Forecasting Dashboard using the completed single-holdout and rolling-origin validation results.

The selected model should prioritize probabilistic forecast quality, especially log loss, while preserving the project's leakage-safe validation rules.

## Candidate Models Compared

Three baseline candidates were compared:

- Class-prior baseline: predicts the training-set outcome distribution for every test match.
- Logistic regression: median imputation, missingness indicators, scaling, and multinomial logistic regression.
- Sigmoid-calibrated logistic regression: the same logistic pipeline wrapped in train-only internal cross-validation calibration.

All models use the leakage-safe team-form feature table and time-aware validation.

## Single-Holdout Results

The single holdout trains before `2022-01-01` and tests on baseline-eligible matches from `2022-01-01` through `2026-06-10`.

| Model | Log Loss | Brier | Accuracy | ECE |
| --- | ---: | ---: | ---: | ---: |
| Class-prior | 1.242354 | 0.633661 | 0.477775 | n/a |
| Logistic regression | 1.250315 | 0.544684 | 0.580688 | 0.026670 |
| Sigmoid-calibrated logistic regression | 1.203647 | 0.548502 | 0.579812 | 0.032929 |

On this holdout, calibrated logistic regression has the best log loss. Uncalibrated logistic regression has slightly better Brier score, accuracy, and ECE.

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
sigmoid-calibrated logistic regression
```

This selection is provisional and tied to the current team-form feature set, validation design, and manually maintained historical data source.

## Why Calibrated Logistic Is Selected

The project prioritizes probability quality. Log loss directly rewards assigning higher probability to the true outcome and strongly penalizes confident wrong forecasts.

Sigmoid-calibrated logistic regression is selected because:

- It has the best single-holdout log loss.
- It has the best rolling-origin mean log loss.
- It beats uncalibrated logistic regression on log loss in all 6 rolling-origin windows.
- It keeps the model simple, interpretable, and compatible with the current feature set.
- Calibration is fitted only inside the training split, preserving the test set as a true holdout.

## Why ECE Caveat Matters

The calibrated model is not uniformly better on calibration diagnostics. It improves log loss but does not consistently improve ECE.

This matters because ECE measures confidence-bin alignment between predicted confidence and empirical accuracy. The calibrated model's better log loss suggests better probability assignment to true classes overall, but the ECE result warns against claiming that all displayed confidence levels are better calibrated.

The dashboard should describe calibrated logistic regression as the current probability-quality baseline, not as a fully solved calibration model.

## Why Class-Prior and Uncalibrated Logistic Remain Useful Benchmarks

The class-prior baseline remains useful because it sets the minimum standard for probabilistic forecasts. A model that cannot beat class priors on log loss is not yet a strong probability model.

Uncalibrated logistic regression remains useful because it shows how much signal comes from the team-form features and linear model before post-hoc calibration. It also performs slightly better on Brier score, accuracy, and ECE in the current single holdout.

Both benchmarks should remain in evaluation reports until stronger models clearly dominate them across multiple metrics.

## Limitations

Current limitations:

- Features are limited to leakage-safe team-form history.
- No Elo, opponent-strength, squad, player, market, or tournament-stage features are included yet.
- The selected baseline is chosen by log loss, not by uniform superiority across all metrics.
- ECE does not consistently improve with sigmoid calibration.
- Rolling-origin windows are broad 24-month windows, not tournament-specific backtests.
- Raw data is manually downloaded and locally maintained.
- No model artifact is written by default.

## Next Modeling Steps

The next recommended feature family is Elo-style team strength. Elo-style ratings can add opponent-adjusted information that rolling form alone does not capture.

After adding Elo features, the same model selection process should be rerun:

- Feature readiness audit.
- Single holdout.
- Rolling-origin backtest.
- Log loss, Brier score, accuracy, and calibration diagnostics.
- Updated model selection report and model card.
