# Model Validation Strategy

## Why Time-Aware Validation

Forecast validation should mimic real decision timing. When predicting a future match, only past information should be available. A time-aware split trains on earlier matches and tests on later matches, which better reflects the actual forecasting problem.

Football teams change over time as players, managers, tactics, and competitive context evolve. A random split can mix eras and make evaluation look stronger than it would be in a real forecast setting.

## Why Random Split Is Not Primary

Random train/test splits are not the primary validation method because they can leak historical context across time. For example, a random split can train on matches from a later version of a team and test on earlier matches from the same team era.

Random splits may still be useful for narrow debugging, but they should not be used as the headline evaluation design.

## Default Holdout

The default holdout idea is:

- Train on baseline-eligible matches before `2022-01-01`.
- Test on baseline-eligible matches from `2022-01-01` through the pre-World Cup cutoff, `2026-06-10`.

This gives a recent out-of-sample window while preserving the first-baseline training cutoff.

## Rolling-Origin Backtesting

Rolling-origin backtesting is the next validation layer after the first 2022-2026 holdout. Each split trains on all data available up to a historical cutoff date and tests on the next future window.

The default baseline backtest uses:

- Initial train end date: `2014-12-31`.
- Test window: 24 months.
- Step size: 24 months.
- Final test end date: `2026-06-10`.

This creates repeated expanding-window forecasts while preserving the first-baseline cutoff.

For every split, preprocessing, imputation, scaling, logistic regression, and calibration are refit using only that split's training rows. The future test window is reserved for evaluation only.

Rolling-origin backtests help evaluate stability across time instead of relying on one holdout period. Mean metrics summarize average performance, standard deviations show volatility, and per-window wins/losses reveal whether one model consistently beats another.

The current selected baseline is sigmoid-calibrated logistic regression because it has the best rolling-origin mean log loss and wins all rolling-origin windows against uncalibrated logistic regression on log loss. This selection remains provisional because ECE does not improve consistently.

## Tournament Backtesting

Future backtests should evaluate prior World Cups and major international tournaments. This is important because tournament matches differ from friendlies and qualifiers in incentives, squad selection, neutral venues, and pressure.

## Metrics

Primary metrics:

- Log loss.
- Multiclass Brier score.
- Calibration diagnostics.

Accuracy is secondary because the project forecasts probabilities.

The first baseline reports a class-prior model, uncalibrated multinomial logistic regression, and a sigmoid-calibrated logistic regression variant on the same time-aware holdout. Calibration diagnostics include expected calibration error, confidence-bin summaries, and classwise calibration tables.

Calibration must be fitted inside the training split only. The holdout test period is reserved for final metric reporting and must not be used to tune or calibrate probabilities.

## Economic and Statistical Intuition

The validation design should resemble a real forecasting decision. At prediction time, the model should only know information available before kickoff. This is both a statistical leakage rule and a practical economic rule: forecasts are useful only if they could have been made before the outcome was known.
