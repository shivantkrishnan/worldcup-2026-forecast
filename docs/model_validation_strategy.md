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

Later validation should include rolling-origin backtests. Each split should train on all data available up to a date and test on the next future window.

This helps evaluate stability across time instead of relying on one holdout period.

## Tournament Backtesting

Future backtests should evaluate prior World Cups and major international tournaments. This is important because tournament matches differ from friendlies and qualifiers in incentives, squad selection, neutral venues, and pressure.

## Metrics

Primary metrics:

- Log loss.
- Multiclass Brier score.
- Calibration diagnostics.

Accuracy is secondary because the project forecasts probabilities.

The first baseline reports both a class-prior model and multinomial logistic regression on the same time-aware holdout, so improvement over a naive probabilistic baseline is visible.

## Economic and Statistical Intuition

The validation design should resemble a real forecasting decision. At prediction time, the model should only know information available before kickoff. This is both a statistical leakage rule and a practical economic rule: forecasts are useful only if they could have been made before the outcome was known.
