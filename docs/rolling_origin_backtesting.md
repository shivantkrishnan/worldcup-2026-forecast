# Rolling-Origin Backtesting

Rolling-origin backtesting is the next validation layer after the first single holdout. The 2022-2026 holdout is useful, but one window can overstate or understate a model's quality because international football changes across cycles.

## Why It Is Needed

A single holdout answers one question: how did the model perform on one future period?

Rolling-origin backtesting asks a stronger question: if we had repeatedly trained the model at different points in history and forecast the next future window, would the result be stable?

This matters for the dashboard because a useful World Cup forecasting model should not depend on one lucky validation window.

## Forecast Timing

Each split mimics a real forecasting decision:

- Train on all matches available up to a historical cutoff date.
- Test only on matches after that cutoff.
- Move the cutoff forward and repeat.

The default backtest uses expanding training windows with 24-month test windows. For example, a model trained through `2014-12-31` forecasts `2015-01-01` through `2016-12-31`, then the next split trains through `2016-12-31` and forecasts the next future window.

## Refit Everything Inside Each Split

Each rolling split must refit the full modeling pipeline independently:

- Feature preprocessing.
- Median imputation.
- Missingness indicators.
- Scaling.
- Logistic regression.
- Sigmoid calibration.

This preserves the same leakage rule as live forecasting: no preprocessing, model fitting, or calibration step may use rows from the future test window.

## Why It Improves Stability Assessment

Rolling-origin results show whether the baseline is consistently useful across eras. The model may do well in one period because of tournament composition, match mix, or unusual class balance. Multiple future windows make that fragility visible.

## How To Interpret Results

Mean metrics summarize average performance across evaluated windows.

Standard deviation shows stability. A low mean log loss with high standard deviation may be less dependable than a slightly weaker model that performs consistently.

Per-window wins and losses show whether a model improvement is broad or concentrated. For example, if calibrated logistic regression beats uncalibrated logistic regression on log loss in most windows, the calibration benefit is more convincing. If it wins only one window, the single-holdout improvement should be treated cautiously.

ECE comparisons should be read as diagnostics, not the only calibration verdict. Expected calibration error depends on binning and sample size, so it should be interpreted alongside log loss, Brier score, and classwise calibration behavior.
