# Calibration Diagnostics

The dashboard reports probabilities, not just predicted winners. Calibration diagnostics check whether those probabilities behave like real frequencies over comparable matches.

## Why Calibration Matters

A World Cup forecasting dashboard will show probabilities such as:

```text
team_a_win: 0.52
draw: 0.25
team_b_win: 0.23
```

Those numbers are only useful if they are trustworthy. If the model says 60 percent across many similar matches, roughly 60 percent of those outcomes should occur. Calibration is what makes probability forecasts credible for users who are comparing uncertainty, not just reading a label.

## Accuracy Can Improve While Log Loss Gets Worse

Accuracy only checks whether the top predicted class is correct. A model can improve accuracy by ranking the winning class first more often while still assigning probabilities that are too extreme.

Log loss is stricter. It rewards probability assigned to the true class and heavily penalizes confident wrong predictions. This means a model with better accuracy can still have worse log loss if its incorrect predictions are overconfident.

## Why Overconfidence Hurts

Consider two wrong predictions for the true class `draw`:

```text
Prediction A: draw probability = 0.30
Prediction B: draw probability = 0.02
```

Both are wrong if another class has the highest probability, but Prediction B is much more damaging under log loss because it nearly ruled out the true outcome. For a probability dashboard, avoiding that kind of overconfidence is central.

## Expected Calibration Error

Expected calibration error (ECE) summarizes confidence calibration. The implementation bins predictions by maximum predicted confidence, compares average confidence to empirical accuracy in each bin, and computes a row-weighted average gap.

ECE is not a replacement for log loss or Brier score. It is a diagnostic that helps answer whether predicted confidence is aligned with observed correctness.

## Classwise Calibration

Classwise calibration bins each class probability separately and compares:

- Average predicted probability for that class.
- Empirical frequency of that class.

This is useful for the three-outcome football setting because draws often have different calibration behavior than wins.

## Sigmoid vs Isotonic Calibration

Sigmoid calibration, often called Platt-style calibration, fits a smooth parametric correction to model scores. It is usually a good first choice because it is sample-efficient and less likely to overfit.

Isotonic calibration fits a more flexible monotonic mapping. It can capture more complex calibration shapes, but it needs more data and can overfit smaller calibration folds.

The first calibrated logistic variant defaults to sigmoid calibration. Isotonic can be compared later once the validation framework is broader.

## No Test-Set Calibration

Calibration must be fit only on training data. The current implementation uses scikit-learn's `CalibratedClassifierCV`, which performs internal cross-validation within the training split.

The test set remains untouched until final evaluation. Using the test set to calibrate probabilities would leak evaluation information and make the reported log loss, Brier score, and calibration diagnostics too optimistic.
