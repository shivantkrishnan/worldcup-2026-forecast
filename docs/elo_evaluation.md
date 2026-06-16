# Elo Feature Set Evaluation

This evaluation compares whether adding leakage-safe Elo-style team strength features improves the current baseline feature set.

## What Is Being Compared

Two feature sets are compared with the same model family and validation design:

- No-Elo feature set: rolling team-form features only.
- Elo feature set: rolling team-form features plus pre-match Elo features.

The selected model family for comparison is sigmoid-calibrated logistic regression because it is the current probability-quality baseline by rolling-origin mean log loss.

## Why Reuse The Same Model And Validation Design

Feature-family evaluation should isolate the feature change. The model family, preprocessing, calibration approach, training cutoff, single-holdout split, and rolling-origin split design should stay the same.

This makes the comparison answer a clean question: do Elo features improve out-of-sample probability quality beyond rolling form?

## Metrics

Improvement should be judged out of sample with:

- Log loss.
- Multiclass Brier score.
- Calibration diagnostics, including ECE.
- Rolling-origin stability across time windows.

Log loss remains the primary model-selection metric because the dashboard forecasts probabilities and should reward probability assigned to the true outcome.

## Interpreting Mixed Results

Elo may improve one metric while hurting another. For example:

- Better log loss but worse ECE suggests stronger probability assignment overall, but weaker confidence-bin alignment.
- Better Brier score but worse log loss may suggest less extreme probabilities without better likelihood on true outcomes.
- Strong single-holdout results but weak rolling-origin results suggest the gain may be period-specific.

Model selection remains provisional unless improvements are stable across rolling-origin windows and consistent with the dashboard's probability-quality goals.

## Current Status

The comparison script is:

```bash
python scripts/compare_elo_feature_set.py
```

It evaluates no-Elo and Elo feature sets without writing model artifacts or processed feature files.

## Evaluation Results

The comparison was run on the local baseline-eligible historical dataset.

Feature counts:

- No-Elo: 57 numeric features.
- With Elo: 63 numeric features.

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

## Selection Outcome

Elo improves the selected baseline by rolling-origin mean log loss, so the current selected baseline feature set changes to rolling team-form plus pre-match Elo features.

The selection remains provisional because ECE worsens on average. Elo improves probability quality by log loss and Brier score, but it does not solve calibration-bin alignment.
