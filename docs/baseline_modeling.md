# Baseline Modeling

The first baseline model is intentionally simple, interpretable, and reproducible. It is designed to establish a clean probabilistic benchmark before adding more complex model families or richer feature sets.

## Why Start Simple

A simple baseline makes it easier to verify the end-to-end pipeline:

- Historical data is loaded locally.
- Canonical matches are cleaned and deduplicated.
- Team-form features are leakage-safe.
- Train/test validation is time-aware.
- Probabilistic metrics are computed consistently.

Complex models are only useful after this foundation is trustworthy.

## Probabilistic Evaluation Over Accuracy

The dashboard forecasts probabilities for `team_a_win`, `draw`, and `team_b_win`. Accuracy is useful, but secondary. A model can have reasonable accuracy while being badly calibrated or overconfident.

Primary metrics:

- Log loss.
- Multiclass Brier score.
- Calibration diagnostics.

Accuracy is reported as a secondary sanity check.

## Time-Aware Train/Test Split

The default split trains before `2022-01-01` and tests on matches from `2022-01-01` through the baseline cutoff. This mimics real forecasting: only past matches can inform predictions for future matches.

Random splits are not the primary evaluation design because they can mix football eras and leak context across time.

## Rolling-Origin Backtesting

The baseline models can also be compared across expanding rolling-origin splits. Each split trains on historical rows up to a cutoff and tests on the next future window. The full preprocessing and modeling pipeline is refit inside each split.

This checks whether class-prior, logistic regression, and calibrated logistic regression performance is stable across time. The backtest report includes mean/std metrics and counts how often each model wins on log loss or ECE.

## Missing Rolling-History Values

Missing values are expected in rolling team-form features, especially for teams with limited prior match history and for full-window rolling statistics.

The first baseline handles missing feature values with:

```text
SimpleImputer(strategy="median", add_indicator=True)
```

The missingness indicators allow the model to learn whether lack of prior history itself carries signal.

## Train-Only Imputation

Imputation must be fitted on training rows only. Fitting an imputer on the full dataset would leak information from the test period into training preprocessing.

The baseline uses a scikit-learn `Pipeline`, so imputation, missingness indicators, scaling, and logistic regression are fit only on the training split.

## Logistic Regression Baseline

Multinomial logistic regression is a good first baseline because it is:

- Interpretable.
- Probabilistic.
- Fast enough for repeated iteration.
- Compatible with imputation and standardized numeric features.
- A useful benchmark before tree-based or rating-based models.

## Calibrated Logistic Regression Variant

The baseline comparison now includes a calibrated logistic regression variant. It uses the same train-only preprocessing pipeline and wraps the logistic model with `CalibratedClassifierCV`.

The default calibration method is sigmoid because it is sample-efficient and less likely to overfit than isotonic calibration on smaller folds. Calibration is fitted only within the training split through internal cross-validation. The test set is used only for final evaluation.

Calibration diagnostics include expected calibration error and confidence-bin summaries. This helps explain cases where logistic regression improves accuracy or Brier score but worsens log loss due to overconfident wrong predictions.

## Current Selected Baseline

The current selected baseline is sigmoid-calibrated logistic regression over rolling team-form plus pre-match Elo features.

Selection is based primarily on rolling-origin log loss stability. The Elo-augmented calibrated logistic model improves rolling-origin mean log loss over the rolling-form-only selected model.

This is a probability-quality selection, not a claim that every metric improves. Elo improves Brier score and accuracy in the current comparison, but worsens mean ECE, so calibration caveats remain important.

## Elo Feature Evaluation

The first Elo feature evaluation compares rolling team-form only against rolling team-form plus pre-match Elo features using the selected sigmoid-calibrated logistic model.

Rolling-origin selected-model results:

- No-Elo log loss: `1.201547 +/- 0.011609`.
- Elo log loss: `1.197716 +/- 0.013559`.
- Elo improves log loss in 5 of 6 rolling-origin windows.
- Elo improves Brier score in 6 of 6 rolling-origin windows.
- Elo improves ECE in 2 of 6 rolling-origin windows.

The current selected baseline feature set therefore includes Elo, but the calibration caveat remains.

## Class-Prior Baseline

The class-prior baseline predicts the training-set class distribution for every test match. It is intentionally simple and provides a sanity check.

The logistic model should beat this baseline on log loss and Brier score before being considered useful.

## Limitations

Current limitations:

- Features are limited to rolling team-form and simple Elo-style strength.
- No player, market, or tournament-stage features yet.
- Missing values are handled with a simple median-imputation strategy.
- No hyperparameter tuning.
- Calibration comparison is limited to the first sigmoid-calibrated logistic variant.
- Rolling-origin backtesting is implemented, but deeper tournament-specific backtests are still future work.
- Current model selection is provisional until richer Elo variants and other feature families are evaluated.
- No artifact writing by default.

## Next Candidates

Possible next model or feature candidates:

- Regularized logistic regression variants.
- Random forest.
- Gradient boosting.
- Home-advantage, margin-of-victory, or tournament-weighted Elo variants.
- Player-form features.
- Market or odds-implied features.
- Additional calibration methods such as isotonic calibration.
