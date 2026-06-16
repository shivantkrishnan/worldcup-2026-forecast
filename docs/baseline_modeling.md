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

## Class-Prior Baseline

The class-prior baseline predicts the training-set class distribution for every test match. It is intentionally simple and provides a sanity check.

The logistic model should beat this baseline on log loss and Brier score before being considered useful.

## Limitations

Current limitations:

- Features are limited to team-form history.
- No Elo, opponent-strength, player, market, or tournament-stage features yet.
- Missing values are handled with a simple median-imputation strategy.
- No hyperparameter tuning.
- No calibration post-processing.
- No artifact writing by default.

## Next Candidates

Possible next model or feature candidates:

- Regularized logistic regression variants.
- Random forest.
- Gradient boosting.
- Elo or rating features.
- Player-form features.
- Market or odds-implied features.
- Calibration methods such as isotonic or Platt-style calibration.
