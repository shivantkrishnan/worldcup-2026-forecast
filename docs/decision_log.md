# Decision Log

This log records methodological, data, validation, and product decisions that affect modeling or interpretation.

## 2026-06-16

### Decision

Use a manually downloaded Kaggle-style international football results CSV as the v1 historical training source at `data/raw/results.csv`.

### Rationale

The project needs a stable local source before adding APIs or scraping. Manual download keeps the first pipeline reproducible and easy to inspect.

### Alternatives Considered

- Scraping public websites.
- Calling live football APIs.
- Automatically downloading Kaggle data.

### Implications for Modeling/Product

The first pipeline depends on a local file and must validate schema and quality before feature engineering.

## 2026-06-16

### Decision

Do not commit raw data.

### Rationale

Raw datasets may be large, change over time, or carry licensing restrictions.

### Alternatives Considered

- Commit raw CSVs directly.
- Commit processed snapshots immediately.

### Implications for Modeling/Product

The repository tracks code, docs, schemas, and instructions. Users must place raw files locally.

## 2026-06-16

### Decision

Use `2026-06-10` as the first-baseline training cutoff date.

### Rationale

The baseline should represent information available before the 2026 World Cup began.

### Alternatives Considered

- Train on all local rows.
- Retrain on completed World Cup matches.

### Implications for Modeling/Product

Completed World Cup matches may update standings and audits, but not first-baseline training.

## 2026-06-16

### Decision

Use a canonical `team_a`/`team_b` match schema.

### Rationale

The model needs a consistent outcome orientation. In the raw historical data, `team_a` maps to `home_team` and `team_b` maps to `away_team`.

### Alternatives Considered

- Keep `home_team`/`away_team` throughout.
- Use winner/loser orientation.

### Implications for Modeling/Product

Result labels are always from `team_a`'s perspective: `team_a_win`, `draw`, `team_b_win`.

## 2026-06-16

### Decision

Exclude scoreless fixture rows from canonical completed-match data.

### Rationale

Canonical completed matches require scores so result labels can be assigned.

### Alternatives Considered

- Keep scoreless rows with missing outcomes.
- Impute scores or labels.

### Implications for Modeling/Product

Scoreless fixtures belong in tournament-state files, not in completed-match training data.

## 2026-06-16

### Decision

Use explicit duplicate resolution and quarantine before feature engineering.

### Rationale

Modeling data must have unique `match_id` values. Metadata duplicates and conflicting duplicates require different handling.

### Alternatives Considered

- Silently drop duplicate rows.
- Keep all duplicate rows.
- Always keep the first duplicate row.

### Implications for Modeling/Product

Metadata duplicates keep one row and quarantine extras. Conflicting duplicate groups are fully quarantined by default.

## 2026-06-16

### Decision

Use time-aware validation over random validation by default.

### Rationale

Football team strength changes over time. Random splits can leak historical context across eras and overstate generalization.

### Alternatives Considered

- Random train/test split.
- Tournament-only holdouts as the only validation design.

### Implications for Modeling/Product

Default holdout trains before 2022 and tests on 2022 to pre-World Cup 2026 matches. Rolling-origin backtests can be added for robustness.

## 2026-06-16

### Decision

Treat player-level data as a modular extension after the team-level baseline.

### Rationale

Player data introduces identity matching, availability, role, and small-sample risks. A team-level baseline is needed first for comparison.

### Alternatives Considered

- Build player features immediately.
- Make the first model player-driven.

### Implications for Modeling/Product

Player features must prove incremental value on log loss, Brier score, and calibration before becoming core model inputs.

## 2026-06-16

### Decision

Compute team-form features from a long team-match panel using shift/lag logic before rolling or expanding calculations.

### Rationale

Canonical match rows contain both teams, but rolling form is naturally team-specific. A long panel gives each team its own historical sequence. Shifting outcomes before rolling ensures the current match result cannot leak into features for that match.

### Alternatives Considered

- Compute features directly from one-row-per-match data.
- Use current-row outcomes in rolling calculations and rely on later filtering.
- Start with complex rating or player-based features.

### Implications for Modeling/Product

The first feature set is interpretable and leakage-safe. Match-level features compare `team_a` and `team_b` by subtracting team-level feature values from `team_a`'s perspective.

## 2026-06-16

### Decision

Audit match-level feature readiness before baseline model training.

### Rationale

The first rolling feature layer intentionally produces missing values for teams with insufficient prior history. Before training, the project needs an explicit report of numeric candidate features, missingness, target distributions, train/test split sizes, and excluded non-numeric columns.

### Alternatives Considered

- Move directly from feature generation to model training.
- Let the model pipeline handle missingness without a pre-training audit.
- Drop early-history rows automatically.

### Implications for Modeling/Product

Missing rolling-history values are expected but must be handled explicitly later. The readiness audit follows the same time-aware split philosophy as model validation and helps decide whether baseline training inputs are acceptable.

## 2026-06-16

### Decision

Implement the first baseline model as a train-only preprocessing pipeline plus multinomial logistic regression, compared against a class-prior baseline.

### Rationale

The first model should be simple, interpretable, and probabilistic. Logistic regression gives a clean benchmark over leakage-safe team-form features. A class-prior baseline verifies that the model improves over always predicting the training-set outcome distribution.

### Alternatives Considered

- Train a tree-based model first.
- Tune hyperparameters before establishing the simplest benchmark.
- Drop rows with missing rolling-history features.
- Impute missing values before splitting.

### Implications for Modeling/Product

Missing rolling-history values are handled inside the model pipeline using median imputation plus missingness indicators fit only on training rows. Evaluation uses the time-aware holdout and prioritizes log loss, multiclass Brier score, and calibration diagnostics.

## 2026-06-16

### Decision

Add calibration diagnostics and compare a sigmoid-calibrated logistic regression variant against the uncalibrated baseline.

### Rationale

The first logistic regression baseline improved Brier score and accuracy but was slightly worse than the class-prior baseline on log loss. That pattern suggests useful separation signal with possible overconfidence or poor calibration.

### Alternatives Considered

- Keep only the uncalibrated logistic baseline.
- Tune model hyperparameters before diagnosing calibration.
- Use isotonic calibration as the first calibration method.

### Implications for Modeling/Product

Baseline reporting now includes expected calibration error, confidence-bin calibration summaries, classwise calibration tables, and calibrated logistic metrics. Calibration is fitted only on training data through internal cross-validation, so the test set remains a genuine holdout.

## 2026-06-16

### Decision

Add rolling-origin backtesting as the next validation layer after the first 2022-2026 holdout.

### Rationale

The calibrated logistic model improved log loss on the single holdout but did not improve ECE, Brier score, or accuracy. Multiple future test windows are needed to see whether that log-loss improvement is stable or concentrated in one period.

### Alternatives Considered

- Rely on the single 2022-2026 holdout.
- Move directly to stronger model families.
- Tune calibration or hyperparameters before checking temporal stability.

### Implications for Modeling/Product

Baseline comparisons now include expanding-window rolling-origin summaries. Each split refits preprocessing, imputation, scaling, logistic regression, and calibration using training rows only. Aggregate results report mean/std metrics and per-window model wins.

## 2026-06-16

### Decision

Select sigmoid-calibrated logistic regression as the current baseline model by rolling-origin mean log loss.

### Rationale

The calibrated logistic model has the best single-holdout log loss and the best rolling-origin mean log loss. It beats uncalibrated logistic regression on log loss in all 6 rolling-origin windows.

### Alternatives Considered

- Select the class-prior baseline because it beats uncalibrated logistic regression on log loss.
- Select uncalibrated logistic regression because it has slightly better single-holdout Brier score, accuracy, and ECE.
- Delay selection until after adding Elo features.

### Implications for Modeling/Product

The selection is provisional because ECE does not consistently improve: calibrated logistic regression beats uncalibrated logistic regression on ECE in only 1 of 6 rolling-origin splits. The next feature family should be Elo-style team strength, followed by the same single-holdout and rolling-origin validation process.

## 2026-06-16

### Decision

Add simple Elo-style team strength as the next opponent-adjusted feature family.

### Rationale

Rolling form summarizes recent team outcomes but does not directly adjust for opponent strength. Elo-style ratings add a compact pre-match state variable that updates more for surprising results than expected results.

### Alternatives Considered

- Add a richer rating system immediately.
- Add player or squad strength before opponent-adjusted team ratings.
- Use only rolling form until a stronger model family is introduced.

### Implications for Modeling/Product

The first Elo implementation uses initial rating `1500.0`, K-factor `20.0`, and the standard expected-score update. Ratings are emitted as pre-match features and update only after the relevant match/date block, so same-date results cannot leak into same-date feature rows. Elo features are ready for later baseline evaluation but are not yet used to retrain or reselect a model.
