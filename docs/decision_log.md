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

## 2026-06-16

### Decision

Select rolling team-form plus pre-match Elo features as the current baseline feature set.

### Rationale

Elo was evaluated against the rolling-form-only feature set using the same sigmoid-calibrated logistic regression model family. Elo improved single-holdout log loss from `1.203647` to `1.202032` and rolling-origin mean log loss from `1.201547` to `1.197716`. It improved rolling-origin log loss in 5 of 6 windows and Brier score in 6 of 6 windows.

### Alternatives Considered

- Keep the rolling-form-only selected baseline until Elo calibration improves.
- Treat Elo as experimental despite better log loss because mean ECE worsened.
- Tune Elo hyperparameters before selecting the feature family.

### Implications for Modeling/Product

The selected baseline changes to sigmoid-calibrated logistic regression over rolling team-form plus pre-match Elo features. The selection remains provisional because Elo improves ECE in only 2 of 6 rolling-origin windows and worsens mean ECE. Next work should refine Elo, especially home advantage, neutral-site handling, K-factor choice, margin-of-victory adjustment, and calibration.

## 2026-06-17

### Decision

Maintain a concise project roadmap and extension backlog in `docs/roadmap.md`.

### Rationale

The project now has enough completed modeling, validation, documentation, and feature-engineering milestones that future work should be separated into core modeling, tournament forecasting, product/UI, supplemental extensions, and out-of-scope items.

### Implications for Modeling/Product

The roadmap keeps the near-term baseline-improvement path focused on Elo refinement while preserving later ideas such as player features, market benchmarks, live in-match modeling, and tournament-specific backtesting as supplemental extensions.

## 2026-06-18

### Decision

Evaluate simple Elo variants by K-factor and fixed home advantage.

### Rationale

The selected Elo baseline improves log loss and Brier score, but it uses a single K-factor and no home/neutral-site adjustment. Historical non-neutral international matches may contain home advantage, while neutral-site matches should not receive that bonus.

### Alternatives Considered

- Move directly to margin-of-victory or tournament-weighted Elo.
- Tune only K-factor without home advantage.
- Keep the simple K=20, no-home Elo setup until model families change.

### Implications for Modeling/Product

The variant comparison keeps underlying ratings as team-strength ratings and applies home advantage only to match-level expected score for non-neutral matches. Variant selection remains based on rolling-origin mean log loss first, with Brier score, accuracy, and ECE as supporting diagnostics.

## 2026-06-18

### Decision

Select the K=10, home-advantage=50 Elo variant as the current baseline Elo setup.

### Rationale

The compact grid compared K-factor values `10`, `20`, `30` and home advantages `0`, `50`, `75`, `100`. K=10 with a 50-point non-neutral home adjustment had the best rolling-origin mean log loss at `1.186855`, improving over the simple K=20/home=0 setup at `1.197724`. It beat simple Elo on log loss in all 6 rolling windows.

### Alternatives Considered

- Keep K=20 and no home advantage because mean ECE is better.
- Select K=20/home=50 because it has better Brier score and accuracy.
- Delay selection until margin-of-victory or tournament-weighted Elo is available.

### Implications for Modeling/Product

The selected baseline remains sigmoid-calibrated logistic regression over rolling team-form plus pre-match Elo features, now using K=10 and home advantage=50. The selection is still provisional because mean ECE worsened from `0.039671` to `0.041903`. Calibration caveats must remain visible.

## 2026-06-18

### Decision

Document K=10 as an empirically selected international-football Elo smoothing parameter.

### Rationale

K controls rating update speed, and international football differs from chess or club leagues because fixtures are sparse, irregular, and heterogeneous across friendlies, qualifiers, and tournaments. The project should not borrow chess/FIDE K conventions mechanically. K=10 was selected from a compact out-of-sample rolling-origin grid because it produced the best mean log loss when paired with a 50-point non-neutral home adjustment.

### Implications for Modeling/Product

The selected K remains provisional. The dashboard methodology should describe K=10 as a validated forecasting choice for this dataset, not as a universal rating rule. Tournament-specific backtests, margin-of-victory Elo, tournament weighting, and host-country effects may change the preferred K later.

## 2026-06-18

### Decision

Document historical home advantage as a temporary Elo expected-score adjustment, not a generic 2026 World Cup home bonus.

### Rationale

Historical non-neutral international matches include crowd, travel, familiarity, venue, and local-context effects. If ignored, Elo can over-credit the home team's underlying strength for results that partly reflect match location. The adjustment applies only inside the match expected-score and update calculation, leaving underlying ratings as team-strength state variables.

### Implications for Modeling/Product

Most 2026 World Cup fixtures should be treated as neutral by default. USA, Canada, and Mexico host effects should be modeled later as explicit tournament-state or venue features rather than as generic historical home advantage.

## 2026-06-18

### Decision

Add tournament-specific baseline validation over prior FIFA World Cups.

### Rationale

Broad rolling-origin windows validate general international-football performance, but the dashboard's core forecasting environment is the World Cup. Holding out prior World Cups tests whether the selected feature/model setup remains credible on tournament matches.

### Implications for Modeling/Product

For each held-out tournament, training rows must occur strictly before the tournament start date and test rows must come only from that World Cup. The validation compares no-Elo, simple Elo, and selected K=10/home=50 Elo calibrated logistic setups. The 2026 World Cup remains excluded from validation by default.

The first run over the 2002, 2006, 2010, 2014, 2018, and 2022 FIFA World Cups supports the selected K=10/home=50 setup by mean log loss (`1.137800`) versus no-Elo (`1.168182`) and simple Elo (`1.172695`). It beats simple Elo on log loss in 6 of 6 World Cup holdouts and no-Elo in 5 of 6. Mean ECE remains worse than no-Elo, so calibration caveats continue to apply.

## 2026-06-18

### Decision

Add the first scheduled-fixture forecast output layer.

### Rationale

Tournament forecasting and the Streamlit predictor need fixture-level probabilities before full Monte Carlo simulation. The output layer should train the selected baseline in memory, build fixture features without requiring future scores, and report the full 3-class probability vector.

### Implications for Modeling/Product

Fixture features use only completed matches strictly before the fixture date, with same-date completed matches excluded under date-only timestamp logic. Future 2026 World Cup fixtures default to neutral if no neutral flag is supplied, so the historical Elo home adjustment is not applied as a generic World Cup host effect. Favorite and confidence labels are display aids layered on top of probabilities.

## 2026-06-18

### Decision

Add the first group-stage Monte Carlo simulation layer.

### Rationale

Fixture-level probabilities are useful on their own, but tournament forecasting requires propagating match uncertainty through group tables. A lightweight simulator can estimate group winner, top-two, and advancement probabilities before the project implements full knockout brackets.

### Implications for Modeling/Product

The simulator consumes full 3-class fixture probabilities, samples W/D/L outcomes, and runs in memory without writing simulation files. Because the current model does not predict scores, tie-breakers are simplified to points, wins, and a seeded random placeholder. Official goal-difference and FIFA tie-break rules remain future work.

## 2026-06-18

### Decision

Use a manually maintained `data/tournament/fixtures_2026.csv` file as the first real 2026 fixture source.

### Rationale

The project needs real scheduled fixtures for prediction and simulation, but scraping, APIs, and paid providers are intentionally out of scope. A validated local CSV keeps tournament-state ingestion transparent and reviewable.

### Implications for Modeling/Product

Fixture data remains separate from historical model training data. Predictions are generated from the selected baseline without retraining on 2026 World Cup matches. The generator prints predictions by default and writes `fixture_predictions_2026.csv` only when `--output` is explicitly provided.
