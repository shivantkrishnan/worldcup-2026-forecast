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
