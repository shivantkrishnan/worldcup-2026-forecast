# Tournament-Specific Backtesting

Tournament-specific backtesting checks whether the selected baseline performs reasonably on World Cup matches, not only on broad rolling-origin windows across all international fixtures.

## Why This Validation Is Needed

Broad rolling-origin validation is useful because it tests stability across time. It mixes friendlies, qualifiers, continental tournaments, and World Cups, which gives larger samples and a realistic general international-football benchmark.

World Cup forecasting is a narrower product goal. A model can perform well on all matches while still being weak on tournament matches, where opponent quality, neutral venues, tactical caution, squad selection, and match stakes differ from friendlies and qualifiers.

## How It Differs From Rolling-Origin Backtesting

Rolling-origin backtesting uses expanding training windows and future time blocks. Tournament-specific backtesting instead holds out a named tournament year:

1. Identify FIFA World Cup matches for the target year.
2. Find the first match date of that tournament.
3. Train the model only on matches before that first tournament date.
4. Test only on matches from that World Cup year.

The feature pipeline remains leakage-safe. Team-form and Elo features are emitted from information available before each match/date block. This makes the validation closer to live tournament forecasting than to a pure pre-tournament simulation, because later tournament matches may have features informed by earlier completed matches in the same tournament while the model itself is not retrained on those matches.

## Tournaments Tested

The first script targets FIFA World Cups:

- 2002
- 2006
- 2010
- 2014
- 2018
- 2022

The 2026 World Cup is excluded by default. Current 2026 matches are reserved for live forecasting, tournament state, and prediction audit, not for validating the first baseline selection.

## Model Setups Compared

The script compares the selected model family, sigmoid-calibrated logistic regression, across three feature setups:

| Setup | Features |
| --- | --- |
| `no_elo_calibrated_logistic` | Rolling team-form features only |
| `simple_elo_calibrated_logistic` | Rolling team-form plus simple Elo, K=20/home=0 |
| `selected_elo_calibrated_logistic` | Rolling team-form plus selected Elo, K=10/home=50 |

## Results Table

Run:

```bash
python scripts/backtest_tournament_baseline.py
```

The script prints per-tournament log loss, multiclass Brier score, accuracy, and expected calibration error, plus aggregate means across evaluated World Cup test sets. It does not write outputs by default.

| Model Setup | Mean Log Loss | Mean Brier | Mean Accuracy | Mean ECE |
| --- | ---: | ---: | ---: | ---: |
| `no_elo_calibrated_logistic` | 1.168182 | 0.597721 | 0.520833 | 0.067903 |
| `simple_elo_calibrated_logistic` | 1.172695 | 0.585348 | 0.539062 | 0.118023 |
| `selected_elo_calibrated_logistic` | 1.137800 | 0.575107 | 0.549479 | 0.117387 |

Per-tournament log-loss comparison:

| Tournament | No Elo | Simple Elo K=20/Home=0 | Selected Elo K=10/Home=50 |
| ---: | ---: | ---: | ---: |
| 2002 | 1.211846 | 1.220869 | 1.189700 |
| 2006 | 1.143146 | 1.123276 | 1.114433 |
| 2010 | 1.107853 | 1.124162 | 1.080576 |
| 2014 | 1.207363 | 1.202048 | 1.179357 |
| 2018 | 1.178852 | 1.161659 | 1.100984 |
| 2022 | 1.160035 | 1.204158 | 1.161749 |

## Whether K=10/Home=50 Remains Preferred

The selected K=10/home=50 Elo variant remains preferred by tournament-specific mean log loss. It beats the simple K=20/home=0 Elo setup in 6 of 6 World Cup holdouts and beats the no-Elo setup in 5 of 6 holdouts.

The caveat remains calibration. The selected setup has much worse tournament mean ECE than the no-Elo setup and is only slightly better than simple Elo on ECE. This supports keeping K=10/home=50 as the current probability-quality baseline by log loss, while continuing to warn that confidence-bin calibration is not solved.

## Limitations

- World Cups provide a small number of matches, so tournament metrics are noisy.
- Squads, tactics, and substitution rules change over time.
- The model has no player, lineup, injury, market, or tactical data yet.
- Neutral and host-country effects are not fully modeled yet.
- The current validation is closer to live tournament forecasting than pure pre-tournament simulation.
- Calibration-bin diagnostics are especially noisy on small tournament test sets.
