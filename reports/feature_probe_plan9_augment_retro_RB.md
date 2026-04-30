# Feature signal probe — plan9_augment_retro_RB

Baseline features: data\features
Overrides:        data\features_probe\plan9_swap_retro_rb.parquet
Drops:            (none)
Model class:      baseline

## Phase 1 — per-stat screening

### RB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| rushing_yards | 2021 | 969 | +0.0535 | -0.0261 | +0.1378 | +0.0011 | NULL |
| rushing_yards | 2022 | 913 | -0.0358 | -0.1000 | +0.0295 | +0.0007 | NULL |
| rushing_yards | 2023 | 900 | +0.0245 | -0.0401 | +0.0910 | +0.0009 | NULL |
| rushing_yards | 2024 | 941 | -0.0842 | -0.1414 | -0.0232 | +0.0006 | SIGNAL |
| rushing_yards | pooled | 3723 | -0.0193 | -0.0620 | +0.0227 | +0.0011 | NULL |
| rushing_tds | 2021 | 969 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2022 | 913 | -0.0003 | -0.0008 | +0.0002 | -0.0003 | NULL |
| rushing_tds | 2023 | 900 | -0.0005 | -0.0008 | -0.0000 | -0.0002 | NULL |
| rushing_tds | 2024 | 941 | +0.0000 | -0.0000 | +0.0001 | +0.0000 | NULL |
| rushing_tds | pooled | 3723 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| receptions | 2021 | 969 | -0.0022 | -0.0038 | -0.0006 | +0.0002 | NULL |
| receptions | 2022 | 913 | -0.0003 | -0.0038 | +0.0036 | +0.0009 | NULL |
| receptions | 2023 | 900 | -0.0001 | -0.0036 | +0.0030 | +0.0007 | NULL |
| receptions | 2024 | 941 | -0.0025 | -0.0059 | +0.0007 | +0.0007 | NULL |
| receptions | pooled | 3723 | -0.0012 | -0.0021 | -0.0004 | +0.0002 | NULL |
| receiving_yards | 2021 | 969 | -0.0079 | -0.0163 | -0.0001 | +0.0000 | NULL |
| receiving_yards | 2022 | 913 | -0.0083 | -0.0299 | +0.0163 | +0.0004 | NULL |
| receiving_yards | 2023 | 900 | +0.0005 | -0.0267 | +0.0238 | +0.0005 | NULL |
| receiving_yards | 2024 | 941 | +0.0057 | -0.0195 | +0.0307 | +0.0004 | NULL |
| receiving_yards | pooled | 3723 | -0.0040 | -0.0085 | +0.0002 | +0.0000 | NULL |
| receiving_tds | 2021 | 969 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2022 | 913 | -0.0000 | -0.0001 | +0.0001 | +0.0002 | NULL |
| receiving_tds | 2023 | 900 | +0.0002 | -0.0001 | +0.0004 | +0.0002 | NULL |
| receiving_tds | 2024 | 941 | -0.0007 | -0.0019 | +0.0004 | -0.0034 | NULL |
| receiving_tds | pooled | 3723 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2021 | 969 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2022 | 913 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2023 | 900 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2024 | 941 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | pooled | 3723 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |

## Phase 1 verdict

1/30 cells SIGNAL, 29/30 NULL, 0/30 REGRESSION.
Phase 2 disabled by --no-composite — composite verdict not computed.

## Probe verdict

Phase 1: 1/30 cells SIGNAL.
Phase 2: not run.

**This probe is a screen, not the gate.** The adoption gate is the final word on whether a feature change ships; SIGNAL is necessary but not sufficient. Run the full backtest + adoption gate before any production routing change.
