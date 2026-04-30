# Feature signal probe — plan9_swap_retro_RB

Baseline features: data\features
Overrides:        data\features_probe\plan9_swap_retro_rb.parquet
Drops:            opp_allowed_rb_fppg_l4
Model class:      baseline

## Phase 1 — per-stat screening

### RB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| rushing_yards | 2021 | 969 | +0.0373 | -0.0757 | +0.1593 | +0.0033 | NULL |
| rushing_yards | 2022 | 913 | -0.0734 | -0.1871 | +0.0389 | +0.0022 | NULL |
| rushing_yards | 2023 | 900 | +0.0005 | -0.1058 | +0.1175 | +0.0024 | NULL |
| rushing_yards | 2024 | 941 | -0.1699 | -0.2705 | -0.0647 | +0.0021 | SIGNAL |
| rushing_yards | pooled | 3723 | -0.0543 | -0.1250 | +0.0101 | +0.0033 | NULL |
| rushing_tds | 2021 | 969 | +0.0000 | -0.0006 | +0.0006 | +0.0004 | NULL |
| rushing_tds | 2022 | 913 | -0.0004 | -0.0011 | +0.0004 | -0.0000 | NULL |
| rushing_tds | 2023 | 900 | -0.0005 | -0.0011 | +0.0001 | +0.0000 | NULL |
| rushing_tds | 2024 | 941 | -0.0003 | -0.0008 | +0.0002 | +0.0002 | NULL |
| rushing_tds | pooled | 3723 | -0.0001 | -0.0004 | +0.0002 | +0.0004 | NULL |
| receptions | 2021 | 969 | -0.0016 | -0.0033 | +0.0000 | +0.0002 | NULL |
| receptions | 2022 | 913 | -0.0002 | -0.0035 | +0.0034 | +0.0006 | NULL |
| receptions | 2023 | 900 | +0.0003 | -0.0026 | +0.0033 | +0.0005 | NULL |
| receptions | 2024 | 941 | -0.0022 | -0.0048 | +0.0005 | +0.0004 | NULL |
| receptions | pooled | 3723 | -0.0009 | -0.0018 | -0.0001 | +0.0002 | NULL |
| receiving_yards | 2021 | 969 | +0.0031 | -0.0007 | +0.0073 | +0.0000 | NULL |
| receiving_yards | 2022 | 913 | -0.0042 | -0.0134 | +0.0056 | +0.0001 | NULL |
| receiving_yards | 2023 | 900 | -0.0001 | -0.0148 | +0.0143 | +0.0001 | NULL |
| receiving_yards | 2024 | 941 | +0.0006 | -0.0145 | +0.0147 | +0.0001 | NULL |
| receiving_yards | pooled | 3723 | +0.0019 | -0.0002 | +0.0041 | +0.0000 | NULL |
| receiving_tds | 2021 | 969 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2022 | 913 | -0.0000 | -0.0003 | +0.0002 | +0.0003 | NULL |
| receiving_tds | 2023 | 900 | +0.0002 | -0.0002 | +0.0005 | +0.0004 | NULL |
| receiving_tds | 2024 | 941 | -0.0002 | -0.0004 | +0.0000 | +0.0002 | NULL |
| receiving_tds | pooled | 3723 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2021 | 969 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2022 | 913 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2023 | 900 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2024 | 941 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | pooled | 3723 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |

## Phase 1 verdict

1/30 cells SIGNAL, 29/30 NULL, 0/30 REGRESSION.
Phase 2 disabled by --no-composite — composite verdict not computed.

## Probe verdict

Phase 1: 1/30 cells SIGNAL.
Phase 2: not run.

**This probe is a screen, not the gate.** The adoption gate is the final word on whether a feature change ships; SIGNAL is necessary but not sufficient. Run the full backtest + adoption gate before any production routing change.
