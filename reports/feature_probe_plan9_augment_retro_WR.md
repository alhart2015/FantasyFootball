# Feature signal probe — plan9_augment_retro_WR

Baseline features: data\features
Overrides:        data\features_probe\plan9_swap_retro_wr.parquet
Drops:            (none)
Model class:      baseline

## Phase 1 — per-stat screening

### WR

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 1633 | -0.0000 | -0.0002 | +0.0002 | +0.0000 | NULL |
| receptions | 2022 | 1616 | +0.0000 | -0.0001 | +0.0001 | +0.0000 | NULL |
| receptions | 2023 | 1616 | -0.0001 | -0.0005 | +0.0002 | +0.0000 | NULL |
| receptions | 2024 | 1563 | +0.0003 | -0.0004 | +0.0011 | +0.0000 | NULL |
| receptions | pooled | 6428 | +0.0000 | -0.0001 | +0.0001 | +0.0000 | NULL |
| receiving_yards | 2021 | 1633 | +0.0173 | -0.0165 | +0.0548 | +0.0004 | NULL |
| receiving_yards | 2022 | 1616 | +0.0147 | -0.0039 | +0.0327 | +0.0002 | NULL |
| receiving_yards | 2023 | 1616 | +0.0019 | -0.0062 | +0.0106 | +0.0000 | NULL |
| receiving_yards | 2024 | 1563 | -0.0004 | -0.0053 | +0.0049 | +0.0000 | NULL |
| receiving_yards | pooled | 6428 | +0.0140 | -0.0033 | +0.0310 | +0.0004 | NULL |
| receiving_tds | 2021 | 1633 | +0.0004 | +0.0000 | +0.0008 | +0.0004 | NULL |
| receiving_tds | 2022 | 1616 | +0.0000 | -0.0001 | +0.0001 | +0.0000 | NULL |
| receiving_tds | 2023 | 1616 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2024 | 1563 | +0.0000 | -0.0000 | +0.0001 | +0.0000 | NULL |
| receiving_tds | pooled | 6428 | +0.0002 | -0.0000 | +0.0004 | +0.0004 | NULL |
| rushing_yards | 2021 | 1633 | +0.0006 | -0.0034 | +0.0053 | +0.0009 | NULL |
| rushing_yards | 2022 | 1616 | +0.0041 | +0.0000 | +0.0087 | +0.0005 | NULL |
| rushing_yards | 2023 | 1616 | -0.0024 | -0.0138 | +0.0090 | -0.0007 | NULL |
| rushing_yards | 2024 | 1563 | +0.0004 | -0.0012 | +0.0021 | +0.0001 | NULL |
| rushing_yards | pooled | 6428 | +0.0022 | -0.0001 | +0.0047 | +0.0009 | NULL |
| rushing_tds | 2021 | 1633 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2022 | 1616 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2023 | 1616 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2024 | 1563 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | pooled | 6428 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2021 | 1633 | -0.0000 | -0.0000 | +0.0000 | +0.0003 | NULL |
| fumbles_lost | 2022 | 1616 | +0.0000 | +0.0000 | +0.0001 | +0.0003 | NULL |
| fumbles_lost | 2023 | 1616 | -0.0000 | -0.0000 | +0.0000 | +0.0001 | NULL |
| fumbles_lost | 2024 | 1563 | -0.0000 | -0.0000 | +0.0000 | +0.0001 | NULL |
| fumbles_lost | pooled | 6428 | +0.0000 | -0.0000 | +0.0000 | +0.0003 | NULL |

## Phase 1 verdict

0/30 cells SIGNAL, 30/30 NULL, 0/30 REGRESSION.
No SIGNAL cells — Phase 2 skipped. Probe predicts the adoption gate would return DO_NOT_ADOPT.

## Probe verdict

Phase 1: 0/30 cells SIGNAL.
Phase 2: not run.

**This probe is a screen, not the gate.** The adoption gate is the final word on whether a feature change ships; SIGNAL is necessary but not sufficient. Run the full backtest + adoption gate before any production routing change.
