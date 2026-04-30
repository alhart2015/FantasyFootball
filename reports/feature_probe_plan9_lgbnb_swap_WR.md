# Feature signal probe — plan9_lgbnb_swap_WR

Baseline features: data\features
Overrides:        data\features_probe\plan9_swap_retro_wr.parquet
Drops:            opp_allowed_wr_fppg_l4
Model class:      lightgbm-nb

## Phase 1 — per-stat screening

### WR

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 1633 | +0.0001 | -0.0006 | +0.0008 | +0.0000 | NULL |
| receptions | 2022 | 1616 | +0.0000 | -0.0003 | +0.0003 | +0.0000 | NULL |
| receptions | 2023 | 1616 | +0.0000 | -0.0002 | +0.0002 | +0.0000 | NULL |
| receptions | 2024 | 1563 | -0.0000 | -0.0001 | +0.0000 | +0.0000 | NULL |
| receptions | pooled | 6428 | -0.0000 | -0.0003 | +0.0003 | +0.0000 | NULL |
| receiving_yards | 2021 | 1633 | +0.0096 | -0.0137 | +0.0354 | +0.0002 | NULL |
| receiving_yards | 2022 | 1616 | +0.0012 | -0.0103 | +0.0127 | +0.0001 | NULL |
| receiving_yards | 2023 | 1616 | +0.0005 | -0.0098 | +0.0109 | +0.0000 | NULL |
| receiving_yards | 2024 | 1563 | -0.0006 | -0.0094 | +0.0088 | +0.0000 | NULL |
| receiving_yards | pooled | 6428 | +0.0032 | -0.0085 | +0.0151 | +0.0002 | NULL |
| receiving_tds | 2021 | 1633 | +0.0005 | +0.0000 | +0.0009 | +0.0004 | NULL |
| receiving_tds | 2022 | 1616 | +0.0000 | -0.0001 | +0.0001 | +0.0000 | NULL |
| receiving_tds | 2023 | 1616 | -0.0000 | -0.0001 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2024 | 1563 | +0.0001 | -0.0001 | +0.0002 | +0.0000 | NULL |
| receiving_tds | pooled | 6428 | +0.0002 | -0.0000 | +0.0004 | +0.0004 | NULL |
| rushing_yards | 2021 | 1633 | +0.0003 | -0.0030 | +0.0040 | +0.0005 | NULL |
| rushing_yards | 2022 | 1616 | +0.0027 | -0.0010 | +0.0065 | +0.0003 | NULL |
| rushing_yards | 2023 | 1616 | -0.0028 | -0.0142 | +0.0088 | -0.0008 | NULL |
| rushing_yards | 2024 | 1563 | +0.0004 | -0.0016 | +0.0026 | +0.0001 | NULL |
| rushing_yards | pooled | 6428 | +0.0009 | -0.0012 | +0.0029 | +0.0005 | NULL |
| rushing_tds | 2021 | 1633 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2022 | 1616 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2023 | 1616 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2024 | 1563 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | pooled | 6428 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2021 | 1633 | -0.0000 | -0.0000 | +0.0000 | +0.0003 | NULL |
| fumbles_lost | 2022 | 1616 | +0.0000 | +0.0000 | +0.0001 | +0.0003 | NULL |
| fumbles_lost | 2023 | 1616 | -0.0000 | -0.0000 | +0.0000 | +0.0001 | NULL |
| fumbles_lost | 2024 | 1563 | +0.0000 | -0.0000 | +0.0000 | +0.0001 | NULL |
| fumbles_lost | pooled | 6428 | +0.0000 | -0.0000 | +0.0000 | +0.0003 | NULL |

## Phase 1 verdict

0/30 cells SIGNAL, 30/30 NULL, 0/30 REGRESSION.
Phase 2 fired.

## Phase 2 — composite ΔRMSE

| Position | Verdict | RMSE delta (95% CI) | Spearman delta (95% CI) | n_paired |
|---|---|---|---|---:|
| WR | DO_NOT_ADOPT | +0.0011 ([-0.0039, +0.0059]) | -0.0004 ([-0.0015, +0.0007]) | 8470 |

## Probe verdict

Phase 1: 0/30 cells SIGNAL.
Phase 2: 0/1 positions ADOPT.

**This probe is a screen, not the gate.** The adoption gate is the final word on whether a feature change ships; SIGNAL is necessary but not sufficient. Run the full backtest + adoption gate before any production routing change.
