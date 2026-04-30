# Feature signal probe — plan9_lgbnb_swap_TE

Baseline features: data\features
Overrides:        data\features_probe\plan9_swap_retro_te.parquet
Drops:            opp_allowed_te_fppg_l4
Model class:      lightgbm-nb

## Phase 1 — per-stat screening

### TE

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 635 | +0.0006 | -0.0019 | +0.0033 | +0.0003 | NULL |
| receptions | 2022 | 705 | -0.0002 | -0.0015 | +0.0011 | +0.0001 | NULL |
| receptions | 2023 | 727 | +0.0009 | -0.0013 | +0.0030 | -0.0002 | NULL |
| receptions | 2024 | 694 | +0.0001 | -0.0006 | +0.0009 | +0.0000 | NULL |
| receptions | pooled | 2761 | +0.0007 | -0.0004 | +0.0019 | +0.0003 | NULL |
| receiving_yards | 2021 | 635 | -0.0007 | -0.0020 | +0.0007 | +0.0000 | NULL |
| receiving_yards | 2022 | 705 | +0.0034 | -0.0299 | +0.0363 | -0.0002 | NULL |
| receiving_yards | 2023 | 727 | -0.0079 | -0.0340 | +0.0198 | +0.0002 | NULL |
| receiving_yards | 2024 | 694 | +0.0020 | -0.0319 | +0.0336 | +0.0002 | NULL |
| receiving_yards | pooled | 2761 | -0.0005 | -0.0011 | +0.0001 | +0.0000 | NULL |
| receiving_tds | 2021 | 635 | +0.0000 | -0.0000 | +0.0001 | +0.0001 | NULL |
| receiving_tds | 2022 | 705 | +0.0000 | -0.0000 | +0.0000 | +0.0001 | NULL |
| receiving_tds | 2023 | 727 | -0.0000 | -0.0001 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2024 | 694 | +0.0000 | -0.0000 | +0.0001 | +0.0001 | NULL |
| receiving_tds | pooled | 2761 | +0.0000 | -0.0000 | +0.0000 | +0.0001 | NULL |
| rushing_yards | 2021 | 635 | +0.0000 | -0.0003 | +0.0003 | +0.0001 | NULL |
| rushing_yards | 2022 | 705 | +0.0001 | -0.0000 | +0.0002 | +0.0001 | NULL |
| rushing_yards | 2023 | 727 | -0.0000 | -0.0001 | +0.0001 | +0.0000 | NULL |
| rushing_yards | 2024 | 694 | -0.0000 | -0.0001 | +0.0001 | +0.0000 | NULL |
| rushing_yards | pooled | 2761 | +0.0000 | -0.0001 | +0.0001 | +0.0001 | NULL |
| rushing_tds | 2021 | 635 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2022 | 705 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2023 | 727 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2024 | 694 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | pooled | 2761 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2021 | 635 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2022 | 705 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2023 | 727 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2024 | 694 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | pooled | 2761 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |

## Phase 1 verdict

0/30 cells SIGNAL, 30/30 NULL, 0/30 REGRESSION.
Phase 2 fired.

## Phase 2 — composite ΔRMSE

| Position | Verdict | RMSE delta (95% CI) | Spearman delta (95% CI) | n_paired |
|---|---|---|---|---:|
| TE | DO_NOT_ADOPT | +0.0083 ([+0.0008, +0.0161]) | -0.0035 ([-0.0066, -0.0007]) | 4257 |

## Probe verdict

Phase 1: 0/30 cells SIGNAL.
Phase 2: 0/1 positions ADOPT.

**This probe is a screen, not the gate.** The adoption gate is the final word on whether a feature change ships; SIGNAL is necessary but not sufficient. Run the full backtest + adoption gate before any production routing change.
