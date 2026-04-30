# Feature signal probe — plan9_lgbnb_augment_QB

Baseline features: data\features
Overrides:        data\features_probe\plan9_swap_retro_qb.parquet
Drops:            (none)
Model class:      lightgbm-nb

## Phase 1 — per-stat screening

### QB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| passing_yards | 2021 | 556 | -0.0142 | -0.0378 | +0.0092 | +0.0000 | NULL |
| passing_yards | 2022 | 544 | +0.0505 | -0.1390 | +0.2388 | -0.0006 | NULL |
| passing_yards | 2023 | 552 | -0.0084 | -0.0254 | +0.0078 | +0.0000 | NULL |
| passing_yards | 2024 | 571 | +0.0282 | -0.0227 | +0.0811 | +0.0001 | NULL |
| passing_yards | pooled | 2223 | +0.0013 | -0.0109 | +0.0139 | +0.0000 | NULL |
| passing_tds | 2021 | 556 | +0.0002 | -0.0006 | +0.0010 | +0.0003 | NULL |
| passing_tds | 2022 | 544 | -0.0000 | -0.0019 | +0.0019 | -0.0003 | NULL |
| passing_tds | 2023 | 552 | +0.0012 | -0.0003 | +0.0029 | +0.0003 | NULL |
| passing_tds | 2024 | 571 | -0.0001 | -0.0010 | +0.0008 | -0.0001 | NULL |
| passing_tds | pooled | 2223 | +0.0001 | -0.0003 | +0.0005 | +0.0003 | NULL |
| interceptions | 2021 | 556 | -0.0004 | -0.0009 | +0.0001 | +0.0005 | NULL |
| interceptions | 2022 | 544 | +0.0007 | -0.0001 | +0.0015 | +0.0009 | NULL |
| interceptions | 2023 | 552 | -0.0000 | -0.0002 | +0.0001 | +0.0001 | NULL |
| interceptions | 2024 | 571 | +0.0000 | -0.0001 | +0.0001 | +0.0000 | NULL |
| interceptions | pooled | 2223 | -0.0000 | -0.0003 | +0.0002 | +0.0005 | NULL |
| rushing_yards | 2021 | 556 | +0.0007 | -0.0090 | +0.0100 | +0.0001 | NULL |
| rushing_yards | 2022 | 544 | +0.0015 | -0.0062 | +0.0086 | +0.0001 | NULL |
| rushing_yards | 2023 | 552 | +0.0000 | -0.0021 | +0.0019 | +0.0000 | NULL |
| rushing_yards | 2024 | 571 | +0.0017 | -0.0001 | +0.0035 | +0.0000 | NULL |
| rushing_yards | pooled | 2223 | +0.0022 | -0.0024 | +0.0073 | +0.0001 | NULL |
| rushing_tds | 2021 | 556 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2022 | 544 | +0.0001 | -0.0010 | +0.0013 | -0.0023 | NULL |
| rushing_tds | 2023 | 552 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2024 | 571 | -0.0000 | -0.0001 | +0.0000 | +0.0000 | NULL |
| rushing_tds | pooled | 2223 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2021 | 556 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2022 | 544 | -0.0000 | -0.0000 | -0.0000 | +0.0000 | NULL |
| fumbles_lost | 2023 | 552 | -0.0000 | -0.0001 | +0.0000 | +0.0001 | NULL |
| fumbles_lost | 2024 | 571 | -0.0000 | -0.0001 | +0.0000 | +0.0001 | NULL |
| fumbles_lost | pooled | 2223 | -0.0000 | -0.0000 | -0.0000 | +0.0000 | NULL |

## Phase 1 verdict

0/30 cells SIGNAL, 30/30 NULL, 0/30 REGRESSION.
Phase 2 fired.

## Phase 2 — composite ΔRMSE

| Position | Verdict | RMSE delta (95% CI) | Spearman delta (95% CI) | n_paired |
|---|---|---|---|---:|
| QB | DO_NOT_ADOPT | +0.0110 ([-0.0003, +0.0227]) | +0.0006 ([-0.0015, +0.0028]) | 2692 |

## Probe verdict

Phase 1: 0/30 cells SIGNAL.
Phase 2: 0/1 positions ADOPT.

**This probe is a screen, not the gate.** The adoption gate is the final word on whether a feature change ships; SIGNAL is necessary but not sufficient. Run the full backtest + adoption gate before any production routing change.
