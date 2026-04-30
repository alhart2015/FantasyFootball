# Feature signal probe — plan9_lgbnb_swap_QB

Baseline features: data\features
Overrides:        data\features_probe\plan9_swap_retro_qb.parquet
Drops:            opp_allowed_qb_fppg_l4
Model class:      lightgbm-nb

## Phase 1 — per-stat screening

### QB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| passing_yards | 2021 | 556 | -0.1167 | -0.2709 | +0.0175 | +0.0009 | NULL |
| passing_yards | 2022 | 544 | -0.0007 | -0.2567 | +0.2537 | +0.0015 | NULL |
| passing_yards | 2023 | 552 | -0.2874 | -0.5318 | -0.0537 | +0.0013 | SIGNAL |
| passing_yards | 2024 | 571 | +0.1978 | -0.1267 | +0.4912 | +0.0020 | NULL |
| passing_yards | pooled | 2223 | -0.0689 | -0.1396 | +0.0066 | +0.0009 | NULL |
| passing_tds | 2021 | 556 | -0.0000 | -0.0000 | +0.0000 | -0.0000 | NULL |
| passing_tds | 2022 | 544 | +0.0000 | -0.0003 | +0.0004 | +0.0000 | NULL |
| passing_tds | 2023 | 552 | -0.0001 | -0.0003 | +0.0000 | +0.0000 | NULL |
| passing_tds | 2024 | 571 | +0.0001 | -0.0012 | +0.0012 | -0.0000 | NULL |
| passing_tds | pooled | 2223 | -0.0000 | -0.0000 | +0.0000 | -0.0000 | NULL |
| interceptions | 2021 | 556 | -0.0003 | -0.0006 | +0.0001 | +0.0002 | NULL |
| interceptions | 2022 | 544 | +0.0004 | -0.0004 | +0.0010 | +0.0004 | NULL |
| interceptions | 2023 | 552 | -0.0000 | -0.0002 | +0.0002 | +0.0001 | NULL |
| interceptions | 2024 | 571 | +0.0000 | -0.0000 | +0.0001 | +0.0000 | NULL |
| interceptions | pooled | 2223 | -0.0000 | -0.0002 | +0.0001 | +0.0002 | NULL |
| rushing_yards | 2021 | 556 | +0.0070 | -0.0120 | +0.0251 | +0.0003 | NULL |
| rushing_yards | 2022 | 544 | +0.0018 | -0.0047 | +0.0079 | +0.0000 | NULL |
| rushing_yards | 2023 | 552 | -0.0001 | -0.0010 | +0.0007 | +0.0000 | NULL |
| rushing_yards | 2024 | 571 | +0.0012 | -0.0005 | +0.0027 | +0.0000 | NULL |
| rushing_yards | pooled | 2223 | +0.0044 | -0.0038 | +0.0130 | +0.0003 | NULL |
| rushing_tds | 2021 | 556 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2022 | 544 | +0.0001 | -0.0010 | +0.0012 | -0.0022 | NULL |
| rushing_tds | 2023 | 552 | +0.0000 | +0.0000 | +0.0001 | +0.0000 | NULL |
| rushing_tds | 2024 | 571 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | pooled | 2223 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2021 | 556 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2022 | 544 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2023 | 552 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2024 | 571 | -0.0000 | -0.0000 | +0.0000 | -0.0000 | NULL |
| fumbles_lost | pooled | 2223 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |

## Phase 1 verdict

1/30 cells SIGNAL, 29/30 NULL, 0/30 REGRESSION.
Phase 2 fired.

## Phase 2 — composite ΔRMSE

| Position | Verdict | RMSE delta (95% CI) | Spearman delta (95% CI) | n_paired |
|---|---|---|---|---:|
| QB | DO_NOT_ADOPT | +0.0004 ([-0.0139, +0.0159]) | -0.0014 ([-0.0046, +0.0015]) | 2692 |

## Probe verdict

Phase 1: 1/30 cells SIGNAL.
Phase 2: 0/1 positions ADOPT.

**This probe is a screen, not the gate.** The adoption gate is the final word on whether a feature change ships; SIGNAL is necessary but not sufficient. Run the full backtest + adoption gate before any production routing change.
