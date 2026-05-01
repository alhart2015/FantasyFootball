# Feature signal probe — pbp_receiver_lgbnb_augment

Baseline features: data\features
Overrides:        data\features_probe\pbp_receiver.parquet
Drops:            (none)
Model class:      lightgbm-nb

## Phase 1 — per-stat screening

### WR

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 1583 | +0.0001 | -0.0028 | +0.0033 | -0.0002 | NULL |
| receptions | 2022 | 1582 | +0.0001 | -0.0020 | +0.0020 | +0.0003 | NULL |
| receptions | 2023 | 1574 | -0.0039 | -0.0061 | -0.0016 | +0.0003 | NULL |
| receptions | 2024 | 1529 | +0.0011 | -0.0024 | +0.0043 | +0.0007 | NULL |
| receptions | pooled | 6268 | -0.0003 | -0.0017 | +0.0012 | -0.0002 | NULL |
| receiving_yards | 2021 | 1583 | -0.0303 | -0.0573 | -0.0041 | +0.0004 | NULL |
| receiving_yards | 2022 | 1582 | +0.0259 | -0.0230 | +0.0705 | +0.0009 | NULL |
| receiving_yards | 2023 | 1574 | -0.0010 | -0.0300 | +0.0278 | +0.0004 | NULL |
| receiving_yards | 2024 | 1529 | +0.0062 | -0.0302 | +0.0420 | +0.0005 | NULL |
| receiving_yards | pooled | 6268 | -0.0005 | -0.0142 | +0.0121 | +0.0004 | NULL |
| receiving_tds | 2021 | 1583 | -0.0001 | -0.0006 | +0.0004 | +0.0006 | NULL |
| receiving_tds | 2022 | 1582 | -0.0000 | -0.0007 | +0.0006 | +0.0006 | NULL |
| receiving_tds | 2023 | 1574 | -0.0000 | -0.0006 | +0.0005 | +0.0005 | NULL |
| receiving_tds | 2024 | 1529 | +0.0006 | +0.0001 | +0.0012 | +0.0006 | NULL |
| receiving_tds | pooled | 6268 | +0.0003 | +0.0000 | +0.0006 | +0.0006 | NULL |
| rushing_yards | 2021 | 1583 | -0.0025 | -0.0124 | +0.0087 | +0.0021 | NULL |
| rushing_yards | 2022 | 1582 | -0.0018 | -0.0146 | +0.0096 | +0.0017 | NULL |
| rushing_yards | 2023 | 1574 | -0.0027 | -0.0137 | +0.0073 | +0.0016 | NULL |
| rushing_yards | 2024 | 1529 | +0.0029 | -0.0073 | +0.0140 | +0.0015 | NULL |
| rushing_yards | pooled | 6268 | -0.0003 | -0.0067 | +0.0054 | +0.0021 | NULL |
| rushing_tds | 2021 | 1583 | +0.0000 | -0.0000 | +0.0000 | +0.0002 | NULL |
| rushing_tds | 2022 | 1582 | +0.0000 | -0.0000 | +0.0001 | +0.0002 | NULL |
| rushing_tds | 2023 | 1574 | +0.0000 | -0.0002 | +0.0003 | -0.0010 | NULL |
| rushing_tds | 2024 | 1529 | +0.0000 | -0.0000 | +0.0001 | +0.0002 | NULL |
| rushing_tds | pooled | 6268 | -0.0000 | -0.0000 | +0.0000 | +0.0002 | NULL |
| fumbles_lost | 2021 | 1583 | -0.0001 | -0.0003 | +0.0001 | +0.0013 | NULL |
| fumbles_lost | 2022 | 1582 | -0.0001 | -0.0003 | +0.0001 | +0.0013 | NULL |
| fumbles_lost | 2023 | 1574 | +0.0002 | +0.0000 | +0.0003 | +0.0012 | NULL |
| fumbles_lost | 2024 | 1529 | +0.0002 | +0.0000 | +0.0003 | +0.0007 | NULL |
| fumbles_lost | pooled | 6268 | +0.0000 | -0.0001 | +0.0002 | +0.0013 | NULL |

### TE

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 629 | +0.0061 | +0.0013 | +0.0108 | -0.0006 | NULL |
| receptions | 2022 | 690 | +0.0025 | -0.0011 | +0.0060 | -0.0001 | NULL |
| receptions | 2023 | 707 | -0.0016 | -0.0040 | +0.0010 | -0.0003 | NULL |
| receptions | 2024 | 686 | +0.0015 | -0.0018 | +0.0047 | +0.0004 | NULL |
| receptions | pooled | 2712 | +0.0028 | +0.0000 | +0.0059 | -0.0006 | NULL |
| receiving_yards | 2021 | 629 | +0.0209 | -0.0230 | +0.0660 | +0.0006 | NULL |
| receiving_yards | 2022 | 690 | +0.1474 | +0.0479 | +0.2517 | -0.0031 | REGRESSION |
| receiving_yards | 2023 | 707 | +0.0004 | -0.0321 | +0.0343 | +0.0002 | NULL |
| receiving_yards | 2024 | 686 | +0.0379 | -0.0090 | +0.0922 | +0.0006 | NULL |
| receiving_yards | pooled | 2712 | +0.0287 | +0.0044 | +0.0535 | +0.0006 | NULL |
| receiving_tds | 2021 | 629 | +0.0002 | -0.0008 | +0.0011 | +0.0007 | NULL |
| receiving_tds | 2022 | 690 | -0.0007 | -0.0015 | +0.0003 | +0.0006 | NULL |
| receiving_tds | 2023 | 707 | -0.0001 | -0.0012 | +0.0010 | +0.0014 | NULL |
| receiving_tds | 2024 | 686 | +0.0005 | -0.0009 | +0.0019 | +0.0012 | NULL |
| receiving_tds | pooled | 2712 | -0.0001 | -0.0006 | +0.0003 | +0.0007 | NULL |
| rushing_yards | 2021 | 629 | -0.0004 | -0.0028 | +0.0031 | +0.0003 | NULL |
| rushing_yards | 2022 | 690 | +0.0025 | +0.0003 | +0.0056 | +0.0005 | NULL |
| rushing_yards | 2023 | 707 | -0.0008 | -0.0024 | +0.0012 | +0.0002 | NULL |
| rushing_yards | 2024 | 686 | +0.0013 | -0.0000 | +0.0028 | +0.0003 | NULL |
| rushing_yards | pooled | 2712 | +0.0012 | -0.0000 | +0.0026 | +0.0003 | NULL |
| rushing_tds | 2021 | 629 | -0.0000 | -0.0001 | +0.0000 | +0.0004 | NULL |
| rushing_tds | 2022 | 690 | -0.0000 | -0.0000 | +0.0002 | +0.0006 | NULL |
| rushing_tds | 2023 | 707 | +0.0002 | +0.0001 | +0.0003 | +0.0011 | NULL |
| rushing_tds | 2024 | 686 | -0.0000 | -0.0001 | +0.0003 | +0.0010 | NULL |
| rushing_tds | pooled | 2712 | -0.0000 | -0.0000 | +0.0000 | +0.0004 | NULL |
| fumbles_lost | 2021 | 629 | +0.0003 | -0.0000 | +0.0006 | +0.0018 | NULL |
| fumbles_lost | 2022 | 690 | +0.0001 | -0.0000 | +0.0003 | +0.0005 | NULL |
| fumbles_lost | 2023 | 707 | +0.0001 | -0.0000 | +0.0002 | +0.0002 | NULL |
| fumbles_lost | 2024 | 686 | +0.0000 | +0.0000 | +0.0001 | +0.0000 | NULL |
| fumbles_lost | pooled | 2712 | +0.0003 | +0.0001 | +0.0004 | +0.0018 | NULL |

## Phase 1 verdict

0/60 cells SIGNAL, 59/60 NULL, 1/60 REGRESSION.
Phase 2 fired.

## Phase 2 — composite ΔRMSE

| Position | Verdict | RMSE delta (95% CI) | Spearman delta (95% CI) | n_paired |
|---|---|---|---|---:|
| WR | DO_NOT_ADOPT | -0.0006 ([-0.0063, +0.0053]) | +0.0008 ([-0.0005, +0.0023]) | 8470 |
| TE | DO_NOT_ADOPT | +0.0054 ([-0.0035, +0.0143]) | -0.0007 ([-0.0044, +0.0028]) | 4257 |

## Probe verdict

Phase 1: 0/60 cells SIGNAL.
Phase 2: 0/2 positions ADOPT.

**This probe is a screen, not the gate.** The adoption gate is the final word on whether a feature change ships; SIGNAL is necessary but not sufficient. Run the full backtest + adoption gate before any production routing change.
