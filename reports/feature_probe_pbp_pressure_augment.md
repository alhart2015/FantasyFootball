# Feature signal probe — pbp_pressure_augment

Baseline features: data\features
Overrides:        data\features_probe\pbp_pressure.parquet
Drops:            (none)
Model class:      baseline

## Phase 1 — per-stat screening

### QB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| passing_yards | 2021 | 556 | -0.0014 | -0.0080 | +0.0046 | +0.0001 | NULL |
| passing_yards | 2022 | 544 | -0.0258 | -0.0545 | +0.0028 | +0.0003 | NULL |
| passing_yards | 2023 | 552 | -0.1549 | -0.3797 | +0.0874 | +0.0022 | NULL |
| passing_yards | 2024 | 571 | +0.1323 | -0.2988 | +0.5490 | +0.0030 | NULL |
| passing_yards | pooled | 2223 | -0.0024 | -0.0054 | +0.0006 | +0.0001 | NULL |
| passing_tds | 2021 | 556 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| passing_tds | 2022 | 544 | +0.0000 | -0.0009 | +0.0009 | +0.0007 | NULL |
| passing_tds | 2023 | 552 | +0.0001 | -0.0009 | +0.0011 | +0.0006 | NULL |
| passing_tds | 2024 | 571 | -0.0000 | -0.0004 | +0.0004 | +0.0002 | NULL |
| passing_tds | pooled | 2223 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| interceptions | 2021 | 556 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| interceptions | 2022 | 544 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| interceptions | 2023 | 552 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| interceptions | 2024 | 571 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| interceptions | pooled | 2223 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2021 | 556 | +0.0004 | -0.0007 | +0.0013 | +0.0000 | NULL |
| rushing_yards | 2022 | 544 | -0.0002 | -0.0010 | +0.0005 | +0.0000 | NULL |
| rushing_yards | 2023 | 552 | +0.0004 | -0.0001 | +0.0010 | +0.0000 | NULL |
| rushing_yards | 2024 | 571 | -0.0000 | -0.0004 | +0.0004 | +0.0000 | NULL |
| rushing_yards | pooled | 2223 | +0.0003 | -0.0002 | +0.0008 | +0.0000 | NULL |
| rushing_tds | 2021 | 556 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2022 | 544 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2023 | 552 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2024 | 571 | -0.0000 | -0.0000 | -0.0000 | +0.0000 | NULL |
| rushing_tds | pooled | 2223 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2021 | 556 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2022 | 544 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2023 | 552 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2024 | 571 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | pooled | 2223 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |

### RB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| rushing_yards | 2021 | 969 | +0.0315 | -0.0322 | +0.0952 | -0.0008 | NULL |
| rushing_yards | 2022 | 913 | -0.0299 | -0.0896 | +0.0340 | +0.0004 | NULL |
| rushing_yards | 2023 | 900 | -0.0582 | -0.1285 | +0.0117 | +0.0006 | NULL |
| rushing_yards | 2024 | 941 | -0.0161 | -0.1253 | +0.0796 | +0.0014 | NULL |
| rushing_yards | pooled | 3723 | -0.0000 | -0.0373 | +0.0366 | -0.0008 | NULL |
| rushing_tds | 2021 | 969 | -0.0003 | -0.0011 | +0.0005 | +0.0021 | NULL |
| rushing_tds | 2022 | 913 | -0.0001 | -0.0010 | +0.0007 | +0.0015 | NULL |
| rushing_tds | 2023 | 900 | +0.0008 | -0.0004 | +0.0019 | +0.0018 | NULL |
| rushing_tds | 2024 | 941 | +0.0004 | -0.0006 | +0.0014 | +0.0014 | NULL |
| rushing_tds | pooled | 3723 | +0.0005 | -0.0000 | +0.0009 | +0.0021 | NULL |
| receptions | 2021 | 969 | -0.0001 | -0.0016 | +0.0015 | +0.0007 | NULL |
| receptions | 2022 | 913 | -0.0019 | -0.0039 | +0.0001 | +0.0008 | NULL |
| receptions | 2023 | 900 | -0.0003 | -0.0032 | +0.0026 | +0.0013 | NULL |
| receptions | 2024 | 941 | -0.0002 | -0.0028 | +0.0025 | +0.0012 | NULL |
| receptions | pooled | 3723 | -0.0009 | -0.0017 | -0.0001 | +0.0007 | NULL |
| receiving_yards | 2021 | 969 | +0.0050 | -0.0238 | +0.0325 | +0.0000 | NULL |
| receiving_yards | 2022 | 913 | -0.0225 | -0.0591 | +0.0123 | -0.0002 | NULL |
| receiving_yards | 2023 | 900 | +0.0211 | -0.0161 | +0.0581 | +0.0005 | NULL |
| receiving_yards | 2024 | 941 | +0.0203 | -0.0106 | +0.0518 | +0.0006 | NULL |
| receiving_yards | pooled | 3723 | -0.0058 | -0.0239 | +0.0127 | +0.0000 | NULL |
| receiving_tds | 2021 | 969 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2022 | 913 | -0.0000 | -0.0007 | +0.0006 | +0.0004 | NULL |
| receiving_tds | 2023 | 900 | +0.0004 | -0.0004 | +0.0013 | +0.0011 | NULL |
| receiving_tds | 2024 | 941 | -0.0003 | -0.0009 | +0.0002 | +0.0009 | NULL |
| receiving_tds | pooled | 3723 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2021 | 969 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2022 | 913 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2023 | 900 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2024 | 941 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | pooled | 3723 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |

### WR

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 1619 | +0.0005 | -0.0013 | +0.0023 | +0.0001 | NULL |
| receptions | 2022 | 1609 | +0.0025 | +0.0002 | +0.0050 | +0.0005 | NULL |
| receptions | 2023 | 1600 | +0.0012 | -0.0006 | +0.0032 | -0.0001 | NULL |
| receptions | 2024 | 1563 | -0.0005 | -0.0018 | +0.0009 | +0.0001 | NULL |
| receptions | pooled | 6391 | +0.0005 | -0.0005 | +0.0016 | +0.0001 | NULL |
| receiving_yards | 2021 | 1619 | +0.0058 | -0.0048 | +0.0159 | +0.0003 | NULL |
| receiving_yards | 2022 | 1609 | +0.0022 | -0.0043 | +0.0099 | +0.0001 | NULL |
| receiving_yards | 2023 | 1600 | +0.0050 | -0.0003 | +0.0107 | +0.0000 | NULL |
| receiving_yards | 2024 | 1563 | -0.0023 | -0.0074 | +0.0034 | +0.0000 | NULL |
| receiving_yards | pooled | 6391 | +0.0022 | -0.0032 | +0.0075 | +0.0003 | NULL |
| receiving_tds | 2021 | 1619 | +0.0000 | -0.0000 | +0.0001 | +0.0001 | NULL |
| receiving_tds | 2022 | 1609 | -0.0000 | -0.0001 | +0.0000 | +0.0001 | NULL |
| receiving_tds | 2023 | 1600 | +0.0000 | -0.0000 | +0.0001 | +0.0001 | NULL |
| receiving_tds | 2024 | 1563 | +0.0000 | -0.0000 | +0.0001 | +0.0001 | NULL |
| receiving_tds | pooled | 6391 | +0.0000 | -0.0000 | +0.0000 | +0.0001 | NULL |
| rushing_yards | 2021 | 1619 | -0.0009 | -0.0021 | +0.0004 | +0.0003 | NULL |
| rushing_yards | 2022 | 1609 | +0.0121 | +0.0015 | +0.0233 | +0.0013 | NULL |
| rushing_yards | 2023 | 1600 | -0.0012 | -0.0024 | +0.0002 | +0.0002 | NULL |
| rushing_yards | 2024 | 1563 | +0.0021 | -0.0028 | +0.0081 | +0.0009 | NULL |
| rushing_yards | pooled | 6391 | -0.0010 | -0.0018 | -0.0002 | +0.0003 | NULL |
| rushing_tds | 2021 | 1619 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2022 | 1609 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2023 | 1600 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2024 | 1563 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | pooled | 6391 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2021 | 1619 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2022 | 1609 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2023 | 1600 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2024 | 1563 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | pooled | 6391 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |

### TE

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 633 | +0.0002 | -0.0002 | +0.0006 | +0.0001 | NULL |
| receptions | 2022 | 704 | +0.0011 | -0.0013 | +0.0036 | -0.0005 | NULL |
| receptions | 2023 | 719 | -0.0006 | -0.0041 | +0.0027 | -0.0005 | NULL |
| receptions | 2024 | 691 | -0.0004 | -0.0034 | +0.0024 | +0.0002 | NULL |
| receptions | pooled | 2747 | +0.0001 | -0.0001 | +0.0003 | +0.0001 | NULL |
| receiving_yards | 2021 | 633 | -0.0000 | -0.0002 | +0.0002 | +0.0000 | NULL |
| receiving_yards | 2022 | 704 | +0.0397 | -0.0112 | +0.0894 | -0.0007 | NULL |
| receiving_yards | 2023 | 719 | -0.0039 | -0.0580 | +0.0460 | -0.0001 | NULL |
| receiving_yards | 2024 | 691 | +0.0161 | -0.0425 | +0.0777 | -0.0000 | NULL |
| receiving_yards | pooled | 2747 | -0.0000 | -0.0001 | +0.0001 | +0.0000 | NULL |
| receiving_tds | 2021 | 633 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2022 | 704 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2023 | 719 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2024 | 691 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | pooled | 2747 | -0.0000 | -0.0000 | -0.0000 | +0.0000 | NULL |
| rushing_yards | 2021 | 633 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2022 | 704 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2023 | 719 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2024 | 691 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | pooled | 2747 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2021 | 633 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2022 | 704 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2023 | 719 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2024 | 691 | -0.0000 | -0.0000 | -0.0000 | +0.0000 | NULL |
| rushing_tds | pooled | 2747 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2021 | 633 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2022 | 704 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2023 | 719 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2024 | 691 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | pooled | 2747 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |

## Phase 1 verdict

0/120 cells SIGNAL, 120/120 NULL, 0/120 REGRESSION.
No SIGNAL cells — Phase 2 skipped. Probe predicts the adoption gate would return DO_NOT_ADOPT.

## Probe verdict

Phase 1: 0/120 cells SIGNAL.
Phase 2: not run.

**This probe is a screen, not the gate.** The adoption gate is the final word on whether a feature change ships; SIGNAL is necessary but not sufficient. Run the full backtest + adoption gate before any production routing change.
