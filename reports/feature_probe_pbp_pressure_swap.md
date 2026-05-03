# Feature signal probe — pbp_pressure_swap

Baseline features: data\features
Overrides:        data\features_probe\pbp_pressure.parquet
Drops:            opp_allowed_qb_fppg_l4, opp_allowed_rb_fppg_l4, opp_allowed_wr_fppg_l4, opp_allowed_te_fppg_l4
Model class:      baseline

## Phase 1 — per-stat screening

### QB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| passing_yards | 2021 | 620 | -0.0179 | -0.0415 | +0.0052 | +0.0002 | NULL |
| passing_yards | 2022 | 612 | -0.1449 | -0.4127 | +0.1196 | +0.0025 | NULL |
| passing_yards | 2023 | 622 | -0.0566 | -0.3236 | +0.2095 | +0.0031 | NULL |
| passing_yards | 2024 | 641 | +0.1617 | -0.1253 | +0.4537 | +0.0025 | NULL |
| passing_yards | pooled | 2495 | -0.0067 | -0.0192 | +0.0043 | +0.0002 | NULL |
| passing_tds | 2021 | 620 | -0.0001 | -0.0003 | +0.0001 | +0.0002 | NULL |
| passing_tds | 2022 | 612 | +0.0003 | -0.0007 | +0.0012 | +0.0008 | NULL |
| passing_tds | 2023 | 622 | -0.0007 | -0.0018 | +0.0002 | -0.0000 | NULL |
| passing_tds | 2024 | 641 | +0.0000 | -0.0002 | +0.0003 | +0.0001 | NULL |
| passing_tds | pooled | 2495 | +0.0001 | -0.0000 | +0.0002 | +0.0002 | NULL |
| interceptions | 2021 | 620 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| interceptions | 2022 | 612 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| interceptions | 2023 | 622 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| interceptions | 2024 | 641 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| interceptions | pooled | 2495 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2021 | 620 | +0.0001 | -0.0002 | +0.0004 | +0.0000 | NULL |
| rushing_yards | 2022 | 612 | -0.0001 | -0.0004 | +0.0002 | +0.0000 | NULL |
| rushing_yards | 2023 | 622 | +0.0003 | -0.0002 | +0.0008 | +0.0000 | NULL |
| rushing_yards | 2024 | 641 | -0.0000 | -0.0002 | +0.0002 | +0.0000 | NULL |
| rushing_yards | pooled | 2495 | +0.0001 | -0.0001 | +0.0003 | +0.0000 | NULL |
| rushing_tds | 2021 | 620 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2022 | 612 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2023 | 622 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2024 | 641 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | pooled | 2495 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2021 | 620 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2022 | 612 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2023 | 622 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2024 | 641 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | pooled | 2495 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |

### RB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| rushing_yards | 2021 | 1092 | +0.0208 | -0.0468 | +0.0927 | -0.0010 | NULL |
| rushing_yards | 2022 | 1032 | +0.0025 | -0.0512 | +0.0592 | +0.0009 | NULL |
| rushing_yards | 2023 | 1015 | -0.0563 | -0.1241 | +0.0125 | +0.0007 | NULL |
| rushing_yards | 2024 | 1068 | -0.0099 | -0.0896 | +0.0818 | +0.0011 | NULL |
| rushing_yards | pooled | 4207 | +0.0109 | -0.0287 | +0.0484 | -0.0010 | NULL |
| rushing_tds | 2021 | 1092 | -0.0000 | -0.0002 | +0.0002 | +0.0004 | NULL |
| rushing_tds | 2022 | 1032 | -0.0002 | -0.0004 | +0.0000 | +0.0004 | NULL |
| rushing_tds | 2023 | 1015 | -0.0002 | -0.0010 | +0.0005 | +0.0009 | NULL |
| rushing_tds | 2024 | 1068 | +0.0001 | -0.0003 | +0.0005 | +0.0005 | NULL |
| rushing_tds | pooled | 4207 | -0.0000 | -0.0002 | +0.0001 | +0.0004 | NULL |
| receptions | 2021 | 1092 | -0.0004 | -0.0017 | +0.0009 | +0.0006 | NULL |
| receptions | 2022 | 1032 | -0.0016 | -0.0037 | +0.0002 | +0.0008 | NULL |
| receptions | 2023 | 1015 | -0.0003 | -0.0028 | +0.0024 | +0.0012 | NULL |
| receptions | 2024 | 1068 | -0.0012 | -0.0038 | +0.0013 | +0.0009 | NULL |
| receptions | pooled | 4207 | -0.0010 | -0.0017 | -0.0002 | +0.0006 | NULL |
| receiving_yards | 2021 | 1092 | +0.0205 | -0.0221 | +0.0610 | +0.0014 | NULL |
| receiving_yards | 2022 | 1032 | -0.0348 | -0.0721 | +0.0025 | +0.0008 | NULL |
| receiving_yards | 2023 | 1015 | +0.0239 | -0.0243 | +0.0713 | +0.0013 | NULL |
| receiving_yards | 2024 | 1068 | +0.0064 | -0.0312 | +0.0392 | +0.0009 | NULL |
| receiving_yards | pooled | 4207 | -0.0026 | -0.0270 | +0.0227 | +0.0014 | NULL |
| receiving_tds | 2021 | 1092 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2022 | 1032 | -0.0002 | -0.0011 | +0.0006 | -0.0026 | NULL |
| receiving_tds | 2023 | 1015 | +0.0005 | -0.0007 | +0.0017 | -0.0045 | NULL |
| receiving_tds | 2024 | 1068 | -0.0003 | -0.0007 | -0.0000 | +0.0002 | NULL |
| receiving_tds | pooled | 4207 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2021 | 1092 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2022 | 1032 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2023 | 1015 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2024 | 1068 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | pooled | 4207 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |

### WR

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 1806 | -0.0006 | -0.0028 | +0.0014 | +0.0003 | NULL |
| receptions | 2022 | 1792 | +0.0024 | +0.0002 | +0.0045 | +0.0004 | NULL |
| receptions | 2023 | 1784 | +0.0006 | -0.0006 | +0.0021 | +0.0001 | NULL |
| receptions | 2024 | 1746 | -0.0002 | -0.0016 | +0.0010 | +0.0001 | NULL |
| receptions | pooled | 7128 | +0.0003 | -0.0009 | +0.0014 | +0.0003 | NULL |
| receiving_yards | 2021 | 1806 | -0.0008 | -0.0204 | +0.0179 | -0.0002 | NULL |
| receiving_yards | 2022 | 1792 | -0.0001 | -0.0220 | +0.0209 | -0.0002 | NULL |
| receiving_yards | 2023 | 1784 | +0.0121 | -0.0059 | +0.0295 | -0.0001 | NULL |
| receiving_yards | 2024 | 1746 | +0.0052 | -0.0092 | +0.0213 | -0.0001 | NULL |
| receiving_yards | pooled | 7128 | -0.0001 | -0.0104 | +0.0109 | -0.0002 | NULL |
| receiving_tds | 2021 | 1806 | +0.0000 | -0.0001 | +0.0001 | +0.0001 | NULL |
| receiving_tds | 2022 | 1792 | -0.0000 | -0.0001 | +0.0000 | +0.0001 | NULL |
| receiving_tds | 2023 | 1784 | +0.0001 | -0.0000 | +0.0001 | +0.0001 | NULL |
| receiving_tds | 2024 | 1746 | +0.0000 | -0.0000 | +0.0001 | +0.0001 | NULL |
| receiving_tds | pooled | 7128 | +0.0000 | -0.0000 | +0.0000 | +0.0001 | NULL |
| rushing_yards | 2021 | 1806 | -0.0003 | -0.0007 | +0.0001 | +0.0001 | NULL |
| rushing_yards | 2022 | 1792 | -0.0002 | -0.0013 | +0.0011 | +0.0002 | NULL |
| rushing_yards | 2023 | 1784 | -0.0010 | -0.0022 | +0.0003 | +0.0002 | NULL |
| rushing_yards | 2024 | 1746 | -0.0014 | -0.0029 | +0.0001 | +0.0003 | NULL |
| rushing_yards | pooled | 7128 | -0.0004 | -0.0006 | -0.0001 | +0.0001 | NULL |
| rushing_tds | 2021 | 1806 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2022 | 1792 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2023 | 1784 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2024 | 1746 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | pooled | 7128 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2021 | 1806 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2022 | 1792 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2023 | 1784 | -0.0002 | -0.0005 | +0.0001 | -0.0023 | NULL |
| fumbles_lost | 2024 | 1746 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | pooled | 7128 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |

### TE

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 732 | -0.0001 | -0.0006 | +0.0003 | +0.0002 | NULL |
| receptions | 2022 | 797 | -0.0009 | -0.0046 | +0.0025 | +0.0000 | NULL |
| receptions | 2023 | 809 | +0.0013 | -0.0034 | +0.0055 | +0.0011 | NULL |
| receptions | 2024 | 786 | -0.0011 | -0.0045 | +0.0024 | +0.0002 | NULL |
| receptions | pooled | 3124 | -0.0002 | -0.0004 | +0.0000 | +0.0002 | NULL |
| receiving_yards | 2021 | 732 | +0.0796 | -0.0698 | +0.2207 | -0.0059 | NULL |
| receiving_yards | 2022 | 797 | +0.0174 | -0.0333 | +0.0657 | +0.0000 | NULL |
| receiving_yards | 2023 | 809 | +0.0122 | -0.0473 | +0.0694 | +0.0009 | NULL |
| receiving_yards | 2024 | 786 | -0.0302 | -0.0780 | +0.0125 | +0.0006 | NULL |
| receiving_yards | pooled | 3124 | +0.0590 | -0.0215 | +0.1340 | -0.0059 | NULL |
| receiving_tds | 2021 | 732 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2022 | 797 | -0.0000 | -0.0000 | -0.0000 | +0.0000 | NULL |
| receiving_tds | 2023 | 809 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2024 | 786 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | pooled | 3124 | -0.0000 | -0.0000 | -0.0000 | +0.0000 | NULL |
| rushing_yards | 2021 | 732 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2022 | 797 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2023 | 809 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2024 | 786 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | pooled | 3124 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2021 | 732 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2022 | 797 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2023 | 809 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2024 | 786 | -0.0000 | -0.0000 | -0.0000 | +0.0000 | NULL |
| rushing_tds | pooled | 3124 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2021 | 732 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2022 | 797 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2023 | 809 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2024 | 786 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | pooled | 3124 | -0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |

## Phase 1 verdict

0/120 cells SIGNAL, 120/120 NULL, 0/120 REGRESSION.
No SIGNAL cells — Phase 2 skipped. Probe predicts the adoption gate would return DO_NOT_ADOPT.

## Probe verdict

Phase 1: 0/120 cells SIGNAL.
Phase 2: not run.

**This probe is a screen, not the gate.** The adoption gate is the final word on whether a feature change ships; SIGNAL is necessary but not sufficient. Run the full backtest + adoption gate before any production routing change.
