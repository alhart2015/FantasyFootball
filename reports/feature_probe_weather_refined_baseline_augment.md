# Feature signal probe — weather_refined_baseline_augment

Baseline features: data\features
Overrides:        data\features_probe\weather_refined_only.parquet
Drops:            (none)
Model class:      baseline

## Phase 1 — per-stat screening

### QB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| passing_yards | 2021 | 535 | -0.2021 | -0.5174 | +0.1292 | +0.0015 | NULL |
| passing_yards | 2022 | 370 | +0.3899 | -0.2305 | +1.0536 | +0.0035 | NULL |
| passing_yards | 2023 | 481 | -0.1796 | -0.6519 | +0.2998 | +0.0017 | NULL |
| passing_yards | 2024 | 562 | -0.0665 | -0.4710 | +0.3283 | +0.0023 | NULL |
| passing_yards | pooled | 1948 | -0.0144 | -0.1855 | +0.1680 | +0.0015 | NULL |
| passing_tds | 2021 | 535 | +0.0031 | -0.0028 | +0.0089 | -0.0019 | NULL |
| passing_tds | 2022 | 370 | +0.0079 | -0.0011 | +0.0161 | -0.0028 | NULL |
| passing_tds | 2023 | 481 | +0.0002 | -0.0039 | +0.0045 | +0.0005 | NULL |
| passing_tds | 2024 | 562 | +0.0028 | -0.0005 | +0.0060 | +0.0005 | NULL |
| passing_tds | pooled | 1948 | +0.0016 | -0.0015 | +0.0049 | -0.0019 | NULL |
| interceptions | 2021 | 535 | +0.0001 | -0.0030 | +0.0035 | -0.0014 | NULL |
| interceptions | 2022 | 370 | -0.0006 | -0.0038 | +0.0024 | -0.0006 | NULL |
| interceptions | 2023 | 481 | +0.0008 | -0.0009 | +0.0025 | +0.0014 | NULL |
| interceptions | 2024 | 562 | -0.0002 | -0.0020 | +0.0015 | -0.0013 | NULL |
| interceptions | pooled | 1948 | -0.0006 | -0.0022 | +0.0011 | -0.0014 | NULL |
| rushing_yards | 2021 | 535 | +0.0144 | -0.0651 | +0.0953 | +0.0010 | NULL |
| rushing_yards | 2022 | 370 | -0.0064 | -0.0857 | +0.0645 | +0.0012 | NULL |
| rushing_yards | 2023 | 481 | +0.0196 | -0.0349 | +0.0748 | +0.0022 | NULL |
| rushing_yards | 2024 | 562 | +0.0579 | +0.0118 | +0.1070 | +0.0018 | REGRESSION |
| rushing_yards | pooled | 1948 | -0.0018 | -0.0404 | +0.0388 | +0.0010 | NULL |
| rushing_tds | 2021 | 535 | -0.0003 | -0.0010 | +0.0003 | +0.0014 | NULL |
| rushing_tds | 2022 | 370 | +0.0008 | -0.0003 | +0.0018 | +0.0023 | NULL |
| rushing_tds | 2023 | 481 | -0.0002 | -0.0005 | +0.0001 | +0.0006 | NULL |
| rushing_tds | 2024 | 562 | -0.0001 | -0.0010 | +0.0007 | +0.0020 | NULL |
| rushing_tds | pooled | 1948 | -0.0001 | -0.0004 | +0.0002 | +0.0014 | NULL |
| fumbles_lost | 2021 | 535 | -0.0005 | -0.0008 | -0.0001 | +0.0012 | NULL |
| fumbles_lost | 2022 | 370 | +0.0007 | -0.0001 | +0.0016 | +0.0022 | NULL |
| fumbles_lost | 2023 | 481 | -0.0002 | -0.0007 | +0.0003 | +0.0013 | NULL |
| fumbles_lost | 2024 | 562 | +0.0002 | -0.0004 | +0.0008 | +0.0014 | NULL |
| fumbles_lost | pooled | 1948 | -0.0001 | -0.0003 | +0.0001 | +0.0012 | NULL |

### RB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| rushing_yards | 2021 | 933 | +0.0997 | -0.0948 | +0.3043 | +0.0036 | NULL |
| rushing_yards | 2022 | 635 | +0.0513 | -0.1217 | +0.2292 | +0.0020 | NULL |
| rushing_yards | 2023 | 789 | -0.0179 | -0.1298 | +0.1075 | +0.0016 | NULL |
| rushing_yards | 2024 | 927 | +0.0189 | -0.0848 | +0.1299 | +0.0016 | NULL |
| rushing_yards | pooled | 3284 | +0.1059 | +0.0032 | +0.2181 | +0.0036 | REGRESSION |
| rushing_tds | 2021 | 933 | -0.0002 | -0.0031 | +0.0026 | +0.0023 | NULL |
| rushing_tds | 2022 | 635 | +0.0035 | -0.0000 | +0.0071 | +0.0045 | NULL |
| rushing_tds | 2023 | 789 | -0.0011 | -0.0029 | +0.0007 | +0.0028 | NULL |
| rushing_tds | 2024 | 927 | +0.0007 | -0.0010 | +0.0026 | +0.0023 | NULL |
| rushing_tds | pooled | 3284 | +0.0014 | -0.0002 | +0.0030 | +0.0023 | NULL |
| receptions | 2021 | 933 | +0.0075 | -0.0019 | +0.0174 | +0.0022 | NULL |
| receptions | 2022 | 635 | +0.0047 | -0.0059 | +0.0154 | +0.0033 | NULL |
| receptions | 2023 | 789 | -0.0002 | -0.0087 | +0.0090 | +0.0026 | NULL |
| receptions | 2024 | 927 | +0.0033 | -0.0043 | +0.0116 | +0.0023 | NULL |
| receptions | pooled | 3284 | +0.0085 | +0.0031 | +0.0137 | +0.0022 | NULL |
| receiving_yards | 2021 | 933 | -0.0462 | -0.2170 | +0.1230 | -0.0098 | NULL |
| receiving_yards | 2022 | 635 | +0.0139 | -0.0417 | +0.0712 | +0.0017 | NULL |
| receiving_yards | 2023 | 789 | -0.0001 | -0.0627 | +0.0675 | +0.0016 | NULL |
| receiving_yards | 2024 | 927 | +0.0388 | -0.0130 | +0.0935 | +0.0014 | NULL |
| receiving_yards | pooled | 3284 | -0.0416 | -0.1182 | +0.0356 | -0.0098 | NULL |
| receiving_tds | 2021 | 933 | -0.0000 | -0.0003 | +0.0002 | +0.0010 | NULL |
| receiving_tds | 2022 | 635 | +0.0002 | -0.0001 | +0.0005 | +0.0011 | NULL |
| receiving_tds | 2023 | 789 | -0.0000 | -0.0020 | +0.0020 | -0.0054 | NULL |
| receiving_tds | 2024 | 927 | -0.0012 | -0.0026 | +0.0002 | -0.0046 | NULL |
| receiving_tds | pooled | 3284 | +0.0001 | -0.0001 | +0.0002 | +0.0010 | NULL |
| fumbles_lost | 2021 | 933 | -0.0001 | -0.0003 | +0.0001 | +0.0011 | NULL |
| fumbles_lost | 2022 | 635 | +0.0001 | -0.0002 | +0.0004 | +0.0012 | NULL |
| fumbles_lost | 2023 | 789 | +0.0002 | -0.0001 | +0.0005 | +0.0011 | NULL |
| fumbles_lost | 2024 | 927 | +0.0001 | -0.0001 | +0.0003 | +0.0006 | NULL |
| fumbles_lost | pooled | 3284 | +0.0000 | -0.0001 | +0.0001 | +0.0011 | NULL |

### WR

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 1401 | -0.0018 | -0.0054 | +0.0017 | +0.0005 | NULL |
| receptions | 2022 | 1062 | +0.0043 | -0.0023 | +0.0112 | +0.0013 | NULL |
| receptions | 2023 | 1293 | +0.0017 | -0.0025 | +0.0058 | +0.0009 | NULL |
| receptions | 2024 | 1424 | -0.0023 | -0.0054 | +0.0011 | +0.0007 | NULL |
| receptions | pooled | 5180 | -0.0003 | -0.0021 | +0.0014 | +0.0005 | NULL |
| receiving_yards | 2021 | 1401 | -0.0300 | -0.1589 | +0.0999 | +0.0030 | NULL |
| receiving_yards | 2022 | 1062 | +0.0428 | -0.0871 | +0.1736 | +0.0036 | NULL |
| receiving_yards | 2023 | 1293 | +0.0215 | -0.0735 | +0.1148 | +0.0026 | NULL |
| receiving_yards | 2024 | 1424 | +0.0104 | -0.0809 | +0.1042 | +0.0024 | NULL |
| receiving_yards | pooled | 5180 | +0.0167 | -0.0552 | +0.0798 | +0.0030 | NULL |
| receiving_tds | 2021 | 1401 | +0.0002 | -0.0013 | +0.0015 | -0.0001 | NULL |
| receiving_tds | 2022 | 1062 | +0.0002 | -0.0009 | +0.0013 | -0.0003 | NULL |
| receiving_tds | 2023 | 1293 | -0.0001 | -0.0013 | +0.0012 | -0.0028 | NULL |
| receiving_tds | 2024 | 1424 | +0.0010 | -0.0002 | +0.0021 | -0.0024 | NULL |
| receiving_tds | pooled | 5180 | +0.0004 | -0.0003 | +0.0010 | -0.0001 | NULL |
| rushing_yards | 2021 | 1401 | +0.0035 | -0.0043 | +0.0104 | +0.0009 | NULL |
| rushing_yards | 2022 | 1062 | +0.0052 | -0.0002 | +0.0107 | +0.0009 | NULL |
| rushing_yards | 2023 | 1293 | -0.0003 | -0.0067 | +0.0049 | +0.0006 | NULL |
| rushing_yards | 2024 | 1424 | -0.0114 | -0.0307 | +0.0049 | -0.0003 | NULL |
| rushing_yards | pooled | 5180 | +0.0043 | +0.0008 | +0.0077 | +0.0009 | NULL |
| rushing_tds | 2021 | 1401 | +0.0000 | -0.0000 | +0.0000 | +0.0001 | NULL |
| rushing_tds | 2022 | 1062 | +0.0002 | -0.0003 | +0.0006 | -0.0019 | NULL |
| rushing_tds | 2023 | 1293 | -0.0000 | -0.0001 | +0.0001 | +0.0007 | NULL |
| rushing_tds | 2024 | 1424 | -0.0002 | -0.0005 | -0.0000 | -0.0008 | NULL |
| rushing_tds | pooled | 5180 | +0.0000 | -0.0000 | +0.0000 | +0.0001 | NULL |
| fumbles_lost | 2021 | 1401 | +0.0001 | +0.0001 | +0.0002 | +0.0008 | NULL |
| fumbles_lost | 2022 | 1062 | -0.0000 | -0.0001 | +0.0001 | +0.0003 | NULL |
| fumbles_lost | 2023 | 1293 | +0.0000 | -0.0000 | +0.0001 | +0.0003 | NULL |
| fumbles_lost | 2024 | 1424 | +0.0000 | -0.0000 | +0.0001 | +0.0004 | NULL |
| fumbles_lost | pooled | 5180 | +0.0000 | -0.0000 | +0.0001 | +0.0008 | NULL |

### TE

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 574 | +0.0045 | +0.0002 | +0.0087 | +0.0013 | NULL |
| receptions | 2022 | 442 | +0.0098 | +0.0004 | +0.0205 | +0.0016 | NULL |
| receptions | 2023 | 586 | -0.0008 | -0.0063 | +0.0047 | -0.0002 | NULL |
| receptions | 2024 | 662 | +0.0029 | -0.0026 | +0.0087 | +0.0012 | NULL |
| receptions | pooled | 2264 | +0.0012 | -0.0007 | +0.0031 | +0.0013 | NULL |
| receiving_yards | 2021 | 574 | +0.0669 | +0.0031 | +0.1340 | +0.0016 | REGRESSION |
| receiving_yards | 2022 | 442 | +0.0575 | -0.0157 | +0.1197 | -0.0019 | NULL |
| receiving_yards | 2023 | 586 | -0.0224 | -0.1882 | +0.1436 | -0.0047 | NULL |
| receiving_yards | 2024 | 662 | +0.0441 | -0.0384 | +0.1250 | +0.0016 | NULL |
| receiving_yards | pooled | 2264 | +0.0252 | -0.0057 | +0.0568 | +0.0016 | NULL |
| receiving_tds | 2021 | 574 | +0.0004 | -0.0000 | +0.0008 | +0.0010 | NULL |
| receiving_tds | 2022 | 442 | -0.0001 | -0.0004 | +0.0003 | +0.0007 | NULL |
| receiving_tds | 2023 | 586 | -0.0001 | -0.0006 | +0.0004 | +0.0007 | NULL |
| receiving_tds | 2024 | 662 | +0.0004 | -0.0000 | +0.0009 | +0.0008 | NULL |
| receiving_tds | pooled | 2264 | +0.0001 | -0.0001 | +0.0003 | +0.0010 | NULL |
| rushing_yards | 2021 | 574 | -0.0005 | -0.0014 | +0.0005 | +0.0004 | NULL |
| rushing_yards | 2022 | 442 | +0.0013 | +0.0001 | +0.0022 | +0.0005 | NULL |
| rushing_yards | 2023 | 586 | +0.0003 | -0.0004 | +0.0010 | +0.0003 | NULL |
| rushing_yards | 2024 | 662 | +0.0002 | -0.0005 | +0.0012 | +0.0003 | NULL |
| rushing_yards | pooled | 2264 | +0.0005 | -0.0000 | +0.0010 | +0.0004 | NULL |
| rushing_tds | 2021 | 574 | +0.0000 | +0.0000 | +0.0002 | +0.0009 | NULL |
| rushing_tds | 2022 | 442 | +0.0000 | -0.0000 | +0.0001 | +0.0004 | NULL |
| rushing_tds | 2023 | 586 | +0.0000 | -0.0000 | +0.0000 | +0.0003 | NULL |
| rushing_tds | 2024 | 662 | +0.0000 | +0.0000 | +0.0001 | +0.0003 | NULL |
| rushing_tds | pooled | 2264 | +0.0000 | +0.0000 | +0.0001 | +0.0009 | NULL |
| fumbles_lost | 2021 | 574 | -0.0000 | -0.0000 | +0.0000 | +0.0001 | NULL |
| fumbles_lost | 2022 | 442 | -0.0000 | -0.0002 | +0.0001 | +0.0004 | NULL |
| fumbles_lost | 2023 | 586 | +0.0001 | -0.0000 | +0.0002 | +0.0006 | NULL |
| fumbles_lost | 2024 | 662 | +0.0000 | -0.0001 | +0.0001 | +0.0004 | NULL |
| fumbles_lost | pooled | 2264 | +0.0000 | -0.0000 | +0.0000 | +0.0001 | NULL |

## Phase 1 verdict

0/120 cells SIGNAL, 117/120 NULL, 3/120 REGRESSION.
No SIGNAL cells — Phase 2 skipped. Probe predicts the adoption gate would return DO_NOT_ADOPT.

## Probe verdict

Phase 1: 0/120 cells SIGNAL.
Phase 2: not run.

**This probe is a screen, not the gate.** The adoption gate is the final word on whether a feature change ships; SIGNAL is necessary but not sufficient. Run the full backtest + adoption gate before any production routing change.
