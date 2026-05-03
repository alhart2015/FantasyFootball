# Feature signal probe — trajectory_baseline_augment

Baseline features: C:\Users\alden\FantasyFootball\data\features
Overrides:        C:\Users\alden\FantasyFootball\data\features_probe\trajectory_probe.parquet
Drops:            (none)
Model class:      baseline

## Phase 1 — per-stat screening

### QB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| passing_yards | 2021 | 493 | -0.2417 | -0.6025 | +0.1133 | -0.0012 | NULL |
| passing_yards | 2022 | 499 | -0.1387 | -0.4431 | +0.1618 | -0.0001 | NULL |
| passing_yards | 2023 | 489 | +0.1574 | -0.0894 | +0.4080 | -0.0001 | NULL |
| passing_yards | 2024 | 529 | +0.0260 | -0.1521 | +0.2009 | -0.0000 | NULL |
| passing_yards | pooled | 2010 | -0.0843 | -0.2762 | +0.1073 | -0.0012 | NULL |
| passing_tds | 2021 | 493 | -0.0021 | -0.0082 | +0.0042 | +0.0037 | NULL |
| passing_tds | 2022 | 499 | -0.0004 | -0.0068 | +0.0067 | +0.0022 | NULL |
| passing_tds | 2023 | 489 | -0.0071 | -0.0141 | -0.0009 | +0.0022 | NULL |
| passing_tds | 2024 | 529 | +0.0062 | +0.0006 | +0.0121 | +0.0037 | NULL |
| passing_tds | pooled | 2010 | +0.0000 | -0.0030 | +0.0030 | +0.0037 | NULL |
| interceptions | 2021 | 493 | +0.0014 | -0.0022 | +0.0049 | +0.0023 | NULL |
| interceptions | 2022 | 499 | +0.0008 | -0.0019 | +0.0037 | +0.0019 | NULL |
| interceptions | 2023 | 489 | +0.0007 | -0.0006 | +0.0020 | +0.0006 | NULL |
| interceptions | 2024 | 529 | -0.0002 | -0.0010 | +0.0005 | +0.0004 | NULL |
| interceptions | pooled | 2010 | +0.0017 | +0.0000 | +0.0032 | +0.0023 | NULL |
| rushing_yards | 2021 | 493 | -0.0733 | -0.1980 | +0.0656 | +0.0070 | NULL |
| rushing_yards | 2022 | 499 | +0.0237 | -0.1066 | +0.1494 | +0.0061 | NULL |
| rushing_yards | 2023 | 489 | -0.1039 | -0.2272 | +0.0310 | +0.0028 | NULL |
| rushing_yards | 2024 | 529 | -0.1270 | -0.2492 | -0.0170 | +0.0046 | SIGNAL |
| rushing_yards | pooled | 2010 | -0.0450 | -0.1142 | +0.0243 | +0.0070 | NULL |
| rushing_tds | 2021 | 493 | -0.0003 | -0.0013 | +0.0007 | +0.0005 | NULL |
| rushing_tds | 2022 | 499 | +0.0006 | -0.0008 | +0.0019 | +0.0012 | NULL |
| rushing_tds | 2023 | 489 | +0.0002 | -0.0006 | +0.0010 | +0.0006 | NULL |
| rushing_tds | 2024 | 529 | -0.0000 | -0.0005 | +0.0004 | +0.0004 | NULL |
| rushing_tds | pooled | 2010 | +0.0001 | -0.0004 | +0.0005 | +0.0005 | NULL |
| fumbles_lost | 2021 | 493 | +0.0045 | +0.0008 | +0.0082 | +0.0113 | NULL |
| fumbles_lost | 2022 | 499 | +0.0008 | -0.0016 | +0.0038 | +0.0042 | NULL |
| fumbles_lost | 2023 | 489 | -0.0000 | -0.0015 | +0.0016 | +0.0021 | NULL |
| fumbles_lost | 2024 | 529 | +0.0001 | -0.0012 | +0.0014 | +0.0017 | NULL |
| fumbles_lost | pooled | 2010 | +0.0021 | +0.0002 | +0.0039 | +0.0113 | NULL |

### RB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| rushing_yards | 2021 | 892 | +0.1226 | -0.0943 | +0.3470 | +0.0051 | NULL |
| rushing_yards | 2022 | 840 | -0.3625 | -0.5860 | -0.1221 | +0.0037 | SIGNAL |
| rushing_yards | 2023 | 864 | -0.1171 | -0.3121 | +0.0899 | +0.0069 | NULL |
| rushing_yards | 2024 | 897 | -0.2237 | -0.4204 | -0.0374 | +0.0064 | SIGNAL |
| rushing_yards | pooled | 3493 | -0.1323 | -0.2376 | -0.0154 | +0.0051 | SIGNAL |
| rushing_tds | 2021 | 892 | -0.0008 | -0.0038 | +0.0019 | +0.0022 | NULL |
| rushing_tds | 2022 | 840 | -0.0012 | -0.0040 | +0.0016 | +0.0039 | NULL |
| rushing_tds | 2023 | 864 | -0.0007 | -0.0035 | +0.0019 | +0.0041 | NULL |
| rushing_tds | 2024 | 897 | -0.0022 | -0.0045 | +0.0003 | +0.0035 | NULL |
| rushing_tds | pooled | 3493 | -0.0020 | -0.0033 | -0.0006 | +0.0022 | NULL |
| receptions | 2021 | 892 | +0.0032 | -0.0047 | +0.0112 | +0.0029 | NULL |
| receptions | 2022 | 840 | +0.0028 | -0.0030 | +0.0083 | +0.0017 | NULL |
| receptions | 2023 | 864 | -0.0015 | -0.0058 | +0.0023 | +0.0011 | NULL |
| receptions | 2024 | 897 | -0.0020 | -0.0063 | +0.0022 | +0.0012 | NULL |
| receptions | pooled | 3493 | +0.0007 | -0.0031 | +0.0044 | +0.0029 | NULL |
| receiving_yards | 2021 | 892 | +0.0250 | -0.0386 | +0.0923 | +0.0030 | NULL |
| receiving_yards | 2022 | 840 | +0.0206 | -0.0331 | +0.0689 | +0.0017 | NULL |
| receiving_yards | 2023 | 864 | -0.0390 | -0.0728 | -0.0050 | +0.0011 | NULL |
| receiving_yards | 2024 | 897 | +0.0469 | +0.0019 | +0.0959 | +0.0016 | NULL |
| receiving_yards | pooled | 3493 | +0.0207 | -0.0141 | +0.0540 | +0.0030 | NULL |
| receiving_tds | 2021 | 892 | +0.0008 | +0.0001 | +0.0014 | +0.0012 | NULL |
| receiving_tds | 2022 | 840 | +0.0002 | +0.0000 | +0.0003 | +0.0001 | NULL |
| receiving_tds | 2023 | 864 | +0.0000 | -0.0003 | +0.0003 | +0.0004 | NULL |
| receiving_tds | 2024 | 897 | -0.0006 | -0.0017 | +0.0007 | -0.0036 | NULL |
| receiving_tds | pooled | 3493 | +0.0002 | -0.0001 | +0.0005 | +0.0012 | NULL |
| fumbles_lost | 2021 | 892 | -0.0000 | -0.0004 | +0.0004 | +0.0014 | NULL |
| fumbles_lost | 2022 | 840 | +0.0007 | +0.0002 | +0.0012 | +0.0011 | NULL |
| fumbles_lost | 2023 | 864 | +0.0000 | -0.0002 | +0.0003 | +0.0005 | NULL |
| fumbles_lost | 2024 | 897 | -0.0000 | -0.0002 | +0.0002 | +0.0003 | NULL |
| fumbles_lost | pooled | 3493 | +0.0004 | +0.0001 | +0.0006 | +0.0014 | NULL |

### WR

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 1518 | -0.0177 | -0.0286 | -0.0067 | +0.0044 | NULL |
| receptions | 2022 | 1530 | -0.0185 | -0.0311 | -0.0051 | +0.0080 | NULL |
| receptions | 2023 | 1489 | -0.0171 | -0.0294 | -0.0039 | +0.0089 | NULL |
| receptions | 2024 | 1467 | -0.0084 | -0.0217 | +0.0058 | +0.0092 | NULL |
| receptions | pooled | 6004 | -0.0139 | -0.0202 | -0.0078 | +0.0044 | NULL |
| receiving_yards | 2021 | 1518 | -0.2167 | -0.3853 | -0.0405 | +0.0070 | SIGNAL |
| receiving_yards | 2022 | 1530 | -0.1469 | -0.3406 | +0.0458 | +0.0083 | NULL |
| receiving_yards | 2023 | 1489 | -0.1453 | -0.3055 | +0.0215 | +0.0080 | NULL |
| receiving_yards | 2024 | 1467 | -0.0500 | -0.2135 | +0.1253 | +0.0076 | NULL |
| receiving_yards | pooled | 6004 | -0.1424 | -0.2303 | -0.0542 | +0.0070 | SIGNAL |
| receiving_tds | 2021 | 1518 | -0.0001 | -0.0012 | +0.0010 | +0.0019 | NULL |
| receiving_tds | 2022 | 1530 | -0.0005 | -0.0017 | +0.0007 | +0.0010 | NULL |
| receiving_tds | 2023 | 1489 | -0.0001 | -0.0011 | +0.0010 | +0.0016 | NULL |
| receiving_tds | 2024 | 1467 | +0.0003 | -0.0009 | +0.0015 | +0.0012 | NULL |
| receiving_tds | pooled | 6004 | +0.0000 | -0.0006 | +0.0007 | +0.0019 | NULL |
| rushing_yards | 2021 | 1518 | +0.0060 | -0.0030 | +0.0160 | +0.0025 | NULL |
| rushing_yards | 2022 | 1530 | -0.0067 | -0.0157 | +0.0020 | +0.0015 | NULL |
| rushing_yards | 2023 | 1489 | -0.0069 | -0.0203 | +0.0069 | +0.0019 | NULL |
| rushing_yards | 2024 | 1467 | -0.0010 | -0.0125 | +0.0110 | +0.0021 | NULL |
| rushing_yards | pooled | 6004 | -0.0013 | -0.0070 | +0.0047 | +0.0025 | NULL |
| rushing_tds | 2021 | 1518 | +0.0001 | -0.0000 | +0.0002 | +0.0008 | NULL |
| rushing_tds | 2022 | 1530 | -0.0000 | -0.0001 | +0.0000 | +0.0003 | NULL |
| rushing_tds | 2023 | 1489 | -0.0001 | -0.0005 | +0.0003 | +0.0018 | NULL |
| rushing_tds | 2024 | 1467 | -0.0000 | -0.0001 | +0.0000 | +0.0003 | NULL |
| rushing_tds | pooled | 6004 | +0.0000 | -0.0000 | +0.0001 | +0.0008 | NULL |
| fumbles_lost | 2021 | 1518 | -0.0001 | -0.0002 | +0.0000 | +0.0005 | NULL |
| fumbles_lost | 2022 | 1530 | -0.0001 | -0.0002 | +0.0000 | +0.0008 | NULL |
| fumbles_lost | 2023 | 1489 | +0.0000 | -0.0002 | +0.0002 | +0.0010 | NULL |
| fumbles_lost | 2024 | 1467 | +0.0001 | -0.0001 | +0.0002 | +0.0007 | NULL |
| fumbles_lost | pooled | 6004 | -0.0000 | -0.0001 | +0.0000 | +0.0005 | NULL |

### TE

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 610 | +0.0034 | -0.0116 | +0.0192 | -0.0007 | NULL |
| receptions | 2022 | 652 | -0.0084 | -0.0210 | +0.0046 | +0.0052 | NULL |
| receptions | 2023 | 682 | -0.0182 | -0.0316 | -0.0057 | +0.0064 | NULL |
| receptions | 2024 | 670 | -0.0204 | -0.0380 | -0.0037 | +0.0078 | NULL |
| receptions | pooled | 2614 | -0.0060 | -0.0130 | +0.0015 | -0.0007 | NULL |
| receiving_yards | 2021 | 610 | -0.0463 | -0.1697 | +0.0682 | +0.0002 | NULL |
| receiving_yards | 2022 | 652 | +0.0689 | -0.0992 | +0.2438 | -0.0026 | NULL |
| receiving_yards | 2023 | 682 | -0.1601 | -0.3014 | -0.0204 | +0.0040 | SIGNAL |
| receiving_yards | 2024 | 670 | -0.1620 | -0.3313 | +0.0037 | +0.0050 | NULL |
| receiving_yards | pooled | 2614 | -0.0814 | -0.1360 | -0.0236 | +0.0002 | SIGNAL |
| receiving_tds | 2021 | 610 | +0.0002 | -0.0003 | +0.0006 | +0.0003 | NULL |
| receiving_tds | 2022 | 652 | -0.0001 | -0.0005 | +0.0002 | +0.0002 | NULL |
| receiving_tds | 2023 | 682 | -0.0002 | -0.0007 | +0.0003 | +0.0003 | NULL |
| receiving_tds | 2024 | 670 | +0.0008 | +0.0001 | +0.0014 | +0.0004 | NULL |
| receiving_tds | pooled | 2614 | +0.0001 | -0.0001 | +0.0003 | +0.0003 | NULL |
| rushing_yards | 2021 | 610 | +0.0025 | +0.0008 | +0.0041 | +0.0009 | NULL |
| rushing_yards | 2022 | 652 | -0.0007 | -0.0025 | +0.0010 | +0.0004 | NULL |
| rushing_yards | 2023 | 682 | +0.0019 | -0.0008 | +0.0045 | +0.0005 | NULL |
| rushing_yards | 2024 | 670 | -0.0011 | -0.0026 | +0.0012 | +0.0004 | NULL |
| rushing_yards | pooled | 2614 | +0.0015 | -0.0001 | +0.0035 | +0.0009 | NULL |
| rushing_tds | 2021 | 610 | -0.0001 | -0.0002 | +0.0007 | +0.0024 | NULL |
| rushing_tds | 2022 | 652 | +0.0001 | +0.0001 | +0.0008 | +0.0028 | NULL |
| rushing_tds | 2023 | 682 | +0.0004 | +0.0003 | +0.0006 | +0.0017 | NULL |
| rushing_tds | 2024 | 670 | +0.0001 | +0.0001 | +0.0007 | +0.0014 | NULL |
| rushing_tds | pooled | 2614 | +0.0000 | -0.0000 | +0.0002 | +0.0024 | NULL |
| fumbles_lost | 2021 | 610 | +0.0004 | +0.0001 | +0.0007 | +0.0033 | NULL |
| fumbles_lost | 2022 | 652 | +0.0001 | -0.0002 | +0.0003 | +0.0011 | NULL |
| fumbles_lost | 2023 | 682 | +0.0002 | +0.0000 | +0.0003 | +0.0009 | NULL |
| fumbles_lost | 2024 | 670 | +0.0000 | -0.0001 | +0.0001 | +0.0004 | NULL |
| fumbles_lost | pooled | 2614 | +0.0003 | +0.0001 | +0.0005 | +0.0033 | NULL |

## Phase 1 verdict

8/120 cells SIGNAL, 112/120 NULL, 0/120 REGRESSION.
Phase 2 fired.

## Phase 2 — composite ΔRMSE

| Position | Verdict | RMSE delta (95% CI) | Spearman delta (95% CI) | n_paired |
|---|---|---|---|---:|
| QB | DO_NOT_ADOPT | +0.0382 ([+0.0155, +0.0600]) | -0.0063 ([-0.0104, -0.0025]) | 2692 |
| RB | DO_NOT_ADOPT | -0.0061 ([-0.0202, +0.0086]) | -0.0012 ([-0.0038, +0.0015]) | 5273 |
| WR | ADOPT | -0.0414 ([-0.0606, -0.0230]) | +0.0058 ([+0.0026, +0.0092]) | 8470 |
| TE | DO_NOT_ADOPT | -0.0097 ([-0.0288, +0.0095]) | +0.0015 ([-0.0039, +0.0064]) | 4257 |

## Probe verdict

Phase 1: 8/120 cells SIGNAL.
Phase 2: 1/4 positions ADOPT.

**This probe is a screen, not the gate.** The adoption gate is the final word on whether a feature change ships; SIGNAL is necessary but not sufficient. Run the full backtest + adoption gate before any production routing change.
