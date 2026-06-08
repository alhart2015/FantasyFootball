# Feature signal probe — vegas_team_context_lgbnb_swap

Baseline features: data\features
Overrides:        data\features_probe\vegas_team_context.parquet
Drops:            implied_team_total, spread
Model class:      lightgbm-nb

## Phase 1 — per-stat screening

### QB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| passing_yards | 2021 | 556 | +0.3877 | -0.0181 | +0.8265 | +0.0020 | NULL |
| passing_yards | 2022 | 544 | -0.0121 | -0.2199 | +0.1670 | +0.0006 | NULL |
| passing_yards | 2023 | 552 | +0.2691 | -0.0337 | +0.5345 | +0.0009 | NULL |
| passing_yards | 2024 | 571 | +0.0834 | -0.0840 | +0.2438 | +0.0004 | NULL |
| passing_yards | pooled | 2223 | +0.2232 | +0.0369 | +0.4034 | +0.0020 | REGRESSION |
| passing_tds | 2021 | 556 | -0.0010 | -0.0107 | +0.0099 | +0.0070 | NULL |
| passing_tds | 2022 | 544 | -0.0083 | -0.0160 | -0.0005 | +0.0070 | NULL |
| passing_tds | 2023 | 552 | +0.0029 | -0.0064 | +0.0120 | +0.0077 | NULL |
| passing_tds | 2024 | 571 | -0.0014 | -0.0086 | +0.0057 | +0.0064 | NULL |
| passing_tds | pooled | 2223 | +0.0016 | -0.0028 | +0.0061 | +0.0070 | NULL |
| interceptions | 2021 | 556 | +0.0015 | -0.0077 | +0.0106 | +0.0058 | NULL |
| interceptions | 2022 | 544 | -0.0064 | -0.0135 | +0.0009 | +0.0073 | NULL |
| interceptions | 2023 | 552 | -0.0017 | -0.0084 | +0.0050 | +0.0080 | NULL |
| interceptions | 2024 | 571 | -0.0016 | -0.0075 | +0.0036 | +0.0089 | NULL |
| interceptions | pooled | 2223 | -0.0025 | -0.0068 | +0.0019 | +0.0058 | NULL |
| rushing_yards | 2021 | 556 | +0.0717 | -0.0222 | +0.1619 | +0.0020 | NULL |
| rushing_yards | 2022 | 544 | -0.0032 | -0.0618 | +0.0574 | +0.0009 | NULL |
| rushing_yards | 2023 | 552 | +0.0327 | -0.0219 | +0.0934 | +0.0008 | NULL |
| rushing_yards | 2024 | 571 | -0.0091 | -0.0476 | +0.0285 | +0.0005 | NULL |
| rushing_yards | pooled | 2223 | +0.0293 | -0.0137 | +0.0679 | +0.0020 | NULL |
| rushing_tds | 2021 | 556 | +0.0010 | -0.0031 | +0.0046 | +0.0072 | NULL |
| rushing_tds | 2022 | 544 | -0.0004 | -0.0025 | +0.0015 | +0.0040 | NULL |
| rushing_tds | 2023 | 552 | -0.0021 | -0.0038 | +0.0001 | +0.0037 | NULL |
| rushing_tds | 2024 | 571 | +0.0032 | +0.0010 | +0.0051 | +0.0051 | NULL |
| rushing_tds | pooled | 2223 | +0.0005 | -0.0011 | +0.0020 | +0.0072 | NULL |
| fumbles_lost | 2021 | 556 | -0.0002 | -0.0025 | +0.0022 | +0.0054 | NULL |
| fumbles_lost | 2022 | 544 | -0.0013 | -0.0035 | +0.0013 | +0.0047 | NULL |
| fumbles_lost | 2023 | 552 | +0.0005 | -0.0018 | +0.0026 | +0.0050 | NULL |
| fumbles_lost | 2024 | 571 | -0.0005 | -0.0024 | +0.0014 | +0.0041 | NULL |
| fumbles_lost | pooled | 2223 | -0.0004 | -0.0015 | +0.0008 | +0.0054 | NULL |

### RB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| rushing_yards | 2021 | 933 | +0.0588 | -0.0524 | +0.1721 | +0.0015 | NULL |
| rushing_yards | 2022 | 635 | +0.0220 | -0.0663 | +0.1125 | +0.0011 | NULL |
| rushing_yards | 2023 | 796 | +0.0884 | -0.0013 | +0.1741 | +0.0009 | NULL |
| rushing_yards | 2024 | 927 | -0.0640 | -0.1079 | -0.0167 | +0.0004 | SIGNAL |
| rushing_yards | pooled | 3291 | +0.0458 | -0.0119 | +0.1006 | +0.0015 | NULL |
| rushing_tds | 2021 | 933 | +0.0002 | -0.0030 | +0.0034 | +0.0049 | NULL |
| rushing_tds | 2022 | 635 | -0.0006 | -0.0030 | +0.0021 | +0.0038 | NULL |
| rushing_tds | 2023 | 796 | -0.0001 | -0.0033 | +0.0028 | +0.0036 | NULL |
| rushing_tds | 2024 | 927 | -0.0032 | -0.0053 | -0.0008 | +0.0035 | NULL |
| rushing_tds | pooled | 3291 | -0.0009 | -0.0025 | +0.0007 | +0.0049 | NULL |
| receptions | 2021 | 933 | +0.0017 | -0.0032 | +0.0065 | +0.0012 | NULL |
| receptions | 2022 | 635 | +0.0028 | -0.0025 | +0.0083 | +0.0009 | NULL |
| receptions | 2023 | 796 | -0.0028 | -0.0067 | +0.0011 | +0.0005 | NULL |
| receptions | 2024 | 927 | +0.0059 | +0.0020 | +0.0095 | +0.0009 | NULL |
| receptions | pooled | 3291 | +0.0030 | -0.0002 | +0.0063 | +0.0012 | NULL |
| receiving_yards | 2021 | 933 | -0.0257 | -0.0793 | +0.0289 | +0.0016 | NULL |
| receiving_yards | 2022 | 635 | +0.0663 | +0.0027 | +0.1319 | +0.0016 | REGRESSION |
| receiving_yards | 2023 | 796 | +0.0016 | -0.0384 | +0.0422 | +0.0009 | NULL |
| receiving_yards | 2024 | 927 | +0.0112 | -0.0153 | +0.0378 | +0.0008 | NULL |
| receiving_yards | pooled | 3291 | +0.0376 | +0.0068 | +0.0676 | +0.0016 | NULL |
| receiving_tds | 2021 | 933 | -0.0003 | -0.0010 | +0.0004 | +0.0015 | NULL |
| receiving_tds | 2022 | 635 | +0.0006 | -0.0014 | +0.0024 | -0.0052 | NULL |
| receiving_tds | 2023 | 796 | +0.0007 | -0.0002 | +0.0014 | +0.0013 | NULL |
| receiving_tds | 2024 | 927 | -0.0003 | -0.0006 | +0.0000 | +0.0006 | NULL |
| receiving_tds | pooled | 3291 | +0.0000 | -0.0004 | +0.0004 | +0.0015 | NULL |
| fumbles_lost | 2021 | 933 | -0.0000 | -0.0005 | +0.0003 | +0.0007 | NULL |
| fumbles_lost | 2022 | 635 | +0.0004 | -0.0001 | +0.0009 | +0.0011 | NULL |
| fumbles_lost | 2023 | 796 | +0.0001 | -0.0003 | +0.0005 | +0.0005 | NULL |
| fumbles_lost | 2024 | 927 | +0.0002 | -0.0001 | +0.0005 | +0.0005 | NULL |
| fumbles_lost | pooled | 3291 | +0.0002 | -0.0000 | +0.0004 | +0.0007 | NULL |

### WR

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 1401 | -0.0029 | -0.0078 | +0.0018 | +0.0011 | NULL |
| receptions | 2022 | 1062 | -0.0059 | -0.0123 | +0.0004 | +0.0014 | NULL |
| receptions | 2023 | 1308 | -0.0046 | -0.0119 | +0.0029 | +0.0018 | NULL |
| receptions | 2024 | 1424 | +0.0054 | -0.0002 | +0.0111 | +0.0020 | NULL |
| receptions | pooled | 5195 | -0.0020 | -0.0047 | +0.0007 | +0.0011 | NULL |
| receiving_yards | 2021 | 1401 | -0.0374 | -0.1439 | +0.0703 | +0.0026 | NULL |
| receiving_yards | 2022 | 1062 | -0.0248 | -0.1664 | +0.1073 | +0.0021 | NULL |
| receiving_yards | 2023 | 1308 | -0.0583 | -0.1880 | +0.0730 | +0.0026 | NULL |
| receiving_yards | 2024 | 1424 | +0.0471 | -0.0374 | +0.1378 | +0.0026 | NULL |
| receiving_yards | pooled | 5195 | -0.0129 | -0.0732 | +0.0455 | +0.0026 | NULL |
| receiving_tds | 2021 | 1401 | -0.0013 | -0.0027 | +0.0001 | +0.0018 | NULL |
| receiving_tds | 2022 | 1062 | -0.0004 | -0.0023 | +0.0015 | +0.0020 | NULL |
| receiving_tds | 2023 | 1308 | -0.0016 | -0.0034 | +0.0003 | +0.0019 | NULL |
| receiving_tds | 2024 | 1424 | -0.0007 | -0.0020 | +0.0007 | +0.0025 | NULL |
| receiving_tds | pooled | 5195 | -0.0005 | -0.0012 | +0.0003 | +0.0018 | NULL |
| rushing_yards | 2021 | 1401 | +0.0032 | -0.0108 | +0.0159 | +0.0025 | NULL |
| rushing_yards | 2022 | 1062 | -0.0015 | -0.0196 | +0.0160 | +0.0018 | NULL |
| rushing_yards | 2023 | 1308 | +0.0026 | -0.0082 | +0.0129 | +0.0016 | NULL |
| rushing_yards | 2024 | 1424 | +0.0045 | -0.0066 | +0.0163 | +0.0013 | NULL |
| rushing_yards | pooled | 5195 | +0.0047 | -0.0036 | +0.0124 | +0.0025 | NULL |
| rushing_tds | 2021 | 1401 | +0.0001 | -0.0001 | +0.0003 | +0.0009 | NULL |
| rushing_tds | 2022 | 1062 | -0.0000 | -0.0002 | +0.0001 | +0.0008 | NULL |
| rushing_tds | 2023 | 1308 | +0.0001 | -0.0000 | +0.0002 | +0.0007 | NULL |
| rushing_tds | 2024 | 1424 | -0.0000 | -0.0001 | +0.0001 | +0.0005 | NULL |
| rushing_tds | pooled | 5195 | +0.0000 | -0.0000 | +0.0001 | +0.0009 | NULL |
| fumbles_lost | 2021 | 1401 | +0.0002 | +0.0000 | +0.0004 | +0.0012 | NULL |
| fumbles_lost | 2022 | 1062 | +0.0001 | -0.0001 | +0.0003 | +0.0006 | NULL |
| fumbles_lost | 2023 | 1308 | +0.0001 | -0.0000 | +0.0002 | +0.0002 | NULL |
| fumbles_lost | 2024 | 1424 | -0.0000 | -0.0001 | +0.0001 | +0.0001 | NULL |
| fumbles_lost | pooled | 5195 | +0.0001 | +0.0000 | +0.0002 | +0.0012 | NULL |

### TE

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 594 | -0.0057 | -0.0149 | +0.0029 | +0.0021 | NULL |
| receptions | 2022 | 641 | -0.0039 | -0.0131 | +0.0050 | +0.0031 | NULL |
| receptions | 2023 | 679 | +0.0135 | +0.0027 | +0.0239 | +0.0030 | NULL |
| receptions | 2024 | 669 | +0.0062 | +0.0006 | +0.0122 | +0.0015 | NULL |
| receptions | pooled | 2583 | +0.0012 | -0.0028 | +0.0052 | +0.0021 | NULL |
| receiving_yards | 2021 | 594 | -0.0537 | -0.2050 | +0.0983 | +0.0028 | NULL |
| receiving_yards | 2022 | 641 | -0.0020 | -0.1244 | +0.1279 | +0.0033 | NULL |
| receiving_yards | 2023 | 679 | +0.1608 | +0.0001 | +0.3185 | +0.0029 | REGRESSION |
| receiving_yards | 2024 | 669 | +0.0566 | -0.0319 | +0.1372 | +0.0015 | NULL |
| receiving_yards | pooled | 2583 | +0.0313 | -0.0346 | +0.1032 | +0.0028 | NULL |
| receiving_tds | 2021 | 594 | -0.0005 | -0.0021 | +0.0011 | +0.0015 | NULL |
| receiving_tds | 2022 | 641 | -0.0007 | -0.0029 | +0.0014 | +0.0034 | NULL |
| receiving_tds | 2023 | 679 | -0.0018 | -0.0040 | +0.0004 | +0.0033 | NULL |
| receiving_tds | 2024 | 669 | -0.0023 | -0.0043 | -0.0002 | +0.0041 | NULL |
| receiving_tds | pooled | 2583 | -0.0008 | -0.0016 | -0.0000 | +0.0015 | NULL |
| rushing_yards | 2021 | 594 | +0.0107 | +0.0065 | +0.0154 | +0.0021 | NULL |
| rushing_yards | 2022 | 641 | -0.0002 | -0.0052 | +0.0086 | +0.0009 | NULL |
| rushing_yards | 2023 | 679 | +0.0087 | +0.0050 | +0.0124 | +0.0011 | NULL |
| rushing_yards | 2024 | 669 | +0.0006 | -0.0012 | +0.0035 | +0.0006 | NULL |
| rushing_yards | pooled | 2583 | +0.0078 | +0.0035 | +0.0129 | +0.0021 | NULL |
| rushing_tds | 2021 | 594 | -0.0000 | -0.0001 | +0.0002 | +0.0003 | NULL |
| rushing_tds | 2022 | 641 | -0.0000 | -0.0001 | +0.0000 | +0.0004 | NULL |
| rushing_tds | 2023 | 679 | -0.0004 | -0.0005 | -0.0003 | +0.0007 | NULL |
| rushing_tds | 2024 | 669 | +0.0000 | -0.0002 | +0.0000 | +0.0009 | NULL |
| rushing_tds | pooled | 2583 | -0.0000 | -0.0001 | +0.0000 | +0.0003 | NULL |
| fumbles_lost | 2021 | 594 | -0.0002 | -0.0006 | +0.0002 | +0.0020 | NULL |
| fumbles_lost | 2022 | 641 | +0.0004 | -0.0001 | +0.0009 | +0.0028 | NULL |
| fumbles_lost | 2023 | 679 | -0.0002 | -0.0007 | +0.0003 | +0.0018 | NULL |
| fumbles_lost | 2024 | 669 | +0.0004 | -0.0001 | +0.0007 | +0.0020 | NULL |
| fumbles_lost | pooled | 2583 | +0.0000 | -0.0002 | +0.0002 | +0.0020 | NULL |

## Phase 1 verdict

1/120 cells SIGNAL, 116/120 NULL, 3/120 REGRESSION.
Phase 2 fired.

## Phase 2 — composite ΔRMSE

| Position | Verdict | RMSE delta (95% CI) | Spearman delta (95% CI) | n_paired |
|---|---|---|---|---:|
| QB | ADOPT | -0.0587 ([-0.0920, -0.0281]) | +0.0145 ([+0.0068, +0.0223]) | 2692 |
| RB | DO_NOT_ADOPT | -0.0113 ([-0.0234, +0.0009]) | +0.0012 ([-0.0013, +0.0035]) | 5273 |
| WR | ADOPT | -0.0130 ([-0.0222, -0.0026]) | +0.0009 ([-0.0007, +0.0025]) | 8470 |
| TE | DO_NOT_ADOPT | -0.0058 ([-0.0149, +0.0044]) | -0.0004 ([-0.0027, +0.0022]) | 4257 |

## Probe verdict

Phase 1: 1/120 cells SIGNAL.
Phase 2: 2/4 positions ADOPT.

**This probe is a screen, not the gate.** The adoption gate is the final word on whether a feature change ships; SIGNAL is necessary but not sufficient. Run the full backtest + adoption gate before any production routing change.
