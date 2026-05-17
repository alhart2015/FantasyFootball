# Feature signal probe — vegas_team_context_lgbnb_augment

Baseline features: data\features
Overrides:        data\features_probe\vegas_team_context.parquet
Drops:            (none)
Model class:      lightgbm-nb

## Phase 1 — per-stat screening

### QB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| passing_yards | 2021 | 556 | +0.4013 | -0.0948 | +0.8763 | +0.0029 | NULL |
| passing_yards | 2022 | 544 | +0.2286 | -0.0770 | +0.5199 | +0.0003 | NULL |
| passing_yards | 2023 | 552 | +0.0489 | -0.0990 | +0.2026 | +0.0004 | NULL |
| passing_yards | 2024 | 571 | +0.0844 | -0.1020 | +0.2632 | +0.0005 | NULL |
| passing_yards | pooled | 2223 | +0.3368 | +0.1082 | +0.5516 | +0.0029 | REGRESSION |
| passing_tds | 2021 | 556 | +0.0073 | +0.0014 | +0.0132 | +0.0029 | NULL |
| passing_tds | 2022 | 544 | +0.0029 | -0.0001 | +0.0057 | +0.0008 | NULL |
| passing_tds | 2023 | 552 | +0.0009 | -0.0009 | +0.0027 | +0.0004 | NULL |
| passing_tds | 2024 | 571 | -0.0009 | -0.0024 | +0.0005 | +0.0003 | NULL |
| passing_tds | pooled | 2223 | +0.0067 | +0.0039 | +0.0095 | +0.0029 | NULL |
| interceptions | 2021 | 556 | +0.0034 | -0.0021 | +0.0089 | +0.0014 | NULL |
| interceptions | 2022 | 544 | -0.0007 | -0.0045 | +0.0034 | +0.0026 | NULL |
| interceptions | 2023 | 552 | +0.0002 | -0.0030 | +0.0033 | +0.0027 | NULL |
| interceptions | 2024 | 571 | +0.0007 | -0.0019 | +0.0032 | +0.0022 | NULL |
| interceptions | pooled | 2223 | +0.0005 | -0.0023 | +0.0033 | +0.0014 | NULL |
| rushing_yards | 2021 | 556 | +0.0791 | -0.0201 | +0.1716 | +0.0022 | NULL |
| rushing_yards | 2022 | 544 | -0.0261 | -0.0871 | +0.0414 | +0.0010 | NULL |
| rushing_yards | 2023 | 552 | +0.0958 | +0.0238 | +0.1721 | +0.0017 | REGRESSION |
| rushing_yards | 2024 | 571 | +0.0035 | -0.0391 | +0.0479 | +0.0007 | NULL |
| rushing_yards | pooled | 2223 | +0.0268 | -0.0162 | +0.0700 | +0.0022 | NULL |
| rushing_tds | 2021 | 556 | +0.0005 | -0.0035 | +0.0040 | +0.0066 | NULL |
| rushing_tds | 2022 | 544 | -0.0006 | -0.0029 | +0.0016 | +0.0015 | NULL |
| rushing_tds | 2023 | 552 | -0.0018 | -0.0036 | +0.0003 | +0.0038 | NULL |
| rushing_tds | 2024 | 571 | +0.0033 | +0.0010 | +0.0054 | +0.0051 | NULL |
| rushing_tds | pooled | 2223 | +0.0005 | -0.0010 | +0.0018 | +0.0066 | NULL |
| fumbles_lost | 2021 | 556 | +0.0009 | -0.0010 | +0.0029 | +0.0032 | NULL |
| fumbles_lost | 2022 | 544 | -0.0001 | -0.0014 | +0.0014 | +0.0021 | NULL |
| fumbles_lost | 2023 | 552 | +0.0001 | -0.0011 | +0.0014 | +0.0018 | NULL |
| fumbles_lost | 2024 | 571 | -0.0000 | -0.0013 | +0.0012 | +0.0016 | NULL |
| fumbles_lost | pooled | 2223 | +0.0001 | -0.0008 | +0.0010 | +0.0032 | NULL |

### RB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| rushing_yards | 2021 | 933 | +0.0837 | -0.0163 | +0.1893 | +0.0012 | NULL |
| rushing_yards | 2022 | 635 | -0.0096 | -0.0804 | +0.0583 | +0.0006 | NULL |
| rushing_yards | 2023 | 796 | +0.0378 | -0.0306 | +0.1037 | +0.0006 | NULL |
| rushing_yards | 2024 | 927 | -0.0264 | -0.0728 | +0.0197 | +0.0006 | NULL |
| rushing_yards | pooled | 3291 | +0.0521 | +0.0009 | +0.1025 | +0.0012 | REGRESSION |
| rushing_tds | 2021 | 933 | -0.0000 | -0.0010 | +0.0011 | +0.0004 | NULL |
| rushing_tds | 2022 | 635 | -0.0001 | -0.0012 | +0.0011 | +0.0004 | NULL |
| rushing_tds | 2023 | 796 | +0.0000 | -0.0012 | +0.0012 | +0.0004 | NULL |
| rushing_tds | 2024 | 927 | -0.0011 | -0.0025 | +0.0005 | +0.0002 | NULL |
| rushing_tds | pooled | 3291 | +0.0000 | -0.0005 | +0.0006 | +0.0004 | NULL |
| receptions | 2021 | 933 | -0.0016 | -0.0060 | +0.0028 | +0.0008 | NULL |
| receptions | 2022 | 635 | +0.0026 | -0.0032 | +0.0085 | +0.0011 | NULL |
| receptions | 2023 | 796 | -0.0032 | -0.0074 | +0.0013 | +0.0008 | NULL |
| receptions | 2024 | 927 | +0.0046 | +0.0003 | +0.0089 | +0.0011 | NULL |
| receptions | pooled | 3291 | +0.0009 | -0.0016 | +0.0034 | +0.0008 | NULL |
| receiving_yards | 2021 | 933 | -0.0394 | -0.0957 | +0.0203 | +0.0018 | NULL |
| receiving_yards | 2022 | 635 | +0.0730 | -0.0030 | +0.1504 | +0.0021 | NULL |
| receiving_yards | 2023 | 796 | -0.0093 | -0.0598 | +0.0435 | +0.0011 | NULL |
| receiving_yards | 2024 | 927 | +0.0206 | -0.0191 | +0.0589 | +0.0011 | NULL |
| receiving_yards | pooled | 3291 | +0.0271 | -0.0036 | +0.0583 | +0.0018 | NULL |
| receiving_tds | 2021 | 933 | +0.0001 | -0.0005 | +0.0007 | +0.0011 | NULL |
| receiving_tds | 2022 | 635 | +0.0003 | -0.0003 | +0.0008 | +0.0011 | NULL |
| receiving_tds | 2023 | 796 | +0.0005 | +0.0000 | +0.0010 | +0.0006 | NULL |
| receiving_tds | 2024 | 927 | -0.0011 | -0.0024 | +0.0002 | -0.0044 | NULL |
| receiving_tds | pooled | 3291 | +0.0002 | -0.0001 | +0.0005 | +0.0011 | NULL |
| fumbles_lost | 2021 | 933 | -0.0001 | -0.0006 | +0.0005 | +0.0018 | NULL |
| fumbles_lost | 2022 | 635 | +0.0003 | -0.0004 | +0.0009 | +0.0019 | NULL |
| fumbles_lost | 2023 | 796 | -0.0002 | -0.0008 | +0.0003 | +0.0014 | NULL |
| fumbles_lost | 2024 | 927 | +0.0001 | -0.0003 | +0.0006 | +0.0018 | NULL |
| fumbles_lost | pooled | 3291 | +0.0000 | -0.0003 | +0.0003 | +0.0018 | NULL |

### WR

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 1401 | +0.0006 | -0.0016 | +0.0030 | +0.0002 | NULL |
| receptions | 2022 | 1062 | -0.0027 | -0.0055 | +0.0000 | +0.0002 | NULL |
| receptions | 2023 | 1308 | -0.0002 | -0.0038 | +0.0034 | +0.0005 | NULL |
| receptions | 2024 | 1424 | +0.0027 | -0.0004 | +0.0058 | +0.0005 | NULL |
| receptions | pooled | 5195 | +0.0007 | -0.0004 | +0.0019 | +0.0002 | NULL |
| receiving_yards | 2021 | 1401 | +0.0569 | -0.0045 | +0.1217 | +0.0009 | NULL |
| receiving_yards | 2022 | 1062 | -0.0280 | -0.0678 | +0.0107 | +0.0003 | NULL |
| receiving_yards | 2023 | 1308 | +0.0171 | -0.0283 | +0.0629 | +0.0006 | NULL |
| receiving_yards | 2024 | 1424 | +0.0371 | -0.0016 | +0.0713 | +0.0005 | NULL |
| receiving_yards | pooled | 5195 | +0.0365 | +0.0050 | +0.0680 | +0.0009 | NULL |
| receiving_tds | 2021 | 1401 | +0.0008 | +0.0001 | +0.0015 | +0.0006 | NULL |
| receiving_tds | 2022 | 1062 | -0.0000 | -0.0003 | +0.0002 | +0.0001 | NULL |
| receiving_tds | 2023 | 1308 | +0.0000 | -0.0002 | +0.0004 | +0.0001 | NULL |
| receiving_tds | 2024 | 1424 | -0.0004 | -0.0007 | -0.0001 | +0.0001 | NULL |
| receiving_tds | pooled | 5195 | +0.0007 | +0.0003 | +0.0010 | +0.0006 | NULL |
| rushing_yards | 2021 | 1401 | +0.0028 | -0.0102 | +0.0142 | +0.0026 | NULL |
| rushing_yards | 2022 | 1062 | -0.0013 | -0.0199 | +0.0175 | +0.0019 | NULL |
| rushing_yards | 2023 | 1308 | +0.0037 | -0.0076 | +0.0151 | +0.0017 | NULL |
| rushing_yards | 2024 | 1424 | +0.0085 | -0.0031 | +0.0210 | +0.0013 | NULL |
| rushing_yards | pooled | 5195 | +0.0054 | -0.0032 | +0.0135 | +0.0026 | NULL |
| rushing_tds | 2021 | 1401 | +0.0001 | -0.0000 | +0.0002 | +0.0006 | NULL |
| rushing_tds | 2022 | 1062 | -0.0000 | -0.0002 | +0.0002 | +0.0006 | NULL |
| rushing_tds | 2023 | 1308 | +0.0000 | -0.0001 | +0.0001 | +0.0005 | NULL |
| rushing_tds | 2024 | 1424 | +0.0000 | -0.0001 | +0.0001 | +0.0004 | NULL |
| rushing_tds | pooled | 5195 | +0.0000 | -0.0000 | +0.0001 | +0.0006 | NULL |
| fumbles_lost | 2021 | 1401 | +0.0002 | +0.0001 | +0.0004 | +0.0016 | NULL |
| fumbles_lost | 2022 | 1062 | +0.0000 | -0.0002 | +0.0002 | +0.0006 | NULL |
| fumbles_lost | 2023 | 1308 | +0.0001 | -0.0001 | +0.0003 | +0.0006 | NULL |
| fumbles_lost | 2024 | 1424 | -0.0000 | -0.0001 | +0.0001 | +0.0003 | NULL |
| fumbles_lost | pooled | 5195 | +0.0001 | -0.0000 | +0.0002 | +0.0016 | NULL |

### TE

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 594 | +0.0077 | -0.0008 | +0.0156 | -0.0028 | NULL |
| receptions | 2022 | 641 | -0.0003 | -0.0051 | +0.0045 | +0.0010 | NULL |
| receptions | 2023 | 679 | +0.0022 | -0.0029 | +0.0072 | +0.0006 | NULL |
| receptions | 2024 | 669 | +0.0008 | -0.0033 | +0.0048 | +0.0007 | NULL |
| receptions | pooled | 2583 | +0.0039 | -0.0002 | +0.0079 | -0.0028 | NULL |
| receiving_yards | 2021 | 594 | +0.0228 | -0.0680 | +0.1168 | +0.0014 | NULL |
| receiving_yards | 2022 | 641 | +0.0384 | -0.0354 | +0.1131 | +0.0008 | NULL |
| receiving_yards | 2023 | 679 | +0.0226 | -0.0194 | +0.0604 | +0.0004 | NULL |
| receiving_yards | 2024 | 669 | +0.0098 | -0.0314 | +0.0566 | +0.0004 | NULL |
| receiving_yards | pooled | 2583 | +0.0180 | -0.0219 | +0.0593 | +0.0014 | NULL |
| receiving_tds | 2021 | 594 | +0.0018 | +0.0005 | +0.0031 | +0.0012 | NULL |
| receiving_tds | 2022 | 641 | +0.0004 | -0.0006 | +0.0012 | +0.0004 | NULL |
| receiving_tds | 2023 | 679 | -0.0001 | -0.0005 | +0.0003 | +0.0001 | NULL |
| receiving_tds | 2024 | 669 | -0.0002 | -0.0008 | +0.0003 | +0.0003 | NULL |
| receiving_tds | pooled | 2583 | +0.0007 | +0.0001 | +0.0012 | +0.0012 | NULL |
| rushing_yards | 2021 | 594 | +0.0091 | +0.0040 | +0.0149 | +0.0041 | NULL |
| rushing_yards | 2022 | 641 | -0.0010 | -0.0096 | +0.0143 | +0.0028 | NULL |
| rushing_yards | 2023 | 679 | +0.0149 | +0.0100 | +0.0201 | +0.0028 | NULL |
| rushing_yards | 2024 | 669 | +0.0015 | -0.0028 | +0.0076 | +0.0017 | NULL |
| rushing_yards | pooled | 2583 | +0.0084 | +0.0028 | +0.0148 | +0.0041 | NULL |
| rushing_tds | 2021 | 594 | +0.0000 | -0.0000 | +0.0000 | +0.0001 | NULL |
| rushing_tds | 2022 | 641 | +0.0000 | -0.0000 | +0.0002 | +0.0001 | NULL |
| rushing_tds | 2023 | 679 | -0.0001 | -0.0001 | -0.0000 | +0.0002 | NULL |
| rushing_tds | 2024 | 669 | -0.0000 | -0.0001 | -0.0000 | +0.0003 | NULL |
| rushing_tds | pooled | 2583 | +0.0000 | -0.0000 | +0.0000 | +0.0001 | NULL |
| fumbles_lost | 2021 | 594 | -0.0002 | -0.0008 | +0.0003 | +0.0026 | NULL |
| fumbles_lost | 2022 | 641 | +0.0003 | -0.0001 | +0.0007 | +0.0033 | NULL |
| fumbles_lost | 2023 | 679 | -0.0002 | -0.0009 | +0.0004 | +0.0024 | NULL |
| fumbles_lost | 2024 | 669 | +0.0004 | -0.0000 | +0.0008 | +0.0026 | NULL |
| fumbles_lost | pooled | 2583 | +0.0000 | -0.0002 | +0.0003 | +0.0026 | NULL |

## Phase 1 verdict

0/120 cells SIGNAL, 117/120 NULL, 3/120 REGRESSION.
Phase 2 fired.

## Phase 2 — composite ΔRMSE

| Position | Verdict | RMSE delta (95% CI) | Spearman delta (95% CI) | n_paired |
|---|---|---|---|---:|
| QB | DO_NOT_ADOPT | +0.0165 ([-0.0012, +0.0345]) | -0.0018 ([-0.0053, +0.0017]) | 2692 |
| RB | DO_NOT_ADOPT | +0.0049 ([-0.0047, +0.0140]) | -0.0006 ([-0.0026, +0.0016]) | 5273 |
| WR | DO_NOT_ADOPT | -0.0004 ([-0.0060, +0.0055]) | -0.0006 ([-0.0020, +0.0005]) | 8470 |
| TE | DO_NOT_ADOPT | +0.0064 ([-0.0010, +0.0139]) | -0.0009 ([-0.0028, +0.0011]) | 4257 |

## Probe verdict

Phase 1: 0/120 cells SIGNAL.
Phase 2: 0/4 positions ADOPT.

**This probe is a screen, not the gate.** The adoption gate is the final word on whether a feature change ships; SIGNAL is necessary but not sufficient. Run the full backtest + adoption gate before any production routing change.
