# Feature signal probe — pbp_family_augment

Baseline features: data\features
Overrides:        data\features_probe\pbp_family.parquet
Drops:            (none)
Model class:      baseline

## Phase 1 — per-stat screening

### QB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| passing_yards | 2021 | 556 | +0.4630 | -0.1639 | +1.0281 | +0.0071 | NULL |
| passing_yards | 2022 | 544 | +0.3268 | -0.1874 | +0.8227 | +0.0042 | NULL |
| passing_yards | 2023 | 552 | +0.1089 | -0.2424 | +0.4407 | +0.0021 | NULL |
| passing_yards | 2024 | 571 | +0.2008 | -0.0853 | +0.5225 | +0.0015 | NULL |
| passing_yards | pooled | 2223 | +0.4537 | +0.1354 | +0.7551 | +0.0071 | REGRESSION |
| passing_tds | 2021 | 556 | +0.0011 | -0.0075 | +0.0093 | +0.0068 | NULL |
| passing_tds | 2022 | 544 | +0.0041 | -0.0037 | +0.0117 | +0.0010 | NULL |
| passing_tds | 2023 | 552 | -0.0008 | -0.0053 | +0.0038 | +0.0029 | NULL |
| passing_tds | 2024 | 571 | +0.0029 | -0.0026 | +0.0084 | +0.0025 | NULL |
| passing_tds | pooled | 2223 | +0.0031 | -0.0013 | +0.0074 | +0.0068 | NULL |
| interceptions | 2021 | 556 | +0.0029 | -0.0021 | +0.0081 | +0.0062 | NULL |
| interceptions | 2022 | 544 | -0.0029 | -0.0074 | +0.0017 | +0.0036 | NULL |
| interceptions | 2023 | 552 | +0.0019 | -0.0029 | +0.0065 | +0.0043 | NULL |
| interceptions | 2024 | 571 | -0.0011 | -0.0045 | +0.0022 | +0.0028 | NULL |
| interceptions | pooled | 2223 | +0.0012 | -0.0015 | +0.0042 | +0.0062 | NULL |
| rushing_yards | 2021 | 556 | +0.0556 | -0.0385 | +0.1409 | -0.0002 | NULL |
| rushing_yards | 2022 | 544 | -0.0065 | -0.0701 | +0.0584 | +0.0012 | NULL |
| rushing_yards | 2023 | 552 | -0.0097 | -0.0780 | +0.0530 | +0.0012 | NULL |
| rushing_yards | 2024 | 571 | -0.0234 | -0.1077 | +0.0660 | +0.0002 | NULL |
| rushing_yards | pooled | 2223 | -0.0115 | -0.0568 | +0.0325 | -0.0002 | NULL |
| rushing_tds | 2021 | 556 | -0.0004 | -0.0013 | +0.0005 | +0.0008 | NULL |
| rushing_tds | 2022 | 544 | -0.0000 | -0.0015 | +0.0014 | -0.0014 | NULL |
| rushing_tds | 2023 | 552 | -0.0007 | -0.0018 | +0.0003 | +0.0009 | NULL |
| rushing_tds | 2024 | 571 | +0.0010 | -0.0005 | +0.0024 | +0.0015 | NULL |
| rushing_tds | pooled | 2223 | -0.0001 | -0.0006 | +0.0004 | +0.0008 | NULL |
| fumbles_lost | 2021 | 556 | -0.0003 | -0.0015 | +0.0009 | +0.0013 | NULL |
| fumbles_lost | 2022 | 544 | -0.0004 | -0.0020 | +0.0013 | +0.0015 | NULL |
| fumbles_lost | 2023 | 552 | -0.0001 | -0.0017 | +0.0015 | +0.0017 | NULL |
| fumbles_lost | 2024 | 571 | -0.0009 | -0.0021 | +0.0004 | +0.0015 | NULL |
| fumbles_lost | pooled | 2223 | -0.0004 | -0.0011 | +0.0002 | +0.0013 | NULL |

### RB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| rushing_yards | 2021 | 969 | -0.0592 | -0.1884 | +0.0717 | +0.0031 | NULL |
| rushing_yards | 2022 | 913 | -0.1498 | -0.2824 | -0.0349 | +0.0030 | SIGNAL |
| rushing_yards | 2023 | 900 | -0.0869 | -0.2363 | +0.0552 | +0.0040 | NULL |
| rushing_yards | 2024 | 941 | -0.0031 | -0.1421 | +0.1403 | +0.0041 | NULL |
| rushing_yards | pooled | 3723 | -0.0709 | -0.1304 | -0.0078 | +0.0031 | SIGNAL |
| rushing_tds | 2021 | 969 | +0.0010 | -0.0009 | +0.0027 | +0.0027 | NULL |
| rushing_tds | 2022 | 913 | -0.0010 | -0.0026 | +0.0005 | +0.0010 | NULL |
| rushing_tds | 2023 | 900 | +0.0005 | -0.0010 | +0.0021 | +0.0016 | NULL |
| rushing_tds | 2024 | 941 | +0.0004 | -0.0010 | +0.0016 | +0.0011 | NULL |
| rushing_tds | pooled | 3723 | +0.0007 | -0.0003 | +0.0017 | +0.0027 | NULL |
| receptions | 2021 | 969 | +0.0040 | -0.0027 | +0.0112 | +0.0036 | NULL |
| receptions | 2022 | 913 | +0.0018 | -0.0034 | +0.0070 | +0.0020 | NULL |
| receptions | 2023 | 900 | -0.0029 | -0.0078 | +0.0016 | +0.0014 | NULL |
| receptions | 2024 | 941 | +0.0032 | -0.0025 | +0.0083 | +0.0016 | NULL |
| receptions | pooled | 3723 | +0.0044 | +0.0006 | +0.0081 | +0.0036 | NULL |
| receiving_yards | 2021 | 969 | -0.0170 | -0.0863 | +0.0548 | +0.0043 | NULL |
| receiving_yards | 2022 | 913 | -0.0263 | -0.0899 | +0.0399 | +0.0031 | NULL |
| receiving_yards | 2023 | 900 | -0.0393 | -0.1091 | +0.0288 | +0.0031 | NULL |
| receiving_yards | 2024 | 941 | +0.0497 | -0.0252 | +0.1148 | +0.0033 | NULL |
| receiving_yards | pooled | 3723 | -0.0000 | -0.0357 | +0.0390 | +0.0043 | NULL |
| receiving_tds | 2021 | 969 | -0.0004 | -0.0010 | +0.0003 | +0.0019 | NULL |
| receiving_tds | 2022 | 913 | -0.0001 | -0.0013 | +0.0011 | +0.0033 | NULL |
| receiving_tds | 2023 | 900 | -0.0001 | -0.0011 | +0.0010 | +0.0027 | NULL |
| receiving_tds | 2024 | 941 | +0.0005 | -0.0003 | +0.0013 | +0.0024 | NULL |
| receiving_tds | pooled | 3723 | +0.0000 | -0.0003 | +0.0004 | +0.0019 | NULL |
| fumbles_lost | 2021 | 969 | -0.0001 | -0.0003 | +0.0001 | +0.0004 | NULL |
| fumbles_lost | 2022 | 913 | -0.0001 | -0.0006 | +0.0003 | +0.0009 | NULL |
| fumbles_lost | 2023 | 900 | +0.0001 | -0.0003 | +0.0005 | +0.0010 | NULL |
| fumbles_lost | 2024 | 941 | -0.0001 | -0.0005 | +0.0003 | +0.0007 | NULL |
| fumbles_lost | pooled | 3723 | -0.0000 | -0.0001 | +0.0001 | +0.0004 | NULL |

### WR

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 1633 | -0.0023 | -0.0064 | +0.0018 | +0.0012 | NULL |
| receptions | 2022 | 1616 | -0.0003 | -0.0046 | +0.0039 | +0.0012 | NULL |
| receptions | 2023 | 1616 | -0.0012 | -0.0057 | +0.0029 | +0.0012 | NULL |
| receptions | 2024 | 1563 | -0.0052 | -0.0096 | -0.0010 | +0.0012 | NULL |
| receptions | pooled | 6428 | -0.0023 | -0.0043 | -0.0004 | +0.0012 | NULL |
| receiving_yards | 2021 | 1633 | -0.0330 | -0.1026 | +0.0366 | +0.0018 | NULL |
| receiving_yards | 2022 | 1616 | +0.0365 | -0.0456 | +0.1215 | +0.0017 | NULL |
| receiving_yards | 2023 | 1616 | -0.0085 | -0.0718 | +0.0499 | +0.0012 | NULL |
| receiving_yards | 2024 | 1563 | -0.0585 | -0.1244 | +0.0082 | +0.0010 | NULL |
| receiving_yards | pooled | 6428 | -0.0110 | -0.0468 | +0.0270 | +0.0018 | NULL |
| receiving_tds | 2021 | 1633 | +0.0003 | -0.0013 | +0.0018 | +0.0041 | NULL |
| receiving_tds | 2022 | 1616 | +0.0008 | -0.0008 | +0.0022 | +0.0027 | NULL |
| receiving_tds | 2023 | 1616 | -0.0005 | -0.0015 | +0.0004 | +0.0017 | NULL |
| receiving_tds | 2024 | 1563 | +0.0003 | -0.0007 | +0.0014 | +0.0018 | NULL |
| receiving_tds | pooled | 6428 | +0.0005 | -0.0003 | +0.0013 | +0.0041 | NULL |
| rushing_yards | 2021 | 1633 | +0.0028 | -0.0042 | +0.0110 | +0.0016 | NULL |
| rushing_yards | 2022 | 1616 | +0.0010 | -0.0050 | +0.0071 | +0.0007 | NULL |
| rushing_yards | 2023 | 1616 | +0.0003 | -0.0052 | +0.0061 | +0.0005 | NULL |
| rushing_yards | 2024 | 1563 | +0.0035 | -0.0024 | +0.0099 | +0.0005 | NULL |
| rushing_yards | pooled | 6428 | +0.0029 | -0.0007 | +0.0072 | +0.0016 | NULL |
| rushing_tds | 2021 | 1633 | -0.0001 | -0.0002 | +0.0000 | +0.0011 | NULL |
| rushing_tds | 2022 | 1616 | +0.0001 | -0.0001 | +0.0003 | +0.0013 | NULL |
| rushing_tds | 2023 | 1616 | -0.0001 | -0.0002 | +0.0000 | +0.0007 | NULL |
| rushing_tds | 2024 | 1563 | +0.0000 | -0.0001 | +0.0002 | +0.0009 | NULL |
| rushing_tds | pooled | 6428 | -0.0000 | -0.0001 | +0.0000 | +0.0011 | NULL |
| fumbles_lost | 2021 | 1633 | -0.0000 | -0.0003 | +0.0002 | +0.0014 | NULL |
| fumbles_lost | 2022 | 1616 | +0.0005 | -0.0000 | +0.0010 | +0.0047 | NULL |
| fumbles_lost | 2023 | 1616 | -0.0000 | -0.0001 | +0.0001 | +0.0005 | NULL |
| fumbles_lost | 2024 | 1563 | -0.0000 | -0.0001 | +0.0001 | +0.0004 | NULL |
| fumbles_lost | pooled | 6428 | +0.0000 | -0.0001 | +0.0001 | +0.0014 | NULL |

### TE

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 633 | +0.0003 | -0.0053 | +0.0057 | +0.0018 | NULL |
| receptions | 2022 | 704 | -0.0029 | -0.0084 | +0.0024 | +0.0014 | NULL |
| receptions | 2023 | 719 | +0.0053 | -0.0013 | +0.0112 | +0.0020 | NULL |
| receptions | 2024 | 691 | -0.0056 | -0.0110 | +0.0003 | +0.0013 | NULL |
| receptions | pooled | 2747 | -0.0020 | -0.0050 | +0.0010 | +0.0018 | NULL |
| receiving_yards | 2021 | 633 | +0.0268 | -0.0756 | +0.1182 | +0.0020 | NULL |
| receiving_yards | 2022 | 704 | +0.1891 | +0.0331 | +0.3316 | -0.0030 | REGRESSION |
| receiving_yards | 2023 | 719 | +0.0231 | -0.0265 | +0.0771 | +0.0007 | NULL |
| receiving_yards | 2024 | 691 | -0.0494 | -0.0944 | -0.0050 | +0.0004 | NULL |
| receiving_yards | pooled | 2747 | -0.0190 | -0.0655 | +0.0244 | +0.0020 | NULL |
| receiving_tds | 2021 | 633 | +0.0002 | -0.0008 | +0.0012 | +0.0009 | NULL |
| receiving_tds | 2022 | 704 | +0.0000 | -0.0007 | +0.0007 | +0.0004 | NULL |
| receiving_tds | 2023 | 719 | +0.0005 | -0.0001 | +0.0011 | +0.0003 | NULL |
| receiving_tds | 2024 | 691 | -0.0002 | -0.0007 | +0.0003 | +0.0002 | NULL |
| receiving_tds | pooled | 2747 | +0.0005 | +0.0000 | +0.0010 | +0.0009 | NULL |
| rushing_yards | 2021 | 633 | -0.0014 | -0.0156 | +0.0207 | +0.0077 | NULL |
| rushing_yards | 2022 | 704 | +0.0078 | -0.0004 | +0.0206 | +0.0072 | NULL |
| rushing_yards | 2023 | 719 | +0.0124 | +0.0050 | +0.0233 | +0.0059 | NULL |
| rushing_yards | 2024 | 691 | +0.0025 | -0.0055 | +0.0154 | +0.0048 | NULL |
| rushing_yards | pooled | 2747 | +0.0091 | +0.0015 | +0.0179 | +0.0077 | NULL |
| rushing_tds | 2021 | 633 | +0.0001 | +0.0000 | +0.0004 | +0.0011 | NULL |
| rushing_tds | 2022 | 704 | +0.0000 | -0.0000 | +0.0002 | +0.0004 | NULL |
| rushing_tds | 2023 | 719 | +0.0001 | +0.0001 | +0.0002 | +0.0002 | NULL |
| rushing_tds | 2024 | 691 | -0.0000 | -0.0001 | +0.0002 | +0.0002 | NULL |
| rushing_tds | pooled | 2747 | +0.0000 | -0.0000 | +0.0001 | +0.0011 | NULL |
| fumbles_lost | 2021 | 633 | +0.0001 | -0.0002 | +0.0003 | +0.0013 | NULL |
| fumbles_lost | 2022 | 704 | +0.0003 | -0.0001 | +0.0007 | +0.0012 | NULL |
| fumbles_lost | 2023 | 719 | +0.0003 | +0.0000 | +0.0005 | +0.0007 | NULL |
| fumbles_lost | 2024 | 691 | -0.0001 | -0.0002 | +0.0001 | +0.0002 | NULL |
| fumbles_lost | pooled | 2747 | +0.0002 | -0.0000 | +0.0004 | +0.0013 | NULL |

## Phase 1 verdict

2/120 cells SIGNAL, 116/120 NULL, 2/120 REGRESSION.
Phase 2 fired.

## Phase 2 — composite ΔRMSE

| Position | Verdict | RMSE delta (95% CI) | Spearman delta (95% CI) | n_paired |
|---|---|---|---|---:|
| QB | DO_NOT_ADOPT | +0.0108 ([-0.0106, +0.0312]) | -0.0020 ([-0.0055, +0.0020]) | 2692 |
| RB | ADOPT | -0.0124 ([-0.0249, -0.0001]) | +0.0021 ([-0.0003, +0.0044]) | 5273 |
| WR | DO_NOT_ADOPT | -0.0038 ([-0.0124, +0.0053]) | -0.0004 ([-0.0021, +0.0012]) | 8470 |
| TE | DO_NOT_ADOPT | +0.0055 ([-0.0048, +0.0163]) | -0.0015 ([-0.0043, +0.0017]) | 4257 |

## Probe verdict

Phase 1: 2/120 cells SIGNAL.
Phase 2: 1/4 positions ADOPT.

**This probe is a screen, not the gate.** The adoption gate is the final word on whether a feature change ships; SIGNAL is necessary but not sufficient. Run the full backtest + adoption gate before any production routing change.
