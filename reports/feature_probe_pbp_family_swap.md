# Feature signal probe — pbp_family_swap

Baseline features: data\features
Overrides:        data\features_probe\pbp_family.parquet
Drops:            opp_allowed_qb_fppg_l4, opp_allowed_rb_fppg_l4, opp_allowed_wr_fppg_l4, opp_allowed_te_fppg_l4
Model class:      baseline

## Phase 1 — per-stat screening

### QB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| passing_yards | 2021 | 620 | +0.1706 | -0.5652 | +0.8616 | +0.0094 | NULL |
| passing_yards | 2022 | 612 | +0.2884 | -0.2465 | +0.8606 | +0.0065 | NULL |
| passing_yards | 2023 | 622 | -0.0538 | -0.4805 | +0.4302 | +0.0046 | NULL |
| passing_yards | 2024 | 641 | +0.4362 | -0.0313 | +0.8885 | +0.0037 | NULL |
| passing_yards | pooled | 2495 | +0.2889 | -0.0656 | +0.6345 | +0.0094 | NULL |
| passing_tds | 2021 | 620 | -0.0005 | -0.0091 | +0.0080 | +0.0063 | NULL |
| passing_tds | 2022 | 612 | +0.0037 | -0.0037 | +0.0109 | +0.0042 | NULL |
| passing_tds | 2023 | 622 | -0.0008 | -0.0063 | +0.0045 | +0.0039 | NULL |
| passing_tds | 2024 | 641 | +0.0049 | -0.0010 | +0.0106 | +0.0035 | NULL |
| passing_tds | pooled | 2495 | +0.0030 | -0.0011 | +0.0068 | +0.0063 | NULL |
| interceptions | 2021 | 620 | +0.0031 | -0.0030 | +0.0090 | +0.0082 | NULL |
| interceptions | 2022 | 612 | -0.0007 | -0.0059 | +0.0049 | +0.0049 | NULL |
| interceptions | 2023 | 622 | +0.0003 | -0.0046 | +0.0049 | +0.0045 | NULL |
| interceptions | 2024 | 641 | -0.0016 | -0.0057 | +0.0025 | +0.0036 | NULL |
| interceptions | pooled | 2495 | +0.0017 | -0.0015 | +0.0048 | +0.0082 | NULL |
| rushing_yards | 2021 | 620 | +0.0099 | -0.0628 | +0.0832 | +0.0017 | NULL |
| rushing_yards | 2022 | 612 | -0.0076 | -0.0678 | +0.0494 | +0.0011 | NULL |
| rushing_yards | 2023 | 622 | -0.0424 | -0.1072 | +0.0204 | +0.0002 | NULL |
| rushing_yards | 2024 | 641 | -0.0039 | -0.0629 | +0.0574 | +0.0012 | NULL |
| rushing_yards | pooled | 2495 | +0.0107 | -0.0265 | +0.0454 | +0.0017 | NULL |
| rushing_tds | 2021 | 620 | -0.0006 | -0.0016 | +0.0004 | -0.0025 | NULL |
| rushing_tds | 2022 | 612 | -0.0000 | -0.0009 | +0.0008 | +0.0007 | NULL |
| rushing_tds | 2023 | 622 | -0.0005 | -0.0014 | +0.0003 | +0.0007 | NULL |
| rushing_tds | 2024 | 641 | +0.0004 | -0.0007 | +0.0015 | +0.0011 | NULL |
| rushing_tds | pooled | 2495 | +0.0003 | -0.0003 | +0.0009 | -0.0025 | NULL |
| fumbles_lost | 2021 | 620 | -0.0008 | -0.0024 | +0.0009 | +0.0024 | NULL |
| fumbles_lost | 2022 | 612 | -0.0003 | -0.0025 | +0.0017 | +0.0031 | NULL |
| fumbles_lost | 2023 | 622 | -0.0002 | -0.0025 | +0.0017 | +0.0029 | NULL |
| fumbles_lost | 2024 | 641 | -0.0007 | -0.0024 | +0.0010 | +0.0025 | NULL |
| fumbles_lost | pooled | 2495 | -0.0006 | -0.0015 | +0.0002 | +0.0024 | NULL |

### RB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| rushing_yards | 2021 | 1092 | -0.0552 | -0.1988 | +0.0800 | +0.0041 | NULL |
| rushing_yards | 2022 | 1032 | -0.1963 | -0.3319 | -0.0619 | +0.0036 | SIGNAL |
| rushing_yards | 2023 | 1015 | -0.0991 | -0.2561 | +0.0470 | +0.0049 | NULL |
| rushing_yards | 2024 | 1068 | -0.0879 | -0.2309 | +0.0519 | +0.0049 | NULL |
| rushing_yards | pooled | 4207 | -0.1047 | -0.1754 | -0.0397 | +0.0041 | SIGNAL |
| rushing_tds | 2021 | 1092 | +0.0015 | +0.0001 | +0.0030 | +0.0018 | NULL |
| rushing_tds | 2022 | 1032 | -0.0006 | -0.0017 | +0.0004 | +0.0006 | NULL |
| rushing_tds | 2023 | 1015 | +0.0005 | -0.0007 | +0.0017 | +0.0012 | NULL |
| rushing_tds | 2024 | 1068 | -0.0001 | -0.0011 | +0.0009 | +0.0008 | NULL |
| rushing_tds | pooled | 4207 | +0.0007 | -0.0001 | +0.0016 | +0.0018 | NULL |
| receptions | 2021 | 1092 | +0.0002 | -0.0048 | +0.0055 | +0.0023 | NULL |
| receptions | 2022 | 1032 | +0.0019 | -0.0028 | +0.0071 | +0.0019 | NULL |
| receptions | 2023 | 1015 | -0.0029 | -0.0071 | +0.0015 | +0.0013 | NULL |
| receptions | 2024 | 1068 | +0.0031 | -0.0019 | +0.0082 | +0.0016 | NULL |
| receptions | pooled | 4207 | +0.0021 | -0.0004 | +0.0051 | +0.0023 | NULL |
| receiving_yards | 2021 | 1092 | -0.0187 | -0.0859 | +0.0449 | +0.0034 | NULL |
| receiving_yards | 2022 | 1032 | -0.0172 | -0.0767 | +0.0495 | +0.0032 | NULL |
| receiving_yards | 2023 | 1015 | -0.0344 | -0.0988 | +0.0314 | +0.0030 | NULL |
| receiving_yards | 2024 | 1068 | +0.0532 | -0.0143 | +0.1257 | +0.0032 | NULL |
| receiving_yards | pooled | 4207 | +0.0090 | -0.0270 | +0.0445 | +0.0034 | NULL |
| receiving_tds | 2021 | 1092 | +0.0003 | -0.0010 | +0.0016 | -0.0057 | NULL |
| receiving_tds | 2022 | 1032 | -0.0000 | -0.0010 | +0.0008 | +0.0024 | NULL |
| receiving_tds | 2023 | 1015 | -0.0001 | -0.0009 | +0.0006 | +0.0017 | NULL |
| receiving_tds | 2024 | 1068 | +0.0005 | -0.0002 | +0.0012 | +0.0016 | NULL |
| receiving_tds | pooled | 4207 | -0.0003 | -0.0010 | +0.0003 | -0.0057 | NULL |
| fumbles_lost | 2021 | 1092 | -0.0002 | -0.0004 | +0.0001 | +0.0005 | NULL |
| fumbles_lost | 2022 | 1032 | -0.0002 | -0.0006 | +0.0002 | +0.0010 | NULL |
| fumbles_lost | 2023 | 1015 | +0.0002 | -0.0002 | +0.0006 | +0.0011 | NULL |
| fumbles_lost | 2024 | 1068 | +0.0000 | -0.0004 | +0.0004 | +0.0006 | NULL |
| fumbles_lost | pooled | 4207 | -0.0000 | -0.0002 | +0.0001 | +0.0005 | NULL |

### WR

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 1820 | -0.0021 | -0.0071 | +0.0022 | +0.0016 | NULL |
| receptions | 2022 | 1800 | -0.0005 | -0.0052 | +0.0041 | +0.0015 | NULL |
| receptions | 2023 | 1801 | +0.0013 | -0.0029 | +0.0058 | +0.0014 | NULL |
| receptions | 2024 | 1746 | -0.0043 | -0.0082 | -0.0003 | +0.0011 | NULL |
| receptions | pooled | 7167 | -0.0017 | -0.0041 | +0.0007 | +0.0016 | NULL |
| receiving_yards | 2021 | 1820 | -0.0402 | -0.1166 | +0.0396 | +0.0020 | NULL |
| receiving_yards | 2022 | 1800 | +0.0182 | -0.0629 | +0.0987 | +0.0019 | NULL |
| receiving_yards | 2023 | 1801 | -0.0024 | -0.0682 | +0.0646 | +0.0013 | NULL |
| receiving_yards | 2024 | 1746 | -0.0489 | -0.1182 | +0.0220 | +0.0013 | NULL |
| receiving_yards | pooled | 7167 | -0.0181 | -0.0562 | +0.0200 | +0.0020 | NULL |
| receiving_tds | 2021 | 1820 | +0.0005 | -0.0012 | +0.0022 | +0.0047 | NULL |
| receiving_tds | 2022 | 1800 | +0.0004 | -0.0012 | +0.0019 | +0.0027 | NULL |
| receiving_tds | 2023 | 1801 | -0.0008 | -0.0017 | +0.0002 | +0.0021 | NULL |
| receiving_tds | 2024 | 1746 | +0.0007 | -0.0005 | +0.0018 | +0.0023 | NULL |
| receiving_tds | pooled | 7167 | +0.0006 | -0.0003 | +0.0014 | +0.0047 | NULL |
| rushing_yards | 2021 | 1820 | +0.0037 | -0.0069 | +0.0142 | +0.0029 | NULL |
| rushing_yards | 2022 | 1800 | +0.0005 | -0.0057 | +0.0068 | +0.0009 | NULL |
| rushing_yards | 2023 | 1801 | +0.0026 | -0.0032 | +0.0084 | +0.0007 | NULL |
| rushing_yards | 2024 | 1746 | +0.0028 | -0.0022 | +0.0084 | +0.0005 | NULL |
| rushing_yards | pooled | 7167 | +0.0119 | +0.0053 | +0.0184 | +0.0029 | NULL |
| rushing_tds | 2021 | 1820 | -0.0001 | -0.0001 | +0.0000 | +0.0008 | NULL |
| rushing_tds | 2022 | 1800 | +0.0001 | -0.0001 | +0.0002 | +0.0010 | NULL |
| rushing_tds | 2023 | 1801 | -0.0001 | -0.0001 | +0.0000 | +0.0007 | NULL |
| rushing_tds | 2024 | 1746 | +0.0000 | -0.0001 | +0.0002 | +0.0009 | NULL |
| rushing_tds | pooled | 7167 | -0.0000 | -0.0001 | +0.0000 | +0.0008 | NULL |
| fumbles_lost | 2021 | 1820 | +0.0001 | -0.0004 | +0.0005 | +0.0059 | NULL |
| fumbles_lost | 2022 | 1800 | +0.0003 | -0.0003 | +0.0007 | +0.0047 | NULL |
| fumbles_lost | 2023 | 1801 | -0.0000 | -0.0002 | +0.0001 | +0.0008 | NULL |
| fumbles_lost | 2024 | 1746 | +0.0001 | -0.0000 | +0.0003 | +0.0007 | NULL |
| fumbles_lost | pooled | 7167 | +0.0003 | +0.0001 | +0.0006 | +0.0059 | NULL |

### TE

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 732 | +0.0003 | -0.0055 | +0.0063 | +0.0016 | NULL |
| receptions | 2022 | 797 | +0.0009 | -0.0037 | +0.0053 | +0.0006 | NULL |
| receptions | 2023 | 809 | +0.0027 | -0.0017 | +0.0069 | +0.0008 | NULL |
| receptions | 2024 | 786 | -0.0055 | -0.0092 | -0.0018 | +0.0006 | NULL |
| receptions | pooled | 3124 | -0.0007 | -0.0035 | +0.0024 | +0.0016 | NULL |
| receiving_yards | 2021 | 732 | +0.0829 | -0.0833 | +0.2565 | -0.0029 | NULL |
| receiving_yards | 2022 | 797 | +0.0299 | -0.0477 | +0.1054 | +0.0013 | NULL |
| receiving_yards | 2023 | 809 | +0.0196 | -0.0430 | +0.0905 | +0.0011 | NULL |
| receiving_yards | 2024 | 786 | -0.0391 | -0.1009 | +0.0240 | +0.0008 | NULL |
| receiving_yards | pooled | 3124 | +0.0343 | -0.0578 | +0.1184 | -0.0029 | NULL |
| receiving_tds | 2021 | 732 | +0.0001 | -0.0010 | +0.0012 | +0.0010 | NULL |
| receiving_tds | 2022 | 797 | +0.0001 | -0.0007 | +0.0008 | +0.0006 | NULL |
| receiving_tds | 2023 | 809 | +0.0002 | -0.0004 | +0.0008 | +0.0004 | NULL |
| receiving_tds | 2024 | 786 | -0.0003 | -0.0009 | +0.0004 | +0.0004 | NULL |
| receiving_tds | pooled | 3124 | +0.0001 | -0.0003 | +0.0006 | +0.0010 | NULL |
| rushing_yards | 2021 | 732 | -0.0039 | -0.0154 | +0.0121 | +0.0087 | NULL |
| rushing_yards | 2022 | 797 | +0.0120 | +0.0046 | +0.0230 | +0.0084 | NULL |
| rushing_yards | 2023 | 809 | +0.0122 | +0.0042 | +0.0221 | +0.0064 | NULL |
| rushing_yards | 2024 | 786 | +0.0073 | -0.0026 | +0.0155 | +0.0053 | NULL |
| rushing_yards | pooled | 3124 | +0.0068 | +0.0040 | +0.0155 | +0.0087 | NULL |
| rushing_tds | 2021 | 732 | +0.0001 | +0.0000 | +0.0004 | +0.0012 | NULL |
| rushing_tds | 2022 | 797 | +0.0000 | -0.0000 | +0.0001 | +0.0005 | NULL |
| rushing_tds | 2023 | 809 | +0.0001 | +0.0000 | +0.0001 | +0.0002 | NULL |
| rushing_tds | 2024 | 786 | +0.0000 | -0.0000 | +0.0002 | +0.0002 | NULL |
| rushing_tds | pooled | 3124 | +0.0000 | -0.0000 | +0.0001 | +0.0012 | NULL |
| fumbles_lost | 2021 | 732 | +0.0001 | -0.0001 | +0.0002 | +0.0007 | NULL |
| fumbles_lost | 2022 | 797 | +0.0002 | -0.0001 | +0.0004 | +0.0005 | NULL |
| fumbles_lost | 2023 | 809 | +0.0002 | +0.0000 | +0.0003 | +0.0003 | NULL |
| fumbles_lost | 2024 | 786 | -0.0000 | -0.0001 | +0.0001 | +0.0001 | NULL |
| fumbles_lost | pooled | 3124 | +0.0002 | +0.0001 | +0.0003 | +0.0007 | NULL |

## Phase 1 verdict

2/120 cells SIGNAL, 118/120 NULL, 0/120 REGRESSION.
Phase 2 fired.

## Phase 2 — composite ΔRMSE

| Position | Verdict | RMSE delta (95% CI) | Spearman delta (95% CI) | n_paired |
|---|---|---|---|---:|
| QB | DO_NOT_ADOPT | -0.0022 ([-0.0261, +0.0223]) | -0.0005 ([-0.0050, +0.0041]) | 2692 |
| RB | ADOPT | -0.0131 ([-0.0259, -0.0003]) | +0.0023 ([-0.0002, +0.0048]) | 5273 |
| WR | DO_NOT_ADOPT | -0.0028 ([-0.0120, +0.0069]) | -0.0009 ([-0.0026, +0.0009]) | 8470 |
| TE | DO_NOT_ADOPT | +0.0079 ([-0.0025, +0.0184]) | -0.0017 ([-0.0046, +0.0013]) | 4257 |

## Probe verdict

Phase 1: 2/120 cells SIGNAL.
Phase 2: 1/4 positions ADOPT.

**This probe is a screen, not the gate.** The adoption gate is the final word on whether a feature change ships; SIGNAL is necessary but not sufficient. Run the full backtest + adoption gate before any production routing change.
