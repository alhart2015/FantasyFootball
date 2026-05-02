# Feature signal probe — pbp_redzone_swap

Baseline features: data\features
Overrides:        data\features_probe\pbp_redzone.parquet
Drops:            opp_allowed_qb_fppg_l4, opp_allowed_rb_fppg_l4, opp_allowed_wr_fppg_l4, opp_allowed_te_fppg_l4
Model class:      baseline

## Phase 1 — per-stat screening

### QB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| passing_yards | 2021 | 609 | -0.2374 | -0.8238 | +0.3885 | +0.0048 | NULL |
| passing_yards | 2022 | 603 | +0.2448 | -0.2716 | +0.7801 | +0.0046 | NULL |
| passing_yards | 2023 | 603 | +0.2413 | -0.2232 | +0.6603 | +0.0023 | NULL |
| passing_yards | 2024 | 627 | +0.1374 | -0.2119 | +0.4507 | +0.0016 | NULL |
| passing_yards | pooled | 2442 | +0.1089 | -0.1568 | +0.3811 | +0.0048 | NULL |
| passing_tds | 2021 | 609 | +0.0037 | -0.0036 | +0.0110 | +0.0025 | NULL |
| passing_tds | 2022 | 603 | +0.0024 | -0.0022 | +0.0064 | +0.0019 | NULL |
| passing_tds | 2023 | 603 | +0.0014 | -0.0027 | +0.0054 | +0.0011 | NULL |
| passing_tds | 2024 | 627 | -0.0013 | -0.0039 | +0.0012 | +0.0009 | NULL |
| passing_tds | pooled | 2442 | +0.0018 | -0.0016 | +0.0050 | +0.0025 | NULL |
| interceptions | 2021 | 609 | -0.0009 | -0.0018 | +0.0000 | +0.0005 | NULL |
| interceptions | 2022 | 603 | +0.0045 | +0.0017 | +0.0073 | +0.0035 | NULL |
| interceptions | 2023 | 603 | +0.0003 | -0.0012 | +0.0018 | +0.0006 | NULL |
| interceptions | 2024 | 627 | -0.0001 | -0.0012 | +0.0009 | +0.0005 | NULL |
| interceptions | pooled | 2442 | -0.0003 | -0.0007 | +0.0001 | +0.0005 | NULL |
| rushing_yards | 2021 | 609 | +0.0662 | +0.0227 | +0.1113 | -0.0020 | REGRESSION |
| rushing_yards | 2022 | 603 | -0.0226 | -0.0478 | +0.0030 | +0.0003 | NULL |
| rushing_yards | 2023 | 603 | -0.0180 | -0.0770 | +0.0407 | -0.0001 | NULL |
| rushing_yards | 2024 | 627 | -0.0361 | -0.0733 | +0.0007 | +0.0006 | NULL |
| rushing_yards | pooled | 2442 | +0.0168 | -0.0080 | +0.0382 | -0.0020 | NULL |
| rushing_tds | 2021 | 609 | +0.0001 | -0.0014 | +0.0017 | +0.0022 | NULL |
| rushing_tds | 2022 | 603 | +0.0006 | -0.0003 | +0.0016 | +0.0011 | NULL |
| rushing_tds | 2023 | 603 | -0.0001 | -0.0006 | +0.0004 | +0.0004 | NULL |
| rushing_tds | 2024 | 627 | +0.0004 | -0.0001 | +0.0009 | +0.0004 | NULL |
| rushing_tds | pooled | 2442 | +0.0006 | -0.0000 | +0.0013 | +0.0022 | NULL |
| fumbles_lost | 2021 | 609 | +0.0011 | +0.0002 | +0.0020 | +0.0006 | NULL |
| fumbles_lost | 2022 | 603 | +0.0002 | -0.0000 | +0.0005 | +0.0001 | NULL |
| fumbles_lost | 2023 | 603 | -0.0001 | -0.0008 | +0.0006 | +0.0005 | NULL |
| fumbles_lost | 2024 | 627 | -0.0003 | -0.0010 | +0.0004 | +0.0005 | NULL |
| fumbles_lost | pooled | 2442 | -0.0000 | -0.0005 | +0.0004 | +0.0006 | NULL |

### RB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| rushing_yards | 2021 | 1068 | +0.1050 | +0.0025 | +0.2112 | +0.0015 | REGRESSION |
| rushing_yards | 2022 | 1019 | -0.0380 | -0.1041 | +0.0251 | +0.0004 | NULL |
| rushing_yards | 2023 | 982 | -0.0231 | -0.0997 | +0.0616 | +0.0009 | NULL |
| rushing_yards | 2024 | 1050 | -0.0299 | -0.0979 | +0.0408 | +0.0009 | NULL |
| rushing_yards | pooled | 4119 | +0.0053 | -0.0456 | +0.0598 | +0.0015 | NULL |
| rushing_tds | 2021 | 1068 | +0.0008 | +0.0001 | +0.0017 | +0.0014 | NULL |
| rushing_tds | 2022 | 1019 | -0.0001 | -0.0010 | +0.0009 | +0.0007 | NULL |
| rushing_tds | 2023 | 982 | -0.0010 | -0.0022 | +0.0001 | +0.0000 | NULL |
| rushing_tds | 2024 | 1050 | -0.0005 | -0.0012 | +0.0003 | +0.0002 | NULL |
| rushing_tds | pooled | 4119 | +0.0004 | -0.0001 | +0.0008 | +0.0014 | NULL |
| receptions | 2021 | 1068 | -0.0016 | -0.0032 | -0.0002 | +0.0002 | NULL |
| receptions | 2022 | 1019 | +0.0029 | -0.0017 | +0.0070 | +0.0008 | NULL |
| receptions | 2023 | 982 | +0.0005 | -0.0021 | +0.0031 | +0.0004 | NULL |
| receptions | 2024 | 1050 | -0.0012 | -0.0037 | +0.0013 | +0.0004 | NULL |
| receptions | pooled | 4119 | +0.0001 | -0.0007 | +0.0010 | +0.0002 | NULL |
| receiving_yards | 2021 | 1068 | -0.0238 | -0.0635 | +0.0179 | +0.0007 | NULL |
| receiving_yards | 2022 | 1019 | +0.0557 | -0.0095 | +0.1155 | +0.0019 | NULL |
| receiving_yards | 2023 | 982 | +0.0576 | +0.0112 | +0.1081 | +0.0008 | REGRESSION |
| receiving_yards | 2024 | 1050 | +0.0212 | +0.0039 | +0.0367 | +0.0002 | NULL |
| receiving_yards | pooled | 4119 | +0.0255 | +0.0050 | +0.0470 | +0.0007 | NULL |
| receiving_tds | 2021 | 1068 | -0.0003 | -0.0009 | +0.0003 | +0.0010 | NULL |
| receiving_tds | 2022 | 1019 | +0.0006 | -0.0006 | +0.0018 | +0.0024 | NULL |
| receiving_tds | 2023 | 982 | +0.0006 | -0.0008 | +0.0019 | -0.0033 | NULL |
| receiving_tds | 2024 | 1050 | +0.0001 | -0.0004 | +0.0006 | +0.0011 | NULL |
| receiving_tds | pooled | 4119 | +0.0000 | -0.0002 | +0.0003 | +0.0010 | NULL |
| fumbles_lost | 2021 | 1068 | +0.0000 | -0.0002 | +0.0003 | +0.0007 | NULL |
| fumbles_lost | 2022 | 1019 | -0.0001 | -0.0003 | +0.0001 | +0.0006 | NULL |
| fumbles_lost | 2023 | 982 | -0.0000 | -0.0002 | +0.0002 | +0.0008 | NULL |
| fumbles_lost | 2024 | 1050 | -0.0000 | -0.0002 | +0.0002 | +0.0008 | NULL |
| fumbles_lost | pooled | 4119 | +0.0000 | -0.0001 | +0.0001 | +0.0007 | NULL |

### WR

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 1781 | +0.0071 | +0.0013 | +0.0132 | +0.0015 | NULL |
| receptions | 2022 | 1780 | +0.0009 | -0.0020 | +0.0035 | +0.0005 | NULL |
| receptions | 2023 | 1747 | +0.0020 | -0.0008 | +0.0050 | +0.0004 | NULL |
| receptions | 2024 | 1711 | +0.0002 | -0.0015 | +0.0019 | +0.0002 | NULL |
| receptions | pooled | 7019 | +0.0039 | +0.0011 | +0.0067 | +0.0015 | NULL |
| receiving_yards | 2021 | 1781 | +0.0113 | -0.0521 | +0.0784 | +0.0008 | NULL |
| receiving_yards | 2022 | 1780 | +0.0126 | -0.0323 | +0.0598 | +0.0006 | NULL |
| receiving_yards | 2023 | 1747 | +0.0188 | -0.0240 | +0.0565 | +0.0004 | NULL |
| receiving_yards | 2024 | 1711 | +0.0003 | -0.0302 | +0.0308 | +0.0003 | NULL |
| receiving_yards | pooled | 7019 | +0.0141 | -0.0162 | +0.0433 | +0.0008 | NULL |
| receiving_tds | 2021 | 1781 | +0.0015 | +0.0000 | +0.0031 | +0.0019 | NULL |
| receiving_tds | 2022 | 1780 | +0.0002 | -0.0006 | +0.0010 | +0.0005 | NULL |
| receiving_tds | 2023 | 1747 | +0.0002 | -0.0003 | +0.0007 | +0.0004 | NULL |
| receiving_tds | 2024 | 1711 | -0.0000 | -0.0004 | +0.0003 | +0.0003 | NULL |
| receiving_tds | pooled | 7019 | +0.0010 | +0.0004 | +0.0017 | +0.0019 | NULL |
| rushing_yards | 2021 | 1781 | -0.0008 | -0.0059 | +0.0054 | +0.0010 | NULL |
| rushing_yards | 2022 | 1780 | +0.0030 | -0.0040 | +0.0100 | +0.0010 | NULL |
| rushing_yards | 2023 | 1747 | +0.0014 | -0.0051 | +0.0086 | +0.0006 | NULL |
| rushing_yards | 2024 | 1711 | +0.0081 | +0.0024 | +0.0137 | +0.0005 | NULL |
| rushing_yards | pooled | 7019 | +0.0033 | +0.0001 | +0.0065 | +0.0010 | NULL |
| rushing_tds | 2021 | 1781 | -0.0000 | -0.0001 | +0.0000 | +0.0003 | NULL |
| rushing_tds | 2022 | 1780 | +0.0001 | -0.0000 | +0.0002 | +0.0006 | NULL |
| rushing_tds | 2023 | 1747 | +0.0000 | -0.0000 | +0.0001 | +0.0002 | NULL |
| rushing_tds | 2024 | 1711 | +0.0000 | -0.0001 | +0.0001 | +0.0003 | NULL |
| rushing_tds | pooled | 7019 | +0.0000 | +0.0000 | +0.0001 | +0.0003 | NULL |
| fumbles_lost | 2021 | 1781 | -0.0000 | -0.0001 | +0.0001 | +0.0005 | NULL |
| fumbles_lost | 2022 | 1780 | -0.0001 | -0.0001 | +0.0000 | +0.0004 | NULL |
| fumbles_lost | 2023 | 1747 | +0.0003 | -0.0001 | +0.0007 | +0.0037 | NULL |
| fumbles_lost | 2024 | 1711 | +0.0003 | -0.0000 | +0.0006 | +0.0024 | NULL |
| fumbles_lost | pooled | 7019 | -0.0000 | -0.0000 | +0.0000 | +0.0005 | NULL |

### TE

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 716 | +0.0045 | -0.0009 | +0.0098 | +0.0015 | NULL |
| receptions | 2022 | 790 | +0.0064 | +0.0005 | +0.0124 | +0.0000 | NULL |
| receptions | 2023 | 784 | -0.0003 | -0.0028 | +0.0025 | +0.0000 | NULL |
| receptions | 2024 | 771 | +0.0007 | -0.0022 | +0.0033 | +0.0001 | NULL |
| receptions | pooled | 3061 | +0.0005 | -0.0022 | +0.0030 | +0.0015 | NULL |
| receiving_yards | 2021 | 716 | -0.0163 | -0.1134 | +0.0813 | +0.0016 | NULL |
| receiving_yards | 2022 | 790 | +0.1192 | +0.0126 | +0.2226 | +0.0017 | REGRESSION |
| receiving_yards | 2023 | 784 | -0.0505 | -0.1194 | +0.0165 | +0.0010 | NULL |
| receiving_yards | 2024 | 771 | -0.0145 | -0.0978 | +0.0635 | +0.0015 | NULL |
| receiving_yards | pooled | 3061 | -0.0188 | -0.0742 | +0.0423 | +0.0016 | NULL |
| receiving_tds | 2021 | 716 | +0.0008 | -0.0003 | +0.0019 | +0.0006 | NULL |
| receiving_tds | 2022 | 790 | +0.0000 | -0.0001 | +0.0002 | +0.0000 | NULL |
| receiving_tds | 2023 | 784 | +0.0000 | -0.0001 | +0.0001 | +0.0000 | NULL |
| receiving_tds | 2024 | 771 | +0.0001 | -0.0002 | +0.0004 | +0.0001 | NULL |
| receiving_tds | pooled | 3061 | +0.0002 | -0.0002 | +0.0007 | +0.0006 | NULL |
| rushing_yards | 2021 | 716 | +0.0002 | -0.0051 | +0.0052 | +0.0019 | NULL |
| rushing_yards | 2022 | 790 | +0.0017 | -0.0019 | +0.0048 | +0.0017 | NULL |
| rushing_yards | 2023 | 784 | -0.0000 | -0.0034 | +0.0033 | +0.0012 | NULL |
| rushing_yards | 2024 | 771 | -0.0004 | -0.0011 | +0.0028 | +0.0012 | NULL |
| rushing_yards | pooled | 3061 | -0.0007 | -0.0018 | +0.0026 | +0.0019 | NULL |
| rushing_tds | 2021 | 716 | +0.0000 | -0.0000 | +0.0002 | +0.0004 | NULL |
| rushing_tds | 2022 | 790 | -0.0000 | -0.0000 | +0.0000 | +0.0003 | NULL |
| rushing_tds | 2023 | 784 | -0.0001 | -0.0002 | -0.0000 | +0.0004 | NULL |
| rushing_tds | 2024 | 771 | -0.0000 | -0.0001 | +0.0000 | +0.0004 | NULL |
| rushing_tds | pooled | 3061 | +0.0000 | +0.0000 | +0.0000 | +0.0004 | NULL |
| fumbles_lost | 2021 | 716 | +0.0004 | +0.0002 | +0.0007 | +0.0012 | NULL |
| fumbles_lost | 2022 | 790 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2023 | 784 | +0.0001 | +0.0000 | +0.0001 | +0.0001 | NULL |
| fumbles_lost | 2024 | 771 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | pooled | 3061 | +0.0002 | +0.0001 | +0.0003 | +0.0012 | NULL |

## Phase 1 verdict

0/120 cells SIGNAL, 116/120 NULL, 4/120 REGRESSION.
No SIGNAL cells — Phase 2 skipped. Probe predicts the adoption gate would return DO_NOT_ADOPT.

## Probe verdict

Phase 1: 0/120 cells SIGNAL.
Phase 2: not run.

**This probe is a screen, not the gate.** The adoption gate is the final word on whether a feature change ships; SIGNAL is necessary but not sufficient. Run the full backtest + adoption gate before any production routing change.
