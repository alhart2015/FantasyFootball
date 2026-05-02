# Feature signal probe — pbp_redzone_augment

Baseline features: data\features
Overrides:        data\features_probe\pbp_redzone.parquet
Drops:            (none)
Model class:      baseline

## Phase 1 — per-stat screening

### QB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| passing_yards | 2021 | 547 | -0.2826 | -0.6584 | +0.0888 | +0.0025 | NULL |
| passing_yards | 2022 | 537 | +0.1755 | -0.3120 | +0.7022 | +0.0049 | NULL |
| passing_yards | 2023 | 536 | +0.1951 | -0.1311 | +0.5764 | +0.0019 | NULL |
| passing_yards | 2024 | 557 | +0.0293 | -0.2655 | +0.3483 | +0.0018 | NULL |
| passing_yards | pooled | 2177 | -0.0135 | -0.1954 | +0.1693 | +0.0025 | NULL |
| passing_tds | 2021 | 547 | +0.0032 | -0.0021 | +0.0089 | +0.0022 | NULL |
| passing_tds | 2022 | 537 | +0.0050 | +0.0006 | +0.0095 | +0.0027 | NULL |
| passing_tds | 2023 | 536 | +0.0005 | -0.0021 | +0.0034 | +0.0003 | NULL |
| passing_tds | 2024 | 557 | -0.0009 | -0.0024 | +0.0007 | +0.0004 | NULL |
| passing_tds | pooled | 2177 | +0.0020 | -0.0006 | +0.0045 | +0.0022 | NULL |
| interceptions | 2021 | 547 | -0.0005 | -0.0010 | +0.0001 | +0.0006 | NULL |
| interceptions | 2022 | 537 | +0.0051 | +0.0019 | +0.0082 | +0.0039 | NULL |
| interceptions | 2023 | 536 | +0.0011 | -0.0007 | +0.0028 | +0.0006 | NULL |
| interceptions | 2024 | 557 | -0.0011 | -0.0026 | +0.0004 | -0.0014 | NULL |
| interceptions | pooled | 2177 | -0.0002 | -0.0005 | +0.0000 | +0.0006 | NULL |
| rushing_yards | 2021 | 547 | +0.0751 | +0.0166 | +0.1315 | -0.0011 | REGRESSION |
| rushing_yards | 2022 | 537 | -0.0553 | -0.1049 | -0.0035 | -0.0011 | SIGNAL |
| rushing_yards | 2023 | 536 | -0.0012 | -0.0403 | +0.0381 | +0.0006 | NULL |
| rushing_yards | 2024 | 557 | -0.0523 | -0.1041 | +0.0008 | -0.0004 | NULL |
| rushing_yards | pooled | 2177 | +0.0101 | -0.0217 | +0.0407 | -0.0011 | NULL |
| rushing_tds | 2021 | 547 | +0.0001 | -0.0015 | +0.0018 | +0.0021 | NULL |
| rushing_tds | 2022 | 537 | +0.0006 | -0.0009 | +0.0022 | -0.0016 | NULL |
| rushing_tds | 2023 | 536 | -0.0002 | -0.0007 | +0.0003 | +0.0004 | NULL |
| rushing_tds | 2024 | 557 | +0.0005 | -0.0003 | +0.0013 | +0.0008 | NULL |
| rushing_tds | pooled | 2177 | +0.0005 | -0.0003 | +0.0012 | +0.0021 | NULL |
| fumbles_lost | 2021 | 547 | +0.0008 | +0.0001 | +0.0014 | +0.0004 | NULL |
| fumbles_lost | 2022 | 537 | +0.0004 | +0.0001 | +0.0008 | +0.0001 | NULL |
| fumbles_lost | 2023 | 536 | -0.0000 | -0.0007 | +0.0008 | +0.0006 | NULL |
| fumbles_lost | 2024 | 557 | -0.0003 | -0.0011 | +0.0004 | +0.0005 | NULL |
| fumbles_lost | pooled | 2177 | -0.0001 | -0.0005 | +0.0002 | +0.0004 | NULL |

### RB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| rushing_yards | 2021 | 949 | +0.0826 | -0.0260 | +0.1924 | +0.0017 | NULL |
| rushing_yards | 2022 | 902 | +0.0149 | -0.0552 | +0.0820 | +0.0008 | NULL |
| rushing_yards | 2023 | 872 | +0.0010 | -0.0731 | +0.0781 | +0.0005 | NULL |
| rushing_yards | 2024 | 923 | -0.0023 | -0.0585 | +0.0543 | +0.0005 | NULL |
| rushing_yards | pooled | 3646 | +0.0442 | -0.0111 | +0.1049 | +0.0017 | NULL |
| rushing_tds | 2021 | 949 | +0.0012 | -0.0001 | +0.0025 | +0.0016 | NULL |
| rushing_tds | 2022 | 902 | +0.0002 | -0.0006 | +0.0011 | +0.0006 | NULL |
| rushing_tds | 2023 | 872 | +0.0004 | -0.0003 | +0.0011 | +0.0004 | NULL |
| rushing_tds | 2024 | 923 | -0.0004 | -0.0011 | +0.0003 | -0.0002 | NULL |
| rushing_tds | pooled | 3646 | +0.0010 | +0.0003 | +0.0017 | +0.0016 | NULL |
| receptions | 2021 | 949 | -0.0006 | -0.0031 | +0.0020 | +0.0002 | NULL |
| receptions | 2022 | 902 | +0.0036 | -0.0002 | +0.0073 | +0.0007 | NULL |
| receptions | 2023 | 872 | +0.0011 | -0.0007 | +0.0029 | +0.0002 | NULL |
| receptions | 2024 | 923 | -0.0021 | -0.0042 | -0.0002 | +0.0002 | NULL |
| receptions | pooled | 3646 | +0.0010 | -0.0003 | +0.0022 | +0.0002 | NULL |
| receiving_yards | 2021 | 949 | -0.0008 | -0.0428 | +0.0443 | +0.0006 | NULL |
| receiving_yards | 2022 | 902 | +0.0295 | -0.0292 | +0.0842 | +0.0011 | NULL |
| receiving_yards | 2023 | 872 | +0.0633 | +0.0239 | +0.1033 | +0.0006 | REGRESSION |
| receiving_yards | 2024 | 923 | +0.0140 | +0.0011 | +0.0281 | +0.0001 | NULL |
| receiving_yards | pooled | 3646 | +0.0210 | -0.0002 | +0.0400 | +0.0006 | NULL |
| receiving_tds | 2021 | 949 | -0.0003 | -0.0010 | +0.0003 | +0.0014 | NULL |
| receiving_tds | 2022 | 902 | +0.0000 | -0.0012 | +0.0015 | -0.0001 | NULL |
| receiving_tds | 2023 | 872 | +0.0004 | -0.0012 | +0.0020 | -0.0033 | NULL |
| receiving_tds | 2024 | 923 | -0.0005 | -0.0017 | +0.0006 | -0.0017 | NULL |
| receiving_tds | pooled | 3646 | +0.0000 | -0.0003 | +0.0003 | +0.0014 | NULL |
| fumbles_lost | 2021 | 949 | +0.0000 | -0.0002 | +0.0002 | +0.0006 | NULL |
| fumbles_lost | 2022 | 902 | -0.0001 | -0.0003 | +0.0001 | +0.0006 | NULL |
| fumbles_lost | 2023 | 872 | -0.0001 | -0.0003 | +0.0001 | +0.0009 | NULL |
| fumbles_lost | 2024 | 923 | -0.0001 | -0.0003 | +0.0001 | +0.0010 | NULL |
| fumbles_lost | pooled | 3646 | -0.0000 | -0.0001 | +0.0001 | +0.0006 | NULL |

### WR

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 1602 | +0.0031 | -0.0020 | +0.0087 | +0.0009 | NULL |
| receptions | 2022 | 1598 | +0.0017 | -0.0012 | +0.0044 | +0.0005 | NULL |
| receptions | 2023 | 1571 | +0.0016 | -0.0010 | +0.0041 | +0.0003 | NULL |
| receptions | 2024 | 1528 | +0.0004 | -0.0013 | +0.0021 | +0.0002 | NULL |
| receptions | pooled | 6299 | +0.0025 | +0.0004 | +0.0049 | +0.0009 | NULL |
| receiving_yards | 2021 | 1602 | -0.0090 | -0.0550 | +0.0379 | +0.0004 | NULL |
| receiving_yards | 2022 | 1598 | +0.0137 | -0.0261 | +0.0546 | +0.0005 | NULL |
| receiving_yards | 2023 | 1571 | +0.0129 | -0.0201 | +0.0449 | +0.0003 | NULL |
| receiving_yards | 2024 | 1528 | +0.0033 | -0.0263 | +0.0318 | +0.0002 | NULL |
| receiving_yards | pooled | 6299 | +0.0059 | -0.0144 | +0.0277 | +0.0004 | NULL |
| receiving_tds | 2021 | 1602 | +0.0010 | -0.0003 | +0.0024 | +0.0013 | NULL |
| receiving_tds | 2022 | 1598 | +0.0007 | +0.0000 | +0.0013 | +0.0009 | NULL |
| receiving_tds | 2023 | 1571 | +0.0001 | -0.0002 | +0.0005 | +0.0003 | NULL |
| receiving_tds | 2024 | 1528 | -0.0000 | -0.0003 | +0.0003 | +0.0002 | NULL |
| receiving_tds | pooled | 6299 | +0.0007 | +0.0002 | +0.0013 | +0.0013 | NULL |
| rushing_yards | 2021 | 1602 | -0.0025 | -0.0135 | +0.0103 | -0.0006 | NULL |
| rushing_yards | 2022 | 1598 | +0.0040 | -0.0029 | +0.0114 | +0.0010 | NULL |
| rushing_yards | 2023 | 1571 | -0.0008 | -0.0068 | +0.0056 | +0.0005 | NULL |
| rushing_yards | 2024 | 1528 | +0.0097 | +0.0029 | +0.0164 | +0.0006 | NULL |
| rushing_yards | pooled | 6299 | -0.0075 | -0.0151 | -0.0005 | -0.0006 | NULL |
| rushing_tds | 2021 | 1602 | -0.0000 | -0.0001 | +0.0000 | +0.0004 | NULL |
| rushing_tds | 2022 | 1598 | +0.0001 | -0.0000 | +0.0002 | +0.0006 | NULL |
| rushing_tds | 2023 | 1571 | +0.0000 | -0.0000 | +0.0001 | +0.0004 | NULL |
| rushing_tds | 2024 | 1528 | -0.0000 | -0.0001 | +0.0000 | +0.0003 | NULL |
| rushing_tds | pooled | 6299 | +0.0000 | -0.0000 | +0.0001 | +0.0004 | NULL |
| fumbles_lost | 2021 | 1602 | -0.0000 | -0.0002 | +0.0001 | +0.0008 | NULL |
| fumbles_lost | 2022 | 1598 | -0.0000 | -0.0001 | +0.0001 | +0.0006 | NULL |
| fumbles_lost | 2023 | 1571 | +0.0001 | -0.0000 | +0.0002 | +0.0007 | NULL |
| fumbles_lost | 2024 | 1528 | -0.0000 | -0.0001 | +0.0000 | +0.0003 | NULL |
| fumbles_lost | pooled | 6299 | +0.0001 | -0.0000 | +0.0001 | +0.0008 | NULL |

### TE

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 619 | +0.0058 | +0.0014 | +0.0101 | +0.0010 | NULL |
| receptions | 2022 | 698 | +0.0054 | +0.0009 | +0.0100 | -0.0001 | NULL |
| receptions | 2023 | 699 | -0.0011 | -0.0040 | +0.0018 | -0.0003 | NULL |
| receptions | 2024 | 676 | +0.0001 | -0.0033 | +0.0037 | +0.0001 | NULL |
| receptions | pooled | 2692 | -0.0014 | -0.0036 | +0.0009 | +0.0010 | NULL |
| receiving_yards | 2021 | 619 | -0.0062 | -0.0188 | +0.0060 | +0.0003 | NULL |
| receiving_yards | 2022 | 698 | +0.0372 | -0.0627 | +0.1252 | +0.0020 | NULL |
| receiving_yards | 2023 | 699 | -0.1130 | -0.1780 | -0.0485 | +0.0007 | SIGNAL |
| receiving_yards | 2024 | 676 | -0.0036 | -0.0975 | +0.1014 | +0.0020 | NULL |
| receiving_yards | pooled | 2692 | -0.0043 | -0.0098 | +0.0012 | +0.0003 | NULL |
| receiving_tds | 2021 | 619 | +0.0008 | -0.0011 | +0.0025 | +0.0011 | NULL |
| receiving_tds | 2022 | 698 | +0.0001 | -0.0004 | +0.0006 | +0.0003 | NULL |
| receiving_tds | 2023 | 699 | -0.0002 | -0.0006 | +0.0002 | +0.0001 | NULL |
| receiving_tds | 2024 | 676 | +0.0002 | -0.0004 | +0.0008 | +0.0003 | NULL |
| receiving_tds | pooled | 2692 | +0.0002 | -0.0005 | +0.0008 | +0.0011 | NULL |
| rushing_yards | 2021 | 619 | -0.0003 | -0.0062 | +0.0058 | +0.0020 | NULL |
| rushing_yards | 2022 | 698 | +0.0017 | -0.0028 | +0.0055 | +0.0019 | NULL |
| rushing_yards | 2023 | 699 | -0.0015 | -0.0051 | +0.0028 | +0.0014 | NULL |
| rushing_yards | 2024 | 676 | +0.0014 | -0.0006 | +0.0041 | +0.0014 | NULL |
| rushing_yards | pooled | 2692 | +0.0004 | -0.0021 | +0.0026 | +0.0020 | NULL |
| rushing_tds | 2021 | 619 | -0.0000 | -0.0000 | +0.0000 | +0.0003 | NULL |
| rushing_tds | 2022 | 698 | -0.0000 | -0.0001 | +0.0000 | +0.0005 | NULL |
| rushing_tds | 2023 | 699 | -0.0000 | -0.0001 | +0.0001 | +0.0007 | NULL |
| rushing_tds | 2024 | 676 | -0.0000 | -0.0001 | +0.0001 | +0.0007 | NULL |
| rushing_tds | pooled | 2692 | -0.0000 | -0.0000 | +0.0000 | +0.0003 | NULL |
| fumbles_lost | 2021 | 619 | +0.0006 | +0.0003 | +0.0009 | +0.0014 | NULL |
| fumbles_lost | 2022 | 698 | -0.0000 | -0.0000 | -0.0000 | +0.0000 | NULL |
| fumbles_lost | 2023 | 699 | +0.0001 | +0.0000 | +0.0002 | +0.0002 | NULL |
| fumbles_lost | 2024 | 676 | +0.0000 | -0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | pooled | 2692 | +0.0002 | +0.0001 | +0.0004 | +0.0014 | NULL |

## Phase 1 verdict

2/120 cells SIGNAL, 116/120 NULL, 2/120 REGRESSION.
Per-year SIGNAL only — no pooled SIGNAL — Phase 2 skipped per default gating. Pass --force-composite to run Phase 2 unconditionally (e.g., to test whether a non-Ridge --model class extracts signal Phase 1's Ridge screen missed).

## Probe verdict

Phase 1: 2/120 cells SIGNAL.
Phase 2: not run.

**This probe is a screen, not the gate.** The adoption gate is the final word on whether a feature change ships; SIGNAL is necessary but not sufficient. Run the full backtest + adoption gate before any production routing change.
