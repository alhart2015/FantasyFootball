# Feature signal probe — trajectory_baseline_swap

Baseline features: C:\Users\alden\FantasyFootball\data\features
Overrides:        C:\Users\alden\FantasyFootball\data\features_probe\trajectory_probe.parquet
Drops:            age, is_rookie, volume_trend_l4_minus_prior_l4, snap_pct_change_l4_vs_prior_l4
Model class:      baseline

## Phase 1 — per-stat screening

### QB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| passing_yards | 2021 | 556 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| passing_yards | 2022 | 544 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| passing_yards | 2023 | 552 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| passing_yards | 2024 | 571 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| passing_yards | pooled | 2223 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| passing_tds | 2021 | 556 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| passing_tds | 2022 | 544 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| passing_tds | 2023 | 552 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| passing_tds | 2024 | 571 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| passing_tds | pooled | 2223 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| interceptions | 2021 | 556 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| interceptions | 2022 | 544 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| interceptions | 2023 | 552 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| interceptions | 2024 | 571 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| interceptions | pooled | 2223 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2021 | 556 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2022 | 544 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2023 | 552 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2024 | 571 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | pooled | 2223 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2021 | 556 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2022 | 544 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2023 | 552 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2024 | 571 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | pooled | 2223 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2021 | 556 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2022 | 544 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2023 | 552 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2024 | 571 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | pooled | 2223 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |

### RB

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| rushing_yards | 2021 | 969 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2022 | 913 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2023 | 900 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2024 | 941 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | pooled | 3723 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2021 | 969 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2022 | 913 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2023 | 900 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2024 | 941 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | pooled | 3723 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receptions | 2021 | 969 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receptions | 2022 | 913 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receptions | 2023 | 900 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receptions | 2024 | 941 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receptions | pooled | 3723 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_yards | 2021 | 969 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_yards | 2022 | 913 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_yards | 2023 | 900 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_yards | 2024 | 941 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_yards | pooled | 3723 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2021 | 969 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2022 | 913 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2023 | 900 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2024 | 941 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | pooled | 3723 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2021 | 969 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2022 | 913 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2023 | 900 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2024 | 941 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | pooled | 3723 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |

### WR

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 1619 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receptions | 2022 | 1609 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receptions | 2023 | 1600 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receptions | 2024 | 1563 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receptions | pooled | 6391 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_yards | 2021 | 1619 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_yards | 2022 | 1609 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_yards | 2023 | 1600 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_yards | 2024 | 1563 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_yards | pooled | 6391 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2021 | 1619 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2022 | 1609 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2023 | 1600 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2024 | 1563 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | pooled | 6391 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2021 | 1619 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2022 | 1609 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2023 | 1600 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2024 | 1563 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | pooled | 6391 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2021 | 1619 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2022 | 1609 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2023 | 1600 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2024 | 1563 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | pooled | 6391 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2021 | 1619 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2022 | 1609 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2023 | 1600 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2024 | 1563 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | pooled | 6391 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |

### TE

| stat | year | n_paired | rmse_delta | ci_lo | ci_hi | r_squared_delta | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| receptions | 2021 | 633 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receptions | 2022 | 704 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receptions | 2023 | 719 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receptions | 2024 | 691 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receptions | pooled | 2747 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_yards | 2021 | 633 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_yards | 2022 | 704 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_yards | 2023 | 719 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_yards | 2024 | 691 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_yards | pooled | 2747 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2021 | 633 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2022 | 704 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2023 | 719 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | 2024 | 691 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| receiving_tds | pooled | 2747 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2021 | 633 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2022 | 704 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2023 | 719 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | 2024 | 691 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_yards | pooled | 2747 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2021 | 633 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2022 | 704 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2023 | 719 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | 2024 | 691 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| rushing_tds | pooled | 2747 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2021 | 633 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2022 | 704 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2023 | 719 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | 2024 | 691 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |
| fumbles_lost | pooled | 2747 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | NULL |

## Phase 1 verdict

0/120 cells SIGNAL, 120/120 NULL, 0/120 REGRESSION.
No SIGNAL cells — Phase 2 skipped. Probe predicts the adoption gate would return DO_NOT_ADOPT.

## Probe verdict

Phase 1: 0/120 cells SIGNAL.
Phase 2: not run.

**This probe is a screen, not the gate.** The adoption gate is the final word on whether a feature change ships; SIGNAL is necessary but not sufficient. Run the full backtest + adoption gate before any production routing change.
