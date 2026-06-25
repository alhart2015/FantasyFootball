# DFS Edge Study — per-position significance (2021-2024)

Companion to the pooled verdict in `dfs_projection_edge_2026-06-24.md` (**STOP**). Per-position head-to-head fractions with player-season **clustered-bootstrap 95% CIs** (N_BOOTSTRAP=2000, seed=20260623, DELTA=3.0 DK-base pts) — the per-position breakdown the verdict report omitted. Universe persisted to `data/dfs_universe_2021-2024.parquet` (gitignored) so re-cuts don't rebuild the 16 model cells.

**Pooled:** 0.476 (95% CI 0.456-0.495); comparable cells 13398, disagreement cells 2521, disagreement clusters 1006.

| Position | Model | Fraction | 95% CI | Verdict | Cmp cells | Disagree cells | Clusters |
|---|---|---|---|---|---|---|---|
| QB | `lightgbm-nb` | 0.431 | 0.375-0.493 | deficit (CI < 0.50) | 1286 | 320 | 152 |
| RB | `baseline` | 0.430 | 0.397-0.463 | deficit (CI < 0.50) | 4272 | 1009 | 356 |
| WR | `ensemble-decomposed` | 0.535 | 0.504-0.566 | **edge** (CI > 0.50) | 5656 | 941 | 376 |
| TE | `baseline` | 0.498 | 0.438-0.550 | not significant (straddles 0.50) | 2184 | 251 | 122 |

**Fraction** = share of disagreement cells (|ours - Sleeper| > DELTA) where our projection is strictly closer to the DK-base actual. > 0.50 = we beat Sleeper. Per-position tests are exploratory/non-confirmatory (no multiple-comparison correction); the pre-registered gate is the pooled test only.
