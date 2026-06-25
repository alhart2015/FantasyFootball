# DFS Projection Edge Study — verdict (2021-2024)

**VERDICT: STOP**

## Primary test (home-grown-only vs Sleeper, pooled, DK base)
- head-to-head fraction: 0.476 (95% CI 0.456 to 0.495), clustered by player-season
- by-week robustness CI: 0.456 to 0.496
- bonus-sensitivity CI (actuals+bonus): 0.451 to 0.489
- ranking-skill diff (Spearman, ours-Sleeper): -0.031 (95% CI -0.038 to -0.023), clustered by player-season
- disagreement clusters (player-seasons): 1006
- pooled (count-weighted) 0.476 vs equal-weight 0.473

## Per-position (EXPLORATORY — non-confirmatory)
- QB: 0.431
- RB: 0.430
- TE: 0.498
- WR: 0.535

## Exploratory 50/50 blend (non-confirmatory)
- verdict INCONCLUSIVE; fraction 0.543 (0.485 to 0.600)

## Coverage & inclusion disagreement
- inclusion: {'ours_only': 898, 'sleeper_only': 556, 'both': 13398}
- coverage: {'universe_cells': 13398, 'universe_wk14_18': 3761, 'universe_wk1_3': 2427, 'universe_wk4_13': 7210}

## Limitations
- Sleeper-alone is a softer proxy than the true DFS field (necessary, not sufficient — spec §4.3/§6.1). Bonuses excluded from the projection comparison (conservative; spec §6.2).
