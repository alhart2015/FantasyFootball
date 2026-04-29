# Plan 8 — adoption gate re-evaluation (run_20260429T003552Z)

Re-evaluation date: 2026-04-29. Source backtest run: `data/backtest/run_20260429T003552Z`. Bootstrap configuration: paired bootstrap with 1000 reps, 95% CI, default seed (per `scripts/adoption_gate.py` Phase 3 defaults). Baseline class: `baseline`. Candidates evaluated: `lightgbm`, `lightgbm-tuned`, `lightgbm-nb`, `ensemble`. Pairing key: `(gsis_id, season, week)`. Per-candidate raw outputs: `reports/adoption_gate_<candidate>.{md,csv}`.

## 16-row summary

`rmse_delta = candidate_rmse − baseline_rmse` (more-negative = candidate wins). `spearman_delta = candidate_rho − baseline_rho` (more-positive = candidate wins). ADOPT requires `rmse_delta` 95% CI strictly below 0 AND `spearman_delta` 95% CI strictly above 0. All other outcomes are `DO_NOT_ADOPT`.

| position | candidate | verdict | rmse_delta (point) | rmse_delta 95% CI | spearman_delta (point) | spearman_delta 95% CI | n_paired |
|---|---|---|---:|---|---:|---|---:|
| QB | lightgbm | DO_NOT_ADOPT | -0.0233 | [-0.1239, +0.0758] | +0.0155 | [+0.0010, +0.0296] | 2676 |
| RB | lightgbm | DO_NOT_ADOPT | +0.1916 | [+0.1438, +0.2421] | -0.0023 | [-0.0082, +0.0028] | 5273 |
| TE | lightgbm | DO_NOT_ADOPT | +0.1553 | [+0.1096, +0.2060] | +0.0043 | [-0.0052, +0.0132] | 4257 |
| WR | lightgbm | DO_NOT_ADOPT | +0.1338 | [+0.0963, +0.1721] | +0.0045 | [-0.0012, +0.0101] | 8460 |
| QB | lightgbm-tuned | ADOPT | -0.1189 | [-0.2063, -0.0310] | +0.0177 | [+0.0046, +0.0304] | 2676 |
| RB | lightgbm-tuned | DO_NOT_ADOPT | +0.1144 | [+0.0798, +0.1520] | -0.0043 | [-0.0098, +0.0009] | 5273 |
| TE | lightgbm-tuned | DO_NOT_ADOPT | +0.0879 | [+0.0468, +0.1322] | +0.0082 | [-0.0003, +0.0170] | 4257 |
| WR | lightgbm-tuned | DO_NOT_ADOPT | +0.0711 | [+0.0397, +0.1046] | +0.0044 | [-0.0012, +0.0099] | 8460 |
| QB | lightgbm-nb | ADOPT | -0.1933 | [-0.2719, -0.1102] | +0.0183 | [+0.0045, +0.0313] | 2676 |
| RB | lightgbm-nb | DO_NOT_ADOPT | +0.0420 | [+0.0133, +0.0740] | -0.0012 | [-0.0068, +0.0039] | 5273 |
| TE | lightgbm-nb | DO_NOT_ADOPT | +0.0028 | [-0.0289, +0.0422] | +0.0071 | [-0.0014, +0.0160] | 4257 |
| WR | lightgbm-nb | DO_NOT_ADOPT | -0.0016 | [-0.0316, +0.0291] | +0.0027 | [-0.0032, +0.0080] | 8460 |
| QB | ensemble | ADOPT | -0.1760 | [-0.2274, -0.1242] | +0.0184 | [+0.0098, +0.0262] | 2676 |
| RB | ensemble | DO_NOT_ADOPT | +0.0212 | [-0.0021, +0.0455] | +0.0003 | [-0.0037, +0.0043] | 5273 |
| TE | ensemble | DO_NOT_ADOPT | -0.0208 | [-0.0454, +0.0097] | +0.0076 | [+0.0016, +0.0137] | 4257 |
| WR | ensemble | ADOPT | -0.0320 | [-0.0531, -0.0092] | +0.0069 | [+0.0028, +0.0109] | 8460 |

## Contender chains

Tie-break rule (per spec §6): when multiple candidates ADOPT for the same position, the strongest contender is the one with the most-negative `rmse_delta.point`. That candidate becomes `default_model_class` for the position.

### QB — three contenders, lightgbm-nb wins

ADOPT verdicts:
- `lightgbm-tuned` — `rmse_delta` point = -0.1189
- `lightgbm-nb` — `rmse_delta` point = **-0.1933** (most negative)
- `ensemble` — `rmse_delta` point = -0.1760

Routing decision: `default_model_class[QB] = lightgbm-nb`.

Note: this contradicts the spec §6 strong prior, which expected ensemble to win QB. Ensemble is still a clear ADOPT (CI well below zero for RMSE and well above zero for Spearman), but lightgbm-nb edges it on the RMSE point estimate by ~0.017 fpts.

### RB — no ADOPT

No candidate met both gates. Routing decision: `default_model_class[RB]` stays `baseline`.

### TE — no ADOPT

No candidate met both gates. Ensemble's `rmse_delta` point is mildly negative (-0.0208) but its CI crosses zero ([-0.0454, +0.0097]); spearman_delta is comfortably above zero. lightgbm-nb is essentially a wash on RMSE (+0.0028, CI crosses zero) and spearman_delta CI also crosses zero. Routing decision: `default_model_class[TE]` stays `baseline`.

### WR — one contender, ensemble wins

ADOPT verdicts:
- `ensemble` — `rmse_delta` point = -0.0320 (sole ADOPTer)

Routing decision: `default_model_class[WR] = ensemble`.

Note: this also contradicts the spec §6 strong prior, which expected baseline to stay default for WR. The improvement is small (~0.03 fpts RMSE) but both CIs strictly clear zero.
