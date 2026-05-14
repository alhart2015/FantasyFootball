# Adoption Gate — `decomposed-baseline` vs `ensemble` (WR)

**Spec:** `docs/superpowers/specs/2026-05-13-wr-target-decomposition-integration-design.md`
**Plan:** `docs/superpowers/plans/2026-05-13-wr-target-decomposition-integration.md`
**Run:** `data/backtest/run_wr_decomp_20260514T012248/`
**Date:** 2026-05-13
**Branch:** `feat/wr-target-decomposition`

## Verdict

**`DO_NOT_ADOPT`** — RMSE delta is +0.0109 fpts (95% CI [-0.0080, +0.0285]); the CI brackets zero, so the result is statistically inconclusive but the point estimate shows candidate (`decomposed-baseline`) regressing relative to incumbent (`ensemble`). Spearman delta is -0.0052 (95% CI [-0.0087, -0.0018]), strictly negative CI indicating a small but consistent rank-correlation degradation.

## Per-position metrics

| Position | n_paired | RMSE delta (fpts) | RMSE 95% CI | Spearman delta | Spearman 95% CI | Verdict |
|---|---:|---:|---|---:|---|:---:|
| WR | 8402 | +0.0109 | [-0.0080, +0.0285] | -0.0052 | [-0.0087, -0.0018] | **DO_NOT_ADOPT** |

(Single-position gate; binding cell per spec §1.3.5 contingency matrix.)

## Per-year breakdown (informational)

| Year | n_paired | RMSE delta | RMSE lo | RMSE hi | Spearman delta | Spearman lo | Spearman hi |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2021 | 2099 | +0.0057 | -0.0297 | +0.0414 | -0.0071 | -0.0139 | -0.0004 |
| 2022 | 2085 | +0.0217 | -0.0157 | +0.0574 | -0.0076 | -0.0149 | +0.0003 |
| 2023 | 2193 | +0.0015 | -0.0316 | +0.0407 | -0.0029 | -0.0088 | +0.0032 |
| 2024 | 2025 | +0.0153 | -0.0207 | +0.0557 | -0.0034 | -0.0109 | +0.0042 |

## Sign convention

Positive RMSE delta = candidate (`decomposed-baseline`) has HIGHER RMSE than incumbent (`ensemble`) — candidate is WORSE.
Negative RMSE delta = candidate has LOWER RMSE — candidate is BETTER.

Positive Spearman delta = candidate has BETTER rank correlation with actuals than incumbent.
Negative Spearman delta = candidate has WORSE rank correlation.

## §1.3.5 mapping

- `ADOPT` (binding RMSE PASS + Spearman not catastrophic): flip `_PositionDispatch[Position.WR].default_model_class` from `"ensemble"` to `"decomposed-baseline"` in Phase 6.
- `MARGINAL` (RMSE PASS, Spearman fail): treat as DO_NOT_ADOPT for routing; ship infra-only.
- `DO_NOT_ADOPT`: keep WR on ensemble. Informational cell determines follow-up.
- `REGRESSION` (RMSE CI strictly > 0): full revert per PR #31 precedent.

**Outcome: DO_NOT_ADOPT.** The RMSE CI [-0.008, +0.029] brackets zero; decomposed-baseline does not demonstrate a reliable improvement over ensemble at composite-fpts level. WR production routing stays on `"ensemble"`. See informational cell (`adoption_gate_wr_decomposed_baseline_vs_baseline.md`) — decomposed-baseline does beat plain baseline cleanly (RMSE -0.0103, CI [-0.0145, -0.0060]), which means the decomposition recipe is sound but the ensemble's LightGBM contribution outpaces the small decomposition lift. Phase 6 recommended path: ship infrastructure-only; next plan slot should evaluate ensemble-child swap (replace ensemble's baseline child with decomposed-baseline child).
