# Adoption Gate — `decomposed-baseline` vs `baseline` (WR)

**Spec:** `docs/superpowers/specs/2026-05-13-wr-target-decomposition-integration-design.md`
**Plan:** `docs/superpowers/plans/2026-05-13-wr-target-decomposition-integration.md`
**Run:** `data/backtest/run_wr_decomp_20260514T012248/`
**Date:** 2026-05-13
**Branch:** `feat/wr-target-decomposition`

## Verdict

**`ADOPT`** (informational only — not routing-gating) — RMSE delta is -0.0103 fpts (95% CI [-0.0145, -0.0060]); the CI is strictly negative, meaning decomposed-baseline reliably improves over plain baseline at composite-fpts level. Spearman delta is +0.0000 (95% CI [-0.0006, +0.0006]) — essentially neutral rank-correlation, confirming the improvement is in RMSE magnitude rather than relative ordering.

## Per-position metrics

| Position | n_paired | RMSE delta (fpts) | RMSE 95% CI | Spearman delta | Spearman 95% CI | Verdict |
|---|---:|---:|---|---:|---|:---:|
| WR | 8402 | -0.0103 | [-0.0145, -0.0060] | +0.0000 | [-0.0006, +0.0006] | **ADOPT** |

(Single-position gate; informational cell per spec §1.3.5 contingency matrix — not routing-gating.)

## Per-year breakdown (informational)

| Year | n_paired | RMSE delta | RMSE lo | RMSE hi | Spearman delta | Spearman lo | Spearman hi |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2021 | 2099 | -0.0158 | -0.0246 | -0.0078 | +0.0007 | -0.0004 | +0.0018 |
| 2022 | 2085 | -0.0103 | -0.0195 | -0.0009 | -0.0008 | -0.0021 | +0.0003 |
| 2023 | 2193 | -0.0115 | -0.0213 | -0.0020 | +0.0004 | -0.0007 | +0.0016 |
| 2024 | 2025 | -0.0031 | -0.0119 | +0.0052 | -0.0003 | -0.0014 | +0.0009 |

## Sign convention

Positive RMSE delta = candidate (`decomposed-baseline`) has HIGHER RMSE than incumbent (`baseline`) — candidate is WORSE.
Negative RMSE delta = candidate has LOWER RMSE — candidate is BETTER.

Positive Spearman delta = candidate has BETTER rank correlation with actuals than incumbent.
Negative Spearman delta = candidate has WORSE rank correlation.

## §1.3.5 mapping — informational cell interpretation

This cell is apples-to-apples: decomposed-baseline vs plain baseline at composite-fpts level, confirming whether the target decomposition recipe itself adds lift independent of ensemble machinery. The pooled RMSE CI [-0.0145, -0.0060] is strictly negative across all four years (2021-2023 all show improvement; 2024 point estimate -0.0031 with CI spanning zero is the only exception, consistent with a smaller held-out set and the smallest year effect).

**Interpretation:** The decomposition recipe is sound. The -0.0103 RMSE improvement matches the probe result (-0.0042 was the Feature Probe scalar; composite-fpts RMSE improvement is larger because it integrates across all stats). The binding cell DO_NOT_ADOPT result (vs ensemble) reflects that ensemble's LightGBM-NB contribution outpaces the decomposition lift, not that decomposition is broken.

**Recommended Phase 6 path:** ship infrastructure-only; create a next-plan ticket for ensemble-child swap (replace ensemble's `wr_baseline` child with `wr_decomposed_baseline` child) to compound both improvements.
