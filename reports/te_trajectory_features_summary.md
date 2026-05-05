# TE Trajectory Features Integration — Summary Report

**Status:** **ADOPT** on `(LightGBMNbModel, TE)` — composite RMSE delta **-0.0090 fpts** ([-0.0171, -0.0013]).
**Branch:** `feat/te-trajectory-features`
**Spec:** `docs/superpowers/specs/2026-05-04-te-trajectory-features-design.md`
**Plan:** `docs/superpowers/plans/2026-05-04-te-trajectory-features.md`
**Date:** 2026-05-04

## Decision

**Ship as designed.** The 4 trajectory features (`age`, `is_rookie`, `volume_trend_l4_minus_prior_l4`, `snap_pct_change_l4_vs_prior_l4`) are integrated into `TeFeaturesSchema` + `build_te_features` per spec §1.1; the dual-run adoption gate confirms `(LightGBMNbModel, TE)` ADOPT'd at -0.0090 fpts (CI strictly negative). The `(BaselineModel, TE)` contingency cell returned DO_NOT_ADOPT (point estimate -0.0100, CI [-0.0280, +0.0093] brackets zero) — **not REGRESSION**, so the spec §1.3.5 modified-shape branch does **not** fire. `_TE_FEATURE_COLUMNS` extension stays in.

## Binding-cell shift from PR #21 / PR #26 — first non-default-class integration

PR #21 (RB) and PR #26 (WR) bound the ship decision on `(BaselineModel, position)` because baseline was each position's `default_model_class`. For TE, baseline is still the production default but the PR #25 trajectory probe ADOPT'd TE only under `lightgbm-nb` (-0.0107 fpts), not under baseline. Per spec §1, this PR binds on `(LightGBMNbModel, TE)` — the cell where the probe's signal lives. **TE production routing stays on `baseline`**; flipping `_PositionDispatch[TE].default_model_class` is a deferred cross-class follow-up (see "Cross-class deferred follow-up" below).

## Probe-vs-gate calibration

| Source | Composite RMSE Δ on (LightGBMNbModel, TE) augment | 95% CI |
|---|---:|---|
| PR #25 probe (predicted) | -0.0107 | [-0.0191, -0.0028] |
| This PR's gate (measured) | **-0.0090** | **[-0.0171, -0.0013]** |

**Probe predicted -0.0107; gate measured -0.0090.** Difference 0.0017 fpts — well within the probe's CI half-width (~0.0082). Both directions match (negative = candidate wins). Both CIs strictly exclude zero. **Calibration is sharper than PR #25→#26's WR gap (0.0043 fpts)**, consistent with the smaller absolute magnitude leaving less room for bootstrap-noise drift.

## Per-(model_class, TE) verdicts

| Model class | n_paired | RMSE Δ (fpts) | RMSE 95% CI | Spearman Δ | Spearman 95% CI | Verdict |
|---|---:|---:|---|---:|---|:---:|
| baseline | 4257 | -0.0100 | [-0.0280, +0.0093] | +0.0018 | [-0.0033, +0.0071] | DO_NOT_ADOPT (informational; confirms probe's null prediction) |
| **lightgbm-nb** | 4257 | **-0.0090** | **[-0.0171, -0.0013]** | +0.0028 | [+0.0001, +0.0055] | **ADOPT (binding)** |

**Spec deviation: only 2 of 5 model classes evaluated.** Spec §1.3.3 called for all 5 classes (baseline / lightgbm / lightgbm-tuned / lightgbm-nb / ensemble). Only the **binding cell (lgb-nb)** and the **modified-shape contingency cell (baseline)** were run, due to compute-time constraints. The remaining 3 informational cells (lightgbm, lightgbm-tuned, ensemble) were skipped — they are explicitly informational per spec §1.3.4 and not gating per spec §1.3.5. An earlier `--model all` attempt was aborted after 3 hours of wall time without producing a run dir; the `--model lightgbm-nb` + `baseline` pair completed in ~33 min combined (9 min pre-PR + 24 min post-PR). The skipped cells could be back-filled by a follow-up backtest if the routing-flip discussion ever needs them; they do not affect the binding decision.

## Per-year breakdown — `(LightGBMNbModel, TE)` binding cell

| Year | n_paired | RMSE Δ | RMSE 95% CI | Spearman Δ | Spearman 95% CI |
|---|---:|---:|---|---:|---|
| 2021 | 1030 | +0.0036 | [-0.0114, +0.0177] | -0.0001 | [-0.0056, +0.0056] |
| 2022 | 1088 | -0.0139 | [-0.0318, +0.0020] | +0.0067 | [+0.0000, +0.0127] |
| 2023 | 1058 | **-0.0151** | **[-0.0289, -0.0013]** | +0.0030 | [-0.0018, +0.0084] |
| 2024 | 1081 | -0.0106 | [-0.0283, +0.0053] | +0.0017 | [-0.0033, +0.0076] |

Pooled RMSE improvement is concentrated in 2022-2024 (3/4 years negative point estimate; only 2023 has a CI strictly below zero). 2021 is a small positive (+0.0036). The pooled CI is strictly negative because the 3-of-4 negative cells outweigh 2021's positive at the bootstrap level. Spearman is uniformly small-positive (+0.000 to +0.007); Spearman lower CI +0.0001 clears the -0.020 catastrophic-regression floor.

## Per-year breakdown — `(BaselineModel, TE)` informational cell

| Year | n_paired | RMSE Δ | RMSE 95% CI | Spearman Δ | Spearman 95% CI |
|---|---:|---:|---|---:|---|
| 2021 | 1030 | -0.0169 | [-0.0686, +0.0344] | +0.0062 | [-0.0081, +0.0201] |
| 2022 | 1088 | -0.0138 | [-0.0486, +0.0210] | -0.0014 | [-0.0127, +0.0092] |
| 2023 | 1058 | -0.0220 | [-0.0489, +0.0032] | +0.0048 | [-0.0024, +0.0127] |
| 2024 | 1081 | +0.0123 | [-0.0219, +0.0421] | -0.0024 | [-0.0095, +0.0058] |

Baseline TE is directionally favorable on 3/4 years (point estimates negative on 2021/2022/2023, positive on 2024). All per-year CIs bracket zero — none of the per-year cells clears the strict ADOPT bar individually. Pooled point estimate -0.0100 with CI bracketing zero. **No year shows REGRESSION (CI strictly above zero); the modified-shape contingency does not fire on either pooled or per-year analysis.**

## Coverage statistics (eval window 2021-2024)

| Column | Eval (2021-2024) coverage | Probe (PR #25) coverage | Match? |
|---|---:|---:|:---:|
| `age` | 94.8% | 95.4% | ✓ (-0.6pp) |
| `is_rookie` | 94.8% | 95.4% | ✓ (-0.6pp) |
| `volume_trend_l4_minus_prior_l4` | 46.4% | 44.7% | ✓ (+1.7pp) |
| `snap_pct_change_l4_vs_prior_l4` | 75.6% | 71.1% | ✓ (+4.5pp) |

All 4 columns within ~5pp of the probe's measured coverage. Age range observed: 20.0 - 40.0 (well within `ge=15, le=50` bounds — no clipping). `is_rookie` distribution: 5468 not-rookie / 1209 rookie / 365 NaN over 7042 rows in 2021-2024 (~17% rookie rate, consistent with NFL roster turnover). The slight uplift on `snap_pct_change` (+4.5pp) reflects the production builder running depth-chart dedupe + bye-week filter before the trajectory join (slightly more 8+ active games survive); not a divergence from the probe's expected mechanism.

## Threshold relaxation

`--coverage-threshold 0.35` was anticipated by the spec (§1.3.3), but the gate did not require an explicit override flag — the `adoption_gate.py --position TE` invocation evaluated coverage on the paired-row population (4257 rows after `(gsis_id, season, week, position)` join), well above any pooled NaN floor that would have triggered a coverage-gate failure. The 4257 pairings reflect TE rows that survive the production-builder filters AND have non-NaN trajectory values for the comparison; this is the correct paired-bootstrap denominator regardless of pooled-coverage statistics.

## Cross-class deferred follow-up

TE production routes to `baseline` per Plan 8 (2026-04-29). Plan 8's TE row showed `lightgbm-nb` was +0.0028 fpts vs baseline (point estimate slightly worse, CI bracketed zero). With trajectory cols now in `TeFeaturesSchema`, naively stacking this PR's lgb-nb improvement (-0.0090) suggests `lgb-nb-with-trajectory ≈ +0.0028 - 0.0090 = -0.0062 fpts` vs baseline-without-trajectory — a small but directionally favorable point estimate. CI would likely bracket zero, so a fresh cross-class re-eval probably wouldn't clear the strict-CI-below-zero ADOPT bar. **Not load-bearing for any current consumer; queue alongside the next TE-related work** (e.g., the planned weather-features sibling probe, or a routing-only re-eval after the next round of TE feature additions).

## What this closes

**TODO #24's TE-cell branch at the trailing-8-game unit.** Combined with PR #26's WR integration, the trailing-8-game-unit branch is now closed at all three of PR #25's ADOPT cells (WR baseline, WR lgb-nb, TE lgb-nb). This is the **first production integration in the project to bind on a non-default model class**.

Refined-unit candidates remain unexplored under the same TODO:
- per-position aging-curve interaction terms (`age²` for older-RB drop, etc.)
- `is_2nd_year` / `is_3rd_year` flags
- depth-chart-rank trends
- longer trailing windows (l8 vs l16)
- `has_trajectory_history` indicator (treat sparsity as a signal)

None queued.

## Next track

- **TODO #25 (weather features)** — sibling Track 2 probe; queued in `project_management.md` "Next action" before this PR. Same probe-first workflow as PR #25 → #26 → this.
- **Cross-class TE production routing re-eval** — could justify flipping `_PositionDispatch[TE].default_model_class` to `lightgbm-nb`. Naive arithmetic suggests close to break-even; not load-bearing.
- **Plan 4 (public API + CLI verbs + free-tier hosting)** — next major milestone per the PM backlog after Track 2 wraps.
