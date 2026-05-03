# PBP Pressure Family Probe — Summary

**Candidate:** `pbp_pressure`
**Branch:** `feat/probe-pbp-pressure`
**Date probes ran:** 2026-05-02
**Override mtime:** 2026-05-02 20:56 UTC-04:00 (`data/features_probe/pbp_pressure.parquet`, 56,652 rows)
**PBP partition mtime:** 2026-05-02 20:56 UTC-04:00 (re-ingested fresh for this probe)

**Bundle definition.** Four PBP team-level pressure features denominated on `qb_dropback == 1` (the canonical nflfastR pressure-event denominator: pass attempts + sacks + scrambles). Per-game rate = `sum(num) / sum(denom)`, then trailing-4 mean across the team's last 4 prior games (within-team rolling+shift, `min_periods=4`):

- `team_sack_rate_allowed_l4` — offense: `sum(sack) / sum(qb_dropback)` for `posteam`.
- `team_qb_scramble_rate_l4` — offense: `sum(qb_scramble) / sum(qb_dropback)` for `posteam`.
- `team_def_sack_rate_l4` — defense: `sum(sack) / sum(qb_dropback)` against `defteam`.
- `team_def_scramble_rate_l4` — defense: `sum(qb_scramble) / sum(qb_dropback)` against `defteam`.

Spec: `docs/superpowers/specs/2026-05-02-pbp-pressure-feature-family-probe-design.md`. Plan: `docs/superpowers/plans/2026-05-02-pbp-pressure-feature-family-probe.md`.

---

## Family verdict: **NULL** (durable)

Per the §4 verdict rule (`family_verdict_from_reports`-equivalent CSV-grep): zero pooled Phase 1 SIGNAL cells across all 4 mode × model reports, and zero Phase 2 ADOPT or MARGINAL cells. The §1.3 criterion 3 is satisfied (both baseline modes ran AND both lgb-nb modes ran with `--force-composite`), so the NULL is durable.

This closes the third and final TODO #3c team-level PBP feature family at the team-level granularity. PR #20 (pace/PROE/AYPS/EPA-resid) was the only one that returned SIGNAL (via RB only). PR #23 (red-zone) and this PR (pressure) both returned durable NULL across baseline + lgb-nb at composite.

---

## Per-mode summary table

| Model | Mode | Position | Phase 1 SIGNAL (pooled) | Phase 1 REGRESSION (pooled) | Phase 2 verdict |
|---|---|---|---:|---:|---|
| baseline | augment | QB | 0/6 | 0/6 | (Phase 2 not run — Phase 1 NULL) |
| baseline | augment | RB | 0/6 | 0/6 | (Phase 2 not run — Phase 1 NULL) |
| baseline | augment | WR | 0/6 | 0/6 | (Phase 2 not run — Phase 1 NULL) |
| baseline | augment | TE | 0/6 | 0/6 | (Phase 2 not run — Phase 1 NULL) |
| baseline | swap | QB | 0/6 | 0/6 | (Phase 2 not run — Phase 1 NULL) |
| baseline | swap | RB | 0/6 | 0/6 | (Phase 2 not run — Phase 1 NULL) |
| baseline | swap | WR | 0/6 | 0/6 | (Phase 2 not run — Phase 1 NULL) |
| baseline | swap | TE | 0/6 | 0/6 | (Phase 2 not run — Phase 1 NULL) |
| lgb-nb (composite, forced) | augment | QB | 0/6 | 0/6 | DO_NOT_ADOPT |
| lgb-nb (composite, forced) | augment | RB | 0/6 | 0/6 | DO_NOT_ADOPT |
| lgb-nb (composite, forced) | augment | WR | 0/6 | 0/6 | DO_NOT_ADOPT |
| lgb-nb (composite, forced) | augment | TE | 0/6 | 0/6 | DO_NOT_ADOPT |
| lgb-nb (composite, forced) | swap | QB | 0/6 | 0/6 | DO_NOT_ADOPT |
| lgb-nb (composite, forced) | swap | RB | 0/6 | 0/6 | DO_NOT_ADOPT |
| lgb-nb (composite, forced) | swap | WR | 0/6 | 0/6 | DO_NOT_ADOPT |
| lgb-nb (composite, forced) | swap | TE | 0/6 | 0/6 | DO_NOT_ADOPT |

Phase 1 cell counts: 6 stats × 4 positions × 4 reports = 480 pooled cells in Phase 1; **0 SIGNAL, 0 REGRESSION** across all of them.

### Phase 2 detail (composite ΔRMSE + ΔSpearman, lgb-nb only)

| Mode | Position | RMSE Δ (95% CI) | Spearman Δ (95% CI) | Verdict | Notes |
|---|---|---|---|---|---|
| augment | QB | **+0.0276** ([+0.0077, +0.0472]) | -0.0014 ([-0.0046, +0.0022]) | DO_NOT_ADOPT | RMSE CI strictly above 0 — small statistically-significant **regression**. |
| augment | RB | +0.0025 ([-0.0053, +0.0108]) | +0.0005 ([-0.0009, +0.0020]) | DO_NOT_ADOPT | Both CIs bracket 0. |
| augment | WR | -0.0005 ([-0.0070, +0.0063]) | +0.0006 ([-0.0009, +0.0021]) | DO_NOT_ADOPT | Both CIs bracket 0; point estimate trivially favorable. |
| augment | TE | +0.0032 ([-0.0070, +0.0128]) | -0.0019 ([-0.0050, +0.0014]) | DO_NOT_ADOPT | Both CIs bracket 0. |
| swap | QB | +0.0125 ([-0.0053, +0.0313]) | -0.0031 ([-0.0070, +0.0005]) | DO_NOT_ADOPT | RMSE CI brackets 0; Spearman CI's upper bound just barely positive. |
| swap | RB | +0.0031 ([-0.0050, +0.0117]) | -0.0003 ([-0.0017, +0.0011]) | DO_NOT_ADOPT | Both CIs bracket 0. |
| swap | WR | -0.0012 ([-0.0074, +0.0058]) | +0.0009 ([-0.0006, +0.0024]) | DO_NOT_ADOPT | Both CIs bracket 0. |
| swap | TE | +0.0053 ([-0.0058, +0.0169]) | **-0.0034** ([-0.0067, -0.0002]) | DO_NOT_ADOPT | Spearman CI strictly below 0 — small statistically-significant **rank regression**, but RMSE brackets 0 so verdict is DO_NOT_ADOPT not REGRESSION. |

The two directional cells (QB augment RMSE regression; TE swap Spearman regression) are flagged for the production-builder follow-up to consider if the family is ever re-opened. Neither flips the family verdict to REGRESSION — the family-level question is "is there orthogonal signal?", not "does this hurt."

---

## Mechanism annotation

The bundle's predicted mechanism was **QB-side pressure exposure**:
- `team_sack_rate_allowed_l4` ↑ → QB `passing_yards` ↓ (sacks subtract yardage), QB `sacks` taken ↑ direct.
- `team_qb_scramble_rate_l4` ↑ → QB `rushing_yards` ↑ (scramble = QB rushing).
- Defensive cols affect QB cells via opponent-strength routing in the per-game join.

**Predicted mechanism not observed.** Zero Phase 1 SIGNAL cells on QB stats (or any other position's stats) across all 4 mode × model reports. The only directional Phase 2 cell is QB augment lgb-nb composite RMSE going the **wrong direction** (regression of +0.0276 fpts) — consistent with the family's predicted-mechanism axis being net-negative-or-null in the production model class, not net-positive. Same pattern as PR #23 (red-zone), which also had QB augment lgb-nb composite RMSE regression (+0.0268 there). Adding team-level PBP features in augment mode under lgb-nb appears to consistently introduce small regressions on QB, which the model class doesn't recover from.

---

## Coverage note

Default `--coverage-threshold 0.95` used; passed cleanly. Pooled non-null fraction across the 4 feature columns: **96.6%**. Per-season:

| Season | Rows | NaN fraction (all 4 feature cols) |
|---|---:|---:|
| 2018 | 7,908 | 24.2% (cold-start; no Y-1 backfill) |
| 2019 | 7,974 | 0.0% |
| 2020 | 8,065 | 0.0% |
| 2021 | 8,232 | 0.0% |
| 2022 | 8,252 | 0.0% |
| 2023 | 8,143 | 0.0% |
| 2024 | 8,078 | 0.0% |

The 2018 cold-start NaN doesn't affect the verdict — the eval window is `--holdout-years 2021-2024`, where coverage is 100%. No `--coverage-threshold` relaxation needed (PR #22 used 0.70, PR #23 used 0.90 — this probe didn't need either).

---

## Decision-log entry

The PBP pressure team-level family is closed across BaselineModel + lightgbm-nb at composite, with the dropback-denominator cut (`qb_dropback == 1`). Refined-unit candidates (alternate denominators like `pass_attempts + sacks` only; goal-line / 3rd-down / two-minute pressure subsets) remain unexplored but are unlikely to clear what the broad cut couldn't, absent independent evidence the unit choice was the binding constraint. None queued.

**This closes Track 2A — all three TODO #3c team-level PBP families have now been probed:**
- PR #20 (pace/PROE/AYPS/EPA-resid bundle) — SIGNAL via RB; integrated into RB schema in PR #21 (-0.0124 fpts adoption gate).
- PR #23 (red-zone bundle) — durable NULL.
- PR #24 (this — pressure bundle) — durable NULL.

The remaining TODO #3c open items are receiver-level / refined-unit candidates (PR #22's air-yards / aDOT family also returned NULL; deeper unit choices like per-route-concept distributions need ingest extensions). Next-action redirect candidates: Track 2B (RB PBP × other model classes — informational pass) or pivot to the model-improvement tracks in TODOs #23 / #24 / #25 / #26.

**Reports:**
- `reports/feature_probe_pbp_pressure_augment.{md,csv}` (baseline augment, all 4 positions)
- `reports/feature_probe_pbp_pressure_swap.{md,csv}` (baseline swap, all 4 positions)
- `reports/feature_probe_pbp_pressure_lgbnb_augment.{md,csv}` (lgb-nb augment, `--force-composite`)
- `reports/feature_probe_pbp_pressure_lgbnb_swap.{md,csv}` (lgb-nb swap, `--force-composite`)

---
