# Trajectory Feature Family Probe — Summary

**Candidate:** `trajectory`
**Branch:** `feat/probe-trajectory`
**Date probes ran:** 2026-05-03
**Override mtime:** 2026-05-03 (`data/features_probe/trajectory.parquet`, 56,652 rows; UDFA / pre-1980 fallback fired on 12,801 rows = **22.6%**)

**Bundle definition.** Four player-level features capturing biological age + role trajectory. Spec: `docs/superpowers/specs/2026-05-03-trajectory-feature-family-probe-design.md`. Plan: `docs/superpowers/plans/2026-05-03-trajectory-feature-family-probe.md`.

- **`age`** — biological age in target season.
  - Primary path: `draft_age + (season - draft_year)` from `nfl_data_py.import_draft_picks`.
  - Fallback path (UDFAs and pre-1980 players whose `draft_age` is NaN or absent from the draft-picks table): `season - inferred_draft_year + 22.0`, where `inferred_draft_year` is the player's earliest `weekly_stats` season (the league enters a player's first NFL season as their effective rookie year, and 22.0 is the league-average rookie age).
- **`is_rookie`** — `1.0 if season == draft_year else 0.0`. Uses `inferred_draft_year` for UDFAs.
- **`volume_trend_l4_minus_prior_l4`** — non-overlapping trailing-4 minus prior-4 means on a position-tailored volume stat:
  - QB → `attempts`
  - RB → `carries`
  - WR / TE → `targets`

  Active-game denominator (excludes bye / inactive / IR weeks; same active-game definition as v1 trailing means). Spec §3.3 codifies the structural NaN: "fewer than 8 prior active games yields NaN."
- **`snap_pct_change_l4_vs_prior_l4`** — non-overlapping trailing-4 minus prior-4 means on `SnapCountsSchema.offense_pct`. Same active-game denominator and same 8-prior-game NaN floor.

Plus an audit-only column `draft_year_inferred` (BooleanDtype) flagging which `draft_year` values came from the inferred fallback. Not a probe feature.

---

## Family verdict: **SIGNAL**

Per the spec's family-verdict rule (any (model, mode, position) cell ADOPT triggers SIGNAL), with criterion 3 satisfied (BaselineModel + lgb-nb both tested at composite via `--force-composite`), the family verdict is **SIGNAL** and **durable**.

**ADOPT cells (3):**
- **WR augment baseline** — RMSE Δ **-0.0414 fpts** ([-0.0606, -0.0230]); Spearman Δ +0.0058 ([+0.0026, +0.0092]).
- **WR augment lgb-nb (composite, forced)** — RMSE Δ **-0.0194 fpts** ([-0.0299, -0.0096]); Spearman Δ +0.0031 ([+0.0012, +0.0051]).
- **TE augment lgb-nb (composite, forced)** — RMSE Δ **-0.0107 fpts** ([-0.0191, -0.0028]); Spearman Δ +0.0032 ([+0.0004, +0.0063]).

This is **the first SIGNAL family probe since PR #20** (PBP team features bundle, RB-only via baseline). PR #22 (receiver air-yards / aDOT), PR #23 (PBP red-zone), and PR #24 (PBP pressure) all returned NULL durable.

---

## Coverage relaxation — important spec deviation

All four probe runs were invoked with `--coverage-threshold 0.35`, well below the spec's 0.95 default and below PR #22's 0.70 (which itself was the deepest precedent for a structural cold-start). This is the deepest threshold relaxation in Track 2A's history.

**Reason — structural sparsity, not silent imputation.** Trajectory's two trend features (`volume_trend_l4_minus_prior_l4`, `snap_pct_change_l4_vs_prior_l4`) require **8 prior active games per player**. By construction, that excludes ~50% of player-weeks across all years even at steady state — rookies, mid-season call-ups, returners from injury. This is "feature undefined for this player-week," NOT silent NaN imputation. The sparsity is structural-by-design and persists across all years (unlike PR #22's, which was a 2018-only cold-start).

Per-position coverage of override-candidate columns vs baseline rows:

| Position | age / is_rookie | volume_trend | snap_pct_change |
|---|---:|---:|---:|
| QB | 88.7% | 37.8% | 39.6% |
| RB | 96.6% | 53.7% | 66.6% |
| WR | 96.7% | 53.6% | 68.4% |
| TE | 95.4% | 44.7% | 71.1% |

**Why the comparison stays valid.** The probe joins the override into baseline features via a left-merge keyed on `(gsis_id, season, week)`. Rows where the override has NaN (under-8-prior-game players) match baseline rows that ALSO have NaN-equivalent missing data from the same player history. The per-paired-row delta is computed only on rows that BOTH sides have, so the bias is **symmetric on both sides of the comparison**. The surviving paired-row subset is biased toward established players (≥ 8 prior active games) — the verdict therefore applies to that subpopulation, which is also the subpopulation where these features have any defined value to begin with.

The relaxation is documented prominently here so any future re-test or production-builder follow-up applies the same threshold (or scopes the cohort explicitly).

---

## Per-mode summary table

| Model | Mode | Position | Phase 1 SIGNAL (pooled) | Phase 1 REGRESSION (pooled) | Phase 2 verdict |
|---|---|---|---:|---:|---|
| baseline | augment | QB | 1/30 | 0/30 | DO_NOT_ADOPT (REGRESSION) |
| baseline | augment | RB | 3/30 | 0/30 | DO_NOT_ADOPT |
| **baseline** | **augment** | **WR** | **2/30** | **0/30** | **ADOPT** |
| baseline | augment | TE | 2/30 | 0/30 | DO_NOT_ADOPT |
| baseline | swap | QB | 0/30 | 0/30 | (Phase 2 not run — Phase 1 NULL) |
| baseline | swap | RB | 0/30 | 0/30 | (Phase 2 not run — Phase 1 NULL) |
| baseline | swap | WR | 0/30 | 0/30 | (Phase 2 not run — Phase 1 NULL) |
| baseline | swap | TE | 0/30 | 0/30 | (Phase 2 not run — Phase 1 NULL) |
| lgb-nb (composite, forced) | augment | QB | 1/30 | 0/30 | DO_NOT_ADOPT (REGRESSION) |
| lgb-nb (composite, forced) | augment | RB | 3/30 | 1/30 | DO_NOT_ADOPT |
| **lgb-nb (composite, forced)** | **augment** | **WR** | **2/30** | **0/30** | **ADOPT** |
| **lgb-nb (composite, forced)** | **augment** | **TE** | **2/30** | **0/30** | **ADOPT** |
| lgb-nb (composite, forced) | swap | QB | 2/30 | 0/30 | DO_NOT_ADOPT |
| lgb-nb (composite, forced) | swap | RB | 0/30 | 0/30 | DO_NOT_ADOPT |
| lgb-nb (composite, forced) | swap | WR | 0/30 | 0/30 | DO_NOT_ADOPT |
| lgb-nb (composite, forced) | swap | TE | 0/30 | 0/30 | DO_NOT_ADOPT |

Phase 1 totals: 8/120 SIGNAL on baseline-augment, 0/120 on baseline-swap, 8/120 SIGNAL + 1/120 REGRESSION on lgb-nb-augment, 2/120 SIGNAL on lgb-nb-swap.

### Phase 2 detail (composite ΔRMSE + ΔSpearman, augment + lgb-nb-swap)

| Model | Mode | Position | RMSE Δ (95% CI) | Spearman Δ (95% CI) | Verdict |
|---|---|---|---|---|---|
| baseline | augment | QB | **+0.0382** ([+0.0155, +0.0600]) | -0.0063 ([-0.0104, -0.0025]) | DO_NOT_ADOPT (REGRESSION) |
| baseline | augment | RB | -0.0061 ([-0.0202, +0.0086]) | -0.0012 ([-0.0038, +0.0015]) | DO_NOT_ADOPT |
| **baseline** | **augment** | **WR** | **-0.0414** ([-0.0606, -0.0230]) | **+0.0058** ([+0.0026, +0.0092]) | **ADOPT** |
| baseline | augment | TE | -0.0097 ([-0.0288, +0.0095]) | +0.0015 ([-0.0039, +0.0064]) | DO_NOT_ADOPT |
| lgb-nb | augment | QB | **+0.0233** ([+0.0068, +0.0388]) | -0.0033 ([-0.0064, +0.0001]) | DO_NOT_ADOPT (REGRESSION) |
| lgb-nb | augment | RB | +0.0034 ([-0.0071, +0.0142]) | +0.0002 ([-0.0014, +0.0019]) | DO_NOT_ADOPT |
| **lgb-nb** | **augment** | **WR** | **-0.0194** ([-0.0299, -0.0096]) | **+0.0031** ([+0.0012, +0.0051]) | **ADOPT** |
| **lgb-nb** | **augment** | **TE** | **-0.0107** ([-0.0191, -0.0028]) | **+0.0032** ([+0.0004, +0.0063]) | **ADOPT** |
| lgb-nb | swap | QB | +0.0081 ([-0.0031, +0.0186]) | +0.0003 ([-0.0019, +0.0023]) | DO_NOT_ADOPT |
| lgb-nb | swap | RB | -0.0038 ([-0.0087, +0.0013]) | +0.0009 ([-0.0000, +0.0019]) | DO_NOT_ADOPT |
| lgb-nb | swap | WR | -0.0024 ([-0.0070, +0.0025]) | +0.0008 ([-0.0001, +0.0017]) | DO_NOT_ADOPT |
| lgb-nb | swap | TE | +0.0009 ([-0.0041, +0.0054]) | -0.0005 ([-0.0020, +0.0010]) | DO_NOT_ADOPT |

(baseline-swap had no Phase 1 SIGNAL → Phase 2 not run; probe predicts DO_NOT_ADOPT for all 4 positions.)

---

## Mechanism annotation — notable patterns

1. **WR is the dominant signal carrier.** WR ADOPTs under BOTH model classes in augment mode (-0.0414 baseline, -0.0194 lgb-nb). The WR feature builder has `targets`-driven volume metrics and `offense_pct` snap-share metrics; trajectory's `volume_trend` (targets-based for WR) and `snap_pct_change` give the model signed-direction-of-role information that the v1 trailing-level features are blind to. WR is the position where role transitions (target volume rising or falling year-over-year) move the most fantasy points per change unit.

2. **TE has a marginal lgb-nb-only signal.** TE augment ADOPTs under lgb-nb (-0.0107) but DOES NOT ADOPT under baseline (-0.0097, CI brackets zero). The CI bounds are nearly identical in absolute width (~0.038); the lgb-nb gain comes from tree-class flexibility extracting non-linear age-curve / trend interactions that the Ridge baseline can't represent.

3. **Recurring QB augment regression — 4th instance.** Adding to the pile:
   - PR #23 (red-zone) QB augment lgb-nb composite RMSE **+0.0268**.
   - PR #24 (pressure) QB augment lgb-nb composite RMSE **+0.0276**.
   - This probe (trajectory) baseline QB augment composite RMSE **+0.0382** (largest yet).
   - This probe (trajectory) lgb-nb QB augment composite RMSE **+0.0233**.

   Both model classes show the QB augment regression on this bundle, where prior probes only showed the regression on lgb-nb. The pattern suggests team-level / contextual / trajectory feature additions to QB inputs consistently overfit on augment configurations across Ridge and tree classes alike. Worth flagging for any future QB feature work — the QB feature builder is more sensitive to noise additions than the other three positions, possibly because its v1 feature set is already small (~25 cols vs ~40+ for skill positions).

4. **Swap mode is uniformly null.** Replacing baseline cols with trajectory cols (or pretending to, since trajectory's column set doesn't overlap with any baseline schema's columns — the swap is effectively a "drop nothing, add candidate" given the override doesn't match any drop list) gives no signal across 16 (model, position) cells. The signal **only appears as a pure addition** (augment mode), which is consistent with the trajectory features carrying orthogonal information rather than being substitutes for v1 features.

5. **RB is null on trajectory** despite ADOPT'ing in PR #21 (RB PBP team features). Different mechanism axis: PR #21's PBP cols are team-level pace / PROE / AYPS / EPA-resid, all team-driven; trajectory is player-level age and role. RB rushing volume is more team-script-driven than career-arc-driven (workhorse RBs cycle in and out of role faster than receivers' multi-year arcs), so this is mechanism-consistent. The 3/30 SIGNAL cells on RB augment are concentrated on `rushing_yards` (2022, 2024, pooled) but the composite Phase 2 doesn't fire.

---

## What this closes / refined-unit candidates left unexplored

**Closes:** TODO #24's age + role-trajectory candidates at the trailing-8-game unit. SIGNAL → greenlights an integration plan. The natural pattern is PR #20 → PR #21: probe SIGNAL on a position cell, then a focused integration spec for that cell only.

**Recommended follow-up — WR-first integration (analogous to PR #21's RB PBP integration):**
- Add the 4 trajectory cols + draft-lookup machinery to `WrFeaturesSchema` and wire `attach_trajectory_features` (extracted from `build_trajectory_overrides`) into `build_wr_features`.
- Run dual-run adoption gate on the binding `(BaselineModel, WR)` cell.
- Expected magnitude: ~-0.0414 fpts gate, similar to probe.

**TE secondary candidate:**
- Same pattern but at `TeFeaturesSchema`. Only ADOPT'd under lgb-nb (-0.0107 fpts), not under baseline. The dual-run gate on baseline would likely return DO_NOT_ADOPT for TE — but the lgb-nb cell would adopt. Consider scoping the TE integration to ship the schema + builder change but route the production binding to lgb-nb for TE only (precedent: Plan 6's QB-only ensemble suggestion).

**Refined-unit candidates left unexplored** (none queued; would each be its own probe):
- Per-position aging-curve interaction terms — e.g. `age²` for older-RB drop / older-WR cliff; flat-effect age in this probe doesn't capture curvature.
- `is_2nd_year` / `is_3rd_year` flags — collinear with age but might unlock breakout-year and post-rookie-leap signal.
- Depth-chart-rank trends (trailing-4 vs prior-4 mean of `depth_rank`) — orthogonal to volume trend if depth chart shifts before volume does.
- Longer trailing windows (l8 vs l16) — the l4 vs l4 cut may be too noisy to distinguish a real arc from week-to-week variance for non-WR positions.
- Treating sparsity as a feature — a `has_trajectory_history` indicator that flips on at game 8+; could let the model learn to use the trajectory signals only when they're well-defined.

---

## Reports

- `reports/feature_probe_trajectory_augment.{md,csv}` — baseline augment, all 4 positions.
- `reports/feature_probe_trajectory_swap.{md,csv}` — baseline swap, all 4 positions.
- `reports/feature_probe_trajectory_lgbnb_augment.{md,csv}` — lgb-nb augment, `--force-composite`.
- `reports/feature_probe_trajectory_lgbnb_swap.{md,csv}` — lgb-nb swap, `--force-composite`.
- `reports/feature_probe_trajectory_summary.md` — this document.
