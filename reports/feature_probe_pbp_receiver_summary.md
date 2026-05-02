# PBP Receiver Family Probe — Summary

**Date:** 2026-05-01
**Branch:** `feat/wr-te-pbp-features`
**Spec:** `docs/superpowers/specs/2026-05-01-wr-te-pbp-receiver-features-design.md`
**Plan:** `docs/superpowers/plans/2026-05-01-wr-te-pbp-receiver-features.md`
**Override:** `data/features_probe/pbp_receiver.parquet` (regenerable; not committed; mtime 2026-05-01 16:59 ET, 399 KB, 37,724 rows)
**Override generator:** `scripts/build_pbp_receiver_override.py`

The four PBP-derived player-level (receiver) features `aDOT_l4`,
`deep_target_share_l4`, `yac_per_reception_l4`, `red_zone_target_share_l4`
were bundled into a single override and probed in two modes (augment, swap)
at the BaselineModel level and at the lightgbm-nb level (via
`--force-composite`) against the v1 baseline features for WR + TE. Swap
mode dropped the NGS season-snapshot analogs `avg_intended_air_yards_std`
and `avg_yac_above_expectation_std`. Family verdict applies the spec §4
rule across the executed reports.

## Coverage

The override has 37,724 rows (one per (gsis_id, season, week) for every
WR/TE rostered per `depth_charts` 2018-2024). 2018 coverage is structurally
low (~50% non-null) because the curated PBP window starts in 2018 and the
trailing-4 lookup has no Y-1 data to backfill at season start. 2019-2024
coverage is 75-87% per (position, season) pair. The probe was invoked with
`--coverage-threshold 0.70` to accommodate the 2018 cold-start.

| Position | Season | aDOT non-null | yac non-null | RZ non-null |
|---|---|---:|---:|---:|
| WR | 2018 | 53% | 50% | 53% |
| WR | 2019 | 78% | 76% | 78% |
| WR | 2020-2024 | 84-87% | 82-85% | 84-87% |
| TE | 2018 | 50% | 43% | 50% |
| TE | 2019 | 78% | 75% | 78% |
| TE | 2020-2024 | 81-85% | 77-83% | 81-85% |

The surviving-receiver row set is biased toward heavier-targeted WRs/TEs
(fewer than 4 prior receiver-active games → row dropped from candidate-side
training). This bias is consistent across baseline + candidate sides of the
probe so the comparison remains valid; the verdict applies to the
substantially-targeted subset of WRs/TEs, not blocking-only TEs or WR3+
deep-bench players.

## Per-mode Phase 1 summary (per-stat Ridge ΔRMSE bootstrap)

Per-position pooled-across-years cell counts. n=12 stats per (position, mode).

| Model | Mode | Pos | Pooled SIGNAL | Pooled REGRESSION | Best Δrmse (most-neg) | Worst Δrmse (most-pos) | Per-year non-NULL |
|---|---|---|---:|---:|---:|---:|---|
| baseline | augment | WR | 0 | 0 | -0.0005 | +0.0003 | none |
| baseline | augment | TE | 0 | 0 | -0.0001 | +0.0287 | per-year REGRESSION on `receiving_yards` 2022 (+0.147 fpts) |
| baseline | swap    | WR | 0 | 0 | -0.0157 | +0.0006 | none |
| baseline | swap    | TE | 0 | 0 | -0.0000 | +0.0310 | per-year REGRESSION on `receiving_yards` 2022 (+0.056 fpts) |
| lightgbm-nb | augment | WR | 0 | 0 | -0.0005 | +0.0003 | none |
| lightgbm-nb | augment | TE | 0 | 0 | -0.0001 | +0.0287 | per-year REGRESSION on `receiving_yards` 2022 (+0.147 fpts) |
| lightgbm-nb | swap    | WR | 0 | 0 | -0.0157 | +0.0006 | none |
| lightgbm-nb | swap    | TE | 0 | 0 | -0.0000 | +0.0310 | per-year REGRESSION on `receiving_yards` 2022 (+0.056 fpts) |

All Phase 1 (per-stat Ridge) cells return NULL at the pooled level across
both positions and both modes. Per-year REGRESSION cells on TE
`receiving_yards` 2022 replicate identically across baseline and lgb-nb
because Phase 1 always uses RidgeCV regardless of `--model` (see addendum
below). Pooled bootstrap correctly washes the per-year effect out at the
family level — pooled CIs bracket zero.

## Phase 2 (composite, lightgbm-nb production model via `--force-composite`)

| Mode | Pos | Composite RMSE Δ | RMSE 95% CI | RMSE verdict | Composite Spearman Δ | Spearman 95% CI | Spearman verdict |
|---|---|---:|---|---|---:|---|---|
| augment | WR | -0.00056 | [-0.0063, +0.0053] | DO_NOT_ADOPT | +0.00083 | [-0.0005, +0.0023] | DO_NOT_ADOPT |
| augment | TE | +0.00544 | [-0.0035, +0.0143] | DO_NOT_ADOPT | -0.00074 | [-0.0044, +0.0028] | DO_NOT_ADOPT |
| swap    | WR | -0.00519 | [-0.0118, +0.0013] | DO_NOT_ADOPT | +0.00064 | [-0.0007, +0.0022] | DO_NOT_ADOPT |
| swap    | TE | +0.00249 | [-0.0069, +0.0112] | DO_NOT_ADOPT | -0.00089 | [-0.0044, +0.0025] | DO_NOT_ADOPT |

The closest cell to a SIGNAL verdict was `swap WR composite RMSE`:
point estimate -0.0052 fpts, 95% CI [-0.0118, +0.0013] — point estimate
is favorable but the upper bound just barely exceeds zero (+0.001 fpts).
Not enough to clear the §4 ADOPT/MARGINAL bar. All other Phase 2 cells
are clearly inconclusive.

## Family verdict

**`NULL`** (computed by the spec §4 rule via `family_verdict_from_reports`).

- **No pooled Phase 1 SIGNAL cells** across all 4 reports (baseline +
  lgb-nb × augment + swap, 2 positions × 12 stats × 4 reports = 96 pooled
  cells, all NULL).
- **No Phase 2 ADOPT or MARGINAL verdicts** across the 2 lgb-nb reports
  (4 RMSE + 4 Spearman composite cells, all DO_NOT_ADOPT).

Per spec §1.3 criterion 3, NULL is durable: both BaselineModel and
lightgbm-nb were tested in both augment and swap modes. The `--force-composite`
addendum (below) preserves the durability claim.

## Addendum: Spec §3.2 + `--force-composite`

Spec §3.2 specified the lgb-nb runs as bare `--model lightgbm-nb` (without
`--force-composite`), expecting them to actually exercise lgb-nb at the
composite level when baseline returned NULL. Empirically that does not
work: Phase 1's per-stat ΔRMSE bootstrap always uses `RidgeCV` regardless
of `--model` (see `feature_probe.py::probe_per_stat` + `_fit_predict_residuals`).
Phase 2 is the only stage that invokes the production model class — and
Phase 2 only fires if Phase 1 returned a pooled SIGNAL cell (or
`--force-composite` was passed).

Without `--force-composite`, the lgb-nb runs returned identical pooled
Phase 1 results to baseline (numerical agreement to ~12 decimals), Phase 2
was skipped, and lgb-nb was never actually tested. PR #19 (Plan 9 retro
option C) introduced `--force-composite` precisely for this case.

The lgb-nb runs were re-executed with `--force-composite` to actually
exercise lgb-nb at the composite level; results above. The spec's
verdict-durability rule (§1.3 criterion 3) is satisfied.

**Spec follow-up (separate change):** spec §3.2 should be updated to
include `--force-composite` on the conditional lgb-nb commands. Same
issue exists in PR #20's spec §3.2; PR #20 didn't hit it because
baseline returned SIGNAL on RB and the conditional lgb-nb path never
fired. This spec is the first invocation of the conditional path.

## Decision

**Family closed at the BaselineModel + lgb-nb level for receivers.**

The four player-level air-yards / aDOT distributions tested here do not
carry orthogonal signal for WR/TE fantasy projections at the unit
boundaries chosen (per-receiver, trailing-4 receiver-active games). Both
augment-mode (add to the existing NGS season-snapshot features) and
swap-mode (replace the NGS aDOT/YAC snapshots with PBP trailing-4
variants) returned DO_NOT_ADOPT at composite under both Ridge and lgb-nb.
The closest cell (WR swap composite RMSE -0.0052, CI bracket zero by
+0.001 fpts) suggests the trailing-4 PBP YAC + aDOT may be a *marginal*
improvement over the season-snapshot NGS variants for WRs, but the effect
is not statistically distinguishable from noise at the gate's per-cell
threshold. Re-investigating this specific direction would require either
a tighter signal estimator or a larger sample; not worth pursuing
standalone.

**What this closes:** TODO #3c's WR/TE refined-unit follow-up at the
air-yards / aDOT cut (this spec's scope). Other refined-unit candidates
that remain unexplored:

- **Per-route-concept distributions** — would require route-running data
  not in the curated PBP subset.
- **Target-quality residuals** — would require per-throw difficulty modeling
  (pressure, separation, pre-snap motion); some inputs in PBP, others not.
- **In-line vs flexed alignment for TE** — would require participation /
  alignment data not in the curated PBP subset.

None of these are queued. Revisit only if independent evidence (a published
study, a successful third-party benchmark, etc.) suggests the unit choice
was the binding constraint rather than absence of orthogonal signal.

**What remains open in TODO #3c:** Other PBP feature families at the
team-level granularity that PR #20 did not include — pressure rate
allowed by O-line, red-zone usage shares (separate from receiver-level
RZ target share tested here), etc. Bundle 3-4 candidates per probe per
the family-level prior.

## Cross-references

- Per-mode reports (committed):
  - `reports/feature_probe_pbp_receiver_augment.{md,csv}` — baseline augment
  - `reports/feature_probe_pbp_receiver_swap.{md,csv}` — baseline swap
  - `reports/feature_probe_pbp_receiver_lgbnb_augment.{md,csv}` — lgb-nb augment with `--force-composite`
  - `reports/feature_probe_pbp_receiver_lgbnb_swap.{md,csv}` — lgb-nb swap with `--force-composite`
- Spec: `docs/superpowers/specs/2026-05-01-wr-te-pbp-receiver-features-design.md`
- Plan: `docs/superpowers/plans/2026-05-01-wr-te-pbp-receiver-features.md`
- Predecessor (team-level family probe, PR #20): `reports/feature_probe_pbp_family_summary.md` — RB SIGNAL + WR/TE NULL at team-level granularity.
- TODO #3c update + project_management.md decision-log entry: appended in the same commit cluster.
