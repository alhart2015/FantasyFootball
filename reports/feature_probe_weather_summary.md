# Weather Feature Family Probe — Summary

**Date:** 2026-05-07
**Branch:** `feat/probe-weather`
**Spec:** `docs/superpowers/specs/2026-05-07-weather-feature-family-probe-design.md`
**Plan:** `docs/superpowers/plans/2026-05-07-weather-feature-family-probe.md`
**Override:** `data/features_probe/weather.parquet` (56,652 rows; audit: `reports/feature_probe_weather_override_audit.md`)

## Family verdict: **SIGNAL** via lgb-nb augment composite — RB and WR

The 4-feature weather bundle (`wind_speed_mph`, `is_high_wind` ≥20 mph, `temperature_f`, `is_grass_surface`) returned ADOPT on **2 of 4 positions** at the (lightgbm-nb composite, augment) cell:

- **(lgb-nb augment, RB)**: composite ΔRMSE **-0.0081 fpts** ([-0.0163, -0.0005]) — CI strictly below 0
- **(lgb-nb augment, WR)**: composite ΔRMSE **-0.0110 fpts** ([-0.0172, -0.0049]) — CI strictly below 0

QB and TE returned DO_NOT_ADOPT under lgb-nb augment composite (point estimates near zero with CIs bracketing 0). BaselineModel returned no Phase-1 SIGNAL across either mode (0/120 cells in augment, 0/120 in swap). lgb-nb swap returned the degenerate all-zero composite (the swap of feature columns with no v1 counterparts is a no-op — the candidate run drops then re-adds the same 4 cols, so base and candidate are identical).

This is the **second SIGNAL family probe in a row** (after PR #25 trajectory) and the **first probe where signal lives only in lgb-nb composite, not in BaselineModel**. The signal is invisible to RidgeCV (Phase 1 + baseline composite) but tree-based models extract it — consistent with the bundle including non-linearities (`is_high_wind` threshold, surface category) that linear regression cannot exploit even with the explicit threshold encoding.

## Per-mode verdict table

| Mode | Model | Phase 1 SIGNAL cells | Phase 2 verdicts (per position) |
|---|---|---:|---|
| augment | baseline | 0 / 120 (1 REGRESSION) | not run (Phase 1 skipped Phase 2) |
| swap | baseline | 0 / 120 | not run (Phase 1 skipped Phase 2) |
| augment | lgb-nb composite | 0 / 120 | QB DO_NOT_ADOPT, **RB ADOPT**, **WR ADOPT**, TE DO_NOT_ADOPT |
| swap | lgb-nb composite | 0 / 120 | all 4 DO_NOT_ADOPT (degenerate all-zero) |

The single Phase 1 REGRESSION cell (baseline + lgb-nb augment, QB rushing_yards 2023, +0.0812 fpts CI [+0.0133, +0.1515]) is one year of one stat; pooled QB rushing_yards is NULL across all 4 modes and pooled QB Phase 2 lgb-nb augment is +0.0077 fpts (CI brackets 0).

## Mechanism annotation

**Mechanism prediction (spec §1.1):** QB / WR / TE pass-volume cells should benefit most; RB rushing relatively insensitive.

**Observed:** WR confirmed (largest signal, -0.0110 fpts), RB unexpectedly adopted (-0.0081 fpts), QB and TE no signal.

**Why no QB signal?** Wind and temperature effects on passing efficiency are likely already partially captured by `roof_dome` (already in `*FeaturesSchema`) and the Vegas-implied `implied_team_total` (which prices in known weather risk for outdoor games). The marginal lift from explicit `wind_speed_mph` / `temperature_f` over those proxies is below the per-cell noise floor for QBs.

**Why RB signal (unexpected)?** Two plausible mechanisms, neither testable from the probe alone:
- `is_grass_surface` (51% True) gives RBs a meaningfully different footing/cut-back regime than turf, especially relevant for break-away yardage. Tree models can split on this where Ridge cannot.
- Cold-weather games shift offensive balance toward rushing (passing efficiency drops, teams lean run); the `temperature_f` continuous feature captures this regime shift via tree splits.

**Why WR signal?** Likely a combination of wind suppressing downfield passing (`is_high_wind` boolean activates the regime change) and surface affecting yards-after-catch on grass vs turf. The lgb-nb-only adoption pattern suggests the signal is non-linear — the threshold `is_high_wind` at ≥20 mph is doing more work than the continuous `wind_speed_mph` alone could in a linear model.

**Why no TE signal?** Plausibly the smaller sample size (n_paired = 3,975 for TE vs 5,273 RB / 8,470 WR) doesn't cross the per-cell signal floor at the noise level the bundle delivers, even if the underlying mechanism is similar. Could be revisited if a TE-specific weather refinement (e.g., per-route-concept × weather interaction) ever has independent evidence.

## Coverage note

Pooled coverage: ~91.6% — the override has 56,652 rows; outdoor weather NaN rate measured at 8.39% (per audit). Default `--coverage-threshold 0.95` would have failed on this margin; **invoked the spec §1.3 fallback at `--coverage-threshold 0.90`** for all 4 probe runs. The relaxation is shallower than PR #25's 0.35 (deepest in Track 2 history) and is on par with PR #23's 0.90. The bias is symmetric across baseline and candidate runs of the probe, so the relaxed threshold is mechanically safe — the missing rows simply drop out of both arms of the comparison.

The 8.39% gap is upstream `nfl_data_py` data quality: outdoor games where `wind` AND `temp` AND (sometimes) `surface` are all NaN despite no dome. Concentrated in older seasons (2018-2019) per anomaly notes; per-season 2021-2024 (the eval window) is uniformly higher. Per-(position, season) coverage breakdown for 2021-2024 is at or above 92% across all 4 positions for every season.

## Recurring QB augment regression check

PRs #23 / #24 / #25 each saw QB augment regress on context-adjacent features — at the composite level, not just per-stat. **The pattern recurred here at the per-stat level only**, not composite:

- baseline + lgb-nb augment: **QB rushing_yards 2023 REGRESSION** at +0.0812 fpts (CI [+0.0133, +0.1515]). Single year, single stat, pooled is NULL.
- Composite QB augment lgb-nb: +0.0077 fpts, CI [-0.0114, +0.0266] — DO_NOT_ADOPT, point near zero, NOT REGRESSION.

This is **milder than the PR #23/#24/#25 pattern**. Those probes saw composite-level QB augment regressions of +0.0268, +0.0276, and +0.0382 fpts respectively. Weather here is +0.0077 — within noise. Plausibly the outdoor-weather fill (which doesn't *change* QB-relevant information beyond what `roof_dome` + `implied_team_total` already deliver) doesn't trigger the QB-specific overfit pattern as strongly as trajectory features did.

Worth flagging: if a follow-up weather integration plan targets QB, re-evaluate this 2023-rushing-yards regression on real production data before shipping.

## Refined-unit candidates left unexplored

Per spec §1.4 — refined units are revisit-only-on-SIGNAL territory. Given SIGNAL fired, these are now in scope as follow-up work but not queued:

- **Precipitation** (would require new ingest source — e.g., NOAA hourly historical data keyed on stadium lat/lon).
- **Kickoff hour / time-of-day** (extractable from existing `schedules.kickoff` UTC timestamp).
- **Cold-weather threshold** (`is_cold_weather = temp < 32`). One additional bool sibling to `is_high_wind` for the temperature mechanism.
- **Multi-class surface encoding** (one bool per surface code: `fieldturf`, `a_turf`, `matrixturf`, `sportturf`, `grass`).
- **Surface × position interactions**.
- **Per-team weather acclimation effects** (cold-weather home teams).
- **Wind direction encoding** (would require new ingest).

Recommended priority order if a refinement plan is scoped: cold-weather threshold (cheapest — single bool, existing data, mirrors `is_high_wind` shape) → multi-class surface (free — same source col) → kickoff-hour (free — `schedules.kickoff`).

## Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-05-07 | Probe shipped at `feat/probe-weather` HEAD `<final-sha>` | All 4 reports complete; verdict SIGNAL via lgb-nb augment (RB + WR). |
| 2026-05-07 | Coverage threshold relaxed to 0.90 for all 4 probe runs | Outdoor-NaN rate 8.39% (audit) exceeds the <5% spec §1.3 projection. Fallback per spec §1.3, on par with PR #23's 0.90 precedent. |
| 2026-05-07 | lgb-nb swap result is degenerate all-zero | Weather features have no v1 counterparts (the spec §5.2 noted swap mode would effectively act as augment). Empirically, the candidate-side drop+add of identical columns produces bit-identical predictions in lgb-nb. Not informative; not a bug. |

## Recommended next direction

**Greenlight a per-position integration plan analogous to PR #21 / PR #26 / PR #27.** Two paths to choose between:

**Option A (recommended): Combined RB + WR integration plan, routed through `LightGBMNbModel` only.**
- The two ADOPT cells share the same model class and binding mode (lgb-nb augment), so a single integration plan can extend `RbFeaturesSchema` + `WrFeaturesSchema` together, plumb the 4 weather cols through `build_rb_features` + `build_wr_features`, and run dual-run gates on `(LightGBMNbModel, RB)` and `(LightGBMNbModel, WR)` in parallel.
- Production routing decision required: TE production routes to `baseline` per the codebase default; both RB and WR also route to `baseline` by default. PR #27 set a precedent for shipping a schema change for a non-default model class while leaving baseline production routing unchanged. The integration plan should follow that pattern (extend the schema; plumb through builders; gate on lgb-nb only; production routing decision deferred).
- Expected gate magnitude: RB ~-0.008 fpts, WR ~-0.011 fpts (probe predictions; actual gate may differ ±0.003 fpts per PR #25/#26/#27 calibration history).

**Option B: Two separate integration plans (RB-first, then WR).**
- Sequential integration. Cleaner per-position attribution. Matches Track 2 historical pattern more closely (PR #21 RB-only, PR #26 WR-only, PR #27 TE-only).
- Cost: ~2x branch / PR overhead. The bundling rationale is that the 4 weather cols are mechanically identical across positions (game-level attributes); per-position builders just need to consume the same 4 cols.

**Recommendation: Option A.** The 4 weather features are game-level (not per-position-derived), so the per-position integration code is essentially identical (call `attach_weather_features` in each builder; add cols to each schema). The marginal cost of bundling is low and the operational efficiency win is high.

**Note on QB / TE:** Do NOT extend `QbFeaturesSchema` or `TeFeaturesSchema` in the follow-up plan. QB DO_NOT_ADOPT (and the recurring augment-regression pattern) and TE DO_NOT_ADOPT (small sample) both point away from extending those positions in the same plan. Leave them to a refined-unit plan if independent evidence later emerges.
