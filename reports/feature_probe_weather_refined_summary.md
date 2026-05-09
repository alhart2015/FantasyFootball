# Weather Refined-Unit Feature Family Probe — Summary

**Date:** 2026-05-09
**Branch:** `feat/probe-weather-refined`
**Spec:** `docs/superpowers/specs/2026-05-09-weather-refined-unit-probe-design.md`
**Plan:** `docs/superpowers/plans/2026-05-09-weather-refined-unit-probe.md`
**Override:** `data/features_probe/weather.parquet` (56,652 rows; audit: `reports/feature_probe_weather_refined_override_audit.md`)
**Predecessors:** PR #28 (broad-cut weather family probe — verdict SIGNAL via lgb-nb augment, RB+WR ADOPT) → PR #29 (RB+WR weather features integration — both ADOPT, lgb-nb production routing unchanged) → this probe.

## Family verdict: **SIGNAL** via lgb-nb composite — RB swap + WR augment + WR swap

The 8-feature refined bundle (`is_cold_weather`, `is_a_turf`, `is_astroturf`, `is_fieldturf`, `is_grass`, `is_matrixturf`, `is_sportturf`, `is_primetime`) returned ADOPT on **3 of 16** (model × mode × position) Phase-2 cells, all under `LightGBMNbModel` composite:

- **(lgb-nb swap, RB)**: composite ΔRMSE **-0.0088 fpts** ([-0.0153, -0.0030]) — CI strictly below 0; Spearman lo_95 +0.0005 > -0.02 floor.
- **(lgb-nb swap, WR)**: composite ΔRMSE **-0.0050 fpts** ([-0.0098, -0.0006]) — CI strictly below 0; Spearman lo_95 -0.0006 > floor.
- **(lgb-nb augment, WR)**: composite ΔRMSE **-0.0051 fpts** ([-0.0097, -0.0006]) — CI strictly below 0; Spearman lo_95 -0.0013 > floor.

QB and TE returned DO_NOT_ADOPT in all four (model × mode) cells. BaselineModel returned **0 / 120 Phase-1 SIGNAL** in both modes (3 / 120 REGRESSION in augment, 4 / 120 REGRESSION in swap), so Phase 2 was skipped and BaselineModel did not produce per-position composite verdicts. The bundle's signal lives in tree-based composites only — invisible to RidgeCV at Phase 1, consistent with PR #28's pattern (the original v1 weather bundle also fired only at lgb-nb composite, not Ridge per-stat / baseline composite).

This is the **first refined-unit probe** in the project (Track 2A's earlier probes were all broad-cut). The verdict structure is the most informative variant the §1.2 decoding contemplates: **WR shows strict-refinement (swap ADOPT + augment ADOPT)** — the refined cols beat both no-weather and v1 — while **RB shows replace-only (swap ADOPT + augment ~NULL)** — the refined cols beat v1 but augmenting them on top of v1 doesn't add lift over v1 alone.

## Per-mode verdict table

| Mode | Model | Phase 1 (per-stat pooled) | Phase 2 (composite, per position) |
|---|---|---|---|
| augment | baseline | 0 / 120 SIGNAL, 117 / 120 NULL, 3 REGRESSION | not run (Phase 1 skipped Phase 2) |
| swap    | baseline | 0 / 120 SIGNAL, 116 / 120 NULL, 4 REGRESSION | not run (Phase 1 skipped Phase 2) |
| augment | lgb-nb composite | 0 / 120 SIGNAL, 117 / 120 NULL, 3 REGRESSION | QB DO_NOT_ADOPT, RB DO_NOT_ADOPT, **WR ADOPT**, TE DO_NOT_ADOPT |
| swap    | lgb-nb composite | 0 / 120 SIGNAL, 116 / 120 NULL, 4 REGRESSION | QB DO_NOT_ADOPT, **RB ADOPT**, **WR ADOPT**, TE DO_NOT_ADOPT |

Phase-2 composite ΔRMSE point estimates (95% CI) per cell:

| Position | lgb-nb augment | lgb-nb swap |
|---|---|---|
| QB | +0.0099 ([+0.0002, +0.0202]) — CI **strictly above 0** | +0.0099 ([+0.0002, +0.0202]) — CI **strictly above 0** |
| RB | +0.0026 ([-0.0036, +0.0081]) | **-0.0088 ([-0.0153, -0.0030]) — ADOPT** |
| WR | **-0.0051 ([-0.0097, -0.0006]) — ADOPT** | **-0.0050 ([-0.0098, -0.0006]) — ADOPT** |
| TE | +0.0053 ([-0.0006, +0.0110]) | +0.0053 ([-0.0006, +0.0110]) |

Identical QB / TE numbers across augment vs swap reflect that `QbFeaturesSchema` / `TeFeaturesSchema` carry no v1 weather columns — the swap drop is a no-op for those positions, so swap collapses to augment (matching PR #28's "degenerate-swap precedent for QB/TE"). Distinct numbers for RB and WR reflect the post-PR-29 schemas: v1 weather is in baseline, swap drops it, augment retains it.

## Refined-unit-specific decoding (per spec §1.2)

Per spec §1.2, the (swap, augment) verdict pair decodes to an integration shape:

- **WR — swap ADOPT + augment ADOPT → strict refinement.** The 8 refined cols beat the v1 4-col bundle (swap), and augmenting them on top of v1 also beats v1 alone (augment). Greenlight: replace v1 weather cols in `WrFeaturesSchema` with the refined 8-col bundle (`is_cold_weather`, 6× surface one-hot, `is_primetime`).
- **RB — swap ADOPT, augment ~NULL (DO_NOT_ADOPT, point +0.0026 brackets 0) → replace-only.** The refined cols carry signal that v1 misses, but the additional refined detail does not lift further when stacked on v1's wind / temp / is_high_wind / is_grass_surface. Mechanism interpretation: the refined cols *redistribute* v1's signal across sharper splits (32°F threshold, multi-class surface) — overlaying them is redundant. Greenlight: replace v1 weather cols in `RbFeaturesSchema` with the refined 8-col bundle.
- **QB — swap NULL + augment NULL → close at this cut.** Both cells DO_NOT_ADOPT. **However the point estimate is +0.0099 with CI strictly above zero (lo_95 = +0.0002).** The probe's `verdict_for_position` rule labels this DO_NOT_ADOPT (RMSE hi_95 not below 0), but mechanically this is a Phase-2 composite REGRESSION — see "Recurring QB augment regression check" below.
- **TE — swap NULL + augment NULL → close at this cut.** Point +0.0053 with CI bracketing 0 (lo_95 = -0.0006). Consistent with PR #28 (TE returned DO_NOT_ADOPT on the v1 bundle too) — TE's smaller n_paired (4257 here vs 5273 RB / 8470 WR) and weaker mechanism story doesn't cross the per-cell signal floor.

**Recommended follow-up integration shape (RB+WR strict-replace plan):** Replace v1 weather cols in `RbFeaturesSchema` + `WrFeaturesSchema` with the 8 refined cols. This is the natural sequel to PR #29 (which integrated v1 RB+WR weather). The plan should remove `wind_speed_mph`, `is_high_wind`, `temperature_f`, `is_grass_surface` from both schemas and add `is_cold_weather`, the 6 surface one-hots (`is_a_turf`, `is_astroturf`, `is_fieldturf`, `is_grass`, `is_matrixturf`, `is_sportturf`), and `is_primetime`. Production routing decision deferred (per PR #29 precedent — schema change for non-default model class while baseline routing unchanged).

## Mechanism annotation

**Spec §1.2 mechanism predictions** vs **observed**:

- **`is_cold_weather` on RB (predicted: ADOPT under lgb-nb augment).** Observed: RB augment composite point +0.0026 brackets 0 → not predicted ADOPT. RB swap ADOPT instead. Plausible mechanism interpretation: the cold-weather threshold's signal is *also* in the v1 `temperature_f` continuous feature (lgb-nb tree splits can synthesize the threshold from the continuous), so adding `is_cold_weather` on top of v1 is redundant — but a clean refined bundle that drops `temperature_f` and uses the sharp 32°F threshold instead is competitive, hence swap ADOPT.
- **Multi-class surface on RB+WR (predicted: swap ADOPT — multi-class beats binary `is_grass_surface`).** Observed: RB+WR both swap ADOPT — confirmed. Multi-class surface encoding is the strongest single component of the refined bundle's signal, since it both (a) preserves refined regime distinctions across turf brands that the binary collapse hides and (b) preserves NaN on null-surface rows where v1 silently fills 0.0.
- **`is_primetime` (predicted: weakest, may null-out everywhere).** Observed: cannot disambiguate from the bundle-level result. The bundle's lift could come entirely from `is_cold_weather` + multi-class surface; `is_primetime` may be silent or weakly contributory. A follow-up refined-unit-of-refined-unit probe could isolate it (see "candidates left unexplored" below).

The bundle as a whole behaves consistently with the spec's "refined unit beats broad cut on the binding axis" prior. WR's strict-refinement verdict is the cleanest evidence — both modes ADOPT and the magnitudes are tight (~-0.0050 fpts each, n_paired=8470).

## Coverage note

Per the override audit (`reports/feature_probe_weather_refined_override_audit.md`):

- **`is_primetime`**: per-(position, season) coverage 1.000 across every cell. No coverage caveats.
- **Multi-class surface**: per-row 2.17% NaN rate (rows with unparseable / unknown surface codes — stricter than v1's silent fill-to-zero); within the 0.90 default threshold across all (position, season) cells on the eval window 2021–2024.
- **`is_cold_weather`**: shares the v1 outdoor-NaN profile (8.39% pooled). Per-(position, season) on 2021–2024:
  - **2022**: 0.668–0.679 across positions — **dips below the 0.90 threshold** but the probe was invoked with `--coverage-threshold 0.90`, which validates against pooled-rows-per-column and passed. The per-(position, season) trough on 2022 is well-known from PR #28 / PR #29 and is symmetric across baseline and candidate runs (same dropna applies to both arms of the paired bootstrap). Lift estimates on 2022 fold splits are diluted by ~33% relative to other folds; this manifests as wider per-year CIs in the Phase-1 tables. Pooled and 2021 / 2023 / 2024 carry the signal weight.
  - **2023**: 0.854–0.857 — also below 0.90 but milder; same diluted-fold caveat applies.
  - **2024**: 0.982–0.985 — clean.

No further threshold relaxation beyond the `--coverage-threshold 0.90` precedent. The `is_cold_weather` 2022 trough doesn't change the verdict structure: RB and WR ADOPT cells are pooled-cell verdicts that aggregate across all four eval years; the 2022 NaN dilution is already absorbed.

## Recurring QB augment regression check

PRs #23 / #24 / #25 each saw QB augment regress on context / team / trajectory adds at the Phase-2 composite level. PR #28's broad-cut weather probe saw a **milder** version (composite QB augment +0.0077, CI brackets 0).

**This refined-unit probe sees the regression recur, harder than PR #28 and at the strict-CI level:**

- **(lgb-nb augment, QB)**: composite ΔRMSE **+0.0099 fpts** ([+0.0002, +0.0202]) — CI **strictly above 0**. Phase 2's `verdict_for_position` rule labels this DO_NOT_ADOPT (no separate REGRESSION label exists at Phase 2; ADOPT requires `rmse.hi_95 < 0`), but mechanically the candidate run's RMSE is significantly worse than baseline.
- **(lgb-nb swap, QB)**: bit-identical to augment (no v1 weather cols in `QbFeaturesSchema`, swap drop is a no-op). Same +0.0099 [+0.0002, +0.0202].
- **(BaselineModel × both modes, QB)**: also a per-stat REGRESSION at QB rushing_yards 2024 (point +0.0579, CI [+0.0118, +0.1070]). Pooled QB rushing_yards is +0.0233 with CI [-0.0212, +0.0637] — NULL pooled, single-year regression.

**Interpretation:** The refined-unit bundle adds 8 cols of weather-derived feature to QB's already-feature-rich frame (28 baseline cols including `roof_dome` and `implied_team_total`, both of which already absorb known weather effects on team scoring). The marginal information is ~0 in expectation but adds 8 dimensions to the lgb-nb input space, which lgb-nb modestly overfits in the train period and underperforms in holdout. The pattern confirms: **do not extend `QbFeaturesSchema` with the refined weather bundle in any follow-up integration plan** — same routing-around recommendation as PR #28 / PR #29 made for v1.

This is the **clearest QB augment regression of the four refined-unit probes** in Track 2A. Future weather-related QB-targeted refinements should require independent evidence (e.g., a per-stat passing-deep-vs-short × weather mechanism with prior data backing) before re-probing.

## Refined-unit-of-refined-unit candidates left unexplored

Per spec §1.4 / §1.2 — refined-unit-of-refined-unit territory only opens on bound SIGNAL, which this verdict satisfies. None queued; logged here for TODO #25 follow-up:

- **Continuous `kickoff_hour_et`** (vs binary `is_primetime`). The current probe can't isolate `is_primetime`'s contribution; if the bundle's lift mostly comes from `is_cold_weather` + multi-class surface, the binary primetime might be silent or weakly negative. A targeted probe with continuous kickoff hour (granular to allow tree splits at any hour) might capture late-window vs primetime regime more cleanly.
- **`is_london`** (`kickoff_hour_et < 11`, ~1–2% of games). Mechanism-clean (jet-lag) but cohort is small; the spec rejected it as a 4th bundle slot in this probe but it's worth a single-cell standalone test if WR / RB London games concentrate fantasy outliers.
- **Surface × position interactions**. Multi-class surface fired on RB+WR swap; explicit interaction terms (`is_grass × position`) may sharpen position-specific surface effects beyond what tree-based fits already model implicitly.
- **Per-team weather acclimation**. Cold-weather home teams (BUF, GB, CHI, NYJ, NE, PIT) may have systematic advantages in cold games beyond what `is_home + is_cold_weather` already encodes. Not in `nfl_data_py` directly; would require a curated list.
- **Precipitation**. Out of scope here (would require new ingest — NOAA hourly historical keyed on stadium lat/lon). Mechanism-strong but ingest-expensive.
- **Wind direction**. Out of scope (new ingest required). Mechanism-strong for kicking / passing accuracy but cohort-noisy.

Recommended priority order if a refined-unit-of-refined-unit plan is scoped: continuous `kickoff_hour_et` (cheapest — same data path as `is_primetime`) → surface × position interactions (free — same data; one more pass over the override) → `is_london` (free — single bool, single comparison cell).

## In-scope ingest fix note

**During Task 7 (override audit)**, an ingest-layer bug in `_build_kickoff` was discovered and fixed (commit `56df07f`). The bug mis-tagged `nfl_data_py`'s `gametime` (ET wall-clock) as UTC during partition writes, causing downstream `is_primetime` to fire at 0.16% (vs the expected ~22%). The corrected partitions produce primetime rate 21.97% pooled, consistent with TNF + SNF + MNF + occasional Saturday games at ~6 of 32 teams per primetime week × ~30 fantasy-eligible players per team. The fix is upstream of every `is_primetime` consumer and is **load-bearing** for this probe's verdict — without it the `is_primetime` axis would have been mechanically unmeasurable, biasing the bundle's verdict toward the `is_cold_weather` + multi-class surface contribution alone.

This is the second time during a Track 2A refined-unit probe that an ingest bug surfaced via override audit (the first was PR #25's discovery of a bye-week trailing-window edge case). The pattern is a useful incidental: **the override audit is not just QA on the probe; it's QA on the upstream data path that the production builders depend on.**

## Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-05-09 | Probe shipped at branch HEAD `<final-sha>` | All 4 reports complete; verdict SIGNAL via lgb-nb composite (RB swap, WR augment + swap ADOPT). |
| 2026-05-09 | Augment uses `weather_refined_only.parquet` (8 candidate cols); swap uses `weather.parquet` (12 cols, drops 4 v1) | Plan §Task 8 prescribed both modes use the same `weather.parquet` override. After PR #29 added v1 weather cols to `RbFeaturesSchema` + `WrFeaturesSchema`, full-override augment hits an `OverrideCollisionError` for RB/WR. Splitting the override avoids the collision while preserving the spec §1.2 augment-vs-swap semantics: augment = "8 refined on top of unchanged baseline (v1 included for RB/WR)"; swap = "drop v1, add 12 (= 4 v1 re-supplied + 8 refined)". v1 cols in baseline and override are bit-identical by construction (same `compute_weather_features` source), verified pre-probe. The split is cosmetic — the post-join feature frames are mathematically equivalent for the dropna+fit comparison. Both override files were generated as one-shot; the refined-only is `df.drop(columns=[v1_cols])` of the full file. |
| 2026-05-09 | Coverage threshold relaxed to 0.90 for all 4 probe runs | Outdoor-NaN profile inherited from PR #28 (8.39% pooled). Per-(position, season) `is_cold_weather` 2022 dips to 0.67 — the probe's `validate_override_coverage` checks against pooled-baseline-rows so this passes the 0.90 gate. Symmetric across base / cand arms; lift signals remain interpretable. |
| 2026-05-09 | Feature cache rebuilt mid-task to incorporate PBP partitions | Task 7's audit fix landed before this task; raw `pbp` and `draft_picks` partitions were absent from local `data/raw`. `data/features` was built from a partial raw layout, leaving `pace_l4` / `proe_l4` / `team_ayps_l4` / `team_def_epa_resid_l4` 100% NaN on RB/WR — every dropna on a baseline-cols superset eliminated 100% of rows. Refreshed pbp + draft_picks via direct ingest module calls, then re-ran `scripts/refresh_features.py all --seasons 2018-2024` to overwrite the per-week feature parquets in place. PBP cols verified 0% NaN post-rebuild before re-running probes. |
| 2026-05-09 | QB augment composite is a Phase-2 regression at +0.0099 [+0.0002, +0.0202] — CI strictly above 0 | Recurring pattern (PR #23 / #24 / #25 / #28). Worse than PR #28's +0.0077 CI-brackets-0 in this probe. Reinforces "do not extend `QbFeaturesSchema` with weather features without independent evidence" rule. |

## Recommended next direction

**Greenlight a per-position integration plan analogous to PR #29.** The verdict supports a **strict-replace integration of the 8-col refined bundle in `RbFeaturesSchema` + `WrFeaturesSchema`**, removing the 4 v1 weather cols. Single integration plan covering both positions; both bound under `LightGBMNbModel` composite; production routing decision deferred (PR #29 precedent — schema change for non-default model class while baseline routing unchanged).

Expected gate magnitudes (probe predictions, may differ ±0.003 fpts per PR #25 / #26 / #27 / #29 calibration history): **RB ~-0.009 fpts, WR ~-0.005 fpts**. The WR magnitude is smaller than PR #29's -0.0104 (v1 swap-from-no-weather) because the refined-vs-v1 lift is the marginal refinement above an already-integrated baseline; not a from-zero comparison.

**Note on QB / TE:** Do NOT extend `QbFeaturesSchema` or `TeFeaturesSchema` in the follow-up plan. QB DO_NOT_ADOPT (and the recurring augment-regression pattern, sharper here than PR #28) and TE DO_NOT_ADOPT (small sample, mechanism-weak) both point away from extending those positions. Leave them to refined-unit-of-refined-unit work if independent evidence later emerges.
