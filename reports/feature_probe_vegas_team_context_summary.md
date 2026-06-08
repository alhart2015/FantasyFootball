# Vegas team-context family probe — summary

**Branch:** `feat/probe-vegas-team-context`
**Date:** 2026-05-17
**Spec:** `docs/superpowers/specs/2026-05-17-vegas-team-context-probe-design.md`
**Plan:** `docs/superpowers/plans/2026-05-17-vegas-team-context-probe.md`

## Headline verdict

**SIGNAL at lgb-nb × swap composite — QB + WR ADOPT.** The 4-col bundle (`preseason_implied_team_total`, `preseason_spread`, `season_avg_implied_team_total`, `season_avg_spread`) replaces `implied_team_total` + `spread` cleanly under tree models on QB (−0.0587 fpts, CI [−0.092, −0.028]) and WR (−0.0130 fpts, CI [−0.022, −0.003]) at Phase 2 composite. RB just misses signal (−0.0113 fpts, CI [−0.023, +0.001]); TE NULL.

The mechanism prediction was wrong about which position carries the signal (RB swap was the call; QB + WR actually carried it) but right about the model class: this is a tree-extractable, non-linear improvement that Ridge cannot see.

## Decision log

| Date | Commit | Decision |
|---|---|---|
| 2026-05-17 | `f49dc59` | Spec written, approved, committed. |
| 2026-05-17 | `db0d151` | Implementation plan committed. |
| 2026-05-17 | `39b4ff6` → `715509b` | Compute fn + preseason broadcast (Task 1), reviewed + dtype contract fixed. |
| 2026-05-17 | `c405cac` | season_avg expanding-mean tests (Task 2). |
| 2026-05-17 | `8f22b5c` | attach_vegas_team_context_features left-merge (Task 3). |
| 2026-05-17 | `6fcbc36` | build_vegas_team_context_overrides with input validation (Task 4). |
| 2026-05-17 | `6f251f2` + `baf2d77` | CLI scaffold + audit test (Tasks 5–6). |
| 2026-05-17 | verification gate | 20/20 new tests pass, mypy + ruff clean, 119/120 in cross-module subset (one pre-existing PR #45 failure, unrelated). |
| 2026-05-17 | `8216094` | Override generated: 56,652 rows, 100% preseason coverage / 94.13% season_avg coverage. |
| 2026-05-17 | `271123b` | BaselineModel probes complete (augment + swap). |
| 2026-05-17 | `7bdb527` | lgb-nb force-composite probes complete (augment + swap). |

## Per-(model, mode) verdict table

| Model | Mode | Phase 1 SIGNAL / cells | Phase 1 REGRESSION cells | Phase 2 verdict per position | Phase 2 fired? |
|---|---|---|---|---|---|
| BaselineModel | augment | 0/120 | 3 (QB pooled `passing_yards` +0.34 fpts; QB 2023 `rushing_yards` +0.10; RB pooled `rushing_yards` +0.05) | not run | no — no pooled SIGNAL |
| BaselineModel | swap | 1/120 (RB 2024 `rushing_yards` −0.064 fpts year-only) | 3 (QB pooled `passing_yards` +0.22; RB 2022 `receiving_yards` +0.07; RB 2023 `receiving_yards` +0.16) | not run | no — per-year SIGNAL only, no pooled |
| lgb-nb composite | augment (`--force-composite`) | 0/120 (tautological — Phase 1 is RidgeCV) | 3 (same as BL augment) | QB DO_NOT_ADOPT (+0.0165, [−0.001, +0.034]); RB DO_NOT_ADOPT (+0.0049, [−0.005, +0.014]); WR DO_NOT_ADOPT (−0.0004, [−0.006, +0.006]); TE DO_NOT_ADOPT (+0.0064, [−0.001, +0.014]) | forced |
| lgb-nb composite | swap (`--force-composite`) | 1/120 (same as BL swap) | 3 (same as BL swap) | **QB ADOPT** (**−0.0587** fpts, [−0.092, −0.028]); RB DO_NOT_ADOPT (−0.0113, [−0.023, +0.001]); **WR ADOPT** (**−0.0130** fpts, [−0.022, −0.003]); TE DO_NOT_ADOPT (−0.0058, [−0.015, +0.004]) | forced |

**Spearman deltas at lgb-nb swap (rank-improvement direction matches RMSE):** QB +0.0145 (CI [+0.007, +0.022]); WR +0.0009 (CI [−0.001, +0.003]); RB +0.0012 (CI [−0.001, +0.004]); TE −0.0004 (CI [−0.003, +0.002]). QB rank improvement is strictly positive; WR / RB / TE all bracket zero.

**Phase 2 fired only for lgb-nb runs** (`--force-composite`). The BaselineModel runs Phase 2 only on a Phase-1 pooled SIGNAL by default; neither augment nor swap returned a pooled SIGNAL under Ridge, so Phase 2 was skipped. The lgb-nb runs forced Phase 2 to test whether trees extract signal Ridge missed — which is exactly what happened on the swap variant.

## Mechanism annotation

**Pre-registered prediction (spec §1.2) vs observed:**

| Prediction | Observed | Verdict |
|---|---|---|
| Most likely SIGNAL: RB swap mode (preseason team-strength as forward-looking RB-rushing-volume proxy). | RB rushing_yards 2024 was the only Phase-1 SIGNAL cell (year-only), and RB Phase 2 just misses ADOPT at lgb-nb swap (CI hits zero). The RB hypothesis is the closest miss but not the headline. | **WRONG**: RB shows the right direction but at year-only and just-misses-composite magnitude. |
| Possible SIGNAL at Phase 1: `passing_yards` / `receiving_yards` at QB / WR. | Phase 1 *REGRESSED* on these cells — QB pooled `passing_yards` was the largest single regression in both augment (+0.34 fpts) and swap (+0.22 fpts). | **WRONG**: Ridge can't extract orthogonal signal — the candidate cols are inferior LINEAR proxies for `implied_team_total` / `spread`. |
| Probably NULL at TE. | TE DO_NOT_ADOPT at all four (model × mode) cells. | **CORRECT**. |
| Tautological at lgb-nb augment. | lgb-nb augment is 0/4 ADOPT; CIs all bracket zero. Trees can't extract additional signal when both the new cols *and* the per-game cols are present. | **CORRECT**: this is the strong confirmation that the SIGNAL is not from the new cols *adding* information — it's from them *replacing* noisy per-game variants under non-linear models. |

**Mechanism interpretation.** The 4 candidate columns and the baseline `implied_team_total` / `spread` measure the *same* mechanism (Vegas's view of team scoring environment) at different temporal granularities: the candidate is preseason / season-to-date; the baseline is per-game-closing. Under Ridge, the per-game version is strictly better (more recent → tighter to game-day truth); replacing it with smoother variants strictly loses information at the linear level (REGRESSION on QB passing_yards both augment and swap). Under lgb-nb's tree splits, the per-game version is *too granular* — trees overfit injury-news-driven per-game line movements that don't generalize. Replacing the per-game cols with the smoother preseason + season-to-date variants gives trees a stabler signal to split on. QB + WR composite improves; RB just misses; TE is unaffected (TE elite-magnitude is target-share-driven, not scoring-environment-driven, consistent with the §1.2 prediction).

**Why QB + WR specifically.** Both positions depend more on team scoring environment than RB (rushing yards is partially decoupled from team total — see PR #29 weather-refined verdict: rushing-axis stats are weather-resistant), and TE doesn't have the volume to express the team-environment signal in fpts. QB ADOPT magnitude (-0.059) is the largest single per-position composite improvement we've seen on any Track 2 family probe (vs trajectory WR -0.037 in PR #25, weather-refined NULL across the board in PR #29).

## QB augment regression check

PRs #23 / #24 / #25 / #28 each saw QB augment regress on context / team / trajectory / weather adds. Does this pattern recur on Vegas team-context?

- BaselineModel × augment QB: REGRESSION on `passing_yards` pooled (+0.34 fpts) — yes, the pattern recurs.
- BaselineModel × swap QB: REGRESSION on `passing_yards` pooled (+0.22 fpts) — yes, swap is less bad than augment but still REGRESSES at Ridge.
- lgb-nb × augment QB: DO_NOT_ADOPT at composite (+0.0165, brackets zero) — not regressing, but not improving.
- lgb-nb × swap QB: **ADOPT** at composite (−0.0587, strictly negative) — the *only* QB-positive cell across 4 (model, mode) combinations.

So the "QB augment regression" pattern at Ridge is real (recurring in 5 probe families now), but it doesn't generalize to tree-model composite: lgb-nb swap escapes the regression. This is consistent with the mechanism interpretation that the per-game `implied_team_total` is noisy for QB passing-yards prediction.

## Coverage relaxation

- Default threshold: 0.95 (PR #28 weather precedent).
- Used: 0.90 (cold-start week-1 NaN on `season_avg_*` accounted for).
- Actual observed coverage (from `reports/feature_probe_vegas_team_context_override_audit.md`):
  - `preseason_implied_team_total`: **100.00%**
  - `preseason_spread`: **100.00%**
  - `season_avg_implied_team_total`: **94.13%**
  - `season_avg_spread`: **94.13%**
- All four cols cleared 0.90; the relaxation was a hedge against worse-than-expected pbp gaps in the rare seasons. Not invoked in practice.

## Refined-unit candidates left unexplored

Per spec §1.4 + §8:

- **External Vegas data** (season win totals, preseason O/U) from a non-pbp source. Pursue if the integration plan finds that the SIGNAL is fragile (e.g., the as-of-time `season_avg_*` carries more weight than `preseason_*`, suggesting we need the genuine May line).
- **Non-linear encodings of existing Vegas cols** (`is_favored`, `is_heavy_favorite`, `is_high_total`, spread × ITT interaction). Tree models already get these natively from continuous cols — confirmed by lgb-nb augment being NULL. Refined-unit territory; deprioritized.
- **Opponent-side rollups** (opp's `season_avg_*`). Information-theoretically a separate hypothesis. Promising follow-up if the integration plan ships and we want to extend.
- **Line-movement features** (open-to-close spread movement). No historical open-line data; needs external ingest.
- **Position-specific encodings** (`preseason_pass_volume_proxy = preseason_implied_team_total × pass_rate_prior`). Could lift QB / WR further. Refined-unit candidate post-integration.
- **2025 eval extension.** Requires `refresh_features --seasons 2025` first. Deferred.

## Mechanism reflection on the 33c hypothesis

The 33c hypothesis from TODO.md was: *the elite-season under-projection lives in feature signal coverage; the missing class is forward-looking team-context*.

**The probe partially validates this hypothesis at the pbp-derivable level.** The 4-col bundle is the cheapest approximation to "what Vegas thought of the team before any in-season data existed" + "the running market view of team quality". It produces a real QB + WR composite improvement under tree models — magnitude comparable to prior ADOPT'd feature families.

But it doesn't fully validate. Critical caveats:

1. **The SIGNAL is at composite ΔRMSE, not at elite-season magnitude.** The 33d diagnostic showed Chase 2024 mean=250.74, actual=403. A composite ΔRMSE improvement of -0.06 fpts at QB / -0.013 fpts at WR is a ~0.5%–2% reduction on per-week RMSE (which is ~4–6 fpts depending on position). This won't move Chase from 250 to 380 by itself. The integration is necessary but not sufficient.
2. **The SIGNAL only shows up under lgb-nb swap.** Production routing for WR is `ensemble-decomposed` (not pure lgb-nb); the ensemble's lgb-nb child contributes but is averaged with Ridge children. The signal magnitude in production routing will be smaller than the raw lgb-nb swap number suggests.
3. **The swap drops `implied_team_total` and `spread` from baseline.** A naïve integration that *augments* would NULL out (per the lgb-nb augment cell). The integration plan must implement swap-equivalence semantics: replace per-game cols with the 4 new cols in the lgb-nb feature set, not add them alongside.

## Recommended next direction

**Greenlight a per-position integration plan for QB + WR (lgb-nb / ensemble-decomposed routes only).** Specifically:

1. **Extend `_shared.build_game_environment` (or sibling)** to emit `preseason_implied_team_total`, `preseason_spread`, `season_avg_implied_team_total`, `season_avg_spread` per team-game row alongside `spread` / `implied_team_total`.
2. **Schema decision: drop or keep `implied_team_total` / `spread` at the schema level?** The probe verdict says SWAP, not augment. Two approaches:
   - **Schema-swap on lgb-nb only:** Keep `implied_team_total` / `spread` in `*FeaturesSchema` (BaselineModel + ensemble Ridge children still need them); rewire just the lgb-nb model class to read the 4 new cols instead of the per-game cols. More surgical; preserves Ridge children's signal.
   - **Schema-augment, model-level swap:** Add the 4 new cols to `*FeaturesSchema` permanently; have lgb-nb's `_X_FEATURE_COLUMNS` explicitly exclude `implied_team_total` / `spread`. Less surgical but simpler schema migration.
   The integration spec should pick one; the cleaner path is probably the schema-swap on lgb-nb only.
3. **Refresh feature caches** under `data/features/{qb,wr}/`.
4. **Run the dual-run adoption gate** on QB + WR with production routing (QB → lgb-nb directly; WR → ensemble-decomposed-child). Confirms the probe Phase 2 verdict matches the gate verdict.
5. **TE: do NOT integrate.** No SIGNAL anywhere. Adding the cols to TE schema would force-test a feature class TE doesn't use.
6. **RB: revisit after QB / WR integration.** The probe RB Phase 2 is −0.0113 fpts with CI [−0.023, +0.001] — barely outside SIGNAL. A separate RB-specific feature variant (`preseason_*` only, dropping `season_avg_*` which adds noise on a cold-start) might cross the threshold. Defer to follow-up probe.

The integration plan is the path that closes 33c at this level of approximation. If after integration the gate confirms ADOPT but the elite-magnitude problem persists, the right next step is **external preseason Vegas data** (genuine May win totals, OC/HC tenure, FA flags) — a richer encoding of the same mechanism.

## Plan-vs-execution deviations

- **Task 1 dtype contract.** Initial implementation lost `Float64` dtype on `season_avg_*` due to `pandas Float64.expanding().mean()` returning numpy `float64`. Caught in code review; fixed at commit `715509b` by adding `.astype("Float64")` to both `season_avg_*` assignments. Documentation of dtype contract also corrected to match `build_game_environment`'s actual inherited dtypes.
- **Task 4 fixture extension.** `_make_schedule_rows([])` raised on empty input; the validation-error tests needed an empty schedules frame. Implementer added a small early-return branch in the fixture rather than constructing dummy non-empty rows that would be discarded anyway. Acceptable extension.
- **Task 8 unique-team-season counts.** The audit logs unique `preseason_spread` values per season (30–32 per season). The 30s are NOT missing data — they're coincidental ties where two teams had identical week-1 closing spreads (e.g., two -3 favorites). 100% preseason_* coverage confirms no missing-line residual.
- **Task 9–10 runtime.** Plan estimated 5–15 min for Baseline runs and 1–2 hr for lgb-nb. Actual: Baseline ~30 s each, lgb-nb ~8 min each. ~10× faster than the conservative estimates.
- **Task 10 stderr handling.** First lgb-nb invocation captured sklearn `UserWarning`s into the .md file alongside the markdown. Killed and re-ran with `-W ignore::UserWarning` + stderr to separate log. No data loss; the run was redirected before any useful Phase-2 numbers were produced.
- **Verdict interpretation flipped mid-execution.** The pre-registered prediction was a NULL-leaning probe (with RB as the most-likely SIGNAL). The lgb-nb swap result flipped the verdict to SIGNAL on QB + WR. The summary captures this honestly: prediction wrong, mechanism real, integration recommended.
