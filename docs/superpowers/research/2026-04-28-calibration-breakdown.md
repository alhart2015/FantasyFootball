# Plan 7 Phase 0 — Calibration Breakdown Diagnostic

**Date:** 2026-04-28
**Branch:** `feat/plan-7-calibration-aware-nb`
**Run:** `data/backtest/run_20260428T152314Z/results.parquet` (Plan 5c C-NB output)
**Diagnostic CLI:** `scripts/diagnose_calibration_breakdown.py`

---

## Summary

**Verdict: STOP Plan 7 at Phase 0.** The diagnostic surfaced that Plan 7's premise — NB-2 count distributions are too narrow at [p10, p90] — is the opposite of the empirical reality. Per-stat NB-2 count distributions are *over*-covering at [p10, p90] by ~16 percentage points across all 16 (position, year) cells. Pinball-loss fitting at q=0.10 / q=0.90 would *narrow* the count distributions, opposite the direction needed to close the composite [p10, p90] coverage gap (-0.062 mean vs Model A).

The composite gap mechanism lives in **upper-tail behavior beyond p90** — outside what Plan 7's loss function targets. Follow-up TODO #30 captures the right next plan: upper-tail count calibration via either alternative pinball quantiles (q=0.90, q=0.95) or a different count-distribution family (ZIP, mixture).

---

## Per-cell breakdown (16 cells)

`gap = 0.80 - empirical_coverage`. **Negative gap = over-coverage** (interval too wide for the central 80% nominal). `share` is variance-of-actual share within each class (informational; raw `Var(actual)` is dominated by yards because yards have continuous-large-variance support and counts have integer-low-variance support).

| Position | Year | count_share | yards_share | count_gap | yards_gap | decision |
|---|---|---|---|---|---|---|
| QB | 2021 | 0.0002 | 0.9998 | -0.1536 | +0.0598 | stop_file_yards_plan |
| QB | 2022 | 0.0002 | 0.9998 | -0.1658 | -0.0210 | stop_file_yards_plan |
| QB | 2023 | 0.0002 | 0.9998 | -0.1618 | +0.0260 | stop_file_yards_plan |
| QB | 2024 | 0.0002 | 0.9998 | -0.1553 | +0.0075 | stop_file_yards_plan |
| RB | 2021 | 0.0003 | 0.9997 | -0.1614 | +0.0195 | stop_file_yards_plan |
| RB | 2022 | 0.0002 | 0.9998 | -0.1660 | -0.0006 | stop_file_yards_plan |
| RB | 2023 | 0.0003 | 0.9997 | -0.1609 | -0.0128 | stop_file_yards_plan |
| RB | 2024 | 0.0002 | 0.9998 | -0.1582 | -0.0096 | stop_file_yards_plan |
| TE | 2021 | 0.0003 | 0.9997 | -0.1777 | +0.0028 | stop_file_yards_plan |
| TE | 2022 | 0.0004 | 0.9996 | -0.1735 | +0.0302 | stop_file_yards_plan |
| TE | 2023 | 0.0003 | 0.9997 | -0.1813 | +0.0071 | stop_file_yards_plan |
| TE | 2024 | 0.0003 | 0.9997 | -0.1879 | +0.0274 | stop_file_yards_plan |
| WR | 2021 | 0.0002 | 0.9998 | -0.1656 | +0.0111 | stop_file_yards_plan |
| WR | 2022 | 0.0002 | 0.9998 | -0.1732 | +0.0336 | stop_file_yards_plan |
| WR | 2023 | 0.0002 | 0.9998 | -0.1761 | +0.0013 | stop_file_yards_plan |
| WR | 2024 | 0.0002 | 0.9998 | -0.1653 | +0.0001 | stop_file_yards_plan |

**Aggregate:** count gap mean **-0.169** (range -0.188 to -0.154); yards gap mean **+0.011** (range -0.021 to +0.060). All 16 cells route to `stop_file_yards_plan` under the gap-based decision rule.

The verdict label `stop_file_yards_plan` is correct in the narrow sense ("of the two classes, yards is more on-direction at [p10, p90] for the composite under-cover gap") but undersells the real finding, which is broader: **neither per-stat p10/p90 fix would close the composite gap.** The composite gap is an upper-tail phenomenon.

---

## Mechanism — why per-stat over-coverage coexists with composite under-coverage

For low-mean count NB-2 (e.g., RB receiving_tds with μ ≈ 0.4, dispersion ≈ 5):

1. **Discrete support concentrates mass at {0, 1}.** P(X=0) ≈ 0.69, P(X=1) ≈ 0.26. So [p10, p90] = [0, 1].
2. **Empirical actuals also concentrate at {0, 1}** (~95% of rows in real data). Coverage of [0, 1] band is therefore ~0.95-0.96 — well above the 80% nominal.
3. **The thin upper tail** (P(X≥2) ≈ 0.05 under the model) under-represents the empirical upper tail (~7-10% of rows have actuals at 2-3). This is what Plan 5c's PM described as "too narrow at the [p10, p90] tails" — the wording is imprecise; the narrowness is at p95-p99, not at p10/p90.
4. **Composite [p10, p90] under-cover comes from upper-tail count outliers.** When a count actual lands at 2-3 TDs (5-10% of rows), composite fantasy points jump 12-18 fp (TD weight × 6). The composite p90 — set mostly by yards width plus expected count mean — is exceeded. So composite under-covers on tail rows precisely where count distributions have under-cover at p95+, not at p90.

The key insight: **per-stat coverage at [p10, p90] does not decompose to composite coverage at [p10, p90].** Composite is a convolution of per-stat distributions; convolution behavior at the central interval (p10/p90) is dominated by the wider distribution (yards) plus the *expected* contribution of the narrower distribution (counts), not by the narrower distribution's own [p10, p90] coverage.

---

## Why Plan 7's premise was misaligned

Plan 7's spec stated: *"Mean μ continues to come from `lgb.LGBMRegressor(objective="poisson")`. Dispersion α is re-fit by minimizing pinball loss at q=0.10 and q=0.90 on a held-out validation year."* The implicit assumption: NB-2 dispersion is fit too high (intervals too narrow) at p10/p90, so pinball fitting will widen.

The diagnostic shows the opposite. For a fixed mu and dispersion that produces over-coverage at p10/p90, pinball-loss minimization at q=0.10/q=0.90 will choose a *narrower* dispersion to bring empirical coverage closer to nominal 0.80. That's the wrong direction for the composite gap.

A pinball fit at upper-tail quantiles (e.g., q=0.95) would bring the count distribution's upper tail closer to empirical. That *might* close part of the composite gap. But it's a different plan, with different mechanics (probably needs different α per quantile or a different distribution family altogether).

---

## What I should have caught earlier

Three things, ordered by leverage:

1. **Trusted Plan 5c's mechanism statement without empirically checking it.** Plan 5c's "NB-2 distribution is too narrow at the [p10, p90] tails on held-out years" is the load-bearing claim. The Phase 0 diagnostic was the first time anyone computed per-stat coverage at p10/p90 on real data. A two-line python check at scoping time would have surfaced the contradiction before the spec was written.

2. **The Phase 0 diagnostic itself had a structural flaw.** Per-stat coverage doesn't decompose composite coverage (convolution, not weighted average). The right diagnostic — counterfactual replacement (swap count distributions in C-NB rows for A's count distributions; re-sample composite; measure coverage closure) — would have given a cleaner answer about what fixing each class actually buys. The diagnostic shipped here measures a related-but-not-identical quantity. It happened to surface the right verdict (stop) for the right reason (per-stat counts already over-covering) — but only after careful interpretation.

3. **Missed the discreteness math at scoping.** At μ ≈ 0.4, NB-2 puts most mass on {0, 1}. [p10, p90] = [0, 1] trivially over-covers at the 80% nominal. This is mechanical, not subtle. A few minutes of pen-and-paper would have caught it.

Cost of getting it wrong: ~30 minutes of conversation + 10 minutes of compute. Cheap relative to the alternative (build Phase 1 + Phase 2 in full, discover the fix moves coverage the wrong direction, then unwind).

---

## Recommendation

**Stop Plan 7.** The diagnostic CLI ships as reusable research output (it can be re-pointed at any future per-row backtest output). The spec and plan stay in `docs/superpowers/{specs,plans}/` as record of decision — future readers see what was tried and why it stopped.

**File TODO #30: upper-tail count calibration follow-up.** Three candidate mechanisms, in order of expected leverage:

1. **Pinball-loss dispersion fit at upper-tail quantiles.** Same machinery as Plan 7 but `quantiles=(0.90, 0.95)` or `(0.95,)` only. Targets the actual gap location. Cheapest if it works.
2. **Switch count family to ZIP (zero-inflated Poisson).** Handles the zero-mass / non-zero-tail separation explicitly rather than via NB-2's overdispersion knob. Fundamental distribution change.
3. **Mixture: explicit point mass at 0 + heavier-tailed continuous-on-positive-integers distribution.** Most flexible, most code surface. Defer until 1 and 2 are tried.

Defer indefinitely if the user decides composite calibration shortfall is acceptable as-is (the "known limitation" framing from Plan 5c's PM). None of the planned downstream consumers (Draft Hub, start/sit, DFS lineup optimizer) depend on a perfectly calibrated [p10, p90] interval — they consume mean and rank.
