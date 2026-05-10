# WR Receiving Stats Target Decomposition Probe — Summary

**Date:** 2026-05-10
**Branch:** `feat/probe-target-decomposition`
**Spec:** `docs/superpowers/specs/2026-05-10-target-decomposition-probe-design.md`
**Plan:** `docs/superpowers/plans/2026-05-10-target-decomposition-probe.md`
**Commit:** [fill in after final commit]
**Predecessor:** PR #31 (refined-unit weather strict-replace integration — full-revert × 2; closes broad-cut weather direction on RB/WR). With the weather track closed, target decomposition (TODO #23) is the next untouched model-improvement axis. This is the **first probe in the project to test a model architecture change** (volume × efficiency factor decomposition) rather than a feature addition.

## Verdict

**Family verdict: SIGNAL (marginal magnitude) — receptions cell only**. Per spec §4: **≥ 1 SIGNAL, no REGRESSION → greenlight the integration plan**, with the explicit caveat that the binding cell's expected composite-fpts Δ is **-0.0042 fpts**, just below the §5 risk #1 ~0.005 fpts flag threshold. Coverage was strictly above the 0.95 threshold across all four eval years (no relaxation invoked), so the PR #31 MARGINAL rule for "coverage relaxation + magnitude < 0.005" does not strictly apply — but the spec §5 risk #1 mitigation flag does. Integration is greenlit; the integration plan's go/no-go gate must weight CI strength (strictly below zero) against the small absolute magnitude.

## Per-stat verdicts (pooled across 2021-2024)

| Stat | n_paired | RMSE direct | RMSE decomposed | Δ-RMSE | 95% CI | Verdict | Expected composite-fpts Δ |
|---|---:|---:|---:|---:|---|:---:|---:|
| receptions | 8460 | 2.0324 | 2.0282 | -0.0042 | [-0.0079, -0.0004] | **SIGNAL** | **-0.0042 fpts** |
| receiving_yards | 8460 | 31.1654 | 31.1600 | -0.0054 | [-0.0601, +0.0492] | NULL | -0.0005 fpts |
| receiving_tds | 8460 | 0.4793 | 0.4788 | -0.0005 | [-0.0011, +0.0002] | NULL | -0.0029 fpts |

Notes:
- ESPN PPR coefficients applied: receptions = +1.0, receiving_yards = +0.1, receiving_tds = +6.0.
- receptions CI is strictly below zero — structurally SIGNAL. Magnitude -0.0042 fpts is just under the §5 risk #1 ~0.005 fpts flag, so flagged here.
- receiving_yards Δ point is favorable (-0.0054 RMSE) but CI brackets zero by a wide margin (±0.05 fpts). The yards/target factor and the targets factor are predicted noisily enough at the WR-pop level that the product's RMSE is statistically indistinguishable from the direct ridge.
- receiving_tds Δ is essentially zero (-0.0005 RMSE; -0.0029 fpts after × 6.0 coefficient). The CI brackets zero — `td_rate_per_target` is zero-inflated and high-variance, the predicted product cannot improve on the direct ridge.
- If signs flip at integration time, the marginal magnitude on the binding cell would land squarely in the PR #31 retrospective scenario.

## Coverage cross-check

| Eval year | Eval n | Eval (targets > 0) | Train n | Train (targets > 0) |
|---:|---:|---:|---:|---:|
| 2021 | 2109 | 0.988 | 3819 | 0.993 |
| 2022 | 2102 | 0.985 | 5220 | 0.993 |
| 2023 | 2201 | 0.981 | 6282 | 0.994 |
| 2024 | 2048 | 0.984 | 7590 | 0.993 |

Coverage threshold (0.95 default) was **met across all four eval years and all four training windows**. Lowest observed: 0.981 (2023 eval). No coverage relaxation was invoked, so the PR #31 retrospective MARGINAL rule ("magnitude < 0.005 fpts under coverage relaxation") does not directly apply. The §5 risk #1 magnitude flag for the receptions cell (-0.0042 fpts < ~0.005) is surfaced independently in the per-stat table and the Decision log.

## Factor residual correlation (Pearson ρ per eval year)

Per spec §5 risk #2, |ρ| > 0.2 in any year is a documented caveat — measures redundancy between the predicted-volume residual and the predicted-efficiency residual, where high correlation would mean the multiplicative decomposition is double-counting a shared signal axis.

| Stat | 2021 | 2022 | 2023 | 2024 |
|---|---:|---:|---:|---:|
| receptions | -0.008 | +0.031 | +0.016 | -0.012 |
| receiving_yards | -0.048 | +0.024 | -0.029 | -0.024 |
| receiving_tds | +0.001 | +0.015 | -0.030 | -0.035 |

All twelve ρ values are well under |0.05|, and none approach the 0.2 caveat threshold. The decomposition cleanly separates volume from efficiency signal axes on the WR receiving cell — no risk #2 caveat fires. This is the strongest mechanism-level finding from the probe: even though only receptions reaches a structural SIGNAL verdict, the orthogonality of the volume vs. efficiency residuals across all three stats × all four years means the integration plan's `ProductDistribution` composition will not suffer from systematic double-counting. The integration plan should still verify this property holds when factor-appropriate sub-models replace the ridge baselines.

## Decision log

The spec §4 branch that fires is **"≥ 1 SIGNAL, no REGRESSION → greenlight integration plan"**. The receptions cell delivered Δ-RMSE -0.0042 with 95% CI [-0.0079, -0.0004] strictly below zero across n_paired = 8460 — a structural SIGNAL with the bootstrap CI clearing zero. receiving_yards and receiving_tds both returned NULL (CIs bracket zero), and no cell returned REGRESSION (no CI is strictly above zero). The integration plan must therefore name and scope, per spec §7 follow-ups: (1) a `DecomposedBaselineModel` subclass of `BaselineModel` with per-stat decomposition opt-in; (2) `ProductDistribution` + a coherent within-row sampling helper so the same per-row `targets` draw flows into all decomposed stats' composed `SampledDistribution`s; (3) factor-appropriate sub-model classes (logistic for `catch_rate`, log-link Gamma for `yards_per_target`, Poisson / NB-2 for `targets`) deliberately deferred to a separate probe + integration cycle gated on this verdict; and (4) the composite-fpts adoption gate on `(DecomposedBaselineModel, WR)` vs production `(EnsembleModel, WR)` with the §1.3.5 per-position contingency matrix.

The integration plan's go/no-go decision must explicitly weight the binding cell's marginal magnitude. The receptions cell composite-fpts Δ is -0.0042 fpts — just below the §5 risk #1 ~0.005 fpts flag threshold. The CI clears zero on the stat-level RMSE, but the composite-fpts dilution at the adoption gate could shrink the absolute fpts magnitude further (the gate compares full-composite fpts predictions, not single-stat RMSE deltas). Two refined-unit candidates remain open under TODO #23 for separate probes if this integration's adoption gate returns DO_NOT_ADOPT: (a) the 3-factor `targets × catch_rate × yards_per_reception` decomposition (more conservative than `targets × yards_per_target` on the yards arm), (b) red-zone-shares × red-zone TD rate for the TD arm. Neither is queued at this PR — the integration plan can opt in to only the SIGNAL stat (receptions) at the model level, leaving receiving_yards and receiving_tds on direct ridges and avoiding the high-variance ratio sub-models for the first integration cycle.

## Recurring "QB augment regression" check

Not applicable — this probe has no augment / swap modes; it tests a model architecture change on WR only. The recurring QB augment regression pattern documented across PRs #23 / #24 / #25 / #28 is feature-additions-on-QB-specific and does not apply here.

## Spec gaps caught + fixed during execution

1. **Task 1 — ruff Unicode rules + mypy strict on sklearn returns.** Source-code docstrings and string literals required ASCII (ruff RUF001 / RUF002 — Greek letters like Δ, ρ disallowed in `.py` files), and typed local variables were needed around sklearn `Ridge.predict` returns to satisfy mypy strict (sklearn returns `Any`).
2. **Task 3 — bootstrap floor on synthetic fixtures.** `paired_bootstrap_rmse_delta` requires `n_paired >= 100`; the synthetic test fixtures needed `n_per_season` bumps to 120 to clear the floor. Real-data run with the full WR cache clears this trivially (8460 paired rows per stat).
3. **Task 3 — UTF-8 encoding on Windows.** `path.write_text` defaults to cp1252 on Windows, but the markdown report bodies use em-dashes; explicit `encoding="utf-8"` required on the write paths.
4. **Task 4 — mypy `mypy_path` collision.** The plan's combined invocation hits a pre-existing "Source file found twice" error when run on `scripts/foo.py` + `tests/test_scripts/test_foo_cli.py` together (mypy_path collision between `scripts/` and `tests/test_scripts/`); the canonical `mypy src tests` invocation is clean, and the script-specific check uses the targeted form.
