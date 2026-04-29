# §1.3 Adoption gate template (Plan 8)

**For author:** copy the body below into your new spec's §1.3 section.
Replace `<CANDIDATE_MODEL_CLASS>` and any other angle-bracket placeholders
with concrete values. Do **not** include / symlink this file — copy inline,
so your spec carries the gate it was evaluated under as record-of-decision.

**Spec history:** introduced in Plan 8
(`docs/superpowers/specs/2026-04-29-plan-8-gate-redesign-design.md`),
replacing the prior §1.3 "three-criteria, no-significance-test" gate that
killed Plans 3e / 5 / 5b / 5c / 7 / 6 from sampling noise on a metric
no consumer needs.

---

## 1.3 Adoption gate

Adoption decisions are **per position**. For each `Position P ∈ {QB, RB, TE, WR}`,
the adoption gate compares `<CANDIDATE_MODEL_CLASS>` against the incumbent
`_PositionDispatch[P].default_model_class`.

**Inputs.** Per-row predictions from both classes for the same
`(gsis_id, season, week)` rows across all held-out years (currently 2021–2024),
pulled from a single backtest run's `results.parquet`. After pairing, position P
contributes ~3,000–8,000 paired rows.

**Statistical machinery.** Paired bootstrap with `n_bootstrap=1000`,
deterministic seed `42`. Resampling unit is the paired player-week — both
candidate and incumbent are scored on the same draw.

**Per-position metrics.**
- **RMSE delta** (`candidate - incumbent`): pooled across all held-out years.
  Negative = candidate wins.
- **Spearman delta**: per-year Spearman computed within each held-out year,
  then averaged unweighted across years.

Per-cell breakdowns (one row per held-out year) are emitted for inspection
but do **not** gate adoption.

**Verdict rule.**
```
PASS_RMSE      := rmse_delta.hi_95     <  0.0
PASS_SPEARMAN  := spearman_delta.lo_95 > -0.02

if  PASS_RMSE and  PASS_SPEARMAN:  ADOPT
if  PASS_RMSE and !PASS_SPEARMAN:  MARGINAL — investigate before adopting
if !PASS_RMSE and  PASS_SPEARMAN:  DO_NOT_ADOPT
if !PASS_RMSE and !PASS_SPEARMAN:  DO_NOT_ADOPT
```

**What this gate does not check:**
- No per-cell pass/fail — per-year deltas are informational; only the
  position-pooled CI gates.
- No Spearman-improvement requirement — only the catastrophic-regression floor.
- No calibration check at all. `weekly_calibration_*` and
  `season_calibration_*` continue to be emitted into the snapshot for
  monitoring; the adoption decision ignores them.
- No "max worse cell" floor — sampling variation on a single year is not
  adoption-blocking.

**Tie-breaking** when multiple candidates ADOPT for the same position: the
candidate with the most-negative `rmse_delta.point` is selected. Document
the contender chain in this spec.

**Adoption is manual.** `scripts/adoption_gate.py` emits a report; a human
reads the verdicts and edits `_PositionDispatch[P].default_model_class` for
any position where the verdict is `ADOPT`. The CLI never writes to source.

**Tooling.** Run:
```
python -m scripts.adoption_gate \
  --run data/backtest/run_<ts> \
  --candidate <CANDIDATE_MODEL_CLASS> \
  --csv-out reports/adoption_gate_<CANDIDATE_MODEL_CLASS>.csv
```

Capture the per-(position) verdict + CI table in this spec's verdict section.
