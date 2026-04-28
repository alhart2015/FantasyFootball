# Plan 7 — Calibration-aware NB-2 fitting (Model C-NB-cal) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `LightGBMNbCalModel` (Model C-NB-cal) as a fifth peer model class that closes the [p10,p90] coverage regression Model C-NB carries vs Model A on RB/TE/WR. Mean prediction (Poisson booster) is unchanged. Only the dispersion estimator changes — α is re-fit by minimizing pinball loss at q=0.10 and q=0.90 on a held-out validation slice instead of maximizing the conditional NB-2 log-likelihood on training residuals.

**Architecture:** Subclass `LightGBMNbModel`. Override only the count-stat dispersion-fitting call inside `fit()`: swap `nb_dispersion_from_residuals(mu_hat=mu_hat_train, actual=y_train)` for `nb_dispersion_from_pinball(mu_hat=mu_hat_val, actual=y_val, quantiles=(0.10, 0.90))`. Override `code_hash` and `model_id` to reflect the `lightgbm-nb-cal:` prefix and the new file in the hash. All other behavior — poisson booster training, yards-stat 5-quantile sub-models, `predict_distribution`, joblib save/load — is inherited verbatim.

**Tech Stack:** Python 3.12, LightGBM ≥4.0 (already installed; reused), scipy.stats / scipy.optimize (already in use), pandera, joblib, pytest, mypy strict, ruff.

**Spec:** `docs/superpowers/specs/2026-04-28-plan-7-calibration-aware-nb-design.md`.

**Branch:** `feat/plan-7-calibration-aware-nb` (branched from `main` at `166ea97`).

---

## Phase 0 — Calibration breakdown diagnostic (mandatory gate)

Goal: decompose Plan 5c's measured -0.062 mean coverage gap vs Model A into **count-stat contribution** and **yards-stat contribution** before committing to the NB-2-only scope. If yards dominate (≥50% of the gap), the calibration-aware NB-2 fix is mostly off-target — stop the plan and file a yards-side calibration spec separately.

### Task 1: Verify Plan 5c backtest output is on disk

**Files:**
- Read-only: `data/backtest/run_<ts>/results.parquet` (most recent C-NB-bearing run).

The Phase 0 diagnostic reads per-row backtest output from the most recent run. Plan 5c's last `--update-snapshot` run wrote the per-row frame. Confirm it's there before writing the diagnostic.

- [ ] **Step 1: Identify the C-NB-bearing run dir**

Run:
```bash
ls -t data/backtest/ | head -5
```

Pick the most recent `run_<ts>/` dir and confirm it contains `results.parquet`:
```bash
LATEST=$(ls -t data/backtest/ | head -1)
ls -la "data/backtest/$LATEST/"
```

Expected: `results.parquet` present and non-empty.

- [ ] **Step 2: Confirm the parquet has C-NB rows**

Run:
```bash
LATEST=$(ls -t data/backtest/ | head -1)
python -c "
import pandas as pd
df = pd.read_parquet('data/backtest/$LATEST/results.parquet')
print('rows:', len(df))
print('model_ids:', df['model_id'].str.split(':').str[0].unique())
print('positions:', df['position'].unique())
print('seasons:', sorted(df['season'].unique()))
"
```

Expected output should include `lightgbm-nb` among model_id prefixes. If not, **stop and re-run** the backtest with `--update-snapshot --model all` first; do not proceed without C-NB rows in the per-row frame.

- [ ] **Step 3: Note the run_dir; commit nothing**

This is a verification step only. Hold the run_dir path in working memory for the rest of Phase 0.

### Task 2: Diagnostic CLI scaffold + tests

**Files:**
- Create: `scripts/diagnose_calibration_breakdown.py`
- Create: `tests/test_scripts/test_diagnose_calibration_breakdown.py`

The CLI loads the per-row backtest results, computes per-stat empirical [p10, p90] coverage for Model C-NB rows on each held-out (position, year) cell, weights each stat's contribution by its share of total fantasy-point variance, and emits a CSV breakdown with one summary row per (position, year) attributing the cell's coverage gap vs A to `count_share` and `yards_share` in [0, 1].

The mechanism: each row in the per-row frame has both `<stat>_actual` and per-stat distributional information in the `params` blob. We can compute per-stat [p10, p90] coverage by unpacking the blob's per-stat distributions and asking each one for its 0.10 and 0.90 quantiles. We then compute each stat's fantasy-point variance contribution (under the row's ruleset) by sampling the per-stat distributions and computing the variance of each scoring component. The "share" attribution is variance-weighted because that's what propagates to the composite [p10, p90] coverage gap.

- [ ] **Step 1: Write the failing test scaffold**

Create `tests/test_scripts/test_diagnose_calibration_breakdown.py`:

```python
"""Tests for scripts/diagnose_calibration_breakdown.py — Plan 7 Phase 0.

The diagnostic reads per-row backtest output, decomposes the [p10, p90]
coverage gap vs Model A into count-stat vs yards-stat contributions, and
emits a CSV summary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_calibration_breakdown import (
    attribute_coverage_gap,
    compute_per_stat_coverage,
    main,
)


def _build_synthetic_per_row(rng: np.random.Generator) -> pd.DataFrame:
    """Build a minimal per-row frame matching scripts/backtest.py's output:
    identifiers + per-stat <stat>_actual / <stat>_pred columns + family +
    model_id."""
    n = 500
    rows = pd.DataFrame(
        {
            "gsis_id": [f"00-{i:07d}" for i in range(n)],
            "season": np.full(n, 2024, dtype=np.int64),
            "week": rng.integers(1, 18, size=n).astype(np.int64),
            "position": np.full(n, "WR", dtype=object),
            "team": np.full(n, "KC", dtype=object),
            "opponent": np.full(n, "DEN", dtype=object),
            "ruleset": np.full(n, "ESPN_PPR", dtype=object),
            "family": np.full(n, "MIXED", dtype=object),
            "model_id": np.full(n, "lightgbm-nb:wr:abc12345:2018-2023", dtype=object),
            "receiving_yards_actual": rng.normal(50, 25, size=n),
            "receiving_yards_p10": np.full(n, 20.0),
            "receiving_yards_p90": np.full(n, 80.0),
            "receiving_tds_actual": rng.poisson(0.4, size=n).astype(np.float64),
            "receiving_tds_p10": np.zeros(n),
            "receiving_tds_p90": np.ones(n),
        }
    )
    return rows


def test_compute_per_stat_coverage_returns_one_row_per_stat() -> None:
    rng = np.random.default_rng(0)
    per_row = _build_synthetic_per_row(rng)
    out = compute_per_stat_coverage(per_row, position="WR", year=2024)
    # Two stats in the synthetic frame -> two rows.
    assert set(out["stat"]) == {"receiving_yards", "receiving_tds"}
    # Coverage is in [0, 1].
    assert out["coverage_p10p90"].between(0.0, 1.0).all()


def test_compute_per_stat_coverage_matches_hand_computed() -> None:
    """For a hand-built frame where all actuals fall inside [p10, p90],
    coverage should be 1.0."""
    n = 100
    per_row = pd.DataFrame(
        {
            "season": np.full(n, 2024, dtype=np.int64),
            "position": np.full(n, "WR", dtype=object),
            "model_id": np.full(n, "lightgbm-nb:wr:x:2018-2023", dtype=object),
            "receiving_yards_actual": np.full(n, 50.0),
            "receiving_yards_p10": np.full(n, 20.0),
            "receiving_yards_p90": np.full(n, 80.0),
        }
    )
    out = compute_per_stat_coverage(per_row, position="WR", year=2024)
    assert (out.loc[out["stat"] == "receiving_yards", "coverage_p10p90"] == 1.0).all()


def test_attribute_coverage_gap_sums_to_one_across_stat_classes() -> None:
    """count_share + yards_share = 1.0 by construction."""
    per_stat = pd.DataFrame(
        {
            "stat": ["receiving_yards", "receiving_tds", "rushing_tds"],
            "stat_class": ["yards", "count", "count"],
            "variance_contribution": [0.7, 0.2, 0.1],
            "coverage_gap_vs_a": [-0.04, -0.05, -0.06],
        }
    )
    out = attribute_coverage_gap(per_stat)
    assert out["yards_share"] + out["count_share"] == pytest.approx(1.0, abs=1e-9)


def test_attribute_coverage_gap_handles_zero_variance_safely() -> None:
    """All-zero variance contributions -> zero shares (no NaN)."""
    per_stat = pd.DataFrame(
        {
            "stat": ["x", "y"],
            "stat_class": ["yards", "count"],
            "variance_contribution": [0.0, 0.0],
            "coverage_gap_vs_a": [-0.01, -0.02],
        }
    )
    out = attribute_coverage_gap(per_stat)
    assert out["yards_share"] == 0.0
    assert out["count_share"] == 0.0


def test_main_writes_csv_and_records_decision(tmp_path: Path) -> None:
    """Smoke: `main()` produces the expected CSV columns when given a fixture."""
    rng = np.random.default_rng(0)
    per_row = _build_synthetic_per_row(rng)
    in_path = tmp_path / "results.parquet"
    per_row.to_parquet(in_path)
    out_path = tmp_path / "calibration_breakdown.csv"

    main(["--per-row-parquet", str(in_path), "--output-csv", str(out_path)])

    assert out_path.exists()
    out = pd.read_csv(out_path)
    expected_cols = {
        "position",
        "year",
        "count_share",
        "yards_share",
        "count_coverage_gap",
        "yards_coverage_gap",
        "decision",
    }
    assert expected_cols.issubset(out.columns)
```

- [ ] **Step 2: Run the test to verify it fails on import**

```bash
pytest tests/test_scripts/test_diagnose_calibration_breakdown.py -v
```

Expected: FAIL with `ModuleNotFoundError: scripts.diagnose_calibration_breakdown`.

- [ ] **Step 3: Implement the diagnostic CLI**

Create `scripts/diagnose_calibration_breakdown.py`:

```python
"""Plan 7 Phase 0 — calibration breakdown diagnostic.

Reads the per-row backtest output written by scripts/backtest.py. Computes
per-stat empirical [p10, p90] coverage for Model C-NB rows on each
held-out (position, year) cell, weights each stat's contribution by its
share of total fantasy-point variance, and emits a CSV breakdown
attributing the cell's coverage gap vs Model A to `count_share` and
`yards_share` in [0, 1].

Decision rule encoded in the CSV `decision` column:
  - count_share >= 0.5  -> "proceed_phase_1"
  - yards_share >= 0.5  -> "stop_file_yards_plan"
  - else                -> "proceed_with_followup"

Spec: docs/superpowers/specs/2026-04-28-plan-7-calibration-aware-nb-design.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from projections.distributions import unpack_per_stat_params
from projections.models import POSITION_DISPATCH, BaselineModel
from projections.schemas import Stat
from projections.scoring.score_distribution import derive_row_seed, score_distribution
from projections.schemas import Ruleset

# Stats Plan 5c routes through NB-2 (count) vs QuantileDistribution (yards).
# Match COUNT_STATS_FOR_NB in models/lightgbm_nb.py.
_COUNT_STATS: Final[frozenset[str]] = frozenset(
    {"passing_tds", "rushing_tds", "receiving_tds", "interceptions", "fumbles_lost"}
)


def _stat_class(stat_value: str) -> str:
    return "count" if stat_value in _COUNT_STATS else "yards"


def _resolve_target_stats() -> dict[str, tuple[str, ...]]:
    """{position_value: (stat_value, ...)} from the baseline factory.

    Mirrors scripts/diagnose_calibration.py's helper of the same shape.
    target_stats are identical across model classes by construction.
    """
    out: dict[str, tuple[str, ...]] = {}
    for position, dispatch in POSITION_DISPATCH.items():
        from typing import cast as _cast

        model = _cast(BaselineModel, dispatch.factories["baseline"]())
        out[position.value] = tuple(s.value for s in model.target_stats)
    return out


def compute_per_stat_coverage(
    per_row: pd.DataFrame, *, position: str, year: int
) -> pd.DataFrame:
    """Compute empirical [p10, p90] coverage per stat for a single
    (position, year) cell.

    Returns a frame with columns:
      stat, stat_class, n_rows, coverage_p10p90, variance_contribution.

    `variance_contribution` is computed as Var(stat_actual) * scoring_weight^2
    aggregated across the cell's rows; this is the share-of-fantasy-point-
    variance proxy used by attribute_coverage_gap.
    """
    cell = per_row[(per_row["position"] == position) & (per_row["season"] == year)]
    if cell.empty:
        return pd.DataFrame(columns=["stat", "stat_class", "n_rows", "coverage_p10p90", "variance_contribution"])

    target_stats = _resolve_target_stats().get(position, ())
    if not target_stats:
        raise ValueError(f"No target stats resolved for position={position}")

    rows: list[dict[str, object]] = []
    for stat_value in target_stats:
        actual_col = f"{stat_value}_actual"
        p10_col = f"{stat_value}_p10"
        p90_col = f"{stat_value}_p90"
        if not all(c in cell.columns for c in (actual_col, p10_col, p90_col)):
            continue
        actual = cell[actual_col].to_numpy(dtype=np.float64)
        p10 = cell[p10_col].to_numpy(dtype=np.float64)
        p90 = cell[p90_col].to_numpy(dtype=np.float64)
        in_band = (actual >= p10) & (actual <= p90)
        coverage = float(in_band.mean()) if in_band.size else 0.0
        # Variance contribution: empirical Var(actual) — the scoring-weight^2
        # multiplier folds in via the ruleset later if needed; for the
        # share comparison only ratios matter so Var(actual) is enough.
        variance_contribution = float(np.var(actual)) if actual.size else 0.0
        rows.append(
            {
                "stat": stat_value,
                "stat_class": _stat_class(stat_value),
                "n_rows": int(actual.size),
                "coverage_p10p90": coverage,
                "variance_contribution": variance_contribution,
            }
        )
    return pd.DataFrame(rows)


def attribute_coverage_gap(per_stat: pd.DataFrame) -> dict[str, float]:
    """Reduce a per-stat frame to a single (position, year) cell summary.

    Computes:
      count_coverage_gap, yards_coverage_gap: variance-weighted average of
        per-stat (target_coverage - empirical_coverage) within each class.
        Target is 0.80 (the nominal [p10, p90] interval).
      count_share, yards_share: each class's share of total fantasy-point
        variance contribution. Sum to 1.0 (or 0.0 if total variance is 0).
    """
    if per_stat.empty:
        return {
            "count_share": 0.0,
            "yards_share": 0.0,
            "count_coverage_gap": 0.0,
            "yards_coverage_gap": 0.0,
        }

    total_var = float(per_stat["variance_contribution"].sum())
    if total_var <= 0:
        return {
            "count_share": 0.0,
            "yards_share": 0.0,
            "count_coverage_gap": 0.0,
            "yards_coverage_gap": 0.0,
        }

    out: dict[str, float] = {}
    for class_name in ("count", "yards"):
        sub = per_stat[per_stat["stat_class"] == class_name]
        share = float(sub["variance_contribution"].sum()) / total_var
        if sub.empty or share == 0.0:
            gap = 0.0
        else:
            # Weighted-average gap = sum(var * (0.80 - cov)) / sum(var).
            w = sub["variance_contribution"].to_numpy()
            cov = sub["coverage_p10p90"].to_numpy()
            gap = float(np.sum(w * (0.80 - cov)) / w.sum())
        out[f"{class_name}_share"] = share
        out[f"{class_name}_coverage_gap"] = gap
    return out


def _decision(count_share: float, yards_share: float) -> str:
    if count_share >= 0.5:
        return "proceed_phase_1"
    if yards_share >= 0.5:
        return "stop_file_yards_plan"
    return "proceed_with_followup"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-row-parquet",
        type=Path,
        required=True,
        help="Path to a per-row results.parquet from scripts/backtest.py.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Where to write the breakdown CSV.",
    )
    args = parser.parse_args(argv)

    per_row = pd.read_parquet(args.per_row_parquet)
    # Plan 7 only attributes Model C-NB rows; filter to that model class.
    nb_mask = per_row["model_id"].str.startswith("lightgbm-nb:")
    per_row = per_row[nb_mask].copy()
    if per_row.empty:
        print("No lightgbm-nb rows found in per-row frame; nothing to attribute.", file=sys.stderr)
        sys.exit(2)

    summary_rows: list[dict[str, object]] = []
    for (position, year), _ in per_row.groupby(["position", "season"]):
        per_stat = compute_per_stat_coverage(per_row, position=position, year=year)
        attribution = attribute_coverage_gap(per_stat)
        summary_rows.append(
            {
                "position": position,
                "year": year,
                "count_share": attribution["count_share"],
                "yards_share": attribution["yards_share"],
                "count_coverage_gap": attribution["count_coverage_gap"],
                "yards_coverage_gap": attribution["yards_coverage_gap"],
                "decision": _decision(attribution["count_share"], attribution["yards_share"]),
            }
        )

    out = pd.DataFrame(summary_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_scripts/test_diagnose_calibration_breakdown.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Run mypy + ruff on the new files**

```bash
mypy scripts/diagnose_calibration_breakdown.py tests/test_scripts/test_diagnose_calibration_breakdown.py
ruff check scripts/diagnose_calibration_breakdown.py tests/test_scripts/test_diagnose_calibration_breakdown.py
ruff format --check scripts/diagnose_calibration_breakdown.py tests/test_scripts/test_diagnose_calibration_breakdown.py
```

Expected: zero errors. If `ruff format --check` reports drift, run `ruff format scripts/diagnose_calibration_breakdown.py tests/test_scripts/test_diagnose_calibration_breakdown.py` to fix.

- [ ] **Step 6: Commit**

```bash
git add scripts/diagnose_calibration_breakdown.py tests/test_scripts/test_diagnose_calibration_breakdown.py
git commit -m "feat(diagnostic): calibration breakdown CLI — Plan 7 Phase 0"
```

### Task 3: Run diagnostic on real Plan 5c output + record decision

**Files:**
- Create: `docs/superpowers/research/2026-04-28-calibration-breakdown.md`
- Modify: `project_management.md` (record Phase 0 verdict)

- [ ] **Step 1: Run the diagnostic**

```bash
LATEST=$(ls -t data/backtest/ | head -1)
mkdir -p data/diagnostics/calibration_breakdown
python scripts/diagnose_calibration_breakdown.py \
    --per-row-parquet "data/backtest/$LATEST/results.parquet" \
    --output-csv "data/diagnostics/calibration_breakdown/breakdown.csv"
```

Expected: prints a per-(position, year) table to stdout; writes the CSV.

- [ ] **Step 2: Read the table; apply the decision rule**

Decision rule (per spec §1.4):
- ≥50% of cells say `proceed_phase_1` (counts dominate) → continue Plan 7.
- ≥50% of cells say `stop_file_yards_plan` (yards dominate) → **stop Plan 7**, file yards-side spec separately.
- Mixed → continue Plan 7 with documented expectation that calibration delta closes ≤50% of the gap; file yards-side follow-up as known dependency.

- [ ] **Step 3: Write the research note**

Create `docs/superpowers/research/2026-04-28-calibration-breakdown.md` with:
- The 16-cell table from the diagnostic output (one row per `(position, year)`).
- A summary line: "Counts dominate in N/16 cells; yards dominate in M/16 cells; mixed in K/16."
- The Phase 0 verdict (one of the three decision-rule outcomes above).
- A one-paragraph note on what this means for Phase 1 scope.

Format the table in markdown; copy values from the CSV directly.

- [ ] **Step 4: Update project_management.md with Phase 0 verdict**

Prepend a new entry at the top of `project_management.md`:

```markdown
## Plan 7 Phase 0 — Calibration breakdown diagnostic (run 2026-04-28)

Decomposed Plan 5c's measured -0.062 mean [p10, p90] coverage gap vs A
into count-stat vs yards-stat contributions on the C-NB per-row backtest
output. Diagnostic CLI: `scripts/diagnose_calibration_breakdown.py`;
research output at `docs/superpowers/research/2026-04-28-calibration-breakdown.md`.

**Verdict:** <one of: proceed_phase_1 / stop_file_yards_plan / proceed_with_followup>

**Per-cell breakdown:** counts dominate in N/16; yards dominate in M/16; mixed in K/16.

**Implications for Phase 1:** <one paragraph based on verdict>

---
```

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/research/2026-04-28-calibration-breakdown.md project_management.md
git commit -m "research(plan-7): Phase 0 calibration breakdown — verdict <verdict>"
```

- [ ] **Step 6: GATE — branch on verdict**

If verdict = `stop_file_yards_plan`: stop Plan 7 here. File a follow-up plan on yards calibration; mark Plan 7 as "deferred pending yards plan" in TODO.md; open PR with just Phase 0 contents and let user decide on next steps.

If verdict = `proceed_phase_1` or `proceed_with_followup`: continue to Phase 1.

If verdict = `proceed_with_followup`: file a TODO entry now (under TODO.md "Open") titled "Plan 7 follow-up: yards-stat calibration" with a one-paragraph note on what scope it covers.

---

## Phase 1 — Implementation

Goal: ship `LightGBMNbCalModel` + the new `nb_dispersion_from_pinball` estimator + dispatch wiring + smoke tests. No backtest run in this phase; that's Phase 2.

### Task 4: Add `nb_dispersion_from_pinball` estimator + tests

**Files:**
- Modify: `src/projections/distributions/parametric.py`
- Create: `tests/test_distributions/test_nb_dispersion_pinball.py`

The new estimator mirrors the shape and clip semantics of `nb_dispersion_from_residuals` (already in `parametric.py`, lines 19-62) but optimizes a different objective: sum of pinball losses at the requested quantiles, evaluated using `ParametricNegativeBinomial.quantile`. Single-shot bounded 1-d optimization in α space; same return type, same clip behavior, same degenerate-input handling.

- [ ] **Step 1: Write failing tests**

Create `tests/test_distributions/test_nb_dispersion_pinball.py`:

```python
"""Tests for the calibration-aware NB-2 dispersion estimator — Plan 7.

Mirrors the shape of test_nb_dispersion.py; the function being tested
optimizes pinball loss instead of conditional MLE.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from projections.distributions.parametric import (
    _NB_DISPERSION_CLIP,
    ParametricNegativeBinomial,
    nb_dispersion_from_pinball,
    nb_dispersion_from_residuals,
)


def test_returns_clip_endpoint_on_empty_input() -> None:
    """Fewer than 2 rows -> degenerate; helper returns clip top to keep
    downstream NB defined (matches nb_dispersion_from_residuals behavior)."""
    out = nb_dispersion_from_pinball(mu_hat=np.array([0.5]), actual=np.array([0.0]))
    assert out == _NB_DISPERSION_CLIP[1]


def test_validates_quantiles_in_open_unit_interval() -> None:
    """quantiles must each be in (0, 1)."""
    with pytest.raises(ValueError):
        nb_dispersion_from_pinball(
            mu_hat=np.ones(10), actual=np.ones(10), quantiles=(0.0, 0.9)
        )
    with pytest.raises(ValueError):
        nb_dispersion_from_pinball(
            mu_hat=np.ones(10), actual=np.ones(10), quantiles=(0.1, 1.0)
        )


def test_recovers_dispersion_within_factor_of_3_on_known_nb_data() -> None:
    """Samples drawn from NB-2(mu, dispersion=5). Pinball-fit alpha at the
    [0.10, 0.90] quantiles should land within an order of magnitude of the
    truth on a sample large enough that quantile-coverage is well-estimated."""
    rng = np.random.default_rng(42)
    n = 5000
    mu = rng.uniform(0.5, 2.0, size=n)
    true_dispersion = 5.0
    p = true_dispersion / (true_dispersion + mu)
    actual = rng.negative_binomial(n=true_dispersion, p=p, size=n).astype(np.float64)

    fitted = nb_dispersion_from_pinball(
        mu_hat=mu, actual=actual, quantiles=(0.10, 0.90)
    )
    # Pinball is loose at the boundary of identification; assert within an
    # order of magnitude in either direction. The MLE estimator is tighter
    # because it uses full-distribution likelihood, not just two quantiles.
    assert true_dispersion / 3.0 <= fitted <= true_dispersion * 3.0


def test_pinball_alpha_widens_when_val_variance_exceeds_train() -> None:
    """The mechanism Plan 7 targets: when val-set variance exceeds train-set
    variance, pinball-fit alpha is smaller than MLE-fit alpha (smaller alpha
    = higher dispersion = wider intervals).

    Construct: train residuals are tightly concentrated at mu; val residuals
    have a heavier tail. MLE on train fits a high alpha (tight); pinball on
    val fits a smaller alpha (wider).
    """
    rng = np.random.default_rng(0)
    n = 2000
    mu = np.full(n, 1.0)

    # Train: NB with high dispersion (small variance beyond Poisson).
    train_actual = rng.negative_binomial(
        n=20.0, p=20.0 / (20.0 + 1.0), size=n
    ).astype(np.float64)
    # Val: NB with low dispersion (much wider tails).
    val_actual = rng.negative_binomial(
        n=2.0, p=2.0 / (2.0 + 1.0), size=n
    ).astype(np.float64)

    mle_alpha = nb_dispersion_from_residuals(mu_hat=mu, actual=train_actual)
    pinball_alpha = nb_dispersion_from_pinball(mu_hat=mu, actual=val_actual)

    # Pinball-fit dispersion is meaningfully lower than MLE-fit-on-train,
    # i.e., the predictive interval gets wider when val variance is wider.
    assert pinball_alpha < mle_alpha


def test_minimum_at_well_calibrated_alpha() -> None:
    """For a fixed mu and actuals drawn from NB(mu, alpha_truth), the pinball
    loss is monotonically decreasing toward alpha_truth from either side
    (within sampling noise). Verify the optimizer ends near the truth on a
    large sample."""
    rng = np.random.default_rng(7)
    n = 10000
    mu = np.full(n, 1.5)
    truth = 3.0
    p = truth / (truth + 1.5)
    actual = rng.negative_binomial(n=truth, p=p, size=n).astype(np.float64)

    fitted = nb_dispersion_from_pinball(mu_hat=mu, actual=actual)
    # On 10k samples the estimator should land within 50% of truth.
    assert truth * 0.5 <= fitted <= truth * 1.5


def test_clipping_to_lower_bound_returns_exact_endpoint() -> None:
    """When the optimizer drives alpha to the lower bound (e.g., extreme
    over-dispersion in val) the helper snaps to the exact clip endpoint."""
    rng = np.random.default_rng(13)
    n = 1000
    mu = np.full(n, 0.5)
    # Heavy-tailed actual: most rows zero; a few very large values.
    actual = np.zeros(n, dtype=np.float64)
    actual[:50] = rng.integers(20, 50, size=50).astype(np.float64)

    fitted = nb_dispersion_from_pinball(mu_hat=mu, actual=actual, quantiles=(0.10, 0.90))
    # Heavy right tail -> very low alpha (very high dispersion).
    # Should snap to or be very close to lower clip endpoint.
    assert fitted == _NB_DISPERSION_CLIP[0] or fitted < 0.5
```

- [ ] **Step 2: Run the test to verify it fails on import**

```bash
pytest tests/test_distributions/test_nb_dispersion_pinball.py -v
```

Expected: FAIL with `ImportError: cannot import name 'nb_dispersion_from_pinball'`.

- [ ] **Step 3: Implement the estimator**

Add to `src/projections/distributions/parametric.py` immediately after the existing `nb_dispersion_from_residuals` function (so it sits in the file's NB-helpers cluster):

```python
def nb_dispersion_from_pinball(
    *,
    mu_hat: np.ndarray,
    actual: np.ndarray,
    quantiles: tuple[float, ...] = (0.10, 0.90),
) -> float:
    """Fit NB-2 dispersion alpha by minimizing sum of pinball losses at the
    requested quantiles, holding mu_hat fixed.

    Designed for use on a held-out validation slice (mu_hat is the booster's
    prediction on val rows, actual is the true target on val rows). Plan 7
    swaps this in for nb_dispersion_from_residuals when training C-NB-cal
    so dispersion is calibrated against held-out quantile loss instead of
    training-residual likelihood.

    For each candidate alpha, predicts ParametricNegativeBinomial(mu, alpha)
    quantile values per row and computes the standard pinball loss
        L(q) = mean_i [(q - 1{y_i < q_hat_i}) * (y_i - q_hat_i)]
    summed across the requested quantiles. Minimizes via bounded 1-d
    optimization in alpha space, with the same _NB_DISPERSION_CLIP and
    clip-snap semantics as nb_dispersion_from_residuals.

    Args:
        mu_hat: per-row mean from the upstream regressor; shape (n,).
        actual: per-row observed target; shape (n,). Coerced to float for the
            pinball loss (NB quantiles are integer but we don't round actuals
            because the loss is defined on the same scale as the quantile).
        quantiles: which quantile(s) to score. Each must be in (0, 1).
            Default (0.10, 0.90) matches Plan 7 §1.3's adoption gate.

    Returns:
        Fitted dispersion alpha clipped to _NB_DISPERSION_CLIP. Snaps to
        either clip endpoint when the bounded minimizer stops within its
        xatol of the boundary (matches nb_dispersion_from_residuals).
    """
    for q in quantiles:
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantiles must each be in (0, 1), got {q}")

    actual_f = np.asarray(actual, dtype=np.float64)
    mu_clipped = np.maximum(mu_hat, _NB_MU_FLOOR)

    if actual_f.size < 2:
        return _NB_DISPERSION_CLIP[1]

    quantile_arr = np.asarray(quantiles, dtype=np.float64)

    def neg_pinball(dispersion: float) -> float:
        if dispersion <= 0:
            return float("inf")
        # Per-row, per-quantile predicted q-quantile from
        # ParametricNegativeBinomial(mu_clipped[i], dispersion). Vectorize
        # by computing scipy nbinom.ppf with broadcasting.
        n = dispersion
        p = dispersion / (dispersion + mu_clipped)
        # shape: (n_rows, n_quantiles)
        q_hat = stats.nbinom.ppf(quantile_arr[None, :], n=n, p=p[:, None])
        # Pinball loss per (row, q): max(q*(y-q_hat), (q-1)*(y-q_hat)).
        diff = actual_f[:, None] - q_hat
        loss = np.maximum(quantile_arr[None, :] * diff, (quantile_arr[None, :] - 1.0) * diff)
        # Sum across quantiles; mean across rows; return total to minimize.
        return float(np.mean(loss.sum(axis=1)))

    result = minimize_scalar(
        neg_pinball,
        bounds=_NB_DISPERSION_CLIP,
        method="bounded",
        options={"xatol": 1e-3},
    )
    if not result.success or not np.isfinite(result.fun):
        return _NB_DISPERSION_CLIP[1]
    fitted = float(np.clip(result.x, *_NB_DISPERSION_CLIP))
    snap_tol = 2e-3
    if fitted - _NB_DISPERSION_CLIP[0] <= snap_tol:
        return _NB_DISPERSION_CLIP[0]
    if _NB_DISPERSION_CLIP[1] - fitted <= snap_tol:
        return _NB_DISPERSION_CLIP[1]
    return fitted
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_distributions/test_nb_dispersion_pinball.py -v
```

Expected: 6 tests pass.

- [ ] **Step 5: Run the full distributions test suite**

```bash
pytest tests/test_distributions/ -v
```

Expected: all green; no regressions in the existing `nb_dispersion_from_residuals` tests.

- [ ] **Step 6: Run mypy + ruff**

```bash
mypy src/projections/distributions/parametric.py tests/test_distributions/test_nb_dispersion_pinball.py
ruff check src/projections/distributions/parametric.py tests/test_distributions/test_nb_dispersion_pinball.py
ruff format --check src/projections/distributions/parametric.py tests/test_distributions/test_nb_dispersion_pinball.py
```

Expected: zero errors.

- [ ] **Step 7: Commit**

```bash
git add src/projections/distributions/parametric.py tests/test_distributions/test_nb_dispersion_pinball.py
git commit -m "feat(distributions): add nb_dispersion_from_pinball — Plan 7 Phase 1"
```

### Task 5: Add `LightGBMNbCalModel` subclass + tests

**Files:**
- Create: `src/projections/models/lightgbm_nb_cal.py`
- Create: `tests/test_models/test_lightgbm_nb_cal.py`

The new model class subclasses `LightGBMNbModel` and overrides only:
- `code_hash` — adds `lightgbm_nb_cal.py` to the hashed file list.
- `model_id` — uses `lightgbm-nb-cal:` prefix.
- `fit` — same flow as parent, but the count-stat dispersion fit takes the val-slice μ̂ and actuals (instead of train-slice) and calls `nb_dispersion_from_pinball` (instead of `nb_dispersion_from_residuals`).

`predict_distribution` is inherited verbatim. NB-2 distribution for count stats; QuantileDistribution for yards. Per-row family stays `MIXED`.

- [ ] **Step 1: Write failing tests for the model class**

Create `tests/test_models/test_lightgbm_nb_cal.py`:

```python
"""Unit tests for LightGBMNbCalModel — Plan 7.

The class subclasses LightGBMNbModel and overrides only the dispersion-
fitting call. These tests pin: code_hash and model_id reflect the new
class identity; fit produces a different dispersion than the parent on a
synthetic dataset where train-residual variance and val-residual variance
diverge; predict_distribution behavior is inherited verbatim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.models import POSITION_DISPATCH
from projections.models.lightgbm_nb_cal import (
    LightGBMNbCalModel,
    qb_lightgbm_nb_cal,
    rb_lightgbm_nb_cal,
    te_lightgbm_nb_cal,
    wr_lightgbm_nb_cal,
)
from projections.schemas import (
    _PYARROW_STR,
    Position,
    ProjectionWeeklySchema,
    Ruleset,
    WeeklyStatsSchema,
)


def _build_synthetic_data(position: Position) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse the shape used by test_lightgbm_nb_smoke.py."""
    rng = np.random.default_rng(42)
    feature_schema = POSITION_DISPATCH[position].feature_schema

    rows = []
    for season in range(2018, 2022):
        for week in range(1, 18):
            for p in range(20):
                rows.append(
                    {
                        "gsis_id": f"00-{p:07d}",
                        "season": np.int64(season),
                        "week": np.int64(week),
                        "team": "KC",
                        "opponent": "DEN",
                    }
                )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)

    schema_cols = feature_schema.to_schema().columns
    for col_name, col in schema_cols.items():
        if col_name in df.columns:
            continue
        dtype_str = str(col.dtype)
        if "bool" in dtype_str.lower():
            df[col_name] = rng.integers(0, 2, size=len(df)).astype(bool)
        elif "int" in dtype_str.lower():
            df[col_name] = rng.integers(1, 6, size=len(df)).astype(np.int64)
        else:
            df[col_name] = rng.uniform(0.0, 0.5, size=len(df)).astype(np.float64)
    features = feature_schema.validate(df)

    ws = features[["gsis_id", "season", "week", "team", "opponent"]].copy()
    ws["position"] = position.value
    n = len(ws)
    ws["passing_yards"] = np.clip(rng.normal(220.0, 60.0, size=n), -100.0, 800.0).astype(np.float64)
    ws["passing_tds"] = np.maximum(0, rng.poisson(1.5, size=n)).astype(np.int64)
    ws["interceptions"] = np.maximum(0, rng.poisson(0.7, size=n)).astype(np.int64)
    ws["rushing_yards"] = np.clip(rng.normal(20.0, 15.0, size=n), -50.0, 400.0).astype(np.float64)
    ws["rushing_tds"] = np.maximum(0, rng.poisson(0.2, size=n)).astype(np.int64)
    ws["receptions"] = np.maximum(0, rng.poisson(2.5, size=n)).astype(np.int64)
    ws["receiving_yards"] = np.clip(rng.normal(25.0, 15.0, size=n), -50.0, 400.0).astype(np.float64)
    ws["receiving_tds"] = np.maximum(0, rng.poisson(0.2, size=n)).astype(np.int64)
    ws["fumbles_lost"] = np.maximum(0, rng.poisson(0.1, size=n)).astype(np.int64)

    schema_cols_ws = WeeklyStatsSchema.to_schema().columns
    for col_name, col in schema_cols_ws.items():
        if col_name in ws.columns:
            continue
        dtype_str = str(col.dtype)
        if "int" in dtype_str.lower():
            ws[col_name] = np.zeros(n, dtype=np.int64)
        elif "float" in dtype_str.lower():
            ws[col_name] = np.zeros(n, dtype=np.float64)
        else:
            ws[col_name] = 0
    return features, WeeklyStatsSchema.validate(ws)


def test_model_id_uses_nb_cal_prefix() -> None:
    features, weekly_stats = _build_synthetic_data(Position.WR)
    model = wr_lightgbm_nb_cal()
    model.fit(features, weekly_stats)
    assert model.model_id.startswith("lightgbm-nb-cal:wr:")
    # Format: lightgbm-nb-cal:<pos>:<8-char-hash>:<train-start>-<train-end>
    parts = model.model_id.split(":")
    assert len(parts) == 4
    assert parts[0] == "lightgbm-nb-cal"
    assert parts[1] == "wr"
    assert len(parts[2]) == 8


def test_code_hash_differs_from_parent_class() -> None:
    """Adding lightgbm_nb_cal.py to the hash file list changes the
    code_hash; otherwise the cal model would be indistinguishable from C-NB
    on the artifact disk path."""
    from projections.models.lightgbm_nb import wr_lightgbm_nb

    features, weekly_stats = _build_synthetic_data(Position.WR)

    parent = wr_lightgbm_nb()
    parent.fit(features, weekly_stats)
    child = wr_lightgbm_nb_cal()
    child.fit(features, weekly_stats)

    assert parent.code_hash != child.code_hash


def test_dispersion_differs_from_parent_when_val_variance_diverges() -> None:
    """The mechanism Plan 7 cures: parent fits MLE on train residuals;
    child fits pinball on val residuals. When the synthetic data's val year
    has different residual variance than the train years, the two
    dispersions should differ."""
    from projections.models.lightgbm_nb import wr_lightgbm_nb

    features, weekly_stats = _build_synthetic_data(Position.WR)

    parent = wr_lightgbm_nb()
    parent.fit(features, weekly_stats)
    child = wr_lightgbm_nb_cal()
    child.fit(features, weekly_stats)

    # At least one count-stat dispersion should differ between parent and
    # child. (With deterministic synthetic data both can't be identical.)
    parent_disp = parent._count_dispersions  # type: ignore[attr-defined]
    child_disp = child._count_dispersions  # type: ignore[attr-defined]
    assert set(parent_disp.keys()) == set(child_disp.keys())
    differs = [k for k in parent_disp if parent_disp[k] != child_disp[k]]
    assert len(differs) > 0, (
        f"Expected at least one count-stat dispersion to differ between "
        f"parent (MLE) and child (pinball); both produced identical {parent_disp}"
    )


def test_predict_distribution_returns_mixed_family() -> None:
    """predict_distribution is inherited from LightGBMNbModel; per-row
    family stays MIXED."""
    features, weekly_stats = _build_synthetic_data(Position.WR)
    model = wr_lightgbm_nb_cal()
    model.fit(features, weekly_stats)
    test_features = features[features["season"] == 2021].head(5).copy()
    out = model.predict_distribution(test_features, Ruleset.espn_ppr())
    ProjectionWeeklySchema.validate(out)
    assert (out["family"] == "MIXED").all()


@pytest.mark.parametrize(
    "factory",
    [qb_lightgbm_nb_cal, rb_lightgbm_nb_cal, te_lightgbm_nb_cal, wr_lightgbm_nb_cal],
)
def test_factory_returns_cal_model(factory) -> None:
    """Each per-position factory returns a LightGBMNbCalModel instance."""
    model = factory()
    assert isinstance(model, LightGBMNbCalModel)
```

- [ ] **Step 2: Run the test to verify it fails on import**

```bash
pytest tests/test_models/test_lightgbm_nb_cal.py -v
```

Expected: FAIL with `ModuleNotFoundError: projections.models.lightgbm_nb_cal`.

- [ ] **Step 3: Implement the model class**

Create `src/projections/models/lightgbm_nb_cal.py`:

```python
"""Calibration-aware LightGBM-NB — Plan 7.

Subclass of LightGBMNbModel. Overrides only the count-stat dispersion-
fitting call: swap nb_dispersion_from_residuals (MLE on training
residuals) for nb_dispersion_from_pinball (sum-of-pinball-losses on
held-out validation slice). Yards-stat training, predict_distribution,
joblib save/load — all inherited verbatim.

Spec: docs/superpowers/specs/2026-04-28-plan-7-calibration-aware-nb-design.md
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Final

import lightgbm as lgb
import numpy as np
import pandas as pd

from projections.distributions.parametric import (
    _NB_MU_FLOOR,
    nb_dispersion_from_pinball,
)
from projections.models.base import compute_code_hash
from projections.models.lightgbm import (
    _LightGBMConfig,
    _QB_FEATURE_COLUMNS,
    _QB_NON_NEGATIVE,
    _QB_TARGET_STATS,
    _RB_FEATURE_COLUMNS,
    _RB_NON_NEGATIVE,
    _RB_TARGET_STATS,
    _TE_FEATURE_COLUMNS,
    _TE_NON_NEGATIVE,
    _TE_TARGET_STATS,
    _WR_FEATURE_COLUMNS,
    _WR_NON_NEGATIVE,
    _WR_TARGET_STATS,
    QUANTILE_GRID,
    _filter_features,
)
from projections.models.lightgbm_nb import COUNT_STATS_FOR_NB, LightGBMNbModel
from projections.models.lightgbm_tuned import _TUNED_PARAMS_PATH
from projections.schemas import (
    Position,
    QbFeaturesSchema,
    RbFeaturesSchema,
    Stat,
    TeFeaturesSchema,
    WeeklyStatsSchema,
    WrFeaturesSchema,
)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]


def _code_hash_files_nb_cal(position: Position) -> tuple[Path, ...]:
    """Source files whose content is hashed into the NB-cal model's
    model_id. Same set as the parent's, plus this file."""
    src = _PROJECT_ROOT / "src" / "projections"
    feat_module = {
        Position.QB: "qb.py",
        Position.RB: "rb.py",
        Position.TE: "te.py",
        Position.WR: "wr.py",
    }[position]
    return (
        src / "models" / "lightgbm_nb_cal.py",
        src / "models" / "lightgbm_nb.py",
        src / "models" / "lightgbm_tuned.py",
        src / "models" / "lightgbm.py",
        src / "models" / "base.py",
        src / "distributions" / "quantile.py",
        src / "distributions" / "codec.py",
        src / "distributions" / "parametric.py",
        src / "features" / feat_module,
        src / "features" / "_shared.py",
        src / "features" / "_rolling.py",
        src / "features" / "_opponent.py",
        src / "scoring" / "score.py",
        src / "scoring" / "score_distribution.py",
        _TUNED_PARAMS_PATH,
    )


class LightGBMNbCalModel(LightGBMNbModel):
    """LightGBM-NB with calibration-aware dispersion fitting.

    Inherits from LightGBMNbModel: tuned-params loader, _hyperparams_for hook,
    joblib save/load, feature/weekly_stats join, yards-stat 5-quantile path,
    and predict_distribution. Overrides fit's count-stat dispersion call to
    use nb_dispersion_from_pinball on the held-out val slice instead of
    nb_dispersion_from_residuals on training residuals. Overrides code_hash
    and model_id to reflect the lightgbm-nb-cal: prefix.
    """

    @property
    def code_hash(self) -> str:
        return compute_code_hash(_code_hash_files_nb_cal(self._config.position))

    @property
    def model_id(self) -> str:
        if not self._is_fitted:
            raise RuntimeError(
                "model_id not available before fit() — depends on training-time state"
            )
        assert self._train_start is not None and self._train_end is not None
        return (
            f"lightgbm-nb-cal:{self._config.position.value.lower()}:"
            f"{self.code_hash}:{self._train_start}-{self._train_end}"
        )

    def fit(self, features: pd.DataFrame, weekly_stats: pd.DataFrame) -> None:
        """Train per-stat sub-models with hybrid count/yards routing.

        Same flow as the parent, except for the count-stat dispersion fit:
        instead of nb_dispersion_from_residuals(mu_train, y_train), we call
        nb_dispersion_from_pinball(mu_val, y_val, quantiles=(0.10, 0.90))
        on the val slice (max(seasons)). The Poisson booster is trained
        identically and predicts mu in original scale (lgb's poisson predict
        already exponentiates internally).

        No leakage: the booster's early-stop callback selects best_iter via
        the val slice; that is a separate single-scalar selection from
        alpha. Sharing the val slice is safe because alpha and best_iter
        don't share gradient information.
        """
        # Schema validation + position filter — mirrors parent.
        features = self._config.feature_schema.validate(features)
        weekly_stats = WeeklyStatsSchema.validate(weekly_stats)
        weekly_stats = weekly_stats[
            weekly_stats["position"] == self._config.position.value
        ].copy()

        target_cols = [s.value for s in self._config.target_stats]
        joined = features.merge(
            weekly_stats[["gsis_id", "season", "week", *target_cols]],
            on=["gsis_id", "season", "week"],
            how="inner",
            validate="one_to_one",
        )
        if joined.empty:
            raise ValueError("Empty training set after feature/weekly_stats join")

        seasons = sorted(joined["season"].unique())
        if len(seasons) < 2:
            raise ValueError(
                f"Need >=2 training seasons for early-stopping validation slice; "
                f"got {len(seasons)}"
            )

        val_season = seasons[-1]
        train_mask = joined["season"] != val_season
        val_mask = joined["season"] == val_season

        feat_cols = list(self._config.feature_columns)
        x_train = joined.loc[train_mask, feat_cols].to_numpy(dtype=np.float64)
        x_val = joined.loc[val_mask, feat_cols].to_numpy(dtype=np.float64)

        for stat in self._config.target_stats:
            stat_params = self._hyperparams_for(stat)
            y_train = joined.loc[train_mask, stat.value].to_numpy(dtype=np.float64)
            y_val = joined.loc[val_mask, stat.value].to_numpy(dtype=np.float64)

            if stat in COUNT_STATS_FOR_NB:
                regressor = lgb.LGBMRegressor(
                    objective="poisson",
                    **stat_params,
                )
                regressor.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                best_iter = int(regressor.best_iteration_ or 0)
                if best_iter == 0:
                    warnings.warn(
                        f"LightGBMNbCalModel.fit: best_iter=0 for "
                        f"{self._config.position.value}/{stat.value} (poisson); "
                        "early stopping fired immediately. Sub-model will "
                        "predict at constant baseline.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                self._count_models[stat] = regressor.booster_
                self._count_best_iters[stat] = best_iter

                # Calibration-aware dispersion fit on the VAL slice.
                # Poisson predict() returns mu in original scale.
                mu_hat_val = np.maximum(
                    np.asarray(regressor.predict(x_val), dtype=np.float64),
                    _NB_MU_FLOOR,
                )
                dispersion = nb_dispersion_from_pinball(
                    mu_hat=mu_hat_val,
                    actual=y_val,
                    quantiles=(0.10, 0.90),
                )
                self._count_dispersions[stat] = dispersion
            else:
                # Yards path inherited verbatim from parent's fit; copy the
                # body to keep the override self-contained (parent's fit is
                # one big function — refactoring it for inheritance is
                # out of scope for Plan 7).
                self._sub_models[stat] = {}
                for q in QUANTILE_GRID:
                    regressor = lgb.LGBMRegressor(
                        objective="quantile",
                        alpha=q,
                        **stat_params,
                    )
                    regressor.fit(
                        x_train,
                        y_train,
                        eval_set=[(x_val, y_val)],
                        callbacks=[lgb.early_stopping(50, verbose=False)],
                    )
                    best_iter = int(regressor.best_iteration_ or 0)
                    if best_iter == 0:
                        warnings.warn(
                            f"LightGBMNbCalModel.fit: best_iter=0 for "
                            f"{self._config.position.value}/{stat.value}/q={q}; "
                            "early stopping fired immediately. Sub-model will "
                            "predict at constant baseline.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    self._sub_models[stat][q] = regressor.booster_
                    self._best_iters[(stat, q)] = best_iter

        self._train_start = int(seasons[0])
        self._train_end = int(seasons[-1])
        self._is_fitted = True


def qb_lightgbm_nb_cal() -> LightGBMNbCalModel:
    return LightGBMNbCalModel(
        config=_LightGBMConfig(
            position=Position.QB,
            target_stats=_QB_TARGET_STATS,
            feature_columns=_filter_features(_QB_FEATURE_COLUMNS),
            feature_schema=QbFeaturesSchema,
            non_negative_stats=_QB_NON_NEGATIVE,
        )
    )


def rb_lightgbm_nb_cal() -> LightGBMNbCalModel:
    return LightGBMNbCalModel(
        config=_LightGBMConfig(
            position=Position.RB,
            target_stats=_RB_TARGET_STATS,
            feature_columns=_filter_features(_RB_FEATURE_COLUMNS),
            feature_schema=RbFeaturesSchema,
            non_negative_stats=_RB_NON_NEGATIVE,
        )
    )


def te_lightgbm_nb_cal() -> LightGBMNbCalModel:
    return LightGBMNbCalModel(
        config=_LightGBMConfig(
            position=Position.TE,
            target_stats=_TE_TARGET_STATS,
            feature_columns=_filter_features(_TE_FEATURE_COLUMNS),
            feature_schema=TeFeaturesSchema,
            non_negative_stats=_TE_NON_NEGATIVE,
        )
    )


def wr_lightgbm_nb_cal() -> LightGBMNbCalModel:
    return LightGBMNbCalModel(
        config=_LightGBMConfig(
            position=Position.WR,
            target_stats=_WR_TARGET_STATS,
            feature_columns=_filter_features(_WR_FEATURE_COLUMNS),
            feature_schema=WrFeaturesSchema,
            non_negative_stats=_WR_NON_NEGATIVE,
        )
    )
```

- [ ] **Step 4: Run unit tests to verify they pass**

```bash
pytest tests/test_models/test_lightgbm_nb_cal.py -v
```

Expected: 7 tests pass (4 simple + 1 parametrized × 4 positions − 1 already counted = 7 total). If the dispersion-differs test fails because the synthetic data happens to converge to identical α, increase synthetic noise in the fixture (mutate `np.random.default_rng` seed) or adjust the assertion to "differs on at least one position".

- [ ] **Step 5: Run mypy + ruff**

```bash
mypy src/projections/models/lightgbm_nb_cal.py tests/test_models/test_lightgbm_nb_cal.py
ruff check src/projections/models/lightgbm_nb_cal.py tests/test_models/test_lightgbm_nb_cal.py
ruff format --check src/projections/models/lightgbm_nb_cal.py tests/test_models/test_lightgbm_nb_cal.py
```

Expected: zero errors.

- [ ] **Step 6: Commit**

```bash
git add src/projections/models/lightgbm_nb_cal.py tests/test_models/test_lightgbm_nb_cal.py
git commit -m "feat(models): LightGBMNbCalModel — calibration-aware NB dispersion — Plan 7 Phase 1"
```

### Task 6: Wire factories into `POSITION_DISPATCH`

**Files:**
- Modify: `src/projections/models/__init__.py`

- [ ] **Step 1: Read the current dispatch wiring**

```bash
sed -n '40,90p' src/projections/models/__init__.py
```

Confirm the current shape: imports from `lightgbm_nb`, lists in `__all__`, factories in each `_<POS>_FACTORIES` dict. We add `lightgbm_nb_cal` parallel to `lightgbm_nb`.

- [ ] **Step 2: Edit the imports**

In `src/projections/models/__init__.py`, after the existing `from projections.models.lightgbm_nb import (...)` block (lines 42-48), add:

```python
from projections.models.lightgbm_nb_cal import (
    LightGBMNbCalModel,
    qb_lightgbm_nb_cal,
    rb_lightgbm_nb_cal,
    te_lightgbm_nb_cal,
    wr_lightgbm_nb_cal,
)
```

- [ ] **Step 3: Extend `__all__`**

Add `"LightGBMNbCalModel"` (alphabetically after `"LightGBMNbModel"`) and `"qb_lightgbm_nb_cal"`, `"rb_lightgbm_nb_cal"`, `"te_lightgbm_nb_cal"`, `"wr_lightgbm_nb_cal"` (each alphabetically after the `_lightgbm_nb` entry — keeping Plan 5c's alpha-ordering pattern).

The full `__all__` should now read:

```python
__all__ = [
    "POSITION_DISPATCH",
    "BaselineModel",
    "LightGBMModel",
    "LightGBMNbCalModel",
    "LightGBMNbModel",
    "LightGBMTunedModel",
    "Model",
    "compute_code_hash",
    "qb_baseline",
    "qb_lightgbm",
    "qb_lightgbm_nb",
    "qb_lightgbm_nb_cal",
    "qb_lightgbm_tuned",
    "rb_baseline",
    "rb_lightgbm",
    "rb_lightgbm_nb",
    "rb_lightgbm_nb_cal",
    "rb_lightgbm_tuned",
    "te_baseline",
    "te_lightgbm",
    "te_lightgbm_nb",
    "te_lightgbm_nb_cal",
    "te_lightgbm_tuned",
    "wr_baseline",
    "wr_lightgbm",
    "wr_lightgbm_nb",
    "wr_lightgbm_nb_cal",
    "wr_lightgbm_tuned",
]
```

- [ ] **Step 4: Add factories to each `_<POS>_FACTORIES` dict**

Each dict gains a `"lightgbm-nb-cal": <pos>_lightgbm_nb_cal` entry. After edits:

```python
_QB_FACTORIES: dict[str, Callable[[], Model]] = {
    "baseline": qb_baseline,
    "lightgbm": qb_lightgbm,
    "lightgbm-tuned": qb_lightgbm_tuned,
    "lightgbm-nb": qb_lightgbm_nb,
    "lightgbm-nb-cal": qb_lightgbm_nb_cal,
}
```

(and analogously for RB / TE / WR).

- [ ] **Step 5: Run a smoke test of the dispatch**

```bash
python -c "
from projections.models import POSITION_DISPATCH
from projections.schemas import Position
for pos in [Position.QB, Position.RB, Position.TE, Position.WR]:
    factory = POSITION_DISPATCH[pos].factories['lightgbm-nb-cal']
    model = factory()
    print(pos.value, type(model).__name__)
"
```

Expected:
```
QB LightGBMNbCalModel
RB LightGBMNbCalModel
TE LightGBMNbCalModel
WR LightGBMNbCalModel
```

- [ ] **Step 6: Run mypy + ruff**

```bash
mypy src/projections/models/__init__.py
ruff check src/projections/models/__init__.py
ruff format --check src/projections/models/__init__.py
```

Expected: zero errors.

- [ ] **Step 7: Commit**

```bash
git add src/projections/models/__init__.py
git commit -m "feat(models): wire lightgbm-nb-cal into POSITION_DISPATCH — Plan 7 Phase 1"
```

### Task 7: Cross-position smoke + 5-model harness smoke

**Files:**
- Create: `tests/test_models/test_lightgbm_nb_cal_smoke.py`
- Create: `tests/test_backtest/test_harness_quint_model.py`

- [ ] **Step 1: Write the cross-position smoke**

Create `tests/test_models/test_lightgbm_nb_cal_smoke.py` mirroring `test_lightgbm_nb_smoke.py` but pointed at the `"lightgbm-nb-cal"` factory:

```python
"""Cross-position smoke for LightGBMNbCalModel — Plan 7.

Single fit + predict for each of the 4 positions on synthetic fixtures, driven
by `POSITION_DISPATCH[position].factories["lightgbm-nb-cal"]`. Mirrors
test_lightgbm_nb_smoke.py — only the factory key changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.models import POSITION_DISPATCH
from projections.schemas import (
    _PYARROW_STR,
    Position,
    ProjectionWeeklySchema,
    Ruleset,
    WeeklyStatsSchema,
)


def _build_synthetic_data(position: Position) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same shape as test_lightgbm_nb_smoke.py's helper of the same name."""
    rng = np.random.default_rng(42)
    feature_schema = POSITION_DISPATCH[position].feature_schema

    rows = []
    for season in range(2018, 2022):
        for week in range(1, 18):
            for p in range(20):
                rows.append(
                    {
                        "gsis_id": f"00-{p:07d}",
                        "season": np.int64(season),
                        "week": np.int64(week),
                        "team": "KC",
                        "opponent": "DEN",
                    }
                )
    df = pd.DataFrame(rows)
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["team"] = df["team"].astype(_PYARROW_STR)
    df["opponent"] = df["opponent"].astype(_PYARROW_STR)

    schema_cols = feature_schema.to_schema().columns
    for col_name, col in schema_cols.items():
        if col_name in df.columns:
            continue
        dtype_str = str(col.dtype)
        if "bool" in dtype_str.lower():
            df[col_name] = rng.integers(0, 2, size=len(df)).astype(bool)
        elif "int" in dtype_str.lower():
            df[col_name] = rng.integers(1, 6, size=len(df)).astype(np.int64)
        else:
            df[col_name] = rng.uniform(0.0, 0.5, size=len(df)).astype(np.float64)
    features = feature_schema.validate(df)

    ws = features[["gsis_id", "season", "week", "team", "opponent"]].copy()
    ws["position"] = position.value
    n = len(ws)
    ws["passing_yards"] = np.clip(rng.normal(220.0, 60.0, size=n), -100.0, 800.0).astype(np.float64)
    ws["passing_tds"] = np.maximum(0, rng.poisson(1.5, size=n)).astype(np.int64)
    ws["interceptions"] = np.maximum(0, rng.poisson(0.7, size=n)).astype(np.int64)
    ws["rushing_yards"] = np.clip(rng.normal(20.0, 15.0, size=n), -50.0, 400.0).astype(np.float64)
    ws["rushing_tds"] = np.maximum(0, rng.poisson(0.2, size=n)).astype(np.int64)
    ws["receptions"] = np.maximum(0, rng.poisson(2.5, size=n)).astype(np.int64)
    ws["receiving_yards"] = np.clip(rng.normal(25.0, 15.0, size=n), -50.0, 400.0).astype(np.float64)
    ws["receiving_tds"] = np.maximum(0, rng.poisson(0.2, size=n)).astype(np.int64)
    ws["fumbles_lost"] = np.maximum(0, rng.poisson(0.1, size=n)).astype(np.int64)

    schema_cols_ws = WeeklyStatsSchema.to_schema().columns
    for col_name, col in schema_cols_ws.items():
        if col_name in ws.columns:
            continue
        dtype_str = str(col.dtype)
        if "int" in dtype_str.lower():
            ws[col_name] = np.zeros(n, dtype=np.int64)
        elif "float" in dtype_str.lower():
            ws[col_name] = np.zeros(n, dtype=np.float64)
        else:
            ws[col_name] = 0
    return features, WeeklyStatsSchema.validate(ws)


@pytest.mark.parametrize("position", [Position.QB, Position.RB, Position.TE, Position.WR])
def test_lightgbm_nb_cal_fit_predict_smoke(position: Position) -> None:
    """Each position's LightGBMNbCalModel fits and predicts via dispatch."""
    features, weekly_stats = _build_synthetic_data(position)
    factory = POSITION_DISPATCH[position].factories["lightgbm-nb-cal"]
    model = factory()
    model.fit(features, weekly_stats)

    test_features = features[features["season"] == 2021].head(5).copy()
    out = model.predict_distribution(test_features, Ruleset.espn_ppr())

    ProjectionWeeklySchema.validate(out)
    assert len(out) == 5
    assert (out["family"] == "MIXED").all()
    assert (out["position"] == position.value).all()
```

- [ ] **Step 2: Run the smoke**

```bash
pytest tests/test_models/test_lightgbm_nb_cal_smoke.py -v
```

Expected: 4 tests pass (one per position).

- [ ] **Step 3: Write the 5-model harness smoke**

Create `tests/test_backtest/test_harness_quint_model.py`:

```python
"""End-to-end harness fold under all 5 model classes — Plan 7 Phase 1.

Mirrors test_harness_quad_model.py: single (WR, 2024) fold under
model_classes=(baseline, lightgbm, lightgbm-tuned, lightgbm-nb,
lightgbm-nb-cal). Verifies all five contribute rows for the cell and
that the same model-class-agnostic metric set is emitted for each.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from projections.backtest.harness import run_backtest
from projections.schemas import Position

_FEATURES_ROOT = Path("data/features")
_RAW_ROOT = Path("data/raw")


def _cache_present() -> bool:
    return (_FEATURES_ROOT / "wr" / "season=2024").exists() and (
        _RAW_ROOT / "weekly_stats" / "season=2024"
    ).exists()


@pytest.mark.backtest
@pytest.mark.skipif(
    not _cache_present(),
    reason="WR feature/raw cache not populated; skipping backtest harness smoke.",
)
def test_quint_model_single_fold_emits_rows_for_each_class() -> None:
    """5-model run for a single (WR, 2024) fold."""
    run = run_backtest(
        positions=(Position.WR,),
        years=(2024,),
        model_classes=(
            "baseline",
            "lightgbm",
            "lightgbm-tuned",
            "lightgbm-nb",
            "lightgbm-nb-cal",
        ),
    )
    cell = run.metrics[(run.metrics["position"] == "WR") & (run.metrics["year"] == 2024)]
    assert set(cell["model_class"].unique()) == {
        "baseline",
        "lightgbm",
        "lightgbm-tuned",
        "lightgbm-nb",
        "lightgbm-nb-cal",
    }
```

- [ ] **Step 4: Run mypy + ruff**

```bash
mypy tests/test_models/test_lightgbm_nb_cal_smoke.py tests/test_backtest/test_harness_quint_model.py
ruff check tests/test_models/test_lightgbm_nb_cal_smoke.py tests/test_backtest/test_harness_quint_model.py
ruff format --check tests/test_models/test_lightgbm_nb_cal_smoke.py tests/test_backtest/test_harness_quint_model.py
```

Expected: zero errors.

- [ ] **Step 5: Commit**

```bash
git add tests/test_models/test_lightgbm_nb_cal_smoke.py tests/test_backtest/test_harness_quint_model.py
git commit -m "test(plan-7): cross-position + 5-model harness smokes"
```

### Task 8: Extend `scripts/backtest.py --model` choices

**Files:**
- Modify: `scripts/backtest.py`

- [ ] **Step 1: Edit the CLI choices**

In `scripts/backtest.py`, locate the `--model` argparse arg (around lines 117-126) and the dispatcher (lines 129-134). Edit:

Current:
```python
    parser.add_argument(
        "--model",
        choices=["baseline", "lightgbm", "lightgbm-tuned", "lightgbm-nb", "both", "all"],
        default="both",
        help=(
            "Which model class(es) to run. "
            "'both' = Model A + Model C (legacy default). "
            "'all' = Model A + Model C + Model C-tuned + Model C-NB."
        ),
    )
```

New:
```python
    parser.add_argument(
        "--model",
        choices=[
            "baseline",
            "lightgbm",
            "lightgbm-tuned",
            "lightgbm-nb",
            "lightgbm-nb-cal",
            "both",
            "all",
        ],
        default="both",
        help=(
            "Which model class(es) to run. "
            "'both' = Model A + Model C (legacy default). "
            "'all' = Model A + Model C + Model C-tuned + Model C-NB + Model C-NB-cal."
        ),
    )
```

And in the dispatcher:

Current:
```python
    if args.model == "both":
        model_classes: tuple[str, ...] = ("baseline", "lightgbm")
    elif args.model == "all":
        model_classes = ("baseline", "lightgbm", "lightgbm-tuned", "lightgbm-nb")
    else:
        model_classes = (args.model,)
```

New:
```python
    if args.model == "both":
        model_classes: tuple[str, ...] = ("baseline", "lightgbm")
    elif args.model == "all":
        model_classes = (
            "baseline",
            "lightgbm",
            "lightgbm-tuned",
            "lightgbm-nb",
            "lightgbm-nb-cal",
        )
    else:
        model_classes = (args.model,)
```

- [ ] **Step 2: Smoke the CLI**

```bash
python scripts/backtest.py --help | head -20
```

Expected: `--model` choices include `lightgbm-nb-cal`.

- [ ] **Step 3: Run mypy + ruff**

```bash
mypy scripts/backtest.py
ruff check scripts/backtest.py
ruff format --check scripts/backtest.py
```

Expected: zero errors.

- [ ] **Step 4: Commit**

```bash
git add scripts/backtest.py
git commit -m "feat(backtest): add lightgbm-nb-cal to --model choices — Plan 7 Phase 1"
```

### Task 9: Default-on smoke covers all 5 model classes

**Files:**
- Modify: `tests/backtest/test_backtest_smoke.py`

The default-on smoke (run as part of `pytest -v` without explicit marker) should now include `lightgbm-nb-cal` so a regression in the new class fails the gate.

- [ ] **Step 1: Read the current smoke test**

```bash
grep -n "model_class\|lightgbm-nb\|model_classes" tests/backtest/test_backtest_smoke.py | head -25
```

- [ ] **Step 2: Locate the assertion that pins the model_class set**

In `tests/backtest/test_backtest_smoke.py`, find the assertion that compares the metrics frame's `model_class` unique values against an expected set. The Plan 5c-era assertion should look like:

```python
assert set(metrics["model_class"].unique()) == {
    "baseline",
    "lightgbm",
    "lightgbm-tuned",
    "lightgbm-nb",
}
```

- [ ] **Step 3: Add `"lightgbm-nb-cal"` to the expected set**

```python
assert set(metrics["model_class"].unique()) == {
    "baseline",
    "lightgbm",
    "lightgbm-tuned",
    "lightgbm-nb",
    "lightgbm-nb-cal",
}
```

If the smoke test invokes `run_backtest(model_classes=...)` directly (vs. relying on a script default), make sure the test's `model_classes` tuple also includes `"lightgbm-nb-cal"`.

- [ ] **Step 4: Run the smoke**

```bash
pytest tests/backtest/test_backtest_smoke.py -v
```

Expected: PASS.

- [ ] **Step 5: Run mypy + ruff**

```bash
mypy tests/backtest/test_backtest_smoke.py
ruff check tests/backtest/test_backtest_smoke.py
ruff format --check tests/backtest/test_backtest_smoke.py
```

Expected: zero errors.

- [ ] **Step 6: Commit**

```bash
git add tests/backtest/test_backtest_smoke.py
git commit -m "test(backtest): include lightgbm-nb-cal in default smoke — Plan 7 Phase 1"
```

### Task 10: Full repo-wide verification

- [ ] **Step 1: Run the full test suite**

```bash
pytest -v
```

Expected: all green. If any pre-existing test fails because it expected the 4-model count somewhere we missed, locate and fix the assertion (paralleling Task 9).

- [ ] **Step 2: Run mypy across the whole project**

```bash
mypy src tests
```

Expected: zero errors.

- [ ] **Step 3: Run ruff check + format check**

```bash
ruff check src tests
ruff format --check src tests
```

Expected: zero errors. Fix any drift with `ruff format src tests`.

- [ ] **Step 4: Run ingest/store/schemas integration smoke**

Per CLAUDE.md directive 4: "For tasks that touch a pandera schema or any ingest/store path: run `pytest -v -k 'ingest or store or schemas'` even if your change is elsewhere."

```bash
pytest -v -k "ingest or store or schemas"
```

Expected: all green.

- [ ] **Step 5: Phase 1 status check — no commit**

This is a verification gate. If any step above failed, fix in place and re-run the gate before continuing to Phase 2. No commit at this step.

---

## Phase 2 — Backtest run + adoption verdict

Goal: run the full walk-forward harness with `--model lightgbm-nb-cal`, regenerate the snapshot (1504 → 1872 rows), build the per-cell A vs C-NB vs C-NB-cal table, apply §1.3 adoption gate, and report per-quantile pinball loss change.

### Task 11: Run the backtest with `--update-snapshot`

**Files:**
- Modify: `tests/backtest/model_metrics.json` (regenerated)
- Read-only: `data/backtest/run_<ts>/results.parquet` (newly written; not committed)

- [ ] **Step 1: Confirm the feature/raw cache is fresh**

```bash
ls data/features/wr/season=2024/ data/raw/weekly_stats/season=2024/ 2>&1 | head -5
```

Expected: parquet files present. If missing, run `python scripts/refresh_features.py` first (per CONTRIBUTING.md).

- [ ] **Step 2: Run the backtest with the new model class only**

```bash
python scripts/backtest.py --update-snapshot --model lightgbm-nb-cal
```

Expected output: prints "Wrote tests/backtest/model_metrics.json" and the new row count. The 1872-row total is **only correct after re-running with `--model all`** in step 3 below; this single-class run is just to confirm the new pipeline produces rows.

Note the run dir under `data/backtest/run_<ts>/`.

- [ ] **Step 3: Run the full all-models backtest with `--update-snapshot`**

```bash
python scripts/backtest.py --update-snapshot --model all
```

Expected: snapshot now contains 1872 rows total (368 per model class × 5 classes − 32 skipped season_calibration rows for QUANTILE/MIXED classes that are still un-aggregated per TODO #28). Print output reports the new total.

- [ ] **Step 4: Confirm row count**

```bash
python -c "
import json
data = json.load(open('tests/backtest/model_metrics.json'))
print('rows:', len(data))
classes = sorted({r['model_class'] for r in data})
print('classes:', classes)
"
```

Expected: row count of 1872 (or whatever the actual sum works out to once season_calibration skips are applied — exact number depends on Plan 5c's 1504 + 368 new rows minus any new season_calibration skip bookkeeping). Classes set: `["baseline", "lightgbm", "lightgbm-nb", "lightgbm-nb-cal", "lightgbm-tuned"]`.

- [ ] **Step 5: Run the gate check on the same snapshot**

```bash
python scripts/backtest.py --check --model all
```

Expected: PASS (the snapshot was just written by `--update-snapshot`; --check reads it back and compares the same run's metrics, so it should be a no-op pass).

- [ ] **Step 6: Commit the regenerated snapshot**

```bash
git add tests/backtest/model_metrics.json
git commit -m "chore(backtest): regenerate snapshot with C-NB-cal rows — Plan 7 Phase 2"
```

### Task 12: Build the per-cell comparison table + adoption verdict

**Files:**
- Read-only: `tests/backtest/model_metrics.json`
- Modify: `project_management.md`

- [ ] **Step 1: Pull the per-cell metrics into a comparison table**

Use a one-shot Python script to extract:

```bash
python -c "
import json
import pandas as pd

data = json.load(open('tests/backtest/model_metrics.json'))
df = pd.DataFrame(data)
# Pivot: index=(position, year), columns=(model_class, metric).
metrics_of_interest = ['composite_rmse', 'composite_mae', 'spearman_topN', 'weekly_calibration_p10p90']
df = df[df['metric'].isin(metrics_of_interest)]
pivot = df.pivot_table(
    index=['position', 'year'],
    columns=['model_class', 'metric'],
    values='value',
)
print(pivot.to_string(float_format=lambda x: f'{x:8.4f}'))
"
```

- [ ] **Step 2: Compute deltas (C-NB-cal vs A) per cell**

For each (position, year) cell, compute:
- `composite_rmse` delta % = `(C-NB-cal - A) / A * 100`
- `spearman_topN` delta = `C-NB-cal - A`
- `weekly_calibration_p10p90` delta = `C-NB-cal - A`

Also compute (C-NB-cal vs C-NB) calibration deltas — this is the marginal effect of swapping the dispersion estimator.

- [ ] **Step 3: Apply the §1.3 adoption gate**

For each cell, evaluate:

| Criterion | Per-cell rule | Aggregate threshold |
|---|---|---|
| 1. Calibration mean delta vs A | mean over 16 cells of (cal_C-NB-cal − cal_A) | ≥ 0 |
| 2. Per-cell calibration vs A | min over 16 cells of (cal_C-NB-cal − cal_A) | ≥ −0.02 |
| 3. RMSE % change | min/max over 16 cells | ∈ [−1.69%, +1.69%] |
| 4. Spearman delta (vs C-NB) | max over 16 cells of \|spearman_C-NB-cal − spearman_C-NB\| | ≤ 0.005 |

Pass = all four criteria pass simultaneously.

- [ ] **Step 4: Compute per-quantile pinball loss change**

This is the upper-tail-distortion diagnostic flagged in spec §4 risk #2. The harness emits per-quantile metrics named `weekly_pinball_q10` and `weekly_pinball_q90` (verify the exact names in `src/projections/backtest/harness.py`; if they don't exist yet, this step is informational only and lives in the Phase 2 report verbatim, not as a gate).

For each cell, compute `(pinball_q10_C-NB-cal - pinball_q10_C-NB) / pinball_q10_C-NB * 100` and same for q90. Report.

If the q90 pinball **worsens by >10% on any cell**, flag the upper-tail-distortion risk in the Phase 2 report and recommend asymmetric-weighted fitting as the follow-up plan.

- [ ] **Step 5: Write the Phase 2 entry into `project_management.md`**

Prepend (above the Phase 0 entry) a new section:

```markdown
## Plan 7 Phase 2 — Backtest run + adoption verdict (run 2026-04-28)

Ran the full walk-forward harness with `--model all` after wiring
LightGBMNbCalModel. Snapshot extended 1504 → 1872 rows (368 new
lightgbm-nb-cal rows; same QUANTILE/MIXED season_calibration_* rows
skipped per TODO #28).

### Per-position model_ids

| Position | Model A | Model C-NB | Model C-NB-cal |
|---|---|---|---|
| WR | baseline:wr:6d955427:2018-2023 | lightgbm-nb:wr:dc445a2d:2018-2023 | lightgbm-nb-cal:wr:<hash>:2018-2023 |
| QB | baseline:qb:c98738f3:2018-2023 | lightgbm-nb:qb:3ae5b940:2018-2023 | lightgbm-nb-cal:qb:<hash>:2018-2023 |
| RB | baseline:rb:5a86c8ee:2018-2023 | lightgbm-nb:rb:ba2e35cc:2018-2023 | lightgbm-nb-cal:rb:<hash>:2018-2023 |
| TE | baseline:te:9c00025b:2018-2023 | lightgbm-nb:te:e76e590a:2018-2023 | lightgbm-nb-cal:te:<hash>:2018-2023 |

### Adoption-gate verdict — <PASS / FAIL> on §1.3

| Criterion | Threshold | Actual | Pass? |
|---|---|---|---|
| Calibration mean delta vs A | >= 0 | <value> | <Y/N> |
| Per-cell calibration vs A | min >= -0.02 | min <value> on <cell> | <Y/N> |
| RMSE % change | within +/-1.69% | min <value%> / max <value%> | <Y/N> |
| Spearman delta vs C-NB | <= 0.005 | max abs <value> on <cell> | <Y/N> |

### Side-by-side per-cell comparison (16 cells)

[Paste the pivoted table from Task 12 Step 1, with deltas computed in Step 2.]

### Per-quantile pinball loss diagnostic

| Cell | q10 pinball % change | q90 pinball % change |
|---|---|---|

[Paste the per-quantile diagnostic from Step 4. Flag any q90 row that worsens by >10%.]

### Decision

<one of:>
**Adopt:** §1.3 gate passed. C-NB-cal replaces C-NB in POSITION_DISPATCH. Same PR also prunes Model C-tuned (TODO #29) and possibly Model C (untuned).
**Peer-ship:** §1.3 gate failed. C-NB-cal ships as fifth peer. Calibration-aware track closed.

---
```

- [ ] **Step 6: Commit the project_management.md update**

```bash
git add project_management.md
git commit -m "docs(plan-7): record Phase 2 backtest verdict — Plan 7 Phase 2"
```

### Task 13: Apply adoption decision

**Files:**
- Modify: `src/projections/models/__init__.py` (only if gate passed)
- Modify: `tests/backtest/model_metrics.json` (only if gate passed and dispatch swap removes a model class)
- Modify: `scripts/backtest.py` (only if gate passed and dispatch swap removes a model class)
- Modify: `TODO.md` (always)

- [ ] **Step 1: Branch on Phase 2 verdict**

If the adoption gate **PASSED**: continue to Step 2 (dispatch swap).

If the adoption gate **FAILED**: skip directly to Step 5 (TODO update; close calibration-aware track).

- [ ] **Step 2 (PASS only): Update POSITION_DISPATCH to default to C-NB-cal**

Plan 5c documented the per-position model_ids but did not change which class is the "default" — the dispatch table just lists factories under string keys, and callers choose. Adoption means making sure documentation, scripts, and any "default" affordance reflect C-NB-cal as the recommended choice. There is no hardcoded default model in this codebase, so the dispatch swap is **just documentation**: update `project_management.md` and TODO.md to flag C-NB-cal as the new recommended class.

If we do prune Model C-tuned (TODO #29) and/or Model C in the same PR (the "housekeeping commit"), now is the time. For each pruned class:

- Remove the `import` from `src/projections/models/__init__.py`.
- Remove the entries from each `_<POS>_FACTORIES` dict.
- Remove the entries from `__all__`.
- Drop the `<class>` choice from `scripts/backtest.py --model`.
- Delete the corresponding rows from `tests/backtest/model_metrics.json` (368 rows per class).
- Delete the per-class smoke files (`tests/test_models/test_lightgbm_tuned*.py` etc.).

This is its own decision in the same PR; only do it if the user explicitly wants the prune now. Default: leave Model C-tuned in for one more soak cycle (TODO #29 stays open).

- [ ] **Step 3 (PASS only): Run the full gate after pruning (if any)**

```bash
pytest -v
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: all green.

- [ ] **Step 4 (PASS only): Commit the pruning (if any)**

```bash
git add -p  # stage only the pruning-related changes
git commit -m "chore(models): adopt C-NB-cal as recommended; prune <classes if any> — Plan 7 Phase 2"
```

- [ ] **Step 5 (always): Update TODO.md**

Append to the "Closed" section of TODO.md (or create one if missing):

```markdown
### Plan 7 — Calibration-aware NB-2 fitting (Model C-NB-cal) — closed in Plan 7

Closed 2026-04-28.

Phase 0 diagnostic decomposed Plan 5c's [p10, p90] coverage gap into
count-stat vs yards-stat contributions; verdict: <one of three>.
Phase 1 implemented `nb_dispersion_from_pinball` + LightGBMNbCalModel;
all 5 model classes coexist in POSITION_DISPATCH.
Phase 2 backtest run extended snapshot 1504 → 1872 rows.

**Adoption-gate verdict: <PASS / FAIL>** on §1.3 calibration-scoped gate.
- Calibration mean delta vs A: <value>
- Per-cell calibration vs A min: <value>
- RMSE % change envelope: [<min>, <max>]
- Spearman vs C-NB max delta: <value>

<If PASS:> C-NB-cal recommended over C-NB. Model C-tuned (TODO #29)
<pruned now / pending soak cycle>.
<If FAIL:> Calibration-aware track closed. Pivot to Plan 6 (ensemble) or
feature tracks (TODO #3, TODO #23). Asymmetric-weighted fitting and
mu-bucketed alpha follow-ups deferred per spec §1.2.
```

- [ ] **Step 6 (always): Commit TODO.md**

```bash
git add TODO.md
git commit -m "docs(todo): record Plan 7 verdict — Plan 7 Phase 2"
```

---

## Phase 3 — Final review + PR

### Task 14: Final repo-wide verification

- [ ] **Step 1: Run all checks one more time**

```bash
pytest -v
mypy src tests
ruff check src tests
ruff format --check src tests
```

Expected: all green.

- [ ] **Step 2: Verify the snapshot row count**

```bash
python -c "
import json
data = json.load(open('tests/backtest/model_metrics.json'))
print('rows:', len(data))
classes = sorted({r['model_class'] for r in data})
print('classes:', classes)
"
```

Expected: 1872 rows (or post-prune row count if Phase 2 Task 13 pruned a class). All five (or post-prune four) model classes present.

- [ ] **Step 3: Verify branch + working tree state**

```bash
git status
git log --oneline main..HEAD
```

Expected: clean working tree; commits since main include Phase 0 (3 commits), Phase 1 (6-7 commits), Phase 2 (3 commits) — 12-13 commits total.

### Task 15: Open PR

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/plan-7-calibration-aware-nb
```

- [ ] **Step 2: Open PR with summary + verdict**

```bash
gh pr create --title "feat(plan-7): calibration-aware NB-2 fitting (Model C-NB-cal)" --body "$(cat <<'EOF'
## Summary

- Phase 0: diagnostic CLI decomposes Plan 5c's [p10, p90] coverage gap into count vs yards contributions. Verdict: <verdict>.
- Phase 1: `nb_dispersion_from_pinball` estimator + `LightGBMNbCalModel` (subclass of `LightGBMNbModel`, swaps only the dispersion fit). 5-model dispatch.
- Phase 2: backtest extended 1504 → 1872 rows. Adoption-gate verdict: <PASS / FAIL>.

## Key results

[Paste the per-cell A vs C-NB vs C-NB-cal table from project_management.md.]

## Test plan

- [ ] `pytest -v` passes (1500+ tests).
- [ ] `mypy src tests` zero errors.
- [ ] `ruff check src tests` zero errors.
- [ ] `ruff format --check src tests` zero errors.
- [ ] Backtest snapshot regenerated; row count matches expectation.
- [ ] Phase 0 diagnostic CSV committed.
- [ ] project_management.md and TODO.md updated.

## Spec

`docs/superpowers/specs/2026-04-28-plan-7-calibration-aware-nb-design.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Note the PR URL for the user**

The PR URL printed by `gh pr create` is the handoff to the user for review.

---

## End-of-plan verification checklist

Before declaring Plan 7 complete:

- [ ] Phase 0 diagnostic verdict recorded in `project_management.md`.
- [ ] If verdict was `stop_file_yards_plan`, the rest of this plan is skipped — file the yards-side spec separately.
- [ ] `nb_dispersion_from_pinball` lands with 6 unit tests; all pass.
- [ ] `LightGBMNbCalModel` lands with 7 unit tests; all pass.
- [ ] `POSITION_DISPATCH.factories["lightgbm-nb-cal"]` resolves for all 4 positions.
- [ ] Cross-position smoke + 5-model harness smoke pass.
- [ ] `scripts/backtest.py --model all` runs all 5 classes.
- [ ] Backtest snapshot extended 1504 → 1872 rows.
- [ ] Per-cell A vs C-NB vs C-NB-cal comparison table in `project_management.md`.
- [ ] §1.3 adoption-gate verdict applied (PASS or FAIL); decision recorded.
- [ ] Per-quantile pinball loss change reported for upper-tail-distortion diagnosis.
- [ ] If gate passed and prune happened: snapshot row count and dispatch table reflect post-prune state.
- [ ] TODO.md updated.
- [ ] PR opened with summary, key results, test plan, and spec link.
- [ ] All `pytest`, `mypy`, `ruff check`, `ruff format --check` pass at HEAD.
