# Per-player weekly performance variance model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-06-14-performance-variance-model-design.md`

**Goal:** A fitted two-component weekly fantasy-point variance model (lognormal season-mean multiplier + position-affine Gamma weekly noise), wired into (A) the risk-aware season-value MC and (B) a predictive league-sim mode for honest post-draft CIs.

**Architecture:** Offline fitter → committed JSON params → a pure vectorized sampler module. Consumer A swaps the season-value MC's deterministic `per_game` for sampled weekly points (optimal-by-sampled fill, CRN-preserved). Consumer B wraps `simulate_league` with a model-sampled `actual_lookup` to produce forward outcome distributions. Phased: core → A → B.

**Tech Stack:** Python 3.12, numpy (vectorized Gamma/lognormal sampling), pandas, pandera, pytest, mypy strict, ruff.

**Run gates on the dev Windows box in PowerShell** with `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` (numpy native-crash mitigation; retry on exit `-1073741819`). See `MEMORY.md`.

**Key facts (do not re-derive):**
- `season_value.py`: `_GAMES = 17`; `_vectorized_lineup_points(avail, meta)` fills the optimal legal lineup over `(n_sims, n)` availability using a *fixed* `pts` (= `season_mean_fpts`); `expected_season_points` / `expected_season_points_crn` / `marginal_season_values` call it via `_factorized_season_value` (which reuses one "clean week" because `per_game` is uniform — **this factorization is invalid once weeks differ via sampling**). `POSITION_SLOTS` (from `roster_eligibility`) and `_FLEX_SLOTS` (from `roster_score`, narrowest-first) define slot fill order.
- `marginal_season_values` shares one `draws` matrix across base + every candidate (CRN) so marginals cancel common noise — preserve this.
- `build_weekly_actuals(weekly_stats, ruleset)` scores a `weekly_stats` frame to per-(gsis,week) half-PPR points (weeks 1–17), used by the fitter and consumer B.
- `simulate_league(seed, *, seat_strategies, strategy_labels, pool, config, proj_lookup, actual_lookup, calendar, jitter)` returns `LeagueOutcome` (per-seat `actual` + `projected` result lists). Lineups set by projection, scored by `actual_lookup` via `weekly_lineup_points(..., score_by="actual")`.
- `load_inputs(*, season, config, data_root)` (`backtest/inputs.py`) builds the pool via `build_draft_basis` and reads `weekly_stats`. It is the home for `is_rookie` (R6).
- STAT positions are `Position.QB/RB/WR/TE`; reference enums, not bare strings.

---

## File Structure

- Create `src/projections/draft/assistant/performance_variance.py` — params dataclass + loader + vectorized sampler. (Phase 1)
- Create `configs/performance_variance_params.json` — fitted params, committed. (Phase 1)
- Create `scripts/fit_performance_variance.py` — offline fitter. (Phase 1)
- Modify `src/projections/draft/backtest/inputs.py` — attach `is_rookie` to the pool. (Phase 1)
- Modify `src/projections/draft/assistant/season_value.py` — sampled-points lineup fill + sampler-backed MC. (Phase 2)
- Create `src/projections/draft/backtest/predictive.py` — model-sampled `actual_lookup` + forward-outcome simulation. (Phase 3)
- Modify `scripts/post_draft_assessment.py` — predictive-CI readout. (Phase 3)
- Tests under `tests/test_draft/test_assistant/` and `tests/test_draft/test_backtest/`.

---

## PHASE 1 — Model + fitter + params + is_rookie

### Task 1: Params dataclass + loader

**Files:**
- Create: `src/projections/draft/assistant/performance_variance.py`
- Test: `tests/test_draft/test_assistant/test_performance_variance.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_draft/test_assistant/test_performance_variance.py
import json
from pathlib import Path

from projections.draft.assistant.performance_variance import VarianceParams


def test_params_round_trip_and_lookup(tmp_path: Path) -> None:
    blob = {
        "weekly_std_affine": {"QB": {"a": 0.20, "b": 5.0}, "WR": {"a": 0.30, "b": 3.0}},
        "mean_mult_log_sd": {"QB|veteran": 0.30, "QB|rookie": 0.45, "default|veteran": 0.39, "default|rookie": 0.58},
    }
    p = tmp_path / "params.json"
    p.write_text(json.dumps(blob))
    vp = VarianceParams.load(p)
    assert vp.weekly_std("QB", 10.0) == 0.20 * 10.0 + 5.0
    assert vp.log_sd("QB", is_rookie=True) == 0.45
    # unknown position falls back to the 'default' cell
    assert vp.log_sd("RB", is_rookie=False) == 0.39
    assert vp.weekly_std("RB", 10.0) > 0  # default affine used for unknown position
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_draft/test_assistant/test_performance_variance.py -k params_round_trip -v`
Expected: FAIL (module/`VarianceParams` not defined).

- [ ] **Step 3: Implement**

```python
# src/projections/draft/assistant/performance_variance.py
"""Two-component weekly fantasy-point variance model (spec 2026-06-14).

Component ii: mean-preserving lognormal season-mean multiplier m (E[m]=1, log-SD by
position x rookie). Component i: per-game Gamma weekly noise, std = a_pos*pg + b_pos.
Params are fit offline (scripts/fit_performance_variance.py) and committed to
configs/performance_variance_params.json; this module only loads + samples.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DEFAULT_PARAMS_PATH = Path("configs/performance_variance_params.json")


@dataclass(frozen=True)
class VarianceParams:
    weekly_std_affine: dict[str, dict[str, float]]  # position -> {"a", "b"}; "default" key required
    mean_mult_log_sd: dict[str, float]  # "<pos>|<veteran|rookie>" -> log-SD; "default|..." required

    @classmethod
    def load(cls, path: Path = _DEFAULT_PARAMS_PATH) -> VarianceParams:
        blob = json.loads(Path(path).read_text())
        return cls(blob["weekly_std_affine"], blob["mean_mult_log_sd"])

    def weekly_std(self, position: str, per_game_mean: float) -> float:
        coef = self.weekly_std_affine.get(position) or self.weekly_std_affine["default"]
        return coef["a"] * per_game_mean + coef["b"]

    def log_sd(self, position: str, *, is_rookie: bool) -> float:
        tier = "rookie" if is_rookie else "veteran"
        return self.mean_mult_log_sd.get(f"{position}|{tier}", self.mean_mult_log_sd[f"default|{tier}"])
```

Note: the test writes its own params blob, so it does not need a `default` affine for QB/WR; add `"default": {"a":0.25,"b":4.4}` in that test's blob if the unknown-position assertion needs it — include it:

```python
    blob["weekly_std_affine"]["default"] = {"a": 0.25, "b": 4.4}
```
(Add that line before writing the file.)

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_draft/test_assistant/test_performance_variance.py -k params_round_trip -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/performance_variance.py tests/test_draft/test_assistant/test_performance_variance.py
git commit -m "feat(variance): VarianceParams dataclass + JSON loader"
```

### Task 2: Vectorized sampler

**Files:**
- Modify: `src/projections/draft/assistant/performance_variance.py`
- Test: `tests/test_draft/test_assistant/test_performance_variance.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from projections.draft.assistant.performance_variance import VarianceParams, sample_weekly_points


def _params() -> VarianceParams:
    return VarianceParams(
        weekly_std_affine={"default": {"a": 0.25, "b": 4.4}, "WR": {"a": 0.30, "b": 3.0}},
        mean_mult_log_sd={"default|veteran": 0.39, "default|rookie": 0.58, "WR|veteran": 0.35, "WR|rookie": 0.55},
    )


def test_sampler_shape_nonneg_deterministic() -> None:
    vp = _params()
    pos = np.array(["WR", "RB"])
    means = np.array([200.0, 150.0])
    rook = np.array([False, True])
    a = sample_weekly_points(vp, pos, means, rook, n_sims=50, n_weeks=14, rng=np.random.default_rng(0))
    b = sample_weekly_points(vp, pos, means, rook, n_sims=50, n_weeks=14, rng=np.random.default_rng(0))
    assert a.shape == (50, 14, 2)
    assert (a >= 0).all()
    assert np.array_equal(a, b)  # seeded determinism


def test_sampler_recovers_mean_and_zero_floor() -> None:
    vp = _params()
    pos = np.array(["WR", "QB"])
    means = np.array([170.0, 0.0])  # second player has no projection
    rook = np.array([False, False])
    s = sample_weekly_points(vp, pos, means, rook, n_sims=4000, n_weeks=17, rng=np.random.default_rng(1))
    # mean-preserving: per-game season mean ~= projected/GAMES for the real player
    assert abs(s[:, :, 0].mean() - 170.0 / 17) < 0.3
    # projected_mean <= 0 -> all zero
    assert (s[:, :, 1] == 0).all()


def test_rookie_wider_than_veteran() -> None:
    vp = _params()
    pos = np.array(["WR", "WR"])
    means = np.array([170.0, 170.0])
    rook = np.array([False, True])
    s = sample_weekly_points(vp, pos, means, rook, n_sims=6000, n_weeks=17, rng=np.random.default_rng(2))
    # season totals: rookie has wider spread (component ii) than veteran at same projection
    season = s.sum(axis=1)  # (n_sims, 2)
    assert season[:, 1].std() > season[:, 0].std()
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_draft/test_assistant/test_performance_variance.py -k sampler -v`
Expected: FAIL (`sample_weekly_points` not defined).

- [ ] **Step 3: Implement**

```python
# append to performance_variance.py
_GAMES = 17


def sample_weekly_points(
    params: VarianceParams,
    positions: np.ndarray,  # (n_players,) str
    projected_means: np.ndarray,  # (n_players,) season points
    is_rookie: np.ndarray,  # (n_players,) bool
    *,
    n_sims: int,
    n_weeks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return (n_sims, n_weeks, n_players) non-negative sampled weekly points.

    Component ii (per sim, player): mean-preserving lognormal m, E[m]=1 ->
    true_mean = projected_mean * m. Component i (per sim, week, player): Gamma with
    per-game mean pg = true_mean/GAMES and std = f_pos(pg). projected_mean<=0 -> 0.
    """
    n = len(positions)
    log_sd = np.array(
        [params.log_sd(str(p), is_rookie=bool(r)) for p, r in zip(positions, is_rookie, strict=True)]
    )  # (n,)
    # mean-preserving lognormal: mu = -sigma^2/2 so E[m]=1
    mu = -0.5 * log_sd**2
    m = rng.lognormal(mean=mu, sigma=log_sd, size=(n_sims, n))  # (n_sims, n)
    true_mean = projected_means[None, :] * m  # (n_sims, n)
    pg = true_mean / _GAMES  # per-game mean, (n_sims, n)

    a = np.array([params.weekly_std_affine.get(str(p), params.weekly_std_affine["default"])["a"] for p in positions])
    b = np.array([params.weekly_std_affine.get(str(p), params.weekly_std_affine["default"])["b"] for p in positions])
    std = np.maximum(a[None, :] * pg + b[None, :], 1e-9)  # (n_sims, n), per-game weekly std

    # Gamma(shape=k, scale=theta) with mean=pg, std -> k=(pg/std)^2, theta=std^2/pg. Broadcast to weeks.
    valid = pg > 0
    k = np.where(valid, (pg / std) ** 2, 1.0)  # placeholder where invalid
    theta = np.where(valid, std**2 / np.where(valid, pg, 1.0), 0.0)
    # draw (n_sims, n_weeks, n): tile shape/scale across weeks
    shape3 = np.broadcast_to(k[:, None, :], (n_sims, n_weeks, n))
    scale3 = np.broadcast_to(theta[:, None, :], (n_sims, n_weeks, n))
    pts = rng.gamma(shape=shape3, scale=scale3)  # (n_sims, n_weeks, n); 0 where theta=0
    pts = np.where(valid[:, None, :], pts, 0.0)
    return pts
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_draft/test_assistant/test_performance_variance.py -k sampler -v`
Expected: PASS (3 tests). If `test_sampler_recovers_mean` is flaky at the 0.3 tolerance, it is set generously for n_sims=4000; do not loosen the model.

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/performance_variance.py tests/test_draft/test_assistant/test_performance_variance.py
git commit -m "feat(variance): vectorized two-component weekly-points sampler"
```

### Task 3: Offline fitter

**Files:**
- Create: `scripts/fit_performance_variance.py`
- Test: `tests/test_scripts/test_fit_performance_variance.py`

- [ ] **Step 1: Write the failing test** (synthetic frames with known structure)

```python
# tests/test_scripts/test_fit_performance_variance.py
import numpy as np
import pandas as pd

from scripts.fit_performance_variance import fit_params


def test_fit_recovers_affine_and_logsd() -> None:
    rng = np.random.default_rng(0)
    # Build synthetic per-player-season rows: WR with std = 0.3*pg + 3, veterans only.
    rows = []
    for i in range(60):
        pg = rng.uniform(6, 18)
        weekly = rng.normal(pg, 0.3 * pg + 3.0, size=17).clip(min=0)
        rows.append({"gsis_id": f"00-{i:07d}", "position": "WR", "season": 2022, "weekly": weekly,
                     "projected_pg": pg, "is_rookie": False})
    params = fit_params(rows)
    wr = params["weekly_std_affine"]["WR"]
    assert abs(wr["a"] - 0.3) < 0.12 and abs(wr["b"] - 3.0) < 1.5
    assert "default|veteran" in params["mean_mult_log_sd"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_scripts/test_fit_performance_variance.py -v`
Expected: FAIL (`fit_params` not defined).

- [ ] **Step 3: Implement** the pure `fit_params(rows)` + a `main()` that builds `rows` from the store and writes JSON.

`fit_params(rows)` logic (rows carry `position`, `weekly` array, `projected_pg`, `is_rookie`):
- Per player-season: `mean_pg = weekly.mean()`, `std_pg = weekly.std()`. For the affine, collect `(mean_pg, std_pg)` per position and `np.polyfit(mean_pg, std_pg, 1)` → `a, b` (require ≥ 20 player-seasons in a position, else fall to the pooled `default` fit).
- For `mean_mult_log_sd`: per (position, rookie tier), collect `log(mean_pg / projected_pg)` (drop non-finite), take `np.std`. Require ≥ 15 player-seasons per cell, else fall back to the pooled `default|tier`. Always emit `default|veteran`, `default|rookie`, and `default` affine from the pooled fit.

```python
# scripts/fit_performance_variance.py  (core; main() omitted here, shown in Step 3b)
from __future__ import annotations

import numpy as np

_MIN_AFFINE = 20
_MIN_LOGSD = 15


def fit_params(rows: list[dict]) -> dict:
    by_pos: dict[str, list[tuple[float, float]]] = {}
    log_by_cell: dict[str, list[float]] = {}
    all_ms: list[tuple[float, float]] = []
    all_log: dict[str, list[float]] = {"veteran": [], "rookie": []}
    for r in rows:
        w = np.asarray(r["weekly"], dtype=float)
        if w.size < 2 or w.mean() <= 0:
            continue
        mean_pg, std_pg = float(w.mean()), float(w.std())
        by_pos.setdefault(r["position"], []).append((mean_pg, std_pg))
        all_ms.append((mean_pg, std_pg))
        ppg = float(r["projected_pg"])
        if ppg > 0:
            ratio = mean_pg / ppg
            if np.isfinite(ratio) and ratio > 0:
                tier = "rookie" if r["is_rookie"] else "veteran"
                log_by_cell.setdefault(f"{r['position']}|{tier}", []).append(float(np.log(ratio)))
                all_log[tier].append(float(np.log(ratio)))

    def affine(ms: list[tuple[float, float]]) -> dict[str, float]:
        arr = np.array(ms)
        a, b = np.polyfit(arr[:, 0], arr[:, 1], 1)
        return {"a": float(a), "b": float(b)}

    weekly_std_affine = {"default": affine(all_ms)}
    for pos, ms in by_pos.items():
        weekly_std_affine[pos] = affine(ms) if len(ms) >= _MIN_AFFINE else weekly_std_affine["default"]

    mean_mult_log_sd = {
        "default|veteran": float(np.std(all_log["veteran"])),
        "default|rookie": float(np.std(all_log["rookie"])),
    }
    for cell, logs in log_by_cell.items():
        if len(logs) >= _MIN_LOGSD:
            mean_mult_log_sd[cell] = float(np.std(logs))
    return {"weekly_std_affine": weekly_std_affine, "mean_mult_log_sd": mean_mult_log_sd}
```

- [ ] **Step 3b: Add `main()`** that builds `rows` from the store: for each season 2019–2025 (2018 reserved as rookie-detection lookback only), read `weekly_stats`, `build_weekly_actuals` → per-(gsis,week) points; group to per-player-season `weekly` arrays + position; `is_rookie` = no `weekly_stats` appearance in any earlier season (from 2018+); `projected_pg` = that season's blended `build_draft_basis` `season_mean_fpts`/17 (skip player-seasons without a projection for the log-SD part, but keep them for the affine which needs no projection). Write `fit_params(rows)` to `configs/performance_variance_params.json` (sorted keys, indent 2). Exact store calls mirror `scripts/post_draft_assessment.py` / `_diag` patterns: `read_partition(Path("data/raw"), "weekly_stats", season=yr)`, `read_latest_partition(Path("data/raw"), "external_projections", season=yr)`, `build_draft_basis(..., league_config=LeagueConfig.model_validate_json(Path("configs/league_espn_half_16team.json").read_text()))`.

- [ ] **Step 4: Run to verify pass** + lint

Run: `pytest tests/test_scripts/test_fit_performance_variance.py -v`
Expected: PASS. Then `ruff check scripts/fit_performance_variance.py`.

- [ ] **Step 5: Commit**

```bash
git add scripts/fit_performance_variance.py tests/test_scripts/test_fit_performance_variance.py
git commit -m "feat(variance): offline fitter (per-game affine + per-cell log-SD)"
```

### Task 4: Generate + commit real params (controller-run verification)

**Files:** Create `configs/performance_variance_params.json` (generated).

- [ ] **Step 1:** Run the fitter on real data (PowerShell, OpenMP-safe, retry):
`python -u scripts/fit_performance_variance.py`
- [ ] **Step 2: Verify (R7a)** the output: per-position `a` positive, `b` positive; `default|rookie` σ > `default|veteran` σ; sampling from the fitted params reproduces per-position weekly CV within ±0.05 of QB 0.56 / RB 0.71 / WR 0.75 / TE 0.72 (check at representative means via a one-off using `sample_weekly_points`) and the sampled `m` arithmetic ratio SD within ±0.05 of 0.39 vet / 0.58 rookie. Print these checks.
- [ ] **Step 3: Commit** `configs/performance_variance_params.json` with a message noting the R7a check results.

### Task 5: `is_rookie` on the pool (R6)

**Files:**
- Modify: `src/projections/draft/backtest/inputs.py`
- Test: `tests/test_draft/test_backtest/test_inputs_is_rookie.py`

- [ ] **Step 1: Write the failing test** — a `load_inputs`-level test is integration-heavy; instead unit-test a pure helper `_attach_is_rookie(pool, prior_gsis)`:

```python
import pandas as pd
from projections.draft.backtest.inputs import _attach_is_rookie

def test_attach_is_rookie() -> None:
    pool = pd.DataFrame({"gsis_id": ["00-0000001", "00-0000002"], "position": ["WR", "RB"]})
    out = _attach_is_rookie(pool, prior_gsis={"00-0000001"})
    by = dict(zip(out["gsis_id"], out["is_rookie"]))
    assert by["00-0000001"] is False or by["00-0000001"] == False  # appeared before -> veteran
    assert bool(by["00-0000002"]) is True  # never appeared -> rookie
```

- [ ] **Step 2: Run to verify it fails** (`_attach_is_rookie` missing).
- [ ] **Step 3: Implement** `_attach_is_rookie(pool, prior_gsis)` (sets `is_rookie = ~gsis.isin(prior_gsis)` as a boolean column) and, in `load_inputs`, build `prior_gsis` by reading `weekly_stats` for seasons `2018..season-1` (union of `gsis_id`), then `pool = _attach_is_rookie(pool, prior_gsis)`. Tolerate missing prior partitions (skip a season whose partition is absent).
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** `feat(backtest): attach is_rookie to the pool from prior-season weekly_stats`.

### Phase 1 gate
Run `pytest -v -k "performance_variance or fit_performance or is_rookie or inputs"`, `mypy src tests`, `ruff check src tests scripts`, `ruff format --check src tests scripts`. All green.

---

## PHASE 2 — Consumer A (risk-aware season_value MC)

### Task 6: Sampled-points optimal-lineup fill

**Files:**
- Modify: `src/projections/draft/assistant/season_value.py`
- Test: `tests/test_draft/test_assistant/test_season_value_variance.py`

- [ ] **Step 1: Write the failing test** — the generalized fill, filled by per-row sampled points, must equal the deterministic fill when every row's points equal the fixed `season_mean_fpts`.

```python
import numpy as np
import pandas as pd
from projections.schemas import RosterSlot
from projections.draft.assistant.season_value import _roster_fill_meta, _vectorized_lineup_points, _lineup_points_sampled

def _roster():
    return pd.DataFrame({
        "gsis_id": [f"00-000000{i}" for i in range(5)],
        "position": ["QB", "RB", "RB", "WR", "WR"],
        "season_mean_fpts": [300.0, 250.0, 200.0, 220.0, 180.0],
    })

def test_sampled_fill_matches_fixed_when_points_constant() -> None:
    roster = _roster()
    slots = {RosterSlot.QB: 1, RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.FLEX: 1}
    meta = _roster_fill_meta(roster, slots)
    avail = np.ones((6, len(roster)), bool)
    fixed = _vectorized_lineup_points(avail, meta)
    pos = roster["position"].to_numpy().astype(str)
    pts = np.broadcast_to(roster["season_mean_fpts"].to_numpy()[None, :], (6, len(roster)))
    sampled = _lineup_points_sampled(pts, avail, pos, slots)
    assert np.allclose(sampled, fixed)
```

- [ ] **Step 2: Run to verify it fails** (`_lineup_points_sampled` missing).
- [ ] **Step 3: Implement** `_lineup_points_sampled(points, avail, pos, roster_slots)` — `points`,`avail` are `(R, n)`; restrictive-first greedy by the row's own points:

```python
def _lineup_points_sampled(
    points: np.ndarray, avail: np.ndarray, pos: np.ndarray, roster_slots: Mapping[RosterSlot, int]
) -> np.ndarray:
    """Optimal legal lineup points per row (R, n), ranked by each row's own `points`.

    Generalizes `_vectorized_lineup_points` to per-row point values (sampled weekly points)
    instead of a fixed per-player value. Same restrictive-first greedy (single slots, then
    FLEX/SUPER_FLEX narrowest-first) — optimal for laminar slots."""
    R, n = points.shape
    total = np.zeros(R, dtype=np.float64)
    used = np.zeros((R, n), dtype=bool)
    eff = np.where(avail, points, -np.inf)
    rows = np.arange(R)[:, None]
    for slot in POSITION_SLOTS:
        count = roster_slots.get(slot, 0)
        if count <= 0:
            continue
        cols = np.flatnonzero(pos == slot.value)
        if cols.size == 0:
            continue
        sub = np.where(used[:, cols], -np.inf, eff[:, cols])  # (R, m)
        k = min(count, cols.size)
        idx = np.argsort(-sub, axis=1)[:, :k]  # (R, k) top-k by points
        vals = sub[rows, idx]
        total += np.where(vals > -np.inf, vals, 0.0).sum(axis=1)
        chosen = cols[idx]  # (R, k) roster columns
        valid = vals > -np.inf
        used[rows, chosen] |= valid
    for slot, eligible in _FLEX_SLOTS:
        count = roster_slots.get(slot, 0)
        if count <= 0:
            continue
        cols = np.flatnonzero(np.isin(pos, [p.value for p in eligible]))
        if cols.size == 0:
            continue
        for _ in range(count):
            sub = np.where(used[:, cols], -np.inf, eff[:, cols])
            best_local = sub.argmax(axis=1)
            best_val = sub[np.arange(R), best_local]
            has = best_val > -np.inf
            total += np.where(has, best_val, 0.0)
            chosen = cols[best_local]
            sel = np.flatnonzero(has)
            used[sel, chosen[sel]] = True
    return total
```

- [ ] **Step 4: Run to verify pass.** (Equivalence with the fixed fill.)
- [ ] **Step 5: Commit** `feat(season_value): sampled-points optimal-lineup fill`.

### Task 7: Sampler-backed expected_season_points + marginals (CRN)

**Files:**
- Modify: `src/projections/draft/assistant/season_value.py`
- Test: `tests/test_draft/test_assistant/test_season_value_variance.py`

This replaces the week-factorized deterministic MC with a full `(n_sims, weeks, players)` sampled MC. New signatures take a `VarianceParams` and per-player `is_rookie`. The CRN draws shared across base+candidates become the sampler's rng-driven `(m, weekly)` draws keyed on the union columns.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np, pandas as pd
from projections.schemas import RosterSlot
from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.assistant.season_value import expected_season_points_var, marginal_season_values_var

def _vp_zero():  # ~zero variance -> reduces to deterministic per_game
    return VarianceParams({"default": {"a": 0.0, "b": 1e-7}}, {"default|veteran": 1e-7, "default|rookie": 1e-7})

def _avail_all():  # everyone always available, no byes
    return PlayerAvailability(p={}, bye={})  # p_week defaults to 1.0; bye_week None  (verify defaults)

def test_var_reduces_to_deterministic_at_zero_variance() -> None:
    roster = pd.DataFrame({"gsis_id":["a","b","c"],"position":["QB","RB","WR"],
                           "season_mean_fpts":[300.0,250.0,200.0],"is_rookie":[False,False,False]})
    slots={RosterSlot.QB:1,RosterSlot.RB:1,RosterSlot.WR:1}
    v = expected_season_points_var(roster, slots, _avail_all(), _vp_zero(),
                                   n_sims=200, n_weeks=14, rng=np.random.default_rng(0))
    # deterministic season points = sum of starters' per_game * 14 weeks = (300+250+200)/17*14
    assert abs(v - (750.0/17*14)) < 5.0

def test_marginal_crn_ranks_better_candidate_first() -> None:
    base = pd.DataFrame({"gsis_id":["a"],"position":["QB"],"season_mean_fpts":[300.0],"is_rookie":[False]})
    cands = pd.DataFrame({"gsis_id":["x","y"],"position":["RB","RB"],
                          "season_mean_fpts":[250.0,120.0],"is_rookie":[False,False]})
    slots={RosterSlot.QB:1,RosterSlot.RB:1}
    out = marginal_season_values_var(base, cands, slots, _avail_all(), _vp_zero(),
                                     n_sims=200, n_weeks=14, rng=np.random.default_rng(0))
    assert out["x"] > out["y"] > 0
```

(Confirm `PlayerAvailability(p={}, bye={})` defaults `p_week`→1.0 and `bye_week`→None by reading `availability.py`; if the real constructor differs, build a trivial availability with all `p=1.0` and no byes per its actual API.)

- [ ] **Step 2: Run to verify they fail** (`expected_season_points_var` / `marginal_season_values_var` missing).
- [ ] **Step 3: Implement** `expected_season_points_var` and `marginal_season_values_var`:
  - `expected_season_points_var(roster, roster_slots, availability, params, *, n_sims, n_weeks, rng, weeks=range(1,15))`: build `(n_sims, n_weeks, n)` points via `sample_weekly_points(params, pos, means, is_rookie, ...)`; build availability `(n_sims, n_weeks, n)` from `p_week` (Bernoulli per sim-week) AND byes (force unavailable in the bye week — map week index to calendar week, set avail False where `bye_week==week`); flatten to `(n_sims*n_weeks, n)`, call `_lineup_points_sampled`, reshape `(n_sims, n_weeks)`, sum over weeks → per-sim season totals, `.mean()`. (No `_GAMES` division — points are already per-game weekly and summed over the played weeks.)
  - `marginal_season_values_var(base, candidates, roster_slots, availability, params, *, n_sims, n_weeks, rng, weeks)`: CRN — draw ONE set of `(m, weekly, availability)` over the union of base+candidate ids using a single rng pass keyed by column; evaluate base and each `base+candidate` against the **same** drawn arrays (slice columns), so the marginal cancels common noise. Mirror the existing `marginal_season_values` column-union/`col_of` pattern but with the sampled arrays instead of `draws`.
  - Keep the old `expected_season_points` / `marginal_season_values` (deterministic) in place for now; the strategy switch is Task 8. This keeps the diff additive and the equivalence test meaningful.
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** `feat(season_value): sampler-backed risk-aware season value + CRN marginals`.

### Task 8: Wire the strategy + speed gate (R4)

**Files:**
- Modify: `src/projections/draft/assistant/strategy.py` (the `_season_marginals` helper that `SeasonValueStrategy`/`SeasonValueTimingStrategy` share), `season_value.py` if a thin adapter is needed.
- Test: `tests/test_draft/test_assistant/test_season_value_variance.py` (speed micro-bench, marked slow).

- [ ] **Step 1:** Add the `VarianceParams` (loaded once) + per-player `is_rookie` into `_season_marginals` so the season-value strategies call `marginal_season_values_var`. `is_rookie` comes from the pool column (Task 5); the params load from `configs/performance_variance_params.json` once at strategy construction.
- [ ] **Step 2: Speed gate** — micro-benchmark a single `recommend` at `n_sims=200`, `n_weeks=14`, mid-draft roster (reuse the `_diag_time.py` harness shape). Assert/observe ≤ 250 ms/pick. If exceeded, lower the strategy's live default `n_sims` until ≤ 250 ms and record the chosen default (R4 fallback). State the measured ms.
- [ ] **Step 3: Validation (R7b)** — run the existing H2H real-outcome backtest smoke (a few seeds) with the risk-aware strategy and confirm season_value remains competitive (does not collapse vs now_or_never/bots). State results.
- [ ] **Step 4: Commit** `feat(strategy): risk-aware season_value uses the variance model; n_sims tuned to <=250ms`.

### Phase 2 gate
`pytest -v -k "season_value or strategy"`, mypy, ruff, format. All green.

---

## PHASE 3 — Consumer B (predictive forward CIs)

### Task 9: Model-sampled predictive league simulation

**Files:**
- Create: `src/projections/draft/backtest/predictive.py`
- Test: `tests/test_draft/test_backtest/test_predictive.py`

- [ ] **Step 1: Write the failing test** — a predictive run with ~zero variance ≈ a deterministic-outcome league (champ in {0,1} per sim, stable); with real variance the per-team outcome distribution has spread.

```python
import numpy as np
from projections.draft.backtest.predictive import sample_actual_lookup

def test_sample_actual_lookup_shape_and_zerovar() -> None:
    # pool: 2 players; zero-variance params -> sampled weekly ~= projected per_game
    import pandas as pd
    from projections.draft.assistant.performance_variance import VarianceParams
    pool = pd.DataFrame({"gsis_id":["a","b"],"position":["QB","RB"],
                         "season_mean_fpts":[340.0,170.0],"is_rookie":[False,False]})
    vp = VarianceParams({"default":{"a":0.0,"b":1e-7}}, {"default|veteran":1e-7,"default|rookie":1e-7})
    lut = sample_actual_lookup(pool, vp, weeks=range(1,15), rng=np.random.default_rng(0))
    assert abs(lut[("a", 1)] - 340.0/17) < 0.5
    assert ("b", 14) in lut
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** `sample_actual_lookup(pool, params, *, weeks, rng) -> dict[tuple[str,int], float]` — one predictive season's per-(gsis,week) points by `sample_weekly_points(params, pool positions, season_mean_fpts, is_rookie, n_sims=1, n_weeks=len(weeks), rng)`, mapping week-index → calendar week. Then a `predictive_outcomes(...)` that, for `n_predictive_sims`, builds a sampled `actual_lookup` and calls `simulate_league` (same `proj_lookup`, seats, calendar) → collects per-seat champ/playoff/wins, returning arrays for CI computation. Lineups stay set by projection (unchanged `simulate_league` behavior); only `actual_lookup` is model-sampled.
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** `feat(backtest): model-sampled predictive league simulation`.

### Task 10: post_draft_assessment predictive-CI readout + validation (R7c)

**Files:**
- Modify: `scripts/post_draft_assessment.py`
- Test: `tests/test_scripts/test_post_draft_assessment.py` (a small smoke that the predictive readout runs on a tiny synthetic input).

- [ ] **Step 1:** Add a `--predictive` flag that runs `predictive_outcomes` over the committed draft pool(s) and prints forward champ%/playoff%/wins with CIs for a season_value draft, labeled distinctly from the historical-actuals tables.
- [ ] **Step 2: Validation (R7c)** — run it and confirm the forward CIs span ≈ the historical cross-season spread (champ ≈ 5–18%, playoff ≈ 50–78%, wins ≈ 7.7–9.2) — materially wider than the ±2–3% draft/schedule-only bootstrap. State the numbers.
- [ ] **Step 3: Commit** `feat(analysis): predictive forward-CI readout in post_draft_assessment (TODO #45)` and update `TODO.md` #45 → done / remaining (per-player shrinkage, correlation) deferred.

### Phase 3 gate + final
`pytest -v` (or the stated subset), mypy, ruff, format. Then `superpowers:finishing-a-development-branch`.

---

## Self-Review

**Spec coverage:** R1→Task 3; R2→Tasks 1–2; R3→Task 7; R4→Task 8; R5→Task 9; R6→Task 5; R7a→Task 4; R7b→Task 8; R7c→Task 10. Two-component model (mean-preserving lognormal + per-game Gamma) in Task 2 matches the corrected spec. Consumer A optimal-by-sampled fill in Task 6 matches the spec assumption. Consumer B keeps real backtest intact (new module) per non-goals.

**Placeholder scan:** every code step has concrete code; commands have expected outcomes. Task 4/8/10 are controller-run verification/validation with explicit checks (not "TBD"). Tasks 8/9 reference functions defined in Tasks 6/7/2.

**Type consistency:** `VarianceParams` (Task 1) used identically in Tasks 2/3/7/9; `sample_weekly_points` signature `(params, positions, projected_means, is_rookie, *, n_sims, n_weeks, rng)` consistent across Tasks 2/7/9; `_lineup_points_sampled(points, avail, pos, roster_slots)` (Task 6) used by Task 7; `is_rookie` pool column (Task 5) consumed in Tasks 7/8/9; new `_var` MC functions are additive (old deterministic ones untouched until Task 8 switches the strategy).
