# Season-Value Timing Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `season_value_timing` draft strategy — `score = marginal_season_value − E[best surviving marginal at position]` (now-or-never's timing layer in season-value units, no extra Monte-Carlo) — and make any strategy pair testable in the H2H harness.

**Architecture:** Compose the two existing strategies. Extract now-or-never's "expected best survivor" accumulation into a shared `survival.expected_best_by_position` helper; the new `SeasonValueTimingStrategy` reuses `season_value`'s marginal MC for the value term and the helper (fed marginals) for the opportunity-cost term. Generalize the harness's two strategy "roles" (A/B) so `(now_or_never, season_value)` stays the byte-identical default while `(season_value_timing, season_value)` becomes a testable field.

**Tech Stack:** Python, numpy, pandas, pandera, pytest. Source under `src/projections/draft/assistant/` and `src/projections/draft/backtest/`; tests under `tests/test_draft/`.

**Source spec:** `docs/superpowers/specs/2026-06-13-season-value-timing-strategy-design.md`

---

## File structure

- `src/projections/draft/assistant/survival.py` — **modify**: add `expected_best_by_position`.
- `src/projections/draft/assistant/strategy.py` — **modify**: refactor `NowOrNeverStrategy` onto the helper; add `SeasonValueTimingStrategy`.
- `src/projections/draft/backtest/draft_field.py` — **modify**: `seat_layout` takes optional `label_a`/`label_b`.
- `src/projections/draft/backtest/harness.py` — **modify**: `_build_strategy` registry; `collect_results`/`run_backtest` take A/B keys; `aggregate` derives labels dynamically.
- `src/projections/draft/backtest/cli.py` + `scripts/h2h_backtest.py` + `scripts/h2h_backtest_chunked.py` — **modify**: `--strategy-a`/`--strategy-b`.
- `src/projections/draft/assistant/cli.py` + `scripts/draft_assistant.py` — **modify**: `--strategy season_value_timing`.
- Tests: `test_assistant_survival.py`, `test_assistant_strategy.py`, `test_backtest/test_draft_field.py`, `test_backtest/test_harness.py`, `test_backtest/test_cli.py`, `test_assistant_cli.py`.

---

## Phase 1 — The strategy

### Task 1: `expected_best_by_position` helper

**Files:**
- Modify: `src/projections/draft/assistant/survival.py`
- Test: `tests/test_draft/test_assistant_survival.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_draft/test_assistant_survival.py`:

```python
import numpy as np

from projections.draft.assistant.survival import expected_best_by_position


def test_expected_best_by_position_hand_computed() -> None:
    # RB: values [50, 40], survival [0.5, 0.5] -> 50*0.5*1 + 40*0.5*(1-0.5) = 35
    # WR: value 52, survival 0.5 -> 52*0.5 = 26
    positions = np.array(["RB", "RB", "WR"])
    values = np.array([50.0, 40.0, 52.0])
    probs = np.array([0.5, 0.5, 0.5])
    tiebreak = np.array(["00-0000001", "00-0000002", "00-0000003"])
    out = expected_best_by_position(positions, values, probs, tiebreak)
    assert out == {"RB": 35.0, "WR": 26.0}


def test_expected_best_by_position_sorts_by_value_then_tiebreak() -> None:
    # Unsorted input: the higher value is consumed first regardless of row order.
    positions = np.array(["RB", "RB"])
    values = np.array([40.0, 50.0])
    probs = np.array([0.5, 0.5])
    tiebreak = np.array(["00-0000002", "00-0000001"])
    out = expected_best_by_position(positions, values, probs, tiebreak)
    assert out["RB"] == 35.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_draft/test_assistant_survival.py -k expected_best -q`
Expected: FAIL — `ImportError: cannot import name 'expected_best_by_position'`.

- [ ] **Step 3: Implement the helper**

In `src/projections/draft/assistant/survival.py`, add imports at the top (after `import math`):

```python
from itertools import groupby

import numpy as np
```

Then add at the end of the file:

```python
def expected_best_by_position(
    positions: np.ndarray,
    values: np.ndarray,
    probs: np.ndarray,
    tiebreak: np.ndarray,
) -> dict[str, float]:
    """Expected value of the best *surviving* player at each position.

    For each position, players are sorted by value descending (deterministic
    `tiebreak`, ascending, breaks ties), and the expected max over survivors is
    accumulated sequentially: ``value_i * p_i * prod_{better j}(1 - p_j)``. The
    accumulation is sequential (not ``np.sum``) so the float result is stable across
    row order. Shared by now_or_never (values = VORP) and the season-value timing
    strategy (values = marginal season points).
    """
    order = np.lexsort((tiebreak, -values, positions))
    out: dict[str, float] = {}
    rows = zip(
        positions[order].tolist(),
        values[order].tolist(),
        probs[order].tolist(),
        strict=True,
    )
    for position, group in groupby(rows, key=lambda r: r[0]):
        expected = 0.0
        prob_all_better_gone = 1.0
        for _, value_i, p_i in group:
            expected += value_i * p_i * prob_all_better_gone
            prob_all_better_gone *= 1.0 - p_i
        out[str(position)] = expected
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_draft/test_assistant_survival.py -k expected_best -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/survival.py tests/test_draft/test_assistant_survival.py
git commit -m "feat(draft): expected_best_by_position helper (shared survivor accumulation)"
```

---

### Task 2: Refactor `NowOrNeverStrategy` onto the helper (bit-identical)

**Files:**
- Modify: `src/projections/draft/assistant/strategy.py:151-174` (the inline accumulation inside `NowOrNeverStrategy.recommend`)
- Test: `tests/test_draft/test_assistant_strategy.py` (existing nn tests are the guard — no new test)

- [ ] **Step 1: Confirm the existing nn tests pass before the change**

Run: `python -m pytest tests/test_draft/test_assistant_strategy.py -q`
Expected: PASS (records the green baseline that must be preserved).

- [ ] **Step 2: Replace the inline accumulation with the helper**

In `strategy.py`, add `expected_best_by_position` to the survival import:

```python
from projections.draft.assistant.survival import SurvivalModel, expected_best_by_position
```

In `NowOrNeverStrategy.recommend`, replace the block from `# E[best survivor at each position]:` through the `for position, group in groupby(...)` loop (the `e_best` construction) — keep the surrounding `pos`/`vorp`/`p`/`gsis` extraction and the final `df["score"] = vorp - ...` line — with:

```python
        # E[best survivor at each position], shared with the season-value timing
        # strategy. Same lexsort + sequential accumulation as before -> bit-identical.
        pos = df["position"].to_numpy()
        vorp = df["vorp"].to_numpy(dtype=float)
        p = internal_p.to_numpy(dtype=float)
        gsis = df["gsis_id"].to_numpy()
        e_best = expected_best_by_position(pos, vorp, p, gsis)

        # score = vorp - E[best survivor at position], reusing the numpy arrays.
        df["score"] = vorp - np.array([e_best[pos_i] for pos_i in pos], dtype=float)
        return _finalize(df, elig, display_p)
```

(The `from itertools import groupby` import in `strategy.py` may now be unused — if `ruff` flags it, remove it.)

- [ ] **Step 3: Run the nn tests to verify they still pass (bit-identical guard)**

Run: `python -m pytest tests/test_draft/test_assistant_strategy.py -q`
Expected: PASS — same set as Step 1. If any nn float assertion now fails, **revert this task** and instead leave nn untouched, duplicating the helper's accumulation inline in Task 3 (spec §4 fallback).

- [ ] **Step 4: Commit**

```bash
git add src/projections/draft/assistant/strategy.py
git commit -m "refactor(draft): now_or_never uses the shared expected_best_by_position"
```

---

### Task 3: `SeasonValueTimingStrategy`

**Files:**
- Modify: `src/projections/draft/assistant/strategy.py` (add the class after `SeasonValueStrategy`)
- Test: `tests/test_draft/test_assistant_strategy.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_draft/test_assistant_strategy.py` (reuses the file's existing `_config`, `_pool`, `_state` and imports `PlayerAvailability`, `LogisticSurvival`):

```python
from projections.draft.assistant.season_value import marginal_season_values
from projections.draft.assistant.strategy import SeasonValueTimingStrategy


def _flat_availability(pool: pd.DataFrame) -> PlayerAvailability:
    # Every player available every week, no byes -> deterministic, MC-stable marginals.
    p = {str(g): 0.95 for g in pool["gsis_id"]}
    return PlayerAvailability(p_by_gsis=p, byes_by_gsis={}, default_p_by_position={})


def _timing(pool: pd.DataFrame, sigma: float = 8.0) -> SeasonValueTimingStrategy:
    return SeasonValueTimingStrategy(
        _flat_availability(pool), n_sims=20, base_seed=0, survival=LogisticSurvival(sigma=sigma)
    )


def test_timing_validates_construction() -> None:
    av = _flat_availability(_pool())
    with pytest.raises(ValueError):
        SeasonValueTimingStrategy(av, n_sims=0, base_seed=0, survival=LogisticSurvival(sigma=8.0))
    with pytest.raises(ValueError):
        SeasonValueTimingStrategy(
            av, n_sims=20, base_seed=0, survival=LogisticSurvival(sigma=8.0), top_k=0
        )


def test_timing_is_deterministic() -> None:
    pool, state, cfg = _pool(), _state(), _config()
    r1 = _timing(pool).recommend(state, pool, cfg)
    r2 = _timing(pool).recommend(state, pool, cfg)
    pd.testing.assert_frame_equal(r1, r2)


def test_timing_score_equals_marginal_minus_opp_cost() -> None:
    # The strategy's score must be exactly marginal - E[best surviving marginal at pos],
    # with no scaling fudge factor. Recompute both pieces independently and compare.
    pool, state, cfg = _pool(), _state(), _config()
    strat = _timing(pool)
    rec = strat.recommend(state, pool, cfg)

    # Recompute marginals with the same rng the strategy uses.
    df = pool[~pool["gsis_id"].isin(state.drafted_ids)].copy()
    pruned = (
        df.sort_values(["position", "vorp"], ascending=[True, False])
        .groupby("position", sort=False)
        .head(8)
    )
    rng = np.random.default_rng([0, state.current_pick])
    base = pool.loc[pool["gsis_id"].isin([str(g) for g in state.my_pick_ids])]
    marg = marginal_season_values(
        base[["gsis_id", "position", "season_mean_fpts"]],
        pruned[["gsis_id", "position", "season_mean_fpts"]],
        cfg.roster_slots,
        _flat_availability(pool),
        n_sims=20,
        rng=rng,
    )
    from projections.draft.assistant.pick_timing import my_next_pick
    from projections.draft.assistant.survival import LogisticSurvival, expected_best_by_position

    nxt = my_next_pick(state.current_pick, state.my_slot, state.n_teams, state.rounds)
    surv = LogisticSurvival(sigma=8.0)
    by_id = {str(r.gsis_id): r for r in rec.itertuples(index=False)}
    pos = np.array([str(r.position) for r in rec.itertuples(index=False)])
    m = np.array([float(marg.get(str(r.gsis_id), 0.0)) for r in rec.itertuples(index=False)])
    p = np.array(
        [
            surv.p_available(float(r.consensus_adp) if pd.notna(r.consensus_adp) else float("nan"), nxt)
            for r in rec.itertuples(index=False)
        ]
    )
    gid = np.array([str(r.gsis_id) for r in rec.itertuples(index=False)])
    opp = expected_best_by_position(pos, m, p, gid)
    for i, r in enumerate(rec.itertuples(index=False)):
        expected = round(m[i] - opp[str(r.position)], 10)
        assert abs(float(r.score) - expected) < 1e-9


def test_timing_promotes_scarce_position_over_safer_higher_marginal() -> None:
    # rb1 is scarce (adp 1 -> ~0 survival to next pick) so its position's opp_cost ~ 0
    # (nothing survives) -> it keeps ~full marginal. wr1 is the highest-marginal player
    # but safe (adp 200 -> survives) so its opp_cost ~ its own marginal -> score ~ 0.
    # season_value ranks wr1 first (highest marginal); timing flips rb1 to the top.
    pool = pd.DataFrame(
        {
            "gsis_id": pd.array(
                ["00-0000010", "00-0000011", "00-0000020", "00-0000021"], dtype=_PYARROW_STR
            ),
            "position": pd.array(["RB", "RB", "WR", "WR"], dtype=_PYARROW_STR),
            "season_mean_fpts": [250.0, 120.0, 255.0, 250.0],
            "vorp": [50.0, 20.0, 55.0, 50.0],
            "replacement_fpts": [200.0, 100.0, 200.0, 200.0],
            "consensus_adp": pd.array([1.0, 90.0, 200.0, 200.0], dtype=pd.Float64Dtype()),
        }
    )
    state, cfg = _state(), _config()
    sv = SeasonValueStrategy(_flat_availability(pool), n_sims=20, base_seed=0)
    sv_top = sv.recommend(state, pool, cfg).iloc[0]
    timing_top = _timing(pool).recommend(state, pool, cfg).iloc[0]
    assert sv_top["gsis_id"] == "00-0000020"  # season_value: highest marginal (wr1)
    assert timing_top["gsis_id"] == "00-0000010"  # timing: scarce rb1 promoted
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_draft/test_assistant_strategy.py -k timing -q`
Expected: FAIL — `ImportError: cannot import name 'SeasonValueTimingStrategy'`.

- [ ] **Step 3: Implement the strategy**

In `strategy.py`, after the `SeasonValueStrategy` class, add:

```python
@dataclass(frozen=True)
class SeasonValueTimingStrategy:
    """Depth-aware + pick-timing (spec §3): season_value's marginal minus the
    opportunity cost of waiting, in season-value units.

    score = marginal_season_value(c) - E[best surviving marginal at pos(c) by my next pick].
    Same per-pick cost as SeasonValueStrategy (one marginal MC); the timing term reuses
    the already-computed marginals + the ADP survival model (no extra MC). Last pick ->
    rank by raw marginal (today's season_value), mirroring nn's raw-VORP fallback.
    """

    availability: PlayerAvailability
    n_sims: int
    base_seed: int
    survival: SurvivalModel
    top_k: int = 8

    def __post_init__(self) -> None:
        if self.n_sims < 1:
            raise ValueError(f"n_sims must be >= 1; got {self.n_sims}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1; got {self.top_k}")

    def recommend(
        self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig
    ) -> pd.DataFrame:
        df, elig = _eligible_subset(state, pool, config)

        my_ids = {str(g) for g in state.my_pick_ids}
        pool_ids = pool["gsis_id"].astype(str)
        base_roster = pool.loc[
            pool_ids.isin(my_ids), ["gsis_id", "position", "season_mean_fpts"]
        ].copy()
        missing = sorted(my_ids - set(pool_ids))
        if missing:
            warnings.warn(
                f"{len(missing)} rostered player(s) absent from the VORP pool; "
                f"excluded from season valuation: {missing}",
                stacklevel=2,
            )

        pruned = (
            df.sort_values(["position", "vorp"], ascending=[True, False])
            .groupby("position", sort=False)
            .head(self.top_k)
        )
        rng = np.random.default_rng([self.base_seed, state.current_pick])
        marginals = marginal_season_values(
            base_roster,
            pruned[["gsis_id", "position", "season_mean_fpts"]],
            config.roster_slots,
            self.availability,
            n_sims=self.n_sims,
            rng=rng,
        )

        out = df.copy()
        # marginal per row: evaluated candidates carry their real marginal, the pruned
        # tail gets 0.0 (cosmetic — never the argmax; spec §3).
        out["score"] = out["gsis_id"].astype(str).map(marginals).fillna(0.0).astype(float)

        next_pick = my_next_pick(state.current_pick, state.my_slot, state.n_teams, state.rounds)
        if next_pick is None:
            # Last pick: no timing signal -> rank by raw marginal (today's season_value).
            p_na: pd.Series[float] = pd.Series(pd.NA, index=out.index, dtype=pd.Float64Dtype())
            return _finalize(out, elig, p_na, starting_need_tier=False)

        adp = out["consensus_adp"]
        internal_p = adp.map(
            lambda a: self.survival.p_available(
                float(a) if pd.notna(a) else float("nan"), next_pick
            )
        ).astype(float)
        display_p = internal_p.where(adp.notna(), other=pd.NA)

        # opp_cost[pos] = E[best surviving marginal at pos]. Computed over `out`: the
        # tail's marginal is 0 and sorts last (by value desc), so it contributes nothing
        # — equivalent to computing over the pruned set only (spec §3).
        pos = out["position"].to_numpy()
        marg = out["score"].to_numpy(dtype=float)
        p = internal_p.to_numpy(dtype=float)
        gsis = out["gsis_id"].to_numpy()
        opp = expected_best_by_position(pos, marg, p, gsis)
        out["score"] = marg - np.array([opp[pos_i] for pos_i in pos], dtype=float)
        return _finalize(out, elig, display_p, starting_need_tier=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_draft/test_assistant_strategy.py -k timing -q`
Expected: PASS (4 passed). If `test_timing_promotes_scarce_position...` does not flip, widen the ADP gap (e.g. rb1 `consensus_adp=1`, wr `consensus_adp=400`) and/or the season_mean gap so the opp_cost contrast is starker, then re-run — the flip is the behavior under test, not the exact fixture numbers.

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/strategy.py tests/test_draft/test_assistant_strategy.py
git commit -m "feat(draft): SeasonValueTimingStrategy — marginal minus season-value opp cost"
```

---

## Phase 2 — Harness generalization (configurable A/B roles)

### Task 4: `seat_layout` takes A/B labels

**Files:**
- Modify: `src/projections/draft/backtest/draft_field.py:25-38` (`seat_layout`)
- Test: `tests/test_draft/test_backtest/test_draft_field.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_draft/test_backtest/test_draft_field.py`:

```python
def test_seat_layout_defaults_unchanged() -> None:
    # Default labels reproduce the historical nn/sv layout exactly.
    odd = seat_layout(1)
    assert {s for s, lab in odd.items() if lab == "now_or_never"} == {2, 6, 10, 14}
    assert {s for s, lab in odd.items() if lab == "season_value"} == {4, 8, 12, 16}
    even = seat_layout(0)  # paired mirror swaps nn<->sv
    assert {s for s, lab in even.items() if lab == "now_or_never"} == {4, 8, 12, 16}


def test_seat_layout_custom_labels() -> None:
    lay = seat_layout(1, label_a="season_value_timing", label_b="season_value")
    assert {s for s, lab in lay.items() if lab == "season_value_timing"} == {2, 6, 10, 14}
    assert {s for s, lab in lay.items() if lab == "season_value"} == {4, 8, 12, 16}
    assert {s for s, lab in lay.items() if lab == "bot"} == {1, 3, 5, 7, 9, 11, 13, 15}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_draft/test_backtest/test_draft_field.py -k seat_layout -q`
Expected: FAIL — `test_seat_layout_custom_labels` errors (`seat_layout()` takes 1 positional arg).

- [ ] **Step 3: Generalize `seat_layout`**

Replace `seat_layout` in `draft_field.py` with:

```python
def seat_layout(
    seed: int, label_a: str = "now_or_never", label_b: str = "season_value"
) -> dict[int, str]:
    """Return a {seat: strategy_label} map for a 16-team snake draft.

    Odd seeds: role A at {2,6,10,14}, role B at {4,8,12,16}. Even seeds mirror
    (A<->B swap) so exposures cancel when summed over paired seeds. The other 8
    seats are bots. Defaults reproduce the historical now_or_never / season_value
    field byte-identically.
    """
    a, b = {2, 6, 10, 14}, {4, 8, 12, 16}
    if seed % 2 == 0:  # paired mirror
        a, b = b, a
    return {s: (label_a if s in a else label_b if s in b else "bot") for s in range(1, 17)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_draft/test_backtest/test_draft_field.py -q`
Expected: PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/backtest/draft_field.py tests/test_draft/test_backtest/test_draft_field.py
git commit -m "feat(backtest): seat_layout takes configurable A/B strategy labels"
```

---

### Task 5: `collect_results` A/B keys + `aggregate` dynamic labels

**Files:**
- Modify: `src/projections/draft/backtest/harness.py`
- Test: `tests/test_draft/test_backtest/test_harness.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_draft/test_backtest/test_harness.py` (reuses `_cfg16`, `_synthetic_pool`, `stub_availability`, `Calendar`):

```python
def test_collect_results_ab_roles_relabel() -> None:
    cfg, pool = _cfg16(), _synthetic_pool(n_per_pos=60)
    cal = Calendar(regular_weeks=tuple(range(1, 6)), playoff_weeks=(6, 7, 8), playoff_size=6)
    proj = {
        (g, wk): float(m)
        for g, m in zip(pool["gsis_id"], pool["season_mean_fpts"], strict=False)
        for wk in range(1, 9)
    }
    a, _p = collect_results(
        seed_lo=0,
        seed_hi=2,
        pool=pool,
        config=cfg,
        availability=stub_availability(pool),
        proj_lookup=proj,
        actual_lookup=dict(proj),
        calendar=cal,
        jitter=8.0,
        strategy_n_sims=5,
        base_seed=0,
        strategy_a="season_value_timing",
        strategy_b="season_value",
    )
    labels = {r.strategy for r in a}
    assert labels == {"season_value_timing", "season_value", "bot"}
    res = aggregate(a, _p, n_seeds=2, base_seed=0)
    assert set(res.by_strategy_actual) == {"season_value_timing", "season_value", "bot"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_draft/test_backtest/test_harness.py -k ab_roles -q`
Expected: FAIL — `collect_results()` got an unexpected keyword `strategy_a`.

- [ ] **Step 3: Add the strategy registry and thread A/B through `collect_results`**

In `harness.py`, add `SeasonValueTimingStrategy` and `RawVorpStrategy` to the strategy import, then add this builder above `collect_results`:

```python
def _build_strategy(
    key: str,
    *,
    availability: PlayerAvailability,
    n_teams: int,
    strategy_n_sims: int,
    base_seed: int,
) -> DraftStrategy | None:
    """Construct a strategy by key from the inputs the harness already has."""
    if key == "bot":
        return None
    if key == "raw_vorp":
        return RawVorpStrategy()
    if key == "now_or_never":
        return NowOrNeverStrategy(LogisticSurvival(sigma=default_sigma(n_teams)))
    if key == "season_value":
        return SeasonValueStrategy(availability, n_sims=strategy_n_sims, base_seed=base_seed)
    if key == "season_value_timing":
        return SeasonValueTimingStrategy(
            availability,
            n_sims=strategy_n_sims,
            base_seed=base_seed,
            survival=LogisticSurvival(sigma=default_sigma(n_teams)),
        )
    raise ValueError(f"unknown strategy key {key!r}")
```

In `collect_results`, add two parameters (after `base_seed`):

```python
    strategy_a: str = "now_or_never",
    strategy_b: str = "season_value",
```

Replace the `sigma`/`nn`/`sv`/`label_to_strategy` construction and the `seat_layout(s)` call with:

```python
    if strategy_a == strategy_b:
        raise ValueError(f"strategy_a and strategy_b must differ; both were {strategy_a!r}")
    label_to_strategy: dict[str, DraftStrategy | None] = {
        strategy_a: _build_strategy(
            strategy_a,
            availability=availability,
            n_teams=config.n_teams,
            strategy_n_sims=strategy_n_sims,
            base_seed=base_seed,
        ),
        strategy_b: _build_strategy(
            strategy_b,
            availability=availability,
            n_teams=config.n_teams,
            strategy_n_sims=strategy_n_sims,
            base_seed=base_seed,
        ),
        "bot": None,
    }
    results_actual: list[LeagueResult] = []
    results_projected: list[LeagueResult] = []
    for s in range(seed_lo, seed_hi):
        layout = seat_layout(s, strategy_a, strategy_b)
        seat_strategies = {seat: label_to_strategy[label] for seat, label in layout.items()}
```

(Keep the rest of the loop body — the `simulate_league(...)` call and the `results_actual += ... / results_projected += ...` — unchanged.)

- [ ] **Step 4: Make `aggregate` derive labels dynamically**

In `aggregate`, replace `labels = ("now_or_never", "season_value", "bot")` with:

```python
    labels = sorted({r.strategy for r in results_actual})
```

- [ ] **Step 5: Thread A/B through `run_backtest`**

In `run_backtest`, add the same two parameters (after `base_seed`):

```python
    strategy_a: str = "now_or_never",
    strategy_b: str = "season_value",
```

and pass them into its `collect_results(...)` call (`strategy_a=strategy_a, strategy_b=strategy_b`).

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_draft/test_backtest/test_harness.py -q`
Expected: PASS — the new `ab_roles` test plus **all existing harness tests** (default `(nn, sv)` is byte-identical, including `test_chunked_collection_matches_monolithic` and the seat-weighted-champion identity).

- [ ] **Step 7: Commit**

```bash
git add src/projections/draft/backtest/harness.py tests/test_draft/test_backtest/test_harness.py
git commit -m "feat(backtest): configurable A/B strategy roles + dynamic aggregate labels"
```

---

### Task 6: Harness CLI `--strategy-a` / `--strategy-b`

**Files:**
- Modify: `src/projections/draft/backtest/cli.py`, `scripts/h2h_backtest_chunked.py`
- Test: `tests/test_draft/test_backtest/test_cli.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_draft/test_backtest/test_cli.py`:

```python
def test_arg_strategy_ab_defaults_and_override() -> None:
    args = _parse_args(["--league-config", "x.json"])
    assert args.strategy_a == "now_or_never"
    assert args.strategy_b == "season_value"
    args2 = _parse_args(
        ["--league-config", "x.json", "--strategy-a", "season_value_timing", "--strategy-b", "season_value"]
    )
    assert args2.strategy_a == "season_value_timing"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_draft/test_backtest/test_cli.py -k strategy_ab -q`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'strategy_a'`.

- [ ] **Step 3: Add the flags to `backtest/cli.py`**

In `src/projections/draft/backtest/cli.py` `_parse_args`, add (before `return p.parse_args(argv)`):

```python
    _STRATEGY_KEYS = ["now_or_never", "season_value", "season_value_timing", "raw_vorp"]
    p.add_argument("--strategy-a", choices=_STRATEGY_KEYS, default="now_or_never")
    p.add_argument("--strategy-b", choices=_STRATEGY_KEYS, default="season_value")
```

In `run(...)`, pass them into `run_backtest(...)`:

```python
        strategy_a=args.strategy_a,
        strategy_b=args.strategy_b,
```

- [ ] **Step 4: Add the flags to the chunked runner**

In `scripts/h2h_backtest_chunked.py` `_parse_args`, add (with the same key list):

```python
    keys = ["now_or_never", "season_value", "season_value_timing", "raw_vorp"]
    p.add_argument("--strategy-a", choices=keys, default="now_or_never")
    p.add_argument("--strategy-b", choices=keys, default="season_value")
```

In `_run_worker`, pass them into `collect_results(...)`:

```python
        strategy_a=args.strategy_a,
        strategy_b=args.strategy_b,
```

In `_run_driver`, add them to the worker subprocess `cmd` list (alongside `--strategy-n-sims`):

```python
            "--strategy-a", args.strategy_a, "--strategy-b", args.strategy_b,
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_draft/test_backtest/test_cli.py -q`
Expected: PASS (existing + new).

- [ ] **Step 6: Commit**

```bash
git add src/projections/draft/backtest/cli.py scripts/h2h_backtest_chunked.py tests/test_draft/test_backtest/test_cli.py
git commit -m "feat(backtest): --strategy-a/--strategy-b on both H2H runners"
```

---

## Phase 3 — Live assistant CLI

### Task 7: `--strategy season_value_timing` in the live assistant

**Files:**
- Modify: `src/projections/draft/assistant/cli.py`
- Test: `tests/test_draft/test_assistant_cli.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_draft/test_assistant_cli.py`:

```python
def test_parse_args_accepts_season_value_timing() -> None:
    from projections.draft.assistant.cli import _parse_args

    args = _parse_args(["--state", "s.json", "--vorp-table", "v.parquet", "--strategy", "season_value_timing"])
    assert args.strategy == "season_value_timing"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_draft/test_assistant_cli.py -k season_value_timing -q`
Expected: FAIL — argparse rejects the choice (`invalid choice: 'season_value_timing'`).

- [ ] **Step 3: Wire the strategy**

In `src/projections/draft/assistant/cli.py`:

Add the import:

```python
from projections.draft.assistant.strategy import (
    DraftStrategy,
    NowOrNeverStrategy,
    RawVorpStrategy,
    SeasonValueStrategy,
    SeasonValueTimingStrategy,
)
```

Add `"season_value_timing"` to the `--strategy` `choices` list in `_parse_args`.

In `generate_recommendation`, replace the `if strategy_name == "season_value": ... else: ...` block with:

```python
    strategy: DraftStrategy
    if strategy_name == "season_value":
        availability = load_store_availability(vorp, season=season, data_root=data_root)
        strategy = SeasonValueStrategy(availability, n_sims=n_sims, base_seed=0)
    elif strategy_name == "season_value_timing":
        availability = load_store_availability(vorp, season=season, data_root=data_root)
        spread = default_sigma(league.n_teams) if sigma is None else sigma
        strategy = SeasonValueTimingStrategy(
            availability, n_sims=n_sims, base_seed=0, survival=LogisticSurvival(sigma=spread)
        )
    else:
        strategy = _build_strategy(strategy_name, league.n_teams, sigma)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_draft/test_assistant_cli.py -q`
Expected: PASS (existing + new).

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/cli.py tests/test_draft/test_assistant_cli.py
git commit -m "feat(draft): --strategy season_value_timing in the live assistant CLI"
```

---

## Phase 4 — Run + report (execution-time, not new code)

### Task 8: Run the H2H backtest for `season_value_timing` and log a test entry

**Files:**
- Modify: `reports/draft_strategy_tests.md`

- [ ] **Step 1: Full gate before running**

Run, fixing any failure before proceeding:
```bash
python -m pytest tests/test_draft -n0 -q
python -m mypy src tests
python -m ruff check src tests && python -m ruff format --check src tests
```
Expected: tests pass (serial `-n0` avoids the known xdist native-crash flake), mypy clean, ruff clean.

- [ ] **Step 2: Run `season_value_timing` vs `season_value` for 2025 and 2024**

Use the resumable runner in PowerShell (the native-crash mitigation), one fresh checkpoint dir per run:
```
$env:OMP_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1; $env:KMP_DUPLICATE_LIB_OK="TRUE"
python scripts/h2h_backtest_chunked.py --season 2025 --league-config configs/league_espn_half_16team.json `
  --n-seeds 200 --strategy-n-sims 200 --jitter 8 --chunk-size 5 --max-retries 15 `
  --strategy-a season_value_timing --strategy-b season_value `
  --checkpoint-dir _h2h_ckpt_timing_2025 --data-root data
python scripts/h2h_backtest_chunked.py --season 2024 --league-config configs/league_espn_half_16team.json `
  --n-seeds 200 --strategy-n-sims 200 --jitter 8 --chunk-size 5 --max-retries 15 `
  --strategy-a season_value_timing --strategy-b season_value `
  --checkpoint-dir _h2h_ckpt_timing_2024 --data-root data
```

- [ ] **Step 3: Paired-bootstrap analysis**

Reuse the `_diag_noise.py` analysis pattern (per-seed `season_value_timing − season_value` and each vs `bot`, win%/playoff%/champ%, ACTUAL + PROJECTED) on each season's checkpoint dir.

- [ ] **Step 4: Record a new Test entry in `reports/draft_strategy_tests.md`**

Add a "Test N — season_value_timing vs season_value (H2H, 2024 + 2025)" entry: setup (mirror-paired A/B field, same config), the per-strategy and paired-diff tables for both seasons, and **what it favors in isolation**. **No verdict** — per the standing process rule, the single decision is at the end of the investigation. Update the standing tally row.

- [ ] **Step 5: Commit**

```bash
git add reports/draft_strategy_tests.md
git commit -m "docs(backtest): Test N — season_value_timing H2H results (2024 + 2025)"
```

---

## Self-Review

**Spec coverage:** §3 strategy → Tasks 1–3 (helper, nn refactor, class); §4 shared helper + nn bit-identical guard → Tasks 1–2 (Task 2 Step 3 names the duplicate-inline fallback); §5 harness A/B (registry, construction, seat_layout, dynamic labels, CLI flags, default byte-identical) → Tasks 4–6; §6 live CLI → Task 7; §7 testing (helper hand-computed, determinism, fallback, reorder flip, score==marginal−opp_cost, `__post_init__` validation, default byte-identical, A/B relabel, CLI parse) → distributed across Tasks 1–7; §8 validation/log (no verdict) → Task 8. All covered.

**Placeholder scan:** every code step shows complete code; the one fixture-tuning note (Task 3 Step 4) is the behavior-under-test, not a placeholder. No TODO/TBD.

**Type consistency:** `expected_best_by_position(positions, values, probs, tiebreak)` is called identically in Task 2 (nn) and Task 3 (timing). `SeasonValueTimingStrategy(availability, n_sims, base_seed, survival, top_k=8)` matches construction in Tasks 3, 5 (`_build_strategy`), and 7 (CLI). `collect_results(..., strategy_a, strategy_b)` matches the CLI/worker call sites in Task 6. `seat_layout(seed, label_a, label_b)` matches its caller in Task 5.
