# Depth-Aware Draft Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `SeasonValueStrategy`, a draft strategy that ranks each available player by the marginal expected season points it adds to the hero's current roster — the first strategy that drafts to PR #60's risk-aware season metric.

**Architecture:** The strategy scores a candidate by `V(my_roster + candidate) − V(my_roster)` where `V` is the season metric (`expected_season_points`). The marginal is a difference of two Monte-Carlo estimates, so all evaluations within one `recommend()` share **one pre-drawn per-player availability matrix** (common random numbers) keyed by `gsis_id`, making the small depth signal low-variance. Candidates are pruned to the top-`top_k`-by-VORP per position (deeper ones add ≈0). The strategy reuses the existing greedy lineup fill (`optimal_lineup_points`) and the clean-week/bye-week factorization; the one new mechanism is the CRN evaluation. Both CLIs construct the strategy's `PlayerAvailability` through one shared store loader.

**Tech Stack:** Python 3.12, pandas (pyarrow-backed strings, `Float64`/`Int64` extension dtypes), numpy, pandera (strict schemas), pytest, mypy strict, ruff.

**Source spec:** `docs/superpowers/specs/2026-06-11-depth-aware-draft-strategy-design.md`

---

## File Structure

- `src/projections/draft/assistant/state.py` — **modify**: add `DraftState.my_pick_ids` (my drafted gsis_ids via snake order).
- `src/projections/draft/assistant/season_value.py` — **modify**: extract a shared `_week_value` + `_factorized_season_value` kernel (behavior-preserving), then add `expected_season_points_crn` (CRN evaluation) and `marginal_season_values` (per-candidate marginal under shared draws).
- `src/projections/draft/assistant/strategy.py` — **modify**: make `_finalize`'s starting-need tier optional; add `SeasonValueStrategy`.
- `src/projections/draft/assistant/availability_loader.py` — **create**: `load_store_availability(pool, *, season, data_root) -> PlayerAvailability` (the store-read + `build_availability`, factored out of `tournament_cli._build_season_valuer`).
- `src/projections/draft/assistant/tournament_cli.py` — **modify**: use the shared loader; register `season_value` as a compare-mode opt-in.
- `src/projections/draft/assistant/cli.py` — **modify**: add `--season`/`--data-root`/`--n-sims`; build `SeasonValueStrategy`; add `season_value` to `--strategy` choices.
- `reports/depth_aware_strategy_validation_2026.md` — **create** (Task 10): validation writeup.
- Tests: `tests/test_draft/test_assistant_state.py`, `test_assistant_season_value.py`, `test_assistant_strategy.py`, `test_assistant_tournament_cli.py`, `test_assistant_cli.py` (all **modify/extend**).

Each task is independently committable and leaves the suite green.

---

### Task 1: `DraftState.my_pick_ids` — my drafted gsis_ids via snake order

`DraftState` stores `picks` (everyone's drafted ids) and `my_roster` (only the *positions* of my picks). The strategy needs the *gsis_ids* of my picks to build my roster's rows from the pool. Derive them the same way `load_draft_state` derives `my_roster`: a pick is mine iff `slot_for(pick_number, n_teams) == my_slot`.

**Files:**
- Modify: `src/projections/draft/assistant/state.py`
- Test: `tests/test_draft/test_assistant_state.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_draft/test_assistant_state.py`:

```python
def test_my_pick_ids_picks_out_my_snake_slots() -> None:
    from projections.draft.assistant.state import DraftState
    from projections.schemas import GsisId, Position

    # 4 teams, my_slot=1 → my picks are #1, #8, #9 (snake).
    picks = tuple(
        GsisId(g)
        for g in [
            "00-0000001",  # #1 mine
            "00-0000002",
            "00-0000003",
            "00-0000004",
            "00-0000005",
            "00-0000006",
            "00-0000007",
            "00-0000008",  # #8 mine
            "00-0000009",  # #9 mine
        ]
    )
    state = DraftState(
        my_slot=1,
        n_teams=4,
        rounds=5,
        picks=picks,
        my_roster=(Position.RB, Position.WR, Position.RB),
    )
    assert state.my_pick_ids == ("00-0000001", "00-0000008", "00-0000009")


def test_my_pick_ids_empty_when_no_picks() -> None:
    from projections.draft.assistant.state import DraftState

    state = DraftState(my_slot=1, n_teams=4, rounds=5, picks=(), my_roster=())
    assert state.my_pick_ids == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_draft/test_assistant_state.py::test_my_pick_ids_picks_out_my_snake_slots -v`
Expected: FAIL with `AttributeError: 'DraftState' object has no attribute 'my_pick_ids'`.

- [ ] **Step 3: Implement the property**

In `src/projections/draft/assistant/state.py`, add a property to `DraftState` (just after `current_pick`):

```python
    @property
    def my_pick_ids(self) -> tuple[GsisId, ...]:
        """The gsis_ids of *my* picks (snake-slot-derived), in pick order.

        Mirrors load_draft_state's roster derivation: pick #k is mine iff its
        snake slot equals my_slot. Parallel to my_roster (positions) but ids.
        """
        return tuple(
            gid
            for index, gid in enumerate(self.picks)
            if slot_for(index + 1, self.n_teams) == self.my_slot
        )
```

`slot_for` is already imported in `state.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_draft/test_assistant_state.py -v`
Expected: PASS (all, including the two new tests).

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/state.py tests/test_draft/test_assistant_state.py
git commit -m "feat(draft): DraftState.my_pick_ids — my drafted ids via snake order"
```

---

### Task 2: Refactor `season_value.py` to a shared, behavior-preserving kernel

Extract the per-week scoring and the clean-week/bye-week factorization into helpers, and rewrite `expected_season_points` to use them. This must be **bit-for-bit behavior-preserving** so the existing season-value tests stay green: `rng.random(n)` called `n_sims` times consumes the bit stream identically to the loop we keep, and the clean-week-first / bye-weeks-sorted call order is unchanged.

**Files:**
- Modify: `src/projections/draft/assistant/season_value.py`
- Test: `tests/test_draft/test_assistant_season_value.py` (existing tests are the regression guard — no new test this task)

- [ ] **Step 1: Run the existing season-value tests to confirm the green baseline**

Run: `pytest tests/test_draft/test_assistant_season_value.py -v`
Expected: PASS (6 tests). These pin the behavior we must preserve.

- [ ] **Step 2: Refactor `expected_season_points` onto the shared kernel**

Replace the body of `src/projections/draft/assistant/season_value.py` from the `expected_season_points` definition downward with:

```python
def _week_value(
    roster: pd.DataFrame, roster_slots: Mapping[RosterSlot, int], available: np.ndarray
) -> float:
    """Optimal weekly lineup points from the available roster rows (UNSCALED).

    The /_GAMES per-game scaling is applied once by the caller (after averaging over
    sims), exactly as the original expected_season_points did — summing pre-scaled
    values per sim would be a different float expression and break exact-equality tests.
    """
    sub = roster.iloc[np.flatnonzero(available)]
    return optimal_lineup_points(sub, roster_slots)


def _factorized_season_value(
    roster: pd.DataFrame,
    availability: PlayerAvailability,
    weeks: Iterable[int],
    week_value_fn: Callable[[np.ndarray], float],
) -> float:
    """Sum the season via the single-week factorization (spec §3.4 of PR #60).

    `week_value_fn(forced_out)` returns E[week points | these roster indices are
    forced out (bye)]. Every non-bye week shares one expectation; each distinct
    roster bye week is recomputed with that player forced out. Exact in
    expectation. Call order (clean week, then bye weeks ascending) is fixed so
    callers that advance a shared RNG inside week_value_fn stay reproducible.
    """
    n = len(roster)
    gsis = roster["gsis_id"].astype(str).to_numpy()
    bye_arr = np.array(
        [b if (b := availability.bye_week(g)) is not None else -1 for g in gsis]
    )
    weeks = list(weeks)
    roster_bye_weeks = sorted({w for w in bye_arr.tolist() if w in weeks})

    clean = week_value_fn(np.zeros(n, dtype=bool))
    total = (len(weeks) - len(roster_bye_weeks)) * clean
    for w in roster_bye_weeks:
        total += week_value_fn(bye_arr == w)
    return total


def expected_season_points(
    roster: pd.DataFrame,
    roster_slots: Mapping[RosterSlot, int],
    availability: PlayerAvailability,
    *,
    n_sims: int,
    rng: np.random.Generator,
    weeks: Iterable[int] = range(1, 18),
) -> float:
    """Expected total season points of `roster` under availability risk."""
    n = len(roster)
    if n == 0:
        return 0.0
    gsis = roster["gsis_id"].astype(str).to_numpy()
    p_arr = np.array([availability.p_week(g) for g in gsis], dtype=np.float64)

    def week_value_fn(forced_out: np.ndarray) -> float:
        acc = 0.0
        for _ in range(n_sims):
            available = (rng.random(n) < p_arr) & ~forced_out
            acc += _week_value(roster, roster_slots, available)
        return acc / n_sims / _GAMES

    return _factorized_season_value(roster, availability, weeks, week_value_fn)
```

- [ ] **Step 3: Update the imports at the top of `season_value.py`**

Ensure the `collections.abc` import line reads:

```python
from collections.abc import Callable, Iterable, Mapping
```

(Add `Callable` to the existing `Iterable, Mapping` import.)

- [ ] **Step 4: Run the existing tests to verify behavior is preserved**

Run: `pytest tests/test_draft/test_assistant_season_value.py -v`
Expected: PASS (all 6 — including `test_determinism`'s exact `a == b` and the brute-force factorization test).

- [ ] **Step 5: Run the season-value consumers to confirm no regression**

Run: `pytest tests/test_draft/test_assistant_valuer.py tests/test_draft/test_assistant_tournament_cli.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/projections/draft/assistant/season_value.py
git commit -m "refactor(draft): extract shared week/factorization kernel in season_value"
```

---

### Task 3: `expected_season_points_crn` — CRN evaluation over a shared draw matrix

Add a CRN variant that takes a pre-drawn `(n_sims × universe)` uniform matrix and a `gsis_id → column` map, so the base roster and every candidate roster share availability draws.

**Files:**
- Modify: `src/projections/draft/assistant/season_value.py`
- Test: `tests/test_draft/test_assistant_season_value.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_draft/test_assistant_season_value.py` (the helpers `_roster`, `_avail` already exist in the file):

```python
def test_crn_matches_expected_season_points_no_bye_exact() -> None:
    # With identity column mapping, a no-bye roster, and the same seed, the CRN
    # kernel is BIT-IDENTICAL to expected_season_points (one rng.random((n_sims,n))
    # equals n_sims successive rng.random(n) draws). Guards column alignment + that
    # CRN reuses the same kernel — i.e. changes variance, not the mean.
    from projections.draft.assistant.season_value import expected_season_points_crn

    roster = _roster(
        [("00-0000001", "RB", 200.0), ("00-0000002", "WR", 180.0), ("00-0000003", "RB", 120.0)]
    )
    slots = {RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.FLEX: 1}
    avail = _avail({"00-0000001": 0.7, "00-0000002": 0.8, "00-0000003": 0.6})
    n_sims = 200
    col_of = {"00-0000001": 0, "00-0000002": 1, "00-0000003": 2}  # roster order

    draws = np.random.default_rng(11).random((n_sims, 3))
    crn = expected_season_points_crn(
        roster, slots, avail, draws=draws, col_of=col_of, weeks=range(1, 18)
    )
    esp = expected_season_points(
        roster, slots, avail, n_sims=n_sims, rng=np.random.default_rng(11), weeks=range(1, 18)
    )
    assert crn == esp


def test_crn_matches_expected_season_points_with_bye_in_expectation() -> None:
    # With a bye, CRN reuses the shared matrix across weeks while expected_season_points
    # draws fresh per week, so they are NOT bit-equal — but equal IN EXPECTATION.
    # Regression guard for the bye handling of the CRN kernel (spec §4, finding #2).
    from projections.draft.assistant.season_value import expected_season_points_crn

    roster = _roster([("00-0000001", "RB", 200.0), ("00-0000002", "WR", 150.0)])
    slots = {RosterSlot.RB: 1, RosterSlot.WR: 1}
    avail = _avail({"00-0000001": 0.85, "00-0000002": 0.85}, bye={"00-0000001": 3})
    n_sims = 4000
    col_of = {"00-0000001": 0, "00-0000002": 1}

    draws = np.random.default_rng(3).random((n_sims, 2))
    crn = expected_season_points_crn(
        roster, slots, avail, draws=draws, col_of=col_of, weeks=range(1, 6)
    )
    esp = expected_season_points(
        roster, slots, avail, n_sims=n_sims, rng=np.random.default_rng(7), weeks=range(1, 6)
    )
    assert abs(crn - esp) / esp < 0.02  # same expectation, independent MC noise


def test_crn_column_selection_is_by_gsis_not_position() -> None:
    # A universe wider than the roster, with non-identity columns: the kernel must
    # pull each player's OWN column. Reordering the universe must not change the value.
    from projections.draft.assistant.season_value import expected_season_points_crn

    roster = _roster([("00-0000002", "RB", 200.0), ("00-0000004", "RB", 120.0)])
    slots = {RosterSlot.RB: 1, RosterSlot.FLEX: 1}
    avail = _avail({"00-0000002": 0.7, "00-0000004": 0.6})
    universe = ["00-0000001", "00-0000002", "00-0000003", "00-0000004"]
    col_of = {g: i for i, g in enumerate(universe)}
    draws = np.random.default_rng(5).random((300, len(universe)))
    val = expected_season_points_crn(roster, slots, avail, draws=draws, col_of=col_of)
    # Empty roster short-circuits to 0.0 — sanity that a real roster does not.
    assert val > 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_draft/test_assistant_season_value.py::test_crn_matches_expected_season_points_no_bye_exact -v`
Expected: FAIL with `ImportError: cannot import name 'expected_season_points_crn'`.

- [ ] **Step 3: Implement `expected_season_points_crn`**

Append to `src/projections/draft/assistant/season_value.py`:

```python
def expected_season_points_crn(
    roster: pd.DataFrame,
    roster_slots: Mapping[RosterSlot, int],
    availability: PlayerAvailability,
    *,
    draws: np.ndarray,
    col_of: Mapping[str, int],
    weeks: Iterable[int] = range(1, 18),
) -> float:
    """Expected season points using a shared pre-drawn availability matrix (CRN).

    `draws` is `(n_sims, universe)` uniforms; `col_of` maps gsis_id -> column.
    Every roster scored against the same `draws` shares per-player draws, so a
    marginal `V(R+c) - V(R)` cancels the common noise (spec §3.3).
    """
    n = len(roster)
    if n == 0:
        return 0.0
    gsis = roster["gsis_id"].astype(str).to_numpy()
    p_arr = np.array([availability.p_week(g) for g in gsis], dtype=np.float64)
    cols = np.array([col_of[g] for g in gsis])
    sub_draws = draws[:, cols]  # (n_sims, n), aligned to roster row order
    n_sims = sub_draws.shape[0]

    def week_value_fn(forced_out: np.ndarray) -> float:
        acc = 0.0
        for s in range(n_sims):
            available = (sub_draws[s] < p_arr) & ~forced_out
            acc += _week_value(roster, roster_slots, available)
        return acc / n_sims / _GAMES

    return _factorized_season_value(roster, availability, weeks, week_value_fn)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_draft/test_assistant_season_value.py -v`
Expected: PASS (all, including the 3 new CRN tests).

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/season_value.py tests/test_draft/test_assistant_season_value.py
git commit -m "feat(draft): expected_season_points_crn — shared-draw season value"
```

---

### Task 4: `marginal_season_values` — per-candidate CRN marginal

Add the function that the strategy calls: given a base roster and a candidate frame, return `{candidate gsis_id: marginal expected season points}`, all under one shared draw matrix.

**Files:**
- Modify: `src/projections/draft/assistant/season_value.py`
- Test: `tests/test_draft/test_assistant_season_value.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_draft/test_assistant_season_value.py`:

```python
def test_marginal_matches_closed_form_insurance() -> None:
    # Base = one risky starter S(p=0.6). Candidate backup B(120, p=0.7), {RB:1}, 1 week.
    # Marginal = points B adds = the insurance term only: (1-p_s)*p_b*B / 17.
    from projections.draft.assistant.season_value import marginal_season_values

    base = _roster([("00-0000001", "RB", 200.0)])
    cands = _roster([("00-0000002", "RB", 120.0)])
    avail = _avail({"00-0000001": 0.6, "00-0000002": 0.7})
    expected = 0.4 * 0.7 * 120 / 17  # ≈ 1.976
    out = marginal_season_values(
        base,
        cands,
        {RosterSlot.RB: 1},
        avail,
        n_sims=8000,
        rng=np.random.default_rng(0),
        weeks=range(1, 2),
    )
    assert abs(out["00-0000002"] - expected) < 0.1


def test_marginal_is_low_variance_under_crn() -> None:
    # CRN makes the marginal stable even at small n_sims: 60 vs 400 sims agree tightly,
    # where an independent-seed difference of the two MC estimates would not.
    from projections.draft.assistant.season_value import marginal_season_values

    base = _roster([("00-0000001", "RB", 200.0)])
    cands = _roster([("00-0000002", "RB", 150.0)])
    avail = _avail({"00-0000001": 0.55, "00-0000002": 0.8})
    lo = marginal_season_values(
        base, cands, {RosterSlot.RB: 1}, avail, n_sims=60, rng=np.random.default_rng(1)
    )["00-0000002"]
    hi = marginal_season_values(
        base, cands, {RosterSlot.RB: 1}, avail, n_sims=400, rng=np.random.default_rng(2)
    )["00-0000002"]
    assert lo > 0.0
    assert abs(lo - hi) < 0.5  # CRN: small-n already close to the large-n estimate


def test_marginal_empty_base_is_solo_value() -> None:
    # With an empty base roster (first pick), the marginal is the candidate's own
    # expected season points (positive).
    from projections.draft.assistant.season_value import marginal_season_values

    base = _roster([])
    cands = _roster([("00-0000002", "RB", 180.0)])
    avail = _avail({"00-0000002": 0.9})
    out = marginal_season_values(
        base, cands, {RosterSlot.RB: 1}, avail, n_sims=300, rng=np.random.default_rng(0)
    )
    assert out["00-0000002"] > 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_draft/test_assistant_season_value.py::test_marginal_matches_closed_form_insurance -v`
Expected: FAIL with `ImportError: cannot import name 'marginal_season_values'`.

- [ ] **Step 3: Implement `marginal_season_values`**

Append to `src/projections/draft/assistant/season_value.py`:

```python
def marginal_season_values(
    base_roster: pd.DataFrame,
    candidates: pd.DataFrame,
    roster_slots: Mapping[RosterSlot, int],
    availability: PlayerAvailability,
    *,
    n_sims: int,
    rng: np.random.Generator,
    weeks: Iterable[int] = range(1, 18),
) -> dict[str, float]:
    """CRN marginal expected-season-points of adding each candidate to `base_roster`.

    Returns {candidate gsis_id: V(base + candidate) - V(base)}. All evaluations
    (base and every candidate) share one pre-drawn availability matrix over the
    union of base + candidate ids, so the marginal isolates the candidate's own
    contribution at low variance (spec §3.3). `base_roster` and `candidates` each
    carry `gsis_id`, `position`, `season_mean_fpts`.
    """
    base_ids = [str(g) for g in base_roster["gsis_id"]]
    cand_ids = [str(g) for g in candidates["gsis_id"]]
    universe = sorted(set(base_ids) | set(cand_ids))
    col_of = {g: i for i, g in enumerate(universe)}
    draws = rng.random((n_sims, len(universe)))

    base_val = expected_season_points_crn(
        base_roster, roster_slots, availability, draws=draws, col_of=col_of, weeks=weeks
    )
    out: dict[str, float] = {}
    for i in range(len(candidates)):
        cand_row = candidates.iloc[[i]]
        cand_roster = pd.concat([base_roster, cand_row], ignore_index=True)
        val = expected_season_points_crn(
            cand_roster, roster_slots, availability, draws=draws, col_of=col_of, weeks=weeks
        )
        out[str(cand_row["gsis_id"].iloc[0])] = val - base_val
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_draft/test_assistant_season_value.py -v`
Expected: PASS (all, including the 3 new marginal tests).

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/season_value.py tests/test_draft/test_assistant_season_value.py
git commit -m "feat(draft): marginal_season_values — CRN per-candidate marginal"
```

---

### Task 5: Make `_finalize`'s starting-need tier optional

`SeasonValueStrategy` must rank purely by marginal `score`, not by the `fills_starting_slot` hard tier (the season metric already values open slots — §3.4). Add a flag; existing callers (`now_or_never`, `raw_vorp`) keep the tier and stay byte-identical.

**Files:**
- Modify: `src/projections/draft/assistant/strategy.py`
- Test: `tests/test_draft/test_assistant_strategy.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_draft/test_assistant_strategy.py`:

```python
def test_finalize_without_starting_tier_orders_by_score() -> None:
    # With starting_need_tier=False, a higher-score row outranks a starting-slot
    # filler with a lower score (the tier is NOT a sort key). fills_starting_slot is
    # still computed/emitted.
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000050", "00-0000051"], dtype=_PYARROW_STR),
            "position": pd.array(["RB", "WR"], dtype=_PYARROW_STR),
            "vorp": [10.0, 20.0],
            "consensus_adp": pd.array([3.0, 4.0], dtype=pd.Float64Dtype()),
            "score": [9.0, 1.0],  # RB has the higher score
        }
    )
    p_na: pd.Series[float] = pd.Series(pd.NA, index=df.index, dtype=pd.Float64Dtype())
    out = _finalize(df, {Position.RB: True, Position.WR: True}, p_na, starting_need_tier=False)
    assert list(out["gsis_id"]) == ["00-0000050", "00-0000051"]  # score desc, tier ignored
    assert set(out.columns) >= {"fills_starting_slot"}  # still emitted
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_draft/test_assistant_strategy.py::test_finalize_without_starting_tier_orders_by_score -v`
Expected: FAIL with `TypeError: _finalize() got an unexpected keyword argument 'starting_need_tier'`.

- [ ] **Step 3: Add the parameter to `_finalize`**

In `src/projections/draft/assistant/strategy.py`, change the `_finalize` signature and the `sort_values` block. The signature becomes:

```python
def _finalize(
    df: pd.DataFrame,
    elig: dict[Position, bool],
    p_available: pd.Series[float],
    *,
    starting_need_tier: bool = True,
) -> pd.DataFrame:
```

Replace the existing single `out = out.sort_values(...)` call with:

```python
    if starting_need_tier:
        out = out.sort_values(
            ["fills_starting_slot", "score", "vorp", "gsis_id"],
            ascending=[False, False, False, True],
        )
    else:
        out = out.sort_values(
            ["score", "vorp", "gsis_id"],
            ascending=[False, False, True],
        )
```

Leave the `fills_starting_slot` computation above it unchanged (it is still emitted either way).

- [ ] **Step 4: Run the full strategy test file to confirm no regression**

Run: `pytest tests/test_draft/test_assistant_strategy.py -v`
Expected: PASS (all existing tests — `now_or_never`/`raw_vorp` ordering unchanged — plus the new one).

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/strategy.py tests/test_draft/test_assistant_strategy.py
git commit -m "feat(draft): optional starting-need tier in _finalize"
```

---

### Task 6: `SeasonValueStrategy`

The strategy itself: prune candidates, compute marginal season values under CRN, rank purely by score.

**Files:**
- Modify: `src/projections/draft/assistant/strategy.py`
- Test: `tests/test_draft/test_assistant_strategy.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_draft/test_assistant_strategy.py` (imports `PlayerAvailability`, `warnings`):

```python
def _depth_pool() -> pd.DataFrame:
    # My roster will be {WR_a safe, WR_b safe, RB_a RISKY}. Candidates: WR_c (high VORP,
    # saturated WR room) vs RB_b (insurance for the risky RB). season_value must take
    # the insurance; now_or_never takes the higher VORP.
    return pd.DataFrame(
        {
            "gsis_id": pd.array(
                ["00-0000201", "00-0000202", "00-0000301", "00-0000203", "00-0000302"],
                dtype=_PYARROW_STR,
            ),
            "position": pd.array(["WR", "WR", "RB", "WR", "RB"], dtype=_PYARROW_STR),
            "season_mean_fpts": [200.0, 195.0, 200.0, 190.0, 185.0],
            "vorp": [60.0, 50.0, 58.0, 55.0, 45.0],
            "replacement_fpts": [140.0, 140.0, 140.0, 140.0, 140.0],
            "consensus_adp": pd.array([3.0, 4.0, 5.0, 20.0, 20.0], dtype=pd.Float64Dtype()),
        }
    )


def _depth_state() -> DraftState:
    # 4 teams, my_slot=1 → my picks at #1, #8, #9. Place WR_a, WR_b, RB_a there;
    # fillers (not in pool) elsewhere. current_pick = 10.
    picks = (
        GsisId("00-0000201"),      # #1 mine (WR_a)
        GsisId("00-9000002"),
        GsisId("00-9000003"),
        GsisId("00-9000004"),
        GsisId("00-9000005"),
        GsisId("00-9000006"),
        GsisId("00-9000007"),
        GsisId("00-0000202"),      # #8 mine (WR_b)
        GsisId("00-0000301"),      # #9 mine (RB_a)
    )
    return DraftState(
        my_slot=1,
        n_teams=4,
        rounds=5,  # == roster_size (RB+WR+FLEX+2*BENCH)
        picks=picks,
        my_roster=(Position.WR, Position.WR, Position.RB),
    )


def _depth_config() -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=4,
        roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.FLEX: 1, RosterSlot.BENCH: 2},
        ruleset=Ruleset.espn_ppr(),
    )


def _depth_avail() -> "PlayerAvailability":
    from projections.draft.assistant.availability import PlayerAvailability

    return PlayerAvailability(
        p={
            "00-0000201": 0.97,  # WR_a (safe)
            "00-0000202": 0.97,  # WR_b (safe)
            "00-0000301": 0.40,  # RB_a (risky starter — depth at RB pays off)
            "00-0000203": 0.97,  # WR_c (candidate, redundant WR)
            "00-0000302": 0.95,  # RB_b (candidate, RB insurance)
        },
        bye={},
    )


def test_season_value_satisfies_protocol() -> None:
    from projections.draft.assistant.strategy import SeasonValueStrategy

    strat = SeasonValueStrategy(_depth_avail(), n_sims=200, base_seed=0)
    assert isinstance(strat, DraftStrategy)


def test_season_value_takes_insurance_where_now_or_never_takes_vorp() -> None:
    from projections.draft.assistant.strategy import SeasonValueStrategy

    state, pool, config = _depth_state(), _depth_pool(), _depth_config()
    season = SeasonValueStrategy(_depth_avail(), n_sims=4000, base_seed=0).recommend(
        state, pool, config
    )
    RecommendationSchema.validate(season)
    assert season.iloc[0]["gsis_id"] == "00-0000302"  # RB_b, the RB insurance pick

    non = NowOrNeverStrategy(LogisticSurvival(sigma=8.0)).recommend(state, pool, config)
    assert non.iloc[0]["gsis_id"] == "00-0000203"  # WR_c, the higher-VORP redundant WR


def test_season_value_is_deterministic() -> None:
    from projections.draft.assistant.strategy import SeasonValueStrategy

    state, pool, config = _depth_state(), _depth_pool(), _depth_config()
    a = SeasonValueStrategy(_depth_avail(), n_sims=300, base_seed=7).recommend(state, pool, config)
    b = SeasonValueStrategy(_depth_avail(), n_sims=300, base_seed=7).recommend(state, pool, config)
    assert list(a["gsis_id"]) == list(b["gsis_id"])
    assert list(a["score"]) == list(b["score"])


def test_season_value_pruning_invariance() -> None:
    # top_k larger than the per-position pool depth → identical to no pruning.
    from projections.draft.assistant.strategy import SeasonValueStrategy

    state, pool, config = _depth_state(), _depth_pool(), _depth_config()
    small = SeasonValueStrategy(_depth_avail(), n_sims=500, base_seed=1, top_k=1).recommend(
        state, pool, config
    )
    big = SeasonValueStrategy(_depth_avail(), n_sims=500, base_seed=1, top_k=50).recommend(
        state, pool, config
    )
    # Only one candidate per position here, so even top_k=1 evaluates all → same #1.
    assert small.iloc[0]["gsis_id"] == big.iloc[0]["gsis_id"]


def test_season_value_warns_on_roster_player_missing_from_pool() -> None:
    # A rostered id absent from the pool is dropped from the valued base, with a warning.
    from projections.draft.assistant.strategy import SeasonValueStrategy

    pool = _depth_pool()
    pool = pool[pool["gsis_id"] != "00-0000301"].copy()  # drop rostered RB_a from the pool
    state, config = _depth_state(), _depth_config()
    with pytest.warns(UserWarning, match="absent from the VORP pool"):
        rec = SeasonValueStrategy(_depth_avail(), n_sims=200, base_seed=0).recommend(
            state, pool, config
        )
    RecommendationSchema.validate(rec)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_draft/test_assistant_strategy.py::test_season_value_satisfies_protocol -v`
Expected: FAIL with `ImportError: cannot import name 'SeasonValueStrategy'`.

- [ ] **Step 3: Implement `SeasonValueStrategy`**

In `src/projections/draft/assistant/strategy.py`, add `import warnings` at the top (after `from __future__`), add to the existing imports:

```python
from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.season_value import marginal_season_values
```

Then append the strategy class at the end of the file:

```python
@dataclass(frozen=True)
class SeasonValueStrategy:
    """Depth-aware: rank by marginal expected season points (spec §3.2).

    Scores each candidate by V(my_roster + candidate) - V(my_roster) under common
    random numbers, prunes to top_k-by-VORP per position, ranks purely by that
    marginal (no fills_starting_slot tier — the season metric already values open
    slots). Holds the MC config like NowOrNeverStrategy holds a SurvivalModel.
    """

    availability: PlayerAvailability
    n_sims: int
    base_seed: int
    top_k: int = 8

    def recommend(
        self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig
    ) -> pd.DataFrame:
        df, elig = _eligible_subset(state, pool, config)

        my_ids = {str(g) for g in state.my_pick_ids}
        pool_ids = pool["gsis_id"].astype(str)
        present = pool_ids.isin(my_ids)
        base_roster = pool.loc[present, ["gsis_id", "position", "season_mean_fpts"]].copy()
        missing = sorted(my_ids - set(pool_ids[present]))
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
        # Evaluated candidates carry their real marginal; pruned-out get 0.0 (marginal
        # is always >= 0, and the argmax is always an evaluated candidate — spec §3.5).
        out["score"] = out["gsis_id"].astype(str).map(marginals).fillna(0.0).astype(float)
        p_na: pd.Series[float] = pd.Series(pd.NA, index=out.index, dtype=pd.Float64Dtype())
        return _finalize(out, elig, p_na, starting_need_tier=False)
```

- [ ] **Step 4: Run the new tests**

Run: `pytest tests/test_draft/test_assistant_strategy.py -v`
Expected: PASS. If `test_season_value_takes_insurance_where_now_or_never_takes_vorp` is borderline, lower `RBa` p to `0.30` in `_depth_avail()` to widen the depth gap (the mechanism is unchanged); do not change the assertion.

- [ ] **Step 5: Update `__init__.py` re-exports if strategies are exported there**

Run: `grep -n "NowOrNeverStrategy" src/projections/draft/assistant/__init__.py`
If `NowOrNeverStrategy` is re-exported there, add `SeasonValueStrategy` alongside it (same import line + `__all__`). If the grep returns nothing, skip.

- [ ] **Step 6: Commit**

```bash
git add src/projections/draft/assistant/strategy.py tests/test_draft/test_assistant_strategy.py src/projections/draft/assistant/__init__.py
git commit -m "feat(draft): SeasonValueStrategy — depth-aware marginal-value strategy"
```

---

### Task 7: Factor `load_store_availability`; point `tournament_cli` at it

Extract the store-read + `build_availability` body out of `tournament_cli._build_season_valuer` into one shared loader both CLIs use, so availability is constructed in exactly one place.

**Files:**
- Create: `src/projections/draft/assistant/availability_loader.py`
- Modify: `src/projections/draft/assistant/tournament_cli.py`
- Test: `tests/test_draft/test_assistant_tournament_cli.py` (existing season tests are the regression guard)

- [ ] **Step 1: Confirm `_build_season_valuer` is not imported elsewhere**

Run: `grep -rn "_build_season_valuer" src tests`
Expected: only references inside `tournament_cli.py`. (If a test imports it directly, keep the name as a thin wrapper in Step 3 instead of removing it.)

- [ ] **Step 2: Create the shared loader**

Create `src/projections/draft/assistant/availability_loader.py`:

```python
"""Load per-player availability from the store (the shared CLI construction point).

Reads historical weekly_stats + the target-season schedules + id_map under
`<data_root>/raw`, then builds a `PlayerAvailability` for `pool`. A missing
weekly_stats history is a hard error (fail loud — spec §6); a missing
target-season schedule degrades to no byes (build_availability warns).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability, build_availability
from projections.store import read_partition

_HISTORY_SEASONS = range(2018, 2025)  # weekly_stats coverage for the availability model


def load_store_availability(
    pool: pd.DataFrame, *, season: int, data_root: Path
) -> PlayerAvailability:
    """Build `PlayerAvailability` for `pool` from store partitions under `data_root`."""
    raw = data_root / "raw"
    frames: list[pd.DataFrame] = []
    for yr in _HISTORY_SEASONS:
        try:
            frames.append(read_partition(raw, "weekly_stats", season=yr))
        except FileNotFoundError:
            continue
    if not frames:
        raise FileNotFoundError(
            f"no weekly_stats partitions under {raw} for seasons "
            f"{_HISTORY_SEASONS.start}-{_HISTORY_SEASONS.stop - 1}; check --data-root"
        )
    weekly_stats = pd.concat(frames, ignore_index=True)
    try:
        schedules = read_partition(raw, "schedules", season=season)
    except FileNotFoundError:
        # A missing target-season schedule degrades to no byes (build_availability
        # warns and the injury model still applies), not a hard fail.
        schedules = pd.DataFrame(columns=["season", "week", "home_team", "away_team"])
    # build_availability only reads gsis_id + team, so full IdMapSchema validation is skipped.
    id_map_path = raw / "id_map.parquet"
    if not id_map_path.exists():
        raise FileNotFoundError(f"id_map.parquet not found at {id_map_path}; check --data-root")
    id_map = pd.read_parquet(id_map_path)
    return build_availability(weekly_stats, schedules, id_map, pool, season=season)
```

- [ ] **Step 3: Rewrite `_build_season_valuer` to delegate**

In `src/projections/draft/assistant/tournament_cli.py`, replace the entire `_build_season_valuer` function (and the now-unused `_HISTORY_SEASONS`, `build_availability`, `read_partition` imports it owned) with a thin delegate:

```python
def _build_season_valuer(
    pool: pd.DataFrame, *, season: int, n_sims: int, base_seed: int, data_root: Path
) -> SeasonValuer:
    availability = load_store_availability(pool, season=season, data_root=data_root)
    return SeasonValuer(availability=availability, n_sims=n_sims, base_seed=base_seed)
```

Update the imports at the top of `tournament_cli.py`: remove `from projections.draft.assistant.availability import build_availability`, remove `from projections.store import read_partition`, remove the `_HISTORY_SEASONS = range(2018, 2025)` line, and add:

```python
from projections.draft.assistant.availability_loader import load_store_availability
```

- [ ] **Step 4: Run the tournament-CLI season tests (regression)**

Run: `pytest tests/test_draft/test_assistant_tournament_cli.py -v`
Expected: PASS — including `test_season_valuer_missing_data_fails_loud` (still raises with "weekly_stats"), `test_compare_with_season_valuer_runs`, and `test_season_valuer_degrades_without_target_schedule` (still warns "no schedules for season 2026").

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/availability_loader.py src/projections/draft/assistant/tournament_cli.py
git commit -m "refactor(draft): shared load_store_availability used by tournament CLI"
```

---

### Task 8: Register `season_value` in the tournament `compare` mode

Add an opt-in so the validation tournament can compare `season_value` against `now_or_never` and `raw_vorp` under the season valuer.

**Files:**
- Modify: `src/projections/draft/assistant/tournament_cli.py`
- Test: `tests/test_draft/test_assistant_tournament_cli.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_draft/test_assistant_tournament_cli.py` a test that reuses the season-store scaffolding. Extract the store-writing block from `test_compare_with_season_valuer_runs` into a module-level helper if not already, or inline it as below:

```python
def test_compare_includes_season_value_strategy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from projections.store import write_partition

    vorp_path, cfg_path = _write_inputs(tmp_path)
    data_root = tmp_path / "data"
    raw = data_root / "raw"
    n = 24
    gsis = [f"00-00000{i:02d}" for i in range(1, n + 1)]
    ws = pd.DataFrame(
        {
            "gsis_id": pd.array(gsis * 10, dtype=_PYARROW_STR),
            "season": [2022] * (n * 10),
            "week": [w for w in range(1, 11) for _ in range(n)],
            "position": pd.array(
                (["RB" if i % 2 else "WR" for i in range(n)]) * 10, dtype=_PYARROW_STR
            ),
        }
    )
    write_partition(raw, "weekly_stats", ws, season=2022)
    sched = pd.DataFrame(
        {
            "season": [2026] * 2,
            "week": [1, 2],
            "home_team": pd.array(["AA", "AA"], dtype=_PYARROW_STR),
            "away_team": pd.array(["BB", "BB"], dtype=_PYARROW_STR),
        }
    )
    write_partition(raw, "schedules", sched, season=2026)
    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "gsis_id": pd.array(gsis, dtype=_PYARROW_STR),
            "team": pd.array(["AA"] * n, dtype=_PYARROW_STR),
        }
    ).to_parquet(raw / "id_map.parquet")

    code = run(
        [
            "--vorp-table", str(vorp_path),
            "--league-config", str(cfg_path),
            "--my-slot", "2",
            "--seeds", "4",
            "--seed", "0",
            "--valuer", "season",
            "--season", "2026",
            "--n-sims", "15",
            "--data-root", str(data_root),
            "compare",
            "--with-season-value",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "season_value" in out and "now_or_never" in out and "raw_vorp" in out
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_draft/test_assistant_tournament_cli.py::test_compare_includes_season_value_strategy -v`
Expected: FAIL — argparse rejects `--with-season-value` (unrecognized argument).

- [ ] **Step 3: Add the flag and wire the strategy**

In `src/projections/draft/assistant/tournament_cli.py`:

Add the import:

```python
from projections.draft.assistant.strategy import (
    DraftStrategy,
    NowOrNeverStrategy,
    RawVorpStrategy,
    SeasonValueStrategy,
)
```

(Replace the existing `from projections.draft.assistant.strategy import NowOrNeverStrategy, RawVorpStrategy` line.)

In `_parse_args`, add to the `compare` subparser (after `--strategy-sigma`):

```python
    cmp_p.add_argument(
        "--with-season-value",
        action="store_true",
        help="Also run the depth-aware SeasonValueStrategy (requires --valuer season data).",
    )
```

In `run`, inside the `if args.mode == "compare":` block, build the strategy dict so it conditionally includes `season_value`. Replace the `run_tournament({...}, ...)` call's strategy mapping with:

```python
        strategies: dict[str, DraftStrategy] = {
            "now_or_never": NowOrNeverStrategy(LogisticSurvival(sigma=sigma)),
            "raw_vorp": RawVorpStrategy(),
        }
        if args.with_season_value:
            availability = load_store_availability(
                pool, season=args.season, data_root=args.data_root
            )
            strategies["season_value"] = SeasonValueStrategy(
                availability, n_sims=args.n_sims, base_seed=args.seed
            )
        result = run_tournament(
            strategies,
            pool=pool,
            config=config,
            my_slot=args.my_slot,
            n_seeds=args.seeds,
            adp_jitter=jitter,
            base_seed=args.seed,
            valuer=valuer,
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_draft/test_assistant_tournament_cli.py -v`
Expected: PASS (all, including the new test).

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/tournament_cli.py tests/test_draft/test_assistant_tournament_cli.py
git commit -m "feat(draft): --with-season-value in tournament compare mode"
```

---

### Task 9: Wire `season_value` into the live assistant CLI

Add `season_value` to the live `cli.py`: new `--season`/`--data-root`/`--n-sims` args and the availability build. The argparse default stays `now_or_never` until the validation gate (Task 11). A missing `weekly_stats` partition hard-fails (fail loud, no silent fallback — spec §6).

**Files:**
- Modify: `src/projections/draft/assistant/cli.py`
- Test: `tests/test_draft/test_assistant_cli.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_draft/test_assistant_cli.py` (check the file's existing imports; it already exercises `run`):

```python
def _season_store(tmp_path: Path, gsis: list[str]) -> Path:
    """Write minimal weekly_stats(2022) + schedules(2026) + id_map; return data_root."""
    import pandas as pd
    from projections.schemas import _PYARROW_STR
    from projections.store import write_partition

    data_root = tmp_path / "data"
    raw = data_root / "raw"
    n = len(gsis)
    ws = pd.DataFrame(
        {
            "gsis_id": pd.array(gsis * 8, dtype=_PYARROW_STR),
            "season": [2022] * (n * 8),
            "week": [w for w in range(1, 9) for _ in range(n)],
            "position": pd.array(
                (["RB" if i % 2 else "WR" for i in range(n)]) * 8, dtype=_PYARROW_STR
            ),
        }
    )
    write_partition(raw, "weekly_stats", ws, season=2022)
    sched = pd.DataFrame(
        {
            "season": [2026, 2026],
            "week": [1, 2],
            "home_team": pd.array(["AA", "AA"], dtype=_PYARROW_STR),
            "away_team": pd.array(["BB", "BB"], dtype=_PYARROW_STR),
        }
    )
    write_partition(raw, "schedules", sched, season=2026)
    raw.mkdir(parents=True, exist_ok=True)
    # Full IdMapSchema frame: the live CLI validates --id-map (we point it here), and
    # load_store_availability reads only gsis_id + team from the same file.
    na = pd.array([pd.NA] * n, dtype=_PYARROW_STR)
    pd.DataFrame(
        {
            "gsis_id": pd.array(gsis, dtype=_PYARROW_STR),
            "espn_id": na,
            "sleeper_id": na,
            "pfr_id": na,
            "full_name": pd.array([f"Player {i}" for i in range(n)], dtype=_PYARROW_STR),
            "position": pd.array(
                ["RB" if i % 2 else "WR" for i in range(n)], dtype=_PYARROW_STR
            ),
            "team": pd.array(["AA"] * n, dtype=_PYARROW_STR),
        }
    ).to_parquet(raw / "id_map.parquet")
    return data_root


def _season_inputs(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    """Write a vorp pool + an empty-draft state file; return (state, vorp, gsis)."""
    import json

    import pandas as pd
    from projections.schemas import _PYARROW_STR

    n = 12
    gsis = [f"00-00000{i:02d}" for i in range(1, n + 1)]
    pool = pd.DataFrame(
        {
            "gsis_id": pd.array(gsis, dtype=_PYARROW_STR),
            "position": pd.array(["RB" if i % 2 else "WR" for i in range(n)], dtype=_PYARROW_STR),
            "season_mean_fpts": [float(200 - i) for i in range(n)],
            "vorp": [float(100 - i) for i in range(n)],
            "replacement_fpts": [100.0] * n,
            "consensus_adp": pd.array([float(i + 1) for i in range(n)], dtype=pd.Float64Dtype()),
        }
    )
    vorp_path = tmp_path / "vorp.parquet"
    pool.to_parquet(vorp_path)

    cfg = {
        "name": "t",
        "n_teams": 4,
        "roster_slots": {"RB": 1, "WR": 1, "FLEX": 1, "BENCH": 2},
        "ruleset": "espn_ppr",
    }
    cfg_path = tmp_path / "league.json"
    cfg_path.write_text(json.dumps(cfg))
    state = {"league_config": str(cfg_path), "my_slot": 1, "picks": []}
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    return state_path, vorp_path, gsis


def test_cli_season_value_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from projections.draft.assistant.cli import run

    state_path, vorp_path, gsis = _season_inputs(tmp_path)
    data_root = _season_store(tmp_path, gsis)
    id_map = data_root / "raw" / "id_map.parquet"
    code = run(
        [
            "--state", str(state_path),
            "--vorp-table", str(vorp_path),
            "--id-map", str(id_map),
            "--strategy", "season_value",
            "--season", "2026",
            "--n-sims", "15",
            "--data-root", str(data_root),
        ]
    )
    assert code == 0
    assert "PLAYER" in capsys.readouterr().out  # the table header printed


def test_cli_season_value_missing_weekly_stats_fails_loud(tmp_path: Path) -> None:
    # A valid --id-map (so _load_id_map passes) but an empty --data-root (no
    # weekly_stats) must hard-fail in availability loading, not silently fall back.
    from projections.draft.assistant.cli import run

    state_path, vorp_path, gsis = _season_inputs(tmp_path)
    data_root = _season_store(tmp_path, gsis)  # writes a valid id_map under raw/
    id_map = data_root / "raw" / "id_map.parquet"
    empty_root = tmp_path / "empty"  # no raw/ partitions
    with pytest.raises(FileNotFoundError, match="weekly_stats"):
        run(
            [
                "--state", str(state_path),
                "--vorp-table", str(vorp_path),
                "--id-map", str(id_map),
                "--strategy", "season_value",
                "--season", "2026",
                "--n-sims", "15",
                "--data-root", str(empty_root),
            ]
        )
```

Ensure `import pytest` and `from pathlib import Path` are present at the top of the test file (add if missing).

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_draft/test_assistant_cli.py::test_cli_season_value_runs -v`
Expected: FAIL — argparse rejects `--strategy season_value` / `--season` (unrecognized).

- [ ] **Step 3: Wire the live CLI**

In `src/projections/draft/assistant/cli.py`:

Add imports:

```python
from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.strategy import SeasonValueStrategy
```

Change `generate_recommendation` to accept the season args and build the strategy after the pool is loaded. Replace the function with:

```python
def generate_recommendation(
    *,
    state_path: Path,
    vorp_path: Path,
    id_map_path: Path,
    strategy_name: str,
    sigma: float | None,
    season: int,
    n_sims: int,
    data_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load inputs, run the chosen strategy.

    Returns `(recommendation, id_map)` — the validated id_map is handed back so
    callers (the CLI display path) reuse it instead of re-reading + re-validating.
    """
    id_map = _load_id_map(id_map_path)
    state, league = load_draft_state(state_path, id_map)

    vorp = pd.read_parquet(vorp_path)
    vorp["gsis_id"] = vorp["gsis_id"].astype(_PYARROW_STR)
    vorp = VorpTableSchema.validate(vorp)

    strategy: DraftStrategy
    if strategy_name == "season_value":
        availability = load_store_availability(vorp, season=season, data_root=data_root)
        strategy = SeasonValueStrategy(availability, n_sims=n_sims, base_seed=0)
    else:
        strategy = _build_strategy(strategy_name, league.n_teams, sigma)
    return strategy.recommend(state, vorp, league), id_map
```

In `_parse_args`, add `season_value` to the `--strategy` choices and add the three new args:

```python
    p.add_argument(
        "--strategy",
        choices=["now_or_never", "raw_vorp", "season_value"],
        default="now_or_never",
        help="Recommendation strategy (default now_or_never).",
    )
```

and after `--sigma`:

```python
    p.add_argument(
        "--season",
        type=int,
        default=2026,
        help="[--strategy season_value] target season for byes + availability.",
    )
    p.add_argument(
        "--n-sims",
        type=int,
        default=300,
        help="[--strategy season_value] Monte-Carlo seasons per candidate.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="[--strategy season_value] store root for weekly_stats/schedules/id_map.",
    )
```

In `run`, pass the new args through:

```python
    rec, id_map = generate_recommendation(
        state_path=args.state,
        vorp_path=args.vorp_table,
        id_map_path=args.id_map,
        strategy_name=args.strategy,
        sigma=args.sigma,
        season=args.season,
        n_sims=args.n_sims,
        data_root=args.data_root,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_draft/test_assistant_cli.py -v`
Expected: PASS (all existing + the 2 new tests).

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/cli.py tests/test_draft/test_assistant_cli.py
git commit -m "feat(draft): season_value strategy in the live assistant CLI"
```

---

### Task 10: Validation run + report

Run the tournament under the season valuer comparing the three strategies on the real 2026 consensus pool, at slots 1 / 6 / 12, and write up the result. This is a manual validation task (no unit test).

**Files:**
- Create: `reports/depth_aware_strategy_validation_2026.md`

- [ ] **Step 1: Confirm the inputs exist**

Run: `ls data/consensus_vorp_2026.parquet configs/league_espn_ppr_12team_skill.json`
Expected: both present. If `data/consensus_vorp_2026.parquet` is missing, regenerate per `project_management.md` (the Draft-Hub-on-consensus entry: `python scripts/generate_vorp_table.py --source consensus --season 2026 --league-config configs/league_espn_ppr_12team_skill.json --out data/consensus_vorp_2026.parquet`).

- [ ] **Step 2: Run the tournament at each slot**

Run (slot 1; repeat with `--my-slot 6` and `--my-slot 12`):

```bash
python scripts/draft_tournament.py \
  --vorp-table data/consensus_vorp_2026.parquet \
  --league-config configs/league_espn_ppr_12team_skill.json \
  --my-slot 1 --seeds 200 --seed 0 \
  --valuer season --season 2026 --n-sims 300 \
  compare --with-season-value
```

Note: `--season 2026` warns "no schedules for season 2026" and runs with no byes unless the 2026 schedule partition has been ingested (see `project_management.md`). That is acceptable for this comparison; record it in the report.

- [ ] **Step 3: Run the determinism check**

Run:

```bash
python scripts/draft_tournament.py \
  --vorp-table data/consensus_vorp_2026.parquet \
  --league-config configs/league_espn_ppr_12team_skill.json \
  --my-slot 6 --seeds 1 --seed 0 --adp-jitter 0 \
  --valuer season --season 2026 --n-sims 300 \
  compare --with-season-value
```

Expected: point CIs (`lo == hi == point`) — identical roster on a single zero-jitter seed.

- [ ] **Step 4: Run the starters-metric guardrail**

Repeat Step 2's slot-1/6/12 runs with `--valuer starters` (drop `--season/--n-sims/--data-root` are still accepted/ignored by the starters valuer; keep `--with-season-value` which still needs the season args to build the strategy — so keep `--season 2026 --n-sims 300 --data-root data`). Record `season_value`'s starters-metric numbers vs `now_or_never`.

- [ ] **Step 5: Write the report**

Create `reports/depth_aware_strategy_validation_2026.md` documenting: the per-slot season-metric means + paired-diff CIs for `season_value` vs `now_or_never` vs `raw_vorp`; whether the primary bar (`season_value` beats `now_or_never`, CI excludes 0, at slots 1/6/12) is met; the starters-metric guardrail numbers; the determinism result; and the no-byes caveat. Interpret a slot-12 miss per the spec's residual-risk note (the turn is where pick-timing matters least).

- [ ] **Step 6: Commit**

```bash
git add reports/depth_aware_strategy_validation_2026.md
git commit -m "docs(draft): depth-aware strategy validation on 2026 consensus pool"
```

---

### Task 11: Flip the default (gated on the validation bar) + update PM/TODO

If Task 10's primary bar is met, make `season_value` the live CLI's argparse default; otherwise leave it selectable. Either way, update the running docs.

**Files:**
- Modify: `src/projections/draft/assistant/cli.py` (conditional)
- Modify: `project_management.md`, `TODO.md`

- [ ] **Step 1: Decide from the validation result**

If `reports/depth_aware_strategy_validation_2026.md` records the primary bar as MET, proceed to Step 2. If NOT met, skip Step 2–4 (default stays `now_or_never`) and go to Step 5; record the decision in the report.

- [ ] **Step 2: Flip the live CLI default**

In `src/projections/draft/assistant/cli.py`, change the `--strategy` argument `default` from `"now_or_never"` to `"season_value"` and update its help text. Note: the zero-config default now loads availability and **hard-fails on missing `weekly_stats`** — that is the intended fail-loud behavior (spec §6).

- [ ] **Step 3: Add a default-flip regression test**

Add to `tests/test_draft/test_assistant_cli.py`:

```python
def test_default_strategy_is_season_value() -> None:
    from projections.draft.assistant.cli import _parse_args

    args = _parse_args(["--state", "s.json", "--vorp-table", "v.parquet"])
    assert args.strategy == "season_value"
```

Run: `pytest tests/test_draft/test_assistant_cli.py::test_default_strategy_is_season_value -v`
Expected: PASS.

- [ ] **Step 4: Commit the flip**

```bash
git add src/projections/draft/assistant/cli.py tests/test_draft/test_assistant_cli.py
git commit -m "feat(draft): default live assistant to season_value strategy"
```

- [ ] **Step 5: Update PM + TODO**

Add a `project_management.md` top entry (Draft Assistant — Depth-Aware Strategy) summarizing what shipped, the validation numbers, and the next direction (opportunity-cost layer in season-value space — spec §7). Update `TODO.md` #38's Draft-Assistant bullet: mark the depth-aware strategy slice done, and add the deferred follow-ups (opportunity-cost-in-season-value-space strategy; numpy fast-path for the weekly fill; conditional survival in the timing layer).

- [ ] **Step 6: Commit the docs**

```bash
git add project_management.md TODO.md
git commit -m "docs(pm,todo): depth-aware draft strategy shipped"
```

---

## Final Verification (run before opening the PR)

- [ ] `pytest -v` — all pass (state the count; the 2 known pre-existing failures from TODO #40 / the flaky Windows scipy segfault are unrelated to this branch).
- [ ] `pytest -v -k "ingest or store or schemas"` — green (this branch touches no ingest/store paths, but `RecommendationSchema` is exercised; run as the integration-seam guard per the project bar).
- [ ] `mypy src tests` — zero errors.
- [ ] `ruff check src tests` — zero violations.
- [ ] `ruff format --check src tests` — no drift.
- [ ] Paste the command outputs into the PR description as evidence.

---

## Self-Review Notes

- **Spec coverage:** §3.1/§3.2 strategy → Task 6; §3.2 step-2 missing-from-pool → Task 6 (warn test); §3.3 CRN → Tasks 2–4; §3.4 ordering → Task 5; §3.5 pruning + fallback (`top_k≥1` argmax) → Task 6 (pruning-invariance); §3.8 shared loader + both CLIs → Tasks 7–9; §4 tests → Tasks 1–9 (incl. CRN mean-equivalence in Task 3, discriminating depth test in Task 6); §6 success bar + default flip → Tasks 10–11. No spec requirement left unmapped.
- **Type consistency:** `expected_season_points_crn(roster, slots, availability, *, draws, col_of, weeks)` and `marginal_season_values(base_roster, candidates, roster_slots, availability, *, n_sims, rng, weeks)` are used with matching signatures in Tasks 3, 4, 6. `_finalize(..., *, starting_need_tier=True)` keyword matches Tasks 5 and 6. `load_store_availability(pool, *, season, data_root)` matches Tasks 7, 8, 9.
- **No placeholders:** every code/test step shows complete code and exact commands.
