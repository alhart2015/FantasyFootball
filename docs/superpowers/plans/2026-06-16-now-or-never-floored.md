# Now-or-Never Floored Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-06-16-now-or-never-floored-design.md`

**Goal:** Add `now_or_never_floored`, a separate `DraftStrategy` that is `now_or_never` plus a one-sided hinge penalty below an absolute VORP bar, A/B-able against the existing strategies.

**Architecture:** A new frozen dataclass `NowOrNeverFlooredStrategy` in `strategy.py` computes `score = vorp − E[best survivor] − λ·max(0, F − vorp)`, reusing the existing `expected_best_by_position` helper and `_finalize`. It is wired into the three strategy-construction seams (assistant CLI via `build_session_strategy`, the live board's `BOARD_STRATEGIES`, and the H2H harness registry `_build_strategy`) with `floor` / `floor_weight` knobs (provisional defaults `40.0` / `1.0`). `now_or_never` is **not edited** — it stays the byte-identical control, and `λ=0` reproduces it exactly.

**Tech Stack:** Python 3.12, numpy, pandas (pyarrow-backed), pandera, pytest, argparse. mypy strict + ruff are gates.

**Key conventions (from CLAUDE.md):** reference `Position`/`RosterSlot` enums never strings; `df = SCHEMA.validate(df)` with reassignment; every code change is TDD (red → green → commit); run `pytest -v`, `mypy src tests`, `ruff check src tests`, `ruff format --check src tests` before declaring done.

---

## File Structure

**Modified:**
- `src/projections/draft/assistant/strategy.py` — add `"now_or_never_floored"` to `STRATEGY_KEYS`; add `NowOrNeverFlooredStrategy`. (~40 LOC added; `NowOrNeverStrategy` untouched.)
- `src/projections/draft/assistant/live.py` — `build_session_strategy` gains `floor`/`floor_weight` kwargs + a `now_or_never_floored` branch; `"now_or_never_floored"` added to `BOARD_STRATEGIES`. (Not added to `MC_STRATEGIES` — it is analytic.)
- `src/projections/draft/assistant/cli.py` — `--floor`/`--floor-weight` args threaded through `generate_recommendation` → `build_session_strategy`.
- `src/projections/draft/backtest/harness.py` — `_build_strategy`, `collect_results`, `run_backtest` gain `floor`/`floor_weight` (defaulted); registry branch added.
- `src/projections/draft/backtest/cli.py` — `--floor`/`--floor-weight` args → `run_backtest`.
- `scripts/h2h_backtest_chunked.py` — `--floor`/`--floor-weight` args; threaded into `_run_worker`'s `collect_results`; added to the manifest `run_key`; forwarded in the worker subprocess command.
- `reports/draft_strategy_tests.md`, `project_management.md`, `TODO.md` — the validation test entry + status (Task 4).

**Test files (modified — exact paths, confirmed present):**
- `tests/test_draft/test_assistant_strategy.py` — unit tests for `NowOrNeverFlooredStrategy`.
- `tests/test_draft/test_assistant_live.py` — `build_session_strategy` / `BOARD_STRATEGIES` wiring.
- `tests/test_draft/test_assistant_cli.py` — CLI flag smoke.
- `tests/test_draft/test_backtest/test_harness.py` — `_build_strategy("now_or_never_floored", …)` construction + defaults; the existing default-`(nn, sv)` aggregation tests stay green.
- `tests/test_draft/test_backtest/test_checkpoint.py` — `verify_or_write_manifest` rejects a changed `floor` (helper-level guard).
- `tests/test_scripts/test_h2h_backtest_chunked.py` — `_run_key(args)` contains `floor`/`floor_weight` (the driver-level provenance guard). `tests/test_scripts/conftest.py` already puts `scripts/` on `sys.path`, so `from scripts.h2h_backtest_chunked import _run_key, _parse_args` works.

---

## Task 1: `NowOrNeverFlooredStrategy` + unit tests

**Files:**
- Modify: `src/projections/draft/assistant/strategy.py` (`STRATEGY_KEYS` tuple ~line 38; add class after `NowOrNeverStrategy` ~line 266)
- Test: `tests/test_draft/test_assistant_strategy.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_draft/test_assistant_strategy.py` (reuses the existing `_pool`, `_state`, `_config`, `_FakeSurvival` helpers and imports):

```python
def test_floored_satisfies_protocol() -> None:
    from projections.draft.assistant.strategy import NowOrNeverFlooredStrategy

    assert isinstance(NowOrNeverFlooredStrategy(_FakeSurvival()), DraftStrategy)


def test_floored_lambda_zero_is_identical_to_now_or_never() -> None:
    """floor_weight=0 ⇒ byte-identical recommendation to now_or_never, for any floor."""
    from projections.draft.assistant.strategy import NowOrNeverFlooredStrategy

    base = NowOrNeverStrategy(_FakeSurvival()).recommend(_state(), _pool(), _config())
    floored = NowOrNeverFlooredStrategy(
        _FakeSurvival(), floor=999.0, floor_weight=0.0
    ).recommend(_state(), _pool(), _config())
    pd.testing.assert_frame_equal(base, floored)


def test_floored_score_is_now_or_never_minus_hinge() -> None:
    """score == vorp − E[best survivor] − λ·max(0, F − vorp), hand-computed on _pool()."""
    from projections.draft.assistant.strategy import NowOrNeverFlooredStrategy

    rec = NowOrNeverFlooredStrategy(_FakeSurvival(), floor=52.0, floor_weight=6.0).recommend(
        _state(), _pool(), _config()
    )
    # now_or_never scores (see test_now_or_never_reorders_cross_position):
    #   rb1 12.6 (vorp 50), rb2 2.6 (vorp 40), wr1 1.25 (vorp 52), wr2 -20.75 (vorp 30)
    # hinge with F=52, λ=6:  rb1 -6*2=-12 ⇒ 0.6 ; rb2 -6*12=-72 ⇒ -69.4 ;
    #   wr1 -6*0=0 ⇒ 1.25 ; wr2 -6*22=-132 ⇒ -152.75
    by_id = dict(zip(rec["gsis_id"], rec["score"], strict=True))
    assert by_id["00-0000010"] == 0.6
    assert by_id["00-0000011"] == -69.4
    assert by_id["00-0000020"] == 1.25
    assert by_id["00-0000021"] == -152.75


def test_floored_demotes_sub_floor_scarce_player_below_better_one() -> None:
    """The pathology fix: under nn, scarce rb1 (vorp 50) outranks wr1 (vorp 52);
    once the floor bites (F=52 so wr1 is at the bar, rb1 below), wr1 leads."""
    from projections.draft.assistant.strategy import NowOrNeverFlooredStrategy

    nn = NowOrNeverStrategy(_FakeSurvival()).recommend(_state(), _pool(), _config())
    assert nn["gsis_id"].iloc[0] == "00-0000010"  # rb1 first under now_or_never

    floored = NowOrNeverFlooredStrategy(_FakeSurvival(), floor=52.0, floor_weight=6.0).recommend(
        _state(), _pool(), _config()
    )
    assert floored["gsis_id"].iloc[0] == "00-0000020"  # wr1 now first (floor flip)


def test_floored_last_pick_fallback_equals_raw_vorp() -> None:
    """No next pick ⇒ raw VORP, floor not applied (identical to nn's fallback), any (F, λ)."""
    from projections.draft.assistant.strategy import NowOrNeverFlooredStrategy

    last = _state(current_pick=7, rounds=1)
    floored = NowOrNeverFlooredStrategy(_FakeSurvival(), floor=999.0, floor_weight=9.0).recommend(
        last, _pool(), _config()
    )
    raw = RawVorpStrategy().recommend(last, _pool(), _config())
    assert list(floored["gsis_id"]) == list(raw["gsis_id"])
    assert floored["p_available_next"].isna().all()


def test_floored_rejects_degenerate_params() -> None:
    from projections.draft.assistant.strategy import NowOrNeverFlooredStrategy

    with pytest.raises(ValueError, match="floor_weight"):
        NowOrNeverFlooredStrategy(_FakeSurvival(), floor_weight=-1.0)
    with pytest.raises(ValueError, match="finite"):
        NowOrNeverFlooredStrategy(_FakeSurvival(), floor=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        NowOrNeverFlooredStrategy(_FakeSurvival(), floor_weight=float("inf"))


def test_floored_null_adp_p_available_is_null() -> None:
    """Null ADP → p=1 internally (hinge is ADP-independent), display p null. Mirrors
    the now_or_never null-ADP test but on the floored class with the floor active."""
    from projections.draft.assistant.strategy import NowOrNeverFlooredStrategy

    pool = _pool()
    pool["consensus_adp"] = pd.array([pd.NA, pd.NA, pd.NA, pd.NA], dtype=pd.Float64Dtype())
    rec = NowOrNeverFlooredStrategy(
        LogisticSurvival(sigma=8.0), floor=45.0, floor_weight=2.0
    ).recommend(_state(), pool, _config())
    RecommendationSchema.validate(rec)
    assert rec["p_available_next"].isna().all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_draft/test_assistant_strategy.py -k floored -v`
Expected: FAIL — `ImportError: cannot import name 'NowOrNeverFlooredStrategy'`.

- [ ] **Step 3: Add the key to `STRATEGY_KEYS` + the default constants**

In `src/projections/draft/assistant/strategy.py`, extend the tuple (keep the others):

```python
STRATEGY_KEYS = (
    "now_or_never",
    "now_or_never_floored",
    "season_value",
    "season_value_var",
    "season_value_timing",
    "raw_vorp",
)
```

Immediately below `STRATEGY_KEYS`, add the **single source of truth** for the floor defaults (every pass-through layer imports these, so Task 4 changes the shipped default in exactly one place):

```python
# PROVISIONAL defaults for the now_or_never_floored knobs — a mid-grid starting point,
# replaced by the A/B winner (spec §8 / plan Task 4). Imported by build_session_strategy,
# the harness registry, and both CLIs so there is ONE literal to update.
_DEFAULT_FLOOR = 40.0
_DEFAULT_FLOOR_WEIGHT = 1.0
```

Then, after checking for any test that asserts the exact `STRATEGY_KEYS` tuple and updating it:

Run: `grep -rn "STRATEGY_KEYS ==" tests/ ; grep -rn "STRATEGY_KEYS" tests/test_draft/test_assistant_strategy.py`
If a test pins the tuple contents/length, update it to include `"now_or_never_floored"`.

- [ ] **Step 4: Add the `math` import and the strategy class**

At the top of `strategy.py` add `import math` (next to the other stdlib imports). Then, immediately after `NowOrNeverStrategy` (the class ending ~line 265), add:

```python
@dataclass(frozen=True)
class NowOrNeverFlooredStrategy:
    """now_or_never plus an absolute quality floor (spec 2026-06-16).

    score = vorp − E[best survivor at position by my next pick]
            − floor_weight · max(0, floor − vorp)

    The hinge demotes sub-`floor` players so the dynamic-scarcity term can no longer
    float a best-of-a-bad-tier player over a better one elsewhere. `floor_weight == 0`
    reproduces `NowOrNeverStrategy` exactly. The ~8-line score prelude is duplicated from
    `NowOrNeverStrategy` *deliberately* — the spec keeps `now_or_never` byte-identical as
    the A/B control, so we copy rather than extract-and-share (which would edit the control).
    `floor` / `floor_weight` defaults are PROVISIONAL — set from the A/B (Task 4).
    """

    survival: SurvivalModel
    floor: float = _DEFAULT_FLOOR
    floor_weight: float = _DEFAULT_FLOOR_WEIGHT

    def __post_init__(self) -> None:
        if not math.isfinite(self.floor) or not math.isfinite(self.floor_weight):
            raise ValueError(
                f"floor and floor_weight must be finite; got floor={self.floor}, "
                f"floor_weight={self.floor_weight}"
            )
        if self.floor_weight < 0:
            raise ValueError(f"floor_weight must be >= 0; got {self.floor_weight}")

    def recommend(
        self, state: DraftState, pool: pd.DataFrame, config: LeagueConfig
    ) -> pd.DataFrame:
        df, elig = _eligible_subset(state, pool, config)
        next_pick = my_next_pick(state.current_pick, state.my_slot, state.n_teams, state.rounds)
        if next_pick is None:
            # Last pick → raw VORP, floor not applied (matches now_or_never's fallback).
            return _raw_vorp_result(df, elig)

        adp = df["consensus_adp"]
        internal_p = adp.map(
            lambda a: self.survival.p_available(
                float(a) if pd.notna(a) else float("nan"), next_pick
            )
        ).astype(float)
        display_p = internal_p.where(adp.notna(), other=pd.NA)

        pos = df["position"].to_numpy()
        vorp = df["vorp"].to_numpy(dtype=float)
        p = internal_p.to_numpy(dtype=float)
        gsis = df["gsis_id"].to_numpy()
        e_best = expected_best_by_position(pos, vorp, p, gsis)

        penalty = self.floor_weight * np.maximum(0.0, self.floor - vorp)
        df["score"] = vorp - np.array([e_best[pos_i] for pos_i in pos], dtype=float) - penalty
        return _finalize(df, elig, display_p)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_draft/test_assistant_strategy.py -k floored -v`
Expected: PASS (6 tests).

- [ ] **Step 6: Run the full strategy module + lint/type on the file**

Run: `pytest tests/test_draft/test_assistant_strategy.py -v && mypy src/projections/draft/assistant/strategy.py && ruff check src/projections/draft/assistant/strategy.py && ruff format --check src/projections/draft/assistant/strategy.py`
Expected: all PASS (the pre-existing now_or_never tests stay green — proof the control is untouched).

- [ ] **Step 7: Commit**

```bash
git add src/projections/draft/assistant/strategy.py tests/test_draft/test_assistant_strategy.py
git commit -m "feat(draft): NowOrNeverFlooredStrategy — absolute scarcity floor

now_or_never plus a one-sided hinge penalty below an absolute VORP bar F
(weight λ). λ=0 reproduces now_or_never byte-identically. now_or_never
itself is untouched (the A/B control).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Assistant CLI + live-board wiring

**Files:**
- Modify: `src/projections/draft/assistant/live.py` (`BOARD_STRATEGIES` ~line 44; `build_session_strategy` ~line 52)
- Modify: `src/projections/draft/assistant/cli.py` (`generate_recommendation` ~line 34; `_parse_args` ~line 93; `run` ~line 147)
- Test: `tests/test_draft/test_assistant_live.py`, `tests/test_draft/test_assistant_cli.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_draft/test_assistant_live.py`:

```python
def test_build_session_strategy_now_or_never_floored() -> None:
    from projections.draft.assistant.live import BOARD_STRATEGIES, build_session_strategy
    from projections.draft.assistant.strategy import NowOrNeverFlooredStrategy

    strat = build_session_strategy(
        "now_or_never_floored",
        league=_league(),  # existing helper in this module
        sigma=None,
        availability=None,  # analytic: must NOT require availability
        n_sims=300,
        base_seed=0,
        floor=55.0,
        floor_weight=2.0,
    )
    assert isinstance(strat, NowOrNeverFlooredStrategy)
    assert strat.floor == 55.0
    assert strat.floor_weight == 2.0
    assert "now_or_never_floored" in BOARD_STRATEGIES


def test_build_session_strategy_floored_defaults() -> None:
    from projections.draft.assistant.live import build_session_strategy
    from projections.draft.assistant.strategy import (
        _DEFAULT_FLOOR,
        _DEFAULT_FLOOR_WEIGHT,
        NowOrNeverFlooredStrategy,
    )

    strat = build_session_strategy(
        "now_or_never_floored",
        league=_league(),
        sigma=None,
        availability=None,
        n_sims=300,
        base_seed=0,
    )
    assert isinstance(strat, NowOrNeverFlooredStrategy)
    # Compare against the constants (not literals) so Task 4's default change stays green.
    assert strat.floor == _DEFAULT_FLOOR
    assert strat.floor_weight == _DEFAULT_FLOOR_WEIGHT
```

> If `test_assistant_live.py` has no `_league()` helper, build a `LeagueConfig` inline as `_config()` does in `test_assistant_strategy.py` (n_teams=12, the same roster_slots). Check with `grep -n "_league\|LeagueConfig(" tests/test_draft/test_assistant_live.py` first.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_draft/test_assistant_live.py -k floored -v`
Expected: FAIL — `build_session_strategy() got an unexpected keyword argument 'floor'`.

- [ ] **Step 3: Wire `build_session_strategy` + `BOARD_STRATEGIES`**

In `src/projections/draft/assistant/live.py`:

(a) Import the new class **and the default constants** — extend the existing `from projections.draft.assistant.strategy import (...)` block to include `NowOrNeverFlooredStrategy`, `_DEFAULT_FLOOR`, `_DEFAULT_FLOOR_WEIGHT`.

(b) Add to `BOARD_STRATEGIES` (after `"now_or_never"`):

```python
BOARD_STRATEGIES: tuple[str, ...] = (
    "now_or_never",
    "now_or_never_floored",
    "raw_vorp",
    "season_value",
    "season_value_timing",
)
```

(c) Add `floor` / `floor_weight` params and a branch to `build_session_strategy` (the new params go after `base_seed`; the branch goes right after the `now_or_never` branch, before the `MC_STRATEGIES` check):

```python
def build_session_strategy(
    name: str,
    *,
    league: LeagueConfig,
    sigma: float | None,
    availability: PlayerAvailability | None,
    n_sims: int,
    base_seed: int,
    floor: float = _DEFAULT_FLOOR,
    floor_weight: float = _DEFAULT_FLOOR_WEIGHT,
) -> DraftStrategy:
    ...
    if name == "now_or_never":
        spread = default_sigma(league.n_teams) if sigma is None else sigma
        return NowOrNeverStrategy(LogisticSurvival(sigma=spread))
    if name == "now_or_never_floored":
        spread = default_sigma(league.n_teams) if sigma is None else sigma
        return NowOrNeverFlooredStrategy(
            LogisticSurvival(sigma=spread), floor=floor, floor_weight=floor_weight
        )
    if name in MC_STRATEGIES:
        ...
```

(Do **not** add `"now_or_never_floored"` to `MC_STRATEGIES` — it is analytic and needs no availability.)

- [ ] **Step 4: Thread the CLI flags**

In `src/projections/draft/assistant/cli.py`:

(a) `generate_recommendation` — add `floor: float = _DEFAULT_FLOOR, floor_weight: float = _DEFAULT_FLOOR_WEIGHT` to the signature (import the two constants from `projections.draft.assistant.strategy`) and pass them to `build_session_strategy`:

```python
    strategy: DraftStrategy = build_session_strategy(
        strategy_name,
        league=league,
        sigma=sigma,
        availability=availability,
        n_sims=n_sims,
        base_seed=0,
        floor=floor,
        floor_weight=floor_weight,
    )
```

(b) `_parse_args` — add after `--sigma`:

```python
    p.add_argument(
        "--floor",
        type=float,
        default=_DEFAULT_FLOOR,
        help="[--strategy now_or_never_floored] absolute VORP quality bar F.",
    )
    p.add_argument(
        "--floor-weight",
        type=float,
        default=_DEFAULT_FLOOR_WEIGHT,
        help="[--strategy now_or_never_floored] hinge weight λ (0 = plain now_or_never).",
    )
```

(c) `run` — pass them through:

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
        floor=args.floor,
        floor_weight=args.floor_weight,
    )
```

- [ ] **Step 5: Add the CLI smoke test**

Append to `tests/test_draft/test_assistant_cli.py` (follow the existing CLI smoke test's fixture setup — locate it with `grep -n "def test_\|_parse_args\|generate_recommendation\|tmp_path" tests/test_draft/test_assistant_cli.py`). Add a parse-level assertion that does not need data:

```python
def test_cli_parses_now_or_never_floored_flags() -> None:
    from projections.draft.assistant.cli import _parse_args

    args = _parse_args(
        [
            "--state", "s.json", "--vorp-table", "v.parquet",
            "--strategy", "now_or_never_floored", "--floor", "55", "--floor-weight", "2.5",
        ]
    )
    assert args.strategy == "now_or_never_floored"
    assert args.floor == 55.0
    assert args.floor_weight == 2.5
```

- [ ] **Step 6: Run to verify pass**

Run: `pytest tests/test_draft/test_assistant_live.py tests/test_draft/test_assistant_cli.py -k "floored or now_or_never_floored" -v`
Expected: PASS.

- [ ] **Step 7: Lint/type the touched files**

Run: `mypy src/projections/draft/assistant/live.py src/projections/draft/assistant/cli.py && ruff check src/projections/draft/assistant/live.py src/projections/draft/assistant/cli.py && ruff format --check src/projections/draft/assistant/live.py src/projections/draft/assistant/cli.py`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/projections/draft/assistant/live.py src/projections/draft/assistant/cli.py tests/test_draft/test_assistant_live.py tests/test_draft/test_assistant_cli.py
git commit -m "feat(draft): wire now_or_never_floored into assistant CLI + live board

build_session_strategy gains floor/floor_weight (default 40/1); the key is
added to BOARD_STRATEGIES (board offers it at the default) and NOT to
MC_STRATEGIES (analytic). CLI gains --floor/--floor-weight.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: H2H harness registry + runners + manifest

**Files:**
- Modify: `src/projections/draft/backtest/harness.py` (`_build_strategy` ~line 47; `collect_results` ~line 78; `run_backtest` ~line 188)
- Modify: `src/projections/draft/backtest/cli.py` (`_parse_args` ~line 23; `run` ~line 107)
- Modify: `scripts/h2h_backtest_chunked.py` (`_parse_args` ~line 55; `_run_worker` ~line 110; new `_run_key` helper; `_run_driver` manifest ~line 177 + worker cmd ~line 196)
- Test: `tests/test_draft/test_backtest/test_harness.py`, `tests/test_draft/test_backtest/test_checkpoint.py`, `tests/test_scripts/test_h2h_backtest_chunked.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_draft/test_backtest/test_harness.py` (it already imports `_build_strategy`):

```python
def test_build_strategy_now_or_never_floored() -> None:
    from projections.draft.assistant.strategy import NowOrNeverFlooredStrategy
    from projections.draft.backtest.harness import _build_strategy

    strat = _build_strategy(
        "now_or_never_floored",
        availability=_availability(),  # existing helper / minimal PlayerAvailability
        n_teams=16,
        strategy_n_sims=10,
        base_seed=0,
        floor=55.0,
        floor_weight=2.0,
    )
    assert isinstance(strat, NowOrNeverFlooredStrategy)
    assert strat.floor == 55.0 and strat.floor_weight == 2.0


def test_build_strategy_floored_defaults_when_unset() -> None:
    from projections.draft.assistant.strategy import (
        _DEFAULT_FLOOR,
        _DEFAULT_FLOOR_WEIGHT,
        NowOrNeverFlooredStrategy,
    )
    from projections.draft.backtest.harness import _build_strategy

    strat = _build_strategy(
        "now_or_never_floored",
        availability=_availability(),
        n_teams=16,
        strategy_n_sims=10,
        base_seed=0,
    )
    assert isinstance(strat, NowOrNeverFlooredStrategy)
    assert strat.floor == _DEFAULT_FLOOR and strat.floor_weight == _DEFAULT_FLOOR_WEIGHT
```

> Locate the existing availability fixture in `test_harness.py`: `grep -rn "PlayerAvailability(\|_availability\|def availability" tests/test_draft/test_backtest/test_harness.py | head`. Reuse it; if it is a pytest fixture, take it as a parameter rather than calling `_availability()`.

Add to `tests/test_draft/test_backtest/test_checkpoint.py` — the **helper-level** guard (proves `verify_or_write_manifest` rejects a changed floor):

```python
def test_manifest_rejects_changed_floor(tmp_path) -> None:
    from projections.draft.backtest.checkpoint import verify_or_write_manifest

    key = {
        "season": 2025, "strategy_a": "now_or_never_floored", "strategy_b": "now_or_never",
        "strategy_n_sims": 200, "jitter": 8.0, "floor": 40.0, "floor_weight": 1.0,
    }
    verify_or_write_manifest(tmp_path, key)  # first write OK
    verify_or_write_manifest(tmp_path, key)  # same key OK
    with pytest.raises(ValueError, match="was built with params"):
        verify_or_write_manifest(tmp_path, {**key, "floor": 60.0})  # changed floor → reject
```

Add to `tests/test_scripts/test_h2h_backtest_chunked.py` — the **driver-level** guard (proves the driver actually puts floor/weight into the manifest key; this is the real §5 protection):

```python
def test_run_key_includes_floor_params() -> None:
    from scripts.h2h_backtest_chunked import _parse_args, _run_key

    args = _parse_args(
        [
            "--league-config", "configs/league_espn_ppr_12team_skill.json",
            "--strategy-a", "now_or_never_floored", "--strategy-b", "now_or_never",
            "--floor", "55", "--floor-weight", "2",
        ]
    )
    key = _run_key(args)
    assert key["floor"] == 55.0
    assert key["floor_weight"] == 2.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_draft/test_backtest/test_harness.py tests/test_draft/test_backtest/test_checkpoint.py tests/test_scripts/test_h2h_backtest_chunked.py -k "floor" -v`
Expected: FAIL — `_build_strategy() got an unexpected keyword argument 'floor'` (harness); `ImportError: cannot import name '_run_key'` (chunked). The checkpoint helper test may already pass (the helper compares dicts) — that's fine; the `_run_key` test is the one that pins the driver wiring and fails until Step 4(c).

- [ ] **Step 3: Wire `_build_strategy`, `collect_results`, `run_backtest`**

In `src/projections/draft/backtest/harness.py`:

(a) Import `NowOrNeverFlooredStrategy`, `_DEFAULT_FLOOR`, `_DEFAULT_FLOOR_WEIGHT` in the existing strategy-import block.

(b) `_build_strategy` — add params + branch:

```python
def _build_strategy(
    key: str,
    *,
    availability: PlayerAvailability,
    n_teams: int,
    strategy_n_sims: int,
    base_seed: int,
    floor: float = _DEFAULT_FLOOR,
    floor_weight: float = _DEFAULT_FLOOR_WEIGHT,
) -> DraftStrategy | None:
    ...
    if key == "now_or_never":
        return NowOrNeverStrategy(LogisticSurvival(sigma=default_sigma(n_teams)))
    if key == "now_or_never_floored":
        return NowOrNeverFlooredStrategy(
            LogisticSurvival(sigma=default_sigma(n_teams)),
            floor=floor,
            floor_weight=floor_weight,
        )
    ...
```

(c) `collect_results` — add `floor: float = _DEFAULT_FLOOR, floor_weight: float = _DEFAULT_FLOOR_WEIGHT` to the signature and pass them into **both** `_build_strategy(strategy_a, ...)` and `_build_strategy(strategy_b, ...)` calls.

(d) `run_backtest` — add `floor: float = _DEFAULT_FLOOR, floor_weight: float = _DEFAULT_FLOOR_WEIGHT` to the signature and forward them to `collect_results`.

- [ ] **Step 4: Wire `backtest/cli.py` and the chunked runner**

In `src/projections/draft/backtest/cli.py`: import `_DEFAULT_FLOOR`, `_DEFAULT_FLOOR_WEIGHT` from `projections.draft.assistant.strategy`; add `--floor`/`--floor-weight` (`type=float`, `default=_DEFAULT_FLOOR` / `_DEFAULT_FLOOR_WEIGHT`) in `_parse_args`, and pass `floor=args.floor, floor_weight=args.floor_weight` into `run_backtest` in `run`.

In `scripts/h2h_backtest_chunked.py`:

(a) `_parse_args` — import `_DEFAULT_FLOOR`, `_DEFAULT_FLOOR_WEIGHT` from `projections.draft.assistant.strategy`; add `--floor` / `--floor-weight` (`type=float`, defaults `_DEFAULT_FLOOR` / `_DEFAULT_FLOOR_WEIGHT`).

(b) `_run_worker` — pass `floor=args.floor, floor_weight=args.floor_weight` into `collect_results`.

(c) Extract the manifest key into a **pure, testable** helper (so the §5 provenance guard is verified, not just the dict-comparing helper). Add to `scripts/h2h_backtest_chunked.py`:

```python
def _run_key(args: argparse.Namespace) -> dict[str, object]:
    """The run identity pinned in the checkpoint manifest. Pure → unit-testable."""
    return {
        "season": args.season,
        "strategy_a": args.strategy_a,
        "strategy_b": args.strategy_b,
        "strategy_n_sims": args.strategy_n_sims,
        "jitter": args.jitter,
        "floor": args.floor,
        "floor_weight": args.floor_weight,
    }
```

Then `_run_driver` calls `verify_or_write_manifest(args.checkpoint_dir, _run_key(args))`.

(d) `_run_driver` — forward both in the worker subprocess `cmd` list (next to `--strategy-a`/`--strategy-b`):

```python
            "--floor", str(args.floor), "--floor-weight", str(args.floor_weight),
```

- [ ] **Step 5: Run to verify pass + default-byte-identical guard**

Run: `pytest tests/test_draft/test_backtest/ tests/test_scripts/test_h2h_backtest_chunked.py -v`
Expected: PASS — including the **existing** chunked-vs-monolithic equivalence test and the default-`(now_or_never, season_value)` aggregation tests (proof the defaulted new params leave the default A/B byte-identical).

- [ ] **Step 6: Lint/type**

Run: `mypy src/projections/draft/backtest/harness.py src/projections/draft/backtest/cli.py scripts/h2h_backtest_chunked.py && ruff check src/projections/draft/backtest/ scripts/h2h_backtest_chunked.py && ruff format --check src/projections/draft/backtest/ scripts/h2h_backtest_chunked.py`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/projections/draft/backtest/harness.py src/projections/draft/backtest/cli.py scripts/h2h_backtest_chunked.py tests/test_draft/
git commit -m "feat(draft): A/B now_or_never_floored in the H2H harness

_build_strategy/collect_results/run_backtest + both CLIs gain floor/
floor_weight (defaulted, so the default nn-vs-sv run is byte-identical).
The chunked runner forwards them to workers and records them in the
checkpoint manifest run_key (resume with a changed floor now fails loud).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Validation A/B run, report, and shipped default

> This task runs the data-dependent H2H A/B and is executed in the environment where the
> 2024/2025 data partitions live (the main checkout, not this worktree's `data/`), in
> **PowerShell** with `KMP_DUPLICATE_LIB_OK=TRUE` + single-thread BLAS, always via the
> chunked runner (memory `h2h-backtest-native-crash`). No new code — it produces numbers,
> a report entry, and the shipped-default update.

- [ ] **Step 1: Confirm data + the strategy is selectable**

Run: `python scripts/draft_assistant.py --help` and confirm `now_or_never_floored` appears in `--strategy` choices and `--floor`/`--floor-weight` exist. Confirm the harness inputs load for both seasons by dry-running `load_inputs` (the harness's own loader, so the path/layout is whatever it actually uses):
`python -c "from projections.draft.backtest.inputs import load_inputs; from projections.draft.league_config import LeagueConfig; c=LeagueConfig.model_validate_json(open('configs/league_espn_half_16team.json').read()); [load_inputs(season=s, config=c, data_root=__import__('pathlib').Path('data')) for s in (2024,2025)]; print('inputs OK 2024+2025')"`
If it raises (missing partitions), ingest the needed season(s) before proceeding. **Use the 16-team config** `configs/league_espn_half_16team.json` — the H2H harness hard-requires `n_teams == 16` (the mirrored seat layout); a 12-team config trips the `collect_results` guard.

- [ ] **Step 2: Coarse screen (cheap, 2025 only)**

For each `(F, λ)` in `F ∈ {0,20,40,60} × λ ∈ {0.5,1,2}`, run a reduced-seed chunked A/B vs `now_or_never` (e.g. `--n-seeds 60`), recording the actual-axis win% / playoff% / champ% paired diff vs `now_or_never` and vs the bot:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"; $env:OMP_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
python scripts/h2h_backtest_chunked.py --season 2025 `
  --league-config configs/league_espn_half_16team.json `
  --n-seeds 60 --strategy-a now_or_never_floored --strategy-b now_or_never `
  --floor 40 --floor-weight 1 --checkpoint-dir _ckpt/floor_F40_L1_2025 --out _ckpt/floor_F40_L1_2025.txt
```

> Use a **distinct `--checkpoint-dir` per `(F, λ, season)`** — the manifest guard rejects reusing a dir across different floor params (by design). Record results in a scratch table. Pick the best 1–2 `(F, λ)`.

- [ ] **Step 3: Full confirmation (best 1–2 configs, both seasons)**

Re-run the chosen `(F, λ)` at `--n-seeds 200 --strategy-n-sims 200` for **2025 and 2024**, paired-bootstrap vs `now_or_never` + bot reference.

- [ ] **Step 4: Write the report entry**

Append a new Test entry to `reports/draft_strategy_tests.md` (follow the Test 7–9 format): the per-strategy / paired-bootstrap actual + projected tables per confirmed `(F, λ)`, the cross-season transfer read, and — per the standing rule — **no adopt/reject verdict**.

- [ ] **Step 5: Set the shipped default**

Update the two constants `_DEFAULT_FLOOR` / `_DEFAULT_FLOOR_WEIGHT` in `strategy.py` to the A/B winner — **one edit**, since every site (dataclass default, `build_session_strategy`, `_build_strategy`, both CLIs) imports them. Re-run `pytest tests/test_draft/test_assistant_strategy.py -k floored` after, since a couple of hand-computed score tests assume specific `(F, λ)` *passed explicitly* (not the default) and so are unaffected; confirm green. If the A/B is inconclusive, leave the constants at `40.0`/`1.0` and say so in the report.

- [ ] **Step 6: Update PM/TODO**

Mark TODO #42 progressed (strategy shipped + A/B logged) and add a `project_management.md` top entry summarizing the slice + the A/B verdict-in-isolation. Commit:

```bash
git add reports/draft_strategy_tests.md project_management.md TODO.md src/projections/draft/assistant/strategy.py src/projections/draft/assistant/live.py src/projections/draft/assistant/cli.py src/projections/draft/backtest/harness.py src/projections/draft/backtest/cli.py
git commit -m "test(draft): H2H A/B for now_or_never_floored + shipped default

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Final verification (run before declaring complete)

Run at repo root and paste output as evidence:

```bash
pytest -v -k "draft or ingest or store or schemas"   # the touched surface + dtype seams
mypy src tests
ruff check src tests
ruff format --check src tests
```

All must pass (Task 4's data-dependent run is the exception — it runs where the data lives).

---

## Self-Review

**Spec coverage:**
- §3 strategy (formula, defaults, `__post_init__` guards, last-pick fallback, `_finalize` tier on) → Task 1. ✓
- §4 reuse `expected_best_by_position` unchanged → Task 1 Step 4 (no helper edit). ✓
- §5 harness registry key + floor/weight threading + manifest provenance → Task 3. ✓
- §6 assistant CLI auto-surface via STRATEGY_KEYS + `build_session_strategy` branch + `BOARD_STRATEGIES` add + not in MC_STRATEGIES → Task 2. ✓
- §7 tests: λ=0≡nn byte-identical, score formula, directed reorder flip, last-pick, construction guards, **explicit null-ADP test** (`test_floored_null_adp_p_available_is_null`, Task 1), harness A/B default byte-identical, **driver-level manifest guard** (`_run_key` test, Task 3) + helper-level guard, CLI/board seam → Tasks 1–3. ✓
- §8 two-stage validation + set default + data dependency → Task 4. ✓
- §2 non-goals (no nn edit, not in MC_STRATEGIES, no board sliders, fixed F) → respected across tasks. ✓

**Placeholder scan:** every code step shows complete code; commands have expected output. No TBD/TODO. ✓

**Type consistency:** `NowOrNeverFlooredStrategy(survival, floor: float, floor_weight: float)` used identically in Tasks 1–3; `build_session_strategy`/`_build_strategy`/`collect_results`/`run_backtest` all gain `floor: float = _DEFAULT_FLOOR, floor_weight: float = _DEFAULT_FLOOR_WEIGHT`; manifest key uses `"floor"`/`"floor_weight"` consistently with the CLI `--floor`/`--floor-weight` (argparse dest `floor`/`floor_weight`). ✓

**Note (DRY):** Task 1 intentionally duplicates ~8 lines of `now_or_never`'s score prelude rather than extracting a shared helper, because the spec keeps `now_or_never` byte-identical as the A/B control. Documented in the class docstring.
