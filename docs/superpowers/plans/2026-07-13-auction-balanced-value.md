# BalancedValueBid (Slice 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-07-13-auction-balanced-value-bid-design.md`

**Goal:** Ship `BalancedValueBid` — an auction bid model that bids a small premium over fair value, capped at `pace ×` the even per-slot share, deliberately without `_budget_urgency` — as the tenth bake-off contestant.

**Architecture:** One new frozen-dataclass strategy behind the existing `AuctionBidStrategy` protocol in `bid_strategy.py`; one line registering it in `tournament_cli._MODELS`; the engine's existing `[min_bid, feasible_max]` clamp handles the floor/reserve. Purely additive — no existing strategy or engine code changes.

**Tech Stack:** Python 3.12, pandas, pandera, pytest, mypy (strict), ruff.

## Global Constraints

- **No existing strategy is modified** — Runs A–F stay byte-identical (additive change only).
- **The hero anchors on `auction_dollars`** (the model SOS value in `view.baseline_dollars`), exactly as `StaticDollarBid` — not the bots' `bot_dollars` seam.
- **`max_bid` returns an unclamped desired bid**; the engine clamps to `[min_bid, feasible_max]`. Do NOT re-implement the reserve or floor.
- **`BalancedValueBid` must NOT apply `_budget_urgency`** — this is the load-bearing design choice.
- **Reference enums, never raw strings** (`RosterSlot.RB`, etc.); nullable-string columns use `_PYARROW_STR`.
- Gates (run at repo root): `pytest -v` (relevant subset ok), `mypy src tests`, `ruff check src tests`, `ruff format --check src tests`.
- Prefix every python invocation with `KMP_DUPLICATE_LIB_OK=TRUE` (dev-box OpenMP workaround).

---

### Task 1: `BalancedValueBid` strategy + unit tests

**Files:**
- Modify: `src/projections/draft/assistant/auction/bid_strategy.py` (add `import math`; add `BalancedValueBid` after `StudsAndDepthBid` at end of file)
- Test: `tests/test_draft/test_assistant_auction_bid_strategy.py` (append tests; reuse the file's existing `_config`/`_pool`/`_baseline`/`_view` helpers and the already-imported `_budget_urgency`)

**Interfaces:**
- Consumes: `AuctionView` (fields `my_budget: int`, `my_open_slots: int`, `baseline_dollars: pd.DataFrame` indexed by gsis_id with an `auction_dollars` column); `LeagueConfig`.
- Produces: `class BalancedValueBid` with fields `premium: float = 0.15`, `pace: float = 2.0` and `max_bid(self, view, player, pool, config) -> int`. Later tasks import it from `projections.draft.assistant.auction.bid_strategy`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_draft/test_assistant_auction_bid_strategy.py`. Add `BalancedValueBid` to the existing `bid_strategy` import block, and `import pytest` at the top if not present.

```python
def test_balanced_premium_wins_contested_value() -> None:
    # mid-tier under the cap -> bid a premium over fair (out-bids a fair-value bidder)
    pool = _pool()
    baseline = _baseline([True, True, False, False], [20, 40, 0, 0])
    view = _view(pool.iloc[:0], budget=100, drafted=set(), baseline=baseline)
    bid = BalancedValueBid(premium=0.15, pace=2.0).max_bid(view, pool.iloc[0], pool, _config())
    assert bid == 23  # round(20 * 1.15); cap = 2*(100/3) = 66.7 does not bind
    assert bid > 20  # strictly above fair value


def test_balanced_cap_forces_spread_on_studs() -> None:
    # a stud whose premium'd value exceeds the pace cap -> bid the cap (< fair)
    pool = _pool()
    baseline = _baseline([True, True, False, False], [80, 40, 0, 0])
    view = _view(pool.iloc[:0], budget=100, drafted=set(), baseline=baseline)
    bid = BalancedValueBid(premium=0.15, pace=2.0).max_bid(view, pool.iloc[0], pool, _config())
    assert bid == 67  # round(2 * 100/3) = round(66.67); 80*1.15=92 does not win
    assert bid < 80  # strictly below fair value (capped)


def test_balanced_cap_tracks_remaining_budget() -> None:
    pool = _pool()
    baseline = _baseline([True, True, False, False], [80, 40, 0, 0])
    strat = BalancedValueBid(premium=0.15, pace=2.0)
    rich = strat.max_bid(_view(pool.iloc[:0], budget=100, drafted=set(), baseline=baseline),
                         pool.iloc[0], pool, _config())
    poor = strat.max_bid(_view(pool.iloc[:0], budget=30, drafted=set(), baseline=baseline),
                         pool.iloc[0], pool, _config())
    assert poor < rich  # cap shrinks with budget: 2*(30/3)=20 vs 2*(100/3)=67


def test_balanced_does_not_apply_urgency() -> None:
    # partial roster + idle cash => _budget_urgency > 1; the bid must NOT be inflated by it.
    pool = _pool()
    baseline = _baseline([True, True, False, False], [20, 40, 0, 0])
    view = _view(pool.iloc[[2]], budget=100, drafted={"00-0000003"}, baseline=baseline)  # 1 held -> 2 open
    config = _config()
    urgency = _budget_urgency(view, config)
    assert urgency > 1.5  # sanity: this state carries a real urgency ramp
    bid = BalancedValueBid(premium=0.15, pace=2.0).max_bid(view, pool.iloc[0], pool, config)
    assert bid == 23  # round(20 * 1.15); cap = 2*(100/2) = 100 does not bind
    assert bid < round(23 * urgency)  # NOT multiplied by the urgency ramp


def test_balanced_is_deterministic() -> None:
    pool = _pool()
    baseline = _baseline([True, True, False, False], [20, 40, 0, 0])
    view = _view(pool.iloc[:0], budget=100, drafted=set(), baseline=baseline)
    strat, config = BalancedValueBid(), _config()
    assert strat.max_bid(view, pool.iloc[0], pool, config) == strat.max_bid(view, pool.iloc[0], pool, config)


def test_balanced_rejects_bad_tuning() -> None:
    with pytest.raises(ValueError):
        BalancedValueBid(premium=-0.1)
    with pytest.raises(ValueError):
        BalancedValueBid(pace=0.0)
    with pytest.raises(ValueError):
        BalancedValueBid(pace=float("inf"))
    with pytest.raises(ValueError):
        BalancedValueBid(premium=float("nan"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `KMP_DUPLICATE_LIB_OK=TRUE pytest tests/test_draft/test_assistant_auction_bid_strategy.py -k balanced -v`
Expected: FAIL — `ImportError: cannot import name 'BalancedValueBid'` (collection error).

- [ ] **Step 3: Add the `import math` and the strategy**

In `src/projections/draft/assistant/auction/bid_strategy.py`, add `import math` to the stdlib import group (ruff will order it). Append at end of file:

```python
@dataclass(frozen=True)
class BalancedValueBid:
    """Balanced-breadth hero: bid a small premium over fair value to win contested players,
    capped at `pace` x the even per-slot share so the budget spreads into a full roster.
    Deliberately does NOT apply _budget_urgency (the ramp over-pays late-round scrubs)."""

    premium: float = 0.15
    pace: float = 2.0

    def __post_init__(self) -> None:
        if not (self.premium >= 0.0 and math.isfinite(self.premium)):
            raise ValueError(f"premium must be finite and >= 0; got {self.premium}")
        if not (self.pace > 0.0 and math.isfinite(self.pace)):
            raise ValueError(f"pace must be finite and > 0; got {self.pace}")

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        fair = float(view.baseline_dollars.loc[player["gsis_id"], "auction_dollars"])
        cap = self.pace * (view.my_budget / max(1, view.my_open_slots))
        return round(min(fair * (1.0 + self.premium), cap))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `KMP_DUPLICATE_LIB_OK=TRUE pytest tests/test_draft/test_assistant_auction_bid_strategy.py -k balanced -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Lint + type-check the touched files**

Run: `KMP_DUPLICATE_LIB_OK=TRUE ruff check src tests && ruff format src tests && mypy src/projections/draft/assistant/auction/ tests/test_draft/test_assistant_auction_bid_strategy.py`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/projections/draft/assistant/auction/bid_strategy.py tests/test_draft/test_assistant_auction_bid_strategy.py
git commit -m "feat(auction): BalancedValueBid strategy (premium + pace cap, no urgency)"
```

---

### Task 2: Register `balanced` + update the contestant-set guard + tournament smoke

**Files:**
- Modify: `src/projections/draft/assistant/auction/tournament_cli.py` (import `BalancedValueBid`; add `"balanced": BalancedValueBid()` to `_MODELS`)
- Test: `tests/test_draft/test_assistant_auction_tournament_cli.py` (update `test_default_models_are_the_nine_contestants` → ten)
- Test: `tests/test_draft/test_assistant_auction_tournament.py` (add a smoke that `balanced` races and appears in `summaries`; reuse its `_config`/`_pool`/`_avail` helpers)

**Interfaces:**
- Consumes: `BalancedValueBid` (Task 1); `_MODELS` (dict[str, AuctionBidStrategy]); `run_auction_tournament`.
- Produces: `_MODELS` now contains key `"balanced"`.

- [ ] **Step 1: Write the failing guard-test update + tournament smoke**

In `tests/test_draft/test_assistant_auction_tournament_cli.py`, rename and extend the guard test:

```python
def test_default_models_are_the_ten_contestants() -> None:
    assert set(_MODELS) == {
        "static",
        "inflation",
        "marginal",
        "anchors",
        "overbid",
        "vorpshare",
        "patient",
        "patient_deep",
        "studsdepth",
        "balanced",
    }
```

In `tests/test_draft/test_assistant_auction_tournament.py`, add (import `BalancedValueBid` from `bid_strategy` at the top):

```python
def test_balanced_contestant_races_and_is_scored() -> None:
    pool = _pool(40)
    cfg = _config(6)
    result = run_auction_tournament(
        {"balanced": BalancedValueBid()},
        pool,
        cfg,
        my_seat=1,
        n_seeds=4,
        price_jitter=0.1,
        base_seed=0,
        n_sims=50,
        availability=_avail(pool),
        params=VarianceParams.load(),
    )
    assert set(result.summaries["balanced"]) == set(METRICS)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `KMP_DUPLICATE_LIB_OK=TRUE pytest tests/test_draft/test_assistant_auction_tournament_cli.py::test_default_models_are_the_ten_contestants tests/test_draft/test_assistant_auction_tournament.py::test_balanced_contestant_races_and_is_scored -v`
Expected: FAIL — guard test: `AssertionError` (`balanced` missing from `_MODELS`); smoke: `ImportError` or `KeyError 'balanced'`.

- [ ] **Step 3: Register the contestant**

In `src/projections/draft/assistant/auction/tournament_cli.py`, add `BalancedValueBid` to the `bid_strategy` import block and add the entry to `_MODELS` (after `"studsdepth"`):

```python
    "studsdepth": StudsAndDepthBid(),
    "balanced": BalancedValueBid(),
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `KMP_DUPLICATE_LIB_OK=TRUE pytest tests/test_draft/test_assistant_auction_tournament_cli.py tests/test_draft/test_assistant_auction_tournament.py -v`
Expected: PASS (incl. the existing `test_every_default_model_satisfies_the_protocol`, which now also covers `balanced`).

- [ ] **Step 5: Lint + type-check**

Run: `KMP_DUPLICATE_LIB_OK=TRUE ruff check src tests && ruff format src tests && mypy src/projections/draft/assistant/auction/ tests/test_draft/test_assistant_auction_tournament_cli.py tests/test_draft/test_assistant_auction_tournament.py`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/projections/draft/assistant/auction/tournament_cli.py tests/test_draft/test_assistant_auction_tournament_cli.py tests/test_draft/test_assistant_auction_tournament.py
git commit -m "feat(auction): register balanced as the tenth bake-off contestant"
```

---

### Task 3: Run G — bake-off of the shipped contestant + record in the report

**Files:**
- Modify: `reports/auction_tournament_validation_2026.md` (append a "Run G" section)

**Interfaces:** none (validation + documentation task).

- [ ] **Step 1: Run the shipped bake-off (12-team half, model prices)**

The pools/configs are untracked artifacts; regenerate if absent: `KMP_DUPLICATE_LIB_OK=TRUE python scripts/generate_preset_vorp_tables.py --season 2026`.

Run (≈7 min):
```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/auction_tournament.py \
  --vorp-table data/vorp_2026/half_12team.parquet \
  --league-config data/vorp_2026/half_12team.league.json \
  --my-seat 1 --season 2026 --seeds 150 --n-sims 300 --seed 0 \
  --bot-prices model --nomination-temp 1.0 \
  compare
```
Capture the printed per-model table (it is the Run G data recorded in Step 3).

- [ ] **Step 2: Verify the pre-registered expectation**

Confirm from the output: `balanced` is the top hero on playoff% (≈ 0.45–0.46) / champ% (≈ 0.05–0.06), CI-separated above `vorpshare`, and above `patient_deep`; `anchors` last. If `balanced` lands materially off this (e.g. below `vorpshare`), STOP and reconcile — the shipped strategy diverged from the prototype; do not record until it matches.

- [ ] **Step 3: Append Run G to the report**

Add a `## Run G — 2026-07-13` section to `reports/auction_tournament_validation_2026.md` with: the exact flags above; the per-model `mean_points`/`playoff%`/`champ%` table from the run; a one-line reading ("`balanced` (premium 0.15 / pace 2×, no urgency) is the top hero, CI-separated on playoff%; `anchors` last; **no winner declared** — Sept decision"). Note the market is model-priced because the 2026 snapshot has no ESPN auction values.

- [ ] **Step 4: Commit**

```bash
git add reports/auction_tournament_validation_2026.md
git commit -m "data(auction): Run G — balanced contestant tops the 12-team field (no winner)"
```

---

## Final verification (after all tasks)

Run the full gates once:
```bash
KMP_DUPLICATE_LIB_OK=TRUE pytest -q          # only the known non-branch failures may remain
KMP_DUPLICATE_LIB_OK=TRUE mypy src tests
KMP_DUPLICATE_LIB_OK=TRUE ruff check src tests
KMP_DUPLICATE_LIB_OK=TRUE ruff format --check src tests
```
Known pre-existing full-suite non-failures (not from this work): `tests/backtest/test_backtest_smoke.py::test_backtest_smoke_one_cell` and the `test_draft_board_smoke.py` AppTest parallel-timeout flake (passes in isolation).
