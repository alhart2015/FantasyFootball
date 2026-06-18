# Auction stars-and-scrubs bidders + position-aware hero — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-06-17-auction-stars-and-scrubs-design.md`

**Goal:** Add three budget-spending stars-and-scrubs bid models and gate the hero with the same positional rules the bots use, then race all six models in the bake-off.

**Architecture:** Three new frozen-dataclass bid models in `bid_strategy.py` (consuming VORP from the pool and dollars from `baseline_dollars`); a one-spot generalization in `simulation.py::_simulate_to_state` that gates every open seat — hero included — through `bot_eligible`; the CLI's default model set grows to six. Bots, snake field, market clearing, and the scorer are untouched.

**Tech Stack:** Python 3, pandas (pyarrow-backed string dtype), pandera schemas, numpy RNG, pytest, mypy (strict), ruff.

## Global Constraints

(Every task implicitly includes these — verbatim from the spec.)
- `GsisId` canonical; reference enums (`Position`, `RosterSlot`) never the raw strings.
- `pd.StringDtype("pyarrow")` (`_PYARROW_STR`) for gsis columns; `Int64`/`pd.NA` rules unchanged.
- `df = SCHEMA.validate(df)` (with reassignment) at any boundary that emits a frame. **No new pandera schema** in this slice.
- `store.read_partition`/`write_partition` only; no `df.to_parquet` outside the store.
- Bid models keep the `max_bid(view, player, pool, config) -> int` signature; **no model implements its own position gate** — the engine owns legality.
- Bots (`market.py::bot_max_bid`, `resolve_bids`) and the snake field (`backtest/draft_field.py`) are **byte-identical / untouched**.
- Determinism: hero bids are a pure function of state (no RNG); only the bot market consumes `rng`.
- Gates per task: `pytest -v <touched>`, `mypy src tests`, `ruff check src tests`, `ruff format --check src tests`.

---

### Task 1: Three stars-and-scrubs bid models + shared VORP helpers

**Files:**
- Modify: `src/projections/draft/assistant/auction/bid_strategy.py`
- Test: `tests/test_draft/test_assistant_auction_bid_strategy.py`

**Interfaces:**
- Consumes: `AuctionView` (fields `my_budget: int`, `my_open_slots: int`, `my_roster: pd.DataFrame` [pool rows for the hero, carries `vorp`], `drafted: frozenset[str]`, `baseline_dollars: pd.DataFrame` [indexed by gsis_id, has `auction_dollars`]); `LeagueConfig` (`min_bid`, `n_teams`); the pool frame (`VorpTableSchema`: `gsis_id`, `position`, `season_mean_fpts`, `vorp`).
- Produces: `AnchorBudgetBid(n_anchors: int = 4)`, `OverbidValueBid(k: float = 1.3, stud_count: int | None = None)`, `VorpShareBid()` — each a frozen dataclass with `max_bid(view, player, pool, config) -> int`; module helpers `_undrafted(pool, drafted) -> pd.DataFrame` and `_vorp_threshold(pool, k) -> float`.

- [ ] **Step 1: Write failing tests for the helpers and the three models.**

Append to `tests/test_draft/test_assistant_auction_bid_strategy.py` (the existing `_config`/`_pool`/`_baseline`/`_view` helpers stay; add the vorp-bearing fixtures and tests below):

```python
import pytest

from projections.draft.assistant.auction.bid_strategy import (
    AnchorBudgetBid,
    OverbidValueBid,
    VorpShareBid,
    _undrafted,
    _vorp_threshold,
)
from projections.schemas import Position  # noqa: F401  (kept for parity; enums over strings)


def _vpool() -> pd.DataFrame:
    # vorp strictly descending by row: 120,110,90,20,10,5
    return pd.DataFrame(
        {
            "gsis_id": pd.array(
                ["00-0000001", "00-0000002", "00-0000003", "00-0000004", "00-0000005", "00-0000006"],
                dtype=_PYARROW_STR,
            ),
            "position": pd.array(["RB", "WR", "QB", "RB", "WR", "TE"], dtype=_PYARROW_STR),
            "season_mean_fpts": [250.0, 240.0, 280.0, 120.0, 110.0, 100.0],
            "vorp": [120.0, 110.0, 90.0, 20.0, 10.0, 5.0],
        }
    )


def _vbaseline() -> pd.DataFrame:
    return pd.DataFrame(
        {"in_pool": [True] * 6, "auction_dollars": [30, 28, 25, 5, 3, 2]},
        index=pd.Index(
            ["00-0000001", "00-0000002", "00-0000003", "00-0000004", "00-0000005", "00-0000006"],
            name="gsis_id",
        ),
    )


def _vconfig() -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=2,
        budget=100,
        min_bid=1,
        roster_slots={
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.QB: 1,
            RosterSlot.BENCH: 2,
        },
        ruleset=Ruleset.espn_ppr(),
    )


def _aview(pool: pd.DataFrame, *, my_ids: tuple[str, ...] = (), budget: int = 100,
           open_slots: int = 8, drafted: tuple[str, ...] = ()) -> AuctionView:
    my_roster = pool[pool["gsis_id"].isin(list(my_ids))]
    return AuctionView(
        my_budget=budget,
        my_open_slots=open_slots,
        my_positions=Counter(),
        my_roster=my_roster,
        drafted=frozenset(drafted),
        budgets_by_seat=(budget, budget),
        baseline_dollars=_vbaseline(),
    )


def test_vorp_threshold_kth_highest() -> None:
    pool = _vpool()
    assert _vorp_threshold(pool, 1) == 120.0
    assert _vorp_threshold(pool, 3) == 90.0
    assert _vorp_threshold(pool, 4) == 20.0


def test_vorp_threshold_pool_smaller_than_k_returns_min() -> None:
    assert _vorp_threshold(_vpool(), 8) == 5.0  # len 6 <= 8 -> pool min (all anchor-grade)


def test_vorp_threshold_nonpositive_k_is_inf() -> None:
    assert _vorp_threshold(_vpool(), 0) == float("inf")


def test_undrafted_filters_drafted_ids() -> None:
    pool = _vpool()
    assert len(_undrafted(pool, frozenset())) == 6
    left = _undrafted(pool, frozenset({"00-0000001"}))
    assert len(left) == 5 and "00-0000001" not in {str(g) for g in left["gsis_id"]}


def test_anchor_bids_above_market_for_a_top_vorp_player() -> None:
    # n_anchors=2, n_teams=2 -> league_anchor_count 4 -> threshold = 4th vorp = 20.
    # Empty roster, budget 100, open 8: reserve=1*(8-2)=6, cap=(100-6)/2=47.
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=8)
    bid = AnchorBudgetBid(n_anchors=2).max_bid(view, pool.iloc[0], pool, _vconfig())
    assert bid == 47
    assert bid > 30  # overbids the $30 market value to actually win the anchor


def test_anchor_bids_min_for_a_scrub() -> None:
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=8)
    bid = AnchorBudgetBid(n_anchors=2).max_bid(view, pool.iloc[4], pool, _vconfig())  # vorp 10 < 20
    assert bid == _vconfig().min_bid


def test_anchor_switches_to_scrubs_once_anchors_held() -> None:
    # Hero already holds 2 anchors (vorp 120,110 >= 20); n_anchors=2 -> anchors_remaining 0.
    pool = _vpool()
    view = _aview(pool, my_ids=("00-0000001", "00-0000002"), budget=60, open_slots=6,
                  drafted=("00-0000001", "00-0000002"))
    bid = AnchorBudgetBid(n_anchors=2).max_bid(view, pool.iloc[2], pool, _vconfig())  # anchor-grade
    assert bid == _vconfig().min_bid


def test_overbid_pays_up_for_studs_value_for_others() -> None:
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=8)
    cfg = _vconfig()
    strat = OverbidValueBid(k=1.5, stud_count=3)  # threshold = 3rd vorp = 90
    assert strat.max_bid(view, pool.iloc[0], pool, cfg) == round(30 * 1.5)  # stud (vorp 120) -> 45
    assert strat.max_bid(view, pool.iloc[3], pool, cfg) == 5  # non-stud (vorp 20) -> value 5


def test_overbid_default_stud_count_is_three_times_teams() -> None:
    assert OverbidValueBid().stud_count is None  # resolved to 3*n_teams at call time


def test_vorpshare_concentrates_on_top_targets() -> None:
    # Empty roster, open 8 -> targets = all 6 undrafted. denom = 120+110+90+20+10+5 = 355.
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=8)
    cfg = _vconfig()
    assert VorpShareBid().max_bid(view, pool.iloc[0], pool, cfg) == round(100 * 120 / 355)  # 34
    assert VorpShareBid().max_bid(view, pool.iloc[5], pool, cfg) == cfg.min_bid  # vorp 5 -> ~1


def test_vorpshare_off_target_player_bids_min() -> None:
    # open 2 -> targets = top-2 vorp (players 1,2). Player 3 (vorp 90) is off-target.
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=2)
    assert VorpShareBid().max_bid(view, pool.iloc[2], pool, _vconfig()) == _vconfig().min_bid


def test_vorpshare_zero_target_vorp_bids_min() -> None:
    pool = pd.DataFrame(
        {
            "gsis_id": pd.array(["00-0000007", "00-0000008"], dtype=_PYARROW_STR),
            "position": pd.array(["RB", "WR"], dtype=_PYARROW_STR),
            "season_mean_fpts": [80.0, 70.0],
            "vorp": [0.0, -5.0],
        }
    )
    view = AuctionView(
        my_budget=100, my_open_slots=2, my_positions=Counter(), my_roster=pool.iloc[:0],
        drafted=frozenset(), budgets_by_seat=(100, 100),
        baseline_dollars=pd.DataFrame(
            {"in_pool": [True, True], "auction_dollars": [1, 1]},
            index=pd.Index(["00-0000007", "00-0000008"], name="gsis_id"),
        ),
    )
    assert VorpShareBid().max_bid(view, pool.iloc[0], pool, _vconfig()) == _vconfig().min_bid
```

- [ ] **Step 2: Run the new tests to verify they fail.**

Run: `python -m pytest tests/test_draft/test_assistant_auction_bid_strategy.py -k "vorp or anchor or overbid or vorpshare or undrafted" -n0 -q`
Expected: FAIL/ERROR — `ImportError: cannot import name 'AnchorBudgetBid'` (and the helpers/models are undefined).

- [ ] **Step 3: Implement the helpers and three models in `bid_strategy.py`.**

Append after the existing `MarginalValueBid` class (the module already imports `pd`, `dataclass`, `Counter`, `LeagueConfig`):

```python
def _undrafted(pool: pd.DataFrame, drafted: frozenset[str]) -> pd.DataFrame:
    """Pool rows whose gsis_id is not yet drafted (same isin pattern as InflationBid)."""
    return pool[~pool["gsis_id"].isin(drafted)]


def _vorp_threshold(pool: pd.DataFrame, k: int) -> float:
    """The k-th highest `vorp` in the pool — the cutoff for 'top-k by VORP'. If the pool has
    fewer than k players, the pool minimum (so every player clears the bar). k<=0 -> +inf
    (nothing clears)."""
    if k <= 0:
        return float("inf")
    vorps = pool["vorp"]
    if len(vorps) <= k:
        return float(vorps.min())
    return float(vorps.nlargest(k).iloc[-1])


@dataclass(frozen=True)
class AnchorBudgetBid:
    """Stars-and-scrubs: pour the budget into `n_anchors` top-VORP players, $1 the rest (spec §B)."""

    n_anchors: int = 4

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        min_bid = config.min_bid
        threshold = _vorp_threshold(pool, self.n_anchors * config.n_teams)
        anchors_held = int((view.my_roster["vorp"] >= threshold).sum())
        anchors_remaining = max(0, self.n_anchors - anchors_held)
        open_slots = view.my_open_slots
        feasible_max = view.my_budget - min_bid * (open_slots - 1)
        if float(player["vorp"]) >= threshold and anchors_remaining > 0:
            reserve = min_bid * max(0, open_slots - anchors_remaining)
            cap = (view.my_budget - reserve) / anchors_remaining
            return round(min(cap, float(feasible_max)))
        return min_bid


@dataclass(frozen=True)
class OverbidValueBid:
    """Pay up for studs (top-VORP), plain value otherwise; the engine clamp handles broke (spec §B)."""

    k: float = 1.3
    stud_count: int | None = None

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        stud_count = self.stud_count if self.stud_count is not None else 3 * config.n_teams
        threshold = _vorp_threshold(pool, stud_count)
        value = int(view.baseline_dollars.loc[player["gsis_id"], "auction_dollars"])
        if float(player["vorp"]) >= threshold:
            return round(value * self.k)
        return value


@dataclass(frozen=True)
class VorpShareBid:
    """Allocate the remaining budget proportionally to VORP across the top-`open_slots` targets."""

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        min_bid = config.min_bid
        targets = _undrafted(pool, view.drafted).nlargest(view.my_open_slots, "vorp")
        if str(player["gsis_id"]) not in {str(g) for g in targets["gsis_id"]}:
            return min_bid
        denom = float(targets["vorp"].clip(lower=0.0).sum())
        if denom <= 0.0:
            return min_bid
        share = max(0.0, float(player["vorp"])) / denom
        return round(view.my_budget * share)
```

- [ ] **Step 4: Run the new tests to verify they pass.**

Run: `python -m pytest tests/test_draft/test_assistant_auction_bid_strategy.py -n0 -q`
Expected: PASS (existing + new tests).

- [ ] **Step 5: Lint/type the file, then commit.**

Run: `python -m ruff check src/projections/draft/assistant/auction/bid_strategy.py && python -m ruff format --check src/projections/draft/assistant/auction/bid_strategy.py && python -m mypy src/projections/draft/assistant/auction/bid_strategy.py`
Expected: clean.

```bash
git add src/projections/draft/assistant/auction/bid_strategy.py tests/test_draft/test_assistant_auction_bid_strategy.py
git commit -m "feat(auction): AnchorBudgetBid / OverbidValueBid / VorpShareBid stars-and-scrubs models"
```

---

### Task 2: Gate the hero like a bot in `_simulate_to_state`

**Files:**
- Modify: `src/projections/draft/assistant/auction/simulation.py` (eligibility-build loop ~lines 100-118; bid loop ~lines 138-161)
- Test: `tests/test_draft/test_assistant_auction_simulation.py`

**Interfaces:**
- Consumes: `bot_eligible`, `bot_position_bounds` (already imported), `pos_by_id` (built ~line 87), `seat_eligible`, `all_positions`, `_feasible_max`, `_build_view`, `bot_max_bid`, `SeatView` (all in scope).
- Produces: no new public symbols; the hero seat now obeys `bot_eligible`. `AnchorBudgetBid` is consumed from Task 1.

- [ ] **Step 1: Replace `test_hero_is_not_gated` with `test_hero_is_gated_like_a_bot` and add gate/determinism tests.**

In `tests/test_draft/test_assistant_auction_simulation.py`: **delete** `test_hero_is_not_gated` (lines ~316-338) and add the following. Also add `AnchorBudgetBid` to the bid_strategy import at the top (`from ...auction.bid_strategy import AnchorBudgetBid, AuctionView, StaticDollarBid`):

```python
def test_hero_is_gated_like_a_bot() -> None:
    # SUPERSEDES test_hero_is_not_gated (sane-bots slice, spec R7a). That slice deliberately left
    # the hero ungated to isolate the bid model; this slice (Goal 1) gates it with the SAME
    # bot_eligible/bot_position_bounds rule. Same cheap-RB pool: pre-gate the max-bidding hero took
    # 3 RB / 0 WR; gated it must stop at the RB max (2 for {RB:1,WR:1,BENCH:1}) and reserve its WR.
    cfg = _config(n_teams=2, roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1})
    pool = _thin_pool(_RB5_WR1, _RB5_WR1_POS)
    baseline = _bd(_RB5_WR1, [10, 9, 8, 7, 6, 5])
    state = _simulate_to_state(
        _MaxBidStub(), 1, pool, cfg,
        baseline_dollars=baseline, price_jitter=0.0, rng=np.random.default_rng(0),
    )
    hero = [p for (_g, p, _pr) in state.rosters[0]]
    assert hero.count("RB") <= 2  # gated: never exceeds the RB max
    assert hero.count("WR") >= 1  # gated: reserves the WR starter (no empty starting slot)


def test_gated_anchor_hero_builds_a_startable_roster() -> None:
    # AnchorBudgetBid through the full engine: respects the gate (startable, within max) and still
    # concentrates budget (pays up for an anchor instead of spreading $1s). _pool carries vorp.
    cfg = _config(n_teams=4, roster_slots={RosterSlot.RB: 2, RosterSlot.WR: 2, RosterSlot.BENCH: 2})
    pool = _pool(40)  # RB/WR only; need >= 4*6 = 24 players
    baseline = _baseline(pool, cfg)
    state = _simulate_to_state(
        AnchorBudgetBid(), 1, pool, cfg,
        baseline_dollars=baseline, price_jitter=0.0, rng=np.random.default_rng(0),
    )
    hero = [p for (_g, p, _pr) in state.rosters[0]]
    prices = [pr for (_g, _p, pr) in state.rosters[0]]
    assert len(hero) == cfg.roster_size                      # filled
    assert hero.count("RB") <= 3 and hero.count("WR") <= 3   # within the gate max (min+bench share)
    assert hero.count("RB") >= 2 and hero.count("WR") >= 2   # minimum starters reserved
    assert max(prices) >= 10                                 # paid up for an anchor, didn't spread


def test_gated_hero_is_deterministic() -> None:
    cfg = _config(n_teams=4, roster_slots={RosterSlot.RB: 2, RosterSlot.WR: 2, RosterSlot.BENCH: 2})
    pool = _pool(40)
    baseline = _baseline(pool, cfg)
    kw = dict(baseline_dollars=baseline, price_jitter=0.15)
    a = _simulate_to_state(AnchorBudgetBid(), 1, pool, cfg, rng=np.random.default_rng(7), **kw)
    b = _simulate_to_state(AnchorBudgetBid(), 1, pool, cfg, rng=np.random.default_rng(7), **kw)
    assert a.rosters == b.rosters  # same seed -> identical draft
```

- [ ] **Step 2: Run to verify the new tests fail (and the old one is gone).**

Run: `python -m pytest tests/test_draft/test_assistant_auction_simulation.py -k "gated or hero_is" -n0 -q`
Expected: `test_hero_is_gated_like_a_bot` FAILS (pre-gate the hero takes 3 RB / 0 WR, so `hero.count("WR") >= 1` fails); the anchor/determinism tests pass or fail depending on import, but the gate test is the red one. (`test_hero_is_not_gated` no longer exists.)

- [ ] **Step 3: Generalize the eligibility-build loop — every open seat uses `bot_eligible`.**

In `simulation.py`, replace the eligibility-build block (the `for seat in range(n):` loop that special-cases `seat == hero0` with `all_positions`). New block (drop the hero branch; update the comment):

```python
        # eligible positions per OPEN seat — hero included — via the SAME bot rule (spec R1); union
        seat_eligible: dict[int, frozenset[Position]] = {}
        union: set[Position] = set()
        for seat in range(n):
            if _open_slots(state, seat, rs) <= 0:
                continue
            counts = {
                Position(p): c
                for p, c in Counter(p for (_g, p, _pr) in state.rosters[seat]).items()
            }
            elig = bot_eligible(
                counts, _open_slots(state, seat, rs), minimums=minimums, maximums=maximums
            )
            seat_eligible[seat] = elig
            union |= elig
```

- [ ] **Step 4: Add the hero position-gate in the bid loop.**

In `simulation.py`, replace the bid-collection loop body so `elig` is computed once for every seat and the hero abstains on an ineligible position (update the leading comment too):

```python
        # collect bids: hero and bots alike abstain (dropped) on an ineligible position unless forced
        bids: dict[int, int] = {}
        for seat in range(n):
            if _open_slots(state, seat, rs) <= 0:
                continue
            fmax = _feasible_max(state, seat, rs, min_bid)
            elig = all_positions if forced else seat_eligible[seat]
            if seat == hero0:
                if pos_by_id[str(nominee_id)] not in elig:
                    continue  # hero is now gated like a bot (spec R2)
                desired = strategy.max_bid(
                    _build_view(state, hero0, pool, bd, config), player, pool, config
                )
                bids[seat] = max(min_bid, min(int(desired), fmax))
            else:
                desired = bot_max_bid(
                    SeatView(open_slots=_open_slots(state, seat, rs), eligible_positions=elig),
                    player,
                    bd,
                    config,
                    rng,
                    price_jitter=price_jitter,
                )
                if desired <= 0:  # abstain -> dropped before the clamp
                    continue
                bids[seat] = max(min_bid, min(int(desired), fmax))
```

- [ ] **Step 5: Run the simulation tests to verify they pass.**

Run: `python -m pytest tests/test_draft/test_assistant_auction_simulation.py -n0 -q`
Expected: PASS — `test_hero_is_gated_like_a_bot`, the anchor-startable, and determinism tests green; the forced-pick + every-bot-startable tests still green (completeness preserved, spec R3).

- [ ] **Step 6: Lint/type, then commit.**

Run: `python -m ruff check src/projections/draft/assistant/auction/simulation.py && python -m ruff format --check src/projections/draft/assistant/auction/simulation.py && python -m mypy src/projections/draft/assistant/auction/simulation.py`
Expected: clean.

```bash
git add src/projections/draft/assistant/auction/simulation.py tests/test_draft/test_assistant_auction_simulation.py
git commit -m "feat(auction): gate the hero like a bot (position-aware bidders); supersede hero-ungated test"
```

---

### Task 3: Race all six models in the CLI `compare`

**Files:**
- Modify: `src/projections/draft/assistant/auction/tournament_cli.py` (`_MODELS` ~lines 33-37; imports ~lines 15-20; module + subparser docstrings)
- Test: `tests/test_draft/test_assistant_auction_tournament_cli.py`

**Interfaces:**
- Consumes: `AnchorBudgetBid`, `OverbidValueBid`, `VorpShareBid` (Task 1); `AuctionBidStrategy`, `run_auction_tournament` (existing).
- Produces: `_MODELS` now has six entries; no new CLI flags.

- [ ] **Step 1: Write the failing test for the six-model default set.**

Add to `tests/test_draft/test_assistant_auction_tournament_cli.py`:

```python
from projections.draft.assistant.auction.bid_strategy import AuctionBidStrategy
from projections.draft.assistant.auction.tournament_cli import _MODELS


def test_default_models_are_the_six_contestants() -> None:
    assert set(_MODELS) == {"static", "inflation", "marginal", "anchors", "overbid", "vorpshare"}


def test_every_default_model_satisfies_the_protocol() -> None:
    assert all(isinstance(m, AuctionBidStrategy) for m in _MODELS.values())
```

- [ ] **Step 2: Run to verify it fails.**

Run: `python -m pytest tests/test_draft/test_assistant_auction_tournament_cli.py -k "six or protocol" -n0 -q`
Expected: FAIL — `_MODELS` has only three keys.

- [ ] **Step 3: Add the three models to `_MODELS`.**

In `tournament_cli.py`, extend the import and the dict:

```python
from projections.draft.assistant.auction.bid_strategy import (
    AnchorBudgetBid,
    AuctionBidStrategy,
    InflationBid,
    MarginalValueBid,
    OverbidValueBid,
    StaticDollarBid,
    VorpShareBid,
)
```

```python
_MODELS: dict[str, AuctionBidStrategy] = {
    "static": StaticDollarBid(),
    "inflation": InflationBid(),
    "marginal": MarginalValueBid(),
    "anchors": AnchorBudgetBid(),
    "overbid": OverbidValueBid(),
    "vorpshare": VorpShareBid(),
}
```

Update the module docstring line "races static/inflation/marginal" → "races the six bid models" and the `compare` subparser help → `"Race the six bid models; record per-metric data."`

- [ ] **Step 4: Run the CLI tests to verify they pass.**

Run: `python -m pytest tests/test_draft/test_assistant_auction_tournament_cli.py -n0 -q`
Expected: PASS.

- [ ] **Step 5: Lint/type, then commit.**

Run: `python -m ruff check src/projections/draft/assistant/auction/tournament_cli.py && python -m ruff format --check src/projections/draft/assistant/auction/tournament_cli.py && python -m mypy src/projections/draft/assistant/auction/tournament_cli.py`
Expected: clean.

```bash
git add src/projections/draft/assistant/auction/tournament_cli.py tests/test_draft/test_assistant_auction_tournament_cli.py
git commit -m "feat(auction): six-model default bake-off in the compare CLI"
```

---

### Task 4: Run the six-model bake-off and record Run C (data, no winner)

**Files:**
- Modify: `reports/auction_tournament_validation_2026.md`

This is a data/verification task — no TDD. **Run chunked** (the i9-14900KF Raptor Lake fault segfaults at `n_sims=500` across many seeds — see memory `h2h-backtest-native-crash`).

- [ ] **Step 1: Full gates before the run.**

Run: `python -m pytest tests/test_draft -n0 -q && python -m mypy src tests && python -m ruff check src tests && python -m ruff format --check src tests`
Expected: green except the known pre-existing `test_backtest_smoke_one_cell` (TODO #40) — note it, don't fix it here.

- [ ] **Step 2: Run the six-model bake-off (chunked).**

Run (start at a low seed count to confirm it completes, then scale; chunk the seed range if segfaults appear):
```bash
python scripts/auction_tournament.py \
  --vorp-table data/vorp_2026/half_16team.parquet \
  --league-config configs/league_espn_half_16team.json \
  --my-seat 1 --season 2026 --seeds 150 --n-sims 300 --seed 0 \
  compare
```
Expected: a per-model table (six rows) + paired diffs. If it segfaults, lower `--n-sims` / split `--seeds` into batches by varying `--seed` and average — record the exact invocation used.

- [ ] **Step 3: Record Run C in the tracking doc.**

Append a "Run C — six models, position-aware hero" section to `reports/auction_tournament_validation_2026.md`: the six per-model rows (exp pts + CI, win/playoff/bye/champ%), the paired diffs, the exact flags, and the §C caveat (gating the hero changes RNG consumption, so Run C absolute levels are **not** level-comparable to Runs A/B — only same-run paired diffs are interpretable). State plainly what the run favored **in isolation** and that **no winner is declared** (September decision).

- [ ] **Step 4: Commit.**

```bash
git add reports/auction_tournament_validation_2026.md
git commit -m "data(auction): Run C — six-model bake-off with position-aware hero (no winner)"
```

---

## Self-Review

**Spec coverage:**
- R1 (every open seat via `bot_eligible`) → Task 2 Step 3. ✅
- R2 (hero abstains on ineligible position) → Task 2 Step 4. ✅
- R3 (union from gated sets; forced-pick completeness) → Task 2 Steps 3-5 + existing forced-pick test. ✅
- R4 (three models + guards) → Task 1 Steps 3 + tests Step 1. ✅
- R5 (signature unchanged; no per-model gate) → Task 1 (models take `(view, player, pool, config)`, engine owns gate). ✅
- R6 (CLI races six) → Task 3. ✅
- R7 (bots/snake byte-identical) → no edits to `market.py`/`draft_field.py`; bot bid path unchanged in Task 2. ✅
- R7a (replace `test_hero_is_not_gated`) → Task 2 Step 1. ✅
- R8 (determinism) → Task 2 `test_gated_hero_is_deterministic`. ✅
- R9 (conventions) → Global Constraints; helpers reuse `_PYARROW_STR`/enum patterns. ✅
- Run C + caveat → Task 4. ✅

**Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output.

**Type consistency:** `AnchorBudgetBid(n_anchors: int = 4)`, `OverbidValueBid(k: float = 1.3, stud_count: int | None = None)`, `VorpShareBid()`, `_undrafted(pool, drafted) -> pd.DataFrame`, `_vorp_threshold(pool, k) -> float` — used identically in Tasks 1-3 and the tests. The engine edit references only in-scope symbols (`pos_by_id`, `seat_eligible`, `all_positions`, `_build_view`, `_feasible_max`, `bot_max_bid`, `SeatView`).
