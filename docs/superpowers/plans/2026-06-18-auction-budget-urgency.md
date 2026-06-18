# Auction Budget-Urgency + Studs/Depth Hero Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-06-18-auction-budget-urgency-design.md` (passed `superpowers-spec-review` 2026-06-18, no Critical/High/Medium).

**Goal:** Add a shared late-draft `_budget_urgency` factor so every auction hero strategy deploys its whole budget, plus a new `StudsAndDepthBid` "good-bot-as-a-hero" contestant, then re-run the eight-model bake-off (Run F) against the realistic market.

**Architecture:** All bid-logic changes live entirely in `src/projections/draft/assistant/auction/bid_strategy.py` plus the CLI model registry. A module-level `_budget_urgency(view, config) -> float` returns `1.0` at the draft start and when broke, escalating to `< 1.0 + URGENCY_GAIN` as the roster fills with idle cash; each strategy multiplies its base bid by it at a single return point. The bots (`market.py`), the engine (`simulation.py`), `resolve_bids`, the scorer, and the snake field are **untouched**. The engine's `[min_bid, feasible_max]` clamp bounds the resulting bid.

**Tech Stack:** Python 3.12, pandas (pyarrow-backed string dtype), numpy, pytest, mypy (strict), ruff.

## Global Constraints

- **Urgency lives only in `bid_strategy.py`.** Do not touch `market.py`, `simulation.py` (`_simulate_to_state`), `resolve_bids`, the scorer, or the snake field. (Spec R5.)
- **No new pandera schema.** (Spec §Non-goals, R7.)
- **`GsisId` is the canonical key**; never join on names. (CLAUDE.md.)
- **Reference enums, never strings**: `RosterSlot.RB`, `Position.QB`, etc. Use `_PYARROW_STR` for nullable string columns. (CLAUDE.md, R7.)
- **Determinism:** every hero bid is a pure function of `view`/`config`; same `(seed, temp, mix, strategy)` ⇒ identical rosters. (Spec R6.)
- **`URGENCY_GAIN` is a module constant, default `3.0`.** (Spec R1.)
- **Empty-roster bids must be unchanged** (urgency is exactly `1.0` at `progress == 0`); only late-draft escalation is new. (Spec R2.)
- **Data-gathering only:** Run F declares **no winner**. (Spec R8.)
- Gates (run at the repo root): `pytest -v` (touched modules), `mypy src tests`, `ruff check src tests`, `ruff format --check src tests`.

---

### Task 1: `_budget_urgency` helper + `URGENCY_GAIN`

**Files:**
- Modify: `src/projections/draft/assistant/auction/bid_strategy.py` (add module constant + helper, after `_vorp_threshold` at line 122)
- Test: `tests/test_draft/test_assistant_auction_bid_strategy.py` (add import + tests)

**Interfaces:**
- Consumes: `AuctionView` (`my_budget: int`, `my_open_slots: int`), `LeagueConfig` (`min_bid: int`, `roster_size: int` property) — both already imported in `bid_strategy.py`.
- Produces: module constant `URGENCY_GAIN: float = 3.0`; `def _budget_urgency(view: AuctionView, config: LeagueConfig) -> float`. Tasks 2 and 3 multiply their base bid by this.

- [ ] **Step 1: Write the failing tests**

Add to the import block at the top of `tests/test_draft/test_assistant_auction_bid_strategy.py` (currently lines 5–16) the new names `URGENCY_GAIN` and `_budget_urgency`:

```python
from projections.draft.assistant.auction.bid_strategy import (
    AnchorBudgetBid,
    AuctionView,
    InflationBid,
    MarginalValueBid,
    OverbidValueBid,
    PatientValueBid,
    StaticDollarBid,
    StudsAndDepthBid,  # added in Task 3; import now so Tasks 1+3 share one import block
    URGENCY_GAIN,
    VorpShareBid,
    _budget_urgency,
    _undrafted,
    _vorp_threshold,
)
```

> NOTE: `StudsAndDepthBid` does not exist until Task 3. If you are executing Task 1 in isolation and `pytest` collection fails on that import, temporarily drop the `StudsAndDepthBid` line and re-add it in Task 3. (Under subagent-driven execution the tasks run in order, so importing it now keeps the import block edited once.)

Append these tests (they use the existing `_vpool`/`_vconfig`/`_aview` helpers at lines 135–201):

```python
# ---------------------------------------------------------------------------
# _budget_urgency (shared late-draft deployment factor)
# ---------------------------------------------------------------------------


def test_budget_urgency_is_one_at_draft_start() -> None:
    # progress == 0 (my_open_slots == roster_size) -> exactly 1.0, regardless of surplus.
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=8)  # roster_size 8 -> progress 0
    assert _budget_urgency(view, _vconfig()) == 1.0


def test_budget_urgency_is_one_when_broke() -> None:
    # surplus = budget - min_bid*open_slots <= 0 -> 1.0 (don't escalate what you can't afford).
    pool = _vpool()
    view = _aview(pool, budget=5, open_slots=8)  # 5 - 1*8 = -3 <= 0
    assert _budget_urgency(view, _vconfig()) == 1.0


def test_budget_urgency_exceeds_one_for_overfunded_partial_roster() -> None:
    pool = _vpool()
    view = _aview(pool, budget=90, open_slots=4)  # surplus 86 > 0, progress 0.5
    assert _budget_urgency(view, _vconfig()) > 1.0


def test_budget_urgency_increases_with_progress() -> None:
    pool = _vpool()
    cfg = _vconfig()
    fewer_slots = _budget_urgency(_aview(pool, budget=100, open_slots=2), cfg)  # progress 0.75
    more_slots = _budget_urgency(_aview(pool, budget=100, open_slots=6), cfg)  # progress 0.25
    assert fewer_slots > more_slots


def test_budget_urgency_increases_with_surplus() -> None:
    pool = _vpool()
    cfg = _vconfig()
    rich = _budget_urgency(_aview(pool, budget=100, open_slots=4), cfg)  # ratio 96/100
    poor = _budget_urgency(_aview(pool, budget=20, open_slots=4), cfg)  # ratio 16/20
    assert rich > poor


def test_budget_urgency_is_bounded_below_one_plus_gain() -> None:
    pool = _vpool()
    cfg = _vconfig()
    extreme = _budget_urgency(_aview(pool, budget=10_000, open_slots=1), cfg)
    assert 1.0 <= extreme < 1.0 + URGENCY_GAIN
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_draft/test_assistant_auction_bid_strategy.py -k budget_urgency -v`
Expected: FAIL — `ImportError: cannot import name '_budget_urgency'` (collection error).

- [ ] **Step 3: Add the constant and helper**

In `src/projections/draft/assistant/auction/bid_strategy.py`, immediately after `_vorp_threshold` (ends at line 121) and before `AnchorBudgetBid` (line 124), insert:

```python
URGENCY_GAIN = 3.0


def _budget_urgency(view: AuctionView, config: LeagueConfig) -> float:
    """Late-draft budget-deployment factor (spec §A). Exactly 1.0 at the draft start
    (`my_open_slots == roster_size` -> progress 0) and when broke (no surplus beyond the
    $1-per-open-slot floor); escalates toward 1.0 + URGENCY_GAIN as the roster fills *and* idle
    cash remains. Bounded [1.0, 1.0 + URGENCY_GAIN) for my_open_slots >= 1 (both factors in [0,1));
    the engine's [min_bid, feasible_max] clamp bounds the resulting bid. The surplus<=0 guard runs
    before the surplus/my_budget term, so my_budget==0 never divides by zero."""
    surplus = view.my_budget - config.min_bid * view.my_open_slots
    if surplus <= 0:
        return 1.0
    progress = 1.0 - view.my_open_slots / config.roster_size
    return 1.0 + URGENCY_GAIN * progress * (surplus / view.my_budget)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_draft/test_assistant_auction_bid_strategy.py -k budget_urgency -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Gates + commit**

Run: `ruff check src tests && ruff format --check src tests && mypy src tests`
Expected: zero violations.

```bash
git add src/projections/draft/assistant/auction/bid_strategy.py tests/test_draft/test_assistant_auction_bid_strategy.py
git commit -m "feat(auction): _budget_urgency late-draft deployment factor + URGENCY_GAIN"
```

---

### Task 2: Apply urgency to the seven existing contestants

**Files:**
- Modify: `src/projections/draft/assistant/auction/bid_strategy.py` (refactor `StaticDollarBid`, `InflationBid`, `MarginalValueBid`, `AnchorBudgetBid`, `OverbidValueBid`, `VorpShareBid`, `PatientValueBid` to a single `round(base * _budget_urgency(view, config))` exit)
- Test: `tests/test_draft/test_assistant_auction_bid_strategy.py` (update the 4 partial-roster tests that now change)

**Interfaces:**
- Consumes: `_budget_urgency` (Task 1).
- Produces: behavior unchanged for empty-roster / progress-0 views (urgency 1.0); late-draft overfunded views bid `round(base * urgency)`.

**Why these 4 tests change (and only these):** under urgency, a test changes iff its view has `my_open_slots < roster_size` (progress > 0) **and** `surplus > 0`. Every other VORP/Patient test uses `_aview(..., open_slots=8)` with `roster_size == 8` (progress 0 → urgency 1.0) or `budget`-broke views, so they are unaffected. The 4 that change: `test_marginal_zero_lift_player_bids_min_bid` (open 1, budget 50), `test_anchor_switches_to_scrubs_once_anchors_held` (open 6, budget 60), `test_vorpshare_off_target_player_bids_min` (open 2, budget 100), `test_vorpshare_zero_target_vorp_bids_min` (open 2, budget 100). All four previously returned `min_bid`; each now returns `round(min_bid * _budget_urgency(view, cfg))`.

- [ ] **Step 1: Update the 4 failing-after-refactor tests first (red)**

These tests assert the *base* branch (zero-lift / anchors-held / off-target → `min_bid`); under urgency they become `round(min_bid * urgency)`. Express the expected value via `_budget_urgency` so it stays correct if `URGENCY_GAIN` is later tuned, and add a one-line "urgency feature" reason. Apply these exact edits:

In `test_marginal_zero_lift_player_bids_min_bid` (currently lines 111–118), replace the last two lines:

```python
    cfg = _config()
    bid = MarginalValueBid().max_bid(view, pool.iloc[2], pool, cfg)
    # urgency feature: late-draft (open 1 of 3, surplus 49) the zero-lift base min_bid is scaled up
    assert bid == round(cfg.min_bid * _budget_urgency(view, cfg))
    assert bid > cfg.min_bid
```

In `test_anchor_switches_to_scrubs_once_anchors_held` (currently lines 243–254), replace the last two lines:

```python
    cfg = _vconfig()
    bid = AnchorBudgetBid(n_anchors=2).max_bid(view, pool.iloc[2], pool, cfg)  # anchor-grade
    # urgency feature: anchors held (anchors_remaining 0) -> base min_bid, scaled up late (open 6/8)
    assert bid == round(cfg.min_bid * _budget_urgency(view, cfg))
```

In `test_vorpshare_off_target_player_bids_min` (currently lines 279–283), replace the last line:

```python
    cfg = _vconfig()
    bid = VorpShareBid().max_bid(view, pool.iloc[2], pool, cfg)
    # urgency feature: off-target base min_bid scaled up late (open 2/8, surplus 98)
    assert bid == round(cfg.min_bid * _budget_urgency(view, cfg))
```

In `test_vorpshare_zero_target_vorp_bids_min` (currently lines 286–307), replace the final assertion line:

```python
    cfg = _vconfig()
    bid = VorpShareBid().max_bid(view, pool.iloc[0], pool, cfg)
    # urgency feature: zero-target-vorp base min_bid scaled up late (open 2/8, surplus 98)
    assert bid == round(cfg.min_bid * _budget_urgency(view, cfg))
```

- [ ] **Step 2: Run to verify they fail (base behavior, pre-refactor)**

Run: `pytest tests/test_draft/test_assistant_auction_bid_strategy.py -k "marginal_zero_lift or anchor_switches or vorpshare_off_target or vorpshare_zero_target" -v`
Expected: FAIL — the strategies still return bare `min_bid`, so `bid == round(min_bid * urgency)` is false (urgency > 1 there) / `bid > min_bid` is false.

- [ ] **Step 3: Refactor the seven strategies to a single urgency-scaled exit**

In `src/projections/draft/assistant/auction/bid_strategy.py`, replace each `max_bid` body. Empty-roster behavior is preserved because `round(base * 1.0) == round(base)` at progress 0.

`StaticDollarBid.max_bid` (lines 54–57):

```python
    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        base = int(view.baseline_dollars.loc[player["gsis_id"], "auction_dollars"])
        return round(base * _budget_urgency(view, config))
```

`InflationBid.max_bid` (lines 64–74):

```python
    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        min_bid = config.min_bid
        money = _surplus_money(view, config)
        bd = view.baseline_dollars
        undrafted_in_pool = bd[bd["in_pool"] & ~bd.index.isin(view.drafted)]
        value = float((undrafted_in_pool["auction_dollars"] - min_bid).sum())
        inflation = money / value if value > 0 else 1.0
        base = int(bd.loc[player["gsis_id"], "auction_dollars"])
        bid = min_bid + (base - min_bid) * inflation
        return round(bid * _budget_urgency(view, config))
```

`MarginalValueBid.max_bid` (lines 81–104) — funnel the three `min_bid` early-returns into one `base`:

```python
    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        min_bid = config.min_bid
        slots = config.roster_slots
        base_pts = optimal_lineup_points(view.my_roster, slots)
        cand = pool[pool["gsis_id"] == player["gsis_id"]]
        with_player = pd.concat([view.my_roster, cand], ignore_index=True)
        lift = optimal_lineup_points(with_player, slots) - base_pts
        base: float = float(min_bid)
        if lift > 0.0:
            money = _surplus_money(view, config)
            bd = view.baseline_dollars
            undrafted_in_pool_ids = bd[bd["in_pool"] & ~bd.index.isin(view.drafted)].index
            # Lift of a single player to an EMPTY lineup == its season_mean_fpts (every in-pool
            # position has a starting slot), so the board's surplus value in lineup-points is the
            # sum of undrafted in-pool projected points. Cheap, and equal to summing single-player
            # optimal_lineup_points one by one.
            on_board = pool[pool["gsis_id"].isin(undrafted_in_pool_ids)]
            value_points = float(on_board["season_mean_fpts"].sum())
            points_per_dollar = value_points / money if money > 0 else 0.0
            if points_per_dollar > 0.0:
                base = min_bid + lift / points_per_dollar
        return round(base * _budget_urgency(view, config))
```

`AnchorBudgetBid.max_bid` (lines 130–143):

```python
    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        min_bid = config.min_bid
        threshold = _vorp_threshold(pool, self.n_anchors * config.n_teams)
        anchors_held = int((view.my_roster["vorp"] >= threshold).sum())
        anchors_remaining = max(0, self.n_anchors - anchors_held)
        open_slots = view.my_open_slots
        base: float = float(min_bid)
        if float(player["vorp"]) >= threshold and anchors_remaining > 0:
            reserve = min_bid * max(0, open_slots - anchors_remaining)
            # unclamped desire; engine clamps to [min_bid, feasible_max] (module contract)
            base = (view.my_budget - reserve) / anchors_remaining
        return round(base * _budget_urgency(view, config))
```

`OverbidValueBid.max_bid` (lines 153–161):

```python
    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        stud_count = self.stud_count if self.stud_count is not None else 3 * config.n_teams
        threshold = _vorp_threshold(pool, stud_count)
        value = int(view.baseline_dollars.loc[player["gsis_id"], "auction_dollars"])
        base = value * self.k if float(player["vorp"]) >= threshold else float(value)
        return round(base * _budget_urgency(view, config))
```

`VorpShareBid.max_bid` (lines 168–179):

```python
    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        min_bid = config.min_bid
        targets = _undrafted(pool, view.drafted).nlargest(view.my_open_slots, "vorp")
        base: float = float(min_bid)
        if str(player["gsis_id"]) in {str(g) for g in targets["gsis_id"]}:
            denom = float(targets["vorp"].clip(lower=0.0).sum())
            if denom > 0.0:
                share = max(0.0, float(player["vorp"])) / denom
                base = view.my_budget * share
        return round(base * _budget_urgency(view, config))
```

`PatientValueBid.max_bid` (lines 190–203) — keep the original `round`-before-compare reserve check so the mid-tier `bid` integer is unchanged at progress 0:

```python
    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        min_bid = config.min_bid
        n = len(pool)
        stud_cut = _vorp_threshold(pool, round(self.stud_frac * n))
        scrub_cut = _vorp_threshold(pool, round((1.0 - self.scrub_frac) * n))
        v = float(player["vorp"])
        base = min_bid
        if not (v >= stud_cut or v < scrub_cut):  # mid-tier: not a stud (let go), not a scrub
            value = int(view.baseline_dollars.loc[player["gsis_id"], "auction_dollars"])
            bid = round(value * (1.0 + self.midtier_premium))
            reserve = view.my_budget - min_bid * (view.my_open_slots - 1)
            if reserve >= bid:
                base = bid
        return round(base * _budget_urgency(view, config))
```

- [ ] **Step 4: Run the full bid-strategy suite to verify green**

Run: `pytest tests/test_draft/test_assistant_auction_bid_strategy.py -v`
Expected: PASS (all existing tests + the 6 urgency tests; the 4 updated tests now match `round(min_bid * urgency)`).

- [ ] **Step 5: Gates + commit**

Run: `ruff check src tests && ruff format --check src tests && mypy src tests`
Expected: zero violations.

```bash
git add src/projections/draft/assistant/auction/bid_strategy.py tests/test_draft/test_assistant_auction_bid_strategy.py
git commit -m "feat(auction): seven contestants deploy budget via _budget_urgency (single-exit)"
```

---

### Task 3: `StudsAndDepthBid` (8th contestant)

**Files:**
- Modify: `src/projections/draft/assistant/auction/bid_strategy.py` (add the dataclass, after `PatientValueBid`)
- Test: `tests/test_draft/test_assistant_auction_bid_strategy.py` (add tests; import already added in Task 1)

**Interfaces:**
- Consumes: `_vorp_threshold`, `_budget_urgency`, `AuctionView`, `LeagueConfig`.
- Produces: `class StudsAndDepthBid` — frozen dataclass with fields `stud_premium: float = 0.2`, `stud_frac: float = 0.10`, `scrub_frac: float = 0.20`; satisfies the `AuctionBidStrategy` protocol (`max_bid(view, player, pool, config) -> int`). Used by the CLI registry in Task 4.

**Tiering arithmetic on the `_vpool` fixture** (vorps 120,110,90,20,10,5; n=6; auction_dollars 30,28,25,5,3,2): `stud_frac 0.10 -> round(0.6)=1 -> stud_cut = _vorp_threshold(pool,1) = 120` (only vorp 120 is a stud). `scrub_frac 0.20 -> round(0.8*6)=5 -> scrub_cut = _vorp_threshold(pool,5) = 5th-highest vorp = 10` (only vorp 5 is a scrub). Mid-tier = `10 <= v < 120` (vorps 110, 90, 20, 10).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_draft/test_assistant_auction_bid_strategy.py`:

```python
# ---------------------------------------------------------------------------
# StudsAndDepthBid (the "good bot as a hero")
# ---------------------------------------------------------------------------


def test_studs_premium_for_a_stud() -> None:
    # vorp 120 >= stud_cut 120 -> auction_dollars 30 * (1 + 0.2). Empty roster -> urgency 1.0.
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=8)
    bid = StudsAndDepthBid().max_bid(view, pool.iloc[0], pool, _vconfig())
    assert bid == round(30 * (1.0 + 0.2))  # 36


def test_studs_fair_value_for_midtier() -> None:
    # vorp 110 in (10, 120) -> fair value = auction_dollars 28, no $1-dump. Empty -> urgency 1.0.
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=8)
    bid = StudsAndDepthBid().max_bid(view, pool.iloc[1], pool, _vconfig())
    assert bid == 28


def test_studs_min_bid_for_a_scrub() -> None:
    # vorp 5 < scrub_cut 10 -> min_bid. Empty roster -> urgency 1.0.
    pool = _vpool()
    view = _aview(pool, budget=100, open_slots=8)
    bid = StudsAndDepthBid().max_bid(view, pool.iloc[5], pool, _vconfig())
    assert bid == _vconfig().min_bid


def test_studs_depth_scales_up_under_overfunded_partial_roster() -> None:
    # Same mid-tier player, but a partial overfunded view (open 5/8, surplus 85) -> urgency > 1.
    pool = _vpool()
    cfg = _vconfig()
    view = _aview(pool, budget=90, open_slots=5)
    bid = StudsAndDepthBid().max_bid(view, pool.iloc[1], pool, cfg)
    assert bid == round(28 * _budget_urgency(view, cfg))
    assert bid > 28  # deploys the surplus rather than leaving it idle


def test_studs_depth_tiny_pool_has_no_studs() -> None:
    # round(stud_frac * 1) == 0 -> _vorp_threshold(pool, 0) == +inf -> nothing clears the stud bar.
    one = _vpool().iloc[[0]].reset_index(drop=True)
    view = _aview(one, budget=100, open_slots=8)
    # vorp 120 is NOT a stud here (stud_cut +inf); scrub_cut = pool min 120 -> v<120 false -> mid.
    bid = StudsAndDepthBid().max_bid(view, one.iloc[0], one, _vconfig())
    assert bid == 30  # fair value (auction_dollars), urgency 1.0


def test_studs_depth_satisfies_protocol() -> None:
    from projections.draft.assistant.auction.bid_strategy import AuctionBidStrategy

    assert isinstance(StudsAndDepthBid(), AuctionBidStrategy)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_draft/test_assistant_auction_bid_strategy.py -k studs -v`
Expected: FAIL — `NameError`/`ImportError` for `StudsAndDepthBid` (if the Task 1 import was deferred, re-add it now).

- [ ] **Step 3: Add the `StudsAndDepthBid` dataclass**

In `src/projections/draft/assistant/auction/bid_strategy.py`, after `PatientValueBid` (ends at line 203), append:

```python
@dataclass(frozen=True)
class StudsAndDepthBid:
    """The 'good bot as a hero' (spec §C): secure a few studs near fair value (a modest premium to
    actually win the anchor), bid fair value across mid-tier depth (no $1-dumping), $1 the scrubs —
    then deploy the whole budget via _budget_urgency as the draft winds down."""

    stud_premium: float = 0.2
    stud_frac: float = 0.10
    scrub_frac: float = 0.20

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        min_bid = config.min_bid
        n = len(pool)
        stud_cut = _vorp_threshold(pool, round(self.stud_frac * n))
        scrub_cut = _vorp_threshold(pool, round((1.0 - self.scrub_frac) * n))
        v = float(player["vorp"])
        auction_dollars = int(view.baseline_dollars.loc[player["gsis_id"], "auction_dollars"])
        if v >= stud_cut:  # stud: modest premium to actually win the anchor (unlike static)
            base: float = auction_dollars * (1.0 + self.stud_premium)
        elif v < scrub_cut:  # scrub: floor it
            base = float(min_bid)
        else:  # mid-tier depth: fair value, no $1-dumping
            base = float(auction_dollars)
        return round(base * _budget_urgency(view, config))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_draft/test_assistant_auction_bid_strategy.py -k studs -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Gates + commit**

Run: `ruff check src tests && ruff format --check src tests && mypy src tests`
Expected: zero violations.

```bash
git add src/projections/draft/assistant/auction/bid_strategy.py tests/test_draft/test_assistant_auction_bid_strategy.py
git commit -m "feat(auction): StudsAndDepthBid contestant (studs premium + fair-value depth)"
```

---

### Task 4: Wire `studsdepth` into the CLI + engine integration tests

**Files:**
- Modify: `src/projections/draft/assistant/auction/tournament_cli.py` (import + `_MODELS` entry; update "seven" → "eight" in the module docstring and the `compare` help)
- Test: `tests/test_draft/test_assistant_auction_tournament_cli.py` (update the model-set test)
- Test: `tests/test_draft/test_assistant_auction_simulation.py` (add two engine integration tests, reusing existing fixtures)

**Interfaces:**
- Consumes: `StudsAndDepthBid` (Task 3); the engine entry `_simulate_to_state(strategy, my_seat, pool, config, *, baseline_dollars, price_jitter, rng, nomination_temp, bot_archetypes)` returning `AuctionState` (`.budgets`, `.rosters`); existing test fixtures `_config`, `_pool`, `_realistic_baseline`, `_MinBidStub` in `test_assistant_auction_simulation.py`.
- Produces: `_MODELS` with eight keys including `"studsdepth"`; the realistic-market defaults (`nomination_temp=1.0`, `_REALISTIC_FIELD`) unchanged.

**Note on the spec's integration acceptance ("spends materially more than `static`"):** Task 2 makes `StaticDollarBid` urgency-aware too, so on a late-draft seed `static` *also* deploys its budget — the head-to-head total-spend gap is confounded. We therefore test the spec's *intent* ("deploys ~the full budget", the Run-E failure being idle cash) directly: spend exceeds a deployment floor and far exceeds a `_MinBidStub` floor-bidder, with a startable roster. This is a stronger, non-flaky check of the same property.

- [ ] **Step 1: Write the failing CLI registry test**

In `tests/test_draft/test_assistant_auction_tournament_cli.py`, replace `test_default_models_are_the_seven_contestants` (lines 49–58):

```python
def test_default_models_are_the_eight_contestants() -> None:
    assert set(_MODELS) == {
        "static",
        "inflation",
        "marginal",
        "anchors",
        "overbid",
        "vorpshare",
        "patient",
        "studsdepth",
    }
```

- [ ] **Step 2: Write the failing engine integration tests**

In `tests/test_draft/test_assistant_auction_simulation.py`, add `StudsAndDepthBid` to the bid_strategy import (currently lines 7–11) and append two tests. The hero is seat 1 → state index 0; spend = `cfg.budget - state.budgets[0]`.

```python
def test_studs_and_depth_deploys_budget_and_is_startable() -> None:
    # Realistic market (mixed field + value-weighted-random nomination). StudsAndDepthBid should
    # deploy most of its budget (the Run-E failure mode was idle cash) and field a startable roster,
    # spending far more than a floor-bidder hero on the same seed.
    cfg = _config(n_teams=8, roster_slots={RosterSlot.RB: 2, RosterSlot.WR: 2, RosterSlot.BENCH: 3})
    pool = _pool(80)
    baseline = _realistic_baseline(pool)
    kw = dict(
        baseline_dollars=baseline,
        price_jitter=0.15,
        nomination_temp=1.0,
        bot_archetypes=[AggressiveBot(), PatientValueBot(), BalancedBot()],
    )
    studs = _simulate_to_state(
        StudsAndDepthBid(), 1, pool, cfg, rng=np.random.default_rng(11), **kw
    )
    floor = _simulate_to_state(_MinBidStub(), 1, pool, cfg, rng=np.random.default_rng(11), **kw)
    studs_spend = cfg.budget - studs.budgets[0]
    floor_spend = cfg.budget - floor.budgets[0]
    assert studs_spend > floor_spend  # actually deploys budget vs a min-bidder
    assert studs_spend >= 0.75 * cfg.budget  # ~full-budget deployment (no large idle cash)
    hero = [p for (_g, p, _pr) in studs.rosters[0]]
    assert len(hero) == cfg.roster_size  # filled
    assert hero.count("RB") >= 2 and hero.count("WR") >= 2  # startable starting lineup


def test_studs_and_depth_is_deterministic() -> None:
    cfg = _config(n_teams=8, roster_slots={RosterSlot.RB: 2, RosterSlot.WR: 2, RosterSlot.BENCH: 3})
    pool = _pool(80)
    bl = _realistic_baseline(pool)
    kw = dict(
        baseline_dollars=bl,
        price_jitter=0.15,
        nomination_temp=1.0,
        bot_archetypes=[AggressiveBot(), PatientValueBot(), BalancedBot()],
    )
    a = _simulate_to_state(StudsAndDepthBid(), 1, pool, cfg, rng=np.random.default_rng(5), **kw)
    b = _simulate_to_state(StudsAndDepthBid(), 1, pool, cfg, rng=np.random.default_rng(5), **kw)
    assert a.rosters == b.rosters
```

- [ ] **Step 3: Run to verify failure**

Run: `pytest tests/test_draft/test_assistant_auction_tournament_cli.py::test_default_models_are_the_eight_contestants tests/test_draft/test_assistant_auction_simulation.py -k "studs_and_depth" -v`
Expected: FAIL — `studsdepth` missing from `_MODELS`; `ImportError`/`NameError` for `StudsAndDepthBid` in the simulation test.

- [ ] **Step 4: Wire the CLI**

In `src/projections/draft/assistant/auction/tournament_cli.py`:

Add `StudsAndDepthBid` to the bid_strategy import (lines 16–25), keeping alphabetical order:

```python
from projections.draft.assistant.auction.bid_strategy import (
    AnchorBudgetBid,
    AuctionBidStrategy,
    InflationBid,
    MarginalValueBid,
    OverbidValueBid,
    PatientValueBid,
    StaticDollarBid,
    StudsAndDepthBid,
    VorpShareBid,
)
```

Add the registry entry (after line 51, inside `_MODELS`):

```python
    "patient": PatientValueBid(),
    "studsdepth": StudsAndDepthBid(),
}
```

Update the two "seven" strings: the module docstring (line 5, "races the seven bid models" → "races the eight bid models") and the `compare` subparser help (line 125, `help="Race the seven bid models; record per-metric data."` → `help="Race the eight bid models; record per-metric data."`).

- [ ] **Step 5: Add `StudsAndDepthBid` to the simulation test import**

In `tests/test_draft/test_assistant_auction_simulation.py` (import block lines 7–11):

```python
from projections.draft.assistant.auction.bid_strategy import (
    AnchorBudgetBid,
    AuctionView,
    StaticDollarBid,
    StudsAndDepthBid,
)
```

- [ ] **Step 6: Run to verify the targeted tests pass**

Run: `pytest tests/test_draft/test_assistant_auction_tournament_cli.py -v tests/test_draft/test_assistant_auction_simulation.py -k "studs_and_depth" -v`
Expected: PASS. If `studs_spend >= 0.75 * cfg.budget` fails, do **not** seed-shop — investigate whether the strategy is genuinely leaving cash idle (a real finding worth a note in the spec's open question); if confirmed working but the synthetic pool simply caps deployment lower, lower the floor with a comment recording the observed spend.

- [ ] **Step 7: Run the full touched-module suite (catches the protocol + no-winner CLI tests)**

Run: `pytest tests/test_draft/test_assistant_auction_tournament_cli.py tests/test_draft/test_assistant_auction_simulation.py tests/test_draft/test_assistant_auction_bid_strategy.py -v`
Expected: PASS (incl. `test_every_default_model_satisfies_the_protocol` now covering 8 models, and `test_format_compare_has_no_winner_line`).

- [ ] **Step 8: Gates + commit**

Run: `ruff check src tests && ruff format --check src tests && mypy src tests`
Expected: zero violations.

```bash
git add src/projections/draft/assistant/auction/tournament_cli.py tests/test_draft/test_assistant_auction_tournament_cli.py tests/test_draft/test_assistant_auction_simulation.py
git commit -m "feat(auction): wire studsdepth into the eight-model CLI + engine integration tests"
```

---

### Task 5: Run F bake-off + tracking doc (no winner)

**Files:**
- Modify: `reports/auction_tournament_validation_2026.md` (append Run F rows to the experiment log + a Run F narrative section; update the "Reproduce" blurb's "seven" → "eight" and the model list)

**Interfaces:**
- Consumes: the eight-model CLI (`scripts/auction_tournament.py` → `tournament_cli.run`), unchanged realistic-market defaults.
- Produces: a recorded Run F (data only; **no winner**).

**Hardware constraint (load-bearing):** the i9-14900KF Raptor Lake fault segfaults large single-process auction runs (memory `h2h-backtest-native-crash`; Runs C–E used 60 seeds × 200 n_sims, which completes cleanly). Eight models is more total work than Run E's seven, so keep the per-process budget at **60 seeds × 200 n_sims**. If it still faults, split into two invocations over disjoint model subsets at the **same `--seed 0`** (per-model means stay comparable across invocations; paired diffs are only valid within an invocation — note the split in the report).

- [ ] **Step 1: Run the eight-model bake-off (realistic market, half × 16)**

Use the same preset/flags as Run E (half_16team, seat 1, realistic-market defaults). The CLI's `--nomination-temp` defaults to `1.0` and the field is `_REALISTIC_FIELD` (mixed) — these are the realistic-market settings; do not override them.

Run:
```bash
python scripts/auction_tournament.py \
    --vorp-table data/vorp_2026/half_16team.parquet \
    --league-config configs/league_espn_half_16team.json \
    --my-seat 1 --season 2026 \
    --seeds 60 --n-sims 200 --price-jitter 0.15 --seed 0 \
    compare
```
Expected: a per-model table (8 rows: static, inflation, marginal, anchors, overbid, vorpshare, patient, studsdepth) with exp-pts/win/playoff/bye/champ means + 95% CIs, plus paired per-seed diffs, ending with no "winner" line. Capture stdout verbatim.

> If the process segfaults (Raptor Lake), re-run with the model set split (e.g. run once, and if needed lower `--seeds` to 40) and record what completed. A crash is a hardware event, not a code bug — do not "fix" the code in response.

- [ ] **Step 2: Append Run F to the experiment log**

In `reports/auction_tournament_validation_2026.md`, add 8 rows (one per model) to the `## Experiment log` table (after the Run E rows, currently ending line 105), using the captured numbers, dated `2026-06-18`, preset `half × 16`, seat 1, 60 seeds, 200 n_sims, price_jitter 0.15, with the note column `Run F (urgency + studsdepth)`. Order rows by exp-pts descending (as prior runs do).

- [ ] **Step 3: Write the Run F narrative**

After the Run E narrative (currently ends ~line 199, before `## Planned experiments`), add a `**Run F — 2026-06-18**` section mirroring Run E's structure: the config line (note `feat/auction-budget-urgency`; `_budget_urgency` applied to all eight; new `studsdepth`), per-model metrics, key paired playoff% diffs, and a headline that **states the observation without declaring a winner**. Explicitly address the spec's open question: if `studsdepth` (and the urgency-refined field) still trails the bots, record that this strengthens the case that the mixed-bot field is mis-calibrated (too strong) until anchored on real published auction values (the next realism slice). End with **No winner declared (September decision).**

- [ ] **Step 4: Update the Reproduce blurb**

In the `## Reproduce` section, update "races `static` / `inflation` / `marginal`" / any "seven" wording to reflect the eight contestants (add `studsdepth`).

- [ ] **Step 5: Verify the report is internally consistent + commit**

Re-read the appended rows and narrative; confirm the row count is 8, the numbers match the captured stdout, and no "the winner is X" phrasing slipped in (grep the file for "winner" — every hit must be a *negation*).

```bash
git add reports/auction_tournament_validation_2026.md
git commit -m "data(auction): Run F — eight-model bake-off (urgency + studsdepth) vs realistic market; no winner"
```

---

## Self-Review

**1. Spec coverage:**
- R1 (`_budget_urgency` + `URGENCY_GAIN`, §A exactly, 1.0 at progress 0 / surplus<=0, bounded) → Task 1.
- R2 (all seven multiply at a single exit; empty-roster unchanged) → Task 2 (+ the 4 changed tests).
- R3 (`StudsAndDepthBid`, frozen dataclass, satisfies protocol, §C behavior) → Task 3.
- R4 (CLI races eight; realistic defaults unchanged; protocol satisfied) → Task 4.
- R5 (bots/engine/`resolve_bids`/snake untouched) → Global Constraints; Tasks touch only `bid_strategy.py`, `tournament_cli.py`, tests, report.
- R6 (determinism) → `test_studs_and_depth_is_deterministic` (Task 4) + helper purity.
- R7 (conventions; no new schema) → Global Constraints + gates.
- R8 (Run F recorded; no winner) → Task 5.
- Edge cases (empty roster, broke, urgency on min_bid, tiny pool, my_budget==0) → Task 1 tests (broke, bounded, progress-0) + Task 3 `test_studs_depth_tiny_pool_has_no_studs`; the `my_budget==0`/division-guard is covered by the broke test and documented in the helper docstring.

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code; the one conditional ("if the 0.75 floor fails") gives a concrete investigate-don't-seed-shop instruction, not a placeholder.

**3. Type consistency:** `_budget_urgency(view, config) -> float` used identically in Tasks 2–3. `StudsAndDepthBid` field defaults (`stud_premium=0.2`, `stud_frac=0.10`, `scrub_frac=0.20`) match §C and the Task 3 tiering arithmetic. `_simulate_to_state` signature in Task 4 matches `simulation.py`. `_MODELS` key `"studsdepth"` consistent across the CLI and the registry test.
