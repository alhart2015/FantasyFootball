# ESPN-anchored auction bots (Slice 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make auction bots price players off real ESPN auction values (rescaled to the league budget; $1 for players ESPN didn't price) while the hero keeps pricing off our SOS model, behind `--bot-prices {espn,model}` (default `espn`, graceful fallback to `model`).

**Architecture:** Reuse the existing Surplus-Of-Surplus (SOS) allocator: extract its core into a shared `_allocate_surplus(value_signal, config)` helper, then build a bot price vector by feeding it `espn_auction_dollars` instead of VORP. Inject that vector as a new engine-internal `bot_dollars` column on the indexed baseline frame the bots read, leaving the hero (and `generate_auction_values`) on `auction_dollars`. Nomination order stays on SOS, so `model` vs `espn` is a clean one-variable A/B.

**Tech Stack:** Python, pandas (nullable `Int64`/`Float64`), pandera schemas, numpy RNG, pytest, mypy-strict, ruff.

**Spec:** `docs/superpowers/specs/2026-06-18-auction-bot-espn-anchor-design.md` (read it; this plan implements R1–R8).

---

## Before You Start

- **Worktree environment.** You are in a git worktree. Ensure the package under test is the *worktree's* `src/`, not the main checkout's. From the worktree root, run `pip install -e .` (or set `PYTHONPATH` to the worktree `src`) inside the project venv so edits are picked up.
- **Baseline green.** Before touching anything, confirm the auction suite is green:
  `pytest tests/test_draft/test_auction.py tests/test_draft/test_assistant_auction_market.py tests/test_draft/test_assistant_auction_simulation.py tests/test_draft/test_assistant_auction_tournament.py tests/test_draft/test_assistant_auction_tournament_cli.py -q`
- **Commits.** Per project memory, before `git commit` prepend the venv scripts dir to PATH so pre-commit's mypy hook resolves to the project Python: on Windows Git Bash, `PATH="$(pwd)/.venv/Scripts:$PATH" git commit ...` (adjust to wherever the active venv lives). End commit messages with the `Co-Authored-By:`/`Claude-Session:` trailers used elsewhere in this branch's history.
- **Full gates after each task** (per CLAUDE.md): `pytest -q -k "auction or schemas"`, `mypy src tests`, `ruff check src tests`, `ruff format --check src tests`.

## File Structure

- `src/projections/draft/auction.py` — **modify.** Extract `_allocate_surplus`; refactor `generate_auction_values` to call it; add `espn_anchored_bot_prices`. (Task 1)
- `src/projections/draft/assistant/auction/market.py` — **modify.** 4 bot read-sites switch `"auction_dollars"` → `"bot_dollars"`. (Task 2)
- `src/projections/draft/assistant/auction/simulation.py` — **modify.** `bot_dollars` param on `_simulate_to_state` + `simulate_auction`; attach the `bot_dollars` column to `bd`. (Task 2)
- `src/projections/draft/assistant/auction/bid_strategy.py` — **modify (doc only).** Update the `AuctionView.baseline_dollars` docstring. (Task 2)
- `src/projections/draft/assistant/auction/tournament.py` — **modify.** `bot_prices` param; compute/thread the bot vector; warn-and-fallback. (Task 3)
- `src/projections/draft/assistant/auction/tournament_cli.py` — **modify.** `--bot-prices` flag (top-level parser) + narrowing + the ESPN-vs-ours diagnostic readout. (Task 3)
- `tests/test_draft/test_auction.py` — **modify.** New `espn_anchored_bot_prices` tests. (Task 1)
- `tests/test_draft/test_assistant_auction_market.py` — **modify.** Fixtures gain `bot_dollars`. (Task 2)
- `tests/test_draft/test_assistant_auction_simulation.py` — **modify.** Seam tests + `_realistic_baseline` gains `bot_dollars`. (Task 2)
- `tests/test_draft/test_assistant_auction_tournament.py` — **modify.** Flag tests. (Task 3)
- `tests/test_draft/test_assistant_auction_tournament_cli.py` — **modify.** CLI flag/diagnostic tests. (Task 3)
- `reports/auction_tournament_validation_2026.md`, `TODO.md`, `project_management.md` — **modify.** Run H. (Task 4)

---

## Task 1: Extract `_allocate_surplus`; add `espn_anchored_bot_prices`

No behavior change to `generate_auction_values` (the existing `test_auction.py` suite is the equivalence guard). Adds the pure bot price-vector producer.

**Files:**
- Modify: `src/projections/draft/auction.py`
- Test: `tests/test_draft/test_auction.py`

- [ ] **Step 1: Write the failing tests for `espn_anchored_bot_prices`**

Append to `tests/test_draft/test_auction.py` (the helpers `_make_config`, `_make_vorp_table`, `_full_pool_vorp_table`, `_PYARROW_STR`, `Position`, `RosterSlot`, `Ruleset` already exist in that file). Add `espn_anchored_bot_prices` to the existing import from `projections.draft.auction`:

```python
from projections.draft.auction import espn_anchored_bot_prices, generate_auction_values


def _hand_pool_with_espn() -> tuple[LeagueConfig, pd.DataFrame]:
    """4-player pool (2 QB + 2 RB), budget 100, min_bid 1 -> surplus 96. Two priced players
    (espn 60, 36) absorb the surplus; two unpriced (NA) park at min_bid. Drift is 0:
      value_signal = [60, 0, 36, 0]; sum=96; extra=[60,0,36,0]; dollars=[61,1,37,1]; sum=100.
    """
    cfg = _make_config(
        n_teams=2, roster_slots={RosterSlot.QB: 1, RosterSlot.RB: 1, RosterSlot.BENCH: 0}
    )
    df = _make_vorp_table(
        [
            {"gsis_id": "00-1000001", "position": "QB", "season_mean_fpts": 200.0, "vorp": 50.0},
            {"gsis_id": "00-1000002", "position": "QB", "season_mean_fpts": 190.0, "vorp": 40.0},
            {"gsis_id": "00-2000001", "position": "RB", "season_mean_fpts": 180.0, "vorp": 30.0},
            {"gsis_id": "00-2000002", "position": "RB", "season_mean_fpts": 170.0, "vorp": 20.0},
        ]
    )
    df["espn_auction_dollars"] = pd.array([60, pd.NA, 36, pd.NA], dtype=pd.Int64Dtype())
    return cfg, df


def test_espn_bot_prices_sum_to_total_budget() -> None:
    cfg, df = _hand_pool_with_espn()
    out = espn_anchored_bot_prices(df, cfg)
    assert int(out.sum()) == cfg.total_budget


def test_espn_bot_prices_unpriced_park_at_min_bid() -> None:
    cfg, df = _hand_pool_with_espn()
    out = espn_anchored_bot_prices(df, cfg)
    assert out["00-1000002"] == cfg.min_bid
    assert out["00-2000002"] == cfg.min_bid


def test_espn_bot_prices_priced_split_surplus_and_are_monotonic() -> None:
    cfg, df = _hand_pool_with_espn()
    out = espn_anchored_bot_prices(df, cfg)
    assert out["00-1000001"] == 61  # min_bid + (60/96)*96
    assert out["00-2000001"] == 37  # min_bid + (36/96)*96
    assert out["00-1000001"] > out["00-2000001"]  # higher ESPN $ -> higher bot $


def test_espn_bot_prices_dtype_is_int64() -> None:
    cfg, df = _hand_pool_with_espn()
    out = espn_anchored_bot_prices(df, cfg)
    assert out.dtype == pd.Int64Dtype()


def test_espn_bot_prices_out_of_pool_get_zero() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)  # has extra out-of-pool rows per position
    # price everyone present at a flat 10 so we can isolate the out-of-pool=0 property
    df["espn_auction_dollars"] = pd.array([10] * len(df), dtype=pd.Int64Dtype())
    pool_ids = set(_select_pool(df, cfg))
    out = espn_anchored_bot_prices(df, cfg)
    out_of_pool = [g for g in df["gsis_id"] if g not in pool_ids]
    assert all(out[g] == 0 for g in out_of_pool)


def test_espn_bot_prices_every_in_pool_at_least_min_bid() -> None:
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    df["espn_auction_dollars"] = pd.array(
        [50 if i < 5 else pd.NA for i in range(len(df))], dtype=pd.Int64Dtype()
    )
    pool_ids = set(_select_pool(df, cfg))
    out = espn_anchored_bot_prices(df, cfg)
    assert all(out[g] >= cfg.min_bid for g in pool_ids)


def test_espn_bot_prices_absent_column_uniform_fallback() -> None:
    cfg, df = _hand_pool_with_espn()
    df = df.drop(columns=["espn_auction_dollars"])
    out = espn_anchored_bot_prices(df, cfg)
    # all-zero weight -> uniform split of total_budget over 4 in-pool players
    assert int(out.sum()) == cfg.total_budget
    in_pool = out[out > 0] if (out > 0).any() else out
    assert sorted(in_pool.tolist()) in ([25, 25, 25, 25], [24, 25, 25, 26])


def test_espn_bot_prices_deep_league_inflation() -> None:
    """One priced player among many unpriced absorbs nearly the whole surplus -> bot $ >> ESPN $."""
    cfg = _make_config()
    df = _full_pool_vorp_table(cfg)
    espn = [pd.NA] * len(df)
    espn[0] = 5  # exactly one priced player, ESPN value $5
    df["espn_auction_dollars"] = pd.array(espn, dtype=pd.Int64Dtype())
    priced_gsis = df["gsis_id"].iloc[0]
    out = espn_anchored_bot_prices(df, cfg)
    assert out[priced_gsis] > 5  # inflated far above its nominal ESPN value
```

`_select_pool` is already imported in `test_auction.py`.

- [ ] **Step 2: Run the new tests; verify they fail**

Run: `pytest tests/test_draft/test_auction.py -k espn_bot_prices -q`
Expected: FAIL with `ImportError: cannot import name 'espn_anchored_bot_prices'`.

- [ ] **Step 3: Refactor `generate_auction_values` to extract `_allocate_surplus`, and add `espn_anchored_bot_prices`**

In `src/projections/draft/auction.py`, **add the helper above `generate_auction_values`** (after the imports / `_OUTPUT_COLUMNS`):

```python
def _allocate_surplus(value_signal: pd.Series, config: LeagueConfig) -> pd.Series:
    """Split the auction surplus across in-pool players in proportion to a non-negative value
    signal, returning whole-dollar prices that sum to ``config.total_budget``.

    ``value_signal`` is a non-null ``float64`` Series indexed over the in-pool players (one entry
    per drafted slot). Every entry is floored at ``min_bid``; the surplus
    ``total_budget - total_pool_size*min_bid`` is distributed proportionally to ``value_signal``
    (uniformly if it sums to <= 0). The index is preserved; the result is ``Int64``. Shared by
    ``generate_auction_values`` (VORP signal) and ``espn_anchored_bot_prices`` (ESPN $ signal).
    """
    total_budget = config.total_budget
    reserve = config.total_pool_size * config.min_bid
    surplus = total_budget - reserve

    signal_sum = float(value_signal.sum())
    if signal_sum > 0:
        extra_float = (value_signal / signal_sum) * surplus
    else:
        extra_float = pd.Series(surplus / config.total_pool_size, index=value_signal.index)

    dollars_float = config.min_bid + extra_float
    rounded = dollars_float.round().astype("int64")

    drift = total_budget - int(rounded.sum())
    if drift != 0:
        fractional = dollars_float - dollars_float.astype("int64")
        if drift > 0:
            order = fractional.sort_values(ascending=False).index
        else:
            adjustable_mask = rounded > config.min_bid
            order = fractional[adjustable_mask].sort_values(ascending=True).index
            if len(order) < abs(drift):
                raise ValueError(
                    f"Cannot close rounding drift of {drift} without violating min_bid "
                    f"floor of ${config.min_bid}. This usually indicates an extreme "
                    f"degenerate input (e.g., very small budget per slot)."
                )
        step = 1 if drift > 0 else -1
        for idx in order[: abs(drift)]:
            rounded.loc[idx] = rounded.loc[idx] + step
    return rounded.astype(pd.Int64Dtype())
```

Then in `generate_auction_values`, **replace the inline allocation block** (currently the lines from `total_budget = league_config.total_budget` through `pool_df["auction_dollars"] = rounded.astype(pd.Int64Dtype())`) with:

```python
    pool_df = vorp_table.loc[pool_mask].copy()
    pool_df["auction_dollars"] = _allocate_surplus(pool_df["vorp"].clip(lower=0.0), league_config)
```

Keep everything else in `generate_auction_values` unchanged (the `_reject_duplicate_gsis_ids`/`_select_pool`/`pool_mask` lines above it, and the sort/pool_rank/non_pool/concat/reference_prices/validate lines below it).

Then **add the bot price-vector producer** (below `generate_auction_values`, above `__all__`):

```python
def espn_anchored_bot_prices(pool: pd.DataFrame, config: LeagueConfig) -> pd.Series:
    """Per-player bot reference dollars anchored on real ESPN auction values (TODO #49c Slice 2).

    Returns a ``gsis_id``-indexed ``Int64`` Series over EVERY row of ``pool``: in-pool players get
    an SOS allocation of the budget over ``espn_auction_dollars`` (NA / absent column -> 0 weight,
    so unpriced players park at ``min_bid``); out-of-pool players get 0 (bots reading it bid
    ``min_bid``, as today). Call on the same ``pool`` frame passed to ``generate_auction_values``.

    Raises ``ValueError`` on degenerate drift (propagated from ``_allocate_surplus``); espn-mode
    callers catch it and fall back to model pricing.
    """
    _reject_duplicate_gsis_ids(pool, "pool")
    pool_set = set(_select_pool(pool, config))
    pool_mask = pool["gsis_id"].isin(pool_set)
    pool_df = pool.loc[pool_mask].copy()

    if "espn_auction_dollars" in pool_df.columns:
        value_signal = (
            pool_df["espn_auction_dollars"]
            .astype("Float64")
            .fillna(0)
            .clip(lower=0.0)
            .astype("float64")
        )
    else:
        value_signal = pd.Series(0.0, index=pool_df.index, dtype="float64")
    in_pool_dollars = _allocate_surplus(value_signal, config)

    out = pd.Series(
        pd.array([0] * len(pool), dtype=pd.Int64Dtype()),
        index=pd.Index(pool["gsis_id"], name="gsis_id"),
    )
    out.loc[pd.Index(pool_df["gsis_id"])] = in_pool_dollars.to_numpy()
    return out
```

Add `espn_anchored_bot_prices` to `__all__`:

```python
__all__ = ["espn_anchored_bot_prices", "generate_auction_values"]
```

- [ ] **Step 4: Run the new tests + the full equivalence suite; verify all pass**

Run: `pytest tests/test_draft/test_auction.py -q`
Expected: PASS — both the new `espn_bot_prices` tests and **all pre-existing tests** (sum invariant, min_bid floor, out-of-pool, pool_rank ordering, scale invariance, degenerate-uniform, drift floor-protection, reference_prices). The pre-existing suite passing unchanged IS the byte-identity guard for the extraction.

- [ ] **Step 5: Run gates and commit**

Run: `mypy src tests && ruff check src tests && ruff format --check src tests`
Expected: clean.

```bash
git add src/projections/draft/auction.py tests/test_draft/test_auction.py
git commit -m "feat(auction): extract _allocate_surplus + add espn_anchored_bot_prices (Slice 2 Task 1)"
```

---

## Task 2: Engine seam — `bot_dollars` column the bots read

Bots read a new `bot_dollars` column; hero/nomination untouched. With `bot_dollars=None` the column equals `auction_dollars`, so behavior is byte-identical (default path).

**Files:**
- Modify: `src/projections/draft/assistant/auction/simulation.py`
- Modify: `src/projections/draft/assistant/auction/market.py`
- Modify: `src/projections/draft/assistant/auction/bid_strategy.py` (docstring)
- Test: `tests/test_draft/test_assistant_auction_simulation.py`, `tests/test_draft/test_assistant_auction_market.py`

- [ ] **Step 1: Write the failing seam test**

Append to `tests/test_draft/test_assistant_auction_simulation.py` (helpers `_config`, `_pool`, `_baseline`, `simulate_auction`, `_PYARROW_STR` already exist there):

```python
def _flat_bot_dollars(pool: pd.DataFrame, value: int) -> pd.Series:
    """A bot_dollars Series over every pool gsis_id, all equal -> bots value every player the same."""
    return pd.Series(
        pd.array([value] * len(pool), dtype=pd.Int64Dtype()),
        index=pd.Index(pool["gsis_id"], name="gsis_id"),
    )


def test_bot_dollars_none_reproduces_baseline() -> None:
    cfg = _config(n_teams=4)
    pool = _pool(40)
    bd = _baseline(pool, cfg)
    a = simulate_auction(
        StaticDollarBid(), 1, pool, cfg,
        baseline_dollars=bd, price_jitter=0.1, rng=np.random.default_rng(0),
    )
    b = simulate_auction(
        StaticDollarBid(), 1, pool, cfg,
        baseline_dollars=bd, price_jitter=0.1, rng=np.random.default_rng(0),
        bot_dollars=None,
    )
    assert a == b  # explicit None is identical to the default


def test_bot_dollars_changes_the_bot_market() -> None:
    # Flat bot_dollars makes bots value every player equally -> a different market than SOS,
    # so the resulting league differs from the bot_dollars=None (SOS) run at the same seed.
    cfg = _config(n_teams=4)
    pool = _pool(40)
    bd = _baseline(pool, cfg)
    sos = simulate_auction(
        StaticDollarBid(), 1, pool, cfg,
        baseline_dollars=bd, price_jitter=0.1, rng=np.random.default_rng(0),
    )
    flat = simulate_auction(
        StaticDollarBid(), 1, pool, cfg,
        baseline_dollars=bd, price_jitter=0.1, rng=np.random.default_rng(0),
        bot_dollars=_flat_bot_dollars(pool, 20),
    )
    assert sos != flat  # bot pricing changed -> different rosters/prices
```

- [ ] **Step 2: Run the seam test; verify it fails**

Run: `pytest tests/test_draft/test_assistant_auction_simulation.py -k bot_dollars -q`
Expected: FAIL with `TypeError: ... got an unexpected keyword argument 'bot_dollars'`.

- [ ] **Step 3: Add the `bot_dollars` param + column in `simulation.py`**

In `src/projections/draft/assistant/auction/simulation.py`, add `bot_dollars: pd.Series | None = None` to the keyword-only params of **both** `_simulate_to_state` and `simulate_auction` (place it after `bot_archetypes`).

In `_simulate_to_state`, immediately **after** the line `nominate_order = bd.sort_values("auction_dollars", ascending=False).index.tolist()`, insert:

```python
    if bot_dollars is None:
        bd["bot_dollars"] = bd["auction_dollars"]
    else:
        bd["bot_dollars"] = (
            bot_dollars.reindex(bd.index).fillna(bd["auction_dollars"]).astype(pd.Int64Dtype())
        )
```

In `simulate_auction`, forward the param in its call to `_simulate_to_state`:

```python
    state = _simulate_to_state(
        strategy,
        my_seat,
        pool,
        config,
        baseline_dollars=baseline_dollars,
        price_jitter=price_jitter,
        rng=rng,
        nomination_temp=nomination_temp,
        bot_archetypes=bot_archetypes,
        bot_dollars=bot_dollars,
    )
```

- [ ] **Step 4: Switch the 4 bot read-sites in `market.py` to `bot_dollars`**

In `src/projections/draft/assistant/auction/market.py`, change exactly these four reads from `"auction_dollars"` to `"bot_dollars"`:

- In `bot_max_bid`: `base = float(baseline_dollars.loc[player["gsis_id"], "bot_dollars"])`
- In `_value_tier`: `inpool = baseline_dollars.loc[baseline_dollars["in_pool"], "bot_dollars"]`
- In `PatientValueBot.max_bid`: `value = float(baseline_dollars.loc[player["gsis_id"], "bot_dollars"])`
- In `BalancedBot.max_bid`: `value = float(baseline_dollars.loc[player["gsis_id"], "bot_dollars"])`

Leave the `in_pool` *mask* in `_value_tier` unchanged (only the value column switches).

- [ ] **Step 5: Update unit-test fixtures that call market/`_value_tier` directly to carry `bot_dollars`**

These tests build raw frames and call the bot functions directly (bypassing the seam), so they need the new column. In `tests/test_draft/test_assistant_auction_market.py`:

`_baseline()` →
```python
def _baseline() -> pd.DataFrame:
    return pd.DataFrame(
        {"in_pool": [True], "auction_dollars": [40], "bot_dollars": [40]},
        index=pd.Index(["00-0000001"], name="gsis_id"),
    )
```

The inline frame in `test_bot_floors_at_min_bid` →
```python
    base = pd.DataFrame(
        {"in_pool": [False], "auction_dollars": [0], "bot_dollars": [0]},
        index=pd.Index(["00-0000001"], name="gsis_id"),
    )
```

`_tiered_baseline()` →
```python
def _tiered_baseline() -> pd.DataFrame:
    ids = [f"00-000000{i}" for i in range(10)]
    dollars = [60, 50, 40, 30, 25, 20, 15, 10, 5, 2]
    return pd.DataFrame(
        {"in_pool": [True] * 10, "auction_dollars": dollars, "bot_dollars": dollars},
        index=pd.Index(ids, name="gsis_id"),
    )
```

In `tests/test_draft/test_assistant_auction_simulation.py`, `_realistic_baseline()` calls `_value_tier` directly in `test_mixed_field_bids_midtier_off_the_dollar_floor`, so add a `bot_dollars` column equal to `auction_dollars`. Change its return to:
```python
    return pd.DataFrame(
        {
            "gsis_id": pd.array(gids, dtype=_PYARROW_STR),
            "auction_dollars": dollars,
            "bot_dollars": dollars,
            "in_pool": [True] * n,
        }
    )
```
(The `_bd(...)` helper frames are passed as `baseline_dollars=` to `_simulate_to_state` and get `bot_dollars` attached by the seam, so `_bd` needs no change.)

- [ ] **Step 6: Update the `AuctionView.baseline_dollars` docstring**

In `src/projections/draft/assistant/auction/bid_strategy.py`, change the `baseline_dollars` field comment in `AuctionView` to note it is the indexed engine frame (`AuctionValuesSchema` columns + the engine-internal `bot_dollars` column the bots read), e.g.:
```python
    baseline_dollars: pd.DataFrame  # indexed engine frame: AuctionValuesSchema cols + bot_dollars
```

- [ ] **Step 7: Run the seam + market + simulation suites; verify pass**

Run: `pytest tests/test_draft/test_assistant_auction_simulation.py tests/test_draft/test_assistant_auction_market.py -q`
Expected: PASS — the two new seam tests, all market tests (now with `bot_dollars` fixtures), and all simulation tests (the seam attaches `bot_dollars` for the `None` default; `_realistic_baseline` carries it for the direct `_value_tier` call).

- [ ] **Step 8: Run gates and commit**

Run: `mypy src tests && ruff check src tests && ruff format --check src tests`
Expected: clean.

```bash
git add src/projections/draft/assistant/auction/simulation.py \
        src/projections/draft/assistant/auction/market.py \
        src/projections/draft/assistant/auction/bid_strategy.py \
        tests/test_draft/test_assistant_auction_simulation.py \
        tests/test_draft/test_assistant_auction_market.py
git commit -m "feat(auction): bots read bot_dollars seam; hero/nomination unchanged (Slice 2 Task 2)"
```

---

## Task 3: `--bot-prices` flag + CLI diagnostic

Wire the bot vector into the tournament behind `bot_prices`, default `espn`, with graceful fallback; add the CLI flag + ESPN-vs-ours diagnostic.

**Files:**
- Modify: `src/projections/draft/assistant/auction/tournament.py`
- Modify: `src/projections/draft/assistant/auction/tournament_cli.py`
- Test: `tests/test_draft/test_assistant_auction_tournament.py`, `tests/test_draft/test_assistant_auction_tournament_cli.py`

- [ ] **Step 1: Write the failing tournament-flag tests**

Append to `tests/test_draft/test_assistant_auction_tournament.py` (helpers `_config`, `_pool`, `_avail`, `run_auction_tournament`, `VarianceParams`, `StaticDollarBid` already imported there; add `pytest` and `pandas as pd` imports if missing — `pandas as pd` is already imported):

```python
import pytest


def _pool_with_inverted_espn(n: int = 40) -> pd.DataFrame:
    """Pool whose ESPN $ are INVERTED vs vorp (best-vorp player gets the lowest ESPN $), so the
    ESPN-anchored bot market diverges hard from SOS."""
    p = _pool(n)
    espn = [int(5 + i) for i in range(n)]  # ascending: worst SOS player priced highest
    p["espn_auction_dollars"] = pd.array(espn, dtype="Int64")
    return p


def test_bot_prices_unknown_raises() -> None:
    pool = _pool(40)
    cfg = _config(6)
    with pytest.raises(ValueError, match="bot_prices"):
        run_auction_tournament(
            {"static": StaticDollarBid()}, pool, cfg,
            my_seat=1, n_seeds=2, price_jitter=0.1, base_seed=0, n_sims=20,
            availability=_avail(pool), params=VarianceParams.load(),
            bot_prices="sos",  # not a valid choice
        )


def test_bot_prices_espn_without_column_warns_and_matches_model() -> None:
    pool = _pool(40)  # no espn_auction_dollars column
    cfg = _config(6)
    common = dict(
        my_seat=1, n_seeds=3, price_jitter=0.1, base_seed=0, n_sims=30,
        availability=_avail(pool), params=VarianceParams.load(),
    )
    with pytest.warns(UserWarning, match="espn"):
        espn = run_auction_tournament({"static": StaticDollarBid()}, pool, cfg, bot_prices="espn", **common)
    model = run_auction_tournament({"static": StaticDollarBid()}, pool, cfg, bot_prices="model", **common)
    assert espn.summaries["static"]["mean_points"].point == model.summaries["static"]["mean_points"].point


def test_bot_prices_espn_with_column_differs_from_model() -> None:
    pool = _pool_with_inverted_espn(40)
    cfg = _config(6)
    common = dict(
        my_seat=1, n_seeds=4, price_jitter=0.1, base_seed=0, n_sims=30,
        availability=_avail(pool), params=VarianceParams.load(),
    )
    espn = run_auction_tournament({"static": StaticDollarBid()}, pool, cfg, bot_prices="espn", **common)
    model = run_auction_tournament({"static": StaticDollarBid()}, pool, cfg, bot_prices="model", **common)
    assert espn.summaries["static"]["mean_points"].point != model.summaries["static"]["mean_points"].point
```

- [ ] **Step 2: Run the flag tests; verify they fail**

Run: `pytest tests/test_draft/test_assistant_auction_tournament.py -k bot_prices -q`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'bot_prices'`.

- [ ] **Step 3: Add `bot_prices` to `run_auction_tournament`**

In `src/projections/draft/assistant/auction/tournament.py`:

Add imports at the top (the module currently has no `warnings` and no `Literal`):
```python
import warnings
from typing import Literal
```
and add `espn_anchored_bot_prices` to the existing `from projections.draft.auction import generate_auction_values` line:
```python
from projections.draft.auction import espn_anchored_bot_prices, generate_auction_values
```

Add the param to `run_auction_tournament`'s keyword-only signature (after `bot_archetypes`):
```python
    bot_prices: Literal["espn", "model"] = "espn",
```

Immediately **after** `baseline_dollars = generate_auction_values(pool, config)  # config-determined; computed once`, insert the bot-vector computation:
```python
    if bot_prices not in ("espn", "model"):
        raise ValueError(f"bot_prices must be 'espn' or 'model'; got {bot_prices!r}")
    bot_dollars: pd.Series | None = None
    if bot_prices == "espn":
        usable = (
            "espn_auction_dollars" in pool.columns
            and bool(pool["espn_auction_dollars"].notna().any())
        )
        if not usable:
            warnings.warn(
                "bot_prices='espn' but pool has no usable espn_auction_dollars; "
                "falling back to model (shared-value) bot pricing.",
                stacklevel=2,
            )
        else:
            try:
                bot_dollars = espn_anchored_bot_prices(pool, config)
            except ValueError as exc:
                warnings.warn(
                    f"espn_anchored_bot_prices failed ({exc}); falling back to model pricing.",
                    stacklevel=2,
                )
                bot_dollars = None
```

Thread `bot_dollars=bot_dollars` into the `simulate_auction(...)` call (add it as a keyword argument alongside `baseline_dollars=baseline_dollars`):
```python
            league = simulate_auction(
                strat,
                my_seat,
                pool,
                config,
                baseline_dollars=baseline_dollars,
                price_jitter=price_jitter,
                rng=np.random.default_rng(base_seed + s),
                nomination_temp=nomination_temp,
                bot_archetypes=bot_archetypes,
                bot_dollars=bot_dollars,
            )
```

- [ ] **Step 4: Run the flag tests; verify they pass**

Run: `pytest tests/test_draft/test_assistant_auction_tournament.py -k bot_prices -q`
Expected: PASS.

- [ ] **Step 5: Write the failing CLI tests**

First inspect `tests/test_draft/test_assistant_auction_tournament_cli.py` for its existing import of `_parse_args`/`run` and fixture helpers; reuse them. Append:

```python
def test_parse_args_bot_prices_defaults_to_espn() -> None:
    args = _parse_args(
        ["--vorp-table", "x.parquet", "--league-config", "c.json",
         "--my-seat", "1", "--season", "2026", "compare"]
    )
    assert args.bot_prices == "espn"


def test_parse_args_bot_prices_accepts_model() -> None:
    args = _parse_args(
        ["--vorp-table", "x.parquet", "--league-config", "c.json",
         "--my-seat", "1", "--season", "2026", "--bot-prices", "model", "compare"]
    )
    assert args.bot_prices == "model"


def test_parse_args_bot_prices_rejects_unknown() -> None:
    import pytest
    with pytest.raises(SystemExit):
        _parse_args(
            ["--vorp-table", "x.parquet", "--league-config", "c.json",
             "--my-seat", "1", "--season", "2026", "--bot-prices", "sos", "compare"]
        )
```

If `_parse_args` is not exported from the CLI module, import it as `from projections.draft.assistant.auction.tournament_cli import _parse_args` (it is module-level).

- [ ] **Step 6: Run the CLI tests; verify they fail**

Run: `pytest tests/test_draft/test_assistant_auction_tournament_cli.py -k bot_prices -q`
Expected: FAIL (`AttributeError: 'Namespace' object has no attribute 'bot_prices'`).

- [ ] **Step 7: Add `--bot-prices` + diagnostic to `tournament_cli.py`**

In `src/projections/draft/assistant/auction/tournament_cli.py`:

Add a `Literal` import and `generate_auction_values` import:
```python
from typing import Literal
from projections.draft.auction import generate_auction_values
```

In `_parse_args`, add the flag to the **top-level** parser (before `sub = p.add_subparsers(...)`):
```python
    p.add_argument(
        "--bot-prices",
        choices=("espn", "model"),
        default="espn",
        help="Bot pricing anchor: 'espn' (real ESPN auction values) or 'model' (shared SOS).",
    )
```

Add a diagnostic helper:
```python
def _format_espn_diagnostic(pool: pd.DataFrame, config: LeagueConfig) -> str:
    """Largest our$-vs-ESPN$ gaps (value_delta = our SOS dollars - ESPN dollars). Skipped when
    the pool carries no usable espn_auction_dollars."""
    usable = (
        "espn_auction_dollars" in pool.columns and bool(pool["espn_auction_dollars"].notna().any())
    )
    if not usable:
        return "ESPN diagnostic: no usable espn_auction_dollars on the pool (skipped)."
    ref = pool.loc[
        pool["espn_auction_dollars"].notna(), ["gsis_id", "espn_auction_dollars"]
    ].rename(columns={"espn_auction_dollars": "reference_dollars"})
    diag = generate_auction_values(pool, config, reference_prices=ref)
    priced = diag[diag["reference_dollars"].notna()].copy()
    priced = priced.sort_values("value_delta")
    lines = ["ESPN vs ours (value_delta = our SOS $ - ESPN $); most negative = ESPN richer:"]
    for _, row in priced.head(5).iterrows():
        lines.append(
            f"  {row['gsis_id']}: ours ${int(row['auction_dollars'])} "
            f"ESPN ${int(row['reference_dollars'])} delta {int(row['value_delta']):+d}"
        )
    for _, row in priced.tail(5).iterrows():
        lines.append(
            f"  {row['gsis_id']}: ours ${int(row['auction_dollars'])} "
            f"ESPN ${int(row['reference_dollars'])} delta {int(row['value_delta']):+d}"
        )
    return "\n".join(lines)
```

In `run`, narrow the flag to a `Literal`, thread it into the tournament, and print the diagnostic when in espn mode:
```python
    bot_prices: Literal["espn", "model"] = "espn" if args.bot_prices == "espn" else "model"
    result = run_auction_tournament(
        _MODELS,
        pool,
        config,
        my_seat=args.my_seat,
        n_seeds=args.seeds,
        price_jitter=args.price_jitter,
        base_seed=args.seed,
        n_sims=args.n_sims,
        availability=availability,
        params=params,
        nomination_temp=args.nomination_temp,
        bot_archetypes=_REALISTIC_FIELD,
        bot_prices=bot_prices,
    )
    print(format_compare(result))
    if bot_prices == "espn":
        print()
        print(_format_espn_diagnostic(pool, config))
    return 0
```

- [ ] **Step 8: Run the CLI tests; verify they pass**

Run: `pytest tests/test_draft/test_assistant_auction_tournament_cli.py -k bot_prices -q`
Expected: PASS.

- [ ] **Step 9: Full gates + auction/schemas sweep, then commit**

Run: `pytest -q -k "auction or schemas" && mypy src tests && ruff check src tests && ruff format --check src tests`
Expected: clean.

```bash
git add src/projections/draft/assistant/auction/tournament.py \
        src/projections/draft/assistant/auction/tournament_cli.py \
        tests/test_draft/test_assistant_auction_tournament.py \
        tests/test_draft/test_assistant_auction_tournament_cli.py
git commit -m "feat(auction): --bot-prices flag (default espn) + ESPN-vs-ours diagnostic (Slice 2 Task 3)"
```

---

## Task 4: Validation bake-off (Run H) — ops + report

No new code. Data prerequisite + the bake-off + the write-up. This is the only task that needs live data; it does not block Tasks 1–3.

**Files:**
- Modify: `reports/auction_tournament_validation_2026.md`, `TODO.md`, `project_management.md`

- [ ] **Step 1: Re-ingest ESPN + regenerate preset tables (R8)**

```bash
python -m projections.ingest.external_projections --season 2026
python -m projections.consensus.refresh --season 2026
python scripts/generate_preset_vorp_tables.py --season 2026
```
Confirm `data/vorp_2026/half_12team.parquet` and `data/vorp_2026/half_16team.parquet` exist and that `espn_auction_dollars` is populated (non-NA for the top ~150–200 players):
```bash
python -c "import pandas as pd; d=pd.read_parquet('data/vorp_2026/half_16team.parquet'); print(d['espn_auction_dollars'].notna().sum(), 'priced of', len(d))"
```
If the 16-team table raises during generation (a position can't fill the 16-team pool, R8), note it and fall back to 12-team only for Run H, recording the limitation.

- [ ] **Step 2: Run the bake-off at half-PPR 12-team and 16-team**

```bash
python -m projections.draft.assistant.auction.tournament_cli \
  --vorp-table data/vorp_2026/half_12team.parquet \
  --league-config configs/league_espn_half_12team_skill.json \
  --my-seat 6 --season 2026 --seeds 40 --bot-prices espn compare
```
(and the same with `half_16team.parquet` + the 16-team config + an appropriate `--my-seat`). Use the league-config paths that match the preset tables — check `configs/` for the exact half-PPR 12/16-team filenames (the presets in `scripts/generate_preset_vorp_tables.py` reference them). Capture stdout (per-model table + the ESPN-vs-ours diagnostic). For a direct A/B reference, also run one with `--bot-prices model`.

- [ ] **Step 3: Write Run H into the validation report**

Append a **Run H** section to `reports/auction_tournament_validation_2026.md` with: the per-model table (all 8 models × the 5 `METRICS` × {12,16}-team), the ESPN-vs-ours `value_delta` summary, and a note on the observed deep-league (16-team) stud inflation. Frame it as data-gathering — **no winner declared** (the September decision is unchanged).

- [ ] **Step 4: Update TODO + PM, then commit**

Mark TODO #49c Slice 2 SHIPPED (bot WTP now ESPN-anchored behind `--bot-prices`, default espn; Run H recorded) and add a PM dated entry. Commit the report + docs (the regenerated `data/vorp_2026/*.parquet` stay untracked per TODO #48).

```bash
git add reports/auction_tournament_validation_2026.md TODO.md project_management.md
git commit -m "report(auction): Run H — ESPN-anchored bots bake-off (Slice 2 Task 4, TODO #49c)"
```

---

## Self-Review

- **Spec coverage:** R1 (extract `_allocate_surplus`, byte-identical) → Task 1 Steps 3–4. R2 (`espn_anchored_bot_prices`) → Task 1. R3 (seam + 4 read-sites) → Task 2. R4 (`bot_prices`, fallback, single call site) → Task 3 Step 3. R5 (CLI flag + diagnostic) → Task 3 Steps 7. R6 (back-compat / clean A/B) → Task 2 Step 1 (`bot_dollars=None` reproduces baseline) + the unchanged existing suites. R7 (`bot_dollars` engine-internal, `Literal`, dtypes) → Tasks 2–3. R8 (data prerequisite + 16-team feasibility) → Task 4 Step 1.
- **Placeholder scan:** every code step shows complete code; the only "inspect existing file" instruction (Task 3 Step 5) is to reuse an existing import, with a concrete fallback given.
- **Type consistency:** `bot_dollars: pd.Series | None` is the same name/type across `simulation.py` (param), `tournament.py` (local + thread), and the producer's return. `bot_prices: Literal["espn","model"]` consistent in `run_auction_tournament` and the CLI narrowing. The producer returns `Int64` indexed by `gsis_id`; the seam `reindex(bd.index).fillna(...).astype(pd.Int64Dtype())` consumes exactly that.
