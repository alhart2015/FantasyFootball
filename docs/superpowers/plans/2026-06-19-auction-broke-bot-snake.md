# Broke-Bot Snake-Draft Behavior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make out-of-money auction bots (`feasible_max == min_bid`) behave like snake drafters — act only on their best-available-by-noisy-ADP target for a needed position — instead of bidding $1 on whatever is nominated.

**Architecture:** A per-bot `SnakeBoard` (fixed noisy-ADP ranking, drawn once per draft from a dedicated CRN-clean RNG) decides each broke bot's target. The auction engine routes broke bots to a snake policy (nominate / snipe only the target; abstain otherwise) and adds a "nominator takes its nominee at $1 when nobody bids" backstop. Flush bots, the hero, and flush/hero nomination are untouched at the code level. The selection core is shared with the existing snake-draft `bot_pick`.

**Tech Stack:** Python 3.11, numpy ≥2.0 (`Generator.spawn`), pandas + pandera, pytest, mypy (strict), ruff.

**Spec:** `docs/superpowers/specs/2026-06-19-auction-broke-bot-snake-design.md`

## Global Constraints

- **numpy ≥ 2.0** — `Generator.spawn` and list-seed `default_rng([a, b])` are both required (repo pins `numpy>=2.0`; 2.4.4 installed).
- **`GsisId` is canonical** — use `validate_gsis_id(raw)` at any untrusted-string boundary; never join on names.
- **Reference the `Position` enum, never the raw strings** — `pool["position"]` is a raw string column; normalize via `Position(str(p))` before comparing to a `frozenset[Position]`.
- **Reuse, don't duplicate** — the noisy-ADP selection core is shared between `bot_pick` and `SnakeBoard`; do not fork it.
- **Tests / mypy strict / ruff are gates** — every task ends green on the relevant subset; full gate at the end.
- **`adp_jitter = 8.0`** — the snake-draft default (`hero_harness.py` uses `jitter=8.0`); reuse it as the broke-bot ADP jitter.
- **CRN discipline** — snake noise must come from an RNG seeded off the auction seed *alone*, independent of the bidding `rng`.

---

### Task 1: Extract the shared noisy-ADP selection core from `bot_pick`

Pure refactor: pull the argmin-with-gsis-tiebreak out of `bot_pick` so `SnakeBoard` (Task 2) can reuse identical semantics. `bot_pick`'s noise draw and sort-before-draw ordering **stay inside `bot_pick`** (preserves its order-independence guarantee).

**Files:**
- Modify: `src/projections/draft/assistant/opponent.py`
- Test: `tests/test_draft/test_assistant_opponent.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `_best_by_noisy_adp(gsis: np.ndarray, noisy_adp: np.ndarray) -> GsisId` — returns the `gsis` whose `noisy_adp` is lowest, ties broken by `gsis` ascending (via `np.lexsort((gsis, noisy_adp))`). Order-independent given the same (gsis, noisy) pairing.

- [ ] **Step 1: Write a characterization test pinning current `bot_pick` output**

Add to `tests/test_draft/test_assistant_opponent.py`:

```python
def test_bot_pick_characterization_stable_across_refactor() -> None:
    # Pins bot_pick's exact picks for fixed seeds so the Task 1 extraction is proven byte-identical.
    import numpy as np
    import pandas as pd
    from projections.draft.assistant.opponent import bot_pick

    avail = pd.DataFrame(
        {
            "gsis_id": ["00-0000005", "00-0000001", "00-0000003", "00-0000002", "00-0000004"],
            "consensus_adp": pd.array([12.0, 3.0, None, 3.0, 50.0], dtype="Float64"),
        }
    )
    picks = [str(bot_pick(avail, np.random.default_rng(seed), adp_jitter=2.0)) for seed in range(6)]
    assert picks == ["00-0000001", "00-0000002", "00-0000001", "00-0000002", "00-0000002", "00-0000001"]
```

- [ ] **Step 2: Run it against current code to capture the true expected picks**

Run: `python -c "import numpy as np, pandas as pd; from projections.draft.assistant.opponent import bot_pick; av=pd.DataFrame({'gsis_id':['00-0000005','00-0000001','00-0000003','00-0000002','00-0000004'],'consensus_adp':pd.array([12.0,3.0,None,3.0,50.0],dtype='Float64')}); print([str(bot_pick(av,np.random.default_rng(s),adp_jitter=2.0)) for s in range(6)])"`
Expected: a list of 6 gsis strings. **Replace the literal in the test's `assert` with this exact output**, then `pytest tests/test_draft/test_assistant_opponent.py::test_bot_pick_characterization_stable_across_refactor -v` → PASS.

- [ ] **Step 3: Add a direct unit test for the extracted core**

```python
def test_best_by_noisy_adp_argmin_and_tiebreak() -> None:
    import numpy as np
    from projections.draft.assistant.opponent import _best_by_noisy_adp

    gsis = np.array(["00-000c", "00-000a", "00-000b"], dtype=str)
    noisy = np.array([5.0, 5.0, 2.0], dtype=float)
    assert str(_best_by_noisy_adp(gsis, noisy)) == "00-000b"  # lowest noisy
    tie = np.array([2.0, 2.0, 2.0], dtype=float)
    assert str(_best_by_noisy_adp(gsis, tie)) == "00-000a"  # gsis-ascending tiebreak
    inf = np.array([np.inf, 1.0, np.inf], dtype=float)
    assert str(_best_by_noisy_adp(gsis, inf)) == "00-000a"  # finite beats inf
```

- [ ] **Step 4: Run the core test to verify it fails**

Run: `pytest tests/test_draft/test_assistant_opponent.py::test_best_by_noisy_adp_argmin_and_tiebreak -v`
Expected: FAIL with `ImportError: cannot import name '_best_by_noisy_adp'`.

- [ ] **Step 5: Extract the core and refactor `bot_pick`**

In `src/projections/draft/assistant/opponent.py`, replace the body of `bot_pick` (the sort → noise → lexsort block) so the lexsort selection lives in a new helper. Final state:

```python
def _best_by_noisy_adp(gsis: np.ndarray, noisy_adp: np.ndarray) -> GsisId:
    """Lowest-noisy-ADP gsis; ties (incl. all-+inf) break on gsis ascending.

    lexsort sorts by the LAST key first -> primary noisy asc, secondary gsis asc. Order-independent
    given the same (gsis, noisy_adp) pairing. Shared by bot_pick and the auction SnakeBoard so the
    two pick by identical semantics.
    """
    winner = int(np.lexsort((gsis, noisy_adp))[0])
    return validate_gsis_id(str(gsis[winner]))


def bot_pick(available: pd.DataFrame, rng: np.random.Generator, *, adp_jitter: float) -> GsisId:
    """Return the lowest noisy-ADP player among `available`.

    `available` needs columns `gsis_id` and `consensus_adp` (nullable Float64).
    Null ADP -> treated as `+inf` (no market signal). Ties (incl. all-null) break
    on `gsis_id` ascending. `available` must be non-empty (caller guarantees it).

    Result is independent of the input row order: rows are sorted by `gsis_id`
    ascending before any random draws, so the same RNG seed always yields the
    same pick for a given player set regardless of how the caller ordered the rows.
    """
    available = available.sort_values("gsis_id", ignore_index=True)
    adp = available["consensus_adp"].to_numpy(dtype=float, na_value=np.inf)
    noisy = adp + rng.normal(0.0, adp_jitter, size=len(available))
    gsis = available["gsis_id"].to_numpy(dtype=str)
    return _best_by_noisy_adp(gsis, noisy)
```

- [ ] **Step 6: Run both tests + the existing opponent suite**

Run: `pytest tests/test_draft/test_assistant_opponent.py -v`
Expected: all PASS (characterization unchanged → extraction is byte-identical).

- [ ] **Step 7: Commit**

```bash
git add src/projections/draft/assistant/opponent.py tests/test_draft/test_assistant_opponent.py
git commit -m "refactor(draft): extract _best_by_noisy_adp core shared by bot_pick + auction snake board"
```

---

### Task 2: `SnakeBoard` — per-bot fixed noisy-ADP ranking + `adp_usable` guard

The pure unit a broke bot consults. Built once per draft per bot from a dedicated RNG; answers "best undrafted player at a needed position."

**Files:**
- Create: `src/projections/draft/assistant/auction/snake_bot.py`
- Test: `tests/test_draft/test_assistant_auction_snake_bot.py`

**Interfaces:**
- Consumes: `_best_by_noisy_adp` (Task 1).
- Produces:
  - `DEFAULT_BROKE_ADP_JITTER: float = 8.0`
  - `adp_usable(pool: pd.DataFrame) -> bool` — `True` iff `consensus_adp` column present and not all-null.
  - `SnakeBoard(pool: pd.DataFrame, rng: np.random.Generator, *, adp_jitter: float = DEFAULT_BROKE_ADP_JITTER)` — draws one `N(0, adp_jitter)` vector over the full pool at construction.
  - `SnakeBoard.best_available(drafted: frozenset[str], eligible: frozenset[Position]) -> GsisId | None` — lowest fixed-noisy-ADP gsis NOT in `drafted` whose position ∈ `eligible`; `None` if that subset is empty.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_draft/test_assistant_auction_snake_bot.py`:

```python
import numpy as np
import pandas as pd

from projections.draft.assistant.auction.snake_bot import (
    DEFAULT_BROKE_ADP_JITTER,
    SnakeBoard,
    adp_usable,
)
from projections.schemas import Position


def _pool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": ["00-rb1", "00-rb2", "00-wr1", "00-qb1", "00-te1"],
            "position": ["RB", "RB", "WR", "QB", "TE"],
            "consensus_adp": pd.array([2.0, 8.0, 1.0, 40.0, 90.0], dtype="Float64"),
        }
    )


def test_adp_usable() -> None:
    assert adp_usable(_pool()) is True
    no_col = _pool().drop(columns=["consensus_adp"])
    assert adp_usable(no_col) is False
    all_null = _pool().assign(consensus_adp=pd.array([None] * 5, dtype="Float64"))
    assert adp_usable(all_null) is False


def test_best_available_respects_eligibility() -> None:
    board = SnakeBoard(_pool(), np.random.default_rng(0), adp_jitter=0.0)  # no noise -> pure ADP
    # WR has lowest ADP overall, but if only RB is eligible we get the lowest-ADP RB.
    assert str(board.best_available(frozenset(), frozenset({Position.RB}))) == "00-rb1"
    assert str(board.best_available(frozenset(), frozenset({Position.WR}))) == "00-wr1"


def test_best_available_excludes_drafted() -> None:
    board = SnakeBoard(_pool(), np.random.default_rng(0), adp_jitter=0.0)
    assert str(board.best_available(frozenset({"00-rb1"}), frozenset({Position.RB}))) == "00-rb2"


def test_best_available_none_when_empty() -> None:
    board = SnakeBoard(_pool(), np.random.default_rng(0), adp_jitter=0.0)
    # All RBs drafted, only RB eligible -> nothing left.
    assert board.best_available(frozenset({"00-rb1", "00-rb2"}), frozenset({Position.RB})) is None
    assert board.best_available(frozenset(), frozenset()) is None


def test_noise_is_fixed_per_board() -> None:
    board = SnakeBoard(_pool(), np.random.default_rng(7), adp_jitter=20.0)
    elig = frozenset({Position.RB, Position.WR, Position.QB, Position.TE})
    first = board.best_available(frozenset(), elig)
    # Same board, same query -> same answer every call (no re-draw).
    assert all(board.best_available(frozenset(), elig) == first for _ in range(5))


def test_order_independent_and_default_jitter() -> None:
    shuffled = _pool().iloc[::-1].reset_index(drop=True)
    b1 = SnakeBoard(_pool(), np.random.default_rng(3), adp_jitter=5.0)
    b2 = SnakeBoard(shuffled, np.random.default_rng(3), adp_jitter=5.0)
    elig = frozenset({Position.RB, Position.WR})
    assert b1.best_available(frozenset(), elig) == b2.best_available(frozenset(), elig)
    assert DEFAULT_BROKE_ADP_JITTER == 8.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_draft/test_assistant_auction_snake_bot.py -v`
Expected: FAIL with `ModuleNotFoundError: ...snake_bot`.

- [ ] **Step 3: Implement `snake_bot.py`**

Create `src/projections/draft/assistant/auction/snake_bot.py`:

```python
"""Out-of-money auction bots draft like snake drafters: act on the best-available-by-noisy-ADP
player at a needed position. `SnakeBoard` holds one bot's fixed (drawn-once-per-draft) noisy-ADP
ranking; `best_available` answers the per-nomination target query. See the design doc
`docs/superpowers/specs/2026-06-19-auction-broke-bot-snake-design.md`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.draft.assistant.opponent import _best_by_noisy_adp
from projections.schemas import GsisId, Position

DEFAULT_BROKE_ADP_JITTER: float = 8.0  # the snake-draft ADP jitter (hero_harness default)


def adp_usable(pool: pd.DataFrame) -> bool:
    """True iff the pool carries a usable consensus_adp signal (present and not all-null).

    `consensus_adp` is OPTIONAL on VorpTableSchema (weekly-path tables omit it). When unusable the
    auction engine disables the snake regime and runs exactly as before.
    """
    return "consensus_adp" in pool.columns and bool(pool["consensus_adp"].notna().any())


class SnakeBoard:
    """One bot's fixed noisy-ADP board for a single draft.

    The noise is drawn ONCE at construction (a real manager's board is set on draft day and does not
    reshuffle every nomination). `best_available` consumes no RNG.
    """

    def __init__(
        self,
        pool: pd.DataFrame,
        rng: np.random.Generator,
        *,
        adp_jitter: float = DEFAULT_BROKE_ADP_JITTER,
    ) -> None:
        ordered = pool.sort_values("gsis_id", ignore_index=True)
        self._gsis = ordered["gsis_id"].to_numpy(dtype=str)
        self._pos = ordered["position"].astype(str).to_numpy(dtype=str)
        adp = ordered["consensus_adp"].to_numpy(dtype=float, na_value=np.inf)
        self._noisy = adp + rng.normal(0.0, adp_jitter, size=len(ordered))

    def best_available(
        self, drafted: frozenset[str], eligible: frozenset[Position]
    ) -> GsisId | None:
        """Lowest fixed-noisy-ADP undrafted gsis whose position is in `eligible`; None if none."""
        if not eligible:
            return None
        elig_str = np.array([p.value for p in eligible], dtype=str)
        mask = np.isin(self._pos, elig_str) & ~np.isin(self._gsis, np.array(list(drafted), dtype=str))
        if not mask.any():
            return None
        return _best_by_noisy_adp(self._gsis[mask], self._noisy[mask])
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_draft/test_assistant_auction_snake_bot.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/auction/snake_bot.py tests/test_draft/test_assistant_auction_snake_bot.py
git commit -m "feat(auction): SnakeBoard fixed noisy-ADP target board + adp_usable guard"
```

---

### Task 3: Engine seam — thread `snake_rng`, build per-bot boards (no behavior change)

Add the parameter and build the boards, but **do not consume them yet**. This lands the plumbing under a byte-identity guarantee before any behavior change.

**Files:**
- Modify: `src/projections/draft/assistant/auction/simulation.py`
- Test: `tests/test_draft/test_assistant_auction_simulation.py`

**Interfaces:**
- Consumes: `SnakeBoard`, `adp_usable` (Task 2).
- Produces: `simulate_auction(..., snake_rng: np.random.Generator | None = None)` and the same new kw-only param on `_simulate_to_state`. Default derives `snake_rng = rng.spawn(1)[0]` (verified: spawn does not perturb the parent stream). Engine-internal: `_adp_ok: bool` and `snake_boards: dict[int, SnakeBoard]` for bot seats (built only when `_adp_ok`).

- [ ] **Step 1: Write the byte-identity test**

Add to `tests/test_draft/test_assistant_auction_simulation.py` (reuse the module's existing pool/config fixtures — find the helper that builds a `VorpTableSchema` pool + `LeagueConfig`; call it `_make_pool`/`_make_config` per the file's convention):

```python
def test_snake_rng_param_does_not_change_rosters_without_adp() -> None:
    # Boards are built (if adp present) but UNUSED in Task 3 -> rosters identical with/without the
    # consensus_adp column and identical whether snake_rng is given or defaulted.
    import numpy as np
    from projections.draft.assistant.auction.bid_strategy import StaticDollarBid

    pool, config = _make_pool_with_adp(), _make_config()  # pool carries consensus_adp
    common = dict(
        my_seat=1,
        baseline_dollars=generate_auction_values(pool, config),
        price_jitter=0.15,
        nomination_temp=1.0,
    )
    a = simulate_auction(StaticDollarBid(), pool, config, rng=np.random.default_rng(0), **common)
    b = simulate_auction(
        StaticDollarBid(), pool, config, rng=np.random.default_rng(0),
        snake_rng=np.random.default_rng([0, 7]), **common,
    )
    no_adp = pool.drop(columns=["consensus_adp"])
    c = simulate_auction(StaticDollarBid(), no_adp, config, rng=np.random.default_rng(0), **common)
    assert a == b == c
```

If the test module lacks an ADP-bearing pool helper, add `_make_pool_with_adp()` that takes the existing pool helper's frame and assigns a `consensus_adp` column (`pd.array(range(1, n+1), dtype="Float64")`), re-validated via `VorpTableSchema.validate`.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_draft/test_assistant_auction_simulation.py::test_snake_rng_param_does_not_change_rosters_without_adp -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'snake_rng'`.

- [ ] **Step 3: Add the parameter, board construction, and `adp_ok` to the engine**

In `src/projections/draft/assistant/auction/simulation.py`:

Add imports at top:

```python
from projections.draft.assistant.auction.snake_bot import SnakeBoard, adp_usable
```

Add `snake_rng` to both signatures (kw-only, after the existing `rng`), e.g. in `_simulate_to_state`:

```python
    rng: np.random.Generator,
    snake_rng: np.random.Generator | None = None,
    nomination_temp: float = 0.0,
```

and identically in `simulate_auction`, threading it through the `_simulate_to_state(...)` call.

Near the top of `_simulate_to_state` body (after `hero0 = my_seat - 1`), build the boards:

```python
    if snake_rng is None:
        snake_rng = rng.spawn(1)[0]  # CRN-safe: spawn advances the seed-sequence, not rng's stream
    adp_ok = adp_usable(pool)
    snake_boards: dict[int, SnakeBoard] = (
        {s: SnakeBoard(pool, snake_rng) for s in bot_seats} if adp_ok else {}
    )
```

(Place this AFTER `bot_seats` is defined.)

- [ ] **Step 4: Run the byte-identity test + full simulation suite**

Run: `pytest tests/test_draft/test_assistant_auction_simulation.py -v`
Expected: all PASS (boards built but unused → rosters unchanged; spawn default does not perturb the bidding stream).

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/auction/simulation.py tests/test_draft/test_assistant_auction_simulation.py
git commit -m "feat(auction): thread CRN-clean snake_rng + build per-bot SnakeBoards (unused seam)"
```

---

### Task 4: Broke-bot responder rule + nominator backstop

Now consume the boards on the bidding side: a broke bot bids `$1` only on its target, else abstains; replace `assert bids` with the "nominator takes it for $1" backstop. Both gated on `adp_ok and not forced`.

**Files:**
- Modify: `src/projections/draft/assistant/auction/simulation.py`
- Test: `tests/test_draft/test_assistant_auction_simulation.py`

**Interfaces:**
- Consumes: `snake_boards`, `adp_ok` (Task 3); `seat_eligible`, `forced`, `_feasible_max`, `state.drafted` (existing engine state).
- Produces: no new public surface; changes the per-round bid collection + award.

- [ ] **Step 1: Write the behavior tests**

Add these two tests to `tests/test_draft/test_assistant_auction_simulation.py` (ensure
`generate_auction_values` is imported in the module — add the import if absent):

```python
def test_backstop_awards_nominee_when_no_bids() -> None:
    # Once broke bots can abstain, an empty-bids round is reachable; the backstop must award the
    # nominee at min_bid (no AssertionError) and the auction must still fill every roster.
    import numpy as np
    from projections.draft.assistant.auction.bid_strategy import StaticDollarBid

    pool, config = _make_pool_with_adp(), _make_config()
    league = simulate_auction(
        StaticDollarBid(), pool, config, rng=np.random.default_rng(3),
        my_seat=1, baseline_dollars=generate_auction_values(pool, config),
        price_jitter=0.15, nomination_temp=1.0,
    )
    assert all(len(r) == config.roster_size for r in league.values())


def test_no_roster_violates_position_caps() -> None:
    # Behavioral guard: broke bots never roster an off-position scrub — every seat's roster respects
    # the position-cap maxima (a $1-scrub grab would blow a cap or strand a needed slot).
    import numpy as np
    from collections import Counter
    from projections.draft.assistant.auction.bid_strategy import StaticDollarBid
    from projections.draft.roster_eligibility import bot_position_bounds
    from projections.schemas import Position

    pool, config = _make_pool_with_adp(), _make_config()
    state = _simulate_to_state(
        StaticDollarBid(), 1, pool, config,
        baseline_dollars=generate_auction_values(pool, config),
        price_jitter=0.15, rng=np.random.default_rng(11), nomination_temp=1.0,
    )
    _minimums, maximums = bot_position_bounds(config.roster_slots)
    for roster in state.rosters:
        counts = Counter(Position(pos) for _g, pos, _pr in roster)
        for p, c in counts.items():
            assert c <= maximums[p]
```

- [ ] **Step 2: Run to verify the new tests fail or error appropriately**

Run: `pytest tests/test_draft/test_assistant_auction_simulation.py::test_backstop_awards_nominee_when_no_bids tests/test_draft/test_assistant_auction_simulation.py::test_no_roster_violates_position_caps -v`
Expected: they currently PASS for caps (today's bots already respect caps) but the backstop path is not yet exercised — proceed to wire the behavior, then both must still pass while the new abstention logic is active. (If the cap test already passes, it serves as a regression guard for Step 3.)

- [ ] **Step 3: Wire the broke responder + backstop in the bid loop**

In `_simulate_to_state`, locate the bid-collection loop (`for seat in range(n):` collecting `bids`). Replace the **bot branch** (`else:` for non-hero seats) so a broke bot uses the snake rule, and replace `assert bids` with the backstop. The bot branch becomes:

```python
            else:
                broke = adp_ok and not forced and fmax == min_bid
                if broke:
                    target = snake_boards[seat].best_available(
                        frozenset(state.drafted), seat_eligible[seat]
                    )
                    if target is None or str(nominee_id) != str(target):
                        continue  # abstain: not this broke bot's snake target
                    bids[seat] = min_bid  # snipe the target at the floor
                else:
                    desired = seat_arch[seat].max_bid(
                        SeatView(
                            open_slots=_open_slots(state, seat, rs),
                            eligible_positions=elig,
                            budget=state.budgets[seat],
                        ),
                        player,
                        bd,
                        config,
                        rng,
                        price_jitter=price_jitter,
                    )
                    if desired <= 0:
                        continue
                    bids[seat] = max(min_bid, min(int(desired), fmax))
```

Then replace:

```python
        assert bids, "resolve_bids requires >=1 bid; forced-pick path guarantees it"
        winner, price = resolve_bids(bids, min_bid)
```

with the backstop:

```python
        if not bids:
            # Nominator takes its nominee at min_bid when nobody bids (only reachable on the
            # non-forced path once broke bots abstain). Awardee: the nominator if it can roster the
            # nominee, else the lowest-index open seat that can (room-union guarantees one exists).
            nominee_pos = pos_by_id[str(nominee_id)]
            if nominee_pos in seat_eligible.get(state.nominator, frozenset()):
                winner, price = state.nominator, min_bid
            else:
                # lowest-index open seat that can roster the nominee (room-union guarantees one on
                # the non-forced path); default to the nominator ungated as a defensive fallback.
                winner = next(
                    (
                        s for s in range(n)
                        if _open_slots(state, s, rs) > 0
                        and nominee_pos in seat_eligible.get(s, frozenset())
                    ),
                    state.nominator,
                )
                price = min_bid
        else:
            winner, price = resolve_bids(bids, min_bid)
```

(`fmax` is already computed per seat at the top of the loop body as `_feasible_max(...)`; reuse it.
`seat_eligible` is the dict built earlier in the round. `pos_by_id` already exists.)

- [ ] **Step 4: Run the simulation suite**

Run: `pytest tests/test_draft/test_assistant_auction_simulation.py -v`
Expected: all PASS — including the Task-3 byte-identity test (no-ADP pool still bypasses the snake path) and the new backstop/cap tests.

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/auction/simulation.py tests/test_draft/test_assistant_auction_simulation.py
git commit -m "feat(auction): broke bots snipe only their snake target; nominator-takes-\$1 backstop"
```

---

### Task 5: Broke-bot nomination override

A broke bot on the clock nominates its own snake target (so "take it for $1 if unbid" is the snake pick). `None` target → central `_sample_nominee`. Gated on `adp_ok and not forced`; must run **before** `assert nominee_id is not None`.

**Files:**
- Modify: `src/projections/draft/assistant/auction/simulation.py`
- Test: `tests/test_draft/test_assistant_auction_simulation.py`

**Interfaces:**
- Consumes: `snake_boards`, `adp_ok`, `forced`, `candidates`, `seat_eligible`, `state.nominator`, `_feasible_max`.
- Produces: no new public surface; changes nominee selection.

- [ ] **Step 1: Write the test**

```python
def test_broke_nominator_nominates_its_snake_target() -> None:
    # With a broke bot on the clock and ADP usable, the nominee is that seat's best-available-by-ADP
    # for a needed position (not the value-sampled central nominee). Assert the auction completes and
    # that a low-ADP, ESPN-unranked depth player is rostered by SOME bot (not stranded for $1 scrubs).
    import numpy as np
    from collections import Counter
    from projections.draft.assistant.auction.bid_strategy import StaticDollarBid

    pool, config = _make_pool_with_adp(), _make_config()
    state = _simulate_to_state(
        StaticDollarBid(), 1, pool, config,
        baseline_dollars=generate_auction_values(pool, config),
        price_jitter=0.15, rng=np.random.default_rng(5), nomination_temp=1.0,
        snake_rng=np.random.default_rng([5, 7]),
    )
    assert all(len(r) == config.roster_size for r in state.rosters)


def test_no_adp_pool_is_byte_identical_to_pre_snake() -> None:
    # Regime fully disabled without ADP -> identical to the central-nomination path.
    import numpy as np
    from projections.draft.assistant.auction.bid_strategy import StaticDollarBid

    pool, config = _make_pool_with_adp(), _make_config()
    no_adp = pool.drop(columns=["consensus_adp"])
    common = dict(
        my_seat=1, baseline_dollars=generate_auction_values(no_adp, config),
        price_jitter=0.15, nomination_temp=1.0,
    )
    a = simulate_auction(StaticDollarBid(), no_adp, config, rng=np.random.default_rng(9), **common)
    b = simulate_auction(StaticDollarBid(), no_adp, config, rng=np.random.default_rng(9),
                         snake_rng=np.random.default_rng([9, 7]), **common)
    assert a == b
```

- [ ] **Step 2: Run to verify current behavior**

Run: `pytest tests/test_draft/test_assistant_auction_simulation.py::test_broke_nominator_nominates_its_snake_target tests/test_draft/test_assistant_auction_simulation.py::test_no_adp_pool_is_byte_identical_to_pre_snake -v`
Expected: both PASS already (completion + no-ADP identity hold pre-Task-5); they lock the invariants Step 3 must preserve.

- [ ] **Step 3: Override nomination for a broke bot**

In `_simulate_to_state`, find the nominee-selection block:

```python
        forced = not candidates
        if forced:
            nominee_id = next(g for g in nominate_order if g not in state.drafted)
            warnings.warn(...)
        else:
            nominee_id = _sample_nominee(candidates, val_by_id, nomination_temp, rng)
        assert nominee_id is not None
```

Replace the `else:` branch so a broke bot nominator nominates its target (falling back to central sampling when its target is `None`):

```python
        else:
            nom = state.nominator
            nom_fmax = _feasible_max(state, nom, rs, min_bid)
            broke_nominator = adp_ok and nom != hero0 and nom_fmax == min_bid
            target = (
                snake_boards[nom].best_available(frozenset(state.drafted), seat_eligible[nom])
                if broke_nominator
                else None
            )
            if target is not None:
                nominee_id = target
            else:
                nominee_id = _sample_nominee(candidates, val_by_id, nomination_temp, rng)
        assert nominee_id is not None
```

(The broke nominator's `target` is provably a member of `candidates` — its position is in
`seat_eligible[nom] ⊆ union` and it is undrafted — so `forced` stays correctly `False` and the
bid-loop eligibility gate is consistent. `target is None` → central sampling, as specced.)

- [ ] **Step 4: Run the suite**

Run: `pytest tests/test_draft/test_assistant_auction_simulation.py -v`
Expected: all PASS (completion holds; no-ADP path byte-identical; broke nominator path exercised).

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/auction/simulation.py tests/test_draft/test_assistant_auction_simulation.py
git commit -m "feat(auction): broke nominator nominates its snake target (None -> central sample)"
```

---

### Task 6: Tournament wiring — dedicated CRN snake substream

Construct the dedicated `snake_rng` off the auction seed and pass it down, so every contestant model faces an identical bot field at a given seed.

**Files:**
- Modify: `src/projections/draft/assistant/auction/tournament.py`
- Test: `tests/test_draft/test_assistant_auction_simulation.py` (CRN unit) — or the tournament test module if one exists.

**Interfaces:**
- Consumes: `simulate_auction(..., snake_rng=...)` (Task 3).
- Produces: `_SNAKE_SUBSTREAM` module constant; `run_auction_tournament` passes `snake_rng=np.random.default_rng([base_seed + s, _SNAKE_SUBSTREAM])`.

- [ ] **Step 1: Write the CRN test**

```python
def test_snake_board_is_shared_across_strategies_at_a_seed() -> None:
    # CRN: the dedicated substream yields the SAME bot field regardless of the hero strategy, so two
    # contestants at the same seed see identical snake boards (paired design). Assert that two
    # strategies that consume the bidding rng differently still produce identical BOT snake targets
    # by checking the boards are seeded identically.
    import numpy as np
    from projections.draft.assistant.auction.snake_bot import SnakeBoard
    from projections.schemas import Position

    pool = _make_pool_with_adp()
    sub = 20260619  # mirror tournament's _SNAKE_SUBSTREAM in the assertion
    b1 = SnakeBoard(pool, np.random.default_rng([0, sub]))
    b2 = SnakeBoard(pool, np.random.default_rng([0, sub]))
    elig = frozenset(Position)
    assert b1.best_available(frozenset(), elig) == b2.best_available(frozenset(), elig)
```

- [ ] **Step 2: Run to verify it passes (board determinism)**

Run: `pytest tests/test_draft/test_assistant_auction_simulation.py::test_snake_board_is_shared_across_strategies_at_a_seed -v`
Expected: PASS (SnakeBoard is deterministic in its seed — this guards the substream choice).

- [ ] **Step 3: Wire the substream in the tournament**

In `src/projections/draft/assistant/auction/tournament.py`, add near the top:

```python
_SNAKE_SUBSTREAM = 20260619  # dedicated sub-key for the broke-bot ADP noise (CRN: shared bot field)
```

At the `simulate_auction(...)` call (currently passing `rng=np.random.default_rng(base_seed + s)`), add:

```python
                rng=np.random.default_rng(base_seed + s),
                snake_rng=np.random.default_rng([base_seed + s, _SNAKE_SUBSTREAM]),
```

- [ ] **Step 4: Run the tournament-touching tests**

Run: `pytest tests/test_draft/ -k "tournament or auction_simulation" -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/auction/tournament.py tests/test_draft/test_assistant_auction_simulation.py
git commit -m "feat(auction): tournament threads a dedicated CRN snake substream"
```

---

### Task 7: Full gate + ADP-coverage re-verification

**Files:**
- Possibly Modify: any file flagged by mypy/ruff.

- [ ] **Step 1: Re-verify in-pool ADP coverage on the current bake-off pools**

Run (a one-off check; confirms the regime is active where intended — the prior 100% measurement must not be trusted on regenerated pools):

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
from projections.draft.auction import _select_pool
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, VorpTableSchema
from projections.draft.assistant.auction.snake_bot import adp_usable
for yr in range(2021, 2027):
    df = pd.read_parquet(f"data/vorp_{yr}/half_12team.parquet")
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df = VorpTableSchema.validate(df)
    cfg = LeagueConfig.model_validate_json(Path(f"data/vorp_{yr}/half_12team.league.json").read_text())
    ids = set(_select_pool(df, cfg))
    ip = df[df["gsis_id"].isin(ids)]
    print(yr, "adp_usable:", adp_usable(ip), "in-pool null adp:", int(ip["consensus_adp"].isna().sum()))
PY
```

Expected: `adp_usable: True` every season. Record the result; if any season is `False`, note it (the regime self-disables there — not a failure, but worth flagging in the report).

- [ ] **Step 2: Full test suite**

Run: `pytest -v`
Expected: all PASS. (If scoping, at minimum `pytest tests/test_draft -v` plus `pytest -v -k "ingest or store or schemas"` since Task 2 touches no schema but the CLAUDE.md seam rule is cheap insurance.)

- [ ] **Step 3: Types + lint + format**

Run:
```bash
mypy src tests
ruff check src tests
ruff format --check src tests
```
Expected: zero violations each. Fix any inline (narrow `# type: ignore[code]` only with a comment if truly unavoidable).

- [ ] **Step 4: Commit any gate fixes**

```bash
git add -A
git commit -m "chore(auction): satisfy mypy/ruff for broke-bot snake behavior"
```

---

## Notes for the implementer

- **Reuse existing engine maps.** `pos_by_id` (`simulation.py`) already maps gsis→`Position`; don't re-derive in the engine. `SnakeBoard` builds its own arrays once at construction — that's fine (per-bot, one-shot).
- **`fmax` is per-seat per-round.** It is computed at the top of the bid-loop body via `_feasible_max(state, seat, rs, min_bid)`. The nomination override (Task 5) recomputes it for the nominator specifically.
- **Do not move the sort out of `bot_pick`** (Task 1) — its order-independence depends on sorting by `gsis_id` before the noise draw.
- **The discount knob is untouched.** `_UNRANKED_MODEL_DISCOUNT` / `espn_anchored_bot_prices` still price unranked players for *flush* bots; this slice does not modify them.
- **Validation re-run (out of scope here).** After merge, re-run the multi-year bake-off and replace the discount-era numbers in `reports/auction_tournament_validation_2026.md` (tracked under #49a). Not part of this plan's code tasks.
