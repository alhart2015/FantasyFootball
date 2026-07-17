# Stack-Ratio Convex-Aggression Hero Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-07-16-auction-stack-ratio-bid-design.md`

**Goal:** Add a `StackRatioBid` hero whose aggression multiplier is a convex function of the raw budget ratio to the field (`mult = 1 + gain·max(0, ratio−1)^curve`), applied to both the value target and the pace cap, reducing exactly to `balanced` when not ahead — then sweep the `(gain, curve)` grid and read out roster shape to see whether convex curves defer spend to depth.

**Architecture:** A new frozen dataclass `StackRatioBid` in `bid_strategy.py` with a `_multiplier(view, config)` helper (mirroring `BigStackBid._advantage`) that computes `ratio = my_budget / mean(opponent budgets)` and the convex multiplier. Its `max_bid` scales both `target = fair·mult` and `cap = pace·per_slot·mult`; the engine already clamps to `feasible_max`. Two scratch runners: a win-rate seat sweep (reusing `run_auction_tournament` + `auction_seat_sweep.aggregate_seat_sweep`) and a `PickRecord`-trace roster-shape analysis.

**Tech Stack:** Python 3.12, numpy/pandas, pytest, mypy strict, ruff. Auction code under `src/projections/draft/assistant/auction/`.

## Global Constraints

- `mult = 1.0 + gain · max(0.0, ratio − 1.0) ** curve`; `ratio = my_budget / max(opp_mean, min_bid)`; `opp_mean = (sum(budgets_by_seat) − my_budget) / max(1, n_teams − 1)`.
- `target = fair · mult`, `cap = pace · per_slot · mult`, `bid = round(min(target, cap))`. Both target and cap lifted by `mult`.
- At `ratio ≤ 1` (or `gain = 0`) → `mult = 1` → **byte-identical** to `BalancedValueBid(premium=0.0, pace=pace)` (default `non_increasing_cap=False`) — spec R1.
- `__post_init__` rejects a non-finite/negative `gain`, a non-finite/non-positive `curve` (so `curve = 0` is rejected), and a non-finite/non-positive `pace` — spec R7.
- Strategies return a *desired* int bid; the engine clamps to `[min_bid, feasible_max]` — never re-implement the reserve. Solvency is the engine's invariant (spec R4).
- The A/B changes ONLY the bid; nomination (`market_adp_jitter=12`), bot field, markets, seeds (20), sims (300) match Run P/Q (spec R5).
- Gates: `pytest`, `mypy src tests` (strict), `ruff check`, `ruff format --check` clean (spec R6).
- Data-gathering: no adopt/reject bar; deliverable is the characterized `(gain, curve)` surface + roster-shape evidence, delta-vs-`balanced` noise-flagged (spec "Interpretation").
- Sweep contestants: `balanced` (control) + `StackRatioBid(gain=g, curve=c)` for `g ∈ {0.5, 1.0, 2.0}` × `c ∈ {1, 2, 3}` (10 total). NOT registered in `_MODELS`.

---

### Task 1: `StackRatioBid` strategy + unit tests

**Files:**
- Modify: `src/projections/draft/assistant/auction/bid_strategy.py` (add `StackRatioBid` after `BigStackBid`, which ends at line 359)
- Test: `tests/test_draft/test_assistant_auction_bid_strategy.py`

**Interfaces:**
- Consumes: `AuctionView` (fields `my_budget`, `my_open_slots`, `budgets_by_seat`, `baseline_dollars`), `BalancedValueBid`, `math` — all already imported in `bid_strategy.py`. `LeagueConfig` (fields `min_bid`, `n_teams`).
- Produces: `StackRatioBid(gain: float = 1.0, curve: float = 2.0, pace: float = 2.0)` with `_multiplier(view, config) -> float` and `max_bid(view, player, pool, config) -> int`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_draft/test_assistant_auction_bid_strategy.py`)

```python
def test_stackratio_falls_back_to_balanced_when_not_ahead() -> None:
    # ratio <= 1 (hero the short stack: 50 vs opp mean 100) -> mult 1 -> identical to balanced, for
    # every curve; and gain=0 -> mult 1 even when AHEAD; and a non-default pace.
    pool = _pool()
    baseline = _baseline([True, True, False, False], [60, 40, 0, 0])
    behind = AuctionView(
        my_budget=50, my_open_slots=3, my_positions=Counter(), my_roster=pool.iloc[:0],
        drafted=frozenset(), budgets_by_seat=(50, 100), baseline_dollars=baseline,
    )
    for curve in (1.0, 2.0, 3.0):
        sr = StackRatioBid(gain=1.0, curve=curve).max_bid(behind, pool.iloc[0], pool, _config())
        assert sr == BalancedValueBid(premium=0.0).max_bid(behind, pool.iloc[0], pool, _config())
    ahead = AuctionView(
        my_budget=100, my_open_slots=3, my_positions=Counter(), my_roster=pool.iloc[:0],
        drafted=frozenset(), budgets_by_seat=(100, 20), baseline_dollars=baseline,
    )
    sr0 = StackRatioBid(gain=0.0, curve=2.0).max_bid(ahead, pool.iloc[0], pool, _config())
    assert sr0 == BalancedValueBid(premium=0.0).max_bid(ahead, pool.iloc[0], pool, _config())
    srp = StackRatioBid(gain=1.0, curve=2.0, pace=1.5).max_bid(behind, pool.iloc[0], pool, _config())
    assert srp == BalancedValueBid(premium=0.0, pace=1.5).max_bid(behind, pool.iloc[0], pool, _config())


def test_stackratio_multiplier_is_convex_and_monotonic() -> None:
    # curve>1 damps a MODERATE lead (ratio 1.14) but not a DOMINANT one (ratio 2.0), and the mult is
    # monotonic increasing in ratio. Tests _multiplier directly (max_bid clamps can mask it).
    pool = _pool()
    baseline = _baseline([True, True, False, False], [60, 40, 0, 0])
    cfg = _config()  # n_teams=2
    moderate = AuctionView(
        my_budget=80, my_open_slots=3, my_positions=Counter(), my_roster=pool.iloc[:0],
        drafted=frozenset(), budgets_by_seat=(80, 70), baseline_dollars=baseline,
    )  # opp_mean 70 -> ratio 80/70 = 1.143
    dominant = AuctionView(
        my_budget=100, my_open_slots=3, my_positions=Counter(), my_roster=pool.iloc[:0],
        drafted=frozenset(), budgets_by_seat=(100, 50), baseline_dollars=baseline,
    )  # opp_mean 50 -> ratio 2.0
    m_lin = StackRatioBid(gain=1.0, curve=1.0)._multiplier(moderate, cfg)
    m_cvx = StackRatioBid(gain=1.0, curve=2.0)._multiplier(moderate, cfg)
    assert m_cvx < m_lin  # convex: 1 + (0.143)^2 < 1 + 0.143
    d_lin = StackRatioBid(gain=1.0, curve=1.0)._multiplier(dominant, cfg)
    d_cvx = StackRatioBid(gain=1.0, curve=3.0)._multiplier(dominant, cfg)
    assert abs(d_lin - 2.0) < 1e-9 and abs(d_cvx - 2.0) < 1e-9  # at ratio 2, (1)^curve = 1 -> 1+gain
    same = StackRatioBid(gain=1.0, curve=2.0)
    assert same._multiplier(dominant, cfg) > same._multiplier(moderate, cfg)  # monotonic in ratio


def test_stackratio_uses_mean_opponent_budget() -> None:
    # ratio keys on the MEAN opponent budget, not max or min or per-slot. 4 teams: hero 200,
    # opponents 40/100/160 -> mean 100 -> ratio 2.0 (max would give 1.25, min would give 5.0).
    cfg = LeagueConfig(
        name="t", n_teams=4, budget=200, min_bid=1,
        roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )
    pool = _pool()
    baseline = _baseline([True, True, False, False], [60, 40, 0, 0])
    view = AuctionView(
        my_budget=200, my_open_slots=3, my_positions=Counter(), my_roster=pool.iloc[:0],
        drafted=frozenset(), budgets_by_seat=(200, 40, 100, 160), baseline_dollars=baseline,
    )
    mult = StackRatioBid(gain=1.0, curve=2.0)._multiplier(view, cfg)
    assert abs(mult - 2.0) < 1e-9  # opp_mean (40+100+160)/3 = 100 -> ratio 2.0 -> 1 + 1*(1)^2 = 2.0


def test_stackratio_rejects_bad_params() -> None:
    with pytest.raises(ValueError, match="gain"):
        StackRatioBid(gain=-1.0)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="gain"):
            StackRatioBid(gain=bad)
    with pytest.raises(ValueError, match="curve"):
        StackRatioBid(curve=0.0)  # curve must be > 0 (curve=0 -> 0**0=1 -> step, not a ramp)
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError, match="curve"):
            StackRatioBid(curve=bad)
    with pytest.raises(ValueError, match="pace"):
        StackRatioBid(pace=0.0)
```

Add `StackRatioBid` to the test file's import block `from projections.draft.assistant.auction.bid_strategy import (...)`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_draft/test_assistant_auction_bid_strategy.py -q -k stackratio -o addopts=""`
Expected: FAIL — `ImportError` / `cannot import name 'StackRatioBid'`.

- [ ] **Step 3: Add `StackRatioBid`**

In `src/projections/draft/assistant/auction/bid_strategy.py`, add this class immediately after `BigStackBid` (which ends at line 359). No new imports are needed (`math`, `dataclass`, `pd`, `AuctionView`, `LeagueConfig` are all present).

```python
@dataclass(frozen=True)
class StackRatioBid:
    """Balanced-breadth hero whose aggression scales CONVEXLY with its budget ratio to the field.

    `mult = 1 + gain * max(0, ratio - 1) ** curve`, where `ratio = my_budget / mean(opponent
    budgets)`, lifts BOTH the value target and the low pace cap. `curve > 1` keeps the hero
    disciplined at a MODERATE lead (mult ~ 1) and only unleashes at a DOMINANT ratio — which arises
    late, when only depth remains — so surplus deploys into depth via the draft's timing, not a
    value-tier gate. At `ratio <= 1` (or `gain = 0`) it is exactly `BalancedValueBid(premium=0.0,
    pace)`. `curve = 1` recovers a linear (BigStackBid field_avg-style) ramp.

    See docs/superpowers/specs/2026-07-16-auction-stack-ratio-bid-design.md.
    """

    gain: float = 1.0
    curve: float = 2.0
    pace: float = 2.0

    def __post_init__(self) -> None:
        if not (self.gain >= 0.0 and math.isfinite(self.gain)):
            raise ValueError(f"gain must be finite and >= 0; got {self.gain}")
        if not (self.curve > 0.0 and math.isfinite(self.curve)):
            raise ValueError(f"curve must be finite and > 0; got {self.curve}")
        if not (self.pace > 0.0 and math.isfinite(self.pace)):
            raise ValueError(f"pace must be finite and > 0; got {self.pace}")

    def _multiplier(self, view: AuctionView, config: LeagueConfig) -> float:
        """Convex aggression multiplier (>= 1.0; exactly 1.0 when the hero is not ahead)."""
        opp_mean = (sum(view.budgets_by_seat) - view.my_budget) / max(1, config.n_teams - 1)
        ratio = view.my_budget / max(opp_mean, config.min_bid)
        return 1.0 + self.gain * max(0.0, ratio - 1.0) ** self.curve

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        fair = float(view.baseline_dollars.loc[player["gsis_id"], "auction_dollars"])
        per_slot = view.my_budget / max(1, view.my_open_slots)
        mult = self._multiplier(view, config)
        target = fair * mult
        cap = self.pace * per_slot * mult
        return round(min(target, cap))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_draft/test_assistant_auction_bid_strategy.py -q -k stackratio -o addopts=""`
Expected: PASS (4 passed).

- [ ] **Step 5: Gates**

Run: `python -m mypy src tests && python -m ruff check src tests && python -m ruff format --check src tests`
Expected: `Success`, `All checks passed!`, formatted. (If `ruff format` reflows the new tests, run `python -m ruff format src tests` first.)

- [ ] **Step 6: Commit**

```bash
git add src/projections/draft/assistant/auction/bid_strategy.py tests/test_draft/test_assistant_auction_bid_strategy.py
git commit -m "feat(auction): StackRatioBid convex-aggression hero (budget-ratio multiplier)"
```

---

### Task 2: Engine smoke — legal roster (R4)

**Files:**
- Test: `tests/test_draft/test_assistant_auction_simulation.py` (append)

**Interfaces:**
- Consumes: `StackRatioBid` (Task 1), `simulate_auction`, existing fixtures `_config(n_teams=4)`, `_pool(40)`, `_baseline(pool, config)`, `np`.

- [ ] **Step 1: Write the test** (append)

```python
def test_stackratio_produces_a_legal_full_roster() -> None:
    # The engine clamps StackRatioBid's (possibly huge) desired bids to feasible_max, so the hero
    # always fills a legal, full, dup-free roster. Spec R4. Both a linear (curve=1) and a convex
    # (curve=3) config run through a real auction.
    from projections.draft.assistant.auction.bid_strategy import StackRatioBid

    cfg = _config(n_teams=4)
    pool = _pool(40)
    for gain, curve in ((2.0, 1.0), (1.0, 3.0)):
        league = simulate_auction(
            StackRatioBid(gain=gain, curve=curve),
            1,
            pool,
            cfg,
            baseline_dollars=_baseline(pool, cfg),
            price_jitter=0.1,
            rng=np.random.default_rng(0),
        )
        assert all(len(r) == cfg.roster_size for r in league.values())
        ids = [g for r in league.values() for g in r]
        assert len(ids) == len(set(ids))
```

- [ ] **Step 2: Run to verify it passes**

Run: `python -m pytest tests/test_draft/test_assistant_auction_simulation.py -q -k stackratio -o addopts=""`
Expected: PASS. (Smoke — the engine clamp invariant already holds; if it fails, investigate `simulation._simulate_to_state` clamping before proceeding.)

- [ ] **Step 3: Gates + commit**

Run: `python -m pytest tests/test_draft/test_assistant_auction_simulation.py -q && python -m mypy src tests && python -m ruff check src tests`
Expected: pass / clean.

```bash
git add tests/test_draft/test_assistant_auction_simulation.py
git commit -m "test(auction): StackRatioBid yields a legal full roster (engine clamps)"
```

---

### Task 3: The win-rate `(gain, curve)` sweep

Race `balanced` + the 9-variant grid under ADP nomination, seat-averaged both markets, and record reg-win/playoff/champ deltas vs `balanced`.

**Files:**
- Create (scratch): `<scratchpad>/stackratio_sweep.py`
- Create (scratch): `<scratchpad>/stackratio_driver.sh`

**Interfaces:**
- Consumes: `run_auction_tournament` (threads `market_adp_jitter`), `auction_seat_sweep.aggregate_seat_sweep` + `_load_chunks`, `BalancedValueBid`, `StackRatioBid`, `_load_tournament_inputs`, `_REALISTIC_FIELD`, `DEFAULT_PRICE_JITTER`, `has_usable_espn_prices`.

- [ ] **Step 1: Write the scratch runner**

Write `<scratchpad>/stackratio_sweep.py`. Mirrors `scripts/auction_seat_sweep.py`'s chunk format so `aggregate_seat_sweep` consumes it; also seat-averages `make_playoffs_pct`/`champ_pct` (which `aggregate_seat_sweep`, reg_win_pct-only, does not surface).

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from projections.draft.assistant.auction.bid_strategy import BalancedValueBid, StackRatioBid
from projections.draft.assistant.auction.market import DEFAULT_PRICE_JITTER
from projections.draft.assistant.auction.tournament import run_auction_tournament
from projections.draft.assistant.auction.tournament_cli import (
    _REALISTIC_FIELD,
    _load_tournament_inputs,
)
from projections.draft.auction import has_usable_espn_prices

# scripts/ is NOT on sys.path at runtime (pyproject mypy_path / the pytest conftest cover only those
# tools). This runner is always invoked from the repo root, so add scripts/ before importing.
sys.path.insert(0, "scripts")

from auction_seat_sweep import (  # noqa: E402  # type: ignore[import-not-found]
    _load_chunks,
    aggregate_seat_sweep,
)

POOL = Path("data/vorp_2026/half_12team.parquet")
CFG = Path("data/vorp_2026/half_12team.league.json")
SEASON, MARKET_ADP_JITTER = 2026, 12.0

CONTESTANTS = {"balanced": BalancedValueBid(premium=0.0)}
for _g in (0.5, 1.0, 2.0):
    for _c in (1.0, 2.0, 3.0):
        CONTESTANTS[f"sr_g{_g}_c{int(_c)}"] = StackRatioBid(gain=_g, curve=_c)


def _run_chunk(a: argparse.Namespace) -> int:
    pool, config, availability, params = _load_tournament_inputs(
        POOL, CFG, season=SEASON, data_root=Path("data")
    )
    if a.market == "espn" and not has_usable_espn_prices(pool):
        raise SystemExit("espn requested but pool has no usable espn prices")
    result = run_auction_tournament(
        CONTESTANTS, pool, config, my_seat=a.seat, n_seeds=a.seeds,
        price_jitter=DEFAULT_PRICE_JITTER, base_seed=0, n_sims=a.n_sims,
        availability=availability, params=params, nomination_temp=1.0,
        bot_archetypes=_REALISTIC_FIELD, bot_prices=a.market, market_adp_jitter=MARKET_ADP_JITTER,
    )
    payload = {
        "market": a.market, "seat": a.seat, "n_seeds": a.seeds, "n_sims": a.n_sims,
        "reg_win_pct": {n: result.summaries[n]["reg_win_pct"].point for n in result.summaries},
        "all_metrics": {
            n: {m: result.summaries[n][m].point for m in result.summaries[n]}
            for n in result.summaries
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {a.out} (market={a.market}, seat={a.seat})")
    return 0


def _seat_avg_metric(
    chunks: list[dict[str, object]], metric: str
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Seat-average a metric under each chunk's 'all_metrics' (playoff/champ, which
    aggregate_seat_sweep does not surface). All contestants race in every chunk."""
    cell: dict[tuple[str, str, int], float] = {}
    for c in chunks:
        m, seat = str(c["market"]), int(c["seat"])  # type: ignore[call-overload]
        am = c.get("all_metrics")
        if not isinstance(am, dict):
            continue
        for name, metrics in am.items():
            if isinstance(metrics, dict) and metric in metrics:
                cell[(m, str(name), seat)] = float(metrics[metric])
    markets = sorted({m for m, _n, _s in cell})
    names = sorted({n for _m, n, _s in cell})
    table: dict[str, dict[str, float]] = {}
    for name in names:
        table[name] = {}
        for mk in markets:
            vals = [v for (m2, n2, _s), v in cell.items() if m2 == mk and n2 == name]
            if vals:
                table[name][mk] = sum(vals) / len(vals)
    return markets, table


def _print_metric(chunks: list[dict[str, object]], metric: str) -> None:
    markets, table = _seat_avg_metric(chunks, metric)
    if not table:
        return
    base = table.get("balanced", {})
    print(f"\n{metric} (seat-avg) + delta vs balanced:")
    for name in sorted(table):
        cells = "".join(f"{table[name].get(m, float('nan')):>9.3f}" for m in markets)
        dcells = (
            ""
            if name == "balanced"
            else "  d:" + "".join(f"{table[name].get(m, 0.0) - base.get(m, 0.0):>+8.3f}" for m in markets)
        )
        print(f"  {name:<16}{cells}{dcells}")


def _aggregate(a: argparse.Namespace) -> int:
    chunks, skipped = _load_chunks(a.chunk_dir)
    markets, seats, rows, best = aggregate_seat_sweep(chunks)
    print(f"seats: {seats} | skipped: {skipped}")
    print("reg_win_pct (seat-avg):")
    print(f"  {'contestant':<16}" + "".join(f"{m:>9}" for m in markets) + f"{'worst':>9}")
    ctrl = next((r for r in rows if r.name == "balanced"), None)
    for row in rows:
        cells = "".join(f"{c:>9.3f}" if c is not None else f"{'-':>9}" for c in row.seat_avg)
        d = ""
        if ctrl is not None and row.name != "balanced":
            d = "  d:" + "".join(
                f"{(a_ - b_):>+8.3f}" if (a_ is not None and b_ is not None) else f"{'-':>8}"
                for a_, b_ in zip(row.seat_avg, ctrl.seat_avg)
            )
        print(f"  {row.name:<16}{cells}{row.worst:>9.3f}{d}")
    _print_metric(chunks, "make_playoffs_pct")
    _print_metric(chunks, "champ_pct")
    return 0


def _args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seat", type=int, required=True)
    r.add_argument("--market", choices=("espn", "model"), required=True)
    r.add_argument("--seeds", type=int, default=20)
    r.add_argument("--n-sims", type=int, default=300)
    r.add_argument("--out", type=Path, required=True)
    r.set_defaults(func=_run_chunk)
    g = sub.add_parser("aggregate")
    g.add_argument("--chunk-dir", type=Path, required=True)
    g.set_defaults(func=_aggregate)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke one tiny chunk**

Run: `python <scratchpad>/stackratio_sweep.py run --seat 1 --market model --seeds 2 --n-sims 20 --out reports/_stackratio/smoke.json` then `python -c "import json;d=json.load(open('reports/_stackratio/smoke.json'));print(sorted(d['reg_win_pct']))"` and `rm reports/_stackratio/smoke.json`
Expected: 10 keys (`balanced` + 9 `sr_g*_c*`); no traceback.

- [ ] **Step 3: Crash-safe driver + launch**

Write `<scratchpad>/stackratio_driver.sh` (resumable: skips valid-JSON chunks; one bounded `python` per `(seat, market)` for the dev-box Raptor Lake fault — memory `h2h-backtest-native-crash`). Substitute `<scratchpad>` with the real session scratch path.

```bash
#!/usr/bin/env bash
# Stack-ratio (gain,curve) sweep: 10 contestants, 12 seats x 2 markets, market_adp_jitter=12.
# Crash-safe, resumable (skips valid-JSON chunks). Run from repo root.
set -u
cd /c/Users/HartAlden/FantasyFootball
RUNNER="<scratchpad>/stackratio_sweep.py"
OUT=reports/_stackratio/2026
LOG=$OUT/driver.log
mkdir -p "$OUT"
echo "=== driver start $(date) ===" >> "$LOG"
for market in model espn; do
  for seat in $(seq 1 12); do
    f="$OUT/${market}_seat${seat}.json"
    if [ -f "$f" ] && python -c "import json,sys;json.load(open(sys.argv[1]))" "$f" 2>/dev/null; then
      echo "skip $f" >> "$LOG"; continue
    fi
    s=$(date +%s)
    python "$RUNNER" run --seat "$seat" --market "$market" --seeds 20 --n-sims 300 --out "$f" \
      >> "$LOG" 2>&1
    rc=$?; e=$(date +%s)
    echo "chunk market=$market seat=$seat rc=$rc $((e-s))s $(date)" >> "$LOG"
    if [ $rc -ne 0 ]; then echo "CHUNK FAILED market=$market seat=$seat rc=$rc" >> "$LOG"; fi
  done
done
echo "=== driver done $(date) ===" >> "$LOG"
```

Launch with the Bash tool `run_in_background: true`: `bash <scratchpad>/stackratio_driver.sh`. Expected ~45–90 min (10 contestants × 24 chunks); every chunk logs `rc=0`. Monitor the log until `=== driver done ===`; re-launch to resume if the box faults (valid chunks are skipped).

- [ ] **Step 4: Aggregate**

Run: `python <scratchpad>/stackratio_sweep.py aggregate --chunk-dir reports/_stackratio/2026`
- **Sanity:** `balanced` reproduces its Run-Q ADP figure (~espn 0.684 / model 0.593). If not, stop.
- Read the reg_win/playoff/champ seat-avg tables + delta-vs-`balanced`. Identify the **best convex (`curve≥2`) variant** by worst-case reg_win — record its `(gain, curve)` for Task 4.

---

### Task 4: Roster-shape trace (R8) + Run writeup

Show WHETHER convex curves defer spend to depth (the user's actual question), then write it up.

**Files:**
- Create (scratch): `<scratchpad>/stackratio_shape.py`
- Modify: `reports/auction_tournament_validation_2026.md` (add a Run)
- Modify: memory `auction-bid-model-investigation-status.md` + `MEMORY.md`

**Interfaces:**
- Consumes: `_simulate_to_state` (accepts `trace: list[PickRecord]`), `PickRecord` (fields `pick`, `value`, `price`, `winner_seat`), `_SNAKE_SUBSTREAM`, `espn_anchored_bot_prices`, `generate_auction_values`, `BalancedValueBid`, `StackRatioBid`, `_load_tournament_inputs`, `_REALISTIC_FIELD`, `DEFAULT_PRICE_JITTER`.

- [ ] **Step 1: Write the roster-shape runner**

Write `<scratchpad>/stackratio_shape.py`. Traces `balanced` + `StackRatioBid(curve=1)` (the linear baseline, the best gain from Task 3) + the best convex variant from Task 3, at a representative seat across seeds, both markets. Reports spend-share by draft quartile + top-5 concentration. Replace `BEST_CONVEX` with Task 3's `(gain, curve)`.

```python
from __future__ import annotations

from pathlib import Path

import numpy as np

from projections.draft.assistant.auction.bid_strategy import BalancedValueBid, StackRatioBid
from projections.draft.assistant.auction.market import DEFAULT_PRICE_JITTER
from projections.draft.assistant.auction.simulation import PickRecord, _simulate_to_state
from projections.draft.assistant.auction.tournament import _SNAKE_SUBSTREAM
from projections.draft.assistant.auction.tournament_cli import (
    _REALISTIC_FIELD,
    _load_tournament_inputs,
)
from projections.draft.auction import espn_anchored_bot_prices, generate_auction_values

POOL = Path("data/vorp_2026/half_12team.parquet")
CFG = Path("data/vorp_2026/half_12team.league.json")
SEAT, SEEDS, JITTER = 1, range(10), 12.0
BEST_CONVEX = StackRatioBid(gain=1.0, curve=2.0)  # <-- REPLACE with Task 3's best convex variant
HEROES = {
    "balanced": BalancedValueBid(premium=0.0),
    "sr_linear_c1": StackRatioBid(gain=1.0, curve=1.0),
    "sr_convex": BEST_CONVEX,
}


def hero_picks(strat, seat, seed, pool, config, baseline, bot):
    trace: list[PickRecord] = []
    _simulate_to_state(
        strat, seat, pool, config, baseline_dollars=baseline, price_jitter=DEFAULT_PRICE_JITTER,
        rng=np.random.default_rng(seed), snake_rng=np.random.default_rng([seed, _SNAKE_SUBSTREAM]),
        nomination_temp=1.0, bot_archetypes=_REALISTIC_FIELD, bot_dollars=bot,
        market_adp_jitter=JITTER, trace=trace,
    )
    return [pr for pr in trace if pr.winner_seat == seat - 1]


def main() -> int:
    pool, config, _a, _p = _load_tournament_inputs(POOL, CFG, season=2026, data_root=Path("data"))
    baseline = generate_auction_values(pool, config)
    markets = {"espn": espn_anchored_bot_prices(pool, config, model_values=baseline), "model": None}
    total = config.n_teams * config.roster_size
    for market, bot in markets.items():
        print(f"\n=== {market} — seat {SEAT}, seeds {list(SEEDS)} — spend share by draft quartile ===")
        print(f"{'hero':<14}{'Q1':>7}{'Q2':>7}{'Q3':>7}{'Q4':>7}{'top5%':>8}{'leftovr':>8}")
        for hname, strat in HEROES.items():
            q = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
            top5, spent_all, n = 0.0, 0.0, 0
            for seed in SEEDS:
                picks = hero_picks(strat, SEAT, seed, pool, config, baseline, bot)
                spent = sum(pr.price for pr in picks)
                for pr in picks:
                    q[min(4, 1 + (pr.pick - 1) * 4 // total)] += pr.price
                top5 += sum(sorted((pr.price for pr in picks), reverse=True)[:5])
                spent_all += spent
                n += 1
            tot = spent_all or 1.0
            print(f"{hname:<14}" + "".join(f"{q[k] / tot:>7.2f}" for k in (1, 2, 3, 4))
                  + f"{top5 / tot:>8.2f}{(200 * n - spent_all) / n:>8.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it**

Run: `python <scratchpad>/stackratio_shape.py`
Expected: for each market, a spend-share-by-quartile + top-5-concentration table for the 3 heroes. Read: does the convex variant shift spend LATER (higher Q3/Q4 share, lower top5%) than the linear `curve=1` — i.e. defer to depth — while `balanced` is the disciplined reference?

- [ ] **Step 3: Write the Run + memory**

Add a Run section to `reports/auction_tournament_validation_2026.md`: the win-rate seat-avg table + delta-vs-`balanced` (both markets, flag ±0.03 noise band; **no adopt bar**), the `(gain, curve)` surface read (does convexity beat both `balanced` AND the linear `curve=1` baseline, esp. in the less-circular ESPN market), and the roster-shape evidence (does convex defer spend to depth). Update the memory files. State plainly: no strategy adopted. Note the deferred alternatives (per-slot ratio, pure power-law).

- [ ] **Step 4: Commit the writeup**

```bash
git add reports/auction_tournament_validation_2026.md
git commit -m "docs(auction): Run — stack-ratio convex-aggression sweep (data, no adoption)"
```

---

## Self-Review

**Spec coverage:**
- `StackRatioBid` mult/target/cap formula → Task 1 Step 3.
- R1 fallback identity (ratio≤1, gain=0, non-default pace) → `test_stackratio_falls_back_to_balanced_when_not_ahead`.
- R2 convexity & monotonicity → `test_stackratio_multiplier_is_convex_and_monotonic`.
- R3 mean-opponent ratio → `test_stackratio_uses_mean_opponent_budget`.
- R4 solvency → Task 2 legal-roster smoke.
- R5 bid-fixed A/B → Task 3 `run_auction_tournament(..., market_adp_jitter=12)`, same seeds/sims/field.
- R6 gates → every code task.
- R7 param validation (incl. curve=0) → `test_stackratio_rejects_bad_params`.
- R8 roster-shape (separate trace, bounded 3-hero set, spend-by-quartile + top-5) → Task 4.
- Interpretation (delta vs balanced, noise-flag, beat linear baseline, no adopt bar) → Tasks 3–4.
- 10-contestant dedicated dict, gain×curve grid → Task 3 `CONTESTANTS`.
- Deferred alternatives noted → Task 4 Step 3.

**Placeholder scan:** none — every code step has complete code; `<scratchpad>` is the session scratch dir; `BEST_CONVEX` in Task 4 is an explicit "replace with Task 3's result" (its value is data-dependent, filled after the sweep).

**Type consistency:** `StackRatioBid(gain, curve, pace)`, `_multiplier(view, config) -> float`, and the `sr_g{g}_c{int(c)}` contestant keys are used identically across Tasks 1, 3, 4. `mult = 1 + gain·max(0, ratio−1)^curve` matches the spec and every test's expected value.
