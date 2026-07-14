# Auction Robust Win-Maximizing Bid Hero — Implementation Plan (Slice 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the diagnosed `BalancedValueBid` cap self-inflation with a non-increasing cap, ship it as a `balanced_flat` contestant, and measure which single config maximizes 12-team half-PPR regular-season win% robustly across both bot markets.

**Architecture:** A one-field change to an existing frozen-dataclass bid strategy (`non_increasing_cap`), a one-line contestant registration, and a dedicated crash-safe tuning script that races a `pace × premium` grid against both markets and picks the best worst-case `reg_win_pct`. No engine/market change (that is Slice 2).

**Tech Stack:** Python 3.12, pandas (pyarrow dtypes), pandera, numpy, pytest, mypy (strict), ruff. Source spec: `docs/superpowers/specs/2026-07-14-auction-robust-win-hero-design.md`.

## Global Constraints

- **No default flip.** `BalancedValueBid.non_increasing_cap` defaults to `False`; the bare `BalancedValueBid()` (`balanced` contestant) stays byte-identical to Runs I/J. The fixed candidate passes `non_increasing_cap=True` explicitly.
- **Enums, never strings.** `Position.RB`, `RosterSlot.BENCH`, etc. `GsisId` is canonical for any player id.
- **Gates are hard** (CLAUDE.md end-of-effort checklist): `pytest -v` (relevant subset ok, state which), `mypy src tests` (zero), `ruff check src tests` (zero), `ruff format --check src tests` (no drift). No test edited to pass code without stated reason + confirmation.
- **Crash safety.** All measurement runs go through the chunked runner — one `(market, seed-chunk)` per OS process (Raptor Lake fault, memory `h2h-backtest-native-crash`). Never one long `n_seeds` process.
- **Metric & selection.** Primary metric `reg_win_pct`; field 12-team half-PPR (`data/vorp_2026/half_12team.parquet` + `configs/half_12team.league.json`); both markets every run (`--bot-prices model` and `--bot-prices espn`); finalist = **best worst-case `reg_win_pct` across the two markets**.
- **Data-gathering only.** No strategy is committed as the September default; runs are recorded in `reports/auction_tournament_validation_2026.md`.

---

### Task 1: Non-increasing cap on `BalancedValueBid`

**Files:**
- Modify: `src/projections/draft/assistant/auction/bid_strategy.py:261-291` (the `BalancedValueBid` dataclass)
- Test: `tests/test_draft/test_assistant_auction_bid_strategy.py` (append after the existing `BalancedValueBid` block, ~line 540)

**Interfaces:**
- Consumes: `AuctionView` (`my_budget`, `my_open_slots`, `baseline_dollars`), `LeagueConfig` (`.budget`, `.roster_size` — a property, already used by `_budget_urgency`).
- Produces: `BalancedValueBid(premium: float = 1.0, pace: float = 2.0, non_increasing_cap: bool = False)`; `max_bid(view, player, pool, config) -> int` unchanged signature.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_draft/test_assistant_auction_bid_strategy.py`:

```python
def test_balanced_non_increasing_cap_defaults_off() -> None:
    # Global constraint: the bare constructor keeps the current inflating behavior (control unchanged).
    assert BalancedValueBid().non_increasing_cap is False


def test_balanced_non_increasing_cap_matches_on_opening_pick() -> None:
    # Opening state: budget/open_slots (100/3) == budget0/roster_size (100/3), so both modes agree.
    pool = _pool()
    baseline = _baseline([True, True, False, False], [80, 40, 0, 0])
    view = _view(pool.iloc[:0], budget=100, drafted=set(), baseline=baseline)  # open_slots=3
    infl = BalancedValueBid(premium=0.15, pace=2.0, non_increasing_cap=False)
    flat = BalancedValueBid(premium=0.15, pace=2.0, non_increasing_cap=True)
    assert flat.max_bid(view, pool.iloc[0], pool, _config()) == 67  # cap = 2*(100/3)=66.7; 80*1.15=92 capped
    assert infl.max_bid(view, pool.iloc[0], pool, _config()) == 67


def test_balanced_non_increasing_cap_blocks_inflation_after_cheap_win() -> None:
    # 1 player held, budget still high: budget/open_slots (90/2=45) exceeds the opening per-slot
    # share (100/3=33.3). The inflating cap balloons; the non-increasing cap holds at the opening pace.
    pool = _pool()
    baseline = _baseline([True, True, False, False], [80, 40, 0, 0])
    view = _view(pool.iloc[[2]], budget=90, drafted={"00-0000003"}, baseline=baseline)  # 1 held -> open=2
    stud = pool.iloc[0]  # $80 in-pool RB
    infl = BalancedValueBid(premium=0.15, pace=2.0, non_increasing_cap=False).max_bid(view, stud, pool, _config())
    flat = BalancedValueBid(premium=0.15, pace=2.0, non_increasing_cap=True).max_bid(view, stud, pool, _config())
    assert infl == 90  # cap = 2*(90/2)=90; min(92, 90)=90 (inflated)
    assert flat == 67  # cap = 2*min(45, 33.3)=66.7; min(92, 66.7)=67 (held to opening pace)
    assert flat < infl


def test_balanced_non_increasing_cap_still_retreats_when_broke() -> None:
    # budget/open_slots (30/3=10) is BELOW the opening share (33.3): the min picks the live ratio,
    # so the non-increasing cap equals the inflating one -> broke-time downside behavior preserved.
    pool = _pool()
    baseline = _baseline([True, True, False, False], [80, 40, 0, 0])
    view = _view(pool.iloc[:0], budget=30, drafted=set(), baseline=baseline)  # open=3
    stud = pool.iloc[0]
    infl = BalancedValueBid(premium=0.15, pace=2.0, non_increasing_cap=False).max_bid(view, stud, pool, _config())
    flat = BalancedValueBid(premium=0.15, pace=2.0, non_increasing_cap=True).max_bid(view, stud, pool, _config())
    assert infl == flat == 20  # cap = 2*(30/3)=20 in both
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_draft/test_assistant_auction_bid_strategy.py -k non_increasing -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'non_increasing_cap'`.

- [ ] **Step 3: Implement the non-increasing cap**

In `src/projections/draft/assistant/auction/bid_strategy.py`, edit the `BalancedValueBid` dataclass. Add the field (after `pace`) and rewrite `max_bid`; extend the docstring. The `__post_init__` validation is unchanged.

```python
    premium: float = 1.0
    pace: float = 2.0
    non_increasing_cap: bool = False

    def __post_init__(self) -> None:
        if not (self.premium >= 0.0 and math.isfinite(self.premium)):
            raise ValueError(f"premium must be finite and >= 0; got {self.premium}")
        if not (self.pace > 0.0 and math.isfinite(self.pace)):
            raise ValueError(f"pace must be finite and > 0; got {self.pace}")

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        fair = float(view.baseline_dollars.loc[player["gsis_id"], "auction_dollars"])
        per_slot = view.my_budget / max(1, view.my_open_slots)
        if self.non_increasing_cap:
            # Never let the cap rise above the OPENING per-slot pace. As a breadth hero wins players
            # cheaper than its per-slot share, budget/open_slots ratchets up and the inflating cap
            # balloons (overpays late, lopsided roster — the diagnosed bug). Clamping to the constant
            # opening share kills the ratchet while still retreating below it when the hero is broke.
            per_slot = min(per_slot, config.budget / config.roster_size)
        cap = self.pace * per_slot
        return round(min(fair * (1.0 + self.premium), cap))
```

Also add one line to the class docstring (before the closing `"""`): `non_increasing_cap=True clamps the pace cap to the opening per-slot share so it can't self-inflate as the hero wins (Slice 1 fix; default False keeps the inflating control).`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_draft/test_assistant_auction_bid_strategy.py -v`
Expected: PASS — the four new tests plus all pre-existing `BalancedValueBid` tests (the default-False path is byte-identical, so `test_balanced_*` are unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/auction/bid_strategy.py tests/test_draft/test_assistant_auction_bid_strategy.py
git commit -m "feat(auction): non-increasing cap on BalancedValueBid (fixes cap self-inflation)"
```

---

### Task 2: Register the `balanced_flat` contestant

**Files:**
- Modify: `src/projections/draft/assistant/auction/tournament_cli.py:48-62` (`_MODELS`), plus the "ten" count strings at `:6` (module docstring) and `:172` (the `compare` subparser help).
- Test: `tests/test_draft/test_assistant_auction_tournament_cli.py:51-63` (`test_default_models_are_the_ten_contestants`).

**Interfaces:**
- Consumes: `BalancedValueBid(non_increasing_cap=True)` from Task 1.
- Produces: `_MODELS["balanced_flat"]`; the registry now has **eleven** contestants.

- [ ] **Step 1: Update the failing test**

In `tests/test_draft/test_assistant_auction_tournament_cli.py`, rename and extend the count test:

```python
def test_default_models_are_the_eleven_contestants() -> None:
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
        "balanced_flat",
    }
    assert _MODELS["balanced_flat"].non_increasing_cap is True  # the Slice 1 cap fix
    assert _MODELS["balanced"].non_increasing_cap is False  # control unchanged
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_draft/test_assistant_auction_tournament_cli.py::test_default_models_are_the_eleven_contestants -v`
Expected: FAIL — `KeyError: 'balanced_flat'` (or set-inequality: `balanced_flat` missing).

- [ ] **Step 3: Register the contestant + fix count strings**

In `tournament_cli.py`, add to `_MODELS` (after the `"balanced"` line, ~:61):

```python
    "balanced": BalancedValueBid(),
    # balanced_flat: the Slice 1 cap-inflation fix — same premium/pace, but the pace cap can't
    # self-inflate as the hero wins (non_increasing_cap=True). See the 2026-07-14 robust-win-hero spec.
    "balanced_flat": BalancedValueBid(non_increasing_cap=True),
```

Then update the two "ten" strings (grep to confirm no others: `grep -n "ten " src/projections/draft/assistant/auction/tournament_cli.py`):
- module docstring `:6`: "races the ten bid models" → "races the eleven bid models".
- `:172` subparser help: `"Race the ten bid models; record per-metric data."` → `"Race the eleven bid models; record per-metric data."`

- [ ] **Step 4: Run the test + CLI tests to verify pass**

Run: `pytest tests/test_draft/test_assistant_auction_tournament_cli.py -v`
Expected: PASS (all, including the renamed count test).

- [ ] **Step 5: Commit**

```bash
git add src/projections/draft/assistant/auction/tournament_cli.py tests/test_draft/test_assistant_auction_tournament_cli.py
git commit -m "feat(auction): register balanced_flat contestant (non-increasing cap)"
```

---

### Task 3: Crash-safe both-market tuning script

**Files:**
- Create: `scripts/auction_cap_tuning.py`
- Test: `tests/test_scripts/test_auction_cap_tuning.py`

**Interfaces:**
- Consumes: `run_auction_tournament` (`tournament.py`), `_load_pool`/`_load_config`/`_REALISTIC_FIELD` (`tournament_cli.py`), `BalancedValueBid` (Task 1), `PatientValueBid`.
- Produces: pure helpers `grid() -> dict[str, AuctionBidStrategy]` and `aggregate_chunks(chunks: list[dict]) -> tuple[list[str], list[tuple[str, list[float], float]], str]` (markets, rows sorted by worst-case desc, best-worst-case name). CLI: `run` (one market+chunk → JSON) and `aggregate` (chunk dir → table).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scripts/test_auction_cap_tuning.py`:

```python
from scripts.auction_cap_tuning import aggregate_chunks, grid


def test_grid_has_flat_variants_plus_controls() -> None:
    g = grid()
    # 4 paces x 3 premiums flat variants + balanced control + patient_deep reference = 14
    assert len(g) == 14
    flat = [k for k in g if k.startswith("flat_")]
    assert len(flat) == 12
    assert all(g[k].non_increasing_cap is True for k in flat)
    assert g["balanced"].non_increasing_cap is False  # inflating control
    assert "patient_deep" in g


def test_aggregate_picks_best_worst_case_across_markets() -> None:
    # Two markets, two chunks each. 'robust' wins the worst-case; 'specialist' wins one market only.
    chunks = [
        {"market": "model", "reg_win_pct": {"robust": 0.50, "specialist": 0.60}},
        {"market": "model", "reg_win_pct": {"robust": 0.52, "specialist": 0.62}},
        {"market": "espn", "reg_win_pct": {"robust": 0.49, "specialist": 0.20}},
        {"market": "espn", "reg_win_pct": {"robust": 0.51, "specialist": 0.22}},
    ]
    markets, rows, best = aggregate_chunks(chunks)
    assert markets == ["espn", "model"]
    assert best == "robust"  # worst-case: robust ~0.50 vs specialist ~0.21
    # rows sorted by worst-case descending
    assert rows[0][0] == "robust"
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_scripts/test_auction_cap_tuning.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.auction_cap_tuning'`.

- [ ] **Step 3: Write the script**

Create `scripts/auction_cap_tuning.py`:

```python
"""Cap-fix tuning: race BalancedValueBid(non_increasing_cap=True) across a pace x premium grid
against both bot markets on the 12-team half-PPR preset, and pick the best worst-case reg_win_pct.

Crash-safe: one (market, seed-chunk) per process (the dev box's Raptor Lake fault wants bounded
processes — memory h2h-backtest-native-crash). `run` writes a per-chunk JSON; `aggregate` combines
the chunk JSONs and prints the reg_win_pct table + finalist. Data-gathering only; no default changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np

from projections.draft.assistant.auction.bid_strategy import (
    AuctionBidStrategy,
    BalancedValueBid,
    PatientValueBid,
)
from projections.draft.assistant.auction.tournament import run_auction_tournament
from projections.draft.assistant.auction.tournament_cli import (
    _REALISTIC_FIELD,
    _load_config,
    _load_pool,
)
from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.performance_variance import VarianceParams
from projections.draft.assistant.rookies import attach_is_rookie

PACES: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5)
PREMIUMS: tuple[float, ...] = (0.5, 1.0, 1.5)


def grid() -> dict[str, AuctionBidStrategy]:
    """The pace x premium flat-cap grid + the inflating control + the standing breadth leader."""
    models: dict[str, AuctionBidStrategy] = {}
    for pace in PACES:
        for prem in PREMIUMS:
            models[f"flat_p{pace}_prem{prem}"] = BalancedValueBid(
                premium=prem, pace=pace, non_increasing_cap=True
            )
    models["balanced"] = BalancedValueBid()  # inflating-cap control (default False)
    models["patient_deep"] = PatientValueBid(scrub_frac=0.0)  # standing multi-year leader reference
    return models


def aggregate_chunks(
    chunks: list[dict],
) -> tuple[list[str], list[tuple[str, list[float], float]], str]:
    """Combine per-chunk reg_win_pct into (markets, rows, best). Equal chunk sizes -> mean of chunk
    means == overall mean. Each row is (name, per-market means, worst-case); rows sorted worst desc."""
    by_market: dict[str, dict[str, list[float]]] = {}
    for c in chunks:
        m = str(c["market"])
        for name, val in c["reg_win_pct"].items():
            by_market.setdefault(m, {}).setdefault(str(name), []).append(float(val))
    markets = sorted(by_market)
    names = sorted({n for m in by_market for n in by_market[m]})
    rows: list[tuple[str, list[float], float]] = []
    for name in names:
        cells = [float(np.mean(by_market[m][name])) for m in markets if name in by_market[m]]
        rows.append((name, cells, min(cells)))
    rows.sort(key=lambda r: r[2], reverse=True)
    best = rows[0][0] if rows else ""
    return markets, rows, best


def _run_chunk(args: argparse.Namespace) -> int:
    pool = _load_pool(args.vorp_table)
    config = _load_config(args.league_config)
    pool = attach_is_rookie(pool, season=args.season, data_root=args.data_root)
    availability = load_store_availability(pool, season=args.season, data_root=args.data_root)
    params = VarianceParams.load()
    market: Literal["espn", "model"] = "espn" if args.bot_prices == "espn" else "model"
    result = run_auction_tournament(
        grid(),
        pool,
        config,
        my_seat=args.my_seat,
        n_seeds=args.seeds,
        price_jitter=0.15,
        base_seed=args.seed,
        n_sims=args.n_sims,
        availability=availability,
        params=params,
        nomination_temp=1.0,
        bot_archetypes=_REALISTIC_FIELD,
        bot_prices=market,
    )
    payload = {
        "market": market,
        "base_seed": args.seed,
        "n_seeds": args.seeds,
        "n_sims": args.n_sims,
        "my_seat": args.my_seat,
        "season": args.season,
        "reg_win_pct": {n: result.summaries[n]["reg_win_pct"].point for n in result.summaries},
        "all_metrics": {
            n: {m: result.summaries[n][m].point for m in result.summaries[n]}
            for n in result.summaries
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out} (market={market}, base_seed={args.seed}, {args.seeds} seeds)")
    return 0


def _aggregate(args: argparse.Namespace) -> int:
    chunks = [json.loads(p.read_text()) for p in sorted(args.chunk_dir.glob("*.json"))]
    if not chunks:
        raise SystemExit(f"no chunk JSONs in {args.chunk_dir}")
    markets, rows, best = aggregate_chunks(chunks)
    print(f"{'model':<22}" + "".join(f"{m:>12}" for m in markets) + f"{'worst':>12}")
    for name, cells, worst in rows:
        print(f"{name:<22}" + "".join(f"{c:>12.3f}" for c in cells) + f"{worst:>12.3f}")
    print(f"\nbest worst-case reg_win_pct across markets: {best} ({rows[0][2]:.3f})")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cap-fix both-market tuning (crash-safe chunks).")
    sub = p.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run", help="Race the grid for one market + one seed chunk -> JSON.")
    r.add_argument("--vorp-table", type=Path, required=True)
    r.add_argument("--league-config", type=Path, required=True)
    r.add_argument("--my-seat", type=int, required=True)
    r.add_argument("--season", type=int, required=True)
    r.add_argument("--seeds", type=int, default=20)
    r.add_argument("--n-sims", type=int, default=300)
    r.add_argument("--seed", type=int, default=0, help="Base seed for this chunk.")
    r.add_argument("--bot-prices", choices=("espn", "model"), required=True)
    r.add_argument("--data-root", type=Path, default=Path("data"))
    r.add_argument("--out", type=Path, required=True, help="Chunk JSON output path.")
    r.set_defaults(func=_run_chunk)
    a = sub.add_parser("aggregate", help="Combine chunk JSONs -> reg_win_pct table + finalist.")
    a.add_argument("--chunk-dir", type=Path, required=True)
    a.set_defaults(func=_aggregate)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_scripts/test_auction_cap_tuning.py -v`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add scripts/auction_cap_tuning.py tests/test_scripts/test_auction_cap_tuning.py
git commit -m "feat(auction): crash-safe both-market cap-tuning script + pure aggregation helpers"
```

---

### Task 4: Run the single-season 2026 grid, pick the finalist

**Files:**
- Create (artifacts, git-ignored/untracked): `reports/_cap_tuning/2026/*.json`
- Modify: `reports/auction_tournament_validation_2026.md` (append "Run K — cap-fix tuning" with the table)

**Interfaces:**
- Consumes: `scripts/auction_cap_tuning.py` (Task 3), `data/vorp_2026/half_12team.parquet`, `configs/half_12team.league.json`.
- Produces: the finalist `(pace, premium)` for `balanced_flat`, recorded for Task 5.

- [ ] **Step 1: Verify inputs exist**

Run: `ls data/vorp_2026/half_12team.parquet configs/half_12team.league.json`
Expected: both present. If the parquet is missing, regenerate: `python scripts/generate_preset_vorp_tables.py --season 2026` (see TODO #48), then re-check.

- [ ] **Step 2: Run both markets in crash-safe chunks (fresh process each)**

Run (Bash tool = Git Bash; three 20-seed chunks × two markets = 6 processes, seat 1):

```bash
mkdir -p reports/_cap_tuning/2026
for mkt in model espn; do
  for base in 0 20 40; do
    python scripts/auction_cap_tuning.py run \
      --vorp-table data/vorp_2026/half_12team.parquet \
      --league-config configs/half_12team.league.json \
      --my-seat 1 --season 2026 --seeds 20 --n-sims 300 --seed "$base" \
      --bot-prices "$mkt" --out "reports/_cap_tuning/2026/${mkt}_seed${base}.json"
  done
done
```

Expected: 6 JSON files written, each printing a `wrote ...` line. If any process dies (Raptor Lake fault), re-run just that one command — the chunk JSONs are independent.

- [ ] **Step 3: Aggregate + read the finalist**

Run: `python scripts/auction_cap_tuning.py aggregate --chunk-dir reports/_cap_tuning/2026`
Expected: a table of every contestant's `reg_win_pct` per market + worst-case column, and a printed `best worst-case ...` line. The finalist is the top `flat_p{pace}_prem{prem}` by worst-case. Sanity-check it beats `balanced` (control) on the worst-case column; if no flat variant beats the control, that is itself the result to report (the cap fix did not help) — record it and stop for user review before Task 5.

- [ ] **Step 4: Seat-6 spot-check of the finalist**

Re-run Step 2's two markets at `--my-seat 6` into `reports/_cap_tuning/2026_seat6/` (one 20-seed chunk each is enough for a directional check), aggregate, and confirm the finalist's ordering vs `balanced` holds at an interior seat (the diagnosis is partly a seat-role effect). Note any seat sensitivity.

- [ ] **Step 5: Record + commit the report entry**

Append a "Run K — cap-fix tuning (2026-07-14)" subsection to `reports/auction_tournament_validation_2026.md` with: the setup line (12-team half, seat 1 + seat-6 check, seeds/n_sims, both markets), the aggregated `reg_win_pct` table, the chosen finalist `(pace, premium)`, and the `balanced_flat` vs `balanced` worst-case delta. State "no winner declared — data-gathering; September decision."

```bash
git add reports/auction_tournament_validation_2026.md
git commit -m "docs(auction): Run K — single-season cap-fix tuning, pick balanced_flat finalist"
```

---

### Task 5: Multi-year validation of the finalist (both markets)

**Files:**
- Create (artifacts): `reports/_cap_tuning/multiyear/*.json`
- Modify: `reports/auction_tournament_validation_2026.md` (append the multi-year table), `project_management.md` + `TODO.md` (#49 status), memory `auction-bid-model-investigation-status`.

**Interfaces:**
- Consumes: the finalist `(pace, premium)` from Task 4; per-season pools `data/vorp_{2021..2026}/half_12team.parquet`.
- Produces: the multi-year mean `reg_win_pct` (both markets) for `balanced_flat(finalist)` vs `balanced` vs `patient_deep`; the Slice 1 verdict for the goal.

- [ ] **Step 1: Verify the per-season pools exist**

Run: `ls data/vorp_20{21,22,23,24,25,26}/half_12team.parquet`
Expected: six files. **If any are missing** (they are untracked artifacts), this needs the multi-year re-ingest (`python scripts/refresh_external_seasons.py` then `generate_preset_vorp_tables.py --season Y`) — a heavier step. If missing, PAUSE and report to the user that single-season (Task 4) stands and multi-year validation is gated on regenerating the tables; do not silently skip.

- [ ] **Step 2: Run the finalist vs controls across seasons, both markets, chunked**

For a leaner grid at multi-year scale, run only the three contestants that matter (edit `grid()` is NOT needed — pass the finalist by re-using the script with a narrowed grid via a small inline `--only` is out of scope; instead run the full `grid()` but read only the three rows). One 20-seed chunk per (season, market) process:

```bash
mkdir -p reports/_cap_tuning/multiyear
for season in 2021 2022 2023 2024 2025 2026; do
  for mkt in model espn; do
    python scripts/auction_cap_tuning.py run \
      --vorp-table "data/vorp_${season}/half_12team.parquet" \
      --league-config configs/half_12team.league.json \
      --my-seat 1 --season "$season" --seeds 20 --n-sims 300 --seed 0 \
      --bot-prices "$mkt" --out "reports/_cap_tuning/multiyear/${season}_${mkt}.json"
  done
done
python scripts/auction_cap_tuning.py aggregate --chunk-dir reports/_cap_tuning/multiyear
```

Expected: a `reg_win_pct` table averaged across all season×market chunks. (Note: `aggregate` averages every chunk equally, so this yields the multi-year both-market mean; per-season detail is in the individual JSONs if needed.)

- [ ] **Step 3: Read the verdict against the goal**

The Slice 1 result = the finalist's multi-year worst-case (across markets) `reg_win_pct` vs `balanced` (control) and vs `patient_deep`, plus the distance to the 0.50 fair-share line. Record whether the cap fix closes part/all/none of the hero-vs-best-bot gap. This is the number the user asked for and the input to the Slice 2 (poisoning) go/no-go.

- [ ] **Step 4: Record + commit**

Append the multi-year table + verdict to `reports/auction_tournament_validation_2026.md`; update `project_management.md` and `TODO.md` #49 with the Slice 1 result and the Slice 2 next step; update the memory file `auction-bid-model-investigation-status` with the finalist config and its multi-year both-market `reg_win_pct`.

```bash
git add reports/auction_tournament_validation_2026.md project_management.md TODO.md
git commit -m "docs(auction): Slice 1 result — multi-year both-market validation of balanced_flat finalist"
```

- [ ] **Step 5: Final gates**

Run the full end-of-effort checklist and paste output into the completion message:
```bash
pytest -v -k "auction or bid_strategy or tournament"
mypy src tests
ruff check src tests
ruff format --check src tests
```
Expected: all green. Then report the both-market win% numbers to the user (slice-at-a-time checkpoint) before designing Slice 2.

---

## Self-Review

**Spec coverage:**
- Non-increasing cap fix (spec §Approach 1) → Task 1. ✓
- `balanced_flat` new contestant + `balanced` control (spec §Approach 2) → Task 2. ✓
- `pace × premium` re-tune in both markets (spec §Approach 3) → Task 3 (grid) + Task 4 (run). ✓
- Measurement: reg_win_pct primary, both markets, single-season tune → multi-year validate, best worst-case selection (spec §Approach 4) → Tasks 4–5. ✓
- Crash safety / chunked runner (spec §Approach 5) → Task 3 design + Tasks 4–5 chunked invocation. ✓
- Testing: unit tests for cap behavior (never inflates, opening-pick equality, broke retreat, default-False control) + gates (spec §Testing) → Task 1 Steps 1–4, Task 5 Step 5. ✓
- Seat spot-check (spec §Open Q1) → Task 4 Step 4. ✓
- Multi-year pool availability caveat (spec §Open Q2 / §Approach 4) → Task 5 Step 1 pause-and-report. ✓
- No default flip; data-gathering only (spec §Non-goals, Global Constraints) → honored (default False in Task 1; report-only in Tasks 4–5). ✓
- Slice 2 not built here (spec §Slice 2) → out of plan scope by design. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to Task N". Every code step shows full code; every run step shows the exact command + expected output. ✓

**Type consistency:** `non_increasing_cap: bool` used identically in Tasks 1–3; `grid()` and `aggregate_chunks()` signatures in Task 3 match their Task-3 tests and the Task 4/5 invocations; `run_auction_tournament(...)` call matches `tournament.py` (bot_prices `Literal["espn","model"]`, `summaries[name][metric].point`). ✓
