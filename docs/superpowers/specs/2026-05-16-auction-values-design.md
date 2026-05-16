# Auction Values — $ Generator — Design

**Status:** draft (brainstorming, 2026-05-16). Ready for user review.
**Date:** 2026-05-16
**Author:** alden + claude
**Sub-project of:** FantasyFootball / Draft Hub (new sub-project seed)
**Branch:** `feat/auction-values` cut from `main` at `4317c66`.

**Depends on:** a separate VORP spec (not yet written) that ships a `vorp_table` parquet with columns `gsis_id`, `position`, `season_mean_fpts`, `vorp`. This spec assumes that contract and errors clearly if the VORP input is missing or malformed.

**Related specs / docs:**
- `draft_ready_checklist.md` §2c.1 — names this work as the "dollar value generator" item.
- `CLAUDE.md` — names "Draft Hub" as a planned sub-project; this spec is the first module in `src/projections/draft/`.

---

## 1. Overview

The Draft Hub needs a way to turn per-player VORP into per-player auction dollars under a configurable league setup. This spec builds the conversion only — no live bid recommender, no nomination strategy, no strategy-aware $ curves. One $ per player, deterministic from `(vorp_table, league_config)`.

The algorithm is the standard "Surplus Of Surplus" (SOS) allocation used by every published expert auction value sheet: reserve `min_bid` per drafted slot, distribute the remaining budget proportionally to positive VORP among the rostered pool.

Three sibling tools (live bid recommender, nomination helper, strategy-aware variants) consume this spec's output but are scoped to follow-up specs.

### 1.1 Goals (in scope)

- **New subpackage `src/projections/draft/`** as the home for all draft-related tooling. First two modules:
  - `src/projections/draft/league_config.py` — `LeagueConfig` pydantic model. Used by this spec and the (not-yet-written) VORP spec.
  - `src/projections/draft/auction.py` — `generate_auction_values(vorp_table, league_config, reference_prices=None)` public function.
- **New schema `AuctionValuesSchema` in `src/projections/schemas.py`** validating the output DataFrame.
- **New CLI script `scripts/generate_auction_values.py`** wrapping the function with I/O. Reads `LeagueConfig` from a JSON file, reads VORP parquet, optionally reads reference-prices CSV, writes output CSV or parquet.
- **Two example `LeagueConfig` JSONs in a new `configs/` directory** at the repo root:
  - `configs/league_espn_ppr_12team.json`
  - `configs/league_espn_half_10team.json`
- **Tests in `tests/test_draft/test_auction.py` and `tests/test_draft/test_league_config.py`** covering algorithmic invariants, schema validation, and CLI end-to-end. See §5.
- **No new ingest, no new store partition, no new model code.** Pure transform.

### 1.2 Non-goals (deferred)

- **No VORP construction.** This spec consumes a VORP parquet. The VORP spec is a sibling. If VORP isn't on disk, the script errors with a message naming the missing file.
- **No strategy-aware $ values.** "Stars-and-scrubs" / "balanced" / aggressiveness knobs live in the future live-bid recommender, not here. The $ generator produces one $ per player.
- **No market-clearing-price calibration.** Algorithm B from the brainstorm (per-position multipliers fit to historical auction data) is deferred to a separate spec that depends on auction-history ingest work that doesn't exist.
- **No ADP-rank anchor.** Algorithm C is deferred to a separate spec that depends on ADP ingest (also not yet built; named in `draft_ready_checklist.md` §2b.3).
- **No live bid recommender.** §2c.2 of the checklist. Separate spec.
- **No nomination strategy helper.** §2c.3. Separate spec.
- **No upside-sensitive VORP** (using `p90` instead of `mean`). Lives in the VORP spec, not here. Auction-$ consumes whatever scalar VORP value the VORP spec produces.
- **No multi-league output in one run.** One `LeagueConfig` per script invocation.
- **No persistence to the store layer.** Output is a CSV/parquet file the user keeps locally; not a partitioned dataset under `store.write_partition`.

---

## 2. Architecture

```
src/projections/draft/                       (new subpackage)
├── __init__.py
├── league_config.py                         (new — LeagueConfig pydantic model)
└── auction.py                               (new — generate_auction_values)

src/projections/schemas.py                   (edited — add AuctionValuesSchema)

scripts/generate_auction_values.py           (new — CLI)

configs/                                     (new directory at repo root)
├── league_espn_ppr_12team.json              (new — example config)
└── league_espn_half_10team.json             (new — example config)

tests/test_draft/                            (new directory)
├── __init__.py
├── test_league_config.py
└── test_auction.py
```

### 2.1 `LeagueConfig`

Pydantic model, frozen for hashability. Lives in its own module because both this spec and the VORP spec consume it.

```python
from pydantic import BaseModel, ConfigDict, Field
from projections.schemas import RosterSlot, Ruleset

class LeagueConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    n_teams: int = Field(gt=1)
    budget: int = Field(gt=0, default=200)
    min_bid: int = Field(ge=1, default=1)
    roster_slots: dict[RosterSlot, int]       # e.g. {QB: 1, RB: 2, WR: 3, TE: 1, FLEX: 1, K: 1, DST: 1, BENCH: 7}
    ruleset: Ruleset

    @property
    def roster_size(self) -> int:
        return sum(count for slot, count in self.roster_slots.items() if slot != RosterSlot.IR)

    @property
    def total_pool_size(self) -> int:
        return self.n_teams * self.roster_size

    @property
    def total_budget(self) -> int:
        return self.n_teams * self.budget
```

`IR` slots are excluded from `roster_size` because IR is post-draft. `BENCH` slots are included.

**JSON serialization** uses pydantic's default `model_dump_json()` / `model_validate_json()`. `Ruleset` deserializes from either a string preset name (`"espn_ppr"`, `"espn_half"`, `"standard"`) or a full object for custom leagues. The string-preset path requires a small validator on `LeagueConfig` that maps strings to `Ruleset.espn_ppr()` / `.espn_half()` / `.standard()`.

### 2.2 `generate_auction_values`

Pure function. No I/O, no side effects, no caching.

```python
def generate_auction_values(
    vorp_table: pd.DataFrame,                 # validated against VorpTableSchema (sibling spec)
    league_config: LeagueConfig,
    reference_prices: pd.DataFrame | None = None,  # cols: gsis_id, reference_dollars
) -> pd.DataFrame:                            # validated against AuctionValuesSchema
    ...
```

Algorithm in §3. Input contract:

- `vorp_table` must have columns `gsis_id`, `position`, `season_mean_fpts`, `vorp`; no duplicate `gsis_id`; positions must cover every `Position` value used in `league_config.roster_slots` (e.g. if config requires K, vorp_table must have K rows).
- `reference_prices`, if provided, must have columns `gsis_id` (string) and `reference_dollars` (Int64). No duplicate `gsis_id`. Players in vorp_table but not in reference_prices get `pd.NA` for the reference column.

The function returns a new DataFrame validated against `AuctionValuesSchema`. One row per player in `vorp_table`, regardless of pool membership.

### 2.3 `AuctionValuesSchema`

Lives in `src/projections/schemas.py` per the "single source of truth" rule.

```python
class AuctionValuesSchema(pa.DataFrameModel):
    gsis_id: Series[pd.StringDtype] = pa.Field(unique=True)
    position: Series[pd.StringDtype] = pa.Field(isin=[p.value for p in Position])
    season_mean_fpts: Series[float]
    vorp: Series[float]
    in_pool: Series[bool]
    auction_dollars: Series[pd.Int64Dtype] = pa.Field(ge=0)
    pool_rank: Series[pd.Int64Dtype] = pa.Field(nullable=True, ge=1)
    reference_dollars: Series[pd.Int64Dtype] = pa.Field(nullable=True, ge=0)
    value_delta: Series[pd.Int64Dtype] = pa.Field(nullable=True)

    class Config:
        strict = "filter"
```

`reference_dollars` and `value_delta` columns are present in the output regardless of whether `reference_prices` was passed; if not passed, both columns are all-`pd.NA`. (Pinning this rather than having two schema flavors keeps consumers simple.)

---

## 3. Algorithm

### Step 1 — Build the rostered pool

The pool is the set of `total_pool_size = n_teams × roster_size` players who would actually be drafted. Selected by position-then-FLEX-then-SUPER_FLEX-then-BENCH:

1. For each position `pos` in `[QB, RB, WR, TE, K, DST]` where `roster_slots.get(pos, 0) > 0`, take the top `n_teams × roster_slots[pos]` players by `season_mean_fpts`. These are guaranteed in-pool. (Positions not present in `roster_slots` are skipped — supports configs that omit K or DST, e.g. best-ball leagues.)
2. Fill `n_teams × roster_slots[FLEX]` slots with the top remaining RB/WR/TE pool by `season_mean_fpts`.
3. Fill `n_teams × roster_slots[SUPER_FLEX]` slots with the top remaining QB/RB/WR/TE pool by `season_mean_fpts`.
4. Fill `n_teams × roster_slots[BENCH]` slots with the top remaining players (any position covered by VORP) by `season_mean_fpts`. (Bench is position-agnostic by default; if a league restricts bench positions, that's a config follow-up — out of scope here.)

Ties broken by `vorp` desc, then `gsis_id` (stable, deterministic).

**Why projection-rank not VORP-rank for pool selection:** VORP comparisons across positions are correctly normalized for fantasy value, but actual drafts assign players to specific roster slots which have positional structure. The 18th QB has higher VORP than the 60th RB; in a league with 1 QB starter and 2 RB starters, the 60th RB is still going on a bench (because every team needs an RB4 backup) and the 18th QB sits on waivers. Pool selection must respect slot structure.

Players outside the pool get `in_pool = False`, `auction_dollars = 0`, `pool_rank = pd.NA`.

### Step 2 — Compute the surplus

```
total_budget = n_teams × budget                              # e.g. 12 × 200 = 2400
reserve      = total_pool_size × min_bid                     # e.g. 12 × 16 × 1 = 192
surplus      = total_budget − reserve                        # e.g. 2208
```

`surplus` is the discretionary dollar pool above the `min_bid` floor.

### Step 3 — Allocate surplus to positive VORP

```
positive_vorp_sum = Σ max(vorp_i, 0) for i in pool

for i in pool:
    extra_i_float    = (max(vorp_i, 0) / positive_vorp_sum) × surplus
    auction_dollars_i_float = min_bid + extra_i_float
```

In-pool players with `vorp ≤ 0` get exactly `min_bid` in floating-point terms (their `extra_i_float = 0`).

### Step 4 — Round to integer and close the drift

```
auction_dollars_i = round(auction_dollars_i_float)           # bankers' rounding via numpy
drift = total_budget − Σ auction_dollars over pool

if drift != 0:
    sort pool players by fractional_part_of(auction_dollars_i_float) descending if drift > 0,
                                                              ascending if drift < 0
    adjust the top |drift| of them by sign(drift)
```

This guarantees `Σ auction_dollars (over pool) == total_budget` exactly. Players outside the pool contribute 0 to the sum.

Out-of-pool rows are not adjusted. Their `auction_dollars` stays 0.

After adjustment, every in-pool player has `auction_dollars ≥ min_bid` (the +1/-1 nudges are small relative to the surplus allocation; for the standard 12-team × $200 setup, drift is typically 0-3 dollars, and adjustments land on players whose pre-round value was already well above the floor).

**Edge case:** if `positive_vorp_sum == 0` (every in-pool player has VORP ≤ 0, which can happen on degenerate test inputs and shouldn't happen in real data), distribute `surplus` uniformly across the pool. Documented + tested.

### Step 5 — Rank and attach reference

```
pool_rank: dense rank over in-pool rows, ordered by
    auction_dollars desc, vorp desc, season_mean_fpts desc, gsis_id asc
```

`pool_rank` is `pd.NA` for not-in-pool rows.

If `reference_prices` provided, left-join on `gsis_id`. `value_delta = auction_dollars − reference_dollars`. Unmatched rows get `pd.NA` in both columns. If `reference_prices` not provided, both columns are added as all-NA Int64.

### Worked example

12-team standard PPR, $200 budget, roster slots `{QB: 1, RB: 2, WR: 3, TE: 1, FLEX: 1, K: 1, DST: 1, BENCH: 7}`, `min_bid = $1`:

- `roster_size = 16`, `total_pool_size = 192`, `total_budget = $2400`, `reserve = $192`, `surplus = $2208`.
- Suppose `positive_vorp_sum = 4400` after pool selection.
- Christian McCaffrey (VORP 130) → `1 + (130/4400)×2208 ≈ $66`.
- Bench WR with VORP 5 → `1 + (5/4400)×2208 ≈ $3.5` → `$4` (rounding).
- WR ranked 35th with VORP -2 (in pool because his projection puts him in the WR top-`12×4`+FLEX pool) → `$1`.
- WR ranked 100th — not in pool → `$0`.
- After integer rounding, drift might be e.g. `-2`; we adjust the two players with the smallest fractional remainders downward by $1.

---

## 4. CLI surface

`scripts/generate_auction_values.py`:

```
python scripts/generate_auction_values.py \
    --season 2026 \
    --league-config configs/league_espn_ppr_12team.json \
    --vorp-input data/vorp/2026.parquet \
    [--reference-prices path/to/sheet.csv] \
    --out reports/auction_values_2026.csv
```

**Flags:**

| Flag | Required | Description |
|---|---|---|
| `--season` | yes | Integer; metadata only — used in log output and (if appropriate) embedded in the output filename. Doesn't affect computation. |
| `--league-config` | yes | Path to `LeagueConfig` JSON. |
| `--vorp-input` | yes | Path to VORP parquet (sibling spec's output). Errors clearly if missing. |
| `--reference-prices` | no | Path to CSV with `gsis_id,reference_dollars` columns. If given, output includes `reference_dollars` and `value_delta`. |
| `--out` | yes | Output destination. `.csv` and `.parquet` both supported (sniff by extension). |

**Script flow:**

1. Parse args.
2. Load `LeagueConfig` from JSON (via `LeagueConfig.model_validate_json`).
3. Read `vorp_input` parquet. Validate against the VORP spec's schema (assume sibling spec exposes `VorpTableSchema`).
4. Read `reference_prices` CSV if given.
5. Call `generate_auction_values(vorp_table, league_config, reference_prices)`.
6. Log per-position summary to stdout — see §6 risk mitigation.
7. Write output. CSV writes are sorted by `pool_rank` ascending (NaN last) for readability; parquet writes preserve the schema's column order without re-sorting.

**Example `LeagueConfig` JSON shape:**

```json
{
  "name": "espn_ppr_12team_2026",
  "n_teams": 12,
  "budget": 200,
  "min_bid": 1,
  "roster_slots": {
    "QB": 1,
    "RB": 2,
    "WR": 3,
    "TE": 1,
    "FLEX": 1,
    "K": 1,
    "DST": 1,
    "BENCH": 7
  },
  "ruleset": "espn_ppr"
}
```

`ruleset` accepts a string preset name (`"espn_ppr"`, `"espn_half"`, `"standard"`) or a full object for custom rules.

---

## 5. Testing

All tests under `tests/test_draft/`. Standard pandera-validated DataFrame contract plus algorithmic invariants.

### 5.1 `test_auction.py`

Synthetic VORP tables, hand-checked numbers.

1. **Sum invariant.** `Σ auction_dollars (over all rows) == n_teams × budget`. Hard equality after rounding-drift correction.
2. **Min-bid floor.** Every in-pool player has `auction_dollars ≥ min_bid`. Every in-pool player with `vorp ≤ 0` has `auction_dollars == min_bid` (modulo drift adjustments, which never push below the floor — pin this).
3. **Out-of-pool zero.** Every not-in-pool row has `auction_dollars == 0`.
4. **Pool size.** Exactly `n_teams × roster_size` rows have `in_pool == True`.
5. **Pool composition respects roster slots.** Synthetic input where Position.QB has only 20 viable players and the league wants top-18 QBs (12 × 1 starter + bench/FLEX surplus from QB pool): the top-18 QBs by `season_mean_fpts` are in-pool; QBs 19-20 are not.
6. **FLEX fills from RB/WR/TE remainder.** With a synthetic input where after filling RB/WR/TE position slots there's a strong RB ranked below the strict RB cutoff, that RB ends up in-pool via the FLEX slot.
7. **SUPER_FLEX fills from QB/RB/WR/TE remainder.** Synthetic config with `SUPER_FLEX: 1`; assertion that QB pool depth grows accordingly.
8. **VORP scale invariance.** Doubling all VORPs leaves `auction_dollars` unchanged (proportional allocation is scale-invariant).
9. **VORP shift sensitivity.** Adding a constant to every VORP changes `auction_dollars`. Pin the expected direction: shifting all VORPs upward by a positive constant raises the share of total surplus going to the previously-low-VORP players (because their share of `positive_vorp_sum` rises).
10. **Higher budget scales surplus.** Identical VORPs with `budget=400` → each in-pool player gets approximately `2×` the original (exactly `min_bid + 2×original_extra`).
11. **`pool_rank` is dense and correct.** `[1, total_pool_size]` exactly, ordered by `auction_dollars` desc with tie-breaks per §3.5.
12. **`reference_prices` pass-through.** When provided, `reference_dollars` and `value_delta` are correct on matched rows and `pd.NA` on unmatched. When not provided, both columns are present and all-NA.
13. **Duplicate `gsis_id` rejection.** Input with duplicate IDs raises a clear `ValueError` before computation starts (not a downstream pandera error).
14. **Pandera validation.** Output round-trips through `AuctionValuesSchema.validate(df)` without filter losses.
15. **Degenerate-input handling.** `positive_vorp_sum == 0` distributes surplus uniformly across the pool; documented invariant.
16. **Missing position in VORP input.** If `LeagueConfig` requires K but `vorp_table` has no K rows, function raises a clear error naming the position.

### 5.2 `test_league_config.py`

17. **Roster size, total_pool_size, total_budget** properties match hand-computed values; `IR` slots excluded from `roster_size`.
18. **JSON round-trip.** `LeagueConfig.model_validate_json(config.model_dump_json()) == config`.
19. **Pydantic validation rejects** `n_teams ≤ 1`, `budget ≤ 0`, empty `roster_slots`, `min_bid < 1`.
20. **Ruleset deserialization.** Strings `"espn_ppr"`, `"espn_half"`, `"standard"` map to the correct presets. A full object for custom rules also works.

### 5.3 CLI integration test

21. **End-to-end with a known fixture.** Synthetic VORP parquet + known `LeagueConfig` JSON → script produces output CSV with expected per-player $ values. Asserts the sum invariant on file contents and pins a small canonical output snippet for regression detection.

### 5.4 What's deliberately not tested

- Calibration against real auction markets. Out of scope — see §6.
- Stability across VORP spec revisions. The VORP spec owns its own contract; this spec consumes whatever shape it produces.
- Strategy variants. Out of scope.

---

## 6. Open items, risks, and explicit out-of-scope

### Open items with explicit defaults

**K/DST behavior.** VORP spec hasn't shipped. The auction spec assumes VORP returns K/DST rows with finite (possibly small) VORP values, treated like any other position. **Fallback:** if VORP returns no K/DST rows and `LeagueConfig.roster_slots` requires them, the script errors with a message naming the missing position(s) and pointing at the VORP spec. No silent zero-fill — that would understate budget consumption and mislead the user.

**`SUPER_FLEX` handling.** Included in the pool-fill algorithm and tested with a synthetic config, even though no shipped `LeagueConfig` JSON uses it. Future-proofs against superflex leagues.

**`IR` slot.** Excluded from `roster_size` and pool-fill. `LeagueConfig` accepts `IR` in `roster_slots` but the algorithm ignores it.

**Bench position restrictions.** Some leagues restrict bench to certain positions. Out of scope; current design treats bench as position-agnostic. Add as a config follow-up if needed.

**Ties at the pool-membership boundary.** Pool selection ranks by `season_mean_fpts` desc; tie-breaks are `vorp` desc then `gsis_id` asc per §3.1 (stable, deterministic). Tested.

### Risks

**Calibration risk against real auction markets.** Algorithm A (pure VORP-to-$) reflects model's view of value, not market clearing prices. `--reference-prices` CSV gives a manual sanity check but doesn't fix the gap. If draft-day experience shows the curve feels wrong, that's the trigger to spec algorithm B (per-position market scaling) or C (ADP-rank anchor) as follow-ups.

**VORP-spec coupling.** Auction-$ output is only as good as the upstream VORP. A broken VORP (wrong replacement level, bad projection input) produces broken $ values silently — the schema invariants pass either way because they don't check VORP correctness. **Mitigation:** the CLI emits a per-position summary to stdout — top-3 $ values, in-pool player count, min/median/max VORP within pool, replacement-level fpts (passed through from VORP if available, else "n/a"). User eyeballs the summary before trusting the output. Cheap, no extra schema.

### Explicit out-of-scope (so they don't get smuggled in)

- Strategy-aware $ values (stars-and-scrubs, balanced, aggressiveness knobs). → live-bid recommender spec.
- Market-calibration multipliers (algorithm B). → separate spec, blocked on historical $ ingest.
- ADP-rank anchor (algorithm C). → separate spec, blocked on ADP ingest.
- Live auction bid recommender (`draft_ready_checklist.md` §2c.2). → separate spec; consumes this spec's output.
- Nomination strategy helper (§2c.3). → separate spec.
- Upside-sensitive VORP. → lives in the VORP spec, not here.
- Multi-league output in one run. → run the script twice.
- Persistence to the store layer. → output stays as a local file.

---

## 7. Acceptance

This spec is complete when:

- `LeagueConfig` and `generate_auction_values` are implemented per §2-3.
- `AuctionValuesSchema` is in `schemas.py`.
- `scripts/generate_auction_values.py` runs end-to-end against a synthetic VORP parquet fixture and a sample `LeagueConfig` JSON.
- All §5 tests pass.
- `mypy src tests` and `ruff check src tests` clean.
- The two example `LeagueConfig` JSONs are present under `configs/`.

No adoption gate, no §1.3.5-style contingency matrix — this is feature work, not a model-change probe. The shipping decision is binary: does the feature work, does the output look sane on the eyeball summary, and do the tests pass.

---

## 8. Follow-ups (sequenced)

In rough priority order, each its own spec:

1. **VORP spec** — required upstream dependency. Builds the `vorp_table` parquet this spec consumes. Should ship before this one is usable; can be specced and reviewed in parallel.
2. **Live auction bid recommender** — primary downstream consumer; turns the static $ sheet into a real-time draft-day tool.
3. **Snake-draft cheat sheet** (`draft_ready_checklist.md` §2b.1) — also consumes VORP; can land in parallel with this spec since they're independent surfaces over the same VORP output.
4. **Algorithm B (market scaling) or C (ADP anchor)** — only if draft-day experience with algorithm A reveals a market-divergence problem worth fixing. Currently low priority.
5. **Nomination strategy helper** — secondary draft-day tool.
