# Waiver-Wire / Undrafted-Pool Assessment — Design

**Date:** 2026-07-17
**Status:** Design (pre-plan)
**Branch:** `feat/waiver-pool-assessment`
**Issue:** [#112](https://github.com/alhart2015/FantasyFootball/issues/112) (migrated from TODO #44)

## 1. Motivation

After a 16-team draft, roughly `n_teams × roster_size = 208` of the 578 consensus-pool players are gone, leaving ~370 on the waiver wire. Two open questions ride on **what's left, by position**:

1. **Draft strategy** — don't over-invest in a position whose waiver stays deep; prioritize the one that dries up. This connects directly to the scarcity thread (Tests 7–19, `reports/draft_strategy_tests.md`): *does the field drain TE to nothing (making elite-TE scarcity real), or does the position stay streamable?*
2. **Seeds the future mid-season waiver/streaming tool** — the same "how good is the best free agent at each position" computation is the core of a waiver valuator.

This is **pure analysis on existing sim machinery** (`draft_mixed_field`, the consensus VORP pool). Per the standing process rule (`reports/draft_strategy_tests.md`), it **gathers data** — no adopt/reject strategy verdict.

## 2. Goal & non-goals

**Goal:** characterize the undrafted pool (pool − union of all rosters) by position after a hero-plus-bots 16-team draft, averaged over many seeds, and produce a per-position readout of *how good the best players you can still grab are* and *how deep the wire is*, plus a depleted-vs-streamable ranking.

**Non-goals (deliberate deferrals):**
- **No per-season historical cut.** Only the 2026 consensus pools exist (`data/vorp_2026/`); historical per-season pools were cleaned up and their regeneration is a separate concern. Runs on 2026; other formats work by pointing the flags elsewhere.
- **No MC hero strategies in v1.** The hero defaults to `now_or_never_floored` (analytic, no availability load). Season-value-family heroes need `availability` wiring (`load_inputs`); deferred. The hero's ~13 picks only lightly perturb the wire, which is dominated by the 15 bots — so the hero choice is second-order.
- **No live waiver tool.** This is the offline draft-time readout that *seeds* that tool, not the tool.
- **No new draft/sim machinery.** Reuses `draft_mixed_field`, `hero_seat_layout`, the VORP pool, and `bootstrap_mean`.

## 3. The core function — `undrafted_pool_by_position` (pure, tested)

New module `src/projections/draft/backtest/waiver_pool.py`:

```
undrafted_pool_by_position(
    rosters: Mapping[int, list[str]],   # {seat: [gsis_id, ...]} from draft_mixed_field
    pool: pd.DataFrame,                 # VorpTableSchema-valid
    config: LeagueConfig,
) -> pd.DataFrame                        # WaiverPoolSchema-valid, one row per position
```

Computes, for **one** draft, per position `p ∈ {QB, RB, WR, TE}`:

- **`top1_vorp`, `top2_vorp`, `top3_vorp`** — the three highest `vorp` values among undrafted players at `p` (the best-available curve; shows how fast quality falls off the wire). Fewer than 3 undrafted at a position → the missing ranks are `NaN` (never happens for the real pool, but handled).
- **`best_avail_proj_pts`** — `season_mean_fpts` of the `top1_vorp` player (the raw-points anchor for the single best available; within a position, VORP-rank == proj-rank since `replacement_fpts` is a per-position constant).
- **`n_above_replacement`** — count of undrafted players at `p` with `vorp > 0` (streamable-quality remaining; `vorp > 0` ⇔ above the pool's own per-position `replacement_fpts`).
- **`drain_rate`** — `drafted_above_replacement / total_above_replacement` for `p` (fraction of the position's startable-quality players that got taken). Normalizes for position depth so QB (few startable) and WR (many) are comparable; high = drained hard = thin wire. Equals `1 − n_above_replacement / total_above_replacement`. **When `total_above_replacement == 0`** (no player at `p` is above replacement in the whole pool — a 0/0), `drain_rate` is `NaN` and `n_above_replacement` is `0`; the test asserts `NaN`.

**Purity:** the drafted set is `union of all rosters.values()`; everything else is a group-by on `pool`. No sim, no RNG, no I/O → trivially unit-testable on a hand-built fixture.

**Output — `WaiverPoolSchema`** (validated per the module-boundary convention): one row per position, indexed by `position` (`pd.StringDtype("pyarrow")`, values in `{QB, RB, WR, TE}`); columns `top1_vorp`, `top2_vorp`, `top3_vorp`, `best_avail_proj_pts` (all `float64`, may be `NaN`), `n_above_replacement` (`int64`), `drain_rate` (`float64`, in `[0, 1]` or `NaN`). Always all four position rows, even when a position is fully drafted (its top-N and `best_avail_proj_pts` are then `NaN`).

## 4. The driver — `scripts/waiver_pool_assessment.py` (thin)

Loads a VORP table + league config, builds a **hero + 15 constrained-ADP bots** seat map via the existing `hero_seat_layout`, runs `draft_mixed_field` over `N` seeds (each a fresh `default_rng(base_seed + seed)`, bot ADP jitter → a different draft), calls `undrafted_pool_by_position` per seed, and **aggregates each (position, metric) into mean + 95% bootstrap CI** with `bootstrap_mean` (house style). Prints a per-position table and, with `--out`, writes the report.

Flags (all defaulted): `--vorp-table` (default `data/vorp_2026/half_16team.parquet`), `--league-config` (default the sibling `.league.json`), `--hero-strategy` (default `now_or_never_floored`; the analytic keys `now_or_never`/`now_or_never_floored`/`raw_vorp` are supported without availability), `--hero-seat` (default 1), `--seeds` (default 200), `--jitter` (default 8.0), `--base-seed` (default 0), `--out` (optional report path).

The hero strategy is constructed from its key by a small local `_build_hero(key, n_teams)` (analytic strategies only — `NowOrNeverFlooredStrategy(LogisticSurvival(default_sigma(n_teams)))` etc.); an MC key raises a clear "needs availability, not supported in v1" error rather than silently mis-building. The floored hero uses the **shipped default floor** (`_DEFAULT_FLOOR=40` / `_DEFAULT_FLOOR_WEIGHT=1`); no floor flags in v1.

## 5. Output & report

**Console table** (per position, sorted by `top1_vorp` descending — deepest wire first, most depleted last), each metric as `mean [lo95, hi95]`:

```
POSITION   TOP1 VORP        TOP2        TOP3     BEST PROJ PTS   #>REPL   DRAIN%
WR         ...              ...         ...      ...             ...      ...
RB         ...
TE         ...
QB         ...
```

**`reports/waiver_pool_2026.md`** — the readout: the ranking, which positions dry up vs stay streamable, and the tie-back to the scarcity thread (does TE drain to nothing?). Records the finding *in isolation* (data-gathering; no strategy verdict), with the reproduce command and caveats (single format/pool, bots = noisy-ADP proxy, hero second-order).

## 6. Testing (TDD)

- **`undrafted_pool_by_position` (unit, hand-built fixture):** a tiny pool (a few players per position with known `vorp`/`season_mean_fpts`/`replacement_fpts`) + a `rosters` dict drafting a known subset; assert `top1/2/3_vorp`, `best_avail_proj_pts`, `n_above_replacement`, `drain_rate` exactly. Edge cases: fewer than 3 undrafted at a position (`NaN` pad), a fully-drafted-above-replacement position (`drain_rate == 1`, `n_above_replacement == 0`), a position with no above-replacement players in the pool.
- **Schema:** output validates against `WaiverPoolSchema` (positions present, dtypes, `drain_rate ∈ [0, 1]`).
- **Driver smoke (`tests/test_scripts/`):** run the driver on a small synthetic pool + config at low `--seeds` (e.g. 2) and assert it produces a 4-row aggregate without error (mirrors the existing script-smoke pattern; no real-data dependency).
- Gates: `pytest`, `mypy src tests`, `ruff check/format` all clean.

## 7. Open questions / future refinements

None blocking. Deferred: per-season historical cut (needs the historical pools regenerated); MC hero strategies (needs `availability` wiring); a stricter "solid starter" depth tier above `vorp > 0`; folding the core function into the live mid-season waiver/streaming tool (the reason it lives in `src/` and is tested, not a throwaway script).
