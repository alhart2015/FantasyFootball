# Multi-source (ESPN + Sleeper) projection blend with stub guard — Design

**Date:** 2026-06-14
**Status:** Approved (brainstorming) → spec review
**Topic:** Make Sleeper a real second projection source in the draft-basis consensus, guarded against degenerate partial stat lines.

## Problem

The H2H draft-strategy backtest values players from a **preseason season projection** built by `build_draft_basis` → `build_consensus`, which scores a player's **stat line** under the league ruleset. Today the only source that contributes a stat line is **ESPN**; Sleeper contributes **ADP only** (its stat columns arrive all-NA in the snapshot).

This single-source dependence has a silent, season-specific failure mode. ESPN's retention of historical *full-season* projections is inconsistent: for completed seasons ESPN sometimes returns only a **1-field stub** for the season-projection entry instead of the full ~40-field stat line. Empirically (verified against the live ESPN `kona_player_info` API and the stored snapshots, 2026-06-13/14):

| Season | pool `season_mean_fpts > 0` | ESPN season projection |
|---|---|---|
| 2024 | 474/475 | full |
| 2023 | **99/514** | **degenerate (1-field stubs; 167/200 have 1 field, only 18 carry any scoring stat)** |
| 2022 | 488/491 | full |
| 2021 | 481/484 | full |

When the pool is degenerate (2023), strategies that rank by `season_mean_fpts`/`vorp` draft blind (most players are valued at 0), producing broken, unfillable rosters that score ~⅓ of a healthy team — while ADP bots (which draft off Sleeper ADP) are unaffected. A direct H2H smoke on 2023 confirmed this: bot teams ~1155 PTS-FOR, strategy teams ~340.

This blocks the in-progress 2021–2025 projected-vs-actual correlation / post-draft-assessment analysis (the immediate motivation) and is a latent reliability hole even for future single-season runs.

## Goals

1. **Make Sleeper a real projection source.** Parse Sleeper's raw stat line (it has one — verified: `rec`, `rec_yd`, `rec_td`, `rush_yd`, `pass_yd`, `pass_td`, `pass_int`, `fum_lost`, …) into the canonical `STAT_FIELDS`, so it flows through the existing ruleset-aware scoring layer exactly like ESPN.
2. **Blend ESPN + Sleeper into a two-source consensus** where both carry a stat line (a more robust projection than either alone), and fall back to whichever single source is present otherwise.
3. **Guard the blend against degenerate partial stat lines** so ESPN's 2023 stubs (or any future source's partial data) cannot contaminate a player's blended projection.
4. **Restore healthy pools for all of 2021–2025** so the correlation analysis can run on a uniform, multi-source basis.
5. Advance TODO #38's long-deferred "real multi-source points consensus" slice, and incidentally resolve the all-NA `pd.concat` FutureWarning (also a TODO #38 item).

## Non-goals (explicitly out of scope)

- **ADP sourcing.** `consensus_adp` in the draft basis stays **Sleeper-only** (`build_draft_basis.sleeper_adp`); ESPN ADP remains an unused sentinel for past seasons. Unchanged.
- **Points-level consensus** (scoring each source separately then averaging the points). We keep the existing **stat-line-mean-then-score-once** semantics of `build_consensus`.
- **Distribution-wrapping / cross-source spread** (using the ESPN↔Sleeper disagreement as a floor/ceiling). A later TODO #38 slice.
- **A third source** (FantasyPros/CBS/NumberFire scraping).
- **K / DST** (TODO #10).
- **The backtest re-run + correlation/post-draft report.** This is the **post-merge payoff** of the new basis, not part of this spec. (It is throwaway analysis that runs existing code; see "Phasing".)

## Chosen approach (Approach A)

Two surgical changes; **no schema change, no `build_draft_basis` change, no draft/backtest code change.**

### Change 1 — `src/projections/ingest/external_projections.py`: populate Sleeper's stat line

- Add a module-level mapping `SLEEPER_STAT_FIELDS: dict[str, Stat]` (raw Sleeper key → canonical `Stat`), verified live against the Sleeper projections API:

  | Sleeper raw key | canonical `STAT_FIELDS` member |
  |---|---|
  | `pass_yd` | `passing_yards` |
  | `pass_td` | `passing_tds` |
  | `pass_int` | `interceptions` |
  | `rush_yd` | `rushing_yards` |
  | `rush_td` | `rushing_tds` |
  | `rec` | `receptions` |
  | `rec_yd` | `receiving_yards` |
  | `rec_td` | `receiving_tds` |
  | `fum_lost` | `fumbles_lost` |

  All nine `STAT_FIELDS` are covered. Reference enums, never raw strings (CLAUDE.md): the mapping values are `Stat`/`STAT_FIELDS` members.
- Add `_sleeper_stats_to_statline(stats: dict[str, float]) -> dict[str, float]` mirroring the existing `_espn_stats_to_statline`: it maps present Sleeper keys to canonical fields and defaults absent fields to 0.0. It stores the **raw fractional values, with no rounding** — exactly as `_espn_stats_to_statline` does (that function deliberately keeps ESPN's raw fractional projections; per its own comment, "Rounding here is irreversible and biases season totals; the scoring layer is the only place that decides how a projected stat becomes points"). `round_count`/`COUNT_FIELDS` are NOT used here; they are retained only for the frozen benchmark spike. Storing Sleeper raw (no rounding) is what makes the two sources' stat lines directly comparable before averaging.
- In the Sleeper parse path, populate the `STAT_FIELDS` columns from `_sleeper_stats_to_statline(...)` and flip the Sleeper source spec from `has_stats=False` to `has_stats=True`. Sleeper rows in the snapshot stop being all-NA on stat columns.
- **Side effect (intended):** the all-NA-column `pd.concat` FutureWarning at the source frame assembly disappears because Sleeper now carries real stat columns. (If any source still legitimately lacks stats, set the stat columns' dtype explicitly before `concat` so no all-NA inference warning remains.)

### Change 2 — `src/projections/consensus/blend.py` (`build_consensus`): stat-bearing gate

- Add a predicate `_is_stat_bearing(row, *, min_fields: int = 2) -> bool`: a single source row qualifies iff it has **≥ `min_fields` non-null values among the scoring `STAT_FIELDS`**.
- Change the per-player stat aggregation so the per-field mean ranges over **stat-bearing source rows only**, not all rows in the `gsis_id` group. `has_points` becomes "≥1 stat-bearing source row exists." The identity-row preference for a stat-bearing row is retained.
- The ADP aggregation is **unchanged** (still mean over all rows with `adp > 0`); ADP and points gating are independent.

### Why `min_fields = 2`

A degenerate ESPN 2023 stub maps to 0–1 scoring fields. A real skill-position projection always has several: QB ≥4 (`passing_yards`, `passing_tds`, `interceptions`, `rushing_yards`), RB ≥4, WR/TE ≥3 (`receptions`, `receiving_yards`, `receiving_tds`). `min_fields = 2` cleanly separates stubs from real lines with margin against false-exclusion of a thin-but-real projection. The threshold is a named constant so it is easy to audit/tune.

### Net behavioral result

- **2023:** ESPN stubs are not stat-bearing → excluded → the blend is **Sleeper-only** for that season (clean, no half-weighted stub field). Pool becomes healthy.
- **2021 / 2022 / 2024 / 2025:** both sources stat-bearing → genuine **per-field mean of ESPN + Sleeper**, scored once under the ruleset.
- Draft basis, VORP, and the entire downstream backtest consume the improved consensus with no code change.

## Requirements

R1. The Sleeper parse populates all nine `STAT_FIELDS` from `SLEEPER_STAT_FIELDS` for rows whose Sleeper payload carries those keys; absent keys default to 0.0; the count-rounding convention matches ESPN's.
R2. After re-ingest, Sleeper rows in an `external_projections` snapshot are **not** all-NA on stat columns, and the snapshot still validates against `ExternalProjectionSchema` (no schema change; existing columns populated).
R3. `build_consensus` includes a source row's stat values in the per-field mean **iff** that row is stat-bearing (≥2 non-null scoring fields). A `gsis_id` group with one stub row and one full row blends to the full row's line only. A group with two full rows blends to their per-field mean. A group with only stub/no-stat rows yields `has_points=False`, `projected_points_ppr=None`.
R4. The existing ESPN-only behavior is preserved: a group containing a single full ESPN row (no Sleeper stats) produces the same projection as before this change (regression guard).
R5. `consensus_adp` and `consensus_rank` are unaffected by the points change (ADP aggregation untouched).
R6. No all-NA `pd.concat` FutureWarning is emitted during ingest.
R7. **Verification (definition of done):** after re-ingesting 2021–2025, every season's **draft-basis pool** — defined as the rows returned by `build_draft_basis(...)`, not the raw snapshot — has `season_mean_fpts > 0` for **≥ 90%** of those rows (2023 must rise from 99/514 to the ~95% the other seasons show).

R8. **Cross-source value sanity (definition of done):** on 2024 and 2025 (both sources stat-bearing), compare, per player on the ESPN∩Sleeper overlap, OUR-scored ESPN-only projected points vs OUR-scored Sleeper-only projected points (each scored under `league_config.ruleset` from that source's raw stat line alone). They must correlate at **r ≥ 0.85** with **median ratio in [0.85, 1.15]**. This proves the Sleeper mapping is complete and unit-/semantically-consistent with ESPN — that the blend produces *correct values*, not merely *populated* ones. (Distinct from the prior r≈0.95 finding, which compared Sleeper's own `pts_half_ppr` to ESPN; R8 checks OUR scoring of Sleeper's raw line.)

## Edge cases / failure modes

- **ADP-only player** (no source stat-bearing): `has_points=False`, `projected_points_ppr=None` — unchanged; such players are ADP-only bot fodder, already tolerated by `generate_vorp_table` / `consensus_to_season_projections` (they receive no VORP and are not draftable by the value strategies). No new handling required; covered by R3/R4 regression.
- **Sleeper-only player** (deep roster, no ESPN row): blended from Sleeper alone — correct.
- **ESPN-only player** (in ESPN, missing from Sleeper): blended from ESPN alone — correct, matches today.
- **Fractional Sleeper counts** (e.g. `rec = 105.0`, or fractional first-down stats): handled by the shared rounding convention at statline construction + `expected_points` (fractional-safe). Non-`STAT_FIELDS` Sleeper keys (`gp`, `cmp_pct`, `pass_fd`, `bonus_rec_wr`, `*_2pt`, all `adp_*`) are ignored by the mapping.
- **Ruleset correctness:** both sources' **raw stat lines** are scored under `league_config.ruleset`; Sleeper's pre-scored `pts_half_ppr`/`pts_ppr`/`pts_std` are **never** used. The blend is therefore correct for any ruleset, not locked to half-PPR.
- **Source disagreement** (ESPN and Sleeper differ on a stat): resolved by the per-field mean — the intended consensus behavior; no special handling.

## Testing expectations

TDD; each unit gets a failing test first. Tests use synthetic payloads (no network), following the existing ESPN parser tests as the template.

- `_sleeper_stats_to_statline`: synthetic Sleeper `stats` dict → correct canonical `StatLine` for all nine fields, storing **raw fractional values (no rounding), matching `_espn_stats_to_statline`**, and ignoring non-mapped keys (`gp`, `cmp_pct`, `pass_fd`, `bonus_rec_wr`, `*_2pt`, `adp_*`).
- Sleeper parse: emits populated `STAT_FIELDS` columns (assert not all-NA); ESPN parse output unchanged.
- `_is_stat_bearing`: 0/1 non-null scoring fields → False; ≥2 → True.
- `build_consensus` stub guard (the four R3/R4 cases): (a) full + stub → full only; (b) two full → per-field mean then scored once; (c) two stub/no-stat → `has_points=False`; (d) single full ESPN row → byte-identical to pre-change projection (regression).
- `build_consensus` ADP unaffected (R5): a group's `consensus_adp`/`consensus_rank` unchanged by adding/removing Sleeper stats.
- No all-NA concat FutureWarning during ingest assembly (R6): assert via `warnings.catch_warnings(record=True)` or equivalent.
- Run the CLAUDE.md ingest/schema integration gate: `pytest -v -k "ingest or store or schemas"` (this change touches an ingest path and exercises a stored-snapshot boundary).

## Phasing

Single coherent phase (the durable data-layer change), gated by R1–R8 and the full project gate suite (`pytest`, `mypy src tests`, `ruff check`, `ruff format --check`).

**Post-merge payoff (NOT in this spec):** re-ingest 2021–2025 (already part of R7 verification), then re-run the five-season H2H backtests on the blended basis and regenerate the projected-vs-actual correlation + post-draft assessment (`_diag_assess.py`, extended to all five seasons). This is throwaway analysis over existing code; it is the motivation for the change but is tracked separately so the production spec stays a clean, testable unit.
