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

  All nine `STAT_FIELDS` are covered. Reference the canonical `STAT_FIELDS` field-name constants (matching the existing `ESPN_STAT_IDS` value convention in the same module), never bare ad-hoc strings.
- Add `_sleeper_stats_to_statline(stats: dict[str, float]) -> dict[str, float] | None` mirroring the existing `_espn_stats_to_statline`. Behavior:
  - **Returns `None` when the Sleeper `stats` dict contains none of the mapped keys** (an ADP-only Sleeper row with no real projection). The caller then leaves that row's stat columns as `NA` — honest "no projection" rather than a fabricated all-zero line. (Mirrors the spirit of ESPN's `if not proj_stats: continue`, except Sleeper keeps the row for its ADP / union coverage.)
  - **Otherwise** maps present mapped keys to canonical fields and defaults absent mapped fields to 0.0, storing the **raw fractional values, with no rounding** — exactly as `_espn_stats_to_statline` does (that function deliberately keeps ESPN's raw fractional projections; per its own comment, "Rounding here is irreversible and biases season totals; the scoring layer is the only place that decides how a projected stat becomes points"). `round_count`/`COUNT_FIELDS` are NOT used here; they are retained only for the frozen benchmark spike. Storing Sleeper raw (no rounding) is what makes the two sources' stat lines directly comparable before averaging.
- In the Sleeper parse path, when `_sleeper_stats_to_statline(...)` returns a dict, add those `STAT_FIELDS` keys to the row; when it returns `None`, omit them (the column becomes `NaN` for that row). Flip the Sleeper source spec from `has_stats=False` to `has_stats=True` so `_to_canonical` carries the parsed stat columns through (real values for stat-bearing rows, `NA` for ADP-only rows).
- **Side effect (intended):** the all-NA-column `pd.concat` FutureWarning disappears once Sleeper's stat columns carry real values. **Caveat:** `espn_draft_rank` is still all-`NA` on the Sleeper frame (Sleeper has no draft rank), so populating stats alone may not fully silence the warning. R6 is verified empirically; if a residual all-NA-column warning remains, set the offending column's dtype explicitly (the `null_col` is already `Float64`) or otherwise make the `concat` dtype-unambiguous so no warning is emitted.

### Change 2 — `src/projections/consensus/blend.py` (`build_consensus`): stat-bearing gate

- Add a predicate `_is_stat_bearing(row, *, min_fields: int = 2) -> bool`: a single source row qualifies iff it has **≥ `min_fields` `STAT_FIELDS` values that are both non-null AND non-zero (`> 0`)**. The non-zero condition is essential (see below).
- Change the per-player stat aggregation so the per-field mean ranges over **stat-bearing source rows only**, not all rows in the `gsis_id` group. `has_points` becomes "≥1 stat-bearing source row exists." The identity-row preference for a stat-bearing row is retained.
- The ADP aggregation is **unchanged** (still mean over all rows with `adp > 0`); ADP and points gating are independent.

### Why the gate tests non-null AND non-zero, with `min_fields = 2`

Both statline constructors **zero-fill absent fields** (`out = {f: 0.0 for f in STAT_FIELDS}`; verified in `_espn_stats_to_statline`). So a degenerate ESPN 2023 stub — whose projection block has 1 field, usually a *non-scoring* one — is stored as an **all-`0.0`, fully non-null** row. A "≥2 non-null" gate would therefore count every stub as stat-bearing and exclude nothing. The gate must test **non-zero**: a real projection has several positive stats (QB ≥4: `passing_yards`/`passing_tds`/`interceptions`/`rushing_yards`; RB ≥2–4; WR/TE ≥3 via `receptions`/`receiving_yards`/`receiving_tds`), while a stub has 0 positive fields (or, for the rare ~9% with a scoring stub, 1). `min_fields = 2` non-zero fields cleanly separates real lines from stubs, with margin against false-exclusion of a thin-but-real projection. (This also subsumes any ADP-only Sleeper row that slipped through as all-zero: 0 non-zero fields → excluded.) The threshold and the `> 0` test are a named constant + documented predicate so they are easy to audit/tune.

**Accepted limitation (absent-vs-zero):** because absent fields are stored as 0.0, if a *stat-bearing* source omits a position-relevant scoring field that the other source reports, the per-field mean averages a real value against a spurious 0, biasing that field low. In practice each source reports the fields relevant to a player's position (a WR's passing stays 0 in both; a WR's receiving is present in both), so collisions land on cross-position-irrelevant zeros. R8's cross-source value-sanity gate (r ≥ 0.85, median ratio ∈ [0.85, 1.15]) would catch any systematic distortion from this. A per-field source-presence model (distinguishing absent from zero) is out of scope (would require schema changes) — YAGNI.

### Net behavioral result

- **2023:** ESPN stubs are not stat-bearing → excluded → the blend is **Sleeper-only** for that season (clean, no half-weighted stub field). Pool becomes healthy.
- **2021 / 2022 / 2024 / 2025:** both sources stat-bearing → genuine **per-field mean of ESPN + Sleeper**, scored once under the ruleset.
- Draft basis, VORP, and the entire downstream backtest consume the improved consensus with no code change.

## Requirements

R1. `_sleeper_stats_to_statline` returns a stat line (raw fractional values, no rounding, absent mapped fields defaulted to 0.0) when the Sleeper `stats` dict carries ≥1 mapped key, and returns `None` (→ stat columns left `NA`) when it carries none. The Sleeper parse stores those columns accordingly; values are stored **raw, with no rounding** (matching `_espn_stats_to_statline`).
R2. After re-ingest, a stat-bearing Sleeper row in an `external_projections` snapshot carries real (non-NA) stat values, and the snapshot still validates against `ExternalProjectionSchema` (no schema change; existing columns populated).
R3. `build_consensus` includes a source row's stat values in the per-field mean **iff** that row is stat-bearing (**≥2 `STAT_FIELDS` values that are non-null AND non-zero**). A `gsis_id` group with one all-zero/stub row and one full row blends to the full row's line only. A group with two full rows blends to their per-field mean. A group with only stub/all-zero/no-stat rows yields `has_points=False`, `projected_points_ppr=None`.
R4. The existing ESPN-only behavior is preserved: a group containing a single full ESPN row (no Sleeper stats) produces the same projection as before this change (regression guard).
R5. `consensus_adp` and `consensus_rank` are unaffected by the points change (ADP aggregation untouched).
R6. No all-NA `pd.concat` FutureWarning is emitted during ingest.
R7. **Verification (definition of done):** after re-ingesting 2021–2025, every season's **draft-basis pool** — defined as the rows returned by `build_draft_basis(...)`, not the raw snapshot — has `season_mean_fpts > 0` for **≥ 90%** of those rows (2023 must rise from 99/514 to the ~95% the other seasons show).

R8. **Cross-source value sanity (definition of done):** on 2024 and 2025 (both sources stat-bearing), compare, per player on the ESPN∩Sleeper overlap, OUR-scored ESPN-only projected points vs OUR-scored Sleeper-only projected points (each scored under `league_config.ruleset` from that source's raw stat line alone). They must correlate at **r ≥ 0.85** with **median ratio in [0.85, 1.15]**. This proves the Sleeper mapping is complete and unit-/semantically-consistent with ESPN — that the blend produces *correct values*, not merely *populated* ones. (Distinct from the prior r≈0.95 finding, which compared Sleeper's own `pts_half_ppr` to ESPN; R8 checks OUR scoring of Sleeper's raw line.)

## Edge cases / failure modes

- **ADP-only player** (no source stat-bearing): `_sleeper_stats_to_statline` returns `None` → stat columns `NA`; the gate excludes the row → `has_points=False`, `projected_points_ppr=None` — unchanged downstream; such players are ADP-only bot fodder, already tolerated by `generate_vorp_table` / `consensus_to_season_projections` (no VORP, not draftable by value strategies). Covered by R3/R4 regression.
- **Sleeper-only player** (deep roster, no ESPN row): blended from Sleeper alone — correct.
- **ESPN-only player** (in ESPN, missing from Sleeper): blended from ESPN alone — correct, matches today.
- **All-zero / stub row** (ESPN 2023 stub, or any fabricated all-zero line): non-zero field count 0–1 < 2 → not stat-bearing → excluded from the blend. This is the core mechanism, not an exception.
- **Fractional Sleeper counts** (e.g. `rec = 105.0`): stored raw (no rounding) and scored by `expected_points` (fractional-safe). Non-`STAT_FIELDS` Sleeper keys (`gp`, `cmp_pct`, `pass_fd`, `rush_fd`, `rec_fd`, `bonus_rec_wr`, `*_2pt`, `rec_0_4`/`rec_5_9`/…, all `adp_*`) are ignored by the mapping.
- **Ruleset correctness:** both sources' **raw stat lines** are scored under `league_config.ruleset`; Sleeper's pre-scored `pts_half_ppr`/`pts_ppr`/`pts_std` are **never** used. The blend is therefore correct for any ruleset, not locked to half-PPR.
- **Source disagreement** (ESPN and Sleeper differ on a stat): resolved by the per-field mean — the intended consensus behavior; no special handling.

## Testing expectations

TDD; each unit gets a failing test first. Tests use synthetic payloads (no network), following the existing ESPN parser tests as the template.

- `_sleeper_stats_to_statline`: (a) a `stats` dict with mapped keys → correct canonical stat line for all nine fields, **raw fractional values (no rounding), matching `_espn_stats_to_statline`**, ignoring non-mapped keys (`gp`, `cmp_pct`, `pass_fd`, `rush_fd`, `rec_fd`, `bonus_rec_wr`, `*_2pt`, `rec_0_4`…, `adp_*`); (b) a `stats` dict with **no** mapped keys (only `adp_*`) → returns `None`.
- Sleeper parse: a row with mapped stats emits populated (non-NA) `STAT_FIELDS`; an ADP-only row leaves them `NA`; ESPN parse output unchanged.
- `_is_stat_bearing`: all-zero row (0 non-zero fields) → False; 1 non-zero field → False; ≥2 non-zero fields → True; an all-NA row → False.
- `build_consensus` stub guard (the four R3/R4 cases): (a) full + all-zero(stub) → full only; (b) two full → per-field mean then scored once; (c) two all-zero/no-stat → `has_points=False`, `projected_points_ppr=None`; (d) single full ESPN row (Sleeper no stats) → byte-identical to pre-change projection (regression).
- `build_consensus` ADP unaffected (R5): a group's `consensus_adp`/`consensus_rank` unchanged by adding/removing Sleeper stats.
- No all-NA concat FutureWarning during ingest assembly (R6): assert via `warnings.catch_warnings(record=True)` or equivalent.
- Run the CLAUDE.md ingest/schema integration gate: `pytest -v -k "ingest or store or schemas"` (this change touches an ingest path and exercises a stored-snapshot boundary).

## Phasing

Single coherent phase (the durable data-layer change), gated by R1–R8 and the full project gate suite (`pytest`, `mypy src tests`, `ruff check`, `ruff format --check`).

**Post-merge payoff (NOT in this spec):** re-ingest 2021–2025 (already part of R7 verification), then re-run the five-season H2H backtests on the blended basis and regenerate the projected-vs-actual correlation + post-draft assessment (`_diag_assess.py`, extended to all five seasons). This is throwaway analysis over existing code; it is the motivation for the change but is tracked separately so the production spec stays a clean, testable unit.
