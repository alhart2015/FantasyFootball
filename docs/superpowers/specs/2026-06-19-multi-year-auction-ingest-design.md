# Multi-year auction-test data ingest — design

**Status:** approved (brainstormed 2026-06-19). Proceeding to plan + execute.
**Owner:** draft-hub / auction + ingest.
**Depends on:** the `external_projections` ingest (`src/projections/ingest/external_projections.py`), the consensus blend (`build_consensus`), the preset generator (`scripts/generate_preset_vorp_tables.py`), and the preset registry (`src/projections/draft/assistant/presets.py`). ESPN auction values land via Slice 1's resolver (`resolve_espn_auction_dollars`).

## Problem

TODO #49a (multi-year averaging) wants the auction bake-off re-run across every season we have projection data for, averaging the per-model metrics so the strategy ranking is a multi-year mean rather than one noisy 2026 draw (Runs A–H are each a single-season snapshot). The blocker: **only 2026 has an `external_projections` snapshot on disk**, so only 2026 has a preset VORP table to feed the bake-off. This spec delivers the **data ingest** that produces per-season bake-off inputs for 2021–2026. The multi-year bake-off *runner + averaging* itself is the separate #49a slice — out of scope here.

## Feasibility (verified 2026-06-19, live API probe via the real parsers)

`parse_espn_players(fetch_espn(Y), Y)` + `parse_sleeper_projections(fetch_sleeper_season(Y))` for Y ∈ {2021, 2023, 2025, 2026}:

| Season | ESPN rows w/ proj | crowd auction | expert (PPR/STD) auction | Sleeper w/ stats | usable `espn_auction_dollars` |
|---|---|---|---|---|---|
| 2021 | 484 | 222 | 0 | 628 | yes — crowd-only (~222) |
| 2023 | 514 | 285 | 140 | 596 | yes — both (~285) |
| 2025 | 460 | 0 | 160 | 547 | yes — expert-only (~160) |
| 2026 | 459 | 201 | 160 | 559 | yes — both (~201) |

**ESPN projections and Sleeper projections come back for every season**, so the `external → consensus → VORP` pipeline can build a real preset table per season. **Every season has *some* usable ESPN auction value**, but the *source flips* year to year (2021 crowd-only; 2025 expert-only; 2023/2026 both). Slice 1's resolver (crowd-if-present-and-positive else ruleset-expert) produces a usable `espn_auction_dollars` column each season transparently — but the anchor's *basis* is not apples-to-apples across years (a documented caveat for the averaging, not a blocker).

## Goals

1. Ingest `external_projections` for 2021–2025 (2026 already on disk) — the existing CLI, run per season.
2. Make the **preset generator + preset registry season-aware** so each season's tables land in `data/vorp_{season}/` (today they hardcode `data/vorp_2026/` regardless of `--season`).
3. Generate per-season preset VORP tables for all six seasons (3 scoring × 3 sizes each) with `espn_auction_dollars` populated.
4. A thin **`scripts/refresh_external_seasons.py`** wrapper that loops ingest → preset-gen across a season list, so #49a (and future refreshes) can reproduce the inputs with one command.
5. **No regression to the live draft board**, which consumes `presets.get_preset(...)` for the current (2026) season.

## Non-goals

- **Not** the multi-year bake-off runner or the averaging logic — that is the #49a test slice (it consumes these per-season tables + per-season availability; a separate spec/plan).
- **Not** changing `resolve_espn_auction_dollars` or the crowd/expert rule — each season uses whatever ESPN served, as-is.
- **Not** backfilling a *consistent* auction-value basis across seasons (ESPN doesn't provide one); the basis-variance is recorded, not corrected.
- **Not** re-ingesting weekly_stats/schedules/id_map — availability/byes for the downstream runner come from the existing store (#49a's concern).
- **Not** K/DST (consensus is skill-only, unchanged).

## Chosen approach (A — season-param the preset registry)

The season-awareness lives in **one place**: `presets.get_preset` gains an optional `season: int = 2026` parameter that drives the table directory and the config name. Everything downstream (the generator's write path, `materialize_league_config`) derives from the preset, so there's a single source of truth and the board's existing `get_preset(scoring, n)` call is unchanged (defaults to 2026 → byte-identical behavior).

### A — `presets.py`

- Replace the module-level `_TABLE_DIR = Path("data/vorp_2026")` with a helper `_table_dir(season: int) -> Path` returning `Path(f"data/vorp_{season}")`.
- `get_preset(scoring_key, n_teams, season: int = 2026)`:
  - `table_path = _table_dir(season) / f"{scoring_key}_{n_teams}team.parquet"`,
  - `league_config` name `f"{scoring_key}_{n_teams}team_{season}"` (was hardcoded `..._2026`; `_skill_config` takes `season`).
  - `season=2026` reproduces today's `table_path`/name exactly.
- `materialize_league_config` is unchanged — it already writes next to `preset.table_path.parent`, which is now season-aware for free.

### B — `scripts/generate_preset_vorp_tables.py`

- `--season` already exists. Pass it through: `get_preset(scoring_key, n_teams, season=args.season)` everywhere the script builds a preset.
- **Write each table to `preset.table_path`** (the canonical location the board reads), creating `preset.table_path.parent` per write. This **removes the existing `out_dir = args.data_root / "vorp_2026"` hardcode** (and the data-root-vs-cwd-relative mismatch it created — the script must write where the board reads, which is the cwd-relative `data/vorp_{season}/`, not under `--data-root`).
- `--data-root` keeps its current meaning: where to **read** the `external_projections` snapshot (and id_map) from. (Decoupling read-root from the canonical write-path is the fix for the Slice-2 footgun where the mkdir'd dir and the actual write target diverged.)
- `build_preset_table(external, scoring_key, n_teams, season)` threads `season` into its `get_preset` call (the rest — `build_consensus` → `consensus_to_season_projections` → `generate_vorp_table` → resolve + merge `espn_auction_dollars` — is unchanged).

### C — `scripts/refresh_external_seasons.py` (new, thin)

A loop, no new logic: for each season in a `--seasons` list (default `2021 2022 2023 2024 2025 2026`), call `refresh_external_projections(data_root, season=Y)` then `generate_preset_vorp_tables.main(["--season", str(Y), "--data-root", <root>])` (or import its `main`/a `run(season, data_root)` entry). Prints a per-season summary. Network-dependent; a per-season failure is logged and the loop continues (one bad season shouldn't abort the rest).

### Data runs (the deliverable, executed at the end)

`python scripts/refresh_external_seasons.py --data-root <store>` → writes `data/raw/external_projections/season={2021..2025}/asof=YYYY-MM-DD/` and regenerates `data/vorp_{2021..2026}/{scoring}_{n}team.parquet` (all untracked artifacts, regenerable). 2026 is re-run for uniformity. Verify each season's `half_12team.parquet` has a non-NA `espn_auction_dollars` (via `has_usable_espn_prices`).

## Requirements

R1. `presets.get_preset(scoring_key, n_teams, season: int = 2026)` returns a `DraftPreset` whose `table_path` is `data/vorp_{season}/{scoring}_{n}team.parquet` and whose `league_config.name` is `{scoring}_{n}team_{season}`. `season=2026` (the default) reproduces the current `table_path` and name **byte-for-byte**. `_skill_config` takes `season`.
R2. `generate_preset_vorp_tables.py --season Y` builds every preset with `season=Y` and writes each table to its `preset.table_path` (creating the parent dir), reading the `external_projections` snapshot from `--data-root`. The hardcoded `vorp_2026` output dir is removed. Default `--season 2026` lands tables in `data/vorp_2026/` exactly as today.
R3. `scripts/refresh_external_seasons.py` loops `--seasons` (default 2021–2026), running ingest + preset-gen per season, continuing past a single-season failure with a logged warning, and printing a per-season status summary.
R4. **No board regression:** every existing caller of `get_preset(scoring, n)` (the live board, `materialize_league_config`, the auction CLI's smoke tests) is unaffected because the new param defaults to 2026.
R5. Conventions: `GsisId` canonical; `Ruleset`/`RosterSlot` referenced as enums; `pd.Int64Dtype` for `espn_auction_dollars`; `SCHEMA.validate(df)` at boundaries (the preset path already validates `VorpTableSchema`); no new direct parquet I/O outside the established preset-write path; mypy-strict + ruff clean.
R6. **Data deliverable:** per-season `external_projections` snapshots (2021–2025) and per-season preset tables (2021–2026) exist in the store, each season's `half_12team.parquet` carrying a usable `espn_auction_dollars`.

## Edge cases / caveats

- **Auction-value basis varies by season** (2021 crowd-only, 2025 expert-only, others both). Each season's bots will anchor on whatever ESPN served that year; the #49a average mixes bases. Documented; the resolver already degrades cleanly per season.
- **Point-in-time validity.** Historical ESPN/Sleeper rows are their *currently-stored* `seasonId=Y` projected lines, interpreted as that season's preseason set. The auction test is **projected-vs-projected** (roster quality under that season's own projections, not actual outcomes), so it's internally consistent per season regardless of whether the stored projection is the exact original preseason value — but the average should be read as "across six projection snapshots," not "six true point-in-time drafts."
- **Current id_map.** Historical players crosswalk through today's `id_map`; a 2021 player still active gets a real gsis, others get the deterministic placeholder (existing behavior). A retired-since player who was a placeholder stays a placeholder — fine for a strategy bake-off.
- **Coverage varies** (160–285 priced players/season). The deep-pool $1-floor share differs by year; observed, not corrected.
- **Sleeper 2021** had the most stat-bearing rows (628); no season returned an empty pull. If a future season *did* return empty for one source, the ingest already writes the other (existing behavior).

## Testing expectations

- **`presets`** (`tests/test_draft/test_presets.py` — create if absent, else extend): `get_preset(s, n, season=2023).table_path == data/vorp_2023/{s}_{n}team.parquet` and `.league_config.name == "{s}_{n}team_2023"`; `get_preset(s, n)` (default) yields the unchanged 2026 path + name (a frozen-string assertion guarding R4); `_table_dir(season)` is correct for a couple of seasons.
- **`generate_preset_vorp_tables`** (`tests/test_scripts/test_generate_preset_vorp_tables.py`, extend): with `--season 2023` on a synthetic external fixture, tables are written under `data/vorp_2023/` (tmp-path'd) and carry `espn_auction_dollars`; the existing default-2026 tests pass unchanged; `build_preset_table(..., season=2023)` validates against `VorpTableSchema`.
- **`refresh_external_seasons`** (new test): the loop calls ingest + preset-gen once per requested season (monkeypatch both to avoid network), and a raised error for one season is caught + logged without aborting the others.
- **No-regression:** the existing preset/board/auction tests pass unchanged (R4) — run `pytest -k "preset or board or auction or schemas"`.
- Per `CLAUDE.md`, run `pytest -k "ingest or store or schemas"` (touches the ingest/preset seam).

## Phasing

~3 tasks, each ≤5 files:
1. **Season-aware preset registry** — `presets.py` (`_table_dir`, `get_preset`/`_skill_config` season param) + `test_presets.py` (per-season path/name + frozen 2026 default).
2. **Season-aware generator + wrapper** — `generate_preset_vorp_tables.py` (thread `season`, write to `preset.table_path`, drop the `vorp_2026` hardcode) + `scripts/refresh_external_seasons.py` (loop) + their tests.
3. **Data runs + verification** — `python scripts/refresh_external_seasons.py` for 2021–2026 (network), verify each season's preset tables carry a usable `espn_auction_dollars`, and record the per-season coverage in `project_management.md` / TODO #49a.

## Open questions for later (the #49a runner, not this slice)

- The multi-year bake-off runner: per-season availability (2025/2026 availability is computed from ≤2024 weekly_stats — forward-looking, same as the current 2026 runs), how to weight/average across seasons, and whether to report per-season *and* pooled.
- Whether to snapshot ESPN's crowd auction value live each future preseason so later multi-year runs can use a consistent behavioral basis going forward.
