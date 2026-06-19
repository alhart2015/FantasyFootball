# Multi-year auction-test data ingest — design

**Status:** approved (brainstormed 2026-06-19; hardened via spec-review 2026-06-19). Proceeding to plan + execute.
**Owner:** draft-hub / auction + ingest.
**Depends on:** the `external_projections` ingest (`src/projections/ingest/external_projections.py`), the consensus blend (`build_consensus`), the preset generator (`scripts/generate_preset_vorp_tables.py`), and the preset registry (`src/projections/draft/assistant/presets.py`). ESPN auction values land via Slice 1's resolver (`resolve_espn_auction_dollars`).

## Problem

TODO #49a (multi-year averaging) wants the auction bake-off re-run across every season we have projection data for, averaging the per-model metrics so the strategy ranking is a multi-year mean rather than one noisy 2026 draw (Runs A–H are each a single-season snapshot). The blocker: **only 2026 has an `external_projections` snapshot on disk**, so only 2026 has a preset VORP table to feed the bake-off. This spec delivers the **data ingest** that produces per-season bake-off inputs for 2021–2026. The multi-year bake-off *runner + averaging* itself is the separate #49a slice — out of scope here.

## Feasibility (verified 2026-06-19, live API probe via the real parsers, all six seasons)

| Season | ESPN rows w/ proj | crowd auction | expert (PPR/STD) auction | Sleeper w/ stats | usable `espn_auction_dollars` |
|---|---|---|---|---|---|
| 2021 | 484 | 222 | 0 / 0 | 628 | yes — crowd-only |
| 2022 | 491 | 267 | 0 / 0 | 617 | yes — crowd-only |
| 2023 | 514 | 285 | 140 / 140 | 596 | yes — both |
| 2024 | 475 | 230 | 139 / 160 | 597 | yes — both |
| 2025 | 460 | 0 | 160 / 160 | 547 | yes — expert-only |
| 2026 | 459 | 201 | 160 / 160 | 559 | yes — both |

**ESPN + Sleeper projections come back for every season**, and **every season has ≥1 usable ESPN auction value** (no all-NA season). The *source flips* year to year: pre-2023 seasons are **crowd-only** (ESPN didn't retain the expert draft-rank auction value); 2025 is **expert-only** (crowd reset to 0); 2023/24/26 have both. Slice 1's resolver (crowd-if-present-and-positive else ruleset-expert) produces a usable `espn_auction_dollars` each season transparently — but the anchor's *basis is not apples-to-apples across years* (documented caveat for the averaging, below).

## Goals

1. Ingest `external_projections` for **2021–2025** (2026 is already on disk and is **not** re-ingested — see Non-goals + Edge cases).
2. Make the **preset registry + generator season-aware** so each season's tables (and league configs) land in `data/vorp_{season}/` (today they hardcode `data/vorp_2026/` regardless of `--season`).
3. Generate per-season preset VORP tables **and their `.league.json` configs** for all six seasons (3 scoring × 3 sizes each) with `espn_auction_dollars` populated.
4. A thin **`scripts/refresh_external_seasons.py`** wrapper that loops ingest → preset-gen across a season list, so #49a (and future refreshes) reproduce the inputs with one command.
5. **No regression to the live draft board**, which consumes `presets.get_preset(...)` for the current (2026) season.

## Non-goals

- **Not** the multi-year bake-off runner or the averaging logic — that is the #49a test slice (separate spec/plan).
- **Not** re-ingesting 2026. Its snapshot (`asof=2026-06-19`) is the baseline Runs A–H were built on; a re-ingest on a later date would write a newer `asof` that `read_latest_partition` silently prefers, drifting the 2026 cell. 2026's tables are *regenerated* from the existing snapshot (deterministic, no network) only to add the full set of `.league.json` configs.
- **Not** changing `resolve_espn_auction_dollars` or the crowd/expert rule — each season uses whatever ESPN served.
- **Not** backfilling a consistent auction-value basis across seasons (ESPN doesn't provide one); the basis-variance is recorded, not corrected.
- **Not** re-ingesting weekly_stats/schedules/id_map (availability/byes are the #49a runner's concern, from the existing store).
- **Not** K/DST (consensus is skill-only, unchanged).

## Chosen approach (A — season-param the preset registry)

Season-awareness lives in **one place**: `presets.get_preset` gains `season: int = 2026`, driving the table directory and config name. Everything downstream (the generator's write path, `materialize_league_config`) derives from the preset, so there's a single source of truth and the board's existing `get_preset(scoring, n)` call is unchanged (defaults to 2026 → byte-identical behavior).

### A — `presets.py`

- Replace `_TABLE_DIR = Path("data/vorp_2026")` (presets.py:40) with `_table_dir(season: int) -> Path` returning `Path(f"data/vorp_{season}")`.
- `get_preset(scoring_key, n_teams, season: int = 2026)`: `table_path = _table_dir(season) / f"{scoring_key}_{n_teams}team.parquet"`; `league_config` name `f"{scoring_key}_{n_teams}team_{season}"` (via `_skill_config(scoring_key, n_teams, season)`). **`season=2026` reproduces today's `table_path` and name byte-for-byte.**
- `materialize_league_config` is unchanged — it writes `{scoring}_{n}team.league.json` next to `preset.table_path.parent` (now season-aware for free). The filename has no season, but the *directory* (`data/vorp_{season}/`) disambiguates, and the `.json`'s `"name"` field carries the season. **Name-change safety:** grep confirms **no consumer asserts or branches on the literal `LeagueConfig.name`** (the live board, `live.py` resume, and the bake-off all read configs structurally, not by name); the only `.name ==` assertions in the suite are on `Ruleset.name`. So the `..._{season}` name is safe, and default-2026 is byte-identical.

### B — `scripts/generate_preset_vorp_tables.py`

- Add `season: int = 2026` default to `build_preset_table(external, scoring_key, n_teams, season=2026)` — **a trailing param WITH a default**, so the existing positional callers (`build_preset_table(external, "half", 12)` in the script loop + the existing 3-arg positional test calls) keep working unchanged. Thread `season` into its `get_preset(scoring_key, n_teams, season=season)` call.
- The script's `main`: pass `--season` through to `build_preset_table(..., season=args.season)`.
- **Write each table to `preset.table_path`** (creating `preset.table_path.parent` with `exist_ok=True` per write) **and write its config via `materialize_league_config(preset)`** (closing the missing-config gap — the #49a runner pairs `--vorp-table data/vorp_{Y}/half_12team.parquet` with `--league-config data/vorp_{Y}/half_12team.league.json`, both of which must exist). **Remove the `out_dir = args.data_root / "vorp_2026"` hardcode** (presets.py-era dead path).
- **Path invariant (read the Slice-2 footgun correctly):** `preset.table_path` is **cwd-relative** (`data/vorp_{season}/…`) — exactly where the live board reads (`draft_board.py` reads `preset.table_path` with `data_root=Path("data")`). The whole preset system assumes the process runs from the **repo root**. `--data-root` is the **read-root** for the `external_projections` snapshot + id_map only. For the canonical workflow, **run from the repo root with the default `--data-root data`**, so the external read (`data/raw`) and the table writes (`data/vorp_{season}`) share the repo's `data/` tree. Do **not** pass an absolute `--data-root` outside the repo's `data/` for table generation — the board/runner read tables cwd-relative, so writes must land under `<repo>/data`. (The Slice-2 worktree failure was exactly this divergence: an absolute `--data-root` made the external-read root and the cwd-relative write target diverge.)
- Update the module docstring + `ArgumentParser(description=...)` (currently say "for 2026"/"data/vorp_2026/") to reflect the now-multi-season behavior.

### C — `scripts/refresh_external_seasons.py` (new, thin)

A loop, no new pipeline logic. `--seasons` (default `2021 2022 2023 2024 2025` — **2026 excluded**, see Non-goals) + `--data-root` (default `data`). For each season Y, in order:
1. `refresh_external_projections(data_root, season=Y)` (the existing function — signature `(data_root, *, season, asof=None)`).
2. `generate_preset_vorp_tables.main(["--season", str(Y), "--data-root", str(data_root)])` (the existing `main(argv)` CLI entry — the single chosen mechanism; no new `run()` refactor).

Wrap each season's two calls in `try/except (ExternalProjectionError, OSError)` — the two failure classes that actually occur (`refresh_external_projections` raises `ExternalProjectionError`; the generator raises `OSError`/`FileNotFoundError` on a missing snapshot). Log a warning and **continue to the next season** (one flaky API season shouldn't discard the rest); print a per-season `ok/failed` summary at the end. (Catching this *named two-class tuple* — not a bare `except` — satisfies the no-broad-except convention.)

### Data runs (the deliverable, executed in Phase 3)

From the repo root:
1. `python scripts/refresh_external_seasons.py` → ingests 2021–2025 + writes `data/vorp_{2021..2025}/{scoring}_{n}team.{parquet,league.json}`.
2. `python scripts/generate_preset_vorp_tables.py --season 2026` → **regenerates 2026's tables + configs from the existing `asof=2026-06-19` snapshot** (no network, deterministic → byte-identical parquets, plus the full `.league.json` set). This adds 2026 to the season-uniform layout **without re-ingesting**.
3. **Hard verification gate (R6):** for each season Y ∈ {2021..2026}, assert `has_usable_espn_prices(read_parquet(data/vorp_{Y}/half_12team.parquet))` is True, and log the priced-player count per season; flag (don't fail) any season with `< 100` priced players as a coverage caveat. A False here is a loud failure, not a silent NA column.

## Requirements

R1. `presets.get_preset(scoring_key, n_teams, season: int = 2026)` returns a `DraftPreset` whose `table_path` is `data/vorp_{season}/{scoring}_{n}team.parquet` and `league_config.name` is `{scoring}_{n}team_{season}`. `season=2026` reproduces the current `table_path` and name **byte-for-byte** (guarded by a frozen-string test). `_skill_config` takes `season`. The board's two-arg `get_preset(scoring, n)` call rides the default → unchanged.

R2. `build_preset_table(external, scoring_key, n_teams, season: int = 2026)` (trailing default — existing positional callers unbroken) threads `season` into `get_preset`. `generate_preset_vorp_tables.py --season Y` writes each table to `preset.table_path` (`mkdir(parents=True, exist_ok=True)` on its parent) **and** writes `materialize_league_config(preset)`; the `out_dir`/`vorp_2026` hardcode is removed; the "for 2026" docstrings are updated. Default `--season 2026` lands tables+configs in `data/vorp_2026/` exactly as today.

R3. `scripts/refresh_external_seasons.py` loops `--seasons` (default 2021–2025), running `refresh_external_projections` then `generate_preset_vorp_tables.main([...])` per season, catching `(ExternalProjectionError, OSError)` per season with a logged warning and continuing, and printing a per-season ok/failed summary.

R4. **No board regression:** every existing caller of `get_preset(scoring, n)` (board, `materialize_league_config`, auction CLI smoke tests) is unaffected (the new param defaults to 2026); no consumer depends on the literal `LeagueConfig.name`.

R5. Conventions: `GsisId` canonical; `Ruleset`/`RosterSlot` referenced as enums; `pd.Int64Dtype` for `espn_auction_dollars`; `VorpTableSchema.validate` at the preset boundary (already present); no new direct parquet I/O outside the preset-write path; `mkdir(exist_ok=True)`; named-exception catches only; mypy-strict + ruff clean.

R6. **Data deliverable + gate:** per-season `external_projections` snapshots (2021–2025) and per-season preset tables **+ `.league.json` configs** (2021–2026) exist in `<repo>/data`. The Phase-3 gate asserts `has_usable_espn_prices` is True for each season's `half_12team` table (loud failure otherwise) and records the per-season priced-player count.

## Edge cases / caveats

- **Auction-value basis varies by season** (pre-2023 crowd-only, 2025 expert-only, others both). Each season's bots anchor on whatever ESPN served; the #49a average mixes bases. Recorded, not corrected.
- **Half-PPR anchors on the PPR *expert* column.** ESPN has no half-PPR auction column, so `resolve_espn_auction_dollars` maps `ESPN_HALF → espn_auction_value_ppr`. In **crowd-present** seasons the crowd value (scoring-agnostic default-league) wins; but in **expert-only** seasons (2025) every half-PPR preset anchors on *PPR* dollars — a within-season scoring-basis mismatch, sharpest exactly where there's no crowd fallback. Inherited from Slice 1 (unchanged here); flagged so the #49a average reads half-PPR 2025 as PPR-anchored.
- **2026 not re-ingested** (Non-goals): preserves the Run-H baseline; 2026 tables are regenerated deterministically from the existing snapshot.
- **Point-in-time validity.** Historical rows are ESPN/Sleeper's *currently-stored* `seasonId=Y` projected lines, interpreted as that season's preseason set. The auction test is **projected-vs-projected** (roster quality under that season's own projections, not actuals), internally consistent per season regardless of whether the stored projection equals the exact original preseason value — read the average as "across six projection snapshots," not "six true point-in-time drafts."
- **Current id_map.** Historical players crosswalk through today's id_map; active players get a real gsis, others the deterministic placeholder (existing behavior). Fine for a strategy bake-off.
- **`model_id` provenance.** Each table's `consensus_to_season_projections` `model_id` is `consensus:<asof>`, not season-tagged; if two seasons are ingested the same UTC day they share a `model_id`. Harmless (VORP ignores it; season provenance lives in the directory name) — noted only so a future `model_id`-keyed reader doesn't conflate seasons.

## Testing expectations

- **`presets`** (`tests/test_draft/test_presets.py`, create or extend): `get_preset(s, n, season=2023).table_path == Path("data/vorp_2023/{s}_{n}team.parquet")` and `.league_config.name == "{s}_{n}team_2023"`; `get_preset(s, n)` (default) yields the unchanged `data/vorp_2026/...` path + `..._2026` name (frozen-string R4 guard); `_table_dir(season)` correct for two seasons; the existing `materialize_league_config` round-trip test passes (default 2026 unchanged).
- **`generate_preset_vorp_tables`** (extend `tests/test_scripts/test_generate_preset_vorp_tables.py`): with `--season 2023` on a synthetic external fixture (monkeypatch `_table_dir`/cwd to a tmp path), tables land under `data/vorp_2023/`, carry `espn_auction_dollars`, **and a matching `.league.json` is written**; `build_preset_table(..., season=2023)` validates against `VorpTableSchema`; the existing default-2026 tests pass unchanged; existing positional `build_preset_table(external, s, n)` calls still work (default-season guard).
- **`refresh_external_seasons`** (new test): monkeypatch `refresh_external_projections` and `generate_preset_vorp_tables.main` to record calls (no network); assert one ingest+gen pair per requested season; a raised `ExternalProjectionError` for one season is caught + logged and the others still run; the summary reports per-season status.
- **No-regression:** `pytest -k "preset or board or auction or schemas"` passes unchanged (R4). Per `CLAUDE.md`, also run `pytest -k "ingest or store or schemas"` (the ingest/preset seam).
- **Phase 3** (data runs) is a network ops task, not CI: its gate is the R6 `has_usable_espn_prices` assertion + priced-count log, run once after the live pulls.

## Phasing

~3 tasks, each ≤5 files:
1. **Season-aware preset registry** — `presets.py` (`_table_dir`, `get_preset`/`_skill_config` season param) + `test_presets.py` (per-season path/name + frozen-2026 default + materialize round-trip).
2. **Season-aware generator + wrapper** — `generate_preset_vorp_tables.py` (season default on `build_preset_table`, write to `preset.table_path` + `materialize_league_config`, drop `out_dir` hardcode, `exist_ok`, docstrings) + `scripts/refresh_external_seasons.py` (loop, named-exception isolation) + their tests.
3. **Data runs + verification (ops)** — `refresh_external_seasons.py` for 2021–2025 + `generate_preset_vorp_tables.py --season 2026` (no re-ingest), run the R6 gate, record per-season coverage in `project_management.md` / TODO #49a.

## Open questions for later (the #49a runner, not this slice)

- The multi-year bake-off runner: per-season availability (2025/2026 availability is computed from ≤2024 weekly_stats — forward-looking, same as the current 2026 runs), how to weight/average across seasons, per-season vs pooled reporting.
- Whether to snapshot ESPN's crowd auction value live each future preseason for a consistent behavioral basis going forward.
