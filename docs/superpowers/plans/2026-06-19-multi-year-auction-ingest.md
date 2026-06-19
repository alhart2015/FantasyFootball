# Multi-year auction-test data ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make preset VORP-table generation season-aware and ingest `external_projections` for 2021–2025, so each season has a per-season preset table + league config (with `espn_auction_dollars`) to feed the multi-year auction bake-off (TODO #49a).

**Architecture:** Season-awareness lives in one place — `presets.get_preset` gains `season: int = 2026` driving `data/vorp_{season}/` and the config name; the generator threads `--season` and writes each table + its `.league.json` to the canonical (cwd-relative) `preset.table_path`; a thin wrapper loops ingest→generate across seasons. The board's 2-arg `get_preset` rides the 2026 default → zero behavior change.

**Tech Stack:** Python, pandas (nullable `Int64`/`Float64`), pandera (`VorpTableSchema`/`ExternalProjectionSchema`), argparse CLIs, pytest, mypy-strict, ruff.

**Spec:** `docs/superpowers/specs/2026-06-19-multi-year-auction-ingest-design.md` (implements R1–R6).

## Global Constraints

- **Run from the repo root.** `preset.table_path` is cwd-relative (`data/vorp_{season}/…`) — the board and the #49a runner read tables there. `--data-root` is the **read-root** for the `external_projections` snapshot + id_map only; for the canonical workflow use the default `--data-root data` so external-read (`data/raw`) and table-writes (`data/vorp_{season}`) share `<repo>/data`. Do **not** pass an absolute `--data-root` outside `<repo>/data` for table generation.
- **`season=2026` must reproduce today's behavior byte-for-byte** (path `data/vorp_2026/…`, config name `..._2026`) — the board depends on it. Frozen-string tests guard this.
- **2026 is NOT re-ingested** (preserves the Run-A–H baseline `asof=2026-06-19`); it is only *regenerated* from the existing snapshot.
- Conventions: `Ruleset`/`RosterSlot` as enums; `pd.Int64Dtype` for `espn_auction_dollars`; `mkdir(parents=True, exist_ok=True)`; named-exception catches only (no bare `except`); mypy-strict + ruff clean.
- Verification (every task): run the fixers first — `ruff format src tests scripts` then `ruff check --fix src tests scripts` (settle formatting + import order) — then the gates must be clean: `mypy src tests`, `ruff check src tests scripts`, `ruff format --check src tests scripts`. Commits: prepend the venv scripts dir to PATH (`PATH="$(pwd)/.venv/Scripts:$PATH" git commit …`) to avoid the pre-commit mypy/venv quirk; end messages with the `Co-Authored-By:`/`Claude-Session:` trailers used on this branch.

---

## Task 1: Season-aware preset registry

**Files:**
- Modify: `src/projections/draft/assistant/presets.py`
- Test: `tests/test_draft/test_presets.py`

**Interfaces:**
- Produces: `get_preset(scoring_key: str, n_teams: int, season: int = 2026) -> DraftPreset` with `table_path = Path(f"data/vorp_{season}/{scoring_key}_{n_teams}team.parquet")` and `league_config.name = f"{scoring_key}_{n_teams}team_{season}"`. `_table_dir(season: int) -> Path`. (`DraftPreset` and `materialize_league_config` unchanged.)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_draft/test_presets.py`:

```python
def test_get_preset_season_param_drives_path_and_name() -> None:
    p = get_preset("half", 12, season=2023)
    assert p.table_path == Path("data/vorp_2023/half_12team.parquet")
    assert p.league_config.name == "half_12team_2023"
    std21 = get_preset("std", 16, season=2021)
    assert std21.table_path == Path("data/vorp_2021/std_16team.parquet")


def test_get_preset_defaults_to_2026_byte_for_byte() -> None:
    p = get_preset("half", 16)  # the board's 2-arg call
    assert p.table_path == Path("data/vorp_2026/half_16team.parquet")
    assert p.league_config.name == "half_16team_2026"
```

- [ ] **Step 2: Run the new tests; verify they fail**

Run: `pytest tests/test_draft/test_presets.py -k "season_param or byte_for_byte" -v`
Expected: `test_get_preset_season_param_drives_path_and_name` FAILS with `TypeError: get_preset() got an unexpected keyword argument 'season'`.

- [ ] **Step 3: Make the registry season-aware**

In `src/projections/draft/assistant/presets.py`, replace the module-level `_TABLE_DIR = Path("data/vorp_2026")` (line ~40) with a helper:

```python
def _table_dir(season: int) -> Path:
    """Per-season VORP-table directory, cwd-relative (the board + generator read/write here)."""
    return Path(f"data/vorp_{season}")
```

Add a `season` param to `_skill_config`:

```python
def _skill_config(scoring_key: str, n_teams: int, season: int) -> LeagueConfig:
    roster = {**_SKILL_STARTERS, RosterSlot.BENCH: _BENCH_BY_SIZE.get(n_teams, _DEFAULT_BENCH)}
    return LeagueConfig(
        name=f"{scoring_key}_{n_teams}team_{season}",
        n_teams=n_teams,
        roster_slots=roster,
        ruleset=_RULESETS[scoring_key],
    )
```

And thread `season` through `get_preset`:

```python
def get_preset(scoring_key: str, n_teams: int, season: int = 2026) -> DraftPreset:
    if scoring_key not in _RULESETS:
        raise ValueError(f"unknown scoring key {scoring_key!r}; expected one of {SCORING_KEYS}")
    if n_teams not in TEAM_SIZES:
        raise ValueError(f"unsupported n_teams {n_teams}; expected one of {TEAM_SIZES}")
    return DraftPreset(
        scoring_key=scoring_key,
        n_teams=n_teams,
        label=f"{_SCORING_LABELS[scoring_key]} / {n_teams}-team",
        league_config=_skill_config(scoring_key, n_teams, season),
        table_path=_table_dir(season) / f"{scoring_key}_{n_teams}team.parquet",
    )
```

- [ ] **Step 4: Run the preset suite; verify all pass**

Run: `pytest tests/test_draft/test_presets.py -v`
Expected: PASS — the 2 new tests AND all existing tests, including `test_get_preset_resolves_ruleset_size_roster_and_path` (which asserts `table_path == Path("data/vorp_2026/half_16team.parquet")`, still true via the default) and `test_materialize_league_config_writes_a_resumable_path`.

- [ ] **Step 5: Gates + commit**

Run: `mypy src tests && ruff check src tests scripts && ruff format --check src tests scripts`
Expected: clean.

```bash
git add src/projections/draft/assistant/presets.py tests/test_draft/test_presets.py
git commit -m "feat(presets): season param on get_preset (default 2026) for per-season VORP tables"
```

---

## Task 2: Season-aware preset generator (+ per-season config write)

**Files:**
- Modify: `scripts/generate_preset_vorp_tables.py`
- Test: `tests/test_scripts/test_generate_preset_vorp_tables.py`

**Interfaces:**
- Consumes: `presets.get_preset(s, n, season=…)`, `presets.materialize_league_config(preset)`, `presets._table_dir` (Task 1).
- Produces: `build_preset_table(external, scoring_key, n_teams, season: int = 2026)` (trailing default — existing 3-arg positional callers unbroken); `main(argv)` writes each `preset.table_path` parquet + `preset.table_path.parent`'s `{s}_{n}team.league.json` under `data/vorp_{season}/`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scripts/test_generate_preset_vorp_tables.py` (it already imports `sys`, `Path`, `pandas as pd`, and `_synthetic_external`). **First add `import pytest` to the file's top-level imports** (it's not there today, and the gate is mypy-strict so `monkeypatch` must be annotated):

```python
def test_build_preset_table_accepts_season_and_still_validates() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts"))
    from generate_preset_vorp_tables import build_preset_table

    table = build_preset_table(_synthetic_external(), "half", 12, season=2023)
    VorpTableSchema.validate(table)
    assert "espn_auction_dollars" in table.columns


def test_main_writes_per_season_tables_and_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts"))
    import generate_preset_vorp_tables as gp
    from projections.draft.assistant import presets

    # The generator reads the external snapshot via read_latest_partition then runs
    # ExternalProjectionSchema.validate; feed the synthetic frame directly (no on-disk partition)
    # and redirect the table dir to tmp_path. `_synthetic_external()` builds stat/draft-rank columns
    # as object dtype with pd.NA — ExternalProjectionSchema declares them float64 and cannot coerce
    # object+<NA>, so pd.to_numeric(errors="coerce") them to real nullable floats first (.astype
    # alone still raises on <NA>).
    external = _synthetic_external()
    external["season"] = 2023
    external["asof"] = "2023-01-01"
    for c in (*STAT_FIELDS, "espn_draft_rank"):
        external[c] = pd.to_numeric(external[c], errors="coerce")
    monkeypatch.setattr(gp, "read_latest_partition", lambda *a, **k: external)
    monkeypatch.setattr(presets, "_table_dir", lambda season: tmp_path / f"vorp_{season}")
    # The 120-player synthetic fixture fills 10/12-team pools but NOT 16-team (FLEX can't fill);
    # restrict main's grid to half/12-team so it builds a buildable preset — the per-season write
    # path (dir + .league.json) is what this test verifies, not the full 9-preset grid.
    monkeypatch.setattr(gp, "SCORING_KEYS", ("half",))
    monkeypatch.setattr(gp, "TEAM_SIZES", (12,))

    rc = gp.main(["--season", "2023", "--data-root", str(tmp_path)])
    assert rc == 0
    tbl = tmp_path / "vorp_2023" / "half_12team.parquet"
    cfg = tmp_path / "vorp_2023" / "half_12team.league.json"
    assert tbl.exists() and cfg.exists()
    assert "espn_auction_dollars" in pd.read_parquet(tbl).columns
    assert json.loads(cfg.read_text())["name"] == "half_12team_2023"  # config carries the season
```

(`STAT_FIELDS` is already imported at the top of this test file.)

- [ ] **Step 2: Run; verify they fail**

Run: `pytest tests/test_scripts/test_generate_preset_vorp_tables.py -k "accepts_season or per_season" -v`
Expected: FAIL — `test_build_preset_table_accepts_season_and_still_validates` fails with `TypeError` (`build_preset_table` rejects `season=`); `test_main_writes_per_season_tables_and_configs` fails because the current `main` calls `get_preset(s, n)` (default 2026) and `build_preset_table(external, s, n)` → tables land in `tmp/vorp_2026/` (not the asserted `tmp/vorp_2023/`) and no `.league.json` is written.

- [ ] **Step 3: Thread `season` + write the config in `generate_preset_vorp_tables.py`**

Add `materialize_league_config` to the presets import:

```python
from projections.draft.assistant.presets import (
    SCORING_KEYS,
    TEAM_SIZES,
    get_preset,
    materialize_league_config,
)
```

Change `build_preset_table`'s signature + its `get_preset` call:

```python
def build_preset_table(
    external: pd.DataFrame, scoring_key: str, n_teams: int, season: int = 2026
) -> pd.DataFrame:
    """One preset's VORP table ... (unchanged docstring body)."""
    preset = get_preset(scoring_key, n_teams, season=season)
    # ... rest of the body unchanged ...
```

Replace the `main` body's loop + the `out_dir` hardcode. The new `main`:

```python
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate the 9 preset VORP tables for a season.")
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    args = p.parse_args(argv)

    external = ExternalProjectionSchema.validate(
        read_latest_partition(args.data_root / "raw", "external_projections", season=args.season)
    )
    for scoring_key in SCORING_KEYS:
        for n_teams in TEAM_SIZES:
            preset = get_preset(scoring_key, n_teams, season=args.season)
            table = build_preset_table(external, scoring_key, n_teams, season=args.season)
            # Write to preset.table_path (cwd-relative data/vorp_{season}/) — where the board and
            # the #49a runner read. --data-root is the external read-root only (Global Constraints).
            preset.table_path.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(preset.table_path, index=False)
            cfg_path = materialize_league_config(preset)
            print(
                f"{preset.label} ({args.season}): {len(table)} players -> "
                f"{preset.table_path}; config -> {cfg_path}"
            )
    return 0
```

Also update the module docstring (line 1) — change "Generate the 9 preset VORP tables for 2026, each correctly re-scored." to "... for a given season ..." and drop the `data/vorp_2026/` literal in favor of `data/vorp_{season}/`.

- [ ] **Step 4: Run the generator suite; verify pass**

Run: `pytest tests/test_scripts/test_generate_preset_vorp_tables.py -v`
Expected: PASS — the 2 new tests AND all existing tests (the existing `build_preset_table(external, "ppr", 10)` positional calls still work via the season default; `resolve_espn_auction_dollars` tests untouched).

- [ ] **Step 5: Gates + commit**

Run: `mypy src tests && ruff check src tests scripts && ruff format --check src tests scripts`
Expected: clean.

```bash
git add scripts/generate_preset_vorp_tables.py tests/test_scripts/test_generate_preset_vorp_tables.py
git commit -m "feat(presets): generate per-season tables + .league.json (drop vorp_2026 hardcode)"
```

---

## Task 3: `refresh_external_seasons.py` wrapper

**Files:**
- Create: `scripts/refresh_external_seasons.py`
- Test: `tests/test_scripts/test_refresh_external_seasons.py`

**Interfaces:**
- Consumes: `refresh_external_projections(data_root, *, season)` (ingest), `generate_preset_vorp_tables.main(argv)` (Task 2).
- Produces: `run(seasons: list[int], data_root: Path) -> dict[int, str]` (per-season `"ok"`/`"failed: …"`), `main(argv) -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scripts/test_refresh_external_seasons.py`:

```python
"""Tests for scripts/refresh_external_seasons.py (loop + per-season failure isolation)."""

from __future__ import annotations

from pathlib import Path

import pytest

import generate_preset_vorp_tables
import refresh_external_seasons as rs  # scripts/ on sys.path via conftest

from projections.ingest.external_projections import ExternalProjectionError

# NOTE: import ExternalProjectionError + generate_preset_vorp_tables DIRECTLY (not via `rs.…`).
# mypy-strict's no_implicit_reexport flags reaching through `rs` for names it merely imported.


def test_loops_each_season_and_isolates_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    ingested: list[int] = []
    gen_calls: list[list[str]] = []

    def fake_ingest(data_root: Path, *, season: int, asof: object = None) -> None:
        if season == 2022:
            raise ExternalProjectionError("boom")
        ingested.append(season)

    def fake_gen(argv: list[str]) -> int:
        gen_calls.append(argv)
        return 0

    monkeypatch.setattr(rs, "refresh_external_projections", fake_ingest)
    monkeypatch.setattr(generate_preset_vorp_tables, "main", fake_gen)

    status = rs.run([2021, 2022, 2023], Path("data"))

    assert status == {2021: "ok", 2022: status[2022], 2023: "ok"}
    assert status[2022].startswith("failed")
    assert ingested == [2021, 2023]  # 2022 raised before reaching gen
    assert gen_calls == [["--season", "2021", "--data-root", "data"],
                         ["--season", "2023", "--data-root", "data"]]
```

- [ ] **Step 2: Run; verify it fails**

Run: `pytest tests/test_scripts/test_refresh_external_seasons.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'refresh_external_seasons'`.

- [ ] **Step 3: Write the wrapper**

Create `scripts/refresh_external_seasons.py`:

```python
"""Loop external_projections ingest + preset-table generation across seasons (TODO #49a inputs).

For each season: pull the ESPN+Sleeper snapshot, then regenerate that season's preset VORP tables
+ league configs under data/vorp_{season}/. 2026 is intentionally NOT in the default list — its
baseline snapshot (asof of the published Runs A-H) must be preserved; regenerate 2026 separately
via `generate_preset_vorp_tables.py --season 2026` (no re-ingest).

Usage (from the repo root; tables write cwd-relative — see the generator's Global Constraints):
    python scripts/refresh_external_seasons.py [--seasons 2021..2025] [--data-root data]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import generate_preset_vorp_tables  # sibling script (scripts/ on sys.path)

from projections.ingest.external_projections import (
    ExternalProjectionError,
    refresh_external_projections,
)

_log = logging.getLogger(__name__)
_DEFAULT_SEASONS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)


def run(seasons: list[int], data_root: Path) -> dict[int, str]:
    """Ingest + regenerate per season; isolate per-season failures (one flaky API season must not
    discard the rest). Returns {season: "ok" | "failed: <reason>"}."""
    status: dict[int, str] = {}
    for year in seasons:
        try:
            refresh_external_projections(data_root, season=year)
            generate_preset_vorp_tables.main(["--season", str(year), "--data-root", str(data_root)])
            status[year] = "ok"
        except (ExternalProjectionError, OSError) as exc:
            _log.warning("season %s failed: %s", year, exc)
            status[year] = f"failed: {exc}"
    return status


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Refresh external_projections + preset tables across seasons (TODO #49a)."
    )
    p.add_argument("--seasons", type=int, nargs="+", default=list(_DEFAULT_SEASONS))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    args = p.parse_args(argv)
    status = run(args.seasons, args.data_root)
    for year, st in status.items():
        print(f"  {year}: {st}")
    return 0 if all(v == "ok" for v in status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run; verify pass**

Run: `pytest tests/test_scripts/test_refresh_external_seasons.py -v`
Expected: PASS.

- [ ] **Step 5: Gates + commit**

Run: `mypy src tests && ruff check src tests scripts && ruff format --check src tests scripts`
Expected: clean.

```bash
git add scripts/refresh_external_seasons.py tests/test_scripts/test_refresh_external_seasons.py
git commit -m "feat(ingest): refresh_external_seasons wrapper (loop ingest+gen, isolate failures)"
```

---

## Task 4: Data runs + verification (ops)

No code. Live network pulls + the R6 gate. Run from the **repo root** with the default `--data-root data`.

- [ ] **Step 1: Ingest + generate 2021–2025**

```bash
python scripts/refresh_external_seasons.py
```
Expected: per-season `ok` lines; writes `data/raw/external_projections/season={2021..2025}/...` and `data/vorp_{2021..2025}/{scoring}_{n}team.{parquet,league.json}`. If a season prints `failed`, re-run that season alone (`--seasons YYYY`) and capture the error.

- [ ] **Step 2: Regenerate 2026 (no re-ingest — preserves the Run-H baseline)**

```bash
python scripts/generate_preset_vorp_tables.py --season 2026
```
Expected: regenerates `data/vorp_2026/*.parquet` (byte-identical content from the existing `asof=2026-06-19` snapshot) + writes the full `.league.json` set.

- [ ] **Step 3: R6 verification gate**

```bash
python - <<'PY'
import pandas as pd
from projections.draft.auction import has_usable_espn_prices
for y in (2021, 2022, 2023, 2024, 2025, 2026):
    d = pd.read_parquet(f"data/vorp_{y}/half_12team.parquet")
    priced = int(d["espn_auction_dollars"].notna().sum()) if "espn_auction_dollars" in d.columns else 0
    ok = has_usable_espn_prices(d)
    flag = "" if priced >= 100 else "  <-- LOW COVERAGE"
    assert ok, f"season {y}: NO usable espn_auction_dollars (R6 FAILURE)"
    print(f"{y}: usable={ok} priced={priced}{flag}")
print("R6 gate PASSED")
PY
```
Expected: `usable=True` for every season; `R6 gate PASSED`. A `priced < 100` season (e.g. 2025 expert-only ~160 is fine; crowd-only years ~220-285) prints a LOW-COVERAGE flag but does not fail.

- [ ] **Step 4: Record coverage + commit docs**

Add a `project_management.md` entry and a TODO #49a note recording: the per-season priced-player counts from Step 3, that 2021–2026 preset tables + configs now exist under `data/vorp_{season}/`, and the auction-value-basis caveat (pre-2023 crowd-only, 2025 expert-only). The `data/` artifacts stay untracked.

```bash
git add project_management.md TODO.md
git commit -m "docs(ingest): record multi-year auction inputs (2021-2026 preset tables) for #49a"
```

---

## Self-Review

- **Spec coverage:** R1 (get_preset season param + frozen 2026) → Task 1. R2 (build_preset_table default, write to preset.table_path + materialize, drop out_dir, docstrings) → Task 2. R3 (refresh wrapper, named-exception isolation) → Task 3. R4 (no board regression / 2026 default) → Task 1 Step 1's frozen test + existing suites. R5 (conventions) → Global Constraints + each task's gates. R6 (per-season tables+configs + hard gate) → Task 4 Steps 1–3.
- **Placeholder scan:** every step has concrete code/commands; the only non-code step (Task 4 Step 4 docs) names exactly what to record.
- **Type consistency:** `get_preset(s, n, season=2026)`, `build_preset_table(external, s, n, season=2026)`, `_table_dir(season)`, `materialize_league_config(preset)`, `run(seasons, data_root) -> dict[int, str]` are consistent across tasks; `read_latest_partition` is monkeypatched on the `gp` module (where it's imported) in Task 2's test.
- Note vs spec §B: the spec threads `season` into `build_preset_table`; the plan keeps it but flags that the VORP *content* is season-independent — the season only drives the config name + write path (handled in `main`). Threading it keeps the call self-consistent and matches the spec; it is not a no-op risk (default preserves 3-arg callers, verified in Task 2 Step 4).
