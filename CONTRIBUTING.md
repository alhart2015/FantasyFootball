# Contributing

Setup, daily commands, workflow, and pattern recipes for the FantasyFootball repo. For Claude-specific behavioral rules, see `CLAUDE.md`. For project status and the running decision log, see `project_management.md`.

## Setup

1. Clone the repo and `cd` into it.
2. Create a virtualenv:
   - **Windows (bash):** `python -m venv .venv && . .venv/Scripts/activate`
   - **Windows (PowerShell):** `python -m venv .venv; .venv\Scripts\Activate.ps1`
   - **macOS/Linux:** `python -m venv .venv && source .venv/bin/activate`
3. Install the project + dev deps in editable mode:
   ```bash
   pip install -e ".[dev]"
   ```
4. Install pre-commit hooks (one-time per clone or worktree):
   ```bash
   pre-commit install
   ```
5. Confirm everything works:
   ```bash
   pytest -v && mypy src tests && ruff check src tests
   ```

### Windows-specific notes

- `.worktrees/` is gitignored. We use it for isolated feature work — see "Workflow" below.
- Each worktree needs its own `.venv` activation per shell.
- After installing `gh` via winget, the binary may not be on the current shell's PATH. Either restart the shell or invoke at `"/c/Program Files/GitHub CLI/gh.exe"`.

## Daily commands

```bash
pytest -v                                                # Run all tests
pytest tests/test_schemas/ -v                            # One test directory
pytest tests/test_scoring/test_score.py -v               # One test file
pytest tests/test_scoring/test_score.py::test_jefferson_real_line -v  # Single test
pytest -v -k "ingest or store"                           # Tests matching a keyword

mypy src tests                                           # Type check (strict)
ruff check src tests                                     # Lint
ruff check --fix src tests                               # Lint + autofix
ruff format src tests                                    # Format
ruff format --check src tests                            # Format dry-run

pre-commit run --all-files                               # Run all hooks against all files
pre-commit autoupdate                                    # Refresh hook versions
```

Pre-commit runs lint + format + typecheck on every commit. You don't need to remember to run those manually before committing.

The Python public API is `from projections import ...`. CLI verbs (`python -m projections refresh|project|backtest|query`) are coming in Plan 4 and aren't built yet.

## Workflow

This project uses the superpowers spec → plan → execute discipline.

### Critical rule: no direct commits to `main`

**Specs, plans, and implementation all live on the feature branch and reach `main` only via PR.** Never commit a spec or plan directly to `main`. Set up the worktree and feature branch first, then write the spec on the branch.

### The full flow

1. **Brainstorm with the user** before any creative work. Use `superpowers:brainstorming`. Out: a clear understanding of what to build and why.
2. **Set up the feature branch and worktree** before writing the spec:
   ```bash
   git branch feat/<short-name>
   git worktree add .worktrees/feat-<short-name> feat/<short-name>
   cd .worktrees/feat-<short-name>
   ```
3. **Write the spec** at `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`. Commit on the feature branch.
4. **Get user approval** on the spec.
5. **Write the plan** at `docs/superpowers/plans/YYYY-MM-DD-<topic>.md`. Plans are TDD-style with bite-sized tasks (2-5 minutes each), real code in every step, ending in a commit. Commit on the feature branch.
6. **Execute** via `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. For tasks with real logic, run two-stage review (spec compliance then code quality).
7. **Open a PR** when implementation is done. The PR contains spec + plan + implementation as one reviewable unit.
8. **Merge** via `gh pr merge` (or the GitHub UI). Pull `main` and clean up the worktree.

### Branch and commit conventions

- Branch names: `feat/<short-name>` (e.g., `feat/dev-tooling`, `feat/projections-foundations`).
- Commit messages: conventional commits.
  - `feat(scoring): add half-PPR ruleset preset`
  - `fix(ingest): drop rows with null gsis_id`
  - `chore: bump pandera floor to 0.20`
  - `refactor(distributions): extract validate_q helper`
  - `docs(readme): expand from placeholder to short orientation`
  - `style: apply ruff format to existing codebase`
  - `test(store): cover idempotency edge cases`
- Keep commits small. One logical change per commit.

## PR checklist

Before opening a PR:

- [ ] **Run `pytest -v` and confirm all tests pass.** This is the only manual gate; we deliberately don't run GitHub Actions CI. mypy and ruff are caught by pre-commit.
- [ ] If the work is foundational (a completed plan, an architectural decision, a convention change), update `project_management.md` (decision log + next-action) and/or `TODO.md`.
- [ ] If a new convention emerged, capture it in `CONTRIBUTING.md` (this file) so it's repo-resident, not memory-only.
- [ ] Confirm the spec and plan are committed on the feature branch.

Open the PR with `gh`:
```bash
gh pr create --title "<conventional-commit-style title>" --body "$(cat <<'EOF'
## Summary
<2-3 bullets of what changed>

## Test Plan
- [x] pytest -v: <N> passed
- [x] mypy src tests: clean
- [x] ruff check src tests: clean
- [ ] Reviewer: confirm <specific verification step>
EOF
)"
```

After merge, sync `main` and clean up:
```bash
git checkout main && git pull
git worktree remove .worktrees/feat-<short-name>
git branch -d feat/<short-name>
```

If `git worktree remove` fails on Windows due to a venv file lock, run `git worktree remove --force` and clean up the leftover directory by hand later.

## Adding a new ingest source

The pattern is established in `src/projections/ingest/weekly_stats.py`. Follow it.

### Skeleton

```python
# src/projections/ingest/<your_source>.py
"""Refresh <source> from `nfl_data_py.import_<source>`."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

from projections.ingest.manifest import record as record_manifest
from projections.schemas import Position, YourSourceSchema, normalize_team_code
from projections.store import write_partition

_PYARROW_STR = pd.StringDtype("pyarrow")

_KEEP = ["gsis_id", "season", "week", "position", "team", ...]
_RENAME = {"player_id": "gsis_id", ...}


def _fetch_raw_<source>(seasons: list[int]) -> pd.DataFrame:
    """Thin wrapper around the nfl_data_py call; tests monkey-patch this."""
    return nfl.import_<source>(seasons)


def _normalize_one_season(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=_RENAME).copy()

    # Coerce dtypes that nfl_data_py sometimes returns as floats.
    for int_col in ("...", ):
        if int_col in df.columns:
            df[int_col] = df[int_col].fillna(0).astype(int)

    # Normalize team codes (handles JAX→JAC, LA→LAR, historical STL/SD/OAK/WSH).
    df["team"] = df["team"].map(lambda v: normalize_team_code(v).value).astype(_PYARROW_STR)

    # Filter rows at unsupported positions BEFORE schema validation.
    df = df[df["position"].isin([p.value for p in Position])].copy()

    # Cast string columns to pyarrow strings (pandera 0.31 requires this for Series[str]).
    df["gsis_id"] = df["gsis_id"].astype(_PYARROW_STR)
    df["position"] = df["position"].astype(_PYARROW_STR)

    # IMPORTANT: reassign — strict="filter" returns a new DataFrame; without
    # reassignment any extras silently slip through.
    df = YourSourceSchema.validate(df)
    return df


def refresh_<source>(data_root: Path, *, seasons: Iterable[int]) -> list[Path]:
    """Fetch and write per-season data. One partition per season. Idempotent."""
    written: list[Path] = []
    for season in seasons:
        raw = _fetch_raw_<source>([season])
        df = _normalize_one_season(raw)
        path = write_partition(data_root / "raw", "<source>", df, season=season, week=None)
        record_manifest(data_root, table="<source>", season=season, df=df)
        written.append(path)
    return written
```

### Test pattern

In `tests/test_ingest/conftest.py`, add a `fake_<source>_df` fixture mimicking the `nfl_data_py` output shape. In a new `tests/test_ingest/test_<source>.py`:

```python
def test_refresh_<source>_writes_partitioned_parquet(
    tmp_path: Path,
    fake_<source>_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projections.ingest.<source>._fetch_raw_<source>",
        lambda seasons: fake_<source>_df,
    )
    refresh_<source>(tmp_path, seasons=[2024])
    df = read_partition(tmp_path / "raw", "<source>", season=2024)
    YourSourceSchema.validate(df)
```

Always test idempotency (run twice, assert row count doesn't double) and team-code normalization (feed `JAX`/`LA` aliases, assert canonical codes in output).

Don't hit the network in tests. The `_fetch_raw_*` indirection exists exactly so monkeypatching is trivial.

### After bumping `nfl_data_py`

CI's synthetic fixtures don't exercise the live API surface, so a `nfl_data_py` version bump can pass `pytest -v` and still break a real refresh. We catch upstream column-rename / column-removal drift with opt-in smokes in `tests/test_ingest/test_api_drift.py`, marked `@pytest.mark.network` and skipped by default.

Run them after any `nfl_data_py` version change in `pyproject.toml`:

```bash
pytest -m network --run-network -q
```

Each smoke fetches a tiny live slice (one season — currently 2023) for one ingest source, asserts every raw column the corresponding `_normalize_one_season` reads is present, and runs the normalize end-to-end so pandera surfaces dtype / value drift too. When a smoke fails, the assertion message names the missing column(s); patch the corresponding ingest module's `_RENAME` / `_KEEP` / schema and re-run. If the drift was non-trivial, log a TODO note alongside the existing TODO #16 entries.

When you add a new ingest source, add a matching smoke test alongside the synthetic-fixture test — the pattern is one `test_<source>_api_columns_and_schema` function per source.

### After touching `src/projections/features/`

The feature cache at `data/features/{position}/season=YYYY/week=WW/part.parquet`
is **not** auto-invalidated when feature builders change. After modifying
any file under `src/projections/features/` you must rebuild the cache
before running the backtest gate:

```bash
python scripts/refresh_features.py all --seasons 2018-2024
```

Then re-snapshot if your change intentionally alters Model A's metrics:

```bash
python scripts/backtest.py --update-snapshot
git diff tests/backtest/model_metrics.json    # review the metric deltas
git add tests/backtest/model_metrics.json
```

Auto-invalidation is TODO #21 (see TODO.md).

### Running the backtest gate before opening a PR

The gate is opt-in:

```bash
pytest -m backtest --run-backtest
```

Full 16-cell run takes about 2 minutes. If the gate fails, the failure
message lists each regressing (position, year, metric) cell with
baseline vs current values.

If the regression is intentional (e.g., a feature change that
legitimately improves the model on some cells but slightly worsens
others within tolerance overrides), update the snapshot and commit:

```bash
python scripts/backtest.py --update-snapshot
git add tests/backtest/model_metrics.json
```

For genuinely-noisy cells (rare-event RMSE on small samples, etc.),
add a per-row override to `tests/backtest/tolerances.json` instead of
loosening the default. Each override row needs a `rationale` field
describing why the cell is noisy.

A default-on smoke test (`tests/backtest/test_backtest_smoke.py`)
runs one (WR, 2024) cell as part of `pytest -v`, taking ~15 seconds.
It catches harness wiring bugs without requiring the full opt-in
gate; auto-skips on fresh checkouts where the feature cache is empty.

## Feature plan workflow

Before scoping a new feature plan touching per-position feature columns, run the feature signal probe. The probe takes ~30s–2min on a single candidate column (Ridge baseline) and predicts whether the proposed feature would clear the adoption gate.

1. Produce the candidate column(s) — typically a one-off script reading from `data/raw/...` partitions and emitting an override parquet at `data/features_probe/<name>.parquet` with columns `gsis_id`, `season`, `week`, plus the candidate column(s).
2. Run the probe (augment-not-swap mode is the safer default):
   ```bash
   python -m scripts.probe_feature_signal \
     --candidate-name "<descriptive_name>" \
     --override data/features_probe/<name>.parquet \
     --csv-out reports/feature_probe_<name>.csv
   ```
3. Inspect the markdown report on stdout (and `reports/feature_probe_<name>.csv` for downstream analysis).
4. If the probe returns no pooled SIGNAL across all 4 positions: do not scope the plan. Decompose the candidate (bundle with other candidates, change model class, or shelve).
5. If the probe returns pooled SIGNAL on at least one (position, stat) cell: Phase 2 fires automatically and predicts the adoption gate verdict. Proceed to spec → plan → execute as normal IF Phase 2 returns ADOPT for at least one position.

The probe is a screening tool, not a substitute for the adoption gate. SIGNAL is necessary but not sufficient — the full backtest + adoption gate is the final word on shipping. See `docs/superpowers/specs/2026-04-30-feature-signal-probe-design.md` for full design.

**Tunable thresholds:**
- `--coverage-threshold 0.80` for overrides with structural NaN patterns (e.g., bye-week trailing-window features). Default 0.95.
- `--effect-size-floor 0.10` for noisier domains where 0.05 fpts effects are too small to act on. Default 0.05; Plan 8's measured per-cell noise floor was ~0.08 fpts.

**Phase 2 gating:**
- Default: Phase 2 runs iff Phase 1 finds a pooled SIGNAL.
- `--no-composite`: skip Phase 2 even on a Phase-1 SIGNAL.
- `--force-composite`: run Phase 2 unconditionally. Use when `--model` is not the Ridge regressor used in Phase 1 — Phase 2's production-model fit may detect signal Phase 1's Ridge screen missed (e.g., trees on a feature Ridge can't use). Adds ~3–10 min per position depending on model class. **Required whenever `--model` is non-default and you want lgb-nb (or any non-Ridge) to actually run.** Without this flag, Phase 1 (always RidgeCV regardless of `--model`) gates Phase 2; if Phase 1 returns no pooled SIGNAL, Phase 2 never fires and the `--model` setting has no effect — the run is tautological with the baseline-Ridge probe. This is the trap that consumed PR #22's WR/TE receiver-family probe before the lgb-nb runs were redone with `--force-composite`. Conditional-lgb-nb specs (per the family-probe pattern) should always include `--force-composite` on the lgb-nb runs.

### Regenerating the PBP family override

The PBP family probe (spec `docs/superpowers/specs/2026-04-30-pbp-feature-family-probe-design.md`) consumes a four-column override parquet at `data/features_probe/pbp_family.parquet`. The output is regenerable from the live PBP partitions and is not committed. To regenerate:

```bash
python -m scripts.build_pbp_family_override --seasons 2018-2024
```

Pass `--force` to overwrite an existing parquet. The script needs `data/raw/pbp/`, `data/raw/depth_charts/`, and `data/raw/schedules/` populated (run the standard ingest refresh first if not). The player-team-week index is built from `depth_charts` (every rostered player per team-week) joined with `schedules` for opponents, matching the per-position feature builders' coverage.

### Regenerating the PBP red-zone override

Sibling to "Regenerating the PBP family override" above. The red-zone
family override at `data/features_probe/pbp_redzone.parquet` is not
committed; regenerate when the spec needs it:

```bash
python -m scripts.build_pbp_redzone_override --seasons 2018-2024
```

Output: ~one row per (gsis_id, season, week) for every fantasy-position
rostered player who had a scheduled game that week. Used as input to
`scripts/probe_feature_signal.py --override
data/features_probe/pbp_redzone.parquet ...`. Spec:
`docs/superpowers/specs/2026-05-02-pbp-redzone-feature-family-probe-design.md`.

### Regenerating the PBP pressure override

Sibling to "Regenerating the PBP family override" and "Regenerating the
PBP red-zone override" above. The pressure family override at
`data/features_probe/pbp_pressure.parquet` is not committed; regenerate
when the spec needs it:

```bash
python -m scripts.build_pbp_pressure_override --seasons 2018-2024
```

Output: ~one row per (gsis_id, season, week) for every fantasy-position
rostered player who had a scheduled game that week. Used as input to
`scripts/probe_feature_signal.py --override
data/features_probe/pbp_pressure.parquet ...`. Spec:
`docs/superpowers/specs/2026-05-02-pbp-pressure-feature-family-probe-design.md`.

### Regenerating the PBP receiver override

The receiver-level override parquet (`data/features_probe/pbp_receiver.parquet`)
is regenerable from the live `data/raw/pbp` and `data/raw/depth_charts`
partitions. It is NOT committed.

```bash
python -m scripts.build_pbp_receiver_override --seasons 2018-2024
```

To overwrite an existing file, pass `--force`. The script reads PBP for
`[seasons.start - 1, seasons.stop)` (one prior season for trailing-4 backfill
at week 1–4 of each season) and writes one row per `(gsis_id, season, week)`
for every WR / TE rostered per `depth_charts`.

## Adding a new pandera schema

Schemas live in `src/projections/schemas.py` (single source of truth). Append your schema after the existing ones.

### Conventions

- Inherit from `pa.DataFrameModel`. Import as `import pandera.pandas as pa` (NOT `import pandera as pa` — pandera 0.20+ moved pandas-specific classes into the `pandera.pandas` submodule).
- Annotate columns with `Series[type]` from `pandera.typing`. Constrain values with `pa.Field(...)`:
  - `isin=[...]` — categorical / enum-backed columns. Compute the list once at module scope (e.g., `_POSITION_VALUES = [p.value for p in Position]`).
  - `ge=...` / `le=...` — numeric ranges.
  - `str_matches=r"..."` — regex on string columns. For canonical IDs, reference the existing `GSIS_ID_PATTERN`.
  - `unique=True` — enforce uniqueness.
  - `nullable=True` — allow NA in the column.
- Add `class Config: strict = "filter"` to drop unexpected columns. **Callers must reassign the result** (`df = SCHEMA.validate(df)`) for the filter to actually take effect.

### Dtype rules

- `Series[str]` requires `pd.StringDtype("pyarrow")` upstream. Object dtype + plain strings will fail validation under pandera 0.31. Cast in your normalize step: `df["col"] = df["col"].astype(pd.StringDtype("pyarrow"))`.
- Mixed `int` + `pd.NA` falls back to `float64` (`2024` becomes `2024.0`). Use `pd.Int64Dtype()` (nullable int) for any integer column that may contain NAs.
- Timezone-aware timestamps in pandas 2.x default to microsecond resolution: use `dtype_kwargs={"unit": "us", "tz": "UTC"}`, not `"unit": "ns"`.
- For `bytes` columns (e.g., serialized distribution params), use `Series[bytes]`. It maps to `object` dtype at runtime.

### Skeleton

```python
class YourSchema(pa.DataFrameModel):
    """One-line description of what this DataFrame represents."""

    gsis_id: Series[str] = pa.Field(str_matches=rf"^{GSIS_ID_PATTERN}$")
    season: Series[int] = pa.Field(ge=1999, le=2100)
    week: Series[int] = pa.Field(ge=1, le=22)
    position: Series[str] = pa.Field(isin=_POSITION_VALUES)
    your_metric: Series[float] = pa.Field(ge=0)

    class Config:
        strict = "filter"
```

Add tests in `tests/test_schemas/test_dataframe_schemas.py` (or a sibling file): one for the happy path, one per constraint that should reject (out-of-range value, bad regex, missing column).
