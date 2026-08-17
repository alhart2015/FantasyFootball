# Pick'em Hub — Implementation Plan

Design: `docs/superpowers/specs/2026-08-16-pickem-hub-design.md`

TDD throughout: write the failing test, then the code, then run the gate. Each task ends in a
commit. Gate for every task: `pytest -v -k <scope>`, `mypy src tests`, `ruff check src tests`,
`ruff format --check src tests`.

## Task 1 — Schedules ingest: scores and game type

Extend `SchedulesSchema` with `home_score` (`Int64`, nullable), `away_score` (`Int64`,
nullable), `game_type` (`str`, nullable). Add all three to `_KEEP` in
`src/projections/ingest/schedules.py` and cast in `_normalize_one_season` (Int64 for scores,
`_PYARROW_STR` for `game_type`).

Extend the `fake_schedules_df` fixture in `tests/conftest.py` with the three raw columns.

Tests (`tests/test_ingest/test_schedules.py`, `tests/test_schemas/`):
- Scores survive the round-trip through `write_partition`/`read_partition`.
- A future game with null scores validates.
- `game_type` is preserved.

Do **not** add `result` — it is exactly `home_score - away_score`.

Commit: `feat(ingest): keep home/away score and game_type on schedules`

## Task 2 — Pick'em schemas

Append to `src/projections/schemas.py`: `PickemSheetSchema`, `PickemSlateSchema`,
`PickemPicksSchema` exactly as specified in the design.

Tests (`tests/test_schemas/`): happy path per schema, plus one rejection per constraint
(unknown team code, probability outside [0, 1], negative `switch_cost`).

Commit: `feat(schemas): add pick'em sheet, slate, and picks schemas`

## Task 3 — Probability

New `src/projections/pickem/__init__.py` and `probability.py`:

```python
def american_to_implied(odds: int) -> float
def devig_pair(home_odds: int, away_odds: int) -> tuple[float, float]
def add_win_probs(schedules: pd.DataFrame) -> pd.DataFrame   # + home_win_prob / away_win_prob
```

Tests: `-110/-110` → 0.5/0.5; `-148/+124` → home favored, pair sums to exactly 1.0; favorite
always holds the larger share; a null moneyline raises `ValueError` naming the game.

Commit: `feat(pickem): devigged win probabilities from moneylines`

## Task 4 — Sheet I/O

`src/projections/pickem/sheet.py`:

```python
def read_sheet(path: Path, *, season: int, week: int) -> pd.DataFrame
def write_template(path: Path, schedules: pd.DataFrame, *, season: int, week: int) -> Path
```

`read_sheet` normalizes team codes via `normalize_team_code`, validates against
`PickemSheetSchema`. `write_template` emits `away_team,home_team,home_spread` with a blank
spread column, one row per real game that week, ordered by kickoff.

Tests: `JAC` normalizes to `JAX`; unknown team raises; blank/missing `home_spread` raises with
a message naming the row; template row count matches the schedule.

Commit: `feat(pickem): read the organizer sheet and emit a pre-filled template`

## Task 5 — Slate

`src/projections/pickem/slate.py`:

```python
def build_slate(sheet: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame
```

Join on `(season, week, home_team, away_team)`. Compute `consensus_home_spread = -spread_line`,
attach win probabilities, label `sheet_favorite`/`sheet_dog` from `sheet_home_spread`
(negative → home favored; 0 → both NA), then `dog_win_prob`, `dog_line_move`, `free_dog`.

Tests, each asserting the sign convention explicitly:
- Home favored on the sheet → `sheet_dog` is the away team.
- Away favored on the sheet → `sheet_dog` is the home team.
- `home_spread == 0` → both NA, `free_dog` False.
- Sheet has dog at +7, consensus at +3 → `dog_line_move == +4`.
- Sheet dog that the market now favors → `free_dog` True and `dog_win_prob > 0.5`.
- A sheet row with no matching schedule row raises, naming the matchup.

Commit: `feat(pickem): join the organizer sheet to consensus lines`

## Task 6 — Optimizer

`src/projections/pickem/optimize.py`:

```python
def choose_picks(slate: pd.DataFrame, *, min_dogs: int = 3) -> pd.DataFrame
```

Greedy per the design. Deterministic tie-break on `game_id`. Raises when eligible dog games
are fewer than `min_dogs`.

Tests:
- All favorites naturally → exactly 3 forced swaps, and they are the cheapest three.
- Already 4 natural dogs → zero forced swaps.
- Free dogs are picked and **not** marked `forced`.
- A `home_spread == 0` game is never used to satisfy the constraint.
- Too few eligible dog games raises.
- **Brute force:** on an 8-game random slate, enumerate all 2^8 pick combinations, keep the
  feasible ones, and assert the greedy result matches the true maximum. Run over several
  seeded slates.

Commit: `feat(pickem): constrained pick optimizer with provable greedy selection`

## Task 7 — Grading

`src/projections/pickem/grade.py`:

```python
def grade_picks(picks: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame
```

Tests: correct pick → True; wrong pick → False; tie → `winner` NA and `correct` False;
unplayed → `correct` NA. Assert `correct` uses `pd.BooleanDtype()` so NA survives.

Commit: `feat(pickem): grade picks against final scores`

## Task 8 — Store

`src/projections/pickem/store.py` — `write_sheet` / `read_sheet_partition` / `write_picks` /
`read_picks`, wrapping `store.write_partition` / `read_partition` against `data/pickem`.

Tests: round-trip both tables; re-writing a week overwrites rather than appends.

Commit: `feat(pickem): persist sheets and picks to the store`

## Task 9 — Board script

`scripts/pickem_board.py` with `--season`, `--week`, `--template PATH`, `--sheet PATH`,
`--min-dogs`, `--data-root`. Prints the slate, the chosen picks, expected correct total, and a
callout section for free dogs and line moves over a threshold. Logic lives in the package; the
script is a thin view.

Tests in `tests/test_scripts/`: template mode writes the expected CSV; picks mode produces the
expected pick set from a fixture slate.

Commit: `feat(pickem): pickem_board CLI`

## Task 10 — Backtest script

`scripts/pickem_backtest.py` — market calibration table (predicted vs. actual by probability
bin) and the expected-score baseline (optimizer run over past seasons with closing lines as
both sheet and consensus).

Tests: calibration binning is correct on synthetic data with a known answer; the baseline
runner handles a multi-week frame.

Commit: `feat(pickem): calibration and baseline backtest`

## Task 11 — Wire-up and docs

- Export the public API from `src/projections/pickem/__init__.py`.
- `CONTRIBUTING.md`: a short "Pick'em weekly workflow" recipe (template → fill → picks → grade).
- `project_management.md`: status entry and decision-log rows.
- Open the PR.

Commit: `docs(pickem): weekly workflow recipe and status update`
