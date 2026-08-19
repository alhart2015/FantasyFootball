"""Read the organizer's weekly sheet, and emit a template to fill in.

The organizer distributes a Google Sheet on Tuesday; picks are due before
Thursday kickoff. There is no API access to it, so the hand-off is a small CSV:

    away_team,home_team,home_spread
    NE,SEA,-3.5
    CHI,CAR,2.5

`home_spread` is in the **standard betting convention** — negative means the
home team is favored — because that is how a human reads a line off an email.
Note this is the negation of nflreadpy's `spread_line`; the conversion lives in
`pickem.slate` and nowhere else.

`write_template` exists to remove the most likely error in the whole pipeline:
a mistyped team code silently failing to join, or worse, joining to the wrong
game. Emitting the real matchups means only the numbers are typed by hand.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.schemas import (
    _PYARROW_STR,
    PickemSheetSchema,
    normalize_team_code,
)

SHEET_COLUMNS = ("away_team", "home_team", "home_spread")


def _normalize_team_column(values: pd.Series, *, column: str, path: Path) -> pd.Series:
    normalized: list[str] = []
    for row_number, raw in enumerate(values, start=2):  # +2: 1-indexed, past the header
        if pd.isna(raw) or str(raw).strip() == "":
            raise ValueError(f"{path}: row {row_number} has an empty {column}")
        try:
            normalized.append(normalize_team_code(str(raw).strip()).value)
        except ValueError as exc:
            raise ValueError(f"{path}: row {row_number} has an unusable {column} — {exc}") from exc
    return pd.Series(normalized, dtype=_PYARROW_STR, index=values.index)


def read_sheet(path: Path, *, season: int, week: int) -> pd.DataFrame:
    """Read and validate the organizer's sheet CSV into a `PickemSheetSchema` frame.

    Team codes are normalized, so the organizer writing `JAC` or `WSH` is fine.
    Season and week are supplied by the caller rather than read from the file —
    one less column to mistype, and the sheet is always for a known week.
    """
    raw = pd.read_csv(path)

    missing = [c for c in SHEET_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"{path}: missing column(s) {missing}. Expected header: {','.join(SHEET_COLUMNS)}"
        )
    if raw.empty:
        raise ValueError(f"{path}: sheet has no rows")

    blank = raw["home_spread"].isna()
    if bool(blank.any()):
        rows = [int(i) + 2 for i in raw.index[blank]]
        raise ValueError(
            f"{path}: home_spread is blank on row(s) {rows}. "
            "Every game needs the organizer's spread — that is what decides which "
            "team counts as the underdog."
        )

    spreads = pd.to_numeric(raw["home_spread"], errors="coerce")
    unparsable = spreads.isna()
    if bool(unparsable.any()):
        rows = [int(i) + 2 for i in raw.index[unparsable]]
        raise ValueError(f"{path}: home_spread is not a number on row(s) {rows}")

    out = pd.DataFrame(
        {
            "season": season,
            "week": week,
            "home_team": _normalize_team_column(raw["home_team"], column="home_team", path=path),
            "away_team": _normalize_team_column(raw["away_team"], column="away_team", path=path),
            "home_spread": spreads.astype(float),
        }
    )

    same = out["home_team"] == out["away_team"]
    if bool(same.any()):
        rows = [int(i) + 2 for i in out.index[same]]
        raise ValueError(f"{path}: row(s) {rows} list the same team as home and away")

    duplicated = out.duplicated(subset=["home_team", "away_team"], keep=False)
    if bool(duplicated.any()):
        rows = [int(i) + 2 for i in out.index[duplicated]]
        raise ValueError(f"{path}: duplicate matchup on row(s) {rows}")

    validated: pd.DataFrame = PickemSheetSchema.validate(out)
    return validated


def write_template(path: Path, schedules: pd.DataFrame, *, season: int, week: int) -> Path:
    """Write a CSV of the week's real matchups with a blank `home_spread` column.

    Ordered by kickoff so the file reads in the same order as the organizer's
    sheet. Games with no confirmed kickoff (flex scheduling) sort last, by
    `game_id`, so the output is deterministic either way.
    """
    week_games = schedules[(schedules["season"] == season) & (schedules["week"] == week)]
    if week_games.empty:
        raise ValueError(f"no scheduled games for season={season} week={week}")

    ordered = week_games.sort_values(
        ["kickoff", "game_id"], na_position="last", kind="stable"
    ).reset_index(drop=True)

    template = pd.DataFrame(
        {
            "away_team": ordered["away_team"].astype(str),
            "home_team": ordered["home_team"].astype(str),
            "home_spread": pd.Series([pd.NA] * len(ordered), dtype=_PYARROW_STR),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(path, index=False)
    return path
