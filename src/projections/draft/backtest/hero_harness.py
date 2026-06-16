"""Hero-vs-bots strategy evaluation.

Runs each strategy as the SOLE hero (one seat) vs a noisy-ADP bot field, scored on the
real-outcome H2H season, swept across all seats with common random numbers across
strategies. The deployment-realistic counterpart to the mixed-field harness (harness.py).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from projections.draft.assistant.availability import PlayerAvailability
from projections.draft.assistant.strategy import _DEFAULT_FLOOR, _DEFAULT_FLOOR_WEIGHT
from projections.draft.backtest.checkpoint import dump_results, load_results
from projections.draft.backtest.draft_field import hero_seat_layout
from projections.draft.backtest.harness import _build_strategy
from projections.draft.backtest.league import Calendar, LeagueResult, simulate_league
from projections.draft.league_config import LeagueConfig
from projections.schemas import HeroResultSchema

_MC_KEYS = frozenset({"season_value", "season_value_var", "season_value_timing"})


def simulate_hero_cell(
    *,
    strategy_key: str,
    hero_seat: int,
    seed: int,
    pool: pd.DataFrame,
    config: LeagueConfig,
    availability: PlayerAvailability | None,
    proj_lookup: Mapping[tuple[str, int], float],
    actual_lookup: Mapping[tuple[str, int], float],
    calendar: Calendar,
    jitter: float = 8.0,
    strategy_n_sims: int = 50,
    base_seed: int = 0,
    floor: float = _DEFAULT_FLOOR,
    floor_weight: float = _DEFAULT_FLOOR_WEIGHT,
) -> tuple[LeagueResult, LeagueResult]:
    """Simulate one (strategy, seat, seed) cell; return the hero seat's (actual, projected).

    The league seed is ``base_seed + seed`` -- independent of strategy and seat, so every
    strategy at a given (seat, seed) faces the identical schedule + bot draws (CRN).
    """
    if strategy_key in _MC_KEYS and availability is None:
        raise ValueError(f"strategy {strategy_key!r} requires availability data (None given)")
    layout = hero_seat_layout(hero_seat=hero_seat, hero_label=strategy_key, n_teams=config.n_teams)
    hero = _build_strategy(
        strategy_key,
        availability=availability,  # type: ignore[arg-type]
        n_teams=config.n_teams,
        strategy_n_sims=strategy_n_sims,
        base_seed=base_seed,
        floor=floor,
        floor_weight=floor_weight,
    )
    seat_strategies = {s: (hero if label != "bot" else None) for s, label in layout.items()}
    outcome = simulate_league(
        base_seed + seed,
        seat_strategies=seat_strategies,
        strategy_labels=layout,
        pool=pool,
        config=config,
        proj_lookup=proj_lookup,
        actual_lookup=actual_lookup,
        calendar=calendar,
        jitter=jitter,
    )
    (a,) = [r for r in outcome.actual if r.seat == hero_seat]
    (p,) = [r for r in outcome.projected if r.seat == hero_seat]
    return a, p


@dataclasses.dataclass(frozen=True)
class HeroCell:
    """One simulated hero-vs-bots cell: the hero's result under both scorings."""

    season: int
    strategy: str
    seat: int
    seed: int
    actual: LeagueResult
    projected: LeagueResult


def consolidate_cells(cells: list[HeroCell]) -> pd.DataFrame:
    """Flatten cells into a long-format, validated HeroResultSchema frame."""
    rows: list[dict[str, object]] = []
    for c in cells:
        for scoring, res in (("actual", c.actual), ("projected", c.projected)):
            rows.append(
                {
                    "season": c.season,
                    "strategy": c.strategy,
                    "seat": c.seat,
                    "seed": c.seed,
                    "scoring": scoring,
                    "wins": res.wins,
                    "losses": res.losses,
                    "made_playoffs": res.made_playoffs,
                    "is_champion": res.is_champion,
                    "points_for": res.points_for,
                }
            )
    df = pd.DataFrame(rows)
    return HeroResultSchema.validate(df)


def _cell_file(checkpoint_dir: Path, strategy: str, seat: int, seed: int) -> Path:
    return checkpoint_dir / f"cell_{strategy}_{seat:02d}_{seed:05d}.json"


def _valid_cell(path: Path) -> tuple[LeagueResult, LeagueResult] | None:
    """Return the cell's (actual, projected) if the checkpoint parses to exactly one of
    each, else None (missing/corrupt → re-run)."""
    if not path.exists():
        return None
    try:
        a, p = load_results(json.loads(path.read_text()))
    except (json.JSONDecodeError, KeyError, TypeError, OSError):
        return None
    if len(a) != 1 or len(p) != 1:
        return None
    return a[0], p[0]


def collect_hero_cells(
    *,
    seed_lo: int,
    seed_hi: int,
    strategies: tuple[str, ...],
    season: int,
    pool: pd.DataFrame,
    config: LeagueConfig,
    availability: PlayerAvailability | None,
    proj_lookup: Mapping[tuple[str, int], float],
    actual_lookup: Mapping[tuple[str, int], float],
    calendar: Calendar,
    jitter: float = 8.0,
    strategy_n_sims: int = 50,
    base_seed: int = 0,
    floor: float = _DEFAULT_FLOOR,
    floor_weight: float = _DEFAULT_FLOOR_WEIGHT,
    checkpoint_dir: Path,
) -> list[HeroCell]:
    """Sweep (strategy, seat, seed) over seats [1, n_teams] and seeds [seed_lo, seed_hi).

    Each cell is checkpointed (atomic JSON); a valid existing checkpoint is loaded, not
    recomputed (resume). Returns the full HeroCell list (computed + resumed).
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cells: list[HeroCell] = []
    for strategy in strategies:
        for seat in range(1, config.n_teams + 1):
            for seed in range(seed_lo, seed_hi):
                out = _cell_file(checkpoint_dir, strategy, seat, seed)
                cached = _valid_cell(out)
                if cached is None:
                    a, p = simulate_hero_cell(
                        strategy_key=strategy,
                        hero_seat=seat,
                        seed=seed,
                        pool=pool,
                        config=config,
                        availability=availability,
                        proj_lookup=proj_lookup,
                        actual_lookup=actual_lookup,
                        calendar=calendar,
                        jitter=jitter,
                        strategy_n_sims=strategy_n_sims,
                        base_seed=base_seed,
                        floor=floor,
                        floor_weight=floor_weight,
                    )
                    tmp = out.with_suffix(".tmp")
                    tmp.write_text(json.dumps(dump_results([a], [p])))
                    tmp.replace(out)  # atomic publish
                else:
                    a, p = cached
                cells.append(
                    HeroCell(
                        season=season,
                        strategy=strategy,
                        seat=seat,
                        seed=seed,
                        actual=a,
                        projected=p,
                    )
                )
    return cells


def load_hero_cells(
    *,
    seed_hi: int,
    strategies: tuple[str, ...],
    season: int,
    n_teams: int,
    checkpoint_dir: Path,
) -> list[HeroCell]:
    """Load already-computed cells (strategy x [1, n_teams] x [0, seed_hi)); NEVER simulate.

    Fails loud on any missing/invalid cell — the run must be complete first. Used by the
    report path so a mismatched/incomplete run errors instead of silently recomputing.
    """
    cells: list[HeroCell] = []
    for strategy in strategies:
        for seat in range(1, n_teams + 1):
            for seed in range(seed_hi):
                out = _cell_file(checkpoint_dir, strategy, seat, seed)
                cached = _valid_cell(out)
                if cached is None:
                    raise FileNotFoundError(
                        f"missing/invalid hero cell {out.name} — finish the run first "
                        f"(strategy={strategy}, seat={seat}, seed={seed})"
                    )
                a, p = cached
                cells.append(
                    HeroCell(
                        season=season,
                        strategy=strategy,
                        seat=seat,
                        seed=seed,
                        actual=a,
                        projected=p,
                    )
                )
    return cells
