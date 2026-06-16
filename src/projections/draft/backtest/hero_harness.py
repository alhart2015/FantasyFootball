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
from projections.draft.assistant.tournament import Interval, _bootstrap_mean
from projections.draft.backtest.checkpoint import dump_results, load_results
from projections.draft.backtest.draft_field import hero_seat_layout
from projections.draft.backtest.harness import StrategyMetrics, _build_strategy
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


def _metrics_from_group(g: pd.DataFrame, *, base_seed: int = 0) -> StrategyMetrics:
    win = (g["wins"] / (g["wins"] + g["losses"])).to_numpy(dtype=float)
    playoff = g["made_playoffs"].to_numpy(dtype=float)
    champ = g["is_champion"].to_numpy(dtype=float)
    pf = g["points_for"].to_numpy(dtype=float)
    return StrategyMetrics(
        championship=_bootstrap_mean(champ, seed=base_seed),
        playoff=_bootstrap_mean(playoff, seed=base_seed),
        win_pct=_bootstrap_mean(win, seed=base_seed),
        points_for=_bootstrap_mean(pf, seed=base_seed),
    )


def seat_averaged_metrics(
    df: pd.DataFrame, *, scoring: str, base_seed: int = 0
) -> dict[str, StrategyMetrics]:
    """Per-strategy metrics averaged over all seats+seeds (the headline)."""
    sub = df[df["scoring"] == scoring]
    return {
        str(s): _metrics_from_group(g, base_seed=base_seed)
        for s, g in sub.groupby("strategy", sort=True)
    }


def per_seat_metrics(
    df: pd.DataFrame, *, scoring: str, base_seed: int = 0
) -> dict[tuple[str, int], StrategyMetrics]:
    """Per-(strategy, seat) metrics — the retained slot-by-slot breakdown."""
    sub = df[df["scoring"] == scoring]
    return {
        (str(s), int(seat)): _metrics_from_group(g, base_seed=base_seed)
        for (s, seat), g in sub.groupby(["strategy", "seat"], sort=True)
    }


_METRIC_COL = {"win_pct": None, "playoff": "made_playoffs", "championship": "is_champion"}


def _metric_series(g: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "win_pct":
        return (g["wins"] / (g["wins"] + g["losses"])).astype(float)
    return g[_METRIC_COL[metric]].astype(float)


def paired_diff(
    df: pd.DataFrame,
    *,
    scoring: str,
    metric: str,
    strategy: str,
    reference: str,
    base_seed: int = 0,
) -> Interval:
    """Bootstrap CI of (strategy - reference) on `metric`, paired on the shared (seat, seed)
    grid (CRN). metric in {win_pct, playoff, championship}."""
    sub = df[df["scoring"] == scoring]
    a = sub[sub["strategy"] == strategy].set_index(["seat", "seed"]).sort_index()
    b = sub[sub["strategy"] == reference].set_index(["seat", "seed"]).sort_index()
    common = a.index.intersection(b.index)
    diff = (
        _metric_series(a.loc[common], metric).to_numpy()
        - _metric_series(b.loc[common], metric).to_numpy()
    )
    return _bootstrap_mean(diff, seed=base_seed)


def _exact(v: float) -> Interval:
    """A degenerate (exact, zero-width) Interval — for structural constants."""
    return Interval(point=v, lo_95=v, hi_95=v)


def bot_baseline(calendar: Calendar, n_teams: int) -> StrategyMetrics:
    """The structural average-team reference for an n_teams league with this playoff size:
    win 0.5, playoff playoff_size/n_teams, champ 1/n_teams. In a 1-hero league these are
    exact by construction (zero-sum win%; one champion; playoff_size berths), not estimated.
    points_for has no structural value -> NaN."""
    return StrategyMetrics(
        championship=_exact(1.0 / n_teams),
        playoff=_exact(calendar.playoff_size / n_teams),
        win_pct=_exact(0.5),
        points_for=_exact(float("nan")),
    )


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
