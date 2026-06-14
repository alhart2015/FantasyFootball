"""Predictive forward-outcome simulation (spec 2026-06-14, Consumer B).

Unlike the historical backtest (which scores on fixed real actuals), this draws each player's
weekly points from the fitted performance-variance model and runs the league many times, yielding
a forward distribution of champ%/playoff%/wins per drafted team — an honest predictive CI that
reflects player-outcome luck, not just draft/schedule luck. The historical real-actuals path is
untouched; this complements it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from projections.draft.assistant.performance_variance import VarianceParams, sample_weekly_points
from projections.draft.assistant.strategy import DraftStrategy
from projections.draft.backtest.draft_field import draft_mixed_field
from projections.draft.backtest.league import Calendar, score_drafted_league
from projections.draft.backtest.schedule import regular_season_schedule
from projections.draft.league_config import LeagueConfig


def sample_actual_lookup(
    pool: pd.DataFrame,
    params: VarianceParams,
    *,
    weeks: Iterable[int],
    rng: np.random.Generator,
) -> dict[tuple[str, int], float]:
    """One predictive season's per-(gsis_id, week) points, sampled from the variance model.

    `pool` carries gsis_id, position, season_mean_fpts, is_rookie. Returns a dict shaped like the
    backtest's `actual_lookup`, with a value for every (player, week) in `weeks`.
    """
    weeks = list(weeks)
    gsis = pool["gsis_id"].astype(str).to_numpy()
    positions = pool["position"].astype(str).to_numpy()
    means = pool["season_mean_fpts"].to_numpy(dtype=np.float64)
    rookie = (
        pool["is_rookie"].to_numpy(dtype=bool)
        if "is_rookie" in pool.columns
        else np.zeros(len(pool), bool)
    )
    pts = sample_weekly_points(
        params, positions, means, rookie, n_sims=1, n_weeks=len(weeks), rng=rng
    )[0]  # (n_weeks, n_players)
    return {
        (str(gsis[i]), wk): float(pts[w, i]) for w, wk in enumerate(weeks) for i in range(len(gsis))
    }


def predictive_outcomes(
    pool: pd.DataFrame,
    config: LeagueConfig,
    proj_lookup: Mapping[tuple[str, int], float],
    params: VarianceParams,
    *,
    seat_strategies: Mapping[int, DraftStrategy | None],
    strategy_labels: Mapping[int, str],
    calendar: Calendar,
    jitter: float,
    draft_seeds: Iterable[int],
    n_predictive_sims: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, np.ndarray]]:
    """Forward outcome distributions per strategy label.

    For each draft seed, draft ONCE (rosters + schedule fixed), then re-score
    `n_predictive_sims` model-sampled seasons (no re-draft). Returns
    {strategy_label: {"champ": arr, "playoff": arr, "wins": arr}} aggregated over all
    (seed, sim, seat-with-that-label) — the forward CI inputs.
    """
    weeks = sorted(set(calendar.regular_weeks) | set(calendar.playoff_weeks))
    acc: dict[str, dict[str, list[float]]] = {}
    for seed in draft_seeds:
        draft_rng = np.random.default_rng(seed)
        rosters = draft_mixed_field(
            dict(seat_strategies), pool, config, rng=draft_rng, jitter=jitter
        )
        sched = regular_season_schedule(
            n_teams=config.n_teams, n_weeks=len(calendar.regular_weeks), rng=draft_rng
        )
        for _ in range(n_predictive_sims):
            actual_lookup = sample_actual_lookup(pool, params, weeks=weeks, rng=rng)
            outcome = score_drafted_league(
                rosters,
                pool,
                config,
                proj_lookup=proj_lookup,
                actual_lookup=actual_lookup,
                calendar=calendar,
                strategy_labels=strategy_labels,
                sched=sched,
            )
            for r in outcome.actual:
                cell = acc.setdefault(r.strategy, {"champ": [], "playoff": [], "wins": []})
                cell["champ"].append(1.0 if r.is_champion else 0.0)
                cell["playoff"].append(1.0 if r.made_playoffs else 0.0)
                cell["wins"].append(float(r.wins))
    return {k: {m: np.array(v) for m, v in cell.items()} for k, cell in acc.items()}
