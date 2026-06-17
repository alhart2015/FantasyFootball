import numpy as np
import pytest

from projections.draft.assistant._compare import Interval, bootstrap_mean, validate_pool_size
from projections.draft.league_config import LeagueConfig
from projections.schemas import RosterSlot, Ruleset


def _config(n_teams: int = 2) -> LeagueConfig:
    return LeagueConfig(
        name="t",
        n_teams=n_teams,
        roster_slots={RosterSlot.RB: 1, RosterSlot.WR: 1, RosterSlot.BENCH: 1},
        ruleset=Ruleset.espn_ppr(),
    )


def test_bootstrap_mean_on_constant_array_is_a_point() -> None:
    iv = bootstrap_mean(np.full(50, 7.0), seed=0)
    assert iv == Interval(point=7.0, lo_95=7.0, hi_95=7.0)


def test_bootstrap_mean_ci_brackets_the_mean() -> None:
    iv = bootstrap_mean(np.arange(100, dtype=float), seed=1)
    assert iv.lo_95 < iv.point < iv.hi_95
    assert abs(iv.point - 49.5) < 1e-9


def test_validate_pool_size_raises_when_too_small() -> None:
    import pandas as pd

    pool = pd.DataFrame({"gsis_id": ["00-0000001"]})  # 1 player; need n_teams*roster_size = 2*3 = 6
    with pytest.raises(ValueError, match="need >= 6"):
        validate_pool_size(pool, _config(2))
