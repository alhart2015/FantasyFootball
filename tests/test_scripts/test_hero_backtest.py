from __future__ import annotations

import scripts.hero_backtest as mod


def test_run_key_includes_sweep_params() -> None:
    args = mod._parse_args(
        [
            "run",
            "--season",
            "2025",
            "--league-config",
            "configs/league_espn_half_16team.json",
            "--n-seeds",
            "40",
            "--strategy-n-sims",
            "50",
            "--strategies",
            "now_or_never,now_or_never_floored",
            "--floor",
            "40",
            "--floor-weight",
            "1",
        ]
    )
    key = mod._run_key(args)
    assert key["season"] == 2025
    assert key["strategies"] == "now_or_never,now_or_never_floored"
    assert key["strategy_n_sims"] == 50
    assert key["floor"] == 40.0


def test_default_strategies_are_the_six() -> None:
    args = mod._parse_args(["run", "--league-config", "configs/league_espn_half_16team.json"])
    assert args.strategies == (
        "raw_vorp,now_or_never,now_or_never_floored,"
        "season_value,season_value_var,season_value_timing"
    )
