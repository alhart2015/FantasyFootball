import json
from pathlib import Path

import pandas as pd
import pytest

from projections.draft.assistant._compare import Interval
from projections.draft.assistant.auction.bid_strategy import AuctionBidStrategy
from projections.draft.assistant.auction.tournament import AuctionTournamentResult
from projections.draft.assistant.auction.tournament_cli import (
    _MODELS,
    _parse_args,
    format_compare,
    run,
)
from projections.schemas import _PYARROW_STR


def _write_pool(path: Path) -> None:
    n = 40
    pos = ["RB" if i % 2 else "WR" for i in range(n)]
    prefix = {"RB": 2, "WR": 3}
    df = pd.DataFrame(
        {
            "gsis_id": pd.array(
                [f"00-{prefix[pos[i]]}{i:06d}" for i in range(n)], dtype=_PYARROW_STR
            ),
            "position": pd.array(pos, dtype=_PYARROW_STR),
            "season_mean_fpts": [float(300 - i) for i in range(n)],
            "vorp": [float(150 - i) for i in range(n)],
            "replacement_fpts": [100.0] * n,
        }
    )
    df.to_parquet(path)


def _write_config(path: Path) -> None:
    cfg = {
        "name": "t",
        "n_teams": 6,  # project_draft requires >= 6 (PLAYOFF_SIZE) and even
        "budget": 100,
        "min_bid": 1,
        "roster_slots": {"RB": 1, "WR": 1, "BENCH": 1},
        "ruleset": "espn_ppr",
    }
    path.write_text(json.dumps(cfg))


def test_default_models_are_the_seven_contestants() -> None:
    assert set(_MODELS) == {
        "static",
        "inflation",
        "marginal",
        "anchors",
        "overbid",
        "vorpshare",
        "patient",
    }


def test_every_default_model_satisfies_the_protocol() -> None:
    assert all(isinstance(m, AuctionBidStrategy) for m in _MODELS.values())


def test_nomination_temp_defaults_to_one() -> None:
    args = _parse_args(
        [
            "--vorp-table",
            "x",
            "--league-config",
            "y",
            "--my-seat",
            "1",
            "--season",
            "2026",
            "compare",
        ]
    )
    assert args.nomination_temp == 1.0


def test_format_compare_has_no_winner_line() -> None:
    iv = Interval(1.0, 0.5, 1.5)
    metric_names = ("mean_points", "reg_win_pct", "make_playoffs_pct", "bye_pct", "champ_pct")
    metrics = {m: iv for m in metric_names}
    result = AuctionTournamentResult(
        summaries={"static": metrics, "inflation": metrics, "marginal": metrics},
        paired_diffs={"static_vs_inflation": metrics},
        n_seeds=10,
        price_jitter=0.15,
        base_seed=0,
        season_base_seed=1_000_000,
        n_sims=500,
        my_seat=1,
        budget=100,
        min_bid=1,
    )
    text = format_compare(result)
    assert "winner" not in text.lower()
    assert "champ" in text.lower()
    assert "static" in text and "inflation" in text and "marginal" in text


def test_cli_compare_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pool_path = tmp_path / "vorp.parquet"
    cfg_path = tmp_path / "league.json"
    _write_pool(pool_path)
    _write_config(cfg_path)

    # Stub the store-backed loaders so the smoke test needs no data store.
    import projections.draft.assistant.auction.tournament_cli as cli
    from projections.draft.assistant.availability import PlayerAvailability

    monkeypatch.setattr(
        cli,
        "load_store_availability",
        lambda pool, **kw: PlayerAvailability(
            p={g: 0.95 for g in pool["gsis_id"].astype(str)}, bye={}
        ),
    )
    monkeypatch.setattr(cli, "attach_is_rookie", lambda pool, **kw: pool.assign(is_rookie=False))

    rc = run(
        [
            "--vorp-table",
            str(pool_path),
            "--league-config",
            str(cfg_path),
            "--my-seat",
            "1",
            "--season",
            "2026",
            "--seeds",
            "2",
            "--n-sims",
            "20",
            "--price-jitter",
            "0.1",
            "compare",
        ]
    )
    assert rc == 0
