import json
from pathlib import Path

import pandas as pd
import pytest

from projections.draft.assistant._compare import Interval
from projections.draft.assistant.auction.bid_strategy import AuctionBidStrategy, PatientValueBid
from projections.draft.assistant.auction.tournament import AuctionTournamentResult
from projections.draft.assistant.auction.tournament_cli import (
    _MODELS,
    _format_espn_diagnostic,
    _parse_args,
    format_compare,
    run,
)
from projections.schemas import _PYARROW_STR
from tests.test_draft.test_assistant_auction_tournament import _config, _pool


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


def test_default_models_are_the_ten_contestants() -> None:
    assert set(_MODELS) == {
        "static",
        "inflation",
        "marginal",
        "anchors",
        "overbid",
        "vorpshare",
        "patient",
        "patient_deep",
        "studsdepth",
        "balanced",
    }


def test_patient_deep_is_the_scrub_frac_zero_tuning() -> None:
    # patient_deep is PatientValueBid tuned to hoard mid-tier breadth (no $1-dumping the bottom
    # half); scrub_frac=0 is the whole point of the contestant — pin it against regressions.
    patient_deep = _MODELS["patient_deep"]
    assert isinstance(patient_deep, PatientValueBid)
    assert patient_deep.scrub_frac == 0.0


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


def test_parse_args_bot_prices_defaults_to_espn() -> None:
    args = _parse_args(
        [
            "--vorp-table",
            "x.parquet",
            "--league-config",
            "c.json",
            "--my-seat",
            "1",
            "--season",
            "2026",
            "compare",
        ]
    )
    assert args.bot_prices == "espn"


def test_parse_args_bot_prices_accepts_model() -> None:
    args = _parse_args(
        [
            "--vorp-table",
            "x.parquet",
            "--league-config",
            "c.json",
            "--my-seat",
            "1",
            "--season",
            "2026",
            "--bot-prices",
            "model",
            "compare",
        ]
    )
    assert args.bot_prices == "model"


def _diag_pool(n: int = 40) -> pd.DataFrame:
    """The tournament test's `_pool` plus a populated `espn_auction_dollars` Int64 column, so the
    diagnostic takes the real (non-skipped) path."""
    pool = _pool(n)
    pool["espn_auction_dollars"] = pd.array([int(60 - i) for i in range(n)], dtype="Int64")
    return pool


def test_format_espn_diagnostic_real_readout() -> None:
    pool = _diag_pool(40)
    out = _format_espn_diagnostic(pool, _config())
    assert "ESPN vs ours" in out  # the real header, not the "skipped" message
    assert "delta" in out
    assert "skipped" not in out


def test_format_espn_diagnostic_skipped_without_espn_column() -> None:
    pool = _diag_pool(40).drop(columns=["espn_auction_dollars"])
    out = _format_espn_diagnostic(pool, _config())
    assert "skipped" in out


def test_parse_args_bot_prices_rejects_unknown() -> None:
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--vorp-table",
                "x.parquet",
                "--league-config",
                "c.json",
                "--my-seat",
                "1",
                "--season",
                "2026",
                "--bot-prices",
                "sos",
                "compare",
            ]
        )
