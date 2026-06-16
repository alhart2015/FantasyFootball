from __future__ import annotations

import pytest

from projections.draft.backtest.league import Calendar
from projections.draft.league_config import LeagueConfig
from projections.schemas import RosterSlot
from tests.test_draft.test_backtest.test_availability_stub import stub_availability
from tests.test_draft.test_backtest.test_draft_field import _synthetic_pool


def _cfg16() -> LeagueConfig:
    return LeagueConfig(
        name="t16",
        n_teams=16,
        budget=200,
        min_bid=1,
        roster_slots={
            RosterSlot.QB: 1,
            RosterSlot.RB: 2,
            RosterSlot.WR: 2,
            RosterSlot.TE: 1,
            RosterSlot.FLEX: 1,
            RosterSlot.BENCH: 4,
        },
        ruleset="espn_half",  # type: ignore[arg-type]
    )


def _inputs():
    pool = _synthetic_pool(n_per_pos=60)
    cal = Calendar(regular_weeks=tuple(range(1, 6)), playoff_weeks=(6, 7, 8), playoff_size=6)
    proj = {
        (g, wk): float(m)
        for g, m in zip(pool["gsis_id"], pool["season_mean_fpts"], strict=False)
        for wk in range(1, 9)
    }
    return pool, cal, dict(proj), dict(proj)


def test_simulate_hero_cell_returns_hero_seat_result() -> None:
    from projections.draft.backtest.hero_harness import simulate_hero_cell

    pool, cal, proj, actual = _inputs()
    cfg = _cfg16()
    a, p = simulate_hero_cell(
        strategy_key="now_or_never",
        hero_seat=4,
        seed=0,
        pool=pool,
        config=cfg,
        availability=stub_availability(pool),
        proj_lookup=proj,
        actual_lookup=actual,
        calendar=cal,
        jitter=8.0,
        strategy_n_sims=5,
        base_seed=0,
    )
    assert a.seat == 4 and p.seat == 4
    assert a.strategy == "now_or_never" and p.strategy == "now_or_never"


def test_simulate_hero_cell_is_deterministic() -> None:
    from projections.draft.backtest.hero_harness import simulate_hero_cell

    pool, cal, proj, actual = _inputs()
    cfg = _cfg16()
    kw = dict(
        strategy_key="now_or_never",
        hero_seat=4,
        seed=1,
        pool=pool,
        config=cfg,
        availability=stub_availability(pool),
        proj_lookup=proj,
        actual_lookup=actual,
        calendar=cal,
        jitter=8.0,
        strategy_n_sims=5,
        base_seed=0,
    )
    a1, _ = simulate_hero_cell(**kw)
    a2, _ = simulate_hero_cell(**kw)
    assert (a1.wins, a1.losses, a1.points_for) == (a2.wins, a2.losses, a2.points_for)


def test_simulate_hero_cell_crn_seed_is_strategy_and_seat_independent(monkeypatch) -> None:
    """The load-bearing CRN invariant (spec §3): the league seed passed to simulate_league
    is base_seed + seed, independent of strategy AND seat. Capture the seed across several
    (strategy, seat) and assert it never varies for a fixed (base_seed, seed)."""
    import projections.draft.backtest.hero_harness as hh
    from projections.draft.backtest.league import LeagueOutcome, LeagueResult

    pool, cal, proj, actual = _inputs()
    cfg = _cfg16()
    captured: list[int] = []

    def _fake_simulate_league(seed, *, strategy_labels, **kw):
        captured.append(seed)
        hero_seat = next(s for s, lbl in strategy_labels.items() if lbl != "bot")
        lbl = strategy_labels[hero_seat]
        r = LeagueResult(
            seat=hero_seat,
            strategy=lbl,
            wins=1,
            losses=1,
            points_for=1.0,
            made_playoffs=False,
            is_champion=False,
        )
        return LeagueOutcome(actual=[r], projected=[r])

    monkeypatch.setattr(hh, "simulate_league", _fake_simulate_league)
    for strat in ("raw_vorp", "now_or_never"):
        for seat in (2, 6, 11):
            hh.simulate_hero_cell(
                strategy_key=strat,
                hero_seat=seat,
                seed=3,
                pool=pool,
                config=cfg,
                availability=stub_availability(pool),
                proj_lookup=proj,
                actual_lookup=actual,
                calendar=cal,
                jitter=8.0,
                strategy_n_sims=5,
                base_seed=100,
            )
    assert captured == [103] * 6


def test_simulate_hero_cell_mc_requires_availability() -> None:
    from projections.draft.backtest.hero_harness import simulate_hero_cell

    pool, cal, proj, actual = _inputs()
    with pytest.raises(ValueError, match="availability"):
        simulate_hero_cell(
            strategy_key="season_value",
            hero_seat=1,
            seed=0,
            pool=pool,
            config=_cfg16(),
            availability=None,
            proj_lookup=proj,
            actual_lookup=actual,
            calendar=cal,
            jitter=8.0,
            strategy_n_sims=5,
            base_seed=0,
        )


def test_consolidate_cells_to_schema() -> None:
    from projections.draft.backtest.hero_harness import HeroCell, consolidate_cells
    from projections.draft.backtest.league import LeagueResult
    from projections.schemas import HeroResultSchema

    a = LeagueResult(
        seat=4,
        strategy="now_or_never",
        wins=8,
        losses=6,
        points_for=1200.0,
        made_playoffs=True,
        is_champion=False,
    )
    p = LeagueResult(
        seat=4,
        strategy="now_or_never",
        wins=9,
        losses=5,
        points_for=1250.0,
        made_playoffs=True,
        is_champion=True,
    )
    cells = [HeroCell(season=2025, strategy="now_or_never", seat=4, seed=0, actual=a, projected=p)]
    df = consolidate_cells(cells)
    HeroResultSchema.validate(df)
    assert set(df["scoring"]) == {"actual", "projected"}
    assert len(df) == 2
    row = df[(df["scoring"] == "actual")].iloc[0]
    assert row["strategy"] == "now_or_never" and row["seat"] == 4 and row["wins"] == 8


def test_collect_hero_cells_resumes_and_skips_completed(tmp_path) -> None:
    from projections.draft.backtest.hero_harness import collect_hero_cells

    pool, cal, proj, actual = _inputs()
    cfg = _cfg16()
    kw = dict(
        seed_lo=0,
        seed_hi=2,
        strategies=("raw_vorp",),
        season=2025,
        pool=pool,
        config=cfg,
        availability=stub_availability(pool),
        proj_lookup=proj,
        actual_lookup=actual,
        calendar=cal,
        jitter=8.0,
        strategy_n_sims=5,
        base_seed=0,
        checkpoint_dir=tmp_path,
    )
    cells1 = collect_hero_cells(**kw)
    assert len(cells1) == 32  # raw_vorp x 16 seats x 2 seeds
    files = list(tmp_path.glob("cell_*.json"))
    assert len(files) == 32
    mtimes = {f: f.stat().st_mtime_ns for f in files}

    cells2 = collect_hero_cells(**kw)  # re-run: all cached → no rewrites
    assert len(cells2) == 32
    assert all(f.stat().st_mtime_ns == mtimes[f] for f in files)


def test_collect_hero_cells_ignores_corrupt_checkpoint(tmp_path) -> None:
    from projections.draft.backtest.hero_harness import collect_hero_cells

    pool, cal, proj, actual = _inputs()
    kw = dict(
        seed_lo=0,
        seed_hi=1,
        strategies=("raw_vorp",),
        season=2025,
        pool=pool,
        config=_cfg16(),
        availability=stub_availability(pool),
        proj_lookup=proj,
        actual_lookup=actual,
        calendar=cal,
        jitter=8.0,
        strategy_n_sims=5,
        base_seed=0,
        checkpoint_dir=tmp_path,
    )
    collect_hero_cells(**kw)
    victim = next(iter(tmp_path.glob("cell_*.json")))
    victim.write_text("{ truncated")  # corrupt → must be re-run, not crash
    cells = collect_hero_cells(**kw)
    assert len(cells) == 16  # raw_vorp x 16 seats x 1 seed


def test_load_hero_cells_fails_loud_on_missing(tmp_path) -> None:
    from projections.draft.backtest.hero_harness import collect_hero_cells, load_hero_cells

    pool, cal, proj, actual = _inputs()
    cfg = _cfg16()
    base = dict(
        season=2025,
        pool=pool,
        config=cfg,
        availability=stub_availability(pool),
        proj_lookup=proj,
        actual_lookup=actual,
        calendar=cal,
        jitter=8.0,
        strategy_n_sims=5,
        base_seed=0,
        checkpoint_dir=tmp_path,
    )
    collect_hero_cells(seed_lo=0, seed_hi=1, strategies=("raw_vorp",), **base)  # 16 cells
    cells = load_hero_cells(
        seed_hi=1,
        strategies=("raw_vorp",),
        season=2025,
        n_teams=cfg.n_teams,
        checkpoint_dir=tmp_path,
    )
    assert len(cells) == 16
    with pytest.raises(FileNotFoundError, match="now_or_never"):
        load_hero_cells(
            seed_hi=1,
            strategies=("now_or_never",),
            season=2025,
            n_teams=cfg.n_teams,
            checkpoint_dir=tmp_path,
        )


def _two_strategy_frame():
    """Hand-built HeroResultSchema frame: 'good' always 10-4, 'bad' always 4-10,
    over 2 seats x 2 seeds, both scorings."""
    import pandas as pd

    rows = []
    for strat, (w, ls) in (("good", (10, 4)), ("bad", (4, 10))):
        for seat in (1, 2):
            for seed in (0, 1):
                for scoring in ("actual", "projected"):
                    rows.append(
                        dict(
                            season=2025,
                            strategy=strat,
                            seat=seat,
                            seed=seed,
                            scoring=scoring,
                            wins=w,
                            losses=ls,
                            made_playoffs=(strat == "good"),
                            is_champion=False,
                            points_for=1000.0 + w,
                        )
                    )
    return pd.DataFrame(rows)


def test_seat_averaged_metrics_win_pct() -> None:
    from projections.draft.backtest.hero_harness import seat_averaged_metrics

    m = seat_averaged_metrics(_two_strategy_frame(), scoring="actual")
    assert abs(m["good"].win_pct.point - 10 / 14) < 1e-9
    assert abs(m["bad"].win_pct.point - 4 / 14) < 1e-9


def test_per_seat_metrics_groups_by_seat() -> None:
    from projections.draft.backtest.hero_harness import per_seat_metrics

    m = per_seat_metrics(_two_strategy_frame(), scoring="actual")
    assert ("good", 1) in m and ("good", 2) in m
    assert abs(m[("good", 1)].win_pct.point - 10 / 14) < 1e-9


def test_paired_diff_sign_and_zero() -> None:
    from projections.draft.backtest.hero_harness import paired_diff

    df = _two_strategy_frame()
    d = paired_diff(df, scoring="actual", metric="win_pct", strategy="good", reference="bad")
    assert d.point > 0
    z = paired_diff(df, scoring="actual", metric="win_pct", strategy="good", reference="good")
    assert z.point == 0.0


def test_bot_baseline_is_structural() -> None:
    from projections.draft.backtest.hero_harness import bot_baseline

    _, cal, _, _ = _inputs()  # cal.playoff_size == 6
    b = bot_baseline(cal, 16)
    assert b.win_pct.point == 0.5
    assert abs(b.playoff.point - 6 / 16) < 1e-9
    assert abs(b.championship.point - 1 / 16) < 1e-9
