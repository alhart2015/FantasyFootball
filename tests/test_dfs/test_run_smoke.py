from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not Path("data/features/wr/season=2023").exists()
    or not Path("data/raw/sleeper_weekly_projections/season=2023").exists(),
    reason="requires built feature/raw + ingested Sleeper partitions",
)


def test_one_season_end_to_end() -> None:
    from projections.dfs.run import run_study
    from projections.schemas import Position, Ruleset

    out = run_study(
        seasons=[2023],
        positions=[Position.WR],
        data_root=Path("data"),
        features_root=Path("data/features"),
        ruleset=Ruleset.draftkings(),
    )
    assert out.primary.verdict in {"ADOPT", "STOP", "INCONCLUSIVE"}
