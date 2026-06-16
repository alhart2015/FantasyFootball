"""Tests for the scoring x size draft preset registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from projections.draft.assistant.presets import (
    DEFAULT_SCORING,
    DEFAULT_TEAMS,
    SCORING_KEYS,
    TEAM_SIZES,
    get_preset,
    materialize_league_config,
)
from projections.draft.league_config import LeagueConfig
from projections.schemas import RosterSlot, Ruleset

_SKILL = {
    RosterSlot.QB: 1,
    RosterSlot.RB: 2,
    RosterSlot.WR: 3,
    RosterSlot.TE: 1,
    RosterSlot.FLEX: 1,
    RosterSlot.BENCH: 9,
}


def test_default_is_half_ppr_16team() -> None:
    assert DEFAULT_SCORING == "half"
    assert DEFAULT_TEAMS == 16
    assert SCORING_KEYS == ("half", "ppr", "std")
    assert TEAM_SIZES == (10, 12, 16)


def test_get_preset_resolves_ruleset_size_roster_and_path() -> None:
    p = get_preset("half", 16)
    assert p.league_config.n_teams == 16
    assert p.league_config.ruleset == Ruleset.espn_half()
    assert p.league_config.roster_slots == _SKILL
    assert p.table_path == Path("data/vorp_2026/half_16team.parquet")
    assert get_preset("ppr", 12).league_config.ruleset == Ruleset.espn_ppr()
    assert get_preset("std", 10).league_config.ruleset == Ruleset.standard()


def test_get_preset_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="scoring"):
        get_preset("nope", 16)
    with pytest.raises(ValueError, match="n_teams"):
        get_preset("half", 11)


def test_materialize_league_config_writes_a_resumable_path(tmp_path: Path) -> None:
    """Resume stores the league config as a PATH, so a preset-started draft must point at a real
    file. The written JSON must round-trip back to the preset's config."""
    preset = get_preset("half", 16)
    path = materialize_league_config(preset, config_dir=tmp_path)
    assert path.exists() and path.is_file()
    loaded = LeagueConfig.model_validate_json(path.read_text())
    assert loaded == preset.league_config
