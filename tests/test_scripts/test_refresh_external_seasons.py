"""Tests for scripts/refresh_external_seasons.py (loop + per-season failure isolation)."""

from __future__ import annotations

from pathlib import Path

import generate_preset_vorp_tables
import pytest
import refresh_external_seasons as rs  # scripts/ on sys.path via conftest

from projections.ingest.external_projections import ExternalProjectionError

# NOTE: import ExternalProjectionError + generate_preset_vorp_tables DIRECTLY (not via `rs.…`).
# mypy-strict's no_implicit_reexport flags reaching through `rs` for names it merely imported.


def test_loops_each_season_and_isolates_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    ingested: list[int] = []
    gen_calls: list[list[str]] = []

    def fake_ingest(data_root: Path, *, season: int, asof: object = None) -> None:
        if season == 2022:
            raise ExternalProjectionError("boom")
        ingested.append(season)

    def fake_gen(argv: list[str]) -> int:
        gen_calls.append(argv)
        return 0

    monkeypatch.setattr(rs, "refresh_external_projections", fake_ingest)
    monkeypatch.setattr(generate_preset_vorp_tables, "main", fake_gen)

    status = rs.run([2021, 2022, 2023], Path("data"))

    assert set(status) == {2021, 2022, 2023}
    assert status[2021] == "ok" and status[2023] == "ok"
    assert status[2022].startswith("failed")  # isolated, not dropped
    assert ingested == [2021, 2023]  # 2022 raised before reaching gen
    assert gen_calls == [
        ["--season", "2021", "--data-root", "data"],
        ["--season", "2023", "--data-root", "data"],
    ]


def test_isolates_generator_side_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A season whose *generation* fails pandera validation (not just ingest) is isolated too —
    the loop must not abort on a SchemaError from generate_preset_vorp_tables.main."""
    import pandas as pd

    from projections.schemas import VorpTableSchema

    def fake_gen(argv: list[str]) -> int:
        if argv[1] == "2022":  # argv is ["--season", str(year), "--data-root", ...]
            VorpTableSchema.validate(pd.DataFrame({"unexpected": [1]}))  # -> pandera SchemaError
        return 0

    monkeypatch.setattr(rs, "refresh_external_projections", lambda *a, **k: None)
    monkeypatch.setattr(generate_preset_vorp_tables, "main", fake_gen)

    status = rs.run([2021, 2022, 2023], Path("data"))

    assert status[2021] == "ok" and status[2023] == "ok"
    assert status[2022].startswith("failed")  # generator SchemaError isolated, not loop-aborting
