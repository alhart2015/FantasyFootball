"""What a page needs on disk before it can be rendered, checked once.

Both pages read the same tree and both need the same message when part of it is absent. They
had a copy each, and the copies had already diverged: the standings page checked the ESPN
credentials and the team page did not, so the same missing input was named in a friendly
sentence on one page and surfaced as a raw resolver error on the other.

Checked up front rather than caught as `FileNotFoundError`, because that exception carries a
bare path and no indication of what the file is for or how to produce it. "No such file or
directory: 'nonexistent'" is not something a reader can act on.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from projections.draft.assistant.performance_variance import DEFAULT_PARAMS_PATH
from projections.ingest.espn_league import EspnCredentials
from projections.web.app import DashboardConfig

#: `(path, how to produce it)` per named input.
Required = Mapping[str, tuple[Path, str]]


def pool_and_id_map(config: DashboardConfig) -> Required:
    """The two inputs every page needs."""
    return {
        "the VORP pool": (config.pool_path, "scripts/generate_league_vorp_table.py"),
        "the id_map": (
            config.data_root / "raw" / "id_map.parquet",
            "projections.ingest.id_map.build_id_map",
        ),
    }


def missing_inputs(config: DashboardConfig, required: Required, *, page: str) -> str | None:
    """Name what is absent, or None when everything is present.

    The ESPN credentials are always checked, and checked in BOTH places they can live:
    `EspnCredentials.resolve` reads `ESPN_SWID`/`ESPN_S2` before the file, so requiring the
    file reported "missing the ESPN credentials" as a falsehood on any machine using
    environment credentials -- and refused to render one page while the other worked fine.
    """
    absent = [
        f"{label} at {path} (build it with {how})"
        for label, (path, how) in required.items()
        if not path.exists()
    ]
    if EspnCredentials.from_env() is None and not config.credentials_path.exists():
        absent.append(
            f"the ESPN credentials — neither ESPN_SWID/ESPN_S2 in the environment nor a file "
            f"at {config.credentials_path}"
        )
    if not absent:
        return None
    return f"Cannot {page} — missing " + "; ".join(absent) + "."


def variance_params(config: DashboardConfig) -> Required:
    """The simulator's fitted parameters.

    `VarianceParams.load()` defaults to a RELATIVE path, resolved against the process CWD
    rather than `data_root`, so starting the dashboard from anywhere but the repo root used to
    500 inside the projection. Named rather than left to raise -- and the CWD-dependence is
    the reason the message quotes the path it actually checked.
    """
    del config
    return {
        "the variance params": (
            DEFAULT_PARAMS_PATH,
            "they ship with the repo — run the dashboard from the repo root",
        )
    }
