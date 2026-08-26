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

import pandas as pd

from projections.draft.assistant.performance_variance import DEFAULT_PARAMS_PATH
from projections.ingest.espn_league import EspnCredentials
from projections.schemas import _PYARROW_STR, IdMapSchema, VorpTableSchema
from projections.web.app import DashboardConfig

#: `(path, how to produce it)` per named input.
Required = Mapping[str, tuple[Path, str]]


def id_map_path(config: DashboardConfig) -> Path:
    """One spelling of the path. It was written out in the pre-check and again in each route,
    so changing it in the pre-check would have named a file the routes do not read."""
    return config.data_root / "raw" / "id_map.parquet"


def pool_and_id_map(config: DashboardConfig) -> Required:
    """The two inputs every page needs."""
    return {
        "the VORP pool": (config.pool_path, "scripts/generate_league_vorp_table.py"),
        "the id_map": (id_map_path(config), "projections.ingest.id_map.build_id_map"),
    }


def load_pool(config: DashboardConfig) -> pd.DataFrame:
    """The VORP pool, validated. Beside the declaration of its path, and shared, because the
    two routes had a copy each -- including the `astype` that makes the schema pass."""
    pool = pd.read_parquet(config.pool_path)
    pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)
    return VorpTableSchema.validate(pool)


def load_id_map(config: DashboardConfig) -> pd.DataFrame:
    """The id_map, validated.

    Validated on BOTH pages. They read the same file, so a page that skipped the check
    surfaced a drifted id_map as a traceback while the other named it -- the divergence this
    module exists to remove, one layer down.
    """
    return IdMapSchema.validate(pd.read_parquet(id_map_path(config)))


def missing_inputs(config: DashboardConfig, required: Required, *, action: str) -> str | None:
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
    return f"Cannot {action} — missing " + "; ".join(absent) + "."


#: The simulator's fitted parameters.
#:
#: A constant rather than a function of the config, because it is not one: `VarianceParams.load()`
#: defaults to a RELATIVE path, resolved against the process CWD rather than `data_root`, so
#: starting the dashboard from anywhere but the repo root used to 500 inside the projection.
#: Naming it in the pre-check is what turns that into a sentence a reader can act on, and the
#: CWD-dependence is exactly why the message quotes the path it actually checked.
VARIANCE_PARAMS: Required = {
    "the variance params": (
        DEFAULT_PARAMS_PATH,
        "they ship with the repo — run the dashboard from the repo root",
    )
}
