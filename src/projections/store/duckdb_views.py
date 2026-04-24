"""DuckDB view layer — register parquet directories as queryable tables."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def _connect_with_views(root: Path) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection and register every directory under
    `root` as a view over its parquet files. Tables that don't exist yet
    simply don't get a view."""
    con = duckdb.connect(database=":memory:")
    if not root.exists():
        return con

    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            # Partitioned table: glob over season=*/week=*/part.parquet.
            # Use as_posix() to force forward slashes — DuckDB's glob on Windows
            # does not handle backslashes reliably.
            # hive_partitioning=false: season/week are already stored as typed
            # integer columns inside the parquet files; letting DuckDB infer them
            # from the zero-padded directory names (week=01) returns strings.
            glob = entry.as_posix() + "/**/part.parquet"
            con.execute(
                f"CREATE OR REPLACE VIEW {entry.name} AS "
                f"SELECT * FROM read_parquet('{glob}', hive_partitioning=false)"
            )
        elif entry.is_file() and entry.suffix == ".parquet":
            # Unpartitioned table.
            path = entry.as_posix()
            con.execute(
                f"CREATE OR REPLACE VIEW {entry.stem} AS "
                f"SELECT * FROM read_parquet('{path}')"
            )
    return con


def query(root: Path, sql: str) -> pd.DataFrame:
    """Run a SQL query against the parquet views under `root`. Returns a pandas
    DataFrame. Connection is opened/closed per call (cheap; in-memory)."""
    con = _connect_with_views(root)
    try:
        return con.execute(sql).fetchdf()
    finally:
        con.close()
