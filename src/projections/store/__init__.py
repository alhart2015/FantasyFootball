"""Store — parquet partitioned reads/writes plus DuckDB views."""

from __future__ import annotations

from projections.store.duckdb_views import query
from projections.store.parquet import delete_partition, read_partition, write_partition

__all__ = ["delete_partition", "query", "read_partition", "write_partition"]
