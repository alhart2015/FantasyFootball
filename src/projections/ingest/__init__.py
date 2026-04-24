"""Ingest layer — the only module that talks to nfl_data_py."""

from __future__ import annotations

from projections.ingest.id_map import build_id_map

__all__ = ["build_id_map"]
