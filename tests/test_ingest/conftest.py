"""Ingest-test conftest.

The shared `fake_*_df` fixtures (raw `nfl_data_py` response mocks) are
defined in `tests/conftest.py` so the top-level smoke test can request
them too. Pytest hierarchical fixture resolution makes them available
to every test under this directory unchanged.
"""

from __future__ import annotations
