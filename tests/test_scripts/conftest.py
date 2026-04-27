"""Add scripts/ to sys.path so test_diagnose_calibration can import the
script as a module. Phase 0 of Plan 3e keeps the diagnostic outside
src/projections/, so we wire imports here rather than via pyproject."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
