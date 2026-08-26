"""Season dashboard — a read-only Flask UI over the projections the repo already computes.

Separate from the Streamlit boards in `scripts/` on purpose: those are interactive session
tools (live draft, live auction), this is a dashboard over batch-computed data. See
`docs/superpowers/specs/2026-08-26-season-web-ui-design.md`.
"""

from projections.web.app import DashboardConfig, create_app, dashboard_config

__all__ = ["DashboardConfig", "create_app", "dashboard_config"]
