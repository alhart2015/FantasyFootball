# FantasyFootball

Probabilistic NFL fantasy football toolkit. The repo is decomposed into sub-projects that share a common projection core: a typed engine that produces per-player, per-week distributions over fantasy points. Downstream sub-projects (Draft Hub, Mid-season Manager, DFS Engine) consume the core and add domain-specific decisions on top.

Status: the Projections Core foundations layer is in place (schemas, distributions, scoring, parquet + DuckDB store, first ingest path). Next: ingest expansion, per-position features, Model A baseline, backtest harness, public API. See `project_management.md` for current status and the decision log.

## Where to look

- **Contributing:** `CONTRIBUTING.md` — setup, daily commands, workflow, pattern recipes.
- **Claude Code instructions:** `CLAUDE.md` — auto-loaded conventions for AI-assisted development.
- **Current status & next actions:** `project_management.md`.
- **Open items:** [GitHub issues](https://github.com/alhart2015/FantasyFootball/issues) — labeled by sub-project (`draft-hub`, `auction`, `dfs`, `mid-season`, `projections-core`, `infra`).
- **Designs:** `docs/superpowers/specs/`.
- **Implementation plans:** `docs/superpowers/plans/`.
