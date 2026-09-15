# FantasyFootball

Probabilistic NFL fantasy football toolkit. The repo is decomposed into sub-projects that share a common projection core: a typed engine that produces per-player, per-week distributions over fantasy points. Downstream sub-projects (Draft Hub, Mid-season Manager, DFS Engine) consume the core and add domain-specific decisions on top.

Status: see the [GitHub issues](https://github.com/alhart2015/FantasyFootball/issues) — they are the source of truth for what is open, labeled by sub-project.

## Where to look

- **Contributing:** `CONTRIBUTING.md` — setup, daily commands, workflow, pattern recipes.
- **Claude Code instructions:** `CLAUDE.md` — auto-loaded conventions for AI-assisted development.
- **Current status & open items:** [GitHub issues](https://github.com/alhart2015/FantasyFootball/issues) — labeled by sub-project (`draft-hub`, `auction`, `dfs`, `mid-season`, `projections-core`, `infra`).
- **Designs:** `docs/superpowers/specs/`.
- **Implementation plans:** `docs/superpowers/plans/`.
