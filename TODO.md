# TODO

Running project management list. Add items as they come up; remove or check off when resolved.

## Open

### 0. Pick lint/format config and write CONTRIBUTING.md

**Context.** Typing posture (pandera schemas, pydantic models, NewType IDs, mypy --strict, enums for reused values) is committed in the projections-core spec. What still needs deciding is the surrounding ergonomic config:
- `ruff` rule set + line length + import sort
- `pyproject.toml` layout (single package vs src layout)
- pre-commit hooks
- CONTRIBUTING.md describing the workflow (test commands, type-check command, how to add a schema)

**Definition of done.** `pyproject.toml`, `.pre-commit-config.yaml`, and `CONTRIBUTING.md` checked in. Doesn't gate the projections-core implementation plan but should land alongside the first PR so conventions are enforced from commit one.

### 1. Explore option D: joint-correlation projections

**Context.** During Projections Core brainstorming we picked option C (full per-player distributions, marginal only). Option D would extend C to model how player outcomes *co-move* — same-game stacks, opponent dependencies, game-script effects. We deferred D because it adds storage and modeling complexity we may not need until DFS tournament work; we want C's schema to make D an additive upgrade rather than a rewrite.

**Why it matters.**
- DFS GPPs (top-heavy tournaments) live and die on correlated ceilings; an uncorrelated "stack" model dramatically underestimates QB+WR1 joint upside.
- Cash-game DFS and start/sit decisions can survive on marginal distributions alone.
- Season-long draft and waiver tools mostly want means and ranks; correlations are nice-to-have, not load-bearing.

**Questions to answer when we explore.**
- *Scope:* which correlations actually move the needle? Likely candidates, in priority order:
  - Same-team QB ↔ pass-catchers (typically ρ ≈ 0.4–0.6 for QB↔WR1)
  - Same-game opposing players (shootouts lift everyone)
  - RB ↔ team defense (negative; if you allowed the opposing RB to score, your D suffers)
  - Weather and pace shared across a game
- *Modeling approaches:*
  - Empirical covariance matrix from historical weekly fantasy points (simple, but noisy and assumes stationarity)
  - Scenario / Monte Carlo from simulated game states (richer, much more code — could lean on `nflfastR` win-probability and play-type models)
  - Factor model: shared "game environment" latent variable (pace, total) plus player-specific noise (compromise)
  - Gaussian copula on marginals from C (clean separation: marginals stay as in C, dependence lives in the copula)
- *Storage:* covariance matrices per slate/week are O(N²); scenario tables are O(N · S) for S draws. Need to pick one before DFS optimizer work.
- *Optimizer interface:* most ILP optimizers accept point projections + ownership; correlated upside requires either a sim-based optimizer or a stacking-rule heuristic on top of ILP. Decide which path.
- *Validation:* how do we measure that correlated projections beat uncorrelated? Backtest against historical DK/FD GPP results — compare uncorrelated lineups vs correlated lineups by realized payout percentile, not just RMSE.

**Inputs / references to gather.**
- `nfl_data_py` play-by-play features useful for game-script modeling (EPA, pace, success rate, win prob).
- Historical DK/FD slate salaries + ownership (for backtest target).
- Existing OSS work: `pydfs-lineup-optimizer`, `pulp`/`cvxpy`-based optimizers, any public correlation matrices.
- Blog/academic refs: RotoGrinders/Fantasy Labs on stacking; any published papers on DFS lineup construction under correlation.

**Definition of done for this exploration.**
A short written recommendation: pick one modeling approach (covariance / scenario sim / factor / copula), one storage format, and a concrete API addition to the C-era projections schema. Include a backtest plan so we know whether D is actually paying off before we commit to building it.
