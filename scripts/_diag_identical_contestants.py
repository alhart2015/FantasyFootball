"""Why do `balanced`/`balanced_flat` and `patient`/`patient_deep` post identical figures? (#146)

Committed so the numbers quoted in the Run Z note are reproducible. Every other figure in that
entry is re-derivable from a chunk directory; these came from an ad-hoc run and were not.

Reports two things:

1. **Artifact agreement** — the max absolute per-seat difference between each pair, over every
   metric, in whichever chunk directories exist. This is the observation.
2. **Bid-level divergence** — how often `BalancedValueBid(non_increasing_cap=True)` actually returns
   a different bid than the default, and whether the resulting hero rosters differ. This is the
   attempted explanation, and it covers the `balanced` pair ONLY; the `patient`/`scrub_frac` half
   remains an unverified hypothesis.

Run FROM THE REPO ROOT (the shared loader reads `configs/` relative to the cwd):
    python scripts/_diag_identical_contestants.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from projections.draft.assistant.auction.bid_strategy import AuctionView, BalancedValueBid
from projections.draft.assistant.auction.registry import ALL_BID_MODELS
from projections.draft.assistant.auction.simulation import _simulate_to_state
from projections.draft.assistant.auction.tournament_cli import _load_tournament_inputs
from projections.draft.auction import build_market_dollars

_ROOT = Path(__file__).resolve().parent.parent
_PAIRS = (("balanced", "balanced_flat"), ("patient", "patient_deep"))
# Every field-mix cell, not a sample: the report claims the agreement is exact across ALL of them.
_CHUNK_DIRS = (
    *(f"reports/_field_mix/p{k}" for k in (0, 2, 3, 5, 6, 8, 11)),
    "reports/_will_bakeoff/postfix_2026",
    "reports/_will_bakeoff/jitter_2026",
)
_SEATS = (1, 6)
_SEEDS = 3


def _artifact_agreement() -> None:
    print("1. artifact agreement (max abs per-seat diff across every metric)")
    for d in _CHUNK_DIRS:
        files = sorted(glob.glob(str(_ROOT / d / "*.json")))
        if not files:
            print(f"   {d}: (absent)")
            continue
        for a, b in _PAIRS:
            diffs = []
            for f in files:
                m = json.loads(Path(f).read_text())["all_metrics"]
                if a in m and b in m:
                    shared = set(m[a]) & set(m[b])
                    if shared:  # a partial block must not abort the rest of the diagnostic
                        diffs.append(max(abs(m[a][k] - m[b][k]) for k in shared))
            if diffs:
                print(f"   {d}  {a} vs {b}: {max(diffs):.6g}  (n={len(diffs)} seats)")


def _bid_divergence() -> None:
    """The `balanced` pair only -- `PatientValueBid`'s scrub_frac is not instrumented here."""
    import sys

    sys.path.insert(0, str(_ROOT / "scripts"))
    from auction_field_bakeoff import build_field

    pool, config, _avail, _params = _load_tournament_inputs(
        _ROOT / "data/vorp_2026/will_half12.parquet",
        _ROOT / "configs/will_half12_pass5.league.json",
        season=2026,
        data_root=_ROOT / "data",
    )
    baseline, bot_dollars = build_market_dollars(pool, config, market="espn")
    # n_patient=2 reproduces the Run-Z 9/2 cell (hoarders evenly spread to seats 2 and 8). Omitting
    # it would take the historical every-5th rule (seats 4 and 9) -- the Run-Y room, which the
    # report itself measures as ~0.004 reg_win_pct away and explicitly says is not interchangeable.
    field = build_field(
        "overbidder",
        0.2,
        4.5,
        "opening",
        n_bots=config.n_teams - 1,
        pace_jitter=0.35,
        n_patient=2,
    )
    calls = differing = 0

    class _Probe(BalancedValueBid):
        def max_bid(self, view: AuctionView, player, pool_, cfg):  # type: ignore[no-untyped-def]
            nonlocal calls, differing
            plain = BalancedValueBid().max_bid(view, player, pool_, cfg)
            flat = BalancedValueBid(non_increasing_cap=True).max_bid(view, player, pool_, cfg)
            calls += 1
            differing += plain != flat
            return plain

    identical_rosters = drafts = 0
    for seat in _SEATS:
        for s in range(_SEEDS):
            rosters = []
            for strategy in (_Probe(), ALL_BID_MODELS["balanced_flat"]):
                state = _simulate_to_state(
                    strategy,
                    seat,
                    pool,
                    config,
                    baseline_dollars=baseline,
                    price_jitter=0.15,
                    rng=np.random.default_rng(s),
                    snake_rng=np.random.default_rng([s, 20260619]),
                    nomination_temp=1.0,
                    bot_archetypes=field,
                    bot_dollars=bot_dollars,
                    market_adp_jitter=12.0,
                )
                rosters.append([(g, pr) for (g, _p, pr) in state.rosters[seat - 1]])
            drafts += 1
            identical_rosters += rosters[0] == rosters[1]

    print("\n2. bid-level divergence (balanced pair only)")
    print(f"   bid calls: {calls}")
    print(f"   calls where the two configs return DIFFERENT bids: {differing}")
    print(f"   identical hero rosters: {identical_rosters}/{drafts} drafts")


if __name__ == "__main__":
    # `VarianceParams.load()` inside the shared loader reads `configs/` relative to the cwd, so a
    # run from elsewhere dies deep in pathlib with nothing tying it back to that. Say so instead.
    if not (Path.cwd() / "configs").is_dir():
        raise SystemExit(f"run this from the repo root ({_ROOT}); cwd is {Path.cwd()}")
    _artifact_agreement()
    _bid_divergence()
