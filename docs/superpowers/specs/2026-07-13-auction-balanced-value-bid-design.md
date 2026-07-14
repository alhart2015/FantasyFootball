# Auction `BalancedValueBid` strategy — design (2026-07-13)

**Branch:** `feat/auction-balanced-value` (off `main` @ `926eb93`)
**Status:** Slice 1 of the "beat-the-bots" auction work. Data-gathering project — **no bid-model winner declared** (the decision is September 2026).
**Source investigation:** this session's all-bot diagnostic + `BalancedValueBid` prototype, re-baselined on `main`. Recorded in memory `auction-bid-model-investigation-status` and PM 2026-07-13.

## Motivation

The auction bid-model bake-off (`reports/auction_tournament_validation_2026.md`) found that **every existing hero strategy finishes at or below the uniform baseline** against the realistic mixed bot field. A this-session investigation root-caused *why* and found the fix:

- **Championships are ≈ +0.91 correlated with total projected points**, and points come from a **balanced, broad roster — not stars-and-scrubs.** Spend concentration *hurts* (`corr(champ, top2_share) ≈ −0.65`, `corr(champ, max_price) ≈ −0.64`); champ% peaks at **exactly one** elite rostered.
- Our existing heroes lose two ways: the value-disciplined ones (`static`/`inflation`/`marginal`/`patient`) bid ≈ fair value and get **out-bid on every contested player** (win nothing good); the concentration ones (`anchors`/`overbid`/`studsdepth`/`vorpshare`) **over-invest in a few studs and starve the other starting slots.**
- The winning policy — validated as the `Balanced` bot archetype (≈25% champ as a field member) — is: **bid a small premium over fair value (so you actually win players) but cap per-player spend at ≈2× your even per-slot share (so the budget spreads into a full roster).**
- **`_budget_urgency` is counterproductive for a well-constructed roster:** it escalates late bids up to `1 + URGENCY_GAIN` (= 4×) and dumps idle cash on replacement-level players.

**Re-baseline on `main`** (12-team half-PPR, 150 seeds × 300 sims, model bot prices): a `BalancedValueBid` hero (premium 0.15 / cap 2×, no urgency) is the **best hero** — playoff **45.6%** / champ **5.8%**, CI-separated above `vorpshare` (36.7% / 4.8%) and far above `main`'s `patient_deep` (19.3% / 0.7%); urgency-on craters the same strategy (21% / 0.9%); `anchors` is worst (5.9% / 0.3%). It reaches **≈baseline but not clearly above** — `main`'s snake-bot field is a tough room — so **nomination warfare (Slice 2) is the lever to *exceed* baseline.** This slice gets the *bidder* right first.

## Goal

Ship `BalancedValueBid` as a new, selectable auction bid-model contestant implementing the empirically-winning policy. It should become the strongest hero in the bake-off.

## Non-goals

- **No engine changes** (nomination, clearing, snake-bot) — Slice 2 covers hero-controlled nomination.
- **No changes to any existing strategy** — their `_budget_urgency` behavior and recorded-experiment numbers (Runs A–F) stay byte-identical. (Whether urgency is self-harm for the existing models too is a separate, later slice.)
- **No ESPN-price work** — the 2026 external snapshot has no ESPN auction values (`--bot-prices espn` falls back to model); revisit when ESPN auction data publishes.
- **No live-board / assistant-CLI wiring** in this slice (an optional thin fast-follow).

## Design

### The strategy — `src/projections/draft/assistant/auction/bid_strategy.py`

```python
@dataclass(frozen=True)
class BalancedValueBid:
    """Balanced-breadth hero: bid a small premium over fair value to win contested players,
    capped at `pace` x the even per-slot share so the budget spreads into a full roster.
    Deliberately does NOT apply _budget_urgency (the ramp over-pays late-round scrubs)."""

    premium: float = 0.15
    pace: float = 2.0

    def __post_init__(self) -> None:
        if not (self.premium >= 0.0 and math.isfinite(self.premium)):
            raise ValueError(f"premium must be finite and >= 0; got {self.premium}")
        if not (self.pace > 0.0 and math.isfinite(self.pace)):
            raise ValueError(f"pace must be finite and > 0; got {self.pace}")

    def max_bid(
        self, view: AuctionView, player: pd.Series, pool: pd.DataFrame, config: LeagueConfig
    ) -> int:
        fair = float(view.baseline_dollars.loc[player["gsis_id"], "auction_dollars"])
        cap = self.pace * (view.my_budget / max(1, view.my_open_slots))
        return round(min(fair * (1.0 + self.premium), cap))
```

- Reads `auction_dollars` (the hero's SOS value anchor), exactly as `StaticDollarBid` does — the hero anchors on model value, independent of the bots' `bot_dollars` seam.
- `premium` and `pace` are dataclass fields (tunable / sweepable); defaults from the validated plateau (premium 0.15–0.25, pace 1.5–2×).
- `max(1, view.my_open_slots)` makes the divide total (open_slots is ≥ 1 whenever the hero bids, but the guard removes the edge case).
- Returns an **unclamped desired bid**; the engine clamps to `[min_bid, feasible_max]` per the module contract — no reserve/floor re-implementation.
- **Deliberately omits `_budget_urgency`** — the single most load-bearing design choice, justified by the re-baseline (urgency-on craters the strategy). This is a per-strategy choice; no existing strategy is touched.
- Guarded by `__post_init__` (fail-loud on bad tuning), matching the existing `NowOrNeverFlooredStrategy` guard style. Requires adding `import math` to `bid_strategy.py` (not currently imported) for the finiteness checks.

### Registration

- Add `"balanced": BalancedValueBid()` to `tournament_cli._MODELS` so the `compare` subcommand races it against the field. This makes it the **tenth** contestant.
- **Update the contestant-set guard test:** `tests/test_draft/test_assistant_auction_tournament_cli.py::test_default_models_are_the_nine_contestants` pins `set(_MODELS)` to the current nine and **will fail** on the new key — update it to include `"balanced"` (ten contestants) and rename it accordingly (e.g. `..._the_ten_contestants`). `test_every_default_model_satisfies_the_protocol` then also covers `balanced` automatically (it iterates `_MODELS.values()`; no change needed there).
- **Deferred:** live-board `BOARD_STRATEGIES` / assistant-CLI wiring (a thin fast-follow; not needed to gather the bake-off data).

## Testing — `tests/test_draft/test_assistant_auction_bid_strategy.py`

- **Premium wins contested value:** a mid-tier player under the cap → bid `== round(fair × (1 + premium)) > fair` (out-bids a fair-value bidder).
- **Cap forces spread:** a stud whose `fair × (1 + premium)` exceeds the cap → bid `== round(cap) < fair` (refuses to blow out on one player).
- **Cap tracks remaining budget/slots:** the cap recomputes from `my_budget` / `my_open_slots` as the roster fills (two states, different caps).
- **No urgency ramp (the contract):** at fixed `my_budget` / `my_open_slots`, the bid is invariant to roster-fill *progress* — pins that `_budget_urgency` is not applied (contrast: an urgency-applying strategy's bid would rise).
- **Determinism:** same inputs → same bid (frozen dataclass, no RNG).
- **Validation:** `__post_init__` rejects negative `premium`, non-positive `pace`, and non-finite values.
- **Engine integration:** a `simulate_auction` smoke with a `BalancedValueBid` hero completes and fills a legal roster; a `run_auction_tournament` smoke includes `balanced` in the summary keys.

## Validation — "Run G" (a fresh bake-off of the *shipped* contestant)

The this-session re-baseline used a throwaway prototype script; the acceptance run must exercise the **shipped** `balanced` contestant via `main`'s CLI. After registration, run `scripts/auction_tournament.py compare` (12-team half, seat 1, 150 seeds × 300 sims, `--bot-prices model`, `nomination_temp=1.0`) and record it as **Run G** in `reports/auction_tournament_validation_2026.md`.

- **Pre-registered expectation** (from the prototype): `balanced` is the top hero — playoff ≈ 45–46% / champ ≈ 5–6%, CI-separated on playoff% above `vorpshare`, and beating `patient_deep`; `anchors` last. A materially different result is a signal the shipped strategy diverged from the prototype and must be reconciled before recording.
- **No winner declared** (Sept decision) — Run G is a data point, not an adoption call.
- The urgency-off-vs-on evidence stays in the prototype notes; this slice does not add an urgency arm to the shipped contestants (per Non-goals).

## Phasing

Single slice — one strategy + `__post_init__` guard + unit tests + one-line CLI registration + a bake-off smoke. Small enough for one implementation pass. Slice 2 (hero-controlled nomination) is a separate spec.
