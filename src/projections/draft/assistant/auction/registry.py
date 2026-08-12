"""The named auction bid models — one registry shared by every consumer.

`BID_MODELS` is the tournament roster (the models the data-gathering harness races).
`EXTRA_BID_MODELS` holds library-tested opt-ins deliberately kept OUT of the tournament
roster: the no-ramp overbid variant and the low-gain convex StackRatio family. The
bake-off script and the live auction board offer `ALL_BID_MODELS` (both sets).

Promoted out of `tournament_cli._MODELS` so the tournament, the analysis scripts, and
`auction/live.py` can't drift apart on what a strategy name means. Every consumer now imports
from here; the old private alias is gone.

The three mappings are read-only. Consumers bind the *same* object, so a `pop`/`__setitem__`
to subset one run would silently redefine the tournament roster and the live board's menu
process-wide. Take a `dict(...)` copy to build a variant.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from projections.draft.assistant.auction.bid_strategy import (
    AnchorBudgetBid,
    AuctionBidStrategy,
    BalancedValueBid,
    InflationBid,
    MarginalValueBid,
    OverbidValueBid,
    PatientValueBid,
    StackRatioBid,
    StaticDollarBid,
    StudsAndDepthBid,
    VorpShareBid,
)

_BID_MODELS: dict[str, AuctionBidStrategy] = {
    "static": StaticDollarBid(),
    "inflation": InflationBid(),
    "marginal": MarginalValueBid(),
    "anchors": AnchorBudgetBid(),
    "overbid": OverbidValueBid(),
    "vorpshare": VorpShareBid(),
    "patient": PatientValueBid(),
    # patient_deep: the scrub_frac=0 tuning — hoard mid-tier breadth (bid real value across the
    # whole non-stud pool, no $1-dumping the bottom half). The multi-year bake-off found this the
    # most era-robust hero; included as a standing contestant.
    "patient_deep": PatientValueBid(scrub_frac=0.0),
    "studsdepth": StudsAndDepthBid(),
    "balanced": BalancedValueBid(),
    # balanced_flat: the Slice 1 cap-inflation fix — same premium/pace, but a pace cap that
    # can't self-inflate as the hero wins (non_increasing_cap=True). See the robust-win-hero spec.
    "balanced_flat": BalancedValueBid(non_increasing_cap=True),
}

_EXTRA_BID_MODELS: dict[str, AuctionBidStrategy] = {
    # The bake-off winner minus its late-draft ramp: stronger AND executable from a printed
    # sheet, since the ramp is the one part a human cannot reproduce. See OverbidValueBid.
    "overbid_noramp": OverbidValueBid(use_urgency=False),
    # The low-gain convex StackRatio variants Run T resolved as the only heroes to beat
    # `balanced` in the less-circular ESPN market. Validated opt-ins, not adopted defaults.
    "sr_g0.1_c2": StackRatioBid(gain=0.1, curve=2.0),
    "sr_g0.2_c2": StackRatioBid(gain=0.2, curve=2.0),
    "sr_g0.3_c2": StackRatioBid(gain=0.3, curve=2.0),
}

BID_MODELS: Mapping[str, AuctionBidStrategy] = MappingProxyType(_BID_MODELS)
EXTRA_BID_MODELS: Mapping[str, AuctionBidStrategy] = MappingProxyType(_EXTRA_BID_MODELS)
ALL_BID_MODELS: Mapping[str, AuctionBidStrategy] = MappingProxyType(
    {**_BID_MODELS, **_EXTRA_BID_MODELS}
)

__all__ = ["ALL_BID_MODELS", "BID_MODELS", "EXTRA_BID_MODELS"]
