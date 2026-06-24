"""Shared player-identity helpers for external-projection ingest and the consensus blend.

`placeholder_name_key` is the single source of truth for the normalized (name, position)
key that reconciles the same rookie across sources (and seeds the deterministic placeholder
gsis_id when a player is not yet in id_map). Ingest and any downstream cross-source matching
import it from here so they agree by construction rather than re-deriving the rule.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

# Generational suffixes dropped from the identity key (Jr/Sr/II/III/IV/V).
NAME_SUFFIXES: frozenset[str] = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})


def placeholder_name_key(full_name: str, position: str) -> str:
    """Normalize (full_name, position) into a stable cross-source key: accents folded to ASCII
    (so 'José'/'Jose' agree across sources), lowercased, punctuation/whitespace removed, common
    generational suffixes (Jr/Sr/II…) dropped. ESPN and Sleeper spell the same rookie nearly
    identically, so this lets both sources' rows reconcile."""
    folded = (
        unicodedata.normalize("NFKD", full_name).encode("ascii", "ignore").decode("ascii").lower()
    )
    tokens = [t for t in re.split(r"[^a-z0-9]+", folded) if t and t not in NAME_SUFFIXES]
    if tokens:
        return "".join(tokens) + "|" + position.lower()
    # Degenerate name (all suffix/punctuation, or non-ASCII that folded to nothing): key on the raw
    # name instead, so two such distinct players don't both collapse to the position-only key
    # '|<pos>' and collide into one placeholder gsis.
    return full_name.strip().lower() + "|" + position.lower()


def normalize_join_id(s: pd.Series) -> pd.Series:
    """Canonicalize a platform-id column for joining against `id_map`.

    `id_map` stores espn_id/sleeper_id float-stringified ('4374302.0'); external
    pulls write clean int-strings ('4374302'). Cast both sides to a plain string
    with surrounding whitespace and any trailing '.0'(/'.00'...) stripped, so the
    merge matches and dtypes line up. Without this the join silently yields ZERO
    matches (TODO #38 — the deeper fix is casting id_map's id columns to Int64
    at ingest).
    """
    return s.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
