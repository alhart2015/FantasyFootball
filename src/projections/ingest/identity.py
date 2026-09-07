"""Shared player-identity helpers for external-projection ingest and the consensus blend.

`placeholder_name_key` is the single source of truth for the normalized (name, position)
key that reconciles the same rookie across sources (and seeds the deterministic placeholder
gsis_id when a player is not yet in id_map). Ingest and any downstream cross-source matching
import it from here so they agree by construction rather than re-deriving the rule.

`drop_placeholder_gsis_rows` is the single source of truth for the other half of that problem:
upstream rows carrying an id that is not a gsis_id at all.
"""

from __future__ import annotations

import logging
import re
import unicodedata

import pandas as pd

from projections.schemas import GSIS_ID_PATTERN

_log = logging.getLogger(__name__)

# Generational suffixes dropped from the identity key (Jr/Sr/II/III/IV/V).
NAME_SUFFIXES: frozenset[str] = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})

_GSIS_RE = re.compile(rf"^{GSIS_ID_PATTERN}$")


def drop_placeholder_gsis_rows(
    df: pd.DataFrame, *, source: str, gsis_col: str = "gsis_id"
) -> pd.DataFrame:
    """Drop rows whose `gsis_col` is null or not a canonical gsis_id, warning when any go.

    **nflverse carries PFR-style placeholder ids** — `WAS569019`, `MEN516487` — for players
    NFL.com has not yet assigned a real gsis_id to: undrafted rookies, practice-squad adds, and
    the current draft class until roughly training camp. Legacy PFR-style ids for very old
    players hit the same filter.

    Every schema keyed on `gsis_id` enforces `GSIS_ID_PATTERN`, so these rows cannot be
    persisted; the question is only whether a source drops twelve players or dies. Dropping is
    right — a placeholder id joins to nothing downstream, so keeping it would trade a loud
    failure for a silent one — but it must be **loud in the log**, or a thin partition for a
    fresh draft class becomes a diagnostic chase.

    This lives here because it was open-coded in three ingest modules and absent from a fourth,
    which is how `depth_charts` came to abort a whole season's ingest over twelve practice-squad
    players (issue #169). A fifth source will get it wrong by omission the same way; one import
    is harder to forget than six lines.
    """
    kept = df[df[gsis_col].notna()].copy()
    n_null = len(df) - len(kept)
    n_pre = len(kept)
    kept = kept[kept[gsis_col].astype(str).str.match(_GSIS_RE)].copy()
    n_placeholder = n_pre - len(kept)
    if n_placeholder:
        _log.warning(
            "%s: filtered %d row(s) with non-GSIS placeholder ids (typical of pre-camp rookies "
            "and practice-squad adds — nflverse holds PFR-style placeholders until NFL assigns "
            "real gsis_ids ~July). Re-ingest after training camps to capture these players.",
            source,
            n_placeholder,
        )
    if n_null:
        _log.debug("%s: dropped %d row(s) with a null %s.", source, n_null, gsis_col)
    return kept


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
