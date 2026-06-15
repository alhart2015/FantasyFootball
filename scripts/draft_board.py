"""Streamlit live draft board (Draft Assistant Slice 3).

Thin view over projections.draft.assistant.live.LiveDraftSession. Run with:
    streamlit run scripts/draft_board.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.live import (
    BOARD_STRATEGIES,
    LiveDraftSession,
    attach_names,
    build_session_strategy,
)
from projections.draft.assistant.pick_timing import slot_for
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, IdMapSchema, VorpTableSchema

_DEFAULT_VORP = "data/consensus_vorp_2026.parquet"
_DEFAULT_ID_MAP = "data/raw/id_map.parquet"
_DEFAULT_LEAGUE = "configs/league_espn_ppr_12team_skill.json"

_MC_STRATEGIES = ("season_value", "season_value_timing")


def _load_inputs(vorp_path: Path, id_map_path: Path, league_path: Path):  # type: ignore[no-untyped-def]
    id_map = IdMapSchema.validate(pd.read_parquet(id_map_path))
    pool = pd.read_parquet(vorp_path)
    pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)
    pool = VorpTableSchema.validate(pool)
    league = LeagueConfig.model_validate_json(Path(league_path).read_text())
    return id_map, pool, league


def _build_session(
    *,
    vorp_path: Path,
    id_map_path: Path,
    league_path: Path,
    my_slot: int,
    mode: str,
    strategy_name: str,
    n_sims: int,
    adp_jitter: float,
    season: int,
    data_root: Path,
) -> LiveDraftSession:
    id_map, pool, league = _load_inputs(vorp_path, id_map_path, league_path)
    availability = None
    if strategy_name in _MC_STRATEGIES:
        availability = load_store_availability(pool, season=season, data_root=data_root)
    strategy = build_session_strategy(
        strategy_name,
        league=league,
        sigma=None,
        availability=availability,
        n_sims=n_sims,
        base_seed=0,
    )
    return LiveDraftSession(
        league=league,
        my_slot=my_slot,
        id_map=id_map,
        pool=pool,
        strategy=strategy,
        strategy_name=strategy_name,
        mode=mode,  # type: ignore[arg-type]
        adp_jitter=adp_jitter,
        n_sims=n_sims,
        season=season,
        league_config_path=Path(league_path),
        vorp_path=vorp_path,
        id_map_path=id_map_path,
        data_root=data_root,
    )


def _sidebar() -> None:
    st.sidebar.header("⚙ Setup")
    mode = st.sidebar.radio(
        "Mode",
        ["copilot", "mock"],
        index=0,
        format_func=lambda m: "Co-pilot (live)" if m == "copilot" else "Mock",
    )
    vorp_path = st.sidebar.text_input("Consensus VORP parquet", _DEFAULT_VORP)
    id_map_path = st.sidebar.text_input("id_map parquet", _DEFAULT_ID_MAP)
    league_path = st.sidebar.text_input("League config JSON", _DEFAULT_LEAGUE)
    my_slot = st.sidebar.number_input("My draft slot", min_value=1, max_value=32, value=1)
    strategy_name = st.sidebar.selectbox("Strategy", BOARD_STRATEGIES, index=0)
    n_sims = st.sidebar.number_input(
        "n_sims (MC strategies)", min_value=50, max_value=2000, value=300, step=50
    )
    adp_jitter = st.sidebar.slider("ADP jitter", 0.0, 20.0, 8.0, 0.5)
    season = st.sidebar.number_input("Season", min_value=2020, max_value=2030, value=2026)

    if st.sidebar.button("Start / restart draft", type="primary"):
        try:
            st.session_state["session"] = _build_session(
                vorp_path=Path(vorp_path),
                id_map_path=Path(id_map_path),
                league_path=Path(league_path),
                my_slot=int(my_slot),
                mode=mode,
                strategy_name=strategy_name,
                n_sims=int(n_sims),
                adp_jitter=float(adp_jitter),
                season=int(season),
                data_root=Path("data"),
            )
        except Exception as exc:  # surface any setup failure to the user
            st.sidebar.error(f"Setup failed: {exc}")


def _status_bar(s: LiveDraftSession) -> None:
    if s.is_complete:
        st.subheader("✅ Draft complete")
        return
    rnd, slot = s.round_and_slot()
    who = "YOU" if s.is_my_pick else f"Team {slot}"
    nxt = s.next_pick_number
    until = "" if nxt is None else f" · your next pick: #{nxt}"
    st.subheader(
        f"Pick {rnd}.{s.on_clock_slot:02d} (#{s.current_pick}) · on the clock: {who}{until}"
    )


def _autosave(s: LiveDraftSession) -> None:
    """No-op until Task 13 wires real autosave."""


@st.cache_data(show_spinner=False)
def _cached_recommendation(
    s_id: int, picks: tuple[str, ...], strategy_name: str, n_sims: int, sigma: float | None
) -> pd.DataFrame:
    """Cache MC recommendations on every result-affecting param.

    All args are hashable and part of the cache key: `s_id` (= id(session)) binds the
    cache to the current session instance, so a restarted draft (new pool) does not
    reuse another session's result; picks/strategy_name/n_sims/sigma cover the rest.
    """
    s: LiveDraftSession = st.session_state["session"]
    return s.recommendation()


def _record_and_rerun(s: LiveDraftSession, gsis_id: str) -> None:
    try:
        s.record_pick(gsis_id)
    except ValueError as exc:
        st.warning(str(exc))
        return
    _autosave(s)
    st.rerun()


def _search_box(s: LiveDraftSession) -> None:
    query = st.text_input("🔍 Record a pick — search a player", key=f"search_{s.current_pick}")
    if not query:
        return
    id_map = s.id_map
    drafted = s.state().drafted_ids
    hits = id_map[
        id_map["full_name"].str.contains(query, case=False, na=False)
        & ~id_map["gsis_id"].isin(drafted)
    ].head(8)
    for row in hits.itertuples(index=False):
        team = "" if pd.isna(row.team) else f" · {row.team}"
        if st.button(f"{row.full_name} ({row.position}{team})", key=f"pick_{row.gsis_id}"):
            _record_and_rerun(s, str(row.gsis_id))


def _board_log_col(s: LiveDraftSession) -> None:
    st.markdown("**Board / pick log**")
    names = dict(zip(s.id_map["gsis_id"], s.id_map["full_name"], strict=False))
    rows = []
    for i, gid in enumerate(s.picks):
        pick_no = i + 1
        owner = slot_for(pick_no, s.league.n_teams)  # slot_for imported at module top
        rows.append(
            {
                "#": pick_no,
                "slot": owner,
                "player": names.get(gid, "—"),
                "mine": "★" if owner == s.my_slot else "",
            }
        )
    st.dataframe(pd.DataFrame(rows), height=520, hide_index=True)


def _recommend_col(s: LiveDraftSession) -> None:
    st.markdown("**★ Recommendations**")
    if s.is_complete:
        st.success("Draft complete.")
        return
    if s.mode == "copilot" and not s.is_my_pick:
        sug = s.suggested_pick()
        if sug is not None:
            name = dict(zip(s.id_map["gsis_id"], s.id_map["full_name"], strict=False)).get(sug, sug)
            st.info(f"Opponent on the clock. ADP suggests: **{name}**")
            if st.button(f"Confirm pick: {name}", type="primary"):
                _record_and_rerun(s, str(sug))
        st.caption("…or search below to record a different pick.")
        return
    with st.spinner("Scoring candidates…"):
        rec = _cached_recommendation(id(s), tuple(s.picks), s.strategy_name, s.n_sims, s.sigma)
    named = attach_names(rec, s.id_map)
    cols = [
        "rank",
        "full_name",
        "position",
        "vorp",
        "consensus_adp",
        "p_available_next",
        "score",
        "fills_starting_slot",
    ]
    st.dataframe(named[cols].head(20), height=480, hide_index=True)


def _roster_col(s: LiveDraftSession) -> None:
    st.markdown("**My Roster**")
    view = s.my_roster_view()
    st.dataframe(view.filled[["slot", "full_name", "position"]], hide_index=True)
    if view.open_slots:
        st.caption(
            "Open starting slots: "
            + ", ".join(f"{slot.value} x{n}" for slot, n in view.open_slots.items())
        )
    st.markdown("**Best available by position**")
    best = s.best_available_by_position(top=3)
    for pos, sub in best.items():
        named = attach_names(sub, s.id_map)
        st.caption(
            f"{pos.value}: "
            + ", ".join(f"{r.full_name} ({r.vorp:.0f})" for r in named.itertuples(index=False))
        )


def main() -> None:
    st.set_page_config(page_title="Draft Board", layout="wide")
    st.title("🏈 Live Draft Board")
    _sidebar()

    s: LiveDraftSession | None = st.session_state.get("session")
    if s is None:
        st.info("Configure the draft in the sidebar and click **Start / restart draft**.")
        return
    _status_bar(s)
    _search_box(s)
    left, center, right = st.columns([1.1, 2.0, 1.3])
    with left:
        _board_log_col(s)
    with center:
        _recommend_col(s)
    with right:
        _roster_col(s)
    st.caption(f"Mode: {s.mode} · strategy: {s.strategy_name} · {len(s.picks)} picks made")


if __name__ == "__main__":
    main()
