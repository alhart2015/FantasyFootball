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
    build_session_strategy,
)
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


def main() -> None:
    st.set_page_config(page_title="Draft Board", layout="wide")
    st.title("🏈 Live Draft Board")
    _sidebar()

    s: LiveDraftSession | None = st.session_state.get("session")
    if s is None:
        st.info("Configure the draft in the sidebar and click **Start / restart draft**.")
        return
    _status_bar(s)
    st.caption(f"Mode: {s.mode} · strategy: {s.strategy_name} · {len(s.picks)} picks made")


if __name__ == "__main__":
    main()
