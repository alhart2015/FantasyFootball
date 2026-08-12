"""Streamlit live draft board (Draft Assistant Slice 3).

Thin view over projections.draft.assistant.live.LiveDraftSession. Run with:
    streamlit run scripts/draft_board.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from projections.draft.assistant.availability_loader import load_store_availability
from projections.draft.assistant.league_projection import N_BYES, PLAYOFF_SIZE
from projections.draft.assistant.live import (
    BOARD_STRATEGIES,
    MC_STRATEGIES,
    LiveDraftSession,
    attach_names,
    build_session_strategy,
)
from projections.draft.assistant.pick_timing import slot_for
from projections.draft.assistant.presets import (
    DEFAULT_SCORING,
    DEFAULT_TEAMS,
    SCORING_KEYS,
    TEAM_SIZES,
    get_preset,
    materialize_league_config,
)
from projections.draft.league_config import LeagueConfig
from projections.schemas import _PYARROW_STR, IdMapSchema, Position, VorpTableSchema

_DEFAULT_ID_MAP = "data/raw/id_map.parquet"


def _load_inputs(vorp_path: Path, id_map_path: Path, league: LeagueConfig):  # type: ignore[no-untyped-def]
    id_map = IdMapSchema.validate(pd.read_parquet(id_map_path))
    pool = pd.read_parquet(vorp_path)
    pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)
    pool = VorpTableSchema.validate(pool)
    return id_map, pool, league


def _build_session(
    *,
    vorp_path: Path,
    id_map_path: Path,
    league: LeagueConfig,
    my_slot: int,
    mode: str,
    strategy_name: str,
    n_sims: int,
    adp_jitter: float,
    season: int,
    data_root: Path,
    league_config_path: Path = Path("."),
) -> LiveDraftSession:
    id_map, pool, league = _load_inputs(vorp_path, id_map_path, league)
    availability = None
    if strategy_name in MC_STRATEGIES:
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
        league_config_path=league_config_path,
        vorp_path=vorp_path,
        id_map_path=id_map_path,
    )


def _install_session(sess: LiveDraftSession, *, autosave_path: Path | None = None) -> None:
    """Place a session in session_state with a fresh unique token.

    The token (not id(sess), which Python recycles after GC) keys both the recommendation
    cache and the autosave filename, so a restart/resume never collides with a prior
    draft. `autosave_path=None` starts a new autosave file; resume passes the resumed file.
    """
    st.session_state["session"] = sess
    st.session_state["session_token"] = uuid.uuid4().hex
    st.session_state["autosave_path"] = autosave_path


def _sidebar() -> None:
    st.sidebar.header("⚙ Setup")
    mode = st.sidebar.radio(
        "Mode",
        ["copilot", "mock"],
        index=0,
        format_func=lambda m: "Co-pilot (live)" if m == "copilot" else "Mock",
    )
    scoring = st.sidebar.selectbox(
        "Scoring",
        SCORING_KEYS,
        index=SCORING_KEYS.index(DEFAULT_SCORING),
        format_func=lambda k: {"half": "Half-PPR", "ppr": "Full PPR", "std": "Standard"}[k],
    )
    n_teams = st.sidebar.selectbox("Teams", TEAM_SIZES, index=TEAM_SIZES.index(DEFAULT_TEAMS))
    preset = get_preset(scoring, int(n_teams))
    my_slot = st.sidebar.number_input("My draft slot", min_value=1, max_value=int(n_teams), value=1)
    id_map_path = st.sidebar.text_input("id_map parquet", _DEFAULT_ID_MAP)
    with st.sidebar.expander("Advanced: custom VORP table"):
        custom_vorp = st.text_input("VORP parquet (overrides preset)", "")
    vorp_path = custom_vorp.strip() or str(preset.table_path)
    strategy_name = st.sidebar.selectbox("Strategy", BOARD_STRATEGIES, index=0)
    n_sims = st.sidebar.number_input(
        "n_sims (MC strategies)", min_value=50, max_value=2000, value=300, step=50
    )
    adp_jitter = st.sidebar.slider("ADP jitter", 0.0, 20.0, 8.0, 0.5)
    season = st.sidebar.number_input("Season", min_value=2020, max_value=2030, value=2026)

    if st.sidebar.button("Start / restart draft", type="primary"):
        try:
            # Persist the preset's in-memory config to a real file so autosave/resume (which
            # store the league config as a PATH) work for preset-started drafts.
            league_config_path = materialize_league_config(preset)
            _install_session(
                _build_session(
                    vorp_path=Path(vorp_path),
                    id_map_path=Path(id_map_path),
                    league=preset.league_config,
                    league_config_path=league_config_path,
                    my_slot=int(my_slot),
                    mode=mode,
                    strategy_name=strategy_name,
                    n_sims=int(n_sims),
                    adp_jitter=float(adp_jitter),
                    season=int(season),
                    data_root=Path("data"),
                )
            )
        except FileNotFoundError as exc:
            st.sidebar.error(
                f"Missing VORP table {vorp_path}. Generate presets with: "
                f"python scripts/generate_preset_vorp_tables.py — ({exc})"
            )
        except Exception as exc:  # surface any other setup failure
            st.sidebar.error(f"Setup failed: {exc}")

    _resume_controls()


def _status_bar(s: LiveDraftSession) -> None:
    if s.is_complete:
        st.subheader("✅ Draft complete")
        return
    rnd, slot = s.round_and_slot()  # slot == on_clock_slot (the on-the-clock team)
    who = "YOU" if s.is_my_pick else f"Team {slot}"
    nxt = s.next_pick_number
    until = "" if nxt is None else f" · your next pick: #{nxt}"
    # Label is Round.PickInRound (counts up every round, snake-direction-independent); the
    # on-clock team slot is shown separately so the snake reversal is still visible.
    st.subheader(
        f"Pick {rnd}.{s.pick_in_round:02d} (#{s.current_pick}) · on the clock: {who}{until}"
    )


_SESSION_DIR = Path("data/draft_sessions")


def _autosave(s: LiveDraftSession) -> None:
    path = st.session_state.get("autosave_path")
    if path is None:
        # Filename keyed on the per-session token (unique via uuid), NOT id(s) — Python
        # recycles addresses after GC, so an id()-named file could clobber a prior draft.
        token = st.session_state.get("session_token", "session")
        path = _SESSION_DIR / f"session_{token}.json"
        st.session_state["autosave_path"] = path
    s.save(Path(path))


def _resume_controls() -> None:
    if not _SESSION_DIR.exists():
        return
    saves = sorted(
        _SESSION_DIR.glob("session_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not saves:
        return
    newest = saves[0]
    st.sidebar.divider()
    st.sidebar.caption(f"Resume autosave: {newest.name}")
    if st.sidebar.button("↩ Resume last draft"):
        try:
            data = json.loads(newest.read_text())  # json/pandas/schemas imported at module top
            id_map = IdMapSchema.validate(pd.read_parquet(data["id_map"]))
            pool = pd.read_parquet(data["vorp_table"])
            pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)
            pool = VorpTableSchema.validate(pool)
            loaded = LiveDraftSession.load(newest, id_map=id_map, pool=pool)
            _install_session(loaded, autosave_path=newest)
        except Exception as exc:  # surface any resume failure to the user
            st.sidebar.error(f"Resume failed: {exc}")


@st.cache_data(show_spinner=False)
def _cached_projection(session_token: str, picks: tuple[str, ...], n_sims: int) -> dict:  # type: ignore[type-arg]
    """Cache the projected-league eval on (session, picks, n_sims). Pulls the live session +
    an optional injected availability from session_state (tests inject; prod loads from store)."""
    s: LiveDraftSession = st.session_state["session"]
    avail = st.session_state.get("_eval_availability")
    res = s.project_league_outcomes(n_sims=n_sims, availability=avail)
    return {slot: vars(proj) for slot, proj in res.items()}


@st.cache_data(show_spinner=False)
def _cached_recommendation(
    session_token: str, picks: tuple[str, ...], strategy_name: str, n_sims: int, sigma: float | None
) -> pd.DataFrame:
    """Cache MC recommendations on every result-affecting param.

    `session_token` (a per-session uuid in session_state, NOT id(session) — which Python
    recycles after GC) binds the cache to the current draft, so a restart/resume with the
    same picks/strategy/n_sims/sigma recomputes against the new pool instead of serving a
    stale entry. The body pulls the live session from session_state on a cache miss.
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


def _selectable(named: pd.DataFrame, cols: list[str], key: str) -> None:
    """Render a single-row-selectable table; selecting a row stages `pending_pick`.

    `named` must carry `gsis_id` with a clean 0..n-1 index (selection rows are positional).
    The on_select callback (a closure over this render's frame + key) fires only for the
    pane the user clicked, so across the two selectable panes the most-recent click wins.
    """
    show = [c for c in cols if c in named.columns]

    def _stage() -> None:
        state = st.session_state.get(key)
        rows = state["selection"]["rows"] if state else []
        if rows:
            st.session_state["pending_pick"] = str(named.iloc[rows[0]]["gsis_id"])

    st.dataframe(
        named[show],
        height=400,
        hide_index=True,
        selection_mode="single-row",
        on_select=_stage,
        key=key,
    )


def _confirm_bar(s: LiveDraftSession) -> None:
    """Shared 'Confirm pick' / 'clear' controls for the staged selection (`pending_pick`)."""
    pending = st.session_state.get("pending_pick")
    if not pending:
        return
    name = s.name(str(pending))
    confirm_col, clear_col = st.columns([4, 1])
    if confirm_col.button(f"✅ Confirm pick: {name}", key="confirm_pending", type="primary"):
        try:
            s.record_pick(str(pending))
        except ValueError as exc:  # already drafted, or my-pick rookie absent from id_map
            st.warning(str(exc))
            st.session_state["pending_pick"] = None  # don't let a rejected selection linger
            return
        st.session_state["pending_pick"] = None
        _autosave(s)
        st.rerun()
    if clear_col.button("✕ clear", key="clear_pending"):
        st.session_state["pending_pick"] = None
        st.rerun()


def _board_log_col(s: LiveDraftSession) -> None:
    st.markdown("**Board / pick log**")
    rows = []
    for i, gid in enumerate(s.picks):
        pick_no = i + 1
        owner = slot_for(pick_no, s.league.n_teams)  # slot_for imported at module top
        rows.append(
            {
                "#": pick_no,
                "slot": owner,
                "player": s.name(gid),
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
            name = s.name(str(sug))
            st.info(f"Opponent on the clock. ADP suggests: **{name}**")
            if st.button(f"Confirm pick: {name}", key="confirm_adp", type="primary"):
                _record_and_rerun(s, str(sug))
        st.caption("…or click a player in **Best available** below to record a different pick.")
        return
    with st.spinner("Scoring candidates…"):
        token = st.session_state.get("session_token", "")
        rec = _cached_recommendation(token, tuple(s.picks), s.strategy_name, s.n_sims, s.sigma)
    named = attach_names(rec, s.player_names).reset_index(drop=True)
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
    st.caption("Click a row to stage it, then **Confirm pick** above.")
    _selectable(named, cols, key=f"sel_rec_{s.current_pick}")


def _roster_col(s: LiveDraftSession) -> None:
    st.markdown("**My Roster**")
    view = s.my_roster_view()
    st.dataframe(view.filled[["slot", "full_name", "position"]], hide_index=True)
    if view.open_slots:
        st.caption(
            "Open starting slots: "
            + ", ".join(f"{slot.value} x{n}" for slot, n in view.open_slots.items())
        )


def _best_available_col(s: LiveDraftSession) -> None:
    st.markdown("**🔎 Best available**")
    if s.is_complete:
        return
    pos_label = st.selectbox(
        "Position", ["All", "QB", "RB", "WR", "TE"], key=f"ba_pos_{s.current_pick}"
    )
    query = st.text_input("Search player", key=f"ba_query_{s.current_pick}", placeholder="name…")
    position = None if pos_label == "All" else Position(pos_label)
    avail = s.available_for_pick(position=position, query=query, top=60)
    if avail.empty:
        st.caption("No matching available players.")
        return
    st.caption("Click a row to stage it, then **Confirm pick** above.")
    _selectable(
        avail, ["full_name", "position", "vorp", "consensus_adp"], key=f"sel_ba_{s.current_pick}"
    )


def _mock_controls(s: LiveDraftSession) -> None:
    if s.mode != "mock":
        return
    if s.is_complete:
        st.success(f"Mock complete — your optimal-lineup score: **{s.roster_scorecard():.1f}**")
        return
    if not s.is_my_pick and st.button("⏭ Advance to my pick", type="secondary"):
        s.mock_advance_to_my_pick()
        _autosave(s)
        st.rerun()


def _results_section(s: LiveDraftSession) -> None:
    if not s.is_complete:
        return
    st.markdown("### 📊 Projected draft results")
    st.caption(
        "Projected-vs-projected league sim (injury + performance variance, optimal lineup all "
        "teams). Measures roster quality under our projections, not real outcomes."
    )
    n_sims = 2000
    if not st.button("Run projected eval", key="run_projected_eval", type="primary"):
        return
    token = st.session_state.get("session_token", "")
    with st.spinner("Simulating seasons…"):
        res = _cached_projection(token, tuple(s.picks), n_sims)
    n = s.league.n_teams
    me = res[s.my_slot]
    base = {"playoff": PLAYOFF_SIZE / n, "bye": N_BYES / n, "champ": 1 / n}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reg-season win%", f"{me['reg_win_pct']:.1%}", "vs 50%")
    c2.metric("Make playoffs", f"{me['make_playoffs_pct']:.1%}", f"vs {base['playoff']:.1%}")
    c3.metric("First-round bye", f"{me['bye_pct']:.1%}", f"vs {base['bye']:.1%}")
    c4.metric("Championship %", f"{me['champ_pct']:.1%}", f"vs {base['champ']:.1%}")
    rows = [
        {
            "slot": slot,
            "you": "★" if slot == s.my_slot else "",
            "reg win%": f"{res[slot]['reg_win_pct']:.0%}",
            "playoff%": f"{res[slot]['make_playoffs_pct']:.0%}",
            "bye%": f"{res[slot]['bye_pct']:.0%}",
            "champ%": f"{res[slot]['champ_pct']:.0%}",
        }
        for slot in range(1, n + 1)
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Draft Board", layout="wide")
    st.title("🏈 Live Draft Board")
    _sidebar()

    s: LiveDraftSession | None = st.session_state.get("session")
    if s is None:
        st.info("Configure the draft in the sidebar and click **Start / restart draft**.")
        return
    _status_bar(s)
    _confirm_bar(s)
    _mock_controls(s)
    left, center, right = st.columns([1.0, 2.5, 1.1])
    with left:
        _board_log_col(s)
    with center:
        _recommend_col(s)
        _best_available_col(s)
    with right:
        _roster_col(s)
    _results_section(s)
    st.caption(f"Mode: {s.mode} · strategy: {s.strategy_name} · {len(s.picks)} picks made")


if __name__ == "__main__":
    main()
