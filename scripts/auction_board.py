"""Streamlit live AUCTION board — the auction companion to scripts/draft_board.py.

Answers the three questions an auction actually asks, live:
  1. **How much do I bid on this player?** — the bid model's number, clamped exactly as the
     simulation engine clamps it, next to what the room is anchored on and the richest rival
     ceiling for that position.
  2. **Who do I nominate?** — the engine's own value-first rule (or a tested poison probe),
     with the shortlist and what each candidate costs you.
  3. **Who bought whom for how much?** — record every award; budgets, rosters, and every
     recommendation update off it.

Thin view over projections.draft.assistant.auction.live.LiveAuctionSession. Run with:
    streamlit run scripts/auction_board.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from projections.draft.assistant.auction.live import (
    BOARD_BID_MODEL_NAMES,
    BOARD_BID_MODELS,
    DEFAULT_BID_MODEL,
    NOMINATION_MODES,
    NOMINATION_NOTES,
    BidAdvice,
    LiveAuctionSession,
)
from projections.draft.assistant.league_projection import N_BYES, PLAYOFF_SIZE
from projections.draft.assistant.live import filter_named_pool
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
_SESSION_DIR = Path("data/auction_sessions")
_SCORING_LABELS = {"half": "Half-PPR", "ppr": "Full PPR", "std": "Standard"}


def _load_inputs(vorp_path: Path, id_map_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_map = IdMapSchema.validate(pd.read_parquet(id_map_path))
    pool = pd.read_parquet(vorp_path)
    pool["gsis_id"] = pool["gsis_id"].astype(_PYARROW_STR)
    return id_map, VorpTableSchema.validate(pool)


def _install_session(sess: LiveAuctionSession, *, autosave_path: Path | None = None) -> None:
    """Place a session in session_state with a fresh unique token (keys the autosave file)."""
    st.session_state["session"] = sess
    st.session_state["session_token"] = uuid.uuid4().hex
    st.session_state["autosave_path"] = autosave_path
    st.session_state["pending_player"] = None


def _autosave(s: LiveAuctionSession) -> None:
    path = st.session_state.get("autosave_path")
    if path is None:
        token = st.session_state.get("session_token", "session")
        path = _SESSION_DIR / f"auction_{token}.json"
        st.session_state["autosave_path"] = path
    s.save(Path(path))


VIEW_MODES = ("Full board", "Minimal (live draft)")


def _sidebar() -> None:
    st.sidebar.radio("View", VIEW_MODES, key="view_mode")
    st.sidebar.divider()
    st.sidebar.header("⚙ Setup")
    scoring = st.sidebar.selectbox(
        "Scoring",
        SCORING_KEYS,
        index=SCORING_KEYS.index(DEFAULT_SCORING),
        format_func=lambda k: _SCORING_LABELS[k],
    )
    n_teams = st.sidebar.selectbox("Teams", TEAM_SIZES, index=TEAM_SIZES.index(DEFAULT_TEAMS))
    preset = get_preset(scoring, int(n_teams))
    my_seat = st.sidebar.number_input("My seat", min_value=1, max_value=int(n_teams), value=1)
    strategy_name = st.sidebar.selectbox(
        "Bid model",
        BOARD_BID_MODEL_NAMES,
        index=BOARD_BID_MODEL_NAMES.index(DEFAULT_BID_MODEL),
    )
    st.sidebar.caption(
        "`balanced` is the robust win% leader across all seats in both markets. "
        "`overbid_noramp` is the printed cheat sheet's plan."
    )
    market = st.sidebar.radio(
        "What the room pays",
        ["espn", "model"],
        index=0,
        format_func=lambda m: "ESPN-anchored" if m == "espn" else "Our own values",
        help="ESPN-anchored prices the ROOM on real ESPN auction values (falls back to our "
        "values if the table has none). Your own max bids always come from our values.",
    )
    nomination_mode = st.sidebar.selectbox("Nomination rule", NOMINATION_MODES, index=0)
    st.sidebar.caption(NOMINATION_NOTES[nomination_mode])
    season = st.sidebar.number_input("Season", min_value=2020, max_value=2030, value=2026)
    with st.sidebar.expander("Advanced"):
        id_map_path = st.text_input("id_map parquet", _DEFAULT_ID_MAP)
        custom_vorp = st.text_input("VORP parquet (overrides preset)", "")
        custom_league = st.text_input("LeagueConfig JSON (overrides preset)", "")
        team_text = st.text_area(
            "Team names (one per line, seat 1 first)",
            "",
            help="Optional. Blank lines fall back to 'Team N'.",
        )
    vorp_path = custom_vorp.strip() or str(preset.table_path)

    if st.sidebar.button("Start / restart auction", type="primary"):
        try:
            if custom_league.strip():
                league_config_path = Path(custom_league.strip())
                league = LeagueConfig.model_validate_json(league_config_path.read_text())
            else:
                league = preset.league_config
                league_config_path = materialize_league_config(preset)
            id_map, pool = _load_inputs(Path(vorp_path), Path(id_map_path))
            _install_session(
                LiveAuctionSession(
                    league=league,
                    my_seat=int(my_seat),
                    id_map=id_map,
                    pool=pool,
                    strategy=BOARD_BID_MODELS[strategy_name],
                    strategy_name=strategy_name,
                    market=market,  # type: ignore[arg-type]
                    nomination_mode=nomination_mode,
                    season=int(season),
                    team_names=tuple(t.strip() for t in team_text.splitlines()),
                    league_config_path=league_config_path,
                    vorp_path=Path(vorp_path),
                    id_map_path=Path(id_map_path),
                )
            )
        except FileNotFoundError as exc:
            st.sidebar.error(
                f"Missing input ({exc}). Generate preset tables with: "
                "python scripts/generate_preset_vorp_tables.py"
            )
        except Exception as exc:  # surface any other setup failure
            st.sidebar.error(f"Setup failed: {exc}")

    _resume_controls()


def _resume_controls() -> None:
    if not _SESSION_DIR.exists():
        return
    saves = sorted(
        _SESSION_DIR.glob("auction_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not saves:
        return
    newest = saves[0]
    st.sidebar.divider()
    st.sidebar.caption(f"Resume autosave: {newest.name}")
    if st.sidebar.button("↩ Resume last auction"):
        try:
            data = json.loads(newest.read_text())
            id_map, pool = _load_inputs(Path(data["vorp_table"]), Path(data["id_map"]))
            _install_session(
                LiveAuctionSession.load(newest, id_map=id_map, pool=pool), autosave_path=newest
            )
        except Exception as exc:  # surface any resume failure to the user
            st.sidebar.error(f"Resume failed: {exc}")


def _status_bar(s: LiveAuctionSession) -> None:
    if s.is_complete:
        st.subheader("✅ Auction complete")
        return
    open_ = s.open_slots(s.my_seat)
    budget = s.budget(s.my_seat)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Budget left", f"${budget}")
    c2.metric("Open slots", open_)
    c3.metric("Your max bid", f"${s.feasible_max(s.my_seat)}")
    c4.metric("Even pace", f"${budget / open_:.0f}/slot" if open_ else "—")
    infl = s.inflation()
    c5.metric(
        "Market",
        f"{infl:.2f}x",
        "over value" if infl > 1 else "under value",
        delta_color="off",
    )
    nom = s.nominating_seat
    who = "**your nomination**" if nom == s.my_seat else f"nomination: {s.team_label(nom)}"
    st.caption(f"Lot {len(s.purchases) + 1} of {s.league.total_pool_size} · {who} (by rotation)")


def _selectable(named: pd.DataFrame, cols: list[str], key: str) -> None:
    """Render a single-row-selectable table; selecting a row stages `pending_player`.

    `named` must carry `gsis_id` with a clean 0..n-1 index (selection rows are positional).
    Mirrors scripts/draft_board.py's helper. NOTE: the original reason given here — "the two
    boards are separate Streamlit apps and neither imports the other" — is not a real
    constraint; `tests/test_scripts/` already imports scripts as `scripts.X`, so a shared
    `scripts/_board_ui.py` would work. This and seven sibling helpers stay duplicated as a
    deliberate open question, not a technical limit. See issue #138.
    """
    show = [c for c in cols if c in named.columns]

    def _stage() -> None:
        state = st.session_state.get(key)
        rows = state["selection"]["rows"] if state else []
        if rows:
            st.session_state["pending_player"] = str(named.iloc[rows[0]]["gsis_id"])

    st.dataframe(
        named[show],
        height=380,
        hide_index=True,
        selection_mode="single-row",
        on_select=_stage,
        key=key,
    )


def _bid_panel(s: LiveAuctionSession) -> None:
    """The staged player: what to bid, and the form that records who actually bought him."""
    pending = st.session_state.get("pending_player")
    if not pending or s.is_complete:
        st.info("Pick a player from **Bid board** below to price him and record the sale.")
        return
    try:
        advice = s.advise(str(pending))
    except ValueError as exc:  # drafted since it was staged, or not in the pool
        st.warning(str(exc))
        st.session_state["pending_player"] = None
        return

    st.markdown(f"### {advice.full_name} · {advice.position}")
    if not advice.eligible:
        # Deliberately "the plan says pass", not "you may not": bot_position_bounds is the
        # engine's roster-discipline heuristic (min + a share of the bench), not a league
        # rule. The league will happily let you roster another one, and record_purchase will
        # record it -- this is the measured strategy declining, and it can be overridden by
        # simply recording the sale to yourself.
        st.error(
            f"PASS — the plan is done buying {advice.position}. It caps you at what the "
            "model will start or bench; the league itself allows more."
        )
    else:
        st.markdown(f"# 🔨 Bid up to ${advice.max_bid}")
        if advice.max_bid < advice.desired:
            st.caption(f"(model wanted ${advice.desired}; capped by your budget)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Worth to us", f"${advice.fair_value}")
    c2.metric("Room's price", f"${advice.market_value}")
    c3.metric("Best rival can bid", f"${advice.room_ceiling}")
    if advice.uncontested:
        # "should", not "cannot": room_ceiling counts only rivals our roster-discipline model
        # says would still buy this position. A real manager with money can bid on anything.
        st.success(
            "You should win this one — your ceiling clears every rival the model expects "
            "to bid here. (A rival who ignores roster discipline still can.)"
        )
    elif advice.eligible and not advice.i_want:
        st.warning("The room is anchored above your ceiling — expect to lose this one. Let it go.")

    st.markdown("**Record the sale**")
    # Widget keys carry the staged player, and the price also carries the chosen winner.
    # Keyed on the lot number alone, Streamlit retains the widget's value when a different
    # player is staged for the same lot: a price typed for a $46 stud survived onto a $3
    # backup, one click from being recorded. The winner belongs in the price key for the
    # same reason -- switching to a poorer seat shrinks max_value under a retained value.
    lot_key = f"{len(s.purchases)}_{advice.gsis_id}"
    # A seat with no open roster slot cannot take the player: record_purchase rejects it,
    # and feasible_max is meaningless there (it reads budget + min_bid at zero slots).
    seats = [seat for seat in s.seats if s.open_slots(seat) > 0]
    winner: int | None = None
    price = s.league.min_bid
    if not seats:
        st.warning("No team has an open roster slot.")
    else:
        winner = st.selectbox(
            "Winning team",
            seats,
            # No default. The hero wins roughly one lot in n_teams, so a pre-selected seat is
            # the wrong answer for most lots and confirm is a single click away -- exactly the
            # mis-record the explicit confirm button exists to prevent.
            index=None,
            placeholder="Select the winning team…",
            format_func=s.team_label,
            key=f"winner_{lot_key}",
        )
        if winner is not None:
            cap = max(s.league.min_bid, s.feasible_max(int(winner)))
            price = st.number_input(
                "Price paid ($)",
                min_value=s.league.min_bid,
                max_value=cap,
                value=max(s.league.min_bid, min(advice.market_value, cap)),
                step=1,
                key=f"price_{lot_key}_{winner}",
            )
    confirm, clear = st.columns([4, 1])
    if winner is None:
        confirm.button(
            "Select the winning team to record this sale",
            key=f"confirm_{lot_key}",
            disabled=True,
        )
    else:
        label = f"✅ {advice.full_name} → {s.team_label(int(winner))} for ${int(price)}"
        if confirm.button(label, key=f"confirm_{lot_key}", type="primary"):
            try:
                s.record_purchase(advice.gsis_id, int(winner), int(price))
            except ValueError as exc:
                st.warning(str(exc))
                return
            st.session_state["pending_player"] = None
            _autosave(s)
            st.rerun()
    if clear.button("✕ clear", key="clear_pending"):
        st.session_state["pending_player"] = None
        st.rerun()


def _nomination_panel(s: LiveAuctionSession) -> None:
    if s.is_complete:
        return
    with st.expander("🎤 Who to nominate", expanded=s.is_my_nomination):
        sug = s.suggested_nomination()
        if sug is None:
            st.caption("Nothing left to nominate.")
            return
        st.markdown(f"**Nominate: {s.name(sug)}**")
        st.caption(NOMINATION_NOTES[s.nomination_mode])
        board = s.nomination_board(top=12)
        st.caption(
            "`i_want` = your roster can take him and your ceiling clears the room's price. "
            "A high-`market`, `i_want`-False player drains rivals' money, not yours."
        )
        st.dataframe(
            board[["full_name", "position", "value", "market", "max_bid", "room_max", "i_want"]],
            hide_index=True,
        )


def _bid_board_panel(s: LiveAuctionSession) -> None:
    st.markdown("**🔎 Bid board** — click a row to price him and record the sale")
    if s.is_complete:
        return
    pos_label = st.selectbox(
        "Position", ["All", "QB", "RB", "WR", "TE"], key=f"bb_pos_{len(s.purchases)}"
    )
    query = st.text_input("Search player", key=f"bb_query_{len(s.purchases)}", placeholder="name…")
    board = _cached_bid_board(
        st.session_state.get("session_token", ""), s.state_key, pos_label, query
    )
    if board.empty:
        st.caption("No matching available players.")
        return
    _selectable(
        board,
        ["full_name", "position", "max_bid", "value", "market", "edge", "room_max", "adp"],
        key=f"sel_bid_{len(s.purchases)}",
    )
    st.caption(
        "`max_bid` = stop there. `value` = worth to us · `market` = what the room pays · "
        "`edge` = max_bid minus market (positive = a lot you can win) · "
        "`room_max` = best rival ceiling."
    )


def _roster_col(s: LiveAuctionSession) -> None:
    view = s.my_roster_view()
    st.markdown(f"**My roster** — ${view.spent} spent")
    st.dataframe(view.filled[["slot", "full_name", "position", "price"]], hide_index=True)
    if view.open_slots:
        st.caption(
            "Open starting slots: "
            + ", ".join(f"{slot.value} x{n}" for slot, n in view.open_slots.items())
        )
    st.markdown("**Budgets**")
    st.dataframe(s.budget_table()[["team", "you", "budget", "players", "max_bid"]], hide_index=True)


def _log_col(s: LiveAuctionSession) -> None:
    st.markdown("**Sold**")
    st.dataframe(s.purchase_log(), height=440, hide_index=True)
    if s.purchases and st.button("↶ Undo last sale", key="undo_last"):
        s.undo()
        st.session_state["pending_player"] = None
        _autosave(s)
        st.rerun()


@st.cache_data(show_spinner=False)
def _cached_bid_board(
    session_token: str, purchases: tuple[tuple[str, int, int], ...], pos_label: str, query: str
) -> pd.DataFrame:
    """Cache the priced bid board on everything that changes it.

    `bid_board` calls `strategy.max_bid` once per player for up to 40 players, and Streamlit
    reruns the whole script on every widget interaction -- so without this, typing one letter
    in the search box re-priced the board, and for `marginal` (a lineup solve per call) that is
    40 lineup solves per keystroke, live, mid-draft. Keyed like `_cached_projection`: the
    session token plus `state_key`, the (player, seat, price) triples that are the only mutable
    state, plus the two filters that shape the result.
    """
    s: LiveAuctionSession = st.session_state["session"]
    position = None if pos_label == "All" else Position(pos_label)
    return s.bid_board(position=position, query=query, top=40)


@st.cache_data(show_spinner=False)
def _cached_projection(
    session_token: str, purchases: tuple[tuple[str, int, int], ...], n_sims: int
) -> dict:  # type: ignore[type-arg]
    """Cache the projected-league eval on (session, purchases, n_sims). Pulls the live session +
    an optional injected availability from session_state (tests inject; prod loads from store).

    `purchases` is the session's `state_key` — (player, seat, price) triples, not bare player
    ids. An auction's rosters are (player, seat) pairs, so a player-only key would serve the
    previous projection after an undo re-awarded the same player to a different team."""
    s: LiveAuctionSession = st.session_state["session"]
    avail = st.session_state.get("_eval_availability")
    res = s.project_league_outcomes(n_sims=n_sims, availability=avail)
    return {seat: vars(proj) for seat, proj in res.items()}


def _results_section(s: LiveAuctionSession) -> None:
    if not s.is_complete:
        return
    st.markdown("### 📊 Projected auction results")
    st.caption(
        "Projected-vs-projected league sim (injury + performance variance, optimal lineup all "
        "teams). Measures roster quality under our projections, not real outcomes."
    )
    if not st.button("Run projected eval", key="run_projected_eval", type="primary"):
        return
    token = st.session_state.get("session_token", "")
    with st.spinner("Simulating seasons…"):
        res = _cached_projection(token, s.state_key, 2000)
    n = s.league.n_teams
    me = res[s.my_seat]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reg-season win%", f"{me['reg_win_pct']:.1%}", "vs 50%")
    # PLAYOFF_SIZE / N_BYES rather than 6 / 2: the bracket the projection actually simulates,
    # so these baselines cannot drift away from it.
    c2.metric("Make playoffs", f"{me['make_playoffs_pct']:.1%}", f"vs {PLAYOFF_SIZE / n:.1%}")
    c3.metric("First-round bye", f"{me['bye_pct']:.1%}", f"vs {N_BYES / n:.1%}")
    c4.metric("Championship %", f"{me['champ_pct']:.1%}", f"vs {1 / n:.1%}")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "team": s.team_label(seat),
                    "you": "★" if seat == s.my_seat else "",
                    "reg win%": f"{res[seat]['reg_win_pct']:.0%}",
                    "playoff%": f"{res[seat]['make_playoffs_pct']:.0%}",
                    "champ%": f"{res[seat]['champ_pct']:.0%}",
                }
                for seat in s.seats
            ]
        ),
        hide_index=True,
    )


# --------------------------------------------------------------------------- minimal mode


def _team_names_gate(s: LiveAuctionSession) -> bool:
    """Name the teams before the draft starts. Returns True once naming is done.

    A gate, not a setting: every confirm button in this mode reads `Player → <name> for $N`,
    and that sentence is the only thing between a misheard winner and a corrupted purchase
    log. `Team 6` does not do that job under time pressure. A resumed session already carries
    names, so this appears once.
    """
    if any(n.strip() for n in s.team_names):
        return True
    st.subheader("① Name the teams")
    st.caption("So the confirm button reads a real name instead of “Team 6”. One time only.")
    with st.form("team_names_form"):
        names = [
            st.text_input(
                f"Seat {seat}" + (" (you)" if seat == s.my_seat else ""),
                value=("You" if seat == s.my_seat else f"Team {seat}"),
                key=f"tn_{seat}",
            )
            for seat in s.seats
        ]
        if st.form_submit_button("Save names & start", type="primary"):
            s.team_names = tuple(n.strip() or f"Team {i + 1}" for i, n in enumerate(names))
            _autosave(s)
            st.rerun()
    # An escape so the gate can never strand a session (e.g. resuming to check one number).
    if st.button("Skip — use Team 1…N"):
        s.team_names = tuple(f"Team {seat}" for seat in s.seats)
        _autosave(s)
        st.rerun()
    return False


def _available_options(s: LiveAuctionSession) -> dict[str, str]:
    """`{"Name (POS)": gsis_id}` for every available player — the autocomplete's options.

    Labelled with the position so two players sharing a surname stay distinguishable, and
    built from the same resolved names the rest of the board displays.
    """
    pool = filter_named_pool(s.available_pool(), s.player_names)
    return {
        f"{row.full_name} ({row.position})": str(row.gsis_id)
        for row in pool.itertuples()
        if isinstance(getattr(row, "full_name", None), str)
    }


def _minimal_record(s: LiveAuctionSession, advice: BidAdvice) -> None:
    """Winner + price + confirm. Same widget-keying safety as the full board's panel."""
    lot_key = f"{len(s.purchases)}_{advice.gsis_id}"
    seats = [seat for seat in s.seats if s.open_slots(seat) > 0]
    if not seats:
        st.warning("No team has an open roster slot.")
        return
    c1, c2 = st.columns([2, 1])
    winner = c1.selectbox(
        "Won by",
        seats,
        index=None,
        placeholder="Which team?",
        format_func=s.team_label,
        key=f"m_winner_{lot_key}",
    )
    if winner is None:
        c2.empty()
        st.button("Pick the winning team", disabled=True, key=f"m_confirm_{lot_key}")
        return
    cap = max(s.league.min_bid, s.feasible_max(int(winner)))
    price = c2.number_input(
        "Price",
        min_value=s.league.min_bid,
        max_value=cap,
        value=max(s.league.min_bid, min(advice.market_value, cap)),
        step=1,
        key=f"m_price_{lot_key}_{winner}",
    )
    label = f"✅ {advice.full_name} → {s.team_label(int(winner))} for ${int(price)}"
    if st.button(label, key=f"m_confirm_{lot_key}", type="primary"):
        try:
            s.record_purchase(advice.gsis_id, int(winner), int(price))
        except ValueError as exc:
            st.warning(str(exc))
            return
        st.session_state["pending_player"] = None
        _autosave(s)
        st.rerun()


def _minimal_view(s: LiveAuctionSession) -> None:
    """The live-draft surface: nominate, read a number, record the sale. Nothing else.

    Deliberately omits the sold log, rosters, budget table, bid board and projected eval --
    all of which the Yahoo draft UI already shows the operator. See the design doc.
    """
    if not _team_names_gate(s):
        return
    if s.is_complete:
        st.subheader("✅ Auction complete")
        st.caption("Switch to **Full board** in the sidebar for the projected-season eval.")
        return

    options = _available_options(s)
    st.subheader("② Who was nominated?")
    picked = st.selectbox(
        "Nominated player",
        list(options),
        index=None,
        placeholder="Type a player name…",
        label_visibility="collapsed",
        key=f"m_nom_{len(s.purchases)}",
    )
    if picked is not None:
        st.session_state["pending_player"] = options[picked]
    pending = st.session_state.get("pending_player")

    if pending and str(pending) in set(options.values()):
        try:
            advice = s.advise(str(pending))
        except ValueError as exc:
            st.warning(str(exc))
            st.session_state["pending_player"] = None
        else:
            st.divider()
            if not advice.eligible:
                st.error(f"PASS — the plan is done buying {advice.position}.")
            else:
                st.markdown(f"# 🔨 ${advice.max_bid}")
                st.caption(f"**{advice.full_name}** · {advice.position} — bid up to this")
                if advice.uncontested:
                    st.success("You should win this one — your ceiling clears every likely rival.")
                elif not advice.i_want:
                    st.warning("Room is anchored above your ceiling — let it go.")
            st.caption(
                f"worth to us ${advice.fair_value} · room pays ${advice.market_value} · "
                f"best rival ${advice.room_ceiling}"
            )
            st.divider()
            _minimal_record(s, advice)

    with st.expander("🎤 Who to nominate", expanded=s.is_my_nomination):
        sug = s.suggested_nomination()
        if sug is None:
            st.caption("Nothing left to nominate.")
        else:
            st.markdown(f"**{s.name(sug)}**")
            st.dataframe(
                s.nomination_board(top=5)[["full_name", "position", "market", "i_want"]],
                hide_index=True,
            )

    foot, undo = st.columns([3, 1])
    foot.caption(
        f"${s.budget(s.my_seat)} left · {s.open_slots(s.my_seat)} slots · "
        f"{len(s.purchases)}/{s.league.total_pool_size} sold"
    )
    if s.purchases and undo.button("↶ Undo", key="m_undo"):
        s.undo()
        st.session_state["pending_player"] = None
        _autosave(s)
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Auction Board", layout="wide")
    st.title("💰 Live Auction Board")
    _sidebar()

    s: LiveAuctionSession | None = st.session_state.get("session")
    if s is None:
        st.info("Configure the auction in the sidebar and click **Start / restart auction**.")
        return
    if st.session_state.get("view_mode") == VIEW_MODES[1]:
        _minimal_view(s)
        return
    _status_bar(s)
    left, center, right = st.columns([1.1, 2.4, 1.2])
    with left:
        _log_col(s)
    with center:
        _bid_panel(s)
        _nomination_panel(s)
        _bid_board_panel(s)
    with right:
        _roster_col(s)
    _results_section(s)
    st.caption(
        f"Bid model: {s.strategy_name} · market: {s.market} · "
        f"{len(s.purchases)} of {s.league.total_pool_size} lots sold"
    )


if __name__ == "__main__":
    main()
