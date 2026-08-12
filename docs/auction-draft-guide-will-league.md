# Auction draft guide — 12-team half-PPR (5-pt pass TD)

For a non-technical user. Everything you need is already committed; the software is optional.

## The league this is built for

| | |
|---|---|
| Teams | 12 |
| Budget | $200, $1 minimum bid |
| Starters | 1 QB, 2 RB, 2 WR, 1 TE, 2 FLEX (+1 K, 1 DST) |
| Bench | 5 |
| Scoring | 0.5 per reception, 6-pt rush/rec TD, **5-pt pass TD**, −2 INT, −2 fumble |

The kicker and defense are left out of everything below on purpose. They are $1 buys that
don't change auction strategy, and the projection model doesn't cover them. Draft them last
for a dollar.

Yardage bonuses (300/350/400 passing, 100/150/200 rushing/receiving) are **not** in the
model. They're worth a few points a season to boom players and don't reorder the board.

## The strategy: `overbid_noramp`

Out of 15 bidding strategies tested across all 12 seats, `overbid_noramp` won. It beat every
other strategy by a margin that clears statistical noise, on every metric.

It is two rules:

> **1. For a stud (marked `*` on the sheet), bid up to 1.3x what he's worth.**
> **2. For everyone else, bid up to exactly what he's worth.**
> **Never a dollar more, at any point in the draft.**

The sheet already does the 1.3x multiplication for you. The MAX BID column is the number to
stop at — don't add anything on top of it.

Two things it beats decisively, both tempting in a room full of overbidders:

- **Don't sit back and wait for bargains.** The "patient" strategy finished at a 54% win rate,
  dead last. Sitting out the early market is the single worst plan against this room.
- **Don't spread your money evenly.** Trying to buy a balanced roster of mid-priced players
  ("balanced", .597) lost significantly to buying a few studs.

### The counter-intuitive part

**Spend everything, early, and don't save any.** A typical draft: about $180 of your $200 goes
to three running backs, and nine or ten roster spots cost $1 each.

That looks wrong. It isn't. Every team in this room ends at $0 — the money always gets spent,
so the only question is whether you spent it on the best available players or on leftovers.
There is nothing worth saving for.

We tested holding money back for a late-draft push (the "ramp"), and it made things **worse**
(−0.012 win rate, CI excludes zero). Bid the number, stop at the number, move on.

## Use it (no software)

Open **`reports/will_auction_cheat_sheet.txt`** and print it. That's the strategy — every
player worth more than $2, by position, with the maximum you should pay.

`reports/will_auction_cheat_sheet.csv` is the same data as a spreadsheet if you'd rather
sort and filter during the draft.

Columns:

| column | meaning |
|---|---|
| **MAX BID** | your ceiling, stud premium already included. The only number that matters. |
| `*` | marks a stud — the top 36 players. His MAX BID is 1.3x his worth. |
| worth | what he's actually worth to us, before the stud premium |
| ESPN | what the market thinks he costs. Much lower than MAX BID = a bargain the room may not contest. |
| proj | projected fantasy points for the season |
| adp | roughly when he'll be nominated |

## Regenerate it yourself (optional)

Only needed if the projections get updated. Requires installing Python.

1. Install [Python 3.11+](https://www.python.org/downloads/) and
   [Git](https://git-scm.com/downloads).
2. Open a terminal and run, one line at a time:

```bash
git clone https://github.com/alhart2015/FantasyFootball.git
cd FantasyFootball
python -m venv .venv
```

3. Turn on the environment — **pick the line for your system**:

```bash
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # Mac / Linux
```

4. Install, then build the sheet:

```bash
pip install -e .
python scripts/auction_cheat_sheet.py \
    --league-config configs/will_half12_pass5.league.json \
    --vorp-table data/vorp_2026/will_half12.parquet \
    --out reports/will_auction_cheat_sheet.csv
```

That rewrites both the `.csv` and the `.txt`.

## About the draft-tracker UI

**There is no auction draft tracker in this repo, and the UI will not help you here.**

`streamlit run scripts/draft_board.py` launches a live board, but it is built for **snake
drafts only** — it tracks pick order and draft slots, has no concept of a budget or a
bid, and its strategy menu contains snake strategies (`season_value`, `now_or_never`, …).
`overbid_noramp` is an auction bid model and does not appear there. Running it for this
league would give you advice for the wrong format.

For this auction, the printed sheet *is* the tool. Track budgets on paper or in the
league site's own auction room.

## Draft day

**Pacing.** Don't. There is no pacing in this plan — you are trying to land three of the best
players on the board and then fill out with dollar guys. If you're eight spots in with $150
still in your pocket, you are losing.

**What a typical draft looks like** (average of 60 simulated drafts):

- Top three buys around **$80 / $63 / $40** — almost all running backs
- Nine or ten players at **$1**
- Ends at **$0** with 13 players

**Positional reality.** The plan spends about **$189 of $200 on running backs** and lands ~6 RB
/ ~4 WR. That's because 23 of the top 30 players by value are running backs.

**Read this before you commit to that:** the RB tilt comes from the projections, not the bidding
rule. Replacement level sits at 92 points for RB vs 135 for WR, which is what makes backs look
so much more valuable. If you don't believe that gap is real, the strategy is still sound but
it's pointing at the wrong players. Your receivers will be waiver-wire types either way — that
is what the plan trades away.

**Where you'll be bidding against the room.** Where our MAX BID is well above the ESPN column,
the room is likely to stop before you do: Derrick Henry ($66 vs $35), Breece Hall ($55 vs $27),
Javonte Williams ($52 vs $24), Kenneth Walker ($58 vs $29).

**Where to walk away.** Where ESPN is close to or above our MAX BID, you'll be in a real fight
and the player isn't worth it to you: Puka Nacua, Ja'Marr Chase, Jaxon Smith-Njigba. Let them go.

**Quarterbacks: exactly one is worth paying for.** Josh Allen is a stud on this board (382
projected points, MAX BID **$36**). After him it falls off a cliff — Lamar Jackson and Drake Maye
are $15, and everyone else is $12 or less, because twelve QBs start and the twelfth is nearly as
good as the third. So either win Allen at $36 or take a $12 quarterback and spend nowhere in
between. Tight end works the same way: Trey McBride $31, Brock Bowers $23, then single digits.

## What this is and isn't

Honest limits, so you can weigh the advice:

- Simulated against a modeled opponent field (9 aggressive overbidders + 2 patient
  bidders), not against your actual league-mates.
- One season's projections (2026), one set of rankings.
- `overbid_noramp` beat every other strategy on every metric with confidence intervals that
  exclude zero. But the opponent model behind that number was rebuilt four times during testing,
  each time because it contained a flaw that handed some strategy a free win. Treat it as a
  well-tested opinion, not a guarantee.
- The rosters it drafts look thin at receiver, by design. That is worth an eye-test before you
  commit to it.
- Over a 13-game season, luck outweighs roster quality. In testing, the roster with the
  *highest* title odds missed the playoffs and one with lower odds won it all.
