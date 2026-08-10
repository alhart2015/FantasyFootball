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

## The strategy: `static`

Out of 14 bidding strategies tested across all 12 seats and 2 market models, `static` won.
It is one rule:

> **Bid up to a player's max bid. Never a dollar more. Then move to the next name.**

That's the whole thing. No budget math during the draft, no adjusting for who's spent what.
The discipline is the strategy.

Two things it beats decisively, both of which are tempting in a room full of overbidders:

- **Don't sit back and wait for bargains.** The "patient" strategy finished at a 50% win
  rate — a coin flip, and the worst result in the study.
- **Don't go all-in on 3 studs and punt the rest.** Pure stars-and-scrubs finished below
  `static` on every metric.

## Use it (no software)

Open **`reports/will_auction_cheat_sheet.txt`** and print it. That's the strategy — every
player worth more than $2, by position, with the maximum you should pay.

`reports/will_auction_cheat_sheet.csv` is the same data as a spreadsheet if you'd rather
sort and filter during the draft.

Columns:

| column | meaning |
|---|---|
| **MAX BID** | your ceiling. This is the number that matters. |
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
`static` is an auction bid model and does not appear there. Running it for this league
would give you advice for the wrong format.

For this auction, the printed sheet *is* the tool. Track budgets on paper or in the
league site's own auction room.

## Draft day

**Pacing.** $200 over 13 spots is ~$15 a spot on average. You will not spend evenly — see
below — but if you have 8 spots left and under $10, you've lost the plot.

**What a typical `static` draft looks like** (average of 60 simulated drafts):

- Top 3 buys: about **$55 / $53 / $49** — three genuine anchors, ~$157 of your $200
- One more player around **$26**
- The remaining nine spots at **$1–5**
- Ends at $0 with 13 players

So in practice this is a top-heavy build. The rule keeps you honest about *which* studs
(only the ones still under their max bid) rather than making you spread the money evenly.

**Positional reality.** The model spends about **$167 of $200 on running backs** and only
~$23 on receivers, and lands ~6 RB / ~4 WR. That's because 23 of the top 30 players by
value are running backs.

**Read this before you commit to that:** that RB tilt comes from the projections, not the
bidding strategy. Replacement level sits at 92 points for RB vs 135 for WR, which is what
makes backs look so much more valuable. If you don't believe that gap is real, the
strategy is still sound but it's pointing you at the wrong players — take the max bids as
a discipline device and apply your own judgment about the RB/WR split.

**Bargains to watch.** Where MAX BID is well above the ESPN column, the room is likely to
let him go cheap: Derrick Henry ($51 vs $35), Breece Hall ($42 vs $27), Javonte Williams
($40 vs $24), D'Andre Swift ($32 vs $9).

**Where to be disciplined.** Where ESPN is above MAX BID, the room will overpay and you
should let it: Puka Nacua ($46 vs $60), Ja'Marr Chase ($43 vs $59), Jaxon Smith-Njigba
($37 vs $55). These are good players — they're just going to cost more than they're worth.

**Quarterbacks are cheap and that's correct.** Even with 5-point passing TDs, twelve
starting QBs means the twelfth is nearly as good as the third. Max bids top out around
$12. Don't pay $30 for Josh Allen because the scoring "looks QB-friendly."

## What this is and isn't

Honest limits, so you can weigh the advice:

- Simulated against a modeled opponent field (9 aggressive overbidders + 2 patient
  bidders), not against your actual league-mates.
- One season's projections (2026), one set of rankings.
- `static` beat the next three strategies by a margin that clears statistical noise on
  championship rate, but on regular-season wins they were within noise of each other. The
  finding is "pay market price for talent," more than "this exact model."
- Over a 13-game season, luck outweighs roster quality. In testing, the roster with the
  *highest* title odds missed the playoffs and one with lower odds won it all.
