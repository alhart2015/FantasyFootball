# Post-draft assessment: projected points-for → actual H2H outcomes (2021–2025)

**Question.** After a draft, how well does a team's **projected** season points-for predict its **actual** head-to-head outcomes (win%, playoff berth, championship), and what confidence can we put on a post-draft assessment?

**TL;DR.** Among teams drafted by `season_value`, projected PF explains ~26% of win% variance (r ≈ 0.51); across all teams (wider draft-quality spread) ~51% (r ≈ 0.71). A `season_value` draft yields ~1.9× base championship odds and +1.5 wins over a .500 team. The single most useful artifact is the **projected-PF quintile table** (§4): read off champ%/playoff%/expected-wins from your team's projected PF. The per-season bootstrap CIs are too tight — see §5.

> ⚠️ **Availability lookahead — numbers below predate PR #149 (2026-08-19).**
> Every backtest recorded here was run with an availability model whose injury prior
> `p` was built from a weekly_stats history that INCLUDED the season being graded (and
> later ones). A player who missed half of 2024 was pre-marked injury-prone in the 2024
> draft. `load_store_availability` now reads only completed seasons STRICTLY BEFORE the
> target, so these figures are **not reproducible** on current code, and any strategy
> that gates on availability was flattered here. Regenerate before using a number below
> as a baseline for a code change.

## Method

- **Backtest:** F1 H2H league sim (`src/projections/draft/backtest/`), 16-team half-PPR, 200 draft seeds per season, `strategy_n_sims=200`, jitter 8. Field = `now_or_never` + `season_value` strategies vs noisy-ADP bots (`seat_layout`: 4 nn / 4 sv / 8 bot per league → 800 sv team-seasons/season, 3,200 all-team/season).
- **Two scorings per team:** PROJECTED points-for (the lineup's projected value under shared beliefs — "who drafted better") and ACTUAL points-for/record (realized weekly outcomes). This report relates **projected PF → actual outcomes**.
- **Draft basis:** the multi-source **ESPN+Sleeper blended** consensus (branch `feat/multi-source-projection-blend`). This is what makes 2021–2023 usable — ESPN does not retain full historical preseason season projections (2023 was 1-field stubs; Sleeper fills the gap). All five seasons' pools are ~100% projected on this basis.
- **CIs:** paired bootstrap over (seed, seat) team-seasons. **Caveat in §5.**
- **Checkpoints (committed):** `reports/data/post_draft_2021_2025/<season>/` (200/200 chunks each; per-season `manifest.json`: nn/sv, 200 sims, jitter 8). Analysis tool: `scripts/post_draft_assessment.py`.

## 1. Correlation: projected PF → actual win%

| Season | `season_value` drafters | all teams |
|---|---|---|
| 2021 | r = +0.566 [0.517, 0.611], r²=0.32 (n=800) | +0.700 [0.681, 0.716], r²=0.49 (n=3200) |
| 2022 | r = +0.513 [0.463, 0.561], r²=0.26 | +0.790 [0.778, 0.802], r²=0.62 |
| 2023 | r = +0.475 [0.416, 0.531], r²=0.23 | +0.762 [0.748, 0.775], r²=0.58 |
| 2024 | r = +0.588 [0.543, 0.631], r²=0.35 | +0.796 [0.784, 0.808], r²=0.63 |
| 2025 | r = +0.625 [0.583, 0.664], r²=0.39 | +0.693 [0.675, 0.711], r²=0.48 |
| **Combined** | **r = +0.506 [0.481, 0.529], r²=0.26 (n=4000)** | **+0.714 [0.706, 0.721], r²=0.51 (n=16000)** |

Among already-good drafters (`season_value`), projected PF explains ~¼ of win% variance — real signal, but ~¾ is single-season luck (schedule, weekly boom/bust, injuries). Across all teams the predictor spans bots→strategies, so the correlation is higher (~½).

## 2. Post-draft assessment — if you draft `season_value`

| Season | champ% | playoff% | expected wins (/14) |
|---|---|---|---|
| 2021 | 16.0% [13.5, 18.5] | 62.1% [58.8, 65.5] | 8.4 [8.3, 8.5] |
| 2022 | 14.6% [12.2, 17.1] | 78.0% [75.1, 80.9] | 9.2 [9.1, 9.3] |
| 2023 | 18.1% [15.5, 20.9] | 73.2% [70.1, 76.4] | 8.9 [8.8, 9.0] |
| 2024 | 5.2% [3.8, 6.9] | 52.1% [48.6, 55.6] | 8.1 [7.9, 8.2] |
| 2025 | 5.2% [3.8, 6.9] | 50.2% [46.8, 53.8] | 7.7 [7.6, 7.8] |
| **Combined** | **11.8% [10.8, 12.8]** | **63.1% [61.7, 64.6]** | **8.5 [8.4, 8.5]** |

16-team base rates: champ 6.25%, playoff 37.5%, wins 7.0. So `season_value` drafts a clearly above-average team: **~1.9× base champ odds, ~1.7× base playoff odds, +1.5 wins** (combined).

## 3. Season-to-season variation

Champ% ranges **5.2% → 18.1%** across the five seasons; 2021–2023 (14–18%) ran well above 2024–2025 (5.2%). This is realized-outcome variance (which players actually boomed/busted that year), not a basis artifact. We don't know what 2026 will look like, so the combined average (11.8%) is the point estimate — but the spread is the real story (see §5).

## 4. Outcomes by projected-PF quintile (combined 2021–2025, n=16,000) — the post-draft tool

| Projected PF | champ% | playoff% | expected wins |
|---|---|---|---|
| 633–976 (bottom 20%) | 0.1% [0.0, 0.2] | 1.5% [1.1, 1.9] | 4.3 [4.2, 4.4] |
| 976–1052 | 1.3% [1.0, 1.8] | 13.3% [12.1, 14.5] | 6.0 [6.0, 6.1] |
| 1052–1112 | 4.5% [3.8, 5.2] | 34.3% [32.7, 36.0] | 7.2 [7.1, 7.3] |
| 1112–1173 | 7.6% [6.7, 8.5] | 56.1% [54.4, 57.9] | 8.2 [8.1, 8.2] |
| 1173–1394 (top 20%) | 17.8% [16.5, 19.1] | 82.2% [80.9, 83.6] | 9.3 [9.3, 9.4] |

**Usage:** after a real draft, compute your team's projected season PF, find the row, read off champ%/playoff%/expected wins. Monotonic and steep — projected PF stratifies outcomes hard. (PF bins are league-specific to this 16-team half-PPR config.)

## 5. CI caveat — the bootstrap CIs are too tight

The backtest resamples **only draft order + schedule**; each player's weekly score is a single fixed historical realization. So the CIs above capture draft/schedule luck but **not** season-to-season projection-quality / player-outcome variance. Evidence: per-season champ% swings 5–18% while each season's bootstrap CI is only ±2–3%. **For a forward-looking 2026 estimate, the honest uncertainty is closer to the cross-season spread (champ ≈ 5–18%, playoff ≈ 50–78%, wins ≈ 7.7–9.2) than the tight per-season bars.** Closing this is **TODO #45** — drawing each simulated week from the player's weekly fantasy-point distribution (Projections Core) instead of a fixed value — which would also widen `scripts/post_draft_assessment.py`'s resampling to player outcomes.

## Reproduce

```
# Re-ingest the blended basis (ESPN+Sleeper) for each season, then:
python scripts/h2h_backtest_chunked.py --season YYYY --league-config configs/league_espn_half_16team.json \
  --n-seeds 200 --strategy-n-sims 200 --jitter 8 --chunk-size 5 \
  --checkpoint-dir reports/data/post_draft_2021_2025/YYYY --data-root data
# Then the committed analysis (defaults to the committed checkpoints):
python scripts/post_draft_assessment.py
```
(On the dev Windows box: run in PowerShell with `KMP_DUPLICATE_LIB_OK=TRUE` + single-thread BLAS; the chunk runner auto-retries the intermittent native crash. See `MEMORY.md`.)
