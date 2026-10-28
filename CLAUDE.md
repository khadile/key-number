# CLAUDE.md

NFL and NBA betting models on one board. NFL: game lines (spreads, totals,
moneylines), player props (pregame), and live (in-game, paper trading). NBA
(since 2026-10-20, section 9): game lines and player props, built daily.
Data comes from nflverse via `nflreadpy`; odds from OddsPapi (this season's
history), PropLine (live) and a one-time SportsGameOdds pull (2025 closing
prop lines, 2026 opens and closes; 2e). The user bets at DraftKings and FanDuel only; Pinnacle is the
sharp reference, never a bet.

## 0. Where it runs (2026-10-11)

- Public repo https://github.com/khadile/key-number, branch `main`. The board is
  built by GitHub Actions (`.github/workflows/board.yml`) Tue/Thu/Sat from 10am ET
  and published to https://khadile.github.io/key-number/ (`board/site.py` wraps
  it in a page). The workflow fires every half hour 14:07-17:37 UTC because
  GitHub drops scheduled firings; `board/slot.py` builds on the first one at or
  after 10am ET (`board/last_build.json`). Run it by hand with
  `gh workflow run board.yml -R khadile/key-number -f force=true`.
  The NBA side has its own workflow (`nba.yml`, section 9). Each sport's build
  commits only its payload (`board/nfl_data.json`, `board/nba_data.json`) and
  then starts the publish workflow (`pages.yml`), which renders the page from
  both (`board/render.py`) and deploys, one deploy at a time; it also runs on a
  push from the laptop and by hand. `board/nfl_board.html` is no longer committed.
- Needs the `ODDSPAPI_KEY` repo secret. Odds data never goes in the repo:
  data/oddspapi (except the three catalog JSONs), data/propline, data/sgo,
  data/live and price-level report parquets are gitignored; on Actions the
  OddsPapi archive lives in the cache (the history endpoint re-serves the
  current season for free if it is lost).
- The local branch `archive-private` holds the full history with the odds
  data. Never push it, and never check it out (switching back to `main` would
  delete the ignored data from disk).
- The Claude scheduled task `nfl-board-build` is paused; do not re-enable it
  while the workflow runs (two builds would both lock picks).
- The live engine still runs on the laptop. After each graded game it commits
  ledger/live_*.jsonl and pushes (`live.engine.sync_ledger`); the public board
  shows it at the next build.

## 1. No leakage, ever

Every rating, feature and probability used for a game in week N is built only
from data that existed before that game kicked off. The backtest replays the
same code the live board runs. If a backtest number looks too good, assume a
leak before you assume an edge.

- Do not use market-derived pbp columns as model inputs: `vegas_wp`,
  `vegas_wpa`, `vegas_home_wp`, `spread_line`, `total_line`. The market is the
  thing we are measured against, not an input. (Calibration in `model/blend.py`
  is the one place the market line is used on purpose, and it is fitted on past
  seasons only.)
- Hyperparameters are tuned on 2018-2021 only. 2022 onward is holdout. Do not
  retune on the holdout to make it look better.

## 2. Lines are snapshotted, never trusted from memory

nflverse `schedules` overwrites lines in place. `pipeline/lines.py` saves a
timestamped copy every run to `data/lines/`. Picks grade against the snapshot
taken at publish time; the last snapshot before kickoff is the closing line
for CLV. Never type a line in by hand.

## 2b. OddsPapi budget (free plan: 250 billable requests/month)

- Every call goes through `pipeline/oddspapi.py`, which logs it and refuses
  billable calls past 240/month. Never call the API any other way.
- `/v4/historical-odds` is free and returns full line timelines (open, every
  move, close) for up to 3 books per call. It only keeps the **current
  season**: run `python -m pipeline.odds_history` every week or the data is
  gone for good. `data/oddspapi/` is kept in git for the same reason.
- Billable calls are cached forever (`fixtures_t31.json`, `markets_s14.json`).
  Refresh the fixture list at most monthly. Steady state is ~5 requests/month;
  the rest is reserved for the props phase.
- If the quota is ever exhausted, the free endpoint is blocked too. Don't.
- `/historical-odds` allows **one call per 5 seconds** (measured 2026-10-08:
  calls 4.8 s apart got 429, 4.9 s+ were served). The client waits 5.5 s after
  each response ends and honours the 429's `retryMs`. Run the archive with one
  worker. Before this, a run drew 1,135 rejections; now zero.
- Only the "core" group (pinnacle, draftkings, fanduel) is archived. One call
  per game returns game lines AND every player prop, pregame and in-game:
  game lines go to `history/`, props to `props/` (~1 MB a game, zstd).
- Player names: `players_s14.json` (one billable call, 2026-10-08) maps
  OddsPapi player ids; `pipeline/prop_players.py` maps them to gsis ids.
  Billable used this month: 3 of 240.
- An upcoming game is re-fetched every run; once a game has ended (kickoff
  + 6 h) it gets one last fetch if its file was written before then. Builds
  run in the morning, so before 2026-10-09 a game's archive stopped at the
  build before kickoff: no closing lines, no in-game prices.
- `python -m pipeline.odds_history --games <game_id>` refreshes just those
  games (one free call each).
- No earlier seasons: the fixture list has 2024-2025 games, but they are
  marked hasOdds false and /historical-odds returns 404 for them (checked
  2026-10-09). PropLine's archive starts April 2026. The 2025 season's
  closing prop lines came from SportsGameOdds instead (2e). The live engine does this itself when it joins
  a game after kickoff, to start from that game's closing prop lines.

## 2d. PropLine budget (free plan: 1,000 requests a day, reset 00:00 UTC)

- SportsGameOdds was tried for live prices on 2026-10-11 and failed: its in-game quotes were
  5-7 minutes old on the trial plan (DK freshest 295 s, FD 341 s, Pinnacle 395 s at halftime;
  PropLine 81/56/586 s), so the engine priced nothing. PropLine is the default; `--odds sgo`
  remains for testing only.
- Lean PropLine schedule (same day): price checks only at decision points (halftime, after a
  score), for live cards and their 30 s / 5 min re-checks, and a 10-minute heartbeat for the
  log. About 50 calls a game, so a full Sunday fits the 950 a day (`LEAN_CALLS_PER_GAME`).
- A decision point whose prices were all too old (nothing priced, or every candidate failed
  only on quote age) retries every 30 s, up to 8 times, until play resumes; it used to close.

- Every call goes through `pipeline/propline.py` (certifi TLS; Python's own
  store rejects their chain). It logs to `data/propline/request_log.jsonl`,
  refuses past 950 a day, keeps at most 8 in flight and spaces calls 0.25 s.
- One per-event `/odds` call carries every market and book; always pass
  `bookmakers=draftkings,fanduel,pinnacle`. Player count never changes cost.
- PropLine is for live games only. Pregame lines come from the free OddsPapi
  archive. The live engine polls a game after new snaps, at most every 20 s,
  heartbeat 2 min (5 at halftime): ~200 calls a game, ~4 games a day.
- PropLine's CLV grader uses the pregame close, so it cannot grade in-game
  bets; live line value is measured from our own polls.
- PropLine refreshes each book on its own cycle. On 2026-10-08 in-game quotes
  were 25-160 s old when they arrived (Pinnacle often the oldest); on 2026-10-11 live game lines
  refreshed only every 2-4 minutes per book (DK 155-264 s old, FD ~2 min, Pinnacle 141-205 s). In
  play the parser skips anything older than 90 s (`MAX_AGE_S`).
- Dead balls (engine 2026-10-11.7): after a score until the next snap, and at halftime, the game
  cannot change, so a quote counts when the book posted it after the play that froze the game
  (ESPN's wallclock) plus a repricing margin (DK/FD 20 s, Pinnacle 60 s; `SINCE_MARGIN_S`), whatever
  its age, and a quote posted before that play does not count however young (it priced another
  game). Under the 90 s rule after-score windows almost never priced ("no game-market prices fresh
  enough", over and over), and when they did, pre-score quotes caused the "model more than 10% from
  the market" fails. After a score the rule covers game markets; at halftime props and game markets.
  It holds until the next snap, so the 30 s fill re-check uses it too.

## 2e. SportsGameOdds archive (Pro trial, pulled 2026-10-09)

- `python -m pipeline.sgo --pull 2024 2025 2026` (one call every 2 s, our own
  cap 2,000 a day, every call logged to `data/sgo/request_log.jsonl`, never the
  key; `SPORTSGAMEODDS_KEY` in .env). Billing is per game returned, so the whole
  archive took 65 calls. Raw games: `data/sgo/raw/` (gitignored, 56 MB); tidy
  tables from `python -m pipeline.sgo_tidy`: `data/sgo/props.parquet`,
  `prop_quotes.parquet` (every book's quote), `game_quotes.parquet`.
- What it holds: nothing before Super Bowl LVIII (Feb 2024), whatever its FAQ
  says. 2024: props mostly at minor books, last quotes only, a pregame quote
  for ~2% of props: unusable. 2025: every prop at ~20 books, but only each
  book's LAST quote, plus a consensus "close" that is really the last
  consensus, in-game prices included (Harold Fannin Jr.'s "close" was 94.5
  when every pregame book had ~37.5). Never use `closeBook*`/`closeFair*` for
  2024-2025. 2026: real per-book `openOdds`/`closeOdds` at every book.
- The clean 2025 close is a book's last quote before kickoff (2 min grace).
  Checked on 2026: it equals the book's official close 99% of the time (line;
  97% line and price). Pinnacle and bet365 leave one on ~62% of 2025 props.
  DraftKings and FanDuel trade props in-game, so their 2025 closes are gone;
  bet365 (a US book that prices props the same way) stands in.
- SGO's settled stats (`score`) match nflverse 98-99.6%; we grade with nflverse.
- No in-game history (the live rules cannot be tested on it) and no 2024-2025
  game-line opens (nothing new for game lines; nflverse already has closes).
- Alt lines (`--alt`, 2026-10-10, 132 more calls): `data/sgo/raw_alt/` (134 MB,
  gitignored) -> `data/sgo/prop_line_history.parquet`. SGO lists every line value
  a book ever hung with its last price and time, which rebuilds a line at any
  moment (`props/line_history.py`). Checked on 2026 against OddsPapi: Pinnacle
  props right 98.9% at the lock (price within ~1 point of win chance).
  DraftKings, FanDuel and bet365 do not rebuild (24-80%: they hang near-even
  alternate lines next to the main one), and neither do game lines
  (`backtest/sgo_lines.py`: books trade them in-game, which overwrites every
  pregame value; 5% of 2026 Tuesday spreads right), so no 2025 Tuesday-line
  test for game lines.
- `expandResults` only adds derived totals (first downs, rates, sums) that
  nflverse already has. The rest of the API is metadata or AllStar-only streaming.

## 2f. Who plays (pipeline/starters.py, 2026-10-15)

- nflverse's schedule QB is a label, not the box score: 10 of 158 team-games in 2026 weeks 1-5 named
  the wrong one (Seattle weeks 3-5: Drew Lock, while Sam Darnold took every snap), and it carries
  labels into upcoming games (Seattle at Denver: Lock). `pipeline/clean.py` now sets each game's QB:
  played games, the starter (the team's first dropback; known before kickoff, so no leak); upcoming
  games, the depth chart's top quarterback who is on that week's active roster and not Out or
  Doubtful (nflverse depth charts, refreshed daily). Game backtest: holdout margin MAE 9.862 -> 9.850,
  total 10.382 -> 10.366; tuning years within noise. The weeks 1-4 game-line simulation on the board
  went 81-74-3 +21.0u -> 78-76-3 +14.8u, all from the corrected QBs (it had run on the wrong ones).
- Props roles (`assign_roles`): a player with no games gets his position's prior share, a starter's,
  so usage alone put backups at QB1 for five teams on 2026-10-15 (BAL Fagnano, NE Morton, NYG Haener,
  WAS Kaliakmanis, SEA Milroe) and practice-squad or just-off-IR receivers at WR2 (Irvin Charles over
  Cooper Kupp). Now roles go to the current depth chart's starters (QB the top available one, RB and
  WR the top three, TE the top two), ordered by projected usage, players with real usage first; a
  fullback ranks behind every running back. Past games (the props backtest and the board's simulated
  props) use each game's starting QB as QB1: backups had been QB1 in 3.3% of past team-games; holdout
  passing-yards MAE 60.9 -> 59.1, other markets unchanged. Distributions recalibrated
  (`python -m props.backtest --no-tune`): miscalibration pass yds 0.040 -> 0.036, rec yds 0.019 ->
  0.016, receptions 0.028 -> 0.023, rush yds 0.020 -> 0.029.
- Tested and not used: skipping games a player left early (under half his usual snaps) when computing
  usage shares. Worse in every market on the holdout (pass yds MAE 60.9 -> 62.2, bias -0.2 -> +5.2):
  a starter who leaves early is more often limited the next week, so the diluted share carries signal.

## 2c. Compare to Tuesday lines with Tuesday information

When the model is compared to the line at publish time, build features with
`as_of_publish=True` (last week's starting QB, no game-day weather). Using the
actual starter or actual weather leaks information the Tuesday line did not
have.

## 3. Picks are locked

`ledger/picks.jsonl` is append-only. A pick, once published, is never edited
or deleted. Grading appends a result record; it does not rewrite the pick.
A change of category is an appended `parlay_type` line and a change of stake
an appended `stake` line, and only before the game kicks off.
The same holds for `ledger/props_picks.jsonl` / `props_results.jsonl` and the
live files (`live_flags`, `live_events`, `live_accepts`, `live_results`).
Demo replays of the live engine write only to `data/live/demo_ledger/`.

## 4. Units

- Prices come from DraftKings and FanDuel only (`SHOP_BOOKS`); Pinnacle is
  the reference. Game-line picks before 2026-10-08 were shopped across 8 books.
- Stakes live in `model/staking.py` (user, 2026-10-15). A straight bet is
  staked 1.0u only when its EV is positive. A negative-EV side is a **lean**:
  it stays on the board with `risk: 0`, meaning no stake. That is advice.
- **The record counts every pick at 1u, leans included** (user, 2026-10-18).
  It is the model's full record. Without the leans it would just repeat the
  Kelly section's "same picks at 1u". `ledger.record` and `grade_all` grade
  straight picks at 1u whatever their stake; the TNF leans already graded at
  0u are recomputed. A note splits the record into +EV picks and negative-EV
  sides. Props picks are all +EV already, so props have no leans.
- The lean rule starts with week 6. Week 6's 27 negative-EV game picks were
  locked at 1u on 2026-10-13. They became leans through appended `stake`
  lines at 07:06 ET on 2026-10-15, before any week 6 kickoff
  (`ledger.read_picks` applies them). Weeks 1-5, replay and live, keep 1u:
  they helped choose the rule, and re-staking them would add ~11u from the
  same data that picked it. The replay forces 1u on its picks for that
  reason. Never re-stake a pick after its game starts.
- **Quarter Kelly is tracked; it is not the stake.**
  - Every +EV pick, prop and live flag carries `kelly_u`: 0.25 × full Kelly
    (pushes refunded) × 100, with 1u = 1% of the bankroll, rounded to
    0.25 / 0.5 / 1 / 2u at geometric midpoints.
  - Its record runs from week 1 as its own section, apart from the flat
    record (user, 2026-10-15). "Same picks at 1u" and "week 6 on" sit beside
    it. Week 6 on is the out-of-sample test, because weeks 1-5 helped choose
    it.
  - It becomes the stake only if it holds up for a few weeks.
  - No cap on total weekly exposure: the bankroll is the user's call.
- Why Kelly is tracked first: sizing by the model's confidence amplifies its
  mistakes as much as its edge. On the 2022-26 backtest at closing prices,
  quarter Kelly lost 6.4% per unit staked against flat's 2.7%, because the
  model overrates its biggest edges. 2026 weeks 1-5 went its way, helped by
  +EV spreads going 13-0-1 at stated ~50-58%, a run calibration says will not
  last. Read the calibration table before promoting it.
- Live plays: 1.0u flat on the record (user's choice). `kelly_u` rides on
  every flag and in `paper_kelly`. Every live flag so far sized 2u, so that
  record is the flat one × 2.
- Golden Parlay: risk 0.5u.
- Picks of the Week: top 3-5 by EV that clear the backtested threshold. If
  fewer clear it, show fewer. Zero is a valid week.
- Golden Parlay (0.5u): best-EV combination of 2-3 positive-EV picks from
  different games priced +150 to +300; stretches to +500 only when nothing
  fits; otherwise none that week. The user wants a ticket with a reasonable
  hit rate, not a long shot.
- +EV Lottery (0.5u): best-EV combination of 3-5 positive-EV legs priced over
  +500, no price cap. Each leg must be +200 or shorter, or "best EV" builds
  +170,000 tickets from underdog moneylines. Counts in the units total (the
  user does not mind the skew).
- Backtest (backtest/parlays.py): no parlay rule was reliably profitable at
  closing prices; parlays only pay when every leg beats its price.
- Week 5's original +1423 parlay was reclassified as that week's Lottery by an
  appended `parlay_type` line; never rewrite ledger lines.

## 4b. Good pick or +EV only (user, 2026-10-18)

The user picks from the board rather than taking every pick, and +EV is not the same as a good pick. Every +EV
pick (game lines and props, not leans or parlays) carries checks (`model/checks.py`; game lines
`board/checks.py`, props `props/checks.py`). It is a **good pick** when every check that applies passes,
otherwise **+EV only**, with the reasons shown on the pick.

Each check was tested on its own market, with the EV at Pinnacle's closing no-vig price as the yardstick
(less noisy than wins and losses). Props get sharp, size, lean and who. Game lines get market and who.

- **sharp** (props): Pinnacle has the prop, so the edge is DK/FD against the sharp price. Without Pinnacle the
  edge would be DK against FD.
- **size** (props): EV of 2% or more. Line-shopping props at 2%+ beat the close 81% vs 70% for all (SGO 2026).
  From week 7 the pick rule itself needs 2%+ (6b), so every new prop passes it; it still grades weeks 1-6.
- **lean** (props): overs on rec yds, rush yds and receptions need 5%+. Pinnacle's closes made those overs 2-5
  points likelier than they turned out in 2025 and 2026; passing yards showed no such bias.
- **market** (game lines): Pinnacle hasn't moved against the pick (1.5 points of win chance) since its Tuesday
  lock. Tested on weeks 1-5, model side at Tuesday prices, 179 bets. When Pinnacle moved toward that side by
  the last update, that update's price on it still closed at +0.8% EV (+3.0% on the +EV picks). When it moved
  against, the close was -4.5% (-1.4%). Props lock at the last update, so they have nothing to compare.
- **Rejected for game lines (2026-10-18, weeks 1-5 at Tuesday prices):**
  - **Reading Pinnacle's juice**, in the pricer or as a check. The juice-aware pricer's 31 picks closed at +1.8%
    EV against the board's 38 at +3.5%. The 15 board picks it would drop closed at +3.6% and the 8 it would add at
    -2.9%. Tuesday juice is small (0.15 points) and barely tracks the close (correlation 0.12). The pricer stays
    as it is: Pinnacle's line read as a coin flip.
  - **A 2% edge minimum.** Model-side bets at 0-2% closed at +4.3%, 2-5% at +1.2%, and at or below 0% at -3.8%.
    The sign of the EV matters, not its size, which also supports the lean rule.
- **who**:
  - Props: not Questionable or Doubtful, not within 3 games of changing teams, not promoted.
  - Game lines: no QB on either team Questionable or Doubtful. Out is settled news the model already uses.
- **Deliberately not a check:** the model-market gap. In 2022-25 the model side covered 51/52/54/44/51% from a
  gap under 1 point to over 4, so a big gap predicts nothing.
- **Tested for props and left out (2026-10-22, user asked):** matchup, form and usage, on 8,104 props with a
  Pinnacle close (SGO 2025 + 2026 weeks 1-6), each prop's over result against the close's no-vig chance, net of
  its season-and-market average miss. Matchup (the opponent's season-to-date rank allowing that stat to that
  position, 3+ games): overs vs bottom-5 defenses +2.5 points (SE 1.6), vs top-5 -2.0 (SE 1.6); 2025 only,
  2026's 72 each showed nothing. A weak lead, not a check. Form (cleared this line 4-5 of the last 5, or 8-10 of
  10): -1.1 / -0.3, noise (hot streaks if anything slightly overpriced; cold ones +1.6 / +1.0). Usage spike
  (last game's targets/carries/attempts 30%+ and 3+ (5+) over his average): -1.2, noise. The market prices
  what everyone can see; the board shows them as facts on the cards, not checks. Pinnacle's 2026 opens moved
  toward the over by the close 61% of the time for cold players (276 moves) vs 52% for others: openers may
  overreact to cold form. One season, one look: re-test at season's end before acting.
  Matchup, pushed harder the same day (props/section.py `matchups()`): every prop with a Pinnacle close, else
  bet365's (7,393 with a defense rank), the defense as a percentile. Passing game (pass yds, rec yds,
  receptions): overs vs the softest fifth +4.5 points beyond the close (SE 1.4), the toughest fifth -2.5 (SE
  1.4); +2.2 and +2.6 points per SD of softness in 2025's two halves; 2026's 462 flat (SE 2.4). Rushing yards
  went the other way: overs vs the softest fifth of run defenses -7.9 (SE 2.8). No prop lines exist before
  2025 in any source we have (SGO 2024: 225 props). Shown as a tag on passing-game picks and leans ("Soft /
  Tough matchup", agreeing with the side or against it); not a check, not tracked (user, 2026-10-22).
  By stat and position (2025): softest-fifth overs QB pass +10.4, RB rec yds +9.0, RB catches +6.5, TE catches
  +5.2, TE rec yds +5.1, WR rec yds +4.5, WR catches -1.8; rushing reversed for RBs (-7.6) and QBs (-11.9).
  Rushing props get a "Soft / Tough run D" tag that agrees with the UNDER against a soft run defense. Picks and
  leans whose edge is a better number than Pinnacle's get a "Better number" tag (props/edge_source.py).
  Not in pricing (user, 2026-10-22: wait for the next calibration). The re-check, set now: 2026 weeks 7 on only
  (Pinnacle closes from our own archive), at the lock and at the close; it passes if catching/passing props
  still show +1 point or more per SD of softer defense and rushing still runs the other way. Pass: a price nudge
  at HALF the measured size, judged by EV at the close like the 2%/4% rule. Fail: tags only, or drop them.
- **Price now:** each pick also shows the best DK/FD price and EV as of the last build.
- **Freezing:** each build appends a verdict line to `ledger/pick_checks.jsonl` (append-only) when a pick's
  verdict changes before kickoff. The record uses the last line before kickoff. Older picks are rebuilt from what
  was known before kickoff: Pinnacle at lock and at the last update, and that week's injury report.
- **Record:** good vs +EV only, flat 1u. Weeks 1-5 are in-sample, because the checks were chosen with them in
  view. Week 6 on (`SINCE_WEEK`) is the test.
- **Weeks 1-5, in-sample:** props good 50-37 (+10.9% ROI) vs +EV only 111-101 (+0.8%), as published
  2026-10-18. Game lines are too few to say.

## 5. Weekly cadence

The board updates three times a week (scheduled build, 10:05am ET, Tue / Thu /
Sat). Picks lock at the last update before their game; Tuesday is the grading
day (user's rules, 2026-10-09; `props/pricing.lock_time`):

- **Tuesday 10am ET**: grade every finished game (Monday Night Football
  included), refresh data, rebuild ratings, then lock the week's game-line
  picks and publish the new week's board. Game lines stay on Tuesday on the
  evidence: replaying weeks 1-4 with game lines locked at the last update
  instead (Thursday/Saturday) cut spreads from 35-26-3 (+6.5u, line value
  +0.40) to 31-30-3 (-1.5u, +0.23); all markets +27.5u -> +11.8u. Totals did
  slightly better later (+0.2 pts, 64 games): watch it, don't split markets
  yet. Re-test with `python -m board.build --offline --dry-run --sim-lock last`
  (or `last-info`) as weeks accumulate.
- **Thursday 10am ET**: lock the Thursday game's props. (Locking them Tuesday
  instead went 0-3, line value -1.00, vs 9-3 at Thursday-morning lines in the
  replay; fewer prop lines are up and DK/FD have not split from Pinnacle yet.)
- **Saturday 10am ET**: bring in Thursday night's results (graded game lines
  and props, live record), refresh injuries and lines, lock Saturday, Sunday
  and Monday props (after Friday's final injury reports), plus the props Picks
  of the Week, Golden Parlay and Lottery. A build outside a game's lock window
  only re-projects; locked picks never change.
- A Wednesday or Friday game locks at the Tuesday or Thursday update before
  it (the week 1 Wednesday opener used to lock the Saturday before).
- **Game days**: `python -m live.engine` on the laptop (paper trading).

## 6. Report the record honestly

The record strip shows ATS, O/U, ML, Picks of the Week, Golden Parlay, units,
and CLV. Show sample sizes. Calibration (stated probability vs actual hit
rate) matters more than win rate. If the backtest says the model only tracks
the closing line, the board says so.

**By category (user, 2026-10-26).** Every record page (NFL lines, NFL props,
NBA lines, NBA props) has a "By category" table under Season to date: each
market, side, Picks of the Week/Day, good picks, +EV picks, leans, EV tiers
and parlays, with record, units, line value and EV at Pinnacle's close
(`closeByCat` in the template; NFL lines' `ev_close` from `board/build.py
ev_close`, NFL props' from `props/section._ev_close`, NBA from the ledgers).
"Read" says beating / losing to the close only past 2 standard errors on 20+
picks. The user picks a subset of the board to bet; this is the table that
should drive that choice, not recent wins and losses.

Leans lose about the vig at the close by construction (their price is already
against them), so a row that includes them reads "losing to the close" whatever
the model does. Through 2026-10-28: NFL leans -3.5% (weeks 5-8 and the 1-4
replay), NBA -3.9%; +EV game-line picks NFL +0.7% (weeks 5-8) / +3.1% (replay),
NBA +0.1%. The Spread and Over/under rows are mostly leans (NBA 124 of 146
game-line picks). Read the +EV picks row and the bet list.

Walk-forward test behind it (2026-10-26, scratchpad walk_nba.py / walk2.py):
- **Fading a cold category is ruinous**: on the NBA props backtest, betting the
  other side of categories with a losing last 3-30 days returned -15% to -18%
  (two vigs). Never suggest fading the model's own picks by recent results.
- **Hot-category carry**: NBA props, a category's long-run level predicts the
  next day (coef ~0.5); its last-3-days streak beyond that adds a little
  (0.13, se 0.04), gone by 14 days. NFL closing lines: nothing either way.
- **Top of the day**: the day's 5 biggest NBA edges returned +9.6% vs +5.0%
  for the rest (both seasons the same way): Picks of the Day being the best
  subset is expected, not luck.
- Recent results are noise next to EV at the close: NBA threes went 7-20 with
  21 of 27 still +EV at the close (+2.1%).

## 6b. Player props (props/)

- Monday night's props lock Monday 10am ET (user, 2026-10-26), from a Monday build (board.yml, slot.py and the
  laptop catch-up now include Monday). Before, they locked Saturday with the weekend and the week 7 game found
  nothing two days out; it was relocked at the user's request Monday 5pm (still no pick: three edges the old
  rule would have taken, off the books). Monday games before 2026-10-26 keep their Saturday lock
  (`pricing.MONDAY_LOCK_FROM`), so the replay and the record are unchanged. Picks of the Week and the parlays
  still lock Saturday, from the Saturday and Sunday games.

- Tracked roles only: QB1, RB1, WR1, WR2, TE1, RB2 in a real timeshare
  (projected run share >= 30%). Roles come from projected usage among players
  not ruled Out/Doubtful, so an injury promotes the next man up (tagged
  `promoted`, which has its own record line).
- Model: team volume (plays, pass rate) x player share x efficiency, decayed
  and shrunk; distributions simulated from real per-play yardage, calibrated
  on 2018-2021. Tuned on 2018-2021 only; 2022-2025 is holdout.
- The game's spread and total ARE inputs to the props model (deliberate: the
  props market is what it is measured against). Never props-market numbers.
- Pricing: anchor = Pinnacle's no-vig location (else DK/FD average), moved k
  toward the model; k is re-measured every build (about 0.005 so far). With k
  that small, picks are mostly DK/FD numbers off Pinnacle's ("Line shop" tag).
- Findings (2026-10-08, weeks 1-4): the model beats simple averages but not
  closing prop lines (Brier worse); lines move toward it ~52%. The edge that
  showed up is line shopping vs Pinnacle; pre-launch sim 146-117 +33.1u with
  75% of moved lines going our way (144-118 +20.2u before the Wednesday
  opener locked Tuesday; nearly all of the difference is one +2245 Lottery
  that now cashes by a leg swap, so the units are luck). Treat as unproven
  until the live record.
- Live props start week 5 (`PROPS_LIVE_START`); weeks 1-4 are a labeled sim.
- Two seasons of closing lines (2026-10-09, `python -m props.sgo_study` ->
  reports/props_sgo_study.json, on the board's props "How it works"):
  - The projections add nothing beyond the close. 2025 (holdout), 4,836
    tracked props at Pinnacle's close: log loss market 0.6869, model alone
    0.7045, best mix (k 0.1) 0.6865; the model gap's logistic weight
    +0.05 +/- 0.07 (2026: -0.16 +/- 0.12). Keep k near zero.
  - The board's rule (DK/FD vs Pinnacle no-vig, k 0.005, EV >= 0) by when the
    price is taken: 2026 opens 382 bets +29.5u (+7.7%), lock (Thu/Sat 10am)
    295 bets +20.0u (+6.8%), both ~70% still +EV at Pinnacle's close; 2026
    closes 209 bets -19.1u. 2025 closes (bet365 vs Pinnacle) +59.5u on 986,
    all from unders (+62.7u; overs -3.2u). The edge, if any, is DK/FD lagging
    Pinnacle during the week; it is gone by kickoff.
  - Closing prices overrate overs in receiving and rushing props by 2-5
    points in both seasons (Pinnacle no-vig vs hit rate); passing yards not.
    Plus-money unders (under >= +100, nearly all receptions): 2025 bet365
    close +3.3% (809), 2026 DK/FD open +5.8% (308), lock +4.0% (242), close
    +7.6% (310); other unders +1.0% (2025) / -3.8% (2026). Every interval
    includes zero and the rule was found on these seasons: a candidate for an
    off-record track, not a pick rule. The model's lean does not help pick
    among them (helped in 2025, not in 2026).
  - Lock time, 2025 (Pinnacle's rebuilt line at Thu/Sat 10am ET, 4,769
    props): 947 lines moved before kickoff, 57% toward the model (60% of the
    275 with the biggest gaps); k 0.012 (95% 0.003-0.021), the same as this
    season's; log loss at the lock line 0.6872 market -> 0.6866 with k 0.15.
    The model knows a little the Saturday line does not, too little to bet on.
  - Anchor: Pinnacle alone beats averaging books (2025, same line: Pinnacle
    0.6845, bet365 0.6874, average 0.6856; 2026's 599 lines are noise).
- Where the edge comes from (2026-10-20, after live weeks 5-6 went 38-51;
  `python -m props.edge_source` -> reports/props_edge_source.json, on the
  props "How it works"). Every tracked prop of weeks 1-6 at the lock, the best
  DK/FD side priced the board's way, judged at Pinnacle's close the same way:
  a better NUMBER than Pinnacle's held up (245 bets, +4.9% at the lock, +4.4%
  at the close, 72% still +EV); the same number with a better PRICE did not
  (137, +1.6% -> -0.5%, 54%). Under 2% nothing held (better number -0.1%,
  same -1.2%); price-only edges held only when large (4%+). The SportsGameOdds
  2026 opens split the same way (0-2%: -0.4% / -1.8% at the close). Weeks 5-6
  had few and small better-number chances (about 19 a week at +2%, against
  about 55 at +5-7% in weeks 1-4), so the one-prop-a-player rule filled with
  price-only edges: the live picks were worth -0.2% at the close (the weeks
  1-4 replay's +4.0%). Week 6's 18-26 had a 7.6% chance at the stated
  chances; injuries and game script explained 2-3 losses a week, not more.
  A rule of 2%+ EV, and 4%+ when the edge is the price alone, keeps 155 of
  the 392 bets (+6.9% at the close, 82% still +EV).
- The pick rule from week 7 (user, 2026-10-20; props/section.py `RULE_WEEK`,
  `clears()`): 2%+ EV, and 4%+ unless DK/FD hang a better number than
  Pinnacle's at the lock (the same number, a worse one, or no Pinnacle number
  all need 4%). Weeks 1-6 keep any positive EV: the replay and the ledger stay
  as they locked. One prop a player, the best by Kelly among those that clear.
  Card leans follow the same bar (`clears` in the payload). The 2% part is the
  main gain; the 4% part rests on thinner samples.
- Shadow record (ledger/props_shadow.jsonl, append-only, from week 7): at each
  lock, the picks the old rule would have added. Off the books: graded each
  build and judged at Pinnacle's close (`_ev_close`, k = 0, the edge_source
  yardstick) on the props Record page beside the rule's own picks. A game with
  picks or shadow picks counts as locked. Revisit after a few weeks: if what
  the rule passes on is worth about nothing at the close, the rule holds; if it
  closes clearly +EV, the rule is costing value (split the shadow into its
  under-2% and 2-4% price-only parts before changing anything).
- Over premium, measured each build (user, 2026-10-27, from the NBA's learned correction; props/shade.py, the
  props Record page). A report beside the lean rule, not a price input: per market, the logit shift between
  Pinnacle's no-vig close and over hits on tracked props, MAP with a prior from 2025's Pinnacle closes (pass
  +0.5 pts at even money, rush -1.8, rec yds -3.4, receptions -5.0; sd 0.10 logit) and a 4-week half-life,
  plus weeks 7+ alone (the lean rule's out-of-sample test), and what an over clearing its bar (5% lean, 2%
  floor) is worth at -110 once the premium is counted. First look (weeks 1-6): raw gaps pass +4.3 (+-3.7),
  rush -6.4 (+-2.7), rec yds -2.4 (+-1.9), receptions -1.4 (+-1.9); learned rush -3.6, rec yds -3.0,
  receptions -3.4, so an over at the 5% bar is worth about -1% to -2%. Read it at checkups; any change to the
  lean bar is the user's call, judged at the close.

## 6c. Live engine (live/)

- Page layout (2026-10-11, for a full slate): today's summary; live plays now (each tagged
  with its game and clock); the slate, one card per game grouped live / upcoming by kickoff /
  final, holding that game's plays and record; every play today as a timeline; the engine log,
  filterable by game and by kind. Game state comes from ESPN (play-by-play, clock, possession,
  box score, injury notes); SGO has none of the play-by-play, so it only supplies prices.
- Runs locally: `python -m live.engine` (every game in the next 48 hours; page http://localhost:8770, `--lan`
  for the phone, `--demo <game_id>` replays a finished game with no PropLine
  calls). The published board cannot reach it; it shows the ledger after games.
- Redesign (2026-10-09, user: fewer, more trustworthy plays): the engine
  watches and prices every play for the log but flags only at decision
  points. Props: once, at halftime (`PROP_WINDOWS`), the single strongest
  play. Game markets: after each score and at halftime, vs Pinnacle's live no-vig number, one
  a game. Rubric (`decide.gate` + two checks): model within 10 pts of no-vig
  (`MAX_GAP`), EV >= 5%, same price on two checks 45 s apart (90 at half),
  quote <= 60 s old (in a dead ball: posted after the play that froze the game, any age),
  no props in a 17+ point game, no game prices beyond
  +/-300, one per player, correlated plays once; props also skip a player
  whose numbers lean on one 20+ yard play (half his yards in that stat, or
  one in the last 2 min) and one the play-by-play says was hurt until he
  touches the ball again (live/rubric_checks.py; the engine reads yards per
  play from box-score changes between polls). Each game starts calibrated to
  its closing prop lines. Cards live 3 min (6 at halftime).
- Halftime props roll (user, 2026-10-11 2:45pm ET): a check every 90 s through the whole break; the
  first prop that holds on two checks in a row (same book, side and line, still passing) is flagged;
  when nothing holds, what passes now is the new first check; play resuming with nothing held is a
  pass (cap 20 checks). On the 64-game replay: rolling 61 plays 38-23 +12.4u vs one pair of checks
  53 plays 32-21 +9.8u; the 9 plays only rolling finds went 6-3. One check only (no confirmation)
  was worse: the plays the second check rejects went 10-10 (reports/live_rolling64.parquet).
- Game markets decide after each score AND at halftime (engine 2026-10-11.7; user: "our model checks
  after each score", so the after-score checks stay). Halftime's window decides both kinds, each
  rolling on its own clock (64-game replay, game markets at halftime: 28 plays 16-12 +4.8u; after
  scores 39 plays 22-17 +3.7u; the Q1 and Q3 breaks lost and are not used). One game-market play a
  game, whichever decision point finds it first. An after-score window that is open when the half
  ends folds into halftime. ESPN posts a touchdown at +6 and folds the try into the same play about a
  minute later, so a touchdown's window waits for the try (`TRY_WAIT_S`, at most 150 s) and prices
  the settled score. Retries (no quote posted after the score yet) do not use up the pair of checks.
  A rolling window retries three times at 30 s, then at its own 90 s pace, so the 20-check cap lasts
  the whole break. Live-path test: a scripted ESPN feed and PropLine quotes drive `GameRunner.step()`.
  Expect more PropLine calls a game than the lean estimate (each score now prices until the books
  post): measure and re-set `LEAN_CALLS_PER_GAME` before a full 13-game Sunday.
- The scoring play's own wallclock starts the dead ball (engine .8): ESPN's score can lead its play
  list (IND-PIT, 15 s), and the previous play's time let pre-score quotes count. Until the play
  arrives, the time the score was seen is used (never too early).
- Game markets judge the book's line on its merits (user, 2026-10-11, engine .9: "the goal is to find
  a line and see if it's a good line"; Pinnacle hangs other numbers than DK/FD, e.g. 5.0 vs 4.5, 56.5
  vs 55.5, so a same-number rule judged almost only moneylines). Pinnacle's reference at DK/FD's
  number: its own price there (main line or one of its alternates, which PropLine carries), else its
  nearest number within 2 points (spreads) or 3 (totals), moved by the model's chance of the result
  landing between the two numbers (`_pin_fair`, `LINE_SLACK`). The gap check then compares the model
  with Pinnacle at Pinnacle's number; EV stays the model's at the book's price. DK and FD are both
  candidates, so the better line wins (line shopping).
- Slack set to 1 point (engine .10) from a paired replay (64 games, the model's draws held fixed;
  earlier unpaired runs were noise-dominated). Game markets after scores + halftime: same number only 68
  plays +9.5u; 0.5 pt 95 plays +7.4u; 1 pt 100 plays +6.2u; 2-3 pts 104 plays +4.0u. The extra plays are
  about break-even, worse as the slack grows (moved spreads lost, moved totals won; noise-level). The
  archive has one line per book, so the replay always moved Pinnacle's number by the model; live, Pinnacle's
  own alternates often give the exact number. `LiveModel` sorts its teams: group_by's order changed run to
  run and the teams share one random stream, so replays moved +/-2u from noise alone.
- PropLine serves Pinnacle's frozen pregame markets alongside its live ones during a game, both with
  fresh timestamps (DEN-LAC 7-0: Pinnacle -3.5 / 41.5 / -182, the pregame numbers, while DK/FD were
  -7.5 / 43.5 / -346). The limits tell them apart: live 1,000-5,000, pregame 20,000-100,000. In a live
  game the parser skips any Pinnacle market above `PIN_LIVE_MAX_LIQ` (10,000). Many earlier "model more
  than 10% from the market" fails were the model against pregame Pinnacle prices.
- Live game markets are spreads and totals only, and line-shop (engine .11, `decide.GAME_MARKETS`). The
  reference is Pinnacle's live number within 1 point, else the other US book's (FD for a DK line, DK for
  FD), moved the same way: Pinnacle's live lines refresh slowest. Paired replay (64 games, one game-market
  play a game at the first decision point that finds one, after scores and halftime;
  reports/live_game_refs2.json): with moneylines, Pinnacle same number 59 plays 29-30 +4.3u, Pinnacle 1 pt
  63 plays -1.9u, Pinnacle-else-US 64 plays -0.4u; without moneylines 49 plays +5.9u, 63 plays +6.4u,
  64 plays 39-25 +9.1u. Moneylines alone went 15-22, 11-18, 10-19 under the three references.
- Replay bug fixed (2026-10-11): `live/replay_windows.select` matched the two checks with a join on the
  line, and a moneyline's line is null, which never matches, so the after-score replay never took a
  moneyline; its "39 plays 22-17 +3.7u" was spreads and totals only. Now `nulls_equal=True`.
- Pure line shopping against Pinnacle (flag DK/FD when its price beats Pinnacle's live no-vig at the
  same number, two checks in a row, no model) rarely fires at decision points: 6 plays in 64 games at
  3% (4-2), 3 at 5% (reports/live_line_shop.json). Not added; revisit if live logs show more.
- Missed decisions can be replayed onto the record (`python -m live.replay_missed <game> --record`),
  marked "replayed", only with a commitment in ledger/live_replayed.jsonl pushed before the game ended
  that names the method; whatever it picks counts, win or lose. 2026-10-11: PHI-JAX (two-checks,
  committed 11:48am) and seven 1pm games (rolling, committed 2:41pm) wait for nflverse's play-by-play.
- Shadow picks (user, 2026-10-09): each game's second-best play of each kind
  that passed the whole rubric and lost only to the one-a-game cap is logged
  to ledger/live_shadow.jsonl and graded in live_shadow_results.jsonl, never
  on the record. Game-market checks keep running after the cap is used, until
  the game's shadow is found. If shadows hold up, a second slot is earned.
- Why one prop and one game market a game: in the replay each game's best
  prop went 35-20 (+12.1u), its 2nd-best 25-22 (+1.8u), 3rd-best 10-21
  (-12.3u); even 2nd+ plays with EV >= 8-10% lost. Game markets: first check
  22-17 (+3.7u), later ones 5-9. Same-game results were not correlated
  (47% same), so the cap is about quality, not correlation.
- Every flag counts in the paper record at the price and line it was flagged
  (user's rule, 2026-10-09: a flag uses one of the game's slots, so a flag
  that does not count is wasted information). The 30 s re-check is reported
  separately: how often the price was still there. A quote too old to judge
  (`stale` in the odds update) means "can't tell yet", never pulled or
  missed; with no fresh quote 150 s after the flag the re-check is "unknown".
  Accept records the user's real bets. Go-live gate: 4 paper weekends, 40+ graded
  plays, positive line value.
- Status: the engine's page has a status light (green: running, ESPN answered
  in the last 45 s; amber: starting, waiting for kickoff, ESPN quiet, or an
  error in the last 2 min; red: the page cannot reach the engine, or a game
  watcher died; gray: idle or all final) and an engine log: every play, every
  price check with the best price the rules could take and why nothing was
  flagged, flags, fills, line value, grades, warnings, errors. Cards show how
  old their quote was. `data/live/status.json` (20 s heartbeat, gitignored)
  gives the board's Live tab a "last heard from" line and the log's last
  lines; the published board cannot show the engine as it runs.
- The board's Flagged plays (Live tab) sit under week headers with the week's record, one dropdown
  per game (user, 2026-10-13: 40+ plays in one list is a scroll), newest first; the header shows plays,
  W-L, units and pending. A game with an ungraded play opens by itself; the open state is kept per
  device (localStorage kn-live-open).
- The Live tab's Record compares the kinds (user, 2026-10-13: do props or game markets have the higher
  edge?): a card each with the live paper record, the edge stated at the flag, line value, and the replay
  of its current rule (reports/live_props_rolling.json: rolling halftime props 61 plays 38-23 +12.4u;
  reports/live_game_refs2.json both_noml: game markets 64 plays 39-25 +9.1u). Tallies only, no prices.
- Restarts are safe: each game keeps `data/live/<game_id>/` (calibration.json
  plus play and odds logs). A restart restores the game's flags from the
  ledger (cap and exposure rules still hold; their fills show "unchecked")
  and its calibration. ESPN's play-by-play ran 40-70 s behind the wallclock.
- Replay of weeks 1-4 against archived in-game prices. Old rule (any play
  after Q1, live/replay.py): lost. The redesign (live/replay_windows.py, 64
  games, every flag counted at its flagged price): halftime props, one a game,
  within 10 pts: 35-20 +12.1u (said 58%, won 64%, every week positive); game
  markets after scores, one a game: 22-17 +3.7u (said 56%, won 56%). End of
  Q1 lost every way (-18u within 10 pts); disagreements of 10-15 pts went
  46-80 -38.8u (said 62%, won 37%); shrinking toward the market left only
  those and lost. Small samples: paper trading decides. Re-score saved
  candidates in seconds: `python -m live.replay_windows --from-saved`.
- In-game line value is not a reliable signal for props: the losing big
  disagreements showed the best line value (lines drift mechanically).

## 7. Attribution

FTN charting data (2023+) must be credited "FTN Data via nflverse" anywhere it
is displayed. NGS participation data (2022 and earlier) is "NFL NextGenStats
via nflverse".

## 8. The board

Published at https://claude.ai/artifact/9zNLyz7eKpoBcTo9Ybejtk ("Key Number"),
built from `board/template.html` by `python -m board.build`, which refreshes
nflverse and OddsPapi data, measures k (how far lines move toward the model
between Tuesday and kickoff), locks the week's picks, grades finished ones and
writes `board/nfl_board.html`. Republish that file to the same URL; omit
`capabilities` on redeploys so `db` + `user` (My Picks) carry forward.

- Run it Tuesday morning: grades last week, locks this week. `--dry-run`
  computes everything without touching the ledger; use it to test changes.
- Live picks start at week 5 (`LIVE_START_WEEK`). Weeks 1-4 are a
  simulation and never enter the ledger. The board shows one combined
  season record (`record.all`, user's choice 2026-10-09) with a one-line note
  that weeks 1-4 were replayed before launch; `record.live`/`record.pre` stay
  in the data.
- Probabilities: `expected = market + k * (model - market)`, key-number
  distribution re-centered so the line itself is 50/50. Picks of the Week rank
  by Kelly fraction (EV / payout), not raw EV%, so long shots do not dominate.
- My Picks are each viewer's own docs under `data/users/<id>/mypicks/picks`;
  they are private to that viewer, the owner included. Props picks live under
  `data/users/<id>/mypicks/props/<game>__<player>__<market>`.
- Three sections: Game lines, Player props (props/section.py, its own ledger,
  lock times in 5) and Live (live/board.py: the live ledger plus the replay
  reports in reports/live_replay*.json).
- Interface: dotted terms and chart marks explain themselves (`GLOSSARY`,
  `data-tip`), detail lives in dropdowns (`details.acc`), props games are
  one dropdown each, and motion is off for reduced-motion viewers. The design
  system (2026-10-20 revamp: Best bets as the landing view, the slip, team
  logos and headshots, team-colored charts) is in `board/DESIGN.md`; read it
  before changing the page. Best bets and the slip only re-show what each
  section computed: no new pricing lives in the page.
- Stakes on the page:
  - Every pick chip and call card carries a stake pill: 1u, or 0u with a
    dashed "Lean" chip.
  - The record page has a separate "Quarter Kelly" section: tiles, a chart
    against the same picks at 1u, and a week-by-week table. The props record
    has the same section.
  - The week view puts one Kelly line under the flat strip.
  - Live record cards show each kind's Kelly record.
- Before every publish, after the LAST template edit: rebuild, load the exact
  built file in a browser, and confirm no console errors on every tab. A stray
  apostrophe inside a single-quoted JS string once blanked the whole page.

## 9. NBA (nba/, user 2026-10-20)

Game lines and player props for every game, every day, on the board's NBA tab
(sport switch NFL / NBA). Halftime live (2-4 games a night, sharing PropLine's
budget with the NFL) comes after the pregame side is stable.

- Data: hoopR (sportsdataverse/hoopR-nba-data: schedules, box scores, consensus
  closing lines 2020-21 on; seasons named by end year, 2027 = 2026-27), ESPN's
  public API (rosters, injury report, box scores the night of, scoreboard), and
  OddsPapi tournament 132 (`nba/odds.py`; catalogs in data/oddspapi/*_t132,
  *_s11; the free history endpoint keeps only the current season, so games are
  archived while they can be). Odds stay off the repo (data/nba/ is gitignored
  and cached on Actions).
- Game lines (`nba/lines.py`): k = 0. Over six seasons of walk-forward tests the
  model never beat the close (margin MAE 11.44 vs 10.51, beta -0.03; first three
  weeks beta +0.07, not significant). The fair line is Pinnacle's no-vig (else
  DraftKings and FanDuel's); a spread pick and a total pick per game, the side
  and book with the best EV; when nothing is +EV the pick is the model's side as
  a lean (0u, counted at 1u). Moneyline only at +EV. Each pick records
  `with_model` so the checkup can read the early-season lean.
- The game model's ratings age by league game days, ridge 10 (2026-10-29, `nba/model.py`). On the calendar the
  summer left every team rated near average: through 2026-10-28 it called every game about home by 2.3 (slope
  0.01 against the market), so the lean was the underdog on 82% of spreads and the under on 68% of totals
  (64 of 64 spread leans were the model's side). Walk-forward with lineups, holdout 2023-24 to 2025-26: margin
  error 11.73 -> 11.26 (close 10.65), first three weeks 11.28 -> 10.48 (9.98), totals 15.03 -> 14.80. It still
  adds nothing to the close (the model's side covered 49-50% of ~7,000 games), and its margins run ~60% of the
  market's, so leans stay mostly underdogs (~85%). Leaning on its disagreement against what it usually says at
  that line balanced dogs and favorites and covered no better (48.9-49.5%): not used. Whether to keep NBA
  game-line leans at all is the user's call.
- Moneylines' margin comes off by Shin's method (half off each side; `model/probs.ml_no_vig`) from 2026-10-29,
  in the NBA fair chance (picks, EV at the close, ladder ML legs, the card's win bar), the NFL's EV at the close
  and the NFL card's win bar (user, 2026-10-28: both sports). NBA moneyline results graded before then keep their
  proportional `ev_close` (append-only); the NFL's is recomputed every build, so its moneyline rows read lower from
  then on (about m x (0.5 - p) off the closing chance, m Pinnacle's margin: ~1 point of EV at +200, ~2.5 at +300).
  Estimated at a 2.7% margin: NFL moneyline dogs at the close +4.1% -> ~+1.7% (weeks 5-8), +5.0% -> ~+2.3%
  (replay); Picks of the Week +1.8% -> ~+0.7%. The bet list's NFL moneyline-dog cells will weaken accordingly. The proportional split overrated long shots: NFL closes 2006-2026 (nflverse,
  5,386 games) said +250-399 dogs win 24.1% (won 23.2%; Shin 23.4%) and +400+ 15.0% (won 14.0%; Shin 14.0%);
  DraftKings' NBA closes 2023-26 said +200-300 28.4% (won 26.7%) and +500+ 12.5% (won 9.1%). Through 2026-10-28,
  9 of 10 NBA moneyline picks were underdogs at +0.0% to +6.8%, inside that error; both of 2026-10-28's
  (WSH +320, LAC +164) go negative. Close games rarely get a moneyline in either sport: DK and FD hang Pinnacle's
  number plus their own margin on both sides (the best close-game price on 2026-10-28 was 0.3-1.2% short).
  The NFL moneyline is unchanged: its win curve on the spread (`model/probs.WinCurve`) is calibrated (2018-2026
  walk-forward log loss 0.6105 vs the moneyline close's 0.6112), and where it rated a +250 dog 2+ points above
  the market's moneyline those dogs won 26.0% vs the market's 21.3%; its +EV dogs at the consensus close returned
  +9.4% (+-7.6) on 425 bets.
- Props (`nba/props.py`): each team's five starters and sixth man; points,
  rebounds, assists, threes, PRA, P+R, P+A, R+A; one pick per player at 1%+ EV
  (`PROP_EV_MIN`). The market sets the level by scaling the player's rate (a
  count is not shifted: shifting made 1.5-threes lines look like +6% edges), the
  model's negative binomial over minutes sets the shape (calibrated on 2025-26
  starters). Ages count league game days, not calendar days (on the calendar the
  summer made last season weigh ~0.3 minutes, so every player was priced at his
  position's average on opening night 2026-27); rates shrink toward the player's
  own two-season record; after an absence, the summer included, he comes in
  short (`BACK_MIN`, fading over four games; validated out of sample on 2023-24
  and 2024-25). Counting ramp-up games at full minutes made things worse: kept
  out. When DraftKings and FanDuel both sit on the same side of
  Pinnacle's line, Pinnacle is taken as behind the news (its props update
  ~100x less often) and the two of them set the level.
- Schedule (`nba/slot.py`, `.github/workflows/nba.yml`, every 15 minutes in
  each window): 10:30am ET grades last night and locks the day's game lines,
  their Picks of the Day and parlays, and props for games before 6:30pm; 5:30pm
  ET, after the 5pm injury report, locks the rest of the props and the props
  Picks of the Day and parlays. Run by hand with
  `gh workflow run nba.yml -R khadile/key-number -f force=true`; a forced
  build before 10:30am locks nothing and does not use up the morning slot.
  Only GitHub writes the NBA ledgers: on the laptop run `python -m nba.build
  --offline --dry-run` (add `--now 2026-10-20T10:35:00-04:00` to replay a slot),
  never a real build, or picks would lock twice.
- Ledgers: ledger/nba_picks.jsonl, nba_results.jsonl, nba_props_picks.jsonl,
  nba_props_results.jsonl; append-only like the NFL's, keyed by `day` (the
  game's Eastern date). Results carry `ev_close` (the locked price against
  Pinnacle's closing no-vig chance): the first evidence the edge is real.
- Angles the user asked about (2026-10-20), `nba/angles.py` (ESPN's odds API keeps the book of record's
  open and close for every game; with prices from 2023-24: Betfair, then ESPN BET, then DraftKings in
  2025-26; reports/nba_angles.json): at the close,
  2023-24 to 2025-26, underdog moneylines lost 10% (+500 and longer 36%), favorites 3.8%, favorites'
  moneyline parlays 7-17%, spreads and totals about the margin. Taking the opening price on sides the line
  later moved toward paid +13%: the edge is the price, not the bet type. The model predicts those moves
  no better than a coin (46% toward it).
- The Ladder (`nba/ladders.py`): 2-3 legs from different games, each 60%+ likely and worth 1%+ at its price,
  main or alternate spreads, totals or moneylines at DraftKings/FanDuel, priced +100 to +150; locked with the
  morning game lines, none on most days. The likely side of an alternate line carries the least margin (the
  vig sits on the long-shot side), so a ladder trailing Pinnacle can have value there. Legs are kind "alt"
  in nba_picks.jsonl, graded but kept out of the picks' record.
- Long shots (`nba/longshots.py`): the "N+" rungs (archived in data/nba/odds/milestones) +200 and longer, priced
  at each props lock against the model's tail at two widths (the tail test on 2025-26: too wide for stars,
  too narrow for role players, and pricing at the market's level is narrower still). Up to 8 a night with
  10%+ value at both widths, 0.25u each in ledger/nba_longshots.jsonl, their own record. Every rung is also
  graded against its price (ledger/nba_longshot_cal.jsonl, counts only): the model-free test of whether books
  misprice long shots.
- Context notes (user, 2026-10-20; both sports, `nba/notes.py`, `board/notes.py`): each pick carries facts
  computed at build time, never written by hand: situation records against the close (NBA since 2020-21,
  NFL since 2018, n >= 60 / 50, "within chance" when inside two standard errors), the opponent's defense
  against the player's role or position (rank of 30/32), and his recent games against the pick's line.
  History, not the model: none of it is used in pricing. NFL game cards got the NBA-style "Books, model
  and injury report" dropdown.
  - Facts follow the market and the role (user, 2026-10-20: "centers for rebounds, PG's for assists", "RB
    receptions -> the defense to RB receiving, QB rushing -> QB rushing defense"). NFL: `RoleDefense`, what
    each defense allowed per game this season to opposing QB1/RB1/RB2/WR1/WR2/TE1 (the props board's own
    roles) in the pick's market; NBA: per 36 minutes to the position group (hoopR labels only G/F/C) in the
    pick's market, or for a card without a pick the player's own stat (`PropNotes.signature`: a center's
    rebounds, a guard's assists or threes, by his rate against the group) and points.
  - Correlations, each with its sample: NBA, his numbers without a 24+ minute teammate ruled out tonight
    (5+ games each side, during that teammate's time on the team), back-to-back minutes, the game's pace;
    NFL, the team's implied points (top or bottom quarter of the week), his target share or carries over
    his last 3 games when 20%+ off, and `Situations`: a role's production by spread and in 15+ mph wind
    since 2018, against the role's typical game (a role's raw ratio to season averages drifts: RB2s ran 13%
    under in every kind of game), shown only at 5%+ on 200+ games and two standard errors. What passes:
    RB1s as 3-7 point underdogs -6% rushing, and wind -8% to -12% on passing and receiving.
- Checkups every other Monday from 2026-11-02 (Claude scheduled task
  `nba-checkup`): the record and EV at the close, overs vs unders, the model's
  lean, morning vs evening locks replayed on the same games, and the edges still
  to test (back-to-backs and rest, a teammate's props when a starter sits,
  referee crews on totals). The decision to pay for more odds data is revisited
  at the first one.
- Over shading (user, 2026-10-25: props ROI bad, threes worst): through five nights NBA props went 48-56;
  threes 2-14 (-12.3u; 11 of 16 were overs, 12 plus money), plus-money sides of every stat 7-27 (said 47%, won
  21%, -19.1u) against minus-money 41-29 (+6.2u). Pinnacle's NBA threes rarely moved, so "EV at the close"
  equalled the EV at the lock and could not flag it. History (2022-24, five books): the even vig split
  overstates the over, most in threes (plus-money overs at 1.5 / 2.5 / 3.5: said 41.7 / 43.2 / 43.8%, won 39.7 /
  40.1 / 39.6%). Fix: `props.OVER_SHADE`, the logit shift fitted on 2022-23 per stat (about -2 to -3 points near
  even; assists ~0), applied to the anchor's no-vig over chance (`unshade`), in picks, cards and EV at the close.
  On the live picks it would have dropped 36 (97% overs, 14-22, -8.8u). Plus-money unders it keeps still went
  5-14: not explained by the shading and not seen in history (US-book shopping made +2.1% on plus money in
  2023-24); watched, not acted on.
- Situational sides model (user, 2026-10-27: "depending on the pick, player, team, situation, would an over or
  under be good here", after the board went 163 unders to 1 over once the blanket over correction was on;
  `nba/sides.py`, research only). A per-stat logistic on the market's price: the props model's disagreement,
  form against the line, minutes trend, starter, home, rest, spread, total, opponent defense by position,
  missing rotation teammates, price shape. At the US books' open in 2022-24 it beats the market both ways
  (10%+ EV +15.2% and +4.5%). This season it does not: worse than Pinnacle near tip-off (log loss 0.6863 ->
  0.6884, 7 nights) and its morning DK/FD picks were worth -5.3% (+-0.5) at Pinnacle's close, a random bet's
  value. The situation is in the price by now. Also measured (8 nights, every tracked prop at the last pre-tip
  build, graded): overs hit 1.3-2.0 points under their even-split price at all three books, but in assists
  (-7 to -10) and threes, not points or rebounds (~+0.7), the reverse of `OVER_SHADE`'s per-stat split; the
  correction fitted on US books 2022-24 had no Pinnacle data. With it, 672 unders vs 7 overs qualified at 1%+
  over 8 nights; on Pinnacle's raw price, 70 vs 61.
- Learned over correction and edge source (user, 2026-10-27; `nba/shade.py`). Every tracked prop's Pinnacle
  price at each props lock is recorded (data/nba/shade/, Actions cache; rebuilt from the committed board at
  each past lock when missing, the workflow fetching full git history only then) with the best DK/FD price
  on its line and the situational model's chance. Each build grades them and re-estimates the over
  correction per market (`props.SHADE`, which `unshade` uses): prior `OVER_SHADE`, sd 0.10 logit, 14-day
  half-life. First estimate (1,669 graded): points -0.5 pts at even money (was -2.2), rebounds -1.5 (-3.0),
  assists -3.5 (-0.6; overs hit 39% against 48.5% said), threes -3.1 (-3.2). Each prop pick carries
  `ev_raw` (the same pick on Pinnacle's raw price) and `edge`: "price" (beats Pinnacle's raw price by 1%+)
  or "correction" (+EV only through the correction); a correction-only pick is its own bet-list category
  ("reb under (correction)"). Results carry `ev_close_raw` (Pinnacle's raw close, the judge that does not
  assume the correction); the props Record page shows it by set, the learned correction, and the situational
  model's shadow record (5%+ at the best DK/FD price; never bet).
- The learned correction starts from zero (2026-10-29, `nba/shade.py PRIOR`; user asked why 2026-10-28's board had
  96 unders and 2 overs). The OVER_SHADE start (US books 2022-23, no Pinnacle) was not borne out: tracked props'
  last pre-tip prices 2026-10-23 to 27 (39 games, from the committed payloads, graded on hoopR) had overs hitting
  at or above their price at Pinnacle, DraftKings and FanDuel in points (+2.9), rebounds (0), threes (+1.4 / +3.0
  / +3.1), PRA (0), P+R (+2.1 / +3.5 at DK/FD) and P+A; only assists (-4.3 / -1.9 / -1.7) and R+A (-1.2 / -1.8)
  leaned under, each +-2.5-3.6. The correction's assists and threes came from the opening nights (2026-10-20 to
  22, 16 games: assists overs ~31%, threes ~38%). Learned from those nights, the old start priced 2026-10-23 to 27
  worse than no correction (log loss 0.69155 vs 0.68983, 1,263 props); from zero 0.68984, keeping the assists
  gain. Combos without a Pinnacle price (P+R, P+A, R+A) never learn, so they had carried the 2022-23 shifts
  outright. On 2026-10-28's 98 picks the zero start would have kept 56 (assists unders 39) and dropped 42 (threes
  12, rebounds 12, R+A 14, P+R 2, two small points unders). Overs stay rare anyway: of the 22 picks that beat
  Pinnacle's raw price that night, 20 were unders, since DraftKings and FanDuel charge more for overs.
- Re-test of the first week (user, 2026-10-27, walk-forward replay of every lock from the committed board):
  balanced price-gap picks with no correction 63-75 (-11.8%; overs 26-36, threes overs 2-12), the fixed
  correction 195-181 (-0.6%), the learned one 187-172 (-1.1%); actual 115-114 (-4.0%, overs 15-24). The
  filtered sets carry it (user: judge the filter, not the board): actual Picks of the Day 20-9 (+28.7%),
  +4.4% at the close and +2.7% (+-1.5) at Pinnacle's RAW close, against -0.7% (+-0.6) for all props. Judge
  changes by Picks of the Day and the bet list, at the raw close first.
- Picks of the Day take price-gap picks first (user, 2026-10-27; `build.potd`): still one per game by Kelly,
  but a correction-only pick takes a slot only for a game with no price pick. Past props POTD by edge: price
  10-5, unknown 9-3, correction 1-1. Tonight's replay: the learned correction alone made Tatum and Holiday
  (correction-only) Picks of the Day; price-first gives back Mitchell Robinson and Doncic. Game lines carry
  no `edge` and are unaffected.
- Starters (user, 2026-10-24): the tracked five are who started the team's last game this season, minus anyone
  Out or Doubtful, then ESPN's depth chart (`espn.depth_starters`), then the model's starts share. The model's
  own pick had been right on 231 of 280 in the first four nights; last-game starters 96%, the depth chart 93%.
  Players with no minutes in the last two seasons are projected from the season before (a season lost to
  injury: Haliburton), so history loads three seasons.
- Props back-test (2026-10-22, user asked; nba/espn_props.py pulls ESPN's public prop lines, nba/prop_backtest.py
  replays the model walk-forward; data in data/nba/, off the repo):
  - Data: ESPN keeps each game's props (open and latest line and price) from 2022-23. Usable: 2022-23 and
    2023-24, 5-6 books a game, ~500k main-line props, opening prices calibrated (results on price slope ~1.0).
    ESPN BET 2024-25 and 2025-26 prices are scrambled (slope ~0; the line numbers are fine: line-only tests can
    use them). Latest prices are stamped after tip: no EV-at-the-close test from ESPN. No Pinnacle in it.
  - Blind replay (`--blind`: no actual scratches or starters, only what was known before the line opened): the
    model alone predicts worse than the open (log loss 0.706 vs 0.686) but a blend 15% toward it beats the open
    in both seasons. Its picks at the same line as every other book (no shopping): 2-5% EV +5.1% (SE 1.4), 5-10%
    +5.6% (SE 2.3) in 2023-24, +2.7% in 2022-23 (k fitted on 2022-23). Almost all unders, and only in REBOUNDS
    (+10.6%, SE 2.1) and ASSISTS (+8.1%, SE 2.5); points and threes ~0. Picks on a line off the other books'
    returned far more (stale openers: real but limited). At the OPEN; the board locks at 5:30pm, later and
    sharper, which this data cannot test.
  - The market: overs lose 7-12% at the opening price, unders 2-6% (rebounds and threes unders nearest even).
    Factors beyond the open, both seasons agreeing (points of over hit rate): bench -1.0, second night of a
    back-to-back -0.8, away -0.5, 10+ point spread -0.4, toughest-fifth matchup -1.6 (assists -2.6, threes -2.3),
    cold (0-2 of last 10 over this line) -1.6, hot (8-10) +0.9 (form persists, unlike NFL), minutes up 4+
    lately -1.4 and down 4+ +1.2 (the open overreacts to minutes swings). Stacked (fit 2022-23): unders the
    factors lean -4 points or more returned +1.3% at the best of 2-3 books in 2023-24 (+2.5% in 2022-23).
  - ESPN BET 2024-25 and 2025-26 reconstructed (`espnbet_reconstruct()`, the over found from its one-sided
    alternate-line ladder): prices only weakly honest (slope ~0.35), but the main LINE is a real pregame close.
    Line-only test, the model's side against the CLOSE at -110 (52.4% to break even): 2023-24 +3% / +8% / +12%
    at 55-60 / 60-65 / 65%+ model confidence, 2024-25 and 2025-26 about break-even (-0.4% to -5%). The one
    edge that held in all three seasons: REBOUND UNDERS at 55%+ (+7.6%, +5.7%, +7.2%; 11,150 bets). Assists
    faded (+5.3, -0.9, -1.9). The close is near the board's 5:30pm lock, so this is the timing that counts.

## 10. Short-lived edges, the bet list, public money (user, 2026-10-26)

**The user's view, which shapes everything below.** Edges that last two-plus years in a market this big are
calibration failures someone richer already took. Real edges are found early and used before they close: an
adapting system, not one that assumes multi-year history predicts outcomes. Books profit from recreational
psychology (boosts, deposit matches, lines shaded toward the popular side), so look for edges in that. The user
does not bet the whole board (limited resources); they pick a subset, and the system's job is to become very good
at that filtering. Brainstorm ideas are seeds: generalize across sports, markets and stats.

**How we reconcile that with not fooling ourselves (agreed in conversation):**
- Judge new ideas by closing-line value (EV at Pinnacle's close), which settles every bet at kickoff, not by wins
  and losses, which need months. A 5-week, 60-bet sample has a win-rate error of about 6.5 points.
- Don't wait for proof at zero stake: probe stakes early, scale with closing-line evidence, cut fast. Be
  conservative on size, aggressive on timing. An edge that lasts 3 weeks can't be proven in 3 weeks; bet the
  reasoning small and let the close decide.
- Never fade the model's own picks because a category ran cold: fading pays the vig twice (walk-forward: -15 to
  -18% ROI, section 6). A category down units while beating the close is usually bad luck (NBA threes 7-20 at
  +2.1% EV at the close, 21 of 27 still +EV).

**Fade the public (tested 2026-10-26, from two Slim Sports reels; Action Network archive, 2023 wk5 to 2026 wk7,
~950 NFL games, graded at the consensus close):** the reels' 2026 numbers reproduced exactly (spreads 55%+ 33-23-3,
70%+ 16-5-2, totals 80%+ 10-3, ML 80%+ 15-34 +1.7u), but across seasons: spreads 70%+ 138-111-4 (+14.3u, z 1.7)
lost in 2023 and 2024 and won in 2025-26; totals coin-flip until 80%+ (+9.4u, mostly 2026); moneyline fades lose
at every threshold (80%+: 132-342, -28.5u). "Follow the money" (money % 15+ ahead of tickets) on totals +14.9u,
z 1.7. Verdict: not a standing edge; the public lost in 2025-26 specifically. Scripts: session scratchpad `fade/`.

**The bet list (model/betlist.py; on Best bets, record row "Bet list", "How the list is choosing" under each
record's By category).** The picks worth a stake, chosen at the lock and never redone (a `betlist` marker in the
picks ledger: `{day}:betlist` / `{day}:props:betlist` for NBA, `{season}_{week}:betlist` for NFL lines,
`{season}_{week}:props:betlist:{Thu|Sat|Mon}` per NFL props lock).
- Cells: category (NBA props: stat and side, combos pooled; lines: spread/ML favorite or underdog, total over or
  under; NFL props: market and side) x EV tier at the lock (1-2%, 2-4%, 4%+).
- Each cell's true edge is a precision-weighted blend of: a prior from its tier pooled across categories (the tier
  from N(0, 3%)), recency-weighted EV at the close (allowance CLOSE_BIAS 3%: the close can misjudge a category,
  e.g. NFL receiving yards +5.3% at the close but -11% ROI on 165), and recency-weighted units at 1u (variance
  ~1 per pick, so units count at volume and can overrule the close in a category with a couple hundred picks).
  Half-life 10 days (NBA), 3 weeks (NFL).
- On the list: estimate +1%+ and 65%+ sure it is above zero; a cell with fewer than 3 graded picks goes by its
  tier; at most 2 picks per cell per lock; caps NBA lines 4/day, NBA props 6/day, NFL lines 6/week, NFL props 4
  per lock. Ranked by cell estimate, then EV.
- User asked for maximum flexibility (e.g. "if 2%+ props are up units a few days running, add them, or only 4%+");
  the tiers and recency do that, at the speed the evidence allows. A few days of units alone moves a cell little;
  units plus closing value moves it fast. If the user wants units to count more, lower CLOSE_BIAS's counterweight
  by raising CLOSE_BIAS (more distrust of the close), not by special-casing.
- First lists (dry run, 2026-10-26): NBA props all 4%+ unders (threes, points, combos), 1-2% tier out (-0.9% at
  the close vs 2-4% +1.7%, 4%+ +5.3%); NBA game lines none (losing to the close); NFL lines ML underdogs 4%+ and
  spread underdogs 2-4%; NFL props 4%+ cells.
- What the user stakes (2026-10-28): game lines' Picks of the Day / Week and the bet list; leans stay for research.
  NBA lines' bet list was empty 2026-10-26 to 28, and its Picks of the Day are small, though not few: 19 in 9 days
  (NFL Picks of the Week: 18 in 4 weeks), median EV at the lock +0.7% (NFL +4.3%), 6 of 19 at 1%+ (NFL 14 of 18),
  at the close -0.2% +-1.3 (NFL +1.8% +-1.8 before the moneyline fix). Of the 22 NBA +EV line picks, 6 reached the
  bet list's 1% floor, 4 of them long-dog moneylines that the vig fix removes, and the graded ones lost at the close
  (Dallas +500 -11.7%), so every tier read negative. The difference is structural: NBA lines are line shopping only
  (k = 0) on same-day lines at 10:30am, when DraftKings and FanDuel have caught up to Pinnacle; the NFL's model moves
  its prices (k measured each week), its moneylines come from a win curve that has found real underdog value, and it
  locks Tuesday, days before kickoff. Where an NBA line edge could exist is timing (locking near the open, untested:
  the archive is on the Actions cache), not a lower bar.
- **Calibrate at every checkup:** bet list vs all picks, by units AND EV at the close. If the list does not beat
  the full board at the close over 4+ weeks, revisit the constants (PRIOR_SD, CELL_SD, CLOSE_BIAS, half-lives,
  MIN_EDGE/MIN_PROB) on the replay, not on a few days. Never re-pick past lists.

**By category tables (section 6) and the walk-forward test** are what the bet list stands on: hot-category carry
is real but small in NBA props (3-day streak coef 0.13), none in NFL lines; the day's top-5 EV beats the rest.

**Public money (pipeline/action_network.py).** Action Network's public API: ticket % and money % per side for the
main spread, total and moneyline (NFL by season/week, NBA by date; `bookIds=15` consensus; history from 2023 wk5).
Every NFL and NBA build pulls it (online only), saves the raw JSON to `data/actionnetwork/nfl` and
`data/nba/actionnetwork` (Actions caches, gitignored), shows "Public" on each game card (most-bet side, tickets %,
money %, bold at 70%+), and stamps each game-line pick at lock with `public: {tickets, money}` on its side.
Context only, never a pricing input.
- Public-tax test (2026 wk1-7, 74 games, best DK/FD price vs Pinnacle no-vig on the same number; scratchpad
  `public_tax.py`): spreads and totals none (DK/FD shade both sides ~-4%, however lopsided); moneylines a hint:
  the 70%+ side -3.9% vs the other side -1.9%, 28% of unpopular-side prices +EV against Pinnacle. Confounded with
  favorites (the public ML side is nearly always the favorite). Re-run at checkups with the build snapshots
  (lock vs close) and control for favorite/underdog before it becomes a tag or a probe.

**Psychology angles queued to test the same way (closing-line value, probe stakes):** star and prime-time overs;
overreaction to last week's blowout or viral game; non-QB injury line moves and backups' props lagging news;
reverse line movement (tickets one way, line the other) and its timing; boosts that are +EV by arithmetic and
same-game-parlay correlation pricing. Also pending: NBA props at 1-2% EV to leans (Nov 2 checkup; at the close
-0.9% +-1.2 vs 2%+ +2.8% +-1.0); a per-category stake (full / 0.25u probe / 0) once the bet list has a few weeks.

## Layout

```
pipeline/   ingest.py (download), clean.py (plays), lines.py (snapshots),
            oddspapi.py (budgeted client), odds_history.py (line + prop timelines),
            propline.py (budgeted live odds client), prop_players.py (id map),
            sgo.py + sgo_tidy.py (SportsGameOdds archive, 2024-2026)
props/      data.py (game logs), model.py, backtest.py, lines.py, study.py, sgo_study.py,
            pricing.py, ledger.py, section.py (the board's props tab)
live/       espn.py, feeds.py, model.py, decide.py, engine.py (+ page.html),
            ledger.py, notify.py (ntfy), replay.py, replay_games.py, board.py
model/      ratings.py, qb.py, features.py, project.py, probs.py, blend.py
board/      build.py, market.py (pricing), ledger.py, defense.py, detail.py,
            weather.py, render.py (both payloads + template.html -> nfl_board.html)
nba/        data.py (hoopR), espn.py, odds.py (OddsPapi), model.py + players.py
            (ratings, APM), backtest.py, lines.py, props.py, ledger.py, build.py, slot.py
backtest/   run.py, clv.py, parlays.py, trends.py, speeds.py -> reports/
ledger/     picks.jsonl (append-only)
board/      JSON -> HTML
data/       raw/ clean/ lines/ snapshots/   (gitignored)
```

Run everything with `.venv/Scripts/python` from the project root.
