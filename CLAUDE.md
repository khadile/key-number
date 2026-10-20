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

## 6b. Player props (props/)

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
  one dropdown each, and motion is off for reduced-motion viewers.
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
- Props (`nba/props.py`): each team's five starters and sixth man; points,
  rebounds, assists, threes, PRA, P+R, P+A, R+A; one pick per player at 1%+ EV
  (`PROP_EV_MIN`). The market sets the level by scaling the player's rate (a
  count is not shifted: shifting made 1.5-threes lines look like +6% edges), the
  model's negative binomial over minutes sets the shape (calibrated on 2025-26
  starters). When DraftKings and FanDuel both sit on the same side of
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
  open and close for every game: DraftKings, ESPN BET or Caesars; reports/nba_angles.json): at the close,
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
- Checkups every other Monday from 2026-11-02 (Claude scheduled task
  `nba-checkup`): the record and EV at the close, overs vs unders, the model's
  lean, morning vs evening locks replayed on the same games, and the edges still
  to test (back-to-backs and rest, a teammate's props when a starter sits,
  referee crews on totals). The decision to pay for more odds data is revisited
  at the first one.

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
