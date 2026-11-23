"""Pregame prices from PropLine, written into the OddsPapi archive (2026-11-23).

OddsPapi's free plan stopped serving games that have not finished (from about 2026-11-20: HTTP 403 "Historical
odds are only available for finished fixtures on the free plan"). Finished games still come back whole, every
price change and the close, so OddsPapi keeps the history; what it no longer gives is the price NOW, which every
lock needs. PropLine does: one call per game returns Pinnacle's, DraftKings' and FanDuel's main game lines and every
player prop as they stand.

Each build appends that snapshot to the same files the board already reads (data/oddspapi/history|props for the NFL,
data/nba/odds/history|props for the NBA), in OddsPapi's own terms (market names, outcome "1"/"2" for home/away,
OddsPapi player ids), stamped with the time of the call. A line the snapshot no longer shows gets an inactive record
at that time (pulled), so an older line cannot pass for the main one. Once the game has ended, OddsPapi's full
timeline replaces the file (odds_history.archive / nba.odds.archive), so closes and line value are OddsPapi's.

Main lines only, no alternates. Budget: one call a game a build (pipeline/propline.py keeps the daily cap).
"""
from datetime import datetime, timedelta, timezone

import polars as pl

from config import secret
from pipeline import propline
from pipeline.odds_history import PROP_SCHEMA, _write

BOOKS = ("pinnacle", "draftkings", "fanduel")
GAME_MARKETS = ("moneyline", "spreads", "totals")
NFL_PROPS = {"player_pass_yds": "playertotals-passyards", "player_rush_yds": "playertotals-rushyards",
             "player_reception_yds": "playertotals-receivingyards", "player_receptions": "playertotals-receptions"}
NBA_PROPS = {"player_points": "playertotals-points", "player_rebounds": "playertotals-rebounds",
             "player_assists": "playertotals-assists", "player_threes": "playertotals-3pointers",
             "player_points_rebounds_assists": "playertotals-ptsassreb", "player_points_rebounds": "playertotals-ptsreb",
             "player_points_assists": "playertotals-ptsass", "player_rebounds_assists": "playertotals-assreb"}
HIST_SCHEMA = {"fixture_id": pl.String, "book": pl.String, "market": pl.String, "handicap": pl.Float64,
               "outcome": pl.String, "ts": pl.Datetime("us", "UTC"), "price": pl.Float64, "active": pl.Boolean}
PROPS_SCHEMA = {**PROP_SCHEMA, "ts": pl.Datetime("us", "UTC")}


def available() -> bool:
    return bool(secret("PROPLINE_KEY"))


def dec(american: float) -> float:
    return 1 + american / 100 if american > 0 else 1 + 100 / -american


def convert(o: dict, fid: str, home: str, away: str, prop_keys: dict, player_of, ts: datetime) -> tuple[list, list]:
    """One PropLine /odds answer -> (game-line rows, prop rows) in the archive's schemas. `home`, `away`: PropLine's
    full team names. `player_of(outcome)` -> OddsPapi player id or None (the prop is then left out)."""
    games, props = [], []
    nick = {home: home.split()[-1], away: away.split()[-1]}

    def team(name: str | None) -> str | None:   # books write "WAS Commanders" as well as "Washington Commanders"
        for full, n in nick.items():
            if name == full or (name or "").endswith(" " + n):
                return full
        return None

    for b in o.get("bookmakers") or []:
        book = b.get("key")
        if book not in BOOKS:
            continue
        for m in b.get("markets") or []:
            if (m.get("line_type") != "main" or m.get("suspended_at") or m.get("team")
                    or m.get("period") not in (None, "", "full_game", "game")):
                continue
            outs = [x for x in m.get("outcomes") or [] if x.get("price") is not None and
                    (not x.get("last_seen_at") or not m.get("last_update") or x["last_seen_at"] >= m["last_update"])]
            key = m.get("key")
            if key == "h2h":
                side = {team(x["name"]): dec(x["price"]) for x in outs}
                if home in side and away in side:
                    games += [(fid, book, "moneyline", 0.0, "1", ts, side[home], True),
                              (fid, book, "moneyline", 0.0, "2", ts, side[away], True)]
            elif key == "spreads":
                side = {team(x["name"]): x for x in outs if x.get("point") is not None}
                if home in side and away in side and float(side[home]["point"]) == -float(side[away]["point"]):
                    h = float(side[home]["point"])   # OddsPapi: team 1 (home) at h, team 2 at -h
                    games += [(fid, book, "spreads", h, "1", ts, dec(side[home]["price"]), True),
                              (fid, book, "spreads", h, "2", ts, dec(side[away]["price"]), True)]
            elif key == "totals":
                side = {x["name"]: x for x in outs if x.get("point") is not None}
                if "Over" in side and "Under" in side and side["Over"]["point"] == side["Under"]["point"]:
                    t = float(side["Over"]["point"])
                    games += [(fid, book, "totals", t, "Over", ts, dec(side["Over"]["price"]), True),
                              (fid, book, "totals", t, "Under", ts, dec(side["Under"]["price"]), True)]
            elif key in prop_keys:
                pairs: dict = {}
                for x in outs:
                    if x.get("point") is None or x.get("name") not in ("Over", "Under") or not x.get("player_id"):
                        continue
                    pairs.setdefault((x["player_id"], float(x["point"])), {})[x["name"]] = x
                for (_, point), sides in pairs.items():
                    if "Over" not in sides or "Under" not in sides:
                        continue
                    pid = player_of(sides["Over"])
                    if pid is None:
                        continue
                    for name in ("Over", "Under"):
                        props.append((fid, book, prop_keys[key], point, name, int(pid), ts,
                                      dec(sides[name]["price"]), True, None))
    return games, props


def merge(path, rows: list, schema: dict, keys: list[str], markets) -> int:
    """Append a snapshot to an archive file. Every (book, market, line, side[, player]) still active in the file for
    the snapshot's books and markets that the snapshot does not show is marked pulled at the snapshot's time.
    Returns the rows written (0: nothing to add)."""
    if not rows:
        return 0
    new = pl.DataFrame(rows, schema=schema, orient="row")
    ts = new["ts"].max()
    if not path.exists():
        _write(path, new)
        return new.height
    old = pl.read_parquet(path)
    old = old.with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC")))
    last = (old.filter(pl.col("book").is_in(list(BOOKS)) & pl.col("market").is_in(list(markets)) & (pl.col("ts") <= ts))
            .sort(["ts", "active", "price"], descending=[False, True, False], maintain_order=True)
            .group_by(keys).last().filter(pl.col("active")))
    pulled = (last.join(new.select(keys), on=keys, how="anti")
              .with_columns(ts=pl.lit(ts).cast(pl.Datetime("us", "UTC")), active=pl.lit(False)))
    out = pl.concat([old, pulled.select(old.columns), new.select(old.columns)], how="vertical_relaxed")
    _write(path, out)
    return new.height + pulled.height


def _odds(sport: str, event_id: str, markets: str) -> dict | None:
    o = propline.get(f"sports/{sport}/events/{event_id}/odds", {"markets": markets, "bookmakers": propline.BOOKS})
    return None if not isinstance(o, dict) or "error" in o else o


def _match(events: list, idx: pl.DataFrame, abbr) -> dict:
    """fixture_id -> PropLine event, on home and away team within a day of the start."""
    out = {}
    for e in events or []:
        try:
            start = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        h, a = abbr(e.get("home_team")), abbr(e.get("away_team"))
        if h is None or a is None:
            continue
        hit = idx.filter((pl.col("home") == h) & (pl.col("away") == a)
                         & ((pl.col("start") - start).dt.total_hours().abs() <= 24))
        if hit.height:
            out[hit["fixture_id"][0]] = e
    return out


def snapshot_nfl(horizon_days: int = 8) -> None:
    """Every NFL game starting in the next `horizon_days`: game lines into history/, the tracked props into props/."""
    if not available():
        print("propline: no PROPLINE_KEY, pregame prices not refreshed")
        return
    from config import CLEAN, RAW
    from live.feeds import MARKETS
    from pipeline.odds_history import HIST, PROPS, TEAMS, fixture_index
    now = datetime.now(timezone.utc)
    idx = (fixture_index().with_columns(start=pl.col("start").str.to_datetime(time_zone="UTC"))
           .rename({"home_team": "home", "away_team": "away"})
           .filter((pl.col("start") > now) & (pl.col("start") <= now + timedelta(days=horizon_days))))
    if idx.height == 0:
        print("propline: no NFL games ahead")
        return
    gsis = {str(int(float(e))): g for e, g in
            pl.read_parquet(RAW / "players.parquet").select("espn_id", "gsis_id").drop_nulls().iter_rows()}
    op = dict(pl.read_parquet(CLEAN / "prop_players.parquet").select("gsis_id", "op_id").drop_nulls()
              .unique("gsis_id").iter_rows())

    def player_of(x):
        pid = str(x.get("player_id") or "")
        return op.get(gsis.get(pid.split(":", 1)[1])) if pid.startswith("espn:") else None

    try:
        found = _match(propline.events(), idx, TEAMS.get)
    except SystemExit as e:
        print(f"propline: {e}")
        return
    n = {"games": 0, "lines": 0, "props": 0, "missing": idx.height - len(found), "failed": 0}
    for fid, e in found.items():
        try:
            o = _odds(propline.SPORT, e["id"], MARKETS)
        except SystemExit as err:   # the daily budget: stop, the rest keep their last prices
            print(f"propline: {err}")
            break
        if o is None:
            n["failed"] += 1
            continue
        g, p = convert(o, fid, e["home_team"], e["away_team"], NFL_PROPS, player_of, now)
        n["lines"] += merge(HIST / f"{fid}.parquet", g, HIST_SCHEMA, ["book", "market", "handicap", "outcome"], GAME_MARKETS)
        n["props"] += merge(PROPS / f"{fid}.parquet", p, PROPS_SCHEMA,
                            ["book", "market", "handicap", "outcome", "player"], NFL_PROPS.values())
        n["games"] += 1
    print(f"propline NFL: {n['games']} games refreshed ({n['lines']} line rows, {n['props']} prop rows), "
          f"not listed {n['missing']}, failed {n['failed']}")


def snapshot_nba(ahead_hours: int = 30) -> None:
    """Every NBA game starting in the next `ahead_hours`: game lines and the eight tracked prop markets."""
    if not available():
        print("propline: no PROPLINE_KEY, pregame prices not refreshed")
        return
    from nba import odds as NO
    now = datetime.now(timezone.utc)
    idx = NO.fixture_index().filter((pl.col("start") > now) & (pl.col("start") <= now + timedelta(hours=ahead_hours)))
    if idx.height == 0:
        print("propline: no NBA games ahead")
        return
    abbr = NO.team_abbr()
    op = NO.op_ids_by_name()

    def player_of(x):
        key, team = NO.norm(x.get("description")), abbr.get(x.get("player_team"))
        return op["team"].get((key, team)) or op["unique"].get(key)

    try:
        events = propline.get("sports/basketball_nba/events", cache_name="events_nba.json", ttl_s=600)
    except SystemExit as e:
        print(f"propline: {e}")
        return
    found = _match(events if isinstance(events, list) else [], idx, abbr.get)
    n = {"games": 0, "lines": 0, "props": 0, "missing": idx.height - len(found), "failed": 0}
    markets = ",".join(list(NBA_PROPS) + ["h2h", "spreads", "totals"])
    for fid, e in found.items():
        try:
            o = _odds("basketball_nba", e["id"], markets)
        except SystemExit as err:
            print(f"propline: {err}")
            break
        if o is None:
            n["failed"] += 1
            continue
        g, p = convert(o, fid, e["home_team"], e["away_team"], NBA_PROPS, player_of, now)
        n["lines"] += merge(NO.HIST / f"{fid}.parquet", g, HIST_SCHEMA, ["book", "market", "handicap", "outcome"], GAME_MARKETS)
        n["props"] += merge(NO.PROPS / f"{fid}.parquet", p, PROPS_SCHEMA,
                            ["book", "market", "handicap", "outcome", "player"], NBA_PROPS.values())
        n["games"] += 1
    print(f"propline NBA: {n['games']} games refreshed ({n['lines']} line rows, {n['props']} prop rows), "
          f"not listed {n['missing']}, failed {n['failed']}")
