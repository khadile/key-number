"""Archive price timelines from OddsPapi: game lines and player props.

/v4/historical-odds is free (no quota) but only keeps the current season, so
every game must be archived while it is still available. One call returns a
book group's whole timeline for a fixture, pregame and in-game. It is split
into the full-game ML / spread / total markets (data/oddspapi/history/
{fixtureId}.parquet) and, for regular-season games, every full-game player
prop (data/oddspapi/props/{fixtureId}.parquet), in-game prices included.
Finished games are fetched once; upcoming games are re-fetched every run
(their history is still growing).

Only the "core" group (Pinnacle, DraftKings, FanDuel) is archived now: the
user bets at DraftKings and FanDuel, and Pinnacle is the sharp reference. The
other groups' files from before 2026-10-08 stay on disk.

    python -m pipeline.odds_history            # archive everything available
    python -m pipeline.odds_history --summary  # open/close lines per game
    python -m pipeline.odds_history --games 2026_05_TB_DAL   # refresh one game
"""
import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import polars as pl

from config import DATA
from pipeline.oddspapi import CACHE, _get

HIST = CACHE / "history"
PROPS = CACHE / "props"
HIST.mkdir(parents=True, exist_ok=True)
PROPS.mkdir(parents=True, exist_ok=True)
# Max 3 books per call, so books come in groups. "core" holds the sharp
# reference (Pinnacle) and the two biggest US books; the others widen line
# shopping across US books. Every group is a free call.
BOOK_GROUPS = {
    "core": "pinnacle,draftkings,fanduel",
    "us2": "betmgm,caesars,espnbet",
    "us3": "fanatics,betrivers,hardrockbet",
}
US_BOOKS = ["draftkings", "fanduel", "betmgm", "caesars", "espnbet", "fanatics", "betrivers", "hardrockbet"]
SHOP_BOOKS = ["draftkings", "fanduel"]  # where the user bets: every pick is priced at one of these
WANTED = {"moneyline", "spreads", "totals"}

# OddsPapi team names -> nflverse abbreviations
TEAMS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Los Angeles Rams": "LA", "Los Angeles Chargers": "LAC", "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "Seattle Seahawks": "SEA",
    "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def _market_map() -> dict[int, tuple[str, float, str]]:
    """outcomeId -> (marketType, handicap, outcomeName) for full-game ML/spread/total."""
    mk = json.loads((CACHE / "markets_s14.json").read_text(encoding="utf-8"))
    out = {}
    for m in mk:
        if m["marketType"] in WANTED and m["period"] == "result":
            for o in m["outcomes"]:
                out[o["outcomeId"]] = (m["marketType"], float(m["handicap"]), o["outcomeName"])
    return out


def _prop_map() -> dict[int, tuple[str, float, str]]:
    """outcomeId -> (marketType, handicap, outcomeName) for every full-game player prop."""
    mk = json.loads((CACHE / "markets_s14.json").read_text(encoding="utf-8"))
    out = {}
    for m in mk:
        if m["playerProp"] and m["period"] == "result":
            for o in m["outcomes"]:
                out[o["outcomeId"]] = (m["marketType"], float(m["handicap"]), o["outcomeName"])
    return out


def _fetch(fixture_id: str, books: str, tries: int = 4) -> dict | None:
    for i in range(tries):  # the endpoint intermittently 404s on data it has
        h = _get("historical-odds", {"fixtureId": fixture_id, "bookmakers": books}, billable=False)
        if "error" not in h:
            return h
        time.sleep(2 + 2 * i)
    return None


def history_path(fixture_id: str, group: str):
    return HIST / (f"{fixture_id}.parquet" if group == "core" else f"{fixture_id}__{group}.parquet")


def prop_path(fixture_id: str):
    return PROPS / f"{fixture_id}.parquet"


def _rows(fixture_id: str, h: dict, mmap: dict) -> list[dict]:
    rows = []
    for book, b in h.get("bookmakers", {}).items():
        for m in b.get("markets", {}).values():
            for oid, o in m.get("outcomes", {}).items():
                info = mmap.get(int(oid))
                if not info:
                    continue
                for pts in o.get("players", {}).values():
                    for p in pts:
                        rows.append({"fixture_id": fixture_id, "book": book, "market": info[0],
                                     "handicap": info[1], "outcome": info[2], "ts": p["createdAt"],
                                     "price": p["price"], "active": p["active"]})
    return rows


PROP_SCHEMA = {"fixture_id": pl.String, "book": pl.String, "market": pl.String, "handicap": pl.Float64,
               "outcome": pl.String, "player": pl.Int64, "ts": pl.String, "price": pl.Float64,
               "active": pl.Boolean, "limit": pl.Float64}


def _prop_rows(fixture_id: str, h: dict, pmap: dict) -> list[tuple]:
    """Player-prop price points. Each outcome holds one series per player (OddsPapi player id)."""
    rows = []
    for book, b in h.get("bookmakers", {}).items():
        for m in b.get("markets", {}).values():
            for oid, o in m.get("outcomes", {}).items():
                info = pmap.get(int(oid))
                if not info:
                    continue
                for pid, pts in o.get("players", {}).items():
                    if pid == "0":
                        continue
                    for p in pts:
                        lim = p.get("limit")
                        rows.append((fixture_id, book, info[0], info[1], info[2], int(pid), p["createdAt"],
                                     float(p["price"]), bool(p["active"]), None if lim is None else float(lim)))
    return rows


def _write(path, df: pl.DataFrame) -> None:
    tmp = path.with_suffix(".tmp")  # write then rename, so an interrupted run never leaves a half file
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(path)


def archive(groups: tuple[str, ...] = ("core",), horizon_days: int = 14,
            season_start: str = "2026-08-01", workers: int = 1, only: list[str] | None = None) -> None:
    """Save every available timeline. An upcoming game is re-fetched every run (its
    history is still growing); once it has ended (6 hours after kickoff) it gets one
    last fetch, if its file was written before then, so the closing lines and in-game
    prices are kept. Builds run in the morning, so a game's last fetch before that rule
    was always the build before kickoff.
    `season_start`: history is only kept for the current season.

    One call at a time: /historical-odds allows about one call every 4
    seconds (the client paces and honours 429 waits), and a busy game's
    response is ~100 MB of JSON, props and in-game prices included.
    `only`: nflverse game ids, to refresh just those games (one call each)."""
    fx = json.loads((CACHE / "fixtures_t31.json").read_text(encoding="utf-8"))
    mmap, pmap = _market_map(), _prop_map()
    types = pl.read_parquet(DATA / "clean" / "games.parquet").select("game_id", "game_type")
    reg = set(fixture_index().join(types, on="game_id").filter(pl.col("game_type") == "REG")["fixture_id"])
    now = datetime.now(timezone.utc)
    horizon = (now + timedelta(days=horizon_days)).isoformat()  # far-off games have no lines yet
    todo = [(f, g) for f in fx if season_start <= f["startTime"] <= horizon and f["statusId"] != 3
            for g in groups]
    if only:
        keep = set(fixture_index().filter(pl.col("game_id").is_in(only))["fixture_id"])
        todo = [j for j in todo if j[0]["fixtureId"] in keep]

    def one(job) -> str:
        f, group = job
        fid = f["fixtureId"]
        path = history_path(fid, group)
        ppath = prop_path(fid) if group == "core" and fid in reg else None
        ended = datetime.fromisoformat(f["startTime"].replace("Z", "+00:00")) + timedelta(hours=6)

        def complete(p) -> bool:   # written after the game ended: nothing more to fetch
            return p.exists() and datetime.fromtimestamp(p.stat().st_mtime, timezone.utc) > ended
        need_lines = not complete(path)
        need_props = ppath is not None and not complete(ppath)
        if not need_lines and not need_props:
            return "skipped"
        h = _fetch(fid, BOOK_GROUPS[group])
        if not h:
            return "missing"
        status = "missing"
        if need_lines and (rows := _rows(fid, h, mmap)):
            _write(path, pl.DataFrame(rows).with_columns(pl.col("ts").str.to_datetime(time_zone="UTC")))
            status = "saved"
        if need_props and (prow := _prop_rows(fid, h, pmap)):
            _write(ppath, pl.DataFrame(prow, schema=PROP_SCHEMA, orient="row")
                   .with_columns(pl.col("ts").str.to_datetime(time_zone="UTC")))
            status = "saved"
        return status

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = Counter(pool.map(one, sorted(todo, key=lambda j: j[0]["startTime"])))
    print(f"history: saved {results['saved']}, already archived {results['skipped']}, unavailable {results['missing']}")


def fixture_index() -> pl.DataFrame:
    """OddsPapi fixtures joined to nflverse game_ids (participant1 is the home team)."""
    fx = json.loads((CACHE / "fixtures_t31.json").read_text(encoding="utf-8"))
    f = pl.DataFrame([{"fixture_id": x["fixtureId"], "start": x["startTime"],
                       "home_team": TEAMS.get(x["participant1Name"]), "away_team": TEAMS.get(x["participant2Name"])}
                      for x in fx])
    games = pl.read_parquet(DATA / "clean" / "games.parquet").select(
        "game_id", "season", "week", "home_team", "away_team", "gameday")
    f = f.with_columns(gameday=pl.col("start").str.slice(0, 10))
    # kickoff dates can differ by a day between UTC and Eastern; match on teams within +-1 day
    j = f.join(games, on=["home_team", "away_team"], how="inner").filter(
        (pl.col("gameday").str.to_date() - pl.col("gameday_right").str.to_date()).dt.total_days().abs() <= 1)
    return j.select("fixture_id", "game_id", "season", "week", "home_team", "away_team", "start")


def summary() -> pl.DataFrame:
    """Main lines per game and book at three moments, in nflverse conventions
    (spread = points the home team is favored by; prices are decimal odds):

    - open:    one hour after the book first posted the game
    - publish: Tuesday 10am ET of game week, when our board goes out. This is
               the fair comparison for the model: lookahead lines posted before
               last week's games could not know results our ratings already use.
    - close:   last price 5 minutes before kickoff (books suspend in the final minutes)
    """
    files = sorted(HIST.glob("*.parquet"))
    if not files:
        return pl.DataFrame()
    idx = fixture_index().with_columns(kickoff=pl.col("start").str.to_datetime(time_zone="UTC"))
    h = pl.concat([pl.read_parquet(p) for p in files]).join(
        idx.select("fixture_id", "game_id", "kickoff"), on="fixture_id")
    h = h.filter(pl.col("ts") < pl.col("kickoff"))  # pregame only
    out = []
    for (fid, book), g in h.group_by("fixture_id", "book"):
        kickoff = g["kickoff"][0]
        first = g["ts"].min()
        rec = {"fixture_id": fid, "book": book, "game_id": g["game_id"][0], "kickoff": kickoff,
               "first_seen": first, "publish_at": publish_time(kickoff)}
        for when, cut in (("open", first + timedelta(hours=1)),
                          ("publish", max(publish_time(kickoff), first + timedelta(hours=1))),
                          ("close", kickoff - timedelta(minutes=5))):  # books suspend in the final minutes
            for k, v in lines_at(g, cut).items():
                rec[f"{when}_{k}"] = v
        out.append(rec)
    return pl.DataFrame(out)


def publish_time(kickoff: datetime) -> datetime:
    """Tuesday 10:00 ET (14:00 UTC) on or before kickoff: when the weekly board goes out."""
    t = kickoff.replace(hour=14, minute=0, second=0, microsecond=0)
    t -= timedelta(days=(t.weekday() - 1) % 7)  # back to Tuesday
    return t if t < kickoff else t - timedelta(days=7)


def lines_at(g: pl.DataFrame, cut: datetime) -> dict:
    """The main spread, total and moneyline a book was showing at `cut`.

    Each (market, line, side) keeps its last record at or before `cut`; a line
    whose last record is inactive had been pulled and is ignored. The main
    line is the one whose two sides are priced closest to even.
    """
    # Records can share a timestamp, so the order must be total: at equal times an
    # inactive record sorts last (a line pulled at that moment is pulled). Without
    # this the "last" price could differ from one run to the next.
    last = (g.filter(pl.col("ts") <= cut)
            .sort(["ts", "active", "price"], descending=[False, True, False], maintain_order=True)
            .group_by("market", "handicap", "outcome")
            .agg(pl.col("price").last(), pl.col("active").last())
            .filter(pl.col("active")))
    rec: dict = {}
    for market, side_a, side_b, name in (("spreads", "1", "2", "spread"), ("totals", "Over", "Under", "total")):
        m = last.filter(pl.col("market") == market)
        # Both sides of a market share its handicap (team 1 at h, team 2 at -h; Over/Under at h).
        a = m.filter(pl.col("outcome") == side_a).select("handicap", pa="price")
        b = m.filter(pl.col("outcome") == side_b).select("handicap", pb="price")
        pair = (a.join(b, on="handicap").with_columns(gap=(1 / pl.col("pa") - 1 / pl.col("pb")).abs())
                .sort(["gap", "handicap"]))  # handicap breaks exact ties the same way every run
        if pair.height:
            best = pair.row(0, named=True)
            rec[name] = -best["handicap"] if market == "spreads" else best["handicap"]
            rec[f"{name}_price_a"], rec[f"{name}_price_b"] = best["pa"], best["pb"]  # home/over, away/under
    ml = last.filter(pl.col("market") == "moneyline")
    prices = dict(zip(ml["outcome"], ml["price"]))
    if "1" in prices and "2" in prices:
        rec["ml_home"], rec["ml_away"] = prices["1"], prices["2"]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--games", nargs="*", help="nflverse game ids: refresh only these")
    args = ap.parse_args()
    if not args.summary:
        archive(only=args.games)
    s = summary()
    s.write_parquet(CACHE / "open_close.parquet")
    print(s.select("game_id", "book", "open_spread", "publish_spread", "close_spread",
                   "open_total", "publish_total", "close_total").sort("game_id", "book").head(30))


if __name__ == "__main__":
    main()
