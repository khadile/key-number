"""NBA odds from OddsPapi: tournament 132 (NBA), sport 11 (basketball).

One-time billable lookups (2026-10-20), cached in data/oddspapi/: tournaments_s11.json, fixtures_t132.json (every
NBA fixture from 2024-25 on), markets_s11.json, players_s11.json. /historical-odds is free (about one call per 5
seconds) and keeps only the current season (last season's games answer 404), so every game is archived while it
can be: upcoming games are re-fetched each build, and a finished game gets one last fetch for its closing and
in-game prices. A game's response is about 60 MB of JSON (DraftKings ~430 markets, FanDuel ~690, Pinnacle ~85);
what is kept, in data/nba/odds/ (off the public repo):

- history/{fixtureId}.parquet: full-game moneyline, spreads, totals, every price change, Pinnacle/DK/FD
- props/{fixtureId}.parquet: full-game player over/unders in the eight tracked markets

    python -m nba.odds                    # archive what is due
    python -m nba.odds --refetch-fixtures # one billable call: new fixture ids (the schedule rarely changes)
"""
import argparse
import json
import re
import unicodedata
from datetime import datetime, timedelta, timezone

import polars as pl

from board.market import Lines
from config import DATA
from nba.data import load as hoopr
from pipeline.odds_history import _fetch, _prop_rows, _rows, _write, PROP_SCHEMA
from pipeline.oddspapi import CACHE, _get

TOURNAMENT, SPORT = 132, 11
ODDS = DATA / "nba" / "odds"
HIST, PROPS = ODDS / "history", ODDS / "props"
for _p in (HIST, PROPS):
    _p.mkdir(parents=True, exist_ok=True)
FIXTURES, MARKETS, PLAYERS = CACHE / f"fixtures_t{TOURNAMENT}.json", CACHE / f"markets_s{SPORT}.json", CACHE / f"players_s{SPORT}.json"
CORE = "pinnacle,draftkings,fanduel"
SHOP_BOOKS = ["draftkings", "fanduel"]
# the tracked prop markets (user, 2026-10-20: starters and the sixth man, core stats and the combos)
PROP_MARKETS = {"playertotals-points": "pts", "playertotals-rebounds": "reb", "playertotals-assists": "ast",
                "playertotals-3pointers": "fg3", "playertotals-ptsassreb": "pra", "playertotals-ptsreb": "pr",
                "playertotals-ptsass": "pa", "playertotals-assreb": "ra"}
SEASON_START = "2026-10-01"


def _markets(props: bool) -> dict[int, tuple[str, float, str]]:
    out = {}
    for m in json.loads(MARKETS.read_text(encoding="utf-8")):
        if m["period"] != "result":
            continue
        if props and m["marketType"] in PROP_MARKETS or not props and not m["playerProp"] and m["marketType"] in ("moneyline", "spreads", "totals"):
            for o in m["outcomes"]:
                out[o["outcomeId"]] = (m["marketType"], float(m["handicap"]), o["outcomeName"])
    return out


def fixture_index() -> pl.DataFrame:
    """OddsPapi fixtures joined to hoopR game ids. participant1 is the home team; dates can differ by a day
    between UTC and Eastern, so teams are matched within one day."""
    fx = json.loads(FIXTURES.read_text(encoding="utf-8"))
    s = hoopr("schedule").filter(pl.col("season") >= 2025)
    name_to_abbr = dict(zip(s["home_display_name"], s["home_abbreviation"]))
    f = pl.DataFrame([{"fixture_id": x["fixtureId"], "start": x["startTime"], "status": x.get("statusId"),
                       "home": name_to_abbr.get(x["participant1Name"]), "away": name_to_abbr.get(x["participant2Name"])}
                      for x in fx]).filter(pl.col("home").is_not_null() & pl.col("away").is_not_null())
    f = f.with_columns(start=pl.col("start").str.to_datetime(time_zone="UTC"))
    g = s.select("game_id", "season", "season_type", pl.col("home_abbreviation").alias("home"),
                 pl.col("away_abbreviation").alias("away"), pl.col("game_date_time").dt.convert_time_zone("UTC").alias("tip"))
    j = f.join(g, on=["home", "away"], how="inner").filter(((pl.col("start") - pl.col("tip")).dt.total_hours().abs() <= 30))
    return j.sort("start").unique("game_id", keep="last").select("fixture_id", "game_id", "season", "home", "away", "start", "tip")


def archive(only: list[int] | None = None, past_hours: int = 40, ahead_hours: int = 40) -> None:
    """Fetch what is due: games starting within `ahead_hours`, and games that started up to `past_hours` ago
    whose files were written before they ended (one last fetch for closes and in-game prices)."""
    idx = fixture_index().filter(pl.col("start") >= datetime.fromisoformat(SEASON_START).replace(tzinfo=timezone.utc))
    now = datetime.now(timezone.utc)
    if only:
        idx = idx.filter(pl.col("game_id").is_in(only))
    else:
        idx = idx.filter((pl.col("start") <= now + timedelta(hours=ahead_hours)) & (pl.col("start") >= now - timedelta(hours=past_hours)))
    mmap, pmap = _markets(False), _markets(True)
    n = {"saved": 0, "skipped": 0, "missing": 0}
    for r in idx.iter_rows(named=True):
        fid, ended = r["fixture_id"], r["start"] + timedelta(hours=4)
        hp, pp = HIST / f"{fid}.parquet", PROPS / f"{fid}.parquet"
        done = lambda p: p.exists() and datetime.fromtimestamp(p.stat().st_mtime, timezone.utc) > ended
        if done(hp) and done(pp) and not only:
            n["skipped"] += 1
            continue
        h = _fetch(fid, CORE)
        if not h:
            n["missing"] += 1
            continue
        if rows := _rows(fid, h, mmap):
            _write(hp, pl.DataFrame(rows).with_columns(pl.col("ts").str.to_datetime(time_zone="UTC")))
        if prow := _prop_rows(fid, h, pmap):
            _write(pp, pl.DataFrame(prow, schema=PROP_SCHEMA, orient="row").with_columns(pl.col("ts").str.to_datetime(time_zone="UTC")))
        n["saved"] += 1
    print(f"nba odds: saved {n['saved']}, already complete {n['skipped']}, unavailable {n['missing']}")


OVERROUND = (1.0, 1.15)   # a real two-way market: both sides' implied chances sum to 100-115%


def lines_at(g: pl.DataFrame, cut: datetime) -> dict:
    """pipeline.odds_history.lines_at, keeping only pairs priced like a real two-way market. On opening night
    (2026-10-20) DraftKings carried a 116.5 "total" at 3.90 / 3.65 (a 53% book) that the closest-to-even rule took
    for the main line."""
    last = (g.filter(pl.col("ts") <= cut)
            .sort(["ts", "active", "price"], descending=[False, True, False], maintain_order=True)
            .group_by("market", "handicap", "outcome")
            .agg(pl.col("price").last(), pl.col("active").last())
            .filter(pl.col("active")))
    rec: dict = {}
    for market, side_a, side_b, name in (("spreads", "1", "2", "spread"), ("totals", "Over", "Under", "total")):
        m = last.filter(pl.col("market") == market)
        a = m.filter(pl.col("outcome") == side_a).select("handicap", pa="price")
        b = m.filter(pl.col("outcome") == side_b).select("handicap", pb="price")
        pair = (a.join(b, on="handicap")
                .with_columns(gap=(1 / pl.col("pa") - 1 / pl.col("pb")).abs(), book=1 / pl.col("pa") + 1 / pl.col("pb"))
                .filter(pl.col("book").is_between(*OVERROUND))
                .sort(["gap", "handicap"]))
        if pair.height:
            best = pair.row(0, named=True)
            rec[name] = -best["handicap"] if market == "spreads" else best["handicap"]
            rec[f"{name}_price_a"], rec[f"{name}_price_b"] = best["pa"], best["pb"]  # home/over, away/under
    ml = last.filter(pl.col("market") == "moneyline")
    prices = dict(zip(ml["outcome"], ml["price"]))
    if "1" in prices and "2" in prices and OVERROUND[0] <= 1 / prices["1"] + 1 / prices["2"] <= OVERROUND[1]:
        rec["ml_home"], rec["ml_away"] = prices["1"], prices["2"]
    return rec


class NbaLines(Lines):
    """board.market.Lines over the NBA archive: .at(book, when) gives the main spread/total/moneyline."""

    def __init__(self, fixture_id: str | None, tip: datetime):
        self.kickoff = tip
        p = HIST / f"{fixture_id}.parquet" if fixture_id else None
        self.h = pl.read_parquet(p).filter(pl.col("ts") < tip) if p and p.exists() else pl.DataFrame()

    def at(self, book: str, when: datetime) -> dict:
        if self.h.height == 0:
            return {}
        g = self.h.filter(pl.col("book") == book)
        return lines_at(g, min(when, self.kickoff - timedelta(minutes=5))) if g.height else {}


def main_lines(P: pl.DataFrame, cut: pl.Expr) -> pl.DataFrame:
    """props.lines.main_lines with the same two-way-market check as lines_at."""
    from props.lines import KEYS
    last = (P.filter(pl.col("ts") <= cut)
            .sort(["ts", "active", "price"], descending=[False, True, False], maintain_order=True)
            .group_by(KEYS + ["handicap", "outcome"])
            .agg(pl.col("price").last(), pl.col("active").last(), pl.col("ts").last().alias("seen"))
            .filter(pl.col("active")))
    o = last.filter(pl.col("outcome") == "Over").select(KEYS + ["handicap", pl.col("price").alias("over"), pl.col("seen").alias("so")])
    u = last.filter(pl.col("outcome") == "Under").select(KEYS + ["handicap", pl.col("price").alias("under"), pl.col("seen").alias("su")])
    pair = (o.join(u, on=KEYS + ["handicap"])
            .with_columns(gap=(1 / pl.col("over") - 1 / pl.col("under")).abs(),
                          p_over=(1 / pl.col("over")) / (1 / pl.col("over") + 1 / pl.col("under")),
                          bk=1 / pl.col("over") + 1 / pl.col("under"), seen=pl.max_horizontal("so", "su"))
            .filter(pl.col("bk").is_between(*OVERROUND)))
    return (pair.sort(["gap", "handicap"]).group_by(KEYS, maintain_order=True).first()
            .select(KEYS + [pl.col("handicap").alias("line"), "over", "under", "p_over", "seen"]))


# ------------------------------------------------------------------ players: OddsPapi ids -> hoopR athlete ids

SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm(name: str | None) -> str:
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`’\-]", "", s)
    return re.sub(r"\s+", " ", SUFFIX.sub("", s)).strip()


def _flip(name: str) -> str:
    last, _, first = name.partition(",")
    return f"{first.strip()} {last.strip()}".strip() if first else name.strip()


def player_map(people: pl.DataFrame) -> pl.DataFrame:
    """op_id -> athlete_id, matched by normalized name within the team, then by a league-wide unique name.
    `people`: athlete_id, name, team (hoopR abbreviation) for everyone who may play."""
    raw = json.loads(PLAYERS.read_text(encoding="utf-8"))["participants"]
    fx = json.loads(FIXTURES.read_text(encoding="utf-8"))
    s = hoopr("schedule").filter(pl.col("season") >= 2025)
    name_to_abbr = dict(zip(s["home_display_name"], s["home_abbreviation"]))
    team_of = {str(f[f"participant{k}Id"]): name_to_abbr.get(f[f"participant{k}Name"]) for f in fx for k in ("1", "2")}
    op = pl.DataFrame([{"op_id": p["playerId"], "key": norm(_flip(p["playerName"] or "")), "op_team": team_of.get(tid)}
                       for tid, v in raw.items() if team_of.get(tid) for p in v["players"] if p.get("playerName")]).unique("op_id")
    pe = people.with_columns(key=pl.col("name").map_elements(norm, return_dtype=pl.String)).unique(["athlete_id", "team"])
    by_team = op.join(pe, left_on=["key", "op_team"], right_on=["key", "team"], how="inner").select("op_id", "athlete_id")
    uniq = pe.group_by("key").agg(pl.col("athlete_id").n_unique().alias("n"), pl.col("athlete_id").first()).filter(pl.col("n") == 1)
    rest = op.join(by_team, on="op_id", how="anti").join(uniq.select("key", "athlete_id"), on="key", how="inner").select("op_id", "athlete_id")
    return pl.concat([by_team, rest]).unique("op_id")


def load_props(game_ids: list[int], pmap: pl.DataFrame, pregame_only: bool = True) -> pl.DataFrame:
    """Price points for the tracked markets, keyed by hoopR game and athlete ids (props.lines.main_lines reads it,
    with gsis_id standing for athlete_id)."""
    idx = fixture_index().filter(pl.col("game_id").is_in(game_ids))
    files = [PROPS / f"{fid}.parquet" for fid in idx["fixture_id"] if (PROPS / f"{fid}.parquet").exists()]
    if not files:
        return pl.DataFrame()
    P = (pl.scan_parquet(files).filter(pl.col("market").is_in(list(PROP_MARKETS)))
         .with_columns(market=pl.col("market").replace_strict(PROP_MARKETS)).collect()
         .join(idx.select("fixture_id", "game_id", pl.col("tip").alias("kickoff")), on="fixture_id")
         .join(pmap.rename({"op_id": "player"}), on="player", how="inner")
         .rename({"athlete_id": "gsis_id"}))
    if pregame_only:
        P = P.filter(pl.col("ts") < pl.col("kickoff"))
    return P.drop("fixture_id", "player")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refetch-fixtures", action="store_true")
    ap.add_argument("--games", nargs="*", type=int)
    a = ap.parse_args()
    if a.refetch_fixtures:
        FIXTURES.unlink(missing_ok=True)
        _get("fixtures", {"tournamentId": TOURNAMENT}, billable=True, cache_name=FIXTURES.name)
    archive(only=a.games)


if __name__ == "__main__":
    main()
