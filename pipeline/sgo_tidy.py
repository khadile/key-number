"""Tidy tables from the SportsGameOdds archive (raw events from pipeline/sgo.py).

    python -m pipeline.sgo_tidy

data/sgo/prop_quotes.parquet: one row per book per side of a full-game player
    over/under (passing, rushing, receiving yards, receptions) per regular-season
    game: the book's LAST quote (line, odds, when), and for 2026 its open and close.
data/sgo/props.parquet: one row per player-market per game, mapped to nflverse
    (game_id, gsis_id), with the settled stat SGO reports (`score`) and the
    consensus fields as SGO gives them.
data/sgo/game_quotes.parquet: the same per-book layout for full-game spreads,
    totals and moneylines.

What a "close" is depends on the season (see pipeline/sgo.py):
- 2026: the book's own `closeOdds`/`closeOverUnder`.
- 2024-2025: there is no per-book close, and SGO's consensus "close" is the last
  consensus, in-game prices included (checked 2026-10-09: Harold Fannin Jr.'s
  "close" was 94.5 when every book that stopped before kickoff had ~37.5). The
  only clean pregame number is a book's last quote when it came before kickoff:
  books that pull props at kickoff (Pinnacle, Caesars, bet365, ...) leave their
  close behind. `pregame` marks those rows; props/sgo_study.py checks the rule
  on 2026, where both exist.
"""
import glob
import gzip
import json
from datetime import datetime

import polars as pl

from config import CLEAN
from pipeline.odds_history import TEAMS
from pipeline.prop_players import norm
from pipeline.sgo import ALT_DIR, RAW_DIR, ROOT

STATS = {"passing_yards": "pass_yds", "rushing_yards": "rush_yds", "receiving_yards": "rec_yds",
         "receiving_receptions": "receptions"}
GAME_ODDS = {"points-home-game-sp-home": ("spread", "home"), "points-away-game-sp-away": ("spread", "away"),
             "points-all-game-ou-over": ("total", "over"), "points-all-game-ou-under": ("total", "under"),
             "points-home-game-ml-home": ("ml", "home"), "points-away-game-ml-away": ("ml", "away")}
TEAM_FIX = {"Washington Football Team": "WAS", "Oakland Raiders": "LV"}


def _am(x) -> int | None:
    if x in (None, ""):
        return None
    try:
        v = int(float(str(x).replace("+", "")))
    except ValueError:
        return None
    return v if abs(v) >= 100 else None


def _f(x) -> float | None:
    try:
        return float(x) if x not in (None, "") else None
    except ValueError:
        return None


def _ts(x) -> datetime | None:
    return datetime.fromisoformat(x.replace("Z", "+00:00")) if x else None


def _abbr(team: dict) -> str | None:
    long = (team.get("names") or {}).get("long")
    return TEAMS.get(long) or TEAM_FIX.get(long)


def read_events(season: int) -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(RAW_DIR / str(season) / "*.json.gz"))):
        with open(f, "rb") as fh:
            e = json.loads(gzip.decompress(fh.read()))
        if e.get("teams") and e["teams"].get("home") and e["teams"].get("away"):  # games, not season-long futures
            out.append(e)
    return out


def game_ids(events: list[dict], season: int) -> dict[str, str]:
    """eventID -> nflverse game_id, by teams and kickoff within 36 hours."""
    g = pl.read_parquet(CLEAN / "games.parquet").filter(pl.col("season") == season).select(
        "game_id", "home_team", "away_team", "kickoff_utc", "game_type")
    by_teams = {}
    for r in g.iter_rows(named=True):
        by_teams.setdefault((r["home_team"], r["away_team"]), []).append(r)
        by_teams.setdefault((r["away_team"], r["home_team"]), []).append(r)  # neutral-site listings flip
    out = {}
    for e in events:
        h, a = _abbr(e["teams"]["home"]), _abbr(e["teams"]["away"])
        t = _ts(e["status"]["startsAt"])
        for r in by_teams.get((h, a), []):
            if r["kickoff_utc"] is not None and abs((r["kickoff_utc"] - t).total_seconds()) < 36 * 3600:
                out[e["eventID"]] = r["game_id"]
    return out


def tidy(season: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    events = read_events(season)
    gid = game_ids(events, season)
    quotes, markets, gq = [], [], []
    for e in events:
        g = gid.get(e["eventID"])
        if g is None:
            continue
        start = e["status"]["startsAt"]
        teams = {"home": _abbr(e["teams"]["home"]), "away": _abbr(e["teams"]["away"])}
        team_of = {pid: p.get("teamID") for pid, p in (e.get("players") or {}).items()}
        name_of = {pid: p.get("name") for pid, p in (e.get("players") or {}).items()}
        long_to_abbr = {e["teams"][s]["teamID"]: teams[s] for s in ("home", "away")}
        for k, o in e.get("odds", {}).items():
            base = {"season": season, "game_id": g, "event_id": e["eventID"], "starts_at": start}
            if k in GAME_ODDS:
                market, side = GAME_ODDS[k]
                for b, v in (o.get("byBookmaker") or {}).items():
                    line = v.get("spread") if market == "spread" else v.get("overUnder")
                    oline = v.get("openSpread") if market == "spread" else v.get("openOverUnder")
                    cline = v.get("closeSpread") if market == "spread" else v.get("closeOverUnder")
                    gq.append({**base, "market": market, "side": side, "book": b, "line": _f(line), "odds": _am(v.get("odds")),
                               "last_at": v.get("lastUpdatedAt"), "open_line": _f(oline), "open_odds": _am(v.get("openOdds")),
                               "close_line": _f(cline), "close_odds": _am(v.get("closeOdds"))})
                continue
            if (o.get("statID") not in STATS or not o.get("playerID") or o.get("periodID") != "game"
                    or o.get("betTypeID") != "ou"):
                continue
            pid, m, side = o["playerID"], STATS[o["statID"]], o["sideID"]
            row = {**base, "sgo_player": pid, "market": m, "side": side}
            for b, v in (o.get("byBookmaker") or {}).items():
                quotes.append({**row, "book": b, "line": _f(v.get("overUnder")), "odds": _am(v.get("odds")),
                               "last_at": v.get("lastUpdatedAt"), "open_line": _f(v.get("openOverUnder")),
                               "open_odds": _am(v.get("openOdds")), "close_line": _f(v.get("closeOverUnder")),
                               "close_odds": _am(v.get("closeOdds"))})
            if side == "over":
                markets.append({**base, "sgo_player": pid, "sgo_name": name_of.get(pid),
                                "team": long_to_abbr.get(team_of.get(pid)), "market": m, "score": _f(o.get("score")),
                                "cancelled": bool(o.get("cancelled")),
                                "cons_line": _f(o.get("closeBookOverUnder") or o.get("bookOverUnder")),
                                "cons_over": _am(o.get("closeBookOdds") or o.get("bookOdds")),
                                "fair_line": _f(o.get("closeFairOverUnder") or o.get("fairOverUnder")),
                                "cons_open_line": _f(o.get("openBookOverUnder")), "cons_open_over": _am(o.get("openBookOdds")),
                                "fair_open_line": _f(o.get("openFairOverUnder"))})
    ts = pl.Datetime("us", "UTC")
    Q = pl.DataFrame(quotes, infer_schema_length=None).with_columns(
        pl.col("starts_at").str.to_datetime(time_zone="UTC").cast(ts), pl.col("last_at").str.to_datetime(time_zone="UTC").cast(ts))
    M = pl.DataFrame(markets, infer_schema_length=None).with_columns(pl.col("starts_at").str.to_datetime(time_zone="UTC").cast(ts))
    G = pl.DataFrame(gq, infer_schema_length=None).with_columns(
        pl.col("starts_at").str.to_datetime(time_zone="UTC").cast(ts), pl.col("last_at").str.to_datetime(time_zone="UTC").cast(ts))
    return Q, M, G


def map_players(M: pl.DataFrame) -> pl.DataFrame:
    """gsis_id for each SGO player, matched among the players who took a snap for that
    team in that game (props of players who sat out are void, so they stay unmapped)."""
    pg = pl.read_parquet(CLEAN / "props_player_games.parquet").select("game_id", "team", "gsis_id", "name", "position")
    pg = pg.with_columns(key=pl.col("name").map_elements(norm, return_dtype=pl.String))
    pg = pg.with_columns(last=pl.col("key").str.split(" ").list.last(), first1=pl.col("key").str.slice(0, 1))
    M = M.with_columns(key=pl.col("sgo_name").map_elements(norm, return_dtype=pl.String))
    M = M.with_columns(last=pl.col("key").str.split(" ").list.last(), first1=pl.col("key").str.slice(0, 1))
    P = M.select("game_id", "team", "sgo_player", "key", "last", "first1").unique()
    m1 = P.join(pg.select("game_id", "team", "key", "gsis_id"), on=["game_id", "team", "key"])
    m1 = m1.filter(pl.len().over("game_id", "sgo_player") == 1).with_columns(how=pl.lit("name"))
    rest = P.join(m1.select("game_id", "sgo_player"), on=["game_id", "sgo_player"], how="anti")
    m2 = rest.join(pg.select("game_id", "team", "last", "first1", "gsis_id"), on=["game_id", "team", "last", "first1"])
    m2 = m2.filter(pl.len().over("game_id", "sgo_player") == 1).with_columns(how=pl.lit("initial"))
    rest = rest.join(m2.select("game_id", "sgo_player"), on=["game_id", "sgo_player"], how="anti")
    m3 = rest.join(pg.select("game_id", "team", "last", "gsis_id"), on=["game_id", "team", "last"])
    m3 = m3.filter(pl.len().over("game_id", "sgo_player") == 1).with_columns(how=pl.lit("last"))
    ids = pl.concat([m.select("game_id", "sgo_player", "gsis_id", "how") for m in (m1, m2, m3)])
    return M.drop("key", "last", "first1").join(ids, on=["game_id", "sgo_player"], how="left")


HIST_BOOKS = ("pinnacle", "draftkings", "fanduel", "bet365", "betmgm", "caesars", "espnbet", "fanatics",
              "hardrockbet", "betrivers", "circa", "bookmakereu", "betonline")


def line_history(season: int) -> pl.DataFrame:
    """Every line value each book hung for a player's over/under, from the alt-line pull
    (data/sgo/raw_alt/): one row per (book, side, line) with the LAST price and time it was
    quoted at that value. A value the book came back to keeps only its last visit."""
    rows = []
    for f in sorted(glob.glob(str(ALT_DIR / str(season) / "*.json.gz"))):
        with open(f, "rb") as fh:
            e = json.loads(gzip.decompress(fh.read()))
        if not (e.get("teams") and e["teams"].get("home") and e["teams"].get("away")):
            continue
        base = {"season": season, "event_id": e["eventID"], "starts_at": e["status"]["startsAt"]}
        for k, o in (e.get("odds") or {}).items():
            if (o.get("statID") not in STATS or not o.get("playerID") or o.get("periodID") != "game"
                    or o.get("betTypeID") != "ou"):
                continue
            for b, v in (o.get("byBookmaker") or {}).items():
                if b not in HIST_BOOKS:
                    continue
                seen = {}
                for src, a in [("main", v)] + [("alt", x) for x in (v.get("altLines") or [])]:
                    line, odds, at = _f(a.get("overUnder")), _am(a.get("odds")), a.get("lastUpdatedAt")
                    if line is None or odds is None or not at:
                        continue
                    if line not in seen or at > seen[line]["last_at"]:
                        seen[line] = {**base, "sgo_player": o["playerID"], "market": STATS[o["statID"]], "side": o["sideID"],
                                      "book": b, "line": line, "odds": odds, "last_at": at, "src": src}
                rows.extend(seen.values())
    ts = pl.Datetime("us", "UTC")
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("starts_at").str.to_datetime(time_zone="UTC").cast(ts), pl.col("last_at").str.to_datetime(time_zone="UTC").cast(ts))


def main() -> None:
    Qs, Ms, Gs = [], [], []
    for season in (2024, 2025, 2026):
        Q, M, G = tidy(season)
        Qs.append(Q)
        Ms.append(M)
        Gs.append(G)
        print(f"{season}: {M['game_id'].n_unique()} games, {M.height} player-markets, {Q.height} book quotes, {G.height} game-line quotes")
    Q, M, G = (pl.concat(x, how="diagonal_relaxed") for x in (Qs, Ms, Gs))
    M = map_players(M)
    reg = pl.read_parquet(CLEAN / "games.parquet").filter(pl.col("game_type") == "REG").select("game_id", "week")
    M = M.join(reg, on="game_id")  # regular season only, like the props model
    Q = Q.join(reg.select("game_id"), on="game_id")
    G = G.join(reg, on="game_id")
    M.write_parquet(ROOT / "props.parquet")
    Q.write_parquet(ROOT / "prop_quotes.parquet")
    G.write_parquet(ROOT / "game_quotes.parquet")
    print(M.group_by("season").agg(games=pl.col("game_id").n_unique(), markets=pl.len(),
                                    mapped=pl.col("gsis_id").is_not_null().mean()).sort("season"))
    print(M.group_by("how").len())
    # line histories from the alt-line pull, keyed like props.parquet (game_id from the event id)
    ev = M.select("event_id", "game_id").unique()
    H = [line_history(s) for s in (2025, 2026) if (ALT_DIR / str(s)).exists()]
    if H:
        H = pl.concat(H).join(ev, on="event_id")
        H.write_parquet(ROOT / "prop_line_history.parquet")
        print(f"line history: {H.height} (book, side, line) entries for {H['game_id'].n_unique()} games")


if __name__ == "__main__":
    main()
