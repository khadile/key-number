"""Do unranked teams cover against Top 25 teams, early in the college season? (user, 2026-11-14)

The user's theory: Top 25 teams are overvalued early in the season, so the unranked side covers when it plays a
ranked team (and only when the other side is unranked). This tests it against the closing spread, season by season.

Data, all free:
- hoopR's college schedules (sportsdataverse/hoopR-mbb-data, GitHub): every game, the score, conferences, neutral.
- ESPN's daily scoreboard: each team's AP rank on the day of the game (hoopR carries it only from 2022-23).
- ESPN's odds history per game (the same source as nba/angles.py): the opening and closing spread and price of the
  book of record.

    python -m ncaab.top25            # fetch what is missing (data/ncaab, Actions cache), write reports/ncaab_top25.json

ESPN is not reachable from a cloud session's container; this runs on Actions (research.yml, job ncaab-top25).
"""
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np
import polars as pl

from config import DATA, REPORTS

DIR = DATA / "ncaab"
DIR.mkdir(parents=True, exist_ok=True)
OUT = REPORTS / "ncaab_top25.json"
SEASONS = range(2020, 2028)       # 2019-20 to 2026-27 (named by end year, as hoopR does)
HOOPR = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb/schedules/parquet/mbb_schedule_{s}.parquet"
BOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={d}&groups=50&limit=500"
CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/events/{g}/competitions/{g}/odds"
POWER = {2, 4, 7, 8, 21, 23}      # ESPN conference ids: ACC, Big East, Big Ten, Big 12, Pac-12, SEC
UA = {"User-Agent": "nfl_model"}


def _json(url: str, tries: int = 4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.5 * (i + 1))
        except OSError:
            time.sleep(1.5 * (i + 1))
    return None


def _dig(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def _num(x):
    if x is None:
        return None
    s = str(x).strip().lower().lstrip("ou").replace("+", "")
    try:
        return float(s)
    except ValueError:
        return 0.0 if s in ("pk", "even") else None


def _am(x):
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in ("EVEN", "EV"):
        return 100.0
    try:
        return float(s.replace("+", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------- data

def schedules() -> pl.DataFrame:
    frames = []
    for s in SEASONS:
        f = DIR / f"hoopr_{s}.parquet"
        if not f.exists() or s == max(SEASONS):
            try:
                with urllib.request.urlopen(urllib.request.Request(HOOPR.format(s=s), headers=UA), timeout=120) as r:
                    f.write_bytes(r.read())
            except OSError as e:
                print(f"hoopR {s}: {e}")
                if not f.exists():
                    continue
        x = pl.read_parquet(f)
        frames.append(x.select(pl.col("game_id").cast(pl.Int64), pl.col("season").cast(pl.Int32), pl.col("season_type").cast(pl.Int32),
                               pl.col("game_date").cast(pl.Date).alias("date"), pl.col("neutral_site").fill_null(False),
                               pl.col("home_id").cast(pl.Int64), pl.col("away_id").cast(pl.Int64),
                               pl.col("home_display_name").alias("home"), pl.col("away_display_name").alias("away"),
                               pl.col("home_conference_id").cast(pl.Int64, strict=False).alias("home_conf"),
                               pl.col("away_conference_id").cast(pl.Int64, strict=False).alias("away_conf"),
                               pl.col("home_score").cast(pl.Float64, strict=False), pl.col("away_score").cast(pl.Float64, strict=False),
                               pl.col("status_type_completed").fill_null(False).alias("done")))
    return pl.concat(frames, how="vertical_relaxed").filter(pl.col("season_type") == 2).unique("game_id")


def _board(d: date) -> list[dict]:
    j = _json(BOARD.format(d=d.strftime("%Y%m%d")))
    out = []
    for ev in (j or {}).get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        row = {"game_id": int(ev["id"]), "day": d.isoformat()}
        for c in comp.get("competitors", []):
            side = c.get("homeAway")
            if side in ("home", "away"):
                rk = _dig(c, "curatedRank", "current")
                row[f"{side}_rank"] = int(rk) if rk is not None else None
                row[f"{side}_tid"] = int(_dig(c, "team", "id") or 0)
        out.append(row)
    return out


def ranks(sched: pl.DataFrame) -> pl.DataFrame:
    """game_id -> each side's AP rank that day (ESPN's scoreboard), fetched once per date and cached."""
    f = DIR / "ranks.parquet"
    have = pl.read_parquet(f) if f.exists() else pl.DataFrame(schema={"game_id": pl.Int64, "day": pl.String})
    done_days = set(have["day"].unique().to_list()) if have.height else set()
    today = date.today()
    days = sorted({d for d in sched["date"].unique().to_list() if d is not None and d < today})
    todo = [d for d in days if d.isoformat() not in done_days or (today - d).days <= 3]
    print(f"scoreboard: {len(todo)} dates to fetch, {len(done_days)} cached")
    rows = []
    with ThreadPoolExecutor(6) as ex:
        for k, r in enumerate(ex.map(_board, todo)):
            rows += r
            if k and k % 200 == 0:
                print(f"  {k} of {len(todo)}", flush=True)
    if rows:
        new = pl.DataFrame(rows, infer_schema_length=None)
        have = pl.concat([have.filter(~pl.col("day").is_in([d.isoformat() for d in todo])), new], how="diagonal_relaxed") if have.height else new
        have.write_parquet(f)
    return have.unique("game_id", keep="last")


def _odds(gid: int) -> dict | None:
    d = _json(CORE.format(g=gid))
    if d is None:
        return {"game_id": gid}
    items = sorted(d.get("items", []), key=lambda it: _dig(it, "provider", "priority") or 99)
    if not items:
        return {"game_id": gid}
    it = items[0]
    row = {"game_id": gid, "provider": _dig(it, "provider", "name")}
    for when in ("open", "close"):
        for side in ("home", "away"):
            o = _dig(it, f"{side}TeamOdds", when) or {}
            row[f"{when}_spread_{side}"] = _num(_dig(o, "pointSpread", "american"))
            row[f"{when}_price_{side}"] = _am(_dig(o, "spread", "american"))
    if row.get("close_spread_home") is None and it.get("spread") is not None:   # some games keep only the current line
        row["close_spread_home"] = float(it["spread"])
        row["close_spread_away"] = -float(it["spread"])
    return row


def odds(gids: list[int]) -> pl.DataFrame:
    f = DIR / "odds.parquet"
    have = pl.read_parquet(f) if f.exists() else pl.DataFrame(schema={"game_id": pl.Int64})
    known = set(have["game_id"].to_list()) if have.height else set()
    todo = [g for g in gids if g not in known]
    print(f"odds: {len(todo)} games to fetch, {len(known)} cached")
    rows = []
    with ThreadPoolExecutor(6) as ex:
        for k, r in enumerate(ex.map(_odds, todo)):
            if r:
                rows.append(r)
            if k and k % 500 == 0:
                print(f"  {k} of {len(todo)}", flush=True)
    if rows:
        new = pl.DataFrame(rows, infer_schema_length=None)
        have = pl.concat([have, new], how="diagonal_relaxed") if have.height else new
        have.write_parquet(f)
    return have.unique("game_id", keep="last")


# ---------------------------------------------------------------- the test

def games() -> pl.DataFrame:
    s = schedules()
    r = ranks(s)
    rank = lambda c: pl.col(c).is_between(1, 25)
    g = s.join(r.select("game_id", "home_rank", "away_rank"), on="game_id", how="left")
    g = g.filter(pl.col("done") & pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
    one = g.filter(rank("home_rank").fill_null(False) ^ rank("away_rank").fill_null(False))
    o = odds(one["game_id"].to_list())
    one = one.join(o, on="game_id", how="left")
    rk_home = rank("home_rank").fill_null(False)
    out = one.with_columns(
        r_home=rk_home,
        r_rank=pl.when(rk_home).then(pl.col("home_rank")).otherwise(pl.col("away_rank")),
        u_name=pl.when(rk_home).then(pl.col("away")).otherwise(pl.col("home")),
        r_name=pl.when(rk_home).then(pl.col("home")).otherwise(pl.col("away")),
        u_conf=pl.when(rk_home).then(pl.col("away_conf")).otherwise(pl.col("home_conf")),
        # from the unranked side: its spread (positive = getting points), its margin, its price
        u_close=pl.when(rk_home).then(pl.col("close_spread_away")).otherwise(pl.col("close_spread_home")),
        u_open=pl.when(rk_home).then(pl.col("open_spread_away")).otherwise(pl.col("open_spread_home")),
        u_price=pl.when(rk_home).then(pl.col("close_price_away")).otherwise(pl.col("close_price_home")),
        u_margin=pl.when(rk_home).then(pl.col("away_score") - pl.col("home_score")).otherwise(pl.col("home_score") - pl.col("away_score")),
        venue=pl.when(pl.col("neutral_site")).then(pl.lit("neutral")).when(rk_home).then(pl.lit("ranked at home")).otherwise(pl.lit("ranked away")),
        month=pl.col("date").dt.month())
    return out.with_columns(u_power=pl.col("u_conf").is_in(list(POWER)))


def _ats(df: pl.DataFrame, spread: str = "u_close") -> dict:
    x = df.filter(pl.col(spread).is_not_null())
    d = (x["u_margin"] + x[spread]).to_numpy()
    w, l, p = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    n = w + l
    price = x["u_price"].fill_null(-110.0).to_numpy()
    dec = np.where(price > 0, 1 + price / 100, 1 + 100 / np.abs(np.where(price == 0, -110, price)))
    ret = np.where(d > 0, dec - 1, np.where(d < 0, -1.0, 0.0))
    ret110 = np.where(d > 0, 100 / 110, np.where(d < 0, -1.0, 0.0))
    return {"n": int(len(d)), "w": w, "l": l, "p": p, "cover": round(w / n, 4) if n else None,
            "se": round(float(np.sqrt(0.25 / n)), 4) if n else None,
            "roi": round(float(ret.mean()), 4) if len(d) else None, "roi_110": round(float(ret110.mean()), 4) if len(d) else None,
            "margin_vs_line": round(float(d.mean()), 2) if len(d) else None}


def report(G: pl.DataFrame) -> dict:
    early = pl.col("month").is_in([11, 12])
    dog = pl.col("u_close") > 0
    out = {"games": G.height, "with_close": int(G["u_close"].is_not_null().sum()), "rows": []}

    def add(group: str, label: str, df: pl.DataFrame, **kw):
        out["rows"].append({"group": group, "set": label, **_ats(df, **kw)})

    for per, sel in (("Nov-Dec (before conference play)", early), ("Jan-Mar", ~early), ("whole season", pl.lit(True))):
        x = G.filter(sel)
        add("all seasons", f"{per}: unranked side", x)
        add("all seasons", f"{per}: unranked underdog", x.filter(dog))
    E = G.filter(early & dog)
    seasons = sorted(G["season"].unique().to_list())
    for s in seasons:
        add("by season, Nov-Dec, unranked underdog", f"{s - 1}-{str(s)[2:]}", E.filter(pl.col("season") == s))
    for s in seasons:
        add("by season, Jan-Mar, unranked underdog", f"{s - 1}-{str(s)[2:]}", G.filter(~early & dog & (pl.col("season") == s)))
    for m, name in ((11, "November"), (12, "December"), (1, "January"), (2, "February"), (3, "March")):
        add("by month, unranked underdog", name, G.filter(dog & (pl.col("month") == m)))
    for lo, hi in ((0.5, 5), (5, 10), (10, 15), (15, 20), (20, 99)):
        add("Nov-Dec by spread", f"getting {lo:g} to {hi - 0.5:g}" if hi < 99 else f"getting {lo:g}+", E.filter(pl.col("u_close").is_between(lo, hi - 0.5)))
    for lo, hi in ((1, 5), (6, 15), (16, 25)):
        add("Nov-Dec by rank", f"ranked {lo}-{hi}", E.filter(pl.col("r_rank").is_between(lo, hi)))
    for v in ("ranked at home", "ranked away", "neutral"):
        add("Nov-Dec by venue", v, E.filter(pl.col("venue") == v))
    add("Nov-Dec by opponent", "unranked from a power conference", E.filter(pl.col("u_power")))
    add("Nov-Dec by opponent", "unranked from elsewhere", E.filter(~pl.col("u_power")))
    mv = E.filter(pl.col("u_open").is_not_null()).with_columns(move=pl.col("u_close") - pl.col("u_open"))
    add("Nov-Dec by line move", "line moved toward the ranked team (more points)", mv.filter(pl.col("move") > 0))
    add("Nov-Dec by line move", "no move", mv.filter(pl.col("move") == 0))
    add("Nov-Dec by line move", "line moved toward the unranked team (fewer points)", mv.filter(pl.col("move") < 0))
    add("Nov-Dec at the opening line", "unranked underdog, opening spread", E.filter(pl.col("u_open") > 0), spread="u_open")
    out["moves"] = {"n": mv.height, "mean_move": round(float(mv["move"].mean()), 3) if mv.height else None,
                    "toward_ranked": int((mv["move"] > 0).sum()), "toward_unranked": int((mv["move"] < 0).sum())}
    out["providers"] = dict(G["provider"].value_counts().iter_rows()) if "provider" in G.columns else {}
    return out


def main() -> None:
    G = games()
    R = report(G)
    OUT.write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    print(f"{R['games']} games with exactly one ranked team, {R['with_close']} with a closing spread; providers {R['providers']}")
    print(f"line moves Nov-Dec (unranked underdog): {R['moves']}")
    group = None
    for r in R["rows"]:
        if r["group"] != group:
            group = r["group"]
            print(f"\n== {group}")
        cv = "–" if r["cover"] is None else f"{100 * r['cover']:.1f}% ± {100 * r['se']:.1f}"
        roi = "–" if r["roi"] is None else f"{100 * r['roi']:+.1f}% (at -110 {100 * r['roi_110']:+.1f}%)"
        print(f"  {r['set']:52} n {r['n']:5d}  {r['w']}-{r['l']}-{r['p']}  covered {cv}  ROI {roi}  vs line {r['margin_vs_line']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
