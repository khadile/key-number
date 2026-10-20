"""Which simple NBA betting angles have paid at DraftKings' real prices? (user, 2026-10-20)

The user asked whether underdogs, favorites' moneylines strung into parlays ("combo ML plays"), "ladder"
parlays of 2-3 likely legs priced +100 to +150, alternate lines and long-shot alt props could carry an edge
over a full season of games. This module tests the ones that history can test.

Prices: ESPN's odds API keeps DraftKings' opening and closing lines for every game (moneyline, spread and
total, each with its price), free, cached here in data/nba/espn_odds.parquet. Results from hoopR.

    python -m nba.angles            # fetch what is missing, then write reports/nba_angles.json
    python -m nba.angles --no-fetch

Every angle is reported as units won per unit staked (ROI) at DraftKings' price, with its standard error, by
season, so a lucky season is visible. A parlay of independent legs returns the product of its legs'
returns, so a parlay angle's ROI follows from its legs': (1 + leg ROI) ^ legs - 1, which the report also
checks directly on same-day combinations.
"""
import argparse
import itertools
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl

from config import DATA, REPORTS
from nba.data import load

OUT = DATA / "nba" / "espn_odds.parquet"
REPORT = REPORTS / "nba_angles.json"
CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{g}/competitions/{g}/odds"
SEASONS = range(2022, 2027)    # 2021-22 to 2025-26


def _am(x) -> float | None:
    """American odds from ESPN's strings ("-192", "+160", "EVEN")."""
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in ("EVEN", "EV", "PK"):
        return 100.0
    try:
        return float(s.replace("+", ""))
    except ValueError:
        return None


def _num(x) -> float | None:
    if x is None:
        return None
    s = str(x).strip().lower().lstrip("ou").replace("+", "")
    try:
        return float(s)
    except ValueError:
        return 0.0 if s in ("pk", "even") else None


def _dig(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def _fetch(gid: int, tries: int = 4) -> dict | None:
    for i in range(tries):
        try:
            req = urllib.request.Request(CORE.format(g=gid), headers={"User-Agent": "nfl_model"})
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.loads(r.read())
            items = sorted(d.get("items", []), key=lambda it: _dig(it, "provider", "priority") or 99)
            if not items:
                return {"game_id": gid}
            it = items[0]
            row = {"game_id": gid, "provider": _dig(it, "provider", "name")}
            for when in ("open", "close"):
                for side in ("home", "away"):
                    o = _dig(it, f"{side}TeamOdds", when) or {}
                    row[f"{when}_ml_{side}"] = _am(_dig(o, "moneyLine", "american"))
                    row[f"{when}_spread_{side}"] = _num(_dig(o, "pointSpread", "american"))
                    row[f"{when}_spread_price_{side}"] = _am(_dig(o, "spread", "american"))
                t = _dig(it, when) or {}
                row[f"{when}_total"] = _num(_dig(t, "total", "american")) or _num(_dig(t, "total", "alternateDisplayValue"))
                row[f"{when}_over"] = _am(_dig(t, "over", "american"))
                row[f"{when}_under"] = _am(_dig(t, "under", "american"))
            return row
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {"game_id": gid}     # no odds kept for this game
            time.sleep(1.5 * (i + 1))
        except OSError:
            time.sleep(1.5 * (i + 1))
    return None


def fetch(seasons=SEASONS, workers: int = 6) -> pl.DataFrame:
    s = load("schedule").filter(pl.col("season").is_in(list(seasons)) & (pl.col("season_type") == 2)
                                & pl.col("status_type_completed").fill_null(False))
    ids = s["game_id"].cast(pl.Int64).to_list()
    have = pl.read_parquet(OUT) if OUT.exists() else pl.DataFrame()
    todo = [g for g in ids if not have.height or g not in set(have["game_id"].to_list())]
    print(f"ESPN odds: {len(todo)} games to fetch, {have.height} cached")
    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(_fetch, todo)):
            if r:
                rows.append(r)
            if k and k % 500 == 0:
                print(f"  {k} of {len(todo)}", flush=True)
    if rows:
        new = pl.DataFrame(rows, infer_schema_length=None)
        have = pl.concat([have, new], how="diagonal_relaxed") if have.height else new
        have.write_parquet(OUT)
    return have


# ---------------------------------------------------------------- the tests

def dec(a: pl.Expr) -> pl.Expr:
    return pl.when(a > 0).then(1 + a / 100).otherwise(1 + 100 / (-a))


def games() -> pl.DataFrame:
    s = load("schedule").select(pl.col("game_id").cast(pl.Int64), "season", pl.col("game_date").cast(pl.Date).alias("date"),
                                pl.col("home_score").cast(pl.Float64).alias("hs"), pl.col("away_score").cast(pl.Float64).alias("as_"))
    o = pl.read_parquet(OUT)
    return o.join(s, on="game_id").filter(pl.col("hs").is_not_null() & (pl.col("hs") + pl.col("as_") > 0))


def _roi(ret: np.ndarray) -> dict:
    """Units per unit staked, its standard error, and the count. `ret`: each bet's return per unit (-1 lost)."""
    n = len(ret)
    if n == 0:
        return {"n": 0}
    return {"n": int(n), "roi": round(float(ret.mean()), 4), "se": round(float(ret.std(ddof=1) / np.sqrt(n)), 4) if n > 1 else None,
            "win": round(float((ret > 0).mean()), 4)}


def ml_bets(g: pl.DataFrame, when: str = "close") -> pl.DataFrame:
    """One row per side per game: the moneyline bet at DraftKings' `when` price and its return."""
    rows = []
    for side, other, won in (("home", "away", pl.col("hs") > pl.col("as_")), ("away", "home", pl.col("as_") > pl.col("hs"))):
        rows.append(g.filter(pl.col(f"{when}_ml_{side}").is_not_null() & pl.col(f"{when}_ml_{other}").is_not_null()).select(
            "game_id", "season", "date", pl.lit(side).alias("side"), pl.col(f"{when}_ml_{side}").alias("american"),
            dec(pl.col(f"{when}_ml_{side}")).alias("dec"), dec(pl.col(f"{when}_ml_{other}")).alias("dec_other"), won.alias("won")))
    b = pl.concat(rows)
    return b.with_columns(ret=pl.when(pl.col("won")).then(pl.col("dec") - 1).otherwise(-1.0),
                          p_novig=(1 / pl.col("dec")) / (1 / pl.col("dec") + 1 / pl.col("dec_other")))


BANDS = [(-10000, -500, "-500 or shorter"), (-500, -300, "-500 to -300"), (-300, -200, "-300 to -200"), (-200, -130, "-200 to -130"),
         (-130, 130, "-130 to +130"), (130, 200, "+130 to +200"), (200, 300, "+200 to +300"), (300, 500, "+300 to +500"), (500, 10000, "+500 or longer")]


def _band(a: float) -> str:
    for lo, hi, name in BANDS:
        if lo <= a < hi or (a == 100 and lo == -130):
            return name
    return "?"


def report(g: pl.DataFrame) -> dict:
    out: dict = {"games": g.height, "seasons": f"{int(g['season'].min()) - 1}-{int(g['season'].min()) % 100:02d} to "
                 f"{int(g['season'].max()) - 1}-{int(g['season'].max()) % 100:02d}",
                 "providers": dict(g.group_by("provider").len().iter_rows())}
    # --- moneylines by price band, at the close and at the open
    for when in ("close", "open"):
        b = ml_bets(g, when).with_columns(band=pl.col("american").map_elements(_band, return_dtype=pl.String))
        rows = []
        for lo, hi, name in BANDS:
            x = b.filter(pl.col("band") == name)
            r = _roi(x["ret"].to_numpy())
            if r["n"]:
                r.update(band=name, said=round(float(x["p_novig"].mean()), 4), seasons={int(s): _roi(x.filter(pl.col("season") == s)["ret"].to_numpy()).get("roi")
                                                                                      for s in sorted(x["season"].unique())})
            rows.append(r)
        out[f"ml_{when}"] = rows
        out[f"ml_{when}_dogs"] = _roi(b.filter(pl.col("american") > 0)["ret"].to_numpy())
        out[f"ml_{when}_favs"] = _roi(b.filter(pl.col("american") < 0)["ret"].to_numpy())
    # --- spreads and totals at the close: favorites vs underdogs, home dogs, overs vs unders
    sp = []
    for side, sign in (("home", 1), ("away", -1)):
        x = g.filter(pl.col(f"close_spread_{side}").is_not_null() & pl.col(f"close_spread_price_{side}").is_not_null())
        m = (pl.col("hs") - pl.col("as_")) * sign + pl.col(f"close_spread_{side}")
        sp.append(x.select("season", pl.lit(side).alias("side"), pl.col(f"close_spread_{side}").alias("line"),
                           dec(pl.col(f"close_spread_price_{side}")).alias("dec"),
                           pl.when(m > 0).then(dec(pl.col(f"close_spread_price_{side}")) - 1).when(m == 0).then(0.0).otherwise(-1.0).alias("ret")))
    sp = pl.concat(sp)
    out["spread"] = {"dogs": _roi(sp.filter(pl.col("line") > 0)["ret"].to_numpy()), "favs": _roi(sp.filter(pl.col("line") < 0)["ret"].to_numpy()),
                     "home_dogs": _roi(sp.filter((pl.col("line") > 0) & (pl.col("side") == "home"))["ret"].to_numpy()),
                     "big_dogs_10": _roi(sp.filter(pl.col("line") >= 10)["ret"].to_numpy()),
                     "big_favs_10": _roi(sp.filter(pl.col("line") <= -10)["ret"].to_numpy())}
    t = g.filter(pl.col("close_total").is_not_null() & pl.col("close_over").is_not_null())
    tot = t["hs"].to_numpy() + t["as_"].to_numpy()
    line = t["close_total"].to_numpy()
    dov, dun = (t.select(dec(pl.col("close_over"))).to_series().to_numpy(), t.select(dec(pl.col("close_under"))).to_series().to_numpy())
    out["totals"] = {"overs": _roi(np.where(tot > line, dov - 1, np.where(tot == line, 0.0, -1.0))),
                     "unders": _roi(np.where(tot < line, dun - 1, np.where(tot == line, 0.0, -1.0)))}
    # --- same-day moneyline parlays of favorites ("combo ML plays"): every 2- and 3-leg combination of the day's
    # favorites in a price band, from different games, at the close
    b = ml_bets(g, "close").filter(pl.col("american") < 0)
    combos = {}
    for lo, hi, name in ((-400, -200, "favorites -400 to -200"), (-250, -150, "favorites -250 to -150"), (-10000, -400, "favorites -400 or shorter")):
        x = b.filter((pl.col("american") >= lo) & (pl.col("american") < hi))
        for legs in (2, 3):
            rets, decs = [], []
            for _, day in x.group_by("date"):
                r_, d_ = day["ret"].to_numpy(), day["dec"].to_numpy()
                for c in itertools.combinations(range(len(r_)), legs):
                    won = all(r_[i] > 0 for i in c)
                    price = float(np.prod(d_[list(c)]))
                    rets.append(price - 1 if won else -1.0)
                    decs.append(price)
            leg = _roi(x["ret"].to_numpy())
            r = _roi(np.array(rets))
            if r["n"]:
                r.update(median_price=round(float(np.median(decs)), 2), leg_roi=leg.get("roi"),
                         implied_from_legs=round((1 + leg["roi"]) ** legs - 1, 4) if leg.get("roi") is not None else None)
            combos[f"{name}, {legs} legs"] = r
    out["ml_parlays"] = combos
    # --- line moves: betting the side the line moved toward, at the open (known only in hindsight: it measures
    # how much a price that beats the close is worth)
    b_o, b_c = ml_bets(g, "open"), ml_bets(g, "close")
    j = b_o.join(b_c.select("game_id", "side", pl.col("p_novig").alias("p_close")), on=["game_id", "side"])
    moved = j.filter(pl.col("p_close") > pl.col("p_novig") + 0.02)
    out["open_beats_close"] = {"bets": _roi(moved["ret"].to_numpy()),
                               "note": "moneylines whose no-vig chance rose 2+ points from open to close, bet at the open"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    if not a.no_fetch:
        fetch()
    r = report(games())
    REPORT.write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1)[:6000])


if __name__ == "__main__":
    main()
