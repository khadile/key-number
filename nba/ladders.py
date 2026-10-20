"""Alternate lines at DraftKings and FanDuel against the fair price (user, 2026-10-20).

The user asked about "ladder" parlays: two or three alternate spreads or totals likely enough to win (say 70-90%)
strung together for +100 to +150. A parlay returns the product of its legs' returns, so it has value only when
its legs do. This prices every alternate spread and total the two books post, both sides, against the fair
line: Pinnacle's no-vig main line as the center and the shape of 7,250 games around their closing lines
(nba/lines.py), and reports what each kind of leg is worth.

    python -m nba.ladders [--hours 24] [--step 2]   # snapshots over the last day of archived prices
"""
import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from nba.lines import GamePricer, _novig, fit
from nba.odds import HIST, OVERROUND, fixture_index, lines_at

BUCKETS = ((0.0, 0.3), (0.3, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0))


def ladder(h: pl.DataFrame, cut: datetime) -> pl.DataFrame:
    """Every two-sided alternate spread and total a book showed at `cut`."""
    last = (h.filter(pl.col("ts") <= cut).sort(["ts", "active", "price"], descending=[False, True, False], maintain_order=True)
            .group_by("book", "market", "handicap", "outcome").agg(pl.col("price").last(), pl.col("active").last())
            .filter(pl.col("active") & pl.col("market").is_in(["spreads", "totals"])))
    a = last.filter(pl.col("outcome").is_in(["1", "Over"])).select("book", "market", "handicap", pa="price")
    b = last.filter(pl.col("outcome").is_in(["2", "Under"])).select("book", "market", "handicap", pb="price")
    return a.join(b, on=["book", "market", "handicap"]).filter((1 / pl.col("pa") + 1 / pl.col("pb")).is_between(*OVERROUND))


def legs(pr: GamePricer, h: pl.DataFrame, cut: datetime, home: str, away: str) -> list[dict]:
    pin = lines_at(h.filter(pl.col("book") == "pinnacle"), cut)
    fair = pr.fair({"pinnacle": pin}) if pin else {}
    if "margin" not in fair or "total" not in fair:
        return []
    out = []
    L = ladder(h.filter(pl.col("book").is_in(["draftkings", "fanduel"])), cut)
    for r in L.iter_rows(named=True):
        if r["market"] == "spreads":
            s = -r["handicap"]                       # home favored by s
            if abs(s - fair["margin"]) > 25:
                continue
            o, push, u = pr.mdist.side_probs(fair["margin"], s)
            sides = (("home", o, r["pa"], -s), ("away", u, r["pb"], s))
            off = s - fair["margin"]
        else:
            t = r["handicap"]
            if abs(t - fair["total"]) > 30:          # DraftKings' 100-120 "totals" are not the game total
                continue
            o, push, u = pr.tdist.side_probs(fair["total"], t)
            sides = (("over", o, r["pa"], t), ("under", u, r["pb"], t))
            off = t - fair["total"]
        for side, p, dec, line in sides:
            out.append({"book": r["book"], "market": "spread" if r["market"] == "spreads" else "total", "side": side,
                        "line": line, "off": off, "p": p, "push": push, "dec": dec,
                        "ev": p * (dec - 1) - (1 - p - push), "hold": 1 / r["pa"] + 1 / r["pb"] - 1})
    return out


def study(hours: int = 24, step: int = 2) -> pl.DataFrame:
    md, td, _ = fit()
    pr = GamePricer(md, td)
    idx = fixture_index()
    now = datetime.now(timezone.utc)
    rows = []
    for f in sorted(HIST.glob("*.parquet")):
        g = idx.filter(pl.col("fixture_id") == f.stem)
        if not g.height:
            continue
        g = g.row(0, named=True)
        h = pl.read_parquet(f)
        end = min(now, g["tip"] - timedelta(minutes=5))
        for k in range(0, hours + 1, step):
            cut = end - timedelta(hours=k)
            for leg in legs(pr, h, cut, g["home"], g["away"]):
                rows.append({"game": f"{g['away']}@{g['home']}", "cut": cut, **leg})
    return pl.DataFrame(rows)


def summarize(D: pl.DataFrame) -> pl.DataFrame:
    D = D.with_columns(bucket=pl.col("p").cut([b[1] for b in BUCKETS[:-1]], labels=[f"{int(lo*100)}-{int(hi*100)}%" for lo, hi in BUCKETS]))
    return (D.group_by("market", "bucket").agg(pl.len().alias("legs"), pl.col("ev").mean().round(4).alias("mean_ev"),
                                               pl.col("ev").median().round(4).alias("median_ev"),
                                               (pl.col("ev") > 0).mean().round(3).alias("share_pos"),
                                               pl.col("ev").max().round(4).alias("best"), pl.col("hold").mean().round(4).alias("hold"))
            .sort("market", "bucket"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--step", type=int, default=2)
    a = ap.parse_args()
    D = study(a.hours, a.step)
    pl.Config.set_tbl_rows(40)
    print(f"{D.height} priced legs from {D['game'].n_unique()} games")
    print(summarize(D))
    pos = D.filter(pl.col("ev") > 0).sort("ev", descending=True)
    print(pos.select("game", "cut", "book", "market", "side", "line", "off", "p", "dec", "ev").head(20))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- the Golden Parlay: likely legs with value, strung together
# The user's Golden Parlay (2026-10-20): "my most confident picks, it should most likely cash", main or alternate
# lines. It replaced both the old NBA Golden (most EV among +EV main-line picks at +150 to +300) and the Ladder
# (likely +EV legs at +100 to +150), which did the same job.

LADDER_P_MIN = 0.60       # a leg the user would call likely
LADDER_EV_MIN = 0.01      # and worth at least 1% at the book's price
LADDER_BAND = (2.0, 4.0)  # +100 to +300; the most likely ticket usually lands near +100 to +150


def ladder_legs(h: pl.DataFrame, cut: datetime, fair: dict, pr: GamePricer, game: dict) -> list[dict]:
    """Every side of every two-sided spread and total (main and alternate) and both moneylines at DraftKings and
    FanDuel at `cut`, priced at the fair line; the ones likely enough and worth enough to be Ladder legs."""
    if "margin" not in fair or "total" not in fair:
        return []
    out = []
    L = ladder(h.filter(pl.col("book").is_in(["draftkings", "fanduel"])), cut)
    for r in L.iter_rows(named=True):
        if r["market"] == "spreads":
            s = -r["handicap"]
            if abs(s - fair["margin"]) > 25:
                continue
            o, push, u = pr.mdist.side_probs(fair["margin"], s)
            opts = (("home", game["home"], o, r["pa"], -s), ("away", game["away"], u, r["pb"], s))
            for side, team, p, dec, line in opts:
                out.append({"market": "spread", "side": side, "team": team, "line": line, "book": r["book"], "dec": dec,
                            "p_win": p, "p_push": push, "ev": p * (dec - 1) - (1 - p - push),
                            "label": f"{team} {'+' if line > 0 else '' if line < 0 else ''}{line:g}" if line else f"{team} PK"})
        else:
            t = r["handicap"]
            if abs(t - fair["total"]) > 30:
                continue
            o, push, u = pr.tdist.side_probs(fair["total"], t)
            for side, p, dec in (("over", o, r["pa"]), ("under", u, r["pb"])):
                out.append({"market": "total", "side": side, "team": None, "line": t, "book": r["book"], "dec": dec,
                            "p_win": p, "p_push": push, "ev": p * (dec - 1) - (1 - p - push), "label": f"{side.title()} {t:g}"})
    for b in ("draftkings", "fanduel"):
        m = lines_at(h.filter(pl.col("book") == b), cut)
        if "ml_home" in m and "p_home" in fair:
            for side, team, p, dec in (("home", game["home"], fair["p_home"], m["ml_home"]), ("away", game["away"], 1 - fair["p_home"], m["ml_away"])):
                out.append({"market": "ml", "side": side, "team": team, "line": None, "book": b, "dec": dec, "p_win": p,
                            "p_push": 0.0, "ev": p * (dec - 1) - (1 - p), "label": f"{team} ML"})
    return [x for x in out if x["p_win"] >= LADDER_P_MIN and x["ev"] >= LADDER_EV_MIN]


def best_ladder(legs_by_game: dict[int, list[dict]], band: tuple[float, float] = LADDER_BAND) -> list[dict] | None:
    """The 2-3 leg combination from different games priced in `band` most likely to cash (each leg already has
    value, so the ticket does; ties go to more value). Per game only its five likeliest legs are tried."""
    import itertools
    import math
    pools = {g: sorted(v, key=lambda x: (-x["p_win"], -x["ev"]))[:5] for g, v in legs_by_game.items() if v}
    best, best_key = None, (0.0, 0.0)
    games_ = list(pools)
    for n in (2, 3):
        for gs in itertools.combinations(games_, n):
            for legs in itertools.product(*(pools[g] for g in gs)):
                dec = math.prod(l["dec"] for l in legs)
                if not band[0] <= dec <= band[1]:
                    continue
                ev = math.prod(l["p_win"] * l["dec"] + l["p_push"] for l in legs) - 1
                hit = math.prod(l["p_win"] + l["p_push"] for l in legs) - math.prod(l["p_push"] for l in legs)
                if ev > 0 and (hit, ev) > best_key:
                    best, best_key = list(legs), (hit, ev)
    return best
