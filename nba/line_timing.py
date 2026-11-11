"""NBA game lines: does the time of the lock matter? (user, 2026-11-11: "why are there no changes even though
the model is significantly underperforming?")

The board's game-line rule is line shopping: Pinnacle's no-vig number is the fair one and the pick is the best
DraftKings or FanDuel price against it (nba/lines.py; the game model sets no price, k = 0). The board locks at
10:30am ET. This replays the same rule on every finished game of the season at other moments before the tip,
from the archive's full price timelines (data/nba/odds/history, Actions cache), and judges each pick three ways:
its EV when taken, its EV at Pinnacle's closing no-vig number, and the result.

Also:
- "ahead": the fair number pushed half the gap past Pinnacle, away from the US books (2026-11-11: DraftKings
  and FanDuel ended 1.4 to 1.6 gaps from where they started, past Pinnacle's morning number).
- steam: picks split by whether Pinnacle moved toward the pick over the three hours before.

    python -m nba.line_timing        # prints the tables, writes reports/nba_line_timing.json (tallies, no prices)
"""
import json
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from config import REPORTS
from nba.lines import SHOP, GamePricer, _novig, center, fit
from nba.odds import HIST, fixture_index, lines_at

ET = ZoneInfo("America/New_York")
HOURS = (20, 14, 10, 8, 6, 4, 2.5, 1.5, 0.75, 0.25)          # before the tip
CLOCK = (("10:30am ET", time(10, 30)), ("1:00pm ET", time(13, 0)), ("3:30pm ET", time(15, 30)), ("5:45pm ET", time(17, 45)))
OUT = REPORTS / "nba_line_timing.json"


def _finals() -> dict[int, tuple[int, int]]:
    from nba.build import finals_and_stats
    return finals_and_stats()[0]


def _units(pk: dict, margin: float, total: float) -> float:
    if pk["market"] == "spread":
        d = (margin if pk["side"] == "home" else -margin) + pk["line"]
    elif pk["market"] == "total":
        d = (total - pk["line"]) * (1 if pk["side"] == "over" else -1)
    else:
        d = margin if pk["side"] == "home" else -margin
    return pk["dec"] - 1 if d > 0 else (0.0 if d == 0 else -1.0)


def _toward(pk: dict, move_m: float | None, move_t: float | None) -> int:
    """+1 Pinnacle moved toward the pick's side over the last three hours, -1 away, 0 under a quarter point."""
    mv = move_t if pk["market"] == "total" else move_m
    if mv is None or abs(mv) < 0.25:
        return 0
    up = pk["side"] in ("home", "over")
    return 1 if (mv > 0) == up else -1


def rows(pricer: GamePricer) -> pl.DataFrame:
    idx, finals, out = fixture_index(), _finals(), []
    now = datetime.now(timezone.utc)
    for g in idx.filter(pl.col("tip") < now - timedelta(hours=5)).iter_rows(named=True):
        p = HIST / f"{g['fixture_id']}.parquet"
        if g["game_id"] not in finals or not p.exists():
            continue
        tip = g["tip"]
        h = pl.read_parquet(p).filter(pl.col("ts") < tip)
        by = {b: h.filter(pl.col("book") == b) for b in ("pinnacle", *SHOP)}
        if not by["pinnacle"].height:
            continue
        hs, as_ = finals[g["game_id"]]
        margin, total = float(hs - as_), float(hs + as_)
        close = pricer.fair({"pinnacle": lines_at(by["pinnacle"], tip - timedelta(minutes=5))})
        day = tip.astimezone(ET).date()
        moments = [(f"{x:g}h before", tip - timedelta(hours=x)) for x in HOURS]
        moments += [(name, datetime.combine(day, t, ET).astimezone(timezone.utc)) for name, t in CLOCK]
        for name, when in moments:
            if when > tip - timedelta(minutes=10):
                continue
            books = {b: lines_at(d, when) for b, d in by.items() if d.height}
            fair = pricer.fair(books)
            then = pricer.fair({"pinnacle": lines_at(by["pinnacle"], when - timedelta(hours=3))})
            mv_m = fair["margin"] - then["margin"] if "margin" in fair and "margin" in then else None
            mv_t = fair["total"] - then["total"] if "total" in fair and "total" in then else None
            gaps, ahead = {}, dict(fair)
            for key, dist, nm in (("spread", pricer.mdist, "margin"), ("total", pricer.tdist, "total")):
                if fair.get(f"{nm}_src") != "pinnacle":
                    fair.pop(nm, None), ahead.pop(nm, None)
                    continue
                us = [center(dist, books[b][key], _novig(books[b][f"{key}_price_a"], books[b][f"{key}_price_b"])) for b in SHOP if key in books.get(b, {})]
                if us:
                    gaps[key] = fair[nm] - float(np.mean(us))
                    ahead[nm] = fair[nm] + 0.5 * gaps[key]
            if "ml_home" not in books.get("pinnacle", {}):
                fair.pop("p_home", None), ahead.pop("p_home", None)
            for rule, f in (("board", fair), ("ahead", ahead)):
                opts = pricer.options(books, f, g["home"], g["away"])
                for market, o in opts.items():
                    if not o or (rule == "ahead" and market == "ml"):
                        continue
                    pk = max(o, key=lambda x: x["ev"])
                    ps = pricer.p_side(pk, close)
                    evc = None if ps is None else ps[0] * (pk["dec"] - 1) - (1 - ps[0] - ps[1])
                    out.append({"game_id": g["game_id"], "day": str(day), "moment": name, "rule": rule, "market": market,
                                "ev": pk["ev"], "ev_close": evc, "units": _units(pk, margin, total), "book": pk["book"],
                                "steam": _toward(pk, mv_m, mv_t), "gap": abs(gaps[market]) if market in gaps else None})
    return pl.DataFrame(out, infer_schema_length=None)


def _line(g: pl.DataFrame) -> dict:
    n = g.height
    c = g["ev_close"].drop_nulls()
    u = g["units"].to_numpy()
    se = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None
    return {"n": n, "ev": round(float(g["ev"].mean()), 4), "ev_close": round(float(c.mean()), 4) if c.len() else None,
            "ev_close_se": round(se(c.to_numpy()), 4) if c.len() > 1 else None, "held": round(float((c > 0).mean()), 3) if c.len() else None,
            "roi": round(float(u.mean()), 4), "roi_se": round(se(u), 4) if n > 1 else None,
            "w": int((u > 0).sum()), "l": int((u < 0).sum())}


def _fmt(r: dict) -> str:
    pm = lambda v, s: "    –" if v is None else f"{100 * v:+5.1f}" + (f"±{100 * s:.1f}" if s is not None else "")
    return (f"n {r['n']:4d}  EV then {100 * r['ev']:+5.1f}%  at close {pm(r['ev_close'], r['ev_close_se'])}%  still +EV {'–' if r['held'] is None else f'{100 * r['held']:.0f}%'}"
            f"  {r['w']}-{r['l']}  ROI {pm(r['roi'], r['roi_se'])}%")


def main() -> None:
    md, td, _ = fit()
    D = rows(GamePricer(md, td))
    games = D["game_id"].n_unique()
    print(f"{games} finished games with Pinnacle's timeline, {D['day'].min()} to {D['day'].max()}")
    order = [f"{x:g}h before" for x in HOURS] + [n for n, _ in CLOCK]
    report = {"games": games, "from": D["day"].min(), "to": D["day"].max(), "rows": []}
    b = D.filter(pl.col("rule") == "board")
    print("\n=== 0. How far DraftKings and FanDuel sit from Pinnacle (points, mean absolute gap of the fair numbers) ===")
    for m in order:
        x = b.filter(pl.col("moment") == m)
        s, t = x.filter(pl.col("market") == "spread")["gap"].drop_nulls(), x.filter(pl.col("market") == "total")["gap"].drop_nulls()
        if s.len():
            print(f"  {m:12} games {s.len():4d}  spread {s.mean():.2f}  total {t.mean() if t.len() else float('nan'):.2f}")
            report["rows"].append({"table": "gap", "moment": m, "n": s.len(), "spread": round(float(s.mean()), 3), "total": round(float(t.mean()), 3) if t.len() else None})
    for rule, title in (("board", "1. The board's rule (best DK/FD price against Pinnacle's no-vig number) by when it is taken"),
                        ("ahead", "2. The same with the fair number half a gap past Pinnacle (EV then is at that number)")):
        print(f"\n=== {title} ===")
        for lo in (0.0, 0.01, 0.02):
            for market in ("spread", "total", "ml", "all"):
                print(f"  -- {market}, EV {100 * lo:.0f}%+ --")
                for m in order:
                    x = D.filter((pl.col("rule") == rule) & (pl.col("moment") == m) & (pl.col("ev") >= lo if lo else pl.col("ev") > 0))
                    x = x if market == "all" else x.filter(pl.col("market") == market)
                    if x.height < 8:
                        continue
                    r = _line(x)
                    print(f"     {m:12} {_fmt(r)}")
                    report["rows"].append({"table": rule, "moment": m, "market": market, "min_ev": lo, **r})
    print("\n=== 3. Every game's best side whatever its EV (what a pick costs when nothing is +EV) ===")
    for m in order:
        x = b.filter((pl.col("moment") == m) & (pl.col("market") != "ml"))
        if x.height >= 8:
            r = _line(x)
            print(f"     {m:12} {_fmt(r)}")
            report["rows"].append({"table": "every", "moment": m, **r})
    print("\n=== 4. Steam: +EV board picks by Pinnacle's move over the three hours before (spreads and totals) ===")
    for m in order:
        for s, lab in ((1, "toward the pick"), (0, "no move"), (-1, "away from it")):
            x = b.filter((pl.col("moment") == m) & (pl.col("ev") > 0) & (pl.col("steam") == s) & (pl.col("market") != "ml"))
            if x.height >= 8:
                r = _line(x)
                print(f"     {m:12} {lab:16} {_fmt(r)}")
                report["rows"].append({"table": "steam", "moment": m, "steam": s, **r})
    print("\n=== 5. By book: +EV board picks, spreads and totals ===")
    for m in ("20h before", "10:30am ET", "5:45pm ET", "1.5h before"):
        for bk in SHOP:
            x = b.filter((pl.col("moment") == m) & (pl.col("ev") > 0) & (pl.col("book") == bk) & (pl.col("market") != "ml"))
            if x.height >= 8:
                r = _line(x)
                print(f"     {m:12} {bk:11} {_fmt(r)}")
                report["rows"].append({"table": "book", "moment": m, "book": bk, **r})
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
