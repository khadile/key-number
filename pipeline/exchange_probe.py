"""Does OddsPapi carry the exchanges' player props? (2026-11-11 checkup: FanDuel prices NBA props better than
Pinnacle, and outside studies rank the exchanges, Novig and Kalshi, above every sportsbook for props.)

Free calls only: /account and /historical-odds (one call per 5 seconds, paced by pipeline/oddspapi.py). For a
few recent and upcoming NBA and NFL games it asks for each candidate book and prints what came back: how many
markets, how many tracked props with both sides priced before the start, how wide the two sides are, how often
they move. Counts and margins only, never a price: the log of a public repository is public.

    python -m pipeline.exchange_probe            # 2 finished + 1 upcoming NBA game, 1 + 1 NFL
"""
import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from pipeline.oddspapi import CACHE, _get, account

CANDIDATES = ["novig.us", "novig", "kalshi", "prophetx", "polymarket", "sporttrade"]
SPORTS = {"nba": (132, 11), "nfl": (31, 14)}


def _catalog(sport_id: int) -> dict[int, tuple[str, float, str, bool]]:
    """outcomeId -> (marketType, handicap, outcomeName, playerProp), full-game markets."""
    out = {}
    for m in json.loads((CACHE / f"markets_s{sport_id}.json").read_text(encoding="utf-8")):
        if m["period"] == "result":
            for o in m["outcomes"]:
                out[o["outcomeId"]] = (m["marketType"], float(m["handicap"]), o["outcomeName"], bool(m["playerProp"]))
    return out


def _fixtures(tournament: int, finished: int, upcoming: int) -> list[tuple[str, datetime, str]]:
    fx = json.loads((CACHE / f"fixtures_t{tournament}.json").read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)
    rows = sorted((datetime.fromisoformat(x["startTime"].replace("Z", "+00:00")), x["fixtureId"]) for x in fx if x.get("startTime"))
    past = [(f, s, "finished") for s, f in rows if s < now - timedelta(hours=6)][-finished:] if finished else []
    nxt = [(f, s, "upcoming") for s, f in rows if s > now + timedelta(hours=1)][:upcoming]
    return past + nxt


def _summary(book: str, b: dict, cat: dict, start: datetime) -> dict:
    rows, types, n_markets = [], Counter(), len(b.get("markets", {}))
    for m in b.get("markets", {}).values():
        for oid, o in m.get("outcomes", {}).items():
            info = cat.get(int(oid))
            if not info:
                types["(not in the full-game catalog)"] += 1
                continue
            for pid, pts in o.get("players", {}).items():
                for p in pts:
                    rows.append((info[0], info[1], info[2], info[3], int(pid), p["createdAt"], float(p["price"]), bool(p["active"]), p.get("limit")))
    if not rows:
        return {"book": book, "markets": n_markets, "points": 0, "other": dict(types)}
    d = pl.DataFrame(rows, schema=["market", "handicap", "outcome", "prop", "player", "ts", "price", "active", "limit"], orient="row",
                     infer_schema_length=None).with_columns(pl.col("ts").str.to_datetime(time_zone="UTC"))
    pre = d.filter(pl.col("ts") < start)
    out = {"book": book, "markets": n_markets, "points": d.height, "pregame_points": pre.height,
           "prop_types": dict(Counter(d.filter(pl.col("prop")).unique(["market", "player", "handicap", "outcome"])["market"].to_list()).most_common(12)),
           "line_types": dict(Counter(d.filter(~pl.col("prop")).unique(["market", "handicap", "outcome"])["market"].to_list()).most_common(8))}
    for name, sel in (("props", pl.col("prop")), ("lines", ~pl.col("prop") & pl.col("market").is_in(["moneyline", "spreads", "totals"]))):
        x = pre.filter(sel)
        if not x.height:
            continue
        keys = ["market", "player", "handicap"]
        last = (x.sort("ts").group_by(keys + ["outcome"]).agg(pl.col("price").last(), pl.col("active").last(), n=pl.len(),
                                                              first=pl.col("ts").first(), last_ts=pl.col("ts").last(), lim=pl.col("limit").last()))
        act = last.filter(pl.col("active"))
        two = (act.group_by(keys).agg(sides=pl.len(), book=(1 / pl.col("price")).sum(), moves=pl.col("n").sum(), first=pl.col("first").min(),
                                      last_ts=pl.col("last_ts").max(), lim=pl.col("lim").max()).filter(pl.col("sides") == 2))
        out[name] = {"series": last.height, "two_sided_at_start": two.height,
                     "players": int(two["player"].n_unique()) if name == "props" else None,
                     "by_market": dict(Counter(two["market"].to_list()).most_common(12)),
                     "two_sides_sum_median": round(float(two["book"].median()), 4) if two.height else None,
                     "two_sides_sum_p10_p90": [round(float(np.quantile(two["book"].to_numpy(), q)), 4) for q in (0.1, 0.9)] if two.height else None,
                     "moves_per_market_median": float(two["moves"].median()) if two.height else None,
                     "first_seen_hours_before_median": round(float(((start - two["first"]).dt.total_minutes() / 60).median()), 1) if two.height else None,
                     "last_move_minutes_before_median": round(float(((start - two["last_ts"]).dt.total_minutes()).median()), 0) if two.height else None,
                     "limit_median": float(two["lim"].drop_nulls().median()) if two["lim"].drop_nulls().len() else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finished", type=int, default=2)
    ap.add_argument("--upcoming", type=int, default=1)
    ap.add_argument("--books", default=",".join(CANDIDATES))
    args = ap.parse_args()
    a = account()
    listed = sorted({bk for s in a.get("subscriptions", []) if s.get("is_active") for bk in s.get("bookmakers", {})}) if "error" not in a else []
    print(f"books on the plan: {len(listed)}")
    cands = [c for c in args.books.split(",") if c]
    if listed:
        near = [bk for bk in listed if any(k in bk for k in ("novig", "kalshi", "prophet", "poly", "sporttrade", "exchange", "betfair", "matchbook", "smarkets"))]
        print("exchange-like names on the plan:", near or "none")
        cands = [c for c in cands if c in listed] + [bk for bk in near if bk not in cands]
    print("asking for:", cands)
    calls, found = 0, {}
    for sport, (tournament, sport_id) in SPORTS.items():
        cat = _catalog(sport_id)
        fx = _fixtures(tournament, args.finished if sport == "nba" else 1, args.upcoming)
        good, first = set(), True
        for fid, start, kind in fx:
            ask = cands if first else sorted(good)   # after a sport's first game, only the books that had data
            first = False
            if not ask:
                break
            groups = [ask[i:i + 3] for i in range(0, len(ask), 3)]
            for grp in groups:
                h = _get("historical-odds", {"fixtureId": fid, "bookmakers": ",".join(grp)}, billable=False)
                calls += 1
                if "error" in h and len(grp) > 1:   # one unknown name can fail the group: ask one at a time
                    print(f"{sport} {kind} {start:%m-%d %H:%M}Z {grp}: error {h.get('error')} {str(h.get('detail'))[:160]}")
                    hs = {}
                    for c in grp:
                        r = _get("historical-odds", {"fixtureId": fid, "bookmakers": c}, billable=False)
                        calls += 1
                        if "error" in r:
                            print(f"   {c}: error {r.get('error')} {str(r.get('detail'))[:160]}")
                        else:
                            hs.update(r.get("bookmakers", {}))
                    h = {"bookmakers": hs}
                elif "error" in h:
                    print(f"{sport} {kind} {start:%m-%d %H:%M}Z {grp}: error {h.get('error')} {str(h.get('detail'))[:160]}")
                    continue
                got = h.get("bookmakers", {})
                print(f"{sport} {kind} {start:%m-%d %H:%M}Z asked {grp}: returned {sorted(got)}")
                for book, b in got.items():
                    s = _summary(book, b, cat, start)
                    if s["points"]:
                        good.add(book)
                    print("  ", json.dumps(s))
            if calls > 40:
                break
        found[sport] = sorted(good)
    print(f"calls {calls} (all free); books with data: {found}")


if __name__ == "__main__":
    main()
