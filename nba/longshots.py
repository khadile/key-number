"""Long-shot player props ("hail marys"): the "N+" rungs DraftKings and FanDuel post (user, 2026-10-20).

The user asked for a lookout on plays like 6+ threes at +1000: rarely right, cheap to make, and maybe priced
wrong more often than the main lines. A rung's fair chance comes from the props model's shape (nba/props.py:
minutes, then a negative binomial given minutes) scaled to the market's level, the player's main line at
Pinnacle (else DraftKings and FanDuel), exactly as the main-line props are priced.

The tail is where the shape matters most, and the model's ranges were calibrated with its own forecast error in
them (a projection misses where the market's line would not), which fattens the tail at the market's level: a
25.5-point scorer's chance of 40+ moves by about a third between the model's width and a market-informed one.
So every rung is priced twice, at the model's width and at a narrower one (minutes 20% tighter, less spread
given minutes), and a long shot counts only when it has value at both.

    python -m nba.longshots            # tonight's rungs at the latest archived prices
"""
import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from nba import props as NP

NARROW_MIN, NARROW_FLOOR = 0.8, 4.5
NARROW_DISP = {"pts": 2.0, "reb": 1.1, "ast": 1.12, "fg3": 1.05}


def narrow(row: dict, market: str) -> NP.Mix:
    """The tighter shape: minutes spread x0.8 (at least 4.5) and less spread given minutes."""
    r = dict(row, sd_min=max(row["sd_min"] * NARROW_MIN, NARROW_FLOOR))
    m = NP.Mix.of(r, market)
    parts = NP.COMBOS.get(market, (market,))
    rates = {k: row[f"r_{k}"] * row.get("pace_factor", 1.0) for k in parts}
    return replace(m, disp=sum(NARROW_DISP[k] * rates[k] for k in parts) / sum(rates.values()))


def price_rungs(x: pl.DataFrame, main_books: dict, rungs: pl.DataFrame, min_price: float = 3.0) -> pl.DataFrame:
    """Every rung priced +200 or longer, for tracked players whose main line gives the market's level."""
    pr = NP.PropPricerNba(0.0)
    rows = []
    rung_by: dict = {}
    for r in rungs.filter(pl.col("price") >= min_price).iter_rows(named=True):
        rung_by.setdefault((r["game_id"], r["gsis_id"], r["market"]), []).append(r)
    for p in x.iter_rows(named=True):
        for m in NP.MARKETS:
            key = (p["game_id"], p["athlete_id"], m)
            if key not in rung_by or key not in main_books:
                continue
            wide, tight = NP.Mix.of(p, m), narrow(p, m)
            lam_w, src = pr.anchor(wide, main_books[key])
            lam_t, _ = pr.anchor(tight, main_books[key])
            if lam_w is None:
                continue
            for r in rung_by[key]:
                line = r["at_least"] - 0.5
                pw, pt = wide.probs(line, lam_w)[0], tight.probs(line, lam_t)[0]
                rows.append({"game_id": p["game_id"], "athlete_id": p["athlete_id"], "name": p["name"], "team": p["team"],
                             "market": m, "at_least": r["at_least"], "book": r["book"], "dec": r["price"],
                             "p_wide": pw, "p_narrow": pt, "ev_wide": pw * r["price"] - 1, "ev_narrow": pt * r["price"] - 1,
                             "anchor": src, "main_line": (main_books[key].get("pinnacle") or next(iter(main_books[key].values())))["line"]})
    return pl.DataFrame(rows)


def main() -> None:
    from nba import build as B, espn
    from nba.odds import load_milestones, load_props, main_lines, player_map, rungs_at
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", help="ISO time (default now)")
    a = ap.parse_args()
    now = datetime.fromisoformat(a.at).astimezone(timezone.utc) if a.at else datetime.now(timezone.utc)
    today = now.astimezone(B.ET).date()
    sched = B.schedule()
    games = list(sched.filter(pl.col("day") == today).iter_rows(named=True))
    roster, inj = espn.rosters(), espn.injuries()
    positions = dict(zip(roster["athlete_id"].to_list(), roster["position"].to_list()))
    prof = NP.profiles(NP.history(B.SEASON, positions), today + timedelta(days=1), B.SEASON)
    x = NP.slate([{"game_id": g["game_id"], "home": g["home"], "away": g["away"], "spread": 0.0, "pace_factor": 1.0} for g in games],
                 prof, roster, inj, today)
    pmap = player_map(roster.select("athlete_id", "name", "team"))
    ids = [g["game_id"] for g in games]
    main_books = B._books(main_lines(load_props(ids, pmap), pl.lit(now)))
    R = price_rungs(x, main_books, rungs_at(load_milestones(ids, pmap), now))
    pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
    R = R.with_columns(band=pl.col("dec").cut([5.0, 11.0, 21.0], labels=["+200 to +400", "+400 to +1000", "+1000 to +2000", "+2000 or longer"]))
    print(f"{R.height} rungs priced +200 or longer")
    print(R.group_by("band", "market").agg(pl.len().alias("rungs"), pl.col("ev_wide").mean().round(3), pl.col("ev_narrow").mean().round(3),
                                           ((pl.col("ev_wide") > 0) & (pl.col("ev_narrow") > 0)).sum().alias("both_pos")).sort("band", "market"))
    print(R.group_by("band").agg(pl.len().alias("rungs"), pl.col("ev_wide").mean().round(3), pl.col("ev_narrow").mean().round(3),
                                 (pl.col("ev_wide") > 0).mean().round(3).alias("pos_wide"), (pl.col("ev_narrow") > 0).mean().round(3).alias("pos_narrow")).sort("band"))
    print(R.filter((pl.col("ev_wide") > 0) & (pl.col("ev_narrow") > 0)).sort("ev_narrow", descending=True)
          .select("name", "market", "at_least", "book", "dec", "main_line", "p_wide", "p_narrow", "ev_wide", "ev_narrow").head(25))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- the lookout, nightly

from config import DATA, LEDGER

SNAP = DATA / "nba" / "longshots"          # every rung priced at each props lock (prices: off the public repo)
SNAP.mkdir(parents=True, exist_ok=True)
PICKS = LEDGER / "nba_longshots.jsonl"     # the flagged long shots, 0.25u each, their own record
RESULTS = LEDGER / "nba_longshot_results.jsonl"
CAL = LEDGER / "nba_longshot_cal.jsonl"    # per graded game: how the books' long shots did against their prices
FLAG_EV = 0.10        # value at both widths: the tail is uncertain, so the bar is high
FLAG_MAX = 8          # a night's list, best first, one per player
STAKE = 0.25
BANDS = ((0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.34))


def snapshot(R: pl.DataFrame, day, now) -> None:
    if not R.height:
        return
    p = SNAP / f"{day.isoformat()}_{now:%H%M}.parquet"
    R.with_columns(day=pl.lit(day.isoformat()), at=pl.lit(now.isoformat())).write_parquet(p)


def flags(R: pl.DataFrame, day, now) -> list[dict]:
    from board.market import american
    if not R.height:
        return []
    f = R.filter((pl.col("ev_wide") >= FLAG_EV) & (pl.col("ev_narrow") >= FLAG_EV)).sort("ev_narrow", descending=True)
    out, seen = [], set()
    for r in f.iter_rows(named=True):
        if r["athlete_id"] in seen:
            continue
        seen.add(r["athlete_id"])
        lab = NP.MARKET_LABEL[r["market"]].lower()
        out.append({"kind": "longshot", "id": f"{r['game_id']}:{r['athlete_id']}:{r['market']}:{r['at_least']}+:{r['book']}",
                    "game_id": r["game_id"], "day": day.isoformat(), "athlete_id": r["athlete_id"], "name": r["name"], "team": r["team"],
                    "market": r["market"], "at_least": int(r["at_least"]), "label": f"{r['name']} {r['at_least']}+ {lab}",
                    "book": r["book"], "book_name": {"draftkings": "DraftKings", "fanduel": "FanDuel"}.get(r["book"], r["book"]),
                    "dec": round(r["dec"], 3), "american": american(r["dec"]), "main_line": r["main_line"],
                    "p_wide": round(r["p_wide"], 4), "p_narrow": round(r["p_narrow"], 4), "ev_wide": round(r["ev_wide"], 4),
                    "ev_narrow": round(r["ev_narrow"], 4), "risk": STAKE, "locked_at": now.isoformat()})
        if len(out) == FLAG_MAX:
            break
    return out


def grade(picks: list[dict], stats: dict, finished: set) -> list[dict]:
    out = []
    for p in picks:
        if p.get("kind") != "longshot" or p["game_id"] not in finished:
            continue
        s = stats.get((p["game_id"], p["athlete_id"]))
        if s is None:
            res, u, actual = "V", 0.0, None
        else:
            actual = s.get(p["market"])
            res = "W" if actual >= p["at_least"] else "L"
            u = STAKE * (p["dec"] - 1) if res == "W" else -STAKE
        out.append({"id": p["id"], "result": res, "units": round(u, 4), "actual": actual})
    return out


def calibration_rows(finished: set, stats: dict, done: set) -> list[dict]:
    """For each newly final game with a snapshot: every rung's book-implied chance against whether it hit, banded
    (no prices kept, only counts), from the last snapshot before tip."""
    if not finished:
        return []
    files = sorted(SNAP.glob("*.parquet"))
    if not files:
        return []
    S = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    S = S.filter(pl.col("game_id").is_in(list(finished - done)))
    if not S.height:
        return []
    S = S.sort("at").group_by("game_id", "athlete_id", "market", "at_least", "book").agg(pl.all().last())
    out = []
    for gid, g in S.group_by("game_id"):
        gid = gid[0]
        bands: dict = {}
        for r in g.iter_rows(named=True):
            s = stats.get((gid, r["athlete_id"]))
            if s is None:
                continue
            implied = 1 / r["dec"]
            for lo, hi in BANDS:
                if lo <= implied < hi:
                    k = f"{lo:.2f}-{hi:.2f}"
                    b = bands.setdefault(k, {"n": 0, "hits": 0, "implied": 0.0, "p_narrow": 0.0, "p_wide": 0.0})
                    b["n"] += 1
                    b["hits"] += int(s.get(r["market"], 0) >= r["at_least"])
                    b["implied"] += implied
                    b["p_narrow"] += r["p_narrow"]
                    b["p_wide"] += r["p_wide"]
        if bands:
            out.append({"game_id": gid, "day": g["day"][0], "bands": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in bands.items()}})
    return out


def calibration_table(rows: list[dict]) -> list[dict]:
    tot: dict = {}
    for r in rows:
        for k, b in r["bands"].items():
            t = tot.setdefault(k, {"n": 0, "hits": 0, "implied": 0.0, "p_narrow": 0.0, "p_wide": 0.0})
            for kk in t:
                t[kk] += b[kk]
    return [{"band": k, "n": int(t["n"]), "hit": round(t["hits"] / t["n"], 4), "implied": round(t["implied"] / t["n"], 4),
             "model_narrow": round(t["p_narrow"] / t["n"], 4), "model_wide": round(t["p_wide"] / t["n"], 4)}
            for k, t in sorted(tot.items()) if t["n"]]
