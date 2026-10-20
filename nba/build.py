"""Build the NBA side of the board, every day (user, 2026-10-20).

    python -m nba.build                # refresh, grade, lock what is due, write board/nba_data.json, render the page
    python -m nba.build --offline      # no downloads: rebuild from what is on disk
    python -m nba.build --dry-run      # compute everything, write nothing to the ledgers
    python -m nba.build --lock-now     # an update run early: lock now what today's later builds would lock

Two builds a day (nba/slot.py runs the scheduled ones):

- Morning, 10:30am ET on: grades last night from ESPN's box scores, locks the day's game lines with their
  Picks of the Day and parlays, and locks props for games that tip before 6:30pm ET.
- Evening, 5:30pm ET on (after the league's 5pm injury report): locks the rest of the day's props, then the
  day's props Picks of the Day and parlays.

Game lines lock in the morning because a market-anchored pick has no edge but the price, and the morning's
DraftKings and FanDuel numbers have had the least sharp money through them; props wait for the injury report,
because who plays moves them most. Both are judgment calls, replayed at the first checkup: the archive keeps every
price, so morning against evening locks can be compared on the same games.
"""
import argparse
import json
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from board.ledger import append, read
from board.market import BOOK_NAMES, american, build_parlays, kelly
from config import PARLAY_RISK, REPORTS, ROOT
from model.staking import flat_risk, kelly_units
from nba import espn, ledger
from nba import props as NP
from nba.data import CURRENT, load
from nba.lines import GamePricer, fit as fit_lines
from nba.odds import NbaLines, fixture_index, load_props, main_lines, player_map

ET = ZoneInfo("America/New_York")
SEASON = CURRENT
OUT = ROOT / "board" / "nba_data.json"
MORNING, EVENING, LATE_TIP = time(10, 30), time(17, 30), time(18, 30)
BOOKS = ["pinnacle", "draftkings", "fanduel"]
SHOP = ["draftkings", "fanduel"]
B2B_MARGIN = -2.2      # points of home margin per back-to-back (home minus away), 2020-21 to 2025-26
PROPS_K = 0.0          # how far prop lines move toward the model between lock and close: measured once there is data
POTD_N = 5
PARLAY_POOL = 16
# A prop is a pick at 1% expected value or more. Below that the edge is inside the error of the no-vig estimate
# (splitting the vig evenly or by the odds moves a -110 side's chance by more): opening night had picks at +0.03%.
PROP_EV_MIN = 0.01


def _f(x, nd=2):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (datetime, date)):
        return x.isoformat()
    return x


def _tip(dt: datetime) -> dict:
    et = dt.astimezone(ET)
    return {"iso": dt.isoformat(), "day": et.strftime("%A, %B ") + str(et.day),
            "time": et.strftime("%I:%M %p").lstrip("0") + " ET", "short": et.strftime("%a ") + et.strftime("%I:%M %p").lstrip("0")}


# ---------------------------------------------------------------- schedule and lock times

def schedule(season: int = SEASON) -> pl.DataFrame:
    s = load("schedule").filter((pl.col("season") == season) & pl.col("season_type").is_in([2, 3, 5]))
    s = s.select(pl.col("game_id").cast(pl.Int64), pl.col("game_date_time").dt.convert_time_zone("UTC").alias("tip"),
                 pl.col("home_abbreviation").alias("home"), pl.col("away_abbreviation").alias("away"),
                 pl.col("neutral_site").fill_null(False).alias("neutral"))
    return s.with_columns(day=pl.col("tip").dt.convert_time_zone("America/New_York").dt.date()).sort("tip", "game_id")


def _at(day: date, t: time) -> datetime:
    return datetime.combine(day, t, ET).astimezone(timezone.utc)


def lines_lock(tip: datetime) -> datetime:
    """Game lines lock at the morning build of the game's day (or half an hour before an earlier tip)."""
    return min(_at(tip.astimezone(ET).date(), MORNING), tip - timedelta(minutes=30))


def props_lock(tip: datetime) -> datetime:
    """Props lock at the evening build (after the 5pm injury report) for games tipping 6:30pm ET or later, and at
    the morning build for earlier games."""
    et = tip.astimezone(ET)
    t = _at(et.date(), EVENING if et.time() >= LATE_TIP else MORNING)
    return min(t, tip - timedelta(minutes=30))


def _due(lock: datetime, tip: datetime, now: datetime, lock_now: bool) -> bool:
    if lock_now and lock.astimezone(ET).date() == now.astimezone(ET).date():
        lock = min(lock, now)
    return lock <= now < tip


# ---------------------------------------------------------------- box scores: grading and this season's minutes

def fetch_boxes(sched: pl.DataFrame, now: datetime) -> int:
    """Save ESPN's box score for every game of the season that should be over and is not saved yet."""
    n = 0
    for r in sched.filter(pl.col("tip") < now - timedelta(hours=2, minutes=30)).iter_rows(named=True):
        p = NP.box_path(r["game_id"])
        if p.exists():
            continue
        try:
            b = espn.box(r["game_id"])
        except RuntimeError as e:
            print("box score unavailable:", r["game_id"], e)
            continue
        if b:
            b["date"] = r["day"].isoformat()
            p.write_text(json.dumps(b), encoding="utf-8")
            n += 1
    return n


def finals_and_stats() -> tuple[dict[int, tuple[int, int]], dict[tuple[int, int], dict]]:
    finals, stats = {}, {}
    for p in NP.BOX_DIR.glob("*.json"):
        b = json.loads(p.read_text(encoding="utf-8"))
        finals[b["game_id"]] = (b["home_score"], b["away_score"])
        for r in b["players"]:
            if r["played"]:
                s = {k: r[k] or 0.0 for k in NP.STATS}
                s.update({c: sum(s[x] for x in parts) for c, parts in NP.COMBOS.items()})
                stats[(b["game_id"], r["athlete_id"])] = s
    return finals, stats


# ---------------------------------------------------------------- the game model (shown, and the side of a lean)

def game_models(games: list[dict], today: date, roster: pl.DataFrame, out_ids: set[int], prof: pl.DataFrame,
                sched: pl.DataFrame) -> dict[int, dict]:
    from nba.model import fit, team_games
    from nba.players import baseline, fit_values, lineup_value, player_games
    tg = team_games(range(SEASON - 2, SEASON + 1))
    r = fit(tg, today, SEASON)
    if r is None:
        return {}
    pg = player_games(range(SEASON - 2, SEASON + 1))
    vals = fit_values(pg, tg, today, SEASON)
    prev = {}
    for g in sched.sort("tip").iter_rows(named=True):     # each team's previous game day, for back-to-backs
        for t in (g["home"], g["away"]):
            prev.setdefault(t, []).append(g["day"])
    def rest(team: str, day: date) -> int:
        before = [d for d in prev.get(team, []) if d < day]
        return min((day - max(before)).days, 7) if before else 7
    out = {}
    for g in games:
        p = r.project(g["home"], g["away"], neutral=g["neutral"])
        if p is None:
            continue
        margin, total, pace = p
        hr, ar = rest(g["home"], g["day"]), rest(g["away"], g["day"])
        b2b = B2B_MARGIN * (float(hr == 1) - float(ar == 1))
        adj = 0.0
        for team, sign in ((g["home"], 1), (g["away"], -1)):
            mins = NP.expected_minutes(prof, roster, team, out_ids, today)
            top = [a for a, _ in sorted(mins.items(), key=lambda kv: -kv[1])[:10]]
            adj += sign * (lineup_value(vals, top) - baseline(vals, pg, team, today))
        out[g["game_id"]] = {"margin": margin + b2b + adj * pace / 100, "team_margin": margin, "b2b": b2b,
                             "lineup": adj * pace / 100, "total": total, "pace": pace, "pace_factor": pace / r.mu_p,
                             "rest": {"home": hr, "away": ar}}
    return out


# ---------------------------------------------------------------- picks

def game_picks(g: dict, books: dict, fair: dict, model: dict | None, pricer: GamePricer, now: datetime) -> list[dict]:
    pk = pricer.picks(books, fair, g["home"], g["away"], (model or {}).get("margin"), (model or {}).get("total"))
    out = []
    for market, p in pk.items():
        out.append({**p, "kind": "pick", "id": f"{g['game_id']}:{market}", "game_id": g["game_id"], "day": g["day"].isoformat(),
                    "home": g["home"], "away": g["away"], "book_name": BOOK_NAMES.get(p["book"], p["book"]),
                    "p_win": round(p["p_win"], 4), "p_push": round(p["p_push"], 4), "ev": round(p["ev"], 4), "dec": round(p["dec"], 4),
                    "fair_margin": _f(fair.get("margin")), "fair_total": _f(fair.get("total")), "fair_p_home": _f(fair.get("p_home"), 4),
                    "market_spread": (books.get("pinnacle") or {}).get("spread"), "market_total": (books.get("pinnacle") or {}).get("total"),
                    "model_margin": _f((model or {}).get("margin")), "model_total": _f((model or {}).get("total")),
                    "locked_at": now.isoformat(), "risk": flat_risk(p["ev"]), "kelly_u": kelly_units(p["p_win"], p["dec"], p["p_push"])})
    return out


def _books(L: pl.DataFrame) -> dict[tuple[int, int, str], dict]:
    out: dict = {}
    for r in L.iter_rows(named=True):
        out.setdefault((r["game_id"], r["gsis_id"], r["market"]), {})[r["book"]] = {
            "line": r["line"], "over": r["over"], "under": r["under"], "p_over": r["p_over"]}
    return out


def mixes_of(x: pl.DataFrame) -> dict[tuple[int, int, str], NP.Mix]:
    return {(r["game_id"], r["athlete_id"], m): NP.Mix.of(r, m) for r in x.iter_rows(named=True) for m in NP.MARKETS}


def prop_picks(x: pl.DataFrame, mixes: dict, books: dict, pricer: NP.PropPricerNba, day: date, now: datetime) -> list[dict]:
    """The best positive-EV prop per tracked player at DraftKings or FanDuel (one pick per player)."""
    out = []
    for r in x.iter_rows(named=True):
        best = None
        for m in NP.MARKETS:
            key = (r["game_id"], r["athlete_id"], m)
            if key not in mixes or key not in books:
                continue
            pk = pricer.price(mixes[key], books[key])
            if pk and pk["ev"] >= PROP_EV_MIN and (best is None or kelly(pk) > kelly(best)):
                best = {**pk, "market": m}
        if not best:
            continue
        side = "Over" if best["side"] == "over" else "Under"
        out.append({
            "kind": "prop", "id": f"{r['game_id']}:{r['athlete_id']}:{best['market']}", "game_id": r["game_id"],
            "day": day.isoformat(), "athlete_id": r["athlete_id"], "name": r["name"], "team": r["team"], "opp": r["opp"],
            "role": r["role"], "status": r["status"], "market": best["market"], "side": best["side"], "line": best["line"],
            "label": f"{r['name']} {side} {best['line']:g} {NP.MARKET_LABEL[best['market']].lower()}",
            "book": best["book"], "book_name": BOOK_NAMES[best["book"]], "dec": round(best["dec"], 4), "american": american(best["dec"]),
            "p_win": round(best["p_win"], 4), "p_push": round(best["p_push"], 4), "ev": round(best["ev"], 4),
            "model_median": round(best["model_median"], 1), "market_median": round(best["market_median"], 1),
            "fair_median": round(best["fair_median"], 1), "anchor": best["anchor"], "exp_min": round(r["exp_min"], 1),
            "locked_at": now.isoformat(), "risk": flat_risk(best["ev"]), "kelly_u": kelly_units(best["p_win"], best["dec"], best["p_push"]),
            # the model as it priced the pick, so its closing chance can be worked out at grading
            "mix": {"exp_min": round(r["exp_min"], 3), "sd_min": round(r["sd_min"], 3),
                    **mixes[(r["game_id"], r["athlete_id"], best["market"])].params()}, "lam": round(best["lam"], 4)})
    return out


def potd(picks: list[dict], n: int = POTD_N) -> list[str]:
    seen, top = set(), []
    for p in sorted((p for p in picks if p["ev"] > 0), key=lambda p: -kelly(p)):
        if p["game_id"] in seen:
            continue
        seen.add(p["game_id"])
        top.append(p["id"])
        if len(top) == n:
            break
    return top


def parlays(picks: list[dict], day: date, prefix: str, now: datetime, want=("golden", "lottery")) -> list[dict]:
    pool = sorted((p for p in picks if p["ev"] > 0), key=lambda p: -kelly(p))[:PARLAY_POOL]
    out = []
    for par in build_parlays(pool):
        if par and par["type"] in want:
            out.append({**par, "kind": "parlay", "id": f"{day.isoformat()}:{prefix}{par['type']}", "day": day.isoformat(),
                        "risk": PARLAY_RISK, "p": round(par["p"], 4), "ev": round(par["ev"], 4), "dec": round(par["dec"], 3),
                        "locked_at": now.isoformat()})
    return out


def with_potd(rows: list[dict]) -> list[dict]:
    ids = {i for r in rows if r.get("kind") == "potd" for i in r["picks"]}
    return [{**r, "potd": r["id"] in ids} if r.get("kind") in ("pick", "prop") else r for r in rows if r.get("kind") != "potd"]


# ---------------------------------------------------------------- closes (grading)

def game_closes(picks: list[dict], tips: dict, fmap: dict, pricer: GamePricer, finals: dict, done: set) -> dict:
    """Each newly gradable pick's closing value: points of line value and EV at Pinnacle's closing fair price."""
    out, cache = {}, {}
    for p in picks:
        if p.get("kind") != "pick" or p["game_id"] not in finals or p["id"] in done:
            continue
        gid = p["game_id"]
        if gid not in cache:
            L = NbaLines(fmap.get(gid), tips[gid])
            close = L.all_at(tips[gid], BOOKS)
            cache[gid] = (close, pricer.fair(close))
        close, fair = cache[gid]
        pin = close.get("pinnacle", {})
        r = {}
        if p["market"] == "spread" and "spread" in pin:
            r["clv"] = round(p["line"] - (-pin["spread"] if p["side"] == "home" else pin["spread"]), 2)
        elif p["market"] == "total" and "total" in pin:
            r["clv"] = round((pin["total"] - p["line"]) if p["side"] == "over" else (p["line"] - pin["total"]), 2)
        ps = pricer.p_side(p, fair)
        if ps:
            r["ev_close"] = round(ps[0] * (p["dec"] - 1) - (1 - ps[0] - ps[1]), 4)
            r["p_close"] = round(ps[0], 4)
        out[p["id"]] = r
    return out


def prop_closes(picks: list[dict], tips: dict, pmap: pl.DataFrame, finished: set, done: set) -> dict:
    todo = [p for p in picks if p.get("kind") == "prop" and p["game_id"] in finished and p["id"] not in done]
    if not todo:
        return {}
    P = load_props(sorted({p["game_id"] for p in todo}), pmap)
    if not P.height:
        return {}
    C = _books(main_lines(P, pl.col("kickoff") - timedelta(minutes=5)))
    pr = NP.PropPricerNba(0.0)
    out = {}
    for p in todo:
        bk = C.get((p["game_id"], p["athlete_id"], p["market"]))
        if not bk:
            continue
        r = {}
        same = bk.get(p["book"])
        if same:
            r["clv"] = round((same["line"] - p["line"]) if p["side"] == "over" else (p["line"] - same["line"]), 2)
        if p.get("mix"):
            mx = p["mix"]
            mix = NP.Mix(NP.minute_draws(mx["exp_min"], mx["sd_min"]), mx["rate"], mx["disp"])
            lam, _ = pr.anchor(mix, bk)
            if lam is not None:
                over, push, under = mix.probs(p["line"], lam)
                pw = over if p["side"] == "over" else under
                r.update(ev_close=round(pw * (p["dec"] - 1) - (1 - pw - push), 4), p_close=round(pw, 4))
        out[p["id"]] = r
    return out


# ---------------------------------------------------------------- page data

def _team_meta(sched: pl.DataFrame, finals: dict) -> dict:
    rec: dict[str, list[int]] = {}
    for g in sched.iter_rows(named=True):
        if g["game_id"] in finals:
            h, a = finals[g["game_id"]]
            for t, won in ((g["home"], h > a), (g["away"], a > h)):
                rec.setdefault(t, [0, 0])[0 if won else 1] += 1
    out = {}
    d = espn._get("teams")
    for t in (x["team"] for s in d["sports"] for lg in s["leagues"] for x in lg["teams"]):
        c = "#" + (t.get("color") or "555555")
        lum = sum(int(c[1 + i:3 + i], 16) * w for i, w in ((0, .299), (2, .587), (4, .114))) / 255
        w, l = rec.get(t["abbreviation"], [0, 0])
        out[t["abbreviation"]] = {"name": t["displayName"], "nick": t.get("shortDisplayName") or t["name"], "color": c,
                                  "ink": "#111111" if lum > 0.6 else "#ffffff", "record": f"{w}-{l}"}
    return out


def _rows(picks: list[dict], results: list[dict]) -> list[dict]:
    res = {r["id"]: r for r in results}
    by_id = {p["id"]: p for p in picks}
    out = []
    for p in picks:
        r = res.get(p["id"], {})
        base = {"day": p["day"], "result": r.get("result"), "ev": p.get("ev"), "ev_close": r.get("ev_close"), "clv": r.get("clv"),
                "american": p.get("american"), "potd": bool(p.get("potd"))}
        if p.get("kind") == "parlay":
            legs = [by_id[l]["label"] for l in p["legs"] if l in by_id]
            out.append({**base, "market": p["type"], "label": "Golden Parlay" if p["type"] == "golden" else "+EV Lottery",
                        "game": f"{len(legs)} legs", "legs": legs, "book": None, "p_win": p.get("p"), "risk": p.get("risk", PARLAY_RISK),
                        "units": round(r["units"], 2) if "units" in r else None})
        else:
            game = f"{p['away']} at {p['home']}" if p.get("kind") == "pick" else f"{p['team']} vs {p['opp']}"
            out.append({**base, "market": p["market"], "label": p["label"], "game": game, "legs": [], "book": p.get("book_name"),
                        "p_win": p["p_win"], "risk": p.get("risk", 1.0), "kelly_u": p.get("kelly_u"), "side": p.get("side"),
                        "with_model": p.get("with_model"), "actual": r.get("actual"),
                        "units": None if r.get("result") is None else round(ledger.units_for(r["result"], p["dec"], 1.0), 2)})
    return out


def _backtest_card() -> dict:
    p = REPORTS / "nba_backtest.parquet"
    if not p.exists():
        return {}
    from nba.backtest import report
    df = pl.read_parquet(p)
    r = report(df)
    lo, hi = int(df["season"].min()), int(df["season"].max())
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} | {"seasons": f"{lo - 1}-{lo % 100:02d} to {hi - 1}-{hi % 100:02d}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--lock-now", action="store_true")
    a = ap.parse_args()
    now = datetime.now(timezone.utc)
    today = now.astimezone(ET).date()

    if not a.offline:
        from nba import data, odds
        data.refresh()
        odds.archive()
    sched = schedule()
    print(f"NBA {today}: {sched.filter(pl.col('day') == today).height} games today")
    if not a.offline:
        n = fetch_boxes(sched, now)
        if n:
            print(f"saved {n} box scores")
    finals, stats = finals_and_stats()

    roster = espn.rosters()
    inj = espn.injuries()
    out_ids = set(inj.filter(pl.col("status").is_in(list(espn.OUT_STATUSES)))["athlete_id"].to_list())
    positions = dict(zip(roster["athlete_id"].to_list(), roster["position"].to_list()))
    hist = NP.history(SEASON, positions)
    prof = NP.profiles(hist, today + timedelta(days=1), SEASON)   # through last night's games

    md, td, dinfo = fit_lines()
    pricer = GamePricer(md, td)
    fidx = fixture_index()
    fmap = dict(zip(fidx["game_id"].to_list(), fidx["fixture_id"].to_list()))
    tips = dict(zip(sched["game_id"].to_list(), sched["tip"].to_list()))

    day_games = [g for g in sched.filter(pl.col("day") == today).iter_rows(named=True)]
    models = game_models(day_games, today, roster, out_ids, prof, sched) if day_games else {}
    lines_now, fair_now = {}, {}
    for g in day_games:
        L = NbaLines(fmap.get(g["game_id"]), g["tip"])
        lines_now[g["game_id"]] = L.all_at(now, BOOKS)
        fair_now[g["game_id"]] = pricer.fair(lines_now[g["game_id"]])

    # --- game lines: lock what is due (today's games past their lock time and not started)
    existing = read(ledger.PICKS)
    locked = {p["game_id"] for p in existing if p.get("kind") == "pick"}
    writes: list[dict] = []
    for g in day_games:
        if g["game_id"] in locked or not _due(lines_lock(g["tip"]), g["tip"], now, a.lock_now):
            continue
        books = lines_now[g["game_id"]]
        if not any(b in books for b in SHOP):
            print(f"no DraftKings/FanDuel lines yet: {g['away']} at {g['home']}")
            continue
        writes += game_picks(g, books, fair_now[g["game_id"]], models.get(g["game_id"]), pricer, now)
    # the day's Picks of the Day and parlays, once every game of the day has locked or started, from games not started
    day_ids = {g["game_id"] for g in day_games}
    locked_now = {p["game_id"] for p in existing + writes if p.get("kind") == "pick"}
    have = {p["id"] for p in existing}
    if day_ids and all(gid in locked_now or tips[gid] <= now for gid in day_ids) and locked_now & day_ids:
        pool = [p for p in existing + writes if p.get("kind") == "pick" and p["game_id"] in day_ids and tips[p["game_id"]] > now]
        if f"{today}:potd" not in have:
            writes.append({"kind": "potd", "id": f"{today}:potd", "day": today.isoformat(), "picks": potd(pool), "at": now.isoformat()})
        want = tuple(t for t in ("golden", "lottery") if f"{today}:{t}" not in have)
        if want:
            writes += parlays(pool, today, "", now, want)
    if writes:
        if a.dry_run:
            print(f"dry run: would lock {sum(w['kind'] == 'pick' for w in writes)} game-line picks and {sum(w['kind'] != 'pick' for w in writes)} daily items")
        else:
            append(ledger.PICKS, writes)
            print(f"game lines: locked {sum(w['kind'] == 'pick' for w in writes)} picks, {[w['id'] for w in writes if w['kind'] != 'pick']}")
    game_rows = existing + (writes if a.dry_run else [])
    if not a.dry_run:
        game_rows = read(ledger.PICKS)

    # --- props: tonight's tracked players, priced against the market
    pmap = player_map(roster.select("athlete_id", "name", "team"))
    games_for_props = [{"game_id": g["game_id"], "home": g["home"], "away": g["away"],
                        "spread": fair_now[g["game_id"]].get("margin", (models.get(g["game_id"]) or {}).get("margin")),
                        "pace_factor": (models.get(g["game_id"]) or {}).get("pace_factor", 1.0)} for g in day_games]
    x = NP.slate(games_for_props, prof, roster, inj, today) if day_games else pl.DataFrame()
    MX = mixes_of(x) if x.height else {}
    P = load_props(sorted(day_ids), pmap) if day_ids else pl.DataFrame()
    now_b = _books(main_lines(P, pl.lit(now))) if P.height else {}
    pp = NP.PropPricerNba(PROPS_K)
    p_existing = read(ledger.PROPS_PICKS)
    p_locked = {p["game_id"] for p in p_existing if p.get("kind") in ("prop", "props_locked")}
    p_writes: list[dict] = []
    due = [g["game_id"] for g in day_games if g["game_id"] not in p_locked and _due(props_lock(g["tip"]), g["tip"], now, a.lock_now)]
    if due and x.height:
        xd = x.filter(pl.col("game_id").is_in(due))
        priced = {gid for (gid, _, _) in now_b}
        no_lines = [gid for gid in due if gid not in priced]
        if no_lines:
            print(f"props: no lines yet for {no_lines}; they lock at the next build")
        p_writes += prop_picks(xd.filter(~pl.col("game_id").is_in(no_lines)), MX, now_b, pp, today, now)
        # a game whose props were priced locks even with no pick (nothing cleared the bar): a marker, so it is final
        p_writes += [{"kind": "props_locked", "id": f"{gid}:props", "game_id": gid, "day": today.isoformat(), "at": now.isoformat()}
                     for gid in due if gid not in no_lines]
    p_have = {p["id"] for p in p_existing}
    p_locked_now = {p["game_id"] for p in p_existing + p_writes if p.get("kind") in ("prop", "props_locked")}
    if day_ids and all(gid in p_locked_now or tips[gid] <= now for gid in day_ids) and p_locked_now & day_ids:
        pool = [p for p in p_existing + p_writes if p.get("kind") == "prop" and p["game_id"] in day_ids and tips[p["game_id"]] > now]
        if f"{today}:props:potd" not in p_have:
            p_writes.append({"kind": "potd", "id": f"{today}:props:potd", "day": today.isoformat(), "picks": potd(pool), "at": now.isoformat()})
        want = tuple(t for t in ("golden", "lottery") if f"{today}:props:{t}" not in p_have)
        if want:
            p_writes += parlays(pool, today, "props:", now, want)
    if p_writes:
        if a.dry_run:
            print(f"props dry run: would lock {sum(w['kind'] == 'prop' for w in p_writes)} props, {[w['id'] for w in p_writes if w['kind'] not in ('prop', 'props_locked')]}")
        else:
            append(ledger.PROPS_PICKS, p_writes)
            print(f"props: locked {sum(w['kind'] == 'prop' for w in p_writes)} picks, {[w['id'] for w in p_writes if w['kind'] not in ('prop', 'props_locked')]}")
    prop_rows = p_existing + (p_writes if a.dry_run else [])
    if not a.dry_run:
        prop_rows = read(ledger.PROPS_PICKS)

    # --- grade
    game_rows, prop_rows = with_potd(game_rows), with_potd([r for r in prop_rows if r.get("kind") != "props_locked"])
    g_done = {r["id"] for r in read(ledger.RESULTS)}
    p_done = {r["id"] for r in read(ledger.PROPS_RESULTS)}
    g_res = ledger.grade_games(game_rows, finals, game_closes(game_rows, tips, fmap, pricer, finals, g_done))
    p_res = ledger.grade_props(prop_rows, stats, set(finals), prop_closes(prop_rows, tips, pmap, set(finals), p_done))
    if a.dry_run:
        g_results = read(ledger.RESULTS) + [r for r in g_res if r["id"] not in g_done]
        p_results = read(ledger.PROPS_RESULTS) + [r for r in p_res if r["id"] not in p_done]
    else:
        n1, n2 = ledger.append_new(ledger.RESULTS, g_res), ledger.append_new(ledger.PROPS_RESULTS, p_res)
        if n1 or n2:
            print(f"graded {len(n1)} game-line picks, {len(n2)} props")
        g_results, p_results = read(ledger.RESULTS), read(ledger.PROPS_RESULTS)

    # --- page data: today's games
    g_by = {}
    gres = {r["id"]: r for r in g_results}
    for p in game_rows:
        if p.get("kind") == "pick":
            g_by.setdefault(p["game_id"], {})[p["market"]] = {**p, **({"result": gres[p["id"]]["result"]} if p["id"] in gres else {})}
    status = {}
    try:
        for e in espn.scoreboard(today.strftime("%Y%m%d")):
            status[e["game_id"]] = e
    except RuntimeError:
        pass
    outs_by_team: dict[str, list] = {}
    for r in inj.iter_rows(named=True):
        outs_by_team.setdefault(r["team"], []).append({"name": r["name"], "status": r["status"], "detail": r["detail"]})
    cards = []
    for g in day_games:
        gid = g["game_id"]
        books = lines_now[gid]
        fair, m = fair_now[gid], models.get(gid)
        st = status.get(gid, {})
        fin = gid in finals
        cards.append({
            "id": gid, "tip": _tip(g["tip"]), "home": g["home"], "away": g["away"],
            "status": "final" if fin else ("live" if st.get("state") == "in" or g["tip"] <= now else "upcoming"),
            "detail": st.get("detail"),
            "score": {"home": finals[gid][0], "away": finals[gid][1]} if fin else (
                {"home": st["home_score"], "away": st["away_score"]} if st.get("state") == "in" else None),
            "fair": {"margin": _f(fair.get("margin"), 1), "total": _f(fair.get("total"), 1), "p_home": _f(fair.get("p_home"), 3),
                     "src": fair.get("margin_src")},
            "model": {"margin": _f(m["margin"], 1), "total": _f(m["total"], 1), "team": _f(m["team_margin"], 1),
                      "b2b": _f(m["b2b"], 1), "lineup": _f(m["lineup"], 1), "pace": _f(m["pace"], 1), "rest": m["rest"]} if m else None,
            "books": [{"book": BOOK_NAMES.get(b, b), "sharp": b == "pinnacle", "spread": r.get("spread"),
                       "spread_home": american(r["spread_price_a"]) if "spread_price_a" in r else None,
                       "spread_away": american(r["spread_price_b"]) if "spread_price_b" in r else None,
                       "total": r.get("total"), "over": american(r["total_price_a"]) if "total_price_a" in r else None,
                       "under": american(r["total_price_b"]) if "total_price_b" in r else None,
                       "ml_home": american(r["ml_home"]) if "ml_home" in r else None,
                       "ml_away": american(r["ml_away"]) if "ml_away" in r else None}
                      for b in SHOP + ["pinnacle"] if (r := books.get(b))],
            "picks": g_by.get(gid, {}),
            "locks": {"lines": lines_lock(g["tip"]).isoformat(), "props": props_lock(g["tip"]).isoformat()},
            "injuries": {"home": outs_by_team.get(g["home"], []), "away": outs_by_team.get(g["away"], [])},
        })

    # --- page data: today's tracked players
    pres = {r["id"]: r for r in p_results}
    p_by = {(p["game_id"], p["athlete_id"]): {**{k: v for k, v in p.items() if k != "mix"},
                                              **({"result": pres[p["id"]]["result"], "actual": pres[p["id"]].get("actual")} if p["id"] in pres else {})}
            for p in prop_rows if p.get("kind") == "prop" and p["day"] == today.isoformat()}
    players = []
    if x.height:
        role_order = {"Starter": 0, "Sixth man": 1}
        for r in x.sort("game_id", "team", pl.col("role").replace_strict(role_order, return_dtype=pl.Int8), pl.col("exp_min"), descending=[False, False, False, True]).iter_rows(named=True):
            mk = []
            for m in NP.MARKETS:
                key = (r["game_id"], r["athlete_id"], m)
                mix = MX.get(key)
                if mix is None:
                    continue
                bk = now_b.get(key) or {}
                priced = pp.price(mix, bk) if bk else None
                q = mix.quantiles()
                mk.append({"market": m, "median": q[2], "q": q, "mean": round(mix.mean(), 1),
                           "books": {BOOK_NAMES[b]: {"line": v["line"], "over": american(v["over"]), "under": american(v["under"])} for b, v in bk.items()},
                           "fair": _f(priced["fair_median"], 1) if priced else None,
                           "lean": ({"side": priced["side"], "line": priced["line"], "book": BOOK_NAMES[priced["book"]],
                                     "american": american(priced["dec"]), "p": round(priced["p_win"], 3), "ev": round(priced["ev"], 4)} if priced else None),
                           "actual": stats.get((r["game_id"], r["athlete_id"]), {}).get(m)})
            players.append({"id": r["athlete_id"], "name": r["name"], "team": r["team"], "opp": r["opp"], "game_id": r["game_id"],
                            "role": r["role"], "status": r["status"], "min": round(r["exp_min"], 1), "games_now": r["games_now"],
                            "markets": mk, "pick": p_by.get((r["game_id"], r["athlete_id"]))})

    def daily(rows, results, prefix):
        res = {r["id"]: r for r in results}
        by_id = {p["id"]: p for p in rows}
        out = {}
        for t in ("golden", "lottery"):
            p = by_id.get(f"{today}:{prefix}{t}")
            if p:
                out[t] = {**p, "leg_picks": [{k: v for k, v in by_id[l].items() if k != "mix"} for l in p["legs"] if l in by_id],
                          **({"result": res[p["id"]]["result"]} if p["id"] in res else {})}
        return out

    live_g = [p for p in game_rows if p.get("kind") in ("pick", "parlay")]
    live_p = [p for p in prop_rows if p.get("kind") in ("prop", "parlay")]
    data = {
        "meta": {"season": SEASON, "label": f"{SEASON - 1}-{str(SEASON)[2:]}", "day": today.isoformat(),
                 "day_label": now.astimezone(ET).strftime("%A, %B ") + str(today.day), "generated": now.isoformat(),
                 "books": [BOOK_NAMES[b] for b in SHOP], "games_today": len(day_games),
                 "slots": {"morning": "10:30am ET", "evening": "5:30pm ET"}, "props_k": PROPS_K,
                 "checkup_first": "2026-11-02", "dist": dinfo},
        "teams": _team_meta(sched, finals),
        "games": cards,
        "parlays": daily(game_rows, g_results, ""),
        "record": {"live": ledger.record_games(live_g, g_results), "rows": _rows(live_g, g_results)},
        "props": {"players": players, "parlays": daily(prop_rows, p_results, "props:"),
                  "record": {"live": ledger.record_props(live_p, p_results), "rows": _rows(live_p, p_results)},
                  "locked": sorted(p_locked_now)},
        "model": {"backtest": _backtest_card()},
    }
    # the page is written on a dry run too (it shows what would lock), only the ledgers are left alone
    OUT.write_text(json.dumps(_clean(data), default=str, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    from board.render import render
    html = render()
    print(f"{'dry run: ' if a.dry_run else ''}wrote {OUT.name} and the board ({len(html) / 1024:.0f} KB): {len(cards)} games, "
          f"{len(players)} tracked players, {len(MX)} markets modeled")


if __name__ == "__main__":
    main()
