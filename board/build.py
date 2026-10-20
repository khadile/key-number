"""Build the weekly board.

    python -m board.build             # refresh data and lines, lock this week's picks, grade, render
    python -m board.build --offline   # no downloads: rebuild from what is on disk

Writes board/nfl_data.json and renders board/nfl_board.html (board/render.py), which holds the NBA side too.
"""
import argparse
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from backtest.run import TRAIN_FIRST, load_params
from board import defense, detail, ledger
from board.checks import game_checks, qb_doubts
from board.market import (BOOK_NAMES, Lines, Pricer, american, build_parlays, fixture_map,
                          picks_of_the_week, reference)
from board.weather import STADIUMS, forecast
from config import PARLAY_RISK, RAW, REPORTS, ROOT, STRAIGHT_RISK
from model import checks as C
from model import features, project
from model.probs import MARGIN_RANGE, TOTAL_RANGE, KeyDist, WinCurve
from model.ratings import load_inputs
from model.staking import flat_risk, kelly_units
from pipeline.odds_history import SHOP_BOOKS, archive, publish_time
from props.pricing import lock_time

LIVE_START_WEEK = 5  # first week whose picks were locked live; earlier 2026 weeks are simulated
TEMPLATE = ROOT / "board" / "template.html"
OUT = ROOT / "board" / "nfl_board.html"
ET = ZoneInfo("America/New_York")


def _f(x, nd=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def _clean(x):
    """NaN and infinity become null: JSON has no NaN, and the page's JSON.parse would fail on it."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _team_meta(teams: list[str], games: pl.DataFrame, season: int) -> dict:
    t = pl.read_parquet(RAW / "teams.parquet").filter(pl.col("team_abbr").is_in(teams))
    played = games.filter((pl.col("season") == season) & pl.col("result").is_not_null() & (pl.col("game_type") == "REG"))
    rec = {k: [0, 0, 0] for k in teams}
    for g in played.iter_rows(named=True):
        for team, m in ((g["home_team"], g["result"]), (g["away_team"], -g["result"])):
            rec[team][0 if m > 0 else (1 if m < 0 else 2)] += 1
    out = {}
    for r in t.iter_rows(named=True):
        c = r["team_color"] or "#555555"
        lum = sum(int(c.lstrip("#")[i:i + 2], 16) * w for i, w in ((0, .299), (2, .587), (4, .114))) / 255
        w, l, tie = rec[r["team_abbr"]]
        out[r["team_abbr"]] = {"name": r["team_name"], "nick": r["team_nick"], "color": c,
                               "ink": "#111111" if lum > 0.6 else "#ffffff",
                               "record": f"{w}-{l}" + (f"-{tie}" if tie else "")}
    return out


def _stake(p: dict) -> dict:
    """What a pick risks. Picks locked before 2026-10-15 carry no stake fields: they staked 1u flat (a parlay
    0.5u), negative EV included, except week 6's, made leans by an appended stake line before kickoff
    (ledger.read_picks). Their quarter-Kelly size is worked out from the locked price and chance."""
    if p.get("kind") == "parlay":
        return {"risk": p.get("risk", PARLAY_RISK), "kelly_u": None}
    k = p.get("kelly_u")
    return {"risk": p.get("risk", STRAIGHT_RISK),
            "kelly_u": k if k is not None else kelly_units(p.get("p_win"), p["dec"], p.get("p_push") or 0.0)}


def _pick_rows(picks: list[dict], results: list[dict], verdict_by: dict | None = None) -> list[dict]:
    """Every pick, graded or still pending, for the record page's week browser.
    `market` is spread / total / ml, or golden / lottery for the two parlays."""
    res = {r["id"]: r for r in results}
    vb = verdict_by or {}
    by_id = {p["id"]: p for p in picks}
    out = []
    for p in picks:
        r = res.get(p["id"], {})
        if p.get("kind") == "parlay":
            market = p.get("type", "golden")
            label = "+EV Lottery" if market == "lottery" else "Golden Parlay"
            legs = [by_id[l]["label"] for l in p["legs"] if l in by_id]
            game = f"{len(legs)} legs"
        else:
            market, label, legs = p["market"], p["label"], []
            _, _, away, home = p["game_id"].split("_")
            game = f"{away} at {home}"
        out.append({"week": p["week"], "market": market, "label": label, "game": game, "legs": legs,
                    "american": p.get("american"), "book": p.get("book_name"), "potw": bool(p.get("potw")),
                    **_stake(p), **vb.get(p["id"], {}),
                    "p_win": p.get("p_win") if p.get("kind") != "parlay" else p.get("p"),
                    "ev": p.get("ev"), "result": r.get("result"),
                    # straight picks at 1u in the record, leans included (board/ledger.record)
                    "units": (None if "result" not in r else round(r["units"], 2) if p.get("kind") == "parlay"
                              else round(ledger.units_for(r["result"], p["dec"], STRAIGHT_RISK), 2)),
                    "clv": r.get("clv")})
    return out


def _kick(dt: datetime) -> dict:
    et = dt.astimezone(ET)
    return {"iso": dt.isoformat(), "day": et.strftime("%A, %B ") + str(et.day),
            "time": et.strftime("%I:%M %p").lstrip("0") + " ET", "short": et.strftime("%a ") + et.strftime("%I:%M %p").lstrip("0")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="no downloads; rebuild from data on disk")
    ap.add_argument("--dry-run", action="store_true", help="compute picks and grades but write nothing to the ledger")
    ap.add_argument("--lock-now", action="store_true",
                    help="the day's update run early: lock now what its 10am build would lock (props due later today)")
    ap.add_argument("--sim-lock", choices=("tuesday", "last", "last-info"), default="tuesday",
                    help="replay of weeks 1-4: lock game lines at Tuesday's prices (default), at the last board "
                         "update before each game, or there with that later information")
    args = ap.parse_args()
    now = datetime.now(timezone.utc)

    if not args.offline:
        from pipeline import clean, ingest
        print("refreshing nflverse current season ...")
        ingest.ingest(only=None, refresh=False)
        clean.main()
        from props import data as props_data
        props_data.build()
        from pipeline import prop_players   # OddsPapi player ids -> gsis (rebuilt here so a fresh checkout has it)
        prop_players.main()
        print("archiving lines and props ...")
        archive()

    inp, games = load_inputs()
    season = int(games.filter(pl.col("result").is_null())["season"].min())
    upcoming = games.filter((pl.col("season") == season) & pl.col("result").is_null())
    week = int(upcoming["week"].min())
    print(f"season {season}, week {week}")

    # --- weather forecasts for this week's outdoor games, fed to the model
    stadium = dict(pl.read_parquet(RAW / "schedules.parquet").select("game_id", "stadium").iter_rows())
    wx = {}
    for g in upcoming.filter(pl.col("week") == week).iter_rows(named=True):
        if not g["dome"] and stadium.get(g["game_id"]) in STADIUMS:
            f = forecast(stadium[g["game_id"]], g["kickoff_utc"])
            if f:
                wx[g["game_id"]] = f
    if wx:
        w = pl.DataFrame([{"game_id": k, "wx_temp": float(v["temp"]), "wx_wind": float(v["wind"])} for k, v in wx.items()])
        games = games.join(w, on="game_id", how="left").with_columns(
            temp=pl.coalesce("wx_temp", "temp"), wind=pl.coalesce("wx_wind", "wind")).drop("wx_temp", "wx_wind")

    # --- model
    p, alpha = load_params()
    rows_live, snaps, qbm = features.build(inp, games, p)
    rows_pub = features.build(inp, games, p, as_of_publish=True)[0]
    pts_model = project.fit(rows_live.filter(pl.col("season").is_between(TRAIN_FIRST, season - 1)), alpha)
    proj_live = project.game_projections(rows_live.filter(pl.col("season") == season), pts_model)
    proj_pub = project.game_projections(rows_pub.filter(pl.col("season") == season), pts_model)
    live_by = {r["game_id"]: r for r in proj_live.iter_rows(named=True)}
    pub_by = {r["game_id"]: r for r in proj_pub.iter_rows(named=True)}

    hist = games.filter((pl.col("season") < season) & pl.col("result").is_not_null() & pl.col("spread_line").is_not_null())
    sm = float((hist["result"] - hist["spread_line"]).std())
    st = float((hist["total"] - hist["total_line"]).std())
    mdist = KeyDist.fit(hist["result"].to_numpy(), hist["spread_line"].to_numpy(), MARGIN_RANGE, sm, True)
    tdist = KeyDist.fit(hist["total"].to_numpy(), hist["total_line"].to_numpy(), TOTAL_RANGE, st, False)
    wins = WinCurve.fit(hist["spread_line"].to_numpy(), hist["result"].to_numpy())

    # --- lines: one cached timeline per game
    fmap = fixture_map()
    season_games = games.filter(pl.col("season") == season).sort("kickoff_utc")
    lines_cache: dict[str, Lines] = {}

    def lines_for(gid: str, kickoff: datetime) -> Lines:
        if gid not in lines_cache:
            lines_cache[gid] = Lines(fmap.get(gid), kickoff)
        return lines_cache[gid]

    finished = season_games.filter(pl.col("result").is_not_null())
    finals = {g["game_id"]: (g["home_score"], g["away_score"]) for g in finished.iter_rows(named=True)}

    # --- how far lines move toward the model between publish and kickoff (sets k)
    study = []
    closes: dict[str, dict] = {}
    for g in finished.iter_rows(named=True):
        L = lines_for(g["game_id"], g["kickoff_utc"])
        fs = L.first_seen("pinnacle")
        close = L.at("pinnacle", g["kickoff_utc"]) if fs else {}
        closes[g["game_id"]] = close or {"spread": g["spread_line"], "total": g["total_line"]}
        if not fs or g["game_id"] not in pub_by:
            continue
        t_pub = max(publish_time(g["kickoff_utc"]), fs + timedelta(hours=1))
        a = L.at("pinnacle", t_pub)
        if "spread" in a and "spread" in close and "total" in a and "total" in close:
            m = pub_by[g["game_id"]]
            study.append({"gap_s": m["model_margin"] - a["spread"], "move_s": close["spread"] - a["spread"],
                          "gap_t": m["model_total"] - a["total"], "move_t": close["total"] - a["total"]})
    S = pl.DataFrame(study)

    def slope(gap, move):
        return float(np.clip((S[gap] * S[move]).sum() / (S[gap] ** 2).sum(), 0, 0.5)) if S.height else 0.0

    k_s, k_t = slope("gap_s", "move_s"), slope("gap_t", "move_t")

    def toward(gap, move):
        moved = S.filter(pl.col(move) != 0)
        t = int((np.sign(moved[gap]) == np.sign(moved[move])).sum())
        clv = float((np.sign(S[gap]) * S[move]).mean())
        return {"moved": moved.height, "toward": t, "away": moved.height - t, "clv": round(clv, 2)}

    clv_study = {"games": S.height, "k_spread": round(k_s, 3), "k_total": round(k_t, 3),
                 "spread": toward("gap_s", "move_s") if S.height else None,
                 "total": toward("gap_t", "move_t") if S.height else None}
    print(f"k spread {k_s:.3f}, k total {k_t:.3f} from {S.height} games")
    pricer = Pricer(mdist, tdist, wins, k_s, k_t)

    def make_picks(week_games: pl.DataFrame, when_fn, proj_by: dict, flag_potw: bool) -> list[dict]:
        out = []
        for g in week_games.iter_rows(named=True):
            gid = g["game_id"]
            if gid not in proj_by:
                continue
            L = lines_for(gid, g["kickoff_utc"])
            when = when_fn(g, L)
            books = L.all_at(when, ["pinnacle"] + SHOP_BOOKS)
            ref = reference(books, {"spread": g["spread_line"], "total": g["total_line"]})
            shop = [b for b in SHOP_BOOKS if b in books]
            if not shop:
                continue
            m = proj_by[gid]
            priced = pricer.price(g, books, m["model_margin"], m["model_total"], ref, shop)
            for market in ("spread", "total", "ml"):
                if market in priced:
                    pk = priced[market]
                    out.append({**pk, "kind": "pick", "id": f"{gid}:{market}", "game_id": gid,
                                "season": int(g["season"]), "week": int(g["week"]),
                                "book_name": BOOK_NAMES.get(pk["book"], pk["book"]),
                                "p_win": round(pk["p_win"], 4), "p_push": round(pk["p_push"], 4),
                                "ev": round(pk["ev"], 4), "dec": round(pk["dec"], 4),
                                "model_margin": round(m["model_margin"], 2), "model_total": round(m["model_total"], 2),
                                "market_spread": ref.get("spread"), "market_total": ref.get("total"),
                                "locked_at": when.isoformat(), "potw": False,
                                # a negative-EV side is a lean (no units); quarter Kelly tracked beside flat
                                "risk": flat_risk(pk["ev"]), "kelly_u": kelly_units(pk["p_win"], pk["dec"], pk["p_push"])})
        if flag_potw:
            top = picks_of_the_week(out)
            ids = {p["id"] for p in top}
            for p in out:
                p["potw"] = p["id"] in ids
        return out

    def week_parlays(picks: list[dict], season_: int, week_: int, want=("golden", "lottery"),
                     locked_at: str | None = None) -> list[dict]:
        """The week's Golden Parlay (+150 to +300) and +EV Lottery (over +500), built
        from its positive-EV picks at their locked prices."""
        out = []
        for par in build_parlays([p for p in picks if p.get("kind") == "pick"]):
            if par and par["type"] in want:
                out.append({**par, "kind": "parlay", "id": f"{season_}_{week_:02d}:{par['type']}",
                            "season": season_, "week": week_, "risk": PARLAY_RISK,
                            "p": round(par["p"], 4), "ev": round(par["ev"], 4), "dec": round(par["dec"], 3),
                            "locked_at": locked_at or max(p["locked_at"] for p in picks if p["id"] in par["legs"])})
        return out

    # --- pre-launch weeks: simulated at Tuesday lines with Tuesday information
    def at_publish(g, L):
        fs = L.first_seen("pinnacle") or L.first_seen("draftkings") or g["kickoff_utc"]
        return max(publish_time(g["kickoff_utc"]), fs + timedelta(hours=1))

    def at_last_update(g, L):   # Thursday 10am ET for a Thursday game, Saturday 10am ET otherwise
        fs = L.first_seen("pinnacle") or L.first_seen("draftkings") or g["kickoff_utc"]
        return max(lock_time(g["kickoff_utc"]), fs + timedelta(hours=1))

    sim_when, sim_proj = {"tuesday": (at_publish, pub_by), "last": (at_last_update, pub_by),
                          "last-info": (at_last_update, live_by)}[args.sim_lock]
    sim = []
    for wk in sorted(finished.filter(pl.col("week") < LIVE_START_WEEK)["week"].unique()):
        wp = make_picks(finished.filter(pl.col("week") == wk), sim_when, sim_proj, True)
        for p in wp:   # every pick staked 1u before week 6's lean rule (model/staking.py), and the replay stays that way
            p["risk"] = STRAIGHT_RISK
        sim += wp + week_parlays(wp, season, wk)
    sim_results = ledger.grade_all(sim, finals, closes)
    pre = ledger.record(sim, sim_results)

    # --- live: lock this week's picks (games not yet started), then its parlays, then grade.
    # A dry run computes the same picks and grades but writes nothing to the ledger.
    existing = ledger.read_picks()
    kickoff_of = dict(zip(season_games["game_id"], season_games["kickoff_utc"]))
    this_week = upcoming.filter((pl.col("week") == week) & (pl.col("kickoff_utc") > now))
    week_locked = {p["game_id"] for p in existing
                   if p["season"] == season and p.get("week") == week and p.get("kind") == "pick"}
    todo = this_week.filter(~pl.col("game_id").is_in(list(week_locked)))
    new_locks: list[dict] = []
    if week >= LIVE_START_WEEK:
        if todo.height:
            new_locks += make_picks(todo, lambda g, L: now, live_by, not week_locked)
        # Parlays the week does not have yet, built from locked picks whose games have not started.
        week_picks = [p for p in existing + new_locks if p.get("kind") == "pick"
                      and p["season"] == season and p["week"] == week and kickoff_of[p["game_id"]] > now]
        have = {p["type"] for p in existing if p.get("kind") == "parlay" and p["season"] == season and p["week"] == week}
        # One-time migration: the week-5 ticket locked before the Golden Parlay was
        # redefined (+150 to +300) is that week's +EV Lottery. Recorded, not rewritten.
        legacy = [p for p in existing if p.get("kind") == "parlay" and p["id"].endswith(":parlay")
                  and p["season"] == season and p["week"] == week and p.get("type") == "golden"]
        migrations = [{"kind": "parlay_type", "id": p["id"], "type": "lottery", "at": now.isoformat(),
                       "note": "Golden Parlay redefined on 2026-10-08 as +150 to +300; this +1423 ticket is the week's +EV Lottery"}
                      for p in legacy]
        if migrations:
            have = (have - {"golden"}) | {"lottery"}
        missing = tuple(t for t in ("golden", "lottery") if t not in have)
        new_parlays = week_parlays(week_picks, season, week, want=missing,
                                   locked_at=now.isoformat() if week_locked else None) if missing else []
        writes = migrations + new_locks + new_parlays
        if writes:
            if args.dry_run:
                print(f"dry run: would write {len(writes)} ledger lines for week {week}: "
                      f"{len(new_locks)} picks, {[p['type'] for p in new_parlays]}, {len(migrations)} reclassified")
            else:
                ledger.append(ledger.PICKS, writes)
                print(f"week {week}: locked {len(new_locks)} picks, parlays {[p['type'] for p in new_parlays]}, "
                      f"reclassified {len(migrations)}")
    else:
        migrations, new_parlays = [], []
    if args.dry_run:
        mig = {m["id"] for m in migrations}
        all_picks = [{**p, "type": "lottery"} if p["id"] in mig else p for p in existing] + new_locks + new_parlays
        done = {r["id"] for r in ledger.read(ledger.RESULTS)}
        live_results = ledger.read(ledger.RESULTS) + [r for r in ledger.grade_all(all_picks, finals, closes)
                                                      if r["id"] not in done]
    else:
        graded = ledger.grade_ledger(finals, closes)
        if graded:
            print(f"graded {len(graded)} picks")
        all_picks = ledger.read_picks()
        live_results = ledger.read(ledger.RESULTS)
    live_picks = [p for p in all_picks if p["season"] == season]
    live = ledger.record(live_picks, live_results)

    # --- game cards and breakdowns for this week
    meta = _team_meta(inp.teams, games, season)
    snap = snaps[(season, week)]
    profiles = detail.team_profiles(snap, inp.teams)
    qr = qbm.ratings(season, week)
    injuries = detail.injuries(season, week)
    last17 = defense.last_n_window(season)
    roles = defense.role_table(season)
    res_by = {r["id"]: r for r in live_results}
    picks_by = {}
    for pk in live_picks:
        if pk.get("kind") == "pick":
            picks_by.setdefault(pk["game_id"], {})[pk["market"]] = {**pk, **_stake(pk), **({"result": res_by[pk["id"]]["result"]} if pk["id"] in res_by else {})}
    # --- is each +EV pick also a good pick? (model/checks.py) A game's verdict is the one frozen before its
    # kickoff when there is one, else rebuilt from what was known at the last update before it; this week's
    # upcoming picks are checked as of now, with the price now.
    kick_of = dict(zip(season_games["game_id"], season_games["kickoff_utc"]))
    game_of = {g["game_id"]: g for g in season_games.iter_rows(named=True)}
    frozen, qbs = C.frozen(kick_of), {}
    def checks_for(pk: dict, model: dict | None = None) -> dict:
        g, k = game_of[pk["game_id"]], kick_of[pk["game_id"]]
        if g["week"] not in qbs:
            qbs[g["week"]] = qb_doubts(season, g["week"])
        return game_checks(pk, g, lines_for(pk["game_id"], k), pricer, now if now < k else lock_time(k), qbs[g["week"]], model)
    verdict_by: dict[str, dict] = {}
    for pk in sim + live_picks:
        if pk.get("kind") != "pick" or pk["ev"] <= 0 or _stake(pk)["risk"] <= 0:
            continue   # leans and parlays get no verdict
        k = kick_of[pk["game_id"]]
        f = frozen.get(pk["id"]) if k <= now else None
        shown = picks_by.get(pk["game_id"], {}).get(pk["market"], {}).get("id") == pk["id"]   # this week's cards
        c = checks_for(pk, live_by.get(pk["game_id"]) if shown and k > now else None) if shown or not f else None
        verdict_by[pk["id"]] = ({"verdict": f["verdict"], "failed": f["failed"]} if f
                                else {"verdict": c["verdict"], "failed": C.failed(c["checks"])})
        if shown:
            picks_by[pk["game_id"]][pk["market"]]["checks"] = c
    upcoming = [pk for pk in live_picks if pk["id"] in verdict_by and kick_of[pk["game_id"]] > now]
    if upcoming and not args.dry_run:
        n = C.freeze([{"kind": "game", "id": pk["id"], "game_id": pk["game_id"], **verdict_by[pk["id"]]} for pk in upcoming], now)
        if n:
            print(f"froze {n} verdicts")

    parlays = {}
    for p in live_picks:
        if p.get("kind") == "parlay" and p["week"] == week:
            parlays[p["type"]] = {**p, "leg_picks": [next(x for x in live_picks if x["id"] == l) for l in p["legs"]],
                                  **({"result": res_by[p["id"]]["result"]} if p["id"] in res_by else {})}

    cards = []
    week_games = games.filter((pl.col("season") == season) & (pl.col("week") == week)).sort("kickoff_utc", "game_id")
    for g in week_games.iter_rows(named=True):
        gid, home, away = g["game_id"], g["home_team"], g["away_team"]
        m = live_by.get(gid)
        L = lines_for(gid, g["kickoff_utc"])
        books = L.all_at(now, ["pinnacle"] + SHOP_BOOKS)
        ref = reference(books, {"spread": g["spread_line"], "total": g["total_line"]})
        fs = L.first_seen("pinnacle")
        opening = L.at("pinnacle", fs + timedelta(hours=1)) if fs else {}
        rows_g = rows_live.filter(pl.col("game_id") == gid)
        hrow = rows_g.filter(pl.col("is_home") == 1).to_dicts()[0]
        arow = rows_g.filter(pl.col("is_home") == 0).to_dicts()[0]
        contrib = pts_model.contributions(hrow, arow)
        book_rows = []
        for b in SHOP_BOOKS + ["pinnacle"]:
            r = books.get(b)
            if not r:
                continue
            book_rows.append({
                "book": BOOK_NAMES.get(b, b), "sharp": b == "pinnacle",
                "spread": r.get("spread"),
                "spread_home": american(r["spread_price_a"]) if "spread_price_a" in r else None,
                "spread_away": american(r["spread_price_b"]) if "spread_price_b" in r else None,
                "total": r.get("total"),
                "over": american(r["total_price_a"]) if "total_price_a" in r else None,
                "under": american(r["total_price_b"]) if "total_price_b" in r else None,
                "ml_home": american(r["ml_home"]) if "ml_home" in r else None,
                "ml_away": american(r["ml_away"]) if "ml_away" in r else None,
            })
        us_ml = [(american(r["ml_home"]), american(r["ml_away"])) for b, r in books.items() if b in SHOP_BOOKS and "ml_home" in r]
        # The model's moneyline view for every game, so a game with no pick reads as a deliberate pass:
        # its side (same as the spread side), the fair price, and the best price available.
        ml_view = None
        if m and ref.get("spread") is not None:
            side = "home" if m["model_margin"] > ref["spread"] else "away"
            p_home = wins.home(pricer.expected_margin(m["model_margin"], ref["spread"]))
            p_side = p_home if side == "home" else 1 - p_home
            key = "ml_home" if side == "home" else "ml_away"
            offers = [(r[key], b) for b, r in books.items() if b in SHOP_BOOKS and key in r]
            best = max(offers) if offers else None
            ml_view = {"team": home if side == "home" else away, "fair": american(1 / p_side),
                       "best": american(best[0]) if best else None,
                       "book": BOOK_NAMES.get(best[1], best[1]) if best else None}
        is_final = g["result"] is not None
        cards.append({
            "id": gid, "week": week, "kickoff": _kick(g["kickoff_utc"]),
            "home": home, "away": away, "venue": stadium.get(gid), "dome": bool(g["dome"]),
            "neutral": bool(g["neutral_site"]), "weather": wx.get(gid),
            "status": "final" if is_final else ("live" if g["kickoff_utc"] <= now else "upcoming"),
            "score": {"home": g["home_score"], "away": g["away_score"]} if is_final else None,
            "proj": {"home": _f(m["proj_home"], 1), "away": _f(m["proj_away"], 1)} if m else None,
            "model": {"spread": _f(m["model_margin"], 1), "total": _f(m["model_total"], 1)} if m else None,
            "market": {"spread": ref.get("spread"), "total": ref.get("total"),
                       "open_spread": opening.get("spread"), "open_total": opening.get("total"),
                       "ml_home": int(np.median([x[0] for x in us_ml])) if us_ml else g["home_moneyline"],
                       "ml_away": int(np.median([x[1] for x in us_ml])) if us_ml else g["away_moneyline"]},
            "books": book_rows,
            "ml_view": ml_view,
            "picks": picks_by.get(gid, {}),
            "contrib": [{"label": k, "pts": round(v, 2)} for k, v in contrib.items() if abs(v) >= 0.05],
            "qb": {side: {"name": g[f"{side}_qb_name"], "rating": _f(qbm.rating_of(qr, g[f"{side}_qb_id"]), 3),
                          "adj": _f((hrow if side == "home" else arow)["qb_adj"], 3)} for side in ("home", "away")},
            "profiles": {"home": profiles[home], "away": profiles[away]},
            "defense": {"home": last17.get(home), "away": last17.get(away)},
            "roles": {"home": roles.get(home, []), "away": roles.get(away, [])},
            "injuries": {"home": injuries.get(home, []), "away": injuries.get(away, [])},
            "movement": L.movement("pinnacle", fs, now) if fs else [],
        })

    # --- player props: same lifecycle, its own ledger (props/section.py)
    from props.section import build_section
    market_lines = {c["id"]: {"spread": c["market"]["spread"], "total": c["market"]["total"]} for c in cards}
    props_data_out = build_section(season, week, games, now, market_lines, args.dry_run, args.lock_now)

    # --- live betting: the engine runs on the laptop; the board shows its record (live/board.py)
    from live.board import board_data as live_board
    live_out = live_board(season)

    # --- defense tab
    def_rows, def_league = defense.season_table(season)

    # --- how it works: backtest headline numbers, computed from saved reports
    model_card = {"clv": clv_study, "sigma_margin": round(sm, 1), "sigma_total": round(st, 1)}
    proj_path = REPORTS / "projections.parquet"
    if proj_path.exists():
        bt = pl.read_parquet(proj_path).filter(pl.col("season").is_between(2022, season - 1) & pl.col("result").is_not_null())
        model_card["backtest"] = {
            "games": bt.height, "seasons": f"2022-{season - 1}",
            "margin_model": _f((bt["model_margin"] - bt["result"]).abs().mean()),
            "margin_close": _f((bt["spread_line"] - bt["result"]).abs().mean()),
            "total_model": _f((bt["model_total"] - bt["total"]).abs().mean()),
            "total_close": _f((bt["total_line"] - bt["total"]).abs().mean()),
        }
        bt = bt.filter(pl.col("result") != pl.col("spread_line"))
        side_home = bt["model_margin"] > bt["spread_line"]
        won = (bt["result"] > bt["spread_line"]) == side_home
        model_card["backtest"]["ats"] = {"w": int(won.sum()), "l": int((~won).sum())}
    rules_path = REPORTS / "parlay_rules.parquet"
    if rules_path.exists():
        R = pl.read_parquet(rules_path)
        best = R.sort("tune_roi", descending=True).row(0, named=True)
        mix = {"any": "any market", "spread": "spreads", "total": "totals", "ml": "moneylines",
               "spread+total": "spreads and totals"}.get(best["mix"], best["mix"])
        select = {"top p_win": "most likely legs", "top EV": "biggest edges"}.get(best["select"], best["select"])
        legs = "legs chosen each week" if best["legs"] == "dynamic" else f"{best['legs']} legs"
        model_card["parlay"] = {"rules": R.height, "profitable_holdout": int((R["hold_roi"] > 0).sum()),
                                "best_tune_roi": round(best["tune_roi"], 3), "best_hold_roi": round(best["hold_roi"], 3),
                                "best_hold_record": f"{best['hold_won']}-{best['hold_n'] - best['hold_won']}",
                                "best_rule": f"{legs}, {mix}, {select}"}
    try:
        from backtest.trends import team_rows
        T = team_rows().filter(pl.col("season").is_between(2016, season - 1))
        hot_o = T.filter((pl.col("overs_n") >= 3) & (pl.col("over_rate_sofar") >= 0.75) & pl.col("over").is_not_null())
        hot_c = T.filter((pl.col("cov_n") >= 3) & (pl.col("cov_rate_sofar") >= 0.75) & pl.col("cover").is_not_null())
        model_card["trends"] = {"hot_over_n": hot_o.height, "hot_over_next": _f(hot_o["over"].mean(), 3),
                                "hot_ats_n": hot_c.height, "hot_ats_next": _f(hot_c["cover"].mean(), 3)}
    except Exception as e:  # noqa: BLE001 - the tab still renders without it
        print("trends skipped:", e)

    # --- my picks: lines to pick against this week, and finals for grading every week
    pickable = [{"id": c["id"], "week": week, "home": c["home"], "away": c["away"], "kickoff": c["kickoff"],
                 "spread": c["market"]["spread"], "total": c["market"]["total"],
                 "ml_home": c["market"]["ml_home"], "ml_away": c["market"]["ml_away"]} for c in cards]
    results = {g["game_id"]: {"week": int(g["week"]), "home": g["home_team"], "away": g["away_team"],
                              "hs": g["home_score"], "as": g["away_score"]}
               for g in season_games.iter_rows(named=True) if g["result"] is not None}
    model_picks = {}
    for pk in live_picks:
        if pk.get("kind") == "pick":
            model_picks.setdefault(pk["game_id"], {})[pk["market"]] = {
                "side": pk["side"], "line": pk["line"], "label": pk["label"], "american": pk["american"]}

    et_days = [g["kickoff_utc"].astimezone(ET) for g in week_games.iter_rows(named=True)]
    data = {
        "meta": {"season": season, "week": week, "generated": now.isoformat(),
                 "range": f"{et_days[0]:%b} {et_days[0].day} - {et_days[-1]:%b} {et_days[-1].day}",
                 "live_start": LIVE_START_WEEK, "books": [BOOK_NAMES[b] for b in SHOP_BOOKS],
                 "data_through": f"Week {week - 1}"},
        "teams": meta,
        "record": {"live": live, "pre": pre,
                   "all": ledger.record(sim + live_picks, sim_results + live_results),
                   "live_rows": (lr := _pick_rows(live_picks, live_results, verdict_by)),
                   "pre_rows": (pr := _pick_rows(sim, sim_results, verdict_by)),
                   "verdicts": {"all": C.tally(pr + lr), "since": C.tally(lr, C.SINCE_WEEK)}},
        "games": cards,
        "parlays": parlays,
        "defense": {"rows": def_rows, "league": def_league, "roles": roles},
        "model": model_card,
        "mypicks": {"games": pickable, "results": results, "model": model_picks},
        "props": props_data_out,
        "live": live_out,
    }
    # the page holds both sports: this build writes the NFL payload, then renders it with the NBA one as the
    # daily NBA build last left it (board/render.py)
    from board.render import NFL_DATA, render
    NFL_DATA.write_text(json.dumps(_clean(data), default=str, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    html = render()
    print(f"wrote {OUT} ({len(html) / 1024:.0f} KB): {len(cards)} games, "
          f"live record {live['all']['w']}-{live['all']['l']}-{live['all']['p']}, "
          f"pre-launch {pre['all']['w']}-{pre['all']['l']}-{pre['all']['p']} ({pre['all']['units']:+.1f}u)")
    if args.sim_lock != "tuesday":
        r = lambda t: f"{t['w']}-{t['l']}-{t['p']} {t['units']:+.1f}u line value {t['clv']}"
        print(f"replay, game lines locked at {args.sim_lock}: ATS {r(pre['ats'])} | O/U {r(pre['ou'])} | ML {r(pre['ml'])} | all {r(pre['all'])}")


if __name__ == "__main__":
    main()
