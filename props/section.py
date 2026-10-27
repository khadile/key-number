"""Everything the board's Player Props tab shows, built from the props model and archive.

    from props.section import build_section
    data = build_section(season, week, games, now, market_lines, dry_run)

The same lifecycle as game lines: projections all week, picks lock at the last
board update before each game (Thursday 10am ET for a Thursday game, Saturday
10am ET for Saturday, Sunday and Monday games), graded once nflverse posts the
stat lines.
Weeks before PROPS_LIVE_START are a labeled simulation at the same lock times.
From RULE_WEEK a pick needs 2%+ expected value, 4%+ when its edge is the price alone (clears()); the picks
the old rule would have added are kept off the books and judged at Pinnacle's close.

Only tracked roles get projected and picked: QB1, RB1 (RB2 in a real
timeshare), WR1, WR2, TE1. A player who moved up because a teammate is out is
tagged "promoted" so that idea gets its own record.
"""
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from board.market import american, build_parlays, kelly
from config import PARLAY_RISK, RAW, REPORTS
from model import betlist as BL
from model import checks as C
from model.staking import flat_risk, kelly_of, kelly_units
from props import ledger
from props.backtest import ROLE_MARKETS, assign_roles, baselines, load
from props.checks import Teams, prop_checks, prop_doubts
from props.data import PLAYER_GAMES
from props.lines import load as load_lines, main_lines
from props.model import STAT_COL, Simulator, TeamModel, defense_frame, player_frame, project, team_frame
from props.pricing import ET, MARKET_LABEL, PropPricer, lock_time, shift_to, side_probs

PROPS_LIVE_START = 5
SIZE = 4000
EV_MIN = 0.0          # weeks 1-6: a prop is a pick when its best price has positive expected value
# From week 7 (user, 2026-10-20; props/edge_source.py): 2% or more, and 4% or more when the edge is the price alone
# (DraftKings or FanDuel at Pinnacle's number, a worse number, or no Pinnacle number to check). In weeks 1-6 edges
# under 2% were gone by Pinnacle's close, and price-only edges mostly were too unless large. The picks the old rule
# would have added go to ledger/props_shadow.jsonl, off the books, to check on live data that they were worth nothing.
RULE_WEEK = 7
EV_FLOOR, EV_FLOOR_PRICE = 0.02, 0.04
# Matchup tags (user, 2026-10-22; CLAUDE.md 4b): a tag on passing-game props, never a check. In 2025, passing-game
# overs against the softest fifth of defenses (what each allowed that position in that stat per game, season to
# date, 3+ games) beat Pinnacle's close by 4.5 points beyond the market's usual miss, and unders against the
# toughest fifth did best too; both halves of the season agreed, 2026's small sample has not confirmed it yet.
# Rushing yards went the other way for QBs and RBs alike (overs against the softest run defenses lost to the close:
# the line over-adjusts), so a rushing tag agrees with the under against a soft run defense (the board flips it).
TAG_MARKETS = {"pass_yds": ("QB",), "rush_yds": ("QB", "RB"), "rec_yds": ("RB", "WR", "TE"), "receptions": ("RB", "WR", "TE")}
POTW_N = 5
BET_BOARD = BL.Board(half_life=3, cap=4)   # the bet list (model/betlist.py): three-week half-life, four per lock


def bet_cat(q: dict) -> str:
    return f"{q['market']} {q['side']}"


def betlist_view(raw: list[dict], picks: list[dict], results: list[dict]) -> dict | None:
    """This week's bet lists (one per lock: Thursday, Saturday, Monday) for the page, picks filled in."""
    ms = [r for r in raw if r.get("kind") == "betlist"]
    if not ms:
        return None
    wk = max((r["season"], r["week"]) for r in ms)
    by_id, res = {p["id"]: p for p in picks}, {r["id"]: r for r in results}
    out, cells = [], None
    for m in sorted((r for r in ms if (r["season"], r["week"]) == wk), key=lambda r: r["at"]):
        cells = m["cells"]
        for i in m["picks"]:
            if (p := by_id.get(i)):
                r = res.get(i, {})
                out.append({"label": p["label"], "game": f"{p['team']} vs {p['opp']}", "american": p["american"], "book": p["book_name"],
                            "ev": p["ev"], **m["why"][i], "result": r.get("result"), "units": round(r["units"], 2) if "units" in r else None})
    return {"week": wk[1], "picks": out, "cells": cells}
BOOK_NAMES = {"draftkings": "DraftKings", "fanduel": "FanDuel", "pinnacle": "Pinnacle"}
ROLE_ORDER = {"QB1": 0, "RB1": 1, "RB2": 2, "WR1": 3, "WR2": 4, "TE1": 5}


# ---------------------------------------------------------------- frames

def _candidates(season: int, week: int, games: pl.DataFrame) -> pl.DataFrame:
    """Players who could suit up for this week's unplayed games: active roster, minus
    anyone the injury report rules Out or Doubtful."""
    r = pl.read_parquet(RAW / "rosters_weekly" / f"{season}.parquet")
    wk = week if r.filter(pl.col("week") == week).height else int(r["week"].max())
    r = r.filter((pl.col("week") == wk) & (pl.col("status") == "ACT") & pl.col("position").is_in(["QB", "RB", "FB", "WR", "TE"]))
    inj_path = RAW / "injuries" / f"{season}.parquet"
    if inj_path.exists():
        out = (pl.read_parquet(inj_path).filter((pl.col("week") == week) & pl.col("report_status").is_in(["Out", "Doubtful"]))
               .select("gsis_id"))
        r = r.join(out, on="gsis_id", how="anti")
    sides = pl.concat([
        games.select("game_id", "season", "week", pl.col("home_team").alias("team"), pl.col("away_team").alias("opp")),
        games.select("game_id", "season", "week", pl.col("away_team").alias("team"), pl.col("home_team").alias("opp"))])
    pos = pl.when(pl.col("position") == "FB").then(pl.lit("RB")).otherwise(pl.col("position"))
    return (r.select("gsis_id", pl.col("full_name").alias("name"), pos.alias("position"), "team")
            .join(sides, on="team").unique(["gsis_id", "game_id"]))


def project_week(p, season: int, week: int, upcoming: pl.DataFrame, market_lines: dict) -> pl.DataFrame:
    """Projections for this week's unplayed games, tracked roles only."""
    if not upcoming.height:
        return pl.DataFrame()
    def side(home: bool):
        return upcoming.select(
            "game_id", "season", "week",
            pl.col("home_team" if home else "away_team").alias("team"),
            pl.col("away_team" if home else "home_team").alias("opp"),
            pl.col("game_id").map_elements(lambda g: (market_lines.get(g) or {}).get("total"), return_dtype=pl.Float64).alias("total_line"),
            pl.col("game_id").map_elements(lambda g: ((market_lines.get(g) or {}).get("spread") or 0.0) * (1 if home else -1),
                                           return_dtype=pl.Float64).alias("spread"),
            pl.lit(1 if home else 0, dtype=pl.Int8).alias("home"))
    trows = pl.concat([side(True), side(False)])
    tf = team_frame(trows, p)
    tp = TeamModel.fit(tf).predict(tf)
    cand = _candidates(season, week, upcoming)
    pf = player_frame(cand, p)
    x = project(pf.filter(pl.col("game_id").is_in(upcoming["game_id"].to_list()) & pl.col("targets").is_null()),
                tp, defense_frame(pf, tf, p), p)
    # roles from the current depth chart, ordered by projected usage (pipeline/starters.py, assign_roles)
    from pipeline.starters import latest_depth
    x = x.join(latest_depth(season), on=["gsis_id", "team", "position"], how="left")
    return assign_roles(x).filter(pl.col("role").is_not_null())


def promoted_flags(x: pl.DataFrame, tracked_hist: pl.DataFrame, available: pl.DataFrame | None = None) -> pl.DataFrame:
    """A player is promoted when he ranks higher at his position than in his team's
    previous game, because a teammate who ranked above him then is not playing now.
    `tracked_hist`: earlier tracked rows (role set), with t and rank. `available`: game_id, team, gsis_id of
    everyone who can play (this week's candidates; a past game's players). Without it only role holders count
    as playing, so a starter back from injury, or a teammate who simply lost his role, read as "promoted"
    (2026-10-15: Lamar Jackson and Caleb Williams returning)."""
    by_game: dict[tuple[str, float], dict[str, tuple[str, int]]] = {}
    season_of: dict[tuple[str, float], int] = {}
    for r in tracked_hist.select("team", "t", "gsis_id", "position", "rank", "season").iter_rows():
        by_game.setdefault((r[0], r[1]), {})[r[2]] = (r[3], r[4])
        season_of[(r[0], r[1])] = r[5]
    team_ts: dict[str, list[float]] = {}
    for team, t in by_game:
        team_ts.setdefault(team, []).append(t)
    playing: dict[tuple[str, str], set] = {}
    for g, team, pid in (available if available is not None else x).select("game_id", "team", "gsis_id").iter_rows():
        playing.setdefault((g, team), set()).add(pid)
    out = []
    for r in x.select("game_id", "team", "t", "gsis_id", "position", "rank", "season").iter_rows(named=True):
        # only this season: a new team in September is roster turnover, not an injury bump
        prior = [t for t in team_ts.get(r["team"], []) if t < r["t"] and season_of[(r["team"], t)] == r["season"]]
        if not prior:
            out.append(False)
            continue
        prev = by_game[(r["team"], max(prior))]
        my_prev = prev[r["gsis_id"]][1] if r["gsis_id"] in prev else 99
        gone = [pid for pid, (pos, rk) in prev.items()
                if pos == r["position"] and rk < my_prev and pid not in playing[(r["game_id"], r["team"])]]
        out.append(bool(gone) and r["rank"] < my_prev)
    return x.with_columns(promoted=pl.Series(out, dtype=pl.Boolean))


def _books(L: pl.DataFrame) -> dict[tuple[str, str, str], dict]:
    out: dict = {}
    for r in L.iter_rows(named=True):
        out.setdefault((r["game_id"], r["gsis_id"], r["market"]), {})[r["book"]] = {
            "line": r["line"], "over": r["over"], "under": r["under"], "p_over": r["p_over"]}
    return out


def _samples(sim: Simulator, x: pl.DataFrame) -> dict[tuple[str, str, str], np.ndarray]:
    if not x.height:
        return {}
    S = sim.simulate(x, SIZE)
    out = {}
    for i, r in enumerate(x.select("game_id", "gsis_id", "role").iter_rows()):
        for m in ROLE_MARKETS[r[2]]:
            out[(r[0], r[1], m)] = np.sort(S[m][i])
    return out


def measure_k(samples: dict, lines_lock: dict, lines_close: dict) -> dict:
    """How far prop lines moved toward the model between lock and close this season
    (anchor = Pinnacle, else DK/FD), in model standard deviations."""
    pr = PropPricer(0.0)
    gaps, moves = [], []
    for key, s in samples.items():
        a, c = lines_lock.get(key), lines_close.get(key)
        if not a or not c:
            continue
        da, _ = pr.anchor(s, a)
        dc, _ = pr.anchor(s, c)
        if da is None or dc is None:
            continue
        sd = float(s.std()) or 1.0
        gaps.append(-da / sd)
        moves.append((dc - da) / sd)
    g, m = np.array(gaps), np.array(moves)
    if len(g) < 30:
        return {"k": 0.0, "n": int(len(g)), "moved": 0, "toward": 0}
    moved = m != 0
    return {"k": round(float(np.clip((g * m).sum() / (g ** 2).sum(), 0, 0.5)), 4), "n": int(len(g)),
            "moved": int(moved.sum()), "toward": int((np.sign(g[moved]) == np.sign(m[moved])).sum()),
            "clv_sd": round(float((np.sign(g) * m).mean()), 4)}


def edge_from(pk: dict, books: dict) -> str:
    """Where a priced side's edge comes from, against Pinnacle's number at the same moment: a better number
    (DraftKings or FanDuel hung a line off Pinnacle's, our side's way), the same number (only a better price), a
    worse number, or no Pinnacle number to compare."""
    pin = books.get("pinnacle")
    if pin is None:
        return "no pinnacle"
    rel = (pin["line"] - pk["line"]) * (1 if pk["side"] == "over" else -1)
    return "better number" if rel > 0 else ("same number" if rel == 0 else "worse number")


def clears(pk: dict, src: str, rule: bool) -> bool:
    """The bar a prop's best price clears to be a pick: positive EV, or with the week-7 rule 2%+ and 4%+ unless
    the edge is a better number than Pinnacle's."""
    if not rule:
        return pk["ev"] > EV_MIN
    return pk["ev"] >= EV_FLOOR and (src == "better number" or pk["ev"] >= EV_FLOOR_PRICE)


def make_picks(x: pl.DataFrame, samples: dict, books: dict, pricer: PropPricer, when_of: dict,
               flag_potw: bool = True, rule: bool = False) -> list[dict]:
    """Best prop per tracked player (DK or FanDuel) that clears the bar (clears()), Picks of the Week flagged."""
    out = []
    for r in x.iter_rows(named=True):
        best = None
        for m in ROLE_MARKETS[r["role"]]:
            key = (r["game_id"], r["gsis_id"], m)
            if key not in samples or key not in books:
                continue
            pk = pricer.price(samples[key], books[key])
            if not pk:
                continue
            src = edge_from(pk, books[key])
            if clears(pk, src, rule) and (best is None or kelly(pk) > kelly(best)):
                best = {**pk, "market": m, "edge_from": src}
        if not best:
            continue
        side = "Over" if best["side"] == "over" else "Under"
        out.append({
            "kind": "prop", "id": f"{r['game_id']}:{r['gsis_id']}:{best['market']}", "game_id": r["game_id"],
            "season": int(r["season"]), "week": int(r["week"]), "gsis_id": r["gsis_id"], "name": r["name"],
            "team": r["team"], "opp": r["opp"], "position": r["position"], "role": r["role"],
            "promoted": bool(r.get("promoted") or False), "market": best["market"], "side": best["side"],
            "line": best["line"], "label": f"{r['name']} {side} {best['line']:g} {MARKET_LABEL[best['market']].lower()}",
            "book": best["book"], "book_name": BOOK_NAMES[best["book"]], "dec": round(best["dec"], 4),
            "american": american(best["dec"]), "p_win": round(best["p_win"], 4), "p_push": round(best["p_push"], 4),
            "ev": round(best["ev"], 4), "model_median": round(best["model_median"], 1),
            "market_median": round(best["market_median"], 1), "fair_median": round(best["fair_median"], 1),
            "anchor": best["anchor"], "edge_from": best["edge_from"], "locked_at": when_of[r["game_id"]].isoformat(), "potw": False,
            # the stake it locked with, and the quarter-Kelly size tracked beside it (model/staking.py)
            "risk": flat_risk(best["ev"]), "kelly_u": kelly_units(best["p_win"], best["dec"], best["p_push"])})
    if flag_potw:
        seen, top = set(), []
        for p in sorted(out, key=lambda p: -kelly(p)):
            if p["game_id"] in seen:
                continue
            seen.add(p["game_id"])
            top.append(p["id"])
            if len(top) == POTW_N:
                break
        for p in out:
            p["potw"] = p["id"] in top
    return out


PARLAY_POOL = 16  # parlays draw from the week's strongest props; 65 legs is 8 million 5-leg tickets


def week_parlays(picks: list[dict], season: int, week: int, locked_at: str, want=("golden", "lottery")) -> list[dict]:
    out = []
    pool = sorted((p for p in picks if p.get("kind") == "prop" and p["ev"] > 0), key=lambda p: -kelly(p))[:PARLAY_POOL]
    for par in build_parlays(pool):
        if par and par["type"] in want:
            out.append({**par, "kind": "parlay", "id": f"{season}_{week:02d}:props:{par['type']}", "season": season,
                        "week": week, "risk": PARLAY_RISK, "p": round(par["p"], 4), "ev": round(par["ev"], 4),
                        "dec": round(par["dec"], 3), "locked_at": locked_at})
    return out


# ---------------------------------------------------------------- board data

def _stat_lines(hist: pl.DataFrame) -> dict[tuple[str, str], dict]:
    return {(r["game_id"], r["gsis_id"]): {m: float(r[c]) for m, c in STAT_COL.items() if r[c] is not None}
            for r in hist.select("game_id", "gsis_id", *STAT_COL.values()).iter_rows(named=True)}


def _rows(picks: list[dict], results: list[dict], verdict_by: dict | None = None, ev_at_close=None) -> list[dict]:
    """`ev_at_close(pick)`: its EV at Pinnacle's close (_ev_close), for graded props that played."""
    res = {r["id"]: r for r in results}
    vb = verdict_by or {}
    by_id = {p["id"]: p for p in picks}
    out = []
    for p in picks:
        r = res.get(p["id"], {})
        if p.get("kind") == "parlay":
            legs = [by_id[l]["label"] for l in p["legs"] if l in by_id]
            out.append({"week": p["week"], "market": p["type"], "label": "Golden Parlay" if p["type"] == "golden" else "+EV Lottery",
                        "game": f"{len(legs)} legs", "legs": legs, "american": p.get("american"), "book": None,
                        "potw": False, "p_win": p.get("p"), "ev": p.get("ev"), "result": r.get("result"),
                        "units": round(r["units"], 2) if "units" in r else None, "clv": None})
        else:
            out.append({"week": p["week"], "market": p["market"], "label": p["label"], "game": f"{p['team']} vs {p['opp']}",
                        "legs": [], "american": p["american"], "book": p["book_name"], "potw": p.get("potw", False), "bet": p.get("bet", False),
                        "promoted": p.get("promoted", False), "p_win": p["p_win"], "ev": p["ev"],
                        "kelly_u": kelly_of(p), **vb.get(p["id"], {}),
                        "result": r.get("result"), "actual": r.get("actual"),
                        "units": round(r["units"], 2) if "units" in r else None, "clv": r.get("clv"),
                        "ev_close": (round(e, 4) if ev_at_close and r.get("result") not in (None, "V")
                                     and (e := ev_at_close(p)) is not None else None)})
    return out


def matchups(season: int, week: int) -> dict[tuple[str, str, str], dict]:
    """(defense, position, market) -> the defense's rank by what it allowed that position in that stat per game
    this season before `week` (1 = the most; defenses with 3+ games), and its tag: soft (the fifth that allowed the
    most), tough (the fifth that allowed the least) or None, for the positions each market is tagged for."""
    pg = pl.read_parquet(PLAYER_GAMES).filter((pl.col("season") == season) & (pl.col("week") < week))
    pg = pg.with_columns(pos=pl.col("position").replace("FB", "RB")).filter(pl.col("pos").is_in(["QB", "RB", "WR", "TE"]))
    out = {}
    for m in TAG_MARKETS:
        c = STAT_COL[m]
        t = (pg.group_by("opp", "pos", "week").agg(pl.col(c).fill_null(0).sum())
             .group_by("opp", "pos").agg(g=pl.len(), per_game=pl.col(c).mean()).filter(pl.col("g") >= 3))
        t = t.with_columns(rank=pl.col("per_game").rank("ordinal", descending=True).over("pos"), n=pl.len().over("pos"))
        for r in t.filter(pl.col("pos").is_in(list(TAG_MARKETS[m]))).iter_rows(named=True):
            q = 1 - (r["rank"] - 1) / max(r["n"] - 1, 1)          # 1: allowed the most
            out[(r["opp"], r["pos"], m)] = {"rank": int(r["rank"]), "n": int(r["n"]), "g": int(r["g"]), "per_game": round(r["per_game"], 1),
                                            "tag": "soft" if q >= 0.8 else ("tough" if q < 0.2 else None)}
    return out


def _mtag(mtab: dict, opp: str, pos: str, market: str) -> dict | None:
    mu = mtab.get((opp, "RB" if pos == "FB" else pos, market))
    return mu if mu and mu["tag"] else None


def _ev_close(q: dict, samples: dict, close_b: dict) -> float | None:
    """A pick's expected value at Pinnacle's closing number, priced the board's way (k = 0): the yardstick of
    props/edge_source.py, which settles long before wins and losses do. None without Pinnacle's close."""
    key = (q["game_id"], q["gsis_id"], q["market"])
    s, pc = samples.get(key), (close_b.get(key) or {}).get("pinnacle")
    if s is None or pc is None:
        return None
    over, push, under = side_probs(s + shift_to(s, pc["line"], pc["p_over"]), q["line"])
    p = over if q["side"] == "over" else under
    return p * (q["dec"] - 1) - (1 - p - push)


def _rule_tally(picks: list[dict], results: list[dict], samples: dict, close_b: dict) -> dict:
    """Picks locked, their graded record, and their value at the lock and at Pinnacle's close."""
    res = {r["id"]: r for r in results}
    rows = [dict(q, **res[q["id"]]) for q in picks if q["id"] in res]
    played = [r for r in rows if r["result"] != "V"]
    evc = [e for e in (_ev_close(r, samples, close_b) for r in played) if e is not None]
    return {**ledger._tally(rows), "locked": len(picks),
            "ev": round(float(np.mean([r["ev"] for r in played])), 4) if played else None,
            "ev_close": round(float(np.mean(evc)), 4) if evc else None,
            "still": round(float(np.mean([e > 0 for e in evc])), 3) if evc else None, "ev_close_n": len(evc)}


def _player_logs(ids: list[str], hist_all: pl.DataFrame, season: int, close_by: dict) -> dict:
    """This season's and last season's game logs for the players on the page."""
    pg = pl.read_parquet(PLAYER_GAMES).filter(pl.col("gsis_id").is_in(ids) & pl.col("season").is_between(season - 1, season))
    out: dict[str, list] = {}
    for r in pg.sort("season", "week").iter_rows(named=True):
        lines = {m: close_by.get((r["game_id"], r["gsis_id"], m)) for m in STAT_COL}
        out.setdefault(r["gsis_id"], []).append({
            "season": r["season"], "week": r["week"], "opp": r["opp"], "snap": r.get("offense_pct"),
            "tgt": r["targets"], "rec": r["receptions"], "rec_yds": r["receiving_yards"],
            "car": (r["d_car"] or 0) + (r["scr"] or 0), "rush_yds": r["rushing_yards"],
            "att": r["att"], "cmp": r["cmp"], "pass_yds": r["passing_yards"],
            "share": round(r["tgt_share"], 3) if r["tgt_share"] is not None else None,
            "lines": {m: v for m, v in lines.items() if v is not None}})
    return out


def _quant(s: np.ndarray) -> list[float]:
    return [round(float(v), 1) for v in np.quantile(s, [0.1, 0.25, 0.5, 0.75, 0.9])]


def _lock_at(kickoff: datetime, now: datetime, lock_now: bool) -> datetime:
    """A game's lock time; with lock_now (a build moved earlier than its 10am slot, user 2026-10-15), a lock
    due later today is due now, so the day's update can run early without leaving that day's games unlocked."""
    t = lock_time(kickoff)
    if lock_now and t.astimezone(ET).date() == now.astimezone(ET).date():
        return min(t, now)
    return t


def build_section(season: int, week: int, games: pl.DataFrame, now: datetime, market_lines: dict,
                  dry_run: bool, lock_now: bool = False) -> dict:
    p, disp = load()
    sim = Simulator(disp, seed=17)
    hist = baselines(_frames_cached(p))
    season_games = games.filter(pl.col("season") == season)
    finished_ids = set(season_games.filter(pl.col("result").is_not_null())["game_id"].to_list())
    kick = dict(zip(season_games["game_id"], season_games["kickoff_utc"]))
    have_stats = set(hist.filter(pl.col("season") == season)["game_id"].unique().to_list())
    graded_ids = finished_ids & have_stats

    # --- archive lines for this season: close (kickoff - 5 min) and each game's lock time
    P = load_lines(list(kick))
    if P.height:
        locks = pl.DataFrame({"game_id": list(kick), "lock": [lock_time(k) for k in kick.values()]}).with_columns(
            pl.col("lock").dt.cast_time_unit("us").dt.convert_time_zone("UTC"))
        P = P.join(locks, on="game_id")
        close_L = main_lines(P, pl.col("kickoff") - timedelta(minutes=5))
        lock_L = main_lines(P, pl.col("lock"))
        now_L = main_lines(P, pl.lit(now))
    else:
        close_L = lock_L = now_L = pl.DataFrame(schema={"game_id": pl.String, "book": pl.String, "market": pl.String,
                                                        "gsis_id": pl.String, "line": pl.Float64, "over": pl.Float64,
                                                        "under": pl.Float64, "p_over": pl.Float64})
    close_b, lock_b, now_b = _books(close_L), _books(lock_L), _books(now_L)
    closes = {(r["game_id"], r["gsis_id"], r["market"], r["book"]): r["line"] for r in close_L.iter_rows(named=True)}
    close_ref = {}
    for (g, pid, m, b), v in closes.items():
        if b == "pinnacle" or (g, pid, m) not in close_ref:
            close_ref[(g, pid, m)] = v

    # --- finished tracked rows this season: k, the simulation, grading
    done = hist.filter((pl.col("season") == season) & pl.col("game_id").is_in(list(graded_ids)) & pl.col("role").is_not_null())
    done = promoted_flags(done, hist.filter(pl.col("role").is_not_null()), hist.select("game_id", "team", "gsis_id"))
    s_done = _samples(sim, done)
    kstudy = measure_k(s_done, lock_b, close_b)
    pricer = PropPricer(kstudy["k"])
    print(f"props: k {kstudy['k']} from {kstudy['n']} lines ({kstudy['toward']} of {kstudy['moved']} moved toward the model)")

    stats = _stat_lines(hist.filter(pl.col("season") == season))
    sim_picks: list[dict] = []
    for wk in sorted(done.filter(pl.col("week") < PROPS_LIVE_START)["week"].unique().to_list()):
        xw = done.filter(pl.col("week") == wk)
        wp = make_picks(xw, s_done, lock_b, pricer, {g: lock_time(kick[g]) for g in xw["game_id"].unique()})
        sim_picks += wp
        lock_main = max((lock_time(kick[g]) for g in xw["game_id"].unique()), default=now)
        sim_picks += week_parlays(wp, season, wk, lock_main.isoformat())
    sim_results = ledger.grade_all(sim_picks, stats, graded_ids, closes)
    pre = ledger.record(sim_picks, sim_results)

    # --- this week: projections for unplayed games, lock what is in its window
    upcoming = season_games.filter((pl.col("week") == week) & pl.col("result").is_null())
    xw = project_week(p, season, week, upcoming, market_lines)
    if xw.height:
        xw = promoted_flags(xw, hist.filter(pl.col("role").is_not_null()),
                            _candidates(season, week, upcoming).select("game_id", "team", "gsis_id"))
    s_week = _samples(sim, xw)
    existing = ledger.read(ledger.PICKS)
    # a game is locked once it has picks, or picks passed on: under the 2%/4% rule a game can lock with none
    locked_games = {r["game_id"] for r in existing + ledger.read(ledger.SHADOW) if r.get("kind") == "prop"}
    new_locks, new_parlays, new_shadow, new_bl = [], [], [], []
    if week >= PROPS_LIVE_START and xw.height:
        due = [g for g in upcoming["game_id"].to_list() if _lock_at(kick[g], now, lock_now) <= now < kick[g] and g not in locked_games]
        if due:
            xd = xw.filter(pl.col("game_id").is_in(due))
            when = {g: now for g in due}
            new_locks = make_picks(xd, s_week, now_b, pricer, when, flag_potw=False, rule=week >= RULE_WEEK)
            if week >= RULE_WEEK:
                # what the old rule would have added, off the books: is what the rule passes on worth nothing?
                kept = {q["id"] for q in new_locks}
                new_shadow = [q for q in make_picks(xd, s_week, now_b, pricer, when, flag_potw=False) if q["id"] not in kept]
        # Parlays and Picks of the Week lock with the weekend slate (Saturday morning), from
        # locked picks whose games have not started.
        have = {r["type"] for r in existing if r.get("kind") == "parlay" and r["season"] == season and r["week"] == week}
        if any(r.get("kind") == "potw" and r.get("season") == season and r.get("week") == week for r in existing):
            have.add("potw")   # Picks of the Week records have no "type"; without this every build re-locked them
        weekend = [g for g in upcoming["game_id"].to_list() if kick[g].astimezone(ET).weekday() in (5, 6)]  # Sat-Sun, Eastern (Monday locks Monday)
        if weekend and all(_lock_at(kick[g], now, lock_now) <= now for g in weekend):
            pool = [r for r in existing + new_locks if r.get("kind") == "prop" and r["season"] == season
                    and r["week"] == week and kick[r["game_id"]] > now]
            if "potw" not in have and pool:
                seen, top = set(), []
                for pk in sorted(pool, key=lambda q: -kelly(q)):
                    if pk["game_id"] not in seen:
                        seen.add(pk["game_id"])
                        top.append(pk)
                    if len(top) == POTW_N:
                        break
                new_parlays.append({"kind": "potw", "id": f"{season}_{week:02d}:props:potw", "season": season,
                                    "week": week, "picks": [q["id"] for q in top], "at": now.isoformat()})
            missing = tuple(t for t in ("golden", "lottery") if t not in have)
            if missing:
                new_parlays += week_parlays(pool, season, week, now.isoformat(), want=missing)
        # this lock's bet list (model/betlist.py), judged on every graded prop of the season, replayed weeks included
        if new_locks:
            bl_graded = {r["id"]: r for r in sim_results + ledger.read(ledger.RESULTS) if r.get("result") in ("W", "L", "P")}
            bl_hist = [{"cat": bet_cat(q), "tier": BL.tier_of(q["ev"], BET_BOARD), "age": week - q["week"], "units": bl_graded[q["id"]]["units"],
                     "ev_close": _ev_close(q, s_done, close_b)}
                    for q in sim_picks + existing if q.get("kind") == "prop" and q["season"] == season and q["id"] in bl_graded]
            bl_cells = BL.estimate(bl_hist, BET_BOARD)
            bl_cands = [{"id": q["id"], "cat": bet_cat(q), "tier": BL.tier_of(q["ev"], BET_BOARD), "ev": q["ev"]} for q in new_locks if q["ev"] > 0]
            new_bl = [BL.marker(f"{season}_{week:02d}:props:betlist:{now.astimezone(ET):%a}", BL.choose(bl_cands, bl_cells, BET_BOARD), bl_cells,
                                {"season": season, "week": week, "at": now.isoformat()})]
        writes = new_locks + new_parlays + new_bl
        if writes:
            if dry_run:
                print(f"props dry run: would lock {len(new_locks)} picks and {len(new_parlays)} weekly items")
            else:
                ledger.append(ledger.PICKS, writes)
                print(f"props: locked {len(new_locks)} picks, {[w.get('type', w['kind']) for w in new_parlays]}")
        if new_shadow:
            if dry_run:
                print(f"props dry run: would keep {len(new_shadow)} picks the old rule would have added, off the books")
            else:
                ledger.append(ledger.SHADOW, new_shadow)
                print(f"props: {len(new_shadow)} picks the old rule would have added, kept off the books")

    # --- live record: grade, then apply Picks-of-the-Week markers
    all_rows = existing + (new_locks + new_parlays + new_bl if dry_run else [])
    if not dry_run:
        g = ledger.grade_ledger(stats, graded_ids, closes)
        if g:
            print(f"props: graded {len(g)}")
        all_rows = ledger.read(ledger.PICKS)
        live_results = ledger.read(ledger.RESULTS)
    else:
        live_results = ledger.read(ledger.RESULTS)
        done_ids = {r["id"] for r in live_results}
        live_results += [r for r in ledger.grade_all(all_rows, stats, graded_ids, closes) if r["id"] not in done_ids]
    potw_ids = {i for r in all_rows if r.get("kind") == "potw" for i in r["picks"]}
    bet_ids = {i for r in all_rows if r.get("kind") == "betlist" for i in r["picks"]}
    live_picks = [{**r, "potw": r["id"] in potw_ids, "bet": r["id"] in bet_ids} if r.get("kind") == "prop" else r
                  for r in all_rows if r.get("kind") in ("prop", "parlay") and r["season"] == season]
    live = ledger.record(live_picks, live_results)

    # --- the week-7 rule on live data: its picks against the ones it passed on, judged at Pinnacle's close
    shadow = list({q["id"]: q for q in ledger.read(ledger.SHADOW) + (new_shadow if dry_run else [])
                   if q["season"] == season}.values())
    rule = {"week": RULE_WEEK, "floor": EV_FLOOR, "floor_price": EV_FLOOR_PRICE,
            "picks": _rule_tally([q for q in live_picks if q.get("kind") == "prop" and q["week"] >= RULE_WEEK],
                                 live_results, s_done, close_b),
            "passed": _rule_tally(shadow, ledger.grade_all(shadow, stats, graded_ids, closes), s_done, close_b)}

    # --- is each prop also a good pick? (model/checks.py) A game's verdict is the one frozen before its kickoff
    # when there is one, else rebuilt from what was known then; this week's picks are checked live, priced now.
    teams, doubts, frozen = Teams(hist), prop_doubts(season), C.frozen(kick)
    def checks_of(q: dict, live: bool = False) -> dict:
        key = (q["game_id"], q["gsis_id"], q["market"])
        return prop_checks(q, doubts.get((q["week"], q["gsis_id"])), teams.streak(q["gsis_id"], q["season"], q["week"], q["team"]),
                           now_b.get(key) if live else None, s_week.get(key) if live else None, pricer)
    verdict_by, checked = {}, {}
    for q in [r for r in sim_picks + live_picks if r.get("kind") == "prop"]:
        f = frozen.get(q["id"]) if kick[q["game_id"]] <= now else None
        if f:
            verdict_by[q["id"]] = {"verdict": f["verdict"], "failed": f["failed"]}
            continue
        c = checks_of(q, live=q["week"] == week and kick[q["game_id"]] > now)
        checked[q["id"]] = c
        verdict_by[q["id"]] = {"verdict": c["verdict"], "failed": C.failed(c["checks"])}
    upcoming_ids = [q for q in live_picks if q.get("kind") == "prop" and kick[q["game_id"]] > now]
    if upcoming_ids and not dry_run:
        n = C.freeze([{"kind": "prop", "id": q["id"], "game_id": q["game_id"], **verdict_by[q["id"]]} for q in upcoming_ids], now)
        if n:
            print(f"props: froze {n} verdicts")

    # --- cards: every tracked player in this week's games with lines, projections, picks
    res_by = {r["id"]: r for r in live_results}
    try:
        mtab = matchups(season, week)
    except Exception as e:   # noqa: BLE001  (a tag is decoration: never stop the build)
        print(f"matchup tags skipped: {type(e).__name__}: {e}")
        mtab = {}
    picks_by_player = {(q["game_id"], q["gsis_id"]): {**q, "kelly_u": kelly_of(q),   # older picks carry no kelly_u
                                                      "matchup": _mtag(mtab, q["opp"], q["position"], q["market"]),
                                                      "checks": checked.get(q["id"]) or checks_of(q),
                                                      **({"result": res_by[q["id"]]["result"]} if q["id"] in res_by else {})}
                       for q in live_picks if q.get("kind") == "prop" and q["week"] == week}
    week_rows = xw
    played_now = done.filter(pl.col("week") == week)
    if played_now.height:
        week_rows = pl.concat([week_rows, played_now.select(week_rows.columns)], how="vertical_relaxed") if xw.height else played_now
    s_all = {**s_week, **{k: v for k, v in s_done.items() if k[0] in set(played_now["game_id"].to_list())}}
    players, games_out = [], {}
    ros = pl.read_parquet(RAW / "rosters_weekly" / f"{season}.parquet").select("gsis_id", "espn_id").drop_nulls().unique("gsis_id")
    espn_ids = dict(zip(ros["gsis_id"].to_list(), ros["espn_id"].to_list()))
    for r in week_rows.sort("game_id", "team", pl.col("role").replace_strict(ROLE_ORDER)).iter_rows(named=True):
        mk = []
        for m in ROLE_MARKETS[r["role"]]:
            key = (r["game_id"], r["gsis_id"], m)
            s = s_all.get(key)
            if s is None:
                continue
            bk = now_b.get(key) or lock_b.get(key) or {}
            priced = pricer.price(s, bk) if bk else None
            src = edge_from(priced, bk) if priced else None
            books = {b: {"line": v["line"], "over": american(v["over"]), "under": american(v["under"])} for b, v in bk.items()}
            mk.append({"market": m, "median": round(float(np.median(s)), 1), "mean": round(float(np.mean(s)), 1),
                       "q": _quant(s), "qq": [round(float(v), 1) for v in np.quantile(s, np.linspace(0.05, 0.95, 19))],
                       "books": books,
                       "fair": round(priced["fair_median"], 1) if priced else None,
                       "lean": ({"side": priced["side"], "line": priced["line"], "book": BOOK_NAMES[priced["book"]],
                                 "american": american(priced["dec"]), "p": round(priced["p_win"], 3), "ev": round(priced["ev"], 4),
                                 "edge_from": src, "clears": clears(priced, src, week >= RULE_WEEK)}
                                if priced else None),
                       "matchup": _mtag(mtab, r["opp"], r["position"], m),
                       "actual": stats.get((r["game_id"], r["gsis_id"]), {}).get(m)})
        pick = picks_by_player.get((r["game_id"], r["gsis_id"]))
        eid = espn_ids.get(r["gsis_id"])
        pl_row = {"img": f"https://a.espncdn.com/combiner/i?img=/i/headshots/nfl/players/full/{eid}.png&w=96&h=70&scale=crop" if eid else None,
                  "id": r["gsis_id"], "name": r["name"], "team": r["team"], "opp": r["opp"], "pos": r["position"],
                  "role": r["role"], "promoted": bool(r.get("promoted") or False), "game_id": r["game_id"],
                  "usage": {"tgt": round(r["tgt_hat"] or 0, 1), "car": round(r["car_hat"] or 0, 1),
                            "att": round(r["att_hat"] or 0, 1), "tshare": round(r["tshare"] or 0, 3),
                            "rshare": round(r["rshare"] or 0, 3)},
                  "markets": mk, "pick": pick}
        players.append(pl_row)
        games_out.setdefault(r["game_id"], []).append(r["gsis_id"])

    ids = [q["id"] for q in players]
    logs = _player_logs(ids, hist, season, close_ref)
    parlays = {}
    for q in live_picks:
        if q.get("kind") == "parlay" and q["week"] == week:
            parlays[q["type"]] = {**q, "leg_picks": [next(x for x in live_picks if x["id"] == l) for l in q["legs"]],
                                  **({"result": res_by[q["id"]]["result"]} if q["id"] in res_by else {})}

    # --- my picks: tracked players' current lines, and stat lines to grade finished games
    pickable = []
    for q in players:
        if kick.get(q["game_id"]) is None or kick[q["game_id"]] <= now:
            continue
        for m in q["markets"]:
            if m["books"]:
                b = "draftkings" if "draftkings" in m["books"] else next(iter(m["books"]))
                pickable.append({"key": f"{q['game_id']}__{q['id']}__{m['market']}", "game_id": q["game_id"],
                                 "player": q["id"], "name": q["name"], "team": q["team"], "market": m["market"],
                                 "line": m["books"][b]["line"], "over": m["books"][b]["over"], "under": m["books"][b]["under"],
                                 "book": BOOK_NAMES[b], "kickoff": kick[q["game_id"]].isoformat(),
                                 "model_side": (m["lean"] or {}).get("side")})
    results = {f"{g}__{pid}": v for (g, pid), v in stats.items()}

    study = {}
    sp = REPORTS / "props_study.json"
    if sp.exists():
        import json
        study = json.loads(sp.read_text())
    sgo = {}
    sg = REPORTS / "props_sgo_study.json"
    if sg.exists():
        import json
        sgo = json.loads(sg.read_text()).get("board", {})
    edge = {}
    ep = REPORTS / "props_edge_source.json"     # python -m props.edge_source: where the props edge comes from
    if ep.exists():
        import json
        edge = json.loads(ep.read_text())
    bt = {}
    pp = REPORTS / "props_params.json"
    if pp.exists():
        import json
        saved = json.loads(pp.read_text())
        bt = {"holdout": saved.get("holdout"), "pit": saved.get("pit")}

    evq = lambda q: _ev_close(q, s_done, close_b)   # noqa: E731
    return {
        "meta": {"live_start": PROPS_LIVE_START, "rule_week": RULE_WEEK, "lock_note": "Saturday 10am ET (Thursday games: Thursday 10am ET; Monday games: Monday 10am ET)",
                 "k": kstudy, "tracked": len(players)},
        "locks": {g: lock_time(kick[g]).isoformat() for g in season_games.filter(pl.col("week") == week)["game_id"].to_list()},
        "record": {"live": live, "pre": pre, "all": ledger.record(sim_picks + live_picks, sim_results + live_results),
                   "live_rows": (lr := _rows(live_picks, live_results, verdict_by, evq)), "pre_rows": (pr := _rows(sim_picks, sim_results, verdict_by, evq)),
                   "verdicts": {"all": C.tally(pr + lr), "since": C.tally(lr, C.SINCE_WEEK)}, "rule": rule},
        "players": players, "games": games_out, "logs": logs, "parlays": parlays,
        "betlist": betlist_view(all_rows, live_picks, live_results),
        "model": {"backtest": bt, "study": study, "sgo": sgo, "edge": edge},
        "mypicks": {"lines": pickable, "results": results},
    }


_FRAMES: dict = {}


def _frames_cached(p):
    from props.backtest import frames
    key = tuple(sorted(p.asdict().items()))
    if key not in _FRAMES:
        _FRAMES[key] = frames(p)
    return _FRAMES[key]
