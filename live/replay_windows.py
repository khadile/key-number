"""Replay the redesigned live rules over finished 2026 games against the real in-game prices.

    python -m live.replay_windows

The redesign (user, 2026-10-09: fewer, more trustworthy plays) watches every play
but only bets at breaks, when the game stops and the books' lines settle:

- player props: at the end of the first quarter, at halftime and at the end of
  the third quarter; this replay compares which of those windows to use;
- game markets (moneyline, spread, total): after every score, once the books
  reopen for the kickoff.

A play is flagged only if it passes every gate of the rubric:
1. discipline: the model's chance is shrunk toward the market's no-vig chance by
   s, the share of the model's disagreement that has held up (fitted on the
   other weeks only, so no week is judged by a number fitted on itself), and the
   price still needs decide.MIN_EV expected value at that shrunk chance;
2. it holds on two price checks (45 s apart at a quarter break, 90 s at
   halftime): a stale or flickering price fails;
3. no props in a game led by more than BLOWOUT at the break; game markets only
   between -PRICE_CAP and +PRICE_CAP (no long shots priced off thin tails);
4. the model and the market within decide.MAX_GAP (bigger gaps are bad data);
5. one play per player, correlated plays count once, at most PER_WINDOW a
   window and decide.MAX_PER_GAME a game.

Every flag counts at the price it was flagged. Line value (did the line move our
way in the next five minutes) and calibration (does a stated 55% win 55%) judge a
rule; win-loss and units are reported but are noisy on this many games.

Writes reports/live_windows.json and reports/live_windows_flags.parquet.
"""
import argparse
import json
import zlib
from datetime import timedelta

import numpy as np
import polars as pl

import live.replay as replay_mod
import live.replay_games as replay_games_mod
from config import RAW, REPORTS
from live import decide
from live.model import LiveModel, LiveParams
from live.replay import CLV_LAG, FILL_LAG, _banked, _final_stats, _prop_grid
from live.replay_games import _fair, _game_grid
from pipeline.odds_history import HIST, fixture_index
from props.lines import load as load_props, main_lines
from props.model import Simulator

OUT = REPORTS / "live_windows.json"
FLAGS = REPORTS / "live_windows_flags.parquet"
CANDS = REPORTS / "live_windows_cands.parquet"     # every candidate price, so rubric changes can be re-tested in seconds
GAPS = (0.15, 0.10)    # how far the model may be from the market: the big disagreements are where it is missing news
WINDOWS = {"q1": 2700.0, "half": 1800.0, "q3": 900.0}   # game seconds left when each break starts
VARIANTS = {"q1": ("q1",), "half": ("half",), "q1+half": ("q1", "half"), "q1+half+q3": ("q1", "half", "q3")}
FEED_LAG = 60          # the engine learns a quarter ended about a minute later (ESPN's delay)
PERSIST = 45           # the second price check, this long after the first (twice that at halftime)
BLOWOUT = 17
PRICE_CAP = 300
PER_WINDOW = 2         # the exploration below; the engine's own caps are decide.PROP_PER_GAME / GAME_PER_GAME
EXPLORE_PER_GAME = 3
STALE_BREAK = 900      # at a break a line can sit unchanged for minutes; pulls are recorded as inactive
S_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)


def _plays(gids: list[str]) -> pl.DataFrame:
    cols = ["game_id", "play_id", "time_of_day", "game_seconds_remaining", "posteam", "home_team", "away_team",
            "total_home_score", "total_away_score", "posteam_score", "defteam_score", "yardline_100", "play_type",
            "qb_kneel", "qb_spike", "two_point_attempt", "pass_attempt", "sack", "complete_pass", "qb_scramble",
            "rush_attempt", "passer_player_id", "receiver_player_id", "rusher_player_id", "passing_yards",
            "receiving_yards", "rushing_yards", "home_score", "away_score"]
    p = pl.read_parquet(RAW / "pbp" / "2026.parquet", columns=cols).filter(pl.col("game_id").is_in(gids))
    return p.with_columns(ts=pl.col("time_of_day").str.to_datetime(time_zone="UTC", strict=False)).sort("game_id", "play_id")


def _score_at(r: dict, home: str, away: str) -> dict:
    """Score at the start of play r, by team."""
    a, b = r["posteam_score"] or 0, r["defteam_score"] or 0
    return {home: a, away: b} if r["posteam"] == home else {home: b, away: a}


def _times(t_from, t_next, first: float, gap: float):
    """Two price checks inside a break, squeezed in if the break is short; None if there is no room."""
    t1, t2 = t_from + timedelta(seconds=first), t_from + timedelta(seconds=first + gap)
    if t2 < t_next - timedelta(seconds=5):
        return t1, t2
    span = (t_next - t_from).total_seconds()
    if span < 40:
        return None
    return t_from + timedelta(seconds=0.4 * span), t_from + timedelta(seconds=0.85 * span)


def _result(side: str, line: float, y: float) -> str:
    d = y - line if side in ("over", "home") else line - y
    return "W" if d > 0 else ("P" if d == 0 else "L")


def game_points(gid: str, week: int, plays: pl.DataFrame, priors: pl.DataFrame, prop_rows: pl.DataFrame,
                game_rows: pl.DataFrame | None, close: dict | None, lp: LiveParams, disp) -> tuple[list, list, list]:
    """Every candidate price at both checks of every decision point, plus the observations s is fitted on."""
    home, away = plays["home_team"][0], plays["away_team"][0]
    model = LiveModel(priors, lp, Simulator(disp, seed=zlib.crc32(gid.encode())), size=3000)
    if close:
        model.calibrate(close)
    info = {r["gsis_id"]: {"name": r["name"], "team": r["team"], "position": r["position"]} for r in priors.iter_rows(named=True)}
    prior_share = {r["gsis_id"]: (r["tshare"], r["rshare"]) for r in priors.iter_rows(named=True)}
    banked, team_so = _banked(plays)
    allp = plays.to_dicts()
    scrim = [i for i, r in enumerate(allp) if r["play_type"] in ("pass", "run") and r["ts"] is not None]
    final = _final_stats(plays)
    hs, as_ = allp[-1]["home_score"], allp[-1]["away_score"]
    margin_final, total_final = hs - as_, hs + as_

    events = []
    for name, B in WINDOWS.items():   # props: the breaks
        before = [i for i in scrim if (allp[i]["game_seconds_remaining"] or 0) > B]
        after = [i for i in scrim if (allp[i]["game_seconds_remaining"] or 0) <= B]
        if not before or not after:
            continue
        i_last, i_next = before[-1], after[0]
        tt = _times(allp[i_last]["ts"], allp[i_next]["ts"], FEED_LAG * (2 if name == "half" else 1),
                    PERSIST * (2 if name == "half" else 1))
        if tt:
            nxt = allp[i_next]
            events.append({"kind": "prop", "name": name, "t1": tt[0], "t2": tt[1], "i": i_next - 1,
                           "state": {"seconds_left": B, "score": _score_at(nxt, home, away), "possession": nxt["posteam"],
                                     "yards_to_endzone": nxt["yardline_100"]}})
    by_next: dict[int, int] = {}       # game markets: after each score, before the next snap
    prev = (0, 0)
    for i, r in enumerate(allp):
        cur = (r["total_home_score"] or 0, r["total_away_score"] or 0)
        if cur != prev and r["ts"] is not None:
            j = next((j for j in scrim if j > i), None)
            if j is not None:
                by_next[j] = i          # the last scoring play before that snap
        prev = cur
    for j, i in by_next.items():
        nxt = allp[j]
        if (nxt["game_seconds_remaining"] or 0) < 60:
            continue
        tt = _times(allp[i]["ts"], nxt["ts"], 90, PERSIST)
        if tt:
            events.append({"kind": "game", "name": f"score@{int(nxt['game_seconds_remaining'])}", "t1": tt[0], "t2": tt[1],
                           "i": j - 1, "state": {"seconds_left": float(nxt["game_seconds_remaining"]),
                                                 "score": _score_at(nxt, home, away), "possession": nxt["posteam"],
                                                 "yards_to_endzone": nxt["yardline_100"]}})
    if not events:
        return [], [], []
    times = []
    for e in events:
        times += [e["t1"], e["t2"], e["t2"] + timedelta(seconds=FILL_LAG), e["t2"] + timedelta(seconds=CLV_LAG)]
    pgrid = _prop_grid(prop_rows, times) if prop_rows.height else {}
    ggrid = _game_grid(game_rows, times) if game_rows is not None and game_rows.height else {}

    cands, obs_p, obs_g = [], [], []
    for n, e in enumerate(events):
        ti = {"t1": 4 * n, "t2": 4 * n + 1}
        fill_i, clv_i = 4 * n + 2, 4 * n + 3
        st, b, tso = e["state"], banked[e["i"]], team_so[e["i"]]
        lead = st["score"][home] - st["score"][away]
        base = {"game_id": gid, "week": week, "kind": e["kind"], "event": e["name"], "at": e["t2"].isoformat(),
                "secs_left": st["seconds_left"], "lead": lead}
        if e["kind"] == "prop":
            proj = model.project(st, b, tso)
            for when, k in ti.items():
                lines = pgrid.get(k, {})
                if when == "t2":    # observations for fitting s: every main line at the break
                    for (pid, m), books in lines.items():
                        s = proj["players"].get(pid, {}).get(m)
                        y = final.get(pid, {}).get(m)
                        if s is None or y is None:
                            continue
                        for bk, L in books.items():
                            if y != L["line"]:
                                obs_p.append({"week": week, "fair": decide.no_vig(L["over"], L["under"]),
                                              "model": float((s > L["line"]).mean()), "y": float(y > L["line"])})
                for c in decide.prop_candidates(gid, proj, lines, info, None):
                    row = {**base, "check": when, "key": c.key, "side": c.side, "book": c.book, "line": c.line, "dec": c.dec,
                           "p": c.p, "push": c.push, "fair": c.fair, "group": c.group, "player": c.player, "market": c.market,
                           "label": c.label(), "team": c.team}
                    if when == "t2":
                        later = pgrid.get(fill_i, {}).get((c.player, c.market), {}).get(c.book)
                        five = pgrid.get(clv_i, {}).get((c.player, c.market), {}).get(c.book)
                        y = final.get(c.player, {}).get(c.market)
                        row.update(fill_line=later["line"] if later else None,
                                   fill_dec=(later["over"] if c.side == "over" else later["under"]) if later else None,
                                   clv=None if five is None else (five["line"] - c.line if c.side == "over" else c.line - five["line"]),
                                   result=None if y is None else _result(c.side, c.line, y))
                        if c.p * (c.dec - 1) - (1 - c.p - c.push) > 0:
                            pb, i_sh = proj["players"][c.player], 1 if c.market == "rush_yds" else 0
                            other = away if c.team == home else home
                            ctx = {"seconds_left": st["seconds_left"], "rem_mean": pb["_rem_mean"], "share_now": pb["_share"][i_sh],
                                   "prior_share": prior_share[c.player][i_sh],
                                   "banked": {"receptions": b.get(c.player, {}).get("rec", 0), "rec_yds": b.get(c.player, {}).get("rec_yds", 0),
                                              "rush_yds": b.get(c.player, {}).get("rush_yds", 0), "pass_yds": b.get(c.player, {}).get("pass_yds", 0)},
                                   "team_lead": st["score"][c.team] - st["score"][other]}
                            row["reason"] = decide.reason(c, ctx)
                    cands.append(row)
        else:
            gm = model.game(st)
            for when, k in ti.items():
                L = ggrid.get(k, {})
                for c in decide.game_candidates(gid, gm, L, home, away):
                    f = _fair(L, c.market, c.side, c.line)
                    if f is None:
                        continue
                    if c.market == "ml":
                        res = "P" if margin_final == 0 else ("W" if (margin_final > 0) == (c.side == "home") else "L")
                    elif c.market == "spread":   # the side's own handicap: home -3.5 covers when the margin beats 3.5
                        d = (margin_final if c.side == "home" else -margin_final) + c.line
                        res = "W" if d > 0 else ("P" if d == 0 else "L")
                    else:
                        res = _result(c.side, c.line, total_final)
                    if when == "t2" and c.side in ("home", "over") and res != "P":
                        obs_g.append({"week": week, "fair": f, "model": c.p, "y": float(res == "W")})
                    row = {**base, "check": when, "key": c.key, "side": c.side, "book": c.book, "line": c.line, "dec": c.dec,
                           "p": c.p, "push": c.push, "fair": f, "group": c.group, "player": None, "market": c.market,
                           "label": c.label(), "team": c.team, "result": res}
                    if when == "t2":
                        later = ggrid.get(fill_i, {}).get(c.book, {})
                        five = ggrid.get(clv_i, {})
                        ok = False
                        if c.market == "ml" and "ml_home" in later:
                            ok = (later["ml_home"] if c.side == "home" else later["ml_away"]) >= c.dec - 1e-9
                        elif c.market == "spread" and "spread" in later:
                            ok = (-later["spread"] if c.side == "home" else later["spread"]) == c.line
                        elif c.market == "total" and "total" in later:
                            ok = later["total"] == c.line
                        f5 = _fair(five, c.market, c.side, c.line)
                        row.update(fill_line=c.line if ok else None, fill_dec=c.dec if ok else None,
                                   clv=None if f5 is None else round(f5 - f, 4))
                        lead_ = int(st["score"][home] - st["score"][away])
                        row["reason"] = (f"model {c.p:.0%} vs Pinnacle {f:.0%}; {home} {lead_:+d} with "
                                         f"{decide._mins(st['seconds_left'])} left, rest played from team strength")
                    cands.append(row)
    return cands, obs_p, obs_g


def fit_s(obs: list[dict], weeks: list[int]) -> dict[int, float]:
    """s for each week, fitted on the other weeks: the shrink toward the market that gives the
    best Brier score. s=0 is the market's number, s=1 the model's own."""
    out = {}
    for w in weeks:
        O = [o for o in obs if o["week"] != w]
        if not O:
            out[w] = 0.0
            continue
        f = np.array([o["fair"] for o in O])
        m = np.array([o["model"] for o in O])
        y = np.array([o["y"] for o in O])
        brier = [float((((f + s * (m - f)) - y) ** 2).mean()) for s in S_GRID]
        out[w] = float(S_GRID[int(np.argmin(brier))])
    return out


def select(C: pl.DataFrame, s_by_week: dict, events: tuple | None, kind: str, cap_used: dict | None = None,
           max_gap: float | None = None, price_cap: float | None = PRICE_CAP, per_window: int | None = None,
           per_game: int | None = None) -> list[dict]:
    """Apply the rubric at each decision point, in time order. events: which prop windows (None: game markets)."""
    if not C.height:
        return []
    C = C.filter(pl.col("kind") == kind)
    if events is not None:
        C = C.filter(pl.col("event").is_in(list(events)))
    if not C.height:
        return []
    C = C.with_columns(s=pl.col("week").replace_strict(s_by_week, return_dtype=pl.Float64))
    C = C.with_columns(p_adj=pl.col("fair") + pl.col("s") * (pl.col("p") - pl.col("fair")))
    C = C.with_columns(ev_adj=pl.col("p_adj") * (pl.col("dec") - 1) - (1 - pl.col("p_adj") - pl.col("push")))
    ok = (pl.col("ev_adj") >= decide.MIN_EV) & ((pl.col("p") - pl.col("fair")).abs() <= (max_gap or decide.MAX_GAP))
    if kind == "prop":
        ok &= pl.col("lead").abs() <= BLOWOUT
    elif price_cap:
        ok &= pl.col("dec").is_between(1 + 100 / price_cap, 1 + price_cap / 100)
    C = C.filter(ok)
    pk = ["game_id", "event", "key", "side", "book", "line"]
    first = C.filter(pl.col("check") == "t1").select(pk).unique()
    C2 = C.filter(pl.col("check") == "t2").join(first, on=pk, how="semi").sort("game_id", "at", pl.col("ev_adj"), descending=[False, False, True])
    flags, used = [], cap_used if cap_used is not None else {}
    per_window, per_game = per_window or PER_WINDOW, per_game or EXPLORE_PER_GAME
    for (gid, ev), grp in C2.group_by(["game_id", "event"], maintain_order=True):
        taken = [f for f in flags if f["game_id"] == gid]
        keys, players, groups = {f["key"] for f in taken}, {f["player"] for f in taken if f["player"]}, {f["group"] for f in taken}
        n_win = 0
        for r in grp.iter_rows(named=True):
            if used.get(gid, 0) >= per_game or n_win >= per_window:
                break
            if r["key"] in keys or (r["player"] and r["player"] in players) or r["group"] in groups:
                continue
            flags.append(r)
            keys.add(r["key"]), groups.add(r["group"])
            if r["player"]:
                players.add(r["player"])
            used[gid] = used.get(gid, 0) + 1
            n_win += 1
    return flags


def score(flags: list[dict]) -> dict:
    """Every flag counts at its flagged price."""
    if not flags:
        return {"flags": 0}
    F = pl.DataFrame(flags)
    g = F.filter(pl.col("result").is_in(["W", "L", "P"]))
    u = g.with_columns(u=pl.when(pl.col("result") == "W").then(pl.col("dec") - 1)
                       .when(pl.col("result") == "L").then(-1.0).otherwise(0.0))["u"]
    wl = g.filter(pl.col("result") != "P")
    clv = F.filter(pl.col("clv").is_not_null())
    moved = clv.filter(pl.col("clv") != 0)
    return {"flags": F.height, "w": int((g["result"] == "W").sum()), "l": int((g["result"] == "L").sum()),
            "p": int((g["result"] == "P").sum()), "units": round(float(u.sum()), 2),
            "roi": round(float(u.mean()), 4) if g.height else None,
            "said": round(float(wl["p_adj"].mean()), 3) if wl.height else None,
            "won": round(float((wl["result"] == "W").mean()), 3) if wl.height else None,
            "clv": round(float(clv["clv"].mean()), 3) if clv.height else None,
            "clv_beat": round(float((moved["clv"] > 0).mean()), 3) if moved.height else None,
            "still_there": round(float(F["fill_dec"].is_not_null().mean()), 3),
            "avg_ev": round(float(F["ev_adj"].mean()), 4)}


def reference() -> dict:
    """The rule the engine ran on 2026-10-08 (any play after Q1, the model's own chances), every flag counted."""
    out = {}
    for tag, path in (("props", REPORTS / "live_replay_flags.parquet"), ("game", REPORTS / "live_replay_games_flags.parquet")):
        if path.exists():
            F = pl.read_parquet(path).filter(pl.col("result").is_in(["W", "L", "P"]))
            u = F.with_columns(u=pl.when(pl.col("result") == "W").then(pl.col("dec") - 1)
                               .when(pl.col("result") == "L").then(-1.0).otherwise(0.0))["u"]
            wl = F.filter(pl.col("result") != "P")
            out[tag] = {"flags": F.height, "w": int((F["result"] == "W").sum()), "l": int((F["result"] == "L").sum()),
                        "p": int((F["result"] == "P").sum()), "units": round(float(u.sum()), 2), "roi": round(float(u.mean()), 4),
                        "said": round(float(wl["p"].mean()), 3), "won": round(float((wl["result"] == "W").mean()), 3)}
    return out


def engine_rule(C: pl.DataFrame, weeks: list[int]) -> tuple[dict, list[dict]]:
    """The rule the engine runs (decide.py): props at its windows, one play, after the big-play and
    still-on-field checks (live/rubric_checks.py); game markets after scores, one."""
    from live import rubric_checks
    raw = {w: 1.0 for w in weeks}
    for w in decide.PROP_WINDOWS:
        gids = C.filter((pl.col("kind") == "prop") & (pl.col("event") == w))["game_id"].unique().to_list()
        C = rubric_checks.apply(C, *rubric_checks.flags_from_pbp(gids, decide.WINDOW_AT[w]), event=w)
    fp = select(C, raw, decide.PROP_WINDOWS, "prop", max_gap=decide.MAX_GAP,
                per_window=decide.PROP_PER_GAME, per_game=decide.PROP_PER_GAME)
    fg = select(C, raw, None, "game", max_gap=decide.MAX_GAP, per_window=decide.GAME_PER_GAME, per_game=decide.GAME_PER_GAME)
    by_week = {}
    for w in weeks:
        by_week[w] = score([f for f in fp if f["week"] == w])
    return {"props": score(fp), "game": score(fg), "all": score(fp + fg), "props_by_week": by_week,
            "windows": [decide.WINDOW_LABEL[w] for w in decide.PROP_WINDOWS], "max_gap": decide.MAX_GAP,
            "prop_per_game": decide.PROP_PER_GAME, "game_per_game": decide.GAME_PER_GAME}, fp + fg


def gap_bands(C: pl.DataFrame, weeks: list[int]) -> dict:
    """Why the rubric caps the disagreement: props at the first quarter and halftime, model 6-10 vs 10-15
    points from the market."""
    fl = select(C, {w: 1.0 for w in weeks}, ("q1", "half"), "prop", max_gap=0.15)
    return {"moderate": score([f for f in fl if abs(f["p"] - f["fair"]) < 0.10]),
            "big": score([f for f in fl if abs(f["p"] - f["fair"]) >= 0.10])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--from-saved", action="store_true", help="re-score the saved candidate prices (seconds, no simulation)")
    args = ap.parse_args()
    if args.from_saved:
        C = pl.read_parquet(CANDS)
        out = json.loads(OUT.read_text())
        out["engine"], fl = engine_rule(C, out["weeks"])
        out["gap_bands"] = gap_bands(C, out["weeks"])
        OUT.write_text(json.dumps(out, indent=1, default=str))
        pl.DataFrame([{**f, "variant": "engine"} for f in fl], infer_schema_length=None).write_parquet(FLAGS)
        print("engine rule:", json.dumps(out["engine"], indent=1))
        return
    replay_mod.STALE_S = STALE_BREAK
    replay_games_mod.STALE_S = STALE_BREAK
    from props.backtest import load as load_props_params
    from props.section import _frames_cached
    p, disp = load_props_params()
    fr = _frames_cached(p).filter((pl.col("season") == 2026) & pl.col("role").is_not_null())
    games = pl.read_parquet(RAW.parent / "clean" / "games.parquet").filter((pl.col("season") == 2026) & pl.col("result").is_not_null())
    gids = sorted(set(games["game_id"]) & set(fr["game_id"]))
    if args.games:
        gids = gids[: args.games]
    week_of = dict(zip(games["game_id"], games["week"]))
    P = load_props(gids, pregame_only=False)
    P_live = P.filter((pl.col("ts") >= pl.col("kickoff")) & pl.col("book").is_in(["draftkings", "fanduel"]))
    closing: dict = {}
    for x in main_lines(P.filter(pl.col("ts") < pl.col("kickoff")), pl.col("kickoff") - timedelta(minutes=5)).iter_rows(named=True):
        closing.setdefault(x["game_id"], {}).setdefault((x["gsis_id"], x["market"]), {})[x["book"]] = {
            "line": x["line"], "over": x["over"], "under": x["under"]}
    fx = dict(fixture_index().select("game_id", "fixture_id").iter_rows())
    pbp = _plays(gids)
    lp = LiveParams.load()
    cands, obs_p, obs_g = [], [], []
    for n, gid in enumerate(gids):
        plays = pbp.filter(pl.col("game_id") == gid)
        if plays["ts"].null_count() == plays.height:
            continue
        kick = games.filter(pl.col("game_id") == gid)["kickoff_utc"][0]
        path = HIST / f"{fx.get(gid)}.parquet"
        grows = pl.read_parquet(path).filter(pl.col("ts") >= kick) if gid in fx and path.exists() else None
        c, op, og = game_points(gid, int(week_of[gid]), plays, fr.filter(pl.col("game_id") == gid),
                                P_live.filter(pl.col("game_id") == gid), grows, closing.get(gid), lp, disp)
        cands += c
        obs_p += op
        obs_g += og
        print(f"{n + 1}/{len(gids)} {gid}: {len(c)} candidate prices")
    C = pl.DataFrame(cands, infer_schema_length=None)
    C.write_parquet(CANDS)
    weeks = sorted({int(w) for w in C["week"].unique()})
    s_p, s_g = fit_s(obs_p, weeks), fit_s(obs_g, weeks)
    print("s (props), fitted on the other weeks:", s_p, "| s (game markets):", s_g)
    raw = {w: 1.0 for w in weeks}
    out = {"games": len(gids), "weeks": weeks, "s_props": s_p, "s_game": s_g, "obs": {"props": len(obs_p), "game": len(obs_g)},
           "props": {}, "game": {}, "reference": reference(),
           "rules": {"min_ev": decide.MIN_EV, "max_gap": decide.MAX_GAP, "per_game": decide.MAX_PER_GAME, "per_window": PER_WINDOW,
                     "blowout": BLOWOUT, "price_cap": PRICE_CAP, "persist_s": PERSIST, "feed_lag_s": FEED_LAG}}
    keep = {}
    for name, wins in VARIANTS.items():
        for shrink, sw in (("shrunk", s_p), ("raw", raw)):
            for gap in GAPS:
                if shrink == "shrunk" and gap != GAPS[0]:
                    continue
                tag = f"{name}|{shrink}|gap{int(100 * gap)}"
                fl = select(C, sw, wins, "prop", max_gap=gap)
                out["props"][tag] = score(fl)
                keep[f"props|{tag}"] = fl
                print(f"props {tag:28s}", out["props"][tag])
    for shrink, sw in (("shrunk", s_g), ("raw", raw)):
        for gap in GAPS:
            for cap in (PRICE_CAP, None):
                if shrink == "shrunk" and (gap != GAPS[0] or cap is None):
                    continue
                tag = f"{shrink}|gap{int(100 * gap)}|{'cap' + str(cap) if cap else 'no cap'}"
                fl = select(C, sw, None, "game", max_gap=gap, price_cap=cap)
                out["game"][tag] = score(fl)
                keep[f"game|{tag}"] = fl
                print(f"game markets {tag:24s}", out["game"][tag])
    out["engine"], fl = engine_rule(C, weeks)
    out["gap_bands"] = gap_bands(C, weeks)
    print("engine rule:", out["engine"]["props"], out["engine"]["game"])
    REPORTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=str))
    if fl:
        pl.DataFrame([{**f, "variant": "engine"} for f in fl], infer_schema_length=None).write_parquet(FLAGS)
    print("saved", OUT)


if __name__ == "__main__":
    main()
