"""Replay the live engine over finished 2026 games against the real in-game prices.

    python -m live.replay            # every finished 2026 game in the odds archive
    python -m live.replay --games 8  # a quick look

The archive holds every in-game price change at DraftKings and FanDuel (props
and game lines). For each game, plays come from nflverse play-by-play, each
stamped with its real time. After every play the engine sees the state the
way the live feed would, DECISION_LAG seconds after the snap, prices what the
books showed at that moment, and flags at most three plays a game. Then:

- fill: the paper bet counts only if the same line was still up FILL_LAG
  seconds later (about how long it takes to place a bet) at a price that
  still had positive expected value; otherwise it is "missed";
- CLV: where the no-vig market priced our side of that same line five
  minutes later; the market moving our way is the early sign of a real edge;
- result: graded against the final stat line.

It also scores the thesis directly: are in-game prop lines calibrated, and is
the model's chance better or worse than the market's (Brier score)?

Writes reports/live_replay.json and reports/live_replay_flags.parquet.
"""
import argparse
import json
import warnings
import zlib
from datetime import timedelta

import numpy as np
import polars as pl

from config import RAW, REPORTS
from live import decide
from live.model import LiveModel, LiveParams
from props.lines import load as load_props
from props.model import Simulator

warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")
DECISION_LAG = 26      # seconds after the snap: the play, then the feed's delay
FILL_LAG = 30          # seconds to place the bet
CLV_LAG = 300          # where the market sits five minutes later
STALE_S = 300          # a line not updated for five minutes in-game is treated as gone
OUT = REPORTS / "live_replay.json"
FLAGS = REPORTS / "live_replay_flags.parquet"


def _pbp(game_ids: list[str]) -> pl.DataFrame:
    cols = ["game_id", "play_id", "time_of_day", "game_seconds_remaining", "posteam", "home_team", "away_team",
            "total_home_score", "total_away_score", "yardline_100", "play_type", "qb_kneel", "qb_spike",
            "two_point_attempt", "pass_attempt", "sack", "complete_pass", "qb_scramble", "rush_attempt",
            "passer_player_id", "receiver_player_id", "rusher_player_id", "passing_yards", "receiving_yards",
            "rushing_yards", "home_score", "away_score"]
    p = pl.read_parquet(RAW / "pbp" / "2026.parquet", columns=cols).filter(pl.col("game_id").is_in(game_ids))
    return p.with_columns(ts=pl.col("time_of_day").str.to_datetime(time_zone="UTC", strict=False)).sort("game_id", "play_id")


def _banked(plays: pl.DataFrame) -> tuple[list[dict], list[dict]]:
    """Per play: cumulative player stats and team usage through that play."""
    banked, team = {}, {}
    out_b, out_t = [], []
    for r in plays.iter_rows(named=True):
        scrim = r["play_type"] in ("pass", "run") and not r["qb_kneel"] and not r["qb_spike"] and not r["two_point_attempt"]
        t = r["posteam"]
        if scrim and t:
            ts = team.setdefault(t, {"plays": 0, "tgt": 0, "runs": 0, "att": 0})
            ts["plays"] += 1
            if r["pass_attempt"] == 1 and not r["sack"]:
                ts["att"] += 1
                if r["receiver_player_id"]:
                    ts["tgt"] += 1
                    b = banked.setdefault(r["receiver_player_id"], {})
                    b["tgt"] = b.get("tgt", 0) + 1
                    if r["complete_pass"] == 1:
                        b["rec"] = b.get("rec", 0) + 1
                        b["rec_yds"] = b.get("rec_yds", 0) + (r["receiving_yards"] or 0)
                if r["passer_player_id"]:
                    b = banked.setdefault(r["passer_player_id"], {})
                    b["att"] = b.get("att", 0) + 1
                    b["pass_yds"] = b.get("pass_yds", 0) + (r["passing_yards"] or 0)
            if r["rush_attempt"] == 1 and r["rusher_player_id"]:
                b = banked.setdefault(r["rusher_player_id"], {})
                b["car"] = b.get("car", 0) + 1
                b["rush_yds"] = b.get("rush_yds", 0) + (r["rushing_yards"] or 0)
                if not r["qb_scramble"]:
                    ts["runs"] += 1
                    b["car_designed"] = b.get("car_designed", 0) + 1
        out_b.append({k: dict(v) for k, v in banked.items()})
        out_t.append({k: dict(v) for k, v in team.items()})
    return out_b, out_t


def _prop_grid(rows: pl.DataFrame, times: list) -> dict[int, dict]:
    """Main prop line per (time, player, market, book) from in-game price points (as-of join)."""
    if not rows.height or not times:
        return {}
    k = ["book", "market", "gsis_id", "handicap", "outcome"]
    keys = rows.select(k).unique()
    grid = keys.join(pl.DataFrame({"t": times, "ti": list(range(len(times)))}).with_columns(
        pl.col("t").dt.cast_time_unit("us")), how="cross").sort("t")
    r = rows.select(k + ["ts", "price", "active"]).sort("ts")
    j = grid.join_asof(r, left_on="t", right_on="ts", by=k, strategy="backward").filter(
        pl.col("active") & ((pl.col("t") - pl.col("ts")).dt.total_seconds() <= STALE_S))
    o = j.filter(pl.col("outcome") == "Over").select("ti", "book", "market", "gsis_id", "handicap", over="price")
    u = j.filter(pl.col("outcome") == "Under").select("ti", "book", "market", "gsis_id", "handicap", under="price")
    pr = (o.join(u, on=["ti", "book", "market", "gsis_id", "handicap"])
          .with_columns(gap=(1 / pl.col("over") - 1 / pl.col("under")).abs()).sort("gap", "handicap")
          .group_by("ti", "book", "market", "gsis_id", maintain_order=True).first())
    out: dict[int, dict] = {}
    for x in pr.iter_rows(named=True):
        out.setdefault(x["ti"], {}).setdefault((x["gsis_id"], x["market"]), {})[x["book"]] = {
            "line": x["handicap"], "over": x["over"], "under": x["under"]}
    return out


def _same_line(grid_t: dict, pid: str, market: str, book: str, line: float) -> dict | None:
    L = (grid_t or {}).get((pid, market), {}).get(book)
    return L if L and L["line"] == line else None


def replay_game(gid: str, plays: pl.DataFrame, priors: pl.DataFrame, prop_rows: pl.DataFrame,
                game_rows: pl.DataFrame | None, sim: Simulator, lp: LiveParams, final_stats: dict,
                k: float | None = None, close_books: dict | None = None) -> tuple[list, list]:
    home, away = plays["home_team"][0], plays["away_team"][0]
    model = LiveModel(priors, lp, sim, size=3000)
    if close_books:
        model.calibrate(close_books)
    info = {r["gsis_id"]: {"name": r["name"], "team": r["team"], "position": r["position"]} for r in priors.iter_rows(named=True)}
    prior_share = {r["gsis_id"]: (r["tshare"], r["rshare"]) for r in priors.iter_rows(named=True)}
    scrim = plays.filter(pl.col("play_type").is_in(["pass", "run"]) & pl.col("ts").is_not_null())
    banked, team_so = _banked(plays)
    idx = {pid: i for i, pid in enumerate(plays["play_id"].to_list())}
    rows = scrim.to_dicts()
    times = [r["ts"] + timedelta(seconds=DECISION_LAG) for r in rows]
    grid = _prop_grid(prop_rows, times + [t + timedelta(seconds=FILL_LAG) for t in times]
                      + [t + timedelta(seconds=CLV_LAG) for t in times])
    n = len(times)
    flags, obs = [], []
    for i, r in enumerate(rows[:-1]):
        nxt = rows[i + 1]
        secs = float(nxt["game_seconds_remaining"] or 0)
        if secs <= 0:
            break
        state = {"seconds_left": secs, "score": {home: nxt["total_home_score"], away: nxt["total_away_score"]},
                 "possession": nxt["posteam"], "yards_to_endzone": nxt["yardline_100"]}
        k = idx[r["play_id"]]
        b, ts_ = banked[k], team_so[k]
        lines_now = grid.get(i, {})
        if not lines_now:
            continue
        proj = model.project(state, b, ts_)
        # thesis check: market vs model at each main line, every 8th play
        if i % 8 == 0:
            later_all = grid.get(2 * n + i, {})
            for (pid, m), books in lines_now.items():
                s = proj["players"].get(pid, {}).get(m)
                y = final_stats.get(pid, {}).get(m)
                if s is None or y is None:
                    continue
                if (pid, m) in later_all:   # does the market move toward the model in the next five minutes?
                    sd = float(s.std()) or 1.0
                    d0, d1 = decide.anchor_shift(s, books), decide.anchor_shift(s, later_all[(pid, m)])
                    obs.append({"kind": "move", "gap": -d0 / sd, "move": (d1 - d0) / sd})
                for bk, L in books.items():
                    if y == L["line"]:
                        continue
                    obs.append({"kind": "cal", "game_id": gid, "secs": secs, "market": m, "book": bk, "fair": decide.no_vig(L["over"], L["under"]),
                                "model": float((s > L["line"]).mean()), "over": float(y > L["line"])})
        if len(flags) >= decide.MAX_PER_GAME or secs > decide.EARLIEST_SECS:
            continue
        cands = decide.prop_candidates(gid, proj, lines_now, info, k)
        new = decide.select(cands, flags)
        for c in new:
            # Fill: 30 seconds later the same line at a price that still has value, or a line
            # that moved our way (taken at that line). Anything else is a missed bet.
            later = (grid.get(n + i, {}) or {}).get((c.player, c.market), {}).get(c.book)
            fill_line, fill_dec = None, None
            if later:
                dec2 = later["over"] if c.side == "over" else later["under"]
                better = later["line"] < c.line if c.side == "over" else later["line"] > c.line
                if later["line"] == c.line and (dec2 >= c.dec - 1e-9 or c.p * (dec2 - 1) - (1 - c.p - c.push) > 0):
                    fill_line, fill_dec = c.line, min(dec2, c.dec)
                elif better:
                    fill_line, fill_dec = later["line"], dec2
            # Line value: how far the book's main line moved our way in the next five minutes.
            five = (grid.get(2 * n + i, {}) or {}).get((c.player, c.market), {}).get(c.book)
            clv_units = None if five is None else (five["line"] - c.line if c.side == "over" else c.line - five["line"])
            fair_later = None
            if five and five["line"] == c.line:
                fo = decide.no_vig(five["over"], five["under"])
                fair_later = fo if c.side == "over" else 1 - fo
            y = final_stats.get(c.player, {}).get(c.market)
            gl = fill_line if fill_line is not None else c.line
            res = None if y is None else ("W" if (y > gl if c.side == "over" else y < gl) else ("P" if y == gl else "L"))
            pb = proj["players"][c.player]
            ctx = {"seconds_left": secs, "banked": {"receptions": b.get(c.player, {}).get("rec", 0),
                                                     "rec_yds": b.get(c.player, {}).get("rec_yds", 0),
                                                     "rush_yds": b.get(c.player, {}).get("rush_yds", 0),
                                                     "pass_yds": b.get(c.player, {}).get("pass_yds", 0)},
                   "rem_mean": pb["_rem_mean"], "share_now": pb["_share"][1 if c.market == "rush_yds" else 0],
                   "prior_share": prior_share[c.player][1 if c.market == "rush_yds" else 0],
                   "team_lead": state["score"][c.team] - state["score"][away if c.team == home else home]}
            c.reason = decide.reason(c, ctx)
            flags.append({"key": c.key, "player": c.player, "group": c.group, "game_id": gid, "at": times[i].isoformat(),
                          "label": c.label(), "market": c.market, "side": c.side, "line": c.line, "book": c.book,
                          "dec": c.dec, "american": decide.american(c.dec), "p": round(c.p, 4), "fair": round(c.fair, 4),
                          "ev": round(c.ev, 4), "reason": c.reason, "secs_left": secs,
                          "filled": fill_dec is not None, "fill_dec": fill_dec, "fill_line": fill_line,
                          "clv_units": clv_units, "fair_later": round(fair_later, 4) if fair_later is not None else None,
                          "actual": y, "result": res})
    return flags, obs


def _final_stats(plays: pl.DataFrame) -> dict[str, dict]:
    b, _ = _banked(plays)
    last = b[-1] if b else {}
    return {pid: {"receptions": v.get("rec", 0), "rec_yds": v.get("rec_yds", 0), "rush_yds": v.get("rush_yds", 0),
                  "pass_yds": v.get("pass_yds", 0)} for pid, v in last.items()}


def summarize(F: pl.DataFrame) -> dict:
    if not F.height:
        return {"flags": 0}
    filled = F.filter(pl.col("filled") & pl.col("result").is_not_null())
    units = filled.with_columns(u=pl.when(pl.col("result") == "W").then(pl.col("fill_dec") - 1)
                                .when(pl.col("result") == "L").then(-1.0).otherwise(0.0))["u"]
    clv = F.filter(pl.col("clv_units").is_not_null())
    moved = clv.filter(pl.col("clv_units") != 0)
    return {"flags": F.height, "filled": filled.height, "missed": F.height - F["filled"].sum(),
            "w": int((filled["result"] == "W").sum()), "l": int((filled["result"] == "L").sum()),
            "p": int((filled["result"] == "P").sum()), "units": round(float(units.sum()), 2),
            "roi": round(float(units.mean()), 4) if filled.height else None,
            "avg_ev": round(float(F["ev"].mean()), 4),
            "clv_n": clv.height, "clv_units": round(float(clv["clv_units"].mean()), 2) if clv.height else None,
            "clv_moved": moved.height, "clv_beat": round(float((moved["clv_units"] > 0).mean()), 3) if moved.height else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--k", type=float, default=None, help="anchor to the market and tilt k toward the model (omit: model's own chances)")
    ap.add_argument("--min-ev", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-calibrate", action="store_true", help="skip starting each game at the closing lines")
    ap.add_argument("--earliest", type=float, default=None, help="no flags until this many seconds are left (2700 = after Q1)")
    args = ap.parse_args()
    if args.min_ev is not None:
        decide.MIN_EV = args.min_ev
    if args.earliest is not None:
        decide.EARLIEST_SECS = args.earliest
    from props.section import _frames_cached
    from props.backtest import load as load_props_params
    p, disp = load_props_params()
    fr = _frames_cached(p).filter((pl.col("season") == 2026) & pl.col("role").is_not_null())
    games = pl.read_parquet(RAW.parent / "clean" / "games.parquet").filter((pl.col("season") == 2026) & pl.col("result").is_not_null())
    gids = sorted(set(games["game_id"]) & set(fr["game_id"]))
    if args.games:
        gids = gids[: args.games]
    P = load_props(gids, pregame_only=False)
    P = P.filter((pl.col("ts") >= pl.col("kickoff")) & pl.col("book").is_in(["draftkings", "fanduel"]))
    pbp = _pbp(gids)
    sim, lp = Simulator(disp, seed=31), LiveParams.load()
    closing = {}
    if not args.no_calibrate:   # each player's closing prop lines, to start every game agreeing with the market
        from props.lines import main_lines
        pre = load_props(gids, pregame_only=True)
        for x in main_lines(pre, pl.col("kickoff") - timedelta(minutes=5)).iter_rows(named=True):
            closing.setdefault(x["game_id"], {}).setdefault((x["gsis_id"], x["market"]), {})[x["book"]] = {
                "line": x["line"], "over": x["over"], "under": x["under"]}
    all_flags, all_obs = [], []
    for n, gid in enumerate(gids):
        plays = pbp.filter(pl.col("game_id") == gid)
        if plays["ts"].null_count() == plays.height:
            continue
        # each game gets its own fixed seed, so a result never depends on which games ran before it
        sim = Simulator(disp, seed=zlib.crc32(gid.encode()))
        fl, ob = replay_game(gid, plays, fr.filter(pl.col("game_id") == gid), P.filter(pl.col("game_id") == gid), None,
                             sim, lp, _final_stats(plays), args.k, closing.get(gid))
        all_flags += fl
        all_obs += ob
        print(f"{n + 1}/{len(gids)} {gid}: {len(fl)} flags")
    F = pl.DataFrame(all_flags) if all_flags else pl.DataFrame()
    MV = pl.DataFrame([o for o in all_obs if o["kind"] == "move"])
    O = pl.DataFrame([{k_: v for k_, v in o.items() if k_ != "kind"} for o in all_obs if o["kind"] == "cal"])
    s = summarize(F)
    k_live = None
    if MV.height:
        g, m = MV["gap"].to_numpy(), MV["move"].to_numpy()
        moved = m != 0
        k_live = {"k": round(float(np.clip((g * m).sum() / (g ** 2).sum(), 0, 0.5)), 4), "n": int(len(g)),
                  "moved": int(moved.sum()), "toward": int((np.sign(g[moved]) == np.sign(m[moved])).sum())}
        print("in-game lines moving toward the model within 5 minutes:", k_live)
    s["k_live"] = k_live
    thesis = {}
    if O.height:
        O = O.with_columns(phase=pl.when(pl.col("secs") > 2700).then(pl.lit("Q1")).when(pl.col("secs") > 1800).then(pl.lit("Q2"))
                           .when(pl.col("secs") > 900).then(pl.lit("Q3")).otherwise(pl.lit("Q4")))
        th = O.group_by("phase").agg(n=pl.len(), market=((pl.col("fair") - pl.col("over")) ** 2).mean(),
                                     model=((pl.col("model") - pl.col("over")) ** 2).mean(),
                                     said_over=pl.col("fair").mean(), over_hit=pl.col("over").mean()).sort("phase")
        thesis = th.to_dicts()
        print("in-game lines vs the model (Brier, lower is better):\n", th)
    print("replay:", s)
    REPORTS.mkdir(exist_ok=True)
    out_path = OUT if not args.tag else REPORTS / f"live_replay_{args.tag}.json"
    out_path.write_text(json.dumps({"games": len(gids), "summary": s, "thesis": thesis, "mode": "model" if args.k is None else f"anchored k={args.k}",
                               "rules": {"min_ev": decide.MIN_EV, "max_gap": decide.MAX_GAP, "per_game": decide.MAX_PER_GAME,
                                         "decision_lag": DECISION_LAG, "fill_lag": FILL_LAG, "clv_lag": CLV_LAG}},
                              indent=1, default=str))
    if F.height:
        F.write_parquet(FLAGS if not args.tag else REPORTS / f"live_replay_flags_{args.tag}.parquet")
    print("saved", out_path)


if __name__ == "__main__":
    main()
