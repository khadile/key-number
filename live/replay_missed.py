"""Replay a live decision point the engine missed, and (with --record) put it on the paper record.

    python -m live.replay_missed 2026_05_PHI_JAX              # what the halftime decision would have been
    python -m live.replay_missed 2026_05_PHI_JAX --record     # and add it to the record

Only with a commitment, written to ledger/live_replayed.jsonl (and pushed, so its time is public) before the
game ended, that names the method; whatever the replay picks, or no play, is what counts: win or lose. A game
that already has its halftime prop on the record gets nothing more (one prop a game).

Methods, both on the archive the live rules were set on (live/replay_windows.py): the game state at the
break from nflverse's play-by-play; DraftKings and FanDuel prices from the OddsPapi archive, including
whether the price was still there 30 seconds later and where the line went in 5 minutes; the engine's
rubric (decide.py, rubric_checks.py); and the engine's own pregame projections, saved during the game
(data/live/<game>/priors_pregame.parquet), not ones rebuilt afterwards.
- two-checks: one pair of checks 90 s apart (the halftime rule until 2026-10-11 2:45pm ET).
- rolling: a check every 90 s from the break until the second half's first snap; the first play that
  holds on two checks in a row is the pick; when nothing holds, what passes now is the new first check
  (the halftime rule from 2026-10-11 2:45pm ET, live/engine.py).

Needs, after the game: its play-by-play (python -c "from pipeline.ingest import ingest; ingest({'pbp'}, False)")
and its archived prices (python -m pipeline.odds_history --games <game_id>).
"""
import argparse
import zlib
from datetime import datetime, timedelta, timezone

import polars as pl

import live.replay as replay_mod
import live.replay_games as replay_games_mod
from config import DATA
from live import decide, ledger, rubric_checks
from live.model import LiveModel, LiveParams
from live.replay import CLV_LAG, FILL_LAG, _banked, _final_stats, _prop_grid
from live.replay_windows import BLOWOUT, FEED_LAG, PERSIST, STALE_BREAK, _plays, _result, _score_at, game_points, select
from pipeline.odds_history import HIST, fixture_index
from props.backtest import load as load_props_params
from props.lines import load as load_props, main_lines
from props.model import Simulator

COMMITS = ledger.LEDGER / "live_replayed.jsonl"
METHODS = ("two-checks", "rolling")


def commitment(gid: str, window: str) -> dict | None:
    rows = [r for r in ledger.read(COMMITS) if r.get("id") == f"{gid}:{window}" and r.get("committed_at")]
    return rows[0] if rows else None


def _inputs(gid: str, test: bool):
    plays = _plays([gid])
    if plays.height == 0 or plays["ts"].null_count() == plays.height:
        raise SystemExit(f"nflverse has not posted {gid}'s play-by-play yet")
    if (plays["game_seconds_remaining"].min() or 0) > 0:
        raise SystemExit(f"{gid}'s play-by-play is not complete yet")
    pri_path = DATA / "live" / gid / "priors_pregame.parquet"
    p_params, disp = load_props_params()
    if pri_path.exists():
        priors = pl.read_parquet(pri_path)
    elif test:
        from props.section import _frames_cached
        priors = _frames_cached(p_params).filter((pl.col("game_id") == gid) & pl.col("role").is_not_null())
    else:
        raise SystemExit(f"missing the engine's pregame projections ({pri_path})")
    P = load_props([gid], pregame_only=False)
    if not P.height:
        raise SystemExit(f"no archived prices for {gid}: run python -m pipeline.odds_history --games {gid}")
    P_live = P.filter((pl.col("ts") >= pl.col("kickoff")) & pl.col("book").is_in(["draftkings", "fanduel"]))
    close: dict = {}
    for x in main_lines(P.filter(pl.col("ts") < pl.col("kickoff")), pl.col("kickoff") - timedelta(minutes=5)).iter_rows(named=True):
        close.setdefault((x["gsis_id"], x["market"]), {})[x["book"]] = {"line": x["line"], "over": x["over"], "under": x["under"]}
    return plays, priors, disp, P, P_live, close or None


def _two_checks(gid, window, plays, priors, disp, P, P_live, close):
    fx = dict(fixture_index().select("game_id", "fixture_id").iter_rows())
    path = HIST / f"{fx.get(gid)}.parquet"
    grows = pl.read_parquet(path).filter(pl.col("ts") >= P["kickoff"][0]) if gid in fx and path.exists() else None
    week = int(gid.split("_")[1])
    c, _, _ = game_points(gid, week, plays, priors, P_live, grows, close, LiveParams.load(), disp)
    C = pl.DataFrame(c, infer_schema_length=None) if c else pl.DataFrame()
    if C.height:
        C = C.filter((pl.col("kind") == "prop") & (pl.col("event") == window))
    if C.height:
        C = rubric_checks.apply(C, *rubric_checks.flags_from_pbp([gid], decide.WINDOW_AT[window]), event=window)
    fl = select(C, {week: 1.0}, (window,), "prop", max_gap=decide.MAX_GAP,
                per_window=decide.PROP_PER_GAME, per_game=decide.PROP_PER_GAME) if C.height else []
    n = C.height if C.height else 0
    return (fl[0] if fl else None), f"{n} candidate prices at the two checks"


def _rolling(gid, window, plays, priors, disp, P, P_live, close):
    """A check every PERSIST*2 (90 s) from the break (plus the feed's delay) to the next snap; the first play
    that holds (same book, side and line, still passing) on two checks in a row."""
    B = decide.WINDOW_AT[window]
    gap = PERSIST * (2 if window == "half" else 1)
    lag = FEED_LAG * (2 if window == "half" else 1)
    home, away = plays["home_team"][0], plays["away_team"][0]
    allp = plays.to_dicts()
    scrim = [i for i, r in enumerate(allp) if r["play_type"] in ("pass", "run") and r["ts"] is not None]
    before = [i for i in scrim if (allp[i]["game_seconds_remaining"] or 0) > B]
    after = [i for i in scrim if (allp[i]["game_seconds_remaining"] or 0) <= B]
    if not before or not after:
        return None, "no break found in the play-by-play"
    i_last, i_next = before[-1], after[0]
    t, t_end = allp[i_last]["ts"] + timedelta(seconds=lag), allp[i_next]["ts"] - timedelta(seconds=5)
    checks = []
    while t < t_end:
        checks.append(t)
        t += timedelta(seconds=gap)
    if not checks:
        return None, "the break was too short for a check"
    times = []
    for c in checks:
        times += [c, c + timedelta(seconds=FILL_LAG), c + timedelta(seconds=CLV_LAG)]
    grid = _prop_grid(P_live, times)
    lp = LiveParams.load()
    model = LiveModel(priors, lp, Simulator(disp, seed=zlib.crc32(gid.encode())), size=3000)
    if close:
        model.calibrate(close)
    info = {r["gsis_id"]: {"name": r["name"], "team": r["team"], "position": r["position"]} for r in priors.iter_rows(named=True)}
    prior_share = {r["gsis_id"]: (r["tshare"], r["rshare"]) for r in priors.iter_rows(named=True)}
    banked, team_so = _banked(plays)
    nxt = allp[i_next]
    st = {"seconds_left": B, "score": _score_at(nxt, home, away), "possession": nxt["posteam"], "yards_to_endzone": nxt["yardline_100"]}
    b, tso = banked[i_next - 1], team_so[i_next - 1]
    proj = model.project(st, b, tso)
    lead = st["score"][home] - st["score"][away]
    big, hurt = rubric_checks.flags_from_pbp([gid], B)
    final = _final_stats(plays)
    key = lambda c: (c.key, c.side, c.book, c.line)

    def passing(k: int) -> list:
        out = []
        if abs(lead) > BLOWOUT:
            return out
        for c in decide.prop_candidates(gid, proj, grid.get(3 * k, {}), info, None):
            ev = c.p * (c.dec - 1) - (1 - c.p - c.push)
            if ev < decide.MIN_EV or abs(c.p - c.fair) > decide.MAX_GAP:
                continue
            if (gid, c.player, c.market) in big or (gid, c.player) in hurt:
                continue
            c.ev = ev
            out.append(c)
        return out

    first, n_priced = None, 0
    for k in range(len(checks)):
        now = passing(k)
        n_priced += len(now)
        if first is not None:
            held = sorted([c for c in now if key(c) in first], key=lambda c: -c.ev)
            if held:
                c = held[0]
                later = grid.get(3 * k + 1, {}).get((c.player, c.market), {}).get(c.book)
                five = grid.get(3 * k + 2, {}).get((c.player, c.market), {}).get(c.book)
                y = final.get(c.player, {}).get(c.market)
                pb, i_sh = proj["players"][c.player], 1 if c.market == "rush_yds" else 0
                other = away if c.team == home else home
                ctx = {"seconds_left": B, "rem_mean": pb["_rem_mean"], "share_now": pb["_share"][i_sh],
                       "prior_share": prior_share[c.player][i_sh],
                       "banked": {"receptions": b.get(c.player, {}).get("rec", 0), "rec_yds": b.get(c.player, {}).get("rec_yds", 0),
                                  "rush_yds": b.get(c.player, {}).get("rush_yds", 0), "pass_yds": b.get(c.player, {}).get("pass_yds", 0)},
                       "team_lead": st["score"][c.team] - st["score"][other]}
                f = {"game_id": gid, "key": c.key, "side": c.side, "book": c.book, "line": c.line, "dec": c.dec, "p": c.p,
                     "push": c.push, "fair": c.fair, "group": c.group, "player": c.player, "market": c.market, "label": c.label(),
                     "team": c.team, "at": checks[k].isoformat(), "secs_left": B, "lead": lead, "check_no": k + 1,
                     "fill_line": later["line"] if later else None,
                     "fill_dec": (later["over"] if c.side == "over" else later["under"]) if later else None,
                     "clv": None if five is None else (five["line"] - c.line if c.side == "over" else c.line - five["line"]),
                     "result": None if y is None else _result(c.side, c.line, y), "reason": decide.reason(c, ctx)}
                return f, f"held on checks {k} and {k + 1} of {len(checks)} ({n_priced} passing prices seen)"
        first = {key(c) for c in now} or None
    return None, f"{len(checks)} checks through the break, nothing held on two in a row ({n_priced} passing prices seen)"


def replay(gid: str, window: str = "half", test: bool = False, method: str = "two-checks"):
    """(the play the method picks or None, a one-line account, the game's plays)."""
    replay_mod.STALE_S = STALE_BREAK           # as live/replay_windows.py: at a break a line can sit for minutes
    replay_games_mod.STALE_S = STALE_BREAK
    plays, priors, disp, P, P_live, close = _inputs(gid, test)
    run = _rolling if method == "rolling" else _two_checks
    f, how = run(gid, window, plays, priors, disp, P, P_live, close)
    return f, how, plays


def _score_at_break(plays: pl.DataFrame, secs: float) -> dict:
    before = plays.filter(pl.col("game_seconds_remaining") >= secs).tail(1).to_dicts()[0]
    return {before["home_team"]: int(before["total_home_score"] or 0), before["away_team"]: int(before["total_away_score"] or 0)}


def record(gid: str, window: str, f: dict | None, how: str, plays: pl.DataFrame, commit: dict, method: str) -> str:
    now = datetime.now(timezone.utc).isoformat()
    end = plays["ts"].drop_nulls().max()
    if datetime.fromisoformat(commit["committed_at"]) >= end:
        raise SystemExit("the commitment was made after the game ended: not recorded")
    if any(r.get("id") == commit["id"] and r.get("done_at") for r in ledger.read(COMMITS)):
        raise SystemExit(f"{commit['id']} was already replayed and recorded")
    if any(r["game_id"] == gid and r.get("kind") == "prop" for r in ledger.read(ledger.FLAGS)):
        ledger.append(COMMITS, {"id": commit["id"], "done_at": now, "decision": "no play: the game already has its prop on the record"})
        return "no play: the game already has its halftime prop on the record (one prop a game)"
    if f is None:
        ledger.append(COMMITS, {"id": commit["id"], "done_at": now, "decision": "no play", "method": method, "detail": how})
        return f"no play ({how}); nothing added to the record"
    fid = f"{gid}:{f['key']}:replay-{window}"
    dec, p, push = float(f["dec"]), float(f["p"]), float(f.get("push") or 0)
    ev = p * (dec - 1) - (1 - p - push)
    gap = PERSIST * (2 if window == "half" else 1)
    note = (f"Replayed after the game ({method}): the engine missed this decision live. Committed "
            f"{commit['committed_at'][:16]}Z, before the result, to count whatever the outcome.")
    ledger.append(ledger.FLAGS, {
        "id": fid, "game_id": gid, "kind": "prop", "key": f["key"], "player": f["player"], "group": f["group"],
        "label": f["label"], "market": f["market"], "side": f["side"], "line": f["line"], "book": f["book"],
        "dec": round(dec, 4), "american": decide.american(dec), "p": round(p, 4), "fair": round(float(f["fair"]), 4),
        "ev": round(ev, 4), "stake": decide.STAKE_U,
        "reason": (f.get("reason") or "") + f"; held on two price checks {gap}s apart",
        "at": f["at"], "score": _score_at_break(plays, decide.WINDOW_AT[window]), "seconds_left": f["secs_left"],
        "mode": "paper", "quote_age_s": None, "window": decide.WINDOW_LABEL[window],
        "replayed": True, "replay_of": commit["id"], "replay_method": method, "replay_note": note})
    filled = f.get("fill_dec") is not None and f.get("fill_line") == f["line"]
    ledger.append(ledger.EVENTS, {"id": fid, "fill": "filled" if filled else "missed",
                                  "fill_dec": f.get("fill_dec") if filled else None, "fill_line": f.get("fill_line")})
    if f.get("clv") is not None:
        ledger.append(ledger.EVENTS, {"id": fid, "clv_units": round(float(f["clv"]), 1)})
    ledger.append(ledger.EVENTS, {"id": fid, "expired": "replayed after the game"})
    actual = _final_stats(plays).get(f["player"], {}).get(f["market"])
    ledger.append(ledger.RESULTS, {"id": fid, "result": f["result"], "actual": actual})
    ledger.append(COMMITS, {"id": commit["id"], "done_at": now, "method": method, "flag_id": fid, "result": f["result"],
                            "actual": actual, "decision": f"{f['label']} {decide.american(dec):+d} {f['book']}"})
    return f"recorded {fid}: {f['label']} {decide.american(dec):+d} {f['book']} -> {f['result']} (actual {actual})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("game_id")
    ap.add_argument("--window", default="half", choices=list(decide.WINDOW_AT))
    ap.add_argument("--method", choices=METHODS, help="default: the commitment's method (two-checks if none)")
    ap.add_argument("--record", action="store_true", help="add it to the paper record (needs a commitment)")
    ap.add_argument("--test", action="store_true", help="dry run on any finished game, with the backtest's projections")
    a = ap.parse_args()
    if a.test and a.record:
        raise SystemExit("--test runs are never recorded")
    commit = commitment(a.game_id, a.window)
    if a.record and commit is None:
        raise SystemExit(f"no commitment for {a.game_id}:{a.window} in {COMMITS}: nothing is recorded without one")
    committed = (commit or {}).get("method", "two-checks")
    method = a.method or committed
    if a.record and method != committed:
        raise SystemExit(f"the commitment says {committed}; a different method cannot be recorded")
    f, how, plays = replay(a.game_id, a.window, a.test, method)
    print(f"{a.game_id} {decide.WINDOW_LABEL[a.window]} ({method}): {how}")
    if f is None:
        print("decision: no play")
    else:
        print(f"decision: {f['label']} {decide.american(float(f['dec'])):+d} {f['book']} · model {float(f['p']):.0%} vs "
              f"{float(f['fair']):.0%} no-vig · at {f['at']} · still there 30 s later: {f.get('fill_line') == f['line']} · "
              f"line moved {f.get('clv')} our way in 5 min · result {f['result']}")
    if a.record:
        print(record(a.game_id, a.window, f, how, plays, commit, method))


if __name__ == "__main__":
    main()
