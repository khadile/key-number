"""Replay a live decision point the engine missed, and (with --record) put it on the paper record.

    python -m live.replay_missed 2026_05_PHI_JAX              # what the halftime decision would have been
    python -m live.replay_missed 2026_05_PHI_JAX --record     # and add it to the record

Only for a decision the engine could not take for a technical reason, and only if the commitment to
count it was written to ledger/live_replayed.jsonl (and pushed, so its time is public) before the game
ended. Whatever the replay picks, or no play, is what counts: win or lose.

The replay is the one that set the live rules (live/replay_windows.py): the game state at the break from
nflverse's play-by-play; DraftKings and FanDuel prices at the two checks from the OddsPapi archive, and
from the same archive whether the price was still there 30 seconds later and where the line went in
5 minutes; the engine's rubric (decide.py, rubric_checks.py); and the engine's own pregame projections,
saved during the game (data/live/<game>/priors_pregame.parquet), not ones rebuilt afterwards.

Needs, after the game: its play-by-play (python -c "from pipeline.ingest import ingest; ingest({'pbp'}, False)")
and its archived prices (python -m pipeline.odds_history --games <game_id>).
"""
import argparse
import json
from datetime import datetime, timedelta, timezone

import polars as pl

from config import DATA
from live import decide, ledger, rubric_checks
from live.model import LiveParams
from live.replay import _final_stats
from live.replay_windows import PERSIST, _plays, game_points, select
from pipeline.odds_history import HIST, fixture_index
from props.backtest import load as load_props_params
from props.lines import load as load_props, main_lines

COMMITS = ledger.LEDGER / "live_replayed.jsonl"


def commitment(gid: str, window: str) -> dict | None:
    rows = [r for r in ledger.read(COMMITS) if r.get("id") == f"{gid}:{window}" and r.get("committed_at")]
    return rows[0] if rows else None


def replay(gid: str, window: str = "half", test: bool = False):
    """(every candidate at the decision point, the play the rules pick or none, the game's plays).
    `test`: without the engine's saved projections, use the backtest's (dry runs only, never recorded)."""
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
    kick = P["kickoff"][0]
    P_live = P.filter((pl.col("ts") >= pl.col("kickoff")) & pl.col("book").is_in(["draftkings", "fanduel"]))
    close: dict = {}
    for x in main_lines(P.filter(pl.col("ts") < pl.col("kickoff")), pl.col("kickoff") - timedelta(minutes=5)).iter_rows(named=True):
        close.setdefault((x["gsis_id"], x["market"]), {})[x["book"]] = {"line": x["line"], "over": x["over"], "under": x["under"]}
    fx = dict(fixture_index().select("game_id", "fixture_id").iter_rows())
    path = HIST / f"{fx.get(gid)}.parquet"
    grows = pl.read_parquet(path).filter(pl.col("ts") >= kick) if gid in fx and path.exists() else None
    week = int(gid.split("_")[1])
    c, _, _ = game_points(gid, week, plays, priors, P_live, grows, close or None, LiveParams.load(), disp)
    C = pl.DataFrame(c, infer_schema_length=None) if c else pl.DataFrame()
    if C.height:
        C = C.filter((pl.col("kind") == "prop") & (pl.col("event") == window))
    if C.height:
        C = rubric_checks.apply(C, *rubric_checks.flags_from_pbp([gid], decide.WINDOW_AT[window]), event=window)
    fl = select(C, {week: 1.0}, (window,), "prop", max_gap=decide.MAX_GAP,
                per_window=decide.PROP_PER_GAME, per_game=decide.PROP_PER_GAME) if C.height else []
    return C, (fl[0] if fl else None), plays


def _score_at_half(plays: pl.DataFrame, secs: float) -> dict:
    before = plays.filter(pl.col("game_seconds_remaining") >= secs).tail(1).to_dicts()[0]
    return {before["home_team"]: int(before["total_home_score"] or 0), before["away_team"]: int(before["total_away_score"] or 0)}


def record(gid: str, window: str, f: dict | None, plays: pl.DataFrame, commit: dict, n_cands: int) -> str:
    now = datetime.now(timezone.utc).isoformat()
    end = plays["ts"].drop_nulls().max()
    if datetime.fromisoformat(commit["committed_at"]) >= end:
        raise SystemExit("the commitment was made after the game ended: not recorded")
    if any(r.get("id") == commit["id"] and r.get("done_at") for r in ledger.read(COMMITS)):
        raise SystemExit(f"{commit['id']} was already replayed and recorded")
    if f is None:
        ledger.append(COMMITS, {"id": commit["id"], "done_at": now, "decision": "no play",
                                "detail": f"{n_cands} candidate prices at the two checks; none passed the rubric"})
        return "no play: nothing passed the rubric at halftime (nothing added to the record)"
    fid = f"{gid}:{f['key']}:replay-{window}"
    if any(r["id"] == fid for r in ledger.read(ledger.FLAGS)):
        raise SystemExit(f"{fid} is already on the record")
    dec, p, push = float(f["dec"]), float(f["p"]), float(f.get("push") or 0)
    ev = p * (dec - 1) - (1 - p - push)
    persist = PERSIST * (2 if window == "half" else 1)
    note = (f"Replayed after the game: the engine missed this decision (SportsGameOdds' in-game quotes were 5-7 minutes "
            f"old). Committed {commit['committed_at'][:16]}Z, before the result, to count whatever the outcome.")
    ledger.append(ledger.FLAGS, {
        "id": fid, "game_id": gid, "kind": "prop", "key": f["key"], "player": f["player"], "group": f["group"],
        "label": f["label"], "market": f["market"], "side": f["side"], "line": f["line"], "book": f["book"],
        "dec": round(dec, 4), "american": decide.american(dec), "p": round(p, 4), "fair": round(float(f["fair"]), 4),
        "ev": round(ev, 4), "stake": decide.STAKE_U,
        "reason": (f.get("reason") or "") + f"; held on two price checks {persist}s apart",
        "at": f["at"], "score": _score_at_half(plays, decide.WINDOW_AT[window]), "seconds_left": f["secs_left"],
        "mode": "paper", "quote_age_s": None, "window": decide.WINDOW_LABEL[window],
        "replayed": True, "replay_of": commit["id"], "replay_note": note})
    filled = f.get("fill_dec") is not None and f.get("fill_line") == f["line"]
    ledger.append(ledger.EVENTS, {"id": fid, "fill": "filled" if filled else "missed",
                                  "fill_dec": f.get("fill_dec") if filled else None, "fill_line": f.get("fill_line")})
    if f.get("clv") is not None:
        ledger.append(ledger.EVENTS, {"id": fid, "clv_units": round(float(f["clv"]), 1)})
    ledger.append(ledger.EVENTS, {"id": fid, "expired": "replayed after the game"})
    actual = _final_stats(plays).get(f["player"], {}).get(f["market"])
    ledger.append(ledger.RESULTS, {"id": fid, "result": f["result"], "actual": actual})
    ledger.append(COMMITS, {"id": commit["id"], "done_at": now, "decision": f"{f['label']} {decide.american(dec):+d} {f['book']}",
                            "flag_id": fid, "result": f["result"], "actual": actual})
    return f"recorded {fid}: {f['label']} {decide.american(dec):+d} {f['book']} -> {f['result']} (actual {actual})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("game_id")
    ap.add_argument("--window", default="half", choices=list(decide.WINDOW_AT))
    ap.add_argument("--record", action="store_true", help="add it to the paper record (needs a commitment)")
    ap.add_argument("--test", action="store_true", help="dry run on any finished game, with the backtest's projections")
    a = ap.parse_args()
    if a.test and a.record:
        raise SystemExit("--test runs are never recorded")
    commit = commitment(a.game_id, a.window)
    if a.record and commit is None:
        raise SystemExit(f"no commitment for {a.game_id}:{a.window} in {COMMITS}: nothing is recorded without one")
    C, f, plays = replay(a.game_id, a.window, a.test)
    n1 = C.filter(pl.col("check") == "t1").height if C.height else 0
    n2 = C.filter(pl.col("check") == "t2").height if C.height else 0
    print(f"{a.game_id} {decide.WINDOW_LABEL[a.window]}: {n1} prices at the first check, {n2} at the second")
    if f is None:
        print("decision: no play passed the rubric")
    else:
        dec = float(f["dec"])
        print(f"decision: {f['label']} {decide.american(dec):+d} {f['book']} · model {float(f['p']):.0%} vs {float(f['fair']):.0%} no-vig"
              f" · at {f['at']} · still there 30 s later: {f.get('fill_line') == f['line']} · line moved {f.get('clv')} our way in 5 min"
              f" · result {f['result']}")
    if a.record:
        print(record(a.game_id, a.window, f, plays, commit, n1 + n2))


if __name__ == "__main__":
    main()
