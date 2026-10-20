"""Where the props edge comes from (user, 2026-10-20, after live weeks 5-6 went 38-51).

Every tracked prop of the season at the board's lock time (props.pricing.lock_time): the best DraftKings or
FanDuel side priced the board's way (the model's samples shifted to Pinnacle's no-vig price at Pinnacle's line,
k = 0), and the same bet judged at Pinnacle's close the same way. The yardstick is the EV at the close, which
settles long before wins and losses do.

Each bet's edge comes from one of two places: a better NUMBER than Pinnacle's (DraftKings or FanDuel hung a line
off Pinnacle's, our side's way) or only a better PRICE at the same number. The split, by the EV at the lock:

- An edge from a better number holds up at the close once it is 2% or more; under 2% it is gone.
- An edge from the price alone is mostly gone at the close unless it is large (4%+).
- Weeks 5-6: DraftKings and FanDuel hung numbers off Pinnacle's far less often than in weeks 1-4, so the board's
  one-prop-a-player rule filled up with small price-only edges.

The SportsGameOdds 2026 opens (independent data, a different moment) gave the same split; see the commit and
CLAUDE.md 6b. A rule on top (EV 2%+, and 4%+ when the edge is the price alone) is the user's call: until then
this is a finding on the board, not a pick rule.

    python -m props.edge_source      # -> reports/props_edge_source.json (the board's props "How it works")
"""
import json
from datetime import datetime, timedelta, timezone

import polars as pl

from config import CLEAN, LEDGER, REPORTS
from props.backtest import baselines, load
from props.lines import load as load_lines, main_lines
from props.model import STAT_COL, Simulator
from props.pricing import lock_time, shift_to, side_probs
from props.section import _frames_cached, _samples

OUT = REPORTS / "props_edge_source.json"
BUCKETS = ((0.0, 0.02, "0-2%"), (0.02, 0.04, "2-4%"), (0.04, 9.0, "4%+"))


def _bucket(ev: float) -> str:
    return next(name for lo, hi, name in BUCKETS if lo <= ev < hi)


def candidates(season: int) -> pl.DataFrame:
    """Best DK/FD side per tracked player-market with positive EV at the lock, with its EV at Pinnacle's close."""
    p, disp = load()
    hist = baselines(_frames_cached(p))
    done = hist.filter((pl.col("season") == season) & pl.col("role").is_not_null())
    smp = _samples(Simulator(disp, seed=17), done)
    L = load_lines(done["game_id"].unique().to_list())
    kick = L.select("game_id", "kickoff").unique().with_columns(
        lock=pl.col("kickoff").map_elements(lock_time, return_dtype=pl.Datetime("us", "UTC")))
    A = main_lines(L.join(kick.select("game_id", "lock"), on="game_id"), pl.col("lock"))
    C = main_lines(L, pl.col("kickoff") - timedelta(minutes=5))
    pg = pl.read_parquet(CLEAN / "props_player_games.parquet").filter(pl.col("season") == season)
    act = {(r["game_id"], r["gsis_id"], m): (float(r[c]), r["week"])
           for r in pg.select("game_id", "gsis_id", "week", *STAT_COL.values()).iter_rows(named=True)
           for m, c in STAT_COL.items() if r[c] is not None}
    pin = {(r["game_id"], r["gsis_id"], r["market"]): r for r in A.filter(pl.col("book") == "pinnacle").iter_rows(named=True)}
    pinc = {(r["game_id"], r["gsis_id"], r["market"]): r for r in C.filter(pl.col("book") == "pinnacle").iter_rows(named=True)}
    bets = []
    for r in A.filter(pl.col("book").is_in(["draftkings", "fanduel"])).iter_rows(named=True):
        key = (r["game_id"], r["gsis_id"], r["market"])
        if key not in smp or key not in pin or key not in pinc or key not in act:
            continue
        s, pn, pc = smp[key], pin[key], pinc[key]
        o, push, u = side_probs(s + shift_to(s, pn["line"], pn["p_over"]), r["line"])
        oc, pushc, uc = side_probs(s + shift_to(s, pc["line"], pc["p_over"]), r["line"])
        y, wk = act[key]
        for side, pw, pwc, price in (("over", o, oc, r["over"]), ("under", u, uc, r["under"])):
            won = y > r["line"] if side == "over" else y < r["line"]
            res = "P" if y == r["line"] else ("W" if won else "L")
            rel = (pn["line"] - r["line"]) * (1 if side == "over" else -1)     # + : a better number than Pinnacle's
            bets.append({"week": wk, "game_id": r["game_id"], "gsis_id": r["gsis_id"], "market": r["market"], "side": side,
                         "source": "better number" if rel > 0 else ("same number" if rel == 0 else "worse number"),
                         "ev": pw * (price - 1) - (1 - pw - push), "ev_close": pwc * (price - 1) - (1 - pwc - pushc),
                         "result": res, "units": 0.0 if res == "P" else (price - 1 if won else -1.0)})
    B = pl.DataFrame(bets)
    best = B.sort("ev", descending=True).group_by("game_id", "gsis_id", "market", maintain_order=True).first()
    return best.filter(pl.col("ev") > 0).with_columns(bucket=pl.col("ev").map_elements(_bucket, return_dtype=pl.String))


def _tally(g: pl.DataFrame) -> dict:
    g = g.filter(pl.col("result") != "P")
    if not g.height:
        return {"n": 0}
    w = int((g["result"] == "W").sum())
    return {"n": g.height, "w": w, "l": g.height - w, "units": round(float(g["units"].sum()), 2),
            "roi": round(float(g["units"].mean()), 4), "ev": round(float(g["ev"].mean()), 4),
            "ev_close": round(float(g["ev_close"].mean()), 4), "still": round(float((g["ev_close"] > 0).mean()), 3)}


def main() -> None:
    picks = [json.loads(line) for line in open(LEDGER / "props_picks.jsonl", encoding="utf-8") if '"kind": "prop"' in line]
    season = max(p["season"] for p in picks) if picks else datetime.now(timezone.utc).year
    X = candidates(season)
    rule = (pl.col("ev") >= 0.02) & ((pl.col("source") == "better number") | (pl.col("ev") >= 0.04))
    out = {
        "season": season, "generated": datetime.now(timezone.utc).isoformat(), "weeks": sorted(X["week"].unique().to_list()),
        "by_source": [{"source": s, **_tally(X.filter(pl.col("source") == s))} for s in ("better number", "same number", "worse number")],
        "by_bucket": [{"source": s, "bucket": b, **_tally(X.filter((pl.col("source") == s) & (pl.col("bucket") == b)))}
                      for s in ("better number", "same number") for _, _, b in BUCKETS],
        "by_week": [{"week": int(w), "better": _tally(g.filter(pl.col("source") == "better number")),
                     "same": _tally(g.filter(pl.col("source") == "same number"))}
                    for (w,), g in sorted(X.group_by("week"), key=lambda t: t[0][0])],
        "rule": {"kept": _tally(X.filter(rule)), "dropped": _tally(X.filter(~rule))},
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {X.height} candidates, weeks {out['weeks']}")
    for r in out["by_source"]:
        print(f"  {r['source']:13} n {r['n']:4}  EV at lock {r.get('ev', 0):+.3f}  at the close {r.get('ev_close', 0):+.3f}  still +EV {r.get('still', 0):.0%}")


if __name__ == "__main__":
    main()
