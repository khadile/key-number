"""Replay the live engine's game markets (moneyline, spread, total) against real in-game lines.

    python -m live.replay_games

Same clock as the props replay: after each play the engine prices the rest of
the game from the score, clock, possession and field position, with both
teams playing at their pregame strength (from the closing spread and total).
DraftKings and FanDuel prices are checked against Pinnacle's in-game no-vig
number, the sharpest live reference in the archive. Writes
reports/live_replay_games.json (and the flags parquet).
"""
import json
import zlib
from datetime import timedelta

import numpy as np
import polars as pl

from config import RAW, REPORTS
from live import decide
from live.model import LiveModel, LiveParams
from live.replay import CLV_LAG, DECISION_LAG, FILL_LAG, STALE_S, _pbp
from pipeline.odds_history import HIST, fixture_index
from props.model import Simulator

OUT = REPORTS / "live_replay_games.json"
FLAGS = REPORTS / "live_replay_games_flags.parquet"


def _game_grid(rows: pl.DataFrame, times: list) -> dict[int, dict]:
    """Main spread, total and moneyline per (time, book), as of each time."""
    if not rows.height:
        return {}
    k = ["book", "market", "handicap", "outcome"]
    keys = rows.select(k).unique()
    grid = keys.join(pl.DataFrame({"t": times, "ti": list(range(len(times)))}).with_columns(
        pl.col("t").dt.cast_time_unit("us")), how="cross").sort("t")
    j = grid.join_asof(rows.select(k + ["ts", "price", "active"]).sort("ts"), left_on="t", right_on="ts", by=k,
                       strategy="backward").filter(pl.col("active") & ((pl.col("t") - pl.col("ts")).dt.total_seconds() <= STALE_S))
    out: dict[int, dict] = {}
    for market, a, b, name in (("spreads", "1", "2", "spread"), ("totals", "Over", "Under", "total")):
        m = j.filter(pl.col("market") == market)
        pa = m.filter(pl.col("outcome") == a).select("ti", "book", "handicap", pa="price")
        pb = m.filter(pl.col("outcome") == b).select("ti", "book", "handicap", pb="price")
        pair = (pa.join(pb, on=["ti", "book", "handicap"]).with_columns(gap=(1 / pl.col("pa") - 1 / pl.col("pb")).abs())
                .sort("gap", "handicap").group_by("ti", "book", maintain_order=True).first())
        for r in pair.iter_rows(named=True):
            d = out.setdefault(r["ti"], {}).setdefault(r["book"], {})
            d[name] = -r["handicap"] if market == "spreads" else r["handicap"]
            d[f"{name}_price_a"], d[f"{name}_price_b"] = r["pa"], r["pb"]
    ml = j.filter(pl.col("market") == "moneyline")
    for (ti, book), g in ml.group_by("ti", "book"):
        pr = dict(zip(g["outcome"], g["price"]))
        if "1" in pr and "2" in pr:
            d = out.setdefault(ti, {}).setdefault(book, {})
            d["ml_home"], d["ml_away"] = pr["1"], pr["2"]
    return out


def _fair(lines: dict, market: str, side: str, line: float | None) -> float | None:
    """Pinnacle's no-vig chance for this side at this line (same number only)."""
    pin = lines.get("pinnacle", {})
    if market == "ml" and "ml_home" in pin:
        f = decide.no_vig(pin["ml_home"], pin["ml_away"])
        return f if side == "home" else 1 - f
    if market == "spread" and "spread" in pin:
        home_line = -pin["spread"]
        if (side == "home" and line == home_line) or (side == "away" and line == -home_line):
            f = decide.no_vig(pin["spread_price_a"], pin["spread_price_b"])
            return f if side == "home" else 1 - f
    if market == "total" and "total" in pin and pin["total"] == line:
        f = decide.no_vig(pin["total_price_a"], pin["total_price_b"])
        return f if side == "over" else 1 - f
    return None


def run(k_game: float | None) -> tuple[list, list]:
    from props.backtest import load as load_props_params
    from props.section import _frames_cached
    p, disp = load_props_params()
    fr = _frames_cached(p).filter((pl.col("season") == 2026) & pl.col("role").is_not_null())
    games = pl.read_parquet(RAW.parent / "clean" / "games.parquet").filter((pl.col("season") == 2026) & pl.col("result").is_not_null())
    gids = sorted(set(games["game_id"]) & set(fr["game_id"]))
    fx = dict(fixture_index().select("game_id", "fixture_id").iter_rows())
    pbp = _pbp(gids)
    sim, lp = Simulator(disp, seed=41), LiveParams.load()
    flags_all, obs = [], []
    for n, gid in enumerate(gids):
        g = games.filter(pl.col("game_id") == gid).row(0, named=True)
        path = HIST / f"{fx.get(gid)}.parquet"
        if gid not in fx or not path.exists():
            continue
        rows = pl.read_parquet(path).filter(pl.col("ts") >= g["kickoff_utc"])
        plays = pbp.filter(pl.col("game_id") == gid)
        scrim = plays.filter(pl.col("play_type").is_in(["pass", "run"]) & pl.col("ts").is_not_null()).to_dicts()
        if len(scrim) < 10:
            continue
        home, away = g["home_team"], g["away_team"]
        model = LiveModel(fr.filter(pl.col("game_id") == gid), lp, Simulator(disp, seed=zlib.crc32(gid.encode())), size=500)
        times = [r["ts"] + timedelta(seconds=DECISION_LAG) for r in scrim]
        nT = len(times)
        grid = _game_grid(rows, times + [t + timedelta(seconds=FILL_LAG) for t in times] + [t + timedelta(seconds=CLV_LAG) for t in times])
        margin_final, total_final = g["home_score"] - g["away_score"], g["home_score"] + g["away_score"]
        flags = []
        for i, r in enumerate(scrim[:-1]):
            nxt = scrim[i + 1]
            secs = float(nxt["game_seconds_remaining"] or 0)
            L = grid.get(i, {})
            if secs <= 0 or not L:
                continue
            state = {"seconds_left": secs, "score": {home: nxt["total_home_score"], away: nxt["total_away_score"]},
                     "possession": nxt["posteam"], "yards_to_endzone": nxt["yardline_100"]}
            gm = model.game(state)
            if i % 6 == 0 and "pinnacle" in L and "ml_home" in L["pinnacle"]:
                a, b = gm["teams"]
                mh = gm["margin"] if a == home else -gm["margin"]
                later_pin = grid.get(2 * nT + i, {}).get("pinnacle", {})
                obs.append({"secs": secs, "model": float((mh > 0).mean()), "fair": decide.no_vig(L["pinnacle"]["ml_home"], L["pinnacle"]["ml_away"]),
                            "fair_later": decide.no_vig(later_pin["ml_home"], later_pin["ml_away"]) if "ml_home" in later_pin else None,
                            "won": float(margin_final > 0)})
            if len(flags) >= decide.MAX_PER_GAME or secs > decide.EARLIEST_SECS or k_game is None:
                continue
            cands = []
            for c in decide.game_candidates(gid, gm, L, home, away):
                # Anchor to Pinnacle's live number at the same line, tilted k toward the model;
                # with no Pinnacle number to anchor to there is no flag.
                f = _fair(L, c.market, c.side, c.line)
                if f is None:
                    continue
                c.p = f + k_game * (c.p - f)
                c.fair = f
                c.ev = c.p * (c.dec - 1) - (1 - c.p - c.push)
                cands.append(c)
            for c in decide.select(cands, flags):
                later = grid.get(nT + i, {}).get(c.book, {})
                five = grid.get(2 * nT + i, {})
                ok = False
                if c.market == "ml" and "ml_home" in later:
                    ok = (later["ml_home"] if c.side == "home" else later["ml_away"]) >= c.dec - 1e-9
                elif c.market == "spread" and "spread" in later:
                    ok = (-later["spread"] if c.side == "home" else later["spread"]) == c.line
                elif c.market == "total" and "total" in later:
                    ok = later["total"] == c.line
                f5 = _fair(five, c.market, c.side, c.line)
                if c.market == "ml":
                    res = "W" if (margin_final > 0) == (c.side == "home") else "L"
                elif c.market == "spread":
                    d = (margin_final if c.side == "home" else -margin_final) + c.line
                    res = "W" if d > 0 else ("P" if d == 0 else "L")
                else:
                    d = total_final - c.line if c.side == "over" else c.line - total_final
                    res = "W" if d > 0 else ("P" if d == 0 else "L")
                lead = state["score"][home] - state["score"][away]
                c.reason = (f"model {c.p:.0%} vs market {c.fair:.0%}; {home} {'+' if lead >= 0 else ''}{lead} with "
                            f"{decide._mins(secs)} left, rest played from team strength")
                flags.append({"key": c.key, "player": None, "group": c.group, "game_id": gid, "at": times[i].isoformat(),
                              "label": c.label(), "market": c.market, "side": c.side, "line": c.line, "book": c.book,
                              "dec": c.dec, "american": decide.american(c.dec), "p": round(c.p, 4), "fair": round(c.fair, 4),
                              "ev": round(c.ev, 4), "reason": c.reason, "secs_left": secs, "filled": ok,
                              "fill_dec": c.dec if ok else None, "fair_later": f5, "result": res})
        flags_all += flags
    return flags_all, obs


def main() -> None:
    _, obs = run(None)          # pass 1: how Pinnacle's live win chance moves relative to the model
    O = pl.DataFrame(obs)
    th = {"n": O.height, "market": round(float(((O["fair"] - O["won"]) ** 2).mean()), 4),
          "model": round(float(((O["model"] - O["won"]) ** 2).mean()), 4)}
    mv = O.filter(pl.col("fair_later").is_not_null())
    gap, move = (mv["model"] - mv["fair"]).to_numpy(), (mv["fair_later"] - mv["fair"]).to_numpy()
    k_game = float(np.clip((gap * move).sum() / (gap ** 2).sum(), 0, 0.5)) if len(gap) else 0.0
    moved = move != 0
    th.update(k_game=round(k_game, 4), moved=int(moved.sum()), toward=int((np.sign(gap[moved]) == np.sign(move[moved])).sum()))
    print("win probability, Brier (lower is better), and drift toward the model:", th)
    lp = LiveParams.load()
    lp.k_game = k_game
    lp.save()
    flags_all, _ = run(k_game)  # pass 2: flags, priced from Pinnacle's live number tilted k toward the model
    F = pl.DataFrame(flags_all) if flags_all else pl.DataFrame()
    s = {"flags": F.height}
    if F.height:
        f = F.filter(pl.col("filled"))
        u = f.with_columns(u=pl.when(pl.col("result") == "W").then(pl.col("fill_dec") - 1)
                           .when(pl.col("result") == "L").then(-1.0).otherwise(0.0))["u"]
        c = F.filter(pl.col("fair_later").is_not_null())
        s.update(filled=f.height, w=int((f["result"] == "W").sum()), l=int((f["result"] == "L").sum()),
                 p=int((f["result"] == "P").sum()), units=round(float(u.sum()), 2), roi=round(float(u.mean()), 4) if f.height else None,
                 clv_n=c.height, clv_prob=round(float((c["fair_later"] - c["fair"]).mean()), 4) if c.height else None)
        F.write_parquet(FLAGS)
    print("game markets replay:", s)
    OUT.write_text(json.dumps({"games": th["n"] and len({f["game_id"] for f in flags_all}), "checkpoints": th["n"],
                               "summary": s, "win_prob": th}, indent=1, default=str))


if __name__ == "__main__":
    main()
