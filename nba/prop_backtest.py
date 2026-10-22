"""Back-test of the NBA props model and of the market's own leanings (user, 2026-10-22), on ESPN's prop-line
history (nba/espn_props.py) and hoopR box scores, 2022-23 to 2025-26.

Walk-forward: on each game date every player's profile is refit on earlier games only (nba/props.py profiles),
his minutes set the board's way (expected_minutes: teammates ruled out hand theirs on; here "ruled out" is who
did not play, which the 5pm injury report mostly knows; a starter's bump; the absence ramp; the blowout trim from
the closing spread), and each prop priced by the model alone (k = 1) at the book's line. The market is that
book's own no-vig chance, at the open and, where the latest line is still a pregame one, at the close.

What it can and cannot test:
- The board's live edge is shopping DraftKings and FanDuel against Pinnacle. ESPN keeps one book per game in
  most seasons and never Pinnacle, so shopping is tested only in 2023-24 (three or four books), among themselves.
- The model against the market: does it know anything the line does not (log loss, a blend k), and do its
  biggest disagreements win at the open or hold to the close?
- The market's leanings, every stat and side: overs and unders, prices, roles, rest, spread, matchups.

    python -m nba.prop_backtest build      # -> data/nba/prop_bt.parquet (odds data: off the public repo)
    python -m nba.prop_backtest build --blind   # minutes from what was known before the line opened
"""
import argparse
from datetime import date

import numpy as np
import polars as pl

from config import DATA
from nba import props as NP
from nba.data import load

SRC = DATA / "nba" / "espn_props"
OUT = DATA / "nba" / "prop_bt.parquet"
SEASONS = (2023, 2024, 2025, 2026)
# a latest line this far from the open is an in-game line, not a pregame close
CLEAN_MOVE = {"pts": 3.0, "reb": 2.0, "ast": 2.0, "fg3": 1.0, "pra": 4.0, "pr": 3.5, "pa": 3.5, "ra": 2.5}


def _dec(am: pl.Expr) -> pl.Expr:
    return pl.when(am > 0).then(1 + am / 100).otherwise(1 + 100 / (-am))


def lines() -> pl.DataFrame:
    """One row per (game, book, player, market): open and latest line, prices and no-vig chance of the over."""
    raw = pl.concat([pl.read_parquet(p).with_columns(season=pl.lit(int(p.stem.split("_")[1])))
                     for p in sorted(SRC.glob("props_*.parquet"))], how="diagonal_relaxed")
    k = ["season", "game_id", "provider", "provider_name", "athlete_id", "market"]
    # the main line: the two entries that share a line (alternate lines come one-sided); over first, under second
    raw = raw.filter(pl.col("line_open").is_not_null() & pl.col("am_open").is_not_null())
    raw = raw.with_columns(n=pl.len().over(*k, "line_open", "line"), i=pl.col("seq").rank("ordinal").over(*k, "line_open", "line"))
    raw = raw.filter(pl.col("n") == 2)
    o = raw.filter(pl.col("i") == 1).select(*k, "line_open", "line", pl.col("am_open").alias("o_am_open"), pl.col("am").alias("o_am"))
    u = raw.filter(pl.col("i") == 2).select(*k, "line_open", "line", pl.col("am_open").alias("u_am_open"), pl.col("am").alias("u_am"))
    L = o.join(u, on=[*k, "line_open", "line"])
    # one main line per player-market and book: the most even pair at the open
    L = L.with_columns(_d=(_dec(pl.col("o_am_open")) - _dec(pl.col("u_am_open"))).abs()).sort("_d").unique(k, keep="first").drop("_d")
    L = L.with_columns(o_open=_dec(pl.col("o_am_open")), u_open=_dec(pl.col("u_am_open")), o_close=_dec(pl.col("o_am")), u_close=_dec(pl.col("u_am")))
    L = L.with_columns(p_open=(1 / pl.col("o_open")) / (1 / pl.col("o_open") + 1 / pl.col("u_open")),
                       hold_open=1 / pl.col("o_open") + 1 / pl.col("u_open") - 1,
                       p_close=(1 / pl.col("o_close")) / (1 / pl.col("o_close") + 1 / pl.col("u_close")),
                       hold_close=1 / pl.col("o_close") + 1 / pl.col("u_close") - 1)
    L = L.filter(pl.col("hold_open").is_between(0.0, 0.15))
    move = pl.col("market").replace_strict(CLEAN_MOVE, return_dtype=pl.Float64)
    return L.with_columns(clean=((pl.col("line") - pl.col("line_open")).abs() <= move) & pl.col("hold_close").is_between(0.0, 0.15))


def results(bx: pl.DataFrame) -> pl.DataFrame:
    r = bx.filter(pl.col("min") > 0).select("game_id", "athlete_id", "team", "pos", "starter", "min", *NP.STATS)
    return r.with_columns(**{m: sum(pl.col(c) for c in parts) for m, parts in NP.COMBOS.items()})


def model_rows(bx: pl.DataFrame, sched: pl.DataFrame, spreads: dict, need: set[tuple[int, int]], blind: bool = False) -> pl.DataFrame:
    """The board's minutes and rates for every player-game with a prop, refit on earlier games each date. `blind`:
    nothing from the game itself (no actual scratches handing minutes on, starter by his recent starts share), the
    fair test against an opening line posted before the injury news."""
    out = []
    games = (bx.select("game_id", "season", "date", "team").unique().join(sched, on="game_id", how="inner"))
    dnp = {(g, a) for g, a in bx.filter(pl.col("min") <= 0).select("game_id", "athlete_id").iter_rows()}
    starters = {(g, a) for g, a in bx.filter(pl.col("starter")).select("game_id", "athlete_id").iter_rows()}
    need_g = {g for g, _ in need}
    for (season, day), gday in sorted(games.group_by("season", "date"), key=lambda t: t[0][1]):
        if not set(gday["game_id"].to_list()) & need_g:
            continue
        prof = NP.profiles(bx, day, int(season))
        if not prof.height:
            continue
        P = {r["athlete_id"]: r for r in prof.iter_rows(named=True)}
        for (gid, team), _ in gday.group_by("game_id", "team"):
            roster = bx.filter((pl.col("game_id") == gid) & (pl.col("team") == team)).select("athlete_id", "team")
            out_ids = set() if blind else {a for a in roster["athlete_id"].to_list() if (gid, a) in dnp}
            mins = NP.expected_minutes(prof, roster, team, out_ids, day)
            sp = abs(spreads.get(gid) or 0.0)
            for a in roster["athlete_id"].to_list():
                if (gid, a) not in need or a in out_ids or a not in P or P[a].get("r_pts") is None:
                    continue
                p = P[a]
                st = (p.get("starts") or 0) >= 0.5 if blind else (gid, a) in starters
                em = mins.get(a, p["exp_min"]) + (NP.STARTER_BUMP if st else 0.0) - NP.minutes_short(p)
                if st and sp > 7:
                    em *= 1 - NP.BLOWOUT * (sp - 7)
                out.append({"game_id": gid, "athlete_id": a, "date": day, "exp_min": float(min(max(em, 0.0), 42.0)),
                            "sd_min": float(min(max((p["sd_min"] or 6.0) * NP.SD_MIN_SCALE, NP.SD_MIN_FLOOR), NP.SD_MIN_CAP)),
                            "games_now": int(p["games_now"] or 0), "pace_factor": 1.0, **{f"r_{k}": float(p[f"r_{k}"]) for k in NP.STATS}})
    return pl.DataFrame(out)


def build(blind: bool = False) -> None:
    L = lines()
    bx = NP.box(list(range(min(SEASONS) - 1, max(SEASONS) + 1)))
    R = results(bx)
    sched = load("schedule").filter(pl.col("season_type") == 2).select(pl.col("id").cast(pl.Int64).alias("game_id"),
                                                                     pl.col("home_abbreviation").alias("home"))
    cl = load("closing_lines").select(pl.col("game_id").cast(pl.Int64), (-pl.col("home_point")).alias("close_spread"),
                                      pl.col("over_under").alias("close_total"))
    spreads = dict(zip(cl["game_id"].to_list(), cl["close_spread"].to_list()))
    need = {(g, a) for g, a in L.select("game_id", "athlete_id").unique().iter_rows()}
    M = model_rows(bx.filter(pl.col("season") >= min(SEASONS) - 1), sched, spreads, need, blind)
    print(f"model rows: {M.height}")
    X = (L.join(R, on=["game_id", "athlete_id"]).join(M.drop("date"), on=["game_id", "athlete_id"], how="left")
         .join(cl, on="game_id", how="left").join(bx.select("game_id", "date").unique("game_id"), on="game_id", how="left"))
    X = X.with_columns(actual=pl.coalesce(*[pl.when(pl.col("market") == m).then(pl.col(m).cast(pl.Float64)) for m in NP.MARKETS]))
    # the model's chance of the over at the open line and at the latest line
    po, pc = [], []
    for r in X.iter_rows(named=True):
        if r["exp_min"] is None:
            po.append(None); pc.append(None)
            continue
        mix = NP.Mix.of(r, r["market"])
        o, push, u = mix.probs(r["line_open"])
        po.append(o / max(o + u, 1e-12))
        o2, _, u2 = mix.probs(r["line"]) if r["line"] != r["line_open"] else (o, push, u)
        pc.append(o2 / max(o2 + u2, 1e-12))
    X = X.with_columns(pm_open=pl.Series(po, dtype=pl.Float64), pm_close=pl.Series(pc, dtype=pl.Float64))
    out = OUT.with_name("prop_bt_blind.parquet") if blind else OUT
    X.write_parquet(out)
    print(f"wrote {out}: {X.height} props, {X['pm_open'].is_not_null().sum()} with a model price")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["build"])
    ap.add_argument("--blind", action="store_true", help="no actual scratches or starters: the fair test at the open")
    a = ap.parse_args()
    build(a.blind)


def espnbet_reconstruct(seasons=(2025, 2026)) -> pl.DataFrame:
    """ESPN BET 2024-25 and 2025-26 (user, 2026-10-22: make them usable). Its stored prices fail the calibration
    test with the over-first order (slope ~0), so the order is not reliable there; but besides the main line it
    lists one-sided alternate lines (always overs), and where the main line falls on that ladder tells which of
    its two prices is the over. Reconstructed this way the prices predict results only weakly (slope ~0.35: they
    are degraded at the source), while the main LINE is a real pregame close (it misses the result by the same
    4.5 points as the opening line, like the clean 2023-24 books). Use these for line-only tests (at -110)."""
    dec = lambda a: (1 + a / 100) if a > 0 else (1 + 100 / -a)  # noqa: E731
    lg = lambda p: float(np.log(min(max(p, 1e-4), 1 - 1e-4) / (1 - min(max(p, 1e-4), 1 - 1e-4))))  # noqa: E731
    raw = pl.concat([pl.read_parquet(SRC / f"props_{s}.parquet").with_columns(season=pl.lit(s)) for s in seasons], how="diagonal_relaxed")
    raw = raw.filter((pl.col("provider_name") == "ESPN BET") & pl.col("line").is_not_null() & pl.col("am").is_not_null() & (pl.col("am") != 0))
    rows = []
    for key, g in raw.group_by("season", "game_id", "athlete_id", "market"):
        lines_, ams = g["line"].to_list(), g["am"].to_list()
        cnt = {l: lines_.count(l) for l in set(lines_)}
        ladder = sorted((l, 1 / dec(a)) for l, a in zip(lines_, ams) if cnt[l] == 1)
        best = None
        for L in (l for l, c in cnt.items() if c == 2):
            q = [1 / dec(a) for a in g.filter(pl.col("line") == L).sort("seq")["am"].to_list()]
            if 0.02 <= sum(q) - 1 <= 0.12 and (best is None or abs(q[0] - q[1]) < abs(best[1][0] - best[1][1])):
                best = (L, q)
        if best is None:
            continue
        L, q = best
        lo, hi = [x for x in ladder if x[0] < L], [x for x in ladder if x[0] > L]
        if not lo or not hi:
            continue
        (l1, p1), (l2, p2) = lo[-1], hi[0]
        qL = 1 / (1 + np.exp(-(lg(p1) + (lg(p2) - lg(p1)) * (L - l1) / (l2 - l1))))
        o, u = (q[0], q[1]) if abs(q[0] - qL) <= abs(q[1] - qL) else (q[1], q[0])
        rows.append({"season": key[0], "game_id": key[1], "athlete_id": key[2], "market": key[3], "line": L, "p": o / (o + u)})
    return pl.DataFrame(rows)
