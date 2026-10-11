"""Game-line history from the SportsGameOdds alt-line pull, and the Tuesday-line test on 2025.

    python -m backtest.sgo_lines            # validate on 2026 (vs OddsPapi), then test 2025

The one signal the game-line model has shown is closing line value at Tuesday's lines: in
the first month of 2026, lines moved toward the model after the Tuesday 10am ET publish (29
of 42 spread moves). OddsPapi only keeps the current season, so this could not be checked on
a full season. SGO's alt-line pull (pipeline/sgo.py --alt) lists every spread and total a
book hung, each with its last price and time, which rebuilds Pinnacle's line at any moment
(props/line_history.py explains the method and its limits). Validated here against OddsPapi's
2026 timelines before it is trusted on 2025.

Writes reports/clv_2025.json.
"""
import glob
import gzip
import json
from datetime import timedelta

import numpy as np
import polars as pl

from config import CLEAN, REPORTS
from pipeline.odds_history import TEAMS
from pipeline.sgo import ALT_DIR

OUT = REPORTS / "clv_2025.json"
SIDES = {("points", "home", "sp"): ("spread", "home"), ("points", "away", "sp"): ("spread", "away"),
         ("points", "all", "ou"): ("total", None)}


def _am(x):
    try:
        v = int(float(str(x).replace("+", "")))
    except (TypeError, ValueError):
        return None
    return v if abs(v) >= 100 else None


def _dec(a):
    return 1 + a / 100 if a > 0 else 1 + 100 / -a


def history(season: int, books=("pinnacle", "circa", "bookmakereu", "draftkings", "fanduel")) -> pl.DataFrame:
    """One row per (game, book, market, line value): both sides' last price and time at that value.
    Spreads are from the home side (home -3 = home favored by 3)."""
    games = pl.read_parquet(CLEAN / "games.parquet").filter(pl.col("season") == season)
    by_teams = {}
    for g in games.iter_rows(named=True):
        by_teams[(g["home_team"], g["away_team"])] = g
    rows = []
    for f in sorted(glob.glob(str(ALT_DIR / str(season) / "*.json.gz"))):
        with open(f, "rb") as fh:
            e = json.loads(gzip.decompress(fh.read()))
        t = e.get("teams") or {}
        if not (t.get("home") and t.get("away")):
            continue
        h, a = TEAMS.get(t["home"]["names"]["long"]), TEAMS.get(t["away"]["names"]["long"])
        g = by_teams.get((h, a))
        if g is None:
            continue
        for oid, o in (e.get("odds") or {}).items():
            if o.get("periodID") != "game" or o.get("statID") != "points":
                continue
            key = (o.get("statID"), o.get("statEntityID"), o.get("betTypeID"))
            if key not in SIDES:
                continue
            market, _ = SIDES[key]
            side = o.get("sideID")
            for b, v in (o.get("byBookmaker") or {}).items():
                if b not in books:
                    continue
                for a_ in [v] + (v.get("altLines") or []):
                    raw = a_.get("spread") if market == "spread" else a_.get("overUnder")
                    odds, at = _am(a_.get("odds")), a_.get("lastUpdatedAt")
                    if raw in (None, "") or odds is None or not at:
                        continue
                    line = float(raw)
                    if market == "spread" and side == "away":
                        line = -line                     # store every spread from the home side
                    rows.append({"game_id": g["game_id"], "kickoff": g["kickoff_utc"], "book": b, "market": market,
                                 "side": side, "line": line, "odds": odds, "at": at})
    H = pl.DataFrame(rows).with_columns(pl.col("at").str.to_datetime(time_zone="UTC").cast(pl.Datetime("us", "UTC")))
    H = H.sort("at").group_by("game_id", "book", "market", "side", "line", maintain_order=True).last()
    a_side = {"spread": "home", "total": "over"}
    A = H.filter(pl.col("side") == pl.col("market").replace_strict(a_side)).select(
        "game_id", "kickoff", "book", "market", "line", pl.col("odds").alias("a_am"), pl.col("at").alias("a_at"))
    B = H.filter(pl.col("side") != pl.col("market").replace_strict(a_side)).select(
        "game_id", "book", "market", "line", pl.col("odds").alias("b_am"), pl.col("at").alias("b_at"))
    E = A.join(B, on=["game_id", "book", "market", "line"])
    E = E.with_columns(a=pl.col("a_am").map_elements(_dec, return_dtype=pl.Float64),
                       b=pl.col("b_am").map_elements(_dec, return_dtype=pl.Float64))
    return E.with_columns(t_end=pl.max_horizontal("a_at", "b_at"), gap=(1 / pl.col("a") - 1 / pl.col("b")).abs(),
                          hold=1 / pl.col("a") + 1 / pl.col("b") - 1).filter(pl.col("hold").is_between(-0.01, 0.12))


def line_at(E: pl.DataFrame, when: pl.DataFrame) -> pl.DataFrame:
    """The main line at T: among values still quoted at T, those that stopped soonest after it
    (within an hour of the first to stop), and of those the one priced closest to even."""
    X = E.join(when, on="game_id").filter(pl.col("t_end") >= pl.col("T"))
    X = X.with_columns(t_first=pl.col("t_end").min().over("game_id", "book", "market"))
    X = X.filter(pl.col("t_end") <= pl.col("t_first") + timedelta(hours=1))
    return X.sort("gap").group_by("game_id", "book", "market", maintain_order=True).first()


def validate() -> dict:
    """2026: rebuilt Pinnacle line at publish time vs OddsPapi's (data/oddspapi/open_close.parquet)."""
    from pipeline.oddspapi import CACHE
    oc = pl.read_parquet(CACHE / "open_close.parquet").filter(pl.col("book") == "pinnacle")
    E = history(2026, books=("pinnacle",))
    pub = publish_times(E)
    R = line_at(E, pub)
    sp = R.filter(pl.col("market") == "spread").select("game_id", pl.col("line").alias("r_spread"))
    tt = R.filter(pl.col("market") == "total").select("game_id", pl.col("line").alias("r_total"))
    # OddsPapi stores the home spread the nflverse way? compare both signs and report
    J = oc.select("game_id", "publish_spread", "publish_total").join(sp, on="game_id", how="left").join(tt, on="game_id", how="left")
    J = J.filter(pl.col("publish_spread").is_not_null())
    same_sp = (J["r_spread"] == J["publish_spread"]).mean()
    same_sp_flip = (J["r_spread"] == -J["publish_spread"]).mean()
    same_t = (J["r_total"] == J["publish_total"]).mean()
    res = {"games": J.height, "spread_match": float(max(same_sp, same_sp_flip)), "spread_sign_flipped": bool(same_sp_flip > same_sp),
           "total_match": float(same_t), "spread_within_half": float(
               ((J["r_spread"] - (-J["publish_spread"] if same_sp_flip > same_sp else J["publish_spread"])).abs() <= 0.5).mean())}
    print("2026 validation (Pinnacle at publish, rebuilt vs OddsPapi):", res)
    return res


def publish_times(E: pl.DataFrame) -> pl.DataFrame:
    from pipeline.odds_history import publish_time
    k = E.select("game_id", "kickoff").unique()
    return k.with_columns(T=pl.col("kickoff").map_elements(publish_time, return_dtype=pl.Datetime("us", "UTC"))).select("game_id", "T")


def test_2025(flip: bool) -> dict:
    """Tuesday-information projections (backtest.run.walk_forward, as_of_publish) vs Pinnacle's
    rebuilt Tuesday line, and the move to nflverse's closing line."""
    from backtest.run import load_params, walk_forward
    from model.ratings import load_inputs

    E = history(2025, books=("pinnacle",))
    R = line_at(E, publish_times(E))
    inp, g = load_inputs()
    p, alpha = load_params()
    proj = walk_forward(inp, g, p, alpha, [2025], as_of_publish=True).select("game_id", "model_margin", "model_total")
    games = pl.read_parquet(CLEAN / "games.parquet").filter((pl.col("season") == 2025) & (pl.col("game_type") == "REG")).select(
        "game_id", "week", "spread_line", "total_line", "result", "total")
    sp = R.filter(pl.col("market") == "spread").select("game_id", pl.col("line").alias("pub_spread"))
    tt = R.filter(pl.col("market") == "total").select("game_id", pl.col("line").alias("pub_total"))
    D = games.join(proj, on="game_id").join(sp, on="game_id", how="left").join(tt, on="game_id", how="left")
    # nflverse spread_line is the home team's expected margin (home favored = positive); SGO home spread is
    # negative when home is favored, so the market margin is -spread
    D = D.with_columns(pub_margin=-pl.col("pub_spread"))
    out = {}
    for kind, model, start, close, outcome in (("spread", "model_margin", "pub_margin", "spread_line", "result"),
                                               ("total", "model_total", "pub_total", "total_line", "total")):
        x = D.filter(pl.col(start).is_not_null() & pl.col(close).is_not_null())
        gap = (x[model] - x[start]).to_numpy()
        move = (x[close] - x[start]).to_numpy()
        moved = move != 0
        side = np.sign(gap)
        toward = int((np.sign(move[moved]) == side[moved]).sum())
        k = float((gap * move).sum() / (gap ** 2).sum())
        rng = np.random.default_rng(5)
        ks = []
        for _ in range(2000):
            i = rng.integers(0, len(gap), len(gap))
            ks.append((gap[i] * move[i]).sum() / (gap[i] ** 2).sum())
        res_ = x[outcome].to_numpy()
        at_pub = np.sign((res_ - x[start].to_numpy()) * side)
        at_close = np.sign((res_ - x[close].to_numpy()) * side)
        big = np.abs(gap) >= 1.5
        out[kind] = {"games": x.height, "moved": int(moved.sum()), "toward": toward, "away": int(moved.sum()) - toward,
                     "k": k, "k_lo": float(np.quantile(ks, 0.025)), "k_hi": float(np.quantile(ks, 0.975)),
                     "clv_pts": float((move * side).mean()), "clv_pts_big": float((move * side)[big].mean()) if big.any() else None,
                     "big": int(big.sum()),
                     "at_publish": [int((at_pub > 0).sum()), int((at_pub < 0).sum()), int((at_pub == 0).sum())],
                     "at_close": [int((at_close > 0).sum()), int((at_close < 0).sum()), int((at_close == 0).sum())]}
    print("2025, Tuesday line -> close (Pinnacle rebuilt; model with Tuesday information):")
    print(json.dumps(out, indent=1))
    return out


def main() -> None:
    v = validate()
    res = {"validation_2026": v}
    if v["spread_match"] >= 0.9 and v["total_match"] >= 0.9:
        res["clv_2025"] = test_2025(v["spread_sign_flipped"])
    else:
        print("rebuild not reliable enough for game lines; no 2025 test")
    OUT.write_text(json.dumps(res, indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    main()
