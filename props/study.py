"""How the props market behaves this season, and where the model stands against it.

    python -m props.study

Uses every finished 2026 game in the OddsPapi archive, tracked roles only
(QB1, RB1-2, WR1-2, TE1). Four questions:

1. Report card: at the closing line, how often do overs hit vs what the no-vig
   price said, and what would blindly betting every over or every under have
   returned at DraftKings and FanDuel?
2. Model vs close: is the model's chance of the over better or worse than the
   market's (Brier score) at the same line?
3. Line movement: from each candidate lock time to the close, do lines move
   toward the model? The slope k (as in the game-line model) prices picks.
4. Lock time: the moment where the model's edge, measured by that movement, is
   biggest with enough lines posted to pick from.

Writes reports/props_study.json for the board.
"""
import json
from datetime import timedelta

import numpy as np
import polars as pl

from config import REPORTS
from props.backtest import ROLE_MARKETS, baselines, frames, load
from props.lines import at_offsets, load as load_lines
from props.model import STAT_COL, Simulator

OFFSETS = {"tue": timedelta(days=5), "thu": timedelta(days=3), "sat": timedelta(hours=27),
           "t3h": timedelta(hours=3), "close": timedelta(minutes=5)}
OUT = REPORTS / "props_study.json"


def tracked_2026(season: int = 2026) -> pl.DataFrame:
    p, _ = load()
    x = baselines(frames(p)).filter((pl.col("season") == season) & pl.col("role").is_not_null())
    return x


def sample_quantiles(x: pl.DataFrame, size: int = 4000, seed: int = 3) -> dict[tuple[str, str, str], np.ndarray]:
    """Sorted simulated samples per (game_id, gsis_id, market) for the tracked rows."""
    _, disp = load()
    sim = Simulator(disp, seed=seed)
    S = sim.simulate(x, size)
    out = {}
    gid, pid, role = x["game_id"].to_list(), x["gsis_id"].to_list(), x["role"].to_list()
    for i in range(x.height):
        for m in ROLE_MARKETS[role[i]]:
            out[(gid[i], pid[i], m)] = np.sort(S[m][i])
    return out


def p_over(s: np.ndarray, line: float) -> float:
    """Chance the stat beats `line` (a whole-number line pushes half the tie away)."""
    return float((s > line).mean())


def delta(s: np.ndarray, line: float, p: float) -> float:
    """How far the market's distribution sits from the model's, in stat units: the
    shift that makes the model's own shape give the market's no-vig chance at its line."""
    q = float(np.quantile(s, min(max(1 - p, 0.001), 0.999)))
    return line - q


def main() -> None:
    x = tracked_2026()
    finished = x.filter(pl.col(STAT_COL["rec_yds"]).is_not_null())
    games = finished["game_id"].unique().to_list()
    P = load_lines(games)
    A = at_offsets(P, OFFSETS)
    print(f"{len(games)} finished games, {A.height} book-lines")
    S = sample_quantiles(finished)
    res = {r["gsis_id"] + r["game_id"]: r for r in finished.iter_rows(named=True)}
    rows = []
    for r in A.iter_rows(named=True):
        key = (r["game_id"], r["gsis_id"], r["market"])
        s = S.get(key)
        row = res.get(r["gsis_id"] + r["game_id"])
        if s is None or row is None:
            continue
        y = float(row[STAT_COL[r["market"]]])
        rows.append({**r, "role": row["role"], "actual": y, "model_p": p_over(s, r["line"]),
                     "model_med": float(np.median(s)), "model_sd": float(s.std()),
                     "delta": delta(s, r["line"], r["p_over"]),
                     "over_won": y > r["line"], "push": y == r["line"]})
    D = pl.DataFrame(rows)
    print(f"{D.height} priced lines for tracked players")

    # 1) report card at the close
    close = D.filter((pl.col("when") == "close") & ~pl.col("push"))
    card = close.group_by("book", "market").agg(
        n=pl.len(), said_over=pl.col("p_over").mean(), over_hit=pl.col("over_won").mean(),
        roi_over=(pl.when(pl.col("over_won")).then(pl.col("over") - 1).otherwise(-1.0)).mean(),
        roi_under=(pl.when(~pl.col("over_won")).then(pl.col("under") - 1).otherwise(-1.0)).mean(),
    ).sort("book", "market")
    print("report card (close):\n", card)

    # 2) model vs market at the same line, closing lines
    brier = close.group_by("market").agg(
        n=pl.len(),
        books=((pl.col("p_over") - pl.col("over_won").cast(pl.Float64)) ** 2).mean(),
        model=((pl.col("model_p") - pl.col("over_won").cast(pl.Float64)) ** 2).mean(),
        blend=(((pl.col("p_over") + pl.col("model_p")) / 2 - pl.col("over_won").cast(pl.Float64)) ** 2).mean(),
    ).sort("market")
    print("Brier at close (lower is better):\n", brier)

    # 3) movement toward the model, reference = Pinnacle else the DK/FD average location
    def ref(df: pl.DataFrame) -> pl.DataFrame:
        k = ["game_id", "gsis_id", "market", "when"]
        pin = df.filter(pl.col("book") == "pinnacle").select(k + [pl.col("delta").alias("d_pin"), "model_sd"])
        us = df.filter(pl.col("book") != "pinnacle").group_by(k).agg(d_us=pl.col("delta").mean(), sd_us=pl.col("model_sd").first())
        j = us.join(pin, on=k, how="full", coalesce=True)
        return j.with_columns(d=pl.coalesce("d_pin", "d_us"), sd=pl.coalesce("model_sd", "sd_us")).select(k + ["d", "sd"])

    R = ref(D)
    closeR = R.filter(pl.col("when") == "close").select("game_id", "gsis_id", "market", pl.col("d").alias("d_close"))
    moves = []
    for when in ("open", "tue", "thu", "sat", "t3h"):
        m = R.filter(pl.col("when") == when).join(closeR, on=["game_id", "gsis_id", "market"])
        if not m.height:
            continue
        gap = (-m["d"] / m["sd"]).to_numpy()                 # model minus market, in model SDs
        mv = ((m["d_close"] - m["d"]) / m["sd"]).to_numpy()  # market's move to the close
        moved = mv != 0
        k = float(np.clip((gap * mv).sum() / (gap ** 2).sum(), 0, 0.5)) if len(gap) else 0.0
        toward = int((np.sign(gap[moved]) == np.sign(mv[moved])).sum())
        moves.append({"when": when, "n": int(len(gap)), "moved": int(moved.sum()), "toward": toward,
                      "k": round(k, 3), "clv_sd": round(float((np.sign(gap) * mv).mean()), 4),
                      "big_gap_n": int((np.abs(gap) > 0.25).sum()),
                      "big_gap_toward": int(((np.abs(gap) > 0.25) & moved & (np.sign(gap) == np.sign(mv))).sum()),
                      "big_gap_moved": int(((np.abs(gap) > 0.25) & moved).sum())})
    M = pl.DataFrame(moves)
    print("movement toward the model:\n", M)

    REPORTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"games": len(games), "lines": D.height, "card": card.to_dicts(),
                               "brier": brier.to_dicts(), "moves": M.to_dicts()}, indent=1, default=str))
    D.write_parquet(REPORTS / "props_study_lines.parquet")
    print("saved", OUT)


if __name__ == "__main__":
    main()
