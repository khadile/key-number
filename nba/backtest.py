"""Walk-forward test of the NBA game model against closing lines.

The model is nba/strength.py: team ratings refit before every date, both sides' available players valued from
their box scores, rest, and for totals the league's recent scoring level, weighed by a ridge. Each season is
predicted with weights fitted on the seasons before it only, so nothing in a prediction has seen its own season.

    python -m nba.backtest            # seasons 2020-21 on -> reports/nba_backtest.parquet, a line per season
    python -m nba.backtest --train    # also refit nba/model_weights.json on every finished season (the live weights)
"""
import argparse
import json

import numpy as np
import polars as pl

from config import REPORTS
from nba import strength as ST
from nba.data import CURRENT, load

OUT = REPORTS / "nba_backtest.parquet"
FIRST = 2018           # first season in the frame (its player history is 2016-17)
MIN_TRAIN = 2000       # games behind a season's weights


def closes() -> pl.DataFrame:
    c = load("closing_lines").select(pl.col("game_id").cast(pl.Int64), (-pl.col("home_point")).alias("close_spread"),
                                     pl.col("over_under").alias("close_total"))
    return c.filter(pl.col("close_spread").is_not_null())


def run(seasons=range(2021, CURRENT + 1), F: pl.DataFrame | None = None) -> pl.DataFrame:
    F = ST.with_env(ST.frame(range(FIRST, max(seasons) + 1))) if F is None else F
    out = []
    for s in seasons:
        tr, te = F.filter(pl.col("season") < s), F.filter(pl.col("season") == s)
        if te.height and tr.height >= MIN_TRAIN:
            out.append(ST.predict(ST.train(tr), te))
    df = pl.concat(out).select("game_id", "season", "date", "home", "away", "model_margin", "team_margin", "lineup_adj", "rest_adj",
                               "model_total", "tm_total", "pace", "result", "total", "home_rest", "away_rest")
    starts = df.group_by("season").agg(pl.col("date").min().alias("opener"))
    return (df.join(closes(), on="game_id", how="left").join(starts, on="season")
              .with_columns(day=(pl.col("date") - pl.col("opener")).dt.total_days()))


def report(df: pl.DataFrame) -> dict:
    c = df.filter(pl.col("close_spread").is_not_null())
    mae = lambda a, b: float((c[a] - c[b]).abs().mean())
    gap, res = (c["model_margin"] - c["close_spread"]).to_numpy(), (c["result"] - c["close_spread"]).to_numpy()
    beta = float((gap * res).sum() / (gap ** 2).sum())
    tgap, tres = (c["model_total"] - c["close_total"]).to_numpy(), (c["total"] - c["close_total"]).to_numpy()
    tbeta = float((tgap * tres).sum() / (tgap ** 2).sum())
    cover = np.where(res == 0, np.nan, (np.sign(res) == np.sign(gap)).astype(float))
    tcover = np.where(tres == 0, np.nan, (np.sign(tres) == np.sign(tgap)).astype(float))
    # straight up: how often the model's favorite and the market's favorite won
    won = lambda x: float(((np.sign(c[x].to_numpy()) == np.sign(c["result"].to_numpy())) & (c["result"].to_numpy() != 0)).mean())
    out = {"games": c.height, "mae_margin_model": mae("model_margin", "result"), "mae_margin_close": mae("close_spread", "result"),
           "mae_total_model": mae("model_total", "total"), "mae_total_close": mae("close_total", "total"),
           "beta_spread": beta, "beta_total": tbeta, "model_side_cover": float(np.nanmean(cover)),
           "model_total_cover": float(np.nanmean(tcover)), "winner_model": won("model_margin"), "winner_close": won("close_spread")}
    early = c.filter(pl.col("day") < 21)
    if early.height > 30:
        g2, r2 = (early["model_margin"] - early["close_spread"]).to_numpy(), (early["result"] - early["close_spread"]).to_numpy()
        out["beta_spread_first3wk"] = float((g2 * r2).sum() / (g2 ** 2).sum())
        out["games_first3wk"] = early.height
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="refit nba/model_weights.json on every finished season")
    a = ap.parse_args()
    F = ST.with_env(ST.frame(range(FIRST, CURRENT + 1)))
    df = run(F=F)
    df.write_parquet(OUT)
    fmt = lambda r: json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
    for s in sorted(df["season"].unique()):
        if df.filter((pl.col("season") == s) & pl.col("close_spread").is_not_null()).height > 30:
            print(s, fmt(report(df.filter(pl.col("season") == s))))
    print("ALL", fmt(report(df)))
    if a.train:
        W = ST.train(F.filter(pl.col("season") < CURRENT))
        ST.WEIGHTS.write_text(json.dumps(W, indent=1), encoding="utf-8")
        print(f"weights: {W['games']} games, seasons {W['seasons']} -> {ST.WEIGHTS.name}")


if __name__ == "__main__":
    main()
