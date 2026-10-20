"""Walk-forward test of the NBA game model against closing lines (hoopR's consensus closes, 2020-21 on).

Before each game date the ratings are refit on earlier games only; the day's games are projected and compared
with the consensus close (about 9 books, 5 minutes before tip) and the result.

    python -m nba.backtest                 # 2021-2026 seasons, default parameters
    python -m nba.backtest --grid          # tune half-life / last-season weight / ridge on 2021-2023
"""
import argparse
import itertools
import json

import numpy as np
import polars as pl

from config import REPORTS
from nba.data import load
from nba.model import fit, team_games
from nba.players import fit_values, lineup_value, player_games

OUT = REPORTS / "nba_backtest.parquet"


def closes() -> pl.DataFrame:
    c = load("closing_lines").select(pl.col("game_id").cast(pl.Int64), (-pl.col("home_point")).alias("close_spread"),
                                     pl.col("over_under").alias("close_total"))
    return c.filter(pl.col("close_spread").is_not_null())


def _baselines(pg: pl.DataFrame, v: dict, upto, games: int = 15, half_life: float = 20.0) -> dict:
    """Each team's recent lineups as played (actual minutes x values), newer counting more, before `upto`."""
    rec = (pg.filter((pl.col("date") < upto) & (pl.col("min") > 0))
             .with_columns(val=pl.col("min") / 48.0 * pl.col("athlete_id").replace_strict(v, default=0.0, return_dtype=pl.Float64))
             .group_by("team", "game_id", "date").agg(pl.col("val").sum())
             .sort("date").group_by("team").tail(games)
             .with_columns(w=0.5 ** ((upto - pl.col("date")).dt.total_days() / half_life)))
    b = rec.group_by("team").agg(base=(pl.col("val") * pl.col("w")).sum() / pl.col("w").sum())
    return dict(zip(b["team"], b["base"]))


def run(seasons=range(2021, 2027), lineups: bool = True, **params) -> pl.DataFrame:
    tg = team_games(range(min(seasons) - 2, max(seasons) + 1))
    pg = player_games(range(min(seasons) - 2, max(seasons) + 1)) if lineups else None
    homes = tg.filter(pl.col("home") >= 0).filter(pl.col("is_home") | (pl.col("home") == 0))
    homes = homes.unique("game_id", keep="first")   # one row per game, the home side (neutral: either)
    rest = tg.select("game_id", "team", "rest")
    played = (pg.filter(pl.col("min") > 0).group_by("game_id", "team").agg(pl.col("athlete_id"))
              if lineups else None)
    who = {(g, t): ids for g, t, ids in played.iter_rows()} if lineups else {}
    out = []
    for season in seasons:
        g = homes.filter(pl.col("season") == season)
        vals, vals_at = None, None
        for d in sorted(g["date"].unique()):
            r = fit(tg, d, season, **params)
            if r is None:
                continue
            if lineups and (vals is None or (d - vals_at).days >= 7):
                vals, vals_at = fit_values(pg, tg, d, season), d
            base = _baselines(pg, vals.v, d) if lineups else {}
            for x in g.filter(pl.col("date") == d).iter_rows(named=True):
                p = r.project(x["team"], x["opp"], neutral=x["home"] == 0)
                if p is None:
                    continue
                adj = 0.0
                if lineups:
                    ah = lineup_value(vals, who.get((x["game_id"], x["team"]), [])) - base.get(x["team"], 0.0)
                    aa = lineup_value(vals, who.get((x["game_id"], x["opp"]), [])) - base.get(x["opp"], 0.0)
                    adj = (ah - aa) * p[2] / 100
                out.append({"game_id": x["game_id"], "season": season, "date": d, "home": x["team"], "away": x["opp"],
                            "model_margin": p[0] + adj, "team_margin": p[0], "lineup_adj": adj, "model_total": p[1], "pace": p[2],
                            "result": x["pts"] - x["opp_pts"], "total": x["pts"] + x["opp_pts"]})
    df = pl.DataFrame(out)
    df = (df.join(rest.rename({"team": "home", "rest": "home_rest"}), on=["game_id", "home"], how="left")
            .join(rest.rename({"team": "away", "rest": "away_rest"}), on=["game_id", "away"], how="left"))
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
    side = np.sign(gap)
    cover = np.where(res == 0, np.nan, (np.sign(res) == side).astype(float))
    out = {"games": c.height, "mae_margin_model": mae("model_margin", "result"), "mae_margin_close": mae("close_spread", "result"),
           "mae_total_model": mae("model_total", "total"), "mae_total_close": mae("close_total", "total"),
           "beta_spread": beta, "beta_total": tbeta, "model_side_cover": float(np.nanmean(cover))}
    early = c.filter(pl.col("day") < 21)
    if early.height > 30:
        g2, r2 = (early["model_margin"] - early["close_spread"]).to_numpy(), (early["result"] - early["close_spread"]).to_numpy()
        out["beta_spread_first3wk"] = float((g2 * r2).sum() / (g2 ** 2).sum())
        out["games_first3wk"] = early.height
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    a = ap.parse_args()
    if a.grid:
        best = []
        for hl, pw, lam in itertools.product((25, 45, 80), (0.2, 0.35, 0.6), (10, 30, 80)):
            r = report(run(range(2021, 2024), half_life=hl, prev_weight=pw, lam=lam, lam_p=lam))
            best.append((r["mae_margin_model"], r["mae_total_model"], hl, pw, lam))
            print(f"hl {hl:3d} prev {pw:.2f} lam {lam:3d}: margin MAE {r['mae_margin_model']:.3f} (close {r['mae_margin_close']:.3f}) "
                  f"total MAE {r['mae_total_model']:.3f} (close {r['mae_total_close']:.3f})", flush=True)
        print("best by margin MAE:", sorted(best)[:3])
        return
    df = run()
    df.write_parquet(OUT)
    for s in sorted(df["season"].unique()):
        print(s, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in report(df.filter(pl.col("season") == s)).items()}))
    print("ALL", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in report(df).items()}))


if __name__ == "__main__":
    main()
