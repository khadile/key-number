"""Who plays: player values from game-level adjusted plus-minus, and the lineup adjustment they give a team.

Each team-game is one row: its net rating (points per 100 possessions, minus the opponent's) explained by the
minutes its players played (minutes / 48, so a team sums to 5) minus the opponent's, plus home court. A weighted
ridge fit (newer games count more; last season's at `prev_weight`) gives each player a value in net points per
100 possessions over an average player; players with little time stay near zero.

A team's rating already reflects the lineups it has used. Tonight's adjustment is the value of tonight's expected
lineup minus that recent baseline: expected minutes are each available player's recent average, rescaled so the
team sums to 240 (a star out hands his minutes to the others), and the baseline is the same sum over the team's
recent games as they were actually played.
"""
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse.linalg import spsolve

from nba.data import load


def player_games(seasons) -> pl.DataFrame:
    pb = load("player_box", seasons).filter(pl.col("season_type").is_in([2, 5]))
    return (pb.select(pl.col("game_id").cast(pl.Int64), "season", pl.col("game_date").cast(pl.Date).alias("date"),
                      pl.col("athlete_id").cast(pl.Int64), pl.col("athlete_display_name").alias("name"),
                      pl.col("team_abbreviation").alias("team"), pl.col("minutes").fill_null(0.0).alias("min"),
                      "did_not_play", "starter", "reason")
              .sort("date", "game_id"))


@dataclass
class Values:
    v: dict[int, float]                 # athlete_id -> net points per 100 over average
    usual: pl.DataFrame                 # athlete_id, team, exp_min (recent average when playing), games


def fit_values(pg: pl.DataFrame, tg: pl.DataFrame, before: date, season: int, half_life: float = 120.0,
               prev_weight: float = 0.6, lam: float = 60.0, window_days: int = 760) -> Values:
    t = tg.filter((pl.col("date") < before) & (pl.col("date") >= before - timedelta(days=window_days)))
    t = t.with_columns(net=100 * (pl.col("pts") - pl.col("opp_pts")) / pl.col("poss"))
    p = pg.filter((pl.col("date") < before) & (pl.col("date") >= before - timedelta(days=window_days)) & (pl.col("min") > 0))
    ids = sorted(p["athlete_id"].unique().to_list())
    col = {a: k for k, a in enumerate(ids)}
    rowkey = {(g, tm): r for r, (g, tm) in enumerate(zip(t["game_id"], t["team"]))}
    opp = dict(zip(zip(t["game_id"], t["team"]), t["opp"]))
    ri, ci, val = [], [], []
    for g, tm, a, m in zip(p["game_id"], p["team"], p["athlete_id"], p["min"]):
        r = rowkey.get((g, tm))
        if r is not None:
            ri.append(r); ci.append(col[a]); val.append(m / 48.0)
            ro = rowkey.get((g, opp.get((g, tm))))
            if ro is not None:
                ri.append(ro); ci.append(col[a]); val.append(-m / 48.0)
    n, k = t.height, len(ids)
    X = sparse.csr_matrix((val, (ri, ci)), shape=(n, k + 1))
    X = sparse.hstack([X[:, :k], sparse.csr_matrix(t["home"].to_numpy().astype(float).reshape(-1, 1))]).tocsr()
    age = (before - t["date"]).dt.total_days().to_numpy().astype(float)
    w = 0.5 ** (age / half_life) * np.where(t["season"].to_numpy() < season, prev_weight, 1.0)
    W = sparse.diags(w)
    A = (X.T @ W @ X + sparse.diags(np.r_[np.full(k, lam), 0.0])).tocsc()
    beta = spsolve(A, X.T @ (w * t["net"].to_numpy()))
    v = dict(zip(ids, beta[:k]))
    # usual minutes: recent games he played, newer counting more
    pr = p.with_columns(w=0.5 ** ((before - pl.col("date")).dt.total_days() / 30.0))
    usual = (pr.sort("date").group_by("athlete_id")
             .agg(team=pl.col("team").last(), exp_min=(pl.col("min") * pl.col("w")).sum() / pl.col("w").sum(),
                  games=pl.len(), last=pl.col("date").max()))
    return Values(v, usual)


def lineup_value(vals: Values, players: list[int], team_total: float = 240.0) -> float:
    """Net value of a lineup: each player's usual minutes rescaled to the team's 240, times his value."""
    u = dict(zip(vals.usual["athlete_id"], vals.usual["exp_min"]))
    mins = np.array([u.get(a, 8.0) for a in players], dtype=float)
    if mins.sum() <= 0:
        return 0.0
    mins *= team_total / mins.sum()
    return float(sum(m / 48.0 * vals.v.get(a, 0.0) for a, m in zip(players, mins)))


def baseline(vals: Values, pg: pl.DataFrame, team: str, before: date, half_life: float = 20.0, games: int = 15) -> float:
    """The team's recent lineups as played (actual minutes), newer counting more: what its rating already reflects."""
    rec = pg.filter((pl.col("team") == team) & (pl.col("date") < before) & (pl.col("min") > 0))
    last_games = rec.select("game_id", "date").unique().sort("date").tail(games)
    if last_games.height == 0:
        return 0.0
    rec = rec.join(last_games, on=["game_id", "date"])
    per = (rec.with_columns(val=pl.col("min") / 48.0 * pl.col("athlete_id").replace_strict(vals.v, default=0.0, return_dtype=pl.Float64))
              .group_by("game_id", "date").agg(pl.col("val").sum()))
    w = 0.5 ** ((before - per["date"]).dt.total_days().to_numpy() / half_life)
    return float((per["val"].to_numpy() * w).sum() / w.sum())
