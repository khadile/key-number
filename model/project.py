"""Ratings -> projected points for each team, then spread and total.

A ridge regression on standardized features maps a team's matchup row to the
points it scores. It is refit for each test season using earlier seasons only.
"""
from dataclasses import dataclass

import numpy as np
import polars as pl

from model.features import FEATURES, GROUPS


@dataclass
class PointsModel:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        X = (df.select(FEATURES).to_numpy() - self.mean) / self.scale
        return self.intercept + X @ self.coef

    def contributions(self, home_row: dict, away_row: dict) -> dict[str, float]:
        """Points of home-team margin attributable to each feature group."""
        diff = {f: (home_row[f] - away_row[f]) / s * c for f, s, c in zip(FEATURES, self.scale, self.coef)}
        return {name: float(sum(diff[f] for f in feats)) for name, feats in GROUPS.items()}


def fit(train: pl.DataFrame, alpha: float = 50.0) -> PointsModel:
    train = train.filter(pl.col("points").is_not_null())
    X = train.select(FEATURES).to_numpy()
    y = train["points"].to_numpy().astype(float)
    mean, scale = X.mean(0), X.std(0)
    scale[scale == 0] = 1.0
    Z = (X - mean) / scale
    coef = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return PointsModel(mean, scale, coef, float(y.mean()))


def game_projections(rows: pl.DataFrame, model: PointsModel) -> pl.DataFrame:
    """Collapse the two team rows of each game into home/away points, spread, total."""
    pts = rows.with_columns(proj=pl.Series(model.predict(rows)))
    home = pts.filter(pl.col("is_home") == 1).select("game_id", proj_home="proj")
    away = pts.filter(pl.col("is_home") == 0).select("game_id", proj_away="proj")
    return home.join(away, on="game_id").with_columns(
        model_margin=pl.col("proj_home") - pl.col("proj_away"),  # same sign as spread_line
        model_total=pl.col("proj_home") + pl.col("proj_away"),
    )
