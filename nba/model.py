"""NBA game model: opponent-adjusted team ratings, refit before every game date.

Each team-game gives an offensive rating (points per 100 possessions) and a pace (possessions per game, the
average of both teams' FGA + 0.44 FTA - OREB + TO). Before date D the ratings are a weighted ridge fit on the
games before D, newer games counting more (half-life `half_life` days) and last season's games counting
`prev_weight` as much, which carries a prior into the new season and fades as this season's games arrive:

    ORtg(team vs opp) = mu + O[team] + D[opp] + h * home        (D: points per 100 the defense gives up)
    pace(team vs opp) = mu_p + P[team] + P[opp]

A game then projects to margin = (ORtg_home - ORtg_away) * pace / 100 and total = (ORtg_home + ORtg_away) *
pace / 100, plus rest (back-to-backs, measured on past seasons) and who plays (nba/players.py).
"""
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from nba.data import load


def team_games(seasons) -> pl.DataFrame:
    """One row per team per game (regular season and play-in), with possessions and rest."""
    tb = load("team_box", seasons).filter(pl.col("season_type").is_in([2, 5]))
    sched = load("schedule").select(pl.col("game_id").cast(pl.Int64), pl.col("neutral_site").fill_null(False))
    t = tb.select(
        pl.col("game_id").cast(pl.Int64), "season", pl.col("game_date").cast(pl.Date).alias("date"),
        pl.col("team_abbreviation").alias("team"), pl.col("opponent_team_abbreviation").alias("opp"),
        (pl.col("team_home_away") == "home").alias("is_home"),
        pl.col("team_score").cast(pl.Float64).alias("pts"), pl.col("opponent_team_score").cast(pl.Float64).alias("opp_pts"),
        (pl.col("field_goals_attempted") + 0.44 * pl.col("free_throws_attempted") - pl.col("offensive_rebounds")
         + pl.col("total_turnovers")).cast(pl.Float64).alias("poss_t"))
    t = t.join(t.select("game_id", "team", pl.col("poss_t").alias("poss_o")), left_on=["game_id", "opp"], right_on=["game_id", "team"])
    t = (t.join(sched, on="game_id", how="left")
         .with_columns(poss=(pl.col("poss_t") + pl.col("poss_o")) / 2,
                       home=pl.when(pl.col("neutral_site")).then(0).when(pl.col("is_home")).then(1).otherwise(-1))
         .drop("poss_t", "poss_o")
         .with_columns(ortg=100 * pl.col("pts") / pl.col("poss"))
         .filter(pl.col("ortg").is_finite() & (pl.col("poss") > 60))   # a few games lack box stats (CHI-ORL 2025-12-01)
         .sort("team", "date"))
    prev = pl.col("date").shift(1).over("team")
    return t.with_columns(rest=(pl.col("date") - prev).dt.total_days().fill_null(7).clip(0, 7)).sort("date", "game_id")


@dataclass
class Ratings:
    teams: list[str]
    mu: float
    h: float
    O: np.ndarray
    D: np.ndarray
    mu_p: float
    P: np.ndarray

    def idx(self, t: str) -> int | None:
        try:
            return self.teams.index(t)
        except ValueError:
            return None

    def project(self, home: str, away: str, neutral: bool = False) -> tuple[float, float, float] | None:
        """(margin, total, pace) for home vs away before rest and lineup adjustments."""
        i, j = self.idx(home), self.idx(away)
        if i is None or j is None:
            return None
        s = 0 if neutral else 1
        oh = self.mu + self.O[i] + self.D[j] + self.h * s
        oa = self.mu + self.O[j] + self.D[i] - self.h * s
        pace = self.mu_p + self.P[i] + self.P[j]
        return (oh - oa) * pace / 100, (oh + oa) * pace / 100, pace


def fit(tg: pl.DataFrame, before: date, season: int, half_life: float = 45.0, prev_weight: float = 0.35,
        lam: float = 30.0, lam_p: float = 30.0, window_days: int = 420) -> Ratings | None:
    """Ratings from games before `before`: this season at full weight, earlier ones at `prev_weight`."""
    h = tg.filter((pl.col("date") < before) & (pl.col("date") >= before - timedelta(days=window_days)))
    if h.height < 50:
        return None
    teams = sorted(set(h["team"]) | set(h["opp"]))
    ix = {t: k for k, t in enumerate(teams)}
    n = len(teams)
    age = (before - h["date"]).dt.total_days().to_numpy().astype(float)
    w = 0.5 ** (age / half_life) * np.where(h["season"].to_numpy() < season, prev_weight, 1.0)
    ti = np.array([ix[t] for t in h["team"]])
    oi = np.array([ix[t] for t in h["opp"]])
    hs = h["home"].to_numpy().astype(float)
    rows = np.arange(h.height)
    # offense/defense: columns mu, O[0..n), D[0..n), h
    X = np.zeros((h.height, 2 * n + 2))
    X[:, 0] = 1
    X[rows, 1 + ti] = 1
    X[rows, 1 + n + oi] = 1
    X[:, -1] = hs
    y = h["ortg"].to_numpy()
    pen = np.r_[0.0, np.full(2 * n, lam), 0.0]
    XtW = X.T * w
    beta = np.linalg.solve(XtW @ X + np.diag(pen), XtW @ y)
    # pace: columns mu_p, P[0..n)
    Xp = np.zeros((h.height, n + 1))
    Xp[:, 0] = 1
    np.add.at(Xp, (rows, 1 + ti), 1)
    np.add.at(Xp, (rows, 1 + oi), 1)
    XpW = Xp.T * w
    bp = np.linalg.solve(XpW @ Xp + np.diag(np.r_[0.0, np.full(n, lam_p)]), XpW @ h["poss"].to_numpy())
    return Ratings(teams, float(beta[0]), float(beta[-1]), beta[1:1 + n], beta[1 + n:1 + 2 * n], float(bp[0]), bp[1:])


REST_FEATURES = ("b2b_diff",)


def rest_terms(home_rest: float, away_rest: float) -> dict[str, float]:
    """Back-to-back for each side (second night of two), as a home-minus-away difference."""
    return {"b2b_diff": float(home_rest == 1) - float(away_rest == 1)}
