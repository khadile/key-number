"""Quarterback ratings and the starter adjustment.

A QB's rating is his opponent-adjusted qb_epa per dropback, decayed by age and
pulled toward a prior. The prior depends on draft round and fades as he builds
a career: a rookie R4+/UDFA starts well below average, a 1,000-dropback
veteran starts at average and is judged on his own plays.

The team pass rating already contains whoever took the snaps. The adjustment
for a game is:

    qb_adj = rating(starter) - sum(share_i * rating(qb_i))

where share_i is each QB's share of the team's weighted dropbacks inside the
rating window. Zero means the starter is the QB already in the numbers.
"""
import numpy as np
import polars as pl

from config import RAW, TUNE_LAST
from model.ratings import Inputs, Params, _decay

PRIOR_FADE_DROPBACKS = 1000.0


def rookie_priors(inp: Inputs) -> dict[str, float]:
    """Mean qb_epa vs league over a QB's first 150 dropbacks, by draft bucket.

    Estimated from QBs who debuted 2017..TUNE_LAST so the holdout seasons never
    inform the prior.
    """
    q = inp.qb.sort("season", "week").with_columns(
        cum=pl.col("n").cum_sum().over("qb_id") - pl.col("n"),
        first=pl.col("season").min().over("qb_id"),
    )
    lg = q.group_by("season").agg(lg=pl.col("wy").sum() / pl.col("w").sum())
    early = (
        q.join(lg, on="season")
        .filter((pl.col("cum") < 150) & pl.col("first").is_between(2017, TUNE_LAST))
        .join(_buckets(), on="qb_id", how="left")
        .with_columns(pl.col("bucket").fill_null("R4+"))
        .group_by("bucket")
        .agg(prior=(pl.col("wy").sum() - (pl.col("lg") * pl.col("w")).sum()) / pl.col("w").sum())
    )
    return dict(zip(early["bucket"], early["prior"]))


def _buckets() -> pl.DataFrame:
    players = pl.read_parquet(RAW / "players.parquet", columns=["gsis_id", "draft_round"])
    return players.select(
        qb_id=pl.col("gsis_id"),
        bucket=pl.when(pl.col("draft_round") == 1).then(pl.lit("R1"))
        .when(pl.col("draft_round").is_in([2, 3])).then(pl.lit("R2-3"))
        .otherwise(pl.lit("R4+")),
    )


class QBModel:
    def __init__(self, inp: Inputs, snaps: dict, p: Params):
        self.p = p
        priors = rookie_priors(inp)
        q = inp.qb.join(_buckets(), on="qb_id", how="left").with_columns(pl.col("bucket").fill_null("R4+"))
        self.ids = sorted(q["qb_id"].unique().to_list())
        self.qidx = {k: i for i, k in enumerate(self.ids)}
        bucket_of = dict(zip(q["qb_id"], q["bucket"]))
        self.rookie_prior = np.array([priors.get(bucket_of[k], priors.get("R4+", -0.15)) for k in self.ids])
        self.default_prior = priors.get("R4+", -0.15)  # a QB nflverse has never seen

        # Opponent adjustment uses the defense rating as of each game's own week.
        adj = np.empty(q.height)
        seasons, weeks = q["season"].to_numpy(), q["week"].to_numpy()
        dfn, home = q["defteam"].to_list(), q["home"].to_numpy()
        for i in range(q.height):
            s = snaps[(seasons[i], weeks[i])]
            adj[i] = s["mu_pass_epa"] + s["def_pass_epa"][inp.idx[dfn[i]]] + s["home_pass_epa"] * home[i]
        self.season, self.week = seasons, weeks
        self.w = q["w"].to_numpy()
        self.wy_adj = q["wy"].to_numpy() - self.w * adj
        self.n = q["n"].to_numpy().astype(float)
        self.qi = np.array([self.qidx[k] for k in q["qb_id"]])
        self.team = np.array([inp.idx[t] for t in q["posteam"]])
        self.n_teams = len(inp.teams)

    def ratings(self, season: int, week: int) -> np.ndarray:
        """Every QB's rating entering (season, week), relative to league average."""
        p, m = self.p, len(self.ids)
        d = _decay(self.season, self.week, season, week, p.qb_half_life, p.qb_gap)
        before = d > 0
        career = np.bincount(self.qi[before], self.n[before], m)
        prior = self.rookie_prior * np.clip(1 - career / PRIOR_FADE_DROPBACKS, 0, 1)
        W = np.bincount(self.qi, self.w * d, m)
        WY = np.bincount(self.qi, self.wy_adj * d, m)
        return (WY + p.alpha_qb * prior) / (W + p.alpha_qb)

    def team_mix(self, season: int, week: int) -> np.ndarray:
        """Rating of the QB blend baked into each team's pass rating."""
        p = self.p
        r = self.ratings(season, week)
        d = _decay(self.season, self.week, season, week, *p.speed("pass", "off"))  # same decay as the team pass rating
        Wt = np.bincount(self.team, self.w * d, self.n_teams)
        mix = np.bincount(self.team, self.w * d * r[self.qi], self.n_teams)
        return np.divide(mix, Wt, out=np.zeros(self.n_teams), where=Wt > 0)

    def rating_of(self, r: np.ndarray, qb_id: str | None) -> float:
        if qb_id is None or qb_id not in self.qidx:
            return self.default_prior
        return float(r[self.qidx[qb_id]])
