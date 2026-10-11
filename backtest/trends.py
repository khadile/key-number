"""Do in-season trends ("x factors") predict the next game against the closing line?

Tests team over/ATS streaks, recent totals residuals, scoring vs our model, and
league-wide early-season tendencies. Regular season only, 2016-2025.

    python -m backtest.trends

Requires reports/projections.parquet from `python -m backtest.run`.
"""
import numpy as np
import polars as pl

from config import CLEAN, REPORTS


def team_rows() -> pl.DataFrame:
    g = pl.read_parquet(CLEAN / "games.parquet").filter(
        pl.col("result").is_not_null() & pl.col("spread_line").is_not_null() & (pl.col("game_type") == "REG")
    ).join(pl.read_parquet(REPORTS / "projections.parquet").select("game_id", "proj_home", "proj_away"),
           on="game_id", how="left")

    def side(home: bool) -> pl.DataFrame:
        t, o, sign = ("home", "away", 1) if home else ("away", "home", -1)
        return g.select(
            "game_id", "season", "week", "total", "total_line",
            team=pl.col(f"{t}_team"), pf=pl.col(f"{t}_score"), pa=pl.col(f"{o}_score"),
            cover_m=(pl.col("result") - pl.col("spread_line")) * sign,
            exp_pf=pl.col(f"proj_{t}"), exp_pa=pl.col(f"proj_{o}"),
        )

    T = pl.concat([side(True), side(False)]).sort("team", "season", "week")
    T = T.with_columns(
        over=pl.when(pl.col("total") > pl.col("total_line")).then(1.0)
        .when(pl.col("total") < pl.col("total_line")).then(0.0),
        cover=pl.when(pl.col("cover_m") > 0).then(1.0).when(pl.col("cover_m") < 0).then(0.0),
        tot_resid=pl.col("total") - pl.col("total_line"),
        off_resid=pl.col("pf") - pl.col("exp_pf"),
        def_resid=pl.col("pa") - pl.col("exp_pa"),
    )
    grp = ["team", "season"]

    def so_far(col: str):  # season-to-date rate, excluding the current game
        hits = pl.col(col).fill_null(0).cum_sum().over(grp) - pl.col(col).fill_null(0)
        n = pl.col(col).is_not_null().cast(pl.Int32)
        return hits, n.cum_sum().over(grp) - n

    oh, on = so_far("over")
    ch, cn = so_far("cover")
    return T.with_columns(
        over_rate_sofar=oh / on, overs_n=on, cov_rate_sofar=ch / cn, cov_n=cn,
        last3_tot=pl.col("tot_resid").shift(1).rolling_mean(3).over(grp),
        last3_def=pl.col("def_resid").shift(1).rolling_mean(3).over(grp),
        last3_off=pl.col("off_resid").shift(1).rolling_mean(3).over(grp),
    )


def bucket(title: str, df: pl.DataFrame, col: str, bins, target: str) -> None:
    print("\n" + title)
    for lo, hi in bins:
        x = df.filter(pl.col(col).is_between(lo, hi, closed="left") & pl.col(target).is_not_null())[target]
        if len(x):
            p = x.mean()
            print(f"  {col} [{lo:+},{hi:+})  n={len(x):5d}  next-game {target} {p:.1%}  (+/- {1.96 * np.sqrt(p * (1 - p) / len(x)):.1%})")


def main() -> None:
    T = team_rows()
    S = T.filter(pl.col("season").is_between(2016, 2025))
    M = T.filter(pl.col("season").is_between(2018, 2025))  # seasons with out-of-sample projections
    bucket("A. Team over record so far (>=3 games) -> next game over", S.filter(pl.col("overs_n") >= 3),
           "over_rate_sofar", [(0, .26), (.26, .5), (.5, .75), (.75, 1.01)], "over")
    bucket("   Perfect over/under record so far (>=4 games)", S.filter(pl.col("overs_n") >= 4),
           "over_rate_sofar", [(0, .01), (1, 1.01)], "over")
    bucket("B. Team ATS record so far (>=3 games) -> next game cover", S.filter(pl.col("cov_n") >= 3),
           "cov_rate_sofar", [(0, .26), (.26, .5), (.5, .75), (.75, 1.01)], "cover")
    bucket("C. Avg (total - closing total), last 3 -> next over", S, "last3_tot",
           [(-99, -7), (-7, -3), (-3, 3), (3, 7), (7, 99)], "over")
    bucket("D. Points allowed above our model, last 3 -> next over", M, "last3_def",
           [(-99, -6), (-6, -2), (-2, 2), (2, 6), (6, 99)], "over")
    bucket("E. Points scored above our model, last 3 -> next over", M, "last3_off",
           [(-99, -6), (-6, -2), (-2, 2), (2, 6), (6, 99)], "over")

    g = pl.read_parquet(CLEAN / "games.parquet").filter(
        pl.col("result").is_not_null() & (pl.col("game_type") == "REG") & pl.col("season").is_between(2016, 2025))
    dog_sign = pl.when(pl.col("spread_line") > 0).then(-1).otherwise(1)
    Y = g.with_columns(
        dog=pl.when(pl.col("spread_line") == 0).then(None)
        .otherwise((pl.col("result") - pl.col("spread_line")) * dog_sign > 0).cast(pl.Float64),
        ov=pl.when(pl.col("total") == pl.col("total_line")).then(None)
        .otherwise(pl.col("total") > pl.col("total_line")).cast(pl.Float64),
        early=pl.col("week") <= 4,
    ).group_by("season").agg(
        dog_w1_4=pl.col("dog").filter(pl.col("early")).mean(), dog_w5_18=pl.col("dog").filter(~pl.col("early")).mean(),
        over_w1_4=pl.col("ov").filter(pl.col("early")).mean(), over_w5_18=pl.col("ov").filter(~pl.col("early")).mean(),
    ).sort("season")
    print("\nF. League-wide: does an early-season tendency carry into weeks 5-18?")
    print(Y)
    print(f"  corr dogs {np.corrcoef(Y['dog_w1_4'], Y['dog_w5_18'])[0, 1]:+.2f}   "
          f"corr overs {np.corrcoef(Y['over_w1_4'], Y['over_w5_18'])[0, 1]:+.2f}   (10 seasons)")


if __name__ == "__main__":
    main()
