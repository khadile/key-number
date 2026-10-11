"""A book's main prop line at any moment before kickoff, rebuilt from SportsGameOdds line entries.

SGO keeps, for every line value a book hung (pipeline/sgo_tidy.line_history), the last price
and time it was quoted at that value. The main line at time T is then the two-way, near-even
entry that was still being quoted at T and stopped soonest after it: the smallest last-quote
time at or after T. Two limits, both measured against OddsPapi's full timelines on 2026
(`python -m props.line_history`):

- A value the book came back to keeps only its last visit, so early-week moments can pick
  the wrong line. Late moments (the Saturday lock) suffer least.
- The price is the entry's LAST price at that value, not the price at T (juice often moves
  before the line does).
- Books that trade props in-game (DraftKings, FanDuel) keep quoting the closing value after
  kickoff, so when the line did not move after T its pregame price is gone: those rows come
  back with `price_known` false. Books that stop at kickoff (Pinnacle, bet365) lose nothing.
"""
from datetime import timedelta

import polars as pl

from pipeline.sgo import ROOT as SGO

KEY = ["season", "game_id", "sgo_player", "market", "book"]


def dec(am: pl.Expr) -> pl.Expr:
    return pl.when(am > 0).then(1 + am / 100).otherwise(1 + 100 / (-am))


def entries(H: pl.DataFrame | None = None) -> pl.DataFrame:
    """Two-way, near-even (main-like) line entries: one row per (book, line) with both prices."""
    H = H if H is not None else pl.read_parquet(SGO / "prop_line_history.parquet")
    o = H.filter(pl.col("side") == "over").select(KEY + ["starts_at", "line", pl.col("odds").alias("o_am"), pl.col("last_at").alias("o_at")])
    u = H.filter(pl.col("side") == "under").select(KEY + ["line", pl.col("odds").alias("u_am"), pl.col("last_at").alias("u_at")])
    E = o.join(u, on=KEY + ["line"]).with_columns(over=dec(pl.col("o_am")), under=dec(pl.col("u_am")))
    E = E.with_columns(hold=1 / pl.col("over") + 1 / pl.col("under") - 1, t_end=pl.max_horizontal("o_at", "u_at"))
    return E.filter(pl.col("over").is_between(1.4, 2.8) & pl.col("under").is_between(1.4, 2.8)
                    & pl.col("hold").is_between(-0.01, 0.15))


def line_at(E: pl.DataFrame, when: pl.DataFrame) -> pl.DataFrame:
    """Each (game, player, market, book)'s main line at the moment in `when` (columns game_id, T).
    Returns line, over, under, p_over (no-vig), and `price_known` (the entry ended before kickoff)."""
    X = E.join(when, on="game_id").filter(pl.col("t_end") >= pl.col("T"))
    X = X.sort("t_end").group_by(KEY, maintain_order=True).first()
    return X.with_columns(p_over=(1 / pl.col("over")) / (1 / pl.col("over") + 1 / pl.col("under")),
                          price_known=pl.col("t_end") < pl.col("starts_at"))


def main() -> None:
    """Validate against OddsPapi's 2026 timelines at each game's real lock time."""
    from config import CLEAN
    from props.lines import load as load_lines, main_lines
    from props.pricing import lock_time

    M = pl.read_parquet(SGO / "props.parquet").filter(pl.col("gsis_id").is_not_null()).select("game_id", "sgo_player", "market", "gsis_id")
    E = entries().filter(pl.col("season") == 2026)
    games = E["game_id"].unique().to_list()
    P = load_lines(games)
    kick = P.select("game_id", "kickoff").unique().with_columns(
        T=pl.col("kickoff").map_elements(lock_time, return_dtype=pl.Datetime("us", "UTC")))
    truth = main_lines(P.join(kick.select("game_id", "T"), on="game_id"), pl.col("T")).select(
        "game_id", "book", "market", "gsis_id", pl.col("line").alias("t_line"), pl.col("over").alias("t_over"),
        pl.col("under").alias("t_under"), pl.col("p_over").alias("t_p_over"))
    for label, off in (("lock (Thu/Sat 10am ET)", None), ("3 hours before kickoff", timedelta(hours=3))):
        when = kick.select("game_id", T=(pl.col("T") if off is None else pl.col("kickoff") - off))
        if off is not None:
            truth_w = main_lines(P.join(when, on="game_id"), pl.col("T")).select(
                "game_id", "book", "market", "gsis_id", pl.col("line").alias("t_line"), pl.col("over").alias("t_over"),
                pl.col("under").alias("t_under"), pl.col("p_over").alias("t_p_over"))
        else:
            truth_w = truth
        R = line_at(E, when).join(M, on=["game_id", "sgo_player", "market"])
        J = truth_w.join(R, on=["game_id", "book", "market", "gsis_id"], how="left")
        print(f"\n== {label}: SGO rebuild vs OddsPapi")
        print(J.group_by("book").agg(
            n=pl.len(), found=pl.col("line").is_not_null().mean(),
            price_known=(pl.col("price_known").fill_null(False)).mean(),
            same_line=(pl.col("line") == pl.col("t_line")).mean(),
            same_line_price_known=((pl.col("line") == pl.col("t_line")) & pl.col("price_known")).sum() / pl.col("price_known").fill_null(False).sum(),
            p_over_err=((pl.col("p_over") - pl.col("t_p_over")).abs()).filter((pl.col("line") == pl.col("t_line")) & pl.col("price_known")).mean(),
        ).sort("book"))


if __name__ == "__main__":
    main()
