"""Player prop lines from the OddsPapi archive (data/oddspapi/props/).

The archive holds every price change for every prop line at Pinnacle,
DraftKings and FanDuel, pregame and in-game. Books post a main line (Over and
Under at one number) plus alternate overs; the main line is the one priced on
both sides, closest to even. `main_lines` reads it at any moment.

Prices are decimal odds. `p_over` is the no-vig chance of the over at that
book (both sides' implied probabilities scaled to sum to one).
"""
import glob
from datetime import timedelta

import polars as pl

from config import CLEAN
from pipeline.odds_history import PROPS, fixture_index

MARKET_OF = {"playertotals-passyards": "pass_yds", "playertotals-rushyards": "rush_yds",
             "playertotals-receivingyards": "rec_yds", "playertotals-receptions": "receptions"}
KEYS = ["game_id", "book", "market", "gsis_id"]


def load(game_ids: list[str] | None = None, pregame_only: bool = True) -> pl.DataFrame:
    """Price points for the four main markets, keyed by nflverse game and player ids."""
    idx = fixture_index().with_columns(kickoff=pl.col("start").str.to_datetime(time_zone="UTC"))
    if game_ids is not None:
        idx = idx.filter(pl.col("game_id").is_in(game_ids))
    files = [PROPS / f"{fid}.parquet" for fid in idx["fixture_id"] if (PROPS / f"{fid}.parquet").exists()]
    if not files:
        return pl.DataFrame()
    players = pl.read_parquet(CLEAN / "prop_players.parquet").select(pl.col("op_id").alias("player"), "gsis_id")
    P = (pl.scan_parquet(files).filter(pl.col("market").is_in(list(MARKET_OF)))
         .with_columns(market=pl.col("market").replace_strict(MARKET_OF))
         .collect()
         .join(idx.select("fixture_id", "game_id", "kickoff"), on="fixture_id")
         .join(players, on="player", how="inner"))
    if pregame_only:
        P = P.filter(pl.col("ts") < pl.col("kickoff"))
    return P.drop("fixture_id", "player")


def main_lines(P: pl.DataFrame, cut: pl.Expr) -> pl.DataFrame:
    """Each (game, book, market, player)'s main line as of `cut` (an expression,
    e.g. pl.col("kickoff") - 5 minutes, or a per-game column joined onto P).

    Every (line, side) keeps its last record at or before the cut; a line whose
    last record is inactive had been pulled. Of the lines priced on both sides,
    the main one is closest to even; equal gaps break on the line, so a rerun
    gives the same answer."""
    last = (P.filter(pl.col("ts") <= cut)
            .sort(["ts", "active", "price"], descending=[False, True, False], maintain_order=True)
            .group_by(KEYS + ["handicap", "outcome"])
            .agg(pl.col("price").last(), pl.col("active").last(), pl.col("ts").last().alias("seen"))
            .filter(pl.col("active")))
    o = last.filter(pl.col("outcome") == "Over").select(KEYS + ["handicap", pl.col("price").alias("over"), pl.col("seen").alias("so")])
    u = last.filter(pl.col("outcome") == "Under").select(KEYS + ["handicap", pl.col("price").alias("under"), pl.col("seen").alias("su")])
    pair = o.join(u, on=KEYS + ["handicap"]).with_columns(
        gap=(1 / pl.col("over") - 1 / pl.col("under")).abs(),
        p_over=(1 / pl.col("over")) / (1 / pl.col("over") + 1 / pl.col("under")),
        seen=pl.max_horizontal("so", "su"))
    return (pair.sort(["gap", "handicap"]).group_by(KEYS, maintain_order=True).first()
            .select(KEYS + [pl.col("handicap").alias("line"), "over", "under", "p_over", "seen"]))


def first_seen(P: pl.DataFrame) -> pl.DataFrame:
    return P.group_by(KEYS).agg(first=pl.col("ts").min())


def at_offsets(P: pl.DataFrame, offsets: dict[str, timedelta]) -> pl.DataFrame:
    """Main lines at several moments before each game's kickoff, plus "open"
    (one hour after the book first posted that player's line), long form with a `when` column."""
    out = []
    fs = first_seen(P)
    Pf = P.join(fs, on=KEYS)
    out.append(main_lines(Pf, pl.col("first") + timedelta(hours=1)).with_columns(when=pl.lit("open")))
    for name, off in offsets.items():
        out.append(main_lines(P, pl.col("kickoff") - off).with_columns(when=pl.lit(name)))
    return pl.concat(out)


def files_available() -> int:
    return len(glob.glob(str(PROPS / "*.parquet")))
