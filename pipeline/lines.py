"""Snapshot the current betting lines so we keep a history nflverse does not.

nflverse `schedules` overwrites lines in place. Each run of this module saves
the current season's lines with a UTC timestamp to data/lines/. From those
files:

- the publish-time line is the snapshot a pick was made against
- the closing line is the last snapshot taken before kickoff (for CLV)

    python -m pipeline.lines           # take a snapshot now
    python -m pipeline.lines --show    # print line history for upcoming games
"""
import argparse
from datetime import datetime, timezone

import nflreadpy as nfl
import polars as pl

from config import LINES
from pipeline.clean import FRANCHISE

LINE_COLS = [
    "game_id", "season", "week", "gameday", "gametime", "away_team", "home_team",
    "spread_line", "home_spread_odds", "away_spread_odds", "total_line",
    "over_odds", "under_odds", "home_moneyline", "away_moneyline",
    "home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name",
]


def take_snapshot() -> pl.DataFrame:
    season = nfl.get_current_season()
    now = datetime.now(timezone.utc)
    s = nfl.load_schedules(season)
    snap = (
        s.filter(pl.col("result").is_null())  # only games not yet played
        .select(LINE_COLS)
        .with_columns(
            pl.col("home_team").replace(FRANCHISE),
            pl.col("away_team").replace(FRANCHISE),
            snapshot_utc=pl.lit(now),
        )
    )
    snap.write_parquet(LINES / f"{now:%Y%m%dT%H%M%SZ}.parquet")
    return snap


def history() -> pl.DataFrame:
    files = sorted(LINES.glob("*.parquet"))
    if not files:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


def line_at(game_id: str, when: datetime) -> dict | None:
    """The most recent snapshot of a game's line at or before `when`."""
    h = history()
    if h.height == 0:
        return None
    rows = h.filter((pl.col("game_id") == game_id) & (pl.col("snapshot_utc") <= when)).sort("snapshot_utc")
    return rows.row(-1, named=True) if rows.height else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if args.show:
        h = history()
        print(h.select("snapshot_utc", "game_id", "spread_line", "total_line", "home_moneyline").sort("game_id", "snapshot_utc"))
        return
    snap = take_snapshot()
    with_lines = snap.filter(pl.col("spread_line").is_not_null()).height
    print(f"snapshot saved: {snap.height} upcoming games, {with_lines} with lines")


if __name__ == "__main__":
    main()
