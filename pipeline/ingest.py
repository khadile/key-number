"""Download every nflverse dataset the model uses into data/raw/.

Past seasons are downloaded once and kept. The current season is always
re-downloaded, since nflverse updates it through the week.

    python -m pipeline.ingest              # everything, current season refreshed
    python -m pipeline.ingest --only pbp   # one dataset
    python -m pipeline.ingest --refresh    # re-download past seasons too
"""
import argparse
import io
import sys
import time
import urllib.request

import nflreadpy as nfl
import polars as pl

from config import FIRST_SEASON, RAW

CURRENT = nfl.get_current_season()

# name -> (loader, first season available). Loaders take one season.
PER_SEASON = {
    "pbp": (nfl.load_pbp, 2016),
    "player_stats": (lambda s: nfl.load_player_stats(s, summary_level="week"), 2016),
    "team_stats": (lambda s: nfl.load_team_stats(s, summary_level="week"), 2016),
    "rosters_weekly": (nfl.load_rosters_weekly, 2016),
    "injuries": (nfl.load_injuries, 2016),
    "depth_charts": (nfl.load_depth_charts, 2016),
    "snap_counts": (nfl.load_snap_counts, 2016),
    "ngs_passing": (lambda s: nfl.load_nextgen_stats(s, stat_type="passing"), 2016),
    "ngs_rushing": (lambda s: nfl.load_nextgen_stats(s, stat_type="rushing"), 2016),
    "ngs_receiving": (lambda s: nfl.load_nextgen_stats(s, stat_type="receiving"), 2016),
    "pfr_pass": (lambda s: nfl.load_pfr_advstats(s, stat_type="pass"), 2018),
    "pfr_rush": (lambda s: nfl.load_pfr_advstats(s, stat_type="rush"), 2018),
    "pfr_rec": (lambda s: nfl.load_pfr_advstats(s, stat_type="rec"), 2018),
    "pfr_def": (lambda s: nfl.load_pfr_advstats(s, stat_type="def"), 2018),
    "ftn_charting": (nfl.load_ftn_charting, 2022),
    # Participation for a season is only published after its playoffs.
    "participation": (nfl.load_participation, 2016),
}

# Small tables that cover all seasons in one file; always refreshed.
SINGLE = {
    "schedules": lambda: nfl.load_schedules(True),
    "players": nfl.load_players,
    "teams": nfl.load_teams,
    "draft_picks": lambda: nfl.load_draft_picks(True),
    "officials": lambda: nfl.load_officials(True),
    "trades": nfl.load_trades,
    # nflreadpy has no QBR loader; read the nflverse release file directly.
    "espn_qbr_week": lambda: _read_release("espn_data", "qbr_week_level.parquet"),
}


def _read_release(tag: str, asset: str) -> pl.DataFrame:
    url = f"https://github.com/nflverse/nflverse-data/releases/download/{tag}/{asset}"
    with urllib.request.urlopen(url, timeout=60) as r:  # follows GitHub's redirect
        return pl.read_parquet(io.BytesIO(r.read()))


def _save(df: pl.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def ingest(only: set[str] | None, refresh: bool) -> list[str]:
    problems = []
    for name, loader in SINGLE.items():
        if only and name not in only:
            continue
        try:
            df = loader()
            _save(df, RAW / f"{name}.parquet")
            print(f"  {name:<16} {df.height:>8,} rows")
        except Exception as e:  # noqa: BLE001 - report and continue
            problems.append(f"{name}: {e}")

    for name, (loader, first) in PER_SEASON.items():
        if only and name not in only:
            continue
        for season in range(max(first, FIRST_SEASON), CURRENT + 1):
            path = RAW / name / f"{season}.parquet"
            if path.exists() and season < CURRENT and not refresh:
                continue
            if name == "participation" and season == CURRENT:
                continue
            try:
                df = loader(season)
            except Exception as e:  # noqa: BLE001
                problems.append(f"{name} {season}: {str(e)[:120]}")
                continue
            if df.height == 0:
                problems.append(f"{name} {season}: empty")
                continue
            _save(df, path)
            print(f"  {name:<16} {season} {df.height:>8,} rows")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    t = time.time()
    print(f"nflverse ingest, current season {CURRENT}")
    problems = ingest(set(args.only) if args.only else None, args.refresh)
    print(f"done in {time.time() - t:.0f}s")
    if problems:
        print("problems:")
        for p in problems:
            print("  " + p)
        sys.exit(1 if not args.only else 0)


if __name__ == "__main__":
    main()
