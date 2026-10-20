"""hoopR NBA data: the NBA's nflverse (sportsdataverse/hoopR-nba-data on GitHub, free, 2002 to now, ESPN-sourced).

Seasons are named by the year they end: 2026 is 2025-26, 2027 is 2026-27. Finished seasons are downloaded once;
the current season and the schedule are refreshed on every build (the repo updates daily).

    python -m nba.data            # download / refresh into data/nba/raw/
"""
import argparse
import time
import urllib.request

import polars as pl

from config import DATA

RAW = DATA / "nba" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
BASE = "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main/nba"
FIRST = 2017          # nine seasons of history before 2026-27; enough for priors and walk-forward tests
CURRENT = 2027        # 2026-27
PER_SEASON = {
    "team_box": "team_box/parquet/team_box_{s}.parquet",
    "player_box": "player_box/parquet/player_box_{s}.parquet",
    "rosters": "game_rosters/parquet/game_rosters_{s}.parquet",
    "officials": "officials/parquet/officials_{s}.parquet",
}
SINGLE = {
    "schedule": "schedules/nba_schedule_master.parquet",
    "closing_lines": "betting_lines/closing_lines_odds_api.parquet",
}


def _get(url: str, dest, tries: int = 4) -> bool:
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "nfl_model"}), timeout=60) as r:
                body = r.read()
            dest.write_bytes(body)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False          # a season with no games yet
            time.sleep(2 ** i)
        except OSError:
            time.sleep(2 ** i)
    raise RuntimeError(f"hoopR download failed: {url}")


def path(kind: str, season: int | None = None):
    return RAW / (f"{kind}_{season}.parquet" if season else f"{kind}.parquet")


def refresh(full: bool = False) -> None:
    """Fetch what is missing, plus the current season and the single files every time."""
    got = []
    for kind, tpl in PER_SEASON.items():
        for s in range(FIRST, CURRENT + 1):
            p = path(kind, s)
            if p.exists() and s < CURRENT and not full:
                continue
            if _get(f"{BASE}/{tpl.format(s=s)}", p):
                got.append(p.name)
    for kind, rel in SINGLE.items():
        if _get(f"{BASE}/{rel}", path(kind)):
            got.append(path(kind).name)
    print(f"hoopR: {len(got)} files fetched into {RAW}")


def load(kind: str, seasons: range | list | None = None) -> pl.DataFrame:
    """One table across seasons (per-season kinds), or the single file."""
    if kind in SINGLE:
        return pl.read_parquet(path(kind))
    seasons = seasons or range(FIRST, CURRENT + 1)
    frames = [pl.read_parquet(path(kind, s)) for s in seasons if path(kind, s).exists()]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="re-download finished seasons too")
    refresh(ap.parse_args().full)


if __name__ == "__main__":
    main()
