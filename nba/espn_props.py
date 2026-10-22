"""NBA player-prop line history from ESPN's public odds API (user, 2026-10-22: back-test the NBA model).

ESPN keeps each game's player props from its odds provider(s): opening and latest line and price per side.
What it holds (checked 2026-10-22 on one game per season):

- 2022-23: DraftKings, Caesars, MGM, SugarHouse. The "latest" price is often from during the game (points
  lines 4-9 points off the open), so only the OPEN is clean.
- 2023-24: ESPN BET, Caesars, MGM, SugarHouse; latest lines within 1-2 points of the open (pregame-like).
- 2024-25: ESPN BET (plus a separate live feed, skipped); latest lines can be in-game (4-13 points off).
- 2025-26 and 2026-27: DraftKings; latest lines within 1-2 points of the open (pregame-like).

Which latest prices are clean is decided per prop downstream (nba/prop_backtest.py), never assumed here.
ESPN does not label the over and the under, and besides the main line it can list one-sided alternate lines
in the same market. Every entry is kept in its order (`seq`); the main line is the pair of entries sharing a line,
over first, under second (checked: when a line drops, the second price was the favorite at the open; and in the
back-test the over's hit rate rises with the first price's chance). Odds data: data/nba/ stays off the public repo.

    python -m nba.espn_props 2023 2024 2025 2026     # seasons by the year they end; resumable
"""
import argparse
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import polars as pl

from config import DATA

OUT = DATA / "nba" / "espn_props"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{e}/competitions/{e}/odds"
MARKETS = {"Total Points": "pts", "Total Rebounds": "reb", "Total Assists": "ast", "Total 3-Point Field Goals": "fg3",
           "Total Points, Rebounds, and Assists": "pra", "Total Points and Rebounds": "pr",
           "Total Points and Assists": "pa", "Total Assists and Rebounds": "ra"}
WORKERS = 4
_pace = threading.Semaphore(WORKERS)


def _get(url: str, tries: int = 4) -> dict | None:
    for k in range(tries):
        try:
            with _pace, urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "key-number"}), timeout=30) as r:
                time.sleep(0.15)
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 * (k + 1))
        except Exception:   # noqa: BLE001  (timeouts, resets: back off and retry)
            time.sleep(2 * (k + 1))
    return None


def _num(x) -> float | None:
    try:
        return float(str(x).replace("+", ""))
    except (TypeError, ValueError):
        return None


def game(e: int) -> list[dict]:
    """Every core player-prop entry at every pregame provider for one game, in ESPN's order."""
    odds = _get(BASE.format(e=e) + "?lang=en&region=us")
    rows = []
    for it in (odds or {}).get("items", []):
        prov = it.get("provider", {})
        if "live" in (prov.get("name") or "").lower():
            continue
        pb = _get(BASE.format(e=e) + f"/{prov['id']}/propBets?lang=en&region=us&limit=1000")
        for seq, p in enumerate((pb or {}).get("items", [])):
            m = MARKETS.get((p.get("type") or {}).get("name"))
            if not m or "athlete" not in p:
                continue
            aid = int(p["athlete"]["$ref"].split("/athletes/")[1].split("?")[0])
            o = p.get("odds", {})
            rows.append({"game_id": e, "provider": int(prov["id"]), "provider_name": prov.get("name"), "athlete_id": aid,
                         "market": m, "seq": seq,
                         "line_open": _num((p.get("open") or {}).get("target", {}).get("value")),
                         "line": _num((p.get("current") or {}).get("target", {}).get("value")),
                         "am_open": _num(o.get("american", {}).get("open")), "am": _num(o.get("american", {}).get("value")),
                         "updated": p.get("lastUpdated")})
    return rows


def pull(season: int) -> None:
    s = pl.read_parquet(DATA / "nba" / "raw" / "schedule.parquet").filter((pl.col("season") == season) & (pl.col("season_type") == 2)
                                                                           & pl.col("status_type_completed"))
    ids = sorted(int(i) for i in s["id"].to_list())
    path, done_p = OUT / f"props_{season}.parquet", OUT / f"done_{season}.json"
    done = set(json.loads(done_p.read_text())) if done_p.exists() else set()
    todo = [i for i in ids if i not in done]
    print(f"{season}: {len(ids)} games, {len(todo)} to pull")
    have = [pl.read_parquet(path)] if path.exists() else []
    for c in range(0, len(todo), 100):
        chunk = todo[c:c + 100]
        with ThreadPoolExecutor(WORKERS) as ex:
            got = list(ex.map(game, chunk))
        rows = [r for g in got for r in g]
        if rows:
            have.append(pl.DataFrame(rows, infer_schema_length=None))
        done.update(chunk)
        if have:
            pl.concat(have, how="diagonal_relaxed").write_parquet(path)
            have = [pl.read_parquet(path)]
        done_p.write_text(json.dumps(sorted(done)))
        print(f"  {season}: {len(done)}/{len(ids)} games, {have[0].height if have else 0} rows", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seasons", nargs="+", type=int)
    for s in ap.parse_args().seasons:
        pull(s)
