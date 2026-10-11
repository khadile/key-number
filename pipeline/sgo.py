"""SportsGameOdds client and archive puller (Pro trial, October 2026).

What the archive holds for NFL (checked 2026-10-09):

- Nothing before Super Bowl LVIII (Feb 2024), whatever the FAQ says about 2020+.
- 2024 season: props, but mostly at minor books (Fliff, "unknown"); each
  book's quote is its LAST one, which for books that trade in-game is an
  in-game price. No open or close fields at all.
- 2025 season: every prop at ~20 books, plus the consensus close (`closeBookOverUnder`,
  `closeBookOdds`, `closeFairOverUnder`, `closeFairOdds`) and the settled stat
  (`score`). Per-book quotes are still last quotes: in-game at DraftKings and
  FanDuel, pregame (their close) at Pinnacle, which pulls props at kickoff.
- 2026 season: per-book `openOdds`/`closeOdds`/`openOverUnder`/`closeOverUnder`
  at every book, plus the consensus open and close.

Billing is per event returned (one object per game, whatever the markets), so a
whole season is a few hundred objects. The trial allows 300 requests a minute
and 3M objects a day; this client stays far under: one call every 2 s, at most
MAX_CALLS_PER_DAY. Every call is logged (never the key) to
data/sgo/request_log.jsonl. Raw events are kept gzipped under data/sgo/raw/
(gitignored); the tidy tables built from them (pipeline/sgo_tidy.py) are kept in git.

    python -m pipeline.sgo --check            # one call: confirms the key, prints the plan and limits
    python -m pipeline.sgo --pull 2024 2025 2026
    python -m pipeline.sgo --pull 2025 2026 --alt   # the same games with every line each book hung

Alt lines (`--alt`, saved to data/sgo/raw_alt/): besides real alternate lines,
SGO lists every line value a book ever hung for a market, each with its LAST
price and time. For books that trade in-game the closing value kept trading, so
closes cannot be rebuilt from them (checked on 2026: 0 of 61 DK/FD closes), but
the pregame entries are a coarse line history: where a book's line sat earlier
in the week, which is what a lock-time test of past seasons needs.
"""
import argparse
import gzip
import json
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import certifi

from config import DATA, secret

BASE = "https://api.sportsgameodds.com/v2"
ROOT = DATA / "sgo"
RAW_DIR = ROOT / "raw"
ALT_DIR = ROOT / "raw_alt"
LOG = ROOT / "request_log.jsonl"
MIN_GAP = 2.0              # seconds between calls (their cap is 300 a minute)
MAX_CALLS_PER_DAY = 20000  # our own guard (their cap is 500,000); a live Sunday is ~3,000 calls
PAGE = 10                  # events per page; a 2026 game with open/close is ~3 MB
# regular season through the Super Bowl
SEASONS = {2024: ("2024-09-01", "2025-02-15"), 2025: ("2025-09-01", "2026-02-15"), 2026: ("2026-09-01", "2027-02-15")}

ROOT.mkdir(parents=True, exist_ok=True)
_TLS = ssl.create_default_context(cafile=certifi.where())
_last = 0.0


def _key() -> str:
    k = secret("SPORTSGAMEODDS_KEY")
    if not k:
        raise SystemExit("SPORTSGAMEODDS_KEY is empty in .env")
    return k


def used_today() -> int:
    day, n = datetime.now(timezone.utc).strftime("%Y-%m-%d"), 0
    if LOG.exists():
        for line in LOG.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            n += r["at"].startswith(day) and r["status"] != 0
    return n


def get(path: str, params: dict, tries: int = 4) -> dict:
    """GET /v2/{path}. Returns the decoded JSON, or {"error": status, "detail": ...}."""
    global _last
    if used_today() >= MAX_CALLS_PER_DAY:
        raise SystemExit(f"SportsGameOdds: our daily guard of {MAX_CALLS_PER_DAY} calls reached")
    url = f"{BASE}/{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"x-api-key": _key(), "User-Agent": "nfl_model",
                                               "Accept-Encoding": "gzip"})
    body = {"error": 0, "detail": "not tried"}
    for attempt in range(tries):
        pause = _last + MIN_GAP - time.monotonic()
        if pause > 0:
            time.sleep(pause)
        _last = time.monotonic()
        t0, nbytes = time.monotonic(), 0
        try:
            with urllib.request.urlopen(req, timeout=180, context=_TLS) as r:
                raw, status = r.read(), r.status
                nbytes = len(raw)
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                body = json.loads(raw)
        except urllib.error.HTTPError as e:
            status = e.code
            body = {"error": e.code, "detail": e.read().decode(errors="replace")[:500]}
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            status = 0
            body = {"error": 0, "detail": str(e)[:300]}
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "path": path, "params": params,
                                "status": status, "ms": round((time.monotonic() - t0) * 1000), "bytes": nbytes,
                                "events": len(body.get("data", [])) if status == 200 and isinstance(body.get("data"), list) else None})
                    + "\n")
        if status == 200:
            return body
        if status in (0, 429, 500, 502, 503, 504) and attempt + 1 < tries:
            time.sleep(5 * (attempt + 1) ** 2)
            continue
        return body
    return body


# ---------------------------------------------------------------- live odds for the engine (trial week, 2026-10)
# One shared call returns every game being watched: each GameRunner asks for its event and gets
# the cached answer if it is younger than LIVE_TTL, so 13 games cost one call every ~12 seconds.
LIVE_BOOKS = "draftkings,fanduel,pinnacle"
LIVE_TTL = 12.0
_live_lock = threading.Lock()
_live = {"at": -1e9, "data": {}}
_watching: set[str] = set()


def watch(event_id: str, on: bool = True) -> None:
    (_watching.add if on else _watching.discard)(event_id)


def live_event(event_id: str) -> dict | None:
    """The latest copy of one event (DraftKings, FanDuel, Pinnacle main lines), shared across games."""
    with _live_lock:
        if time.monotonic() - _live["at"] > LIVE_TTL or event_id not in _live["data"]:
            ids = sorted(_watching | {event_id})
            body = get("events", {"eventIDs": ",".join(ids), "bookmakerID": LIVE_BOOKS, "limit": len(ids)}, tries=2)
            if "error" in body:
                return None
            _live.update(at=time.monotonic(), data={e["eventID"]: e for e in body.get("data", [])})
        return _live["data"].get(event_id)


def nfl_events(lo: str, hi: str) -> list[dict]:
    """NFL events starting in [lo, hi] (ISO), with teams and players but just one market, to map ids."""
    body = get("events", {"leagueID": "NFL", "startsAfter": lo, "startsBefore": hi,
                          "oddID": "points-home-game-ml-home", "limit": 50})
    return [] if "error" in body else body.get("data", [])


def pull(season: int, refresh: bool = False, alt: bool = False) -> int:
    """Every NFL event of a season, saved one file per event. Returns how many were written.
    Finished games never change, so existing files are kept unless `refresh`."""
    lo, hi = SEASONS[season]
    hi = min(hi, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    out = (ALT_DIR if alt else RAW_DIR) / str(season)
    out.mkdir(parents=True, exist_ok=True)
    params = {"leagueID": "NFL", "startsAfter": lo, "startsBefore": hi, "includeOpenCloseOdds": "true",
              "limit": PAGE // 2 if alt else PAGE}
    if alt:
        params["includeAltLines"] = "true"
    cursor, wrote, seen = None, 0, 0
    while True:
        q = dict(params, **({"cursor": cursor} if cursor else {}))
        body = get("events", q)
        if "error" in body:
            raise SystemExit(f"SportsGameOdds pull {season} failed: {body}")
        data = body.get("data", [])
        for e in data:
            seen += 1
            st = e.get("status", {})
            f = out / f"{e['eventID']}.json.gz"
            # a game still to be played is re-pulled next time; a finished one is final
            if f.exists() and not refresh:
                continue
            if not st.get("ended") and not st.get("cancelled"):
                continue
            f.write_bytes(gzip.compress(json.dumps(e).encode("utf-8"), 6))
            wrote += 1
        cursor = body.get("nextCursor")
        print(f"  {season}: {seen} events seen, {wrote} written", flush=True)
        if not cursor or not data:
            return wrote


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--pull", nargs="*", type=int)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--alt", action="store_true", help="include every line each book hung (raw_alt/)")
    args = ap.parse_args()
    if args.check:
        u = get("account/usage", {})
        if "error" in u:
            print("check failed:", u)
            return
        d = u["data"]
        print(f"key works: tier {d.get('tier')}, active {d.get('isActive')}")
        for per, lim in d.get("rateLimits", {}).items():
            print(f"  {per}: {lim}")
    for s in args.pull or []:
        print(f"pulling {s} ...")
        print(f"{s}: {pull(s, args.refresh, args.alt)} new events")
    print(f"calls today (our log): {used_today()} of our {MAX_CALLS_PER_DAY} guard")


if __name__ == "__main__":
    main()
