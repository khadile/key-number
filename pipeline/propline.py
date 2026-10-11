"""PropLine client with a hard daily request budget.

The free plan allows 1,000 requests a day (reset 00:00 UTC), 20 in flight per
key and about 5 a second sustained. Every call goes through `get()`, which:

- refuses to run once today's usage reaches DAILY_BUDGET (headroom under 1,000),
  counting both our own log and the usage the server reports back,
- keeps at most MAX_IN_FLIGHT calls open and spaces calls MIN_GAP apart,
- honours Retry-After on 429/503 instead of retrying straight away,
- caches slow-changing answers (event lists, id maps) on disk with a TTL,
- logs every call (never the key) to data/propline/request_log.jsonl.

One per-event /odds call carries every market and every book asked for, so a
game costs one request per poll however many players it covers. Always pass
`bookmakers=` (BOOKS) to keep payloads small.

    python -m pipeline.propline --usage   # today's usage from our log (no call)
    python -m pipeline.propline --check   # one call: confirms the key, prints the server's count
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

BASE = "https://api.prop-line.com/v1"
CACHE = DATA / "propline"
LOG = CACHE / "request_log.jsonl"
DAILY_LIMIT = 1000   # free plan
DAILY_BUDGET = 950   # leave headroom for a manual check or a retry storm
MAX_IN_FLIGHT = 8    # their cap is 20
MIN_GAP = 0.25       # seconds between call starts, about 4 a second
SPORT = "football_nfl"
BOOKS = "draftkings,fanduel,pinnacle"  # our two books plus the sharp reference

CACHE.mkdir(parents=True, exist_ok=True)
_LOG_LOCK = threading.Lock()
_GAP_LOCK = threading.Lock()
_SLOTS = threading.BoundedSemaphore(MAX_IN_FLIGHT)
_last_start = 0.0
_server = {"used": None, "remaining": None, "day": None}  # last usage the server reported
# Python's default store on this machine rejects PropLine's chain as expired;
# certifi's bundle verifies it. Verification stays on either way.
_TLS = ssl.create_default_context(cafile=certifi.where())


def _key() -> str:
    k = secret("PROPLINE_KEY")
    if not k:
        raise SystemExit("PROPLINE_KEY is empty in .env")
    return k


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def used_today() -> int:
    """Requests spent today: our log's count, or the server's if it reported more."""
    day, ours, theirs = _today(), 0, 0
    if LOG.exists():
        for line in LOG.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["at"].startswith(day) and r["status"] != 0:  # 0 = never reached the server
                ours += 1
                theirs = max(theirs, r.get("daily_used") or 0)
    return max(ours, theirs)


def remaining_today() -> int:
    return max(0, DAILY_BUDGET - used_today())


def _wait_turn() -> None:
    global _last_start
    with _GAP_LOCK:
        pause = _last_start + MIN_GAP - time.monotonic()
        if pause > 0:
            time.sleep(pause)
        _last_start = time.monotonic()


def _log(entry: dict) -> None:
    with _LOG_LOCK, LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def get(path: str, params: dict | None = None, cache_name: str | None = None,
        ttl_s: float = 0, tries: int = 3):
    """GET /v1/{path}. Returns the decoded JSON, or {"error": status, ...}."""
    params = params or {}
    if cache_name and ttl_s > 0:
        f = CACHE / cache_name
        if f.exists() and time.time() - f.stat().st_mtime < ttl_s:
            return json.loads(f.read_text(encoding="utf-8"))
    if used_today() >= DAILY_BUDGET:
        raise SystemExit(f"PropLine daily budget of {DAILY_BUDGET} reached; refusing call to {path}")
    url = f"{BASE}/{path}" + (f"?{urllib.parse.urlencode(params)}" if params else "")
    req = urllib.request.Request(url, headers={"X-API-Key": _key(), "User-Agent": "nfl_model",
                                               "Accept-Encoding": "gzip"})
    for attempt in range(tries):
        with _SLOTS:
            _wait_turn()
            t0 = time.monotonic()
            retry_after, headers = None, {}
            try:
                with urllib.request.urlopen(req, timeout=30, context=_TLS) as r:
                    raw, status, headers = r.read(), r.status, r.headers
                    if headers.get("Content-Encoding") == "gzip":
                        raw = gzip.decompress(raw)
                    body = json.loads(raw)
            except urllib.error.HTTPError as e:
                status, headers = e.code, e.headers
                body = {"error": e.code, "detail": e.read().decode(errors="replace")[:500]}
                retry_after = e.headers.get("Retry-After")
            except (urllib.error.URLError, TimeoutError) as e:
                status, raw = 0, b""
                body = {"error": 0, "detail": str(e)[:300]}
        used = headers.get("X-Daily-Used") if headers else None
        left = headers.get("X-Daily-Remaining") if headers else None
        if used is not None:
            _server.update(used=int(used), remaining=int(left) if left is not None else None, day=_today())
        _log({"at": datetime.now(timezone.utc).isoformat(), "path": path, "params": params,
              "status": status, "ms": round((time.monotonic() - t0) * 1000),
              "daily_used": int(used) if used is not None else None,
              "daily_remaining": int(left) if left is not None else None})
        if status == 200:
            if cache_name:
                (CACHE / cache_name).write_text(json.dumps(body), encoding="utf-8")
            return body
        if status in (429, 503) and attempt + 1 < tries:
            wait = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 2.0
            if wait > 120:  # the daily cap: Retry-After counts down to midnight UTC
                return body
            time.sleep(wait)
            continue
        if status == 0 and attempt + 1 < tries:
            time.sleep(2 + 2 * attempt)
            continue
        return body
    return body


def server_usage() -> dict:
    """The usage the server reported on our last call this process (None if no call yet)."""
    return dict(_server)


def events() -> list[dict]:
    """Upcoming and in-progress NFL events (cached 10 minutes)."""
    return get(f"sports/{SPORT}/events", cache_name="events_nfl.json", ttl_s=600)


def ids() -> list[dict]:
    """Event ids across books plus ESPN's event id (cached 6 hours)."""
    return get(f"sports/{SPORT}/ids", cache_name="ids_nfl.json", ttl_s=6 * 3600)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usage", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        ev = get(f"sports/{SPORT}/events")
        if isinstance(ev, dict) and "error" in ev:
            print("PropLine check failed:", ev)
            return
        print(f"key works: {len(ev)} NFL events listed")
        s = server_usage()
        print(f"server says: {s['used']} used today, {s['remaining']} remaining")
    print(f"our log: {used_today()} used today (budget {DAILY_BUDGET} of {DAILY_LIMIT}, resets 00:00 UTC)")


if __name__ == "__main__":
    main()
