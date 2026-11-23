"""Why are upcoming games coming back empty from /historical-odds? (2026-11-23)

From 2026-11-20 every upcoming NBA game and from 2026-11-21 every upcoming NFL game came back "unavailable"
from the free history endpoint, while finished games still saved. This asks once, for one upcoming and one
finished game in each sport, and prints only what the endpoint said: the HTTP status, its error text, how many
books and price points came back and how recent the newest one is. No prices (research.yml logs are public).
It also tallies the request log by day and status, to date when the failures began, and asks the free
/account endpoint for the plan's state.

    python -m pipeline.oddspapi_diag      # 5 free calls
"""
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone

from pipeline.oddspapi import CACHE, LOG, _get, account, billable_used_this_month

NFL_BOOKS = "pinnacle,draftkings,fanduel"


def _scrub(s: str) -> str:
    return re.sub(r"apiKey=[^&\s\"']+", "apiKey=***", s)


def _pick(fixtures: list[dict], now: datetime) -> tuple[dict | None, dict | None]:
    """The next upcoming fixture with odds, and the most recent one that has ended."""
    def t(f):
        return datetime.fromisoformat(f["startTime"].replace("Z", "+00:00"))
    up = sorted((f for f in fixtures if now < t(f) < now + timedelta(days=3)), key=t)
    done = sorted((f for f in fixtures if now - timedelta(days=4) < t(f) < now - timedelta(hours=8)), key=t)
    return (up[0] if up else None), (done[-1] if done else None)


def probe(label: str, f: dict | None, books: str, now: datetime) -> None:
    if f is None:
        print(f"{label}: no fixture in range")
        return
    h = _get("historical-odds", {"fixtureId": f["fixtureId"], "bookmakers": books}, billable=False, tries=2)
    start = f["startTime"][:16]
    if "error" in h:
        print(f"{label} (starts {start}): status {h['error']}  {_scrub(str(h.get('detail')))[:300]}")
        return
    pts, newest = 0, ""
    per_book = Counter()
    for book, b in (h.get("bookmakers") or {}).items():
        for m in (b.get("markets") or {}).values():
            for o in (m.get("outcomes") or {}).values():
                for series in (o.get("players") or {}).values():
                    for p in series:
                        pts += 1
                        per_book[book] += 1
                        newest = max(newest, p.get("createdAt") or "")
    age = ""
    if newest:
        age = f", newest {(now - datetime.fromisoformat(newest.replace('Z', '+00:00'))).total_seconds() / 3600:.1f} h old"
    print(f"{label} (starts {start}): status 200, keys {sorted(h)[:8]}, books {dict(per_book)}, {pts} price points{age}")


def log_tally(days: int = 8) -> None:
    if not LOG.exists():
        print("no request log in the cache")
        return
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    c = Counter()
    for line in LOG.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["at"][:10] >= since and r["path"] == "historical-odds":
            c[(r["at"][:10], r["status"])] += 1
    print("historical-odds calls in the log by day and status:")
    for (day, status), n in sorted(c.items()):
        print(f"  {day}  {status}  {n}")


def main() -> None:
    now = datetime.now(timezone.utc)
    a = account()
    if "error" in a:
        print("account:", a["error"], _scrub(str(a.get("detail")))[:300])
    else:
        for s in a.get("subscriptions", []):
            print(f"plan: active {s.get('is_active')}, requests {s.get('request_count')} of {s.get('request_limit')}, "
                  f"valid {s.get('valid_from')} -> {s.get('valid_until')}, sports {s.get('sport_ids')}")
    print(f"our log: {billable_used_this_month()} billable this month")
    nfl = json.loads((CACHE / "fixtures_t31.json").read_text(encoding="utf-8"))
    nba = json.loads((CACHE / "fixtures_t132.json").read_text(encoding="utf-8"))
    up, done = _pick(nfl, now)
    probe("NFL upcoming", up, NFL_BOOKS, now)
    probe("NFL finished", done, NFL_BOOKS, now)
    up, done = _pick(nba, now)
    probe("NBA upcoming", up, NFL_BOOKS, now)
    probe("NBA finished", done, NFL_BOOKS, now)
    log_tally()


if __name__ == "__main__":
    main()
