"""OddsPapi v4 client with a hard request budget.

The free plan allows a small number of billable requests per month, so every
billable call goes through `_get(..., billable=True)`, which:

- refuses to run if it would cross the monthly budget we set (BUDGET), and
- writes every response to data/oddspapi/ so the same data is never bought twice.

/v4/historical-odds and /v4/account do not count toward the quota.

    python -m pipeline.oddspapi --account     # free: plan, books, usage
"""
import argparse
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from config import DATA, secret

BASE = "https://api.oddspapi.io/v4"
CACHE = DATA / "oddspapi"
LOG = CACHE / "request_log.jsonl"
BUDGET = 240  # leave headroom under the 250/month plan
# Minimum seconds between calls, per endpoint, counted from when the last response ended.
# /historical-odds allows one call per 5 seconds (measured 2026-10-08: every call 4.8 s
# after the last was rejected with 429, every call 4.9 s or more after was served).
PACE = {"historical-odds": 5.5}
DEFAULT_PACE = 1.0

CACHE.mkdir(parents=True, exist_ok=True)
_LOG_LOCK = threading.Lock()
_PACE_LOCK = threading.Lock()
_last_start: dict[str, float] = {}


def _key() -> str:
    k = secret("ODDSPAPI_KEY")
    if not k:
        raise SystemExit("ODDSPAPI_KEY is empty in .env")
    return k


def billable_used_this_month() -> int:
    if not LOG.exists():
        return 0
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    # A 429 (rate limited) or 0 (never reached the server) is not a served request.
    return sum(1 for line in LOG.read_text().splitlines()
               if (r := json.loads(line))["billable"] and r["at"].startswith(month)
               and r.get("status") not in (0, 429))


def _wait_turn(path: str) -> None:
    """Hold this call until the endpoint's minimum gap since the last call has passed."""
    gap = PACE.get(path, DEFAULT_PACE)
    with _PACE_LOCK:
        pause = _last_start.get(path, 0.0) + gap - time.monotonic()
        if pause > 0:
            time.sleep(pause)
        _last_start[path] = time.monotonic()


def _retry_ms(detail: str) -> float | None:
    """The wait a 429 asks for, e.g. {"error": {"retryMs": 2871, ...}}."""
    try:
        return float(json.loads(detail)["error"]["retryMs"])
    except (ValueError, KeyError, TypeError):
        return None


def _get(path: str, params: dict, billable: bool, cache_name: str | None = None, tries: int = 4):
    if billable and billable_used_this_month() >= BUDGET:
        raise SystemExit(f"OddsPapi monthly budget of {BUDGET} reached; refusing billable call to {path}")
    if cache_name and (CACHE / cache_name).exists():
        return json.loads((CACHE / cache_name).read_text(encoding="utf-8"))
    query = urllib.parse.urlencode({**params, "apiKey": _key()})
    req = urllib.request.Request(f"{BASE}/{path}?{query}", headers={"User-Agent": "nfl_model"})
    for attempt in range(tries):
        _wait_turn(path)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = json.loads(r.read())
                status = r.status
        except urllib.error.HTTPError as e:
            body, status = {"error": e.code, "detail": e.read().decode(errors="replace")[:500]}, e.code
        except (urllib.error.URLError, TimeoutError) as e:
            body, status = {"error": 0, "detail": str(e)[:300]}, 0
        with _PACE_LOCK:  # the server's window runs from when a (100 MB) response ends, not starts
            _last_start[path] = time.monotonic()
        with _LOG_LOCK, LOG.open("a") as f:  # params logged without the key
            f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "path": path,
                                "params": params, "status": status, "billable": billable}) + "\n")
        if status == 429 and attempt + 1 < tries:
            # Rate limited: wait exactly as long as the server asks (plus a margin), then retry.
            time.sleep((_retry_ms(body["detail"]) or 5000) / 1000 + 0.5)
            continue
        break
    if status == 200 and cache_name:
        (CACHE / cache_name).write_text(json.dumps(body), encoding="utf-8")
    return body


def account() -> dict:
    return _get("account", {}, billable=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", action="store_true")
    args = ap.parse_args()
    if args.account:
        a = account()
        if "error" in a:
            print(a)
            return
        for s in a.get("subscriptions", []):
            if s.get("is_active"):
                print("books:", ", ".join(s.get("bookmakers", {})) or "none listed")
                print("sports:", s.get("sport_ids"))
                print(f"requests: {s.get('request_count')} used of {s.get('request_limit')}  "
                      f"(valid {s.get('valid_from')} -> {s.get('valid_until')})")
        print(f"our log: {billable_used_this_month()} billable this month, budget {BUDGET}")


if __name__ == "__main__":
    main()
