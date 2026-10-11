"""Is a scheduled board build due? Used by the GitHub workflow (.github/workflows/board.yml).

GitHub's scheduler is best-effort and drops many firings, so the workflow fires every half
hour from 10am to about 1:30pm ET on Tuesday, Thursday and Saturday, and the first firing
that lands builds. board/last_build.json records the Eastern date of the last scheduled
build; later firings that day skip.

    python -m board.slot --check [--force]   # writes due=true|false to $GITHUB_OUTPUT
    python -m board.slot --mark              # after a successful build
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
STATE = Path(__file__).with_name("last_build.json")
BUILD_DAYS = (1, 3, 5)   # Tuesday, Thursday, Saturday (Monday = 0)


def due(now: datetime) -> bool:
    et = now.astimezone(ET)
    last = json.loads(STATE.read_text()).get("date") if STATE.exists() else None
    return et.weekday() in BUILD_DAYS and et.hour >= 10 and last != et.date().isoformat()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--mark", action="store_true")
    a = ap.parse_args()
    now = datetime.now(timezone.utc)
    if a.check:
        d = a.force or due(now)
        print(f"build due: {d} ({now.astimezone(ET):%a %Y-%m-%d %H:%M} ET)")
        out = os.environ.get("GITHUB_OUTPUT")
        if out:
            with open(out, "a") as f:
                f.write(f"due={'true' if d else 'false'}\n")
    if a.mark:
        STATE.write_text(json.dumps({"date": now.astimezone(ET).date().isoformat(), "at": now.isoformat()}) + "\n")


if __name__ == "__main__":
    main()
