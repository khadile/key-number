"""Is a scheduled NBA build due? Used by the GitHub workflow (.github/workflows/nba.yml).

Two builds a day: the morning one from 10:30am ET and the evening one from 5:30pm ET. GitHub's scheduler is
best-effort and drops firings, so the workflow fires every 15 minutes through each window and the first firing
past the slot's start builds; board/nba_last.json records the last date each slot was built, and later firings
skip. A morning build that never ran is made up by the evening one (it locks whatever is due by then).

    python -m nba.slot --check [--force]    # writes due=true|false and slot=morning|evening to $GITHUB_OUTPUT
    python -m nba.slot --mark               # after a successful build
"""
import argparse
import json
import os
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
STATE = Path(__file__).resolve().parent.parent / "board" / "nba_last.json"
SLOTS = (("evening", time(17, 30)), ("morning", time(10, 30)))


def slot_of(now: datetime) -> str | None:
    t = now.astimezone(ET).time()
    return next((name for name, start in SLOTS if t >= start), None)


def due(now: datetime) -> bool:
    s = slot_of(now)
    last = json.loads(STATE.read_text()) if STATE.exists() else {}
    return s is not None and last.get(s) != now.astimezone(ET).date().isoformat()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--mark", action="store_true")
    a = ap.parse_args()
    now = datetime.now(timezone.utc)
    s = slot_of(now) or "morning"
    if a.check:
        d = a.force or due(now)
        print(f"NBA build due: {d} ({s}, {now.astimezone(ET):%a %Y-%m-%d %H:%M} ET)")
        out = os.environ.get("GITHUB_OUTPUT")
        if out:
            with open(out, "a") as f:
                f.write(f"due={'true' if d else 'false'}\nslot={s}\n")
    if a.mark and slot_of(now):   # a forced build before 10:30am is no slot's build: the morning one still runs
        last = json.loads(STATE.read_text()) if STATE.exists() else {}
        last.update({s: now.astimezone(ET).date().isoformat(), "at": now.isoformat()})
        STATE.write_text(json.dumps(last) + "\n")


if __name__ == "__main__":
    main()
