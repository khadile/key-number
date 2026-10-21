"""Start a board build GitHub's scheduler skipped (user, 2026-10-21). Runs on the laptop every 15 minutes
(Windows Task Scheduler, task "Key Number catch-up"); does nothing while the laptop is off.

A slot counts as built when the repo's slot file says so: board/nba_last.json (NBA morning 10:30am ET,
evening 5:30pm ET) and board/last_build.json (NFL Tuesday, Thursday, Saturday 10am ET). Thirty minutes past a
slot with no build, it dispatches that workflow WITHOUT force, so the workflow's own slot check still decides
and never builds a slot twice. Nothing is dispatched while a run of that workflow is queued or running, and at
most twice a slot (40+ minutes apart), so a build that keeps failing is left for a person to look at.

    python -m board.catchup            # one check; the log is ~/key-number-catchup/catchup.log

The log lives in the home folder, not AppData: Store Python redirects AppData writes into its own sandbox.
"""
import json
import os
import subprocess
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
REPO = "khadile/key-number"
GH = r"C:\Program Files\GitHub CLI\gh.exe"
HOME = Path.home() / "key-number-catchup"
GRACE = timedelta(minutes=30)
RETRY = timedelta(minutes=40)
NBA_SLOTS = (("evening", time(17, 30)), ("morning", time(10, 30)))   # latest first, as nba/slot.py
NFL_DAYS, NFL_AT = (1, 3, 5), time(10, 0)                            # Tue, Thu, Sat (board/slot.py)


def gh(*args: str) -> str:
    return subprocess.run([GH, *args], capture_output=True, text=True, check=True,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout


def slot_file(path: str) -> dict:
    try:
        return json.loads(gh("api", f"repos/{REPO}/contents/{path}", "-H", "Accept: application/vnd.github.raw"))
    except subprocess.CalledProcessError:
        return {}


def busy(workflow: str) -> bool:
    runs = json.loads(gh("run", "list", "-R", REPO, "--workflow", workflow, "-L", "5", "--json", "status"))
    return any(r["status"] in ("queued", "in_progress", "waiting", "pending", "requested") for r in runs)


def missed(now: datetime) -> list[tuple[str, str]]:
    """(workflow, slot key) for each slot past its grace period without a build today."""
    et, today = now.astimezone(ET), now.astimezone(ET).date().isoformat()
    out = []
    name, start = next(((n, s) for n, s in NBA_SLOTS if et.time() >= s), (None, None))
    if name and et - GRACE >= et.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0):
        if slot_file("board/nba_last.json").get(name) != today:
            out.append(("nba.yml", f"nba:{today}:{name}"))
    nfl_at = et.replace(hour=NFL_AT.hour, minute=NFL_AT.minute, second=0, microsecond=0)
    if et.weekday() in NFL_DAYS and et >= nfl_at + GRACE and slot_file("board/last_build.json").get("date") != today:
        out.append(("board.yml", f"nfl:{today}"))
    return out


def main() -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    state_p, log_p = HOME / "state.json", HOME / "catchup.log"
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    now = datetime.now(timezone.utc)

    def log(msg: str) -> None:
        with log_p.open("a", encoding="utf-8") as f:
            f.write(f"{now.astimezone(ET):%Y-%m-%d %H:%M} ET  {msg}\n")

    try:
        for workflow, key in missed(now):
            tries = state.get(key, [])
            if busy(workflow):
                continue
            if len(tries) >= 2 or (tries and now - datetime.fromisoformat(tries[-1]) < RETRY):
                if len(tries) >= 2 and not state.get(key + ":gave_up"):
                    state[key + ":gave_up"] = True
                    log(f"{key}: still not built after 2 dispatches; leaving it for a person")
                continue
            gh("workflow", "run", workflow, "-R", REPO)
            state[key] = tries + [now.isoformat()]
            log(f"{key}: no build {GRACE.seconds // 60} min after its slot; dispatched {workflow}")
    except Exception as e:   # noqa: BLE001  (no network, gh logged out): try again next time
        log(f"check failed: {type(e).__name__}: {e}")
    state_p.write_text(json.dumps({k: v for k, v in state.items() if k.split(":")[1] >= (now - timedelta(days=7)).date().isoformat()}))


if __name__ == "__main__":
    main()
