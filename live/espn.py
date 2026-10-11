"""ESPN's public game feed: the live game state and box score.

Free, no key, unofficial (it can change without notice). One request returns
the score, clock, every play so far (each stamped with its real time,
`wallclock`, which is how feed delay is measured) and the box score with each
player's targets, carries and yards keyed by ESPN athlete id.
"""
import json
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import certifi

SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={eid}"
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={day}"
TEAM_FIX = {"WSH": "WAS", "LAR": "LA"}  # ESPN abbreviation -> nflverse
_TLS = ssl.create_default_context(cafile=certifi.where())


def _get(url: str, tries: int = 3) -> dict | None:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (nfl_model)"})
            with urllib.request.urlopen(req, timeout=12, context=_TLS) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(1.5 * (i + 1))
    return None


def summary(event_id: str) -> dict | None:
    return _get(SUMMARY.format(eid=event_id))


def scoreboard(day: str) -> dict | None:
    """day: YYYYMMDD."""
    return _get(SCOREBOARD.format(day=day))


def _secs(display: str | None) -> int:
    if not display or ":" not in display:
        return 0
    m, s = display.split(":")
    return int(m) * 60 + int(float(s))


def game_seconds_left(period: int, clock: int) -> int:
    """Regulation seconds remaining (overtime counts its own clock)."""
    return (4 - period) * 900 + clock if period <= 4 else clock


def _num(v) -> float:
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return 0.0


def parse(d: dict) -> dict:
    """The parts the engine uses, in nflverse team codes."""
    comp = d["header"]["competitions"][0]
    st = comp["status"]
    teams, score, poss = {}, {}, None
    for c in comp["competitors"]:
        abbr = TEAM_FIX.get(c["team"]["abbreviation"], c["team"]["abbreviation"])
        teams[c["homeAway"]] = abbr
        score[abbr] = int(_num(c.get("score", 0)))
        if c.get("possession"):
            poss = abbr
    period = int(st.get("period") or 0)
    clock = _secs(st.get("displayClock"))
    plays = []
    ids: set = set()
    drives = d.get("drives", {})
    for dr in drives.get("previous", []) + ([drives["current"]] if drives.get("current") else []):
        team = TEAM_FIX.get(dr.get("team", {}).get("abbreviation", ""), dr.get("team", {}).get("abbreviation"))
        for p in dr.get("plays", []):
            if p["id"] in ids:   # the drive in progress is listed under both "previous" and "current"
                continue
            ids.add(p["id"])
            s = p.get("start", {})
            plays.append({"id": p["id"], "seq": int(_num(p.get("sequenceNumber", 0))), "type": p.get("type", {}).get("text"),
                          "text": p.get("text", ""), "period": p.get("period", {}).get("number"),
                          "clock": _secs(p.get("clock", {}).get("displayValue")), "home": p.get("homeScore"),
                          "away": p.get("awayScore"), "scoring": bool(p.get("scoringPlay")), "wallclock": p.get("wallclock"),
                          "drive_team": team, "down": s.get("down"), "distance": s.get("distance"),
                          "yards_to_endzone": s.get("yardsToEndzone"), "yards": p.get("statYardage")})
    plays.sort(key=lambda p: p["seq"])
    if poss is None and plays:
        poss = plays[-1]["drive_team"]
    box: dict[str, dict] = {}
    team_box: dict[str, dict] = {}
    for tp in d.get("boxscore", {}).get("players", []):
        abbr = TEAM_FIX.get(tp["team"]["abbreviation"], tp["team"]["abbreviation"])
        tb = team_box.setdefault(abbr, {"att": 0, "car": 0, "tgt": 0})
        for cat in tp.get("statistics", []):
            keys = cat.get("keys", [])
            for a in cat.get("athletes", []):
                aid = a["athlete"]["id"]
                vals = dict(zip(keys, a.get("stats", [])))
                b = box.setdefault(aid, {"name": a["athlete"].get("displayName"), "team": abbr})
                if cat["name"] == "passing":
                    c_a = str(vals.get("completions/passingAttempts", "0/0")).split("/")
                    b.update(cmp=_num(c_a[0]), att=_num(c_a[1]) if len(c_a) > 1 else 0.0,
                             pass_yds=_num(vals.get("passingYards", 0)))
                    tb["att"] += b["att"]
                elif cat["name"] == "rushing":
                    b.update(car=_num(vals.get("rushingAttempts", 0)), rush_yds=_num(vals.get("rushingYards", 0)))
                    tb["car"] += b["car"]
                elif cat["name"] == "receiving":
                    b.update(rec=_num(vals.get("receptions", 0)), rec_yds=_num(vals.get("receivingYards", 0)),
                             tgt=_num(vals.get("receivingTargets", 0)))
                    tb["tgt"] += b["tgt"]
    return {"state": st["type"]["state"], "final": bool(st["type"].get("completed")), "detail": st["type"].get("shortDetail"),
            "period": period, "clock": clock, "seconds_left": game_seconds_left(period, clock) if period else 3600,
            "teams": teams, "score": score, "possession": poss, "plays": plays, "box": box, "team_box": team_box,
            "received": datetime.now(timezone.utc).isoformat()}
