"""Today's NBA rosters and injury report from ESPN's public site API (free; the same athlete ids as hoopR)."""
import json
import re
import time
import urllib.request

import polars as pl

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
OUT_STATUSES = {"Out", "Doubtful"}            # treated as not playing
UNSURE = {"Questionable", "Day-To-Day", "Game Time Decision", "GTD"}


def _get(path: str, tries: int = 4) -> dict:
    for i in range(tries):
        try:
            req = urllib.request.Request(f"{BASE}/{path}", headers={"User-Agent": "nfl_model"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except OSError:
            time.sleep(2 ** i)
    raise RuntimeError(f"ESPN request failed: {path}")


def teams() -> pl.DataFrame:
    d = _get("teams")
    rows = [t["team"] for s in d["sports"] for lg in s["leagues"] for t in lg["teams"]]
    return pl.DataFrame([{"team_id": int(t["id"]), "team": t["abbreviation"], "name": t["displayName"]} for t in rows])


def rosters() -> pl.DataFrame:
    """Everyone on a roster now: athlete_id, name, team, position."""
    out = []
    for t in teams().iter_rows(named=True):
        d = _get(f"teams/{t['team_id']}/roster")
        for a in d.get("athletes", []):
            items = a.get("items", [a]) if isinstance(a, dict) and "items" in a else [a]
            for p in items:
                out.append({"athlete_id": int(p["id"]), "name": p.get("fullName") or p.get("displayName"),
                            "team": t["team"], "position": (p.get("position") or {}).get("abbreviation")})
        time.sleep(0.2)
    return pl.DataFrame(out).unique("athlete_id")


POS_ORDER = ("pg", "sg", "sf", "pf", "c")


def depth_starters() -> dict[str, list[int]]:
    """ESPN's depth chart: each team's first name at each position (pg, sg, sf, pf, c). Measured 2026-10-23 against
    each team's latest starting five: 93% right (who started the team's last game: 96%)."""
    out = {}
    for t in teams().iter_rows(named=True):
        try:
            d = _get(f"teams/{t['team_id']}/depthcharts")
        except Exception:   # noqa: BLE001  (one team's chart missing never stops the build)
            continue
        pos = ((d.get("depthchart") or [{}])[0]).get("positions", {})
        ids = [int(pos[p]["athletes"][0]["id"]) for p in POS_ORDER if (pos.get(p) or {}).get("athletes")]
        out[t["team"]] = list(dict.fromkeys(ids))
        time.sleep(0.2)
    return out


def scoreboard(day: str) -> list[dict]:
    """A day's games (day: YYYYMMDD, Eastern): game_id, tip (UTC ISO), state (pre / in / post), home, away, scores."""
    out = []
    for e in _get(f"scoreboard?dates={day}").get("events", []):
        c = e["competitions"][0]
        side = {x["homeAway"]: x for x in c["competitors"]}
        out.append({"game_id": int(e["id"]), "tip": e["date"], "state": e["status"]["type"]["state"],
                    "final": bool(e["status"]["type"]["completed"]), "detail": e["status"]["type"].get("shortDetail"),
                    "home": side["home"]["team"]["abbreviation"], "away": side["away"]["team"]["abbreviation"],
                    "home_score": int(side["home"].get("score") or 0), "away_score": int(side["away"].get("score") or 0)})
    return out


BOX_STATS = {"MIN": "min", "PTS": "pts", "REB": "reb", "AST": "ast", "3PT": "fg3"}


def box(game_id: int) -> dict | None:
    """A finished game's final score and player lines from ESPN (None until it is final). Each player: athlete_id,
    name, team, starter, played, min, pts, reb, ast, fg3 (made threes)."""
    s = _get(f"summary?event={game_id}")
    c = s["header"]["competitions"][0]
    if not c["status"]["type"].get("completed"):
        return None
    side = {x["homeAway"]: x for x in c["competitors"]}
    players = []
    for t in s.get("boxscore", {}).get("players", []):
        for st in t.get("statistics", []):
            col = {lab: i for i, lab in enumerate(st.get("labels", []))}
            for a in st.get("athletes", []):
                v = a.get("stats") or []
                row = {"athlete_id": int(a["athlete"]["id"]), "name": a["athlete"].get("displayName"),
                       "team": t["team"]["abbreviation"], "starter": bool(a.get("starter")),
                       "played": bool(v) and not a.get("didNotPlay")}
                for lab, k in BOX_STATS.items():
                    x = v[col[lab]] if row["played"] and lab in col and col[lab] < len(v) else None
                    x = x.split("-")[0] if isinstance(x, str) else x   # "3-7": made threes
                    try:
                        row[k] = float(x) if x not in (None, "", "--") else (0.0 if row["played"] else None)
                    except ValueError:
                        row[k] = 0.0 if row["played"] else None
                players.append(row)
    return {"game_id": int(game_id), "home": side["home"]["team"]["abbreviation"], "away": side["away"]["team"]["abbreviation"],
            "home_score": int(side["home"]["score"]), "away_score": int(side["away"]["score"]), "players": players}


def injuries() -> pl.DataFrame:
    """The league injury report: athlete_id, team, status (Out, Doubtful, Questionable, Day-To-Day, ...), detail."""
    d = _get("injuries")
    abbr = dict(zip(*[teams()[c].cast(pl.String) for c in ("team_id", "team")]))
    out = []
    for t in d.get("injuries", []):
        for i in t.get("injuries", []):
            a = i.get("athlete", {})
            m = next((re.search(r"/id/(\d+)", l.get("href", "")) for l in a.get("links", []) if "/id/" in l.get("href", "")), None)
            if not m:
                continue
            out.append({"athlete_id": int(m.group(1)), "name": a.get("displayName"), "team": abbr.get(str(t.get("id"))),
                        "status": i.get("status"), "detail": i.get("shortComment") or i.get("longComment") or ""})
    return pl.DataFrame(out, schema={"athlete_id": pl.Int64, "name": pl.String, "team": pl.String, "status": pl.String, "detail": pl.String})
