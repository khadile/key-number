"""Public betting splits from Action Network (user, 2026-10-26): the share of tickets and of money on each side of
each game's spread, total and moneyline, at their consensus book.

    from pipeline.action_network import splits
    splits("nfl", season=2026, week=8)      # NFL is asked by week (its date parameter is ignored)
    splits("nba", day=date(2026, 10, 27))

Returns {(home full name, away full name): {"start": iso, "spread": {"home": {...}, "away": {...}}, "total": {"over",
"under"}, "ml": {"home", "away"}}} with each side {line, odds, tickets, money} (percent). Only the main line carries
splits; alternates come back with zeros and are dropped. Every pull is saved raw (`snapshot_dir`), so the week's
movement can be studied later; those files stay off the public repo (Actions cache only).

Context, not a pricing input. Fading the public did not hold up 2023-2026 (CLAUDE.md, "Public money"): what
this feeds is the public-tax test (are DraftKings and FanDuel worse than Pinnacle on the popular side?) and a tag.
"""
import json
import time
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

URL = "https://api.actionnetwork.com/web/v2/scoreboard/{league}?bookIds=15&{q}"
MARKETS = {"spread": "spread", "total": "total", "moneyline": "ml"}


def _get(league: str, q: str) -> dict:
    req = urllib.request.Request(URL.format(league=league, q=q), headers={"User-Agent": "Mozilla/5.0"})
    return json.loads(urllib.request.urlopen(req, timeout=20).read())


def parse(raw: dict) -> dict:
    out = {}
    for g in raw.get("games", []):
        teams = {t["id"]: t for t in g.get("teams", [])}
        home, away = teams.get(g["home_team_id"]), teams.get(g["away_team_id"])
        if not home or not away:
            continue
        ev = (g.get("markets") or {}).get("15", {}).get("event", {})
        rec = {"start": g["start_time"], "home_abbr": home.get("abbr"), "away_abbr": away.get("abbr"),
               "home_nick": home.get("display_name"), "away_nick": away.get("display_name")}
        for k, name in MARKETS.items():
            sides = {}
            for o in ev.get(k, []):
                bi = o.get("bet_info") or {}
                t, m = (bi.get("tickets") or {}).get("percent"), (bi.get("money") or {}).get("percent")
                if not t and not m:
                    continue   # an alternate line: no splits
                sides[o["side"]] = {"line": o.get("value"), "odds": o.get("odds"), "tickets": t, "money": m}
            if len(sides) == 2:
                rec[name] = sides
        out[(home["full_name"], away["full_name"])] = rec
    return out


def splits(league: str, season: int | None = None, week: int | None = None, day: date | None = None,
           snapshot_dir: Path | None = None) -> dict:
    q = f"season={season}&week={week}&seasonType=reg" if league == "nfl" else f"date={day:%Y%m%d}"
    raw = _get(league, q)
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / f"{datetime.now(timezone.utc):%Y%m%dT%H%M}Z_{q.replace('&', '_').replace('=', '')}.json").write_text(json.dumps(raw))
    return parse(raw)


def side_split(sp: dict | None, market: str, side: str) -> dict | None:
    """{tickets, money} on one side of one market, for a pick (market spread/total/ml, side home/away/over/under)."""
    s = (sp or {}).get(market, {}).get(side)
    return {"tickets": s["tickets"], "money": s["money"]} if s else None


def by_nick(sp: dict) -> dict:
    """Re-key by (home nickname, away nickname), e.g. ("Chiefs", "Raiders"): full names differ between sources
    (ESPN's "LA Clippers" is Action Network's "Los Angeles Clippers"), nicknames do not."""
    return {(v["home_nick"], v["away_nick"]): v for v in sp.values()}


def safe(fn, *a, **k) -> dict:
    """Splits are decoration: a failure is printed and the board builds without them."""
    for attempt in range(2):
        try:
            return fn(*a, **k)
        except Exception as e:   # noqa: BLE001
            if attempt:
                print(f"action network splits skipped: {type(e).__name__}: {e}")
            time.sleep(2)
    return {}
