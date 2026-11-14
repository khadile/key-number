"""The college basketball tracker (user, 2026-11-14): this season's early-season leads, flagged daily, graded at the
closing line. Paper only: no stake (the user's choice), a record at 1u so the leads can be judged.

The leads come from ncaab/explore.py (reports/ncaab_explore_2027.json): every simple fact about a game, and every
pair, scored on this season's games and checked against both halves of the season and the same weeks of the last
four seasons. One idea held up:

- fade_cover: a team that covered the spread in its last game by under 10 points is faded: bet its opponent. This
  season (through Nov 11) 74-44 against the close; the same weeks of the last four seasons 399-329 (54.8%). Only
  when exactly one of the two teams did (both, and the game says nothing). Covering by 10 or more was its own
  bucket in the study and did not fade (backing those teams went 41-35, past seasons 255-266).
- big_win (weaker, tracked beside it): back a team that won its last game by 20 or more. This season 94-63, past
  seasons 427-393 (52.1%). Only when exactly one team did.

Each build (ncaab.yml, twice a day on Actions; ESPN is not reachable from a cloud session's container):
- reads ESPN's scoreboard for every day of the season so far and today (scores, ranks, who played whom),
- reads ESPN's odds for today's games (the book of record's current spread and price) and for finished games (the
  close), cached in data/ncaab (Actions cache),
- locks a flag for each of today's games, not yet started, with a line and a lead: once per game and lead,
  appended to ledger/ncaab_flags.jsonl and never changed,
- grades finished flags into ledger/ncaab_results.jsonl: the result at the locked spread, the closing spread, the
  points of line value, and the EV at the close (the chance of covering the locked number if the close is fair),
- replays the leads on the season's games before tracking began (2026-11-03 to 11-13; `replay`): each game locked
  at ESPN's opening line (its close when no open was kept), graded the same way, shown on the page as its own
  labeled record. Never in the ledger. Through Nov 11 these are the games the leads were found on (in-sample);
  Nov 12-13 came after,
- writes board/ncaab_data.json for the page.

    python -m ncaab.build            # on Actions
    python -m ncaab.build --dry-run  # computes everything, writes no ledger lines
    python -m ncaab.build --no-lock  # grades and refreshes the page, locks nothing (a run by hand at night)
"""
import argparse
import json
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import polars as pl

from config import LEDGER, ROOT
from ncaab.top25 import CORE, DIR, HOOPR, UA, _am, _dig, _json, _num

ET = ZoneInfo("America/New_York")
SEASON, SEASON_START = 2027, date(2026, 11, 2)
LABEL = "2026-27"
BOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={d}&groups=50&limit=500"
FLAGS, RESULTS = LEDGER / "ncaab_flags.jsonl", LEDGER / "ncaab_results.jsonl"
OUT = ROOT / "board" / "ncaab_data.json"
SB = DIR / "scoreboard.parquet"
LINES = DIR / "lines_v2.parquet"   # v2: the close falls back to the item's own spread, as the study read it
D1 = DIR / "d1_teams.json"
SIGMA = 11.0     # how far a college final margin lands from the closing spread (points, one standard deviation)
LIVE_FROM = date(2026, 11, 14)   # the tracker's first day: games before it are the replay
FOUND_TO = "2026-11-11"          # the explore study's last day: replayed games through it are in-sample
ANGLES = {
    "fade_cover": {"name": "Fade the cover", "rule": "Bet against a team that covered the spread in its last game by under 10 points (only when its opponent did not).",
                   "season": {"w": 74, "l": 44}, "past": {"w": 399, "l": 329}, "p": 0.548},
    "big_win": {"name": "Back the big win", "rule": "Back a team that won its last game by 20 or more (only when its opponent did not).",
                "season": {"w": 94, "l": 63}, "past": {"w": 427, "l": 393}, "p": 0.521},
}


def read(p) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def append(p, rows: list[dict]) -> None:
    if rows:
        with p.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------- data

def d1_teams(today: date) -> set[int] | None:
    """Division I teams: those hoopR's schedule gives a conference (365). The leads were found on Division I games
    only, with each team's last game being its last Division I game, so the tracker keeps to the same games: ESPN's
    scoreboard also lists Division I teams' games against lower divisions. Read once a day, cached."""
    have = json.loads(D1.read_text()) if D1.exists() else None
    if have and have.get("day") == today.isoformat():
        return set(have["ids"])
    try:
        import io
        with urllib.request.urlopen(urllib.request.Request(HOOPR.format(s=SEASON), headers=UA), timeout=120) as r:
            x = pl.read_parquet(io.BytesIO(r.read()))
        ids = set()
        for side in ("home", "away"):
            ids |= set(x.filter(pl.col(f"{side}_conference_id").cast(pl.Int64, strict=False).is_not_null())[f"{side}_id"].cast(pl.Int64).to_list())
        if len(ids) > 300:
            D1.write_text(json.dumps({"day": today.isoformat(), "ids": sorted(ids)}))
            return ids
        print(f"hoopR listed only {len(ids)} Division I teams")
    except OSError as e:
        print(f"hoopR schedule: {e}")
    if have:
        return set(have["ids"])
    print("WARNING: no Division I list: every game counts")
    return None


def _day(d: date) -> list[dict]:
    j = _json(BOARD.format(d=d.strftime("%Y%m%d")))
    out = []
    for ev in (j or {}).get("events", []):
        if _dig(ev, "season", "type") not in (None, 2):
            continue
        comp = (ev.get("competitions") or [{}])[0]
        st = _dig(ev, "status", "type") or {}
        row = {"game_id": int(ev["id"]), "day": d.isoformat(), "tip": ev.get("date"), "state": st.get("state"), "done": bool(st.get("completed")),
               "neutral": bool(comp.get("neutralSite"))}
        for c in comp.get("competitors", []):
            side = c.get("homeAway")
            if side not in ("home", "away"):
                continue
            t = c.get("team") or {}
            rk = _dig(c, "curatedRank", "current")
            row.update({f"{side}_id": int(t.get("id") or 0), f"{side}_abbr": t.get("abbreviation"), f"{side}_name": t.get("shortDisplayName") or t.get("displayName"),
                        f"{side}_logo": t.get("logo"), f"{side}_rank": int(rk) if rk is not None and 1 <= int(rk) <= 25 else None,
                        f"{side}_score": float(c["score"]) if c.get("score") not in (None, "") else None})
        if "home_id" in row and "away_id" in row:
            out.append(row)
    return out


def scoreboard(today: date) -> pl.DataFrame:
    """Every game of the season so far and today's, one row each; finished days are cached, the last three and
    today are read again."""
    have = pl.read_parquet(SB) if SB.exists() else None
    cached = set(have["day"].unique().to_list()) if have is not None else set()
    days = [SEASON_START + timedelta(days=i) for i in range((today - SEASON_START).days + 1)]
    todo = [d for d in days if d.isoformat() not in cached or (today - d).days <= 3]
    rows = []
    with ThreadPoolExecutor(6) as ex:
        for r in ex.map(_day, todo):
            rows += r
    if rows:
        new = pl.DataFrame(rows, infer_schema_length=None)
        have = pl.concat([have.filter(~pl.col("day").is_in([d.isoformat() for d in todo])), new], how="diagonal_relaxed") if have is not None else new
        have.write_parquet(SB)
    print(f"scoreboard: {len(todo)} days read, {have.height if have is not None else 0} games")
    return have.unique("game_id", keep="last") if have is not None else pl.DataFrame()


def _line(gid: int) -> dict:
    d = _json(CORE.format(g=gid))
    row = {"game_id": gid}
    items = sorted((d or {}).get("items", []), key=lambda it: _dig(it, "provider", "priority") or 99)
    if not items:
        return row
    it = items[0]
    row["provider"] = _dig(it, "provider", "name")
    for side in ("home", "away"):
        o = _dig(it, f"{side}TeamOdds") or {}
        for when in ("open", "close", "current"):
            w = o.get(when) or {}
            row[f"{when}_spread_{side}"] = _num(_dig(w, "pointSpread", "american"))
            row[f"{when}_price_{side}"] = _am(_dig(w, "spread", "american"))
    if it.get("spread") is not None:   # the item's own line (home side) when a slot is empty, as ncaab/explore.py read it
        for when in ("current", "close"):
            if row.get(f"{when}_spread_home") is None:
                row[f"{when}_spread_home"], row[f"{when}_spread_away"] = float(it["spread"]), -float(it["spread"])
    return row


def lines(sb: pl.DataFrame, today: date) -> pl.DataFrame:
    """Closing lines of finished games (once each) and today's current lines (every build)."""
    have = pl.read_parquet(LINES) if LINES.exists() else None
    final = set(have.filter(pl.col("final"))["game_id"].to_list()) if have is not None else set()
    done = sb.filter(pl.col("done"))["game_id"].to_list()
    upcoming = sb.filter(~pl.col("done") & (pl.col("day") == today.isoformat()))["game_id"].to_list()
    todo = [g for g in done if g not in final] + upcoming
    with ThreadPoolExecutor(8) as ex:
        rows = [r for r in ex.map(_line, todo) if r]
    done_set = set(done)
    for r in rows:
        r["final"] = r["game_id"] in done_set
    if rows:
        new = pl.DataFrame(rows, infer_schema_length=None)
        have = pl.concat([have.filter(~pl.col("game_id").is_in(todo)), new], how="diagonal_relaxed") if have is not None else new
        have.write_parquet(LINES)
    print(f"lines: {len(todo)} games read ({len(upcoming)} today), {have.height if have is not None else 0} kept")
    return have if have is not None else pl.DataFrame({"game_id": []})


# ---------------------------------------------------------------- the leads

def last_games(sb: pl.DataFrame, L: pl.DataFrame, d1: set[int] | None = None) -> pl.DataFrame:
    """Each team's games in order, every game as in ncaab/explore.py (one against a lower division too: its margin
    counts for "won by 20+"), with the result against the closing line, null when the game had none. The study
    fetched lines only for Division I games, so a game against a lower division has none here either."""
    g = sb.filter(pl.col("done") & pl.col("home_score").is_not_null()).join(L.select("game_id", "close_spread_home", "close_spread_away"), on="game_id", how="left")
    if d1:
        lower = ~(pl.col("home_id").is_in(list(d1)) & pl.col("away_id").is_in(list(d1)))
        g = g.with_columns(pl.when(lower).then(None).otherwise(pl.col(c)).alias(c) for c in ("close_spread_home", "close_spread_away"))
    long = pl.concat([
        g.select("game_id", "tip", pl.col("home_id").alias("tid"), (pl.col("home_score") - pl.col("away_score")).alias("margin"), pl.col("close_spread_home").alias("spread")),
        g.select("game_id", "tip", pl.col("away_id").alias("tid"), (pl.col("away_score") - pl.col("home_score")).alias("margin"), pl.col("close_spread_away").alias("spread"))])
    return long.with_columns(ats=pl.col("margin") + pl.col("spread")).sort("tid", "tip")


def flags_for(game: dict, prev: dict, line: dict, now: datetime) -> list[dict]:
    """The leads that fire for one game: a list of flags (team to back, its spread and price at this moment)."""
    # as in ncaab/explore.py: a team's last game is its previous game of any kind; the lead fires when one team
    # has the fact and the other does not (no previous game, an unlined one, or the opposite result all count as not)
    out = []
    ph, pa = prev.get(game["home_id"]), prev.get(game["away_id"])
    cov = lambda p: bool(p) and p["ats"] is not None and 0 < p["ats"] < 10      # the study's "covered" (10+ was its own bucket)
    big = lambda p: bool(p) and p["margin"] is not None and p["margin"] >= 20
    picks = []
    if cov(ph) != cov(pa):
        side = "away" if cov(ph) else "home"
        picks.append(("fade_cover", side, f"{game['home_name' if side == 'away' else 'away_name']} covered its last game by {abs((ph if side == 'away' else pa)['ats']):.1f}"))
    if big(ph) != big(pa):
        side = "home" if big(ph) else "away"
        picks.append(("big_win", side, f"{game[f'{side}_name']} won its last game by {int((ph if side == 'home' else pa)['margin'])}"))
    for angle, side, why in picks:
        sp, pr = line.get(f"current_spread_{side}"), line.get(f"current_price_{side}")
        if sp is None:
            continue
        opp = "away" if side == "home" else "home"
        out.append({"kind": "flag", "id": f"{game['game_id']}:{angle}", "angle": angle, "game_id": game["game_id"], "day": game["day"], "tip": game["tip"],
                    "team": game[f"{side}_name"], "team_abbr": game[f"{side}_abbr"], "team_id": game[f"{side}_id"], "team_logo": game.get(f"{side}_logo"),
                    "opp": game[f"{opp}_name"], "opp_logo": game.get(f"{opp}_logo"), "side": side, "neutral": game["neutral"],
                    "game": f"{game['away_name']} at {game['home_name']}" + (" (neutral)" if game["neutral"] else ""),
                    "spread": sp, "american": pr if pr is not None else -110.0, "book": line.get("provider"),
                    "label": f"{game[f'{side}_name']} {'+' if sp > 0 else ''}{sp:g}" if sp else f"{game[f'{side}_name']} pick'em",
                    "why": why, "p_win": ANGLES[angle]["p"], "ranked": bool(game.get(f"{side}_rank") or game.get(f"{opp}_rank")), "locked_at": now.isoformat()})
    return out


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def grade(flags: list[dict], sb: pl.DataFrame, L: pl.DataFrame, done_ids: set) -> list[dict]:
    games = {r["game_id"]: r for r in sb.filter(pl.col("done")).iter_rows(named=True)}
    close = {r["game_id"]: r for r in L.iter_rows(named=True)}
    out = []
    for f in flags:
        g = games.get(f["game_id"])
        if f["id"] in done_ids or not g or g.get("home_score") is None:
            continue
        side, opp = f["side"], "away" if f["side"] == "home" else "home"
        margin = g[f"{side}_score"] - g[f"{opp}_score"]
        d = margin + f["spread"]
        res = "W" if d > 0 else "L" if d < 0 else "P"
        a = f["american"] or -110.0
        dec = 1 + a / 100 if a > 0 else 1 + 100 / -a
        c = (close.get(f["game_id"]) or {}).get(f"close_spread_{side}")
        r = {"id": f["id"], "result": res, "units": round(dec - 1 if res == "W" else -1.0 if res == "L" else 0.0, 4),
             "margin": margin, "close_spread": c, "graded_at": datetime.now(timezone.utc).isoformat()}
        if c is not None:
            r["clv"] = round(f["spread"] - c, 2)                       # points better than the close
            p = _phi((f["spread"] - c) / SIGMA)                         # covers the locked number if the close is fair
            r["ev_close"] = round(p * dec - 1, 4)
            dc = margin + c
            r["close_result"] = "W" if dc > 0 else "L" if dc < 0 else "P"
        out.append(r)
    return out


def replay(sb: pl.DataFrame, L: pl.DataFrame, seq: pl.DataFrame, now: datetime) -> list[dict]:
    """The leads on every finished game before tracking began, each locked at ESPN's opening line (its close when
    the open was not kept) and graded like a live flag. Recomputed every build, never written to the ledger."""
    by_line = {r["game_id"]: r for r in L.iter_rows(named=True)}
    hist: dict[int, list[dict]] = {}
    for r in seq.iter_rows(named=True):          # sorted by team, then tip
        hist.setdefault(r["tid"], []).append(r)
    games = sb.filter(pl.col("done") & pl.col("home_score").is_not_null() & (pl.col("day") < LIVE_FROM.isoformat()))
    flags = []
    for g in games.iter_rows(named=True):
        ln = by_line.get(g["game_id"]) or {}
        at = "open" if ln.get("open_spread_home") is not None else "close"
        line = {"provider": ln.get("provider")}
        for side in ("home", "away"):
            line[f"current_spread_{side}"], line[f"current_price_{side}"] = ln.get(f"{at}_spread_{side}"), ln.get(f"{at}_price_{side}")
        prev = {}
        for tid in (g["home_id"], g["away_id"]):
            before = [r for r in hist.get(tid, []) if r["tip"] < g["tip"]]
            if before:
                prev[tid] = before[-1]
        for f in flags_for(g, prev, line, now):
            f.update(replay=True, at=at, locked_at=None)
            flags.append(f)
    res = {r["id"]: r for r in grade(flags, sb, L, set())}
    keep = ("id", "angle", "day", "tip", "label", "game", "spread", "american", "book", "at", "why", "result", "units", "margin",
            "close_spread", "clv", "ev_close", "close_result")
    return [{k: v for k, v in {**f, **res.get(f["id"], {})}.items() if k in keep} for f in flags]


# ---------------------------------------------------------------- record and page

def _tally(rows: list[dict]) -> dict:
    g = [r for r in rows if r.get("result") in ("W", "L", "P")]
    w, l, p = (sum(r["result"] == k for r in g) for k in "WLP")
    u = sum(r["units"] for r in g)
    ec = [r["ev_close"] for r in g if r.get("ev_close") is not None]
    cl = [r["clv"] for r in g if r.get("clv") is not None]
    cw = sum(r.get("close_result") == "W" for r in g)
    clo = sum(r.get("close_result") == "L" for r in g)
    return {"n": len(g), "w": w, "l": l, "p": p, "units": round(u, 2), "roi": round(u / len(g), 4) if g else None,
            "ev_close": round(sum(ec) / len(ec), 4) if ec else None, "ev_close_n": len(ec), "clv": round(sum(cl) / len(cl), 2) if cl else None,
            "close": {"w": cw, "l": clo}, "pending": len(rows) - len(g)}


def _replay_block(rp: list[dict]) -> dict:
    found = [r for r in rp if r["day"] <= FOUND_TO]
    after = [r for r in rp if r["day"] > FOUND_TO]
    by = lambda rows: {a: _tally([r for r in rows if r["angle"] == a]) for a in ANGLES}
    return {"from": min((r["day"] for r in rp), default=None), "to": (LIVE_FROM - timedelta(days=1)).isoformat(), "found_to": FOUND_TO,
            "after_from": (date.fromisoformat(FOUND_TO) + timedelta(days=1)).isoformat(),
            "at_open": sum(r.get("at") == "open" for r in rp), "all": _tally(rp), "angles": by(rp),
            "found": by(found), "after": by(after),
            "rows": sorted([r for r in rp if r.get("result")], key=lambda r: (r["day"], r["tip"]), reverse=True)}


def payload(flags: list[dict], results: list[dict], sb: pl.DataFrame, today: date, now: datetime, rp: list[dict] | None = None) -> dict:
    res = {r["id"]: r for r in results}
    rows = [{**f, **res.get(f["id"], {})} for f in flags]
    for r in rows:
        tip = datetime.fromisoformat(r["tip"].replace("Z", "+00:00")).astimezone(ET)
        r["tip_short"] = tip.strftime("%a %-I:%M %p")
    today_rows = [r for r in rows if r["day"] == today.isoformat()]
    gt = sb.filter(pl.col("day") == today.isoformat())
    return {
        "meta": {"generated": now.isoformat(), "day": today.isoformat(), "day_label": now.astimezone(ET).strftime("%A, %B ") + str(today.day),
                 "season": LABEL, "angles": ANGLES, "sigma": SIGMA, "games_today": gt.height},
        "today": sorted(today_rows, key=lambda r: (r["tip"], r["angle"])),
        "record": {"all": _tally(rows), "angles": {a: _tally([r for r in rows if r["angle"] == a]) for a in ANGLES},
                   "rows": sorted([r for r in rows if r.get("result")], key=lambda r: (r["day"], r["tip"]), reverse=True)},
        "replay": _replay_block(rp or []),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-lock", action="store_true", help="grade and refresh the page, lock nothing")
    a = ap.parse_args()
    now = datetime.now(timezone.utc)
    today = now.astimezone(ET).date()
    every = scoreboard(today)              # every game: each team's sequence, as in the study
    d1 = d1_teams(today)
    sb = every.filter(pl.col("home_id").is_in(list(d1)) & pl.col("away_id").is_in(list(d1))) if d1 else every
    print(f"Division I games (flagged and graded): {sb.height} of {every.height}")
    L = lines(sb, today)
    seq = last_games(every, L, d1)
    flags = [f for f in read(FLAGS) if f.get("kind") == "flag"]
    have = {f["id"] for f in flags}
    new = []
    for g in sb.filter((pl.col("day") == today.isoformat()) & ~pl.col("done") & (pl.col("state") == "pre")).iter_rows(named=True):
        tip = datetime.fromisoformat(g["tip"].replace("Z", "+00:00"))
        if tip <= now:
            continue
        before = seq.filter(pl.col("tip") < g["tip"]).group_by("tid").agg(pl.all().sort_by("tip").last())
        prev = {r["tid"]: r for r in before.filter(pl.col("tid").is_in([g["home_id"], g["away_id"]])).iter_rows(named=True)}
        line = next(iter(L.filter(pl.col("game_id") == g["game_id"]).iter_rows(named=True)), {})
        new += [f for f in flags_for(g, prev, line, now) if f["id"] not in have]
    if a.no_lock:
        new = []
    done_ids = {r["id"] for r in read(RESULTS)}
    graded = grade(flags + new, sb, L, done_ids)
    if a.dry_run:
        print(f"dry run: would lock {len(new)} flags and grade {len(graded)}")
    else:
        append(FLAGS, new)
        append(RESULTS, graded)
        print(f"locked {len(new)} flags, graded {len(graded)}")
    rp = replay(sb, L, seq, now)
    data = payload(flags + new, read(RESULTS) + (graded if a.dry_run else []), sb, today, now, rp)
    OUT.write_text(json.dumps(data, separators=(",", ":"), default=str), encoding="utf-8")
    rec = data["record"]
    print(f"record: {rec['all']['w']}-{rec['all']['l']}-{rec['all']['p']}, {rec['all']['units']:+.2f}u; " +
          "; ".join(f"{a}: {t['w']}-{t['l']}" for a, t in rec["angles"].items()) + f"; today {len(data['today'])} flags of {data['meta']['games_today']} games")
    R = data["replay"]
    print(f"replay {R['from']} to {R['to']} ({R['at_open']} at the open): " + "; ".join(
        f"{a}: {t['w']}-{t['l']}-{t['p']} at the line, {t['close']['w']}-{t['close']['l']} at the close, EV at the close {t['ev_close'] if t['ev_close'] is None else round(100 * t['ev_close'], 1)}%"
        for a, t in R["angles"].items()))


if __name__ == "__main__":
    main()
