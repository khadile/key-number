"""Context notes for NFL picks (user, 2026-10-20): quick facts to help choose among picks, computed from data.

- Matchup (props): what the opponent's defense has allowed to the player's role (WR1, WR2, TE1, RB1: receiving
  yards a game and its rank of 32, board/defense.py) or, for rushing and passing yards, a game on the ground or
  through the air.
- Form (props): his games over the pick's line this season, and how often he beat his closing line.
- Trends (game lines): how sides in the same situation (prime-time underdogs, divisional favorites, teams off a
  bye, ...) have done against the closing line since 2018 (nflverse), with the record and the sample. History, not
  the model: none of it is used in pricing, and a record within two standard errors of 50% is marked as chance.
"""
from datetime import datetime

import numpy as np
import polars as pl

STAT_OF = {"pass_yds": "pass_yds", "rush_yds": "rush_yds", "rec_yds": "rec_yds", "receptions": "rec"}
LABEL = {"pass_yds": "passing yards", "rush_yds": "rushing yards", "rec_yds": "receiving yards", "receptions": "receptions"}
MIN_N = 50


def _ord(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _say(name: str, res: list[str], win: str, what: str) -> str | None:
    w = sum(r == win for r in res)
    p = sum(r == "P" for r in res)
    l = len(res) - w - p
    n = w + l
    if n < MIN_N:
        return None
    rate = w / n
    z = (rate - 0.5) / (0.5 / np.sqrt(n))
    return f"{name[0].upper() + name[1:]} since 2018: {w}-{l}{f'-{p}' if p else ''} {what} ({rate:.1%})" + ("" if abs(z) >= 2 else " (within chance)")


def _primetime(gameday: str, gametime: str | None) -> bool:
    wd = datetime.fromisoformat(gameday).weekday()
    return wd in (0, 3) or (wd == 6 and (gametime or "") >= "20:00")


SIDE = [
    ("home favorites", lambda r: r["home"] and r["line"] < 0), ("home underdogs", lambda r: r["home"] and r["line"] > 0),
    ("road favorites", lambda r: not r["home"] and r["line"] < 0), ("road underdogs", lambda r: not r["home"] and r["line"] > 0),
    ("prime-time favorites", lambda r: r["prime"] and r["line"] < 0), ("prime-time underdogs", lambda r: r["prime"] and r["line"] > 0),
    ("divisional favorites", lambda r: r["div"] and r["line"] < 0), ("divisional underdogs", lambda r: r["div"] and r["line"] > 0),
    ("teams off a bye against a rested-normally opponent", lambda r: r["rest"] >= 13 and r["opp_rest"] < 13),
    ("underdogs of 7+", lambda r: r["line"] >= 7), ("favorites of 7+", lambda r: r["line"] <= -7),
]
TOTAL = [
    ("prime-time games", lambda r: r["prime"]), ("divisional games", lambda r: r["div"]),
    ("totals of 48+", lambda r: r["total_line"] >= 48), ("totals under 41", lambda r: r["total_line"] < 41),
    ("games in a dome", lambda r: r["dome"]), ("outdoor games with 15+ mph wind", lambda r: not r["dome"] and (r["wind"] or 0) >= 15),
]


class GameTrends:
    def __init__(self, games: pl.DataFrame):
        g = games.filter((pl.col("season") >= 2018) & pl.col("result").is_not_null() & pl.col("spread_line").is_not_null())
        self.sides, self.totals = [], []
        for x in g.iter_rows(named=True):
            prime, div = _primetime(x["gameday"], x["gametime"]), bool(x["div_game"])
            for home in (True, False):
                line = -x["spread_line"] if home else x["spread_line"]
                d = (x["result"] if home else -x["result"]) + line
                self.sides.append({"home": home, "line": line, "prime": prime, "div": div,
                                   "rest": x["home_rest"] if home else x["away_rest"], "opp_rest": x["away_rest"] if home else x["home_rest"],
                                   "res": "W" if d > 0 else ("P" if d == 0 else "L")})
            if x["total_line"] is not None:
                t = x["total"] - x["total_line"]
                self.totals.append({"prime": prime, "div": div, "total_line": x["total_line"], "dome": bool(x["dome"]), "wind": x["wind"],
                                    "res": "O" if t > 0 else ("P" if t == 0 else "U")})

    def notes(self, pick: dict, game: dict, k: int = 3) -> list[str]:
        """`game`: an nflverse games row (gameday, gametime, div_game, home_rest, away_rest, dome, wind)."""
        prime, div = _primetime(game["gameday"], game["gametime"]), bool(game["div_game"])
        out = []
        if pick["market"] == "spread":
            home = pick["side"] == "home"
            me = {"home": home, "line": pick["line"], "prime": prime, "div": div,
                  "rest": game["home_rest"] if home else game["away_rest"], "opp_rest": game["away_rest"] if home else game["home_rest"]}
            for name, f in SIDE:
                if f(me) and (s := _say(name, [r["res"] for r in self.sides if f(r)], "W", "against the spread")):
                    out.append(s)
        elif pick["market"] == "total":
            me = {"prime": prime, "div": div, "total_line": pick["line"], "dome": bool(game["dome"]), "wind": game.get("wind")}
            win = "O" if pick["side"] == "over" else "U"
            for name, f in TOTAL:
                if f(me) and (s := _say(name, [r["res"] for r in self.totals if f(r)], win, f"{'over' if win == 'O' else 'under'} the close")):
                    out.append(s.replace(" since", f", {pick['side']}s, since", 1))
        # the most lopsided records first: the ones a reader would want to see
        return sorted(out, key=lambda s: "within chance" in s)[:k]


def prop_notes(pick: dict, roles: dict, def_rows: list[dict], logs: list[dict], season: int) -> list[str]:
    """`roles[team]`: what each defense allowed by receiver role; `def_rows`: season defense table; `logs`: his games."""
    out = []
    opp, m = pick["opp"], pick["market"]
    if m in ("rec_yds", "receptions"):
        r = next((x for x in roles.get(opp, []) if x["role"] == pick.get("role")), None)
        if r and r.get("g"):
            most = r["n"] - r["rk"] + 1
            where = f"the {_ord(most)}-most" if most <= r["n"] / 2 else f"the {_ord(r['rk'])}-fewest"
            out.append(f"{opp} allows {r['ypg']:.0f} receiving yards a game to {pick['role']}s, {where} of {r['n']} ({r['g']} games)")
    elif m in ("rush_yds", "pass_yds"):
        key = "rush_yds_pg" if m == "rush_yds" else "pass_yds_pg"
        rows = sorted((x for x in def_rows if x.get(key) is not None), key=lambda x: -x[key])
        teams = [x["team"] for x in rows]
        if opp in teams:
            most = teams.index(opp) + 1
            x = rows[most - 1]
            where = f"the {_ord(most)}-most" if most <= len(rows) / 2 else f"the {_ord(len(rows) - most + 1)}-fewest"
            out.append(f"{opp} allows {x[key]:.0f} {'rushing' if m == 'rush_yds' else 'passing'} yards a game, {where} of {len(rows)} ({x['g']} games)")
    games = [g for g in logs if g["season"] == season]
    stat = STAT_OF[m]
    if len(games) >= 3:
        v = [g[stat] or 0 for g in games]
        hit = sum((x > pick["line"]) if pick["side"] == "over" else (x < pick["line"]) for x in v)
        out.append(f"{'Over' if pick['side'] == 'over' else 'Under'} {pick['line']:g} in {hit} of {len(v)} games this season; averaged {np.mean(v):.1f}")
        beat = [g for g in games if (g.get("lines") or {}).get(m) is not None]
        if len(beat) >= 3:
            b = sum((g[stat] > g["lines"][m]) if pick["side"] == "over" else (g[stat] < g["lines"][m]) for g in beat)
            out.append(f"{'Beat' if pick['side'] == 'over' else 'Stayed under'} his closing {LABEL[m]} line in {b} of {len(beat)}")
    return out
