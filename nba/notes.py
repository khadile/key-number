"""Context notes for NBA picks (user, 2026-10-20): quick facts to help choose among picks, computed from data.

- Matchup (props): what the opponent has allowed in that stat per 36 minutes to the player's position group
  (guards, forwards, centers) and where that ranks among 30, this season once each team has 5 games, else last season.
- Form (props): how often the player cleared the pick's line in his last 10 games.
- Trends (game lines): how sides in the same situation (home underdogs, the second night of a back-to-back,
  10-point favorites, ...) have done against the consensus closing line since 2020-21, with the record and the
  sample. History, not the model: none of it is used in pricing, and a record within two standard errors of
  50% is marked as noise.

Every note carries its numbers; nothing is stated without them.
"""
from datetime import date

import numpy as np
import polars as pl

from nba import props as NP

GROUP_NAME = {"G": "guards", "F": "forwards", "C": "centers"}
MIN_TEAM_GAMES = 5


def _ord(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _record(w: int, l: int, p: int = 0) -> str:
    return f"{w}-{l}" + (f"-{p}" if p else "")


# ---------------------------------------------------------------- props

class PropNotes:
    def __init__(self, hist: pl.DataFrame, before: date, season: int):
        h = hist.filter((pl.col("date") < before) & (pl.col("min") > 0) & (pl.col("season") >= season - 1))
        h = h.with_columns(pg=pl.col("pos").map_elements(NP._pos, return_dtype=pl.String),
                           pra=pl.col("pts") + pl.col("reb") + pl.col("ast"), pr=pl.col("pts") + pl.col("reb"),
                           pa=pl.col("pts") + pl.col("ast"), ra=pl.col("reb") + pl.col("ast"))
        self.h = h
        teams = h.select("game_id", "team").unique()
        opp = teams.join(teams, on="game_id", suffix="_o").filter(pl.col("team") != pl.col("team_o")).select("game_id", "team", pl.col("team_o").alias("opp"))
        g = h.join(opp, on=["game_id", "team"])
        per = g.group_by("season", "game_id", "opp", "pg").agg(pl.col("min").sum(), *[pl.col(m).sum() for m in NP.MARKETS])
        self.allowed = {}
        for s in (season, season - 1):
            x = per.filter(pl.col("season") == s)
            if not x.height:
                continue
            games = x.group_by("opp").agg(pl.col("game_id").n_unique().alias("g"))
            games = games.filter(pl.col("g") >= 0.5 * games["g"].max())   # real teams only (not All-Star sides)
            # per 36 minutes played by the position group: comparable whoever plays how much
            avg = (x.group_by("opp", "pg").agg(*[(pl.col(m).sum() / pl.col("min").sum() * 36).alias(m) for m in NP.MARKETS])
                   .join(games, on="opp", how="inner"))
            self.allowed[s] = avg
        self.season = season

    def matchup(self, opp: str, pos: str, market: str) -> str | None:
        grp = NP._pos(pos)
        for s, label in ((self.season, "this season"), (self.season - 1, "last season")):
            a = self.allowed.get(s)
            if a is None:
                continue
            row = a.filter((pl.col("opp") == opp) & (pl.col("pg") == grp))
            if not row.height or (s == self.season and row["g"][0] < MIN_TEAM_GAMES):
                continue
            x = a.filter(pl.col("pg") == grp).sort(market, descending=True)
            rank = x["opp"].to_list().index(opp) + 1
            v = row[market][0]
            lab = NP.MARKET_LABEL[market].lower()
            where = f"the {_ord(rank)}-most" if rank <= 15 else f"the {_ord(len(x) - rank + 1)}-fewest"
            return f"{opp} allowed {GROUP_NAME[grp]} {v:.1f} {lab} per 36 minutes, {where} of {len(x)} ({label}, {row['g'][0]} games)"
        return None

    def form(self, athlete_id: int, market: str, line: float, side: str, n: int = 10) -> str | None:
        x = self.h.filter(pl.col("athlete_id") == athlete_id).sort("date").tail(n)
        if x.height < 5:
            return None
        v = x[market].to_numpy()
        hit = int((v > line).sum() if side == "over" else (v < line).sum())
        cur = x.filter(pl.col("season") == self.season).height
        span = f"last {x.height} games" + (f" ({cur} this season)" if 0 < cur < x.height else "")
        return f"{'Over' if side == 'over' else 'Under'} {line:g} in {hit} of his {span}; averaged {v.mean():.1f}"


# ---------------------------------------------------------------- game-line trends

SITUATIONS_SIDE = [
    ("home favorites", lambda r: r["home"] and r["line"] < 0), ("home underdogs", lambda r: r["home"] and r["line"] > 0),
    ("road favorites", lambda r: not r["home"] and r["line"] < 0), ("road underdogs", lambda r: not r["home"] and r["line"] > 0),
    ("teams on the second night of a back-to-back", lambda r: r["rest"] == 1),
    ("teams facing a team on a back-to-back", lambda r: r["opp_rest"] == 1 and r["rest"] > 1),
    ("teams with 2+ more days of rest", lambda r: r["rest"] - r["opp_rest"] >= 2),
    ("underdogs of 10+", lambda r: r["line"] >= 10), ("favorites of 10+", lambda r: r["line"] <= -10),
    ("underdogs in a season's first three weeks", lambda r: r["early"] and r["line"] > 0),
    ("favorites in a season's first three weeks", lambda r: r["early"] and r["line"] < 0),
]
SITUATIONS_TOTAL = [
    ("totals of 235+", lambda r: r["total_line"] >= 235), ("totals under 215", lambda r: r["total_line"] < 215),
    ("games with a team on a back-to-back", lambda r: r["home_rest"] == 1 or r["away_rest"] == 1),
    ("games in a season's first three weeks", lambda r: r["early"]),
]
MIN_N = 60


class GameTrends:
    """Records against the consensus close, 2020-21 to last season (hoopR), by situation."""

    def __init__(self):
        from nba.lines import _results
        from nba.model import team_games
        r = _results()
        tg = team_games(range(2021, 2027)).select("game_id", "team", "is_home", "rest", "date", "season")
        h = tg.filter(pl.col("is_home")).select("game_id", "date", "season", pl.col("rest").alias("home_rest"))
        a = tg.filter(~pl.col("is_home")).select("game_id", pl.col("rest").alias("away_rest"))
        g = r.join(h, on="game_id").join(a, on="game_id")
        opener = g.group_by("season").agg(pl.col("date").min().alias("open"))
        g = g.join(opener, on="season").with_columns(early=(pl.col("date") - pl.col("open")).dt.total_days() < 21)
        sides, totals = [], []
        for x in g.iter_rows(named=True):
            for home in (True, False):
                line = -x["close_spread"] if home else x["close_spread"]
                margin = x["margin"] if home else -x["margin"]
                d = margin + line
                sides.append({"home": home, "line": line, "rest": x["home_rest"] if home else x["away_rest"],
                              "opp_rest": x["away_rest"] if home else x["home_rest"], "early": x["early"],
                              "res": "W" if d > 0 else ("P" if d == 0 else "L")})
            t = x["total"] - x["close_total"]
            totals.append({"total_line": x["close_total"], "home_rest": x["home_rest"], "away_rest": x["away_rest"], "early": x["early"],
                           "res": "O" if t > 0 else ("P" if t == 0 else "U")})
        self.sides, self.totals = sides, totals

    @staticmethod
    def _say(name: str, res: list[str], win: str, what: str) -> str | None:
        w = sum(r == win for r in res)
        p = sum(r == "P" for r in res)
        l = len(res) - w - p
        n = w + l
        if n < MIN_N:
            return None
        rate = w / n
        z = (rate - 0.5) / (0.5 / np.sqrt(n))
        flag = "" if abs(z) >= 2 else " (within chance)"
        return f"{name[0].upper() + name[1:]} since 2020-21: {_record(w, l, p)} {what} ({rate:.1%}){flag}"

    def notes(self, pick: dict, ctx: dict, k: int = 3) -> list[str]:
        """`pick`: market, side, line; `ctx`: home_rest, away_rest, early, total_line."""
        out = []
        if pick["market"] == "spread":
            home = pick["side"] == "home"
            me = {"home": home, "line": pick["line"], "rest": ctx["home_rest"] if home else ctx["away_rest"],
                  "opp_rest": ctx["away_rest"] if home else ctx["home_rest"], "early": ctx["early"]}
            for name, f in SITUATIONS_SIDE:
                if f(me):
                    s = self._say(name, [r["res"] for r in self.sides if f(r)], "W", "against the spread")
                    if s:
                        out.append(s)
        elif pick["market"] == "total":
            me = {"total_line": pick["line"], "home_rest": ctx["home_rest"], "away_rest": ctx["away_rest"], "early": ctx["early"]}
            for name, f in SITUATIONS_TOTAL:
                if f(me):
                    win = "O" if pick["side"] == "over" else "U"
                    s = self._say(name, [r["res"] for r in self.totals if f(r)], win, f"{'over' if win == 'O' else 'under'} the close")
                    if s:
                        out.append(s.replace(" since", f", {pick['side']}s, since", 1))
        return out[:k]
