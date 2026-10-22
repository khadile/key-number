"""Context notes for NBA picks (user, 2026-10-20): quick facts to help choose among picks, computed from data.

- Matchup (props): what the opponent has allowed in that stat per 36 minutes to the player's position group
  (guards, forwards, centers) and where that ranks among 30, this season once each team has 5 games, else last season.
  A player without a pick gets his own stats' matchups (user, 2026-10-20: "centers for rebounds, PG's for assists,
  guards for threes"): the stat he produces most above his position's average (a passing guard's assists, a
  shooter's threes, a big's rebounds), then points. The box scores label positions only G, F or C.
- Teammate out (props): his numbers in the games he played without a regular (20+ minutes) ruled out tonight,
  against the games with him, this season and last.
- Back-to-backs (props): his minutes and the stat on the second night of a back-to-back against other nights.
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
CORE = ("pts", "reb", "ast", "fg3")
# a stat counts as his when he makes at least this much of it per 36 minutes: enough for a book to post it
MIN_RATE = {"reb": 7.0, "ast": 4.5, "fg3": 1.6}
# the stats a position is read in, after points: a center's rebounds; a guard's assists or threes; a forward's any
POS_STATS = {"C": ("reb",), "G": ("ast", "fg3"), "F": ("reb", "fg3", "ast")}
MIN_SPLIT = 5                       # games on each side of a with/without or back-to-back split
REGULAR_MIN = 24.0                  # a teammate whose absence is worth a note plays this many minutes


def _ord(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _where(rank: int, n: int) -> str:
    """Rank 1 = the most of n."""
    if rank <= n / 2:
        return "the most" if rank == 1 else f"the {_ord(rank)}-most"
    k = n - rank + 1
    return "the fewest" if k == 1 else f"the {_ord(k)}-fewest"


def _record(w: int, l: int, p: int = 0) -> str:
    return f"{w}-{l}" + (f"-{p}" if p else "")


# ---------------------------------------------------------------- props

class PropNotes:
    REGULAR_MIN = REGULAR_MIN

    def __init__(self, hist: pl.DataFrame, before: date, season: int):
        h = hist.filter((pl.col("date") < before) & (pl.col("min") > 0) & (pl.col("season") >= season - 1))
        h = h.with_columns(pg=pl.col("pos").map_elements(NP._pos, return_dtype=pl.String),
                           pra=pl.col("pts") + pl.col("reb") + pl.col("ast"), pr=pl.col("pts") + pl.col("reb"),
                           pa=pl.col("pts") + pl.col("ast"), ra=pl.col("reb") + pl.col("ast"))
        self.h = h
        teams = h.select("game_id", "team").unique()
        opp = teams.join(teams, on="game_id", suffix="_o").filter(pl.col("team") != pl.col("team_o")).select("game_id", "team", pl.col("team_o").alias("opp"))
        g = h.join(opp, on=["game_id", "team"])
        self.games = g
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
        self.before = before
        # each position group's per-minute rate in every stat, to find the stats a player makes more of than his peers
        base = h.filter(pl.col("min") >= 10)
        self.grp_rate = {r["pg"]: r for r in base.group_by("pg").agg(*[(pl.col(m).sum() / pl.col("min").sum()).alias(m) for m in CORE]).iter_rows(named=True)}
        # each team-game's rest: days since that team's previous game (1 = the second night of a back-to-back)
        tg = h.select("team", "game_id", "date").unique().sort("team", "date")
        self.rest = tg.with_columns(rest=(pl.col("date") - pl.col("date").shift(1).over("team")).dt.total_days()).select("game_id", "team", "rest")

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
            return f"{opp} allowed {GROUP_NAME[grp]} {v:.1f} {lab} per 36 minutes, {_where(rank, len(x))} of {len(x)} ({label}, {row['g'][0]} games)"
        return None

    def signature(self, athlete_id: int, pos: str) -> list[str]:
        """The stats that are his: of the ones his position is read in (a center's rebounds, a guard's assists or
        threes), the one he makes most of against his position group, if he makes enough of it; then points."""
        x = self.h.filter(pl.col("athlete_id") == athlete_id)
        mins = float(x["min"].sum())
        grp = NP._pos(pos)
        g = self.grp_rate.get(grp)
        if mins < 150 or not g:
            return ["pts"]
        own = {m: float(x[m].sum()) / mins for m in CORE}
        cand = [m for m in POS_STATS[grp] if own[m] * 36 >= MIN_RATE[m] and g[m] > 0]
        best = max(cand, key=lambda m: own[m] / g[m], default=None)
        return [best, "pts"] if best else ["pts"]

    def card(self, athlete_id: int, pos: str, opp: str) -> list[str]:
        """A card's matchups when he has no pick: his own stat first, then points."""
        out = [(m, self.matchup(opp, pos, m)) for m in self.signature(athlete_id, pos)]
        return [n for _, n in out if n]

    def without(self, athlete_id: int, team: str, mates: list[tuple[int, str, str]], market: str) -> str | None:
        """His numbers without the most-used teammate ruled out tonight. `mates`: (id, name, status), most minutes
        first. Only games inside the teammate's time on this team count, so a trade is not an absence."""
        lab = NP.MARKET_LABEL[market].lower()
        p = self.h.filter((pl.col("athlete_id") == athlete_id) & (pl.col("team") == team))
        for mid, name, status in mates:
            t = self.h.filter((pl.col("athlete_id") == mid) & (pl.col("team") == team))
            if t.height < MIN_SPLIT:
                continue
            mine = p.filter(pl.col("date") >= t["date"].min())
            with_ids = set(t["game_id"].to_list())
            w = mine.filter(pl.col("game_id").is_in(list(with_ids)))
            wo = mine.filter(~pl.col("game_id").is_in(list(with_ids)))
            if wo.height < MIN_SPLIT or w.height < MIN_SPLIT:
                continue
            return (f"Without {name} ({status.lower()} tonight): {wo[market].mean():.1f} {lab} and {wo['min'].mean():.1f} minutes a game "
                    f"in {wo.height} games, vs {w[market].mean():.1f} and {w['min'].mean():.1f} with him (this season and last)")
        return None

    def b2b(self, athlete_id: int, market: str) -> str | None:
        """His minutes and the stat on the second night of a back-to-back against other nights."""
        x = self.h.filter(pl.col("athlete_id") == athlete_id).join(self.rest, on=["game_id", "team"], how="left")
        b, o = x.filter(pl.col("rest") == 1), x.filter(pl.col("rest") > 1)
        if b.height < MIN_SPLIT + 1 or o.height < MIN_SPLIT:
            return None
        lab = NP.MARKET_LABEL[market].lower()
        return (f"Second night of a back-to-back: {b['min'].mean():.1f} minutes and {b[market].mean():.1f} {lab} a game in {b.height} such games, "
                f"vs {o['min'].mean():.1f} and {o[market].mean():.1f} otherwise (this season and last)")

    def recent(self, athlete_id: int, n: int = 5) -> dict:
        """His last `n` games, oldest first: date, opponent and every market's value (for the card's strip)."""
        x = self.games.filter(pl.col("athlete_id") == athlete_id).sort("date").tail(n)
        return {"games": [{"d": d.isoformat(), "opp": o} for d, o in zip(x["date"].to_list(), x["opp"].to_list())],
                **{m: [float(v) for v in x[m].to_list()] for m in NP.MARKETS}}

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


def defense_table(hist: pl.DataFrame, season: int, before: date) -> dict:
    """The Defenses tab (user, 2026-10-22: this season's games only): per team, what opponents scored per game in
    points, rebounds, assists and threes, overall and by the scorer's position group (box scores label G, F, C),
    with ranks among the teams (1 = allowed the least). Small samples early: each row carries its games."""
    h = hist.filter((pl.col("season") == season) & (pl.col("date") < before) & (pl.col("min") > 0))
    if not h.height:
        return {"rows": [], "games": 0}
    h = h.with_columns(pg=pl.col("pos").map_elements(NP._pos, return_dtype=pl.String))
    teams = h.select("game_id", "team").unique()
    opp = (teams.join(teams, on="game_id", suffix="_o").filter(pl.col("team") != pl.col("team_o"))
           .select("game_id", "team", pl.col("team_o").alias("defense")))
    g = h.join(opp, on=["game_id", "team"])
    games = g.group_by("defense").agg(pl.col("game_id").n_unique().alias("g"))
    games = games.filter(pl.col("g") >= 0.5 * games["g"].max())   # real teams only (not All-Star sides)
    out = games.join(g.group_by("defense").agg(*[pl.col(m).sum().alias(m) for m in CORE]), on="defense")
    for grp in ("G", "F", "C"):
        part = g.filter(pl.col("pg") == grp).group_by("defense").agg(*[pl.col(m).sum().alias(f"{m}_{grp}") for m in CORE])
        out = out.join(part, on="defense", how="left")
    cols = [c for c in out.columns if c not in ("defense", "g")]
    out = out.with_columns(*[(pl.col(c).fill_null(0) / pl.col("g")).round(1).alias(c) for c in cols])
    out = out.with_columns(*[pl.col(c).rank("min").cast(pl.Int32).alias(f"{c}_rk") for c in cols])
    return {"rows": out.rename({"defense": "team"}).sort("pts").to_dicts(), "games": int(h["game_id"].n_unique()),
            "through": str(h["date"].max())}
