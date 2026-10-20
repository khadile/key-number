"""Context notes for NFL picks (user, 2026-10-20): quick facts to help choose among picks, computed from data.

- Matchup (props): what the opponent's defense has allowed this season to the player's role in the pick's own
  market (user, 2026-10-20: an RB's receptions read the defense against RB receiving, a QB's rushing its defense
  against quarterbacks running): receptions, receiving yards, rushing yards (with yards a carry) or passing
  yards a game to QB1s, RB1s, RB2s, WR1s, WR2s or TE1s, the props board's own usage roles, ranked among 32.
  A player without a pick gets his role's two markets (a QB's passing and rushing, an RB's rushing and catches,
  a receiver's catches and yards).
- Game environment (props): the team's implied points from the spread and total when they rank in the top or
  bottom quarter of the week; how the role's production in that market has moved with the spread (favorites run,
  underdogs throw) and, outdoors in 15+ mph wind, with the wind, since 2018: each game against the player's own
  season average, the situation against the role's typical game (shown when the effect is 5%+ on 200+ games and
  two standard errors from nothing).
- Usage (props): his target share or carries over his last three games when they moved 20%+ from his season.
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


MAIN_MARKET = {"QB": "pass_yds", "RB": "rush_yds", "WR": "rec_yds", "TE": "rec_yds"}
ROLE_WORD = {"QB1": "quarterbacks", "RB1": "RB1s", "RB2": "RB2s", "WR1": "WR1s", "WR2": "WR2s", "TE1": "TE1s"}
HIST_COL = {"pass_yds": "passing_yards", "rush_yds": "rushing_yards", "rec_yds": "receiving_yards", "receptions": "receptions"}
# the markets a card without a pick reads, by role
CARD_MARKETS = {"QB1": ("pass_yds", "rush_yds"), "RB1": ("rush_yds", "receptions"), "RB2": ("rush_yds", "receptions"),
                "WR1": ("receptions", "rec_yds"), "WR2": ("receptions", "rec_yds"), "TE1": ("receptions", "rec_yds")}
# the team's own expected margin (+ favored): favorites run, underdogs throw
SPREAD_BUCKETS = (("favored by 7+", 7.0, 99.0), ("favored by 3 to 7", 3.0, 7.0), ("underdogs of 3 to 7", -7.0, -3.0), ("underdogs of 7+", -99.0, -7.0))
MIN_EFFECT, MIN_GAMES = 0.05, 200


def _where(rank: int, n: int) -> str:
    """Rank 1 = the most of n."""
    if rank <= n / 2:
        return "the most" if rank == 1 else f"the {_ord(rank)}-most"
    k = n - rank + 1
    return "the fewest" if k == 1 else f"the {_ord(k)}-fewest"


def _in_bucket(spread: float, lo: float, hi: float) -> bool:
    return lo <= spread < hi if lo > 0 else lo < spread <= hi


class RoleDefense:
    """What each defense allowed per game this season to opponents' QB1, RB1, RB2, WR1, WR2 and TE1: the props
    board's own roles (usage within each team), so a WR2's note compares WR2s. `hist`: this season's player-games
    with role and opp (props.backtest.baselines(frames)), before this week."""

    def __init__(self, hist: pl.DataFrame):
        x = hist.filter(pl.col("role").is_in(list(ROLE_WORD)))
        per = (x.group_by("opp", "role").agg(g=pl.col("game_id").n_unique(), car=pl.col("carries").sum(),
                                              *[pl.col(c).cast(pl.Float64).sum().alias(m) for m, c in HIST_COL.items()])
                 .with_columns(ypc=pl.col("rush_yds") / pl.col("car").clip(1))
                 .with_columns(*[(pl.col(m) / pl.col("g")).alias(m) for m in HIST_COL]))
        self.t = {(r["opp"], r["role"]): r for r in per.iter_rows(named=True)}
        self.order = {(role, m): g.sort(m, descending=True)["opp"].to_list()
                      for (role,), g in per.group_by("role") for m in HIST_COL}

    def note(self, opp: str, role: str | None, market: str) -> tuple[str, float] | None:
        """The note and how lopsided it is (0 = mid-table, 1 = the top or the bottom)."""
        r = self.t.get((opp, role))
        if r is None or r["g"] < 3 or (role != "QB1" and market == "pass_yds"):
            return None
        teams = self.order[(role, market)]
        n, rank = len(teams), teams.index(opp) + 1
        v, who, tail = r[market], ROLE_WORD[role], f"{_where(rank, n)} of {n} ({r['g']} games)"
        if market == "receptions":
            s = f"{opp} allows {v:.1f} catches a game to {who}, {tail}"
        elif market == "rec_yds":
            s = f"{opp} allows {v:.0f} receiving yards a game to {who}, {tail}"
        elif market == "rush_yds":
            carry = f" ({r['ypc']:.1f} a carry)" if role != "QB1" else ""
            s = f"{opp} allows {v:.0f} rushing yards a game to {who}{carry}, {tail}"
        else:
            s = f"{opp} allows {v:.0f} passing yards a game, {tail}"
        return s, abs(rank - (n + 1) / 2) / max((n - 1) / 2, 1)

    def card(self, opp: str, role: str | None) -> list[str]:
        """A card's matchups when he has no pick: his role's two markets, the more lopsided first."""
        out = [x for m in CARD_MARKETS.get(role, ()) if (x := self.note(opp, role, m))]
        return [s for s, _ in sorted(out, key=lambda x: -x[1])]


class Situations:
    """League history, 2018 to last season, from the props board's player-games: how a role's production in a
    market moved with the spread and with wind. Each game is measured against the player's own season-to-date
    average, and a situation against the role's typical game: a role's raw ratio drifts with who holds it (RB2s ran
    13% under their season average in every kind of game, RB1s' receiving yards 11%), which is not the situation's
    doing. A record of what happened, never used in pricing (the props market already knows the spread)."""

    def __init__(self, hist: pl.DataFrame, sched: pl.DataFrame):
        x = (hist.filter(pl.col("role").is_in(list(ROLE_WORD)))
                 .join(sched.select("game_id", "wind", "roof"), on="game_id", how="left"))
        self.rows: dict = {}
        for m, c in HIST_COL.items():
            b = f"b_std_{m}"
            y = x.filter(pl.col(b).is_not_null() & (pl.col(b) >= (1.0 if m == "receptions" else 10.0)))
            for role in ROLE_WORD:
                r = y.filter(pl.col("role") == role)
                self.typical = getattr(self, "typical", {})
                self.typical[(role, m)] = float(r[c].cast(pl.Float64).sum() / r[b].sum()) if r.height else None
                for name, lo, hi in SPREAD_BUCKETS:
                    s = r.filter((pl.col("spread") >= lo) & (pl.col("spread") < hi)) if lo > 0 else r.filter((pl.col("spread") > lo) & (pl.col("spread") <= hi))
                    self._keep(("spread", role, m, name), s, c, b)
                self._keep(("wind", role, m), r.filter(pl.col("roof").is_in(["outdoors", "open"]) & (pl.col("wind") >= 15)), c, b)

    def _keep(self, key, s: pl.DataFrame, col: str, base: str) -> None:
        """Keep an effect of 5%+ on 200+ games that is also two standard errors from nothing (a ratio of sums'
        delta-method error): QB rushing swings too much game to game for most of its splits to clear it."""
        typ = self.typical.get((key[1], key[2]))
        if s.height < MIN_GAMES or not typ:
            return
        a, b = s[col].cast(pl.Float64).to_numpy(), s[base].to_numpy()
        r = a.sum() / b.sum()
        eff, se = r / typ - 1, float(np.sqrt(((a - r * b) ** 2).sum()) / b.sum()) / typ
        if abs(eff) >= MIN_EFFECT and abs(eff) >= 2 * se:
            self.rows[key] = (eff, s.height)

    def notes(self, role: str | None, market: str, spread: float | None, wind: float | None, outdoors: bool) -> list[str]:
        out = []
        who = ROLE_WORD.get(role)
        if who is None:
            return out
        if spread is not None:
            for name, lo, hi in SPREAD_BUCKETS:
                if _in_bucket(spread, lo, hi) and (e := self.rows.get(("spread", role, market, name))):
                    out.append(f"{who[0].upper() + who[1:]} on teams {name} since 2018: {100 * e[0]:+.0f}% {LABEL[market]} "
                               f"against a typical game for the role ({e[1]:,} games)")
        if outdoors and (wind or 0) >= 15 and (e := self.rows.get(("wind", role, market))):
            out.append(f"Forecast wind {wind:.0f} mph: {who} outdoors in 15+ mph wind since 2018 had {100 * e[0]:+.0f}% {LABEL[market]} "
                       f"against a typical game for the role ({e[1]:,} games)")
        return out


def usage_note(logs: list[dict], season: int, market: str) -> str | None:
    """His target share (receiving markets) or carries (rushing) over the last three games, when they moved 20%+
    from his earlier games this season."""
    g = [x for x in logs if x["season"] == season]
    if len(g) < 5:
        return None
    last, early = g[-3:], g[:-3]
    if market in ("rec_yds", "receptions"):
        a, b = np.mean([x.get("share") or 0 for x in last]), np.mean([x.get("share") or 0 for x in early])
        if b >= 0.05 and abs(a / b - 1) >= 0.2 and abs(a - b) >= 0.03:
            return f"Target share {100 * a:.0f}% over his last 3 games, {100 * b:.0f}% before"
    elif market == "rush_yds":
        a, b = np.mean([x.get("car") or 0 for x in last]), np.mean([x.get("car") or 0 for x in early])
        if b >= 3 and abs(a / b - 1) >= 0.2 and abs(a - b) >= 2:
            return f"{a:.1f} carries a game over his last 3 games, {b:.1f} before"
    return None


def implied_note(team: str, implied: dict[str, float]) -> str | None:
    """The team's implied points from the spread and total, when it is in the week's top or bottom quarter."""
    if team not in implied or len(implied) < 8:
        return None
    order = sorted(implied, key=lambda t: -implied[t])
    rank, n = order.index(team) + 1, len(order)
    if n / 4 < rank <= n - n / 4:
        return None
    word = _where(rank, n).replace("most", "highest").replace("fewest", "lowest")
    return f"{team} implied {implied[team]:.1f} points from the spread and total, {word} of {n} teams this week"


def matchup_note(opp: str, role: str | None, market: str, roles: dict, def_rows: list[dict]) -> str | None:
    """What the opponent's defense allows in this market: to the player's receiver role (rank of 32), or on the
    ground or through the air."""
    if market in ("rec_yds", "receptions"):
        r = next((x for x in roles.get(opp, []) if x["role"] == role), None)
        if r and r.get("g"):
            most = r["n"] - r["rk"] + 1
            where = f"the {_ord(most)}-most" if most <= r["n"] / 2 else f"the {_ord(r['rk'])}-fewest"
            return f"{opp} allows {r['ypg']:.0f} receiving yards a game to {role}s, {where} of {r['n']} ({r['g']} games)"
    elif market in ("rush_yds", "pass_yds"):
        key = "rush_yds_pg" if market == "rush_yds" else "pass_yds_pg"
        rows = sorted((x for x in def_rows if x.get(key) is not None), key=lambda x: -x[key])
        teams = [x["team"] for x in rows]
        if opp in teams:
            most = teams.index(opp) + 1
            x = rows[most - 1]
            where = f"the {_ord(most)}-most" if most <= len(rows) / 2 else f"the {_ord(len(rows) - most + 1)}-fewest"
            return f"{opp} allows {x[key]:.0f} {'rushing' if market == 'rush_yds' else 'passing'} yards a game, {where} of {len(rows)} ({x['g']} games)"
    return None


def prop_notes(pick: dict, roles: dict, def_rows: list[dict], logs: list[dict], season: int,
               rd: "RoleDefense | None" = None, sit: "Situations | None" = None, env: dict | None = None) -> list[str]:
    """`roles[team]`: what each defense allowed by receiver role; `def_rows`: season defense table; `logs`: his games;
    `rd`, `sit`: the role defense and the league's situations; `env`: spread (his team's expected margin), wind,
    outdoors and the week's implied team points."""
    m = pick["market"]
    x = rd.note(pick["opp"], pick.get("role"), m) if rd else None
    out = [x[0]] if x else [n for n in (matchup_note(pick["opp"], pick.get("role"), m, roles, def_rows),) if n]
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
    out += [n for n in (usage_note(logs, season, m),) if n]
    if env:
        out += [n for n in (implied_note(pick["team"], env.get("implied", {})),) if n]
        if sit:
            out += sit.notes(pick.get("role"), m, env.get("spread"), env.get("wind"), env.get("outdoors", False))
    return out
