"""Where the engine's game state and prices come from.

LiveFeed: ESPN's game feed (free, polled every few seconds) plus PropLine's
odds (budgeted: one request per poll covers every market and book for a game,
polled after new plays rather than on a fixed clock).

DemoFeed: replays a finished 2026 game from nflverse play-by-play and the
OddsPapi archive's in-game prices at any speed, so the engine and the live
page can be tried without a game on or a single PropLine request.

Both return the same shapes:
    state():  {seconds_left, score {team: pts}, possession, yards_to_endzone, period, clock,
               final, detail, new_plays [...], banked {gsis: {...}}, team_so_far {team: {...}}}
    odds():   {"props": {(gsis, market): {book: {line, over, under, age}}},
               "game": {book: {spread, spread_price_a, ..., ml_home, ml_away}}, "at": iso,
               "books": {book: seconds since PropLine last refreshed it, None if frozen},
               "stale": {(gsis or "game", market, book)} quotes dropped only for being too old}  (live only)
A market missing from "props"/"game" but listed in "stale" may still be up at the book: the engine
treats it as "can't tell yet", never as pulled.
Prices are decimal odds.
"""
import time
from datetime import datetime, timedelta, timezone

import polars as pl

from config import RAW
from live import espn
from pipeline import propline, sgo
from pipeline.odds_history import TEAMS

PROP_KEYS = {"player_pass_yds": "pass_yds", "player_rush_yds": "rush_yds", "player_reception_yds": "rec_yds",
             "player_receptions": "receptions"}
MARKETS = ",".join(list(PROP_KEYS) + ["h2h", "spreads", "totals"])
GAME_KEYS = {"h2h": "ml", "spreads": "spread", "totals": "total"}
MAX_AGE_S = 90
# During a dead ball (a score until the next snap; halftime) the game cannot change, so what makes a quote current is
# that the book posted it after the play that froze the game, not its age. The margin covers the books' repricing
# (weeks 1-4, first new game-line price after a score: DraftKings 7 s, FanDuel 9 s, Pinnacle 29 s median and 62 s
# at the 75th percentile). PropLine refreshes each book's live game lines only every 2-4 minutes.
SINCE_MARGIN_S = {"pinnacle": 60}
SINCE_MARGIN_DEFAULT_S = 20


def dec(american: float) -> float:
    return 1 + american / 100 if american > 0 else 1 + 100 / -american


def espn_to_gsis() -> dict[str, str]:
    p = pl.read_parquet(RAW / "players.parquet").select("espn_id", "gsis_id").drop_nulls()
    return {str(int(float(e))): g for e, g in p.iter_rows()}


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def parse_propline(o: dict, home: str, away: str, now: datetime, since: datetime | None = None,
                   kinds: tuple = ("prop", "game")) -> dict:
    """PropLine per-event odds -> main lines the engine can price, stale and pulled prices dropped.
    In play a quote counts when it is at most MAX_AGE_S old. During a dead ball (`since`: the wallclock of the play
    that froze the game; `kinds`: the markets the open decision point decides) a quote for those markets counts
    when the book posted it at least SINCE_MARGIN_S after that play, whatever its age (marked by_state), and one
    posted before does not, however young: it priced a different game."""
    props: dict = {}
    game: dict = {}
    fresh: dict = {}   # book -> age of its freshest main market: PropLine refreshes each book on its own cycle
    stale: set = set()  # (player id or "game", market, book) dropped only for age
    ok_books: set = set()  # books with at least one quote that counts
    nick = {home: None, away: None}
    for full, abbr in TEAMS.items():
        if abbr in nick:
            nick[abbr] = full.split()[-1]
    for b in o.get("bookmakers", []):
        book = b["key"]
        if b.get("pregame_only") and o.get("live"):
            fresh[book] = None
            continue                         # frozen pregame quote during the game
        for m in b.get("markets", []):
            # Pinnacle's other spread and total numbers: the reference at whatever number DK/FD hang (user, 2026-10-11)
            alt = (book == "pinnacle" and m.get("line_type") == "alternate" and m.get("key") in ("spreads", "totals")
                   and not m.get("team"))
            if (m.get("line_type") != "main" and not alt) or m.get("suspended_at"):
                continue                     # other alternates, milestones and suspended markets
            if m.get("period") not in (None, "", "full_game", "game"):
                continue                     # quarter and half markets are out of scope
            upd = _parse_ts(m.get("last_update"))
            age = (now - upd).total_seconds() if upd else 9e9
            fresh[book] = min(fresh.get(book, 9e9), age)
            dead = since is not None and ("prop" if m.get("key") in PROP_KEYS else "game") in kinds
            by_state = bool(o.get("live") and dead and upd
                            and upd >= since + timedelta(seconds=SINCE_MARGIN_S.get(book, SINCE_MARGIN_DEFAULT_S)))
            if o.get("live") and not by_state and (dead or age > MAX_AGE_S):
                if alt:
                    continue
                if m["key"] in PROP_KEYS:
                    stale.update((x["player_id"], PROP_KEYS[m["key"]], book) for x in m.get("outcomes", []) if x.get("player_id"))
                elif m["key"] in GAME_KEYS and not m.get("team"):
                    stale.add(("game", GAME_KEYS[m["key"]], book))
                continue
            ok_books.add(book)
            outs = [x for x in m.get("outcomes", []) if not x.get("last_seen_at") or not m.get("last_update")
                    or x["last_seen_at"] >= m["last_update"]]
            key = m["key"]
            if alt:
                if key == "spreads":   # {home handicap: (home price, away price)}
                    pr = {}
                    for x in outs:
                        side = "home" if nick[home] and x["name"].endswith(nick[home]) else ("away" if nick[away] and x["name"].endswith(nick[away]) else None)
                        if side and x.get("point") is not None:
                            pr[side] = (float(x["point"]), dec(x["price"]))
                    if len(pr) == 2 and pr["home"][0] == -pr["away"][0]:
                        game.setdefault(book, {}).setdefault("alt_spread", {})[pr["home"][0]] = (pr["home"][1], pr["away"][1])
                else:                  # {total: (over price, under price)}
                    pr = {x["name"]: (float(x["point"]), dec(x["price"])) for x in outs if x.get("point") is not None}
                    if "Over" in pr and "Under" in pr and pr["Over"][0] == pr["Under"][0]:
                        game.setdefault(book, {}).setdefault("alt_total", {})[pr["Over"][0]] = (pr["Over"][1], pr["Under"][1])
                continue
            if key in PROP_KEYS:
                by_player: dict = {}
                for x in outs:
                    if x.get("point") is None or x["name"] not in ("Over", "Under") or not x.get("player_id"):
                        continue
                    by_player.setdefault((x["player_id"], x["point"]), {})[x["name"]] = x["price"]
                for (pid, point), sides in by_player.items():
                    if "Over" in sides and "Under" in sides:
                        props.setdefault((pid, PROP_KEYS[key]), {})[book] = {
                            "line": float(point), "over": dec(sides["Over"]), "under": dec(sides["Under"]), "age": age,
                            "by_state": by_state}
            elif key == "h2h":
                pr = {}
                for x in outs:
                    side = "home" if nick[home] and x["name"].endswith(nick[home]) else ("away" if nick[away] and x["name"].endswith(nick[away]) else None)
                    if side:
                        pr[side] = dec(x["price"])
                if len(pr) == 2:
                    game.setdefault(book, {}).update(ml_home=pr["home"], ml_away=pr["away"], age=age, age_ml=age,
                                                     by_state_ml=by_state)
            elif key == "spreads":
                pr = {}
                for x in outs:
                    side = "home" if nick[home] and x["name"].endswith(nick[home]) else ("away" if nick[away] and x["name"].endswith(nick[away]) else None)
                    if side and x.get("point") is not None:
                        pr[side] = (float(x["point"]), dec(x["price"]))
                if len(pr) == 2:
                    game.setdefault(book, {}).update(spread=-pr["home"][0], spread_price_a=pr["home"][1],
                                                     spread_price_b=pr["away"][1], age=age, age_spread=age,
                                                     by_state_spread=by_state)
            elif key == "totals" and not m.get("team"):
                pr = {x["name"]: (float(x["point"]), dec(x["price"])) for x in outs if x.get("point") is not None}
                if "Over" in pr and "Under" in pr:
                    game.setdefault(book, {}).update(total=pr["Over"][0], total_price_a=pr["Over"][1],
                                                     total_price_b=pr["Under"][1], age=age, age_total=age,
                                                     by_state_total=by_state)
    return {"props": props, "game": game, "at": now.isoformat(), "books": fresh, "stale": stale, "ok_books": ok_books,
            "since": since.isoformat() if since else None}


class LiveFeed:
    """ESPN + PropLine for one game."""

    def __init__(self, espn_event: str, propline_event: str, home: str, away: str, e2g: dict[str, str]):
        self.eid, self.pid, self.home, self.away, self.e2g = espn_event, propline_event, home, away, e2g
        self.seen: set[str] = set()
        self.last = None

    def state(self) -> dict | None:
        d = espn.summary(self.eid)
        if d is None:
            return None
        g = espn.parse(d)
        new = [p for p in g["plays"] if p["id"] not in self.seen]
        self.seen.update(p["id"] for p in new)
        last = g["plays"][-1] if g["plays"] else {}
        banked = {}
        for aid, b in g["box"].items():
            gsis = self.e2g.get(aid)
            if gsis:
                banked[gsis] = {"tgt": b.get("tgt", 0), "rec": b.get("rec", 0), "rec_yds": b.get("rec_yds", 0),
                                "car": b.get("car", 0), "rush_yds": b.get("rush_yds", 0), "att": b.get("att", 0),
                                "cmp": b.get("cmp", 0), "pass_yds": b.get("pass_yds", 0)}
        team_so = {t: {"plays": tb["att"] + tb["car"], "tgt": tb["tgt"], "runs": tb["car"], "att": tb["att"]}
                   for t, tb in g["team_box"].items()}
        self.last = g
        return {"seconds_left": g["seconds_left"], "score": g["score"], "possession": g["possession"],
                "yards_to_endzone": last.get("yards_to_endzone"), "period": g["period"], "clock": g["clock"],
                "final": g["final"], "live": g["state"] == "in", "detail": g["detail"], "new_plays": new,
                "banked": banked, "team_so_far": team_so, "box_names": {self.e2g.get(a): b.get("name") for a, b in g["box"].items()}}

    def odds(self, since: datetime | None = None, kinds: tuple = ("prop", "game")) -> dict | None:
        o = propline.get(f"sports/{propline.SPORT}/events/{self.pid}/odds",
                         {"markets": MARKETS, "bookmakers": propline.BOOKS})
        if not isinstance(o, dict) or "error" in o:
            return None
        out = parse_propline(o, self.home, self.away, datetime.now(timezone.utc), since, kinds)
        out["props"] = {(self._gsis(pid), m): v for (pid, m), v in out["props"].items() if self._gsis(pid)}
        out["stale"] = {(k if k == "game" else self._gsis(k), m, b) for k, m, b in out["stale"]}
        return out

    def _gsis(self, player_id: str) -> str | None:
        return self.e2g.get(player_id.split(":")[-1]) if player_id else None


# ---------------------------------------------------------------- SportsGameOdds (trial week, 2026-10)
SGO_STATS = {"passing_yards": "pass_yds", "rushing_yards": "rush_yds", "receiving_yards": "rec_yds",
             "receiving_receptions": "receptions"}
SGO_MAX_AGE_S = 120   # SGO refreshes every 30-60 s on the trial plan, so allow one missed cycle


def _sgo_dec(x) -> float | None:
    try:
        a = float(str(x).replace("+", ""))
    except (TypeError, ValueError):
        return None
    return dec(a) if abs(a) >= 100 else None


def parse_sgo(e: dict, now: datetime, pmap: dict[str, str]) -> dict:
    """One SGO event -> the same shape as parse_propline, props keyed by gsis id (`pmap`: SGO player id
    -> gsis). Only quotes marked available; a quote SGO has not refreshed in SGO_MAX_AGE_S is "stale"."""
    props: dict = {}
    game: dict = {}
    fresh: dict = {}
    stale: set = set()
    live = bool((e.get("status") or {}).get("live"))
    sides: dict = {}
    for k, o in (e.get("odds") or {}).items():
        if o.get("periodID") != "game":
            continue
        for book, v in (o.get("byBookmaker") or {}).items():
            if not v.get("available"):
                continue
            upd = _parse_ts(v.get("lastUpdatedAt"))
            age = (now - upd).total_seconds() if upd else 9e9
            fresh[book] = min(fresh.get(book, 9e9), age)
            price = _sgo_dec(v.get("odds"))
            if price is None:
                continue
            stat, side, bt = o.get("statID"), o.get("sideID"), o.get("betTypeID")
            if stat in SGO_STATS and o.get("playerID") and bt == "ou":
                gs = pmap.get(o["playerID"])
                if not gs:
                    continue
                key = (gs, SGO_STATS[stat], book)
                if live and age > SGO_MAX_AGE_S:
                    stale.add(key)
                    continue
                sides.setdefault(key, {})[side] = (float(v["overUnder"]), price, age)
            elif stat == "points" and bt in ("ml", "sp", "ou") and o.get("statEntityID") in ("home", "away", "all"):
                mk = {"ml": "ml", "sp": "spread", "ou": "total"}[bt]
                if live and age > SGO_MAX_AGE_S:
                    stale.add(("game", mk, book))
                    continue
                sides.setdefault(("game", mk, book), {})[side] = (
                    float(v["spread"]) if bt == "sp" else float(v["overUnder"]) if bt == "ou" else None, price, age)
    for (who, mk, book), sd in sides.items():
        if who != "game":
            if "over" in sd and "under" in sd and sd["over"][0] == sd["under"][0]:
                props.setdefault((who, mk), {})[book] = {"line": sd["over"][0], "over": sd["over"][1],
                                                         "under": sd["under"][1], "age": max(sd["over"][2], sd["under"][2])}
            continue
        g = game.setdefault(book, {})
        if mk == "ml" and "home" in sd and "away" in sd:
            g.update(ml_home=sd["home"][1], ml_away=sd["away"][1], age=max(sd["home"][2], sd["away"][2]))
        elif mk == "spread" and "home" in sd and "away" in sd:
            g.update(spread=-sd["home"][0], spread_price_a=sd["home"][1], spread_price_b=sd["away"][1],
                     age=max(sd["home"][2], sd["away"][2]))
        elif mk == "total" and "over" in sd and "under" in sd and sd["over"][0] == sd["under"][0]:
            g.update(total=sd["over"][0], total_price_a=sd["over"][1], total_price_b=sd["under"][1],
                     age=max(sd["over"][2], sd["under"][2]))
    game = {b: g for b, g in game.items() if g}
    return {"props": props, "game": game, "at": now.isoformat(), "books": fresh, "stale": stale}


class SGOFeed(LiveFeed):
    """ESPN for the game + SportsGameOdds for prices (the trial week's stand-in for PropLine)."""

    def __init__(self, espn_event: str, sgo_event: str, home: str, away: str, e2g: dict[str, str],
                 pmap: dict[str, str], players: pl.DataFrame):
        super().__init__(espn_event, sgo_event, home, away, e2g)
        self.pmap, self.players, self.n_listed = pmap, players, -1
        sgo.watch(sgo_event)

    def odds(self, since: datetime | None = None, kinds: tuple = ("prop", "game")) -> dict | None:
        e = sgo.live_event(self.pid)
        if e is None:
            return None
        if len(e.get("players") or {}) != self.n_listed:   # books post props through the morning: re-match
            self.n_listed = len(e.get("players") or {})
            self.pmap = sgo_player_map(e, self.players)
        if (e.get("status") or {}).get("ended"):
            sgo.watch(self.pid, False)
        return parse_sgo(e, datetime.now(timezone.utc), self.pmap)


def sgo_player_map(e: dict, players: pl.DataFrame) -> dict[str, str]:
    """SGO player id -> gsis id for one event, by name within team (`players`: gsis_id, name, team)."""
    from pipeline.prop_players import norm
    abbr = {e["teams"][s]["teamID"]: TEAMS.get(e["teams"][s]["names"]["long"]) for s in ("home", "away")}
    by_name, by_last = {}, {}
    for r in players.iter_rows(named=True):
        k = norm(r["name"])
        by_name.setdefault((r["team"], k), set()).add(r["gsis_id"])
        parts = k.split()
        if parts:
            by_last.setdefault((r["team"], parts[-1], parts[0][:1]), set()).add(r["gsis_id"])
    out = {}
    for pid, p in (e.get("players") or {}).items():
        team = abbr.get(p.get("teamID"))
        k = norm(p.get("name"))
        hit = by_name.get((team, k))
        if not hit and k.split():
            hit = by_last.get((team, k.split()[-1], k.split()[0][:1]))
        if hit and len(hit) == 1:
            out[pid] = next(iter(hit))
    return out


class DemoFeed:
    """A finished game replayed from nflverse plays and archived in-game prices.
    `speed` = game seconds of wall time per real second."""

    def __init__(self, game_id: str, speed: float = 20.0):
        from live.replay import _banked, _pbp, _prop_grid
        from live.replay_games import _game_grid
        from pipeline.odds_history import HIST, fixture_index
        from props.lines import load as load_props
        self.gid, self.speed = game_id, speed
        p = _pbp([game_id])
        self.plays = p.filter(pl.col("ts").is_not_null())
        self.home, self.away = self.plays["home_team"][0], self.plays["away_team"][0]
        self.banked, self.team_so = _banked(self.plays)
        self.rows = self.plays.to_dicts()
        P = load_props([game_id], pregame_only=False)
        self.kickoff = P["kickoff"][0] if P.height else self.rows[0]["ts"]
        self.prop_rows = P.filter((pl.col("ts") >= pl.col("kickoff")) & pl.col("book").is_in(["draftkings", "fanduel"]))
        self.close_rows = P.filter(pl.col("ts") < pl.col("kickoff"))
        fx = dict(fixture_index().select("game_id", "fixture_id").iter_rows())
        path = HIST / f"{fx.get(game_id)}.parquet"
        self.game_rows = pl.read_parquet(path).filter(pl.col("ts") >= self.kickoff) if path.exists() else pl.DataFrame()
        self._pg, self._gg = _prop_grid, _game_grid
        self.t0 = time.monotonic()
        self.game_t0 = self.rows[0]["ts"]
        self.i = 0
        self.final = False

    def now_game(self) -> datetime:
        return self.game_t0 + timedelta(seconds=(time.monotonic() - self.t0) * self.speed)

    def state(self) -> dict:
        t = self.now_game()
        new = []
        while self.i < len(self.rows) and self.rows[self.i]["ts"] <= t:
            r = self.rows[self.i]
            new.append({"id": str(r["play_id"]), "type": r["play_type"], "text": f"{r['posteam'] or ''} {r['play_type'] or ''}",
                        "wallclock": r["ts"].isoformat()})
            self.i += 1
        k = max(self.i - 1, 0)
        nxt = self.rows[min(self.i, len(self.rows) - 1)]
        self.final = self.i >= len(self.rows)
        secs = float(nxt["game_seconds_remaining"] or 0) if not self.final else 0.0
        b = self.banked[k] if self.banked else {}
        return {"seconds_left": secs, "score": {self.home: nxt["total_home_score"], self.away: nxt["total_away_score"]},
                "possession": nxt["posteam"], "yards_to_endzone": nxt["yardline_100"], "period": None, "clock": None,
                "final": self.final, "live": not self.final, "detail": "Final" if self.final else f"{int(secs // 60)} min left",
                "new_plays": new, "banked": b, "team_so_far": self.team_so[k] if self.team_so else {}}

    def odds(self, since: datetime | None = None, kinds: tuple = ("prop", "game")) -> dict:
        t = self.now_game()
        g = self._pg(self.prop_rows, [t]).get(0, {})
        gg = self._gg(self.game_rows, [t]).get(0, {}) if self.game_rows.height else {}
        for v in g.values():
            for L in v.values():
                L["age"] = 5.0
        return {"props": g, "game": gg, "at": t.isoformat()}

    def closing(self) -> dict:
        from props.lines import main_lines
        out: dict = {}
        if self.close_rows.height:
            for x in main_lines(self.close_rows, pl.col("kickoff") - timedelta(minutes=5)).iter_rows(named=True):
                out.setdefault((x["gsis_id"], x["market"]), {})[x["book"]] = {"line": x["line"], "over": x["over"], "under": x["under"]}
        return out
