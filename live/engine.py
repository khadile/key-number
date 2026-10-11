"""The live engine: watch games, price every tracked prop and game market after each play,
flag the best few, and keep score. Runs on your laptop; the published board only shows
its record.

    python -m live.engine                          # every game in the next 48 hours (PropLine: as many as its budget covers)
    python -m live.engine --games 2026_05_TB_DAL   # specific games
    python -m live.engine --demo 2026_04_PIT_CLE   # replay a finished game: no PropLine requests
    python -m live.engine --lan                    # also serve the page to your phone on home Wi-Fi

Then open http://localhost:8770.

Paper mode (the default): every flag is auto-accepted for the model's record,
at the price it was flagged; the engine also checks whether that price was
still there 30 seconds later (how often it could really be bet). Accept records the
bets you actually place, at the price you got; that is your record. Every play
risks 1 unit. Nothing here places a bet.

Odds source (`--odds`): SportsGameOdds during its trial week (default from 2026-10-11:
one shared call every ~12 s covers every game, so the whole slate is watched), PropLine
otherwise (`--odds propline`).

Request budget (PropLine free plan, 1,000 a day reset at midnight UTC; the
client stops at 950): ESPN is free and polled every 7 seconds. PropLine is
polled for a game only after a new snap, at most every 20 seconds, with a
heartbeat every 2 minutes (5 at halftime): about 200 requests a game, so the
free plan covers about four games a day. Below 20% of the day's budget the gap
doubles.
"""
import argparse
import json
import socket
import threading
import time
import zlib
from collections import deque
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import polars as pl

from config import CLEAN, DATA, RAW
from live import decide, ledger, notify, rubric_checks
from live.feeds import MAX_AGE_S, DemoFeed, LiveFeed, SGOFeed, espn_to_gsis, sgo_player_map
from live.model import LiveModel, LiveParams
from pipeline import propline, sgo
from props.model import Simulator


class OddsSource:
    """Which odds API the engine prices from, and its request budget."""

    def __init__(self, kind: str = "sgo"):
        self.kind = kind
        self.name = "SportsGameOdds" if kind == "sgo" else "PropLine"
        self.budget = sgo.MAX_CALLS_PER_DAY if kind == "sgo" else propline.DAILY_BUDGET

    def used_today(self) -> int:
        return sgo.used_today() if self.kind == "sgo" else propline.used_today()

    def remaining_today(self) -> int:
        return max(0, self.budget - self.used_today())


SRC = OddsSource("propline")

PORT = 8770  # the board's local preview uses 8765
# Bump together with PAGE_VERSION in live/page.html. The engine serves page.html fresh from disk but
# loads its own code once, so a mismatch means "the page was updated; restart the engine".
ENGINE_VERSION = "2026-10-11.9"
ESPN_POLL = 7
ODDS_MIN_GAP = 20
ODDS_HEARTBEAT = 120
ODDS_HALFTIME = 300
CALLS_PER_GAME = 220    # PropLine, polling every play (the old schedule)
# Lean schedule (PropLine, 2026-10-11): price checks only at decision points (halftime, after a score),
# for live cards and their 30-second and 5-minute re-checks, plus a slow heartbeat for the log. About
# 50 calls a game, so a 13-game Sunday fits PropLine's 950 a day.
LEAN_CALLS_PER_GAME = 60
LEAN_HEARTBEAT = 600    # seconds between log-only price checks outside decision points
CARD_POLL = 30          # while a card is live or awaiting its re-check
WINDOW_RETRY_S = 30     # a decision point whose prices were all too old tries again this soon...
WINDOW_RETRIES = 8      # ...this many times, until play resumes
# Halftime props roll (user, 2026-10-11): a check every PERSIST (90 s) through the whole break; the first play
# that holds on two checks in a row is flagged; when nothing holds, what passes now becomes the new first
# check. Play resuming ends the window: no flag, a pass. Capped so one break cannot drain PropLine's budget.
ROLLING_MAX_CHECKS = 20
# ESPN posts a touchdown at +6 and folds the try into the same play about a minute later: an after-score window
# waits for it (the score changing again, or the next play), at most this long, so the model prices the settled score.
TRY_WAIT_S = 150
# How far Pinnacle's nearest spread or total number may sit from the one DK/FD hang (user, 2026-10-11: judge the
# book's line on its merits, don't require the same number). Pinnacle's price there is moved by the model's chance
# of the result landing between the two numbers (_pin_fair).
LINE_SLACK = {"spread": 2.0, "total": 3.0}
FILL_AFTER = 25
FILL_GIVEUP = 150       # no fresh quote for the market this long after a flag: its fill is "unknown"
CLV_AFTER = 300
ACCEPT_GRACE = 600      # an Accept still counts this long after a card expires
QUIET_LOG_S = 60        # log an "all quiet" line when nothing else was logged for this long
EVENT_KEEP = 6000       # lines of engine log kept for the live page (a full slate, every game)
LOG_DIR = DATA / "live"
STATUS = LOG_DIR / "status.json"
PAGE = Path(__file__).with_name("page.html")
SCRIM = {"Pass", "Rush", "Pass Reception", "Pass Incompletion", "Sack", "Passing Touchdown", "Rushing Touchdown",
         "Interception", "Fumble Recovery (Own)", "Fumble Recovery (Opponent)", "pass", "run"}
STAT_KEY = {"pass_yds": "pass_yds", "rush_yds": "rush_yds", "rec_yds": "rec_yds", "receptions": "rec"}
BOOK_SHORT = {"draftkings": "DK", "fanduel": "FD", "pinnacle": "Pinnacle"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Engine:
    def __init__(self, mode: str, url: str):
        self.mode, self.url = mode, url
        self.cards: dict[str, dict] = {}
        self.games: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.events: deque = deque(maxlen=EVENT_KEEP)   # the engine log shown on the live page
        self.seq = 0
        self.started = _now()
        self.runners: list = []
        self.last_error_at: str | None = None
        self.ready = False   # projections loaded and game watchers started

    def log(self, kind: str, msg: str, game: str | None = None) -> None:
        """One line in the engine log. kind: info, play, odds, price, flag, card, fill, grade, warn, error."""
        with self.lock:
            self.seq += 1
            self.events.append({"seq": self.seq, "at": _now().isoformat(), "kind": kind, "game": game, "msg": msg})
            if kind == "error":
                self.last_error_at = _now().isoformat()

    def add_card(self, card: dict) -> None:
        with self.lock:
            self.cards[card["id"]] = card

    def update_card(self, cid: str, **kw) -> None:
        with self.lock:
            if cid in self.cards:
                self.cards[cid].update(kw)

    def accept(self, cid: str, american: int | None = None) -> dict:
        with self.lock:
            c = self.cards.get(cid)
        if not c:
            return {"ok": False, "error": "unknown play"}
        if c.get("accepted"):
            return {"ok": True, "already": True}
        if c["status"] != "active" and (_now() - datetime.fromisoformat(c["expired_at"])).total_seconds() > ACCEPT_GRACE:
            return {"ok": False, "error": "this play expired more than 10 minutes ago"}
        am = american if american is not None else c["american"]
        dec = 1 + am / 100 if am > 0 else 1 + 100 / -am
        ledger.append(ledger.ACCEPTS, {"id": cid, "american": am, "dec": dec, "line": c["line"], "book": c["book"],
                                       "at": _now().isoformat()})
        self.update_card(cid, accepted={"american": am, "at": _now().isoformat()})
        return {"ok": True}

    def pass_(self, cid: str) -> dict:
        ledger.append(ledger.EVENTS, {"id": cid, "passed": True})
        self.update_card(cid, passed=True)
        return {"ok": True}

    def snapshot(self, since: int = 0) -> dict:
        with self.lock:
            watched = set(self.games)
            cards = sorted([c for c in self.cards.values() if c["game_id"] in watched or c["status"] == "active"],
                           key=lambda c: c["at"], reverse=True)[:600]
            games = [{**g, "alive": any(r.gid == g["game_id"] and r.is_alive() for r in self.runners)} for g in self.games.values()]
            events = [e for e in self.events if e["seq"] > since][-(2500 if since == 0 else 600):]
            seq = self.seq
        rec = ledger.records()
        mine = [r for r in rec["flags"] if r["game_id"] in watched and r.get("units") is not None]
        by_game = {g: ledger._tally([r for r in mine if r["game_id"] == g]) for g in watched}
        return {"mode": self.mode, "now": _now().isoformat(), "url": self.url, "games": games, "cards": cards,
                "today": ledger._tally(mine), "by_game": by_game,
                "record": {"paper": rec["paper"], "you": rec["you"], "missed": rec["missed"], "fill_checked": rec["fill_checked"]},
                "budget": {"used": SRC.used_today(), "budget": SRC.budget, "source": SRC.name},
                "alerts": notify.enabled(), "rules": {"min_ev": decide.MIN_EV, "max_gap": decide.MAX_GAP,
                                                     "windows": [decide.WINDOW_LABEL[w] for w in decide.PROP_WINDOWS],
                                                     "prop_per_game": decide.PROP_PER_GAME, "game_per_game": decide.GAME_PER_GAME,
                                                     "per_game": decide.MAX_PER_GAME, "ttl": decide.CARD_TTL_S},
                "engine": {"started": self.started.isoformat(), "seq": seq, "last_error_at": self.last_error_at,
                           "ready": self.ready, "version": ENGINE_VERSION},
                "events": events}

    def write_status(self) -> None:
        """A small heartbeat file the board build reads: when the engine last ran and what it was watching."""
        snap = self.snapshot()
        out = {"at": snap["now"], "mode": self.mode, "started": snap["engine"]["started"],
               "games": [{k: g.get(k) for k in ("game_id", "home", "away", "detail", "score", "flags", "calibrated", "phase", "alive")}
                         for g in snap["games"]],
               "budget": snap["budget"], "last_error_at": self.last_error_at,
               "recent": [{k: e[k] for k in ("at", "kind", "msg")} for e in list(self.events)[-12:]]}
        tmp = STATUS.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, default=str), encoding="utf-8")
        tmp.replace(STATUS)


class GameRunner(threading.Thread):
    def __init__(self, engine: Engine, game: dict, feed, priors: pl.DataFrame, demo: bool = False):
        super().__init__(daemon=True)
        self.e, self.g, self.feed, self.demo = engine, game, feed, demo
        self.gid, self.home, self.away = game["game_id"], game["home"], game["away"]
        lp = LiveParams.load()
        self.k_game = lp.k_game
        from props.backtest import load as load_props
        _, disp = load_props()
        self.model = LiveModel(priors, lp, Simulator(disp, seed=zlib.crc32(self.gid.encode())), size=2000)
        self.info = {r["gsis_id"]: {"name": r["name"], "team": r["team"], "position": r["position"]} for r in priors.iter_rows(named=True)}
        self.prior_share = {r["gsis_id"]: (r["tshare"], r["rshare"]) for r in priors.iter_rows(named=True)}
        self.flags: list[dict] = []
        self.calibrated = False
        self.cal_tried = False
        self.last_odds = 0.0
        self.last_said = time.monotonic()
        self.budget_warned = False
        self.window: dict | None = None      # the open decision point (decide.py), if any
        self.shadowed: set[str] = {s["kind"] for s in ledger.read(ledger.SHADOW) if s["game_id"] == game["game_id"]}
        self.windows_done: set[str] = set()
        self.last_wall: datetime | None = None   # wallclock of the newest play seen: when the game last changed
        self.dead: dict | None = None   # a dead ball: {since: the play that froze the game, kinds: markets it applies to}
        self.prev_secs: float | None = None
        self.prev_points: float | None = None
        # what the rubric's play checks need (live/rubric_checks.py): yards per play from the box
        # score's change between polls, the last touch of each player, and who was hurt
        self.prev_banked: dict | None = None
        self.play_yards: dict = {}
        self.last_touch: dict = {}
        self.hurt: dict = {}
        self.tick = 0
        self.abbrev: dict = {}
        for pid, inf in self.info.items():
            for a in rubric_checks.short_names(inf["name"]):
                self.abbrev.setdefault(a, []).append(pid)
        # one folder per game (not per day), so a restart after midnight UTC finds the saved calibration
        self.log = LOG_DIR / f"{self.gid}{'_demo' if demo else ''}"
        self.log.mkdir(parents=True, exist_ok=True)
        self.cal_path = self.log / "calibration.json"
        kick = game.get("kickoff")
        self.e.games[self.gid] = {"game_id": self.gid, "home": self.home, "away": self.away, "detail": "Waiting",
                                  "kickoff": kick.isoformat() if hasattr(kick, "isoformat") else kick,
                                  "score": {}, "flags": 0, "calibrated": False, "phase": "pre", "health": {}}
        if not demo:
            self._restore()
        self.say("info", f"Watching {self.away} at {self.home}: {len(self.info)} tracked players"
                         + (", calibrated" if self.calibrated else ""))

    # ---------------------------------------------------------------- plumbing
    def _logline(self, name: str, row: dict) -> None:
        with (self.log / f"{name}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({**row, "at": _now().isoformat()}, default=str) + "\n")

    def _status(self, **kw) -> None:
        with self.e.lock:
            self.e.games[self.gid].update(kw)

    def say(self, kind: str, msg: str) -> None:
        self.e.log(kind, msg, self.gid)
        self.last_said = time.monotonic()

    def _group(self, f: dict) -> str:
        """Exposure group of a flag, as decide.prop_candidates assigns it."""
        if f["kind"] != "prop":
            return f"game:{self.gid}:total" if f["market"] == "total" else f"game:{self.gid}"
        who = self.info.get(f["player"], {})
        if f["market"] in ("pass_yds", "rec_yds", "receptions") and who.get("position") != "RB":
            return f"pass:{who.get('team')}:{f['side']}"
        return f["player"]

    def _restore(self) -> None:
        """After a restart mid-game, keep the flags already made (the per-game cap and no
        re-alerts still hold) and the kickoff calibration, if it was saved."""
        ev: dict[str, dict] = {}
        for x in ledger.read(ledger.EVENTS):
            ev.setdefault(x["id"], {}).update({k: v for k, v in x.items() if k not in ("id", "logged_at")})
        flags = [f for f in ledger.read(ledger.FLAGS) if f["game_id"] == self.gid]
        for f in flags:
            e = ev.get(f["id"], {})
            fill = e.get("fill") or "unchecked"   # a fill not checked before the restart cannot be checked fairly now
            card = {**f, "group": f.get("group") or self._group(f), "push": 0.0, "status": "expired",
                    "expired_at": f["at"], "expires_at": f["at"], "expired_why": e.get("expired", "engine restarted"),
                    "fill": fill, "fill_american": decide.american(e["fill_dec"]) if e.get("fill_dec") else None,
                    "fill_line": e.get("fill_line"), "clv_units": e.get("clv_units"), "restored": True}
            self.flags.append(card)
            self.e.add_card(card)
        if self.cal_path.exists():
            saved = json.loads(self.cal_path.read_text(encoding="utf-8"))
            self.model.offset = {tuple(k.split("|")): v for k, v in saved["offset"].items()}
            self.model.rem0 = {tuple(k.split("|")): v for k, v in saved["rem0"].items()}
            self.calibrated = self.cal_tried = bool(self.model.offset)
            self._status(calibrated=self.calibrated)
        if flags:
            self._status(flags=len(flags))
            self.say("info", f"Restored {len(flags)} earlier flag(s) for this game from the ledger")

    def calibrate(self, close: dict, source: str) -> None:
        self.cal_tried = True
        if not close:
            self.say("warn", f"No closing prop lines from {source}; the model runs uncalibrated")
            return
        self.model.calibrate(close)
        self.calibrated = bool(self.model.offset)
        self._status(calibrated=self.calibrated)
        if not self.demo:
            self.cal_path.write_text(json.dumps({"offset": {f"{k[0]}|{k[1]}": v for k, v in self.model.offset.items()},
                                                 "rem0": {f"{k[0]}|{k[1]}": v for k, v in self.model.rem0.items()}}), encoding="utf-8")
        self.say("info", f"Calibrated to {len(self.model.offset)} closing prop lines ({source})")

    # ---------------------------------------------------------------- main loop
    def run(self) -> None:
        if self.demo:
            self.calibrate(self.feed.closing(), "archived closing lines")
        poll = 1.0 if self.demo else ESPN_POLL
        while True:
            try:
                if self.step(poll):
                    return
            except Exception as e:  # noqa: BLE001 - one bad update must not end a game's watch
                import traceback
                self._logline("errors", {"error": repr(e), "trace": traceback.format_exc()[-1500:]})
                self._status(health={**self.e.games[self.gid]["health"], "error": repr(e)[:200]})
                self.say("error", f"{type(e).__name__}: {str(e)[:160]} (still watching)")
                time.sleep(poll * 2)

    def score_line(self, st: dict) -> str:
        return f"{self.away} {int(st['score'].get(self.away, 0))}, {self.home} {int(st['score'].get(self.home, 0))}"

    def step(self, poll: float) -> bool:
        """One pass of the loop; True when the game is over and graded."""
        st = self.feed.state()
        if st is None:
            self._status(health={**self.e.games[self.gid]["health"], "espn": "error"})
            self.say("warn", "ESPN did not answer; trying again")
            time.sleep(poll * 2)
            return False
        delay = None
        new = st["new_plays"]
        if 0 < len(new) <= 4 and new[-1].get("wallclock") and not self.demo:   # a catch-up batch says nothing about lag
            wc = datetime.fromisoformat(new[-1]["wallclock"].replace("Z", "+00:00"))
            delay = round((_now() - wc).total_seconds(), 1)
            self._logline("plays", {"n": len(new), "delay_s": delay, "last": new[-1]})
        phase = "final" if st["final"] else "live" if st["live"] else "pre"
        self._status(detail=st["detail"], score=st["score"], seconds_left=st["seconds_left"], phase=phase,
                     health={"espn_at": _now().isoformat(), "feed_delay_s": delay,
                             "odds_at": self.e.games[self.gid]["health"].get("odds_at")})
        if len(new) > 4:
            self.say("play", f"Caught up on {len(new)} plays · {st['detail']} · {self.score_line(st)}")
        else:
            for p in new:
                txt = " ".join(str(p.get("text") or p.get("type") or "").split())
                if txt:
                    lag = f"  [feed {delay:.0f}s behind]" if p is new[-1] and delay is not None else ""
                    self.say("play", (txt if len(txt) <= 110 else txt[:107] + "...") + lag)
        if st["final"]:
            self.say("info", f"Final: {self.score_line(st)}. Grading.")
            self.grade(st)
            if not self.demo:
                ok, why = sync_ledger(f"Live: {self.away} at {self.home} final, graded")
                self.say("info" if ok else "warn", why)
            self._status(detail="Final")
            return True
        if not st["live"]:
            kick = self.g.get("kickoff")
            if not self.calibrated and not self.cal_tried and kick and (kick - _now()).total_seconds() <= 600:
                od = self.feed.odds()     # one pregame request: the closing lines this game starts from
                if od:
                    self.calibrate(od["props"], f"{SRC.name}, before kickoff")
            if time.monotonic() - self.last_said >= 300:
                self.say("info", f"Waiting for kickoff · {st['detail']}")
            time.sleep(30)
            return False
        if not self.calibrated and not self.cal_tried:
            # started after kickoff with nothing saved: start from the archived closing lines
            self.say("info", "Joined after kickoff: fetching this game's closing prop lines from the OddsPapi archive")
            self.calibrate(*closing_from_archive(self.gid))
        self._track(st, new)
        self._watch_breaks(st, new)
        now = time.monotonic()
        snap = any(p.get("type") in SCRIM for p in new)
        half = "half" in (st.get("detail") or "").lower()
        remaining = SRC.remaining_today()
        gap = ODDS_MIN_GAP * (2 if remaining < 0.2 * SRC.budget else 1)
        heartbeat = ODDS_HALFTIME if half else ODDS_HEARTBEAT
        check = self.window is not None and now >= self.window["next_check"]   # a decision point's price check
        if SRC.kind == "propline" and not self.demo:
            # lean: decision points, live cards and their re-checks, and a slow heartbeat for the log
            t_now = _now()
            cards_due = False
            for c in self.e.cards.values():
                if c["game_id"] != self.gid or c.get("restored"):
                    continue
                age = (t_now - datetime.fromisoformat(c["at"])).total_seconds()
                if (c["status"] == "active" or (c.get("fill") == "pending" and age >= FILL_AFTER)
                        or (c.get("clv_units") is None and CLV_AFTER <= age < CLV_AFTER + 180)):
                    cards_due = True
                    break
            lean_beat = LEAN_HEARTBEAT * (3 if remaining < 0.15 * SRC.budget else 1)
            due = check or (cards_due and now - self.last_odds >= CARD_POLL) or now - self.last_odds >= lean_beat
        else:
            due = self.demo or check or (snap and now - self.last_odds >= gap) or now - self.last_odds >= heartbeat
        if due and not self.demo and remaining <= 0:
            if not self.budget_warned:
                self.say("warn", f"{SRC.name}'s daily budget is used up; still following the game, no prices until midnight UTC")
                self.budget_warned = True
        elif due:
            # in a dead ball, prices count by when they were posted (after the play that froze the game), not by age
            od = self.feed.odds(since=self.dead["since"], kinds=self.dead["kinds"]) if self.dead else self.feed.odds()
            self.last_odds = now
            if od:
                src = "archived prices" if self.demo else f"{SRC.name} {SRC.used_today()}/{SRC.budget} today"
                self.say("odds", _odds_line(od, src))
                self._status(health={**self.e.games[self.gid]["health"], "odds_at": _now().isoformat()})
                self.process(st, od)
            else:
                self.say("warn", f"{SRC.name} did not answer; the next play tries again")
        if time.monotonic() - self.last_said >= QUIET_LOG_S:
            self.say("info", f"ESPN ok · {st['detail']} · {self.score_line(st)} · no new plays")
        time.sleep(poll)
        return False

    # ---------------------------------------------------------------- each odds update
    def _price_now(self, card: dict, od: dict) -> dict | str | None:
        """The card's market now: {line, dec}; None if the book no longer lists it; "stale" if the
        book's only quote for it was too old to judge (PropLine refreshes each book on its own cycle)."""
        p = self._price_fresh(card, od)
        if p is None and (card["player"] if card["kind"] == "prop" else "game", card["market"], card["book"]) in od.get("stale", ()):
            return "stale"
        return p

    def _price_fresh(self, card: dict, od: dict) -> dict | None:
        if card["kind"] == "prop":
            L = od["props"].get((card["player"], card["market"]), {}).get(card["book"])
            if not L:
                return None
            return {"line": L["line"], "dec": L["over"] if card["side"] == "over" else L["under"]}
        L = od["game"].get(card["book"], {})
        if card["market"] == "ml" and "ml_home" in L:
            return {"line": None, "dec": L["ml_home"] if card["side"] == "home" else L["ml_away"]}
        if card["market"] == "spread" and "spread" in L:
            return {"line": -L["spread"] if card["side"] == "home" else L["spread"],
                    "dec": L["spread_price_a"] if card["side"] == "home" else L["spread_price_b"]}
        if card["market"] == "total" and "total" in L:
            return {"line": L["total"], "dec": L["total_price_a"] if card["side"] == "over" else L["total_price_b"]}
        return None

    def process(self, st: dict, od: dict) -> None:
        t = _now()
        self._logline("odds", {"props": len(od["props"]), "game_books": list(od["game"])})
        for card in [c for c in self.e.cards.values() if c["game_id"] == self.gid]:
            age = (t - datetime.fromisoformat(card["at"])).total_seconds()
            now_p = self._price_now(card, od)
            stale = now_p == "stale"
            if stale:
                now_p = None
            # expiry: the deal is gone, or its time is up
            if card["status"] == "active":
                why = None
                if now_p is None and not stale:
                    why = "pulled"
                elif now_p is None:
                    why = "time" if age > decide.CARD_TTL_S else None
                elif now_p["line"] != card["line"]:
                    why = f"line moved to {now_p['line']:g}"
                elif card["p"] * (now_p["dec"] - 1) - (1 - card["p"] - card.get("push", 0)) < decide.MIN_EV / 2:
                    why = f"price now {decide.american(now_p['dec']):+d}"
                elif t >= datetime.fromisoformat(card["expires_at"]):
                    why = "time"
                if why:
                    self.e.update_card(card["id"], status="expired", expired_at=t.isoformat(), expired_why=why)
                    ledger.append(ledger.EVENTS, {"id": card["id"], "expired": why})
                    self.say("card", f"Card closed: {card['label']} ({why})")
            if card.get("restored") and (card.get("clv_units") is not None or age > CLV_AFTER + 180):
                continue   # made before a restart: its fill cannot be judged now, its line value only near the 5-minute mark
            # paper fill: still bettable 25+ seconds after the flag?
            if card.get("fill") == "pending" and age >= FILL_AFTER and stale and age < FILL_GIVEUP:
                pass   # the book's quote is too old to judge; the next update tries again
            elif card.get("fill") == "pending" and age >= FILL_AFTER and stale:
                self.e.update_card(card["id"], fill="unknown")
                ledger.append(ledger.EVENTS, {"id": card["id"], "fill": "unknown", "fill_dec": None, "fill_line": None})
                self.say("fill", f"Paper bet: {card['label']} unknown (no fresh quote within {FILL_GIVEUP}s; not counted)")
            elif card.get("fill") == "pending" and age >= FILL_AFTER:
                fill, fdec, fline = "missed", None, None
                if now_p:
                    better = (now_p["line"] is not None and card["line"] is not None and
                              (now_p["line"] < card["line"] if card["side"] == "over" else now_p["line"] > card["line"]))
                    if now_p["line"] == card["line"] and (now_p["dec"] >= card["dec"] - 1e-9 or
                                                         card["p"] * (now_p["dec"] - 1) - (1 - card["p"]) > 0):
                        fill, fdec, fline = "filled", min(now_p["dec"], card["dec"]), card["line"]
                    elif better and card["kind"] == "prop":
                        fill, fdec, fline = "filled", now_p["dec"], now_p["line"]
                self.e.update_card(card["id"], fill=fill, fill_american=decide.american(fdec) if fdec else None, fill_line=fline)
                ledger.append(ledger.EVENTS, {"id": card["id"], "fill": fill, "fill_dec": fdec, "fill_line": fline})
                self.say("fill", f"Paper bet: {card['label']} " + (f"filled at {decide.american(fdec):+d}" + (f" on {fline:g}" if fline != card["line"] else "")
                                                                  if fill == "filled" else "missed (the price was gone 30 seconds later)"))
            # line value five minutes on
            if card.get("clv_units") is None and age >= CLV_AFTER and now_p and now_p["line"] is not None and card["line"] is not None:
                cu = now_p["line"] - card["line"] if card["side"] == "over" else card["line"] - now_p["line"]
                if card["market"] == "spread":
                    cu = now_p["line"] - card["line"]
                self.e.update_card(card["id"], clv_units=cu)
                ledger.append(ledger.EVENTS, {"id": card["id"], "clv_units": cu})
                self.say("fill", f"Line value: {card['label']} {cu:+g} after five minutes")
        # Price everything every update, so the log shows what the engine sees even when it may not flag.
        state = {"seconds_left": st["seconds_left"], "score": st["score"], "possession": st["possession"],
                 "yards_to_endzone": st["yards_to_endzone"]}
        proj = self.model.project(state, st["banked"], st["team_so_far"])
        lead = int(st["score"].get(self.home, 0) - st["score"].get(self.away, 0))
        cands = decide.prop_candidates(self.gid, proj, od["props"], self.info)
        pin = od["game"].get("pinnacle", {})
        g = proj["game"]
        margin = g["margin"] if g["teams"][0] == self.home else -g["margin"]
        for c in decide.game_candidates(self.gid, g, od["game"], self.home, self.away):
            r = _pin_fair(pin, c, margin, g["total"])   # judged against Pinnacle's live no-vig number at the book's line
            if r is None:
                continue
            c.fair, c.ref = r
            cands.append(c)

        def quote(c):
            if c.kind == "prop":
                return od["props"].get((c.player, c.market), {}).get(c.book, {})
            g = od["game"].get(c.book, {})
            return {"age": g.get(f"age_{c.market}", g.get("age")), "by_state": g.get(f"by_state_{c.market}", False)}
        passing, fails = [], {"prop": {}, "game": {}}
        for c in cands:
            q_ = quote(c)
            why = decide.gate(c, lead, None if q_.get("by_state") else q_.get("age"))
            if why is None and c.kind == "prop":
                why = self._play_checks(c, st)
            if why is None:
                passing.append(c)
            else:
                fails[c.kind][why] = fails[c.kind].get(why, 0) + 1
        best = max(passing, key=lambda c: c.ev) if passing else None
        seen = (f"checked {len(cands)} prices, {len(passing)} pass the rubric"
                + (f"; best {best.label()} {decide.american(best.dec):+d} {BOOK_SHORT.get(best.book, best.book)}, EV {best.ev:+.1%}" if best else ""))
        W = self.window
        if W is None:   # between decision points: price everything for the log, flag nothing
            self.say("price", f"{seen} · {self._watching()}")
            return
        if W.get("await_try"):
            self.say("price", f"{seen} · {W['label']}: waiting for the try before deciding")
            return
        # A window decides one kind or both (halftime: the prop and the game market; after a score: the game market),
        # each on its own clock. A play is flagged when it holds (same book, side and line, still passing) on two
        # checks W["persist"] apart. A rolling window (halftime) starts over with whatever passes now when nothing
        # holds; an after-score window ends after its pair. Play resuming ends any window (_watch_breaks).
        key = lambda c: (c.key, c.side, c.book, c.line)
        speed = self.feed.speed if self.demo else 1.0
        now_m = time.monotonic()
        if W["checks"] >= ROLLING_MAX_CHECKS:
            self.say("price", f"{W['label']}: {ROLLING_MAX_CHECKS} checks and nothing held up on two in a row; pass")
            self._close_window()
            return
        after = "the end of the half" if W["rolling"] else "the score"
        notes = []
        for kind in list(W["kinds"]):
            if now_m < W["due"][kind]:
                continue
            what = "props" if kind == "prop" else "game markets"
            mine = [c for c in passing if c.kind == kind]
            fk = fails[kind]
            priced = sum(c.kind == kind for c in cands)
            if priced == 0 or (not mine and set(fk) <= {"quote too old"}):
                # nothing current enough to judge is not the same as nothing worth betting: wait for the books
                W["retries"][kind] = W["retries"].get(kind, 0) + 1
                if not W["rolling"] and W["retries"][kind] > WINDOW_RETRIES:
                    W["kinds"].remove(kind)
                    notes.append(f"{what}: no prices posted after {after} in {WINDOW_RETRIES} tries; no flag")
                    continue
                # a rolling window retries quickly three times, then at its own pace, so one kind that never gets
                # a price cannot spend the break's ROLLING_MAX_CHECKS before play resumes
                wait = WINDOW_RETRY_S / speed if not W["rolling"] or W["retries"][kind] <= 3 else W["persist"]
                W["due"][kind] = now_m + wait
                notes.append(f"{what}: {self._why_unpriced(kind, od, after)}, again in {wait * speed:.0f}s"
                             + ("" if W["rolling"] else f" ({W['retries'][kind]}/{WINDOW_RETRIES})"))
                continue
            W["priced"] = True
            top = max(mine, key=lambda c: c.ev) if mine else None
            best = (f" (best {top.label()} {decide.american(top.dec):+d} {BOOK_SHORT.get(top.book, top.book)}, EV {top.ev:+.1%})"
                    if top else "")
            why_not = (f" ({', '.join(f'{n} {w}' for w, n in sorted(fk.items(), key=lambda x: -x[1])[:3])})" if fk else "")
            first = W["first"][kind]
            if first is None:
                if not mine and not W["rolling"]:
                    W["kinds"].remove(kind)
                    notes.append(f"{what}: none pass the rubric{why_not}; no flag")
                    continue
                W["first"][kind] = {key(c) for c in mine} or None
                W["due"][kind] = now_m + W["persist"]
                notes.append(f"{what}: {len(mine)} pass{best}, confirming in {W['persist']:.0f}s" if mine
                             else f"{what}: none pass the rubric{why_not}, again in {W['persist']:.0f}s")
                continue
            held = [c for c in mine if key(c) in first]
            new = decide.select(held, self.flags)
            self._shadow(held, new, W, st, kind)
            if new:
                for c in new:
                    q_ = quote(c)
                    c.reason = (self.reason(c, proj, st) + f"; held on two price checks {W['persist']:.0f}s apart"
                                + (f"; its price was posted after {after}" if q_.get("by_state") else ""))
                    self.flag(c, st, q_.get("age"), W["label"])
                W["kinds"].remove(kind)
                notes.append(f"{what}: flagged {', '.join(c.label() for c in new)}")
                continue
            n_prop, n_game = self._counts()
            capped = (n_prop >= decide.PROP_PER_GAME) if kind == "prop" else (n_game >= decide.GAME_PER_GAME)
            if held and capped:
                W["kinds"].remove(kind)
                notes.append(f"{what}: {len(held)} held up, but this game's play is already used")
                continue
            if not W["rolling"]:
                W["kinds"].remove(kind)
                notes.append(f"{what}: nothing held up ({len(held)} of {len(first)} still pass"
                             + (", but they repeat an exposure already taken" if held else "") + "); no flag")
                continue
            W["first"][kind] = {key(c) for c in mine} or None
            W["due"][kind] = now_m + W["persist"]
            notes.append(f"{what}: nothing held from the last check ({len(held)} of {len(first)} still pass"
                         + (", but they repeat an exposure already taken" if held else "") + ")"
                         + (f"; {len(mine)} pass now{best}, confirming in {W['persist']:.0f}s" if mine
                            else f"; none pass now, again in {W['persist']:.0f}s"))
        if notes:
            W["checks"] += 1
            self.say("price", f"{W['label']}, check {W['checks']}: " + " · ".join(notes))
        if not W["kinds"]:
            self._close_window()
            return
        W["next_check"] = min(W["due"][k] for k in W["kinds"])

    def _track(self, st: dict, new: list) -> None:
        """Per-play pieces for the rubric's last two checks. Yards gained on each play come from the box
        score's change between polls (every few seconds, so one play at a time); the first poll after a
        start is a running total, not a play, so it only sets the baseline."""
        self.tick += 1
        if self.prev_banked is not None:
            secs = st["seconds_left"]
            for pid, b in st["banked"].items():
                old = self.prev_banked.get(pid, {})
                for m in rubric_checks.YARD_MARKETS:
                    d = float(b.get(m, 0) or 0) - float(old.get(m, 0) or 0)
                    if d:
                        self.play_yards.setdefault((pid, m), []).append((secs, d))
                if any((b.get(k, 0) or 0) > (old.get(k, 0) or 0) for k in ("tgt", "car", "att")):
                    self.last_touch[pid] = self.tick
        self.prev_banked = {pid: dict(b) for pid, b in st["banked"].items()}
        for p in new:
            for nm in rubric_checks.injured_in(" ".join(str(p.get("text") or "").split())):
                for pid in self.abbrev.get(nm, ()):
                    self.hurt[pid] = self.tick
                    self.say("info", f"Injury: {self.info[pid]['name']} was hurt on a play; no plays on him until he touches the ball again")

    def _play_checks(self, c, st: dict) -> str | None:
        """The rubric's checks that need the plays: still on the field, and not one big play."""
        if c.player in self.hurt and self.last_touch.get(c.player, -1) <= self.hurt[c.player]:
            return "player hurt and not back"
        if c.market in rubric_checks.YARD_MARKETS:
            banked = float(st["banked"].get(c.player, {}).get(STAT_KEY[c.market], 0) or 0)
            if rubric_checks.big_play(self.play_yards.get((c.player, c.market), []), banked, st["seconds_left"]):
                return "numbers lean on one big play"
        return None

    def _shadow(self, held: list, new: list, W: dict, st: dict, kind: str) -> None:
        """Each game's second-best play of each kind: it passed the whole rubric, both checks and the
        exposure rules, and only the one-a-game cap kept it off. Logged and graded off the record
        (ledger/live_shadow.jsonl), to learn whether a second slot would earn its keep."""
        if kind in self.shadowed:
            return
        taken = self.flags + [{"key": c.key, "player": c.player, "group": c.group, "kind": c.kind} for c in new]
        rest = decide.select([c for c in held if c not in new], taken, caps={kind: 1})
        if not rest:
            return
        c = rest[0]
        t = _now()
        row = {"id": f"{self.gid}:shadow:{c.key}:{int(t.timestamp())}", "game_id": self.gid, "kind": c.kind, "key": c.key,
               "player": c.player, "label": c.label(), "market": c.market, "side": c.side, "line": c.line, "book": c.book,
               "dec": round(c.dec, 4), "american": decide.american(c.dec), "p": round(c.p, 4), "fair": round(c.fair, 4),
               "ev": round(c.ev, 4), "at": t.isoformat(), "seconds_left": st["seconds_left"], "window": W["label"],
               "score": dict(st["score"]), "mode": self.e.mode}
        ledger.append(ledger.SHADOW, row)
        self.shadowed.add(kind)
        self.say("shadow", f"Shadow pick (off the record): {row['label']} {row['american']:+d} {BOOK_SHORT.get(c.book, c.book)} · "
                           f"model {c.p:.0%} vs {c.fair:.0%} no-vig · EV {c.ev:+.1%} · the game's next-best "
                           f"{'prop' if kind == 'prop' else 'game-market'} play, kept off by the one-a-game cap")

    @staticmethod
    def _why_unpriced(kind: str, od: dict, after: str) -> str:
        """Why a decision point priced nothing of this kind: no current quotes yet, or nothing to compare."""
        if kind == "prop":
            return f"no prices posted after {after} yet"
        have = set(od["game"])
        ours = have & {"draftkings", "fanduel"}
        if not ours and "pinnacle" not in have:
            return f"no prices posted after {after} yet"
        if "pinnacle" not in have:
            return f"waiting for Pinnacle to post after {after}"
        if not ours:
            return f"waiting for DK/FD to post after {after}"
        return (f"no Pinnacle number near DK/FD's (within {LINE_SLACK['spread']:g} on spreads, "
                f"{LINE_SLACK['total']:g} on totals) and no moneyline")

    def _close_window(self) -> None:
        if self.window and self.window["rolling"]:
            self.windows_done.add(self.window["name"])
        self.window = None

    def _counts(self) -> tuple[int, int]:
        return sum(f["kind"] == "prop" for f in self.flags), sum(f["kind"] == "game" for f in self.flags)

    def _watching(self) -> str:
        n_prop, n_game = self._counts()
        todo = [decide.WINDOW_LABEL[w].lower() for w in decide.PROP_WINDOWS if w not in self.windows_done]
        props = (f"props decide at {' and '.join(todo)}" if todo and n_prop < decide.PROP_PER_GAME
                 else "props: done for this game")
        games = ("game markets after the next score" + (" and at halftime" if "half" not in self.windows_done else "")
                 if n_game < decide.GAME_PER_GAME else "game markets: done for this game")
        return f"watching, no flags between decision points ({props}; {games})"

    def _watch_breaks(self, st: dict, new: list) -> None:
        """Open a decision point: halftime, for the prop and the game market (each rolls through the break); a score,
        for the game market (one pair of checks). A score or the end of the half also starts a dead ball (self.dead),
        which lasts until the next snap: quotes posted after the play that froze the game count as current, whatever
        their age, and quotes from before it do not (live/feeds.parse_propline)."""
        now = time.monotonic()
        speed = self.feed.speed if self.demo else 1.0
        secs, pts = st["seconds_left"], sum(st["score"].values())
        # play resumed: a scrimmage play that did not score (a scoring play is what froze the game)
        snap = any(p.get("type") in SCRIM and not p.get("scoring") for p in new)
        for p in new:
            w = _wall(p)
            if w and (self.last_wall is None or w > self.last_wall):
                self.last_wall = w
        scored = [w for w in (_wall(p) for p in new if p.get("scoring")) if w]
        if snap:
            self.dead = None
        elif scored and self.dead and not self.dead["play"]:
            # ESPN's score can lead its play list by a poll or more (2026-10-11, IND-PIT: 15 s); once the scoring play
            # arrives, its own time starts the dead ball
            self.dead["since"], self.dead["play"] = min(self.dead["since"], scored[-1]), True
        if (self.prev_secs is None and "half" in decide.PROP_WINDOWS and "half" not in self.windows_done
                and secs is not None and secs < 1800 and "half" not in (st.get("detail") or "").lower()):
            self.windows_done.add("half")   # joined in the second half: the halftime decision is behind us
            self.say("info", "Joined after halftime: this game's halftime decision (the prop and the game market) has "
                             "passed; game markets still check after each score")
        W = self.window
        if W and snap and not self.demo and (W["rolling"] or now - W["opened"] > 20):
            left = " or ".join("the prop" if k == "prop" else "the game market" for k in W["kinds"])
            if W["rolling"] or W.get("priced"):
                self.say("info", f"{W['label']}: play resumed after {W['checks']} checks; nothing held up on two checks "
                                 f"in a row for {left}; " + ("pass" if W["rolling"] else "no flag"))
            else:
                self.say("info", f"{W['label']}: play resumed before the books posted prices after the score; no flag")
            self._close_window()
        jump = pts - self.prev_points if self.prev_points is not None else 0
        W = self.window
        others = [p for p in new if not p.get("scoring")]   # a late-arriving touchdown play is not the try
        if W and W.get("await_try") and (pts != W["pts"] or others or now - W["opened"] >= TRY_WAIT_S / speed):
            # the try is in (ESPN folds it into the touchdown play) or play moved on: decide on the settled score
            if pts - W["pts"] != 1 and self.dead and self.last_wall and self.last_wall > self.dead["since"]:
                self.dead["since"] = self.last_wall   # a missed kick or a two-point try: earlier prices priced another score
            W["await_try"] = False
            W["label"] = f"After the score ({self.score_line(st)})"
            W["next_check"] = W["due"]["game"] = now
            jump = 0   # the try belongs to the touchdown's dead ball
        elif jump > 0:
            # the scoring play's time; when ESPN's score is ahead of its play list, the time we saw the score (later
            # than the play, so prices posted before the score never count)
            since = scored[-1] if scored else _now()
            # a catch-up batch can hold the score and the next drive's first snap: then play has already resumed
            resumed = bool(scored) and any(p.get("type") in SCRIM and not p.get("scoring") and (_wall(p) or since) > since
                                           for p in new)
            if self.dead and "prop" in self.dead["kinds"]:
                # a score as the half ends: halftime decides on it, starting over from the new score
                self.dead.update(since=max(self.dead["since"], since), play=bool(scored))
                if W and W["rolling"]:
                    W["first"], W["due"], W["next_check"] = {k: None for k in W["kinds"]}, {k: now for k in W["kinds"]}, now
            else:
                self.dead = None if resumed else {"since": since, "kinds": ("game",), "play": bool(scored)}
                if W and not W["rolling"]:
                    self.say("info", f"{W['label']}: another score before play resumed; deciding on the new one")
                    self._close_window()
        brk = _break_name(st.get("detail"), secs, self.prev_secs)
        if brk == "half" and not snap and self.last_wall and not (self.dead and "prop" in self.dead["kinds"]):
            since = max(x for x in (self.last_wall, self.dead and self.dead["since"]) if x)
            self.dead = {"since": since, "kinds": ("prop", "game"), "play": True}
        n_prop, n_game = self._counts()
        kinds = [k for k, n, cap in (("prop", n_prop, decide.PROP_PER_GAME), ("game", n_game, decide.GAME_PER_GAME)) if n < cap]
        if (brk in decide.PROP_WINDOWS and brk not in self.windows_done and kinds
                and (self.window is None or not self.window["rolling"])):
            if self.window is not None:
                self.say("info", f"{self.window['label']}: the break began; halftime decides from here")
            persist = decide.PERSIST_S * (2 if brk == "half" else 1) / speed
            self.window = {"kinds": kinds, "rolling": True, "name": brk, "label": decide.WINDOW_LABEL[brk], "opened": now,
                           "checks": 0, "first": {k: None for k in kinds}, "due": {k: now for k in kinds}, "retries": {},
                           "next_check": now, "persist": persist}
            what = " and ".join("player props" if k == "prop" else "game markets" for k in kinds)
            self.say("info", f"{decide.WINDOW_LABEL[brk]}: decision point for {what} · a check every {persist:.0f}s "
                             f"through the break with prices posted after the half ended; a play must hold on two "
                             f"checks in a row; at most one of each")
        elif (jump > 0 and self.window is None
              and (n_game < decide.GAME_PER_GAME or "game" not in self.shadowed) and secs > 60):
            touchdown = jump == 6
            wait = TRY_WAIT_S / speed if touchdown else 0.0
            self.window = {"kinds": ["game"], "rolling": False, "name": f"score {pts}",
                           "label": f"After the score ({self.score_line(st)})", "opened": now, "checks": 0,
                           "first": {"game": None}, "due": {"game": now + wait}, "retries": {}, "next_check": now + wait,
                           "persist": decide.PERSIST_S / speed, "await_try": touchdown, "pts": pts}
            self.say("info", f"{self.window['label']}: checking the moneyline, spread and total with prices posted "
                             f"after the score" + (", once the try is in" if touchdown else ""))
        self.prev_secs, self.prev_points = secs, pts
        self._status(next_decision=self._watching() if self.window is None else self.window["label"])

    def reason(self, c, proj: dict, st: dict) -> str:
        if c.kind == "prop":
            b = st["banked"].get(c.player, {})
            pb = proj["players"][c.player]
            i = 1 if c.market == "rush_yds" else 0
            other = self.away if c.team == self.home else self.home
            ctx = {"seconds_left": st["seconds_left"], "rem_mean": pb["_rem_mean"], "share_now": pb["_share"][i],
                   "prior_share": self.prior_share[c.player][i],
                   "banked": {"receptions": b.get("rec", 0), "rec_yds": b.get("rec_yds", 0), "rush_yds": b.get("rush_yds", 0),
                              "pass_yds": b.get("pass_yds", 0)},
                   "team_lead": st["score"].get(c.team, 0) - st["score"].get(other, 0)}
            return decide.reason(c, ctx)
        lead = int(st["score"].get(self.home, 0) - st["score"].get(self.away, 0))
        return (f"model {c.p:.0%} vs Pinnacle {c.fair:.0%}" + (f" ({c.ref})" if c.ref else "")
                + f"; {self.home} {lead:+d} with {decide._mins(st['seconds_left'])} left, rest played from team strength")

    def flag(self, c, st: dict, quote_age: float | None = None, window: str | None = None) -> None:
        t = _now()
        cid = f"{self.gid}:{c.key}:{int(t.timestamp())}"
        ttl = decide.CARD_TTL_S * (2 if window == decide.WINDOW_LABEL["half"] else 1)
        card = {"id": cid, "game_id": self.gid, "kind": c.kind, "key": c.key, "player": c.player, "group": c.group,
                "label": c.label(), "market": c.market, "side": c.side, "line": c.line, "book": c.book,
                "dec": round(c.dec, 4), "american": decide.american(c.dec), "p": round(c.p, 4), "push": round(c.push, 4),
                "fair": round(c.fair, 4), "ev": round(c.ev, 4), "stake": decide.STAKE_U, "reason": c.reason,
                "at": t.isoformat(), "expires_at": (t + timedelta(seconds=ttl)).isoformat(), "window": window,
                "status": "active", "fill": "pending", "score": dict(st["score"]), "seconds_left": st["seconds_left"],
                "mode": self.e.mode, "quote_age_s": round(quote_age) if quote_age is not None else None}
        self.flags.append(card)
        self.e.add_card(card)
        ledger.append(ledger.FLAGS, {k: card[k] for k in ("id", "game_id", "kind", "key", "player", "group", "label", "market",
                                                          "side", "line", "book", "dec", "american", "p", "fair", "ev", "stake",
                                                          "reason", "at", "score", "seconds_left", "mode", "quote_age_s", "window")})
        self._status(flags=len(self.flags))
        age = f" · quote {card['quote_age_s']}s old" if card["quote_age_s"] is not None else ""
        self.say("flag", f"{card['label']} {card['american']:+d} {BOOK_SHORT.get(card['book'], card['book'])} · "
                         f"model {c.p:.0%} vs {c.fair:.0%} no-vig · EV {c.ev:+.1%} · paper 1u{age} · {c.reason}")
        notify.send(f"{card['label']} {card['american']:+d} {card['book']}",
                    f"Model {c.p:.0%} vs {c.fair:.0%} no-vig | EV {c.ev:+.1%} | 1u\nWhy: {c.reason}\nConfirm the price in your app first.",
                    self.e.url)

    # ---------------------------------------------------------------- final
    def grade(self, st: dict) -> None:
        margin = st["score"].get(self.home, 0) - st["score"].get(self.away, 0)
        total = st["score"].get(self.home, 0) + st["score"].get(self.away, 0)
        done = {r["id"] for r in ledger.read(ledger.RESULTS)}
        accepts = {a["id"]: a for a in ledger.read(ledger.ACCEPTS)}
        for card in [c for c in self.e.cards.values() if c["game_id"] == self.gid]:
            actual = None
            if card["kind"] == "prop":
                # a tracked player missing from the box score had no stats: graded as zero
                actual = float(st["banked"].get(card["player"], {}).get(STAT_KEY[card["market"]], 0.0))
            res = ledger.grade(card, card["line"], actual, margin, total)   # the paper record: the flagged line
            if card["id"] not in done:
                ledger.append(ledger.RESULTS, {"id": card["id"], "result": res, "actual": actual})
            if card["id"] in accepts and card["id"] + ":you" not in done:
                a = accepts[card["id"]]
                ledger.append(ledger.RESULTS, {"id": card["id"] + ":you", "result": ledger.grade(card, a["line"], actual, margin, total),
                                               "actual": actual})
            self.e.update_card(card["id"], result=res, actual=actual)
        done_sh = {r["id"] for r in ledger.read(ledger.SHADOW_RESULTS)}
        for s in [s for s in ledger.read(ledger.SHADOW) if s["game_id"] == self.gid and s["id"] not in done_sh]:
            actual = (float(st["banked"].get(s["player"], {}).get(STAT_KEY[s["market"]], 0.0)) if s["kind"] == "prop" else None)
            res = ledger.grade(s, s["line"], actual, margin, total)
            ledger.append(ledger.SHADOW_RESULTS, {"id": s["id"], "result": res, "actual": actual})
            self.say("grade", f"Shadow pick {s['label']}: {res} (off the record)")
        mine = [c for c in self.e.cards.values() if c["game_id"] == self.gid]
        tally = {r: sum(c.get("result") == r for c in mine) for r in "WLP"}
        self.say("grade", f"Graded {len(mine)} flag(s): {tally['W']}-{tally['L']}" + (f"-{tally['P']}" if tally["P"] else "")
                 + " (every flag counts at its flagged price)" if mine else "No flags this game: nothing passed the rubric")


def _odds_line(od: dict, src: str) -> str:
    """The engine log's line for one odds update: what came back and how fresh each book's quotes were."""
    fresh = od.get("books") or {}
    ok = od.get("ok_books")
    since = od.get("since")

    def ages(mark_old: bool) -> str:
        return ", ".join(f"{BOOK_SHORT.get(b, b)} frozen" if a is None else
                         f"{BOOK_SHORT.get(b, b)} {a:.0f}s"
                         + ((" (posted before the play stopped, skipped)" if since else " (too old, skipped)")
                            if mark_old and (b not in ok if ok is not None else a > MAX_AGE_S) else "")
                         for b, a in sorted(fresh.items()))
    if not od["props"] and not od["game"]:
        return ("Prices: nothing current enough to price"
                + ((f" (quotes {ages(False)} old, all posted before the play stopped at "
                    f"{datetime.fromisoformat(since).strftime('%H:%M:%S')} UTC)" if since else
                    f" (quotes {ages(False)} old; the limit is {MAX_AGE_S}s)") if fresh else "") + f" · {src}")
    books = ", ".join(BOOK_SHORT.get(b, b) for b in sorted(od["game"])) or "no book"
    return (f"Prices: {len(od['props'])} player lines, game lines at {books}"
            + (f" · newest quote {ages(True)}" if fresh else "") + f" · {src}")


def _wall(play: dict) -> datetime | None:
    """A play's wallclock (ESPN; a demo's archived timestamp) as a UTC datetime."""
    w = play.get("wallclock")
    if not w:
        return None
    try:
        t = datetime.fromisoformat(str(w).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _break_name(detail: str | None, secs: float, prev: float | None) -> str | None:
    """Which break the game is in, from ESPN's status ("Halftime", "End of 1st Quarter") or, when a
    status was missed (or in a demo), from the clock crossing a quarter's end."""
    d = (detail or "").lower()
    if "halftime" in d:
        return "half"
    if d.startswith("end of 1st") or d.startswith("end of the 1st"):
        return "q1"
    if d.startswith("end of 3rd") or d.startswith("end of the 3rd"):
        return "q3"
    if prev is not None:
        for name, at in decide.WINDOW_AT.items():
            if secs <= at < prev:
                return name
    return None


def closing_from_archive(gid: str) -> tuple[dict, str]:
    """Closing prop lines for a game the engine joins after kickoff. Refreshes the game's
    OddsPapi archive first (one free call), so its timeline runs up to kickoff."""
    from pipeline.odds_history import archive
    from props.lines import load as load_lines, main_lines
    try:
        archive(only=[gid])
    except Exception:  # noqa: BLE001 - an older archive still beats no calibration
        pass
    P = load_lines([gid], pregame_only=True)
    out: dict = {}
    note = "archived closing lines, OddsPapi"
    if P.height:
        M = main_lines(P, pl.col("kickoff") - timedelta(minutes=5))
        for x in M.iter_rows(named=True):
            out.setdefault((x["gsis_id"], x["market"]), {})[x["book"]] = {"line": x["line"], "over": x["over"], "under": x["under"]}
        if M.height:
            mins = (P["kickoff"][0] - M["seen"].max()).total_seconds() / 60
            note += f", last update {mins:.0f} min before kickoff"
    return out, note


def _pin_fair(pin: dict, c, margin=None, total=None) -> tuple[float, str] | None:
    """Pinnacle's no-vig chance for this side at the book's line, and a note when it was moved there.
    Pinnacle's own price at the same number (its main or an alternate line) when it has one. Otherwise its
    nearest number within LINE_SLACK, moved to the book's number by the model's chance of the result landing
    between the two (user, 2026-10-11: books rarely hang Pinnacle's number, and a same-number rule meant spreads
    and totals were almost never judged). margin: model samples of home minus away; total: of the total."""
    if c.market == "ml":
        if "ml_home" not in pin:
            return None
        f = decide.no_vig(pin["ml_home"], pin["ml_away"])
        return (f if c.side == "home" else 1 - f), ""
    if c.market == "spread":   # in the home side's handicap h: home covers when margin + h > 0
        lines = dict(pin.get("alt_spread", {}))
        if "spread" in pin:
            lines[-pin["spread"]] = (pin["spread_price_a"], pin["spread_price_b"])
        x, s, first = (c.line if c.side == "home" else -c.line), margin, c.side == "home"
        win = lambda h: float((s + h > 1e-9).mean())
        lose = lambda h: float((s + h < -1e-9).mean())
    else:
        lines = dict(pin.get("alt_total", {}))
        if "total" in pin:
            lines[pin["total"]] = (pin["total_price_a"], pin["total_price_b"])
        x, s, first = c.line, total, c.side == "over"
        win = lambda t: float((s > t + 1e-9).mean())
        lose = lambda t: float((s < t - 1e-9).mean())
    if not lines:
        return None
    near = min(lines, key=lambda h: (abs(h - x), h))
    if abs(near - x) > LINE_SLACK[c.market] + 1e-9:
        return None
    f = decide.no_vig(*lines[near])
    f = f if first else 1 - f
    if near == x:
        return f, ""
    if s is None:
        return None
    side = win if first else lose
    push = 1 - win(near) - lose(near)
    fair = min(max(f * (1 - push) + side(x) - side(near), 0.01), 0.99)
    show = (lambda h: f"{(h if first else -h):+g}") if c.market == "spread" else (lambda h: f"{h:g}")
    return fair, f"Pinnacle's {show(near)} moved to {show(x)} by the model"


# ---------------------------------------------------------------- web page and API

def serve(engine: Engine, host: str, port: int = PORT) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith("/api/state"):
                q = parse_qs(urlparse(self.path).query)
                try:
                    since = int(q.get("since", ["0"])[0])
                except ValueError:
                    since = 0
                return self._json(engine.snapshot(since))
            if self.path in ("/", "/index.html"):
                b = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
                return
            self.send_error(404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/accept":
                am = body.get("american")
                return self._json(engine.accept(body.get("id", ""), int(am) if am not in (None, "") else None))
            if self.path == "/api/pass":
                return self._json(engine.pass_(body.get("id", "")))
            self.send_error(404)

    class Server(ThreadingHTTPServer):
        # Python's default (SO_REUSEADDR) lets a second engine bind the same port on Windows without an
        # error; two engines would then both flag the same games into the ledger. Refuse instead.
        allow_reuse_address = False
        daemon_threads = True

    try:
        srv = Server((host, port), H)
    except OSError:
        raise SystemExit(f"Port {port} is already in use: another live engine is still running. Stop it first "
                         f"(Ctrl+C in its window, or close that window), then start this one again.")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


# ---------------------------------------------------------------- choosing games

def todays_games(only: list[str] | None) -> list[dict]:
    g = pl.read_parquet(CLEAN / "games.parquet")
    sched = pl.read_parquet(RAW / "schedules.parquet").select("game_id", "espn")
    g = g.join(sched, on="game_id", how="left")
    now = _now()
    if only:
        g = g.filter(pl.col("game_id").is_in(only))
    else:   # every unplayed game kicking off in the next 48 hours (a Sunday-morning start covers Monday night) or under way
        g = g.filter(pl.col("result").is_null() & (pl.col("kickoff_utc") <= now + timedelta(hours=48))
                     & (pl.col("kickoff_utc") >= now - timedelta(hours=4)))
    return [{"game_id": r["game_id"], "home": r["home_team"], "away": r["away_team"], "kickoff": r["kickoff_utc"],
             "espn": r["espn"], "season": r["season"], "week": r["week"]} for r in g.sort("kickoff_utc").iter_rows(named=True)]


def priors_for(games: list[dict]) -> dict[str, pl.DataFrame]:
    """Pregame projections for tracked players, from the props model."""
    from props.backtest import load as load_props
    from props.section import project_week
    p, _ = load_props()
    allg = pl.read_parquet(CLEAN / "games.parquet")
    out = {}
    for (season, week), grp in _group(games):
        up = allg.filter(pl.col("game_id").is_in([x["game_id"] for x in grp]))
        lines = {r["game_id"]: {"spread": r["spread_line"], "total": r["total_line"]} for r in up.iter_rows(named=True)}
        x = project_week(p, season, week, up.filter(pl.col("result").is_null()), lines)
        for gid in [x_["game_id"] for x_ in grp]:
            out[gid] = x.filter(pl.col("game_id") == gid) if x.height else pl.DataFrame()
    return out


def _group(games):
    d: dict = {}
    for g in games:
        d.setdefault((g["season"], g["week"]), []).append(g)
    return d.items()


_SYNC = threading.Lock()


def sync_ledger(message: str) -> tuple[bool, str]:
    """Send the live ledger to GitHub so the public board (built there) shows it: pull, commit
    ledger/live_*.jsonl, push. Only the engine writes those files, so this never conflicts with
    the scheduled builds. Best effort: a failure is logged and the files stay committed locally."""
    import subprocess
    root = str(Path(__file__).resolve().parent.parent)

    def git(*a):
        return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, timeout=120)

    with _SYNC:
        try:
            if git("remote").stdout.strip() == "":
                return False, "No GitHub remote: live results stay on this laptop"
            git("add", "--", "ledger/live_*.jsonl")   # a pattern: naming a file that does not exist yet fails the whole add
            c = git("commit", "-m", message, "--", "ledger/")
            if c.returncode != 0 and "nothing" not in (c.stdout + c.stderr):
                return False, f"Could not commit the live ledger: {(c.stderr or c.stdout).strip()[:160]}"
            p = git("pull", "--rebase", "--autostash")
            if p.returncode != 0:
                git("rebase", "--abort")
                return False, f"Could not update from GitHub (results kept locally): {p.stderr.strip()[:160]}"
            u = git("push")
            if u.returncode != 0:
                return False, f"Could not push to GitHub (results kept locally): {u.stderr.strip()[:160]}"
            return True, "Live ledger sent to GitHub; the public board shows it at its next build"
        except (OSError, subprocess.SubprocessError) as e:
            return False, f"GitHub sync failed (results kept locally): {e}"


def _stay_awake() -> None:
    """Ask Windows not to idle-sleep while the engine runs (released when the process exits).
    A closed lid or choosing Sleep still sleeps; no power setting is changed."""
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except (AttributeError, OSError):
        pass   # not Windows


def sgo_events(games: list[dict]) -> dict[str, tuple[str, dict]]:
    """nflverse game id -> (SGO event id, event) by teams and kickoff within 6 hours."""
    from pipeline.odds_history import TEAMS
    kicks = [g["kickoff"] for g in games if g.get("kickoff")]
    if not kicks:
        return {}
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    evs = sgo.nfl_events((min(kicks) - timedelta(hours=6)).strftime(fmt), (max(kicks) + timedelta(hours=6)).strftime(fmt))
    out = {}
    for g in games:
        for e in evs:
            t = e.get("teams") or {}
            if not (t.get("home") and t.get("away")):
                continue
            h, a = TEAMS.get(t["home"]["names"]["long"]), TEAMS.get(t["away"]["names"]["long"])
            start = datetime.fromisoformat(e["status"]["startsAt"].replace("Z", "+00:00"))
            if {h, a} == {g["home"], g["away"]} and g.get("kickoff") and abs((start - g["kickoff"]).total_seconds()) < 6 * 3600:
                out[g["game_id"]] = (e["eventID"], e)
    return out


def propline_event(game: dict) -> str | None:
    ids = propline.ids()
    if isinstance(ids, dict):
        return None
    for e in ids:
        if str(e.get("espn_event_id")) == str(game["espn"]):
            return str(e["id"])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", help="nflverse game ids (default: every game in the next 48 hours)")
    ap.add_argument("--demo", help="replay this finished game instead of a live one")
    ap.add_argument("--speed", type=float, default=20.0, help="demo speed: game seconds per real second")
    ap.add_argument("--lan", action="store_true", help="serve on your home network too (for your phone)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--odds", choices=("sgo", "propline"), default="propline",
                    help="live prices: PropLine (default; ~50 requests a game on the lean schedule) or SportsGameOdds "
                         "(its in-game quotes were 5-7 minutes old on the trial plan, 2026-10-11: too stale to bet)")
    args = ap.parse_args()
    global SRC
    SRC = OddsSource(args.odds)
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    url = f"http://{_lan_ip() if args.lan else 'localhost'}:{args.port}"
    if args.demo:   # a demo replay keeps its own throwaway ledger, so it never touches the real record
        demo_dir = LOG_DIR / "demo_ledger"
        for f in demo_dir.glob("*.jsonl"):
            f.unlink()
        ledger.use(demo_dir)
    engine = Engine("demo" if args.demo else "paper", url)
    serve(engine, host, args.port)
    print(f"live page: {url}  (Ctrl+C to stop)")
    runners = []
    if args.demo:
        from props.backtest import load as load_props
        from props.section import _frames_cached
        p, _ = load_props()
        pri = _frames_cached(p).filter((pl.col("game_id") == args.demo) & pl.col("role").is_not_null())
        feed = DemoFeed(args.demo, args.speed)
        runners.append(GameRunner(engine, {"game_id": args.demo, "home": feed.home, "away": feed.away}, feed, pri, demo=True))
    else:
        games = todays_games(args.games)
        if not games:
            print("no games in the next 12 hours")
            return
        if SRC.kind == "propline":
            # only games before PropLine's reset (midnight UTC, 8pm ET) spend today's budget; a game already
            # past halftime needs about half a game's calls
            now_ = _now()
            reset = (now_ + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            today = [g for g in games if g["kickoff"] < reset]
            cost = lambda g: LEAN_CALLS_PER_GAME * (0.5 if g["kickoff"] < now_ - timedelta(minutes=90) else 1)
            left, keep = max(0, propline.remaining_today()), []
            for g in today:
                if cost(g) <= left:
                    keep.append(g)
                    left -= cost(g)
            if len(keep) < len(today):
                msg = (f"PropLine budget covers {len(keep)} of today's {len(today)} games; skipping "
                       + ", ".join(f"{g['away']} at {g['home']}" for g in today if g not in keep))
                print(msg)
                engine.log("warn", msg)
            games = keep + [g for g in games if g["kickoff"] >= reset]
        engine.log("info", f"Engine started in paper mode · prices from {SRC.name} · {SRC.used_today()}/{SRC.budget} requests used today")
        priors = priors_for(games)
        e2g = espn_to_gsis()
        sgo_ev = sgo_events(games) if SRC.kind == "sgo" else {}
        for g in games:
            pe = sgo_ev.get(g["game_id"], (None,))[0] if SRC.kind == "sgo" else propline_event(g)
            if not pe or g["espn"] is None or not priors.get(g["game_id"], pl.DataFrame()).height:
                msg = f"Skipping {g['away']} at {g['home']}: no {SRC.name} event, ESPN id or projections"
                print(msg)
                engine.log("warn", msg, g["game_id"])
                continue
            if SRC.kind == "sgo":
                who = priors[g["game_id"]].select("gsis_id", "name", "team")
                pmap = sgo_player_map(sgo_ev[g["game_id"]][1], who)
                engine.log("info", f"{g['away']} at {g['home']}: {len(set(pmap.values()))} of {who.height} tracked players have SportsGameOdds props so far", g["game_id"])
                feed = SGOFeed(str(g["espn"]), pe, g["home"], g["away"], e2g, pmap, who)
            else:
                feed = LiveFeed(str(g["espn"]), pe, g["home"], g["away"], e2g)
            runners.append(GameRunner(engine, g, feed, priors[g["game_id"]]))
    engine.runners = runners
    if not args.demo:
        _stay_awake()
    for r in runners:
        r.start()
        print(f"watching {r.gid}")
    engine.ready = True
    try:
        done_said = False
        while True:   # heartbeat for the board's Live tab; the game threads do the work
            if not args.demo:
                engine.write_status()
            if not done_said and not any(r.is_alive() for r in runners):
                print("all games final; graded. The page stays up until Ctrl+C.")
                engine.log("info", "All games final and graded. The page stays up until you stop the engine.")
                done_said = True
            time.sleep(20)
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
