"""Pricing player props: the market's number, nudged toward the model, against DK and FanDuel.

The same honest rule as game lines. The model's distribution supplies the
shape (how spread out a receiver's yards are); the market supplies the
location. The market's location is where its no-vig price says the median is:
Pinnacle's main line when Pinnacle has the player, else the average of
DraftKings and FanDuel. The expected location moves from there toward the
model by k, the share of the gap that lines have actually closed between lock
and kickoff this season (measured each week; about 0.01-0.04 so far):

    location = market + k * (model - market)

A pick is the side and book with the best expected value at DraftKings or
FanDuel's own line and price. Most of the edge, when there is any, comes from
one of those two books hanging a number off Pinnacle's.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

SHOP = ("draftkings", "fanduel")
ET = ZoneInfo("America/New_York")
MARKET_LABEL = {"pass_yds": "Pass yds", "rush_yds": "Rush yds", "rec_yds": "Rec yds", "receptions": "Receptions"}


def shift_to(s: np.ndarray, line: float, p_over: float) -> float:
    """Units to add to the model's samples so they give `p_over` at `line`."""
    q = float(np.quantile(s, min(max(1 - p_over, 0.001), 0.999)))
    return line - q


def side_probs(s: np.ndarray, line: float) -> tuple[float, float, float]:
    """(over, push, under) for sorted samples; a whole-number line can push."""
    over = float((s > line + 1e-9).mean())
    under = float((s < line - 1e-9).mean())
    return over, 1 - over - under, under


@dataclass
class PropPricer:
    k: float

    def anchor(self, s: np.ndarray, books: dict[str, dict]) -> tuple[float | None, str | None]:
        """The market's location relative to the model (units), and where it came from."""
        if "pinnacle" in books:
            b = books["pinnacle"]
            return shift_to(s, b["line"], b["p_over"]), "pinnacle"
        us = [shift_to(s, books[b]["line"], books[b]["p_over"]) for b in SHOP if b in books]
        return (float(np.mean(us)), "dk/fd") if us else (None, None)

    def price(self, s: np.ndarray, books: dict[str, dict]) -> dict | None:
        """Best side at DraftKings or FanDuel. `s`: the model's samples for this player
        and market (calibrated, receptions jittered). `books`: {book: {line, over, under, p_over}}."""
        d_mkt, src = self.anchor(s, books)
        if d_mkt is None:
            return None
        shift = (1 - self.k) * d_mkt          # market + k * (model - market), as a shift of the model
        x = s + shift
        best = None
        for b in SHOP:
            if b not in books:
                continue
            L = books[b]
            over, push, under = side_probs(x, L["line"])
            for side, p, dec in (("over", over, L["over"]), ("under", under, L["under"])):
                ev = p * (dec - 1) - (1 - p - push)
                if best is None or ev > best["ev"]:
                    best = {"side": side, "line": L["line"], "book": b, "dec": dec, "p_win": p, "p_push": push,
                            "ev": ev}
        if best is None:
            return None
        med_model = float(np.median(s))
        best.update(model_median=med_model, market_median=med_model + d_mkt, fair_median=med_model + shift,
                    anchor=src, model_mean=float(np.mean(s)))
        return best


# The board updates Tuesday, Thursday and Saturday at 10am ET (the scheduled build). Picks lock
# at the last update before their game; Tuesday's update grades last week (user's rule, 2026-10-09).
UPDATE_DAYS_BACK = {5: 0, 6: 1, 0: 2, 1: 0, 2: 1, 3: 0, 4: 1}   # kickoff weekday (Mon=0) -> days back to Sat/Tue/Thu


def lock_time(kickoff: datetime) -> datetime:
    """When a game's picks lock: 10am ET at the last board update before it. Thursday games
    lock that Thursday morning; Saturday, Sunday and Monday games lock Saturday morning, after
    Friday's final injury reports and with Thursday night's results in."""
    et = kickoff.astimezone(ET)
    t = (et.replace(hour=10, minute=0, second=0, microsecond=0)
         - timedelta(days=UPDATE_DAYS_BACK[et.weekday()])).astimezone(kickoff.tzinfo)
    return t if t < kickoff else kickoff - timedelta(hours=3)


def lock_window_open(kickoff: datetime, now: datetime) -> bool:
    """A build locks a game's props once its lock time has passed and the game has not started."""
    return lock_time(kickoff) <= now < kickoff
