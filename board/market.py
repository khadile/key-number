"""Lines across books, and pricing the model's picks against them.

How a pick's probability is set (the honest part): closing lines are more
accurate than the model, so the model's number is never taken at face value.
Instead, the expected final margin is the market line moved part of the way
toward the model:

    expected = market + k * (model - market)

where k is how far lines have actually moved toward the model between our
publish time and kickoff this season (measured from archived OddsPapi lines).
If lines stop moving toward the model, k falls toward 0 and every edge
shrinks with it.
"""
import itertools
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from model.probs import KeyDist, WinCurve
from pipeline.odds_history import CACHE, HIST, SHOP_BOOKS, fixture_index, lines_at

BOOK_NAMES = {
    "pinnacle": "Pinnacle", "draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM",
    "caesars": "Caesars", "espnbet": "ESPN BET", "fanatics": "Fanatics", "betrivers": "BetRivers",
    "hardrockbet": "Hard Rock",
}


def american(dec: float) -> int:
    return int(round((dec - 1) * 100)) if dec >= 2 else int(round(-100 / (dec - 1)))


def fmt_line(x: float) -> str:
    return "PK" if x == 0 else (f"+{x:g}" if x > 0 else f"{x:g}")


class Lines:
    """Every archived price timeline for one game, all books."""

    def __init__(self, fixture_id: str | None, kickoff: datetime):
        self.kickoff = kickoff
        files = sorted(HIST.glob(f"{fixture_id}*.parquet")) if fixture_id else []
        if files:
            h = pl.concat([pl.read_parquet(f) for f in files])
            self.h = h.filter(pl.col("ts") < kickoff)
        else:
            self.h = pl.DataFrame()

    def books(self) -> list[str]:
        return sorted(self.h["book"].unique().to_list()) if self.h.height else []

    def at(self, book: str, when: datetime) -> dict:
        if self.h.height == 0:
            return {}
        g = self.h.filter(pl.col("book") == book)
        cut = min(when, self.kickoff - timedelta(minutes=5))
        return lines_at(g, cut) if g.height else {}

    def all_at(self, when: datetime, books: list[str]) -> dict[str, dict]:
        return {b: r for b in books if (r := self.at(b, when))}

    def first_seen(self, book: str) -> datetime | None:
        g = self.h.filter(pl.col("book") == book) if self.h.height else self.h
        return g["ts"].min() if g.height else None

    def movement(self, book: str, start: datetime, end: datetime, step_hours: int = 6) -> list[dict]:
        """The book's main spread and total sampled over time, for the line-movement chart."""
        out, t = [], start
        end = min(end, self.kickoff - timedelta(minutes=5))
        while t <= end:
            r = self.at(book, t)
            if "spread" in r or "total" in r:
                out.append({"t": t.isoformat(), "spread": r.get("spread"), "total": r.get("total")})
            t += timedelta(hours=step_hours)
        r = self.at(book, end)
        if r and (not out or out[-1]["t"] != end.isoformat()):
            out.append({"t": end.isoformat(), "spread": r.get("spread"), "total": r.get("total")})
        return out


def fixture_map() -> dict[str, str]:
    idx = fixture_index()
    return dict(zip(idx["game_id"], idx["fixture_id"]))


def reference(lines: dict[str, dict], fallback: dict) -> dict:
    """The market's number: Pinnacle when it has the game, else the median of DraftKings and FanDuel, else nflverse."""
    pin = lines.get("pinnacle", {})
    ref = {}
    for key in ("spread", "total"):
        if key in pin:
            ref[key] = pin[key]
        else:
            vals = [r[key] for b, r in lines.items() if key in r and b in SHOP_BOOKS]
            ref[key] = float(np.median(vals)) if vals else fallback.get(key)
    return ref


@dataclass
class Pricer:
    mdist: KeyDist
    tdist: KeyDist
    wins: WinCurve
    k_spread: float
    k_total: float

    def expected_margin(self, model: float, market: float) -> float:
        return market + self.k_spread * (model - market)

    def expected_total(self, model: float, market: float) -> float:
        return market + self.k_total * (model - market)

    def price(self, game: dict, lines: dict[str, dict], model_margin: float, model_total: float,
              ref: dict, shop_books: list[str]) -> dict:
        """The model's side in each market, priced at the best book. Returns
        {spread, total, ml} picks (ml only when it has positive EV) plus fair numbers."""
        out: dict = {"fair": {}}
        home, away = game["home_team"], game["away_team"]
        if ref.get("spread") is not None:
            mu = self.expected_margin(model_margin, ref["spread"])
            out["fair"]["spread"] = mu
            side = "home" if model_margin > ref["spread"] else "away"
            best = None
            for b in shop_books:
                r = lines.get(b, {})
                if "spread" not in r:
                    continue
                s = r["spread"]
                over, push, under = self.mdist.side_probs(mu, s)
                p_win, dec, line = (over, r["spread_price_a"], -s) if side == "home" else (under, r["spread_price_b"], s)
                ev = p_win * (dec - 1) - (1 - p_win - push)
                if best is None or ev > best["ev"]:
                    team = home if side == "home" else away
                    best = {"market": "spread", "side": side, "team": team, "line": line,
                            "label": f"{team} {fmt_line(line)}", "dec": dec, "american": american(dec),
                            "book": b, "p_win": p_win, "p_push": push, "ev": ev}
            if best:
                out["spread"] = best
            mu_win = self.wins.home(mu)
            ml_side = side
            best_ml = None
            for b in shop_books:
                r = lines.get(b, {})
                if "ml_home" not in r:
                    continue
                p, dec = (mu_win, r["ml_home"]) if ml_side == "home" else (1 - mu_win, r["ml_away"])
                ev = p * (dec - 1) - (1 - p)
                if best_ml is None or ev > best_ml["ev"]:
                    team = home if ml_side == "home" else away
                    best_ml = {"market": "ml", "side": ml_side, "team": team, "line": None,
                               "label": f"{team} ML", "dec": dec, "american": american(dec),
                               "book": b, "p_win": p, "p_push": 0.0, "ev": ev}
            out["win_home"] = mu_win
            if best_ml and best_ml["ev"] > 0:
                out["ml"] = best_ml
        if ref.get("total") is not None:
            mu_t = self.expected_total(model_total, ref["total"])
            out["fair"]["total"] = mu_t
            side = "over" if model_total > ref["total"] else "under"
            best = None
            for b in shop_books:
                r = lines.get(b, {})
                if "total" not in r:
                    continue
                t = r["total"]
                over, push, under = self.tdist.side_probs(mu_t, t)
                p_win, dec = (over, r["total_price_a"]) if side == "over" else (under, r["total_price_b"])
                ev = p_win * (dec - 1) - (1 - p_win - push)
                if best is None or ev > best["ev"]:
                    best = {"market": "total", "side": side, "team": None, "line": t,
                            "label": f"{side.title()} {t:g}", "dec": dec, "american": american(dec),
                            "book": b, "p_win": p_win, "p_push": push, "ev": ev}
            if best:
                out["total"] = best
        return out


def kelly(p: dict) -> float:
    """Edge relative to the payout (the Kelly fraction). Ranking by raw EV% favors
    longshots: a +400 underdog a hair above fair shows a bigger EV% than a -110
    spread with a real edge. Kelly ranks them by how much the edge is worth per
    unit of risk, so lottery tickets do not crowd out real edges."""
    return p["ev"] / (p["dec"] - 1)


def picks_of_the_week(picks: list[dict], k: int = 5) -> list[dict]:
    """Top k picks by Kelly fraction, positive EV only, at most one per game."""
    seen, out = set(), []
    for p in sorted((p for p in picks if p["ev"] > 0), key=lambda p: -kelly(p)):
        if p["game_id"] in seen:
            continue
        seen.add(p["game_id"])
        out.append(p)
        if len(out) == k:
            break
    return out


def parlay_ev(legs: list[dict]) -> tuple[float, float, float]:
    """(decimal price, chance it cashes, EV per unit) for independent legs.
    A pushed leg drops out; any losing leg loses the parlay. Closed form:
    each leg returns dec with p_win, 1 with p_push, 0 otherwise, so the
    expected return is the product of the legs' expected returns."""
    ret = math.prod(l["p_win"] * l["dec"] + l["p_push"] for l in legs)
    no_loss = math.prod(l["p_win"] + l["p_push"] for l in legs)
    all_push = math.prod(l["p_push"] for l in legs)
    return float(math.prod(l["dec"] for l in legs)), float(no_loss - all_push), float(ret - 1)


# Decimal prices: +150 = 2.5, +300 = 4.0, +500 = 6.0
GOLDEN_BAND = (2.5, 4.0)       # the Golden Parlay aims for +150 to +300 ...
GOLDEN_STRETCH = (4.0, 6.0)    # ... and stretches to +500 only when nothing fits
LOTTERY_BAND = (6.0, float("inf"))  # the +EV Lottery: anything longer than +500, no cap. It stays
                                    # reasonable by construction: at most 3 legs, each a positive-EV pick.


GOLDEN_LEGS = (2, 3)
LOTTERY_LEGS = (3, 4, 5)
LOTTERY_LEG_MAX_DEC = 3.0  # each lottery leg +200 or shorter; with long underdog legs "best EV"
                           # runs to +170,000 five-leg tickets


def build_parlays(picks: list[dict]) -> tuple[dict | None, dict | None]:
    """The Golden Parlay and the +EV Lottery from a week's picks.

    Legs: positive-EV picks only, at most one per game (legs from the same
    game move together, and books price them that way). The Golden Parlay is
    the best expected value among 2-3 leg combinations priced +150 to +300
    (or up to +500 if none is); the Lottery is the best expected value among
    3-5 leg combinations priced above +500, with no cap. Either can be empty.
    """
    pool = [p for p in picks if p["ev"] > 0]

    def scored(sizes, legs_pool):
        for n in sizes:
            for legs in itertools.combinations(legs_pool, n):
                if len({l["game_id"] for l in legs}) < n:
                    continue
                dec, cash, ev = parlay_ev(list(legs))
                if ev > 0:
                    yield list(legs), dec, cash, ev

    golden_pool = list(scored(GOLDEN_LEGS, pool))
    lottery_pool = list(scored(LOTTERY_LEGS, [p for p in pool if p["dec"] <= LOTTERY_LEG_MAX_DEC]))

    def best(pool_, lo: float, hi: float):
        inside = [s for s in pool_ if lo < s[1] <= hi or (lo == GOLDEN_BAND[0] and s[1] == lo)]
        return max(inside, key=lambda s: (s[3], s[2])) if inside else None

    def pack(s, kind: str) -> dict | None:
        if s is None:
            return None
        legs, dec, cash, ev = s
        return {"type": kind, "legs": [l["id"] for l in legs], "dec": dec, "american": american(dec), "p": cash, "ev": ev}

    golden = best(golden_pool, *GOLDEN_BAND) or best(golden_pool, *GOLDEN_STRETCH)
    lottery = best(lottery_pool, *LOTTERY_BAND)
    return pack(golden, "golden"), pack(lottery, "lottery")
