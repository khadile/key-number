"""How much each pick risks (user, 2026-10-15).

- Flat units go only on positive-EV picks. A side the model leans to at negative EV is shown as a lean and
  risks nothing: the board's 141 negative-EV spread and total picks in 2026 weeks 1-5 lost 11.2u (ROI -8%),
  and over the 2022-26 backtest at closing prices they lost 0.5% on 1,618, never helping.
- A quarter-Kelly size is tracked beside the flat record, not yet the stake: Kelly stakes in proportion to the
  edge for the price, so a solid favorite gets more and a long shot less (-147 at 62%: 2u; +575 at 16%: 0.25u),
  with 1u = 1% of the bankroll, rounded to 0.25 / 0.5 / 1 / 2u. It is tracked first because sizing by the
  model's confidence amplifies its mistakes as much as its edge: on the 2022-26 backtest at closing prices
  quarter Kelly lost -6.4% per unit against flat's -2.7% (the model overrates its edges), while 2026 weeks
  1-5 favored it. No cap on a week's total exposure: the bankroll is the user's call.
- The lean rule starts with 2026 week 6, whose games were all unplayed when it was adopted (07:00 ET,
  2026-10-15); weeks 1-5 keep their 1u stakes, since they helped choose it (re-staking them would add back
  about 11u from the data that picked the rule). The quarter-Kelly record runs from week 1, kept apart from
  the flat record (user, 2026-10-15), with its week-6-on part shown too: the part that tests it.
"""
from math import sqrt

from config import STRAIGHT_RISK

KELLY_FRACTION = 0.25
KELLY_TIERS = (0.25, 0.5, 1.0, 2.0)   # units, 1u = 1% of the bankroll
RULES_FROM = (2026, 6)                # (season, week) the stake rules start
RULES_FROM_DATE = "2026-10-15"        # the same start for live flags, which carry a time rather than a week


def tracked(r: dict) -> bool:
    """Whether a pick or live flag came after the stake rules were set (week 6 on), out of their sample."""
    if r.get("week") is None:
        return (r.get("at") or r.get("locked_at") or "") >= RULES_FROM_DATE
    return (r.get("season") or 0, r["week"]) >= RULES_FROM


def kelly(p_win: float, dec: float, p_push: float = 0.0) -> float:
    """The full Kelly fraction of the bankroll for a bet paying `dec` (a push refunds the stake)."""
    b = dec - 1
    if b <= 0 or p_win is None:
        return 0.0
    q = 1 - p_win - (p_push or 0.0)
    return max((p_win * b - q) / b, 0.0)


def kelly_units(p_win: float, dec: float, p_push: float = 0.0) -> float:
    """Quarter Kelly in units, rounded to the nearest tier (geometric midpoints); 0 when there is no edge."""
    k = KELLY_FRACTION * kelly(p_win, dec, p_push) * 100
    if k <= 0:
        return 0.0
    for t, nxt in zip(KELLY_TIERS, KELLY_TIERS[1:]):
        if k < sqrt(t * nxt):
            return t
    return KELLY_TIERS[-1]


def flat_risk(ev: float | None) -> float:
    """The flat stake: a unit on a positive-EV pick, nothing on a lean."""
    return STRAIGHT_RISK if ev is not None and ev > 0 else 0.0


def payout(result: str, dec: float) -> float:
    """Units won per unit staked."""
    return dec - 1 if result == "W" else (-1.0 if result == "L" else 0.0)


def kelly_of(r: dict) -> float:
    """A row's quarter-Kelly stake: as stored when it locked, else worked out from its locked price and chance."""
    k = r.get("kelly_u")
    return k if k is not None else kelly_units(r.get("p_win"), r["dec"], r.get("p_push") or 0.0)


def kelly_tally(rows: list[dict]) -> dict:
    """The quarter-Kelly record of graded rows (each with result, dec, and kelly_u or p_win/p_push)."""
    staked = units = 0.0
    w = l = p = n = 0
    for r in rows:
        if r.get("result") not in ("W", "L", "P"):
            continue
        k = kelly_of(r)
        if k <= 0:
            continue
        n += 1
        w, l, p = w + (r["result"] == "W"), l + (r["result"] == "L"), p + (r["result"] == "P")
        staked += k
        units += k * payout(r["result"], r["dec"])
    return {"w": w, "l": l, "p": p, "n": n, "staked": round(staked, 2), "units": round(units, 2),
            "roi": round(units / staked, 4) if staked else None}
