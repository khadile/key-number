"""The good-pick checks for prop picks (model/checks.py).

Sharp: props are priced from Pinnacle's no-vig line moved toward the model by k, and k is about 0.005, so a
prop anchored on Pinnacle has the same EV at Pinnacle's own price to the first decimal. One anchored on the
DraftKings/FanDuel average has no sharp price behind it.
"""
from collections import defaultdict

import numpy as np
import polars as pl

from board.market import BOOK_NAMES, american
from config import RAW
from model import checks as C
from props.pricing import SHOP, PropPricer, side_probs


def prop_doubts(season: int) -> dict[tuple[int, str], str]:
    """(week, gsis_id) -> Questionable or Doubtful, from that week's injury report."""
    path = RAW / "injuries" / f"{season}.parquet"
    if not path.exists():
        return {}
    i = pl.read_parquet(path).filter((pl.col("season") == season) & pl.col("report_status").is_in(list(C.UNSURE)))
    return {(r["week"], r["gsis_id"]): r["report_status"] for r in i.select("week", "gsis_id", "report_status").iter_rows(named=True)}


class Teams:
    """Each player's games in order, to tell whether he just changed teams."""

    def __init__(self, hist: pl.DataFrame):
        self.games: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        for pid, s, w, t in hist.select("gsis_id", "season", "week", "team").sort("season", "week").iter_rows():
            self.games[pid].append((s, w, t))

    def streak(self, pid: str, season: int, week: int, team: str) -> tuple[int, str | None]:
        """(games in a row with `team` before this one, the team before that if he changed teams)."""
        n, prev = 0, None
        for s, w, t in reversed(self.games.get(pid, [])):
            if (s, w) >= (season, week):
                continue
            if t != team:
                prev = t
                break
            n += 1
        return n, prev


def now_price(pk: dict, books: dict, s: np.ndarray, pricer: PropPricer) -> dict | None:
    """The best DraftKings/FanDuel price on the pick's side now, and its EV by the board's own pricing."""
    d, _ = pricer.anchor(s, books)
    if d is None:
        return None
    x = s + (1 - pricer.k) * d
    best = None
    for b in SHOP:
        if b not in books:
            continue
        L = books[b]
        over, push, under = side_probs(x, L["line"])
        p, dec = (over, L["over"]) if pk["side"] == "over" else (under, L["under"])
        ev = p * (dec - 1) - (1 - p - push)
        if best is None or ev > best["ev"]:
            best = {"book": BOOK_NAMES[b], "label": f"{'Over' if pk['side'] == 'over' else 'Under'} {L['line']:g}",
                    "american": american(dec), "ev": round(ev, 4)}
    return best


def prop_checks(pk: dict, doubt: str | None, streak: tuple[int, str | None], books_now: dict | None = None,
                s: np.ndarray | None = None, pricer: PropPricer | None = None) -> dict:
    """Checks, verdict and (given today's lines and samples) the price now."""
    problems = []
    if doubt:
        problems.append(f"{doubt} on the injury report.")
    n, prev = streak
    if prev and n < 3:
        problems.append(f"New to the team: {'first game' if n == 0 else f'{n} game' + ('s' if n > 1 else '')} since {prev}.")
    if pk.get("promoted"):
        problems.append("Promoted: a teammate ahead of him is out, so the role is new to him.")
    checks = [C.sharp(pk["ev"] if pk.get("anchor") == "pinnacle" else None), C.size(pk["ev"]),
              C.lean(pk["market"], pk["side"], pk["ev"]), C.who(problems, "Not on the injury report, and in his usual role.")]
    out = {"verdict": C.verdict(checks), "checks": checks}
    if books_now and s is not None and pricer is not None:
        out["now"] = now_price(pk, books_now, s, pricer)
    return out
