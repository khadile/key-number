"""The bet list (user, 2026-10-26): the subset of the board worth staking, chosen by what has been working lately.

The user bets a few picks, not the board. Each pick falls in a cell: its category (market and side, e.g.
"3-pointers under") and its EV tier at the lock (1-2%, 2-4%, 4%+). Every cell's true edge is estimated from
three things, weighted by how much each can be trusted:

  1. EV at Pinnacle's close of its graded picks: settles at kickoff, so it moves the estimate within days;
     a small allowance (CLOSE_BIAS) because the close itself is not the truth (it overprices prop overs a bit).
  2. Units won per pick: the result the user cares about, but a pick's units vary about 1u either way, so a few
     days of them barely count; weeks of them do. A cell up units AND beating the close moves fastest.
  3. Its tier's record across every category (one pool per tier), which a cell leans on until it has its own.

Recent picks count more (exponential decay, half-life per board), so a cell that turns cold leaves the list and
one that turns hot joins it, with no fixed rule about which categories or tiers qualify. All of it is a normal
approximation: estimate = precision-weighted average of the prior, the close and the units.

A pick makes the list when its cell's estimate is at least MIN_EDGE and 65%+ likely above zero; the best cells
first, at most PER_CELL from one cell, up to the board's cap per lock. Locked with the picks (a betlist marker in the ledger) and never redone.
"""
import math
from dataclasses import dataclass

PRIOR_SD = 0.03        # how far a tier's edge can plausibly sit from zero
CELL_SD = 0.03         # how far a cell can sit from its tier
CLOSE_BIAS = 0.03      # how far the close can misjudge a category (NFL rec yds: +5% at the close, -11% ROI on 165)
MIN_EDGE = 0.01
MIN_CELL_N = 3         # a cell with fewer graded picks of its own goes by its tier's verdict (one lucky close is not a trend)
PER_CELL = 2           # at most this many picks from one cell per lock: six threes unders are one bet, not six
MIN_PROB = 0.65
UNIT_SD = 1.0          # a 1u pick near even money: units vary about 1u per pick
CLOSE_SD = 0.06        # spread of single picks' EV at the close when a group has too few to measure it


@dataclass
class Board:
    half_life: float           # in the board's time unit (days for NBA, weeks for NFL)
    cap: int                   # most picks on the list per lock
    tiers: tuple = ((0.01, "1-2%"), (0.02, "2-4%"), (0.04, "4%+"))


def tier_of(ev: float, board: Board) -> str | None:
    t = None
    for lo, name in board.tiers:
        if ev >= lo:
            t = name
    return t


def _wstats(vals: list[float], ws: list[float], fallback_sd: float) -> tuple[float, float, float] | None:
    """Weighted mean, its variance, effective sample size."""
    if not vals:
        return None
    W = sum(ws)
    m = sum(w * v for w, v in zip(ws, vals)) / W
    n_eff = W * W / sum(w * w for w in ws)
    var = sum(w * (v - m) ** 2 for w, v in zip(ws, vals)) / W if len(vals) > 1 else 0.0
    sd = math.sqrt(var) if n_eff >= 8 else fallback_sd   # too few to trust their own spread
    return m, (sd * sd) / n_eff, n_eff


def _post(prior_m: float, prior_v: float, rows: list[dict], half_life: float) -> dict:
    ws_c, ec, ws_u, un = [], [], [], []
    for r in rows:
        w = 0.5 ** (r["age"] / half_life)
        if r.get("ev_close") is not None:
            ws_c.append(w); ec.append(r["ev_close"])
        if r.get("units") is not None:
            ws_u.append(w); un.append(r["units"])
    prec, num = 1 / prior_v, prior_m / prior_v
    c, u = _wstats(ec, ws_c, CLOSE_SD), _wstats(un, ws_u, UNIT_SD)
    if c:
        v = c[1] + CLOSE_BIAS ** 2
        prec += 1 / v; num += c[0] / v
    if u:
        prec += 1 / u[1]; num += u[0] / u[1]
    m, sd = num / prec, math.sqrt(1 / prec)
    return {"edge": m, "sd": sd, "p_pos": 0.5 * (1 + math.erf(m / sd / math.sqrt(2))),
            "n": len(rows), "close": c[0] if c else None, "close_n": len(ec), "units": sum(un), "roi": u[0] if u else None}


def estimate(history: list[dict], board: Board) -> dict[tuple[str, str], dict]:
    """history: graded picks as {cat, tier, age, units (at 1u), ev_close}. Returns every cell seen, plus each tier's
    pooled estimate under (\"*\", tier)."""
    out = {}
    tiers = {t for _, t in board.tiers}
    for t in tiers:
        out[("*", t)] = _post(0.0, PRIOR_SD ** 2, [r for r in history if r["tier"] == t], board.half_life)
    for cat, t in {(r["cat"], r["tier"]) for r in history}:
        if t not in tiers:
            continue
        pt = out[("*", t)]
        out[(cat, t)] = _post(pt["edge"], pt["sd"] ** 2 + CELL_SD ** 2,
                              [r for r in history if r["cat"] == cat and r["tier"] == t], board.half_life)
    return out


def cell(cells: dict, cat: str, t: str) -> dict:
    """A cell's estimate; a category new to this tier falls back on the tier."""
    return cells.get((cat, t)) or {**cells[("*", t)], "n": 0, "close": None, "close_n": 0, "units": 0.0, "roi": None}


def qualifies(cells: dict, cat: str, t: str) -> bool:
    e = cell(cells, cat, t)
    if e["n"] < MIN_CELL_N and cat != "*":
        e = cells[("*", t)]
    return e["edge"] >= MIN_EDGE and e["p_pos"] >= MIN_PROB


def choose(candidates: list[dict], cells: dict, board: Board) -> list[dict]:
    """candidates: {id, cat, tier, ev}. The ones whose cell clears the bar, best cells first, up to the cap."""
    ok = []
    for c in candidates:
        if c["tier"] is None:
            continue
        e = cell(cells, c["cat"], c["tier"])
        if qualifies(cells, c["cat"], c["tier"]):
            ok.append({**c, "edge": e["edge"], "p_pos": e["p_pos"]})
    ok.sort(key=lambda c: (-c["edge"], -c["ev"]))
    out, per = [], {}
    for c in ok:
        k = (c["cat"], c["tier"])
        if per.get(k, 0) < PER_CELL:
            per[k] = per.get(k, 0) + 1
            out.append(c)
    return out[:board.cap]


def table(cells: dict) -> list[dict]:
    """Every cell for the page: rounded, best first."""
    rows = [{"cat": k[0], "tier": k[1], "edge": round(v["edge"], 4), "sd": round(v["sd"], 4), "p_pos": round(v["p_pos"], 3),
             "n": v["n"], "close": None if v["close"] is None else round(v["close"], 4), "close_n": v["close_n"],
             "units": round(v["units"], 2), "in": qualifies(cells, *k)}
            for k, v in cells.items()]
    return sorted(rows, key=lambda r: (r["cat"] != "*", -r["edge"]))


def marker(kind_id: str, chosen: list[dict], cells: dict, extra: dict) -> dict:
    """The ledger record: which picks made the list, and the cells as they stood (so the choice can be audited)."""
    return {"kind": "betlist", "id": kind_id, "picks": [c["id"] for c in chosen],
            "why": {c["id"]: {"cat": c["cat"], "tier": c["tier"], "edge": round(c["edge"], 4), "p_pos": round(c["p_pos"], 3)} for c in chosen},
            "cells": table(cells), **extra}
