"""NBA picks, results and records: the NFL ledgers' rules, a day at a time.

ledger/nba_picks.jsonl and ledger/nba_results.jsonl (game lines), ledger/nba_props_picks.jsonl and
ledger/nba_props_results.jsonl (player props) are append-only: a pick is written once when it locks and never
edited; grading appends a result line. Every pick carries `day`, its game's Eastern date.

Stakes as in the NFL (model/staking.py): 1u on a positive-EV pick, nothing on a lean, and every straight pick
counted at 1u in the record, leans included; quarter Kelly tracked beside it. Prop results add V (void: the
player did not play), which risks nothing and stays out of the record.

Closing value: besides points of line value, each graded pick carries `ev_close`, its expected value at the
closing fair price (Pinnacle's no-vig close; for props, the close anchored the same way as at the lock). With a
market-anchored model the edge is the price, so this is the number that shows first whether the edge is real.
"""
from datetime import datetime, timezone

import numpy as np

from board.ledger import append, grade_straight, read
from config import LEDGER, PARLAY_RISK, STRAIGHT_RISK
from model.staking import kelly_of, kelly_tally

PICKS = LEDGER / "nba_picks.jsonl"
RESULTS = LEDGER / "nba_results.jsonl"
PROPS_PICKS = LEDGER / "nba_props_picks.jsonl"
PROPS_RESULTS = LEDGER / "nba_props_results.jsonl"
PROP_MARKETS = ("pts", "reb", "ast", "fg3", "pra", "pr", "pa", "ra")


def units_for(res: str, dec: float, risk: float) -> float:
    return risk * (dec - 1) if res == "W" else (-risk if res == "L" else 0.0)


def _parlay(p: dict, graded: dict, by_id: dict) -> dict | None:
    legs = [graded.get(l) for l in p["legs"]]
    if any(l is None for l in legs):
        return None
    risk = p.get("risk", PARLAY_RISK)
    live = [l for l in legs if l["result"] not in ("P", "V")]
    if any(l["result"] == "L" for l in legs):
        res, u = "L", -risk
    elif not live:
        res, u = "P", 0.0
    else:
        res, u = "W", risk * (float(np.prod([by_id[l["id"]]["dec"] for l in live])) - 1)
    return {"id": p["id"], "result": res, "units": u}


def grade_games(picks: list[dict], finals: dict[int, tuple[int, int]], closes: dict) -> list[dict]:
    """Game-line results. `finals[game_id]` = (home, away) score; `closes[pick id]` = {clv, ev_close}."""
    out, by_id = [], {p["id"]: p for p in picks}
    for p in picks:
        if p.get("kind") not in ("pick", "alt") or p["game_id"] not in finals:   # alt: a Ladder leg
            continue
        res = grade_straight(p, *finals[p["game_id"]])
        out.append({"id": p["id"], "result": res, "units": units_for(res, p["dec"], STRAIGHT_RISK), **closes.get(p["id"], {})})
    graded = {r["id"]: r for r in out}
    out += [r for p in picks if p.get("kind") == "parlay" and (r := _parlay(p, graded, by_id))]
    return out


def grade_props(picks: list[dict], stats: dict[tuple[int, int], dict], finished: set[int], closes: dict) -> list[dict]:
    """Prop results. `stats[(game_id, athlete_id)]`: his line (absent: did not play, void)."""
    out, by_id = [], {p["id"]: p for p in picks}
    for p in picks:
        if p.get("kind") != "prop" or p["game_id"] not in finished:
            continue
        row = stats.get((p["game_id"], p["athlete_id"]))
        actual = None if row is None else row.get(p["market"])
        if actual is None:
            res = "V"
        else:
            d = actual - p["line"] if p["side"] == "over" else p["line"] - actual
            res = "W" if d > 0 else ("P" if d == 0 else "L")
        out.append({"id": p["id"], "result": res, "units": units_for(res, p["dec"], p.get("risk", STRAIGHT_RISK)),
                    "actual": actual, **closes.get(p["id"], {})})
    graded = {r["id"]: r for r in out}
    out += [r for p in picks if p.get("kind") == "parlay" and (r := _parlay(p, graded, by_id))]
    return out


def append_new(path, results: list[dict]) -> list[dict]:
    done = {r["id"] for r in read(path)}
    now = datetime.now(timezone.utc).isoformat()
    new = [dict(r, graded_at=now) for r in results if r["id"] not in done]
    append(path, new)
    return new


def _tally(rows: list[dict]) -> dict:
    rows = [r for r in rows if r.get("result") in ("W", "L", "P")]
    w = sum(r["result"] == "W" for r in rows)
    l = sum(r["result"] == "L" for r in rows)
    p = sum(r["result"] == "P" for r in rows)
    units = sum(r["units"] for r in rows)
    risked = sum(r.get("risk", PARLAY_RISK) if r.get("kind") == "parlay" else STRAIGHT_RISK for r in rows)
    evc = [r["ev_close"] for r in rows if r.get("ev_close") is not None]
    clv = [r["clv"] for r in rows if r.get("clv") is not None]
    return {"w": w, "l": l, "p": p, "n": len(rows), "units": round(units, 2),
            "roi": round(units / risked, 4) if risked else None,
            "ev_close": round(float(np.mean(evc)), 4) if evc else None, "ev_close_n": len(evc),
            "clv": round(float(np.mean(clv)), 2) if clv else None,
            "ev": round(float(np.mean([r["ev"] for r in rows if r.get("ev") is not None])), 4) if rows else None}


def _cal(rows: list[dict]) -> list[dict]:
    cal = []
    decided = [r for r in rows if r.get("kind") in ("pick", "prop") and r["result"] in ("W", "L")]
    for lo, hi in ((0.30, 0.45), (0.45, 0.50), (0.50, 0.53), (0.53, 0.56), (0.56, 0.60), (0.60, 1.0)):
        b = [r for r in decided if lo <= r["p_win"] / max(1 - r["p_push"], 1e-9) < hi]
        if b:
            cal.append({"lo": lo, "hi": hi, "n": len(b), "said": float(np.mean([r["p_win"] / (1 - r["p_push"]) for r in b])),
                        "won": float(np.mean([r["result"] == "W" for r in b]))})
    return cal


def _graded(picks: list[dict], results: list[dict]) -> list[dict]:
    res = {r["id"]: r for r in results}
    out = [dict(p, **res[p["id"]]) for p in picks if p["id"] in res]
    for r in out:   # every straight pick counts at 1u, leans included
        if r.get("kind") != "parlay":
            r.update(units=units_for(r["result"], r["dec"], STRAIGHT_RISK), risk=STRAIGHT_RISK)
    return out


def _kelly(rows: list[dict]) -> dict:
    pos = [r for r in rows if r.get("kind") != "parlay" and kelly_of(r) > 0]
    return {**kelly_tally(pos), "flat": _tally(pos)}


def record_games(picks: list[dict], results: list[dict]) -> dict:
    rows = [r for r in _graded(picks, results) if r.get("kind") != "alt"]   # a Ladder leg counts only in its Ladder
    st = [r for r in rows if r.get("kind") == "pick"]
    days = sorted({r["day"] for r in rows})
    return {
        "ats": _tally([r for r in st if r["market"] == "spread"]), "ou": _tally([r for r in st if r["market"] == "total"]),
        "ml": _tally([r for r in st if r["market"] == "ml"]), "potd": _tally([r for r in st if r.get("potd")]),
        "parlay": _tally([r for r in rows if r.get("kind") == "parlay" and r.get("type") == "golden"]),
        "lottery": _tally([r for r in rows if r.get("kind") == "parlay" and r.get("type") == "lottery"]),
        "all": _tally(rows), "split": {"pos": _tally([r for r in st if r["ev"] > 0]), "neg": _tally([r for r in st if r["ev"] <= 0])},
        "with_model": {"with": _tally([r for r in st if r.get("with_model") is True]),
                       "against": _tally([r for r in st if r.get("with_model") is False])},
        "kelly": _kelly(rows),
        "days": [{"day": d, "all": _tally([r for r in rows if r["day"] == d]),
                  "pos": _tally([r for r in st if r["day"] == d and r["ev"] > 0]),
                  **{k: _tally([r for r in st if r["day"] == d and r["market"] == m]) for k, m in (("ats", "spread"), ("ou", "total"), ("ml", "ml"))},
                  "potd": _tally([r for r in st if r["day"] == d and r.get("potd")])} for d in days],
        "calibration": _cal(st),
    }


def record_longshots(picks: list[dict], results: list[dict]) -> dict:
    """The long-shot watchlist at its own stake (0.25u each): record, units, and how often the model said they
    would hit against how often they did."""
    res = {r["id"]: r for r in results}
    rows = [dict(p, **res[p["id"]]) for p in picks if p.get("kind") == "longshot" and p["id"] in res and res[p["id"]]["result"] in ("W", "L")]
    n = len(rows)
    staked = sum(r["risk"] for r in rows)
    units = sum(r["units"] for r in rows)
    return {"n": n, "w": sum(r["result"] == "W" for r in rows), "l": sum(r["result"] == "L" for r in rows), "p": 0,
            "units": round(units, 2), "staked": round(staked, 2), "roi": round(units / staked, 4) if staked else None,
            "said_lo": round(float(np.mean([r["p_narrow"] for r in rows])), 4) if n else None,
            "said_hi": round(float(np.mean([r["p_wide"] for r in rows])), 4) if n else None,
            "implied": round(float(np.mean([1 / r["dec"] for r in rows])), 4) if n else None,
            "voids": sum(1 for p in picks if p.get("kind") == "longshot" and res.get(p["id"], {}).get("result") == "V")}


def record_props(picks: list[dict], results: list[dict]) -> dict:
    rows = _graded(picks, results)
    st = [r for r in rows if r.get("kind") == "prop"]
    days = sorted({r["day"] for r in rows})
    return {
        **{m: _tally([r for r in st if r["market"] == m]) for m in PROP_MARKETS},
        "over": _tally([r for r in st if r["side"] == "over"]), "under": _tally([r for r in st if r["side"] == "under"]),
        "potd": _tally([r for r in st if r.get("potd")]),
        "parlay": _tally([r for r in rows if r.get("kind") == "parlay" and r.get("type") == "golden"]),
        "lottery": _tally([r for r in rows if r.get("kind") == "parlay" and r.get("type") == "lottery"]),
        "all": _tally(rows), "straight": _tally(st), "kelly": _kelly(rows),
        "voids": sum(r["result"] == "V" for r in st),
        # by the size of the edge at the lock: the NFL props rule's question, asked of NBA's own picks
        "split": {"big": _tally([r for r in st if r["ev"] >= 0.02]), "small": _tally([r for r in st if r["ev"] < 0.02])},
        "days": [{"day": d, "all": _tally([r for r in rows if r["day"] == d]),
                  "big": _tally([r for r in st if r["day"] == d and r["ev"] >= 0.02]),
                  **{m: _tally([r for r in st if r["day"] == d and r["market"] == m]) for m in ("pts", "reb", "ast", "fg3")},
                  "combo": _tally([r for r in st if r["day"] == d and r["market"] in ("pra", "pr", "pa", "ra")]),
                  "potd": _tally([r for r in st if r["day"] == d and r.get("potd")])} for d in days],
        "calibration": _cal(st),
    }
