"""Locked picks, graded results, and the record.

ledger/picks.jsonl and ledger/results.jsonl are append-only. A pick is written
once when its week is published and never edited; grading appends a result
line; a change of category is an appended "parlay_type" line. The same grading
functions score the pre-launch simulation, so the two records are computed
identically.
"""
import json
from datetime import datetime, timezone

import numpy as np

from config import LEDGER, PARLAY_RISK, STRAIGHT_RISK

PICKS = LEDGER / "picks.jsonl"
RESULTS = LEDGER / "results.jsonl"


def read(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def read_picks() -> list[dict]:
    """Locked picks with any later reclassification applied.

    The file is never rewritten, so a change of category is its own appended
    line ({"kind": "parlay_type", "id": ..., "type": ...}). Parlays locked
    before types existed default to "golden".
    """
    rows = read(PICKS)
    types = {r["id"]: r["type"] for r in rows if r.get("kind") == "parlay_type"}
    out = []
    for r in rows:
        if r.get("kind") == "parlay_type":
            continue
        if r.get("kind") == "parlay":
            r = {**r, "type": types.get(r["id"], r.get("type", "golden"))}
        out.append(r)
    return out


# ------------------------------------------------------------------ grading

def grade_straight(p: dict, home_score: int, away_score: int) -> str:
    margin = home_score - away_score
    if p["market"] == "spread":
        d = (margin if p["side"] == "home" else -margin) + p["line"]
    elif p["market"] == "total":
        total = home_score + away_score
        d = total - p["line"] if p["side"] == "over" else p["line"] - total
    else:  # moneyline: a tie pushes
        d = margin if p["side"] == "home" else -margin
    return "W" if d > 0 else ("P" if d == 0 else "L")


def units_for(result: str, dec: float, risk: float) -> float:
    return risk * (dec - 1) if result == "W" else (0.0 if result == "P" else -risk)


def clv_points(p: dict, close: dict | None) -> float | None:
    """Points of closing line value: how much better our number was than the close."""
    if not close:
        return None
    if p["market"] == "spread" and close.get("spread") is not None:
        h_close = -close["spread"] if p["side"] == "home" else close["spread"]
        return p["line"] - h_close
    if p["market"] == "total" and close.get("total") is not None:
        return close["total"] - p["line"] if p["side"] == "over" else p["line"] - close["total"]
    return None


def grade_all(picks: list[dict], finals: dict[str, tuple[int, int]], closes: dict[str, dict]) -> list[dict]:
    """Results for every pick whose game is final. Parlays grade from their legs."""
    out, by_id = [], {p["id"]: p for p in picks}
    for p in picks:
        if p.get("kind") == "parlay":
            continue
        if p["game_id"] not in finals:
            continue
        h, a = finals[p["game_id"]]
        res = grade_straight(p, h, a)
        out.append({"id": p["id"], "result": res, "units": units_for(res, p["dec"], STRAIGHT_RISK),
                    "clv": clv_points(p, closes.get(p["game_id"]))})
    graded = {r["id"]: r for r in out}
    for p in picks:
        if p.get("kind") != "parlay":
            continue
        legs = [graded.get(l) for l in p["legs"]]
        if any(l is None for l in legs):
            continue
        risk = p.get("risk", PARLAY_RISK)
        if any(l["result"] == "L" for l in legs):
            res, u = "L", -risk
        elif all(l["result"] == "P" for l in legs):
            res, u = "P", 0.0
        else:
            dec = float(np.prod([by_id[l["id"]]["dec"] for l in legs if l["result"] == "W"]))
            res, u = "W", risk * (dec - 1)
        out.append({"id": p["id"], "result": res, "units": u, "clv": None})
    return out


def grade_ledger(finals: dict[str, tuple[int, int]], closes: dict[str, dict]) -> list[dict]:
    """Append results for newly finished live picks."""
    picks = read_picks()
    done = {r["id"] for r in read(RESULTS)}
    now = datetime.now(timezone.utc).isoformat()
    new = [dict(r, graded_at=now) for r in grade_all(picks, finals, closes) if r["id"] not in done]
    append(RESULTS, new)
    return new


# ------------------------------------------------------------------ records

def _tally(rows: list[dict]) -> dict:
    w = sum(r["result"] == "W" for r in rows)
    l = sum(r["result"] == "L" for r in rows)
    p = sum(r["result"] == "P" for r in rows)
    units = sum(r["units"] for r in rows)
    risked = sum(r.get("risk", PARLAY_RISK) if r.get("kind") == "parlay" else STRAIGHT_RISK for r in rows)
    clvs = [r["clv"] for r in rows if r.get("clv") is not None]
    return {"w": w, "l": l, "p": p, "n": len(rows), "units": round(units, 2),
            "roi": round(units / risked, 4) if risked else None,
            "clv": round(float(np.mean(clvs)), 2) if clvs else None, "clv_n": len(clvs)}


def record(picks: list[dict], results: list[dict]) -> dict:
    """The record strip: ATS, O/U, ML, Picks of the Week, Golden Parlay, units, CLV, and by week."""
    res = {r["id"]: r for r in results}
    rows = [dict(p, **res[p["id"]]) for p in picks if p["id"] in res]
    straight = [r for r in rows if r.get("kind") != "parlay"]
    lottery = [r for r in rows if r.get("kind") == "parlay" and r.get("type") == "lottery"]
    out = {
        "ats": _tally([r for r in straight if r["market"] == "spread"]),
        "ou": _tally([r for r in straight if r["market"] == "total"]),
        "ml": _tally([r for r in straight if r["market"] == "ml"]),
        "potw": _tally([r for r in straight if r.get("potw")]),
        "parlay": _tally([r for r in rows if r.get("kind") == "parlay" and r.get("type", "golden") == "golden"]),
        "lottery": _tally(lottery),
        "all": _tally(rows),
        "weeks": [],
    }
    for wk in sorted({r["week"] for r in rows}):
        wr = [r for r in rows if r["week"] == wk]
        out["weeks"].append({"week": wk, "all": _tally(wr),
                             "ats": _tally([r for r in wr if r.get("market") == "spread"]),
                             "ou": _tally([r for r in wr if r.get("market") == "total"]),
                             "potw": _tally([r for r in wr if r.get("potw")])})
    # Calibration: does a stated 55% win 55%? (pushes excluded)
    cal = []
    decided = [r for r in straight if r["result"] != "P"]
    for lo in (0.45, 0.50, 0.53, 0.56, 0.60):
        hi = {0.45: 0.50, 0.50: 0.53, 0.53: 0.56, 0.56: 0.60, 0.60: 1.0}[lo]
        b = [r for r in decided if lo <= r["p_win"] / max(1 - r["p_push"], 1e-9) < hi]
        if b:
            cal.append({"lo": lo, "hi": hi, "n": len(b),
                        "said": float(np.mean([r["p_win"] / (1 - r["p_push"]) for r in b])),
                        "won": float(np.mean([r["result"] == "W" for r in b]))})
    out["calibration"] = cal
    return out
