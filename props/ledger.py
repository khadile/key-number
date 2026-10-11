"""Locked prop picks, graded results, and the props record.

ledger/props_picks.jsonl and ledger/props_results.jsonl are append-only, like
the game-line ledger: a pick is written once when its game's props lock and is
never edited; grading appends a result line.

Results: W, L, P (stat landed exactly on a whole-number line) or V (void: the
player did not play, which books refund). Voids risk nothing and stay out of
the record. Line value is how many yards (or catches) better our number was
than the same book's closing line.
"""
import json
from datetime import datetime, timezone

import numpy as np

from config import LEDGER, PARLAY_RISK, STRAIGHT_RISK

PICKS = LEDGER / "props_picks.jsonl"
RESULTS = LEDGER / "props_results.jsonl"
MARKETS = ("pass_yds", "rush_yds", "rec_yds", "receptions")


def read(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def append(path, rows: list[dict]) -> None:
    if rows:
        with path.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")


def grade_one(p: dict, actual: float | None) -> str:
    if actual is None:
        return "V"
    d = actual - p["line"] if p["side"] == "over" else p["line"] - actual
    return "W" if d > 0 else ("P" if d == 0 else "L")


def units_for(res: str, dec: float, risk: float) -> float:
    return risk * (dec - 1) if res == "W" else (-risk if res == "L" else 0.0)


def line_value(p: dict, close_line: float | None) -> float | None:
    if close_line is None:
        return None
    return close_line - p["line"] if p["side"] == "over" else p["line"] - close_line


def grade_all(picks: list[dict], stats: dict[tuple[str, str], dict], finished: set[str],
              closes: dict[tuple[str, str, str, str], float]) -> list[dict]:
    """Results for every pick whose game is final. `stats[(game_id, gsis_id)]` holds the
    player's stat line (absent = did not play); `closes[(game, player, market, book)]`
    is that book's closing line."""
    out, by_id = [], {p["id"]: p for p in picks}
    for p in picks:
        if p.get("kind") != "prop" or p["game_id"] not in finished:
            continue
        row = stats.get((p["game_id"], p["gsis_id"]))
        actual = None if row is None else row.get(p["market"])
        res = grade_one(p, actual)
        out.append({"id": p["id"], "result": res, "units": units_for(res, p["dec"], STRAIGHT_RISK),
                    "actual": actual, "clv": line_value(p, closes.get((p["game_id"], p["gsis_id"], p["market"], p["book"])))})
    graded = {r["id"]: r for r in out}
    for p in picks:
        if p.get("kind") != "parlay":
            continue
        legs = [graded.get(l) for l in p["legs"]]
        if any(l is None for l in legs):
            continue
        risk = p.get("risk", PARLAY_RISK)
        live = [l for l in legs if l["result"] not in ("P", "V")]
        if any(l["result"] == "L" for l in legs):
            res, u = "L", -risk
        elif not live:
            res, u = "P", 0.0
        else:
            dec = float(np.prod([by_id[l["id"]]["dec"] for l in live]))
            res, u = "W", risk * (dec - 1)
        out.append({"id": p["id"], "result": res, "units": u, "actual": None, "clv": None})
    return out


def grade_ledger(stats, finished, closes) -> list[dict]:
    picks = read(PICKS)
    done = {r["id"] for r in read(RESULTS)}
    now = datetime.now(timezone.utc).isoformat()
    new = [dict(r, graded_at=now) for r in grade_all(picks, stats, finished, closes) if r["id"] not in done]
    append(RESULTS, new)
    return new


def _tally(rows: list[dict]) -> dict:
    rows = [r for r in rows if r["result"] != "V"]
    w = sum(r["result"] == "W" for r in rows)
    l = sum(r["result"] == "L" for r in rows)
    p = sum(r["result"] == "P" for r in rows)
    units = sum(r["units"] for r in rows)
    risked = sum(r.get("risk", PARLAY_RISK) if r.get("kind") == "parlay" else STRAIGHT_RISK for r in rows)
    clvs = [r["clv"] for r in rows if r.get("clv") is not None]
    moved = [c for c in clvs if c != 0]
    return {"w": w, "l": l, "p": p, "n": len(rows), "units": round(units, 2),
            "roi": round(units / risked, 4) if risked else None,
            "clv": round(float(np.mean(clvs)), 2) if clvs else None, "clv_n": len(clvs),
            # of the picks whose line moved before kickoff, the share that moved our way
            "moved": len(moved), "beat": round(float(np.mean([c > 0 for c in moved])), 3) if moved else None}


def record(picks: list[dict], results: list[dict]) -> dict:
    res = {r["id"]: r for r in results}
    rows = [dict(p, **res[p["id"]]) for p in picks if p["id"] in res]
    straight = [r for r in rows if r.get("kind") == "prop"]
    out = {m: _tally([r for r in straight if r["market"] == m]) for m in MARKETS}
    out.update(potw=_tally([r for r in straight if r.get("potw")]),
               parlay=_tally([r for r in rows if r.get("kind") == "parlay" and r.get("type") == "golden"]),
               lottery=_tally([r for r in rows if r.get("kind") == "parlay" and r.get("type") == "lottery"]),
               promoted=_tally([r for r in straight if r.get("promoted")]),
               all=_tally(rows), weeks=[])
    for wk in sorted({r["week"] for r in rows}):
        wr = [r for r in rows if r["week"] == wk]
        out["weeks"].append({"week": wk, "all": _tally(wr), "potw": _tally([r for r in wr if r.get("potw")]),
                             **{m: _tally([r for r in wr if r.get("market") == m]) for m in MARKETS}})
    cal = []
    decided = [r for r in straight if r["result"] in ("W", "L")]
    for lo, hi in ((0.40, 0.50), (0.50, 0.53), (0.53, 0.56), (0.56, 0.60), (0.60, 1.0)):
        b = [r for r in decided if lo <= r["p_win"] / max(1 - r["p_push"], 1e-9) < hi]
        if b:
            cal.append({"lo": lo, "hi": hi, "n": len(b),
                        "said": float(np.mean([r["p_win"] / (1 - r["p_push"]) for r in b])),
                        "won": float(np.mean([r["result"] == "W" for r in b]))})
    out["calibration"] = cal
    return out
